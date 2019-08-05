#include "search.h"

#include <mpi.h>
#include <algorithm>
#include <future>
#include <fstream>
#include <unistd.h>
#include <list>
#include <vector>
 
#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

#include "gpu/GpuAutoTune.h"
#include "gpu/StandardGpuResources.h"
#include "gpu/GpuIndexIVFPQ.h"

#include "utils.h"
#include "Buffer.h"
#include "readSplittedIndex.h"
#include "ExecPolicy.h"
#include "QueryQueue.h"
#include "QueueManager.h"

static void comm_handler(Buffer* query_buffer, Buffer* distance_buffer, Buffer* label_buffer, bool& finished_receiving, Config& cfg) {
	long queries_received = 0;
	long queries_sent = 0;
	
	float dummy;
	MPI_Send(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD); //signal that we are ready to receive queries

	while (! finished_receiving || queries_sent < queries_received) {
		if (! finished_receiving) {
			MPI_Status status;
			int message_arrived;
			MPI_Iprobe(GENERATOR, 0, MPI_COMM_WORLD, &message_arrived, &status);

			if (message_arrived) {
				int message_size;
				MPI_Get_count(&status, MPI_FLOAT, &message_size);

				if (message_size == 1) {
					finished_receiving = true;
					float dummy;
					MPI_Recv(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
				} else {
					query_buffer->waitForSpace(1);
					float* recv_data = reinterpret_cast<float*>(query_buffer->peekEnd());
					MPI_Recv(recv_data, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
					query_buffer->add(1);
					queries_received += cfg.block_size;
					
					assert(status.MPI_ERROR == MPI_SUCCESS);
				}
			}
		}
		
		auto ready = std::min(distance_buffer->entries(), label_buffer->entries());

		if (ready >= 1) {
			queries_sent += ready * cfg.block_size;
			
			faiss::Index::idx_t* label_ptr = reinterpret_cast<faiss::Index::idx_t*>(label_buffer->peekFront());
			float* dist_ptr = reinterpret_cast<float*>(distance_buffer->peekFront());
			
			//TODO: Optimize this to an Immediate Synchronous Send
			MPI_Ssend(label_ptr, cfg.k * cfg.block_size * ready, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
			MPI_Ssend(dist_ptr, cfg.k * cfg.block_size * ready, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);
			
			label_buffer->consume(ready);
			distance_buffer->consume(ready);
		}
	}

	MPI_Ssend(&dummy, 1, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
	deb("Finished receiving queries");	
}

static faiss::IndexIVFPQ* load_index(float start_percent, float end_percent, Config& cfg) {
	deb("Started loading");
	
	char index_path[500];
	sprintf(index_path, "%s/index_%d_%d_%d", INDEX_ROOT, cfg.nb, cfg.ncentroids, cfg.m);

	deb("Loading file: %s", index_path);

	FILE* index_file = fopen(index_path, "r");
	auto cpu_index = static_cast<faiss::IndexIVFPQ*>(read_index(index_file, start_percent, end_percent));

	deb("Ended loading");

	//TODO: Maybe we shouldnt set nprobe here
	cpu_index->nprobe = cfg.nprobe;
	return cpu_index;
}

static void store_profile_data(std::vector<double>& procTimes, Config& cfg) {
	system("mkdir -p prof"); //to make sure that the "prof" dir exists
	
	//now we write the time data on a file
	char file_path[100];
	sprintf(file_path, "prof/%d_%d_%d_%d_%d_%d", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
	std::ofstream file;
	file.open(file_path);

	int blocks = procTimes.size() / BENCH_REPEATS;
	file << blocks << std::endl;

	int ptr = 0;

	for (int b = 1; b <= blocks; b++) {
		std::vector<double> times;

		for (int repeats = 1; repeats <= BENCH_REPEATS; repeats++) {
			times.push_back(procTimes[ptr++]);
		}

		std::sort(times.begin(), times.end());

		int mid = BENCH_REPEATS / 2;
		file << times[mid] << std::endl;
	}

	file.close();
}


static void printResult(long num_queries, float* distance_array, faiss::Index::idx_t* label_array) {
	static long qn = 0;
		for (int q = 0; q < num_queries; q++) {
			std::printf("Query %ld\n", qn);
			qn++;
			
			for (int i = 0; i < cfg.k; i++) {
				std::printf("%ld: %f\n", label_array[q * cfg.k + i], distance_array[q * cfg.k + i]);
			}
		}
}

static void merge(long num_queries, std::vector<float*>& all_distances, std::vector<faiss::Index::idx_t*>& all_labels, float* distance_array, faiss::Index::idx_t* label_array) {
	static long qn = 0;
	
	if (num_queries == 0) return;
	
	auto oda = distance_array;
	auto ola = label_array;
	
	std::vector<long> idxs(all_distances.size());

	for (int q = 0; q < num_queries; q++) {
		for (int i = 0; i < all_distances.size(); i++) {
			idxs[i] = q * cfg.k;
		}

		for (int topi = 0; topi < cfg.k; topi++) {
			int from = 0;

			for (int i = 1; i < idxs.size(); i++) {
				if (all_distances[i][idxs[i]] < all_distances[from][idxs[from]]) {
					from = i;
				}
			}

			*distance_array = all_distances[from][idxs[from]];
			*label_array = all_labels[from][idxs[from]];

			distance_array++;
			label_array++;
			idxs[from]++;
		}
	}
}

static void cpu_process(QueueManager* qm, std::mutex* cleanup_mutex) {
	while (qm->sent_queries() < cfg.test_length) {
		for (QueryQueue* qq : qm->queues()) {
			if (qq->on_gpu) continue;
			
			qq->lock();
			
			long nq = std::min(qq->size(), 120l);
			assert(nq >= 0);
			qq->search(nq);
			qq->unlock();
		}

		if (cleanup_mutex->try_lock()) {
			qm->shrinkQueryBuffer();
			qm->mergeResults();
			cleanup_mutex->unlock();
		}
	}
}

static void gpu_process(QueueManager* qm, std::mutex* cleanup_mutex) {
	while (qm->sent_queries() < cfg.test_length) {
		bool emptyQueue = true;
		
		for (QueryQueue* qq : qm->queues()) {
			if (! qq->on_gpu) continue;
			
			qq->lock();

			long nq = std::min(qq->size(), 120l);
			assert(nq >= 0);
			
			qq->search(nq);
			qq->unlock();
			
			emptyQueue = emptyQueue && nq == 0;
		}
//
		if (cleanup_mutex->try_lock()) {
			qm->shrinkQueryBuffer();
			qm->mergeResults();
			cleanup_mutex->unlock();
		}

		if (emptyQueue) {
			QueryQueue* qq = qm->biggestCPUQueue();
			
			if (qq->size() > 1000) { //arbitrary threshold
				qm->switchToGPU(qq);
			}
		}
	}
	
	std::printf("bases exchanged: %ld\n", qm->bases_exchanged);
	for (int j = 0; j < qm->bases_exchanged; j++) {
		for (int i = 0; i < qm->_queues.size(); i++) {
			if (i == qm->switches[j].first) {
				std::printf("<");
			} else if (i == qm->switches[j].second) {
				std::printf(">");
			}

			std::printf("%ld ", qm->log[j][i]);
		}
		std::printf("\n");
	}
}

static void search_cpu(Buffer& query_buffer, Buffer& distance_buffer, Buffer& label_buffer, int shard, int nshards) {
	//TODO: a potencial problem is that we might be reading the entire database first, and then cutting it. This might be a problem in huge databases.
	//TODO: we are assuming that we are dividing the database in two parts
	float start_percent = float(shard) / nshards;
	float end_percent = float(shard + 1) / nshards;

	auto cpu_index = load_index(start_percent, end_percent, cfg);

	bool finished = false;
	std::thread receiver { comm_handler, & query_buffer, & distance_buffer, & label_buffer, std::ref(finished), std::ref(cfg) };
	
	long sent = 0;
	while (sent < cfg.test_length) {
		deb("waiting for queries");
		query_buffer.waitForData(1);
		long nblocks = query_buffer.entries();
		long nqueries = nblocks * cfg.block_size;
		deb("%ld queries available", nqueries);

		faiss::Index::idx_t* labels = (faiss::Index::idx_t*) label_buffer.peekEnd();
		float* distances = (float*) distance_buffer.peekEnd();
		float* query_start = (float*) query_buffer.peekFront();

		label_buffer.waitForSpace(nblocks);
		distance_buffer.waitForSpace(nblocks);

		cpu_index->search(nqueries, query_start, cfg.k, distances, labels);
		
		query_buffer.consume(nblocks);
		label_buffer.add(nblocks);
		distance_buffer.add(nblocks);

		sent += nqueries;
		deb("sent %ld queries", nqueries);
	}


	receiver.join();
}

static void merge_procedure(long& buffer_start_id, long& sent, std::vector<Buffer*>& all_distance_buffers, std::vector<Buffer*>& all_label_buffers, Buffer& distance_buffer, Buffer& label_buffer) {
	//merge
	auto num_queries = buffer_start_id - sent;
	
	if (num_queries == 0) return;
	assert(num_queries > 0);
	
	std::vector<float*> all_distances;
	std::vector<faiss::Index::idx_t*> all_labels;
	for (int i = 0; i < all_distance_buffers.size(); i++) {
		auto dptr = (float*) all_distance_buffers[i]->peekFront();
		auto lptr = (faiss::Index::idx_t*) all_label_buffers[i]->peekFront();
		all_distances.push_back(dptr);
		all_labels.push_back(lptr);
	}

	distance_buffer.waitForSpace(num_queries);
	label_buffer.waitForSpace(num_queries);

	merge(num_queries, all_distances, all_labels, reinterpret_cast<float*>(distance_buffer.peekEnd()), reinterpret_cast<faiss::Index::idx_t*>(label_buffer.peekEnd()));

	distance_buffer.add(num_queries / cfg.block_size);
	label_buffer.add(num_queries / cfg.block_size);

	for (int i = 0; i < all_distance_buffers.size(); i++) {
		all_distance_buffers[i]->consume(num_queries);
		all_label_buffers[i]->consume(num_queries);
	}

	sent += num_queries;
	deb("merged queries up to %ld", sent);
}

//TODO: pass as parameter instead of being a global
std::condition_variable should_merge;
std::mutex should_merge_mutex;

static void merger(long& buffer_start_id, std::vector<Buffer*>& all_distance_buffers, std::vector<Buffer*>& all_label_buffers, Buffer& distance_buffer, Buffer& label_buffer) {
	std::unique_lock<std::mutex> lck { should_merge_mutex };
	long sent = 0;
	
	while (sent < cfg.test_length) {
		if (buffer_start_id > sent) {
			merge_procedure(buffer_start_id, sent, all_distance_buffers, all_label_buffers, distance_buffer, label_buffer);
		} else {
			should_merge.wait(lck, [&buffer_start_id, &sent] { return buffer_start_id > sent; });
		}
	}
	
	deb("Sent: %ld", sent);
}

//TODO: dont use globals like this, instead move it to a class or something
std::mutex sync_mutex;

static void gpu_process_fixed(Buffer& query_buffer, long& buffer_start_id, long& sent, std::vector<long>& proc_ids, std::vector<faiss::IndexIVFPQ*>& all_gpu_bases, faiss::gpu::GpuIndexIVFPQ* gpu_index, std::vector<Buffer*>& all_distance_buffers, std::vector<Buffer*> all_label_buffers) {
	long log[1000][proc_ids.size()];
	std::vector<std::pair<int, int>> switches;
	long bases_exchanged = 0;
	
	long current_base = 1;
	while (sent < cfg.test_length) {
		for (int i = 1; i <= all_gpu_bases.size(); i++) {
			if (proc_ids[i] == cfg.test_length) continue;
			
			sync_mutex.lock();
			long buffer_idx = proc_ids[i] - buffer_start_id;
			long available_queries = query_buffer.entries() * cfg.block_size - buffer_idx;
			auto buffer_ptr = (float*)(query_buffer.peekFront()) + buffer_idx * cfg.d;
			sync_mutex.unlock();
			
			if (available_queries == 0) continue;
			assert(available_queries > 0);
			
			if (current_base != i) {
				switches.push_back(std::make_pair(current_base, i));
				for (int i = 0; i < proc_ids.size(); i++) {
					log[bases_exchanged][i] = query_buffer.entries() * cfg.block_size - (proc_ids[i] - buffer_start_id);
				}
				
				gpu_index->copyFrom(all_gpu_bases[i - 1]); // we subtract 1 because this vector doesnt include the cpu entry (whereas the other ones - like proc_ids - do).
				current_base = i;
				bases_exchanged++;
			}
			
			while (available_queries >= 1) {
				long nqueries = std::min(available_queries, 120l);
				auto lb = all_label_buffers[i]; 
				auto db = all_distance_buffers[i]; 
				
				lb->waitForSpace(nqueries);
				db->waitForSpace(nqueries);
				
				deb("searching %ld queries on base %d", nqueries, i);
				gpu_index->search(nqueries, buffer_ptr, cfg.k, (float*) db->peekEnd(), (faiss::Index::idx_t*) lb->peekEnd());
				
				lb->add(nqueries);
				db->add(nqueries);
				
				available_queries -= nqueries;
				proc_ids[i] += nqueries;
				buffer_ptr += nqueries * cfg.d;
			}
		}
	}
	
	std::printf("bases exchanged: %ld\n", bases_exchanged);
	for (int j = 0; j < bases_exchanged; j++) {
		for (int i = 0; i < proc_ids.size(); i++) {
			if (i == switches[j].first) {
				std::printf("<");
			} else if (i == switches[j].second) {
				std::printf(">");
			}
			
			std::printf("%ld ", log[j][i]);
		}
		std::printf("\n");
	}
	
}

static void cpu_process_fixed(Buffer& query_buffer, long& buffer_start_id, long& sent, std::vector<long>& proc_ids, faiss::IndexIVFPQ* cpu_index, std::vector<Buffer*>& all_distance_buffers, std::vector<Buffer*> all_label_buffers, Buffer& distance_buffer, Buffer& label_buffer) {
	while (sent < cfg.test_length) {
//		query_buffer.waitForData(1);
		
		long buffer_idx = proc_ids[0] - buffer_start_id;	
		long available_queries = query_buffer.entries() * cfg.block_size - buffer_idx;
		
		if (available_queries > 0) {
			auto buffer_ptr = (float*) (query_buffer.peekFront()) + buffer_idx * cfg.d;
			long nqueries = std::min(available_queries, 120l);
			auto lb = all_label_buffers[0];
			auto db = all_distance_buffers[0];
	
			lb->waitForSpace(nqueries);
			db->waitForSpace(nqueries);
	
			deb("searching %ld queries on the cpu", nqueries);
			cpu_index->search(nqueries, buffer_ptr, cfg.k, (float*) db->peekEnd(), (faiss::Index::idx_t*) lb->peekEnd());
	
			lb->add(nqueries);
			db->add(nqueries);
	
			proc_ids[0] += nqueries;
		}

		auto min_id = *std::min_element(proc_ids.begin(), proc_ids.end());
		if (min_id > buffer_start_id) {
			sync_mutex.lock();
			query_buffer.consume((min_id - buffer_start_id) / cfg.block_size);
			buffer_start_id = min_id;
			sync_mutex.unlock();

			merge_procedure(buffer_start_id, sent, all_distance_buffers, all_label_buffers, distance_buffer, label_buffer);
		}
		
	}
}

//TODO: make the algorithms classes instead
static void search_cpu_fixed(Buffer& query_buffer, Buffer& distance_buffer, Buffer& label_buffer, faiss::gpu::StandardGpuResources& res, int shard, int nshards) {
	//TODO: a potencial problem is that we might be reading the entire database first, and then cutting it. This might be a problem in huge databases.
	//TODO: we are assuming that we are dividing the database in two parts

	deb("search_gpu called");

	float start_percent = float(shard) / nshards;
	float end_percent = float(shard + 1) / nshards;
	float total_share = end_percent - start_percent;

	faiss::IndexIVFPQ* cpu_base;
	std::vector<faiss::IndexIVFPQ*> all_gpu_bases;
	faiss::gpu::GpuIndexIVFPQ* gpu_base;
	
	std::vector<Buffer*> all_distance_buffers;
	std::vector<Buffer*> all_label_buffers;
	std::vector<long> proc_ids;
	
	auto cpu_share = (cfg.total_pieces - cfg.gpu_pieces) / static_cast<float>(cfg.total_pieces);
	
	cpu_base = load_index(start_percent, start_percent + total_share * cpu_share, cfg);
	deb("loading cpu base from %.2f to %.2f", start_percent, start_percent + total_share * cpu_share);
	all_label_buffers.push_back(new Buffer(sizeof(faiss::Index::idx_t) * cfg.k, 1000000));
	all_distance_buffers.push_back(new Buffer(sizeof(float) * cfg.k, 1000000));
	proc_ids.push_back(0);
	
	auto gpu_share = total_share - cpu_share;
	auto gpu_slice = gpu_share / cfg.gpu_pieces;
	
	for (int i = 0; i < cfg.gpu_pieces; i++) {
		auto start = start_percent + cpu_share + gpu_slice * i;
		auto end = start + gpu_slice;
		all_gpu_bases.push_back(load_index(start, end, cfg));
		deb("loading gpu base from %.2f to %.2f", start, end);
		all_label_buffers.push_back(new Buffer(sizeof(faiss::Index::idx_t) * cfg.k, 1000000));
		all_distance_buffers.push_back(new Buffer(sizeof(float) * cfg.k, 1000000));
		proc_ids.push_back(0);
	}

	auto gpu_index = static_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(&res, 0, all_gpu_bases[0], nullptr));
	
	bool finished = false;
	std::thread receiver { comm_handler, &query_buffer, &distance_buffer, &label_buffer, std::ref(finished), std::ref(cfg) };

	long sent = 0;
	long buffer_start_id = 0;
	std::thread cpu_process { cpu_process_fixed, std::ref(query_buffer), std::ref(buffer_start_id), std::ref(sent), std::ref(proc_ids), cpu_base, std::ref(all_distance_buffers), std::ref(all_label_buffers), std::ref(distance_buffer), std::ref(label_buffer) };
	std::thread gpu_process { gpu_process_fixed, std::ref(query_buffer), std::ref(buffer_start_id), std::ref(sent), std::ref(proc_ids), std::ref(all_gpu_bases), gpu_index, std::ref(all_distance_buffers), std::ref(all_label_buffers) };
	
	receiver.join();
	cpu_process.join();
	gpu_process.join();
}

static void search_gpu(Buffer& query_buffer, Buffer& distance_buffer, Buffer& label_buffer, faiss::gpu::StandardGpuResources& res, int shard, int nshards) {
	//TODO: a potencial problem is that we might be reading the entire database first, and then cutting it. This might be a problem in huge databases.
	//TODO: we are assuming that we are dividing the database in two parts
	
	deb("search_gpu called");
	
	
	long current_base = 0;
	long bases_exchanged = 0;
	
	float start_percent = float(shard) / nshards;
	float end_percent = float(shard + 1) / nshards;
	float step = (end_percent - start_percent) / cfg.total_pieces;
	
	std::vector<faiss::IndexIVFPQ*> cpu_bases;
	std::vector<Buffer*> all_distance_buffers;
	std::vector<Buffer*> all_label_buffers;
	std::vector<long> proc_ids;
	long buffer_start_id = 0;
	
	for (int i = 0; i < cfg.total_pieces; i++) {
		cpu_bases.push_back(load_index(start_percent + i * step, start_percent + (i + 1) * step, cfg)); 
		all_label_buffers.push_back(new Buffer(sizeof(faiss::Index::idx_t) * cfg.k, 1000000));
		all_distance_buffers.push_back(new Buffer(sizeof(float) * cfg.k, 1000000));
		proc_ids.push_back(0);
	}

	auto gpu_index = static_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_bases[0], nullptr));
	
	long log[1000][proc_ids.size()];
	std::vector<std::pair<int, int>> switches;
	
	bool finished = false;
	std::thread receiver { comm_handler, &query_buffer, &distance_buffer, &label_buffer, std::ref(finished), std::ref(cfg) };	
	std::thread merge_thread { merger, std::ref(buffer_start_id), std::ref(all_distance_buffers), std::ref(all_label_buffers), std::ref(distance_buffer), std::ref(label_buffer) };
	
	while (buffer_start_id < cfg.test_length) {
		query_buffer.waitForData(1);
		
		//TODO: subclass buffer and create a QueryBuffer, that deals better with queries (aka. no more "* cfg.d" etc)
		for (int i = 0; i < cpu_bases.size(); i++) {
			if (current_base != i) {
				switches.push_back(std::make_pair(current_base, i));
				for (int i = 0; i < proc_ids.size(); i++) {
					log[bases_exchanged][i] = query_buffer.entries()
							* cfg.block_size - (proc_ids[i] - buffer_start_id);
				}
				
				gpu_index->copyFrom(cpu_bases[i]);
				current_base = i;
				bases_exchanged++;
			}
			
			long buffer_idx = proc_ids[i] - buffer_start_id;	
			long available_queries = query_buffer.entries() * cfg.block_size - buffer_idx;
			auto buffer_ptr = (float*)(query_buffer.peekFront()) + buffer_idx * cfg.d;
			
			while (available_queries >= 1) {
				long nqueries = std::min(available_queries, 120l);
				auto lb = all_label_buffers[i];
				auto db = all_distance_buffers[i];
				
				lb->waitForSpace(nqueries);
				db->waitForSpace(nqueries);
				
				deb("searched %ld queries on base %d", nqueries, i);
				gpu_index->search(nqueries, buffer_ptr, cfg.k, (float*) db->peekEnd(), (faiss::Index::idx_t*) lb->peekEnd());
				
				db->add(nqueries);
				lb->add(nqueries);
				
				available_queries -= nqueries;
				proc_ids[i] += nqueries;
				buffer_ptr += nqueries * cfg.d;
			}
		}
		
		auto min_id = *std::min_element(proc_ids.begin(), proc_ids.end());
		if (min_id > buffer_start_id) {
			query_buffer.consume((min_id - buffer_start_id) / cfg.block_size);
			buffer_start_id = min_id;
			deb("buffer start id is now: %ld", buffer_start_id);
			should_merge.notify_one();
		}
	}
	
	deb("search_gpu finished");
	
	merge_thread.join();
	receiver.join();
	
	std::printf("bases exchanged: %ld\n", bases_exchanged);
	
	for (int j = 0; j < bases_exchanged; j++) {
		for (int i = 0; i < proc_ids.size(); i++) {
			if (i == switches[j].first) {
				std::printf("<");
			} else if (i == switches[j].second) {
				std::printf(">");
			}

			std::printf("%ld ", log[j][i]);
		}
		std::printf("\n");
	}
}

static void search_hybrid(Buffer& query_buffer, Buffer& distance_buffer, Buffer& label_buffer, faiss::gpu::StandardGpuResources& res, int shard, int nshards) {
	QueueManager qm(& query_buffer, & label_buffer, & distance_buffer);

	//TODO: a potential problem is that we might be reading the entire database first, and then cutting it. This might be a problem in huge databases.
	float start_percent = float(shard) / nshards;
	float end_percent = float(shard + 1) / nshards;
	
	long pieces = cfg.total_pieces;
	float gpu_pieces = cfg.gpu_pieces;
	float step = (end_percent - start_percent) / pieces;

	for (int i = 0; i < pieces; i++) {
		auto cpu_index = load_index(start_percent + i * step, start_percent + (i+1) * step, cfg);
		deb("Creating index from %f to %f", start_percent + i * step, start_percent + (i+1) * step);
		QueryQueue* qq = new QueryQueue(cpu_index, &qm, i);
		
		if (i < gpu_pieces) {
			qq->create_gpu_index(res);
		}
	}
	
	std::mutex cleanup_mutex;
	
	bool finished = false;
	std::thread receiver { comm_handler, &query_buffer, &distance_buffer, &label_buffer, std::ref(finished), std::ref(cfg) };
	std::thread gpu_thread { gpu_process, &qm, &cleanup_mutex };
	std::thread cpu_thread { cpu_process, &qm, &cleanup_mutex };

	receiver.join();
	cpu_thread.join();
	gpu_thread.join();
}


void search(int shard, int nshards, Config& cfg) {
	deb("search called");
	
	const long block_size_in_bytes = sizeof(float) * cfg.d * cfg.block_size;
	Buffer query_buffer(block_size_in_bytes, 100 * 1024 * 1024 / block_size_in_bytes); //100 MB
	
	const long distance_block_size_in_bytes = sizeof(float) * cfg.k * cfg.block_size;
	const long label_block_size_in_bytes = sizeof(faiss::Index::idx_t) * cfg.k * cfg.block_size;
	Buffer distance_buffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	Buffer label_buffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 
	
	faiss::gpu::StandardGpuResources res;
	res.setTempMemory(1500 * 1024 * 1024);
	
	
	
	if (cfg.search_algorithm == SearchAlgorithm::Cpu) {
		search_cpu(query_buffer, distance_buffer, label_buffer, shard, nshards);
	} else if (cfg.search_algorithm == SearchAlgorithm::Hybrid) {
		search_hybrid(query_buffer, distance_buffer, label_buffer, res, shard, nshards);
	} else if (cfg.search_algorithm == SearchAlgorithm::Gpu) {
		search_gpu(query_buffer, distance_buffer, label_buffer, res, shard, nshards);
	} else if (cfg.search_algorithm == SearchAlgorithm::CpuFixed) {
		search_cpu_fixed(query_buffer, distance_buffer, label_buffer, res, shard, nshards);
	}
}
