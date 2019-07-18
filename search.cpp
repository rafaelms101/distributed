#include "search.h"

#include <mpi.h>
#include <algorithm>
#include <future>
#include <fstream>
#include <unistd.h>
#include <list>
 
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

#define THRES 80l

static void cpu_process(QueueManager* qm) {
	while (qm->sent_queries() < cfg.test_length) {
		for (QueryQueue* qq : qm->queues()) {
			if (qq->on_gpu) continue;
			auto size = qq->size();
			long nq = std::min(size.size, THRES);
			assert(nq >= 0);
			qq->search(nq);
		}

		qm->shrinkQueryBuffer();
		qm->mergeResults();
	}
}

static void gpu_process(QueueManager* qm) {
	while (qm->sent_queries() < cfg.test_length) {
		bool emptyQueue = true;
		
		for (QueryQueue* qq : qm->queues()) {
			if (! qq->on_gpu) continue;

			auto size = qq->size();
			long nq = std::min(size.size, THRES);
			auto after_size = qq->size();
			
			if (after_size.size < size.size) {
				std::printf("[ERROR1] buffer start: %ld -> %ld, entries: %ld -> %ld, start query id: %ld -> %ld\n", size.buffer_start_query_id, after_size.buffer_start_query_id, size.queries_in_buffer, after_size.queries_in_buffer, size.starting_query_id, after_size.starting_query_id);
			}
//			
//			assert(nq <= after_size.size); //FAILED HERE
			
			if (nq < 0) {
				std::printf("[ERROR2] buffer start: %ld , entries: %ld , start query id: %ld \n", size.buffer_start_query_id,  size.queries_in_buffer,  size.starting_query_id);
			}
			
			assert(nq >= 0);

			qq->search(nq);
			emptyQueue = emptyQueue && nq == 0;
//			assert(qq->size().size >= 0);
		}

//		qm->shrinkQueryBuffer();
//		qm->mergeResults();

//		if (emptyQueue) {
//			QueryQueue* qq = qm->biggestCPUQueue();
			
//			if (qq->size() > 1000) { //arbitrary threshold
//				qm->switchToGPU(qq);
//			}
//		}
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

		for (int q = 0; q < nqueries; q++) {
			std::printf("Query %ld\n", sent + q);
			for (int j = 0; j < cfg.k; j++) {
				std::printf("%ld: %f\n", labels[q * cfg.k + j], distances[q * cfg.k + j]);
			}
		}


		query_buffer.consume(nblocks);

		label_buffer.add(nblocks);
		distance_buffer.add(nblocks);

		sent += nqueries;
		deb("sent %ld queries", nqueries);
	}


	receiver.join();
}

void merge(long num_queries, std::vector<float*>& all_distances, std::vector<faiss::Index::idx_t*>& all_labels, float* distance_array, faiss::Index::idx_t* label_array) {
	if (num_queries == 0) return;

	std::vector<long> idxs(2);

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

static void search_hybrid(Buffer& query_buffer, Buffer& distance_buffer, Buffer& label_buffer, faiss::gpu::StandardGpuResources& res, int shard, int nshards) {
	QueueManager qm(& query_buffer, & label_buffer, & distance_buffer);

	//TODO: a potencial problem is that we might be reading the entire database first, and then cutting it. This might be a problem in huge databases.
	//TODO: we are assuming that we are dividing the database in two parts
	float start_percent = float(shard) / nshards;
	float end_percent = float(shard + 1) / nshards;
	
	long pieces = 2;
	float gpu_pieces = 1;
	float step = (end_percent - start_percent) / pieces;

	for (int i = 0; i < pieces; i++) {
		auto cpu_index = load_index(start_percent + i * step, start_percent + (i+1) * step, cfg);
		deb("Creating index from %f to %f", start_percent + i * step, start_percent + (i+1) * step);
		QueryQueue* qq = new QueryQueue(cpu_index, &qm);
		
		if (i < gpu_pieces) {
			qq->create_gpu_index(res);
		}
	}

	bool finished = false;
	std::thread receiver { comm_handler, & query_buffer, & distance_buffer, & label_buffer, std::ref(finished), std::ref(cfg) };
	std::thread gpu_thread { gpu_process, & qm };
	std::thread cpu_thread { cpu_process, & qm };

	receiver.join();
	cpu_thread.join();
	gpu_thread.join();
}

void search(int shard, int nshards, Config& cfg) {
	const long block_size_in_bytes = sizeof(float) * cfg.d * cfg.block_size;
	Buffer query_buffer(block_size_in_bytes, 100 * 1024 * 1024 / block_size_in_bytes); //100 MB
	
	const long distance_block_size_in_bytes = sizeof(float) * cfg.k * cfg.block_size;
	const long label_block_size_in_bytes = sizeof(faiss::Index::idx_t) * cfg.k * cfg.block_size;
	Buffer distance_buffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	Buffer label_buffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 
	
	faiss::gpu::StandardGpuResources res;
	res.setTempMemory(1500 * 1024 * 1024);
	
	search_hybrid(query_buffer, distance_buffer, label_buffer, res, shard, nshards);
//	search_cpu(query_buffer, distance_buffer, label_buffer, shard, nshards);
}
