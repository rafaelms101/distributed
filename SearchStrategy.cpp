#include "SearchStrategy.h"

#include <thread>
#include <vector>
#include <algorithm>
#include <fstream>
#include <unistd.h>

#include "QueryQueue.h"

SearchStrategy::SearchStrategy(int num_queues, float _base_start, float _base_end, faiss::gpu::StandardGpuResources* _res) :
		base_start(_base_start), base_end(_base_end), res(_res) {
	const long block_size_in_bytes = sizeof(float) * cfg.d * cfg.block_size;
	const long distance_block_size_in_bytes = sizeof(float) * cfg.k * cfg.block_size;
	const long label_block_size_in_bytes = sizeof(faiss::Index::idx_t) * cfg.k * cfg.block_size;

	distance_buffer = new SyncBuffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	label_buffer = new SyncBuffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 

	for (int i = 0; i < num_queues; i++) {
		query_buffer.push_back(new SyncBuffer(block_size_in_bytes, 500 * 1024 * 1024 / block_size_in_bytes)); //500MB
		all_distance_buffers.push_back(new SyncBuffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes)); //100 MB 
		all_label_buffers.push_back(new SyncBuffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes)); //100 MB 
	}

	load_bench_data(true, best_block_point_cpu);
	load_bench_data(false, best_block_point_gpu);
}

void SearchStrategy::load_bench_data(bool cpu, long& best) {
	char file_path[100];
	sprintf(file_path, "%s/%s_%d_%d_%d_%d_%d_%d", PROF_ROOT, cpu ? "cpu" : "gpu", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
	std::ifstream file;
	file.open(file_path);

	if (!file.good()) {
		std::printf("File %s/%s_%d_%d_%d_%d_%d_%d doesn't exist\n", PROF_ROOT, cpu ? "cpu" : "gpu", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
		std::exit(-1);
	}

	int total_size;
	file >> total_size;

	std::vector<double> times(total_size + 1);

	best = -1;
	double btpb = 999999999;

	for (int i = 1; i <= total_size; i++) {
		file >> times[i];
		
		double tpb = times[i] / i; 
		
		if (tpb < btpb) {
			btpb = tpb;
			best = i;
		}
	}

	file.close();
}

void SearchStrategy::merge(long num_queries, std::vector<float*>& all_distances, std::vector<faiss::Index::idx_t*>& all_labels, float* distance_array, faiss::Index::idx_t* label_array) {
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

void SearchStrategy::merger() {
	long sent = 0;

	faiss::Index::idx_t* I = new faiss::Index::idx_t[cfg.block_size * cfg.k];
	float* D = new float[cfg.block_size * cfg.k];
	
	while (sent < cfg.test_length) {
		for (int i = 0; i < all_distance_buffers.size(); i++) {
			all_distance_buffers[i]->waitForData(1);
			all_label_buffers[i]->waitForData(1);
		}
		
		std::vector<float*> all_distances;
		std::vector<faiss::Index::idx_t*> all_labels;
		for (int i = 0; i < all_distance_buffers.size(); i++) {
			all_distances.push_back((float*) all_distance_buffers[i]->front());
			all_labels.push_back((faiss::Index::idx_t*) all_label_buffers[i]->front());
		}
		
		merge(cfg.block_size, all_distances, all_labels, D, I);
		
		distance_buffer->insert(1, (byte*) D);
		label_buffer->insert(1, (byte*) I);
		
		for (int i = 0; i < all_distance_buffers.size(); i++) {
			all_distance_buffers[i]->remove(1);
			all_label_buffers[i]->remove(1);
		}
		
		sent += cfg.block_size;
	}
}

//
//void HybridSearchStrategy::gpu_process(std::mutex* cleanup_mutex) {
//	while (qm->sent_queries() < cfg.test_length) {
//		bool emptyQueue = true;
//
//		for (QueryQueue* qq : qm->queues()) {
//			if (!qq->on_gpu) continue;
//
//			qq->lock();
//
//			long nq = std::min(qq->size(), best_query_point_gpu);
//			assert(nq >= 0);
//
//			qq->search(nq);
//			qq->unlock();
//
//			emptyQueue = emptyQueue && nq == 0;
//		}
//
//		if (cleanup_mutex->try_lock()) {
//			qm->shrinkQueryBuffer();
//			qm->mergeResults();
//			cleanup_mutex->unlock();
//		}
//
//		if (emptyQueue && ! qm->gpu_loading) {
//			QueryQueue* cpu_queue = qm->biggestCPUQueue();
//			QueryQueue* gpu_queue = qm->firstGPUQueue();
//
//			auto cpu_size = cpu_queue->size();
//			auto gpu_size = gpu_queue->size();
//			
//			if (cpu_size > 1000 && gpu_size * 5 < cpu_size) { //arbitrary threshold
//				qm->switchToGPU(cpu_queue, gpu_queue);
//			}
//		}
//	}
//	
//	while (qm->gpu_loading) {
//		usleep(1 * 1000 * 1000);
//	}
//
//	if (cfg.shard == 0) {
//		std::printf("bases exchanged: %ld\n", qm->bases_exchanged);
//		for (int j = 0; j < qm->bases_exchanged; j++) {
//			std::printf("b: ");
//			for (int i = 0; i < qm->_queues.size(); i++) {
//				if (i == qm->switches[j].first) {
//					std::printf("<");
//				} else if (i == qm->switches[j].second) {
//					std::printf(">");
//				}
//
//				std::printf("%ld ", qm->log[j][i]);
//			}
//			std::printf("\n");
//		}
//	}
//}
//
//void HybridSearchStrategy::cpu_process(std::mutex* cleanup_mutex) {
//	while (qm->sent_queries() < cfg.test_length) {
//		for (QueryQueue* qq : qm->queues()) {
//			if (qq->on_gpu) continue;
//
//			qq->lock();
//
//			long nq = std::min(qq->size(), best_query_point_cpu);
//			assert(nq >= 0);
//			qq->search(nq);
//			qq->unlock();
//		}
//
//		if (cleanup_mutex->try_lock()) {
//			qm->shrinkQueryBuffer();
//			qm->mergeResults();
//			cleanup_mutex->unlock();
//		}
//	}
//}
//
//void HybridSearchStrategy::setup() {
//	qm = new QueueManager(&query_buffer, &label_buffer, &distance_buffer);
//
//	long pieces = cfg.total_pieces;
//	float gpu_pieces = cfg.gpu_pieces;
//	float step = (base_end - base_start) / pieces;
//
//	for (int i = 0; i < pieces; i++) {
//		auto cpu_index = load_index(base_start + i * step, base_start + (i + 1) * step, cfg);
//		deb("Creating index from %f to %f", base_start + i * step, base_start + (i + 1) * step);
//		QueryQueue* qq = new QueryQueue(cpu_index, qm, i);
//
//		if (i < gpu_pieces) {
//			qq->create_gpu_index(*res);
//		}
//	}
//}
//
//void HybridSearchStrategy::start_search_process() {
//	std::mutex cleanup_mutex;
//
//	std::thread gpu_thread { &HybridSearchStrategy::gpu_process, this, &cleanup_mutex };
//	std::thread cpu_thread { &HybridSearchStrategy::cpu_process, this, &cleanup_mutex };
//
//	cpu_thread.join();
//	gpu_thread.join();
//}

void CpuOnlySearchStrategy::setup() {
	cpu_index = load_index(base_start, base_end, cfg);
}

void CpuOnlySearchStrategy::start_search_process() {
	faiss::Index::idx_t* I = new faiss::Index::idx_t[cfg.eval_length * cfg.k];
	float* D = new float[cfg.eval_length * cfg.k];

	long sent = 0;
	
	while (sent < cfg.test_length) {
		deb("waiting for queries");
		long nblocks = std::min(query_buffer[0]->num_entries(), 200L);
		
		if (nblocks == 0) {
			auto sleep_time_us = std::min(query_buffer[0]->arrivalInterval() * 1000000, 1000.0);
			usleep(sleep_time_us);
			continue;
		}
		
		long nqueries = nblocks * cfg.block_size;
		deb("%ld queries available", nqueries);

		cpu_index->search(nqueries, (float*) query_buffer[0]->front(), cfg.k, D, I);

		query_buffer[0]->remove(nblocks);
		label_buffer->insert(nblocks, (byte*) I);
		distance_buffer->insert(nblocks, (byte*) D);

		sent += nqueries;
		deb("sent %ld queries", nqueries);
	}
}
//
//void GpuOnlySearchStrategy::merger() {
//	std::unique_lock < std::mutex > lck { should_merge_mutex };
//	long sent = 0;
//
//	while (sent < cfg.test_length) {
//		if (buffer_start_id > sent) {
//			merge_procedure(buffer_start_id, sent, all_distance_buffers, all_label_buffers, distance_buffer, label_buffer);
//		} else {
//			should_merge.wait(lck, [this, &sent] {return buffer_start_id > sent;});
//		}
//	}
//
//	deb("Sent: %ld", sent);
//}
//
//void GpuOnlySearchStrategy::setup() {
//	deb("search_gpu called");
//
//	float step = (base_end - base_start) / cfg.total_pieces;
//
//	for (int i = 0; i < cfg.total_pieces; i++) {
//		cpu_bases.push_back(load_index(base_start + i * step, base_start + (i + 1) * step, cfg));
//		
//		if (i < cfg.gpu_pieces) {
//			gpu_indexes.push_back(static_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(res, cfg.shard % cfg.gpus_per_node, cpu_bases[i], nullptr)));
//			baseMap.push_back(i);
//			reverseBaseMap.push_back(i);
//		} else {
//			baseMap.push_back(-1);
//		}
//		
//		all_label_buffers.push_back(new Buffer(sizeof(faiss::Index::idx_t) * cfg.k, 1000000));
//		all_distance_buffers.push_back(new Buffer(sizeof(float) * cfg.k, 1000000));
//		proc_ids.push_back(0);
//	}
//}
//
//void GpuOnlySearchStrategy::start_search_process() {
//	long bases_exchanged = 0;
//
//	long log[1000][proc_ids.size()];
//	std::vector<std::pair<int, int>> switches;
//
//	std::thread merge_thread { &GpuOnlySearchStrategy::merger, this };
//
//	while (buffer_start_id < cfg.test_length) {
//		query_buffer.waitForData(1);
//
//		//TODO: subclass buffer and create a QueryBuffer, that deals better with queries (aka. no more "* cfg.d" etc)
//		for (int i = 0; i < cpu_bases.size(); i++) {
//			long buffer_idx = proc_ids[i] - buffer_start_id;
//			long available_queries = query_buffer.entries() * cfg.block_size - buffer_idx;
//			
//			if (available_queries == 0) continue;
//			
//			if (baseMap[i] == -1) {
//				switches.push_back(std::make_pair(reverseBaseMap[0], i));
//				for (int i = 0; i < proc_ids.size(); i++) {
//					log[bases_exchanged][i] = query_buffer.entries() * cfg.block_size - (proc_ids[i] - buffer_start_id);
//				}
//				
//				baseMap[reverseBaseMap[0]] = -1;
//				reverseBaseMap[0] = -1;
//				gpu_indexes[0]->copyFrom(cpu_bases[i]);
//				reverseBaseMap[0] = i;
//				baseMap[i] = 0;
//				bases_exchanged++;
//			}
//			
//			auto buffer_ptr = (float*) (query_buffer.peekFront()) + buffer_idx * cfg.d;
//
//			while (available_queries >= 1) {
//				long nqueries = std::min(available_queries, best_query_point_gpu);
//				auto lb = all_label_buffers[i];
//				auto db = all_distance_buffers[i];
//
//				lb->waitForSpace(nqueries);
//				db->waitForSpace(nqueries);
//
//				deb("searched %ld queries on base %d", nqueries, i);
//				
//				gpu_indexes[baseMap[i]]->search(nqueries, buffer_ptr, cfg.k, (float*) db->peekEnd(), (faiss::Index::idx_t*) lb->peekEnd());
//				
//				db->add(nqueries);
//				lb->add(nqueries);
//
//				available_queries -= nqueries;
//				proc_ids[i] += nqueries;
//				buffer_ptr += nqueries * cfg.d;
//			}
//		}
//
//		auto min_id = *std::min_element(proc_ids.begin(), proc_ids.end());
//		if (min_id > buffer_start_id) {
//			query_buffer.consume((min_id - buffer_start_id) / cfg.block_size);
//			buffer_start_id = min_id;
//			deb("buffer start id is now: %ld", buffer_start_id);
//			should_merge.notify_one();
//		}
//	}
//
//	deb("search_gpu finished");
//
//	merge_thread.join();
//
//	if (cfg.shard == 0) {
//		std::printf("bases exchanged: %ld\n", bases_exchanged);
//	
//		for (int j = 0; j < bases_exchanged; j++) {
//			std::printf("b: ");
//			for (int i = 0; i < proc_ids.size(); i++) {
//				if (i == switches[j].first) {
//					std::printf("<");
//				} else if (i == switches[j].second) {
//					std::printf(">");
//				}
//	
//				std::printf("%ld ", log[j][i]);
//			}
//			std::printf("\n");
//		}
//	}
//}
//
void FixedSearchStrategy::setup() {
	float total_share = base_end - base_start;
	auto cpu_share = (cfg.total_pieces - cfg.gpu_pieces) / static_cast<float>(cfg.total_pieces);

	cpu_index = load_index(base_start, base_start + total_share * cpu_share, cfg);
	deb("loading cpu base from %.2f to %.2f", base_start, base_start + total_share * cpu_share);
	
	auto tmp_index = load_index(base_start + total_share * cpu_share, base_start + total_share, cfg);
	gpu_index = static_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(res, cfg.shard % cfg.gpus_per_node, tmp_index, nullptr));
}

void FixedSearchStrategy::start_search_process() {
	std::thread merger_process { &FixedSearchStrategy::merger, this };
	std::thread cpu_process { &FixedSearchStrategy::process, this, cpu_index, query_buffer[0], all_distance_buffers[0], all_label_buffers[0], best_block_point_cpu };
	std::thread gpu_process { &FixedSearchStrategy::process, this, gpu_index, query_buffer[1], all_distance_buffers[1], all_label_buffers[1], best_block_point_gpu  };

	merger_process.join();
	cpu_process.join();
	gpu_process.join();
}

void FixedSearchStrategy::process(faiss::Index* index, SyncBuffer* query_buffer, SyncBuffer* distance_buffer, SyncBuffer* label_buffer, long cutoff_point) {
	faiss::Index::idx_t* I = new faiss::Index::idx_t[cutoff_point * cfg.block_size * cfg.k];
	float* D = new float[cutoff_point * cfg.block_size * cfg.k];
	
	long blocks_remaining = cfg.test_length / cfg.block_size;

	while (blocks_remaining >= 1) {
		query_buffer->waitForData(1);
		long nb = std::min(query_buffer->num_entries(), cutoff_point);

		deb("searching %ld queries on the gpu", nb * cfg.block_size);
		index->search(nb * cfg.block_size, (float*) query_buffer->front(), cfg.k, D, I);

		blocks_remaining -= nb;

		query_buffer->remove(nb);
		label_buffer->insert(nb, (byte*) I);
		distance_buffer->insert(nb, (byte*) D);
	}
}
