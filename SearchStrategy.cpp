#include "SearchStrategy.h"

#include <thread>
#include <vector>
#include <algorithm>
#include <fstream>
#include <unistd.h>
#include "faiss/gpu/GpuCloner.h"

SearchStrategy::SearchStrategy(int num_queues, float _base_start, float _base_end, bool usesCPU, bool usesGPU, faiss::gpu::StandardGpuResources* _res) :
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

//	cfg.nb /= cfg.total_pieces;
	if (usesCPU) load_bench_data(true, best_block_point_cpu, best_block_point_cpu_time);
	if (usesGPU) load_bench_data(false, best_block_point_gpu, best_block_point_gpu_time);
//	cfg.nb *= cfg.total_pieces;
}

void SearchStrategy::load_bench_data(bool cpu, long& best, double& best_time) {
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
			best_time = times[i];
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
	
	while (sent < cfg.num_blocks * cfg.block_size) {
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


void HybridSearchStrategy::gpu_process() {
	faiss::Index::idx_t* I = new faiss::Index::idx_t[best_block_point_gpu * cfg.block_size * cfg.k];
	float* D = new float[best_block_point_gpu * cfg.block_size * cfg.k];

	while (remaining_blocks >= 1) {
		//REMEMBER, IT MIGHT BE THAT INSIDE HERE REMAINING_BLOCKS IS, IN FACT, 0
		
		auto nb = std::min(query_buffer[on_gpu]->num_entries(), best_block_point_gpu);
		
		if (nb == 0) {
			// find the biggest queue
			long largest_index = 0;
			long largest_size = 0;
			
			for (int i = 0; i < query_buffer.size(); i++) {
				long size = query_buffer[i]->num_entries();
				if (size > largest_size) {
					largest_size = size;
					largest_index = i;
				}
			}
			
			// exchange it IF there are enough queries
			long gpu_size = query_buffer[on_gpu]->num_entries();
			
			if (largest_size >= 600 && gpu_size <= 200) { //3000 and 1000 queries
				//exchange it
				deb("Switching %d[size=%d] with %d[size=%d]", on_gpu, gpu_size * cfg.block_size, largest_index, largest_size * cfg.block_size);
				on_gpu = largest_index;
				gpu_index->copyFrom(cpu_index[largest_index]);
				deb("Switch finished");
			}
		} else {
			mutvec[on_gpu]->lock();
			
			deb("Processing %d queries on the GPU[%d]", nb * cfg.block_size, on_gpu);
			gpu_index->search(nb * cfg.block_size, (float*) query_buffer[on_gpu]->front(), cfg.k, D, I);
			
			query_buffer[on_gpu]->remove(nb);
			all_distance_buffers[on_gpu]->insert(nb, (byte*) D);
			all_label_buffers[on_gpu]->insert(nb, (byte*) I);

			remaining_blocks -= nb;

			mutvec[on_gpu]->unlock();
		}
	}
}

void HybridSearchStrategy::cpu_process() {
	faiss::Index::idx_t* I = new faiss::Index::idx_t[best_block_point_cpu * cfg.block_size * cfg.k];
	float* D = new float[best_block_point_cpu * cfg.block_size * cfg.k];
	
	while (remaining_blocks >= 1) {
		//REMEMBER, IT MIGHT BE THAT INSIDE HERE REMAINING_BLOCKS IS, IN FACT, 0
		
		for (int i = 0; i < cpu_index.size(); i++) {
			if (i == on_gpu) continue;
			
			auto nb = std::min(query_buffer[i]->num_entries(), best_block_point_cpu); //TODO: maybe reduce this to like 20
			
			if (nb == 0) continue;
			
			mutvec[i]->lock();
			
			deb("Processing %d queries on the CPU[%d]", nb * cfg.block_size, i);
			cpu_index[i]->search(nb * cfg.block_size, (float*) query_buffer[i]->front(), cfg.k, D, I);

			query_buffer[i]->remove(nb);
			all_distance_buffers[i]->insert(nb, (byte*) D);
			all_label_buffers[i]->insert(nb, (byte*) I);
							
			remaining_blocks -= nb;

			mutvec[i]->unlock();
		}
	}
}

void HybridSearchStrategy::start_search_process() {
	std::thread merger_process { &HybridSearchStrategy::merger, this };
	std::thread cpu_thread { &HybridSearchStrategy::cpu_process, this };
	std::thread gpu_thread { &HybridSearchStrategy::gpu_process, this };

	merger_process.join();
	cpu_thread.join();
	gpu_thread.join();
}

void HybridSearchStrategy::setup() {
	deb("search_gpu called");

	float step = (base_end - base_start) / cfg.total_pieces;

	for (int i = 0; i < cfg.total_pieces; i++) {
//		cpu_index.push_back(load_index(base_start + i * step, base_start + (i + 1) * step, cfg));
		cfg.nb /= cfg.total_pieces;
		cpu_index.push_back(load_index(0, 1, cfg));
		cfg.nb *= cfg.total_pieces;
		mutvec.push_back(new std::mutex());
	}

	gpu_index = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(res, cfg.shard % cfg.gpus_per_node, cpu_index[0], nullptr));
	on_gpu = 0;
	remaining_blocks = cfg.num_blocks * cpu_index.size();
}

void CpuOnlySearchStrategy::setup() {
	cpu_index = load_index(base_start, base_end, cfg);
}

void CpuOnlySearchStrategy::start_search_process() {
	faiss::Index::idx_t* I = new faiss::Index::idx_t[best_block_point_cpu * cfg.block_size * cfg.k];
	float* D = new float[best_block_point_cpu * cfg.block_size * cfg.k];

	long sent = 0;
	
	while (sent < cfg.num_blocks * cfg.block_size) {
		deb("waiting for queries");
		long nblocks = std::min(query_buffer[0]->num_entries(), best_block_point_cpu);
		
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

void GpuOnlySearchStrategy::setup() {
	deb("search_gpu called");

//	float step = (base_end - base_start) / cfg.gpu_pieces;

	auto index = load_index(0, 1, cfg);
	
	for (int i = 0; i < cfg.gpu_pieces; i++) {
//		cpu_index.push_back(load_index(base_start + i * step, base_start + (i + 1) * step, cfg));
		cpu_index.push_back(index);
		remaining_blocks.push_back(cfg.num_blocks);
	}

	gpu_index = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(res, cfg.shard % cfg.gpus_per_node, cpu_index[0], nullptr));
	
//	deb("search_gpu called");
//
//	float step = (base_end - base_start) / cfg.gpu_pieces;
//
//	for (int i = 0; i < cfg.gpu_pieces; i++) {
//		cpu_index.push_back(load_index(base_start + i * step, base_start + (i + 1) * step, cfg));
//		remaining_blocks.push_back(cfg.num_blocks);
//	}
//	
//	gpu_index = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(res, cfg.shard % cfg.gpus_per_node, cpu_index[0], nullptr));
}

void GpuOnlySearchStrategy::start_search_process() {
	faiss::Index::idx_t* I = new faiss::Index::idx_t[best_block_point_gpu * cfg.block_size * cfg.k];
	float* D = new float[best_block_point_gpu * cfg.block_size * cfg.k];

	bool allFinished = false;
	long on_gpu = 0;
	
	std::thread merger_process { &GpuOnlySearchStrategy::merger, this };
	
	while (! allFinished) {
		allFinished = true;
		
		for (int i = 0; i < cpu_index.size(); i++) {
			if (remaining_blocks[i] == 0) continue;
			
			allFinished = false;
			
			if (on_gpu != i) {
				auto before = now();
				gpu_index->copyFrom(cpu_index[i]);
				std::printf("%d) Switch: %lf\n", cfg.shard, now() - before);
				on_gpu = i;
			}
			
			while (remaining_blocks[i] >= 1) {
				query_buffer[i]->waitForData(1);
				
				long nb = std::min(query_buffer[i]->num_entries(), best_block_point_gpu);
				gpu_index->search(nb * cfg.block_size, (float*) query_buffer[i]->front(), cfg.k, D, I);
				
				query_buffer[i]->remove(nb);
				all_distance_buffers[i]->insert(nb, (byte*) D);
				all_label_buffers[i]->insert(nb, (byte*) I);
				
//				num_blocks_to_be_processed -= nb;
				remaining_blocks[i] -= nb;
			}
		}
	}
	
	merger_process.join();
}

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
	
	long blocks_remaining = cfg.num_blocks;

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

void BestSearchStrategy::gpu_process() {
	faiss::Index::idx_t* I = new faiss::Index::idx_t[best_block_point_gpu * cfg.block_size * cfg.k];
	float* D = new float[best_block_point_gpu * cfg.block_size * cfg.k];

	
	long gpu_thr = best_block_point_gpu / best_block_point_gpu_time;
	long cpu_thr = best_block_point_cpu / best_block_point_cpu_time;
	long simple_threshold = long(2 * switch_time * cpu_thr * gpu_thr / (gpu_thr - cpu_thr));
	
	long switch_threshold = 2 * (switch_time + (best_block_point_gpu_time - best_block_point_cpu_time) / 2) 
				/ (best_block_point_cpu_time / best_block_point_cpu - best_block_point_gpu_time / best_block_point_gpu);
	
	deb("switch threshold is %d. simple is %d", switch_threshold, simple_threshold);
	
	while (remaining_blocks >= 1) {
		//REMEMBER, IT MIGHT BE THAT INSIDE HERE REMAINING_BLOCKS IS, IN FACT, 0
		auto nb = std::min(query_buffer[on_gpu]->num_entries(), best_block_point_gpu);

		
		if (nb == 0) {
			// find the biggest queue
			long largest_index = 0;
			long largest_size = 0;
			
			for (int i = 0; i < query_buffer.size(); i++) {
				long size = query_buffer[i]->num_entries();
				if (size > largest_size) {
					largest_size = size;
					largest_index = i;
				}
			}
			
			// exchange it IF <complex logic>			
			if (largest_index != on_gpu && largest_size > switch_threshold) {
				//exchange it
				deb("Switching %d[size=%d] with %d[size=%d]", on_gpu, query_buffer[on_gpu]->num_entries() * cfg.block_size, largest_index, largest_size * cfg.block_size);
				on_gpu = largest_index;
				gpu_index->copyFrom(cpu_index[largest_index]);
				deb("Switch finished");
			}
		} else {
			mutvec[on_gpu]->lock();
			
			deb("Processing %d queries on the GPU[%d]", nb * cfg.block_size, on_gpu);
			gpu_index->search(nb * cfg.block_size, (float*) query_buffer[on_gpu]->front(), cfg.k, D, I);
			
			query_buffer[on_gpu]->remove(nb);
			all_distance_buffers[on_gpu]->insert(nb, (byte*) D);
			all_label_buffers[on_gpu]->insert(nb, (byte*) I);

			remaining_blocks -= nb;

			mutvec[on_gpu]->unlock();
		}
	}
}

void BestSearchStrategy::cpu_process() {
	faiss::Index::idx_t* I = new faiss::Index::idx_t[best_block_point_cpu * cfg.block_size * cfg.k];
	float* D = new float[best_block_point_cpu * cfg.block_size * cfg.k];
	
	while (remaining_blocks >= 1) {
		//REMEMBER, IT MIGHT BE THAT INSIDE HERE REMAINING_BLOCKS IS, IN FACT, 0
		
		for (int i = 0; i < cpu_index.size(); i++) {
			if (i == on_gpu) continue;
			
			auto nb = std::min(query_buffer[i]->num_entries(), best_block_point_cpu); //TODO: maybe reduce this to like 20
			
			if (nb == 0) continue;
			
			mutvec[i]->lock();
			
			deb("Processing %d queries on the CPU[%d]", nb * cfg.block_size, i);
			cpu_index[i]->search(nb * cfg.block_size, (float*) query_buffer[i]->front(), cfg.k, D, I);

			query_buffer[i]->remove(nb);
			all_distance_buffers[i]->insert(nb, (byte*) D);
			all_label_buffers[i]->insert(nb, (byte*) I);
							
			remaining_blocks -= nb;

			mutvec[i]->unlock();
		}
	}
}

void BestSearchStrategy::start_search_process() {
	std::thread merger_process { &BestSearchStrategy::merger, this };
	std::thread cpu_thread { &BestSearchStrategy::cpu_process, this };
	std::thread gpu_thread { &BestSearchStrategy::gpu_process, this };

	merger_process.join();
	cpu_thread.join();
	gpu_thread.join();
}

void BestSearchStrategy::setup() {
	deb("search_gpu called");

	float step = (base_end - base_start) / cfg.total_pieces;

	for (int i = 0; i < cfg.total_pieces; i++) {
//		cpu_index.push_back(load_index(base_start + i * step, base_start + (i + 1) * step, cfg));
		cfg.nb /= cfg.total_pieces;
		cpu_index.push_back(load_index(0, 1, cfg));
		cfg.nb *= cfg.total_pieces;
		mutvec.push_back(new std::mutex());
	}

	gpu_index = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(res, cfg.shard % cfg.gpus_per_node, cpu_index[0], nullptr));
	on_gpu = 0;
	remaining_blocks = cfg.num_blocks * cpu_index.size();
	
	auto before = now();
	gpu_index->copyFrom(cpu_index[0]);
	switch_time = now() - before;
	deb("switch time = %lf", switch_time);
}
