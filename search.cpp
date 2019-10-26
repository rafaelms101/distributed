#include "search.h"

#include <mpi.h>
#include <algorithm>
#include <future>
#include <fstream>
#include <unistd.h>
#include <thread> 

#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

#include "../faiss/gpu/GpuAutoTune.h"
#include "../faiss/gpu/StandardGpuResources.h"
#include "../faiss/gpu/GpuIndexIVFPQ.h"

#include "utils.h"
#include "Buffer.h"
#include "SyncBuffer.h"
#include "readSplittedIndex.h"
#include "ExecPolicy.h"
#include "SearchStrategy.h"

static void comm_handler_single(SyncBuffer* query_buffer, SyncBuffer* distance_buffer, SyncBuffer* label_buffer, bool& finished_receiving, Config& cfg) {
	deb("Receiver");
	
	byte tmp_buffer[cfg.block_size * cfg.d * sizeof(float)];
	
	long queries_received = 0;
	long queries_sent = 0;
	
	float dummy;
	MPI_Send(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD); //signal that we are ready to receive queries

	deb("Now waiting for queries");

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
					MPI_Recv(tmp_buffer, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
					query_buffer->insert(1, tmp_buffer);
					queries_received += cfg.block_size;
					
					assert(status.MPI_ERROR == MPI_SUCCESS);
				}
			}
		}
		
		auto ready = std::min(distance_buffer->num_entries(), label_buffer->num_entries());

		if (ready >= 1) {
			queries_sent += ready * cfg.block_size;
			
			void* label_ptr = label_buffer->front();
			void* dist_ptr = distance_buffer->front();
			
			//TODO: Optimize this to an Immediate Synchronous Send
			MPI_Ssend(label_ptr, cfg.k * cfg.block_size * ready, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
			MPI_Ssend(dist_ptr, cfg.k * cfg.block_size * ready, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);
			
			label_buffer->remove(ready);
			distance_buffer->remove(ready);
		}
	}

	MPI_Ssend(&dummy, 1, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
	deb("Finished receiving queries");	
}

static void comm_handler_both(int blocks_gpu, SyncBuffer* cpu_buffer, SyncBuffer* gpu_buffer, SyncBuffer* cpu_distance_buffer, SyncBuffer* cpu_label_buffer, SyncBuffer* gpu_distance_buffer, SyncBuffer* gpu_label_buffer, bool& finished_receiving, Config& cfg) {
	deb("Receiver");
	
	byte tmp_buffer[cfg.block_size * cfg.d * sizeof(float)];
	
	long queries_received = 0;
	long queries_sent = 0;
	int blocks_until_cpu = blocks_gpu;
	
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
				} else if (blocks_until_cpu >= 1) {
					MPI_Recv(tmp_buffer, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
					gpu_buffer->insert(1, tmp_buffer);
					queries_received += cfg.block_size;
					
					blocks_until_cpu--;
					
					assert(status.MPI_ERROR == MPI_SUCCESS);
				} else {
					MPI_Recv(tmp_buffer, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
					cpu_buffer->insert(1, tmp_buffer);
					queries_received += cfg.block_size;

					blocks_until_cpu = blocks_gpu;
					
					assert(status.MPI_ERROR == MPI_SUCCESS);
				}
			}
		}
		
		auto readyGPU = std::min(gpu_distance_buffer->num_entries(), gpu_label_buffer->num_entries());

		if (readyGPU >= 1) {
			queries_sent += readyGPU * cfg.block_size;
			
			void* label_ptr = gpu_label_buffer->front();
			void* dist_ptr = gpu_distance_buffer->front();
			
			//TODO: Optimize this to an Immediate Synchronous Send
			MPI_Ssend(label_ptr, cfg.k * cfg.block_size * readyGPU, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
			MPI_Ssend(dist_ptr, cfg.k * cfg.block_size * readyGPU, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);
			
			gpu_label_buffer->remove(readyGPU);
			gpu_distance_buffer->remove(readyGPU);
		}
		
		auto readyCPU = std::min(cpu_distance_buffer->num_entries(), cpu_label_buffer->num_entries());

		if (readyCPU >= 1) {
			queries_sent += readyCPU * cfg.block_size;

			void* label_ptr = cpu_label_buffer->front();
			void* dist_ptr = cpu_distance_buffer->front();

			//TODO: Optimize this to an Immediate Synchronous Send
			MPI_Ssend(label_ptr, cfg.k * cfg.block_size * readyCPU, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
			MPI_Ssend(dist_ptr, cfg.k * cfg.block_size * readyCPU, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);

			cpu_label_buffer->remove(readyCPU);
			cpu_distance_buffer->remove(readyCPU);
		}
	}

	MPI_Ssend(&dummy, 1, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
	deb("Finished receiving queries");	
}

static void comm_handler_out(std::vector<SyncBuffer*>& query_buffers, SyncBuffer* distance_buffer, SyncBuffer* label_buffer, bool& finished_receiving, Config& cfg) {
	deb("Receiver");
	
	byte tmp_buffer[cfg.block_size * cfg.d * sizeof(float)];
	
	long queries_received = 0;
	long queries_sent = 0;
	
	float dummy;
	MPI_Send(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD); //signal that we are ready to receive queries

	deb("Now waiting for queries");

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
					MPI_Recv(tmp_buffer, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
					
					for (auto buffer : query_buffers) {
						buffer->insert(1, tmp_buffer);
					}
					
					queries_received += cfg.block_size;
					assert(status.MPI_ERROR == MPI_SUCCESS);
				}
			}
		}
		
		auto ready = std::min(distance_buffer->num_entries(), label_buffer->num_entries());

		if (ready >= 1) {
			queries_sent += ready * cfg.block_size;
			
			void* label_ptr = label_buffer->front();
			void* dist_ptr = distance_buffer->front();
			
			//TODO: Optimize this to an Immediate Synchronous Send
			MPI_Ssend(label_ptr, cfg.k * cfg.block_size * ready, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
			MPI_Ssend(dist_ptr, cfg.k * cfg.block_size * ready, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);
			
			label_buffer->remove(ready);
			distance_buffer->remove(ready);
		}
	}

	MPI_Ssend(&dummy, 1, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
	deb("Finished receiving queries");	
}

static void main_driver(bool& finished, SyncBuffer* query_buffer, SyncBuffer* label_buffer, SyncBuffer* distance_buffer, ExecPolicy* policy, long blocks_to_be_processed, faiss::Index* cpu_index, faiss::Index* gpu_index) {
	deb("Starting to process queries");
	
	faiss::Index::idx_t* I = new faiss::Index::idx_t[cfg.num_blocks * cfg.block_size * cfg.k];
	float* D = new float[cfg.num_blocks * cfg.block_size * cfg.k];
	int processed = 0;
	
	while (! finished || query_buffer->num_entries() >= 1) {
		long num_blocks = policy->numBlocksRequired(*query_buffer, cfg);
		num_blocks = std::min(num_blocks, blocks_to_be_processed);
		
		if (num_blocks == 0) {
			auto sleep_time_us = std::min(query_buffer->arrivalInterval() * 1000000, 1000.0);
			usleep(sleep_time_us);
			continue;
		}
		
		query_buffer->waitForData(num_blocks);

		blocks_to_be_processed -= num_blocks;
		int nqueries = num_blocks * cfg.block_size;

		policy->process_buffer(cpu_index, gpu_index, nqueries, *query_buffer, I, D);
		processed += nqueries;
		
		deb("Processed %d queries. %d already processed. %d left to be processed",  nqueries, processed, blocks_to_be_processed);

		label_buffer->insert(num_blocks, (byte*) I);
		distance_buffer->insert(num_blocks, (byte*) D);
	}
	
	delete[] I;
	delete[] D;
	
	policy->cleanup(cfg);
}

void search_both(int shard, ExecPolicy* cpu_policy, ExecPolicy* gpu_policy, long num_blocks, double gpu_throughput, double cpu_throughput) {
	const long block_size_in_bytes = sizeof(float) * cfg.d * cfg.block_size;
	const long distance_block_size_in_bytes = sizeof(float) * cfg.k * cfg.block_size;
	const long label_block_size_in_bytes = sizeof(faiss::Index::idx_t) * cfg.k * cfg.block_size;

	SyncBuffer cpu_query_buffer(block_size_in_bytes, 500 * 1024 * 1024 / block_size_in_bytes); //500MB
	SyncBuffer cpu_distance_buffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	SyncBuffer cpu_label_buffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 

	SyncBuffer gpu_query_buffer(block_size_in_bytes, 500 * 1024 * 1024 / block_size_in_bytes); //500MB
	SyncBuffer gpu_distance_buffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	SyncBuffer gpu_label_buffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 

	
	faiss::Index* cpu_index = load_index(0, 1, cfg);
	
	faiss::gpu::StandardGpuResources res;
	if (cfg.temp_memory_gpu > 0) res.setTempMemory(cfg.temp_memory_gpu);
	faiss::Index* gpu_index = faiss::gpu::index_cpu_to_gpu(&res, shard % cfg.gpus_per_node, cpu_index, nullptr);
	 
	long gpu_blocks_per_cpu_block = std::nearbyint(gpu_throughput / cpu_throughput);
	long blocks_cpu = num_blocks / (gpu_blocks_per_cpu_block + 1);
	long blocks_gpu = num_blocks - blocks_cpu;
	
	cpu_policy->setup();
	gpu_policy->setup();
	
	bool finished = false;
	std::thread receiver { comm_handler_both, gpu_blocks_per_cpu_block, &cpu_query_buffer, &gpu_query_buffer, &cpu_distance_buffer, &cpu_label_buffer, &gpu_distance_buffer, &gpu_label_buffer, std::ref(finished), std::ref(cfg) };	

	std::thread gpu_thread { main_driver, std::ref(finished), &gpu_query_buffer, &gpu_label_buffer, &gpu_distance_buffer, gpu_policy, blocks_gpu, cpu_index, gpu_index };
	std::thread cpu_thread { main_driver, std::ref(finished), &cpu_query_buffer, &cpu_label_buffer, &cpu_distance_buffer, cpu_policy, blocks_cpu, cpu_index, gpu_index };
	
	receiver.join();
	gpu_thread.join();
	cpu_thread.join();
}

void search_single(int shard, ExecPolicy* policy, long num_blocks) {
	const long block_size_in_bytes = sizeof(float) * cfg.d * cfg.block_size;
	const long distance_block_size_in_bytes = sizeof(float) * cfg.k * cfg.block_size;
	const long label_block_size_in_bytes = sizeof(faiss::Index::idx_t) * cfg.k * cfg.block_size;

	SyncBuffer query_buffer(block_size_in_bytes, 500 * 1024 * 1024 / block_size_in_bytes); //500MB
	SyncBuffer distance_buffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	SyncBuffer label_buffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 

	faiss::Index* cpu_index = load_index(0, 1, cfg);
	faiss::Index* gpu_index = nullptr;
		

	faiss::gpu::StandardGpuResources* res = nullptr;
	
	if (policy->usesGPU()) {
		deb("Transfering base to the GPU. Shard=%d", shard);
		
		res = new faiss::gpu::StandardGpuResources();
		if (cfg.temp_memory_gpu > 0) res->setTempMemory(cfg.temp_memory_gpu);
		gpu_index = faiss::gpu::index_cpu_to_gpu(res, shard % cfg.gpus_per_node, cpu_index, nullptr);
		deb("Base transferred to the GPU");
	} 

	policy->setup();
	
	bool finished = false;
	std::thread receiver { comm_handler_single, &query_buffer, &distance_buffer, &label_buffer, std::ref(finished), std::ref(cfg) };
	
	main_driver(finished, &query_buffer, &label_buffer, &distance_buffer, policy, num_blocks, cpu_index, gpu_index);

	receiver.join();
}

void search_out(int shard, SearchAlgorithm search_algorithm) {
	deb("search called");

	SearchStrategy* strategy;

	float base_start = 0;
	float base_end = 1;

	faiss::gpu::StandardGpuResources res;
	if (cfg.temp_memory_gpu > 0) res.setTempMemory(cfg.temp_memory_gpu);
	
	if (search_algorithm == SearchAlgorithm::Cpu) {
		strategy = new CpuOnlySearchStrategy(1, base_start, base_end);
	} else if (search_algorithm == SearchAlgorithm::Hybrid) {
		strategy = new HybridSearchStrategy(cfg.total_pieces, base_start, base_end, &res);
	} else if (search_algorithm == SearchAlgorithm::Gpu) {
		strategy = new GpuOnlySearchStrategy(cfg.gpu_pieces, base_start, base_end, &res);
	} else if (search_algorithm == SearchAlgorithm::Fixed) {
		strategy = new FixedSearchStrategy(2, base_start, base_end, &res);
	}

	strategy->setup();

	bool finished = false;
	std::thread receiver { comm_handler_out, std::ref(strategy->queryBuffers()), strategy->distanceBuffer(), strategy->labelBuffer(), std::ref(finished), std::ref(cfg) };

	strategy->start_search_process();

	receiver.join();
}
