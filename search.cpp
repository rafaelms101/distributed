#include "search.h"

#include <mpi.h>
#include <algorithm>
#include <future>
#include <fstream>
#include <unistd.h>
 
#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

#include "../faiss/gpu/GpuAutoTune.h"
#include "../faiss/gpu/StandardGpuResources.h"
#include "../faiss/gpu/GpuIndexIVFPQ.h"

#include "utils.h"
#include "Buffer.h"
#include "readSplittedIndex.h"
#include "ExecPolicy.h"

static void comm_handler(int blocks_gpu, Buffer* cpu_buffer, Buffer* gpu_buffer, Buffer* cpu_distance_buffer, Buffer* cpu_label_buffer, Buffer* gpu_distance_buffer, Buffer* gpu_label_buffer, bool& finished_receiving, Config& cfg) {
	deb("Receiver");
	
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
					gpu_buffer->waitForSpace(1);
					float* recv_data = reinterpret_cast<float*>(gpu_buffer->peekEnd());
					MPI_Recv(recv_data, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
					gpu_buffer->add(1);
					queries_received += cfg.block_size;
					
					blocks_until_cpu--;
					
					assert(status.MPI_ERROR == MPI_SUCCESS);
				} else {
					cpu_buffer->waitForSpace(1);
					float* recv_data = reinterpret_cast<float*>(cpu_buffer->peekEnd());
					MPI_Recv(recv_data, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
					cpu_buffer->add(1);
					queries_received += cfg.block_size;

					blocks_until_cpu = blocks_gpu;
					
					assert(status.MPI_ERROR == MPI_SUCCESS);
				}
			}
		}
		
		auto readyGPU = std::min(gpu_distance_buffer->entries(), gpu_label_buffer->entries());

		if (readyGPU >= 1) {
			queries_sent += readyGPU * cfg.block_size;
			
			faiss::Index::idx_t* label_ptr = reinterpret_cast<faiss::Index::idx_t*>(gpu_label_buffer->peekFront());
			float* dist_ptr = reinterpret_cast<float*>(gpu_distance_buffer->peekFront());
			
			//TODO: Optimize this to an Immediate Synchronous Send
			MPI_Ssend(label_ptr, cfg.k * cfg.block_size * readyGPU, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
			MPI_Ssend(dist_ptr, cfg.k * cfg.block_size * readyGPU, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);
			
			gpu_label_buffer->consume(readyGPU);
			gpu_distance_buffer->consume(readyGPU);
		}
		
		auto readyCPU = std::min(cpu_distance_buffer->entries(), cpu_label_buffer->entries());

		if (readyCPU >= 1) {
			queries_sent += readyCPU * cfg.block_size;

			faiss::Index::idx_t* label_ptr = reinterpret_cast<faiss::Index::idx_t*>(cpu_label_buffer->peekFront());
			float* dist_ptr = reinterpret_cast<float*>(cpu_distance_buffer->peekFront());

			//TODO: Optimize this to an Immediate Synchronous Send
			MPI_Ssend(label_ptr, cfg.k * cfg.block_size * readyCPU, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
			MPI_Ssend(dist_ptr, cfg.k * cfg.block_size * readyCPU, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);

			cpu_label_buffer->consume(readyCPU);
			cpu_distance_buffer->consume(readyCPU);
		}
	}

	MPI_Ssend(&dummy, 1, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
	deb("Finished receiving queries");	
}

static faiss::Index* load_index(int shard, int nshards, Config& cfg) {
	deb("Started loading");
	
	char index_path[500];
	sprintf(index_path, "%s/index_%d_%d_%d", INDEX_ROOT, cfg.nb, cfg.ncentroids, cfg.m);

	deb("Loading file: %s", index_path);

	FILE* index_file = fopen(index_path, "r");
	auto cpu_index = read_index(index_file, shard, nshards);

	deb("Ended loading");

	//TODO: Maybe we shouldnt set nprobe here
	dynamic_cast<faiss::IndexIVFPQ*>(cpu_index)->nprobe = cfg.nprobe;
	return cpu_index;
}

static void store_profile_data(bool gpu, std::vector<double>& procTimes, int shard_number, Config& cfg) {
	char checkup_command[100];
	sprintf(checkup_command, "mkdir -p %s", PROF_ROOT);
	system(checkup_command); //to make sure that the "prof" dir exists
	
	//now we write the time data on a file
	char file_path[100];
	sprintf(file_path, "%s/%s_%d_%d_%d_%d_%d_%d_%d", gpu ? "gpu" : "cpu", PROF_ROOT, cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size, shard_number);
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

static void main_driver(bool& finished, Buffer* query_buffer, Buffer* label_buffer, Buffer* distance_buffer, ExecPolicy* policy, int blocks_to_be_processed, faiss::Index* cpu_index, faiss::Index* gpu_index) {
	faiss::Index::idx_t* I = new faiss::Index::idx_t[cfg.eval_length * cfg.k];
	float* D = new float[cfg.eval_length * cfg.k];

	while (! finished || query_buffer->entries() >= 1) {
		int num_blocks = policy->numBlocksRequired(*query_buffer, cfg);
		num_blocks = std::min(num_blocks, blocks_to_be_processed);
		query_buffer->waitForData(num_blocks);

		blocks_to_be_processed -= num_blocks;
		int nqueries = num_blocks * cfg.block_size;

		deb("Processing %d queries", nqueries);
		policy->process_buffer(cpu_index, gpu_index, nqueries, *query_buffer, I, D);

		label_buffer->transfer(I, num_blocks);
		distance_buffer->transfer(D, num_blocks);
	}
	
	delete[] I;
	delete[] D;
	
	policy->cleanup(cfg);
}

//comm_handler(int blocks_gpu, Buffer* cpu_buffer, Buffer* gpu_buffer, Buffer* cpu_distance_buffer, Buffer* cpu_label_buffer, Buffer* gpu_distance_buffer, Buffer* gpu_label_buffer, bool& finished_receiving, Config& cfg) {
void search(int blocks_gpu, ProcType ptype, int shard, Config& cfg) {
	std::vector<double> procTimesGpu;
	
	cfg.gpu_exec_policy->setup();
	cfg.cpu_exec_policy->setup();

	const long block_size_in_bytes = sizeof(float) * cfg.d * cfg.block_size;
	const long distance_block_size_in_bytes = sizeof(float) * cfg.k * cfg.block_size;
	const long label_block_size_in_bytes = sizeof(faiss::Index::idx_t) * cfg.k * cfg.block_size;
	
	Buffer cpu_query_buffer(block_size_in_bytes, 500 * 1024 * 1024 / block_size_in_bytes); //500MB
	Buffer cpu_distance_buffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	Buffer cpu_label_buffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 
	
	Buffer gpu_query_buffer(block_size_in_bytes, 500 * 1024 * 1024 / block_size_in_bytes); //500MB
	Buffer gpu_distance_buffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	Buffer gpu_label_buffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 
	
	faiss::gpu::StandardGpuResources res;
	res.setTempMemory(cfg.temp_memory_gpu);

	auto cpu_index = load_index(0, 1, cfg);
	auto gpu_index = faiss::gpu::index_cpu_to_gpu(&res, shard % cfg.gpus_per_node, cpu_index, nullptr);

	deb("Search node is ready");
	
	assert(cfg.test_length % cfg.block_size == 0);
	
	bool finished = false;
	
	int size_total = cfg.test_length / cfg.block_size;
 	int size_cpu = size_total / (blocks_gpu + 1);
	int size_gpu = size_total - size_cpu;
	
	std::thread receiver { comm_handler, blocks_gpu, &cpu_query_buffer, &gpu_query_buffer, &cpu_distance_buffer, &cpu_label_buffer, &gpu_distance_buffer, &gpu_label_buffer, std::ref(finished), std::ref(cfg) };
	std::thread gpu_thread { main_driver, std::ref(finished), &gpu_query_buffer, &gpu_label_buffer, &gpu_distance_buffer, cfg.gpu_exec_policy, size_gpu, nullptr, gpu_index };
	std::thread cpu_thread { main_driver, std::ref(finished), &cpu_query_buffer, &cpu_label_buffer, &cpu_distance_buffer, cfg.cpu_exec_policy, size_cpu, cpu_index, nullptr  };

	receiver.join();
	cpu_thread.join();
	gpu_thread.join();

	deb("Finished search node");
}
