#include "search.h"

#include <mpi.h>
#include <algorithm>
#include <future>
#include <fstream>
#include <unistd.h>
 
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

static void comm_handler(Buffer* query_buffer, Buffer* distance_buffer, Buffer* label_buffer, bool& finished_receiving, Config& cfg) {
	deb("Receiver");
	
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

static void store_profile_data(std::vector<double>& procTimes, int shard_number, Config& cfg) {
	char checkup_command[100];
	sprintf(checkup_command, "mkdir -p %s", PROF_ROOT);
	system(checkup_command); //to make sure that the "prof" dir exists
	
	//now we write the time data on a file
	char file_path[100];
	sprintf(file_path, "%s/%d_%d_%d_%d_%d_%d_%d", PROF_ROOT, cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size, shard_number);
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

static void process_buffer(ProcType ptype, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D, std::vector<double>& procTimesGpu, Config& cfg) {
	float* query_buffer = reinterpret_cast<float*>(buffer.peekFront());
	
	//now we proccess our query buffer
	if (ptype == ProcType::Bench) {
		static bool finished = false;
		
		if (! finished) {
			auto before = now();
			gpu_index->search(nq, query_buffer, cfg.k, D, I);
			auto time_spent = now() - before;

			procTimesGpu.push_back(time_spent);
			finished = time_spent >= 1;
		}
	} else gpu_index->search(nq, query_buffer, cfg.k, D, I);

	buffer.consume(nq / cfg.block_size);
}

void search(int shard, int nshards, ProcType ptype, Config& cfg) {
	std::vector<double> procTimesGpu;
	
	cfg.exec_policy->setup();

	const long block_size_in_bytes = sizeof(float) * cfg.d * cfg.block_size;
	Buffer query_buffer(block_size_in_bytes, 100 * 1024 * 1024 / block_size_in_bytes); //100 MB
	
	const long distance_block_size_in_bytes = sizeof(float) * cfg.k * cfg.block_size;
	const long label_block_size_in_bytes = sizeof(faiss::Index::idx_t) * cfg.k * cfg.block_size;
	Buffer distance_buffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	Buffer label_buffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 
	
	faiss::gpu::StandardGpuResources res;
	res.setTempMemory(1500 * 1024 * 1024);

	auto cpu_index = load_index(shard, nshards, cfg);
	auto gpu_index = faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index, nullptr);
	
	faiss::Index::idx_t* I = new faiss::Index::idx_t[cfg.eval_length * cfg.k];
	float* D = new float[cfg.eval_length * cfg.k];
	
	deb("Search node is ready");
	
	assert(cfg.test_length % cfg.block_size == 0);

	int qn = 0;
	bool finished = false;
	
	std::thread receiver { comm_handler, &query_buffer, &distance_buffer, &label_buffer, std::ref(finished), std::ref(cfg) };
	
	while (! finished || query_buffer.entries() >= 1) {
		int remaining_blocks = (cfg.test_length - qn) / cfg.block_size;
		int num_blocks = cfg.exec_policy->numBlocksRequired(ptype, query_buffer, cfg);
		if (ptype != ProcType::Bench) num_blocks = std::min(num_blocks, remaining_blocks);
		query_buffer.waitForData(num_blocks);

		int nqueries = num_blocks * cfg.block_size;
		
		deb("Processing %d queries", nqueries);
		process_buffer(ptype, gpu_index, nqueries, query_buffer, I, D, procTimesGpu, cfg); 
		
		label_buffer.transfer(I, num_blocks);
		distance_buffer.transfer(D, num_blocks);

		qn += nqueries;
	}
	
	receiver.join();

	delete[] I;
	delete[] D;
	
	if (ptype == ProcType::Bench) {
		store_profile_data(procTimesGpu, shard, cfg);
	}

	deb("Finished search node");
}
