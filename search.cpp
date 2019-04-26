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

static void query_receiver(Buffer* buffer, bool* finished, Config& cfg) {
	deb("Receiver");
	int queries_received = 0;
	
	float dummy;
	MPI_Send(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD); //signal that we are ready to receive queries

	while (true) {
		MPI_Status status;
		MPI_Probe(GENERATOR, 0, MPI_COMM_WORLD, &status);
		
		int message_size;
		MPI_Get_count(&status, MPI_FLOAT, &message_size);
		
		if (message_size == 1) {
			*finished = true;
			float dummy;
			MPI_Recv(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
			break;
		}

		buffer->waitForSpace(1);

		float* recv_data = reinterpret_cast<float*>(buffer->peekEnd());
		MPI_Recv(recv_data, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
		
		buffer->add(1);

		assert(status.MPI_ERROR == MPI_SUCCESS);

		queries_received += cfg.block_size;
	}
	
	deb("Finished receiving queries");
}

static faiss::Index* load_index(int shard, int nshards, Config& cfg) {
	deb("Started loading");

	char index_path[500];
	sprintf(index_path, "index/index_%d_%d_%d", cfg.nb, cfg.ncentroids, cfg.m);

	deb("Loading file: %s", index_path);

	FILE* index_file = fopen(index_path, "r");
	auto cpu_index = read_index(index_file, shard, nshards);

	deb("Ended loading");

	//TODO: Maybe we shouldnt set nprobe here
	dynamic_cast<faiss::IndexIVFPQ*>(cpu_index)->nprobe = cfg.nprobe;
	return cpu_index;
}

//TODO: currently, we use the same files in all nodes. This doesn't work with ProfileData, since each machine is different. Need to create a way for each node to load a different file.
namespace {
	struct ProfileData {
		double* times;
		int min_block;
		int max_block;
	};
}

static ProfileData getProfilingData(Config& cfg) {	
	char file_path[100];
	sprintf(file_path, "prof/%d_%d_%d_%d_%d_%d", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
	std::ifstream file;
	file.open(file_path);
	
	if (! file.good()) {
		std::printf("File prof/%d_%d_%d_%d_%d_%d", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	int total_size;
	file >> total_size;
	
	double times[total_size + 1];
	times[0] = 0;
	
	for (int i = 1; i <= total_size; i++) {
		file >> times[i];
	}

	file.close();
	
	ProfileData pd;
	double time_per_block[total_size + 1];
	time_per_block[0] = 0;

	for (int i = 1; i <= total_size; i++) {
		time_per_block[i] = times[i] / i;
	}

	double tolerance = 0.1;
	int minBlock = 1;

	for (int nb = 1; nb <= total_size; nb++) {
		if (time_per_block[nb] < time_per_block[minBlock]) minBlock = nb;
	}

	double threshold = time_per_block[minBlock] * (1 + tolerance);

	int nb = 1;
	while (time_per_block[nb] > threshold) nb++;
	pd.min_block = nb;

	nb = total_size;
	while (time_per_block[nb] > threshold) nb--;
	pd.max_block = nb;

	pd.times = new double[minBlock + 1];
	pd.times[0] = 0;
	for (int nb = 1; nb <= minBlock; nb++) pd.times[nb] = time_per_block[nb];
	
	assert(pd.max_block * cfg.block_size != BENCH_SIZE);
	
	return pd;
}


static void store_profile_data(std::vector<double>& procTimes, Config& cfg) {
	//now we write the time data on a file
	char file_path[100];
	sprintf(file_path, "prof/%d_%d_%d_%d_%d_%d", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
	std::ofstream file;
	file.open(file_path);

	int blocks = procTimes.size() / bench_repeats;
	file << blocks << std::endl;

	int ptr = 0;

	for (int b = 1; b <= blocks; b++) {
		std::vector<double> times;

		for (int repeats = 1; repeats <= bench_repeats; repeats++) {
			times.push_back(procTimes[ptr++]);
		}

		std::sort(times.begin(), times.end());

		int mid = bench_repeats / 2;
		file << times[mid] << std::endl;
	}

	file.close();
}

static int numBlocksRequired(ProcType ptype, Buffer& buffer, ProfileData& pdGPU, Config& cfg) {
	switch (ptype) {
		case ProcType::Dynamic: {
			buffer.waitForData(1);
			
			int num_blocks = buffer.entries();
			
			if (num_blocks < pdGPU.min_block) {
				double work_without_waiting = num_blocks / pdGPU.times[num_blocks];
				double work_with_waiting = pdGPU.min_block / ((pdGPU.min_block - num_blocks) * buffer.block_rate() + pdGPU.times[pdGPU.min_block]);

				if (work_with_waiting > work_without_waiting) {
					usleep((pdGPU.min_block - num_blocks) * buffer.block_rate() * 1000000);
				}
			}

			return std::min(buffer.entries(), pdGPU.max_block);
		}
		case ProcType::Static: {
			return cfg.processing_size;
		}
		case ProcType::Bench: {
			static int nrepeats = 0;
			static int nb = 1;

			if (nrepeats >= bench_repeats) {
				nrepeats = 0;
				nb++;
			}

			nrepeats++;
			return nb;
		}
	}
	
	return 0;
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
	
	ProfileData pdGPU; 
	if (ptype == ProcType::Dynamic) pdGPU = getProfilingData(cfg);

	const long block_size_in_bytes = sizeof(float) * cfg.d * cfg.block_size;
	Buffer buffer(block_size_in_bytes, 100 * 1024 * 1024 / block_size_in_bytes); //100 MB
	
	faiss::gpu::StandardGpuResources res;
	res.setTempMemory(1250 * 1024 * 1024);

	auto cpu_index = load_index(shard, nshards, cfg);
	auto gpu_index = faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index, nullptr);
	
	faiss::Index::idx_t* I = new faiss::Index::idx_t[cfg.test_length * cfg.k];
	float* D = new float[cfg.test_length * cfg.k];
	
	deb("Search node is ready");
	
	assert(cfg.test_length % cfg.block_size == 0);

	int qn = 0;
	bool finished = false;
	
	std::thread receiver { query_receiver, &buffer, &finished, std::ref(cfg) };
	
	while (! finished || buffer.entries() >= 1) {
		int remaining_blocks = (cfg.test_length - qn) / cfg.block_size;
		int num_blocks = numBlocksRequired(ptype, buffer, pdGPU, cfg);
		if (ptype != ProcType::Bench) num_blocks = std::min(num_blocks, remaining_blocks);
		buffer.waitForData(num_blocks);

		int nqueries = num_blocks * cfg.block_size;
		
		deb("Processing %d queries", nqueries);
		process_buffer(ptype, gpu_index, nqueries, buffer, I, D, procTimesGpu, cfg); 
		
		//TODO: Optimize this to an Immediate Synchronous Send
		//TODO: Merge these two sends into one
		MPI_Ssend(I, cfg.k * nqueries, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
		MPI_Ssend(D, cfg.k * nqueries, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);
		
		qn += nqueries;
	}
	
	long dummy = 1;
	MPI_Ssend(&dummy, 1, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
	
	receiver.join();

	delete[] I;
	delete[] D;
	
	if (ptype == ProcType::Bench) {
		store_profile_data(procTimesGpu, cfg);
	}

	deb("Finished search node");
}
