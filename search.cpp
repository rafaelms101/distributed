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

static inline double _search(faiss::Index* index, int nq, float* queries, float* D, faiss::Index::idx_t* I, Config& cfg) {
	double before = now();
	if (nq > 0) index->search(nq, queries, cfg.k, D, I);
	double after = now();
	
	return after - before;
}

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

static ProfileData getProfilingData(bool cpu, Config& cfg) {	
	char file_path[100];
	sprintf(file_path, "prof/%d_%d_%d_%d_%d_%d_%s", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size, cpu ? "cpu" : "gpu");
	std::ifstream file;
	file.open(file_path);
	
	if (! file.good()) {
		std::printf("File prof/%d_%d_%d_%d_%d_%d_%s", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size, cpu ? "cpu" : "gpu");
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
	
	if  (cpu) {
		pd.times = new double[total_size + 1];
		pd.times[0] = 0;
		for (int nb = 1; nb <= total_size; nb++) pd.times[nb] = times[nb];
		pd.max_block = total_size;
		pd.min_block = 0;
	} else {
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
		while (time_per_block[nb] > threshold)
			nb--;
		pd.max_block = nb;
		
		pd.times = new double[minBlock + 1];
		pd.times[0] = 0;
		for (int nb = 1; nb <= minBlock; nb++) pd.times[nb] = time_per_block[nb];
	}
	
	return pd;
}


static void store_profile_data(const char* type, std::vector<double>& procTimes, Config& cfg) {
	//now we write the time data on a file
	char file_path[100];
	sprintf(file_path, "prof/%d_%d_%d_%d_%d_%d_%s", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size, type);
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

static int bsearch(int min, int max, double* data, double val) {
	while (max - min >= 2) {
		int mid = (min + max) / 2;

		if (data[mid] > val) {
			max = mid - 1;
		} else if (data[mid] < val) {
			min = mid;
		} else return mid;
	}
	
	if (data[max] <= val) return max;
	else return min;
}

static int numBlocksRequired(ProcType ptype, Buffer& buffer, ProfileData& pdGPU, Config& cfg) {
	switch (ptype) {
		case ProcType::Dynamic: {
			buffer.waitForData(1);
			
			int num_blocks = buffer.entries();
			
			if (num_blocks < pdGPU.min_block) {
				double work_without_waiting = num_blocks / pdGPU.times[num_blocks];
				double work_with_waiting = (pdGPU.min_block - num_blocks) * buffer.block_rate() + pdGPU.times[pdGPU.min_block];

				if (work_with_waiting > work_without_waiting) {
					usleep((pdGPU.min_block - num_blocks) * buffer.block_rate() * 1000000);
				}
			}

			return num_blocks;
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
			return nb * 2;
		}
	}
}

static std::pair<int, int> computeSplitCPU_GPU(ProcType ptype, Buffer& buffer, int num_blocks, ProfileData& pdGPU, ProfileData& pdCPU, Config& cfg) {
	int nq_cpu, nq_gpu;
	
	switch (ptype) {
		case ProcType::Dynamic: {
			int bgpu = std::min(num_blocks, pdGPU.max_block);
			int bcpu = std::min(num_blocks - bgpu, pdCPU.max_block);

			if (bcpu > 0) bcpu = bsearch(0, bcpu, pdCPU.times, pdGPU.times[pdGPU.max_block]);
			nq_cpu = bcpu * cfg.block_size;
			
			if (nq_cpu != 0) deb("%d CPU queries", nq_cpu);
			
			nq_gpu = bgpu * cfg.block_size;

			break;
		}
		case ProcType::Static: {
			int nq = num_blocks * cfg.block_size;
			nq_gpu = static_cast<int>(nq * cfg.gpu_slice);
			nq_cpu = nq - nq_gpu;
			
			break;
		}
		case ProcType::Bench: {
			nq_cpu = num_blocks / 2 * cfg.block_size;
			nq_gpu = nq_cpu;

			break;
		}
	}
	
	return std::make_pair(nq_cpu, nq_gpu);
}

static void process_buffer(ProcType ptype, faiss::Index* cpu_index, faiss::Index* gpu_index, int nq_cpu, int nq_gpu, Buffer& buffer, faiss::Index::idx_t* I, float* D, std::vector<double>& procTimesGpu, std::vector<double>& procTimesCpu, Config& cfg) {
	float* query_buffer = reinterpret_cast<float*>(buffer.peekFront());
	
	//now we proccess our query buffer
	if (ptype == ProcType::Bench) {
		static bool gpu_finished = false;
		static bool cpu_finished = false;

		std::future<double> future_time_spent;
		if (! gpu_finished) {
			future_time_spent = std::async(std::launch::async, _search, gpu_index, nq_gpu, query_buffer + nq_cpu * cfg.d, D + cfg.k * nq_cpu, I + cfg.k * nq_cpu, std::ref(cfg));
		}

		if (! cpu_finished) {
			auto time_spent = _search(cpu_index, nq_cpu, query_buffer, D, I, cfg);
			procTimesCpu.push_back(time_spent);

			if (time_spent >= 1) {
				cpu_finished = true;
				deb("Stopped benching CPU at %d", nq_cpu);
			}
		}

		if (! gpu_finished) {
			auto time_spent = future_time_spent.get();
			procTimesGpu.push_back(time_spent);

			if (time_spent >= 1) {
				gpu_finished = true;
				deb("Stopped benching GPU at %d", nq_gpu);
			}
		}
	} else {
		auto future = std::async(std::launch::async, _search, gpu_index, nq_gpu, query_buffer + nq_cpu * cfg.d, D + cfg.k * nq_cpu, I + cfg.k * nq_cpu, std::ref(cfg));
		_search(cpu_index, nq_cpu, query_buffer, D, I, cfg);
		future.get();
	}

	int nqueries = nq_cpu + nq_gpu;
	buffer.consume(nqueries / cfg.block_size);
			
}

void search(int shard, int nshards, ProcType ptype, Config& cfg) {
	std::vector<double> procTimesGpu;
	std::vector<double> procTimesCpu;
	
	ProfileData pdGPU, pdCPU; 
	if (ptype == ProcType::Dynamic) {
		pdCPU = getProfilingData(true, cfg);
		pdGPU = getProfilingData(false, cfg);
	}

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
		int num_blocks = numBlocksRequired(ptype, buffer, pdGPU, cfg);
		
		std::pair<int, int> p = computeSplitCPU_GPU(ptype, buffer, num_blocks, pdGPU, pdCPU, cfg);
		auto nq_cpu = p.first;
		auto nq_gpu = p.second;
		
		int remaining_blocks = (cfg.test_length - qn) / 20;
		buffer.waitForData(std::min(num_blocks, remaining_blocks));
		
		deb("Processing %d queries", nq_cpu + nq_gpu);
		process_buffer(ptype, cpu_index, gpu_index, nq_cpu, nq_gpu, buffer, I, D, procTimesGpu, procTimesCpu, cfg); 

		int nqueries = nq_cpu + nq_gpu;
		
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
		store_profile_data("cpu", procTimesCpu, cfg);
		store_profile_data("gpu", procTimesGpu, cfg);
	}

	deb("Finished search node");
}
