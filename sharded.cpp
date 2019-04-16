#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <assert.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <limits>
#include <condition_variable>
#include <algorithm>
#include <future>
#include <fstream>
#include <utility>
#include <tuple>
#include <unistd.h>
#include <functional>
#include <unordered_map>
#include <unistd.h>
#include <thread>   

#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

#include "gpu/GpuAutoTune.h"
#include "gpu/StandardGpuResources.h"
#include "gpu/GpuIndexIVFPQ.h"

#include "readSplittedIndex.h"
#include "QueryBuffer.h"
#include "generator.h"
#include "utils.h"

#define AGGREGATOR 0
#define GENERATOR 1

void aggregator(int nshards, ProcType ptype, Config& cfg);
void search(int shard, int nshards, ProcType ptype, Config& cfg);

ProcType handle_parameters(int argc, char* argv[], Config& cfg) {
	std::string usage = "./sharded b | d <qr> | s <qr> <num_blocks> <gpu_slice>";

	if (argc < 2) {
		std::printf("Wrong arguments.\n%s\n", usage.c_str());
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	ProcType ptype;
	if (!strcmp("d", argv[1])) ptype = ProcType::Dynamic;
	else if (!strcmp("b", argv[1])) ptype = ProcType::Bench;
	else if (!strcmp("s", argv[1])) ptype = ProcType::Static;
	else {
		std::printf("Invalid processing type.Expected b | s | d\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	if (ptype == ProcType::Dynamic) {
		if (argc != 3) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			MPI_Abort(MPI_COMM_WORLD, -1);
		}

		cfg.query_rate = atof(argv[2]);
	} else if (ptype == ProcType::Static) {
		if (argc != 5) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			MPI_Abort(MPI_COMM_WORLD, -1);
		}

		cfg.query_rate = atof(argv[2]);
		cfg.processing_size = atoi(argv[3]);
		cfg.gpu_slice = atof(argv[4]);

		assert(cfg.gpu_slice >= 0 && cfg.gpu_slice <= 1);
	} else if (ptype == ProcType::Bench) {
		//nothing
	}

	return ptype;
}

Config inline loadConfig() {
	Config cfg; 
	cfg.d = 128;
	cfg.nb = 500000000;
	cfg.ncentroids = 8192;
	cfg.m = 8;
	cfg.k = 100;
	cfg.nprobe = 8;
	cfg.block_size = 20;
	cfg.test_length = 100000;
	cfg.eval_length = 10000;
	cfg.nq = 10000;
	
	return cfg;
}

int main(int argc, char* argv[]) {
	Config cfg = loadConfig();
	
	int provided;
	MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
	assert(provided == MPI_THREAD_MULTIPLE);
	
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
	ProcType ptype = handle_parameters(argc, argv, cfg);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    
    MPI_Group search_group;
    int ranks[] = {0};
    MPI_Group_excl(world_group, 1, ranks, &search_group);
    
    MPI_Comm search_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, search_group, 0, &search_comm);
    

    if (world_rank == 1) {
    	generator(world_size - 2, ptype, cfg);
    } else if (world_rank == 0) {
    	aggregator(world_size - 2, ptype, cfg);
    } else {
    	search(world_rank - 2, world_size - 2, ptype, cfg);
    }
    
    // Finalize the MPI environment.
    MPI_Finalize();
}

inline double _search(faiss::Index* index, int nq, float* queries, float* D, faiss::Index::idx_t* I, Config& cfg) {
	double before = now();
	if (nq > 0) index->search(nq, queries, cfg.k, D, I);
	double after = now();
	
	return after - before;
}

void query_receiver(QueryBuffer* buffer, bool* finished, Config& cfg) {
	deb("Receiver");
	int queries_received = 0;
	
	float dummy;
	MPI_Send(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD); //signal that we are ready to receive queries
	
	//TODO: I will left it like this for the time being since buffer will always have free space available. In the future we should use events so that this thread sleeps while waiting for the buffer to have space
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
		
		while (! buffer->hasSpace());

		float* recv_data = reinterpret_cast<float*>(buffer->peekEnd());
		MPI_Recv(recv_data, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0,
				MPI_COMM_WORLD, &status);
		buffer->add();

		assert(status.MPI_ERROR == MPI_SUCCESS);

		queries_received += cfg.block_size;
	}
	
	deb("Finished receiving queries");
}

auto load_index(int shard, int nshards, Config& cfg) {
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

//TODO: separate each stage in its own file
//TODO: currently, we use the same files in all nodes. This doesn't work with ProfileData, since each machine is different. Need to create a way for each node to load a different file.
struct ProfileData {
	double* times;
	int min_block;
	int max_block;
};

ProfileData getProfilingData(bool cpu, Config& cfg) {	
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


void store_profile_data(const char* type, std::vector<double>& procTimes, Config& cfg) {
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

int bsearch(int min, int max, double* data, double val) {
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

int numBlocksRequired(ProcType ptype, QueryBuffer& buffer, ProfileData& pdGPU, Config& cfg) {
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

std::pair<int, int> computeSplitCPU_GPU(ProcType ptype, QueryBuffer& buffer, int num_blocks, ProfileData& pdGPU, ProfileData& pdCPU, Config& cfg) {
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

void process_buffer(ProcType ptype, faiss::Index* cpu_index, faiss::Index* gpu_index, int nq_cpu, int nq_gpu, QueryBuffer& buffer, faiss::Index::idx_t* I, float* D, std::vector<double>& procTimesGpu, std::vector<double>& procTimesCpu, Config& cfg) {
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
	QueryBuffer buffer(block_size_in_bytes, 100 * 1024 * 1024 / block_size_in_bytes); //100 MB
	
	faiss::gpu::StandardGpuResources res;
	res.setTempMemoryFraction(0.14);

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
		auto [nq_cpu, nq_gpu] = computeSplitCPU_GPU(ptype, buffer, num_blocks, pdGPU, pdCPU, cfg);
		
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

struct PartialResult {
	float* dists;
	long* ids;
	bool own_fields;
	float* base_dists;
	long* base_ids;
};

void merge_results(std::vector<PartialResult>& results, faiss::Index::idx_t* answers, int nshards, int k) {
	int counter[nshards];
	for (int i = 0; i < nshards; i++) counter[i] = 0;

	for (int topi = 0; topi < k; topi++) {
		float bestDist = std::numeric_limits<float>::max();
		long bestId = -1;
		int fromShard = -1;

		for (int shard = 0; shard < nshards; shard++) {
			if (counter[shard] == k) continue;

			if (results[shard].dists[counter[shard]] < bestDist) {
				bestDist = results[shard].dists[counter[shard]];
				bestId = results[shard].ids[counter[shard]];
				fromShard = shard;
			}
		}

		answers[topi] = bestId;
		counter[fromShard]++;
	}
}

void aggregate_query(std::queue<PartialResult>* queue, int nshards, faiss::Index::idx_t* answers, int k) {
	std::vector<PartialResult> results(nshards);
	
	for (int shard = 0; shard < nshards; shard++) {
		results[shard] = queue[shard].front();
		queue[shard].pop();
	}
				
	merge_results(results, answers, nshards, k);
	
	for (int shard = 0; shard < nshards; shard++) {
		if (results[shard].own_fields) {
			delete [] results[shard].base_dists;
			delete [] results[shard].base_ids;
		}	
	}
}


faiss::Index::idx_t* load_gt(Config& cfg) {
	//	 load ground-truth and convert int to long
	char idx_path[1000];
	char gt_path[500];
	sprintf(gt_path, "%s/gnd", src_path);
	sprintf(idx_path, "%s/idx_%dM.ivecs", gt_path, cfg.nb / 1000000);

	int n_out;
	int db_k;
	int *gt_int = ivecs_read(idx_path, &db_k, &n_out);

	faiss::Index::idx_t* gt = new faiss::Index::idx_t[cfg.k * cfg.nq];

	for (int i = 0; i < cfg.nq; i++) {
		for (int j = 0; j < cfg.k; j++) {
			gt[i * cfg.k + j] = gt_int[i * db_k + j];
		}
	}

	delete[] gt_int;
	return gt;
}

void send_times(std::deque<double>& end_times, int eval_length) {
	double end_times_array[eval_length];

	for (int i = 0; i < eval_length; i++) {
		end_times_array[i] = end_times.front();
		end_times.pop_front();
	}

	MPI_Send(end_times_array, eval_length, MPI_DOUBLE, GENERATOR, 0, MPI_COMM_WORLD);
}

//TODO: make this work for generic k's
void show_recall(faiss::Index::idx_t* answers, Config& cfg) {
	auto gt = load_gt(cfg);

	int n_1 = 0, n_10 = 0, n_100 = 0;
	
	for (int i = cfg.test_length - cfg.eval_length; i < cfg.test_length; i++) {
		int answer_id = i % cfg.eval_length;
		int nq = i % cfg.nq;
		int gt_nn = gt[nq * cfg.k];
		
		for (int j = 0; j < cfg.k; j++) {
			if (answers[answer_id * cfg.k + j] == gt_nn) {
				if (j < 1) n_1++;
				if (j < 10) n_10++;
				if (j < 100) n_100++;
			}
		}
	}

	printf("R@1 = %.4f\n", n_1 / float(cfg.eval_length));
	printf("R@10 = %.4f\n", n_10 / float(cfg.eval_length));
	printf("R@100 = %.4f\n", n_100 / float(cfg.eval_length));
	
	delete [] gt;
}

void aggregator(int nshards, ProcType ptype, Config& cfg) {
	std::deque<double> end_times;

	faiss::Index::idx_t* answers = new faiss::Index::idx_t[cfg.eval_length * cfg.k];
	
	std::queue<PartialResult> queue[nshards];
	std::queue<PartialResult> to_delete;
	
	deb("Aggregator node is ready");
	
	int shards_finished = 0;
	
	long qn = 0;
	while (true) {
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		int message_size;
		MPI_Get_count(&status, MPI_LONG, &message_size);
		
		if (message_size == 1) {
			shards_finished++;
			float dummy;
			MPI_Recv(&dummy, 1, MPI_LONG, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (shards_finished == nshards) break;
			continue;
		}

		int qty = message_size / cfg.k;

		auto I = new faiss::Index::idx_t[cfg.k * qty];
		auto D = new float[cfg.k * qty];
		
		MPI_Recv(I, cfg.k * qty, MPI_LONG, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D, cfg.k * qty, MPI_FLOAT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		int from = status.MPI_SOURCE - 2;
		for (int q = 0; q < qty; q++) {
			queue[from].push({D + cfg.k * q, I + cfg.k * q, q == qty - 1, D, I});
		}

		while (true) {
			bool hasEmpty = false;

			for (int i = 0; i < nshards; i++) {
				if (queue[i].empty()) {
					hasEmpty = true;
					break;
				}
			}
			
			if (hasEmpty) break;

			aggregate_query(queue, nshards, answers + (qn % cfg.eval_length) * cfg.k, cfg.k);
			qn++;
			
			if (end_times.size() >= cfg.eval_length) end_times.pop_front();
			end_times.push_back(now());
		}
	}
	
	if (ptype != ProcType::Bench) {
		show_recall(answers, cfg); 
		send_times(end_times, cfg.eval_length);
	}
	
	deb("Finished aggregator");
}
