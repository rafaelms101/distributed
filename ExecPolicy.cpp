#include "ExecPolicy.h"

#include <fstream>
#include <algorithm>
#include <future>   

int CPUGreedyPolicy::numBlocksRequired(Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);
	int num_blocks = buffer.entries();
	return std::min(num_blocks, 200);
}

void CPUGreedyPolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D) {
	float* query_buffer = reinterpret_cast<float*>(buffer.peekFront());
	cpu_index->search(nq, query_buffer, cfg.k, D, I);
	buffer.consume(nq / cfg.block_size);
}

void HybridPolicy::setup() {
	gpuPolice->setup();
	timesCPU = BenchExecPolicy::load_prof_times(false, shard, cfg);
	timesGPU = BenchExecPolicy::load_prof_times(true, shard, cfg);
}

int HybridPolicy::bsearch(std::vector<double>& times, double val) {
	int begin = 0;
	int end = times.size() - 1;
	
	while (begin < end - 1) {
		int mid = (end + begin) / 2;
		
		if (times[mid] > val) {
			end = mid - 1;
		} else {
			begin = mid;
		}
	}
	
	if (times[end] <= val) return end;
	else return begin;
}

int HybridPolicy::numBlocksRequired(Buffer& buffer, Config& cfg) {
	blocks_gpu = gpuPolice->numBlocksRequired(buffer, cfg);
	auto expected_time = timesGPU[blocks_gpu];
	blocks_cpu = bsearch(timesCPU, expected_time);
	
	return blocks_gpu + blocks_cpu;
}

static bool gpu_search(faiss::Index* gpu_index, int nq_gpu, float* query_buffer, faiss::Index::idx_t* I, float* D) {
	if (nq_gpu > 0) gpu_index->search(nq_gpu, query_buffer, cfg.k, D, I);
	return true;
}

void HybridPolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D) {
	auto nq_cpu = blocks_cpu * cfg.block_size;
	auto nq_gpu = nq - nq_cpu;
	
	deb("gpu=%d, cpu=%d", nq_gpu, nq_cpu);
	
	float* query_buffer = reinterpret_cast<float*>(buffer.peekFront());
	std::future<bool> fut = std::async(std::launch::async, gpu_search, gpu_index, nq_gpu, query_buffer, I, D);
	if (nq_cpu > 0) cpu_index->search(nq_cpu, query_buffer + nq_gpu * cfg.d, cfg.k, D + nq_gpu * cfg.k, I + nq_gpu * cfg.k);
	bool ret = fut.get();
	buffer.consume(nq / cfg.block_size);
}


int HybridBatch::numBlocksRequired(Buffer& buffer, Config& cfg) {
	return cfg.test_length / cfg.block_size;
}

void HybridBatch::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D) {
	auto nq_gpu = int(nq * gpuSpeedup / (1 + gpuSpeedup));
	auto nq_cpu = nq - nq_gpu;
	
	deb("gpu=%d, cpu=%d", nq_gpu, nq_cpu);

	float* query_buffer = reinterpret_cast<float*>(buffer.peekFront());
	std::future<bool> fut = std::async(std::launch::async, gpu_search, gpu_index, nq_gpu, query_buffer, I, D);
	if (nq_cpu > 0) cpu_index->search(nq_cpu, query_buffer + nq_gpu * cfg.d, cfg.k, D + nq_gpu * cfg.k, I + nq_gpu * cfg.k);
	bool ret = fut.get();
	buffer.consume(nq / cfg.block_size);
}

void BenchExecPolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D) {
	float* query_buffer = reinterpret_cast<float*>(buffer.peekFront());

	//now we proccess our query buffer
	if (! finished_gpu) {
		auto before = now();
		gpu_index->search(nq, query_buffer, cfg.k, D, I);
		auto time_spent = now() - before;
		procTimesGpu.push_back(time_spent);
		finished_gpu = time_spent >= 1;
	}
	
	if (! finished_cpu) {
		auto before = now();
		cpu_index->search(nq, query_buffer, cfg.k, D, I);
		auto time_spent = now() - before;
		procTimesCpu.push_back(time_spent);
		finished_cpu = time_spent >= 1;
	}


	buffer.consume(nq / cfg.block_size);
}

int BenchExecPolicy::numBlocksRequired(Buffer& buffer, Config& cfg) {
	if (nrepeats >= BENCH_REPEATS) {
		nrepeats = 0;
		nb++;
	}

	nrepeats++;
	return nb;
}

void BenchExecPolicy::store_profile_data(bool gpu, Config& cfg) {
	char checkup_command[100];
	sprintf(checkup_command, "mkdir -p %s", PROF_ROOT);
	system(checkup_command); //to make sure that the "prof" dir exists

	//now we write the time data on a file
	char file_path[100];
	sprintf(file_path, "%s/%s_%d_%d_%d_%d_%d_%d_%d", PROF_ROOT, gpu ? "gpu" : "cpu", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size, shard);
	std::ofstream file;
	file.open(file_path);

	std::vector<double>& procTimes = gpu ? procTimesGpu : procTimesCpu;
	
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

std::vector<double> BenchExecPolicy::load_prof_times(bool gpu, int shard_number, Config& cfg) {
	char file_path[100];
	sprintf(file_path, "%s/%s_%d_%d_%d_%d_%d_%d_%d", PROF_ROOT, gpu ? "gpu" : "cpu", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size, shard_number);
	std::ifstream file;
	file.open(file_path);

	if (!file.good()) {
		std::printf("File %s/%s_%d_%d_%d_%d_%d_%d_%d doesn't exist\n", PROF_ROOT, gpu ? "gpu" : "cpu", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size, shard_number);
		std::exit(-1);
	}

	int total_size;
	file >> total_size;

	std::vector<double> times(total_size + 1);

	times[0] = 0;

	for (int i = 1; i <= total_size; i++) {
		file >> times[i];
	}

	file.close();

	return times;
}

void BenchExecPolicy::cleanup(Config& cfg) {
	store_profile_data(true, cfg);
	store_profile_data(false, cfg);
}

void GPUPolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D) {
	float* query_buffer = reinterpret_cast<float*>(buffer.peekFront());
	gpu_index->search(nq, query_buffer, cfg.k, D, I);
	buffer.consume(nq / cfg.block_size);
}

std::pair<int, int> DynamicExecPolicy::longest_contiguous_region(double min, double tolerance, std::vector<double>& time_per_block) {
	int start, bestStart, bestEnd;
	int bestLength = 0;
	int length = 0;

	double threshold = min * (1 + tolerance);

	for (int i = 1; i < time_per_block.size(); i++) {
		if (time_per_block[i] <= threshold) {
			length++;

			if (length > bestLength) {
				bestStart = start;
				bestEnd = i;
				bestLength = length;
			}
		} else {
			start = i + 1;
			length = 0;
		}
	}

	return std::pair<int, int>(bestStart, bestEnd);
}

void DynamicExecPolicy::setup() {
	std::vector<double> times(BenchExecPolicy::load_prof_times(true, shard, cfg));
	std::vector<double> time_per_block(times.size());

	for (int i = 1; i < times.size(); i++) {
		time_per_block[i] = times[i] / i;
	}

	double tolerance = 0.1;
	int minBlock = 1;

	for (int nb = 1; nb < times.size(); nb++) {
		if (time_per_block[nb] < time_per_block[minBlock]) minBlock = nb;
	}

	std::pair<int, int> limits = longest_contiguous_region(time_per_block[minBlock], 0.1, time_per_block);
	pdGPU.min_block = limits.first;
	pdGPU.max_block = limits.second;

	pdGPU.times = new double[pdGPU.min_block + 1];
	pdGPU.times[0] = 0;
	for (int nb = 1; nb <= pdGPU.min_block; nb++)
		pdGPU.times[nb] = times[nb];

	deb("min=%d, max=%d", pdGPU.min_block * cfg.block_size, pdGPU.max_block * cfg.block_size);
	//	std::printf("min=%d\n", pd.min_block * cfg.block_size);
	assert(pdGPU.max_block <= cfg.eval_length);
}

int MinExecPolicy::numBlocksRequired(Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);

	int num_blocks = buffer.entries();

	if (num_blocks < pdGPU.min_block) {
		double work_without_waiting = num_blocks / pdGPU.times[num_blocks];
		double work_with_waiting = pdGPU.min_block / ((pdGPU.min_block - num_blocks) * buffer.block_interval() + pdGPU.times[pdGPU.min_block]);

		if (work_with_waiting > work_without_waiting) {
			usleep((pdGPU.min_block - num_blocks) * buffer.block_interval() * 1000000);
		}
	}

	return std::min(buffer.entries(), pdGPU.min_block);
}

int MaxExecPolicy::numBlocksRequired(Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);

	int num_blocks = buffer.entries();

	if (num_blocks < pdGPU.min_block) {
		double work_without_waiting = num_blocks / pdGPU.times[num_blocks];
		double work_with_waiting = pdGPU.min_block / ((pdGPU.min_block - num_blocks) * buffer.block_interval() + pdGPU.times[pdGPU.min_block]);

		if (work_with_waiting > work_without_waiting) {
			usleep((pdGPU.min_block - num_blocks) * buffer.block_interval() * 1000000);
		}
	}

	return std::min(buffer.entries(), pdGPU.max_block);
}	

int QueueExecPolicy::numBlocksRequired(Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);

	while (true) {
		//deciding between executing what we have or wait one extra block
		int num_blocks = buffer.entries();
		int nq = num_blocks * cfg.block_size;

		if (num_blocks >= pdGPU.min_block) {
			processed += pdGPU.min_block * cfg.block_size;
			return pdGPU.min_block;
		}

		if (nq + processed == cfg.test_length) return num_blocks;

		//case 1: execute right now
		auto queries_after_execute = pdGPU.times[num_blocks] / buffer.block_interval();

		if (queries_after_execute <= nq) {
			processed += nq;
			return num_blocks;
		}

		//case 2: wait for one more block
		auto queries_after_wait = pdGPU.times[num_blocks + 1] / buffer.block_interval();

		double increase_rate_execute = double(queries_after_execute - nq) / pdGPU.times[num_blocks];
		double increase_rate_wait = double(queries_after_wait - nq) / (buffer.block_interval() + pdGPU.times[num_blocks + 1]);

		if (increase_rate_execute <= increase_rate_wait) {
			processed += nq;
			return num_blocks;
		}

		buffer.waitForData(num_blocks + 1);
	}
}

int QueueMaxExecPolicy::numBlocksRequired(Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);

	while (true) {
		//deciding between executing what we have or wait one extra block
		int num_blocks = buffer.entries();
//		if (processed % 1000 == 0) std::printf("processed: %d\n", processed);
		int nq = num_blocks * cfg.block_size;

		if (num_blocks >= pdGPU.max_block) {
			processed += pdGPU.max_block * cfg.block_size;
			return pdGPU.max_block;
		}

		if (nq + processed == cfg.test_length) return num_blocks;

		//case 1: execute right now
		int queries_after_execute = pdGPU.times[num_blocks] / buffer.block_interval();

		//case 2: wait for one more block
		int queries_after_wait = pdGPU.times[num_blocks + 1] / buffer.block_interval();

		if (queries_after_execute <= nq) {
			processed += nq;
			return num_blocks;
		}

		if (queries_after_wait <= nq) {
			buffer.waitForData(num_blocks + 1);
			continue;
		}

		double increase_rate_execute = double(queries_after_execute - nq) / pdGPU.times[num_blocks];
		double increase_rate_wait = double(queries_after_wait - nq) / (buffer.block_interval() + pdGPU.times[num_blocks + 1]);

		if (increase_rate_execute <= increase_rate_wait) {
			processed += nq;
			return num_blocks;
		}

		buffer.waitForData(num_blocks + 1);
	}
}

int MinGreedyExecPolicy::numBlocksRequired(Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);
	int num_blocks = buffer.entries();
	return std::min(num_blocks, pdGPU.min_block);
}

int GreedyExecPolicy::numBlocksRequired(Buffer& buffer, Config& cfg) {
	buffer.waitForData(1);
	int num_blocks = buffer.entries();
	return std::min(num_blocks, 200);
}


