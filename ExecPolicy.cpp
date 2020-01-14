#include "ExecPolicy.h"

#include <fstream>
#include <algorithm>
#include <future>   

long CPUGreedyPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	long num_blocks = buffer.num_entries();
	return std::min(num_blocks, 20000L);
}

//TODO: add this to the HybridPolicy class
static long cpu_blocks(std::vector<double>& cpu, std::vector<double>& gpu, long nb) {
	long nbgpu = nb;
	long nbcpu = 0;
	
	double best_time = gpu[nbgpu];
	long bestnb = 0;
	
	nbgpu--;
	nbcpu++;
	
	while (nbcpu < cpu.size()) {
		double time = std::max(cpu[nbcpu], gpu[nbgpu]);
		if (time < best_time) {
			best_time = time;
			bestnb = nbcpu;
		}
		
		nbgpu--;
		nbcpu++;
	}
	
	return bestnb;
}

//TODO: there is a lot of duplication with the setup of DynamicExecPolicy
void HybridPolicy::setup() {
	timesCPU = BenchExecPolicy::load_prof_times(false, cfg);
	timesGPU = BenchExecPolicy::load_prof_times(true, cfg);
	
	std::vector<double> time_per_block(timesGPU.size());

	for (long i = 1; i < timesGPU.size(); i++) {
		time_per_block[i] = timesGPU[i] / i;
	}
	
	double tolerance = 0.1;
	std::pair<long, long> limits = longest_contiguous_region(tolerance, time_per_block);
	long minBlock = limits.first;
	
	max_blocks = minBlock + cpu_blocks(timesCPU, timesGPU, minBlock);
	
	nbToCpu.push_back(0);
	
	for (long i = 1; i <= max_blocks; i++) {
		nbToCpu.push_back(cpu_blocks(timesCPU, timesGPU, i));
	}
}

long HybridPolicy::bsearch(std::vector<double>& times, double val) {
	long begin = 0;
	long end = times.size() - 1;
	
	while (begin < end - 1) {
		long mid = (end + begin) / 2;
		
		if (times[mid] > val) {
			end = mid - 1;
		} else {
			begin = mid;
		}
	}
	
	if (times[end] <= val) return end;
	else return begin;
}

long HybridPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	long num_blocks = buffer.num_entries();
	num_blocks = std::min(num_blocks, max_blocks);
	
	return num_blocks;
}

static bool gpu_search(faiss::Index* gpu_index, long nq_gpu, float* query_buffer, faiss::Index::idx_t* I, float* D) {
	if (nq_gpu > 0) gpu_index->search(nq_gpu, query_buffer, cfg.k, D, I);
	return true;
}

void HybridPolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, SyncBuffer& buffer, faiss::Index::idx_t* I, float* D) {
	auto nq_cpu = nbToCpu[nq / cfg.block_size] * cfg.block_size;
	auto nq_gpu = nq - nq_cpu;
	
	deb("gpu=%d, cpu=%d", nq_gpu, nq_cpu);
	
	float* query_buffer = (float*) buffer.front();
	std::future<bool> fut = std::async(std::launch::async, gpu_search, gpu_index, nq_gpu, query_buffer, I, D);
	if (nq_cpu > 0) cpu_index->search(nq_cpu, query_buffer + nq_gpu * cfg.d, cfg.k, D + nq_gpu * cfg.k, I + nq_gpu * cfg.k);
	bool ret = fut.get();
	buffer.remove(nq / cfg.block_size);
}

void HybridCompositePolicy::setup() {
	timesCPU = BenchExecPolicy::load_prof_times(false, cfg);
	timesGPU = BenchExecPolicy::load_prof_times(true, cfg);

	nbToCpu.push_back(0);
	
	for (long i = 1; i <= timesGPU.size(); i++) {
		nbToCpu.push_back(cpu_blocks(timesCPU, timesGPU, i));
	}
	
	policy->setup(); 
}

//TODO: HybridCompositePolicy is basically a copy of HybridPolicy. Maybe merge the two?
long HybridCompositePolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	return policy->numBlocksRequired(buffer, cfg);
}

void HybridCompositePolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, SyncBuffer& buffer, faiss::Index::idx_t* I, float* D) {
	long nb = nq / cfg.block_size;
	
	auto nq_cpu = (nb < timesGPU.size() ? nbToCpu[nb] : (nbToCpu[timesGPU.size() - 1] * nb / timesGPU.size())) * cfg.block_size;
	auto nq_gpu = nq - nq_cpu;

	deb("gpu=%d, cpu=%d", nq_gpu, nq_cpu);

	float* query_buffer = (float*) buffer.front();
	std::future<bool> fut = std::async(std::launch::async, gpu_search, gpu_index, nq_gpu, query_buffer, I, D);
	if (nq_cpu > 0) cpu_index->search(nq_cpu, query_buffer + nq_gpu * cfg.d, cfg.k, D + nq_gpu * cfg.k, I + nq_gpu * cfg.k);
	bool ret = fut.get();
	buffer.remove(nq / cfg.block_size);
}

HybridBatch::HybridBatch(long _block_size) : block_size(_block_size) {
	auto timesCPU = BenchExecPolicy::load_prof_times(false, cfg);
	auto timesGPU = BenchExecPolicy::load_prof_times(true, cfg);

	nbCPU = cpu_blocks(timesCPU, timesGPU, block_size);
}

long HybridBatch::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	return block_size;
}

void HybridBatch::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, SyncBuffer& buffer, faiss::Index::idx_t* I, float* D) {
	auto nq_cpu = nbCPU * cfg.block_size;
	auto nq_gpu = nq - nq_cpu;
	
	deb("gpu=%d, cpu=%d", nq_gpu, nq_cpu);

	float* query_buffer = (float*) buffer.front();
	std::future<bool> fut = std::async(std::launch::async, gpu_search, gpu_index, nq_gpu, query_buffer, I, D);
	if (nq_cpu > 0) cpu_index->search(nq_cpu, query_buffer + nq_gpu * cfg.d, cfg.k, D + nq_gpu * cfg.k, I + nq_gpu * cfg.k);
	bool ret = fut.get();
	buffer.remove(nq / cfg.block_size);
}

void BenchExecPolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, SyncBuffer& buffer, faiss::Index::idx_t* I, float* D) {
	float* query_buffer = (float*) buffer.front();

	//now we proccess our query buffer
	if (cfg.bench_gpu && ! finished_gpu) {
		auto before = now();
		gpu_index->search(nq, query_buffer, cfg.k, D, I);
		auto time_spent = now() - before;
		procTimesGpu.push_back(time_spent);
		finished_gpu = time_spent >= 1;
	}
	
	if (cfg.bench_cpu && ! finished_cpu) {
		auto before = now();
		cpu_index->search(nq, query_buffer, cfg.k, D, I);
		auto time_spent = now() - before;
		procTimesCpu.push_back(time_spent);
		finished_cpu = time_spent >= 1;
	}


	buffer.remove(nq / cfg.block_size);
}

long BenchExecPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	if (nrepeats >= BENCH_REPEATS) {
		nrepeats = 0;
		nb += cfg.bench_step / cfg.block_size;
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
	sprintf(file_path, "%s/%s_%d_%d_%d_%d_%d_%d", PROF_ROOT, gpu ? "gpu" : "cpu", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
	std::ofstream file;
	file.open(file_path);

	std::vector<double>& procTimes = gpu ? procTimesGpu : procTimesCpu;
	
	long blocks = procTimes.size() / BENCH_REPEATS;
	file << blocks << std::endl;

	long ptr = 0;

	for (long b = 1; b <= blocks; b++) {
		std::vector<double> times;

		for (long repeats = 1; repeats <= BENCH_REPEATS; repeats++) {
			times.push_back(procTimes[ptr++]);
		}

		std::sort(times.begin(), times.end());

		long mid = BENCH_REPEATS / 2;
		file << times[mid] << std::endl;
	}

	file.close();
}

std::vector<double> BenchExecPolicy::load_prof_times(bool gpu, Config& cfg) {
	char file_path[100];
	sprintf(file_path, "%s/%s_%d_%d_%d_%d_%d_%d", PROF_ROOT, gpu ? "gpu" : "cpu", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
	std::ifstream file;
	file.open(file_path);

	if (!file.good()) {
		std::printf("File %s/%s_%d_%d_%d_%d_%d_%d doesn't exist\n", PROF_ROOT, gpu ? "gpu" : "cpu", cfg.nb, cfg.ncentroids, cfg.m, cfg.k, cfg.nprobe, cfg.block_size);
		std::exit(-1);
	}

	long total_size;
	file >> total_size;

	std::vector<double> times(total_size + 1);

	times[0] = 0;

	for (long i = 1; i <= total_size; i++) {
		file >> times[i];
	}

	file.close();

	return times;
}

void BenchExecPolicy::cleanup(Config& cfg) {
	if (cfg.bench_gpu) store_profile_data(true, cfg);
	if (cfg.bench_cpu) store_profile_data(false, cfg);
}

void GPUPolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, SyncBuffer& buffer, faiss::Index::idx_t* I, float* D) {
	float* query_buffer = (float*) buffer.front();
	gpu_index->search(nq, query_buffer, cfg.k, D, I);
	buffer.remove(nq / cfg.block_size);
}

void CPUPolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, SyncBuffer& buffer, faiss::Index::idx_t* I, float* D) {
	float* query_buffer = (float*) buffer.front();
	auto before = now();
	cpu_index->search(nq, query_buffer, cfg.k, D, I);
	cfg.raw_search_time += now() - before;
	buffer.remove(nq / cfg.block_size);
}

void StaticExecPolicy::process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, SyncBuffer& buffer, faiss::Index::idx_t* I, float* D) {
	float* query_buffer = (float*) buffer.front();
	(gpu ? gpu_index : cpu_index)->search(nq, query_buffer, cfg.k, D, I);
	buffer.remove(nq / cfg.block_size);
}

void DynamicExecPolicy::setup() {
	std::vector<double> times(BenchExecPolicy::load_prof_times(true, cfg));
	std::vector<double> time_per_block(times.size());

	for (long i = 1; i < times.size(); i++) {
		time_per_block[i] = times[i] / i;
	}

	double tolerance = 0.1;

	std::pair<long, long> limits = longest_contiguous_region(tolerance, time_per_block);
	pdGPU.min_block = limits.first;
	pdGPU.max_block = limits.second;

	pdGPU.times = new double[pdGPU.min_block + 1];
	pdGPU.times[0] = 0;
	for (long nb = 1; nb <= pdGPU.min_block; nb++)
		pdGPU.times[nb] = times[nb];

	deb("min=%d, max=%d", pdGPU.min_block * cfg.block_size, pdGPU.max_block * cfg.block_size);
	//	std::printf("min=%d\n", pd.min_block * cfg.block_size);
	assert(pdGPU.max_block <= cfg.num_blocks);
}

long MinExecPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	long num_blocks = buffer.num_entries();

	if (num_blocks < pdGPU.min_block) {
		double work_without_waiting = num_blocks / pdGPU.times[num_blocks];
		double work_with_waiting = pdGPU.min_block / ((pdGPU.min_block - num_blocks) * buffer.arrivalInterval() + pdGPU.times[pdGPU.min_block]);

		if (work_with_waiting > work_without_waiting) {
			usleep((pdGPU.min_block - num_blocks) * buffer.arrivalInterval() * 1000000);
		}
	}

	return std::min(buffer.num_entries(), pdGPU.min_block);
}

long MaxExecPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	long num_blocks = buffer.num_entries();

	if (num_blocks < pdGPU.min_block) {
		double work_without_waiting = num_blocks / pdGPU.times[num_blocks];
		double work_with_waiting = pdGPU.min_block / ((pdGPU.min_block - num_blocks) * buffer.arrivalInterval() + pdGPU.times[pdGPU.min_block]);

		if (work_with_waiting > work_without_waiting) {
			usleep((pdGPU.min_block - num_blocks) * buffer.arrivalInterval() * 1000000);
		}
	}

	return std::min(buffer.num_entries(), pdGPU.max_block);
}	

long QueueExecPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	while (true) {
		//deciding between executing what we have or wait one extra block
		long num_blocks = buffer.num_entries();
		long nq = num_blocks * cfg.block_size;

		if (num_blocks >= pdGPU.min_block) {
			blocks_processed += pdGPU.min_block;
			return pdGPU.min_block;
		}

		if (num_blocks + blocks_processed == cfg.num_blocks) return num_blocks;

		//case 1: execute right now
		auto queries_after_execute = pdGPU.times[num_blocks] / buffer.arrivalInterval();

		if (queries_after_execute <= nq) {
			blocks_processed += num_blocks;
			return num_blocks;
		}

		//case 2: wait for one more block
		auto queries_after_wait = pdGPU.times[num_blocks + 1] / buffer.arrivalInterval();

		double increase_rate_execute = double(queries_after_execute - nq) / pdGPU.times[num_blocks];
		double increase_rate_wait = double(queries_after_wait - nq) / (buffer.arrivalInterval() + pdGPU.times[num_blocks + 1]);

		if (increase_rate_execute <= increase_rate_wait) {
			blocks_processed +=  num_blocks;
			return num_blocks;
		}

		return 0;
	}
}

long BestExecPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	constexpr long lookahead_blocks = 1; 

	while (true) {
		//deciding between executing what we have or wait one extra block
		long num_blocks = buffer.num_entries();
		long nq = num_blocks * cfg.block_size;

		if (num_blocks >= pdGPU.min_block) {
			blocks_processed += pdGPU.min_block;
			return pdGPU.min_block;
		}

		if (num_blocks + blocks_processed == cfg.num_blocks) return num_blocks;

		//case 1: execute right now
		auto queries_after_execute = pdGPU.times[num_blocks] / buffer.arrivalInterval();
		auto time_after_execute = pdGPU.times[num_blocks];

		//case 2: wait for one more block
		auto queries_after_wait = pdGPU.times[num_blocks + lookahead_blocks] / buffer.arrivalInterval();
		auto time_after_wait = lookahead_blocks * buffer.arrivalInterval() + pdGPU.times[num_blocks + lookahead_blocks]; 

		double increase_rate_execute = (queries_after_execute - nq) / time_after_execute;
		double increase_rate_wait = (queries_after_wait - nq) / time_after_wait;

		if (increase_rate_execute <= increase_rate_wait) {
			blocks_processed += num_blocks;
			return num_blocks;
		}

		return 0;
	}
}

long GeorgeExecPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	while (true) {
		//deciding between executing what we have or wait one extra block
		long num_blocks = buffer.num_entries();
		long nq = num_blocks * cfg.block_size;

		if (num_blocks >= pdGPU.min_block) {
			blocks_processed += pdGPU.min_block;
			return pdGPU.min_block;
		}
		
		if (num_blocks + blocks_processed == cfg.num_blocks) return num_blocks;
		
		double time_to_max = (pdGPU.min_block - num_blocks) * buffer.arrivalInterval();
		
		double extra_time_wait = num_blocks * time_to_max;
		double extra_time_execute = (pdGPU.times[num_blocks] - time_to_max) * (pdGPU.min_block - num_blocks);
		
		if (extra_time_execute < extra_time_wait) {
			blocks_processed += num_blocks;
			return num_blocks;
		} else {
			return 0;
		}
	}
}

long QueueMaxExecPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	while (true) {
		//deciding between executing what we have or wait one extra block
		long num_blocks = buffer.num_entries();
//		if (processed % 1000 == 0) std::printf("processed: %d\n", processed);
		long nq = num_blocks * cfg.block_size;

		if (num_blocks >= pdGPU.max_block) {
			blocks_processed += pdGPU.max_block;
			return pdGPU.max_block;
		}

		if (num_blocks + blocks_processed == cfg.num_blocks) return num_blocks;

		//case 1: execute right now
		long queries_after_execute = pdGPU.times[num_blocks] / buffer.arrivalInterval();

		//case 2: wait for one more block
		long queries_after_wait = pdGPU.times[num_blocks + 1] / buffer.arrivalInterval();

		if (queries_after_execute <= nq) {
			blocks_processed += num_blocks;
			return num_blocks;
		}

		if (queries_after_wait <= nq) {
			buffer.waitForData(num_blocks + 1);
			continue;
		}

		double increase_rate_execute = double(queries_after_execute - nq) / pdGPU.times[num_blocks];
		double increase_rate_wait = double(queries_after_wait - nq) / (buffer.arrivalInterval() + pdGPU.times[num_blocks + 1]);

		if (increase_rate_execute <= increase_rate_wait) {
			blocks_processed += num_blocks;
			return num_blocks;
		}

		buffer.waitForData(num_blocks + 1);
	}
}

long MinGreedyExecPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	long num_blocks = buffer.num_entries();
	return std::min(num_blocks, pdGPU.min_block);
}

long GreedyExecPolicy::numBlocksRequired(SyncBuffer& buffer, Config& cfg) {
	long num_blocks = buffer.num_entries();
	return num_blocks;
}


