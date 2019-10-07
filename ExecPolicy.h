#ifndef EXECPOLICY_H_
#define EXECPOLICY_H_

#include "utils.h"
#include "Buffer.h"
#include "config.h"
#include "ProfileData.h"
#include "unistd.h"
#include "../faiss/IndexIVFPQ.h"

//TODO: remove the need to pass explicitly the CFG
class ExecPolicy {
public:
	virtual ~ExecPolicy() {}
	
	virtual void setup() {}
	virtual int numBlocksRequired(Buffer& buffer, Config& cfg) = 0;
	virtual void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D) = 0;
	virtual void cleanup(Config& cfg) {}
};

class CPUPolicy : public ExecPolicy {
public:
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
};

class CPUGreedyPolicy : public CPUPolicy {	
public:
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};

class GPUPolicy : public ExecPolicy {
public:
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
};

class HybridBatch : public ExecPolicy {
	double gpuRatio;
	int procSize;
	
public:
	HybridBatch(double _gpuRatio, int _procSize) : gpuRatio(_gpuRatio), procSize(_procSize) {}
	int numBlocksRequired(Buffer& buffer, Config& cfg);
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
};

class HybridPolicy : public ExecPolicy {
	std::vector<double> timesCPU;
	std::vector<double> timesGPU;
	std::vector<int> gpuToCpu;
	int blocks_cpu = 0;
	int blocks_gpu = 0;
	int max_blocks = 0;
	
	int bsearch(std::vector<double>& times, double val);
	
public:
	HybridPolicy() {}
	void setup();
	int numBlocksRequired(Buffer& buffer, Config& cfg);
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	void cleanup(Config& cfg) { }
};

class StaticExecPolicy : public ExecPolicy {
private:
	int block_size;
	bool gpu;

public:
	StaticExecPolicy(bool _gpu, int bs) : block_size(bs), gpu(_gpu) {}
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	int numBlocksRequired(Buffer& buffer, Config& cfg) { return block_size; }
};

class BenchExecPolicy : public ExecPolicy {
private:
	bool finished_cpu = false;
	bool finished_gpu = false;
	int nrepeats = BENCH_REPEATS;
	int nb = 0;
	std::vector<double> procTimesGpu;
	std::vector<double> procTimesCpu;

	void store_profile_data(bool gpu, Config& cfg);
	
public:
	BenchExecPolicy() {} 
	int numBlocksRequired(Buffer& buffer, Config& cfg);
	void cleanup(Config& cfg);
	
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	static std::vector<double> load_prof_times(bool gpu, Config& cfg);
};



class DynamicExecPolicy : public GPUPolicy {
protected:
	ProfileData pdGPU;
	
private:
	std::pair<int, int> longest_contiguous_region(double min, double tolerance, std::vector<double>& time_per_block);
	
public:
	DynamicExecPolicy() {} 
	void setup();
};


class MinExecPolicy : public DynamicExecPolicy {
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};

class MaxExecPolicy : public DynamicExecPolicy {
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};

class QueueExecPolicy : public DynamicExecPolicy {
private:
	int processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};

class BestExecPolicy : public DynamicExecPolicy {
private:
	int processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};

class GeorgeExecPolicy : public DynamicExecPolicy {
private:
	int processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};

class QueueMaxExecPolicy : public DynamicExecPolicy {
private:
	int processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};



class MinGreedyExecPolicy : public DynamicExecPolicy {
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};

class GreedyExecPolicy : public GPUPolicy {
public:
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};

#endif /* EXECPOLICY_H_ */
