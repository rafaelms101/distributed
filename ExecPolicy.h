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
	virtual long numBlocksRequired(Buffer& buffer, Config& cfg) = 0;
	virtual void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, Buffer& buffer, faiss::Index::idx_t* I, float* D) = 0;
	virtual void cleanup(Config& cfg) {}
	virtual bool usesGPU() = 0;
};

class CPUPolicy : public ExecPolicy {
public:
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	bool usesGPU() { return false; }
};

class CPUGreedyPolicy : public CPUPolicy {	
public:
	long numBlocksRequired(Buffer& buffer, Config& cfg);
};

class GPUPolicy : public ExecPolicy {
public:
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	bool usesGPU() { return true; }
};

class HybridBatch : public ExecPolicy {
	long nbCPU;
	long block_size;
	
public:
	HybridBatch(long block_size);
	long numBlocksRequired(Buffer& buffer, Config& cfg);
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	bool usesGPU() { return true; }
};

class HybridPolicy : public ExecPolicy {
	std::vector<double> timesCPU;
	std::vector<double> timesGPU;
	std::vector<long> nbToCpu;
	long max_blocks = 0;
	
	long bsearch(std::vector<double>& times, double val);
	
public:
	HybridPolicy() {}
	void setup();
	long numBlocksRequired(Buffer& buffer, Config& cfg);
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	void cleanup(Config& cfg) { }
	bool usesGPU() { return true; }
};

class HybridCompositePolicy : public ExecPolicy {
	std::vector<double> timesCPU;
	std::vector<double> timesGPU;
	std::vector<long> nbToCpu;
	ExecPolicy* policy; 
	
public:
	HybridCompositePolicy(ExecPolicy* _policy) : policy(_policy) {}
	void setup();
	long numBlocksRequired(Buffer& buffer, Config& cfg);
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	void cleanup(Config& cfg) { policy->cleanup(cfg); }	
	bool usesGPU() { return true; }
};

class StaticExecPolicy : public ExecPolicy {
private:
	long block_size;
	bool gpu;

public:
	StaticExecPolicy(bool _gpu, long bs) : block_size(bs), gpu(_gpu) {}
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	long numBlocksRequired(Buffer& buffer, Config& cfg) { return block_size; }
	bool usesGPU() { return gpu; }
};

class BenchExecPolicy : public ExecPolicy {
private:
	bool finished_cpu = false;
	bool finished_gpu = false;
	long nrepeats = BENCH_REPEATS;
	long nb = 0;
	std::vector<double> procTimesGpu;
	std::vector<double> procTimesCpu;

	void store_profile_data(bool gpu, Config& cfg);
	
public:
	BenchExecPolicy() {} 
	long numBlocksRequired(Buffer& buffer, Config& cfg);
	void cleanup(Config& cfg);
	
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, long nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	static std::vector<double> load_prof_times(bool gpu, Config& cfg);
	bool usesGPU() { return true; }
};



class DynamicExecPolicy : public GPUPolicy {
protected:
	ProfileData pdGPU;
	
public:
	DynamicExecPolicy() {} 
	void setup();
};


class MinExecPolicy : public DynamicExecPolicy {
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	long numBlocksRequired(Buffer& buffer, Config& cfg);
};

class MaxExecPolicy : public DynamicExecPolicy {
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	long numBlocksRequired(Buffer& buffer, Config& cfg);
};

class QueueExecPolicy : public DynamicExecPolicy {
private:
	long processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	long numBlocksRequired(Buffer& buffer, Config& cfg);
};

class BestExecPolicy : public DynamicExecPolicy {
private:
	long processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	long numBlocksRequired(Buffer& buffer, Config& cfg);
};

class GeorgeExecPolicy : public DynamicExecPolicy {
private:
	long processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	long numBlocksRequired(Buffer& buffer, Config& cfg);
};

class QueueMaxExecPolicy : public DynamicExecPolicy {
private:
	long processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	long numBlocksRequired(Buffer& buffer, Config& cfg);
};



class MinGreedyExecPolicy : public DynamicExecPolicy {
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	long numBlocksRequired(Buffer& buffer, Config& cfg);
};

class GreedyExecPolicy : public GPUPolicy {
public:
	long numBlocksRequired(Buffer& buffer, Config& cfg);
};

#endif /* EXECPOLICY_H_ */
