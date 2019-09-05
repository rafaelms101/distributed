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
	int numBlocksRequired(Buffer& buffer, Config& cfg);
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
};

class GPUPolicy : public ExecPolicy {
public:
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
};

class HybridPolicy : public ExecPolicy {
	GPUPolicy* gpuPolice;
	std::vector<double> timesCPU;
	std::vector<double> timesGPU;
	int blocks_cpu = 0;
	int blocks_gpu = 0;
	int shard;
	
	int bsearch(std::vector<double>& times, double val);
	
public:
	HybridPolicy(GPUPolicy* _gpuPolice, int _shard) : gpuPolice(_gpuPolice), shard(_shard) {}
	void setup();
	int numBlocksRequired(Buffer& buffer, Config& cfg);
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	void cleanup(Config& cfg) { gpuPolice->cleanup(cfg); }
};

class StaticExecPolicy : public GPUPolicy {
private:
	int block_size;

public:
	StaticExecPolicy(int bs) : block_size(bs) {}
	
	int numBlocksRequired(Buffer& buffer, Config& cfg) { return block_size; }
};

class BenchExecPolicy : public ExecPolicy {
private:
	int shard;
	bool finished_cpu = false;
	bool finished_gpu = false;
	int nrepeats = 0;
	int nb = 1;
	std::vector<double> procTimesGpu;
	std::vector<double> procTimesCpu;

	void store_profile_data(bool gpu, Config& cfg);
	
public:
	BenchExecPolicy(int _shard) : shard(_shard) {} 
	int numBlocksRequired(Buffer& buffer, Config& cfg);
	void cleanup(Config& cfg);
	
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
	static std::vector<double> load_prof_times(bool gpu, int shard_number, Config& cfg);
	
};



class DynamicExecPolicy : public GPUPolicy {
protected:
	ProfileData pdGPU;
	
private:
	std::pair<int, int> longest_contiguous_region(double min, double tolerance, std::vector<double>& time_per_block);
	int shard;
	
public:
	DynamicExecPolicy(int _shard) : shard(_shard) {} 
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

class GreedyExecPolicy : public DynamicExecPolicy {
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(Buffer& buffer, Config& cfg);
};

#endif /* EXECPOLICY_H_ */
