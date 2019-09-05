#ifndef EXECPOLICY_H_
#define EXECPOLICY_H_

#include "utils.h"
#include "Buffer.h"
#include "config.h"
#include "ProfileData.h"
#include "unistd.h"
#include "../faiss/IndexIVFPQ.h"


class ExecPolicy {
public:
	virtual ~ExecPolicy() {}
	
	virtual void setup() {}
	virtual int numBlocksRequired(Buffer& buffer, Config& cfg) = 0;
	virtual void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
};

class CPUPolicy : public ExecPolicy {
	int numBlocksRequired(Buffer& buffer, Config& cfg);
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
};

class StaticExecPolicy : public ExecPolicy {
private:
	int block_size;

public:
	StaticExecPolicy(int bs) : block_size(bs) {}
	
	int numBlocksRequired(Buffer& buffer, Config& cfg) { return block_size; }
};

class BenchExecPolicy : public ExecPolicy {
private:
	bool finished = false;
	int nrepeats = 0;
	int nb = 1;
	std::vector<double> procTimesGpu;
	
public:
	int numBlocksRequired(Buffer& buffer, Config& cfg);
	void process_buffer(faiss::Index* cpu_index, faiss::Index* gpu_index, int nq, Buffer& buffer, faiss::Index::idx_t* I, float* D);
};

class DynamicExecPolicy : public ExecPolicy {
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
