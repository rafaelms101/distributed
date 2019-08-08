#ifndef EXECPOLICY_H_
#define EXECPOLICY_H_

#include "utils.h"
#include "Buffer.h"
#include "config.h"
#include "ProfileData.h"
#include "unistd.h"


class ExecPolicy {
public:
	virtual ~ExecPolicy() {}
	
	virtual void setup() {}
	virtual int numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg) = 0;
};


class StaticExecPolicy : public ExecPolicy {
private:
	int block_size;

public:
	StaticExecPolicy(int bs) : block_size(bs) {}
	
	int numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg) { return block_size; }
};

class BenchExecPolicy : public ExecPolicy {
private:
	int nrepeats = 0;
	int nb = 1;
	
public:
	int numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg);
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
	int numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg);
};

class MaxExecPolicy : public DynamicExecPolicy {
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg);
};

class QueueExecPolicy : public DynamicExecPolicy {
private:
	int processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg);	
};

class QueueMaxExecPolicy : public DynamicExecPolicy {
private:
	int processed = 0;
	
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg);	
};



class MinGreedyExecPolicy : public DynamicExecPolicy {
public:
	using DynamicExecPolicy::DynamicExecPolicy;
	int numBlocksRequired(ProcType ptype, Buffer& buffer, Config& cfg);	
};

#endif /* EXECPOLICY_H_ */
