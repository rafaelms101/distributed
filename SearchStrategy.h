#ifndef SEARCHSTRATEGY_H_
#define SEARCHSTRATEGY_H_

#include <map>

#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/gpu/GpuIndexIVFPQ.h"

#include "SyncBuffer.h"
#include "utils.h"

class SearchStrategy {
protected:
	std::vector<SyncBuffer*> query_buffer; 
	std::vector<SyncBuffer*> all_distance_buffers;
	std::vector<SyncBuffer*> all_label_buffers;
	
	SyncBuffer* distance_buffer; 
	SyncBuffer* label_buffer;
	
	float base_start; 
	float base_end;
	
	faiss::gpu::StandardGpuResources* res;
	
	long best_block_point_cpu;
	double best_block_point_cpu_time;
	long best_block_point_gpu;
	double best_block_point_gpu_time;
	
	void load_bench_data(bool cpu, long& best, double& best_time);

	void merge(long num_queries, std::vector<float*>& all_distances, std::vector<faiss::Index::idx_t*>& all_labels, float* distance_array, faiss::Index::idx_t* label_array);
	void merge_procedure(long& buffer_start_id, long& sent, std::vector<SyncBuffer*>& all_distance_buffers, std::vector<SyncBuffer*>& all_label_buffers, SyncBuffer& distance_buffer, SyncBuffer& label_buffer);
	void merger();
	
public:
	SearchStrategy(int num_queues, float _base_start, float _base_end, faiss::gpu::StandardGpuResources* _res = nullptr);
	
	virtual ~SearchStrategy() {};
	virtual void setup() = 0; //load bases and such
	virtual void start_search_process() = 0; //process queries
	
	std::vector<SyncBuffer*>& queryBuffers() { return query_buffer; }
	SyncBuffer* distanceBuffer() { return distance_buffer; }
	SyncBuffer* labelBuffer() { return label_buffer; }
};

class HybridSearchStrategy : public SearchStrategy {
	std::vector<faiss::IndexIVFPQ*> cpu_index;
	faiss::gpu::GpuIndexIVFPQ* gpu_index;
	long on_gpu;
	std::atomic<long> remaining_blocks;
	std::vector<std::mutex*> mutvec;
	
	void gpu_process();
	void cpu_process();
	
public:
	using SearchStrategy::SearchStrategy;
	
	void setup();
	void start_search_process();
};

class CpuOnlySearchStrategy : public SearchStrategy {
	faiss::IndexIVFPQ* cpu_index;
	
public:
	using SearchStrategy::SearchStrategy;
	
	void setup();
	void start_search_process();
};

class GpuOnlySearchStrategy : public SearchStrategy {
	std::vector<faiss::IndexIVFPQ*> cpu_index;
	faiss::gpu::GpuIndexIVFPQ* gpu_index;
	std::vector<long> remaining_blocks;
	
public:
	using SearchStrategy::SearchStrategy;
	
	void setup();
	void start_search_process();
};

class FixedSearchStrategy : public SearchStrategy {
	faiss::IndexIVFPQ* cpu_index;
	faiss::gpu::GpuIndexIVFPQ* gpu_index;
	
	void process(faiss::Index* index, SyncBuffer* query_buffer, SyncBuffer* distance_buffer, SyncBuffer* label_buffer, long cutoff_point); 
	
public:
	using SearchStrategy::SearchStrategy;
	
	void setup();
	void start_search_process();
};

#endif
