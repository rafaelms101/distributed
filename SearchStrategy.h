#ifndef SEARCHSTRATEGY_H_
#define SEARCHSTRATEGY_H_

#include <map>

#include "QueueManager.h"
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
	long best_block_point_gpu;
	
	void load_bench_data(bool cpu, long& best);

	void merge(long num_queries, std::vector<float*>& all_distances, std::vector<faiss::Index::idx_t*>& all_labels, float* distance_array, faiss::Index::idx_t* label_array);
	void merge_procedure(long& buffer_start_id, long& sent, std::vector<Buffer*>& all_distance_buffers, std::vector<Buffer*>& all_label_buffers, Buffer& distance_buffer, Buffer& label_buffer);
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
	QueueManager* qm;
	
	void gpu_process(std::mutex* cleanup_mutex);
	void cpu_process(std::mutex* cleanup_mutex);
	
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
	std::condition_variable should_merge;
	std::mutex should_merge_mutex;
	
	std::vector<int> baseMap;
	std::vector<int> reverseBaseMap;
	std::vector<faiss::IndexIVFPQ*> cpu_bases;
	std::vector<Buffer*> all_distance_buffers;
	std::vector<Buffer*> all_label_buffers;
	std::vector<long> proc_ids;
	std::vector<faiss::gpu::GpuIndexIVFPQ*> gpu_indexes;
	long buffer_start_id = 0;
	
	void merger();
	
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
