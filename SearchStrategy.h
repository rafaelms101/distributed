#ifndef SEARCHSTRATEGY_H_
#define SEARCHSTRATEGY_H_


#include "QueueManager.h"
#include "utils.h"

class SearchStrategy {
protected:
	Buffer& query_buffer; 
	Buffer& distance_buffer; 
	Buffer& label_buffer;
	float base_start; 
	float base_end;
	faiss::gpu::StandardGpuResources* res;
	
	long best_query_point_cpu;
	long best_query_point_gpu;
	
	void load_bench_data(bool cpu, long& best);
	
public:
	SearchStrategy(Buffer& _query_buffer, Buffer& _distance_buffer, Buffer& _label_buffer, float _base_start, float _base_end, faiss::gpu::StandardGpuResources* _res = nullptr) : 
		query_buffer(_query_buffer),
		distance_buffer(_distance_buffer),
		label_buffer(_label_buffer),
		base_start(_base_start),
		base_end(_base_end),
		res(_res) {
		load_bench_data(true, best_query_point_cpu);
		load_bench_data(false, best_query_point_gpu);
		
		std::printf("cpu: %ld, gpu: %ld\n", best_query_point_cpu, best_query_point_gpu);
	}
	
	virtual ~SearchStrategy() {};
	virtual void setup() = 0; //load bases and such
	virtual void start_search_process() = 0; //process queries
	
	
	void merge(long num_queries, std::vector<float*>& all_distances, std::vector<faiss::Index::idx_t*>& all_labels, float* distance_array, faiss::Index::idx_t* label_array);
	void merge_procedure(long& buffer_start_id, long& sent, std::vector<Buffer*>& all_distance_buffers, std::vector<Buffer*>& all_label_buffers, Buffer& distance_buffer, Buffer& label_buffer);
};

class HybridSearchStrategy : public SearchStrategy {
	constexpr static long queries_threshold = 120l;
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
	
	std::vector<faiss::IndexIVFPQ*> cpu_bases;
	std::vector<Buffer*> all_distance_buffers;
	std::vector<Buffer*> all_label_buffers;
	std::vector<long> proc_ids;
	faiss::gpu::GpuIndexIVFPQ* gpu_index;
	long buffer_start_id = 0;
	
	constexpr static long queries_threshold = 120l;
	
	void merger();
	
public:
	using SearchStrategy::SearchStrategy;
	
	void setup();
	void start_search_process();
};

class CpuFixedSearchStrategy : public SearchStrategy {
	faiss::IndexIVFPQ* cpu_index;
	std::vector<faiss::IndexIVFPQ*> all_gpu_bases;
	faiss::gpu::GpuIndexIVFPQ* gpu_base;
	std::vector<Buffer*> all_distance_buffers;
	std::vector<Buffer*> all_label_buffers;
	std::vector<long> proc_ids;
	faiss::gpu::GpuIndexIVFPQ* gpu_index;
	long buffer_start_id = 0;
	long sent = 0;
	std::mutex sync_mutex;
	
	constexpr static long queries_threshold = 120l;
	
	void cpu_process();
	void gpu_process();
	
public:
	using SearchStrategy::SearchStrategy;
	
	void setup();
	void start_search_process();
};

#endif
