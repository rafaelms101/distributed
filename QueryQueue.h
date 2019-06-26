#ifndef QUERYQUEUE_H_
#define QUERYQUEUE_H_

#include "faiss/IndexIVFPQ.h"

#include "faiss/gpu/GpuIndexIVFPQ.h"

#include "Buffer.h"
#include "config.h"

class QueryQueue {
	faiss::IndexIVFPQ* _cpu_index;
	Buffer* query_buffer;
	Buffer* _label_buffer;
	Buffer* _distance_buffer;
	
public:
	static long buffer_start_query_id; 
	faiss::gpu::GpuIndexIVFPQ* gpu_index;
	long start_query_id;
	bool on_gpu;
	
	QueryQueue(faiss::IndexIVFPQ* index, Buffer* _query_buffer);
	Buffer* label_buffer();
	Buffer* distance_buffer();
	long size();
	faiss::IndexIVFPQ* cpu_index();
	void search();
	long results_size();
	void clear_result_buffer(int nqueries);
};

#endif /* QUERYQUEUE_H_ */
