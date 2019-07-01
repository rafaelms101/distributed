#ifndef QUERYQUEUE_H_
#define QUERYQUEUE_H_

#include "faiss/IndexIVFPQ.h"

#include "faiss/gpu/GpuIndexIVFPQ.h"

#include "Buffer.h"
#include "config.h"
#include "QueueManager.h"

class QueueManager;

class QueryQueue {
	QueueManager* qm;
	faiss::IndexIVFPQ* _cpu_index;
	Buffer* _label_buffer;
	Buffer* _distance_buffer;
	
	
	
public:
	faiss::gpu::GpuIndexIVFPQ* gpu_index;
	long start_query_id;
	bool on_gpu;
	
	QueryQueue(faiss::IndexIVFPQ*, QueueManager*);
	Buffer* label_buffer();
	Buffer* distance_buffer();
	long size();
	faiss::IndexIVFPQ* cpu_index();
	
	long results_size();
	void clear_result_buffer(int nqueries);
	void create_gpu_index(faiss::gpu::StandardGpuResources&);
	long search(int nqueries);
};

#endif /* QUERYQUEUE_H_ */
