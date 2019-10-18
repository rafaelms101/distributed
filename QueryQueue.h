#ifndef QUERYQUEUE_H_
#define QUERYQUEUE_H_

#include "faiss/IndexIVFPQ.h"

#include "faiss/gpu/GpuIndexIVFPQ.h"

#include "Buffer.h"
#include "config.h"
#include "QueueManager.h"

class QueueManager;
struct Size;

class QueryQueue {
	QueueManager* qm;
	faiss::IndexIVFPQ* _cpu_index;
	Buffer* _label_buffer;
	Buffer* _distance_buffer;
	long _start_query_id;
	std::mutex queue_mutex;
	
	
public:
	int id;
	
	long start_query_id();

	faiss::gpu::GpuIndexIVFPQ* gpu_index;

	bool on_gpu;
	
	QueryQueue(faiss::IndexIVFPQ*, QueueManager*, int);
	Buffer* label_buffer();
	Buffer* distance_buffer();
	long size();
	faiss::IndexIVFPQ* cpu_index();
	void lock();
	void unlock();
	
	long results_size();
	void clear_result_buffer(int nqueries);
	void create_gpu_index(faiss::gpu::StandardGpuResources&);
	void search(int nqueries);
};

#endif /* QUERYQUEUE_H_ */
