#ifndef QUEUEMANAGER_H_
#define QUEUEMANAGER_H_

#include <list>

#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/gpu/GpuAutoTune.h"

#include "Buffer.h"
#include "QueryQueue.h"

struct Size {
	long queries_in_buffer; 
	long starting_query_id; 
	long buffer_start_query_id;
	long size;
};

class QueryQueue;

class QueueManager {
	Buffer* label_buffer;
	Buffer* distance_buffer;
	long start_out_id = 0;
	long _sent_queries = 0;
	bool _gpu_loading = false;
	std::list<QueryQueue*> _queues;
	
public:
	std::mutex mutex_buffer_start;
	long buffer_start_query_id = 0;
	Buffer* query_buffer;
	
	QueueManager(Buffer* _query_buffer, Buffer* _label_buffer, Buffer* _distance_buffer);
	void addQueryQueue(QueryQueue* qq);
	int sent_queries();
	std::list<QueryQueue*>& queues();
	bool gpu_loading();
	long cpu_load();
	QueryQueue* biggestCPUQueue();
	QueryQueue* firstGPUQueue();
	float* ptrToQueryBuffer(long query_id);
	Size numberOfQueries(long starting_query_id);
	void switchToGPU(QueryQueue*);
	void shrinkQueryBuffer();
	void mergeResults();
};

#endif /* QUEUEMANAGER_H_ */
