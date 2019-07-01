#ifndef QUEUEMANAGER_H_
#define QUEUEMANAGER_H_

#include <list>

#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/gpu/GpuAutoTune.h"

#include "Buffer.h"
#include "QueryQueue.h"

class QueryQueue;

class QueueManager {
	Buffer* query_buffer;
	long buffer_start_query_id = 0;
	std::mutex mutex_buffer_start;
	Buffer* label_buffer;
	Buffer* distance_buffer;
	long start_out_id = 0;
	long _sent_queries = 0;
	bool _gpu_loading = false;
	std::list<QueryQueue*> _queues;
	
public:
	QueueManager(Buffer* _query_buffer, Buffer* _label_buffer, Buffer* _distance_buffer);
	void addQueryQueue(QueryQueue* qq);
	int sent_queries();
	std::list<QueryQueue*>& queues();
	bool gpu_loading();
	long cpu_load();
	QueryQueue* biggestCPUQueue();
	QueryQueue* firstGPUQueue();
	float* ptrToQueryBuffer(long query_id);
	long numberOfQueries(long starting_query_id);
	void switchToGPU(QueryQueue*);
	void shrinkQueryBuffer();
	void mergeResults();
};

#endif /* QUEUEMANAGER_H_ */
