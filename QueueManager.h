#ifndef QUEUEMANAGER_H_
#define QUEUEMANAGER_H_

#include <list>

#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/gpu/GpuAutoTune.h"

#include "Buffer.h"
#include "QueryQueue.h"

class QueryQueue;

class QueueManager {
	Buffer* label_buffer;
	Buffer* distance_buffer;
	long start_out_id = 0;
	long _sent_queries = 0;
	
	Buffer* query_buffer;
	long buffer_start_query_id = 0;
	std::mutex mutex_buffer_start;
	
	void transferProcess(faiss::gpu::GpuIndexIVFPQ* gpu_index, QueryQueue* to_gpu);
	
public:
	bool gpu_loading = false;
	std::list<QueryQueue*> _queues;
	long bases_exchanged = 0;
	long log[1000][100];
	std::vector<std::pair<int, int>> switches;
	
	QueueManager(Buffer* _query_buffer, Buffer* _label_buffer, Buffer* _distance_buffer);
	void addQueryQueue(QueryQueue* qq);
	int sent_queries();
	std::list<QueryQueue*>& queues();
	long cpu_load();
	QueryQueue* biggestCPUQueue();
	QueryQueue* firstGPUQueue();
	float* ptrToQueryBuffer(long query_id);
	long numberOfQueries(long starting_query_id);
	void switchToGPU(QueryQueue* to_gpu, QueryQueue* to_cpu);
	void shrinkQueryBuffer();
	void mergeResults();
};

#endif /* QUEUEMANAGER_H_ */
