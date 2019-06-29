#include "QueueManager.h"

#include "utils.h"

void QueueManager::shrinkQueryBuffer() {
	long min_query_id = std::numeric_limits<long>::max();

	for (QueryQueue* qq : _queues) {
		if (qq->start_query_id < min_query_id) {
			min_query_id = qq->start_query_id;
		}
	}
	
	if (min_query_id == buffer_start_query_id) return;
	
	assert(min_query_id > buffer_start_query_id);
	
	std::unique_lock<std::mutex> { mutex_buffer_start };
	
	query_buffer->consume((min_query_id - buffer_start_query_id) / cfg.block_size);
	buffer_start_query_id = min_query_id;
}

void QueueManager::mergeResults() {
	long num_queries = std::numeric_limits<long>::max();

	for (QueryQueue* qq : _queues) {
		if (qq->results_size() < num_queries) {
			num_queries = qq->results_size();
		}
	}

	if (num_queries == 0) return;
	

	float* all_distances[_queues.size()];
	faiss::Index::idx_t* all_labels[_queues.size()];

	int i = 0;
	for (QueryQueue* qq : _queues) {
		//TODO: potentially needs a lock
		all_distances[i] = reinterpret_cast<float*>(qq->distance_buffer()->peekFront());
		all_labels[i] = reinterpret_cast<faiss::Index::idx_t*>(qq->label_buffer()->peekFront());
		i++;
	}
	
	std::vector<long> idxs(_queues.size());

	distance_buffer->waitForSpace(num_queries);
	label_buffer->waitForSpace(num_queries);

	float* distance_array = reinterpret_cast<float*>(distance_buffer->peekEnd());
	faiss::Index::idx_t* label_array = reinterpret_cast<faiss::Index::idx_t*>(label_buffer->peekEnd());
	
	for (int q = 0; q < num_queries; q++) {
		for (int i = 0; i < _queues.size(); i++) {
			idxs[i] = q * cfg.k;
		}

		for (int topi = 0; topi < cfg.k; topi++) {
			int from = 0;

			for (int i = 1; i < idxs.size(); i++) {
				if (all_distances[i][idxs[i]] < all_distances[from][idxs[from]]) {
					from = i;
				}
			}

			*distance_array = all_distances[from][idxs[from]];
			*label_array = all_labels[from][idxs[from]];
			
			distance_array++;
			label_array++;
			idxs[from]++;
		}
	}
	
	distance_buffer->add(num_queries / cfg.block_size);
	label_buffer->add(num_queries / cfg.block_size);
	
	for (QueryQueue* qq : _queues) {
		qq->clear_result_buffer(num_queries);
	}

	_sent_queries += num_queries;
}

QueueManager::QueueManager(Buffer* _query_buffer, Buffer* _label_buffer, Buffer* _distance_buffer) :
		query_buffer(_query_buffer), label_buffer(_label_buffer), distance_buffer(_distance_buffer) {

}

void QueueManager::addQueryQueue(QueryQueue* qq) {
	_queues.push_back(qq);
}

void QueueManager::replaceGPUIndex(QueryQueue* qq) {
	deb("gpu: changing from queue %s to queue %s",gpu_queue->id(), qq->id());
	_gpu_loading = true;
	gpu_queue->on_gpu = false;
	qq->on_gpu = true;
	gpu_queue = qq;
	gpu_index->copyFrom(qq->cpu_index());
	_gpu_loading = false;
	deb("finished changing");
}

int QueueManager::sent_queries() {
	return _sent_queries;
}

std::list<QueryQueue*>& QueueManager::queues() {
	return _queues;
}

bool QueueManager::gpu_loading() {
	return _gpu_loading;
}

long QueueManager::cpu_load() {
	long load = 0;

	for (QueryQueue* qq : _queues) {
		if (qq->on_gpu) continue;

		load += qq->size();
	}

	return load;
}

QueryQueue* QueueManager::biggestQueue() {
	QueryQueue* biggest = _queues.front();

	for (QueryQueue* qq : _queues) {
		if (qq->size() > biggest->size()) {
			biggest = qq;
		}
	}

	return biggest;
}

void QueueManager::processGPU() {
	long remaining = cfg.test_length - _sent_queries;
	long threshold = std::min(std::min(100l, gpu_queue->size()), remaining);
	if (gpu_queue->size() >= threshold) gpu_queue->search(threshold);
	shrinkQueryBuffer();
	mergeResults();
}

void QueueManager::processCPU(QueryQueue* qq) {
	if (qq->size() >= 1) qq->search(qq->size());
}

void QueueManager::setStartingGPUQueue(QueryQueue* qq, faiss::gpu::StandardGpuResources& res) {
	gpu_index = static_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(& res, 0, qq->cpu_index(), nullptr));
	qq->on_gpu = true;
	gpu_queue = qq;
}

float* QueueManager::ptrToQueryBuffer(long query_id) {
	std::unique_lock<std::mutex> { mutex_buffer_start };
	return reinterpret_cast<float*>(query_buffer->peekFront()) + (query_id - buffer_start_query_id) * cfg.d;
}

long QueueManager::numberOfQueries(long starting_query_id) {
	std::unique_lock<std::mutex> { mutex_buffer_start };
	return query_buffer->entries() * cfg.block_size - (starting_query_id - buffer_start_query_id);
}

faiss::gpu::GpuIndexIVFPQ* QueueManager::gpuIndex() {
	return gpu_index;
}
