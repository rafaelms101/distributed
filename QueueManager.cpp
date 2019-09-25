#include "QueueManager.h"

#include "utils.h"
#include <thread>

void QueueManager::shrinkQueryBuffer() {
	mutex_buffer_start.lock();
	
	long min_query_id = std::numeric_limits<long>::max();

	for (QueryQueue* qq : _queues) {
		auto qid = qq->start_query_id();

		if (qid < min_query_id) {
			min_query_id = qid;
		}
	}
	
	if (min_query_id > buffer_start_query_id) {
		assert(min_query_id > buffer_start_query_id);
		assert((min_query_id - buffer_start_query_id) % cfg.block_size == 0);
		query_buffer->consume((min_query_id - buffer_start_query_id) / cfg.block_size);
		buffer_start_query_id = min_query_id;
	}
	
	mutex_buffer_start.unlock();
}

//TODO: move this function to outside this class, since the idea is somewhat generic and can be reused in a lot of scenarios
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

int QueueManager::sent_queries() {
	return _sent_queries;
}

std::list<QueryQueue*>& QueueManager::queues() {
	return _queues;
}

long QueueManager::cpu_load() {
	long load = 0;

	for (QueryQueue* qq : _queues) {
		if (qq->on_gpu) continue;

		load += qq->size();
	}

	return load;
}

QueryQueue* QueueManager::biggestCPUQueue() {
	QueryQueue* biggest = _queues.front();

	for (QueryQueue* qq : _queues) {
		if (! qq->on_gpu && qq->size() > biggest->size()) {
			biggest = qq;
		}
	}

	return biggest;
}

QueryQueue* QueueManager::firstGPUQueue() {
	for (QueryQueue* qq : _queues) {
		if (qq->on_gpu) return qq;
	}
	
	return nullptr;
}

float* QueueManager::ptrToQueryBuffer(long query_id) {
	mutex_buffer_start.lock();
	auto ret = reinterpret_cast<float*>(query_buffer->peekFront()) + (query_id - buffer_start_query_id) * cfg.d;
	mutex_buffer_start.unlock();
	return ret;
}

long QueueManager::numberOfQueries(long starting_query_id) {
	mutex_buffer_start.lock();
	auto sz = query_buffer->entries() * cfg.block_size - (starting_query_id - buffer_start_query_id);
	mutex_buffer_start.unlock();
	
	return sz;
}

static void transferToGPU(faiss::gpu::GpuIndexIVFPQ* gpu_index, QueryQueue* to_gpu, QueueManager* qm) {
	gpu_index->copyFrom(to_gpu->cpu_index());
	to_gpu->gpu_index = gpu_index;
	to_gpu->on_gpu = true;
	qm->gpu_loading = false;
}

void QueueManager::switchToGPU(QueryQueue* to_gpu, QueryQueue* to_cpu) {
	switches.push_back(std::make_pair(to_cpu->id, to_gpu->id));
	
	for (auto q : _queues) {
		log[bases_exchanged][q->id] = q->size();
	}
	
	
	to_cpu->on_gpu = false;
	auto gpu_index = to_cpu->gpu_index;
	to_cpu->gpu_index = nullptr;
	
	bases_exchanged++;
	
	gpu_loading = true;
	std::thread thread(transferToGPU, gpu_index, to_gpu, this);
	thread.detach();
}
