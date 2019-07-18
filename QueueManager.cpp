#include "QueueManager.h"

#include "utils.h"

void QueueManager::shrinkQueryBuffer() {
	std::unique_lock<std::mutex> { mutex_buffer_start };
	long min_query_id = std::numeric_limits<long>::max();

	for (QueryQueue* qq : _queues) {
		auto qid = qq->start_query_id();

		if (qid < min_query_id) {
			min_query_id = qid;
		}
	}
	
	if (min_query_id == buffer_start_query_id) return;
	
	assert(min_query_id > buffer_start_query_id);
	assert((min_query_id - buffer_start_query_id) % cfg.block_size == 0);

	
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

	auto oda = distance_array;
	auto ola = label_array;
//
//	if (_sent_queries >= 325) {
//		deb("joining: ");
//
//		for (int i = 0; i < idxs.size(); i++) {
//			deb("Vector %d", i);
//			for (int q = 0; q < 5; q++) {
//				deb("Query %ld", q + _sent_queries);
//				for (int j = 0; j < cfg.k; j++) {
//					deb("%f", all_distances[i][q * cfg.k + j]);
//				}
//			}
//		}
//	}


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
	

//	for (int q = 0; q < num_queries; q++) {
//		std::printf("Query %ld\n", q + _sent_queries);
//		for (int j = 0; j < cfg.k; j++) {
//			std::printf("%ld: %f\n", ola[q * cfg.k + j], oda[q * cfg.k + j]);
//		}
//	}

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

bool QueueManager::gpu_loading() {
	return _gpu_loading;
}

long QueueManager::cpu_load() {
	long load = 0;

	for (QueryQueue* qq : _queues) {
		if (qq->on_gpu) continue;

		load += qq->size().size;
	}

	return load;
}

QueryQueue* QueueManager::biggestCPUQueue() {
	QueryQueue* biggest = _queues.front();

	for (QueryQueue* qq : _queues) {
		if (! qq->on_gpu && qq->size().size > biggest->size().size) {
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
	std::unique_lock<std::mutex> { mutex_buffer_start };
	return reinterpret_cast<float*>(query_buffer->peekFront()) + (query_id - buffer_start_query_id) * cfg.d;
}

Size QueueManager::numberOfQueries(long starting_query_id) {
	std::unique_lock<std::mutex> { mutex_buffer_start };
	Size size;
	size.buffer_start_query_id = buffer_start_query_id; //AUMENTA
	size.queries_in_buffer = query_buffer->entries() * cfg.block_size; //CONTINUA IGUAL
	size.starting_query_id = starting_query_id;
	size.size = size.queries_in_buffer - (size.starting_query_id - size.buffer_start_query_id);
//	auto sz = query_buffer->entries() * cfg.block_size - (starting_query_id - buffer_start_query_id);

//	if (sz < 0) {
//		std::printf("@%d * %d - (%ld - %ld) = %ld\n", query_buffer->entries(), cfg.block_size, starting_query_id, buffer_start_query_id, sz);
//	}
//
//	assert(sz >= 0);
	return size;
}

void QueueManager::switchToGPU(QueryQueue* to_gpu) {
	QueryQueue* to_cpu = firstGPUQueue();
	
	to_cpu->on_gpu = false;
	auto gpu_index = to_cpu->gpu_index;
	to_cpu->gpu_index = nullptr;
	
	_gpu_loading = true;
	gpu_index->copyFrom(to_gpu->cpu_index());
	to_gpu->gpu_index = gpu_index;
	to_gpu->on_gpu = true;
	_gpu_loading = false;
}
