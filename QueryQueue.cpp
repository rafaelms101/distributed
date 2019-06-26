#include "QueryQueue.h"

long QueryQueue::buffer_start_query_id = 0;

QueryQueue::QueryQueue(faiss::IndexIVFPQ* index, Buffer* _query_buffer) : start_query_id(0), on_gpu(false), _cpu_index(index), gpu_index(nullptr), query_buffer(_query_buffer) {
	_label_buffer = new Buffer(sizeof(faiss::Index::idx_t) * cfg.k, 1000000);
	_distance_buffer = new Buffer(sizeof(float) * cfg.k, 1000000);
}

Buffer* QueryQueue::label_buffer() {
	return _label_buffer;
}

Buffer* QueryQueue::distance_buffer() {
	return _distance_buffer;
}

long QueryQueue::size() {
	long qty = query_buffer->entries() * cfg.block_size;
	return qty - (start_query_id - buffer_start_query_id);
}

faiss::IndexIVFPQ* QueryQueue::cpu_index() {
	return _cpu_index;
}

void QueryQueue::search() {
	faiss::Index* index = on_gpu ? dynamic_cast<faiss::Index*>(gpu_index) : dynamic_cast<faiss::Index*>(_cpu_index);
	faiss::Index::idx_t* labels = (faiss::Index::idx_t*) _label_buffer->peekEnd();
	float* distances = (float*) _distance_buffer->peekEnd();
	float* query_start = (float*) query_buffer->peekFront();
	long nqueries = size();

	_label_buffer->waitForSpace(nqueries);
	_distance_buffer->waitForSpace(nqueries);

	index->search(nqueries, query_start + (start_query_id - buffer_start_query_id) * cfg.d, cfg.k, distances, labels);

	_label_buffer->add(nqueries);
	_distance_buffer->add(nqueries);

	start_query_id += nqueries;
}

long QueryQueue::results_size() {
	return _distance_buffer->entries();
}

void QueryQueue::clear_result_buffer(int nqueries) {
	_distance_buffer->consume(nqueries);
	_label_buffer->consume(nqueries);
}
