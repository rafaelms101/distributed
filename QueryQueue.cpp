#include "QueryQueue.h"

#include "utils.h"

QueryQueue::QueryQueue(faiss::IndexIVFPQ* index, QueueManager* _qm) :
		_start_query_id(0), on_gpu(false), _cpu_index(index), qm(_qm), gpu_index(nullptr) {
	_label_buffer = new Buffer(sizeof(faiss::Index::idx_t) * cfg.k, 1000000);
	_distance_buffer = new Buffer(sizeof(float) * cfg.k, 1000000);
	qm->addQueryQueue(this);
}

Buffer* QueryQueue::label_buffer() {
	return _label_buffer;
}

Buffer* QueryQueue::distance_buffer() {
	return _distance_buffer;
}

long QueryQueue::size() {
	auto sz = qm->numberOfQueries(_start_query_id);
//	assert(sz >= 0);
	return sz;
}

faiss::IndexIVFPQ* QueryQueue::cpu_index() {
	return _cpu_index;
}

void QueryQueue::search(int nqueries) {
	if (nqueries == 0) return;
	
	
	faiss::Index* index = on_gpu ? static_cast<faiss::Index*>(gpu_index) : static_cast<faiss::Index*>(_cpu_index);
	faiss::Index::idx_t* labels = (faiss::Index::idx_t*) _label_buffer->peekEnd();
	float* distances = (float*) _distance_buffer->peekEnd();
	float* query_start = qm->ptrToQueryBuffer(_start_query_id);

	//TODO: THIS IS UNSAFE, what would happen if the gpu and cpu tried to execute this on the same queue at the same time????????????????
	_start_query_id += nqueries;
	
	_label_buffer->waitForSpace(nqueries);
	_distance_buffer->waitForSpace(nqueries);

//	std::printf("searching %d queries\n", nqueries);
	index->search(nqueries, query_start, cfg.k, distances, labels);	

	_label_buffer->add(nqueries);
	_distance_buffer->add(nqueries);
//
//	_start_query_id += nqueries;
}

long QueryQueue::results_size() {
	return _distance_buffer->entries();
}

void QueryQueue::clear_result_buffer(int nqueries) {
	_distance_buffer->consume(nqueries);
	_label_buffer->consume(nqueries);
}

void QueryQueue::create_gpu_index(faiss::gpu::StandardGpuResources& res) {
	gpu_index = static_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(& res, 0, _cpu_index, nullptr));
	on_gpu = true;
}

long QueryQueue::start_query_id() {
	return _start_query_id;
}
