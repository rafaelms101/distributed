#include "QueryQueue.h"

#include "utils.h"

QueryQueue::QueryQueue(char* id, faiss::IndexIVFPQ* index, QueueManager* _qm) :
		start_query_id(0), on_gpu(false), _cpu_index(index), _id(id), qm(_qm) {
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
	auto nq = qm->numberOfQueries(start_query_id);
	
	if (nq < 0 || nq > 1000) {
		deb("%s/%s: %d queries", on_gpu ? "gpu" : "cpu", _id, nq);
		
		if (nq < 0) exit(-1);
	}
	
	return nq;
}

faiss::IndexIVFPQ* QueryQueue::cpu_index() {
	return _cpu_index;
}

void QueryQueue::search() {
	faiss::Index* index = on_gpu ? dynamic_cast<faiss::Index*>(qm->gpuIndex()) : dynamic_cast<faiss::Index*>(_cpu_index);
	faiss::Index::idx_t* labels = (faiss::Index::idx_t*) _label_buffer->peekEnd();
	float* distances = (float*) _distance_buffer->peekEnd();
	float* query_start = qm->ptrToQueryBuffer(start_query_id);
	long nqueries = size();

	processed += nqueries;

	_label_buffer->waitForSpace(nqueries);
	_distance_buffer->waitForSpace(nqueries);

	index->search(nqueries, query_start, cfg.k, distances, labels);

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

char* QueryQueue::id() {
	return _id;
}
