#include "SearchStrategy.h"

#include <thread>
#include <vector>
#include <algorithm>
#include <fstream>

#include "QueryQueue.h"

void SearchStrategy::load_bench_data(bool cpu, long& best) {
	char file_path[100];
	sprintf(file_path, "%s_bench", cpu ? "cpu" : "gpu");
	std::ifstream file;
	file.open(file_path);

	if (! file.good()) {
		std::printf("File %s_bench", cpu ? "cpu" : "gpu");
		std::exit(-1);
	}

	int total_size;
	file >> total_size;

	best = 0;
	double best_time_per_query = 9999999;
	
	for (int i = 2; i <= total_size; i++) {
		long qty;
		file >> qty;
		double total_time;
		file >> total_time;
		
		if (total_time / qty < best_time_per_query) {
			best = qty;
			best_time_per_query = total_time / qty;
		}
	}

	file.close();
}

void SearchStrategy::merge(long num_queries, std::vector<float*>& all_distances, std::vector<faiss::Index::idx_t*>& all_labels, float* distance_array,
		faiss::Index::idx_t* label_array) {
	if (num_queries == 0) return;

	auto oda = distance_array;
	auto ola = label_array;

	std::vector<long> idxs(all_distances.size());

	for (int q = 0; q < num_queries; q++) {
		for (int i = 0; i < all_distances.size(); i++) {
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
}

void SearchStrategy::merge_procedure(long& buffer_start_id, long& sent, std::vector<Buffer*>& all_distance_buffers, std::vector<Buffer*>& all_label_buffers,
		Buffer& distance_buffer, Buffer& label_buffer) {
	//merge
	auto num_queries = buffer_start_id - sent;

	if (num_queries == 0) return;
	assert(num_queries > 0);

	std::vector<float*> all_distances;
	std::vector<faiss::Index::idx_t*> all_labels;
	for (int i = 0; i < all_distance_buffers.size(); i++) {
		auto dptr = (float*) all_distance_buffers[i]->peekFront();
		auto lptr = (faiss::Index::idx_t*) all_label_buffers[i]->peekFront();
		all_distances.push_back(dptr);
		all_labels.push_back(lptr);
	}

	distance_buffer.waitForSpace(num_queries);
	label_buffer.waitForSpace(num_queries);

	merge(num_queries, all_distances, all_labels, reinterpret_cast<float*>(distance_buffer.peekEnd()),
			reinterpret_cast<faiss::Index::idx_t*>(label_buffer.peekEnd()));

	distance_buffer.add(num_queries / cfg.block_size);
	label_buffer.add(num_queries / cfg.block_size);

	for (int i = 0; i < all_distance_buffers.size(); i++) {
		all_distance_buffers[i]->consume(num_queries);
		all_label_buffers[i]->consume(num_queries);
	}

	sent += num_queries;
	deb("merged queries up to %ld", sent);
}

void HybridSearchStrategy::gpu_process(std::mutex* cleanup_mutex) {
	while (qm->sent_queries() < cfg.test_length) {
		bool emptyQueue = true;

		for (QueryQueue* qq : qm->queues()) {
			if (!qq->on_gpu) continue;

			qq->lock();

			long nq = std::min(qq->size(), best_query_point_gpu);
			assert(nq >= 0);

			qq->search(nq);
			qq->unlock();

			emptyQueue = emptyQueue && nq == 0;
		}

		if (cleanup_mutex->try_lock()) {
			qm->shrinkQueryBuffer();
			qm->mergeResults();
			cleanup_mutex->unlock();
		}

		if (emptyQueue) {
			QueryQueue* qq = qm->biggestCPUQueue();

			if (qq->size() > 1000) { //arbitrary threshold
				qm->switchToGPU(qq);
			}
		}
	}

//	std::printf("bases exchanged: %ld\n", qm->bases_exchanged);
//	for (int j = 0; j < qm->bases_exchanged; j++) {
//		for (int i = 0; i < qm->_queues.size(); i++) {
//			if (i == qm->switches[j].first) {
//				std::printf("<");
//			} else if (i == qm->switches[j].second) {
//				std::printf(">");
//			}
//
//			std::printf("%ld ", qm->log[j][i]);
//		}
//		std::printf("\n");
//	}
}

void HybridSearchStrategy::cpu_process(std::mutex* cleanup_mutex) {
	while (qm->sent_queries() < cfg.test_length) {
		for (QueryQueue* qq : qm->queues()) {
			if (qq->on_gpu) continue;

			qq->lock();

			long nq = std::min(qq->size(), best_query_point_cpu);
			assert(nq >= 0);
			qq->search(nq);
			qq->unlock();
		}

		if (cleanup_mutex->try_lock()) {
			qm->shrinkQueryBuffer();
			qm->mergeResults();
			cleanup_mutex->unlock();
		}
	}
}

void HybridSearchStrategy::setup() {
	qm = new QueueManager(&query_buffer, &label_buffer, &distance_buffer);

	long pieces = cfg.total_pieces;
	float gpu_pieces = cfg.gpu_pieces;
	float step = (base_end - base_start) / pieces;

	for (int i = 0; i < pieces; i++) {
		auto cpu_index = load_index(base_start + i * step, base_start + (i + 1) * step, cfg);
		deb("Creating index from %f to %f", base_start + i * step, base_start + (i + 1) * step);
		QueryQueue* qq = new QueryQueue(cpu_index, qm, i);

		if (i < gpu_pieces) {
			qq->create_gpu_index(*res);
		}
	}
}

void HybridSearchStrategy::start_search_process() {
	std::mutex cleanup_mutex;

	std::thread gpu_thread { &HybridSearchStrategy::gpu_process, this, &cleanup_mutex };
	std::thread cpu_thread { &HybridSearchStrategy::cpu_process, this, &cleanup_mutex };

	cpu_thread.join();
	gpu_thread.join();
}

void CpuOnlySearchStrategy::setup() {
	cpu_index = load_index(base_start, base_end, cfg);
}

void CpuOnlySearchStrategy::start_search_process() {
	long sent = 0;
	while (sent < cfg.test_length) {
		deb("waiting for queries");
		query_buffer.waitForData(1);
		long nblocks = query_buffer.entries();
		long nqueries = nblocks * cfg.block_size;
		deb("%ld queries available", nqueries);

		faiss::Index::idx_t* labels = (faiss::Index::idx_t*) label_buffer.peekEnd();
		float* distances = (float*) distance_buffer.peekEnd();
		float* query_start = (float*) query_buffer.peekFront();

		label_buffer.waitForSpace(nblocks);
		distance_buffer.waitForSpace(nblocks);

		cpu_index->search(nqueries, query_start, cfg.k, distances, labels);

		query_buffer.consume(nblocks);
		label_buffer.add(nblocks);
		distance_buffer.add(nblocks);

		sent += nqueries;
		deb("sent %ld queries", nqueries);
	}
}

void GpuOnlySearchStrategy::merger() {
	std::unique_lock < std::mutex > lck { should_merge_mutex };
	long sent = 0;

	while (sent < cfg.test_length) {
		if (buffer_start_id > sent) {
			merge_procedure(buffer_start_id, sent, all_distance_buffers, all_label_buffers, distance_buffer, label_buffer);
		} else {
			should_merge.wait(lck, [this, &sent] {return buffer_start_id > sent;});
		}
	}

	deb("Sent: %ld", sent);
}

void GpuOnlySearchStrategy::setup() {
	deb("search_gpu called");

	float step = (base_end - base_start) / cfg.total_pieces;

	for (int i = 0; i < cfg.total_pieces; i++) {
		cpu_bases.push_back(load_index(base_start + i * step, base_start + (i + 1) * step, cfg));
		
		if (i < cfg.gpu_pieces) {
			gpu_indexes.push_back(static_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(res, 0, cpu_bases[i], nullptr)));
			baseMap.push_back(i);
			reverseBaseMap.push_back(i);
		} else {
			baseMap.push_back(-1);
		}
		
		all_label_buffers.push_back(new Buffer(sizeof(faiss::Index::idx_t) * cfg.k, 1000000));
		all_distance_buffers.push_back(new Buffer(sizeof(float) * cfg.k, 1000000));
		proc_ids.push_back(0);
	}
}

void GpuOnlySearchStrategy::start_search_process() {
	long bases_exchanged = 0;

	long log[1000][proc_ids.size()];
	std::vector<std::pair<int, int>> switches;

	std::thread merge_thread { &GpuOnlySearchStrategy::merger, this };

	while (buffer_start_id < cfg.test_length) {
		query_buffer.waitForData(1);

		//TODO: subclass buffer and create a QueryBuffer, that deals better with queries (aka. no more "* cfg.d" etc)
		for (int i = 0; i < cpu_bases.size(); i++) {
			long buffer_idx = proc_ids[i] - buffer_start_id;
			long available_queries = query_buffer.entries() * cfg.block_size - buffer_idx;
			auto buffer_ptr = (float*) (query_buffer.peekFront()) + buffer_idx * cfg.d;

			if (available_queries >= 1 && baseMap[i] == -1) {
				switches.push_back(std::make_pair(reverseBaseMap[0], i));
				for (int i = 0; i < proc_ids.size(); i++) {
					log[bases_exchanged][i] = query_buffer.entries() * cfg.block_size - (proc_ids[i] - buffer_start_id);
				}

				gpu_indexes[0]->copyFrom(cpu_bases[i]);
				baseMap[i] = 0;
				baseMap[reverseBaseMap[0]] = -1;
				reverseBaseMap[0] = i;
				
				bases_exchanged++;
			}

			while (available_queries >= 1) {
				long nqueries = std::min(available_queries, best_query_point_gpu);
				auto lb = all_label_buffers[i];
				auto db = all_distance_buffers[i];

				lb->waitForSpace(nqueries);
				db->waitForSpace(nqueries);

				deb("searched %ld queries on base %d", nqueries, i);
				gpu_indexes[baseMap[i]]->search(nqueries, buffer_ptr, cfg.k, (float*) db->peekEnd(), (faiss::Index::idx_t*) lb->peekEnd());

				db->add(nqueries);
				lb->add(nqueries);

				available_queries -= nqueries;
				proc_ids[i] += nqueries;
				buffer_ptr += nqueries * cfg.d;
			}
		}

		auto min_id = *std::min_element(proc_ids.begin(), proc_ids.end());
		if (min_id > buffer_start_id) {
			query_buffer.consume((min_id - buffer_start_id) / cfg.block_size);
			buffer_start_id = min_id;
			deb("buffer start id is now: %ld", buffer_start_id);
			should_merge.notify_one();
		}
	}

	deb("search_gpu finished");

	merge_thread.join();
//
//	std::printf("bases exchanged: %ld\n", bases_exchanged);
//
//	for (int j = 0; j < bases_exchanged; j++) {
//		for (int i = 0; i < proc_ids.size(); i++) {
//			if (i == switches[j].first) {
//				std::printf("<");
//			} else if (i == switches[j].second) {
//				std::printf(">");
//			}
//
//			std::printf("%ld ", log[j][i]);
//		}
//		std::printf("\n");
//	}
}

void CpuFixedSearchStrategy::cpu_process() {
	while (sent < cfg.test_length) {
		//		query_buffer.waitForData(1);
		long buffer_idx = proc_ids[0] - buffer_start_id;
		long available_queries = query_buffer.entries() * cfg.block_size - buffer_idx;

		if (available_queries > 0) {
			auto buffer_ptr = (float*) (query_buffer.peekFront()) + buffer_idx * cfg.d;
			long nqueries = std::min(available_queries, best_query_point_cpu);
			auto lb = all_label_buffers[0];
			auto db = all_distance_buffers[0];

			lb->waitForSpace(nqueries);
			db->waitForSpace(nqueries);

			deb("searching %ld queries on the cpu", nqueries);
			cpu_index->search(nqueries, buffer_ptr, cfg.k, (float*) db->peekEnd(), (faiss::Index::idx_t*) lb->peekEnd());

			lb->add(nqueries);
			db->add(nqueries);

			proc_ids[0] += nqueries;
		}

		auto min_id = *std::min_element(proc_ids.begin(), proc_ids.end());
		if (min_id > buffer_start_id) {
			sync_mutex.lock();
			query_buffer.consume((min_id - buffer_start_id) / cfg.block_size);
			buffer_start_id = min_id;
			sync_mutex.unlock();

			merge_procedure(buffer_start_id, sent, all_distance_buffers, all_label_buffers, distance_buffer, label_buffer);
		}

	}
}

void CpuFixedSearchStrategy::gpu_process() {
	long log[1000][proc_ids.size()];
	std::vector<std::pair<int, int>> switches;
	long bases_exchanged = 0;
	long current_base = 1;

	while (sent < cfg.test_length) {
		for (int i = 1; i <= all_gpu_bases.size(); i++) {
			if (proc_ids[i] == cfg.test_length) continue;

			sync_mutex.lock();
			long buffer_idx = proc_ids[i] - buffer_start_id;
			long available_queries = query_buffer.entries() * cfg.block_size - buffer_idx;
			auto buffer_ptr = (float*) (query_buffer.peekFront()) + buffer_idx * cfg.d;
			sync_mutex.unlock();

			if (available_queries == 0) continue;
			assert(available_queries > 0);

			if (current_base != i) {
				switches.push_back(std::make_pair(current_base, i));
				for (int i = 0; i < proc_ids.size(); i++) {
					log[bases_exchanged][i] = query_buffer.entries() * cfg.block_size - (proc_ids[i] - buffer_start_id);
				}

				gpu_index->copyFrom(all_gpu_bases[i - 1]); // we subtract 1 because this vector doesnt include the cpu entry (whereas the other ones - like proc_ids - do).
				current_base = i;
				bases_exchanged++;
			}

			while (available_queries >= 1) {
				long nqueries = std::min(available_queries, best_query_point_gpu);
				auto lb = all_label_buffers[i];
				auto db = all_distance_buffers[i];

				lb->waitForSpace(nqueries);
				db->waitForSpace(nqueries);

				deb("searching %ld queries on base %d", nqueries, i);
				gpu_index->search(nqueries, buffer_ptr, cfg.k, (float*) db->peekEnd(), (faiss::Index::idx_t*) lb->peekEnd());

				lb->add(nqueries);
				db->add(nqueries);

				available_queries -= nqueries;
				proc_ids[i] += nqueries;
				buffer_ptr += nqueries * cfg.d;
			}
		}
	}

	std::printf("bases exchanged: %ld\n", bases_exchanged);
	for (int j = 0; j < bases_exchanged; j++) {
		for (int i = 0; i < proc_ids.size(); i++) {
			if (i == switches[j].first) {
				std::printf("<");
			} else if (i == switches[j].second) {
				std::printf(">");
			}

			std::printf("%ld ", log[j][i]);
		}
		std::printf("\n");
	}
}

void CpuFixedSearchStrategy::setup() {
	deb("search_gpu called");

	float total_share = base_end - base_start;
	auto cpu_share = (cfg.total_pieces - cfg.gpu_pieces) / static_cast<float>(cfg.total_pieces);

	cpu_index = load_index(base_start, base_start + total_share * cpu_share, cfg);
	deb("loading cpu base from %.2f to %.2f", base_start, base_start + total_share * cpu_share);
	all_label_buffers.push_back(new Buffer(sizeof(faiss::Index::idx_t) * cfg.k, 1000000));
	all_distance_buffers.push_back(new Buffer(sizeof(float) * cfg.k, 1000000));
	proc_ids.push_back(0);

	auto gpu_share = total_share - cpu_share;
	auto gpu_slice = gpu_share / cfg.gpu_pieces;

	for (int i = 0; i < cfg.gpu_pieces; i++) {
		auto start = base_start + cpu_share + gpu_slice * i;
		auto end = start + gpu_slice;
		all_gpu_bases.push_back(load_index(start, end, cfg));
		deb("loading gpu base from %.2f to %.2f", start, end);
		all_label_buffers.push_back(new Buffer(sizeof(faiss::Index::idx_t) * cfg.k, 1000000));
		all_distance_buffers.push_back(new Buffer(sizeof(float) * cfg.k, 1000000));
		proc_ids.push_back(0);
	}

	gpu_index = static_cast<faiss::gpu::GpuIndexIVFPQ*>(faiss::gpu::index_cpu_to_gpu(res, 0, all_gpu_bases[0], nullptr));
}

void CpuFixedSearchStrategy::start_search_process() {
	std::thread cpu_process { &CpuFixedSearchStrategy::cpu_process, this };
	std::thread gpu_process { &CpuFixedSearchStrategy::gpu_process, this };

	cpu_process.join();
	gpu_process.join();
}
