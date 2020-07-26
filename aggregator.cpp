#include "aggregator.h"

#include <vector>
#include <mpi.h>
#include <list>
#include <queue>
#include <map>

#include "faiss/IndexIVFPQ.h"

namespace {
	struct BlockResult {
		float* dists;
		faiss::Index::idx_t* ids;
	};
}

static void aggregate_query(float** dists, faiss::Index::idx_t** ids, int nshards, faiss::Index::idx_t* answers) {
	int counter[nshards];
	for (int i = 0; i < nshards; i++) counter[i] = 0;

	for (int topi = 0; topi < cfg.k; topi++) {
		float bestDist = std::numeric_limits<float>::max();
		long bestId = -1;
		int fromShard = -1;

		for (int shard = 0; shard < nshards; shard++) {
			if (counter[shard] == cfg.k) continue;

			if (dists[shard][counter[shard]] < bestDist) {
				bestDist = dists[shard][counter[shard]];
				bestId = ids[shard][counter[shard]];
				fromShard = shard;
			}
		}

		answers[topi] = bestId;
		counter[fromShard]++;
	}
}

static void aggregate_block(std::list<BlockResult>& results, int nshards, faiss::Index::idx_t* answers, double* end_times) {
	float* distances[cfg.block_size * nshards];
	faiss::Index::idx_t* ids[cfg.block_size * nshards];
	int shard = 0;
	
	for (const BlockResult& result : results) {
		for (int q = 0; q < cfg.block_size; q++) {
			distances[q * nshards + shard] = &result.dists[q * cfg.k];
			ids[q * nshards + shard] = &result.ids[q * cfg.k];
		}	
		
		shard++;
	}
	
	for (int q = 0; q < cfg.block_size; q++) {
		aggregate_query(&distances[q * nshards], &ids[q * nshards], nshards, &answers[q * cfg.k]);
		end_times[q] = now();
	}
}


static faiss::Index::idx_t* load_gt(Config& cfg) {
	long n_out;
	int db_k;
	int *gt_int = ivecs_read(cfg.gnd_path.c_str(), &db_k, &n_out);

	faiss::Index::idx_t* gt = new faiss::Index::idx_t[cfg.k * cfg.distinct_queries];

	for (int i = 0; i < cfg.distinct_queries; i++) {
		for (int j = 0; j < cfg.k; j++) {
			gt[i * cfg.k + j] = gt_int[i * db_k + j];
		}
	}

	delete[] gt_int;
	return gt;
}

//TODO: make this work for generic k's
static void show_recall(faiss::Index::idx_t* answers, Config& cfg) {
	auto gt = load_gt(cfg);

	int n_1 = 0, n_10 = 0, n_100 = 0;
	
	for (int id = 0; id < cfg.num_blocks * cfg.block_size; id++) {
		int nq = id % cfg.distinct_queries;
		int gt_nn = gt[nq * cfg.k];
		
		for (int j = 0; j < cfg.k; j++) {
			if (answers[id * cfg.k + j] == gt_nn) {
				if (j < 1) n_1++;
				if (j < 10) n_10++;
				if (j < 100) n_100++;
			}
		}
	}
	
	std::printf("R@1 = %.4f\n", n_1 / float(cfg.num_blocks * cfg.block_size));
	
	if (cfg.k >= 10) {
		std::printf("R@10 = %.4f\n", n_10 / float(cfg.num_blocks * cfg.block_size));
	}
	
	if (cfg.k >= 100) {
		std::printf("R@100 = %.4f\n", n_100 / float(cfg.num_blocks * cfg.block_size));
	}
	
	
	delete [] gt;
}

void aggregator() {
	auto target_delta = cfg.num_blocks * cfg.block_size / 10;
	auto target = target_delta;
	
	double end_times[cfg.num_blocks * cfg.block_size];

	faiss::Index::idx_t* answers = new faiss::Index::idx_t[cfg.num_blocks * cfg.block_size * cfg.k];
	std::map<long, std::list<BlockResult>> block_queue;

	long queries_remaining = cfg.num_blocks * cfg.block_size;
	
	std::vector<long> remaining_queries_per_shard(cfg.nshards);
	for (int shard = 0; shard < cfg.nshards; shard++) {
		remaining_queries_per_shard[shard] = cfg.num_blocks * cfg.block_size;
	}

	while (queries_remaining >= 1) {
		MPI_Status status;
		
		auto I = new faiss::Index::idx_t[cfg.k * cfg.block_size];
		auto D = new float[cfg.k * cfg.block_size];
		long block_id;

		MPI_Recv(&block_id, 1, MPI_LONG, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(I, cfg.k * cfg.block_size, MPI_LONG, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D, cfg.k * cfg.block_size, MPI_FLOAT, status.MPI_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		int shard = status.MPI_SOURCE - 2;
		remaining_queries_per_shard[shard] -= cfg.block_size;
		
		auto it = block_queue.find(block_id);
		
		if (it == block_queue.end()) {
			block_queue[block_id] = {{D, I}};
		} else {
			block_queue[block_id].push_back({D, I});
		}

		if (block_queue[block_id].size() == cfg.nshards) {
			aggregate_block(block_queue[block_id], cfg.nshards, &answers[block_id * cfg.block_size * cfg.k], &end_times[block_id * cfg.block_size]);
			for (const BlockResult& result : block_queue[block_id]) {
				delete[] result.dists;
				delete[] result.ids;
			}

			block_queue.erase(block_id);
			queries_remaining -= cfg.block_size;
		}
	}
	
	if (cfg.exec_type != ExecType::Bench) {
		if (cfg.show_recall) show_recall(answers, cfg);
		MPI_Send(end_times, cfg.num_blocks * cfg.block_size, MPI_DOUBLE, GENERATOR, 0, MPI_COMM_WORLD);
	}
}
