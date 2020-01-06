#include "aggregator.h"

#include <vector>
#include <queue>
#include <mpi.h>

#include "faiss/IndexIVFPQ.h"

namespace {
	struct PartialResult {
		float* dists;
		long* ids;
		bool own_fields;
		float* base_dists;
		long* base_ids;
	};
}

static void merge_results(std::vector<PartialResult>& results, faiss::Index::idx_t* answers, int nshards, int k) {
	int counter[nshards];
	for (int i = 0; i < nshards; i++) counter[i] = 0;

	for (int topi = 0; topi < k; topi++) {
		float bestDist = std::numeric_limits<float>::max();
		long bestId = -1;
		int fromShard = -1;

		for (int shard = 0; shard < nshards; shard++) {
			if (counter[shard] == k) continue;

			if (results[shard].dists[counter[shard]] < bestDist) {
				bestDist = results[shard].dists[counter[shard]];
				bestId = results[shard].ids[counter[shard]];
				fromShard = shard;
			}
		}

		answers[topi] = bestId;
		counter[fromShard]++;
	}
}

static void aggregate_query(std::queue<PartialResult>* queue, int nshards, faiss::Index::idx_t* answers, int k) {
	std::vector<PartialResult> results(nshards);
	
	for (int shard = 0; shard < nshards; shard++) {
		results[shard] = queue[shard].front();
		queue[shard].pop();
	}
				
	merge_results(results, answers, nshards, k);
	
	for (int shard = 0; shard < nshards; shard++) {
		if (results[shard].own_fields) {
			delete [] results[shard].base_dists;
			delete [] results[shard].base_ids;
		}	
	}
}

static faiss::Index::idx_t* load_gt_tinydb(Config& cfg) {
	//	 load ground-truth and convert int to long
	char idx_path[1000];
	char gt_path[500];
	sprintf(gt_path, "%s/gnd", SRC_PATH);
	sprintf(idx_path, "%s/gt", gt_path);

	int n_out;
	int db_k;
	int *gt_int = ivecs_read(idx_path, &db_k, &n_out);

	faiss::Index::idx_t* gt = new faiss::Index::idx_t[cfg.k * cfg.nq];

	for (int i = 0; i < cfg.nq; i++) {
		for (int j = 0; j < cfg.k; j++) {
			gt[i * cfg.k + j] = gt_int[i * db_k + j];
		}
	}

	delete[] gt_int;
	return gt;
}

static faiss::Index::idx_t* load_gt(Config& cfg) {
	//	 load ground-truth and convert int to long
	char idx_path[1000];
	char gt_path[500];
	sprintf(gt_path, "%s/gnd", SRC_PATH);
	sprintf(idx_path, "%s/idx_%dM.ivecs", gt_path, cfg.nb / 1000000);

	int n_out;
	int db_k;
	int *gt_int = ivecs_read(idx_path, &db_k, &n_out);

	faiss::Index::idx_t* gt = new faiss::Index::idx_t[cfg.k * cfg.nq];

	for (int i = 0; i < cfg.nq; i++) {
		for (int j = 0; j < cfg.k; j++) {
			gt[i * cfg.k + j] = gt_int[i * db_k + j];
		}
	}

	delete[] gt_int;
	return gt;
}

static void send_times(std::deque<double>& end_times, int eval_length) {
	double end_times_array[eval_length];

	for (int i = 0; i < eval_length; i++) {
		end_times_array[i] = end_times.front();
		end_times.pop_front();
	}

	MPI_Send(end_times_array, eval_length, MPI_DOUBLE, GENERATOR, 0, MPI_COMM_WORLD);
}

//TODO: make this work for generic k's
static void show_recall(faiss::Index::idx_t* answers, Config& cfg) {
	auto gt = load_gt_tinydb(cfg);

	int n_1 = 0, n_10 = 0, n_100 = 0;
	
	for (int i = 0; i < cfg.num_blocks * cfg.block_size; i++) {
		int answer_id = i % (cfg.num_blocks * cfg.block_size);
		int nq = i % cfg.nq;
		int gt_nn = gt[nq * cfg.k];
		
		for (int j = 0; j < cfg.k; j++) {
			if (answers[answer_id * cfg.k + j] == gt_nn) {
				if (j < 1) n_1++;
				if (j < 10) n_10++;
				if (j < 100) n_100++;
			}
		}
	}

	deb("R@1 = %.4f", n_1 / float(cfg.num_blocks * cfg.block_size));
	deb("R@10 = %.4f", n_10 / float(cfg.num_blocks * cfg.block_size));
	deb("R@100 = %.4f", n_100 / float(cfg.num_blocks * cfg.block_size));
	
	delete [] gt;
}

void aggregator(int nshards, Config& cfg) {
	auto target_delta = cfg.num_blocks * cfg.block_size / 10;
	auto target = target_delta;
	
	std::deque<double> end_times;

	faiss::Index::idx_t* answers = new faiss::Index::idx_t[cfg.num_blocks * cfg.block_size * cfg.k];
	
	std::queue<PartialResult> queue[nshards];
	std::queue<PartialResult> to_delete;
	
	deb("Aggregator node is ready");

	long queries_remaining = cfg.num_blocks * cfg.block_size;
	long qn = 0;
	
	double time = 0;
	
	std::vector<long> remaining_queries_per_shard(nshards);
	for (int shard = 0; shard < nshards; shard++) {
		remaining_queries_per_shard[shard] = cfg.num_blocks * cfg.block_size;
	}
	
	while (queries_remaining >= 1) {
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		
		int message_size;
		MPI_Get_count(&status, MPI_LONG, &message_size);

		int qty = message_size / cfg.k;
		
		auto I = new faiss::Index::idx_t[cfg.k * qty];
		auto D = new float[cfg.k * qty];

		MPI_Recv(I, cfg.k * qty, MPI_LONG, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D, cfg.k * qty, MPI_FLOAT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		int shard = status.MPI_SOURCE - 2;
		remaining_queries_per_shard[shard] -= qty;
		
		for (int q = 0; q < qty; q++) {
			queue[shard].push({ D + cfg.k * q, I + cfg.k * q, q == qty - 1, D, I });
		}
		
		while (true) {
			bool hasEmpty = false;

			for (int i = 0; i < nshards; i++) {
				if (queue[i].empty()) {
					hasEmpty = true;
					break;
				}
			}
			
			if (hasEmpty) break;

			double before = now();
			
			aggregate_query(queue, nshards, answers + (qn % (cfg.num_blocks * cfg.block_size)) * cfg.k, cfg.k);
			queries_remaining--;
			qn++;
			
			end_times.push_back(now());
			
			if (qn >= target) {
//				std::printf("%d queries remaining\n", queries_remaining);
				target += target_delta;
			}
			
			time += now() - before;
		}
	}
	
	deb("aggregating took %lf", time);
	
	if (cfg.exec_type != ExecType::Bench) {
//		show_recall(answers, cfg);
		send_times(end_times, cfg.num_blocks * cfg.block_size);
	}
	
	deb("Finished aggregator");
}
