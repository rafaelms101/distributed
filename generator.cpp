#include "generator.h"

#include <mpi.h>
#include <sys/mman.h>
#include <cassert>
#include <sys/stat.h>
#include <random>
#include <limits>

static unsigned char* bvecs_read(const char *fname, size_t* filesize) {
	FILE *f = fopen(fname, "rb");
	if (!f) {
		fprintf(stderr, "could not open %s\n", fname);
		perror("");
		abort();
	}

	struct stat st;
	fstat(fileno(f), &st);
	*filesize = st.st_size;
	
	unsigned char* dataset = (unsigned char*) mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fileno(f), 0);
    
    fclose(f);
    
    return dataset;
}

static float* to_float_array(unsigned char* vector, int ne, int d) {
	float* res = new float[ne * d];
	
	for (int i = 0; i < ne; i++) {
		vector += 4;
		
		for (int cd = 0; cd < d; cd++) {
			res[i * d + cd] = *vector;
			vector++;
		}
	}
	
	return res;
}

static float* load_queries(int d, int nq) {
	char query_path[500];
	sprintf(query_path, "%s/bigann_query.bvecs", SRC_PATH);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	//loading queries
	size_t fz;
	unsigned char* queries = bvecs_read(query_path, &fz);
	float* xq = to_float_array(queries, nq, d);
	munmap(queries, fz);
	
	return xq;
}

static void send_queries(int nshards, float* query_buffer, int queries_in_buffer, int d) {
	for (int node = 0; node < nshards; node++) {
		MPI_Ssend(query_buffer, queries_in_buffer * d, MPI_FLOAT, node + 2, 0, MPI_COMM_WORLD);
	}
}

static double constant_interval(double val) {
	return val;
}

static double poisson_interval(double query_rate) {
	static std::default_random_engine generator;
	
	std::exponential_distribution<double> distribution(0.001 / query_rate);
	
	return distribution(generator) / 1000;
}

static double fast_slow_fast_interval(int eval_length) {
	constexpr double slow_time = 0.001;
	constexpr double fast_time = 0.0001;
	double delta_time = (slow_time - fast_time) / (eval_length / 2);

	static bool going_up = false;
	static double curr_time = slow_time;

	if (going_up) {
		curr_time = curr_time + delta_time;
	} else {
		curr_time = curr_time - delta_time;
	}

	if (curr_time > slow_time) {
		deb("Now going up in speed");
		curr_time = slow_time - delta_time;
		going_up = ! going_up;
	} else if (curr_time < fast_time) {
		deb("Now going down in speed");
		curr_time = fast_time + delta_time;
		going_up = ! going_up;
	}

	return curr_time;
}

static std::pair<int, double> next_query(int test_length, double (*next_interval)(double), double func_arg, Config& cfg) {
	static int qty = 0;
	static double query_time = now();
	
	double time_now = now();
	
	if (qty >= test_length) return {-2, -1};
	
	if (time_now < query_time) return {-1, -1};
	
	double old_time = query_time;
	query_time = time_now + next_interval(func_arg);
	
	qty++;
	
	if (qty % 10000 == 0 || (qty % 1000 == 0 && qty <= 10000)) {
		deb("Sent %d/%d queries", qty, test_length);
	}
	
	return {(qty - 1) % cfg.nq, old_time}; 
}

static void send_finished_signal(int nshards) {
	float dummy = 1;
	for (int node = 0; node < nshards; node++) {
		MPI_Ssend(&dummy, 1, MPI_FLOAT, node + 2, 0, MPI_COMM_WORLD);
	}
}

static void bench_generator(int num_queries, int nshards, Config& cfg) {
	auto num_blocks = num_queries / cfg.block_size;
	
	assert(num_queries <= cfg.nq);

	float* xq = load_queries(cfg.d, cfg.nq);

	for (int i = 1; i <= num_blocks; i++) {
		for (int repeats = 1; repeats <= BENCH_REPEATS; repeats++) {
			for (int b = 1; b <= i; b++) {
				send_queries(nshards, xq, cfg.block_size, cfg.d);
			}
		}
	}

	send_finished_signal(nshards);
}


static void single_block_size_generator(int nshards, double (*next_interval)(double), double func_arg, Config& cfg) {
	assert(cfg.test_length % cfg.block_size == 0);
	
	float* xq = load_queries(cfg.d, cfg.nq);

	double start_time[cfg.eval_length], end_time[cfg.eval_length];
	
	int begin_timing = cfg.test_length - cfg.eval_length + 1;

	float* query_buffer = new float[cfg.test_length * cfg.d];
	float* to_be_deleted = query_buffer;

	int queries_in_buffer = 0;
	int qn = 0;
	
	bool first = true;
	
	while (true) {
		std::pair<int, double> p = next_query(cfg.test_length, next_interval, func_arg, cfg);
		auto id = p.first;
		auto query_time = p.second;
		
		if (id == -1) {
			continue;
		}
			
		if (id == -2) {
			if (queries_in_buffer >= 1) send_queries(nshards, query_buffer, queries_in_buffer, cfg.d);
			break;
		}

		qn++;
		
		if (qn >= begin_timing) {
			int offset = qn - begin_timing;
			start_time[offset] = query_time;
		}
		
		memcpy(query_buffer + queries_in_buffer * cfg.d, xq + id * cfg.d, cfg.d * sizeof(float));
		queries_in_buffer++;

		if (queries_in_buffer < cfg.block_size) continue;

		
		if (first) {
			first = false;
			deb("First block sent");
		}
		send_queries(nshards, query_buffer, queries_in_buffer, cfg.d);
		
		query_buffer = query_buffer + queries_in_buffer * cfg.d;
		queries_in_buffer = 0;
	}

	send_finished_signal(nshards);
	
	MPI_Recv(end_time, cfg.eval_length, MPI_DOUBLE, AGGREGATOR, 0, MPI_COMM_WORLD,
			MPI_STATUS_IGNORE);

	double total = 0;

	for (int i = 0; i < cfg.eval_length; i++) {
		total += end_time[i] - start_time[i];
	}

	std::printf("%lf\n", total / cfg.eval_length);
	
	delete [] to_be_deleted;
	delete [] xq;
}

//TODO: make generator a class
void generator(int nshards, ProcType ptype, Config& cfg) {
	double best_time_per_query = std::numeric_limits<double>::max();
			
	if (ptype != ProcType::Bench) {
		std::vector<double> times(load_prof_times(cfg));
		best_time_per_query = times[1];
		for (int i = 2; i < times.size(); i++) {
			double time_per_query = times[i] / (i * cfg.block_size);
			if (time_per_query < best_time_per_query) best_time_per_query = time_per_query;
		}
		
		deb("max_rate = %lf", best_time_per_query);
	}
	
	int shards_ready = 0;
	
	//Waiting for search nodes to be ready
	while (shards_ready < nshards) {
		float dummy;
		MPI_Recv(&dummy, 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		shards_ready++;
	}

	if (ptype == ProcType::Bench) bench_generator(BENCH_SIZE, nshards, cfg);
	else {
		switch (cfg.request_distribution) {
			case RequestDistribution::Constant_Slow: {
				single_block_size_generator(nshards, constant_interval, best_time_per_query / 0.1, cfg);
				break;
			}
			case RequestDistribution::Constant_Average: {
				single_block_size_generator(nshards, constant_interval, best_time_per_query / 0.5, cfg);
				break;
			}
			case RequestDistribution::Constant_Fast: {
				single_block_size_generator(nshards, constant_interval, best_time_per_query / 1, cfg);
				break;
			}
			case RequestDistribution::Variable_Poisson: {
				//to be implemented
				break;
			}
		}
	}
}
