#include "generator.h"

#include <mpi.h>
#include <sys/mman.h>
#include <cassert>
#include <sys/stat.h>
#include <random>
#include <limits>
#include <cstring>

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

static double poisson_constant_interval(double mean_interval) {
	static int index = 0;
	static double time = 0;
	static double interval_length = cfg.test_duration / cfg.poisson_intervals.size();
	
	if (time >= interval_length) {
		index++;
		time -= interval_length;
	}
	
	time += cfg.poisson_intervals[index];
	return cfg.poisson_intervals[index];
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

static int next_query(const int test_length, const double begin, double* start_query_time, Config& cfg) {
	static int qty = 0;
	
	double time_now = now();
	
	if (qty >= test_length) return -2;
	
	if (time_now < begin + start_query_time[qty]) return -1;
	
	qty++;
	
	if (qty % 10000 == 0 || (qty % 1000 == 0 && qty <= 10000)) {
		deb("Sent %d/%d queries", qty, test_length);
	}
	
	return (qty - 1) % cfg.nq; 
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

static void compute_stats(double* start_time, double* end_time, Config& cfg) {
	double aggregate_total = 0;
	double total = 0;
	
	double interval_duration = (start_time[cfg.eval_length - 1] - start_time[0]) * 2 / cfg.poisson_intervals.size();
	double interval_boundary = start_time[0] + interval_duration;
	int nq = 0;
	
	for (int q = 0; q < cfg.eval_length; q++) {
		if (cfg.request_distribution == RequestDistribution::Variable_Poisson && start_time[q] >= interval_boundary) {
			std::printf("%lf\n", total / nq);
			interval_boundary += interval_duration;
			total = 0;
			nq = 0;
		}
		
		nq++;
		auto response_time = end_time[q] - start_time[q];
		total += response_time;
		aggregate_total += response_time;
	}

	std::printf("%lf\n", aggregate_total / cfg.eval_length);
	std::printf("%lf\n", end_time[cfg.eval_length - 1] - start_time[0]);
}

static void single_block_size_generator(int nshards, double* query_start_time, Config& cfg) {
	assert(cfg.test_length % cfg.block_size == 0);
	
	float* xq = load_queries(cfg.d, cfg.nq);
	float* query_buffer = new float[cfg.test_length * cfg.d];
	float* to_be_deleted = query_buffer;
	int queries_in_buffer = 0;
	int qn = 0;
	const double begin_time = now();
	
	while (true) {
		auto id = next_query(cfg.test_length, begin_time, query_start_time, cfg);

		if (id == -1) continue;
			
		if (id == -2) {
			if (queries_in_buffer >= 1) send_queries(nshards, query_buffer, queries_in_buffer, cfg.d);
			break;
		}

		qn++;
		
		memcpy(query_buffer + queries_in_buffer * cfg.d, xq + id * cfg.d, cfg.d * sizeof(float));
		queries_in_buffer++;

		if (queries_in_buffer < cfg.block_size) continue;

		send_queries(nshards, query_buffer, queries_in_buffer, cfg.d);
		query_buffer = query_buffer + queries_in_buffer * cfg.d;
		queries_in_buffer = 0;
	}

	send_finished_signal(nshards);
	
	double end_time[cfg.eval_length];
	MPI_Recv(end_time, cfg.eval_length, MPI_DOUBLE, AGGREGATOR, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	for (int i = 0; i < cfg.test_length; i++) {
		query_start_time[i] = begin_time + query_start_time[i];
	}
	
	compute_stats(&query_start_time[cfg.test_length - cfg.eval_length], end_time, cfg);
	
	delete [] to_be_deleted;
	delete [] xq;
}

static double* query_start_time(double (*next_interval)(double), double param, Config& cfg) {
	double* query_start = new double[cfg.test_length];
	double time = 0;
	
	for (int i = 0; i < cfg.test_length; i++) {
		query_start[i] = time;
		time += next_interval(param);
	}

	return query_start;
}

//TODO: make generator a class
void generator(int nshards, ProcType ptype, Config& cfg) {
	double* query_start;
	
	if (ptype != ProcType::Bench) {
		switch (cfg.request_distribution) {
			case RequestDistribution::Constant: {
				query_start = query_start_time(constant_interval, cfg.query_interval, cfg);
				break;
			}
			case RequestDistribution::Variable_Poisson: {
				query_start = query_start_time(poisson_constant_interval, cfg.query_interval, cfg);
				break;
			}
		}
	}
	
	int shards_ready = 0;
	
	//Waiting for search nodes to be ready
	while (shards_ready < nshards) {
		float dummy;
		MPI_Recv(&dummy, 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		shards_ready++;
	}

	if (ptype == ProcType::Bench) bench_generator(BENCH_SIZE, nshards, cfg);
	else single_block_size_generator(nshards, query_start, cfg);
}
