#include "generator.h"

#include <mpi.h>
#include <sys/mman.h>
#include <cassert>
#include <sys/stat.h>
#include <random>
#include <limits>
#include <cstring>
#include "ExecPolicy.h"

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

static void send_queries(int nshards, float* query_buffer, int d) {
	MPI_Bcast(query_buffer, cfg.block_size * d, MPI_FLOAT, 0, cfg.search_comm);
}

static double constant_interval(double val) {
	return val;
}

static double poisson_constant_interval(double mean_interval) {
	static int block_size = cfg.num_blocks * cfg.block_size / cfg.poisson_intervals;
	static int nq = block_size;
	static double current_interval = 0;
	
	if (nq == block_size) {
		nq = 0;
		current_interval = poisson_interval(mean_interval);
	}
	
	nq++;
	return current_interval;
}

static int next_query(const int test_length, double* start_query_time, Config& cfg) {
	static int qn = 0;
	
	double time_now = now();
	
	if (qn >= test_length) return -2;
	
	if (time_now < start_query_time[qn]) return -1;
	
	qn++;
	
	if (qn % 10000 == 0 || (qn % 1000 == 0 && qn <= 10000)) {
		deb("Sent %d/%d queries", qn, test_length);
	}
	
	return (qn - 1) % cfg.nq; 
}

static void bench_generator(int num_queries, int nshards, Config& cfg) {
	float* xq = load_queries(cfg.d, cfg.nq);
	int nq = cfg.bench_step;
	
	assert(num_queries % cfg.bench_step == 0);
	
	while (nq <= num_queries) {
		auto num_blocks = nq / cfg.block_size;
		
		for (int repeats = 1; repeats <= BENCH_REPEATS; repeats++) {
			for (int b = 1; b <= num_blocks; b++) {
				send_queries(nshards, xq, cfg.d);
			}
		}

		nq += cfg.bench_step; 
	}
}

static void compute_stats(double* start_time, double* end_time, Config& cfg) {
	double total = 0;
	double block_total = 0;
	int nq_block = 0;
	int block_size = cfg.num_blocks * cfg.block_size / cfg.poisson_intervals;
	
	for (int q = 0; q < cfg.num_blocks * cfg.block_size; q++) {
		auto response_time = end_time[q] - start_time[q];
		total += response_time;
		block_total += response_time;
		nq_block++;
		
		if (cfg.request_distribution == RequestDistribution::Variable_Poisson
				&& nq_block == block_size) {
			std::printf("%lf\n", block_total / nq_block);
			block_total = 0;
			nq_block = 0;
		}
	}
	
	std::printf("%lf\n", total / (cfg.num_blocks * cfg.block_size));
	std::printf("%lf\n", end_time[(cfg.num_blocks * cfg.block_size) - 1] - start_time[0]);
	
	std::fflush(stdout);
}

static void single_block_size_generator(int nshards, double* query_start_time, Config& cfg) {
	float* xq = load_queries(cfg.d, cfg.nq);
	float* query_buffer = new float[cfg.num_blocks * cfg.block_size * cfg.d];
	float* to_be_deleted = query_buffer;
	int queries_in_buffer = 0;
	int qn = 0;
	double begin_time = now();
	
	for (int i = 0; i < cfg.num_blocks * cfg.block_size; i++) {
		query_start_time[i] += begin_time;
	}
	
	while (true) {
		auto id = next_query(cfg.num_blocks * cfg.block_size, query_start_time, cfg);

		if (id == -1) continue;
			
		if (id == -2) {
			if (queries_in_buffer >= 1) send_queries(nshards, query_buffer, cfg.d);
			break;
		}

		qn++;
		
		memcpy(query_buffer + queries_in_buffer * cfg.d, xq + id * cfg.d, cfg.d * sizeof(float));
		queries_in_buffer++;

		if (queries_in_buffer < cfg.block_size) continue;

		send_queries(nshards, query_buffer, cfg.d);
		query_buffer = query_buffer + queries_in_buffer * cfg.d;
		queries_in_buffer = 0;
	}

	double end_time[cfg.num_blocks * cfg.block_size];
	MPI_Recv(end_time, cfg.num_blocks * cfg.block_size, MPI_DOUBLE, AGGREGATOR, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	compute_stats(&query_start_time[0], end_time, cfg);
	
	delete [] to_be_deleted;
	delete [] xq;
}

static double* query_start_time(double (*next_interval)(double), double param, Config& cfg) {
	double* query_start = new double[cfg.num_blocks * cfg.block_size];
	double time = 0;
	
	for (int i = 0; i < cfg.num_blocks * cfg.block_size; i++) {
		query_start[i] = time;
		time += next_interval(param);
	}

	return query_start;
}

//TODO: make generator a class
void generator(int nshards, Config& cfg) {
	double* query_start;
	
	if (cfg.exec_type != ExecType::Bench) {
		double query_interval = 0;

		if (cfg.query_load > 0) {
			auto times = BenchExecPolicy::load_prof_times(true, cfg);

			double best = times[1];
			for (int i = 1; i < times.size(); i++) {
				times[i] = times[i] / (i * cfg.block_size);
				deb("%d: %lf", i * cfg.block_size, times[i]);
				if (times[i] < best) best = times[i];
			}

			deb("best interval: %lf", best);
			query_interval = best / cfg.query_load;
		}

		deb("interval: %lf", query_interval);
		
		
		switch (cfg.request_distribution) {
			case RequestDistribution::Constant: {
				query_start = query_start_time(constant_interval, query_interval, cfg);
				break;
			}
			case RequestDistribution::Variable_Poisson: {
				query_start = query_start_time(poisson_constant_interval, query_interval, cfg);
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

	if (cfg.exec_type == ExecType::Bench) bench_generator(BENCH_SIZE, nshards, cfg);
	else single_block_size_generator(nshards, query_start, cfg);
}
