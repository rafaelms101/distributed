#ifndef UTILS_H_
#define UTILS_H_

#include <cstdio>

#define DEBUG

#ifdef DEBUG
	#define deb(...) do {std::printf("%lf) ", now()); std::printf(__VA_ARGS__); std::printf("\n");} while(0)
#else
	#define deb(...) ;
#endif

enum class ProcType {Static, Dynamic, Bench};

constexpr int AGGREGATOR = 0;
constexpr int GENERATOR = 1;

constexpr int BENCH_SIZE = 10000; 

constexpr char src_path[] = "/home/rafael/mestrado/bigann/";

constexpr int bench_repeats = 3;

struct Config {
	int d;
	int nb;
	int ncentroids;
	int m;
	int k;
	int nprobe;
	int block_size;
	int test_length;
	int eval_length;
	int nq;
	
	double query_rate;
	
	int processing_size;
	
};

double now();
int *ivecs_read(const char *fname, int *d_out, int *n_out);
float * fvecs_read (const char *fname, int *d_out, int *n_out);

#endif /* UTILS_H_ */
