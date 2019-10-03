#ifndef UTILS_H_
#define UTILS_H_

#include <cstdio>
#include <vector>

#include "config.h"

#include "faiss/IndexIVFPQ.h"

//#define DEBUG

#ifdef DEBUG
	#define deb(...) do {std::printf("%lf) ", now()); std::printf(__VA_ARGS__); std::printf("\n");} while(0)
#else
	#define deb(...) ;
#endif

enum class ProcType {Static, Dynamic, Bench};

constexpr int AGGREGATOR = 0;
constexpr int GENERATOR = 1;

double now();
int *ivecs_read(const char *fname, int *d_out, int *n_out);
float *fvecs_read (const char *fname, int *d_out, int *n_out);
std::vector<double> load_prof_times(Config&); 
faiss::IndexIVFPQ* load_index(float start_percent, float end_percent, Config& cfg);
void load_bench_data(bool cpu, long& best, double& best_time_per_query);
void load_bench_data(bool cpu, long& best);
std::vector<double> load_prof_times(bool gpu, Config& cfg);

#endif /* UTILS_H_ */
