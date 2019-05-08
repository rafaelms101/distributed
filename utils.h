#ifndef UTILS_H_
#define UTILS_H_

#include <cstdio>
#include <vector>

#include "config.h"

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

#endif /* UTILS_H_ */
