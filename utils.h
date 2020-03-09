#ifndef UTILS_H_
#define UTILS_H_

#include <cstdio>
#include <vector>

#include "faiss/IndexIVFPQ.h"

#include "config.h"

typedef unsigned char byte;

//#define DEBUG

#ifdef DEBUG
	#define deb(...) do {std::printf("%lf) ", now()); std::printf(__VA_ARGS__); std::printf("\n");} while(0)
#else
	#define deb(...) ;
#endif

constexpr int AGGREGATOR = 0;
constexpr int GENERATOR = 1;

double now();
//TODO: change to receive string instead of const char*
int *ivecs_read(const char* fname, int* d_out, long* n_out);
float *fvecs_read (const char* fname, int* d_out, long* n_out);
unsigned char* bvecs_read(const char* fname, int* d_out, long* n_out);
float* to_float_array(unsigned char* vector, int ne, int d);
double poisson_interval(double mean_interval);
std::pair<int, int> longest_contiguous_region(double tolerance, std::vector<double>& time_per_block);
bool file_exists(char* name);
faiss::IndexIVFPQ* load_index(float start_percent, float end_percent, Config& cfg);

#endif /* UTILS_H_ */
