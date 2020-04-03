#include "utils.h"

#include <sys/time.h>
#include <cstdlib>
#include <cassert>
#include <sys/stat.h>
#include <cstring>
#include <fstream>
#include <random>
#include <sys/stat.h>
#include <mpi.h>
#include <sys/mman.h>

#include "readSplittedIndex.h"
#include "faiss/IndexIVFPQ.h"

#include "config.h"

double now() {
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}


// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char *fname, int *d_out, long *n_out) {
	assert(sizeof(int) == sizeof(float));
    return (int*)fvecs_read(fname, d_out, n_out);
}

float* fvecs_read (const char *fname, int *d_out, long *n_out) {
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;	

    assert(fread(&d, 1, sizeof(int), f) == sizeof(int) || !"could not read d");
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = (int) d; *n_out = (int) n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

unsigned char* bvecs_read(const char *fname, int* d_out, long* n_out) {
	FILE *f = fopen(fname, "rb");
	
	if (!f) {
		fprintf(stderr, "could not open %s\n", fname);
		perror("");
		abort();
	}

	struct stat st;
	fstat(fileno(f), &st);
	auto filesize = st.st_size;
	
	unsigned char* dataset = (unsigned char*) mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fileno(f), 0);
	int* intptr = (int*) dataset;
	*d_out = intptr[0];
	*n_out = filesize / (4 + *d_out);
    
    fclose(f);
    
    return dataset;
}

float* to_float_array(unsigned char* vector, int ne, int d) {
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

double poisson_interval(double mean_interval) {
	double r = 0;
	
	while (r == 0) r = static_cast<double>(rand()) / RAND_MAX;
	auto ret = -std::log(r) * mean_interval;
	ret = std::min(ret, 5 * mean_interval);
	ret = std::max(ret, mean_interval / 5);
	return ret;
}

std::pair<int, int> longest_contiguous_region(double tolerance, std::vector<double>& time_per_block) {
	int min_block = 1;

	for (int nb = 1; nb < time_per_block.size(); nb++) {
		if (time_per_block[nb] < time_per_block[min_block]) min_block = nb;
	}
	
	double min = time_per_block[min_block];
	int start, bestStart, bestEnd;
	int bestLength = 0;
	int length = 0;

	double threshold = min * (1 + tolerance);

	for (int i = 1; i < time_per_block.size(); i++) {
		if (time_per_block[i] <= threshold) {
			length++;

			if (length > bestLength) {
				bestStart = start;
				bestEnd = i;
				bestLength = length;
			}
		} else {
			start = i + 1;
			length = 0;
		}
	}

	return std::pair<int, int>(bestStart, bestEnd);
}

bool file_exists(const char* name) {
  struct stat buffer;   
  return stat(name, &buffer) == 0; 
}

faiss::IndexIVFPQ* load_index(float start_percent, float end_percent, Config& cfg) {
	deb("Started loading");

	if (! file_exists(cfg.index_path.c_str())) {
		std::printf("%s doesnt exist\n", cfg.index_path.c_str());
	}
	
//	std::printf("%d) Loading file: %s\n", cfg.shard, index_path);
	
	auto before = now();
	
	FILE* index_file = fopen(cfg.index_path.c_str(), "r");
	auto cpu_index = dynamic_cast<faiss::IndexIVFPQ*>(read_index(index_file, start_percent, end_percent));

//	std::printf("%d) Load finished. Took %lf\n", cfg.shard, now() - before);

	//TODO: Maybe we shouldnt set nprobe here
	dynamic_cast<faiss::IndexIVFPQ*>(cpu_index)->nprobe = cfg.nprobe;
	return cpu_index;
}
