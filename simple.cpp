#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <assert.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <limits>
#include <bits/stdc++.h>

#include <faiss/index_io.h>
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

#include "gpu/GpuAutoTune.h"
#include "gpu/StandardGpuResources.h"
#include "gpu/GpuIndexIVFPQ.h"

#include "config.h"

double elapsed () {
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
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

unsigned char* bvecs_read(const char *fname, size_t* filesize) {
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

float * fvecs_read (const char *fname, int *d_out, int *n_out) {
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

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, int *d_out, int *n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

float* load_queries(int d) {
	char query_path[500];
	sprintf(query_path, "%s/bigann_query.bvecs", SRC_PATH);

	//loading queries
	size_t fz;
	unsigned char* queries = bvecs_read(query_path, &fz);
	float* xq = to_float_array(queries, 10000, d);
	munmap(queries, fz);

	return xq;
}

faiss::Index* load_index(Config& cfg) {
	char index_path[500];
	sprintf(index_path, "%s/index_%d_%d_%d", INDEX_ROOT, cfg.nb, cfg.ncentroids, cfg.m);

	FILE* index_file = fopen(index_path, "r");
	auto cpu_index = faiss::read_index(index_file);

	dynamic_cast<faiss::IndexIVFPQ*>(cpu_index)->nprobe = cfg.nprobe;
	return cpu_index;
}

faiss::Index::idx_t* load_gt(int nb, int k) {
	//	 load ground-truth and convert int to long
	char idx_path[1000];
	char gt_path[500];
	sprintf(gt_path, "%s/gnd", SRC_PATH);
	sprintf(idx_path, "%s/idx_%dM.ivecs", gt_path, nb / 1000000);

	int n_out;
	int db_k;
	int *gt_int = ivecs_read(idx_path, &db_k, &n_out);

	faiss::Index::idx_t* gt = new faiss::Index::idx_t[k * 10000];

	for (int i = 0; i < 10000; i++) {
		for (int j = 0; j < k; j++) {
			gt[i * k + j] = gt_int[i * db_k + j];
		}
	}

	delete[] gt_int;
	return gt;
}

int main(int argc, char** argv) {
	Config cfg;
	
	if (argc != 7) {
		std::printf("Wrong usage.\n./simple <cpu|gpu> <begin> <end> <step> <ncentroids> <nprobe>\n");
	}

	bool gpu = ! strcmp("gpu", argv[1]);
	int begin = atoi(argv[2]);
	int end = atoi(argv[3]);
	int step = atoi(argv[4]);
	
	cfg.ncentroids = atoi(argv[5]);
	cfg.nprobe = atoi(argv[6]);

	faiss::gpu::StandardGpuResources res;
	res.setTempMemory(1350 * 1024 * 1024); 

	auto index = load_index(cfg);
	if (gpu) index = faiss::gpu::index_cpu_to_gpu(&res, 0, index, nullptr);
	float* original_queries = load_queries(cfg.d);

	float* queries = new float[end * 128];

	float* D = new float[cfg.k * end];
	faiss::Index::idx_t* I = new faiss::Index::idx_t[cfg.k * end];

	for (int i = 0; i < end * 128; i++) {
		queries[i] = original_queries[i % (10000 * 128)];
	}

	for (int nq = begin; nq <= end; nq += step) {
		int repeats = 0;
		std::vector<double> times;

		while (repeats <= 3) {
			double before = elapsed();
			index->search(nq, queries, cfg.k, D, I);
			double after = elapsed();

			times.push_back(after - before);

			repeats++;
		}

		std::sort(times.begin(), times.end());

		double median;
		
		if (times.size() % 2 == 0) {
			int right = times.size() / 2;
			int left = right - 1;
			median = (times[left] + times[right]) / 2;
		} else {
			int mid = times.size() / 2;
			median = times[mid];
		}
		
		std::printf("%d %lf\n", nq, median);
	}
}
