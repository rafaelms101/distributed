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

#include <faiss/index_io.h>
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

#include "gpu/GpuAutoTune.h"
#include "gpu/StandardGpuResources.h"
#include "gpu/GpuIndexIVFPQ.h"

#include <bits/stdc++.h>


int nb = 500000000;
int ncentroids = 8192;
int m = 8;
int k = 10;
int d = 128;
int nprobe = 8;
char src_path[] = "/home/rafael/mestrado/bigann/";

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

float* load_queries() {
	char query_path[500];
	sprintf(query_path, "%s/bigann_query.bvecs", src_path);

	//loading queries
	size_t fz;
	unsigned char* queries = bvecs_read(query_path, &fz);
	float* xq = to_float_array(queries, 10000, d);
	munmap(queries, fz);

	return xq;
}

auto load_index() {
	char index_path[500];
	sprintf(index_path, "index/index_%d_%d_%d", nb, ncentroids, m);

	FILE* index_file = fopen(index_path, "r");
	auto cpu_index = faiss::read_index(index_file);

	dynamic_cast<faiss::IndexIVFPQ*>(cpu_index)->nprobe = nprobe;
	return cpu_index;
}

faiss::Index::idx_t* load_gt() {
	//	 load ground-truth and convert int to long
	char idx_path[1000];
	char gt_path[500];
	sprintf(gt_path, "%s/gnd", src_path);
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
	int nq = atoi(argv[1]);

	faiss::gpu::StandardGpuResources res;
	res.noTempMemory();
	auto cpu_index = load_index();
	auto gpu_index = faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index, nullptr);

	float* original_queries = load_queries();

	int length = nq * 128;
	float* queries = new float[length];

	for (int i = 0; i < length; i++) {
		queries[i] = original_queries[i % (10000 * 128)];
	}

	int repeats = 0;

	float* D = new float[k * nq];
	faiss::Index::idx_t* I = new faiss::Index::idx_t[k * nq];

	std::vector<double> times;

	while (repeats <= 10) {
		double before = elapsed();
		gpu_index->search(nq, queries, k, D, I);
		double after = elapsed();

		times.push_back(after - before);

		repeats++;
	}

	std::sort(times.begin(), times.end());


	if (times.size() % 2 == 0) {
		int right = times.size() / 2;
		int left = right - 1;

		std::printf("%lf\n", (times[left] + times[right]) / 2);

	} else {
		int mid = times.size() / 2;
		std::printf("%lf\n", times[mid]);
	}

	std::printf("finished\n");
}
