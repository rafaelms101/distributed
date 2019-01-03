#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <sys/mman.h>
#include <faiss/index_io.h>

#include "faiss/Clustering.h"

#include "gpu/GpuAutoTune.h"
#include "gpu/StandardGpuResources.h"
#include "gpu/GpuIndexIVFPQ.h"

#include <omp.h>
#include "readSplittedIndex.h"



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

unsigned char* bvecs_read(const char *fname, size_t* filesize)
{
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

float * fvecs_read (const char *fname, int *d_out, int *n_out){
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
int *ivecs_read(const char *fname, int *d_out, int *n_out)
{
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
	if (argc != 5) {
		std::printf("Usage: ./train size centroids pq nprobe nqueries gpu\n");
		std::exit(-1);
	}
	
    double t0 = elapsed();
    
    int d = 128;                           
    int nb = atoi(argv[1]);                                                                                
    int ncentroids = atoi(argv[2]);
    int nt = std::max(1000000, 256 * ncentroids);
    int m = atoi(argv[3]);    
    int nprobe = atoi(argv[4]);
    
    char src_path[] = "/home/rafael/mestrado/bigann/";
    char learn_path[500];
    char data_path[500];
    char query_path[500];
    char gt_path[500];
    
    sprintf(learn_path, "%s/bigann_learn.bvecs", src_path);
    sprintf(data_path, "%s/bigann_base.bvecs", src_path);
    sprintf(query_path, "%s/bigann_query.bvecs", src_path);
    sprintf(gt_path, "%s/gnd", src_path);
    		
    faiss::gpu::StandardGpuResources res;
    res.setTempMemory(1536 * 1024 * 1024);
    
    char index_path[500];
    sprintf(index_path, "index/index_%d_%d_%d", nb, ncentroids, m);
    
    faiss::Index* index;

	faiss::IndexFlatL2 quantizer(d);

	{ //training
		printf("[%.3f s] Loading training set\n", elapsed() - t0);

		size_t fz;
		unsigned char* train = bvecs_read(learn_path, &fz);
		float* xt = to_float_array(train, nt, d);

		munmap(train, fz);

		auto clus = faiss::Clustering(d, ncentroids);
		clus.verbose = true;
		clus.max_points_per_centroid = 10000000;

		faiss::IndexFlatL2 tmp_quantizer(d);
		auto cpu_gpu = faiss::gpu::index_cpu_to_gpu(&res, 0, &tmp_quantizer,
				nullptr);
		clus.train(nt, xt, *cpu_gpu);

		quantizer.add(ncentroids, &clus.centroids[0]);

		auto i = new faiss::IndexIVFPQ(&quantizer, d, ncentroids, m, 8);
		i->nprobe = nprobe;
		i->verbose = true;
		index = i;
		index->train(nt, xt);

		delete[] xt;
	}

	{ //indexing
		index = faiss::gpu::index_cpu_to_gpu(&res, 0, index, nullptr);
		index->verbose = true;

		printf("[%.3f s] Loading database\n", elapsed() - t0);

		size_t fz;
		unsigned char* base = bvecs_read(data_path, &fz);

		printf("[%.3f s] Indexing database, size %d*%d\n", elapsed() - t0, nb,
				d);

		int max_buffer_size = 1 * 1024 * 1024 * 1024; //1gb
		int elements_by_slice = std::min(max_buffer_size / sizeof(float) / d,
				(size_t) nb);
		int nslices = (nb + elements_by_slice - 1) / elements_by_slice;

		int left = nb;

		for (int si = 0; si < nslices; si++) {
			int add_size = std::min(elements_by_slice, left);
			float* buffer = to_float_array(
					base + (d + 4) * elements_by_slice * si, add_size, d);
			index->add(add_size, buffer);
			left -= add_size;

			delete[] buffer;
		}

		index = faiss::gpu::index_gpu_to_cpu(index);

		munmap(base, fz);
	}

	faiss::write_index(index, index_path);
}
