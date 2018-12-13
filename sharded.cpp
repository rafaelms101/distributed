#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

#include "gpu/GpuAutoTune.h"
#include "gpu/StandardGpuResources.h"
#include "gpu/GpuIndexIVFPQ.h"

#include "readSplittedIndex.h"
#include "pqueue.h"

char src_path[] = "/home/rafaelm/Downloads/bigann/";
int nq = 10000;
int d = 128;
int k = 1000;
int nb = 1000000; 
int ncentroids = 256; 
int m = 8;
bool gpu = false;
int nprobe = 4;

void generator(MPI_Comm search_comm);
void aggregator(int nshards);
void search(MPI_Comm search_comm, int nshards);

// aggregator, generator, search

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

int main() {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    
    MPI_Group search_group;
    int ranks[] = {0};
    MPI_Group_excl(world_group, 1, ranks, &search_group);
    
    MPI_Comm search_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, search_group, 0, &search_comm);
    

    if (world_rank == 1) {
    	generator(search_comm);
    } else if (world_rank == 0) {
    	aggregator(world_size - 2);
    } else {
    	search(search_comm, world_size - 2);
    }
    
    // Finalize the MPI environment.
    MPI_Finalize();
}

void generator(MPI_Comm search_comm) {
	char query_path[500];
	sprintf(query_path, "%s/bigann_query.bvecs", src_path);
	
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	//loading queries
	size_t fz;
	unsigned char* queries = bvecs_read(query_path, &fz);
	float* xq = to_float_array(queries, nq, d);
	munmap(queries, fz);
	
	MPI_Bcast(&nq, 1, MPI_INT, 0, search_comm);
	MPI_Bcast(xq, nq * d, MPI_FLOAT, 0, search_comm);

	std::printf("sent everything\n");
}

void search(MPI_Comm search_comm, int nshards) {
	faiss::gpu::StandardGpuResources res;
	res.setTempMemory(1536 * 1024 * 1024);
	
	int search_rank;
	MPI_Comm_rank(search_comm, &search_rank);
	int shard = search_rank - 1;
	
	char index_path[500];
	sprintf(index_path, "/tmp/index_%d_%d_%d", nb, ncentroids, m);
	FILE* index_file = fopen(index_path, "r");
	auto index = read_index(index_file, shard, nshards);
	dynamic_cast<faiss::IndexIVFPQ*>(index)->nprobe = nprobe;
	
	if (gpu) index = faiss::gpu::index_cpu_to_gpu(&res, 0, index, nullptr);
	
	int nq;
	MPI_Bcast(&nq, 1, MPI_INT, 0, search_comm);
	
	float* xq = new float[nq * d];
	MPI_Bcast(xq, nq * d, MPI_FLOAT, 0, search_comm);
	
	std::printf("received everything\n");
	
	

	faiss::Index::idx_t *I = new faiss::Index::idx_t[k * nq];
	float *D = new float[k * nq];
	index->search(nq, xq, k, D, I);

	std::printf("done searching\n");
	
	MPI_Send(&nq, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	MPI_Send(I, k * nq, MPI_LONG, 0, 1, MPI_COMM_WORLD);
	MPI_Send(D, k * nq, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
	
	delete xq;
	std::printf("done sending results\n");
}

void aggregator(int nshards) {
	// load ground-truth and convert int to long
	char idx_path[1000];
	char gt_path[500];
	sprintf(gt_path, "%s/gnd", src_path);
	sprintf(idx_path, "%s/idx_%dM.ivecs", gt_path, nb / 1000000);

	int *gt_int = ivecs_read(idx_path, &k, &nq);

	faiss::Index::idx_t* gt = new faiss::Index::idx_t[k * nq];

	for (int i = 0; i < k * nq; i++) {
		gt[i] = gt_int[i];
	}

	delete[] gt_int;
		
	int total = 0;

	std::pair<float, long>* bases = new std::pair<float, long>[nq * k];
	pqueue* pq = static_cast<pqueue*>(malloc(sizeof(pqueue) * nq));
	
	for (int q = 0; q < nq; q++) {
		pq[q] = pqueue(bases + k * q, k);
	}
	
	faiss::Index::idx_t* finalI = new faiss::Index::idx_t[k * nq];
	float* finalD = new float[k * nq];
	
	while (total != nshards) {
		MPI_Status status;
		int nq;
		MPI_Recv(&nq, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				
		faiss::Index::idx_t* I = new faiss::Index::idx_t[k * nq];
		float* D = new float[k * nq];
		
		MPI_Recv(I, k * nq, MPI_LONG, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D, k * nq, MPI_FLOAT, status.MPI_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		for (int q = 0; q < nq; q++) {
			for (int j = 0; j < k; j++) {
				pq[q].add(D[q*k+j], I[q*k+j]);
			}
			
		}
		
		total++;
		
		delete D;
		delete I;
	}
	
	for (int q = 0; q < nq; q++) {
		for (int j = k - 1; j >= 0; j--) {
			auto v = pq[q].pop();
			finalD[q*k+j] = v.first;
			finalI[q*k+j] = v.second;
		}
	}
	
	std::printf("aggregated everything\n");
	
	int n_1 = 0, n_10 = 0, n_100 = 0;
	for (int i = 0; i < nq; i++) {
		int gt_nn = gt[i * k];
		for (int j = 0; j < k; j++) {
			if (finalI[i * k + j] == gt_nn) {
				if (j < 1)
					n_1++;
				if (j < 10)
					n_10++;
				if (j < 100)
					n_100++;
			}
		}
	}

	printf("R@1 = %.4f\n", n_1 / float(nq));
	printf("R@10 = %.4f\n", n_10 / float(nq));
	printf("R@100 = %.4f\n", n_100 / float(nq));
}
