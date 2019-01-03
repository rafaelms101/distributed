#include <mpi.h>
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

#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

#include "gpu/GpuAutoTune.h"
#include "gpu/StandardGpuResources.h"
#include "gpu/GpuIndexIVFPQ.h"

#include "readSplittedIndex.h"
#include "pqueue.h"
#include <utility>

#include <tuple>

char src_path[] = "/home/rafael/mestrado/bigann/";
int nq = 10000;
int d = 128;
int k = 1000;
int nb = 1000000; 
int ncentroids = 256; 
int m = 8;
bool gpu = true;
int nprobe = 4;

void generator(MPI_Comm search_comm);
void aggregator(int nshards);
void search(MPI_Comm search_comm, int nshards);

// aggregator, generator, search

double elapsed ()
{
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
//	std::printf("started\n");
	srand(1);
	
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

double uniform_interval(double val) {
	return val;
}

double poisson_interval(double lambda) {
	double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	return - log(r) / lambda;
}

void mock_generator(MPI_Comm search_comm) {
	static int qb = 0;

	int q = 2;
	MPI_Bcast(&q, 1, MPI_INT, 0, search_comm);
	
	float query[q * d];
	for (int i = 0; i < q * d; i++) query[i] = 0;
	
	qb++;

	MPI_Bcast(query, q * d, MPI_FLOAT, 0, search_comm);
	std::printf("sent:%d\n", qb);
}

int next_query() {
	static int qn = 0;
	static double start = 0;
	static double interval = 0;
	
	if (qn == nq) return -2;
	
	double now = elapsed();
	
	if (now - start < interval) return -1;
	
	start = now;
	interval = uniform_interval(0);
	
	qn++;
	return qn - 1; 
}

float* load_queries() {
	char query_path[500];
	sprintf(query_path, "%s/bigann_query.bvecs", src_path);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	//loading queries
	size_t fz;
	unsigned char* queries = bvecs_read(query_path, &fz);
	float* xq = to_float_array(queries, nq, d);
	munmap(queries, fz);
	
	return xq;
}

void generator(MPI_Comm search_comm) {
//	mock_generator(search_comm);
	
	double start_time[nq];
	double end_time[nq];
	
	float* xq = load_queries();
	
	int id = -1;
	
	while (true) {
		int id = next_query();
		
		if (id == -1) continue;
		if (id == -2) break;
		
		start_time[id] = elapsed();
		
		// do what you must
		int qty = 1;

		MPI_Bcast(&qty, 1, MPI_INT, 0, search_comm);
		MPI_Bcast(xq + id * d, d, MPI_FLOAT, 0, search_comm);
	}
	
	MPI_Recv(end_time, nq, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	double total = 0;
	
	for (int i = 0; i < nq; i++) {
		total += end_time[i] - start_time[i];
	}
	
	std::printf("Average response time: %lf\n", total / nq);
	std::printf("Total time: %lf\n", end_time[nq - 1] - start_time[0]);
}

void mock_search(MPI_Comm search_comm) {
	while (true) {
		int qty;
		MPI_Bcast(&qty, 1, MPI_INT, 0, search_comm);

		float query_buffer[qty * d];
		MPI_Bcast(query_buffer, qty * d, MPI_FLOAT, 0, search_comm);
		
		std::printf("received queries\n");

		faiss::Index::idx_t I[k * qty];
		float D[k * qty];
		
		for (int i = 0; i < k * qty; i++) {
			I[i] = i;
			D[i] = i;
		}

		MPI_Send(&qty, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(I, k * qty, MPI_LONG, 0, 1, MPI_COMM_WORLD);
		MPI_Send(D, k * qty, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
//
//		std::printf("sent  results\n");
	}
}

void search(MPI_Comm search_comm, int nshards) {
//	std::printf("search\n");
//	mock_search(search_comm);
//	return;

//	
	faiss::gpu::StandardGpuResources res;
	res.setTempMemory(1536 * 1024 * 1024);

	int search_rank;
	MPI_Comm_rank(search_comm, &search_rank);
	int shard = search_rank - 1;

	char index_path[500];
	sprintf(index_path, "index/index_%d_%d_%d", nb, ncentroids, m);
	FILE* index_file = fopen(index_path, "r");
	auto index = read_index(index_file, shard, nshards);
	dynamic_cast<faiss::IndexIVFPQ*>(index)->nprobe = nprobe;

	if (gpu) index = faiss::gpu::index_cpu_to_gpu(&res, 0, index, nullptr);

	int qn = 0;

	while (qn != nq) {
		int qty;
		MPI_Bcast(&qty, 1, MPI_INT, 0, search_comm);

		float query_buffer[qty * d];
		MPI_Bcast(query_buffer, qty * d, MPI_FLOAT, 0, search_comm);
		
//		std::printf("received queries\n");

		faiss::Index::idx_t* I = new faiss::Index::idx_t[k * qty];
		float* D = new float[k * qty];
		
		index->search(qty, query_buffer, k, D, I);
		
		MPI_Send(&qty, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(I, k * qty, MPI_LONG, 0, 1, MPI_COMM_WORLD);
		MPI_Send(D, k * qty, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
		
//		std::printf("sent  results\n");
		qn += qty;
		
		delete[] I;
		delete[] D;
	}
}

struct PartialResult {
	float* dists;
	long* ids;
	bool own_fields;
};

void merge_results(std::vector<PartialResult>& results, faiss::Index::idx_t* ids, int nshards) {
	int counter[nshards];
	for (int i = 0; i < nshards; i++) counter[i] = 0;

	for (int topi = 0; topi < k; topi++) {
		float bestDist = std::numeric_limits<float>::max();
		long bestId;
		int fromShard;

		for (int shard = 0; shard < nshards; shard++) {
			if (counter[shard] == k) continue;

			if (results[shard].dists[counter[shard]] < bestDist) {
				bestDist = results[shard].dists[counter[shard]];
				bestId = results[shard].ids[counter[shard]];
				fromShard = shard;
			}
		}

		ids[topi] = bestId;
		counter[fromShard]++;
	}
}

void mock_aggregator() {
	while (true) {
		MPI_Status status;
		int qty;
		MPI_Recv(&qty, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		auto I = new faiss::Index::idx_t[k * qty];
		auto D = new float[k * qty];

		MPI_Recv(I, k * qty, MPI_LONG, status.MPI_SOURCE, 1, MPI_COMM_WORLD,
		MPI_STATUS_IGNORE);
		MPI_Recv(D, k * qty, MPI_FLOAT, status.MPI_SOURCE, 2, MPI_COMM_WORLD,
		MPI_STATUS_IGNORE);
		
		std::printf("received  results\n");
	}
}

void aggregator(int nshards) {
	double end_time[nq];

//	 load ground-truth and convert int to long
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
	
	std::queue<PartialResult> queue[nshards];
	std::queue<PartialResult> to_delete[nshards];
	
	int n_1 = 0, n_10 = 0, n_100 = 0;
	int qn = 0;
	
	while (qn != nq) {
		MPI_Status status;
		int qty;
		MPI_Recv(&qty, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		
		
		int from = status.MPI_SOURCE - 2;
		
		auto I = new faiss::Index::idx_t[k * qty];
		auto D = new float[k * qty];
		
		MPI_Recv(I, k * qty, MPI_LONG, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D, k * qty, MPI_FLOAT, status.MPI_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		
		for (int q = 0; q < qty; q++) {
			queue[from].push({D + k * q, I + k * q, q == 0});
		}
		
		
		while (true) {
			bool hasEmpty = false;
			
			for (int i = 0; i < nshards; i++) {
				if (queue[i].empty()) {
					hasEmpty = true;
					break;
				}
			}
			
			if (hasEmpty) break;
			
			
			std::vector<PartialResult> results(nshards);
			for (int shard = 0; shard < nshards; shard++) {
				results[shard] = queue[shard].front();
				if (results[shard].own_fields) to_delete[shard].push(results[shard]);
				queue[shard].pop();
			}
						
			faiss::Index::idx_t ids[k];
			merge_results(results, ids, nshards);
			
			end_time[qn] = elapsed();

//			int gt_nn = gt[qn * k];
//
//			for (int j = 0; j < k; j++) {
//				if (ids[j] == gt_nn) {
//					if (j < 1)
//						n_1++;
//					if (j < 10)
//						n_10++;
//					if (j < 100)
//						n_100++;
//				}
//			}
//		
//
//			std::printf("%.2f\n", 100.0 * n_100 / qn);
			
			qn++;
		}
		
		for (int shard = 0; shard < nshards; shard++) {
			while (! to_delete[shard].empty()) {
				delete[] to_delete[shard].front().dists;
				delete[] to_delete[shard].front().ids;
				to_delete[shard].pop();
			}
		}
	}

	MPI_Send(end_time, nq, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
}
