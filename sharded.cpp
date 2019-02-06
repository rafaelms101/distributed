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

#include <fstream>
#include <utility>
#include <tuple>
#include <unistd.h>

#include <pthread.h>

#include <unordered_map>


//#define BENCHMARK
#define AGGREGATOR 0
#define GENERATOR 1

char src_path[] = "/home/rafael/mestrado/bigann/";

faiss::Index* gpu_index;

#define NQ 10000

int test_length = 80000;

double exp_length = 5;

int d = 128;
int k = 10;
int nb = 1000000; 
int ncentroids = 256; 
int m = 8;
int nprobe = 4;

int block_size = 0;
double gpu_slice = 0;
double query_rate = 0;

void generator(int nshards);
void single_block_size_generator(int nshards, int block_size);
void aggregator(int nshards);
void search(int shard, int nshards, bool dynamic);

enum MESSAGE_TAGS {
	
};

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

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::printf("Wrong usage. ./sharded <qr> s|d [<block_size> <gpu_slice>]\n");
		std::exit(-1);
	}
	
	query_rate = atof(argv[1]);
	bool dynamic = ! strcmp("d", argv[2]);
	
	
	if (! dynamic) {
		if (argc != 5 || strcmp("s", argv[2])) {
			std::printf("Wrong usage. ./sharded <qr> s|d [<block_size> <gpu_slice>]\n");
			std::exit(-1);
		}
		
		block_size = atoi(argv[3]);
		gpu_slice = atof(argv[4]);
		
		assert(gpu_slice >= 0 && gpu_slice <= 1);
	}
	
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
//    	std::printf("generator[%d]\n", world_rank);
    	generator(world_size - 2);
    } else if (world_rank == 0) {
//    	std::printf("aggregator[%d]\n", world_rank);
    	aggregator(world_size - 2);
    } else {
//    	std::printf("search[%d]\n", world_rank);
    	search(world_rank - 2, world_size - 2, dynamic);
    }
    
    
    
    // Finalize the MPI environment.
    MPI_Finalize();
}

double constant_interval(double val) {
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

float* load_queries() {
	char query_path[500];
	sprintf(query_path, "%s/bigann_query.bvecs", src_path);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	//loading queries
	size_t fz;
	unsigned char* queries = bvecs_read(query_path, &fz);
	float* xq = to_float_array(queries, NQ, d);
	munmap(queries, fz);
	
	return xq;
}


double next_query_interval() {
	return constant_interval(query_rate);
}

int next_query() {
	static int qty = 0;
	static double query_time = elapsed();
	
	double now = elapsed();
	
	if (qty >= test_length) return -2;
	
	if (now < query_time) return -1;
	
	query_time = now + next_query_interval();
	
	qty++;
	
	return rand() % 10000; 
}

inline void send_queries(int nshards, float* query_buffer, int queries_in_buffer) {
	for (int node = 0; node < nshards; node++) {
		double send_time = elapsed();
		
		MPI_Request request;
		MPI_Isend(&send_time, 1, MPI_DOUBLE, node + 2, 0, MPI_COMM_WORLD,
				&request);
		MPI_Request_free(&request);
		MPI_Isend(&queries_in_buffer, 1, MPI_INT, node + 2, 0, MPI_COMM_WORLD,
				&request);
		MPI_Request_free(&request);
		MPI_Isend(query_buffer, queries_in_buffer * d, MPI_FLOAT, node + 2, 0,
				MPI_COMM_WORLD, &request);
		MPI_Request_free(&request);
	}
}

//TODO: mock nodes must be updated
void single_block_size_generator(int nshards, int block_size) {
	float* xq = load_queries();

	double end_time[10000];
	double start_time[10000];
	
	int begin_timing = test_length - 10000 + 1;

	float* query_buffer = new float[test_length * d];
	float* to_be_deleted = query_buffer;

	int queries_in_buffer = 0;

	int qn = 0;
	
	while (true) {
		int id = next_query();
		
		if (id == -1) {
			continue;
		}
			
		if (id == -2) {
			if (queries_in_buffer >= 1) {
				send_queries(nshards, query_buffer, queries_in_buffer);
			}

			break;
		}

		qn++;
		
		if (qn >= begin_timing) {
			int offset = qn - begin_timing;
			start_time[offset] = elapsed();
			
//			if (offset % 500 == 0) std::printf("start[%d]=%lf\n", offset, start_time[offset]);
		}
		
		memcpy(query_buffer + queries_in_buffer * d, xq + id * d, d * sizeof(float));
		queries_in_buffer++;

		if (queries_in_buffer != block_size) continue;

		send_queries(nshards, query_buffer, queries_in_buffer);
		query_buffer = query_buffer + queries_in_buffer * d;
		queries_in_buffer = 0;
	}

	MPI_Recv(end_time, 10000, MPI_DOUBLE, AGGREGATOR, 0, MPI_COMM_WORLD,
			MPI_STATUS_IGNORE);

	double total = 0;

	for (int i = 0; i < 10000; i++) {
//		std::printf("query #%d took: %lf\n", i, end_time[i] - start_time[i]);
		total += end_time[i] - start_time[i];
	}

	std::printf("%lf\n", total / 10000);
	
	delete [] to_be_deleted;
	delete [] xq;
}

void generator(int nshards) {
	single_block_size_generator(nshards, 20);
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
	}
}

struct SearchArg {
	int nq;
	float* queries;
	float* D;
	faiss::Index::idx_t* I;
};

void* gpu_search(void* arg) {
	SearchArg* a = (SearchArg*) arg;
	
	gpu_index->search(a->nq, a->queries, k, a->D, a->I);
	pthread_exit(NULL);
}

inline void receive_queries(float* query_buffer, int* nqueries) {
	double send_time;
	MPI_Recv(&send_time, 1, MPI_DOUBLE, GENERATOR, 0, MPI_COMM_WORLD,
	MPI_STATUS_IGNORE);

	int qty;
	MPI_Status status;
	MPI_Recv(&qty, 1, MPI_INT, GENERATOR, 0, MPI_COMM_WORLD, &status);

	MPI_Recv(query_buffer + *nqueries * d, qty * d, MPI_FLOAT,
	GENERATOR, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	*nqueries = *nqueries + qty;
}

struct ExecutionProfile {
	double gpu_slice;
	double time;
	bool optimal;
};

void search(int shard, int nshards, bool dynamic) {
	const int max_block_size = 100000;
	const double gpu_slice_granularity = 0.05;
	
	ExecutionProfile execution_time[max_block_size];
	for (int i = 0; i < max_block_size; i++) {
		execution_time[i].optimal = false;
		execution_time[i].gpu_slice = 1 + gpu_slice_granularity;
		execution_time[i].time = -1;
	}
	
	int processed = 0;
	
	faiss::gpu::StandardGpuResources res;
//	res.setTempMemory(1536 * 1024 * 1024);

	char index_path[500];
	sprintf(index_path, "index/index_%d_%d_%d", nb, ncentroids, m);
	FILE* index_file = fopen(index_path, "r");
	auto cpu_index = read_index(index_file, shard, nshards);
	dynamic_cast<faiss::IndexIVFPQ*>(cpu_index)->nprobe = nprobe;

	gpu_index = faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index, nullptr);

	int qn = 0;

	float* query_buffer = new float[max_block_size * d];
	int nqueries = 0;
	
	int target_block_size = block_size;
	
	while (qn != test_length) {	
		int available;
		MPI_Iprobe(GENERATOR, 0, MPI_COMM_WORLD, &available, MPI_STATUS_IGNORE);
		
		while (available && nqueries < max_block_size) {
			receive_queries(query_buffer, &nqueries);
			MPI_Iprobe(GENERATOR, 0, MPI_COMM_WORLD, &available, MPI_STATUS_IGNORE);
		}
		
//		std::printf("waiting=%d\n", nqueries);
		
		if (dynamic) {
			if (nqueries > target_block_size) {
				target_block_size = nqueries;
			} else {
//				target_block_size = nqueries;
				target_block_size = std::max(20,
						static_cast<int>(0.5 * (target_block_size + nqueries)));
			}
		}

		target_block_size = std::max(20, target_block_size);
		target_block_size = std::min(target_block_size, test_length - processed);
		
		while (nqueries < target_block_size) {
			receive_queries(query_buffer, &nqueries);
		}
		
		assert(nqueries <= max_block_size);
		
		if (dynamic) {
			if (execution_time[nqueries].optimal)
				gpu_slice = execution_time[nqueries].gpu_slice;
			else
				gpu_slice = execution_time[nqueries].gpu_slice - gpu_slice_granularity;
		}
		
		if (nqueries != 20) std::printf("handling %d\n", nqueries);

		//now we proccess our query buffer
		faiss::Index::idx_t* I = new faiss::Index::idx_t[k * nqueries];
		float* D = new float[k * nqueries];
		
		int nq_gpu = nqueries * gpu_slice;
		int nq_cpu = nqueries - nq_gpu;
		
		pthread_t thread;

		SearchArg args;
		
		double start = elapsed();
		
		if (nq_gpu > 0) {
			args.nq = nq_gpu;
			args.queries = query_buffer + nq_cpu * d;
			args.D = D + k * nq_cpu;
			args.I = I + k * nq_cpu;
			pthread_create(&thread, NULL, gpu_search, &args);
		}
		
		
		if (nq_cpu > 0) {
			cpu_index->search(nq_cpu, query_buffer, k, D, I);
		}
		
		
		if (nq_gpu > 0) {
			pthread_join(thread, NULL);
		}	
		
		double end = elapsed();
		double et = end - start;
		
//		std::printf("et[%d] = %lf + %lf\n", nqueries, start, end - start);
		
		MPI_Send(&nqueries, 1, MPI_INT, AGGREGATOR, 0, MPI_COMM_WORLD);
		MPI_Send(I, k * nqueries, MPI_LONG, AGGREGATOR, 1, MPI_COMM_WORLD);
		MPI_Send(D, k * nqueries, MPI_FLOAT, AGGREGATOR, 2, MPI_COMM_WORLD);

		if (dynamic) {
			if (execution_time[nqueries].time == -1
					|| et < execution_time[nqueries].time) {
				execution_time[nqueries].time = et;
				execution_time[nqueries].gpu_slice = gpu_slice;

				//			if (gpu_slice != 1) std::printf("updated gpu_slice of %d to %lf\n", nqueries, gpu_slice);
			} else {
				execution_time[nqueries].optimal = true;
			}
		}
		

		qn += nqueries;
		processed += nqueries;
		nqueries = 0;
		
		delete[] I;
		delete[] D;
	}
	
	delete [] query_buffer;
}

struct PartialResult {
	float* dists;
	long* ids;
	bool own_fields;
	float* base_dists;
	long* base_ids;
};

void merge_results(std::vector<PartialResult>& results, faiss::Index::idx_t* ids, int nshards) {
	int counter[nshards];
	for (int i = 0; i < nshards; i++) counter[i] = 0;

	for (int topi = 0; topi < k; topi++) {
		float bestDist = std::numeric_limits<float>::max();
		long bestId = -1;
		int fromShard = -1;

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

void aggregate_query(std::queue<PartialResult>* queue, int nshards) {
	std::vector<PartialResult> results(nshards);
	
	for (int shard = 0; shard < nshards; shard++) {
		results[shard] = queue[shard].front();
		queue[shard].pop();
	}
				
	faiss::Index::idx_t ids[k];
	merge_results(results, ids, nshards);
	
	for (int shard = 0; shard < nshards; shard++) {
		if (results[shard].own_fields) {
			delete [] results[shard].base_dists;
			delete [] results[shard].base_ids;
		}	
	}
}

void aggregator(int nshards) {
	int begin_timing = test_length - 10000 + 1;
	double end_time[10000];

//	 load ground-truth and convert int to long
	char idx_path[1000];
	char gt_path[500];
	sprintf(gt_path, "%s/gnd", src_path);
	sprintf(idx_path, "%s/idx_%dM.ivecs", gt_path, nb / 1000000);

	int whatever;
	int *gt_int = ivecs_read(idx_path, &k, &whatever);

	faiss::Index::idx_t* gt = new faiss::Index::idx_t[k * NQ];

	for (int i = 0; i < k * NQ; i++) {
		gt[i] = gt_int[i];
	}

	delete[] gt_int;
	
	std::queue<PartialResult> queue[nshards];
	std::queue<PartialResult> to_delete;
	
	int qn = 0;
	
	while (qn != test_length) {
		MPI_Status status;
		int qty;
		MPI_Recv(&qty, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		
		int from = status.MPI_SOURCE - 2;
		
		auto I = new faiss::Index::idx_t[k * qty];
		auto D = new float[k * qty];
		
		MPI_Recv(I, k * qty, MPI_LONG, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D, k * qty, MPI_FLOAT, status.MPI_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		for (int q = 0; q < qty; q++) {
			queue[from].push({D + k * q, I + k * q, q == qty - 1, D, I});
		}
		
		qty = 0;
//		double start = elapsed();

		while (true) {
			bool hasEmpty = false;

			for (int i = 0; i < nshards; i++) {
				if (queue[i].empty()) {
					hasEmpty = true;
					break;
				}
			}
			
			if (hasEmpty) break;

			aggregate_query(queue, nshards);
			qn++;
			
			if (qn >= begin_timing) {
				int offset = qn - begin_timing;
				end_time[offset] = elapsed();

//				if (offset % 500 == 0) std::printf("end[%d]=%lf\n", offset, end_time[offset]);
			}
			
			qty++;
		}
		
//		double end = elapsed();
//		std::printf("at[%d] = %lf + %lf\n", qty, start, end - start);
	}
	
	MPI_Send(end_time, 10000, MPI_DOUBLE, GENERATOR, 0, MPI_COMM_WORLD);
}
