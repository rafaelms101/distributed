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
#include "CircularBuffer.h"

#include <fstream>
#include <utility>
#include <tuple>
#include <unistd.h>
#include <functional>

#include <unordered_map>

#include <thread>   

#define DEBUG


#ifdef DEBUG

#define deb(...) do {std::printf("%lf) ", elapsed()); std::printf(__VA_ARGS__); std::printf("\n");} while(0)
#else
#define deb(...) ;
#endif

//#define BENCHMARK
#define AGGREGATOR 0
#define GENERATOR 1

char src_path[] = "/home/rafael/mestrado/bigann/";

#define NQ 10000

int test_length = 80000;

int d = 128;
int k = 10;
int nb = 1000000; 
int ncentroids = 256; 
int m = 8;
int nprobe = 4;

int block_size = 20;

double gpu_slice = 1;
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
	int provided;
	MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
	assert(provided == MPI_THREAD_MULTIPLE);

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

std::pair<int, double> next_query() {
	static int qty = 0;
	static double query_time = elapsed() + 2;
	
	double now = elapsed();
	
	if (qty >= test_length) return {-2, -1};
	
	if (now < query_time) return {-1, -1};
	
	double old_time = query_time;
	query_time = now + next_query_interval();
	
	qty++;
	
	if (qty % 10000 == 0 || (qty % 1000 == 0 && qty <= 10000)) {
		deb("Sent %d/%d queries", qty, test_length);
	}
	
	return {(qty - 1) % 10000, old_time}; 
}

void send_queries(int nshards, float* query_buffer, int queries_in_buffer) {
	for (int node = 0; node < nshards; node++) {
		assert(queries_in_buffer == block_size);
		MPI_Ssend(query_buffer, queries_in_buffer * d, MPI_FLOAT, node + 2, 0, MPI_COMM_WORLD);
	}
}

//TODO: mock nodes must be updated
void single_block_size_generator(int nshards, int block_size) {
	assert(test_length % block_size == 0);
	
	float* xq = load_queries();

	double end_time[10000];
	double start_time[10000];
	
	int begin_timing = test_length - 10000 + 1;

	float* query_buffer = new float[test_length * d];
	float* to_be_deleted = query_buffer;

	int queries_in_buffer = 0;

	int qn = 0;
	
	bool first = true;
	
	while (true) {
		auto [id, query_time] = next_query();
		
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
			start_time[offset] = query_time;
		}
		
		memcpy(query_buffer + queries_in_buffer * d, xq + id * d, d * sizeof(float));
		queries_in_buffer++;

		if (queries_in_buffer < block_size) continue;

		
		if (first) {
			first = false;
			deb("First block sent");
		}
		send_queries(nshards, query_buffer, queries_in_buffer);
		
		query_buffer = query_buffer + queries_in_buffer * d;
		queries_in_buffer = 0;
	}

	MPI_Recv(end_time, 10000, MPI_DOUBLE, AGGREGATOR, 0, MPI_COMM_WORLD,
			MPI_STATUS_IGNORE);

	double total = 0;

	for (int i = 0; i < 10000; i++) {
		total += end_time[i] - start_time[i];
	}

	std::printf("%lf\n", total / 10000);
	
	delete [] to_be_deleted;
	delete [] xq;
}

void generator(int nshards) {
	single_block_size_generator(nshards, block_size);
}

void gpu_search(faiss::Index* gpu_index, int nq, float* queries, float* D, faiss::Index::idx_t* I) {
	if (nq > 0) gpu_index->search(nq, queries, k, D, I);
}

void query_receiver(CircularBuffer* buffer) {
	int queries_received = 0;
	
//	assert(block_size * d * sizeof(float) == buffer->bs());
	
	while (queries_received < test_length) {
		//TODO: instead of a busy loop, use events
		//TODO: I will left it like this for the time being since buffer will always have free space available.

//		assert(buffer->hasSpace());

		if (buffer->hasSpace()) {
			float* recv_data = reinterpret_cast<float*>(buffer->peekEnd());

			MPI_Status status;
			MPI_Recv(recv_data, block_size * d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);

			buffer->add();

			assert(status.MPI_ERROR == MPI_SUCCESS);

			queries_received += block_size;
		}
	}
}

auto load_index(int shard, int nshards) {
	char index_path[500];
	sprintf(index_path, "index/index_%d_%d_%d", nb, ncentroids, m);
	FILE* index_file = fopen(index_path, "r");
	auto cpu_index = read_index(index_file, shard, nshards);
	dynamic_cast<faiss::IndexIVFPQ*>(cpu_index)->nprobe = nprobe;
	return cpu_index;
}

void search(int shard, int nshards, bool dynamic) {
	CircularBuffer buffer(sizeof(float) * d * block_size, test_length / block_size);
	std::thread receiver { query_receiver, &buffer };
	
	faiss::gpu::StandardGpuResources res;

	auto cpu_index = load_index(shard, nshards);
	auto gpu_index = faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index, nullptr);

	faiss::Index::idx_t* I = new faiss::Index::idx_t[test_length * k];
	float* D = new float[test_length * k];
	
	deb("Search node is ready");
	
	assert(test_length % block_size == 0);
	
	int qn = 0;
	while (qn < test_length) {	
		while (buffer.empty()) {}
		
		//TODO: this is wrong since the buffer MIGHT not be continuous.
		float* query_buffer = reinterpret_cast<float*>(buffer.peekFront());
		int num_blocks = buffer.entries();
		int nqueries = num_blocks * block_size;
		
		deb("Search node processing %d queries", nqueries);

		//now we proccess our query buffer
		int nq_gpu = static_cast<int>(nqueries * gpu_slice);
		int nq_cpu = static_cast<int>(nqueries - nq_gpu);
		
		assert(nqueries == nq_cpu + nq_gpu);

		std::thread gpu_thread{gpu_search, gpu_index, nq_gpu, query_buffer + nq_cpu * d, D + k * nq_cpu, I + k * nq_cpu};
		
		if (nq_cpu > 0) {
			cpu_index->search(nq_cpu, query_buffer, k, D, I);
		}

		gpu_thread.join();
		
		buffer.consume(num_blocks);

		//TODO: Optimize this to a Immediate Synchronous Send
		//TODO: Merge these two sends into one
		MPI_Ssend(I, k * nqueries, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
		MPI_Ssend(D, k * nqueries, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);
		
		qn += nqueries;
	}
	
	receiver.join();

	delete[] I;
	delete[] D;
}

struct PartialResult {
	float* dists;
	long* ids;
	bool own_fields;
	float* base_dists;
	long* base_ids;
};

void merge_results(std::vector<PartialResult>& results, faiss::Index::idx_t* answers, int nshards) {
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

		answers[topi] = bestId;
		counter[fromShard]++;
	}
}

void aggregate_query(std::queue<PartialResult>* queue, int nshards, faiss::Index::idx_t* answers) {
	std::vector<PartialResult> results(nshards);
	
	for (int shard = 0; shard < nshards; shard++) {
		results[shard] = queue[shard].front();
		queue[shard].pop();
	}
				
	merge_results(results, answers, nshards);
	
	for (int shard = 0; shard < nshards; shard++) {
		if (results[shard].own_fields) {
			delete [] results[shard].base_dists;
			delete [] results[shard].base_ids;
		}	
	}
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

	faiss::Index::idx_t* gt = new faiss::Index::idx_t[k * NQ];

	for (int i = 0; i < NQ; i++) {
		for (int j = 0; j < k; j++) {
			gt[i * k + j] = gt_int[i * db_k + j];
		}
	}

	delete[] gt_int;
	return gt;
}

void aggregator(int nshards) {
	int begin_timing = test_length - 10000 + 1;
	double end_time[10000];

	auto gt = load_gt();
	
	faiss::Index::idx_t* answers = new faiss::Index::idx_t[10000 * k];
	
	std::queue<PartialResult> queue[nshards];
	std::queue<PartialResult> to_delete;
	
	deb("Aggregator node is ready");
	
	int qn = 0;
	while (qn < test_length) {
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		int message_size;
		MPI_Get_count(&status, MPI_LONG, &message_size);

		int qty = message_size / k;

		auto I = new faiss::Index::idx_t[k * qty];
		auto D = new float[k * qty];
		
		MPI_Recv(I, k * qty, MPI_LONG, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D, k * qty, MPI_FLOAT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		int from = status.MPI_SOURCE - 2;
		for (int q = 0; q < qty; q++) {
			queue[from].push({D + k * q, I + k * q, q == qty - 1, D, I});
		}

		int p = 0;
		
		while (true) {
			bool hasEmpty = false;

			for (int i = 0; i < nshards; i++) {
				if (queue[i].empty()) {
					hasEmpty = true;
					break;
				}
			}
			
			if (hasEmpty) break;

			aggregate_query(queue, nshards, answers + (qn % 10000) * k);
			qn++;
			p++;
			
			if (qn >= begin_timing) {
				int offset = qn - begin_timing;
				
				assert(offset < 10000);
				
				end_time[offset] = elapsed();
			}
		}
	}

	int n_1 = 0, n_10 = 0, n_100 = 0;
	for (int i = 0; i < 10000; i++) {
		int gt_nn = gt[i * k];
		for (int j = 0; j < k; j++) {
			if (answers[i * k + j] == gt_nn) {
				if (j < 1)
					n_1++;
				if (j < 10)
					n_10++;
				if (j < 100)
					n_100++;
			}
		}
	}

	printf("R@1 = %.4f\n", n_1 / float(NQ));
	printf("R@10 = %.4f\n", n_10 / float(NQ));
	printf("R@100 = %.4f\n", n_100 / float(NQ));
	
	
	delete [] gt;
	
	MPI_Send(end_time, 10000, MPI_DOUBLE, GENERATOR, 0, MPI_COMM_WORLD);
}
