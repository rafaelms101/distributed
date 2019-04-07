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
#include <condition_variable>
#include <algorithm>

#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

#include "gpu/GpuAutoTune.h"
#include "gpu/StandardGpuResources.h"
#include "gpu/GpuIndexIVFPQ.h"

#include "readSplittedIndex.h"
#include "QueryBuffer.h"

#include <fstream>
#include <utility>
#include <tuple>
#include <unistd.h>
#include <functional>

#include <unordered_map>
#include <unistd.h>

#include <thread>   

#define DEBUG


#ifdef DEBUG

//TODO: rename to db or dbg
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

const int block_size = 20;

int processing_size;
double gpu_slice = 1;
double query_rate;

constexpr int bench_repeats = 3;

enum class ProcType {Static, Dynamic, Bench};

void generator(int nshards, ProcType ptype);
void single_block_size_generator(int nshards, int block_size);
void aggregator(int nshards, ProcType ptype);
void search(int shard, int nshards, ProcType ptype);



// aggregator, generator, search
//TODO: rename to now
static double elapsed ()
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
	// Initialize the MPI environment
	int provided;
	MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
	assert(provided == MPI_THREAD_MULTIPLE);
	
	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
	std::string usage = "./sharded b | d <qr> | s <qr> <num_blocks> <gpu_slice>";

	if (argc < 2) {
		if (world_rank == 0) std::printf("Wrong arguments.\n%s\n", usage.c_str());
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	ProcType ptype; 
	if (! strcmp("d", argv[1])) ptype = ProcType::Dynamic;
	else if (! strcmp("b", argv[1])) ptype = ProcType::Bench;
	else if (! strcmp("s", argv[1])) ptype = ProcType::Static;
	else {
		if (world_rank == 0) std::printf("Invalid processing type.Expected b | s | d\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	if (ptype == ProcType::Dynamic) {
		if (argc != 3) {
			if (world_rank == 0) std::printf("Wrong arguments.\n%s\n", usage.c_str());
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		
		query_rate = atof(argv[2]);
	} else if (ptype == ProcType::Static) {
		if (argc != 5) {
			if (world_rank == 0) std::printf("Wrong arguments.\n%s\n", usage.c_str());
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		
		query_rate = atof(argv[2]);
		processing_size = atoi(argv[3]);
		gpu_slice = atof(argv[4]);

		assert(gpu_slice >= 0 && gpu_slice <= 1);
	} 

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    
    MPI_Group search_group;
    int ranks[] = {0};
    MPI_Group_excl(world_group, 1, ranks, &search_group);
    
    MPI_Comm search_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, search_group, 0, &search_comm);
    

    if (world_rank == 1) {
    	generator(world_size - 2, ptype);
    } else if (world_rank == 0) {
    	aggregator(world_size - 2, ptype);
    } else {
    	search(world_rank - 2, world_size - 2, ptype);
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
	static double query_time = elapsed();
	
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

void send_finished_signal(int nshards) {
	float dummy = 1;
	for (int node = 0; node < nshards; node++) {
		MPI_Ssend(&dummy, 1, MPI_FLOAT, node + 2, 0, MPI_COMM_WORLD);
	}
}

void bench_generator(int num_blocks, int block_size, int nshards) {
	assert(num_blocks * block_size <= NQ);

	float* xq = load_queries();

	for (int i = 1; i <= num_blocks; i++) {
		for (int repeats = 1; repeats <= bench_repeats; repeats++) {
			for (int b = 1; b <= i; b++) {
				send_queries(nshards, xq, block_size);
			}
		}
	}

	send_finished_signal(nshards);
}


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

	send_finished_signal(nshards);
	
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

void generator(int nshards, ProcType ptype) {
	int shards_ready = 0;
	
	//Waiting for search nodes to be ready
	while (shards_ready < nshards) {
		float dummy;
		MPI_Recv(&dummy, 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		shards_ready++;
	}

	if (ptype == ProcType::Bench) bench_generator(3000 / 20, block_size, nshards);
	else single_block_size_generator(nshards, block_size);
}

inline void _search(faiss::Index* index, int nq, float* queries, float* D, faiss::Index::idx_t* I) {
	if (nq > 0) index->search(nq, queries, k, D, I);
}

void query_receiver(QueryBuffer* buffer, bool* finished) {
	deb("Receiver");
	int queries_received = 0;
	
	float dummy;
	MPI_Send(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD); //signal that we are ready to receive queries
	
	//TODO: I will left it like this for the time being since buffer will always have free space available. In the future we should use events so that this thread sleeps while waiting for the buffer to have space
	while (true) {
		MPI_Status status;
		MPI_Probe(GENERATOR, 0, MPI_COMM_WORLD, &status);
		
		int message_size;
		MPI_Get_count(&status, MPI_FLOAT, &message_size);
		
		if (message_size == 1) {
			*finished = true;
			float dummy;
			MPI_Recv(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
			break;
		}
		
		while (! buffer->hasSpace());

		float* recv_data = reinterpret_cast<float*>(buffer->peekEnd());
		MPI_Recv(recv_data, block_size * d, MPI_FLOAT, GENERATOR, 0,
				MPI_COMM_WORLD, &status);
		buffer->add();

		assert(status.MPI_ERROR == MPI_SUCCESS);

		queries_received += block_size;
	}
	
	deb("Finished receiving queries");
}

auto load_index(int shard, int nshards) {
	deb("Started loading");

	char index_path[500];
	sprintf(index_path, "index/index_%d_%d_%d", nb, ncentroids, m);

	deb("Loading file: %s", index_path);

	FILE* index_file = fopen(index_path, "r");
	auto cpu_index = read_index(index_file, shard, nshards);

	deb("Ended loading");

	dynamic_cast<faiss::IndexIVFPQ*>(cpu_index)->nprobe = nprobe;
	return cpu_index;
}

//TODO: separate each stage in its own file
//TODO: currently, we use the same files in all nodes. This doesn't work with ProfileData, since each machine is different. Need to create a way for each node to load a different file.
struct ProfileData {
	double* times;
	int min_block;
	int max_block;
};

ProfileData getProfilingData() {	
	char file_path[100];
	sprintf(file_path, "prof/%d_%d_%d_%d_%d_%d", nb, ncentroids, m, k, nprobe, block_size);
	std::ifstream file;
	file.open(file_path);
	
	if (! file.good()) {
		std::printf("File prof/%d_%d_%d_%d_%d_%d was not found\n", nb, ncentroids, m, k, nprobe, block_size);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	int total_size;
	file >> total_size;
	
	double times[total_size + 1];
	double time_per_block[total_size + 1];
	
	times[0] = 0;
	time_per_block[0] = 0;
	
	for (int i = 1; i <= total_size; i++) {
		file >> times[i];
		time_per_block[i] = times[i] / i;
	}

	file.close();
	
	double tolerance = 0.1;
	int minBlock = 1;
	
	for (int nb = 2; nb <= total_size; nb++) {
		if (time_per_block[nb] < time_per_block[minBlock]) minBlock = nb;
	}
	
	double threshold = time_per_block[minBlock] * (1 + tolerance);
	ProfileData pd;

	int nb = 1;
	while (time_per_block[nb] > threshold) nb++;
	pd.min_block = nb;
	
	nb = total_size;
	while (time_per_block[nb] > threshold) nb--;
	pd.max_block = nb;
	
	deb("min: %d, max: %d", pd.min_block * block_size, pd.max_block * block_size);
	
	pd.times = new double[minBlock + 1];
	pd.times[0] = 0;
	for (int nb = 1; nb <= minBlock; nb++) pd.times[nb] = time_per_block[nb];
	
	return pd;
}

void store_profile_data(std::vector<double>& procTimes) {
	//now we write the time data on a file
	char file_path[100];
	sprintf(file_path, "prof/%d_%d_%d_%d_%d_%d", nb, ncentroids, m, k, nprobe,
			block_size);
	std::ofstream file;
	file.open(file_path);

	int blocks = procTimes.size() / bench_repeats;
	file << blocks << std::endl;

	int ptr = 0;
	
	for (int b = 1; b <= blocks; b++) {
		std::vector<double> times;
		
		for (int repeats = 1; repeats <= bench_repeats; repeats++) {
			times.push_back(procTimes[ptr++]);
		}
		
		std::sort(times.begin(), times.end());
		
		int mid = bench_repeats / 2;
		file << times[mid] << std::endl;
	}

	file.close();
}

void search(int shard, int nshards, ProcType ptype) {
	std::vector<double> procTimes;
	
	ProfileData pd; 
	if (ptype == ProcType::Dynamic) pd = getProfilingData();

	const long block_size_in_bytes = sizeof(float) * d * block_size;
	QueryBuffer buffer(block_size_in_bytes, 100 * 1024 * 1024 / block_size_in_bytes); //100 MB
	
	faiss::gpu::StandardGpuResources res;

	auto cpu_index = load_index(shard, nshards);
	auto gpu_index = faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index, nullptr);

	faiss::Index::idx_t* I = new faiss::Index::idx_t[test_length * k];
	float* D = new float[test_length * k];
	
	deb("Search node is ready");
	
	assert(test_length % block_size == 0);

	int qn = 0;
	bool finished = false;
	
	std::thread receiver { query_receiver, &buffer, &finished };
	
	while (! finished || buffer.entries() >= 1) {
		int num_blocks;
		
		if (ptype == ProcType::Dynamic) {
			buffer.waitForData(1);
			num_blocks = buffer.entries();
			
			if (num_blocks < pd.min_block) {
				double work_without_waiting = num_blocks / pd.times[num_blocks];
				double work_with_waiting = (pd.min_block - num_blocks) * buffer.block_rate() + pd.times[pd.min_block];
				
				if (work_with_waiting > work_without_waiting) {
					usleep((pd.min_block - num_blocks) * buffer.block_rate() * 1000000);
				}
			} 
			
			num_blocks = std::min(buffer.entries(), pd.max_block);
		} else if (ptype == ProcType::Static) {
			assert((test_length - qn) % 20 == 0);
			num_blocks = processing_size;
		} else if (ptype == ProcType::Bench) {
			static int nrepeats = 0;
			static int nb = 1;
			
			if (nrepeats >= bench_repeats) {
				nrepeats = 0;
				nb++;
			}
						
			nrepeats++;
			num_blocks = nb;
		}
		
		if (ptype != ProcType::Bench) num_blocks = std::min(num_blocks, (test_length - qn) / 20);
		buffer.waitForData(num_blocks);
		
		//TODO: this is wrong since the buffer MIGHT not be continuous.
		float* query_buffer = reinterpret_cast<float*>(buffer.peekFront());
		int nqueries = num_blocks * block_size;
		
		deb("Search node processing %d queries", nqueries);

		//now we proccess our query buffer
		int nq_gpu = static_cast<int>(nqueries * gpu_slice);
		int nq_cpu = static_cast<int>(nqueries - nq_gpu);
		
		assert(nqueries == nq_cpu + nq_gpu);

//		std::thread gpu_thread{gpu_search, gpu_index, nq_gpu, query_buffer + nq_cpu * d, D + k * nq_cpu, I + k * nq_cpu};

		if (nq_cpu > 0) {
			_search(cpu_index, nq_cpu, query_buffer, D, I);
		}

		if (nq_gpu > 0) {
			double before = elapsed();
			_search(gpu_index, nq_gpu, query_buffer + nq_cpu * d, D + k * nq_cpu, I + k * nq_cpu);
			double after = elapsed();
			
			if (ptype == ProcType::Bench) {
				procTimes.push_back(after - before);
			}
		}

//		gpu_thread.join();
		
		buffer.consume(num_blocks);

		//TODO: Optimize this to a Immediate Synchronous Send
		//TODO: Merge these two sends into one
		MPI_Ssend(I, k * nqueries, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
		MPI_Ssend(D, k * nqueries, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);
		
		qn += nqueries;
	}
	
	long dummy = 1;
	MPI_Ssend(&dummy, 1, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
	
	receiver.join();

	delete[] I;
	delete[] D;
	
	if (ptype == ProcType::Bench) store_profile_data(procTimes);

	deb("Finished search node");
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

void aggregator(int nshards, ProcType ptype) {
	std::deque<double> end_times;

	auto gt = load_gt();
	faiss::Index::idx_t* answers = new faiss::Index::idx_t[10000 * k];
	
	std::queue<PartialResult> queue[nshards];
	std::queue<PartialResult> to_delete;
	
	deb("Aggregator node is ready");
	
	int shards_finished = 0;
	
	int qn = 0;
	while (true) {
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		int message_size;
		MPI_Get_count(&status, MPI_LONG, &message_size);
		
		if (message_size == 1) {
			shards_finished++;
			float dummy;
			MPI_Recv(&dummy, 1, MPI_LONG, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (shards_finished == nshards) break;
			continue;
		}

		int qty = message_size / k;

		auto I = new faiss::Index::idx_t[k * qty];
		auto D = new float[k * qty];
		
		MPI_Recv(I, k * qty, MPI_LONG, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(D, k * qty, MPI_FLOAT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		int from = status.MPI_SOURCE - 2;
		for (int q = 0; q < qty; q++) {
			queue[from].push({D + k * q, I + k * q, q == qty - 1, D, I});
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

			aggregate_query(queue, nshards, answers + (qn % 10000) * k);
			qn++;
			
			if (end_times.size() >= 10000) end_times.pop_front();
			end_times.push_back(elapsed());
		}
	}
	
	if (ptype != ProcType::Bench) {
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
		
		double end_times_array[10000];

		for (int i = 0; i < 10000; i++) {
			end_times_array[i] = end_times.front();
			end_times.pop_front();
		}

		MPI_Send(end_times_array, 10000, MPI_DOUBLE, GENERATOR, 0, MPI_COMM_WORLD);
	}
	
	delete [] gt;
	
	deb("Finished aggregator");
}
