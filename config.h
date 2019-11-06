#ifndef CONFIG_H_
#define CONFIG_H_

#include <vector>
#include <mpi.h>

constexpr long BENCH_SIZE = 1000; //upper limit on maximum block size when in benchmark mode

constexpr char SRC_PATH[] = "../bigann"; //folder where the bigann database is stored
constexpr char INDEX_ROOT[] = "../index"; //folder where the indexed databases are stored
constexpr char PROF_ROOT[] = "prof"; //folder where the indexed databases are stored
constexpr long BENCH_REPEATS = 3; //number of times that a certain number of queries will be executed while in benchmark mode

enum class RequestDistribution {Constant, Variable_Poisson};
enum class ExecType {Single, Both, OutOfCore, Bench};
enum class SearchAlgorithm {Cpu, Gpu, Hybrid, CpuFixed, Fixed};

class ExecPolicy;
class SearchStrategy;

struct Config {
	//database config
	const long d = 128; //vector dimension
	const long nb = 500000000; //database size
	const long ncentroids = 4096; //number of centroids
	const long m = 8; //vector size after compression, in bytes
	const long nq = 10000; //total number of distinct queries
	
	//runtime config
	const long k = 10;
	const long nprobe = 16;
	const long block_size = 5;
	const long bench_step = block_size;
	const long gpus_per_node = 1;
	const long seed = 2;
	const long temp_memory_gpu = 0;
	const long poisson_intervals = 100;
	
	RequestDistribution request_distribution;
	long num_blocks = 100000 / block_size; 
	bool bench_cpu = false;
	bool bench_gpu = false;
	double query_load;
	
	
	//mode specific config
	double gpu_throughput;
	double cpu_throughput;
	long gpu_pieces;
	long total_pieces;
	
	ExecType exec_type;
	
	ExecPolicy* exec_policy = nullptr;
	SearchStrategy* search_strategy = nullptr;
	
	SearchAlgorithm search_algorithm;
	
	int shard;
	
	double raw_search_time = 0;
	
	MPI_Comm search_comm;
};

extern Config cfg;

#endif 
