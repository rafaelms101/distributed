#ifndef CONFIG_H_
#define CONFIG_H_

#include <vector>
#include <mpi.h>
#include <string>

constexpr long BENCH_SIZE = 1000; //upper limit on maximum block size when in benchmark mode
constexpr long BENCH_REPEATS = 3; //number of times that a certain number of queries will be executed while in benchmark mode

enum class RequestDistribution {Constant, Variable_Poisson};
enum class ExecType {Single, Both, OutOfCore, Bench};
enum class SearchAlgorithm {Cpu, Gpu, Hybrid, CpuFixed, Fixed, Best};

class ExecPolicy;
class SearchStrategy;


struct SingleExec {
	double duration;
	long numQueries;
};

struct SingleTransfer {
	double duration;
};

struct Config {
	void loadConfig(std::string filename);
	
	//database config
	long d; //vector dimension
	std::string index_path;  
	std::string queries_path; 
	long distinct_queries;
	std::string gnd_path ;
	long dataset_size_reduction;
	
	
	//runtime config
	const long k = 10;
	const long nprobe = 16;
	const long block_size = 5;
	const long bench_step = block_size;
	const long gpus_per_node = 1;
	const long seed = 2;
	const long temp_memory_gpu = 0;
	const long poisson_intervals = 100;
	bool show_recall = false;
	
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
	
	std::vector<SingleExec> execs;
	std::vector<SingleTransfer> transfers;

	ExecType exec_type;
	
	ExecPolicy* exec_policy = nullptr;
	SearchStrategy* search_strategy = nullptr;
	
	SearchAlgorithm search_algorithm;
	
	int shard;
	int nshards;
	
	double raw_search_time = 0;
	
	MPI_Comm search_comm;
};

extern Config cfg;

#endif 
