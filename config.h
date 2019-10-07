#ifndef CONFIG_H_
#define CONFIG_H_

constexpr int BENCH_SIZE = 1000; //upper limit on maximum block size when in benchmark mode

constexpr char SRC_PATH[] = "/home/rafael/mestrado/bigann"; //folder where the bigann database is stored
constexpr char INDEX_ROOT[] = "index"; //folder where the indexed databases are stored
constexpr int BENCH_REPEATS = 3; //number of times that a certain number of queries will be executed while in benchmark mode
constexpr char PROF_ROOT[] = "prof";

enum class RequestDistribution {Constant, Variable_Poisson};
enum class SearchAlgorithm {Cpu, Gpu, Hybrid, CpuFixed, Fixed};

class ExecPolicy;

struct Config {
	//database config
	const int d = 128; //vector dimension
	const int nb = 500000000; //database size
	const int ncentroids = 4096; //number of centroids
	const int m = 8; //vector size after compression, in bytes
	const int nq = 10000; //total number of distinct queries
	
	//runtime config
	const int k = 10;
	const int nprobe = 16;
	const int block_size = 5;
	const int test_length = 100000; //how many queries will be sent in total
	const int eval_length = 100000; //of the sent queries, how many will be used to compute the average response time
	
	int shard;
	
	
	const int poisson_intervals = 100;
	
	//mode specific config
	RequestDistribution request_distribution;
	double query_load;
	int processing_size;
	bool only_min;
	
	SearchAlgorithm search_algorithm;
	
	//hybrid params
	long gpu_pieces;
	long total_pieces;
};

extern Config cfg;

#endif 
