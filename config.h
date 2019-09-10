#ifndef CONFIG_H_
#define CONFIG_H_

#include <vector>

constexpr int BENCH_SIZE = 5000; //upper limit on maximum block size when in benchmark mode

constexpr char SRC_PATH[] = "/home/rafael/mestrado/bigann"; //folder where the bigann database is stored
constexpr char INDEX_ROOT[] = "index"; //folder where the indexed databases are stored
constexpr char PROF_ROOT[] = "prof"; //folder where the indexed databases are stored
constexpr int BENCH_REPEATS = 3; //number of times that a certain number of queries will be executed while in benchmark mode

enum class RequestDistribution {Constant, Variable_Poisson, Batch};

class ExecPolicy;

struct Config {
	//database config
	int d = 128; //vector dimension
	int nb = 500000000; //database size
	int ncentroids = 4096; //number of centroids
	int m = 8; //vector size after compression, in bytes
	int nq = 10000; //total number of distinct queries
	
	//runtime config
	int k = 10;
	int nprobe = 16;
	int block_size = 5;
	double test_duration = 2;
	
	// filled elsewhere
	int test_length; //how many queries will be sent in total
	int eval_length; //of the sent queries, how many will be used to compute the average response time
	
	long temp_memory_gpu = 800 * 1024 * 1024;
	std::vector<double> poisson_intervals = std::vector<double>(100);
	
	
	//mode specific config
	RequestDistribution request_distribution;
	double query_interval;
	int processing_size;
	bool only_min;
	
	ExecPolicy* exec_policy = nullptr;
};

extern Config cfg;

#endif 
