#ifndef CONFIG_H_
#define CONFIG_H_

constexpr int BENCH_SIZE = 10000; //upper limit on maximum block size when in benchmark mode

constexpr char SRC_PATH[] = "/home/rafael/mestrado/bigann"; //folder where the bigann database is stored
constexpr char INDEX_ROOT[] = "index"; //folder where the indexed databases are stored
constexpr int BENCH_REPEATS = 3; //number of times that a certain number of queries will be executed while in benchmark mode

struct Config {
	//database config
	int d = 128; //vector dimension
	int nb = 500000000; //database size
	int ncentroids = 8192; //number of centroids
	int m = 8; //vector size after compression, in bytes
	int nq = 10000; //total number of distinct queries
	
	//runtime config
	int k = 10;
	int nprobe = 16;
	int block_size = 20;
	int test_length = 100000; //how many queries will be sent in total
	int eval_length = 30000; //of the sent queries, how many will be used to compute the average response time
	
	//mode specific config
	double query_rate;
	
	int processing_size;
	double gpu_slice;
};

#endif 
