#include <cstring>
#include <mpi.h>
#include <cassert>
#include <cuda_runtime.h>

#include "generator.h"
#include "search.h"
#include "aggregator.h"
#include "utils.h"
#include "config.h"
#include "ExecPolicy.h"

static void process_query_distribution(char** argv) {
	cfg.query_load = std::atof(argv[1]);
	
	if (cfg.query_load < 0) {
		std::printf("Invalid load (%s)\n", argv[1]);
		std::exit(-1);
	}
	
	if (!std::strcmp(argv[0], "c")) {
		cfg.request_distribution = RequestDistribution::Constant;
	} else if (!std::strcmp(argv[0], "p")) {
		cfg.request_distribution = RequestDistribution::Variable_Poisson;
		
		if (cfg.query_load <= 0) {
			std::printf("Invalid load for a poisson distribution (%s)\n", argv[1]);
			std::exit(-1);
		}
	} else {
		std::printf("Wrong query distribution. Use 'p' or 'c'\n");
		std::exit(-1);
	}
	
	
}

static void process_exec_policy(char** argv) {
	cfg.total_pieces = 1;
	
	if (!std::strcmp(argv[0], "min")) {
		cfg.exec_policy = new MinExecPolicy();
	} else if (!std::strcmp(argv[0], "max")) {
		cfg.exec_policy = new MaxExecPolicy();
	} else if (!std::strcmp(argv[0], "q")) {
		cfg.exec_policy = new QueueExecPolicy();
	} else if (!std::strcmp(argv[0], "gmin")) {
		cfg.exec_policy = new MinGreedyExecPolicy();
	} else if (!std::strcmp(argv[0], "g")) {
		cfg.exec_policy = new GreedyExecPolicy();
	} else if (!std::strcmp(argv[0], "qmax")) {
		cfg.exec_policy = new QueueMaxExecPolicy();
	} else if (!std::strcmp(argv[0], "c")) {
		cfg.exec_policy = new CPUGreedyPolicy();
	} else if (!std::strcmp(argv[0], "h")) {
		cfg.exec_policy = new HybridPolicy();
	} else if (!std::strcmp(argv[0], "ge")) {
		cfg.exec_policy = new GeorgeExecPolicy();
	} else if (!std::strcmp(argv[0], "b")) {
		cfg.exec_policy = new BestExecPolicy();
	} else if (!std::strcmp(argv[0], "hg")) {
		cfg.exec_policy = new HybridCompositePolicy(new GreedyExecPolicy());
	} else if (!std::strcmp(argv[0], "s")) {
		int block_size = std::atoi(argv[1]) / cfg.block_size;
		bool gpu = ! std::strcmp(argv[2], "gpu");
		cfg.exec_policy = new StaticExecPolicy(gpu, block_size);
	} else {
		std::printf("Wrong algorithm specified (%s)\n", argv[0]);
		std::exit(-1);
	}
}

static void process_search_strategy(char** argv) {
	if (!std::strcmp(argv[0], "c")) {
		cfg.search_algorithm = SearchAlgorithm::Cpu;
		cfg.total_pieces = 1;
	} else if (!std::strcmp(argv[0], "g")) {
		cfg.search_algorithm = SearchAlgorithm::Gpu;
		cfg.gpu_pieces = atoi(argv[1]);
		cfg.total_pieces = cfg.gpu_pieces;
	} else if (!std::strcmp(argv[0], "h")) {
		cfg.search_algorithm = SearchAlgorithm::Hybrid;
		cfg.gpu_pieces = atoi(argv[1]);
		cfg.total_pieces = atoi(argv[2]);
	}  else if (!std::strcmp(argv[0], "b")) {
		cfg.search_algorithm = SearchAlgorithm::Best;
		cfg.gpu_pieces = atoi(argv[1]);
		cfg.total_pieces = atoi(argv[2]);
	} else if (!std::strcmp(argv[0], "f")) {
		cfg.search_algorithm = SearchAlgorithm::Fixed;
		cfg.gpu_pieces = atoi(argv[1]);
		cfg.total_pieces = atoi(argv[2]);
	} else {
		std::printf("Wrong algorithm specified (%s)\n", argv[0]);
		std::exit(-1);
	} 
}

static void handle_parameters(int argc, char* argv[], int shard) {
	std::string usage = "./sharded b <cpu|gpu|both> <parts> | sharded <c|p> <query_interval> <s|o> <alg params> | sharded <c|p> <query_interval> b <gpu_throughput> <cpu_throughput> <alg params>";

	if (argc < 2) {
		std::printf("Wrong arguments.\n%s\n", usage.c_str());
		std::exit(-1);
	}

	if (! strcmp("b", argv[1])) {
		if (argc != 4) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}
		
		cfg.exec_type = ExecType::Bench;
		cfg.exec_policy = new BenchExecPolicy();
		
		
		
		if (! strcmp("cpu", argv[2])) {
			cfg.bench_cpu = true;
			cfg.bench_gpu = false;
		} else if (! strcmp("gpu", argv[2])) {
			cfg.bench_cpu = false;
			cfg.bench_gpu = true;
		} else if (! strcmp("both", argv[2])) {
			cfg.bench_cpu = true;
			cfg.bench_gpu = true;
		} else {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}

		cfg.total_pieces = atoi(argv[3]);
	} else {
		if (argc < 5) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}
		
		process_query_distribution(&argv[1]);
		
		if (! strcmp("s", argv[3])) {
			cfg.exec_type = ExecType::Single;
			process_exec_policy(&argv[4]);
		} else if (! strcmp("b", argv[3])) {
			cfg.exec_type = ExecType::Both;
			cfg.gpu_throughput = atof(argv[4]);
			cfg.cpu_throughput = atof(argv[5]);
			process_exec_policy(&argv[6]);
		} else if (! strcmp("o", argv[3])) {
			cfg.exec_type = ExecType::OutOfCore;
			process_search_strategy(&argv[4]);
		} else {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}
	}
}

static void logSearchStart(int shard) {
	char hostname[1024];
	gethostname(hostname, 1024);
	int n;
	cudaGetDeviceCount(&n);
	std::printf("%d) Starting at %s. Visible gpus: %d\n", shard, hostname, n);
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    
    MPI_Group search_group;
    int ranks[] = {0};
    MPI_Group_excl(world_group, 1, ranks, &search_group);
    
    MPI_Comm_create_group(MPI_COMM_WORLD, search_group, 112758, &cfg.search_comm);
    
    
    int shard = world_rank - 2;
    int nshards = world_size - 2;
    
    handle_parameters(argc, argv, shard);
    
    srand(cfg.seed);
    
    if (cfg.exec_type == ExecType::Bench) {
		assert(BENCH_SIZE % cfg.block_size == 0);

		long n = BENCH_SIZE / cfg.bench_step;
		cfg.num_blocks = BENCH_REPEATS * cfg.bench_step * n * (n + 1) / 2 / cfg.block_size;
		
		assert(cfg.num_blocks >= 1);
    }
//    
//    //TODO: remember to pass the right amount of blocks in the bench case
    if (world_rank == 1) {
    	generator(nshards, cfg);
    } else if (world_rank == 0) {
    	aggregator(nshards, cfg);
    } else {
//    	logSearchStart(shard);
    	cfg.shard = shard;
    	
    	if (cfg.exec_type == ExecType::Bench) {
    		search_single(shard, cfg.exec_policy, cfg.num_blocks);
    	} else if (cfg.exec_type == ExecType::Single) {
    		search_single(shard, cfg.exec_policy, cfg.num_blocks);
    	} else if (cfg.exec_type == ExecType::Both) {
    		search_both(shard, new CPUGreedyPolicy(), cfg.exec_policy, cfg.num_blocks, cfg.gpu_throughput, cfg.cpu_throughput);
    	} else if (cfg.exec_type == ExecType::OutOfCore) {
    		search_out(shard, cfg.search_algorithm);
    	}
    }
    
    // Finalize the MPI environment.
    MPI_Finalize();
}
