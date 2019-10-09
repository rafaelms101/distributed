#include <cstring>
#include <mpi.h>
#include <cassert>

#include "generator.h"
#include "search.h"
#include "aggregator.h"
#include "utils.h"
#include "config.h"
#include "ExecPolicy.h"

static void process_query_distribution(char** argv) {
	assert(cfg.test_length % cfg.block_size == 0);
	
	cfg.query_load = std::atof(argv[1]);
	
	if (! std::strcmp(argv[0], "c")) {
		cfg.request_distribution = RequestDistribution::Constant;
	} else if (! std::strcmp(argv[0], "p")) {
		cfg.request_distribution = RequestDistribution::Variable_Poisson;
	} else if (! std::strcmp(argv[0], "b")) {
		cfg.request_distribution = RequestDistribution::Batch;
		assert(cfg.query_load == 0);
	} else {
		std::printf("Wrong query distribution. Use 'p' or 'c'\n");
		std::exit(-1);
	}
}

static ProcType handle_parameters(int argc, char* argv[], int shard) {
	std::string usage = "./sharded b | d <c|p> <query_interval> <min|max|q|g|gmin|c|...> <seed> <gpu_throughput> <cpu_throughput>  | s <c|p> <query_interval> <queries_per_block> <gpu|cpu|h> <seed> <gpu_throughput> <cpu_throughput> ";

	if (argc < 2) {
		std::printf("Wrong arguments.\n%s\n", usage.c_str());
		std::exit(-1);
	}

	ProcType ptype = ProcType::Bench;
	if (! strcmp("d", argv[1])) ptype = ProcType::Dynamic;
	else if (! strcmp("b", argv[1])) assert(false);
	else if (! strcmp("s", argv[1])) ptype = ProcType::Static;
	else {
		std::printf("Invalid processing type.Expected b | s | d\n");
		std::exit(-1);
	}

	if (ptype == ProcType::Dynamic) {
		if (argc != 8) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}

		process_query_distribution(&argv[2]);
		
		if (shard >= 0) {
			if (! std::strcmp(argv[4], "min")) {
				cfg.gpu_exec_policy = new MinExecPolicy();
			} else if (! std::strcmp(argv[4], "max")) {
				cfg.gpu_exec_policy = new MaxExecPolicy();
			} else if (! std::strcmp(argv[4], "q")) {
				cfg.gpu_exec_policy = new QueueExecPolicy();
			} else if (! std::strcmp(argv[4], "gmin")) {
				cfg.gpu_exec_policy = new MinGreedyExecPolicy();
			} else if (! std::strcmp(argv[4], "g")) {
				cfg.gpu_exec_policy = new GreedyExecPolicy();
			} else if (! std::strcmp(argv[4], "qmax")) {
				cfg.gpu_exec_policy = new QueueMaxExecPolicy();
			} else if (! std::strcmp(argv[4], "c")) {
				cfg.gpu_exec_policy = new CPUGreedyPolicy();
			} else if (! std::strcmp(argv[4], "h")) {
				cfg.gpu_exec_policy = new HybridPolicy();
			} else if (! std::strcmp(argv[4], "ge")) {
				cfg.gpu_exec_policy = new GeorgeExecPolicy();
			} else if (! std::strcmp(argv[4], "b")) {
				cfg.gpu_exec_policy = new BestExecPolicy();
			} else if (! std::strcmp(argv[4], "hg")) {
				cfg.gpu_exec_policy = new HybridCompositePolicy(new GreedyExecPolicy());
			} else if (! std::strcmp(argv[4], "hge")) {
				cfg.gpu_exec_policy = new HybridCompositePolicy(new GeorgeExecPolicy());
			} 
		}
		
		srand(std::atoi(argv[5]));
		cfg.gpu_throughput = atoi(argv[6]);
		cfg.cpu_throughput = atoi(argv[7]);
	} else if (ptype == ProcType::Static) {
		if (argc != 9) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}

		process_query_distribution(&argv[2]);
	
		int nq = atoi(argv[4]); 
		deb("%d <= %d", nq, cfg.eval_length);
		assert(nq <= cfg.eval_length);
		assert(nq % cfg.block_size == 0);
		cfg.processing_size = nq / cfg.block_size;
		
		bool gpu = ! std::strcmp(argv[5], "gpu");
		bool hybrid = ! std::strcmp(argv[5], "h");
		
		if (hybrid) cfg.gpu_exec_policy = new HybridBatch(cfg.processing_size);
		else cfg.gpu_exec_policy = new StaticExecPolicy(gpu, cfg.processing_size);

		srand(std::atoi(argv[6]));
		cfg.gpu_throughput = atoi(argv[7]);
		cfg.cpu_throughput = atoi(argv[8]);
	} else if (ptype == ProcType::Bench) {
		cfg.gpu_exec_policy = new BenchExecPolicy();
		cfg.test_length = cfg.eval_length = BENCH_SIZE * BENCH_REPEATS;
	}

	return ptype;
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
    
    MPI_Comm search_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, search_group, 0, &search_comm);
    
    
    int shard = world_rank - 2;
    int nshards = world_size - 2;
    ProcType ptype = handle_parameters(argc, argv, shard);
    

    if (world_rank == 1) {
    	generator(world_size - 2, ptype, cfg);
    } else if (world_rank == 0) {
    	aggregator(world_size - 2, ptype, cfg);
    } else {
    	cfg.cpu_exec_policy = new CPUGreedyPolicy();
    	int blocks_gpu = std::nearbyint(double(cfg.gpu_throughput) / cfg.cpu_throughput);
    	cfg.gpu_test_length = blocks_gpu * cfg.block_size;
    	search(blocks_gpu, ptype, shard, cfg);
    }
    
    // Finalize the MPI environment.
    MPI_Finalize();
}
