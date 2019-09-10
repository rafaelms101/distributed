#include <cstring>
#include <mpi.h>
#include <cassert>

#include "generator.h"
#include "search.h"
#include "aggregator.h"
#include "utils.h"
#include "config.h"
#include "ExecPolicy.h"

static void fill_poisson_rates() {
	for (int i = 0; i < cfg.poisson_intervals.size(); i++) {
		cfg.poisson_intervals[i] = poisson_interval(cfg.query_interval);
	}
}

static void fill_test_length() {
	double interval_length = cfg.test_duration / cfg.poisson_intervals.size();
	cfg.test_length = 0;
	
	double time = 0;
	
	for (double interval : cfg.poisson_intervals) {
		while (time < interval_length) {
			cfg.test_length++;
			time += interval;
		}
		
		time = time - interval_length;
	}
}

static void process_query_distribution(char* type) {
	if (! std::strcmp(type, "c")) {
		cfg.request_distribution = RequestDistribution::Constant;

		assert(cfg.query_interval > 0);
		
		cfg.test_length = int(cfg.test_duration / cfg.query_interval);
	} else if (! std::strcmp(type, "p")) {
		cfg.request_distribution = RequestDistribution::Variable_Poisson;
		fill_poisson_rates();
		fill_test_length();
	} else if (! std::strcmp(type, "b")) {
		cfg.request_distribution = RequestDistribution::Batch;
		assert(cfg.query_interval == 0);
		cfg.test_length = 10000;
	} else {
		std::printf("Wrong query distribution. Use 'p' or 'c'\n");
		std::exit(-1);
	}
	
	cfg.test_length = cfg.test_length - cfg.test_length % cfg.block_size;
	cfg.eval_length = cfg.test_length;
	
//	std::printf("test length: %d\n", cfg.test_length);
	
	deb("test_length: %d", cfg.test_length);
}

static ProcType handle_parameters(int argc, char* argv[], int shard) {
	std::string usage = "./sharded b | d <c|p> <query_interval> <min|max|q|g|gmin|c> <seed> | s <c|p> <query_interval> <queries_per_block> <seed>";

	if (argc < 2) {
		std::printf("Wrong arguments.\n%s\n", usage.c_str());
		std::exit(-1);
	}

	ProcType ptype = ProcType::Bench;
	if (! strcmp("d", argv[1])) ptype = ProcType::Dynamic;
	else if (! strcmp("b", argv[1])) ptype = ProcType::Bench;
	else if (! strcmp("s", argv[1])) ptype = ProcType::Static;
	else {
		std::printf("Invalid processing type.Expected b | s | d\n");
		std::exit(-1);
	}

	if (ptype == ProcType::Dynamic) {
		if (argc != 6) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}

		cfg.query_interval = std::atof(argv[3]);
		process_query_distribution(argv[2]);
		
		if (shard >= 0) {
			if (! std::strcmp(argv[4], "min")) {
				cfg.exec_policy = new MinExecPolicy(shard);
			} else if (! std::strcmp(argv[4], "max")) {
				cfg.exec_policy = new MaxExecPolicy(shard);
			} else if (! std::strcmp(argv[4], "q")) {
				cfg.exec_policy = new QueueExecPolicy(shard);
			} else if (! std::strcmp(argv[4], "gmin")) {
				cfg.exec_policy = new MinGreedyExecPolicy(shard);
			} else if (! std::strcmp(argv[4], "g")) {
				cfg.exec_policy = new GreedyExecPolicy();
			} else if (! std::strcmp(argv[4], "qmax")) {
				cfg.exec_policy = new QueueMaxExecPolicy(shard);
			} else if (! std::strcmp(argv[4], "c")) {
				cfg.exec_policy = new CPUGreedyPolicy();
			} else if (! std::strcmp(argv[4], "h")) {
				cfg.exec_policy = new HybridBatch(3.3);
			} 
		}
		
		srand(std::atoi(argv[5]));
	} else if (ptype == ProcType::Static) {
		if (argc != 6) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}

		cfg.query_interval = std::atof(argv[3]);
		process_query_distribution(argv[2]);
	
		int nq = atoi(argv[4]); 
		deb("%d <= %d", nq, cfg.eval_length);
		assert(nq <= cfg.eval_length);
		assert(nq % cfg.block_size == 0);
		cfg.processing_size = nq / cfg.block_size;
		
		cfg.exec_policy = new StaticExecPolicy(cfg.processing_size);
		
		srand(std::atoi(argv[5]));
	} else if (ptype == ProcType::Bench) {
		cfg.exec_policy = new BenchExecPolicy(shard);
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
    	search(shard, nshards, ptype, cfg);
    }
    
    // Finalize the MPI environment.
    MPI_Finalize();
}
