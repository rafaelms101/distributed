#include <cstring>
#include <mpi.h>
#include <cassert>

#include "generator.h"
#include "search.h"
#include "aggregator.h"
#include "utils.h"
#include "config.h"
#include "ExecPolicy.h"

ProcType handle_parameters(int argc, char* argv[]) {
	std::string usage = "./sharded b | d <c|p> <load> <min|max|q|gmin> | s <c|p> <load> <queries_per_block>";

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
		if (argc != 5) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}

		if (! std::strcmp(argv[2], "c")) {
			cfg.request_distribution = RequestDistribution::Constant;
		} else if (! std::strcmp(argv[2], "p")) {
			cfg.request_distribution = RequestDistribution::Variable_Poisson;
		} else {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}
		
		cfg.load_factor = std::atof(argv[3]);
		
		
		if (! std::strcmp(argv[4], "min")) {
			cfg.exec_policy = new MinExecPolicy;
		} else if (! std::strcmp(argv[4], "max")) {
			cfg.exec_policy = new MaxExecPolicy;
		} else if (! std::strcmp(argv[4], "q")) {
			cfg.exec_policy = new QueueExecPolicy;
		} else if (! std::strcmp(argv[4], "gmin")) {
			cfg.exec_policy = new MinGreedyExecPolicy;
		} 
	} else if (ptype == ProcType::Static) {
		if (argc != 5) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}

		if (! std::strcmp(argv[2], "c")) {
			cfg.request_distribution = RequestDistribution::Constant;
		} else if (! std::strcmp(argv[2], "p")) {
			cfg.request_distribution = RequestDistribution::Variable_Poisson;
		} else {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(- 1);
		}

		cfg.load_factor = std::atof(argv[3]);
		
		int nq = atoi(argv[4]); 
		assert(nq <= cfg.eval_length);
		assert(nq % cfg.block_size == 0);
		cfg.processing_size = nq / cfg.block_size;
		
		cfg.exec_policy = new StaticExecPolicy(cfg.processing_size);
	} else if (ptype == ProcType::Bench) {
		cfg.exec_policy = new BenchExecPolicy;
		assert(BENCH_SIZE <= cfg.eval_length);
	}

	return ptype;
}

int main(int argc, char* argv[]) {
	srand(42);
//	srand(time(NULL));

	MPI_Init(&argc, &argv);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
	ProcType ptype = handle_parameters(argc, argv);

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
    	generator(world_size - 2, ptype, cfg);
    } else if (world_rank == 0) {
    	aggregator(world_size - 2, ptype, cfg);
    } else {
    	search(world_rank - 2, world_size - 2, ptype, cfg);
    }
    
    // Finalize the MPI environment.
    MPI_Finalize();
}
