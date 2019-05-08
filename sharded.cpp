#include <cstring>
#include <mpi.h>
#include <cassert>

#include "generator.h"
#include "search.h"
#include "aggregator.h"
#include "utils.h"
#include "config.h"

ProcType handle_parameters(int argc, char* argv[], Config& cfg) {
	std::map<std::string, RequestDistribution> m = { { "cs", RequestDistribution::Constant_Slow }, { "ca", RequestDistribution::Constant_Average }, { "cf",
			RequestDistribution::Constant_Fast }, { "vp", RequestDistribution::Variable_Poisson } };
	
	std::string usage = "./sharded b | d <cs|ca|cf|vp> <min|max> | s <cs|ca|cf|vp> <queries_per_block>";

	if (argc < 2) {
		std::printf("Wrong arguments.\n%s\n", usage.c_str());
		std::exit(-1);
	}

	ProcType ptype = ProcType::Bench;
	if (!strcmp("d", argv[1])) ptype = ProcType::Dynamic;
	else if (!strcmp("b", argv[1])) ptype = ProcType::Bench;
	else if (!strcmp("s", argv[1])) ptype = ProcType::Static;
	else {
		std::printf("Invalid processing type.Expected b | s | d\n");
		std::exit(-1);
	}

	if (ptype == ProcType::Dynamic) {
		if (argc != 4) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}

		std::string rd(argv[2]);
		cfg.request_distribution = m[rd];
		cfg.only_min = ! std::strcmp(argv[3], "min");
	} else if (ptype == ProcType::Static) {
		if (argc != 4) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			std::exit(-1);
		}

		std::string rd(argv[2]);
		cfg.request_distribution = m[rd];
		
		int nq = atoi(argv[3]); 
		assert(nq <= cfg.eval_length);
		assert(nq % cfg.block_size == 0);
		cfg.processing_size = nq / cfg.block_size;
	} else if (ptype == ProcType::Bench) {
		//nothing
		assert(BENCH_SIZE <= cfg.eval_length);
	}

	return ptype;
}

int main(int argc, char* argv[]) {
	Config cfg;

	MPI_Init(&argc, &argv);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
	ProcType ptype = handle_parameters(argc, argv, cfg);

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
