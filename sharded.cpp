#include <cstring>
#include <mpi.h>
#include <cassert>

#include "generator.h"
#include "search.h"
#include "aggregator.h"
#include "utils.h"
#include "config.h"
#include "ExecPolicy.h"

void handle_parameters(int argc, char* argv[]) {
	std::string usage = "./sharded <c|p> <query_interval> <alg> <seed>";

	if (argc != 5) {
		std::printf("Wrong arguments.\n%s\n", usage.c_str());
		std::exit(-1);
	}

	if (! std::strcmp(argv[1], "c")) {
		cfg.request_distribution = RequestDistribution::Constant;
	} else if (! std::strcmp(argv[2], "p")) {
		cfg.request_distribution = RequestDistribution::Variable_Poisson;
	} else {
		std::printf("Wrong arguments.\n%s\n", usage.c_str());
		std::exit(- 1);
	}

	cfg.query_interval = std::atof(argv[2]);

	srand(std::atoi(argv[4]));
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
	handle_parameters(argc, argv);

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
    	generator(world_size - 2, cfg);
    } else if (world_rank == 0) {
    	aggregator(world_size - 2, cfg);
    } else {
    	search(world_rank - 2, world_size - 2, cfg);
    }
    
    // Finalize the MPI environment.
    MPI_Finalize();
}
