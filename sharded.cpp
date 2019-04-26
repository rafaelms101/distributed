#include <string>
#include <mpi.h>
#include <cassert>

#include "generator.h"
#include "search.h"
#include "aggregator.h"
#include "utils.h"

ProcType handle_parameters(int argc, char* argv[], Config& cfg) {
	std::string usage = "./sharded b | d <qr> | s <qr> <num_queries>";

	if (argc < 2) {
		std::printf("Wrong arguments.\n%s\n", usage.c_str());
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	ProcType ptype = ProcType::Bench;
	if (!strcmp("d", argv[1])) ptype = ProcType::Dynamic;
	else if (!strcmp("b", argv[1])) ptype = ProcType::Bench;
	else if (!strcmp("s", argv[1])) ptype = ProcType::Static;
	else {
		std::printf("Invalid processing type.Expected b | s | d\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	if (ptype == ProcType::Dynamic) {
		if (argc != 3) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			MPI_Abort(MPI_COMM_WORLD, -1);
		}

		cfg.query_rate = atof(argv[2]);
	} else if (ptype == ProcType::Static) {
		if (argc != 4) {
			std::printf("Wrong arguments.\n%s\n", usage.c_str());
			MPI_Abort(MPI_COMM_WORLD, -1);
		}

		cfg.query_rate = atof(argv[2]);
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

Config inline loadConfig() {
	Config cfg; 
	cfg.d = 128;
	cfg.nb = 500000000;
	cfg.ncentroids = 8192;
	cfg.m = 8;
	cfg.k = 100;
	cfg.nprobe = 8;
	cfg.block_size = 20;
	cfg.test_length = 100000;
	cfg.eval_length = 10000;
	cfg.nq = 10000;
	
	return cfg;
}

int main(int argc, char* argv[]) {
	Config cfg = loadConfig();

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
