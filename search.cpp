#include "search.h"

#include <mpi.h>
#include <algorithm>
#include <future>
#include <fstream>
#include <unistd.h>
#include <list>
#include <vector>
 
#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

#include "gpu/GpuAutoTune.h"
#include "gpu/StandardGpuResources.h"
#include "gpu/GpuIndexIVFPQ.h"

#include "utils.h"
#include "Buffer.h"
#include "ExecPolicy.h"
#include "SearchStrategy.h"

static void comm_handler(Buffer* query_buffer, Buffer* distance_buffer, Buffer* label_buffer, bool& finished_receiving, Config& cfg) {
	long queries_received = 0;
	long queries_sent = 0;
	
	float dummy;
	MPI_Send(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD); //signal that we are ready to receive queries

	while (! finished_receiving || queries_sent < queries_received) {
		if (! finished_receiving) {
			MPI_Status status;
			int message_arrived;
			MPI_Iprobe(GENERATOR, 0, MPI_COMM_WORLD, &message_arrived, &status);

			if (message_arrived) {
				int message_size;
				MPI_Get_count(&status, MPI_FLOAT, &message_size);

				if (message_size == 1) {
					finished_receiving = true;
					float dummy;
					MPI_Recv(&dummy, 1, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
				} else {
					query_buffer->waitForSpace(1);
					float* recv_data = reinterpret_cast<float*>(query_buffer->peekEnd());
					MPI_Recv(recv_data, cfg.block_size * cfg.d, MPI_FLOAT, GENERATOR, 0, MPI_COMM_WORLD, &status);
					query_buffer->add(1);
					queries_received += cfg.block_size;
					
					assert(status.MPI_ERROR == MPI_SUCCESS);
				}
			}
		}
		
		auto ready = std::min(distance_buffer->entries(), label_buffer->entries());

		if (ready >= 1) {
			queries_sent += ready * cfg.block_size;
			
			faiss::Index::idx_t* label_ptr = reinterpret_cast<faiss::Index::idx_t*>(label_buffer->peekFront());
			float* dist_ptr = reinterpret_cast<float*>(distance_buffer->peekFront());
			
			//TODO: Optimize this to an Immediate Synchronous Send
			MPI_Ssend(label_ptr, cfg.k * cfg.block_size * ready, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
			MPI_Ssend(dist_ptr, cfg.k * cfg.block_size * ready, MPI_FLOAT, AGGREGATOR, 1, MPI_COMM_WORLD);
			
			label_buffer->consume(ready);
			distance_buffer->consume(ready);
		}
	}

	MPI_Ssend(&dummy, 1, MPI_LONG, AGGREGATOR, 0, MPI_COMM_WORLD);
	deb("Finished receiving queries");	
}

void search(int shard, int nshards, Config& cfg) {
	deb("search called");
	
	const long block_size_in_bytes = sizeof(float) * cfg.d * cfg.block_size;
	Buffer query_buffer(block_size_in_bytes, 100 * 1024 * 1024 / block_size_in_bytes); //100 MB
	
	const long distance_block_size_in_bytes = sizeof(float) * cfg.k * cfg.block_size;
	const long label_block_size_in_bytes = sizeof(faiss::Index::idx_t) * cfg.k * cfg.block_size;
	Buffer distance_buffer(distance_block_size_in_bytes, 100 * 1024 * 1024 / distance_block_size_in_bytes); //100 MB 
	Buffer label_buffer(label_block_size_in_bytes, 100 * 1024 * 1024 / label_block_size_in_bytes); //100 MB 
	
	faiss::gpu::StandardGpuResources res;
	res.setTempMemory(1500 * 1024 * 1024);
	
	SearchStrategy* strategy;
	
	float shard_size = 1.0 / nshards;
	float base_start = shard * shard_size;
	float base_end = base_start + shard_size;
	
	if (cfg.search_algorithm == SearchAlgorithm::Cpu) {
		strategy = new CpuOnlySearchStrategy(query_buffer, distance_buffer, label_buffer, base_start, base_end);
	} else if (cfg.search_algorithm == SearchAlgorithm::Hybrid) {
		strategy = new HybridSearchStrategy(query_buffer, distance_buffer, label_buffer, base_start, base_end, &res);
	} else if (cfg.search_algorithm == SearchAlgorithm::Gpu) {
		strategy = new GpuOnlySearchStrategy(query_buffer, distance_buffer, label_buffer, base_start, base_end, &res);
	} else if (cfg.search_algorithm == SearchAlgorithm::CpuFixed) {
		strategy = new CpuFixedSearchStrategy(query_buffer, distance_buffer, label_buffer, base_start, base_end, &res);
	} else if (cfg.search_algorithm == SearchAlgorithm::Fixed) {
		strategy = new FixedSearchStrategy(query_buffer, distance_buffer, label_buffer, base_start, base_end, &res);
	}
	
	strategy->setup();
	
	bool finished = false;
	std::thread receiver { comm_handler, &query_buffer, &distance_buffer, &label_buffer, std::ref(finished), std::ref(cfg) };
	
	strategy->start_search_process();
	
	receiver.join();
}
