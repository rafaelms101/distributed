#ifndef SEARCH_H_
#define SEARCH_H_

#include "utils.h"
#include "config.h"

#include "ExecPolicy.h"

void search_single(int shard, ExecPolicy* policy, long num_blocks);
void search_both(int shard, ExecPolicy* cpu_policy, ExecPolicy* gpu_policy, long num_blocks, double gpu_throughput, double cpu_throughput); 
void search_out(int shard, SearchAlgorithm search_algorithm);

#endif /* SEARCH_H_ */
