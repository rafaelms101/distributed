#ifndef READSPLITTEDINDEX_H_
#define READSPLITTEDINDEX_H_

#include "faiss/IndexIVFPQ.h"

faiss::Index *read_index (FILE * f, int shard, int total_shards, int io_flags = 0);

#endif /* READSPLITTEDINDEX_H_ */
