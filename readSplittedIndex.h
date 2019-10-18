#ifndef READSPLITTEDINDEX_H_
#define READSPLITTEDINDEX_H_

#include "faiss/IndexIVFPQ.h"

faiss::Index *read_index (FILE * f, float start_percent, float size_percent, int io_flags = 0);

#endif /* READSPLITTEDINDEX_H_ */
