/*
 * readSplittedIndex.h
 *
 *  Created on: Dec 12, 2018
 *      Author: rafaelm
 */

#ifndef READSPLITTEDINDEX_H_
#define READSPLITTEDINDEX_H_


#include <faiss/index_io.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <bits/stdint-uintn.h>
#include <cstring>
#include <assert.h>

#include "faiss/FaissAssert.h"
#include "faiss/Index.h"
#include "faiss/IndexIVF.h"
#include "faiss/ProductQuantizer.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexFlat.h"
#include <vector>


#define WRITEANDCHECK(ptr, n) {                             \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);          \
        FAISS_THROW_IF_NOT_MSG(ret == (n), "write error");  \
    }

#define READANDCHECK(ptr, n) {                              \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);          \
        FAISS_THROW_IF_NOT_MSG(ret == (n), "read error");   \
    }

#define WRITE1(x) WRITEANDCHECK(&(x), 1)
#define READ1(x)  READANDCHECK(&(x), 1)

#define WRITEVECTOR(vec) {                      \
        size_t size = (vec).size ();            \
        WRITEANDCHECK (&size, 1);               \
        WRITEANDCHECK ((vec).data (), size);    \
    }

#define READVECTOR(vec) {                       \
        long size;                            \
        READANDCHECK (&size, 1);                \
        FAISS_THROW_IF_NOT (size >= 0 && size < (1L << 40));  \
        (vec).resize (size);                    \
        READANDCHECK ((vec).data (), size);     \
    }

struct IOReader {
    // fread
    virtual size_t operator()(void *ptr, size_t size, size_t nitems) = 0;

    // return a file number that can be memory-mapped
    virtual int fileno() = 0;

    virtual ~IOReader() {}
};

struct FileIOReader: IOReader {
    FILE *f = nullptr;

    FileIOReader(FILE *rf): f(rf) {}

    ~FileIOReader() = default;

    size_t operator()(
            void *ptr, size_t size, size_t nitems) override {
        return fread(ptr, size, nitems, f);
    }

    int fileno() override {
        return ::fileno (f);
    }

};


faiss::Index *read_index (FILE * f, int shard, int total_shards, int io_flags = 0);
faiss::Index *read_index (IOReader *f, int shard, int total_shards, int io_flags = 0);
faiss::Index* read_quantizer(IOReader *f);


static uint32_t fourcc (const char sx[4]) {
    assert(4 == strlen(sx));
    const unsigned char *x = (unsigned char*)sx;
    return x[0] | x[1] << 8 | x[2] << 16 | x[3] << 24;
}

static void read_index_header (faiss::Index *idx, IOReader *f) {
    READ1 (idx->d);
    READ1 (idx->ntotal);
    faiss::Index::idx_t dummy;
    READ1 (dummy);
    READ1 (dummy);
    READ1 (idx->is_trained);
    READ1 (idx->metric_type);
    idx->verbose = false;
}

static void read_ivf_header (
    faiss::IndexIVF *ivf, IOReader *f)
{
    read_index_header(ivf, f);
    READ1 (ivf->nlist);
    READ1 (ivf->nprobe);
    ivf->quantizer = read_quantizer(f);
    ivf->own_fields = true;
//    if (ids) { // used in legacy "Iv" formats
//        ids->resize (ivf->nlist);
//        for (size_t i = 0; i < ivf->nlist; i++)
//            READVECTOR ((*ids)[i]);
//    }
    READ1 (ivf->maintain_direct_map);
    READVECTOR (ivf->direct_map);
}

static void read_ProductQuantizer (faiss::ProductQuantizer *pq, IOReader *f) {
    READ1 (pq->d);
    READ1 (pq->M);
    READ1 (pq->nbits);
    pq->set_derived_values ();
    READVECTOR (pq->centroids);
}

static void read_ArrayInvertedLists_sizes (
         IOReader *f, std::vector<size_t> & sizes)
{
//    size_t nlist = sizes.size();
    uint32_t list_type;
    READ1(list_type);
    if (list_type == fourcc("full")) {
        size_t os = sizes.size();
        READVECTOR (sizes);
        FAISS_THROW_IF_NOT (os == sizes.size());
    } 
//    else if (list_type == fourcc("sprs")) {
//        std::vector<size_t> idsizes;
//        READVECTOR (idsizes);
//        for (size_t j = 0; j < idsizes.size(); j += 2) {
//            FAISS_THROW_IF_NOT (idsizes[j] < sizes.size());
//            sizes[idsizes[j]] = idsizes[j + 1];
//        }
//    } 
    else {
        FAISS_THROW_MSG ("invalid list_type");
    }
}


faiss::InvertedLists *read_InvertedLists (IOReader *f, int io_flags, int shard, int total_shards, faiss::Index::idx_t* ntotal) {
	*ntotal = 0;
	
	uint32_t h;
	READ1 (h);
	
	if (h == fourcc("ilar") && !(io_flags & faiss::IO_FLAG_MMAP)) {
		auto ails = new faiss::ArrayInvertedLists(0, 0);
		READ1(ails->nlist);
		READ1(ails->code_size);
		ails->ids.resize(ails->nlist);
		ails->codes.resize(ails->nlist);
		std::vector<size_t> sizes(ails->nlist);
		read_ArrayInvertedLists_sizes(f, sizes);
		

		for (size_t i = 0; i < ails->nlist; i++) {
			ails->ids[i].resize(sizes[i]);
			ails->codes[i].resize(sizes[i] * ails->code_size);
			
			size_t n = ails->ids[i].size();
			
			if (n > 0) {
				READANDCHECK(ails->codes[i].data(), n * ails->code_size);
				READANDCHECK(ails->ids[i].data(), n);
				
				// here we throw away the entries that are not IN this shard
				int c = 0;
				for (int j = shard; j < n; j += total_shards, c++) {
					ails->ids[i][c] = ails->ids[i][j];
					
					for (int d = 0; d < ails->code_size; d++) {
						ails->codes[i][c * ails->code_size + d] = ails->codes[i][j * ails->code_size + d];
					}
				}
				
				ails->ids[i].resize(c);
				ails->codes[i].resize(c * ails->code_size);
				*ntotal += c;
			}
		}
		
		return ails;
	} else {
        FAISS_THROW_MSG ("read_InvertedLists: unsupported invlist type");
    }
//	else if (h == fourcc("ilar") && (io_flags & IO_FLAG_MMAP)) {
//		// then we load it as an OnDiskInvertedLists
//
//		FileIOReader *reader = dynamic_cast<FileIOReader*>(f);
//		FAISS_THROW_IF_NOT_MSG(reader, "mmap only supported for File objects");
//		FILE *fdesc = reader->f;
//
//		auto ails = new OnDiskInvertedLists();
//		READ1(ails->nlist);
//		READ1(ails->code_size);
//		ails->read_only = true;
//		ails->lists.resize(ails->nlist);
//		std::vector<size_t> sizes(ails->nlist);
//		read_ArrayInvertedLists_sizes(f, sizes);
//		size_t o0 = ftell(fdesc), o = o0;
//		{ // do the mmap
//			struct stat buf;
//			int ret = fstat(fileno(fdesc), &buf);
//			FAISS_THROW_IF_NOT_FMT(ret == 0, "fstat failed: %s",
//					strerror(errno));
//			ails->totsize = buf.st_size;
//			ails->ptr = (uint8_t*) mmap(nullptr, ails->totsize, PROT_READ,
//					MAP_SHARED, fileno(fdesc), 0);
//			FAISS_THROW_IF_NOT_FMT(ails->ptr != MAP_FAILED,
//					"could not mmap: %s", strerror(errno));
//		}
//
//		for (size_t i = 0; i < ails->nlist; i++) {
//			OnDiskInvertedLists::List & l = ails->lists[i];
//			l.size = l.capacity = sizes[i];
//			l.offset = o;
//			o += l.size
//					* (sizeof(OnDiskInvertedLists::idx_t) + ails->code_size);
//		}
//		FAISS_THROW_IF_NOT(o <= ails->totsize);
//		// resume normal reading of file
//		fseek(fdesc, o, SEEK_SET);
//		return ails;
//	}
}

static void read_InvertedLists (faiss::IndexIVF *ivf, IOReader *f, int io_flags, int shard, int total_shards) {
	faiss::InvertedLists *ils = read_InvertedLists (f, io_flags, shard, total_shards, &ivf->ntotal);
    FAISS_THROW_IF_NOT (!ils || (ils->nlist == ivf->nlist &&
                                 ils->code_size == ivf->code_size));
    ivf->invlists = ils;
    ivf->own_invlists = true;
}

static faiss::IndexIVFPQ *read_ivfpq (IOReader *f, int shard, int total_shards, int io_flags)
{
	faiss::IndexIVFPQ * ivpq = new faiss::IndexIVFPQ();

    std::vector<std::vector<faiss::Index::idx_t>> ids;
    read_ivf_header(ivpq, f);
    READ1(ivpq->by_residual);
    READ1(ivpq->code_size);
    read_ProductQuantizer(&ivpq->pq, f);
    read_InvertedLists(ivpq, f, io_flags, shard, total_shards);
   
    // precomputed table not stored. It is cheaper to recompute it
    ivpq->use_precomputed_table = 0;
    if (ivpq->by_residual) ivpq->precompute_table();
  
    return ivpq;
}



faiss::Index* read_quantizer(IOReader *f) {
	faiss::Index * idx = nullptr;
	uint32_t h;
	READ1(h);
	    
	faiss::IndexFlat *idxf;
	if (h == fourcc("IxFI"))
		idxf = new faiss::IndexFlatIP();
	else
		idxf = new faiss::IndexFlatL2();
	read_index_header(idxf, f);
	READVECTOR(idxf->xb);
	FAISS_THROW_IF_NOT(idxf->xb.size() == idxf->ntotal * idxf->d);
	// leak!
	idx = idxf;
	
	return idx;
}

faiss::Index *read_index (IOReader *f, int shard, int total_shards, int io_flags) {
	faiss::Index * idx = nullptr;
	uint32_t h;
	READ1(h);
	idx = read_ivfpq(f, shard, total_shards, io_flags);
	return idx;
}



faiss::Index *read_index (FILE * f, int shard, int total_shards, int io_flags) {
	FileIOReader reader(f);
	return read_index(&reader, shard, total_shards, io_flags);
}

#endif /* READSPLITTEDINDEX_H_ */
