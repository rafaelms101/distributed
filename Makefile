# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ../faiss/makefile.inc

OPT = -O3
FAISS_HOME = ../faiss
FAISS_INCLUDE = -I $(FAISS_HOME)/.. -I $(FAISS_HOME)
FAISS_LIB = $(FAISS_HOME)/gpu/libgpufaiss.a $(FAISS_HOME)/libfaiss.a

%.o: %.cpp
	$(CXX) -g $(OPT) $(CUDACFLAGS) $(EXTRA) -o $@ -c $^ $(FAISS_INCLUDE) -std=c++17

sharded: utils.o readSplittedIndex.o generator.o search.o aggregator.o QueryBuffer.o sharded.o $(FAISS_LIB)
	mpic++ $(OPT) $(LDFLAGS) $(NVCCLDFLAGS) -o $@ $^ $(LIBS) $(NVCCLIBS)

train: train.o $(FAISS_LIB)
	mpic++ $(OPT) $(LDFLAGS) $(NVCCLDFLAGS) -o $@ $^ $(LIBS) $(NVCCLIBS)

simple: simple.o $(FAISS_LIB)
	g++ -g $(OPT) $(LDFLAGS) $(NVCCLDFLAGS) -o $@ $^ $(LIBS) $(NVCCLIBS)

clean:
	rm -f *.o demo_ivfpq_indexing_gpu sharded train simple

.PHONY: clean
