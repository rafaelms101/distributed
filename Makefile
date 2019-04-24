# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

FAISS_HOME = ../faiss

-include $(FAISS_HOME)/makefile.inc

OPT = -O3

FAISS_INCLUDE = -I $(FAISS_HOME)/.. -I $(FAISS_HOME)
FAISS_LIB = $(FAISS_HOME)/libfaiss.a

%.o: %.cpp
	$(CXX) -g $(OPT) $(CPPFLAGS) -o $@ -c $^ $(FAISS_INCLUDE) -std=c++17

sharded: utils.o readSplittedIndex.o generator.o search.o aggregator.o QueryBuffer.o sharded.o $(FAISS_LIB)
	mpic++ $(OPT) $(LDFLAGS) $(CPPFLAGS) -o $@ $^ $(LIBS)

train: train.o $(FAISS_LIB)
	mpic++ $(OPT) $(LDFLAGS) $(CPPFLAGS) -o $@ $^ $(LIBS)

simple: simple.o $(FAISS_LIB)
	g++ $(OPT) $(LDFLAGS) $(CPPFLAGS) -o $@ $^ $(LIBS)

recall: recall.o $(FAISS_LIB)
		g++ $(OPT) $(LDFLAGS) $(CPPFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o demo_ivfpq_indexing_gpu sharded train simple

.PHONY: clean
