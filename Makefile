# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

FAISS_HOME = ../faiss

-include $(FAISS_HOME)/makefile.inc

OPT = -O3
MPI = mpic++

FAISS_INCLUDE = -I $(FAISS_HOME)/.. -I $(FAISS_HOME)
FAISS_LIB = $(FAISS_HOME)/libfaiss.a

all: sharded

%.o: %.cpp *.h
	$(MPI) -g -std=c++11  $(OPT) $(CPPFLAGS) -o $@ -c $< $(FAISS_INCLUDE) 

sharded: config.o utils.o readSplittedIndex.o generator.o search.o aggregator.o ExecPolicy.o Buffer.o SyncBuffer.o SearchStrategy.o sharded.o $(FAISS_LIB)
	$(MPI) -g -std=c++11 $(OPT) $(LDFLAGS) $(CPPFLAGS) -o $@ $^ $(LIBS) -lboost_filesystem -lboost_system -Wall

clean:
	rm -f *.o sharded

.PHONY: clean
