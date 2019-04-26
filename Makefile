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
	$(CXX) -g $(OPT) $(CPPFLAGS) -o $@ -c $^ $(FAISS_INCLUDE)

sharded: utils.o readSplittedIndex.o generator.o search.o aggregator.o Buffer.o sharded.o $(FAISS_LIB)
	mpic++ -std=c++11 $(OPT) $(LDFLAGS) $(CPPFLAGS) -o $@ $^ $(LIBS)

train: train.o $(FAISS_LIB)
	$(CXX) $(OPT) $(LDFLAGS) $(CPPFLAGS) -o $@ $^ $(LIBS)

simple: simple.o $(FAISS_LIB)
	$(CXX) $(OPT) $(LDFLAGS) $(CPPFLAGS) -o $@ $^ $(LIBS)

recall: recall.o $(FAISS_LIB)
		$(CXX) $(OPT) $(LDFLAGS) $(CPPFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o sharded train simple recall

.PHONY: clean
