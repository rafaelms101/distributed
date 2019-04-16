# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ../faiss/makefile.inc

OPT = -O3

%.o: %.cpp
	$(CXX) -g $(OPT) $(CUDACFLAGS) $(EXTRA) -o $@ -c $^ -I.. -I../faiss -std=c++17

demo_ivfpq_indexing_gpu: demo_ivfpq_indexing_gpu.o
	$(CXX) $(OPT) $(LDFLAGS) $(NVCCLDFLAGS) -o $@ $^ $(LIBS) $(NVCCLIBS)

sharded: utils.o generator.o search.o aggregator.o QueryBuffer.o sharded.o ../faiss/gpu/libgpufaiss.a ../faiss/libfaiss.a 
	mpic++ $(OPT) $(LDFLAGS) $(NVCCLDFLAGS) -o $@ $^ $(LIBS) $(NVCCLIBS)

train: train.o ../faiss/gpu/libgpufaiss.a ../faiss/libfaiss.a
	mpic++ $(OPT) $(LDFLAGS) $(NVCCLDFLAGS) -o $@ $^ $(LIBS) $(NVCCLIBS)

simple: simple.o ../faiss/gpu/libgpufaiss.a ../faiss/libfaiss.a
	g++ -g $(OPT) $(LDFLAGS) $(NVCCLDFLAGS) -o $@ $^ $(LIBS) $(NVCCLIBS)

profile: profile.o ../faiss/gpu/libgpufaiss.a ../faiss/libfaiss.a
	g++ -g $(OPT) $(LDFLAGS) $(NVCCLDFLAGS) -o $@ $^ $(LIBS) $(NVCCLIBS)

clean:
	rm -f *.o demo_ivfpq_indexing_gpu sharded train simple

.PHONY: clean
