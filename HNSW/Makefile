CXX := g++
CXXFLAGS := -O2 -mavx

SRCS := hnsw.cpp run_hnsw.cpp beam_benchmark.cpp hnsw_save.cpp
OBJS := $(SRCS:.cpp=.o)
TARGETS := beam_benchmark.out run_hnsw.out hnsw_save.out

.PHONY: all clean

all: $(TARGETS)

beam_benchmark.out: beam_benchmark.cpp hnsw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

run_hnsw.out: run_hnsw.cpp hnsw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

hnsw_save.out: hnsw_save.cpp hnsw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(OBJS) $(TARGETS)