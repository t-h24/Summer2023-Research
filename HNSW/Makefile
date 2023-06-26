CXX := g++
CXXFLAGS := -Wall

SRCS := hnsw.cpp run_hnsw.cpp beam_benchmark.cpp
OBJS := $(SRCS:.cpp=.o)
TARGETS := beam_benchmark.out hnsw_classic.out

.PHONY: all clean

all: $(TARGETS)

beam_benchmark.out: beam_benchmark.cpp hnsw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

hnsw_classic.out: hnsw.cpp run_hnsw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(OBJS) $(TARGETS)