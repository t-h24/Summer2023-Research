CXX := g++
CXXFLAGS := -Wall

SRCS := hnsw.cpp run_hnsw.cpp beam_benchmark.cpp
OBJS := $(SRCS:.cpp=.o)
TARGETS := beam_benchmark.out run_hnsw.out

.PHONY: all clean

all: $(TARGETS)

beam_benchmark.out: beam_benchmark.cpp hnsw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

run_hnsw.out: hnsw.cpp run_hnsw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(OBJS) $(TARGETS)