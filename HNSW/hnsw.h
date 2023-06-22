#include <vector>
#include <map>
#include <fstream>

#ifndef HNSW_H
#define HNSW_H

class Config {
public:
    int generation_seed = 0;
    int graph_seed = 100000;

    int dimensions = 2;
    int num_nodes = 40;
    int optimal_connections = 3;
    int max_connections = 6;
    int ef_construction = 8;
    double scaling_factor = 0.5;

    int num_queries = 10;
    int num_return = 5;

    bool debug_insert = false;
    bool debug_graph = false;
    bool debug_search = false;

    bool export_graph = true;
    bool export_queries = true;

    int debug_query_search_index = -1;
};

class Node {
public:
    int index;
    int dimensions;
    int level;
    float* values;
    std::ofstream* debug_file;

    Node(int index, int dimensions, float* values);

    double distance(Node* other);

    ~Node();
};

class HNSWLayer {
public:
    std::map<int, std::deque<Node*>> mappings;
};

class HNSW {
public:
    int node_size;
    Node** nodes;
    std::vector<HNSWLayer*> layers;
    Node* entry_point;

    HNSW(int node_size, Node** nodes);

    int get_layers();

    ~HNSW();
};

Node** generate_nodes(int dimensions, int amount, int seed);

// Main algorithms
HNSW* insert(Config* config, HNSW* hnsw, Node* query, int est_con, int max_con, int ef_con, float normal_factor);
std::deque<Node*> search_layer(Config* config, HNSW* hnsw, Node* query, std::deque<Node*> entry_points, int num_to_return, int layer_num);
std::deque<Node*> select_neighbors_simple(Config* config, HNSW* hnsw, Node* query, std::deque<Node*> candidates, int num, bool drop);
std::deque<Node*> nn_search(Config* config, HNSW* hnsw, Node* query, int num_to_return, int ef_con, std::vector<int>& path);

// Executing HNSW
bool sanity_checks(Config* config);
HNSW* init_hnsw(Config* config, Node** nodes);
void insert_nodes(Config* config, Node** nodes, HNSW* hnsw);
void print_hnsw(Config* config, HNSW* hnsw);
void run_query_search(Config* config, HNSW* hnsw, Node** queries);
void export_graph(Config* config, HNSW* hnsw, Node** nodes);

#endif