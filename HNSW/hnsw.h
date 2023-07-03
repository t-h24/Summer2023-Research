#include <vector>
#include <map>
#include <fstream>
#include <queue>
#include <random>
#include <functional>

#ifndef HNSW_H
#define HNSW_H

class Config {
public:
    std::string load_file = "";

    int graph_seed = 0;
    int query_seed = 100000;
    int level_seed = 1000000;

    int gen_min = 0;
    int gen_max = 100000;
    int gen_decimals = 2;

    int dimensions = 128;
    int num_nodes = 10000;
    int optimal_connections = 20;
    int max_connections = 40;
    int ef_construction = 60;
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

Node** get_nodes(Config* config);

Node** get_queries(Config* config, Node** graph_nodes);

// Main algorithms
HNSW* insert(Config* config, HNSW* hnsw, Node* query, int est_con, int max_con, int ef_con, float normal_factor, std::function<double()> rand);
std::deque<Node*> search_layer(Config* config, HNSW* hnsw, Node* query, std::deque<Node*> entry_points, int num_to_return, int layer_num);
std::deque<Node*> select_neighbors_simple(Config* config, HNSW* hnsw, Node* query, std::deque<Node*> candidates, int num, bool drop);
std::deque<Node*> nn_search(Config* config, HNSW* hnsw, Node* query, int num_to_return, int ef_con, std::vector<int>& path);

// Executing HNSW
bool sanity_checks(Config* config);
HNSW* init_hnsw(Config* config, Node** nodes);
void insert_nodes(Config* config, HNSW* hnsw, Node** nodes);
void print_hnsw(Config* config, HNSW* hnsw);
void run_query_search(Config* config, HNSW* hnsw, Node** queries);
void export_graph(Config* config, HNSW* hnsw, Node** nodes);

#endif