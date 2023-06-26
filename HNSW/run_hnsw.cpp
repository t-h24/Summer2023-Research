#include <iostream>
#include "hnsw.h"

using namespace std;

int main() {
    Config* config = new Config();

    // Sanity checks
    if(!sanity_checks(config))
        return 1;

    // Generate num_nodes amount of nodes
    Node** nodes = generate_nodes(config->dimensions, config->num_nodes, config->generation_seed);
    cout << "Beginning HNSW construction" << endl;

    // Insert nodes into HNSW
    HNSW* hnsw = init_hnsw(config, nodes);
    insert_nodes(config, hnsw, nodes);

    // Print HNSW graph
    print_hnsw(config, hnsw);
    
    // Generate num_queries amount of nodes
    Node** queries = generate_nodes(config->dimensions, config->num_queries, config->graph_seed);
    cout << "Beginning search" << endl;

    if (config->debug_query_search_index >= 0) {
        ofstream* debug_file = new ofstream("runs/query_search.txt");
        queries[config->debug_query_search_index]->debug_file = debug_file;
    }

    // Run query search and print results
    run_query_search(config, hnsw, queries);

    if (config->debug_query_search_index >= 0) {
       queries[config->debug_query_search_index]->debug_file->close();
       delete queries[config->debug_query_search_index]->debug_file;
    }

    // Export graph to file
    export_graph(config, hnsw, nodes);

    // Delete nodes
    for (int i = 0; i < config->num_nodes; i++) {
        delete nodes[i];
    }
    delete[] nodes;

    // Delete queries
    for (int i = 0; i < config->num_queries; ++i)
        delete queries[i];
    delete[] queries;

    // Delete hnsw and config
    delete hnsw;
    delete config;

    return 0;
}
