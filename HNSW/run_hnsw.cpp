#include <iostream>
#include "hnsw.h"

using namespace std;

/** 
 * This class is used to run a single instance of the HNSW algorithm.
*/
int main() {
    time_t now = time(NULL);
    cout << "HNSW run started at " << ctime(&now);

    Config* config = new Config();

    // Sanity checks
    if(!sanity_checks(config))
        return 1;

    // Get num_nodes amount of graph nodes
    float** nodes = new float*[config->num_nodes];
    load_nodes(config, nodes);
    cout << "Beginning HNSW construction" << endl;

    // Insert nodes into HNSW
    HNSW* hnsw = init_hnsw(config, nodes);
    insert_nodes(config, hnsw);

    // Print HNSW graph
    print_hnsw(config, hnsw);
    
    // Generate num_queries amount of queries
    float** queries = new float*[config->num_queries];
    load_queries(config, nodes, queries);
    cout << "Beginning search" << endl;

    // Run query search and print results
    run_query_search(config, hnsw, queries);

    // Export graph to file
    export_graph(config, hnsw, nodes);

    // Delete nodes
    for (int i = 0; i < config->num_nodes; i++)
        delete nodes[i];
    delete[] nodes;

    // Delete queries
    for (int i = 0; i < config->num_queries; ++i)
        delete queries[i];
    delete[] queries;

    // Delete hnsw and config
    delete hnsw;
    delete config;

    now = time(NULL);
    cout << "HNSW run ended at " << ctime(&now);

    return 0;
}
