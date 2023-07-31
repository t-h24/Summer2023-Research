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
    Node** nodes = get_nodes(config);
    cout << "Beginning HNSW construction" << endl;

    // Insert nodes into HNSW
    HNSW* hnsw = init_hnsw(config, nodes);
    insert_nodes(config, hnsw, nodes);

    // Print HNSW graph
    print_hnsw(config, hnsw);
    
    // Generate num_queries amount of queries
    Node** queries = get_queries(config, nodes);
    cout << "Beginning search" << endl;

    if (config->debug_query_search_index >= 0) {
        ofstream* debug_file = new ofstream(config->export_dir + "query_search.txt");
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

    now = time(NULL);
    cout << "HNSW run ended at " << ctime(&now);

    return 0;
}
