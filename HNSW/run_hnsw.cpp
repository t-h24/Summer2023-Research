#include <iostream>
#include "hnsw.h"

using namespace std;

const bool LOAD_FROM_FILE = false;
const string LOAD_DIR = "exports/";
const string LOAD_NAME = "random_graph";
const int LOAD_INDEX = 0;

void load_hnsw_graph(HNSW* hnsw, ifstream& graph_file, float** nodes, int num_nodes, int num_layers) {
    // Load node neighbors
    for (int i = 0; i < num_nodes; ++i) {
        int levels;
        graph_file.read(reinterpret_cast<char*>(&levels), sizeof(levels));
        hnsw->mappings[i].resize(levels);

        // Load level
        for (int j = 0; j < levels; ++j) {
            int num_neighbors;
            graph_file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
            hnsw->mappings[i][j].reserve(num_neighbors);

            // Load neighbors
            for (int k = 0; k < num_neighbors; ++k) {
                int index;
                float distance;
                graph_file.read(reinterpret_cast<char*>(&index), sizeof(index));
                graph_file.read(reinterpret_cast<char*>(&distance), sizeof(distance));
                hnsw->mappings[i][j].emplace_back(distance, index);
            }
        }
    }

    // Load entry point
    int entry_point;
    graph_file.read(reinterpret_cast<char*>(&entry_point), sizeof(entry_point));
    hnsw->entry_point = entry_point;
}

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

    HNSW* hnsw = init_hnsw(config, nodes);
    if (LOAD_FROM_FILE) {
        // Check file and parameters
        const string graph_file_name = LOAD_DIR + LOAD_NAME + "_graph_" + to_string(LOAD_INDEX) + ".bin";
        const string info_file_name = LOAD_DIR + LOAD_NAME + "_info_" + to_string(LOAD_INDEX) + ".txt";
        ifstream graph_file(graph_file_name);
        ifstream info_file(info_file_name);
        cout << "Loading saved graph from " << graph_file_name << endl;

        if (!graph_file) {
            cout << "File " << graph_file_name << " not found!" << endl;
            return 1;
        }
        if (!info_file) {
            cout << "File " << info_file_name << " not found!" << endl;
            return 1;
        }

        int opt_con, max_con, max_con_0, ef_con;
        int num_nodes;
        int num_layers;
        info_file >> opt_con >> max_con >> max_con_0 >> ef_con;
        info_file >> num_nodes;
        info_file >> num_layers;

        // Check if number of nodes match
        if (num_nodes != config->num_nodes) {
            cout << "Mismatch between loaded and expected number of nodes" << endl;
            return 1;
        }

        // Check if construction parameters match
        if (opt_con != config->optimal_connections || max_con != config->max_connections ||
            max_con_0 != config->max_connections_0 || ef_con != config->ef_construction) {
            cout << "Mismatch between loaded and expected construction parameters" << endl;
            return 1;
        }

        hnsw->layers = num_layers;
        load_hnsw_graph(hnsw, graph_file, nodes, num_nodes, num_layers);
    } else {
        // Insert nodes into HNSW
        insert_nodes(config, hnsw);
    }

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
