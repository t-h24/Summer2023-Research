#include <iostream>
#include <algorithm>
#include <chrono>
#include "hnsw.h"

using namespace std;

const string EXPORT_DIR = "exports/";
const string EXPORT_NAME = "random_graph";

/**
 * This class is used to export multiple HNSW graphs to file.
*/
int main() {
    time_t now = time(NULL);
    cout << "HNSW save run started at " << ctime(&now);

    Config* config = new Config();

    // Setup config
    config->export_graph = false;
    config->export_queries = false;

    // Get num_nodes amount of graph nodes
    Node** nodes = get_nodes(config);

    // Generate num_queries amount of queries
    Node** queries = get_queries(config, nodes);

    cout << "Construction parameters: opt_con, max_con, max_con_0, ef_con" << endl;

    // Initialize different config values
    const int SIZE = 3;
    int optimal_connections[SIZE] = {7, 14, 25};
    int max_connections[SIZE] = {11, 18, 30};
    int max_connections_0[SIZE] = {14, 28, 50};
    int ef_constructions[SIZE] = {21, 42, 75};

    // Run HNSW with different ef_construction values
    for (int i = 0; i < SIZE; ++i) {
        config->optimal_connections = optimal_connections[i];
        config->max_connections = max_connections[i];
        config->max_connections_0 = max_connections_0[i];
        config->ef_construction = ef_constructions[i];
        dist_comps = 0;

        // Sanity checks
        if(!sanity_checks(config)) {
            cout << "Config error!" << endl;
            return 1;
        }

        auto start = chrono::high_resolution_clock::now();

        // Insert nodes into HNSW
        cout << "Inserting with construction parameters: "
            << config->optimal_connections << ", " << config->max_connections << ", "
            << config->max_connections_0 << ", " << config->ef_construction << endl; 
        HNSW* hnsw = init_hnsw(config, nodes);
        insert_nodes(config, hnsw, nodes);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Time taken: " << duration / 1000.0 << " seconds" << endl;
        cout << "Distance computations: " << dist_comps << endl;

        // Export graph to file
        ofstream graph_file(EXPORT_DIR + EXPORT_NAME + "_graph_" + to_string(i) + ".bin");

        // Export levels
        for (size_t i = 0; i < config->num_nodes; ++i) {
            int index = nodes[i]->level;
            graph_file.write(reinterpret_cast<const char*>(&index), sizeof(index));
        }

        // Export edges
        for (int level = 0; level < hnsw->get_layers(); ++level) {
            HNSWLayer* layer = hnsw->layers[level];

            // Write entry size
            int entry_size = layer->mappings.size();
            graph_file.write(reinterpret_cast<const char*>(&entry_size), sizeof(entry_size));
            
            // Add all neighbor mappings, even empty ones to keep the graph structure 
            for (auto it = layer->mappings.begin(); it != layer->mappings.end(); ++it) {
                int node_index = it->first;
                graph_file.write(reinterpret_cast<const char*>(&node_index), sizeof(node_index));

                // Write neighbor size
                int n_size = it->second->size();
                graph_file.write(reinterpret_cast<const char*>(&n_size), sizeof(n_size));

                for (auto n_pair : *it->second) {
                    int neighbor_index = n_pair.second->index;
                    float distance = n_pair.first;
                    graph_file.write(reinterpret_cast<const char*>(&neighbor_index), sizeof(neighbor_index));
                    graph_file.write(reinterpret_cast<const char*>(&distance), sizeof(distance));
                }
            }
        }

        // Save entry point
        int entry_point = hnsw->entry_point->index;
        graph_file.write(reinterpret_cast<const char*>(&entry_point), sizeof(entry_point));
        graph_file.close();

        // Export construction parameters
        ofstream info_file(EXPORT_DIR + EXPORT_NAME + "_info_" + to_string(i) + ".txt");
        info_file << config->optimal_connections << " " << config->max_connections << " "
            << config->max_connections_0 << " " << config->ef_construction << endl;
        info_file << config->num_nodes << endl;
        info_file << hnsw->get_layers() << endl;
        info_file << dist_comps << endl;
        info_file << duration / 1000.0 << endl;

        cout << "Exported graph to file" << endl;

        /*
        // DEBUG START
        // Print each node's level
        for (int i = 0; i < config->num_nodes; ++i) {
            cout << nodes[i]->level << " ";
        }
        cout << "Debug levels done" << endl;

        // Print edges
        for (int level = 0; level < hnsw->get_layers(); ++level) {
            HNSWLayer* layer = hnsw->layers[level];
            for (auto it = layer->mappings.begin(); it != layer->mappings.end(); ++it) {
                if (it->second->empty())
                    continue;
                int node_index = it->first;
                cout << node_index << " ";

                int n_size = it->second->size();
                cout << n_size << " ";

                for (auto n_pair : *it->second) {
                    int neighbor_index = n_pair.second->index;
                    float distance = n_pair.first;
                    cout << neighbor_index << " " << distance << " ";
                }
                cout << endl;
            }
        }
        cout << "Debug edges done" << endl;
        // Print entry point
        cout << "Entry point: " << hnsw->entry_point->index << endl;
        // DEBUG END
        */

        delete hnsw;
    }

    now = time(NULL);
    cout << "HNSW save run ended at " << ctime(&now);

    return 0;
}
