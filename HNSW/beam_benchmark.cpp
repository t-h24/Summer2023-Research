#include <iostream>
#include <algorithm>
#include "hnsw.h"

using namespace std;

const bool PRINT_NEIGHBORS = false;

vector<deque<Node*>> return_queries(Config* config, HNSW* hnsw, Node** queries) {
    vector<deque<Node*>> results;
    vector<int>* paths = new vector<int>[config->num_queries];
    for (int i = 0; i < config->num_queries; ++i) {
        Node* query = queries[i];
        deque<Node*> found = nn_search(config, hnsw, query, config->num_return, config->ef_construction, paths[i]);
        results.push_back(found);
    }

    delete[] paths;
    return results;
}

int main() {
    Config* config = new Config();

    // Setup config
    config->num_nodes = 5000;
    config->num_queries = 10;
    config->num_return = 50;
    config->optimal_connections = 50;
    config->max_connections = 100;
    config->ef_construction = 100;

    // Generate NUM_NODES amount of nodes
    Node** nodes = generate_nodes(config->dimensions, config->num_nodes, config->generation_seed);
    cout << "Beginning HNSW construction" << endl;

    // Generate NUM_QUERIES amount of nodes
    Node** queries = generate_nodes(config->dimensions, config->num_queries, config->graph_seed);

    // Generate ef_construction list from config->ef_construction to config->num_nodes
    int EF_CON_SIZE = 1;
    int ef_construction = config->ef_construction;
    while (ef_construction < config->num_nodes) {
        ef_construction *= 2;
        ++EF_CON_SIZE;
    }
    int ef_constructions[EF_CON_SIZE];
    ef_constructions[0] = config->ef_construction;
    for (int i = 1; i < EF_CON_SIZE; ++i) {
        ef_constructions[i] = ef_constructions[i - 1] * 2;
    }
    ef_constructions[EF_CON_SIZE - 1] = config->num_nodes;

    // Run HNSW with different ef_construction values
    vector<deque<Node*>> neighbors[EF_CON_SIZE];
    for (int i = 0; i < EF_CON_SIZE; ++i) {
        config->ef_construction = ef_constructions[i];

        // Sanity checks
        if(!sanity_checks(config)) {
            cout << "Config error!" << endl;
            return 1;
        }

        // Insert nodes into HNSW
        cout << "Inserting with ef_construction = " << ef_constructions[i] << endl;
        HNSW* hnsw = init_hnsw(config, nodes);
        insert_nodes(config, hnsw, nodes);

        // Run query search for EF_CONSTRUCTION changes
        cout << "Searching with ef_construction = " << ef_constructions[i] << endl;
        vector<deque<Node*>> results = return_queries(config, hnsw, queries);
        neighbors[i] = results;

        delete hnsw;
    }

    // Find differences between different ef_construction values and optimal (max ef_construction)
    for (int i = 0; i < EF_CON_SIZE - 1; ++i) {
        int differences = 0;
        for (int j = 0; j < config->num_queries; ++j) {
            deque<Node*> intersection;
            set_intersection(neighbors[i][j].begin(), neighbors[i][j].end(),
                neighbors[EF_CON_SIZE - 1][j].begin(), neighbors[EF_CON_SIZE - 1][j].end(), back_inserter(intersection));
            differences += neighbors[i][j].size() - intersection.size();

            // Print out neighbors[i][j]
            if (PRINT_NEIGHBORS) {
                cout << "Neighbors for query " << j << " with ef_construction = " << ef_constructions[i] << endl;
                for (size_t k = 0; k < neighbors[i][j].size(); ++k) {
                    Node* neighbor = neighbors[i][j][k];
                    cout << neighbor->index << " ";
                }
                cout << endl;
            }
        }

        cout << "Differences between ef_construction = " << ef_constructions[i] << " and optimal: " << differences << endl;
    }

    // Delete nodes
    for (int i = 0; i < config->num_nodes; i++) {
        delete nodes[i];
    }
    delete[] nodes;

    // Delete queries
    for (int i = 0; i < config->num_queries; ++i)
        delete queries[i];
    delete[] queries;

    // Delete config
    delete config;
}