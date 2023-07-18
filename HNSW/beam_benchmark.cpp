#include <iostream>
#include <algorithm>
#include <chrono>
#include "hnsw.h"

using namespace std;

const bool PRINT_NEIGHBORS = false;
const bool PRINT_MISSING = false;

vector<vector<Node*>> return_queries(Config* config, HNSW* hnsw, Node** queries) {
    vector<vector<Node*>> results;
    vector<int>* paths = new vector<int>[config->num_queries];
    for (int i = 0; i < config->num_queries; ++i) {
        Node* query = queries[i];
        vector<Node*> found = nn_search(config, hnsw, query, config->num_return, config->ef_construction_search, paths[i]);
        results.push_back(found);
    }

    delete[] paths;
    return results;
}

int main() {
    time_t now = time(0);
    cout << "Benchmark run started at " << ctime(&now);

    Config* config = new Config();

    // Setup config
    config->export_graph = false;
    config->export_queries = false;
    config->num_queries = 100;
    config->num_return = 20;

    // Get num_nodes amount of graph nodes
    Node** nodes = get_nodes(config);

    // Generate num_queries amount of queries
    Node** queries = get_queries(config, nodes);

    // Initialize different config values
    const int SIZE = 3;
    int optimal_connections[SIZE] = {7, 14, 25};
    int max_connections[SIZE] = {11, 18, 30};
    int max_connections_0[SIZE] = {14, 28, 50};
    int ef_constructions[SIZE] = {21, 42, 75};

    const int SEARCH_SIZE = 2;
    int ef_construction_searches[SEARCH_SIZE] = {300, 500};

    // Run HNSW with different ef_construction values
    vector<vector<Node*>> neighbors[SIZE * SEARCH_SIZE + 1];
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
        cout << "Inserting with ef_construction = " << ef_constructions[i] << endl;
        HNSW* hnsw = init_hnsw(config, nodes);
        insert_nodes(config, hnsw, nodes);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Time taken: " << duration / 1000.0 << " seconds" << endl;
        cout << "Distance computations: " << dist_comps << endl;

        for (int j = 0; j < SEARCH_SIZE; ++j) {
            config->ef_construction_search = ef_construction_searches[j];
            start = chrono::high_resolution_clock::now();
            dist_comps = 0;

            // Run query search
            cout << "Searching with ef_construction = " << ef_construction_searches[j] << endl;
            vector<vector<Node*>> results = return_queries(config, hnsw, queries);
            neighbors[i * SEARCH_SIZE + j] = results;

            end = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            cout << "Time taken: " << duration / 1000.0 << " seconds" << endl;
            cout << "Distance computations: " << dist_comps << endl;
        }

        delete hnsw;
    }

    // Calcuate actual neighbors per query
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < config->num_queries; ++i) {
        Node* query = queries[i];
        neighbors[SIZE * SEARCH_SIZE].push_back(vector<Node*>());

        auto close_dist_comp = [query](Node* a, Node* b) {	
            return query->distance(a) < query->distance(b);	
        };
        priority_queue<Node*, vector<Node*>, decltype(close_dist_comp)> pq(close_dist_comp);

        for (int j = 0; j < config->num_nodes; ++j) {
            pq.push(nodes[j]);

            if (pq.size() > config->num_return)
                pq.pop();
        }

        neighbors[SIZE * SEARCH_SIZE][i].reserve(config->num_return);
        neighbors[SIZE * SEARCH_SIZE][i].resize(config->num_return);
        size_t idx = pq.size();
        while (idx > 0) {
            --idx;
            neighbors[SIZE * SEARCH_SIZE][i][idx] = pq.top();
            pq.pop();
        }

        // Print out neighbors[SIZE][i]
        if (PRINT_NEIGHBORS) {
            cout << "Neighbors in ideal case for query " << i << endl;
            for (size_t j = 0; j < neighbors[SIZE * SEARCH_SIZE][i].size(); ++j) {
                Node* neighbor = neighbors[SIZE * SEARCH_SIZE][i][j];
                cout << neighbor->index << " (" << queries[i]->distance(neighbor) << ") ";
            }
            cout << endl;
        }
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Brute force time: " << duration / 1000.0 << " seconds" << endl;

    // Find differences between different ef_construction values and optimal
    for (int i = 0; i < SIZE * SEARCH_SIZE; ++i) {
        int ef_con = ef_constructions[i / SEARCH_SIZE];
        int ef_con_s = ef_construction_searches[i % SEARCH_SIZE];

        int similar = 0;
        for (int j = 0; j < config->num_queries; ++j) {
            Node* query = queries[j];
            auto comp = [query](Node* a, Node* b) {
                return query->distance(a) < query->distance(b);
            };

            vector<Node*> intersection;
            set_intersection(neighbors[i][j].begin(), neighbors[i][j].end(),
                neighbors[SIZE * SEARCH_SIZE][j].begin(), neighbors[SIZE * SEARCH_SIZE][j].end(), back_inserter(intersection), comp);
            similar += intersection.size();

            // Print out neighbors[i][j]
            if (PRINT_NEIGHBORS) {
                cout << "Neighbors for query " << j << " with ef_con = " << ef_con << ", ef_con_s = " << ef_con_s << endl;
                for (size_t k = 0; k < neighbors[i][j].size(); ++k) {
                    Node* neighbor = neighbors[i][j][k];
                    cout << neighbor->index << " (" << queries[j]->distance(neighbor) << ") ";
                }
                cout << endl;
            }

            // Print missing neighbors between intersection and neighbors[SIZE][j]
            if (PRINT_MISSING) {
                cout << "Missing neighbors for query " << j << " with ef_con = " << ef_con << ", ef_con_s = " << ef_con_s << endl;
                size_t idx = 0;
                for (size_t k = 0; k < neighbors[SIZE * SEARCH_SIZE][j].size(); ++k) {
                    Node* neighbor = neighbors[SIZE * SEARCH_SIZE][j][k];
                    if (idx < intersection.size() && neighbor->index == intersection[idx]->index) {
                        ++idx;
                    } else {
                        cout << neighbor->index << " (" << queries[j]->distance(neighbor) << ") ";
                    }
                }
                cout << endl;
            }
        }

        cout << "Similarities between ef_con = " << ef_con << ", ef_con_s = " << ef_con_s << " and optimal: " << similar 
            << " (" << (double) similar / (config->num_queries * config->num_return) * 100 << "%)" << endl;
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

    now = time(0);
    cout << "Benchmark run ended at " << ctime(&now);
}
