#include <iostream>
#include <algorithm>
#include "hnsw.h"

using namespace std;

const bool PRINT_NEIGHBORS = false;

vector<vector<Node*>> return_queries(Config* config, HNSW* hnsw, Node** queries) {
    vector<vector<Node*>> results;
    vector<int>* paths = new vector<int>[config->num_queries];
    for (int i = 0; i < config->num_queries; ++i) {
        Node* query = queries[i];
        vector<Node*> found = nn_search(config, hnsw, query, config->num_return, config->ef_construction, paths[i]);
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
    config->num_queries = 100;
    config->num_return = 50;

    // Get num_nodes amount of graph nodes
    Node** nodes = get_nodes(config);

    // Generate num_queries amount of queries
    Node** queries = get_queries(config, nodes);

    // Initialize different config values
    const int SIZE = 5;
	int optimal_connections[SIZE] = {7, 10, 14, 21, 25};
    int max_connections[SIZE] = {14, 20, 28, 42, 50};
    int ef_constructions[SIZE] = {21, 30, 42, 63, 75};
    int ef_construction_searches[SIZE] = {150, 150, 150, 150, 150};

    // Run HNSW with different ef_construction values
    vector<vector<Node*>> neighbors[SIZE + 1];
    for (int i = 0; i < SIZE; ++i) {
        config->optimal_connections = optimal_connections[i];
        config->max_connections = max_connections[i];
        config->ef_construction = ef_constructions[i];
        config->ef_construction_search = ef_construction_searches[i];

        // Sanity checks
        if(!sanity_checks(config)) {
            cout << "Config error!" << endl;
            return 1;
        }

        now = time(0);

        // Insert nodes into HNSW
        cout << "Inserting with ef_construction = " << ef_constructions[i] << endl;
        HNSW* hnsw = init_hnsw(config, nodes);
        insert_nodes(config, hnsw, nodes);

        // Run query search for EF_CONSTRUCTION changes
        cout << "Searching with ef_construction = " << ef_constructions[i] << endl;
        vector<vector<Node*>> results = return_queries(config, hnsw, queries);
        neighbors[i] = results;

        time_t after = time(0);
        cout << "Time elapsed: " << difftime(after, now) << " seconds" << endl;

        delete hnsw;
    }

    // Calcuate actual neighbors per query
    for (int i = 0; i < config->num_queries; ++i) {
        Node* query = queries[i];
        neighbors[SIZE].push_back(vector<Node*>());

        auto close_dist_comp = [query](Node* a, Node* b) {	
            return query->distance(a) < query->distance(b);	
        };
        priority_queue<Node*, vector<Node*>, decltype(close_dist_comp)> pq(close_dist_comp);

        for (int j = 0; j < config->num_nodes; ++j) {
            pq.push(nodes[j]);

            if (pq.size() > config->num_return)
                pq.pop();
        }

        neighbors[SIZE][i].reserve(config->num_return);
        neighbors[SIZE][i].resize(config->num_return);
        size_t idx = pq.size();
        while (idx > 0) {
            --idx;
            neighbors[SIZE][i][idx] = pq.top();
            pq.pop();
        }

        // Print out neighbors[SIZE][i]
        if (PRINT_NEIGHBORS) {
            cout << "Neighbors in ideal case for query " << i << endl;
            for (size_t j = 0; j < neighbors[SIZE][i].size(); ++j) {
                Node* neighbor = neighbors[SIZE][i][j];
                cout << neighbor->index << " (" << queries[i]->distance(neighbor) << ") ";
            }
            cout << endl;
        }
    }

    // Find differences between different ef_construction values and optimal
    for (int i = 0; i < SIZE; ++i) {
        int similar = 0;
        for (int j = 0; j < config->num_queries; ++j) {
            vector<Node*> intersection;
            set_intersection(neighbors[i][j].begin(), neighbors[i][j].end(),
                neighbors[SIZE][j].begin(), neighbors[SIZE][j].end(), back_inserter(intersection));
            similar += intersection.size();

            // Print out neighbors[i][j]
            if (PRINT_NEIGHBORS) {
                cout << "Neighbors for query " << j << " with ef_construction = " << ef_constructions[i] << endl;
                for (size_t k = 0; k < neighbors[i][j].size(); ++k) {
                    Node* neighbor = neighbors[i][j][k];
                    cout << neighbor->index << " (" << queries[j]->distance(neighbor) << ") ";
                }
                cout << endl;
            }
        }

        cout << "Similarities between ef_construction = " << ef_constructions[i] << " and optimal: " << similar << endl;
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
