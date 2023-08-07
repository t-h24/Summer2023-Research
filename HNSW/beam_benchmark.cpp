#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include "hnsw.h"

using namespace std;

const bool LOAD_FROM_FILE = false;
const string LOAD_DIR = "exports/";
const string LOAD_NAME = "random_graph";

const bool PRINT_NEIGHBORS = false;
const bool PRINT_MISSING = false;

const bool EXPORT_RESULTS = false;
const string EXPORT_DIR = "exports/";
const string EXPORT_NAME = "random_graph";

void load_hnsw_graph(HNSW* hnsw, ifstream& graph_file, Node** nodes, int num_nodes, int num_layers) {
    // Load node levels
    for (int i = 0; i < num_nodes; ++i) {
        int level;
        graph_file.read(reinterpret_cast<char*>(&level), sizeof(level));
        nodes[i]->level = level;
    }

    // Load edges
    for (int i = 1; i < num_layers; ++i) {
        HNSWLayer* layer = new HNSWLayer();
        hnsw->layers.push_back(layer);
    }
    for (int i = 0; i < num_layers; ++i) {
        int num_entries;
        graph_file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));

        for (int j = 0; j < num_entries; ++j) {
            int node_index;
            graph_file.read(reinterpret_cast<char*>(&node_index), sizeof(node_index));

            int num_neighbors;
            graph_file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));

            vector<pair<float, Node*>>* neighbors = new vector<pair<float, Node*>>();
            neighbors->reserve(num_neighbors);
            for (int k = 0; k < num_neighbors; ++k) {
                int neighbor_index;
                float distance;
                graph_file.read(reinterpret_cast<char*>(&neighbor_index), sizeof(neighbor_index));
                graph_file.read(reinterpret_cast<char*>(&distance), sizeof(distance));
                neighbors->push_back(make_pair(distance, nodes[neighbor_index]));
            }

            hnsw->layers[i]->mappings[node_index] = neighbors;
        }
    }

    // Load entry point
    int entry_point;
    graph_file.read(reinterpret_cast<char*>(&entry_point), sizeof(entry_point));
    hnsw->entry_point = nodes[entry_point];
}

vector<vector<pair<float, Node*>>> return_queries(Config* config, HNSW* hnsw, Node** queries) {
    vector<vector<pair<float, Node*>>> results;
    vector<int>* paths = new vector<int>[config->num_queries];
    for (int i = 0; i < config->num_queries; ++i) {
        Node* query = queries[i];
        vector<pair<float, Node*>> found = nn_search(config, hnsw, query, config->num_return, config->ef_construction_search, paths[i]);
        results.push_back(found);
    }

    delete[] paths;
    return results;
}

/**
 * This class is used to run HNSW with different parameters, comparing the recall
 * versus ideal for each set of parameters.
*/
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

    cout << "Construction parameters: opt_con, max_con, max_con_0, ef_con" << endl;
    cout << "Search parameters: ef_con_s" << endl;

    // Initialize different config values
    const int SIZE = 3;
    int optimal_connections[SIZE] = {7, 14, 25};
    int max_connections[SIZE] = {11, 18, 30};
    int max_connections_0[SIZE] = {14, 28, 50};
    int ef_constructions[SIZE] = {21, 42, 75};

    const int SEARCH_SIZE = 2;
    int ef_construction_searches[SEARCH_SIZE] = {300, 500};

    vector<double> search_durations;
    vector<long long> search_dist_comps;
    search_durations.reserve(SIZE * SEARCH_SIZE);
    search_dist_comps.reserve(SIZE * SEARCH_SIZE); 

    // Run HNSW with different ef_construction values
    vector<vector<pair<float, Node*>>> neighbors[SIZE * SEARCH_SIZE];
    vector<vector<int>> actual_neighbors;
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
        HNSW* hnsw = NULL;

        if (LOAD_FROM_FILE) {
            // Get files to load from
            const string graph_file_name = LOAD_DIR + LOAD_NAME + "_graph_" + to_string(i) + ".bin";
            const string info_file_name = LOAD_DIR + LOAD_NAME + "_info_" + to_string(i) + ".txt";
            ifstream graph_file(graph_file_name);
            ifstream info_file(info_file_name);

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
            long long dist_comps;
            double construct_duration;
            info_file >> opt_con >> max_con >> max_con_0 >> ef_con;
            info_file >> num_nodes;
            info_file >> num_layers;
            info_file >> dist_comps;
            info_file >> construct_duration;

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

            // Load graph from file
            auto start = chrono::high_resolution_clock::now();

            cout << "Loading graph with construction parameters: "
                << config->optimal_connections << ", " << config->max_connections << ", "
                << config->max_connections_0 << ", " << config->ef_construction << endl;

            hnsw = init_hnsw(config, nodes);
            hnsw->layers[0]->mappings.clear();
            load_hnsw_graph(hnsw, graph_file, nodes, num_nodes, num_layers);

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            cout << "Load time: " << duration / 1000.0 << " seconds" << endl;
            cout << "Construction time: " << construct_duration << " seconds" << endl;
            cout << "Distance computations: " << dist_comps << endl;
        } else {
            // Insert nodes into HNSW
            auto start = chrono::high_resolution_clock::now();

            cout << "Inserting with construction parameters: "
                << config->optimal_connections << ", " << config->max_connections << ", "
                << config->max_connections_0 << ", " << config->ef_construction << endl; 
            hnsw = init_hnsw(config, nodes);
            insert_nodes(config, hnsw, nodes);

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            cout << "Time taken: " << duration / 1000.0 << " seconds" << endl;
            cout << "Distance computations: " << dist_comps << endl;
        }

        for (int j = 0; j < SEARCH_SIZE; ++j) {
            config->ef_construction_search = ef_construction_searches[j];
            auto start = chrono::high_resolution_clock::now();
            dist_comps = 0;

            // Run query search
            cout << "Searching with ef_con_s = " << ef_construction_searches[j] << endl;
            vector<vector<pair<float, Node*>>> results = return_queries(config, hnsw, queries);
            neighbors[i * SEARCH_SIZE + j] = results;

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            cout << "Time taken: " << duration / 1000.0 << " seconds" << endl;
            cout << "Distance computations: " << dist_comps << endl;

            search_durations.push_back(duration);
            search_dist_comps.push_back(dist_comps);
        }

        delete hnsw;
    }

    bool use_groundtruth = config->groundtruth_file != "";
    if (use_groundtruth && config->query_file == "") {
        cout << "Warning: Groundtruth file will not be used because queries were generated" << endl;
        use_groundtruth = false;
    }

    if (use_groundtruth) {
        // Load actual nearest neighbors
        load_ivecs(config->groundtruth_file, actual_neighbors, config->num_queries, config->num_return);

        if (PRINT_NEIGHBORS) {
            for (int i = 0; i < config->num_queries; ++i) {
                cout << "Neighbors in ideal case for query " << i << endl;
                for (size_t j = 0; j < actual_neighbors[i].size(); ++j) {
                    cout << actual_neighbors[i][j] << " (" << queries[i]->distance(nodes[actual_neighbors[i][j]]) << ") ";
                }
                cout << endl;
            }
        }
    } else {
        // Calcuate actual nearest neighbors per query
        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < config->num_queries; ++i) {
            Node* query = queries[i];
            actual_neighbors.push_back(vector<int>());
            priority_queue<pair<float, Node*>> pq;

            for (int j = 0; j < config->num_nodes; ++j) {
                pq.emplace(query->distance(nodes[j]), nodes[j]);
                if (pq.size() > config->num_return)
                    pq.pop();
            }

            // Place actual nearest neighbors
            actual_neighbors[i].reserve(config->num_return);
            actual_neighbors[i].resize(config->num_return);

            size_t idx = pq.size();
            while (idx > 0) {
                --idx;
                actual_neighbors[i][idx] = pq.top().second->index;
                pq.pop();
            }

            // Print out neighbors
            if (PRINT_NEIGHBORS) {
                cout << "Neighbors in ideal case for query " << i << endl;
                for (size_t j = 0; j < actual_neighbors[i].size(); ++j) {
                    cout << actual_neighbors[i][j] << " (" << queries[i]->distance(nodes[actual_neighbors[i][j]]) << ") ";
                }
                cout << endl;
            }
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Brute force time: " << duration / 1000.0 << " seconds" << endl;
    }

    ofstream* results_file = NULL;

    // Find differences between different ef_construction values and optimal
    for (int i = 0; i < SIZE * SEARCH_SIZE; ++i) {
        int opt_con = optimal_connections[i / SEARCH_SIZE];
        int max_con = max_connections[i / SEARCH_SIZE];
        int max_con_0 = max_connections_0[i / SEARCH_SIZE];
        int ef_con = ef_constructions[i / SEARCH_SIZE];
        int ef_con_s = ef_construction_searches[i % SEARCH_SIZE];

        cout << "Results for construction parameters: " << opt_con << ", " << max_con << ", "
            << max_con_0 << ", " << ef_con << " and search parameters: " << ef_con_s << endl;

        // Setup export file per set of parameters
        if (EXPORT_RESULTS && i % SEARCH_SIZE == 0) {
            if (results_file != NULL) {
                results_file->close();
                delete results_file;
            }

            results_file = new ofstream(EXPORT_DIR + EXPORT_NAME + "_results_"
                + to_string(config->num_queries) + "_" + to_string(config->num_return) + "_" + to_string(i / SEARCH_SIZE) + ".csv");
            *results_file << EXPORT_NAME << " of size " << config->num_nodes << " with construction parameters: " << opt_con << ", " << max_con << ", "
                << max_con_0 << ", " << ef_con << endl;
        }

        int similar = 0;
        for (int j = 0; j < config->num_queries; ++j) {
            // Find similar neighbors
            unordered_set<int> actual_set(actual_neighbors[j].begin(), actual_neighbors[j].end());
            unordered_set<int> intersection;

            for (size_t k = 0; k < neighbors[i][j].size(); ++k) {
                auto n_pair = neighbors[i][j][k];
                if (actual_set.find(n_pair.second->index) != actual_set.end()) {
                    intersection.insert(n_pair.second->index);
                }
            }
            similar += intersection.size();

            // Print out neighbors[i][j]
            if (PRINT_NEIGHBORS) {
                cout << "Neighbors for query " << j << ": ";
                for (size_t k = 0; k < neighbors[i][j].size(); ++k) {
                    auto n_pair = neighbors[i][j][k];
                    cout << n_pair.second->index << " (" << n_pair.first << ") ";
                }
                cout << endl;
            }

            // Print missing neighbors between intersection and actual_neighbors
            if (PRINT_MISSING) {
                cout << "Missing neighbors for query " << j << ": ";
                if (intersection.size() == actual_neighbors[j].size()) {
                    cout << "None" << endl;
                    continue;
                }
                for (size_t k = 0; k < actual_neighbors[j].size(); ++k) {
                    if (intersection.find(actual_neighbors[j][k]) == intersection.end()) {
                        cout << actual_neighbors[j][k] << " (" << queries[j]->distance(nodes[actual_neighbors[j][k]]) << ") ";
                    }
                }
                cout << endl;
            }
        }

        double recall = (double) similar / (config->num_queries * config->num_return);
        cout << "Correctly found neighbors: " << similar << " ("
            << recall * 100 << "%)" << endl;

        if (EXPORT_RESULTS) {
            *results_file << ef_con_s << ", " << search_dist_comps[i] / config->num_queries << ", "
                << recall << ", " << search_durations[i] / config->num_queries << endl;
        }
    }

    if (results_file != NULL) {
        results_file->close();
        delete results_file;
        cout << "Results exported to " << EXPORT_DIR << EXPORT_NAME << "_results_"
            << config->num_queries << "_" << config->num_return << "_*.csv" << endl;
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
