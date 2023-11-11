#include <iostream>
#include <math.h>
#include <algorithm>
#include <set>
#include <immintrin.h>
#include <unordered_set>
#include "hnsw.h"

using namespace std;

long long int dist_comps = 0;
ofstream* debug_file = NULL;

int correct_nn_found = 0;
bool log_neighbors = false;
vector<int> cur_groundtruth;
ofstream* when_neigh_found_file;

// 0 = building, 1 = searching
int program_state = 0;

HNSW::HNSW(int node_size, float** nodes) : node_size(node_size), nodes(nodes), layers(0) {}

float calculate_l2_sq(float* a, float* b, int size) {
    ++dist_comps;

    int parts = size / 8;

    // Initialize result to 0
    __m256 result = _mm256_setzero_ps();

    // Process 8 floats at a time
    for (size_t i = 0; i < parts; ++i) {
        // Load vectors from memory into AVX registers
        __m256 vec_a = _mm256_loadu_ps(&a[i * 8]);
        __m256 vec_b = _mm256_loadu_ps(&b[i * 8]);

        // Compute differences and square
        __m256 diff = _mm256_sub_ps(vec_a, vec_b);
        __m256 diff_sq = _mm256_mul_ps(diff, diff);

        result = _mm256_add_ps(result, diff_sq);
    }

    // Process remaining floats
    float remainder = 0;
    for (size_t i = parts * 8; i < size; ++i) {
        float diff = a[i] - b[i];
        remainder += diff * diff;
    }

    // Sum all floats in result
    float sum[8];
    _mm256_storeu_ps(sum, result);
    for (size_t i = 1; i < 8; ++i) {
        sum[0] += sum[i];
    }

    return sum[0] + remainder;
}

void load_fvecs(const string& file, const string& type, float** nodes, int num, int dim, bool has_groundtruth) {
    ifstream f(file, ios::binary | ios::in);
    if (!f) {
        cout << "File " << file << " not found!" << endl;
        exit(-1);
    }
    cout << "Loading " << num << " " << type << " from file " << file << endl;

    // Read dimension
    int read_dim;
    f.read(reinterpret_cast<char*>(&read_dim), 4);
    if (dim != read_dim) {
        cout << "Mismatch between expected and actual dimension: " << dim << " != " << read_dim << endl;
        exit(-1);
    }

    // Check size
    f.seekg(0, ios::end);
    if (num > f.tellg() / (dim * 4 + 4)) {
        cout << "Requested number of " << type << " is greater than number in file: "
            << num << " > " << f.tellg() / (dim * 4 + 4) << endl;
        exit(-1);
    }
    if (type == "nodes" && num != f.tellg() / (dim * 4 + 4) && has_groundtruth) {
        cout << "You must load all " << f.tellg() / (dim * 4 + 4) << " nodes if you want to use a groundtruth file" << endl;
        exit(-1);
    }

    f.seekg(0, ios::beg);
    for (int i = 0; i < num; i++) {
        // Skip dimension size
        f.seekg(4, ios::cur);

        // Read point
        nodes[i] = new float[dim];
        f.read(reinterpret_cast<char*>(nodes[i]), dim * 4);
    }
    f.close();
}

void load_ivecs(const string& file, vector<vector<int>>& results, int num, int num_return) {
    ifstream f(file, ios::binary | ios::in);
    if (!f) {
        cout << "File " << file << " not found!" << endl;
        exit(-1);
    }
    cout << "Loading groundtruth from file " << file << endl;

    // Read width
    int width;
    f.read(reinterpret_cast<char*>(&width), 4);
    if (num_return > width) {
        cout << "Requested num_return is greater than width in file: " << num_return << " > " << width << endl;
        exit(-1);
    }

    // Check size
    f.seekg(0, ios::end);
    if (num > f.tellg() / (width * 4 + 4)) {
        cout << "Requested number of queries is greater than number in file: "
            << num << " > " << f.tellg() / (width * 4 + 4) << endl;
        exit(-1);
    }

    results.reserve(num);
    f.seekg(0, ios::beg);
    for (int i = 0; i < num; i++) {
        // Skip list size
        f.seekg(4, ios::cur);

        // Read point
        int values[num_return];
        f.read(reinterpret_cast<char*>(values), num_return * 4);
        results.push_back(vector<int>(values, values + num_return));

        // Skip remaining values
        f.seekg((width - num_return) * 4, ios::cur);
    }
    f.close();
}

void load_nodes(Config* config, float** nodes) {
    if (config->load_file != "") {
        if (config->load_file.size() >= 6 && config->load_file.substr(config->load_file.size() - 6) == ".fvecs") {
            // Load nodes from fvecs file
            load_fvecs(config->load_file, "nodes", nodes, config->num_nodes, config->dimensions, config->groundtruth_file != "");
            return;
        }
    
        // Load nodes from file
        ifstream f(config->load_file, ios::in);
        if (!f) {
            cout << "File " << config->load_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << config->num_nodes << " nodes from file " << config->load_file << endl;

        for (int i = 0; i < config->num_nodes; i++) {
            nodes[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> nodes[i][j];
            }
        }

        f.close();
        return;
    }

    cout << "Generating " << config->num_nodes << " random nodes" << endl;

    mt19937 gen(config->graph_seed);
    uniform_real_distribution<float> dis(config->gen_min, config->gen_max);

    for (int i = 0; i < config->num_nodes; i++) {
        nodes[i] = new float[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            nodes[i][j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
    }
}

void load_queries(Config* config, float** nodes, float** queries) {
    mt19937 gen(config->query_seed);
    if (config->query_file != "") {
        if (config->query_file.size() >= 6 && config->query_file.substr(config->query_file.size() - 6) == ".fvecs") {
            // Load queries from fvecs file
            load_fvecs(config->query_file, "queries", queries, config->num_queries, config->dimensions, config->groundtruth_file != "");
            return;
        }

        // Load queries from file
        ifstream f(config->query_file, ios::in);
        if (!f) {
            cout << "File " << config->query_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << config->num_queries << " queries from file " << config->query_file << endl;

        for (int i = 0; i < config->num_queries; i++) {
            queries[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> queries[i][j];
            }
        }

        f.close();
        return;
    }

    if (config->load_file == "") {
        // Generate random queries (same as get_nodes)
        cout << "Generating " << config->num_queries << " random queries" << endl;
        uniform_real_distribution<float> dis(config->gen_min, config->gen_max);

        for (int i = 0; i < config->num_queries; i++) {
            queries[i] = new float[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                queries[i][j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
            }
        }

        return;
    }
    
    // Generate queries randomly based on bounds of graph_nodes
    cout << "Generating queries based on file " << config->load_file << endl;
    float* lower_bound = new float[config->dimensions];
    float* upper_bound = new float[config->dimensions];
    copy(nodes[0], nodes[0] + config->dimensions, lower_bound);
    copy(nodes[0], nodes[0] + config->dimensions, upper_bound);

    // Calculate lowest and highest value for each dimension using graph_nodes
    for (int i = 1; i < config->num_nodes; i++) {
        for (int j = 0; j < config->dimensions; j++) {
            if (nodes[i][j] < lower_bound[j]) {
                lower_bound[j] = nodes[i][j];
            }
            if (nodes[i][j] > upper_bound[j]) {
                upper_bound[j] = nodes[i][j];
            }
        }
    }
    uniform_real_distribution<float>* dis_array = new uniform_real_distribution<float>[config->dimensions];
    for (int i = 0; i < config->dimensions; i++) {
        dis_array[i] = uniform_real_distribution<float>(lower_bound[i], upper_bound[i]);
    }

    // Generate queries based on the range of values in each dimension
    for (int i = 0; i < config->num_queries; i++) {
        queries[i] = new float[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            queries[i][j] = round(dis_array[j](gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
    }

    delete[] lower_bound;
    delete[] upper_bound;
    delete[] dis_array;
}

/**
 * Alg 1
 * INSERT(hnsw, q, M, Mmax, efConstruction, mL)
 * Extra arguments: rand (for generating random value between 0 and 1)
 * Note: max_con is not used for level 0, instead max_connections_0 is used
*/
HNSW* insert(Config* config, HNSW* hnsw, int query, int opt_con, int max_con, int ef_con, float normal_factor, function<double()> rand) {
    vector<pair<float, int>> entry_points;
    entry_points.reserve(ef_con);
    float dist = calculate_l2_sq(hnsw->nodes[query], hnsw->nodes[hnsw->entry_point], config->dimensions);
    entry_points.push_back(make_pair(dist, hnsw->entry_point));
    int top = hnsw->layers - 1;
    
    // Get node level
    int node_level = -log(rand()) * normal_factor;
    hnsw->mappings[query].resize(node_level + 1);

    // Update layer count
    if (node_level > top) {
        hnsw->layers = node_level + 1;
        if (config->debug_insert)
            cout << "Layer count increased to " << hnsw->layers << endl;
    }

    if (config->debug_insert)
        cout << "Inserting node " << query << " at level " << node_level << " with entry point " << entry_points[0].second << endl;

    // Get closest element by using search_layer to find the closest point at each level
    for (int level = top; level >= node_level + 1; level--) {
        search_layer(config, hnsw, hnsw->nodes[query], entry_points, 1, level);

        if (config->debug_insert)
            cout << "Closest point at level " << level << " is " << entry_points[0].second << " (" << entry_points[0].first << ")" << endl;
    }

    for (int level = min(top, node_level); level >= 0; level--) {
        if (level == 0)
            max_con = config->max_connections_0;

        // Get nearest elements
        search_layer(config, hnsw, hnsw->nodes[query], entry_points, ef_con, level);

        // Initialize mapping vector
        vector<pair<float, int>>& neighbors = hnsw->mappings[query][level];
        neighbors.reserve(max_con + 1);
        neighbors.resize(min(opt_con, (int)entry_points.size()));

        //Select opt_con number of neighbors from entry_points
        copy_n(entry_points.begin(), min(opt_con, (int)entry_points.size()), neighbors.begin());

        if (config->debug_insert) {
            cout << "Neighbors at level " << level << " are ";
            for (auto n_pair : neighbors)
                cout << n_pair.second << " (" << n_pair.first << ") ";
            cout << endl;
        }

        //Connect neighbors to this node
        for (auto n_pair : neighbors) {
            vector<pair<float, int>>& neighbor_mapping = hnsw->mappings[n_pair.second][level];

            // Place query in correct position in neighbor_mapping
            float new_dist = calculate_l2_sq(hnsw->nodes[query], hnsw->nodes[n_pair.second], config->dimensions);
            auto new_pair = make_pair(new_dist, query);
            auto pos = lower_bound(neighbor_mapping.begin(), neighbor_mapping.end(), new_pair);
            neighbor_mapping.insert(pos, new_pair);
        }

        // Trim neighbor connections if needed
        for (auto n_pair : neighbors) {
            vector<pair<float, int>>& neighbor_mapping = hnsw->mappings[n_pair.second][level];
            if (neighbor_mapping.size() > max_con) {
                // Pop last element (size will be max_con after this)
                neighbor_mapping.pop_back();
            }
        }

        if (config->single_entry_point)
            // Resize entry_points to 1
            entry_points.resize(1);
    }

    if (node_level > top) {
        hnsw->entry_point = query;
    }
    return hnsw;
}

/**
 * Alg 2
 * SEARCH-LAYER(hnsw, q, ep, ef, lc)
 * Note: Result is stored in entry_points (ep)
*/
void search_layer(Config* config, HNSW* hnsw, float* query, vector<pair<float, int>>& entry_points, int num_to_return, int layer_num) {
    set<int> visited;
    auto ge_comp = [](const pair<float, int>& a, const pair<float, int>& b) {
        return a.first > b.first;
    };
    priority_queue<pair<float, int>, vector<pair<float, int>>, decltype(ge_comp)> candidates(ge_comp);
    priority_queue<pair<float, int>> found;

    // Array of when each neighbor was found
    vector<int> when_neigh_found(config->num_return, -1);
    int nn_found = 0;

    // Add entry points to visited, candidates, and found
    for (auto entry : entry_points) {
        visited.insert(entry.second);
        candidates.emplace(entry);
        found.emplace(entry);

        if (program_state == 1 && log_neighbors) {
            auto loc = find(cur_groundtruth.begin(), cur_groundtruth.end(), entry.second);
            if (loc != cur_groundtruth.end()) {
                // Get neighbor index (xth closest) and log distance comp
                int index = distance(cur_groundtruth.begin(), loc);
                when_neigh_found[index] = dist_comps;
                ++nn_found;
                ++correct_nn_found;
                if (config->gt_smart_termination && nn_found == config->num_return)
                    // End search
                    priority_queue<pair<float, int>, vector<pair<float, int>>, decltype(ge_comp)> candidates(ge_comp);
            }
        }
    }

    int iteration = 0;
    while (!candidates.empty()) {
        if (debug_file != NULL) {
            // Export search data
            *debug_file << "Iteration " << iteration << endl;
            for (int index : visited)
                *debug_file << index << ",";
            *debug_file << endl;

            priority_queue<pair<float, int>, vector<pair<float, int>>, decltype(ge_comp)> temp_candidates(candidates);
            while (!temp_candidates.empty()) {
                *debug_file << temp_candidates.top().second << ",";
                temp_candidates.pop();
            }
            *debug_file << endl;

            priority_queue<pair<float, int>> temp_found(found);
            while (!temp_found.empty()) {
                *debug_file << temp_found.top().second << ",";
                temp_found.pop();
            }
            *debug_file << endl;
        }
        ++iteration;

        // Get and remove closest element in candiates to query
        int closest = candidates.top().second;
        float close_dist = candidates.top().first;
        candidates.pop();

        // Get furthest element in found to query
        int furthest = found.top().second;
        float far_dist = found.top().first;

        // If closest is further than furthest, stop
        if (close_dist > far_dist)
            break;

        // Get neighbors of closest in HNSWLayer
        vector<pair<float, int>>& neighbors = hnsw->mappings[closest][layer_num];

        for (auto n_pair : neighbors) {
            int neighbor = n_pair.second;
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);

                // Get furthest element in found to query
                float far_inner_dist = found.top().first;

                // If distance from query to neighbor is less than the distance from query to furthest,
                // or if the size of found is less than num_to_return,
                // add to candidates and found
                float neighbor_dist = calculate_l2_sq(query, hnsw->nodes[neighbor], config->dimensions);
                if (neighbor_dist < far_inner_dist || found.size() < num_to_return) {
                    candidates.emplace(neighbor_dist, neighbor);
                    found.emplace(neighbor_dist, neighbor);

                    if (program_state == 1 && log_neighbors) {
                        auto loc = find(cur_groundtruth.begin(), cur_groundtruth.end(), neighbor);
                        if (loc != cur_groundtruth.end()) {
                            // Get neighbor index (xth closest) and log distance comp
                            int index = distance(cur_groundtruth.begin(), loc);
                            when_neigh_found[index] = dist_comps;
                            ++nn_found;
                            ++correct_nn_found;
                            if (config->gt_smart_termination && nn_found == config->num_return)
                                // End search
                                priority_queue<pair<float, int>, vector<pair<float, int>>, decltype(ge_comp)> candidates(ge_comp);
                        }
                    }

                    // If found is greater than num_to_return, remove furthest
                    if (found.size() > num_to_return)
                        found.pop();
                }
            }
        }
    }

    // Place found elements into entry_points
    entry_points.clear();
    entry_points.resize(found.size());

    size_t idx = found.size();
    while (idx > 0) {
        --idx;
        entry_points[idx] = found.top();
        found.pop();
    }

    // Export when_neigh_found data
    if (program_state == 1 && log_neighbors)
        for (int i = 0; i < config->num_return; ++i) {
            *when_neigh_found_file << when_neigh_found[i] << " ";
        }
}

/**
 * Alg 5
 * K-NN-SEARCH(hnsw, q, K, ef)
 * Extra argument: path (for traversal debugging)
*/
vector<pair<float, int>> nn_search(Config* config, HNSW* hnsw, pair<int, float*>& query, int num_to_return, int ef_con, vector<int>& path) {
    vector<pair<float, int>> entry_points;
    entry_points.reserve(ef_con);
    float dist = calculate_l2_sq(query.second, hnsw->nodes[hnsw->entry_point], config->dimensions);
    entry_points.push_back(make_pair(dist, hnsw->entry_point));
    int top = hnsw->layers - 1;

    if (config->debug_search)
        cout << "Searching for " << num_to_return << " nearest neighbors of node " << query.first << endl;

    // Get closest element by using search_layer to find the closest point at each level
    for (int level = top; level >= 1; level--) {
        search_layer(config, hnsw, query.second, entry_points, 1, level);
        path.push_back(entry_points[0].second);

        if (config->debug_search)
            cout << "Closest point at level " << level << " is " << entry_points[0].second << " (" << entry_points[0].first << ")" << endl;
    }

    if (config->debug_query_search_index == query.first) {
        debug_file = new ofstream(config->export_dir + "query_search.txt");
    }
    if (config->gt_dist_log)
        log_neighbors = true;
    search_layer(config, hnsw, query.second, entry_points, ef_con, 0);
    if (config->gt_dist_log)
        log_neighbors = false;
    if (config->debug_query_search_index == query.first) {
        debug_file->close();
        delete debug_file;
        debug_file = NULL;
        cout << "Exported query search data to " << config->export_dir << "query_search.txt for query " << query.first << endl;
    }

    if (config->debug_search) {
        cout << "All closest points at level 0 are ";
        for (auto n_pair : entry_points)
            cout << n_pair.second << " (" << n_pair.first << ") ";
        cout << endl;
    }

    // Select closest elements
    entry_points.resize(min(entry_points.size(), (size_t)num_to_return));
    return entry_points;
}

bool sanity_checks(Config* config) {
    if (config->optimal_connections > config->max_connections) {
        cout << "Optimal connections cannot be greater than max connections" << endl;
        return false;
    }
    if (config->optimal_connections > config->ef_construction) {
        cout << "Optimal connections cannot be greater than beam width" << endl;
        return false;
    }
    if (config->num_return > config->num_nodes) {
        cout << "Number of nodes to return cannot be greater than number of nodes" << endl;
        return false;
    }
    if (config->ef_construction > config->num_nodes) {
        config->ef_construction = config->num_nodes;
        cout << "Warning: Beam width was set to " << config->num_nodes << endl;
    }
    if (config->num_return > config->ef_search) {
        config->num_return = config->ef_search;
        cout << "Warning: Number of queries to return was set to " << config->ef_search << endl;
    }
    return true;
}

HNSW* init_hnsw(Config* config, float** nodes) {
    HNSW* hnsw = new HNSW(config->num_nodes, nodes);
    hnsw->mappings.resize(config->num_nodes);

    // Insert first node into first layer with empty connections vector
    hnsw->layers = 1;
    hnsw->mappings[0].resize(1);
    hnsw->entry_point = 0;
    return hnsw;
}

void insert_nodes(Config* config, HNSW* hnsw) {
    mt19937 rand(config->level_seed);
    uniform_real_distribution<double> dis(0.0000001, 0.9999999);

    double normal_factor = 1 / -log(config->scaling_factor);
    for (int i = 1; i < config->num_nodes; i++) {
        insert(config, hnsw, i, config->optimal_connections, config->max_connections, config->ef_construction,
            normal_factor, [&]() {
                return dis(rand);
            }
        );
    }
}

void print_hnsw(Config* config, HNSW* hnsw) {
    if (config->print_graph) {
        vector<int> nodes_per_layer(hnsw->layers);
        for (int i = 0; i < config->num_nodes; ++i) {
            for (int j = 0; j < hnsw->mappings[i].size(); ++j)
                ++nodes_per_layer[j];
        }

        cout << "Nodes per layer: " << endl;
        for (int i = 0; i < hnsw->layers; ++i)
            cout << "Level " << i << ": " << nodes_per_layer[i] << endl;
        cout << endl;

        for (int i = 0; i < hnsw->layers; ++i) {
            cout << "Layer " << i << " connections: " << endl;
            for (int j = 0; j < config->num_nodes; ++j) {
                if (hnsw->mappings[j].size() <= i)
                    continue;

                cout << j << ": ";
                for (auto n_pair : hnsw->mappings[j][i])
                    cout << n_pair.second << " ";
                cout << endl;
            }
        }
    }
}

void run_query_search(Config* config, HNSW* hnsw, float** queries) {
    vector<vector<int>> paths(config->num_queries);
    ofstream* export_file = NULL;
    if (config->export_queries)
        export_file = new ofstream(config->export_dir + "queries.txt");
    
    ofstream* indiv_file = NULL;
    if (config->export_indiv)
        indiv_file = new ofstream(config->export_dir + "indiv.txt");

    if (config->gt_dist_log)
        when_neigh_found_file = new ofstream(config->export_dir + "when_neigh_found.txt");

    bool use_groundtruth = config->groundtruth_file != "";
    if (use_groundtruth && config->query_file == "") {
        cout << "Warning: Groundtruth file will not be used because queries were generated" << endl;
        use_groundtruth = false;
    }

    vector<vector<int>> actual_neighbors;
    if (use_groundtruth) {
        load_ivecs(config->groundtruth_file, actual_neighbors, config->num_queries, config->num_return);

        if (config->gt_dist_log)
            // Sort groundtruth neighbors by distance
            for (int i = 0; i < config->num_queries; ++i) {
                vector<int>& neighbors = actual_neighbors[i];
                sort(neighbors.begin(), neighbors.end(), [&](int a, int b) {
                    float dist_a = calculate_l2_sq(queries[i], hnsw->nodes[a], config->dimensions);
                    float dist_b = calculate_l2_sq(queries[i], hnsw->nodes[b], config->dimensions);
                    return dist_a < dist_b;
                });
            }
    } else
        actual_neighbors.resize(config->num_queries);

    int total_found = 0;
    program_state = 1;
    for (int i = 0; i < config->num_queries; ++i) {
        pair<int, float*> query = make_pair(i, queries[i]);
        if ((config->print_actual || config->print_indiv_found || config->print_total_found || config->export_indiv
            || config->gt_dist_log) && !use_groundtruth) {
            // Get actual nearest neighbors
            priority_queue<pair<float, int>> pq;
            for (int j = 0; j < config->num_nodes; ++j) {
                float dist = calculate_l2_sq(query.second, hnsw->nodes[j], config->dimensions);
                pq.emplace(dist, j);
                if (pq.size() > config->num_return)
                    pq.pop();
            }

            // Place actual nearest neighbors
            actual_neighbors[i].resize(config->num_return);

            int idx = config->num_return;
            while (idx > 0) {
                --idx;
                actual_neighbors[i][idx] = pq.top().second;
                pq.pop();
            }
        }
        cur_groundtruth = actual_neighbors[i];
        dist_comps = 0;
        vector<pair<float, int>> found = nn_search(config, hnsw, query, config->num_return, config->ef_search, paths[i]);
        if (config->gt_dist_log)
            *when_neigh_found_file << endl;
        
        if (config->print_results) {
            // Print out found
            cout << "Found " << found.size() << " nearest neighbors of [" << query.second[0];
            for (int dim = 1; dim < config->dimensions; ++dim)
                cout << " " << query.second[dim];
            cout << "] : ";
            for (auto n_pair : found)
                cout << n_pair.second << " ";
            cout << endl;
            // Print path
            cout << "Path taken: ";
            for (int path : paths[i])
                cout << path << " ";
            cout << endl;
        }

        if (config->print_actual) {
            // Print out actual
            cout << "Actual " << config->num_return << " nearest neighbors of [" << query.second[0];
            for (int dim = 1; dim < config->dimensions; ++dim)
                cout << " " << query.second[dim];
            cout << "] : ";
            for (int index : actual_neighbors[i])
                cout << index << " ";
            cout << endl;
        }

        if (config->print_indiv_found || config->print_total_found || config->export_indiv) {
            unordered_set<int> actual_set(actual_neighbors[i].begin(), actual_neighbors[i].end());
            int matching = 0;
            for (auto n_pair : found) {
                if (actual_set.find(n_pair.second) != actual_set.end())
                    ++matching;
            }

            if (config->print_indiv_found)
                cout << "Found " << matching << " (" << matching /  (double)config->num_return * 100 << "%) for query " << i << endl;
            if (config->print_total_found)
                total_found += matching;
            if (config->export_indiv)
                *indiv_file << matching / (double)config->num_return << " " << dist_comps << endl;
        }

        if (config->export_queries) {
            *export_file << "Query " << i << endl << query.second[0];
            for (int dim = 1; dim < config->dimensions; ++dim)
                *export_file << "," << query.second[dim];
            *export_file << endl;
            for (auto n_pair : found)
                *export_file << n_pair.second << ",";
            *export_file << endl;
            for (int node : paths[i])
                *export_file << node << ",";
            *export_file << endl;
        }
    }

    if (config->gt_dist_log) {
        cout << "Total neighbors found (gt comparison): " << correct_nn_found << " (" << correct_nn_found / (double)(config->num_queries * config->num_return) * 100 << "%)" << endl;
    }
    if (config->print_total_found) {
        cout << "Total neighbors found: " << total_found << " (" << total_found / (double)(config->num_queries * config->num_return) * 100 << "%)" << endl;
    }

    cout << "Finished search" << endl;
    if (export_file != NULL) {
        export_file->close();
        delete export_file;
        cout << "Exported queries to " << config->export_dir << "queries.txt" << endl;
    }
    if (indiv_file != NULL) {
        indiv_file->close();
        delete indiv_file;
        cout << "Exported individual query results to " << config->export_dir << "indiv.txt" << endl;
    }

    if (config->gt_dist_log) {
        when_neigh_found_file->close();
        delete when_neigh_found_file;
        cout << "Exported when neighbors were found to " << config->export_dir << "when_neigh_found.txt" << endl;
    }
}

void export_graph(Config* config, HNSW* hnsw, float** nodes) {
    if (config->export_graph) {
        ofstream file(config->export_dir + "graph.txt");

        // Export number of layers
        file << hnsw->layers << endl;

        // Export nodes
        file << "Nodes" << endl;
        for (int i = 0; i < config->num_nodes; ++i) {
            file << i << " " << hnsw->mappings[i].size() - 1 << ": " << nodes[i][0];
            for (int dim = 1; dim < config->dimensions; ++dim)
                file << "," << nodes[i][dim];
            file << endl;
        }

        // Export edges
        file << "Edges" << endl;
        for (int i = 0; i < config->num_nodes; ++i) {
            file << i << endl;
            for (int level = 0; level < hnsw->mappings[i].size(); ++level) {
                for (auto n_pair : hnsw->mappings[i][level])
                    file << n_pair.second << ",";
                file << endl;
            }
        }

        file.close();
        cout << "Exported graph to " << config->export_dir << "graph.txt" << endl;
    }
}
