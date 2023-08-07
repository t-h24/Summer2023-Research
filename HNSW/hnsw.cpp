#include <iostream>
#include <math.h>
#include <algorithm>
#include <set>
#include <immintrin.h>
#include <unordered_set>
#include "hnsw.h"

using namespace std;

long long int dist_comps = 0;

Node::Node(int index, int dimensions, float* values) : index(index), dimensions(dimensions), 
    values(new float[dimensions]), debug_file(NULL) {
    for (int i = 0; i < dimensions; i++) {
        this->values[i] = values[i];
    }
}

float Node::distance(Node* other) {
    ++dist_comps;
    return calculate_l2_sq(this->values, other->values, this->dimensions);
}

Node::~Node() {
    delete[] values;
}

HNSWLayer::~HNSWLayer() {
    for (auto it = mappings.begin(); it != mappings.end(); it++) {
        delete it->second;
    }
}

HNSW::HNSW(int node_size, Node** nodes) : node_size(node_size), nodes(nodes) {}

int HNSW::get_layers() {
    return layers.size();
}

HNSW::~HNSW() {
    for (size_t i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
}

float calculate_l2_sq(float* a, float* b, int size) {
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

void load_fvecs(const string& file, const string& type, Node** nodes, int num, int dim, bool has_groundtruth) {
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
        float values[dim];
        f.read(reinterpret_cast<char*>(values), dim * 4);
        nodes[i] = new Node(i, dim, values);
    }
    f.close();
}

void load_ivecs(const string& file, vector<vector<int>> &results, int num, int num_return) {
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
    results.resize(num);
    f.seekg(0, ios::beg);
    for (int i = 0; i < num; i++) {
        // Skip list size
        f.seekg(4, ios::cur);

        // Read point
        int values[num_return];
        f.read(reinterpret_cast<char*>(values), num_return * 4);
        results[i] = vector<int>(values, values + num_return);

        // Skip remaining values
        f.seekg((width - num_return) * 4, ios::cur);
    }
    f.close();
}

Node** get_nodes(Config* config) {
    if (config->load_file != "") {
        if (config->load_file.size() >= 6 && config->load_file.substr(config->load_file.size() - 6) == ".fvecs") {
            // Load nodes from fvecs file
            Node** nodes = new Node*[config->num_nodes];
            load_fvecs(config->load_file, "nodes", nodes, config->num_nodes, config->dimensions, config->groundtruth_file != "");
            return nodes;
        }
    
        // Load nodes from file
        ifstream f(config->load_file, ios::in);
        if (!f) {
            cout << "File " << config->load_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << config->num_nodes << " nodes from file " << config->load_file << endl;

        Node** nodes = new Node*[config->num_nodes];
        for (int i = 0; i < config->num_nodes; i++) {
            float values[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> values[j];
            }
            nodes[i] = new Node(i, config->dimensions, values);
        }

        f.close();
        return nodes;
    }

    cout << "Generating " << config->num_nodes << " random nodes" << endl;

    mt19937 gen(config->graph_seed);
    uniform_real_distribution<float> dis(config->gen_min, config->gen_max);

    Node** nodes = new Node*[config->num_nodes];
    for (int i = 0; i < config->num_nodes; i++) {
        float values[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            values[j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
        nodes[i] = new Node(i, config->dimensions, values);
    }

    return nodes;
}

Node** get_queries(Config* config, Node** graph_nodes) {
    mt19937 gen(config->query_seed);
    if (config->query_file != "") {
        if (config->query_file.size() >= 6 && config->query_file.substr(config->query_file.size() - 6) == ".fvecs") {
            // Load queries from fvecs file
            Node** queries = new Node*[config->num_queries];
            load_fvecs(config->query_file, "queries", queries, config->num_queries, config->dimensions, config->groundtruth_file != "");
            return queries;
        }

        // Load queries from file
        ifstream f(config->query_file, ios::in);
        if (!f) {
            cout << "File " << config->query_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading " << config->num_queries << " queries from file " << config->query_file << endl;

        Node** queries = new Node*[config->num_queries];
        for (int i = 0; i < config->num_queries; i++) {
            float values[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> values[j];
            }
            queries[i] = new Node(i, config->dimensions, values);
        }

        f.close();
        return queries;
    }

    if (config->load_file == "") {
        // Generate random queries (same as get_nodes)
        cout << "Generating " << config->num_queries << " random queries" << endl;
        uniform_real_distribution<float> dis(config->gen_min, config->gen_max);

        Node** queries = new Node*[config->num_queries];
        for (int i = 0; i < config->num_queries; i++) {
            float values[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                values[j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
            }
            queries[i] = new Node(i, config->dimensions, values);
        }

        return queries;
    }
    
    // Generate queries randomly based on bounds of graph_nodes
    cout << "Generating queries based on file " << config->load_file << endl;
    float* lower_bound = new float[config->dimensions];
    float* upper_bound = new float[config->dimensions];
    copy(graph_nodes[0]->values, graph_nodes[0]->values + config->dimensions, lower_bound);
    copy(graph_nodes[0]->values, graph_nodes[0]->values + config->dimensions, upper_bound);

    // Calculate lowest and highest value for each dimension using graph_nodes
    for (int i = 1; i < config->num_nodes; i++) {
        for (int j = 0; j < config->dimensions; j++) {
            if (graph_nodes[i]->values[j] < lower_bound[j]) {
                lower_bound[j] = graph_nodes[i]->values[j];
            }
            if (graph_nodes[i]->values[j] > upper_bound[j]) {
                upper_bound[j] = graph_nodes[i]->values[j];
            }
        }
    }
    uniform_real_distribution<float>* dis_array = new uniform_real_distribution<float>[config->dimensions];
    for (int i = 0; i < config->dimensions; i++) {
        dis_array[i] = uniform_real_distribution<float>(lower_bound[i], upper_bound[i]);
    }

    // Generate queries based on the range of values in each dimension
    Node** queries = new Node*[config->num_queries];
    for (int i = 0; i < config->num_queries; i++) {
        float values[config->dimensions];
        for (int j = 0; j < config->dimensions; j++) {
            values[j] = round(dis_array[j](gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
        }
        queries[i] = new Node(i, config->dimensions, values);
    }

    delete[] lower_bound;
    delete[] upper_bound;
    delete[] dis_array;

    return queries;
}

/**
 * Alg 1
 * INSERT(hnsw, q, M, Mmax, efConstruction, mL)
 * Extra arguments: rand (for generating random value between 0 and 1)
 * Note: max_con is not used for level 0, instead max_connections_0 is used
*/
HNSW* insert(Config* config, HNSW* hnsw, Node* query, int opt_con, int max_con, int ef_con, float normal_factor, function<double()> rand) {
    vector<pair<float, Node*>> entry_points;
    entry_points.reserve(ef_con);
    entry_points.push_back(make_pair(query->distance(hnsw->entry_point), hnsw->entry_point));
    int top = hnsw->get_layers() - 1;
    
    // Get node level
    int node_level = -log(rand()) * normal_factor;
    query->level = node_level;

    if (config->debug_insert)
        cout << "Inserting node " << query->index << " at level " << node_level << " with entry point " << entry_points[0].second->index << endl;

    // Add layers if needed
    if (node_level > top)
        for (int i = top + 1; i <= node_level; i++) {
            if (config->debug_insert)
                cout << "Adding layer " << i << endl;

            HNSWLayer* layer = new HNSWLayer();
            hnsw->layers.push_back(layer);

            // Initialize mapping vector for current query
            layer->mappings[query->index] = new vector<pair<float, Node*>>();
        }

    // Get closest element by using search_layer to find the closest point at each level
    for (int level = top; level >= node_level + 1; level--) {
        search_layer(config, hnsw, query, &entry_points, 1, level);

        if (config->debug_insert)
            cout << "Closest point at level " << level << " is " << entry_points[0].second->index << " (" << entry_points[0].first << ")" << endl;
    }

    for (int level = min(top, node_level); level >= 0; level--) {
        if (level == 0)
            max_con = config->max_connections_0;

        // Get nearest elements
        search_layer(config, hnsw, query, &entry_points, ef_con, level);

        // Initialize mapping vector
        vector<pair<float, Node*>>* neighbors = hnsw->layers[level]->mappings[query->index] = new vector<pair<float, Node*>>();
        neighbors->reserve(max_con + 1);
        neighbors->resize(min(opt_con, (int)entry_points.size()));

        //Select opt_con number of neighbors from entry_points
        copy_n(entry_points.begin(), min(opt_con, (int)entry_points.size()), neighbors->begin());

        if (config->debug_insert) {
            cout << "Neighbors at level " << level << " are ";
            for (auto n_pair : *neighbors)
                cout << n_pair.second->index << " (" << n_pair.first << ") ";
            cout << endl;
        }

        //Connect neighbors to this node
        for (auto n_pair : *neighbors) {
            vector<pair<float, Node*>>* neighbor_mapping = hnsw->layers[level]->mappings[n_pair.second->index];

            // Place query in correct position in neighbor_mapping
            auto new_pair = make_pair(query->distance(n_pair.second), query);
            auto pos = lower_bound(neighbor_mapping->begin(), neighbor_mapping->end(), new_pair);
            neighbor_mapping->insert(pos, new_pair);
        }

        // Trim neighbor connections if needed
        for (auto n_pair : *neighbors) {
            if (hnsw->layers[level]->mappings[n_pair.second->index]->size() > max_con) {
                // Pop last element
                hnsw->layers[level]->mappings[n_pair.second->index]->pop_back();
            }
        }
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
void search_layer(Config* config, HNSW* hnsw, Node* query, vector<pair<float, Node*>>* entry_points, int num_to_return, int layer_num) {
    set<int> visited;
    priority_queue<pair<float, Node*>> candidates;
    priority_queue<pair<float, Node*>> found;

    // Add entry points to visited, candidates, and found
    for (auto entry : *entry_points) {
        visited.insert(entry.second->index);
        candidates.emplace(-entry.first, entry.second);
        found.emplace(entry);
    }

    int iteration = 0;
    while (!candidates.empty()) {
        if (query->debug_file != NULL && layer_num == 0) {
            // Export search data
            *query->debug_file << "Iteration " << iteration << endl;
            for (int index : visited)
                *query->debug_file << index << ",";
            *query->debug_file << endl;

            priority_queue<pair<float, Node*>> temp_candidates(candidates);
            while (!temp_candidates.empty()) {
                *query->debug_file << temp_candidates.top().second->index << ",";
                temp_candidates.pop();
            }
            *query->debug_file << endl;

            priority_queue<pair<float, Node*>> temp_found(found);
            while (!temp_found.empty()) {
                *query->debug_file << temp_found.top().second->index << ",";
                temp_found.pop();
            }
            *query->debug_file << endl;
        }
        ++iteration;

        // Get and remove closest element in candiates to query
        Node* closest = candidates.top().second;
        float close_dist = -candidates.top().first;
        candidates.pop();

        // Get furthest element in found to query
        Node* furthest = found.top().second;
        float far_dist = found.top().first;

        // If closest is further than furthest, stop
        if (close_dist > far_dist)
            break;

        // Get neighbors of closest in HNSWLayer
        vector<pair<float, Node*>>* neighbors = hnsw->layers[layer_num]->mappings[closest->index];

        for (auto n_pair : *neighbors) {
            Node* neighbor = n_pair.second;
            if (visited.find(neighbor->index) == visited.end()) {
                visited.insert(neighbor->index);

                // Get furthest element in found to query
                Node* furthest_inner = found.top().second;
                float far_inner_dist = found.top().first;

                // If distance from query to neighbor is less than the distance from query to furthest,
                // or if the size of found is less than num_to_return,
                // add to candidates and found
                float neighbor_dist = query->distance(neighbor);
                if (neighbor_dist < far_inner_dist || found.size() < num_to_return) {
                    candidates.emplace(-neighbor_dist, neighbor);
                    found.emplace(neighbor_dist, neighbor);

                    // If found is greater than num_to_return, remove furthest
                    if (found.size() > num_to_return)
                        found.pop();
                }
            }
        }
    }

    // Place found elements into entry_points
    entry_points->clear();
    entry_points->resize(found.size());

    size_t idx = found.size();
    while (idx > 0) {
        --idx;
        (*entry_points)[idx] = found.top();
        found.pop();
    }
}

/**
 * Alg 5
 * K-NN-SEARCH(hnsw, q, K, ef)
 * Extra argument: path (for traversal debugging)
*/
vector<pair<float, Node*>> nn_search(Config* config, HNSW* hnsw, Node* query, int num_to_return, int ef_con, vector<int>& path) {
    vector<pair<float, Node*>> entry_points;
    entry_points.reserve(ef_con);
    entry_points.push_back(make_pair(query->distance(hnsw->entry_point), hnsw->entry_point));
    int top = hnsw->get_layers() - 1;

    if (config->debug_search)
        cout << "Searching for " << num_to_return << " nearest neighbors of node " << query->index << endl;

    // Get closest element by using search_layer to find the closest point at each level
    for (int level = top; level >= 1; level--) {
        search_layer(config, hnsw, query, &entry_points, 1, level);
        path.push_back(entry_points[0].second->index);

        if (config->debug_search)
            cout << "Closest point at level " << level << " is " << entry_points[0].second->index << " (" << entry_points[0].first << ")" << endl;
    }

    search_layer(config, hnsw, query, &entry_points, ef_con, 0);

    if (config->debug_search) {
        cout << "All closest points at level 0 are ";
        for (auto n_pair : entry_points)
            cout << n_pair.second->index << " (" << n_pair.first << ") ";
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
    if (config->num_return > config->ef_construction_search) {
        config->num_return = config->ef_construction_search;
        cout << "Warning: Number of queries to return was set to " << config->ef_construction_search << endl;
    }
    return true;
}

HNSW* init_hnsw(Config* config, Node** nodes) {
    HNSW* hnsw = new HNSW(config->num_nodes, nodes);
    hnsw->layers.push_back(new HNSWLayer());

    // Insert first node into first layer with no connections (empty vector is inserted)
    nodes[0]->level = 0;
    hnsw->layers[0]->mappings.insert(pair<int, vector<pair<float, Node*>>*>(0, new vector<pair<float, Node*>>()));
    hnsw->entry_point = nodes[0];
    return hnsw;
}

void insert_nodes(Config* config, HNSW* hnsw, Node** nodes) {
    mt19937 rand(config->level_seed);
    uniform_real_distribution<double> dis(0.0000001, 0.9999999);

    double normal_factor = 1 / -log(config->scaling_factor);
    for (int i = 1; i < config->num_nodes; i++) {
        Node* query = nodes[i];
        insert(config, hnsw, query, config->optimal_connections, config->max_connections, config->ef_construction,
            normal_factor, [&]() {
                return dis(rand);
            }
        );
    }
}

void print_hnsw(Config* config, HNSW* hnsw) {
    if (config->debug_graph) {
        cout << "Nodes per layer: " << endl;
        for (int i = 0; i < hnsw->get_layers(); i++) {
            cout << "Level " << i << ": " << hnsw->layers[i]->mappings.size() << endl;
        }
        cout << endl;

        for (int i = hnsw->layers.size() - 1; i >= 0; i--) {
            cout << "Layer " << i << " connections: " << endl;
            for (auto const& mapping : hnsw->layers[i]->mappings) {
                cout << mapping.first << ": ";
                for (auto n_pair : *mapping.second)
                    cout << n_pair.second->index << " ";
                cout << endl;
            }
        }
    }
}

void run_query_search(Config* config, HNSW* hnsw, Node** queries) {
    vector<int>* paths = new vector<int>[config->num_queries];
    ofstream file(config->export_dir + "queries.txt");

    bool use_groundtruth = config->groundtruth_file != "";
    if (use_groundtruth && config->query_file == "") {
        cout << "Warning: Groundtruth file will not be used because queries were generated" << endl;
        use_groundtruth = false;
    }

    vector<vector<int>> actual_neighbors;
    if (use_groundtruth)
        load_ivecs(config->groundtruth_file, actual_neighbors, config->num_queries, config->num_return);

    int total_found = 0;
    for (int i = 0; i < config->num_queries; ++i) {
        Node* query = queries[i];
        vector<pair<float, Node*>> found = nn_search(config, hnsw, query, config->num_return, config->ef_construction_search, paths[i]);
        
        if (config->print_results) {
            // Print out found
            cout << "Found " << found.size() << " nearest neighbors of [" << query->values[0];
            for (int dim = 1; dim < config->dimensions; ++dim)
                cout << " " << query->values[dim];
            cout << "] : ";
            for (auto n_pair : found)
                cout << n_pair.second->index << " ";
            cout << endl;
            // Print path
            cout << "Path taken: ";
            for (int path : paths[i])
                cout << path << " ";
            cout << endl;
        }

        if (config->print_actual || config->print_indiv_found || config->print_total_found) {
            if (!use_groundtruth) {
                // Get actual nearest neighbors
                actual_neighbors.push_back(vector<int>());
                priority_queue<pair<float, Node*>> pq;
                for (int j = 0; j < config->num_nodes; ++j) {
                    pq.emplace(query->distance(hnsw->nodes[j]), hnsw->nodes[j]);
                    if (pq.size() > config->num_return)
                        pq.pop();
                }

                // Place actual nearest neighbors
                actual_neighbors[i].reserve(config->num_return);
                actual_neighbors[i].resize(config->num_return);

                int idx = config->num_return;
                while (idx > 0) {
                    --idx;
                    actual_neighbors[i][idx] = pq.top().second->index;
                    pq.pop();
                }
            }

            if (config->print_actual) {
                // Print out actual
                cout << "Actual " << config->num_return << " nearest neighbors of [" << query->values[0];
                for (int dim = 1; dim < config->dimensions; ++dim)
                    cout << " " << query->values[dim];
                cout << "] : ";
                for (int index : actual_neighbors[i])
                    cout << index << " ";
                cout << endl;
            }

            if (config->print_indiv_found || config->print_total_found) {
                unordered_set<int> actual_set(actual_neighbors[i].begin(), actual_neighbors[i].end());
                int matching = 0;
                for (auto n_pair : found) {
                    if (actual_set.find(n_pair.second->index) != actual_set.end())
                        ++matching;
                }

                if (config->print_indiv_found)
                    cout << "Found " << matching << " (" << matching /  (double)config->num_return * 100 << "%) for query " << i << endl;
                if (config->print_total_found)
                    total_found += matching;
            }
        }

        if (config->export_queries) {
            file << "Query " << i << endl << query->values[0];
            for (int dim = 1; dim < config->dimensions; ++dim)
                file << "," << query->values[dim];
            file << endl;
            for (auto n_pair : found)
                file << n_pair.second->index << ",";
            file << endl;
            for (int node : paths[i])
                file << node << ",";
            file << endl;
        }
    }

    if (config->print_total_found) {
        cout << "Total neighbors found: " << total_found << " (" << total_found / (double)(config->num_queries * config->num_return) * 100 << "%)" << endl;
    }

    cout << "Finished search" << endl;

    file.close();
    delete[] paths;
}

void export_graph(Config* config, HNSW* hnsw, Node** nodes) {
    if (config->export_graph) {
        auto level_comp = [](Node* a, Node* b) {
            return a->level < b->level;
        };
        vector<Node*> nodes_vec(nodes, nodes + config->num_nodes);
        sort(nodes_vec.begin(), nodes_vec.end(), level_comp);
        ofstream file(config->export_dir + "graph.txt");

        // Export nodes
        file << "Nodes" << endl;
        int start_loc = 0;
        bool skipped = false;
        // Each level contains its nodes and all nodes from higher levels
        for (int level = 0; level < hnsw->get_layers(); ++level) {
            skipped = false;
            file << "Level " << level << endl;
            for (size_t i = start_loc; i < nodes_vec.size(); ++i) {
                file << nodes_vec[i]->index << ": " << nodes_vec[i]->values[0];
                for (int dim = 1; dim < config->dimensions; ++dim)
                    file << "," << nodes_vec[i]->values[dim];
                file << endl;

                if (!skipped && nodes_vec[i]->level > level) {
                    start_loc = i;
                    skipped = true;
                }
            }
        }

        // Export edges
        file << "Edges" << endl;
        for (int level = 0; level < hnsw->get_layers(); ++level) {
            file << "Level " << level << endl;
            HNSWLayer* layer = hnsw->layers[level];
            
            // Append neighbors of each node in a single line
            for (auto it = layer->mappings.begin(); it != layer->mappings.end(); ++it) {
                if (it->second->empty())
                    continue;
                file << it->first << ":";
                for (auto n_pair : *it->second)
                    file << n_pair.second->index << ",";
                file << endl;
            }
        }
        file.close();
    }
}
