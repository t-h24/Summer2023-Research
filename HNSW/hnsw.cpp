#include <iostream>
#include <math.h>
#include <algorithm>
#include <set>
#include "hnsw.h"

using namespace std;

Node::Node(int index, int dimensions, float* values) : index(index), dimensions(dimensions), 
    values(new float[dimensions]), debug_file(NULL) {
    for (int i = 0; i < dimensions; i++) {
        this->values[i] = values[i];
    }
}

double Node::distance(Node* other) {
    double sum = 0;
    for (int i = 0; i < dimensions; i++) {
        sum += pow(this->values[i] - other->values[i], 2);
    }
    return sum;
}

Node::~Node() {
    delete[] values;
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

Node** get_nodes(Config* config) {
    if (config->load_file != "") {
        // Load nodes from file
        ifstream f(config->load_file, ios::in);
        if (!f) {
            cout << "File " << config->load_file << " not found!" << endl;
            exit(1);
        }
        cout << "Loading nodes from file " << config->load_file << endl;

        Node** nodes = new Node*[config->num_nodes];
        for (int i = 0; i < config->num_nodes; i++) {
            float values[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                f >> values[j];
            }
            nodes[i] = new Node(i, config->dimensions, values);
        }

        return nodes;
    }

    cout << "Generating random nodes" << endl;

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
    if (config->load_file == "") {
        // Generate random nodes (same as get_nodes)
        cout << "Generating random queries" << endl;
        uniform_real_distribution<float> dis(config->gen_min, config->gen_max);

        Node** nodes = new Node*[config->num_queries];
        for (int i = 0; i < config->num_queries; i++) {
            float values[config->dimensions];
            for (int j = 0; j < config->dimensions; j++) {
                values[j] = round(dis(gen) * pow(10, config->gen_decimals)) / pow(10, config->gen_decimals);
            }
            nodes[i] = new Node(i, config->dimensions, values);
        }

        return nodes;
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
*/
HNSW* insert(Config* config, HNSW* hnsw, Node* query, int opt_con, int max_con, int ef_con, float normal_factor, function<double()> rand) {
    deque<Node*> found;
    deque<Node*> entry_points = { hnsw->entry_point };
    int top = hnsw->get_layers() - 1;
    
    // Get node level
    int node_level = -log(rand()) * normal_factor;
    query->level = node_level;

    if (config->debug_insert)
        cout << "Inserting node " << query->index << " at level " << node_level << " with entry point " << entry_points[0]->index << endl;

    // Add layers if needed
    if (node_level > top)
        for (int i = top + 1; i <= node_level; i++) {
            if (config->debug_insert)
                cout << "Adding layer " << i << endl;

            HNSWLayer* layer = new HNSWLayer();
            hnsw->layers.push_back(layer);
        }

    // Get closest element by using search_layer to find the closest point at each level
    for (int level = top; level >= node_level + 1; level--) {
        found = search_layer(config, hnsw, query, entry_points, 1, level);
        entry_points = found;

        if (config->debug_insert)
            cout << "Closest point at level " << level << " is " << entry_points[0]->index << endl;
    }

    for (int level = min(top, node_level); level >= 0; level--) {
        // Get nearest elements
        found = search_layer(config, hnsw, query, entry_points, ef_con, level);
        deque<Node*> neighbors = select_neighbors_simple(config, hnsw, query, found, opt_con, false);

        if (config->debug_insert) {
            cout << "Neighbors at level " << level << " are ";
            for (Node* neighbor : neighbors)
                cout << neighbor->index << " ";
            cout << endl;
        }

        // Add neighbors to HNSW layer mappings
        hnsw->layers[level]->mappings[query->index] = neighbors;

        //Connect neighbors to this node
        for (Node* neighbor : neighbors)
            hnsw->layers[level]->mappings[neighbor->index].push_back(query);

        // Trim neighbor connections if needed
        for (Node* neighbor : neighbors) {
            if (hnsw->layers[level]->mappings[neighbor->index].size() > max_con) {
                deque<Node*> trimmed = select_neighbors_simple(config, hnsw, neighbor, hnsw->layers[level]->mappings[neighbor->index], max_con, true);
                hnsw->layers[level]->mappings[neighbor->index] = trimmed;
            }
        }

        entry_points = found;
    }

    if (node_level > top) {
        hnsw->entry_point = query;
    }
    return hnsw;
}

/**
 * Alg 2
 * SEARCH-LAYER(hnsw, q, ep, ef, lc)
*/
deque<Node*> search_layer(Config* config, HNSW* hnsw, Node* query, deque<Node*> entry_points, int num_to_return, int layer_num) {
    auto close_dist_comp = [query](Node* a, Node* b) {	
        return query->distance(a) > query->distance(b);	
    };
    auto far_dist_comp = [query](Node* a, Node* b) {
        return query->distance(a) < query->distance(b);	
    };
    set<int> visited;
    priority_queue<Node*, deque<Node*>, decltype(close_dist_comp)> candidates(close_dist_comp);
    priority_queue<Node*, deque<Node*>, decltype(far_dist_comp)> found(far_dist_comp);

    // Add entry points to visited, candidates, and found
    for (Node* entry_point : entry_points) {
        visited.insert(entry_point->index);
        candidates.push(entry_point);
        found.push(entry_point);
    }

    int iteration = 0;
    while (candidates.size() > 0) {
        if (query->debug_file != NULL && layer_num == 0) {
            // Export search data
            *query->debug_file << "Iteration " << iteration << endl;
            for (int index : visited)
                *query->debug_file << index << ",";
            *query->debug_file << endl;

            priority_queue<Node*, deque<Node*>, decltype(close_dist_comp)> temp_candidates(candidates);
            while (!temp_candidates.empty()) {
                *query->debug_file << temp_candidates.top()->index << ",";
                temp_candidates.pop();
            }
            *query->debug_file << endl;

            priority_queue<Node*, deque<Node*>, decltype(far_dist_comp)> temp_found(found);
            while (!temp_found.empty()) {
                *query->debug_file << temp_found.top()->index << ",";
                temp_found.pop();
            }
            *query->debug_file << endl;
        }
        ++iteration;

        // Get and remove closest element in candiates to query
        Node* closest = candidates.top();
        candidates.pop();

        // Get furthest element in found to query
        Node* furthest = found.top();

        // If closest is further than furthest, stop
        if (query->distance(closest) > query->distance(furthest))
            break;

        // Get neighbors of closest in HNSWLayer
        deque<Node*>& neighbors = hnsw->layers[layer_num]->mappings[closest->index];

        for (Node* neighbor : neighbors) {
            if (visited.find(neighbor->index) == visited.end()) {
                visited.insert(neighbor->index);

                // Get furthest element in found to query
                Node* furthest_inner = found.top();

                // If distance from query to neighbor is less than the distance from query to furthest,
                // or if the size of found is less than num_to_return,
                // add to candidates and found
                if (query->distance(neighbor) < query->distance(furthest_inner) || found.size() < num_to_return) {
                    candidates.push(neighbor);
                    found.push(neighbor);

                    // If found is greater than num_to_return, remove furthest
                    if (found.size() > num_to_return)
                        found.pop();
                }
            }
        }
    }

    deque<Node*> result;
    while (found.size() > 0) {
        result.push_back(found.top());
        found.pop();
    }
    return result;
}

/**
 * Alg 3
 * SELECT-NEIGHBORS-SIMPLE(hnsw, q, C, M)
 * Extra argument: drop (for debugging)
*/
deque<Node*> select_neighbors_simple(Config* config, HNSW* hnsw, Node* query, deque<Node*> candidates, int num, bool drop) {
    if (candidates.size() <= num)
        return candidates;

    auto dist_comp = [query](Node* a, Node* b) {
        return query->distance(a) > query->distance(b);
    };
    priority_queue<Node*, deque<Node*>, decltype(dist_comp)> queue(dist_comp, deque<Node*>(candidates.begin(), candidates.end()));

    // Fetch num closest elements
    deque<Node*> neighbors;
    for (int i = 0; i < num; i++) {
        neighbors.push_back(queue.top());
        queue.pop();
    }

    if (config->debug_insert && drop) {
        cout << "Dropped neighbors for node " << query->index << " are ";
        while (!queue.empty()) {
            cout << queue.top()->index << " ";
            queue.pop();
        }
        cout << endl;
    }
    return neighbors;
}

/**
 * Alg 5
 * K-NN-SEARCH(hnsw, q, K, ef)
 * Extra argument: path (for traversal debugging)
*/
deque<Node*> nn_search(Config* config, HNSW* hnsw, Node* query, int num_to_return, int ef_con, vector<int>& path) {
    deque<Node*> found;
    deque<Node*> entry_points = { hnsw->entry_point };
    int top = hnsw->get_layers() - 1;

    if (config->debug_search)
        cout << "Searching for " << num_to_return << " nearest neighbors of node " << query->index << endl;

    // Get closest element by using search_layer to find the closest point at each level
    for (int level = top; level >= 1; level--) {
        found = search_layer(config, hnsw, query, entry_points, 1, level);
        entry_points = found;
        path.push_back(found[0]->index);

        if (config->debug_search)
            cout << "Closest point at level " << level << " is " << entry_points[0]->index << endl;
    }

    found = search_layer(config, hnsw, query, entry_points, ef_con, 0);
    return select_neighbors_simple(config, hnsw, query, found, num_to_return, false);
}

bool sanity_checks(Config* config) {
    if (config->optimal_connections > config->max_connections) {
        cout << "Optimal connections cannot be greater than max connections" << endl;
        return false;
    }
    if (config->ef_construction < config->max_connections) {
        cout << "Max connections must be less than beam width" << endl;
        return false;
    }
    return true;
}

HNSW* init_hnsw(Config* config, Node** nodes) {
    HNSW* hnsw = new HNSW(config->num_nodes, nodes);
    hnsw->layers.push_back(new HNSWLayer());

    // Insert first node into first layer with no connections (empy deque is inserted)
    nodes[0]->level = 0;
    hnsw->layers[0]->mappings.insert(pair<int, deque<Node*>>(0, deque<Node*>()));
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
            });
    }
}

void print_hnsw(Config* config, HNSW* hnsw) {
    if (config->debug_graph) {
        for (int i = hnsw->layers.size() - 1; i >= 0; i--) {
            cout << "Layer " << i << " connections: " << endl;
            for (auto const& mapping : hnsw->layers[i]->mappings) {
                cout << mapping.first << ": ";
                for (Node* node : mapping.second)
                    cout << node->index << " ";
                cout << endl;
            }
        }
    }
}

void run_query_search(Config* config, HNSW* hnsw, Node** queries) {
    vector<int>* paths = new vector<int>[config->num_queries];
    ofstream file("runs/queries.txt");

    for (int i = 0; i < config->num_queries; ++i) {
        Node* query = queries[i];
        deque<Node*> found = nn_search(config, hnsw, query, config->num_return, config->ef_construction, paths[i]);

        // Print out found
        cout << "Found " << found.size() << " nearest neighbors of [" << query->values[0];
        for (int dim = 1; dim < config->dimensions; ++dim)
            cout << " " << query->values[dim];
        cout << "] : ";
        for (Node* node : found)
            cout << node->index << " ";
        cout << endl;

        // Print path
        cout << "Path taken: ";
        for (int path : paths[i])
            cout << path << " ";
        cout << endl;

        if (config->export_queries) {
            file << "Query " << i << endl << query->values[0];
            for (int dim = 1; dim < config->dimensions; ++dim)
                file << "," << query->values[dim];
            file << endl;
            for (Node* node : found)
                file << node->index << ",";
            file << endl;
            for (int node : paths[i])
                file << node << ",";
            file << endl;
        }
    }
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
        ofstream file("runs/graph.txt");

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
            for (size_t i = 0; i < nodes_vec.size(); ++i) {
                if (layer->mappings[i].empty())
                    continue;
                file << i << ":";
                for (Node* neighbor : layer->mappings[i])
                    file << neighbor->index << ",";
                file << endl;
            }
        }
    }
}
