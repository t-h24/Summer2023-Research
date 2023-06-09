#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include <algorithm>
#include <queue>
#include <set>

using namespace std;

const int DIMENSIONS = 2;
const int NUM_NODES = 200;
const int OPTIMAL_CONNECTIONS = 15;
const int MAX_CONNECTIONS = 30;
const int EF_CONSTRUCTION = 30;
const double SCALING_FACTOR = 0.5;

const int NUM_QUERIES = 10;
const int NUM_RETURN = 15;

const bool DEBUG_INSERT = false;
const bool DEBUG_GRAPH = false;
const bool DEBUG_SEARCH = false;

class Node {
public:
    int index;
    float values[DIMENSIONS];

    Node(int index, float values[DIMENSIONS]) : index(index) {
        for (int i = 0; i < DIMENSIONS; i++) {
            this->values[i] = values[i];
        }
    }

    double distance(Node* other) {
        double sum = 0;
        for (int i = 0; i < DIMENSIONS; i++) {
            sum += pow(this->values[i] - other->values[i], 2);
        }
        return sum;
    }
};

class HNSWLayer {
public:
    map<int, vector<Node*>> mappings;
};

class HNSW {
public:
    Node** nodes;
    vector<HNSWLayer*> layers;
    Node* entry_point;

    HNSW(Node** nodes) : nodes(nodes) {}

    int get_layers() {
        return layers.size();
    }
};

Node** generate_nodes(int amount) {
    // Pre-set seed (consistent outputs)
    srand(0);
    Node** nodes = new Node*[amount];
    for (int i = 0; i < amount; i++) {
        float values[DIMENSIONS];
        for (int j = 0; j < DIMENSIONS; j++) {
            values[j] = (float)rand() / RAND_MAX * 1000;
        }
        nodes[i] = new Node(i, values);
    }

    return nodes;
}

HNSW* insert(HNSW* hnsw, Node* query, int est_con, int max_con, int ef_con, float normal_factor);
vector<Node*> search_layer(HNSW* hnsw, Node* query, vector<Node*> entry_points, int num_to_return, int layer_num);
vector<Node*> select_neighbors_simple(HNSW* hnsw, Node* query, vector<Node*> candidates, int num, bool drop);

/**
 * Alg 1
 * INSERT(hnsw, q, M, Mmax, efConstruction, mL)
*/
HNSW* insert(HNSW* hnsw, Node* query, int opt_con, int max_con, int ef_con, float normal_factor) {
    vector<Node*> found;
    vector<Node*> entry_points = { hnsw->entry_point };
    int top = hnsw->get_layers() - 1;
    
    // Get node level
    double random = (double)rand() / RAND_MAX;
    int node_level = -log(random) * normal_factor;

    if (DEBUG_INSERT)
        cout << "Inserting node " << query->index << " at level " << node_level << " with entry point " << entry_points[0]->index << endl;

    // Add layers if needed
    if (node_level > top)
        for (int i = top + 1; i <= node_level; i++) {
            if (DEBUG_INSERT)
                cout << "Adding layer " << i << endl;

            HNSWLayer* layer = new HNSWLayer();
            hnsw->layers.push_back(layer);
        }

    // Get closest element by using search_layer to find the closest point at each level
    for (int level = top; level >= node_level + 1; level--) {
        found = search_layer(hnsw, query, entry_points, 1, level);
        entry_points = found;

        if (DEBUG_INSERT)
            cout << "Closest point at level " << level << " is " << entry_points[0]->index << endl;
    }

    for (int level = min(top, node_level); level >= 0; level--) {
        // Get nearest elements
        found = search_layer(hnsw, query, entry_points, ef_con, level);
        vector<Node*> neighbors = select_neighbors_simple(hnsw, query, found, opt_con, false);

        if (DEBUG_INSERT) {
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
                vector<Node*> trimmed = select_neighbors_simple(hnsw, neighbor, hnsw->layers[level]->mappings[neighbor->index], max_con, true);
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
vector<Node*> search_layer(HNSW* hnsw, Node* query, vector<Node*> entry_points, int num_to_return, int layer_num) {
    auto close_dist_comp = [query](Node* a, Node* b) {	
        return query->distance(a) > query->distance(b);	
    };
    auto far_dist_comp = [query](Node* a, Node* b) {
        return query->distance(a) < query->distance(b);	
    };
    set<int> visited;
    priority_queue<Node*, vector<Node*>, decltype(close_dist_comp)> candidates(close_dist_comp);
    priority_queue<Node*, vector<Node*>, decltype(far_dist_comp)> found(far_dist_comp);

    // Add entry points to visited, candidates, and found
    for (Node* entry_point : entry_points) {
        visited.insert(entry_point->index);
        candidates.push(entry_point);
        found.push(entry_point);
    }

    while (candidates.size() > 0) {
        // Get and remove closest element in candiates to query
        Node* closest = candidates.top();
        candidates.pop();

        // Get furthest element in found to query
        Node* furthest = found.top();

        // If closest is further than furthest, stop
        if (query->distance(closest) > query->distance(furthest))
            break;

        // Get neighbors of closest in HNSWLayer
        vector<Node*>& neighbors = hnsw->layers[layer_num]->mappings[closest->index];

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

    vector<Node*> result;
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
vector<Node*> select_neighbors_simple(HNSW* hnsw, Node* query, vector<Node*> candidates, int num, bool drop) {
    if (candidates.size() <= num)
        return candidates;

    auto dist_comp = [query](Node* a, Node* b) {
        return query->distance(a) > query->distance(b);
    };
    priority_queue<Node*, vector<Node*>, decltype(dist_comp)> queue(dist_comp, candidates);

    // Fetch num closest elements
    vector<Node*> neighbors;
    for (int i = 0; i < num; i++) {
        neighbors.push_back(queue.top());
        queue.pop();
    }

    if (DEBUG_INSERT && drop) {
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
*/
vector<Node*> nn_search(HNSW* hnsw, Node* query, int num_to_return, int ef_con) {
    vector<Node*> found;
    vector<Node*> entry_points = { hnsw->entry_point };
    int top = hnsw->get_layers() - 1;

    if (DEBUG_SEARCH)
        cout << "Searching for " << num_to_return << " nearest neighbors of node " << query->index << endl;

    // Get closest element by using search_layer to find the closest point at each level
    for (int level = top; level >= 1; level--) {
        found = search_layer(hnsw, query, entry_points, 1, level);
        entry_points = found;

        if (DEBUG_SEARCH)
            cout << "Closest point at level " << level << " is " << entry_points[0]->index << endl;
    }

    found = search_layer(hnsw, query, entry_points, num_to_return, 0);
    return found;
}

int main() {
    // Generate NUM_NODES amount of nodes
    Node** nodes = generate_nodes(NUM_NODES);
    cout << "Beginning HNSW construction" << endl;

    HNSW* hnsw = new HNSW(nodes);
    hnsw->layers.push_back(new HNSWLayer());

    // Insert first node into first layer with no connections
    hnsw->layers[0]->mappings.insert(pair<int, vector<Node*>>(0, vector<Node*>()));
    hnsw->entry_point = nodes[0];

    // Insert nodes
    double normal_factor = 1 / -log(SCALING_FACTOR);
    for (int i = 1; i < NUM_NODES; i++) {
        Node* query = nodes[i];
        insert(hnsw, query, OPTIMAL_CONNECTIONS, MAX_CONNECTIONS, EF_CONSTRUCTION, normal_factor);
    }

    // Print results
    if (DEBUG_GRAPH) {
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
    
    // Generate NUM_QUERIES amount of nodes
    Node** queries = generate_nodes(NUM_QUERIES);
    cout << "Beginning search" << endl;

    for (int i = 0; i < NUM_QUERIES; ++i) {
        Node* query = queries[i];
        vector<Node*> found = nn_search(hnsw, query, NUM_RETURN, EF_CONSTRUCTION);

        // Print out found
        cout << "Found " << found.size() << " nearest neighbors of node " << query->index << ": ";
        for (Node* node : found)
            cout << node->index << " ";
        cout << endl;
    }
    return 0;
}
