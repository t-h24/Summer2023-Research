#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include <algorithm>

using namespace std;

const int DIMENSIONS = 10;
const int NUM_NODES = 100;
const int MAX_CONNECTIONS = 20;
const int EF_CONSTRUCTION = 30;

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
vector<Node*> select_neighbors_simple(HNSW* hnsw, Node* query, vector<Node*> candidates, int num);

/**
 * Alg 1
 * INSERT(hnsw, q, M, Mmax, efConstruction, mL)
*/
HNSW* insert(HNSW* hnsw, Node* query, int est_con, int max_con, int ef_con, float normal_factor) {
    vector<Node*> found;
    Node* entry_point = hnsw->nodes[0]; //TODO
    vector<Node*> entry_points;
    entry_points.push_back(entry_point);
    int top = hnsw->get_layers() - 1;
    
    // Get node level
    double random = (double)rand() / RAND_MAX;
    int node_level = -log(random) * normal_factor;

    // Get closest element by using search_layer to find the closest point at each level
    for (int level = top; level >= node_level + 1; level--) {
        found = search_layer(hnsw, query, entry_points, 1, level);
        entry_point = found[0];
    }

    for (int level = min(top, node_level); level >= 0; level--) {
        // Get nearest elements
        found = search_layer(hnsw, query, entry_points, ef_con, level);
        vector<Node*> neighbors = select_neighbors_simple(hnsw, query, found, est_con);

        // Add neighbors to HNSW layer mappings
        hnsw->layers[level]->mappings[query->index] = neighbors;

        //Connect neighbors to this node
        for (Node* neighbor : neighbors)
            hnsw->layers[level]->mappings[neighbor->index].push_back(query);

        // Trim neighbor connections if needed
        for (Node* neighbor : neighbors) {
            if (hnsw->layers[level]->mappings[neighbor->index].size() > max_con) {
                vector<Node*> trimmed = select_neighbors_simple(hnsw, neighbor, hnsw->layers[level]->mappings[neighbor->index], max_con);
                hnsw->layers[level]->mappings[neighbor->index] = trimmed;
            }
        }

        entry_points = found;
    }

    //TODO if l>L set enter-point for hnsw to q
    return hnsw;
}

/**
 * Alg 2
 * SEARCH-LAYER(hnsw, q, ep, ef, lc)
*/
vector<Node*> search_layer(HNSW* hnsw, Node* query, vector<Node*> entry_points, int num_to_return, int layer_num) {
    vector<Node*> visited;
    vector<Node*> candidates;
    vector<Node*> found;

    // Add entry points to visited, candidates, and found
    for (Node* entry_point : entry_points) {
        visited.push_back(entry_point);
        candidates.push_back(entry_point);
        found.push_back(entry_point);
    }

    while (candidates.size() > 0) {
        //TODO optimize with priority queue
        // Get closest element in candiates to query
        Node* closest = candidates[0];
        for (int i = 1; i < candidates.size(); i++) {
            if (query->distance(candidates[i]) < query->distance(closest))
                closest = candidates[i];
        }

        // Get furthest element in found to query
        Node* furthest = found[0];
        for (int i = 1; i < found.size(); i++) {
            if (query->distance(found[i]) > query->distance(furthest))
                furthest = found[i];
        }

        // If closest is further than furthest, stop
        if (query->distance(closest) > query->distance(furthest))
            break;

        // Get neighbors of closest in HNSWLayer
        vector<Node*> neighbors = hnsw->layers[layer_num]->mappings[closest->index];

        for (Node* neighbor : neighbors) {
            if (find(visited.begin(), visited.end(), neighbor) == visited.end()) {
                visited.push_back(neighbor);

                // Get furthest element in found to query using distance()
                Node* furthestInner = found[0];
                for (int i = 1; i < found.size(); i++) {
                    if (query->distance(found[i]) > query->distance(furthestInner))
                        furthestInner = found[i];
                }

                // If distance from query to neighbor is less than the distance from query to furthest,
                // or if the size of found is less than num_to_return,
                // add to candidates and found
                if (query->distance(neighbor) < query->distance(furthestInner) || found.size() < num_to_return) {
                    candidates.push_back(neighbor);
                    found.push_back(neighbor);

                    // If found is greater than num_to_return, remove furthest
                    if (found.size() > num_to_return) {
                        Node* furthestRemove = found[0];
                        for (int i = 1; i < found.size(); i++) {
                            if (query->distance(found[i]) > query->distance(furthestRemove))
                                furthestRemove = found[i];
                        }

                        found.erase(remove(found.begin(), found.end(), furthestRemove), found.end());
                    }
                }
            }
        }
    }
    return found;
}

/**
 * Alg 3
 * SELECT-NEIGHBORS-SIMPLE(hnsw, q, C, M)
*/
vector<Node*> select_neighbors_simple(HNSW* hnsw, Node* query, vector<Node*> candidates, int num) {
    vector<Node*> neighbors;
    //TODO: Not efficient, use priority queue
    for (int i = 0; i < num; i++) {
        Node* closest = candidates[0];
        for (int j = 1; j < candidates.size(); j++) {
            if (query->distance(candidates[j]) < query->distance(closest))
                closest = candidates[j];
        }
        neighbors.push_back(closest);
        candidates.erase(remove(candidates.begin(), candidates.end(), closest), candidates.end());
    }
    return neighbors;
}

int main() {
    // Generate NUM_NODES amount of nodes
    Node** nodes = generate_nodes(NUM_NODES);

    HNSW* hnsw = new HNSW(nodes);
    hnsw->layers.push_back(new HNSWLayer());

    // Insert first node into first layer with no connections
    hnsw->layers[0]->mappings.insert(pair<int, vector<Node*>>(0, vector<Node*>()));

    // Insert first nodes
    for (int i = 1; i < NUM_NODES; i++) {
        cout << "Inserting node " << i << endl; //TODO: test
        Node* query = nodes[i];
        insert(hnsw, query, 0, MAX_CONNECTIONS, EF_CONSTRUCTION, 0.5);//TODO what is est_con?
    }

    return 0;
}
