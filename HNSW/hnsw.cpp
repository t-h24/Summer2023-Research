#include <iostream>
#include <vector>

using namespace std;

const int DIMENSIONS = 10;
const int NUM_NODES = 100;

class Node {
public:
    float values[DIMENSIONS];

    Node(float values[DIMENSIONS]) {
        for (int i = 0; i < DIMENSIONS; i++) {
            this->values[i] = values[i];
        }
    }
};

vector<Node*> generate_nodes(int amount) {
    // Pre-set seed (consistent outputs)
    srand(0);
    vector<Node*> nodes;
    for (int i = 0; i < amount; i++) {
        float values[DIMENSIONS];
        for (int j = 0; j < DIMENSIONS; j++) {
            values[j] = (float)rand() / RAND_MAX * 1000;
        }
        nodes.push_back(new Node(values));
    }

    return nodes;
}

int main() {
    // Generate NUM_NODES amount of nodes
    vector<Node*> nodes = generate_nodes(NUM_NODES);

    
    return 0;
}
