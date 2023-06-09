#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

int TOTAL = 10000;
int alpha = 1;
int L = 20;
int R = 20;

class Graph {
private:
    struct Node {
        int val;
        vector<Node*> outEdge;
    };
public:
    Graph(){}
    Node* findNode(int val) {
        for (Node* one : allNodes) {
            if (one->val == val) {
                return one;
            }
        }
        return nullptr;
    }
    void addEdge(int me, const vector<int>& neighbors) {
        Node* thisNode = findNode(me);
        if (thisNode != nullptr) {
            for (int neighbor : neighbors) {
                Node* other = findNode(neighbor);
                if (other != nullptr) {
                    thisNode->outEdge.push_back(other);
                }
            }
        }
    }
    void addNode(int val, const vector<int>& neighbors) {
        Node* newNode = new Node();
        allNodes.push_back(newNode);
        newNode->val = val;
        addEdge(val, neighbors);
    }
    void randomize(int R) {
        size_t total = allNodes.size();
        for (Node* each : allNodes) {
            vector<int> neighbors = {};
            for (int i = 0; i < R; i++) {
                int random = rand() % total + 1; // find a random node
                neighbors.push_back(random);
            }
            addEdge(each->val, neighbors);
        }
    }
    vector<int> getNeighbors(int i) {
        vector<int> result;
        for (Node* node : allNodes) {
            if (node->val == i) {
                for (Node* neighbor : node->outEdge) {
                    result.push_back(neighbor->val);
                }
            }
        }
        return result;
    }
    void clearNeighbors(int i) {
        for (Node* node : allNodes) {
            if (node->val == i) {
                node->outEdge = {};
            }
        }
    }
    void displayGraph() {
        for (Node* one : allNodes) {
            cout << "Node: " << one->val << endl;
            cout << "Out-Edges: ";
            for (Node* neighbor : one->outEdge) {
                cout << neighbor->val << " ";
            }
            cout << endl;
        }
    }
    ~Graph() {
        for (Node* one : allNodes) {
            delete one;
        }
    }
private:
    vector<Node*> allNodes = {};
};

void constructGraph(const vector<int>& allNodes, Graph& graph) {
    cout << "Constructing graph" << endl;
    for (int node : allNodes) {
        graph.addNode(node, {});
    }
}

void randomEdges(Graph& graph, int R) {
    cout << "Randomizing edges" << endl;
    graph.randomize(R);
}

bool findInVec(int val, const vector<int>& vec) {
    for (int i : vec) {
        if (i == val) { return true; }
    }
    return false;
}

vector<int> setDiff(const vector<int>& setOne, const vector<int>& setTwo) {
    vector<int> diff;
    for (int i : setOne) {
        if (!findInVec(i, setTwo)) {
            diff.push_back(i);
        }
    }
    return diff;
}

long findDistance(int i, int query) {
    return abs(query - i);
}

tuple<vector<int>, vector<int>> GreedySearch(Graph& graph, int start, int query, size_t k, size_t l) {
    cout << "In GreedySearch, start = " << start << ", query = " << query << endl;
    vector<int> List;
    List.push_back(start);
    vector<int> Visited;
    while (setDiff(List, Visited).size() != 0) {
        vector<int> diff = setDiff(List, Visited);
        int bestCandidate = diff[0];
        for (int i : diff) {
            if (findDistance(i, query) < findDistance(bestCandidate, query)) {
                bestCandidate = i;
            }
        }
        // update L and V
        Visited.push_back(bestCandidate);
        for (int j : graph.getNeighbors(bestCandidate)) {
            List.push_back(j);
        }
        while (List.size() > l) {
            size_t worstCandidiate = 0;
            for (size_t ind = 1; ind < List.size(); ind++) {
                if (findDistance(List[ind], query) > findDistance(List[worstCandidiate], query)) {
                    worstCandidiate = ind;
                }
            }
            List[worstCandidiate] = List[List.size()-1];
            List.pop_back();
        }
    }
    return {List, Visited};
}

void RobustPrune(Graph& graph, int point, vector<int>& candidates, long threshold, int R) {
    cout << "In RobustPrune, point = " << point << endl;
    vector<int> neighbors = graph.getNeighbors(point);
    for (int i : neighbors) {
        candidates.push_back(i);
    }
    graph.clearNeighbors(point);
    while (candidates.size() != 0) {
        int bestCandidate = candidates[0];
        for (int j : candidates) {
            if (findDistance(j, point) < findDistance(bestCandidate, point)) {
                bestCandidate = j;
            }
        }
        graph.addEdge(point, {bestCandidate});
        if (graph.getNeighbors(point).size() == R) {
            break;
        }
        for (size_t ind = 0; ind < candidates.size(); ind++) {
            if (findDistance(bestCandidate, candidates[ind]) * threshold < findDistance(point, candidates[ind])) {
                candidates[candidates.size()-1] = candidates[ind];
                candidates.pop_back();
            }
        }
    }
}

Graph Vamana(const vector<int>& allNodes, long alpha, int L, int R) {
    Graph graph;
    cout << "Start of Vamana" << endl;
    constructGraph(allNodes, graph);
    randomEdges(graph, R);
    int s = allNodes.size() / 2;
    vector<int> sigma;
    for (int i = 1; i < allNodes.size()+1; i++) {
        sigma.push_back(i);
    }
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(sigma.begin(), sigma.end(), default_random_engine(seed));
    for (int i : sigma) {
        tuple<vector<int>, vector<int>> tup = GreedySearch(graph, s, allNodes[i], 1, L);
        vector<int> L = get<0>(tup);
        vector<int> V = get<1>(tup);
        RobustPrune(graph, allNodes[i], V, alpha, R);
        for (int j : graph.getNeighbors(allNodes[i])) {
            if (graph.getNeighbors(j).size() + 1 > R) {
                vector<int> unionV = graph.getNeighbors(j);
                unionV.push_back(allNodes[i]);
                RobustPrune(graph, j, unionV, alpha, R);
            } else {
                graph.addEdge(j, {allNodes[i]});
            }
        }
    }
    cout << "End of Vamana" << endl;
    return graph;
}

int main() {
    vector<int> allNodes = {};
    for (int i = 1; i < TOTAL; i++) {
        allNodes.push_back(i);
    }
    Graph G = Vamana(allNodes, alpha, L, R);
    G.displayGraph();
}
