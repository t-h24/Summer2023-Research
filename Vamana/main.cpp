#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <immintrin.h>

using namespace std;

// avx parallel computation
// avoid distance computation - maybe store??
// flat profile calculate runtime

int TOTAL = 500000;
int alpha = 1;
int L = 20;
int R = 20;
string FILENAME = "../../hdd1/home/sw4293/sift_base.txt";
//string FILENAME = "sift_base.txt";
size_t DIMENSION = 128;

class DataNode {
    friend ostream& operator<<(ostream& os, const DataNode& rhs) {
        for (size_t i = 0; i < DIMENSION; i++) {
            os << rhs.coordinates[i] << ' ';
        }
        os << endl;
        return os;
    }
public:
    DataNode() {}
    DataNode(float* coord) {
        dimension = DIMENSION;
        coordinates = coord;
    }
    void sumArraysAVX(float* array1, float* array2, float* result, int size) const {
//        cout << "In sumArraysAVX" << endl;
        int vectorSize = sizeof(__m256) / sizeof(float);
        for (int i = 0; i < size; i += vectorSize) {
            __m256 vec1 = _mm256_load_ps(&array1[i]);
            __m256 vec2 = _mm256_load_ps(&array2[i]);
            __m256 sub = _mm256_sub_ps(vec1, vec2);
            _mm256_store_ps(&result[i], sub);
        }
    }
    long long int findDistanceAVX(const DataNode& other) const {
        float distance = 0;
//        cout << "In findDistance" << endl;
        if (dimension == other.dimension) {
            void* ptr = aligned_alloc(32, DIMENSION*8);
            float* result = new(ptr) float[DIMENSION];
            sumArraysAVX(coordinates, other.coordinates, result, DIMENSION);
            for (size_t i = 0; i < DIMENSION; i++) {
                if (result[i] < 0) result[i] *= -1;
                distance += result[i];
            }
        }
        return distance;
    }
    long long int findDistance(const DataNode& other) const {
        float distance = 0;
//        cout << "In findDistance" << endl;
        if (dimension == other.dimension) {
            for (size_t i = 0; i < DIMENSION; i++) {
                float result = coordinates[i] - other.coordinates[i];
                if (result < 0) result *= -1;
                distance += result;
            }
        }
        return distance;
    }
    bool operator==(const DataNode& other) {
        if (dimension != other.dimension) return false;
        for (size_t ind = 0; ind < dimension; ind++) {
            if (coordinates[ind] != other.coordinates[ind]) return false;
        }
        return true;
    }
private:
    size_t dimension;
    float* coordinates;
};

struct Node {
    DataNode val;
    set<size_t> outEdge;
};

class Graph {
public:
    Graph(){}
    size_t findNode(const DataNode& val) {
        for (size_t i = 0; i < TOTAL; i++) {
            if (allNodes[i].val == val) {
                return i;
            }
        }
        return TOTAL;
    }
    void addNode(const DataNode& val, set<size_t>& neighbors, size_t pos) {
        Node newNode = Node();
        newNode.val = val;
        newNode.outEdge = neighbors;
        allNodes[pos] = newNode;
    }
    void randomize(int R) {
        for (size_t i = 0; i < TOTAL; i++) {
            set<size_t> neighbors = {};
            for (size_t j = 0; j < R; j++) {
                size_t random = rand() % TOTAL; // find a random node
                neighbors.insert(random);
            }
            allNodes[i].outEdge = neighbors;
        }
    }
    set<size_t> getNeighbors(const DataNode& i) {
        set<size_t> result;
        size_t thisNode = findNode(i);
        for (size_t neighbor : allNodes[thisNode].outEdge) {
            result.insert(neighbor);
        }
        return result;
    }
    void clearNeighbors(size_t i) {
        allNodes[i].outEdge = {};
    }
    long findDistance(size_t i, const DataNode& query) const {
        return allNodes[i].val.findDistance(query);
    }
    Node getNode(size_t i) const {
        return allNodes[i];
    }
    set<size_t> getNodeNeighbor(size_t i) const {
        return allNodes[i].outEdge;
    }
    void setEdge(size_t i, set<size_t> edges) {
        Node newNode = Node();
        newNode.val = allNodes[i].val;
        newNode.outEdge = edges;
        allNodes[i] = newNode;
    }
    void display() const {
        for (size_t i = 0; i < TOTAL; i++) {
            cout << i << " : ";
            for (size_t neighbor : allNodes[i].outEdge) {
                cout << neighbor << ' ';
            }
            cout << endl;
        }
    }
private:
    Node* allNodes = new Node[TOTAL];
};

void constructGraph(vector<DataNode>& allNodes, Graph& graph) {
    cout << "Constructing graph" << endl;
    size_t j = 0;
    for (DataNode& node : allNodes) {
        set<size_t> neighbors;
        graph.addNode(node, neighbors, j);
        j++;
        if (j % 100 == 0) {
            cout << j << " processed" << endl;
        }
    }
}

void randomEdges(Graph& graph, int R) {
    cout << "Randomizing edges" << endl;
    graph.randomize(R);
    cout << "Randomized edges" << endl;
}

template<typename T>
bool findInSet(const set<T>& set, T target) {
    for (T i : set) {
        if (i == target) {
            return true;
        }
    }
    return false;
}

template<typename Y>
set<Y> setDiff(const set<Y>& setOne, const set<Y>& setTwo) {
    set<Y> diff;
    for (Y i : setOne) {
        if (!findInSet(setTwo, i)) {
            diff.insert(i);
        }
    }
    return diff;
}

tuple<set<size_t>, set<size_t>> GreedySearch(Graph& graph, size_t start, const DataNode& query, size_t k, size_t l) {
    cout << "In GreedySearch, start " << start << endl;
    set<size_t> List = {};
    List.insert(start);
    set<size_t> Visited = {};
    while (setDiff(List, Visited).size() != 0) {
        set<size_t> diff = setDiff(List, Visited);
        size_t bestCandidate = *diff.begin();
        for (size_t i : diff) {
            if (graph.findDistance(i, query) < graph.findDistance(bestCandidate, query)) {
                bestCandidate = i;
            }
        }
        // update L and V
        Visited.insert(bestCandidate);
        for (size_t j : graph.getNodeNeighbor(bestCandidate)) {
            List.insert(j);
        }
        while (List.size() > l) {
            size_t worstCandidiate = *List.begin();
            for (size_t k : List) {
                if (graph.findDistance(k, query) > graph.findDistance(worstCandidiate, query)) {
                    worstCandidiate = k;
                }
            }
            for (size_t m : List) {
                if (m == worstCandidiate) {
                    List.erase(m);
                    break;
                }
            }
        }
    }
    return {List, Visited};
}

void RobustPrune(Graph& graph, size_t point, set<size_t>& candidates, long threshold, int R) {
    cout << "In RobustPrune, point " << point << endl;
    set<size_t> neighbors = graph.getNodeNeighbor(point);
    for (size_t i : neighbors) {
        candidates.insert(i);
    }
    graph.clearNeighbors(point);
    while (candidates.size() != 0) {
        size_t bestCandidate = *candidates.begin();
        for (size_t j : candidates) {
            if (graph.findDistance(j, graph.getNode(point).val) < graph.findDistance(bestCandidate, graph.getNode(point).val)) {
                bestCandidate = j;
            }
        }
        candidates.erase(bestCandidate);
        set<size_t> edges = graph.getNodeNeighbor(point);
        edges.insert(bestCandidate);
        graph.setEdge(point, edges);
        if (graph.getNodeNeighbor(point).size() == R) {
            break;
        }
        set<size_t> copy = candidates;
        for (size_t k : copy) {
            if (graph.findDistance(point, graph.getNode(k).val) > threshold * graph.findDistance(bestCandidate, graph.getNode(k).val)) {
                candidates.erase(k);
            }
        }
    }
}

Graph Vamana(vector<DataNode>& allNodes, long alpha, int L, int R) {
    Graph graph;
    cout << "Start of Vamana" << endl;
    constructGraph(allNodes, graph);
    randomEdges(graph, R);
    size_t s = allNodes.size() / 2;
    vector<size_t> sigma;
    for (size_t i = 0; i < allNodes.size(); i++) {
        sigma.push_back(i);
    }
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(sigma.begin(), sigma.end(), default_random_engine(seed));
    size_t count = 0;
    for (size_t i : sigma) {
        cout << "Num of node processed: " << count << endl;
        count++;
        tuple<set<size_t>, set<size_t>> tup = GreedySearch(graph, s, allNodes[i], 1, L);
        set<size_t> L = get<0>(tup);
        set<size_t> V = get<1>(tup);
        RobustPrune(graph, i, V, alpha, R);
        for (size_t j : graph.getNeighbors(allNodes[i])) {
            if (graph.getNode(j).outEdge.size() + 1 > R) {
                set<size_t> unionV = graph.getNode(j).outEdge;
                unionV.insert(graph.findNode(allNodes[i]));
                RobustPrune(graph, j, unionV, alpha, R);
            } else {
                set<size_t> edges = graph.getNodeNeighbor(j);
                edges.insert(i);
                graph.setEdge(j, edges);
            }
        }
    }
    cout << "End of Vamana" << endl;
    return graph;
}

void getNodes(vector<DataNode>& allNodes, const string& fileName, size_t dimension) {
    cout << "Getting nodes" << endl;
    fstream f;
    f.open(fileName);
    if (!f) {cout << "File not open" << endl;}
    int each;
    for (size_t i = 0; i < TOTAL; i++) {
        void* ptr = aligned_alloc(32, DIMENSION*8);
        float* coord = new(ptr) float[DIMENSION];
        for (size_t i = 0; i < DIMENSION; i++) {
            f >> each;
            coord[i] = each;
        }
        DataNode data = DataNode(coord);
        allNodes.push_back(data);
    }
    cout << "End of get nodes" << endl;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    vector<DataNode> allNodes = {};
    getNodes(allNodes, FILENAME, DIMENSION);
    Graph G = Vamana(allNodes, alpha, L, R);
    auto stop = std::chrono::high_resolution_clock::now();
    G.display();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "Duration: "<< duration.count()/1000000 << endl;
}
