#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <queue>
#include <immintrin.h>
using namespace std;

/// reduce K_QUERY to 1, 50, etc -> find out what's the issue
/// construction L and query L are different -> try diff vals
/// L should not be related to K -> try to find a good number (for both L)
/// first fix construction L to a good number and then for query L
/// try max outedge != K -> check what they do in paper
/// 2 round vamana
/// try different / larger dataset

int distanceCalculationCount = 0;
int TOTAL = 1000000;
int QUERY_TOTAL = 100000;
int alpha = 1.2;
int K = 30; // Num of NNs when building Vamana graph
int K_QUERY = 100; // Num of NNs found for each query
int K_TRUTH = 100; // Num of NNs provided by ground truth for each query
int R = 50; // Max outedge
int L = 100; // beam search width
int L_QUERY = 100;
size_t DIMENSION = 128;
string FILENAME = "../../hdd1/home/sw4293/sift_base.txt";
string QUERY_FILE = "../../hdd1/home/sw4293/sift_query.txt";
string GROUND_TRUTH = "../../hdd1/home/sw4293/sift_groundtruth.txt";
//string FILENAME = "glove.6B.50d.txt";
//string QUERY_FILE = "sift_query.txt";
//string GROUND_TRUTH = "sift_groundtruth.txt";
//string FILENAME = "sifttest_base.txt";

class DataNode {
    friend ostream& operator<<(ostream& os, const DataNode& rhs) {
        for (size_t i = 0; i < DIMENSION; i++) {
            os << rhs.coordinates[i] << ' ';
        }
        os << endl;
        return os;
    }
    friend bool operator==(const DataNode& lhs, const DataNode& rhs) {
        if (lhs.dimension != rhs.dimension) return false;
        for (size_t ind = 0; ind < lhs.dimension; ind++) {
            if (lhs.coordinates[ind] != rhs.coordinates[ind]) return false;
        }
        return true;
    }
public:
    DataNode();
    DataNode(double* coord);
//    void sumArraysAVX(float* array1, float* array2, float* result, int size) const;
    long long int findDistanceAVX(const DataNode& other) const;
    double findDistance(const DataNode& other) const;
    bool compare(double* coord) const;
    void addCoord(double* coord) const;
    void setWord(const string& theWord);
private:
    size_t dimension;
    double* coordinates;
    string word;
};

struct Node {
    DataNode val;
    set<size_t> outEdge;
};

class Graph {
public:
    Graph();
    size_t findNode(const DataNode& val);
    void addNode(const DataNode& val, set<size_t>& neighbors, size_t pos);
    void randomize(int R);
    set<size_t> getNeighbors(const DataNode& i);
    void clearNeighbors(size_t i);
    double findDistance(size_t i, const DataNode& query) const;
    Node getNode(size_t i) const;
    set<size_t> getNodeNeighbor(size_t i) const;
    void setEdge(size_t i, set<size_t> edges);
    void display() const;
    vector<vector<size_t>> query(size_t start);
    void queryBruteForce(size_t start);
    void sanityCheck(vector<vector<size_t>> allResults) const;
    void queryTest(size_t start);
private:
    Node* allNodes = new Node[TOTAL];
};

void constructGraph(vector<DataNode>& allNodes, Graph& graph);
void randomEdges(Graph& graph, int R);
template<typename T>
bool findInSet(const set<T>& set, T target);
template<typename Y>
set<Y> setDiff(const set<Y>& setOne, const set<Y>& setTwo);
vector<size_t> GreedySearch(Graph& graph, size_t start, const DataNode& query, size_t L);
void RobustPrune(Graph& graph, size_t point, vector<size_t>& candidates, long threshold, int R);
Graph Vamana(vector<DataNode>& allNodes, long alpha, int L, int R);
void getNodes(vector<DataNode>& allNodes, const string& fileName, size_t dimension);
void getNodesGlove(vector<DataNode>& allNodes, const string& fileName, size_t dimension);
size_t findStart(const Graph& g);

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    vector<DataNode> allNodes = {};
//    getNodes(allNodes, FILENAME, DIMENSION);
    getNodesGlove(allNodes, FILENAME, DIMENSION);
    Graph G = Vamana(allNodes, alpha, K, R);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    G.display();
    size_t s = findStart(G);
    G.queryTest(s);
//    start = std::chrono::high_resolution_clock::now();
//    G.queryBruteForce(s);
//    distanceCalculationCount = 0;
//    vector<vector<size_t>> allResults = G.query(s);
//    stop = std::chrono::high_resolution_clock::now();
//    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
////    G.sanityCheck(allResults);
//    cout << "Duration of Vamana: "<< duration.count()/1000000 << " second(s)" << endl;
//    cout << "Duration of Each Query: "<< duration2.count()/1000000/QUERY_TOTAL << " second(s)"<< endl;
//    cout << "Number of distance calculation per query: " << distanceCalculationCount/QUERY_TOTAL << endl;
}

DataNode::DataNode() {}
DataNode::DataNode(double* coord) {
    dimension = DIMENSION;
    coordinates = coord;
}
void DataNode::setWord(const string& theWord) {
    word = theWord;
}
//void DataNode::sumArraysAVX(int* array1, int* array2, int* result, int size) const {
////        cout << "In sumArraysAVX" << endl;
//    int vectorSize = sizeof(__m256) / sizeof(float);
//    for (int i = 0; i < size; i += vectorSize) {
//        __m256 vec1 = _mm256_load_ps(&array1[i]);
//        __m256 vec2 = _mm256_load_ps(&array2[i]);
//        __m256 sub = _mm256_sub_ps(vec1, vec2);
//        _mm256_store_ps(&result[i], sub);
//    }
//}
//long long int DataNode::findDistanceAVX(const DataNode& other) const {
//    float distance = 0;
////        cout << "In findDistance" << endl;
//    if (dimension == other.dimension) {
//        void* ptr = aligned_alloc(32, DIMENSION*8);
//        float* result = new(ptr) float[DIMENSION];
//        sumArraysAVX(coordinates, other.coordinates, result, DIMENSION);
//        for (size_t i = 0; i < DIMENSION; i++) {
//            if (result[i] < 0) result[i] *= -1;
//            distance += result[i];
//        }
//    }
//    return distance;
//}


double DataNode::findDistance(const DataNode& other) const {
    distanceCalculationCount++;
    double distance = 0;
//    cout << "In findDistance ";
    if (dimension == other.dimension) {
        for (size_t i = 0; i < DIMENSION; i++) {
            double result = coordinates[i] - other.coordinates[i];
            result *= result;
            distance += result;
        }
    }
    return sqrt(distance);
}

bool DataNode::compare(double* coord) const {
    for (size_t i = 0; i < DIMENSION; i++) {
        if (coord[i] != coordinates[i]) return false;
    }
    return true;
}

void DataNode::addCoord(double* coord) const {
    for (size_t i = 0; i < DIMENSION; i++) {
        coord[i] += coordinates[i];
    }
}

Graph::Graph(){}
size_t Graph::findNode(const DataNode& val) {
    for (size_t i = 0; i < TOTAL; i++) {
        if (allNodes[i].val == val) {
            return i;
        }
    }
    return TOTAL;
}
void Graph::addNode(const DataNode& val, set<size_t>& neighbors, size_t pos) {
    Node newNode = Node();
    newNode.val = val;
    newNode.outEdge = neighbors;
    allNodes[pos] = newNode;
}
void Graph::randomize(int R) {
    for (size_t i = 0; i < TOTAL; i++) {
        set<size_t> neighbors = {};
        for (size_t j = 0; j < R; j++) {
            size_t random = rand() % TOTAL; // find a random node
            while (random == i) random = rand() % TOTAL;
            neighbors.insert(random);
        }
        setEdge(i, neighbors);
    }
}
set<size_t> Graph::getNeighbors(const DataNode& i) {
    set<size_t> result;
    size_t thisNode = findNode(i);
    for (size_t neighbor : allNodes[thisNode].outEdge) {
        result.insert(neighbor);
    }
    return result;
}
void Graph::clearNeighbors(size_t i) {
    allNodes[i].outEdge = {};
}
double Graph::findDistance(size_t i, const DataNode& query) const {
    //return allNodes[i].val.findDistanceAVX(query);
    return allNodes[i].val.findDistance(query);
}
Node Graph::getNode(size_t i) const {
    return allNodes[i];
}
set<size_t> Graph::getNodeNeighbor(size_t i) const {
    return allNodes[i].outEdge;
}
void Graph::setEdge(size_t i, set<size_t> edges) {
    Node newNode = Node();
    newNode.val = allNodes[i].val;
    newNode.outEdge = edges;
    allNodes[i] = newNode;
}
void Graph::display() const {
    for (size_t i = 0; i < TOTAL; i++) {
        cout << i << " : ";
        for (size_t neighbor : allNodes[i].outEdge) {
            cout << neighbor << ' ';
        }
        cout << endl;
    }
}

void Graph::sanityCheck(vector<vector<size_t>> allResults) const {
    // vector<set<size_t>> gives all NNs
    fstream groundTruth;
    groundTruth.open(GROUND_TRUTH);
    if (!groundTruth) {cout << "Ground truth file not open" << endl;}
    int each;
    int totalCorrect = 0;
    float result;
    for (size_t j = 0; j < QUERY_TOTAL; j++) {
        int correct = 0;
        vector<size_t> allTruths = {};
//        cout << "Ground truths: ";
        for (size_t i = 0; i < K_TRUTH; i++) {
            groundTruth >> each;
            allTruths.push_back(each);
//            cout << each << ' ';
        }
//        cout << endl;
        vector<size_t> eachResult = {};
        for (int count = 0; count < K_QUERY; count++) {
            for (size_t ea : allResults[j]) {
                if (allTruths[count] == ea) correct++;
            }
        }
        result = correct * 100 / K_QUERY;
        totalCorrect += result;
        cout << "Found " << result << "% among " << K_QUERY << " closest neighbors" << endl;
    }
    result = totalCorrect / QUERY_TOTAL;
    cout << "Average correctness: " << result << '%' << endl;
}

vector<vector<size_t>> Graph::query(size_t start) {
    fstream f;
    f.open(QUERY_FILE);
    if (!f) {cout << "Query file not open" << endl;}
    DataNode* queries = new DataNode[QUERY_TOTAL];
    double each;
    for (size_t j = 0; j < QUERY_TOTAL; j++) {
        //void* ptr = aligned_alloc(32, DIMENSION*8);
        //float* coord = new(ptr) float[DIMENSION];
        double* coord = new double[DIMENSION];
        for (size_t i = 0; i < DIMENSION; i++) {
            f >> each;
            coord[i] = each;
        }
        DataNode data = DataNode(coord);
        queries[j] = data;
    }
    cout << "All queries read" << endl;
    vector<vector<size_t>> allResults = {};
    for (size_t k = 0; k < QUERY_TOTAL; k++) {
        cout << "Processing " << k+1 << endl;
        DataNode thisQuery = queries[k];
        auto startTime = std::chrono::high_resolution_clock::now();
        vector<size_t> result = GreedySearch(*this, start, thisQuery, L_QUERY);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        cout << "Duration of GreedySearch: "<< duration.count()/1000000 << " second(s)" << endl;
        allResults.push_back(result);
    }
    cout << "All queries processed" << endl;
    return allResults;
}


void Graph::queryTest(size_t start) {
    vector<Node*> queryNodes = {};
    int queryCount = 0;
    size_t correct = 0;
    while(queryCount < 100) {
        size_t random = rand() % TOTAL;
        queryNodes.push_back(&allNodes[random]);
        queryCount++;
    }
    for (Node* each : queryNodes) {
        DataNode dataNode = each->val;
        auto startTime = std::chrono::high_resolution_clock::now();
        vector<size_t> result = GreedySearch(*this, start, dataNode, L_QUERY);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        cout << "Duration of GreedySearch: "<< duration.count()/1000000 << " second(s)" << endl;
        size_t closestNode = 0;
        double shortestDistance = findDistance(0, dataNode);
        for (size_t i = 0; i < TOTAL; i++) {
            double distance = findDistance(i, dataNode);
            if (distance < shortestDistance) {
                closestNode = i;
                shortestDistance = distance;
            }
        }
        cout << closestNode << ' ' << result[0] << endl;
        for (auto i : result) {
            if (i == closestNode) correct++;
        }
    }
    cout << "Total correct number: " << correct << endl;
}

void Graph::queryBruteForce(size_t start){
    fstream f;
    f.open(QUERY_FILE);
    if (!f) {cout << "Query file not open" << endl;}
    DataNode* queries = new DataNode[QUERY_TOTAL];
    double each;
    for (size_t j = 0; j < QUERY_TOTAL; j++) {
        //void* ptr = aligned_alloc(32, DIMENSION*8);
        //float* coord = new(ptr) float[DIMENSION];
        double* coord = new double[DIMENSION];
        for (size_t i = 0; i < DIMENSION; i++) {
            f >> each;
            coord[i] = each;
        }
        DataNode data = DataNode(coord);
        queries[j] = data;
    }
    cout << "All queries read" << endl;
    fstream groundTruth;
    groundTruth.open(GROUND_TRUTH);
    if (!groundTruth) {cout << "Ground truth file not open" << endl;}
    int totalCorrect = 0;
    float result;
    for (size_t j = 0; j < QUERY_TOTAL; j++) {
        vector<size_t> allTruths = {};
        for (size_t i = 0; i < K_TRUTH; i++) {
            groundTruth >> each;
            allTruths.push_back(each);
        }
        DataNode query = queries[j];
        size_t closest = 0;
        double closestDist = findDistance(closest, query);
        for (size_t k = 0; k < TOTAL; k++) {
            double newDist = findDistance(k, query);
            if (newDist < closestDist) {
                closest = k;
                closestDist = newDist;
            }
        }
        if (allTruths[0] == closest) {
            totalCorrect++;
        } else {
            cout << allTruths[0] << ' ' << closest << endl;
        }
    }
    result = totalCorrect;
    cout << "Average correctness: " << result << '%' << endl;
}

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

vector<size_t> GreedySearch(Graph& graph, size_t start, const DataNode& query, size_t L) {

/// L, V, diff between L and V
/// L -> priority queue with distance and index
/// V -> vector with index
/// diff -> priority queue with distance and index -> 
    
    vector<size_t> result;
    priority_queue<tuple<double, size_t>> List; // max priority queue
    set<size_t> ListSet = {};
    double distance = graph.findDistance(start, query);
    List.push({distance, start}); // L <- {s}
    ListSet.insert(start);
    vector<size_t> Visited = {};
    priority_queue<tuple<double, size_t>> diff; // min priority queue
    diff.push({-1 * distance, start});
    while (diff.size() != 0) {
        tuple<double, size_t> top = diff.top(); // get the best candidate
        Visited.push_back(get<1>(top));
        for (size_t j : graph.getNodeNeighbor(get<1>(top))) {
            double dist = graph.findDistance(j, query);
            tuple<double, size_t> newNode = {dist, j};
            bool inserted = false;
            for (size_t i : ListSet) {
                if (i == j) {inserted = true;}
            }
            if (!inserted) List.push(newNode);
            ListSet.insert(j);
        }
        while (List.size() > L) List.pop();
        priority_queue<tuple<double, size_t>> copy = List;
        diff = {};
        while (copy.size() != 0) {
            tuple<double, size_t> next = copy.top();
            copy.pop();
            bool exists = false;
            for (size_t k : Visited) {
                if (k == get<1>(next)) exists = true;
            }
            if (!exists) diff.push({-1 * get<0>(next), get<1>(next)});
        }
    }
    while (List.size() != 0) {
        result.push_back(get<1>(List.top()));
        List.pop();
    }
    return result;
}

void RobustPrune(Graph& graph, size_t point, vector<size_t>& candidates, long threshold, int R) {
//    cout << "In RobustPrune, point " << point << endl;
    set<size_t> neighbors = graph.getNodeNeighbor(point);
    for (size_t i : neighbors) {
        candidates.push_back(i);
    }
    for (size_t j = 0; j < candidates.size(); j++) {
        if (candidates[j] == point) {
            candidates[j] = candidates[candidates.size()-1];
            candidates.pop_back();
            break;
        }
    }
    graph.clearNeighbors(point);
    while (candidates.size() != 0) {
        // find p* <- closest neighbor to p
        size_t bestCandidate = *candidates.begin();
        for (size_t j : candidates) {
            if (graph.findDistance(j, graph.getNode(point).val) < graph.findDistance(bestCandidate, graph.getNode(point).val)) {
                bestCandidate = j;
            }
        }
        for (size_t j = 0; j < candidates.size(); j++) {
            if (candidates[j] == bestCandidate) {
                candidates[j] = candidates[candidates.size()-1];
                candidates.pop_back();
                break;
            }
        }
        // add best candidate back to p's neighborhood
        set<size_t> edges = graph.getNodeNeighbor(point);
        edges.insert(bestCandidate);
        graph.setEdge(point, edges);
        // neighborhood is full
        if (graph.getNodeNeighbor(point).size() == R) {
            break;
        }
        vector<size_t> copy;
        for (size_t k : candidates) {
            if (graph.findDistance(point, graph.getNode(k).val) < threshold * graph.findDistance(bestCandidate, graph.getNode(k).val)) {
                copy.push_back(k);
            }
        }
        candidates = copy;
    }
}

size_t findStart(const Graph& g) {
    double* coord = new double[DIMENSION];
    for (size_t j = 0; j < TOTAL; j++) {
        g.getNode(j).val.addCoord(coord);
    }
    for (size_t i = 0; i < DIMENSION; i++) {
        coord[i] /= TOTAL;
    }
    DataNode center = DataNode(coord);
    size_t closest = 0;
    float closest_dist = MAXFLOAT;
    for (size_t m = 0; m < TOTAL; m++) {
        float this_dist = g.findDistance(m, center);
        if (this_dist < closest_dist) {
            closest_dist = this_dist;
            closest = m;
        }
    }
    return closest;
}

Graph Vamana(vector<DataNode>& allNodes, long alpha, int L, int R) {
    Graph graph;
    cout << "Start of Vamana" << endl;
    constructGraph(allNodes, graph);
    randomEdges(graph, R);
    cout << "Random graph: " << endl;
//    graph.display();
    size_t s = findStart(graph);
    cout << "The centroid is #" << s << endl;
    vector<size_t> sigma;
    for (size_t i = 0; i < allNodes.size(); i++) {
        sigma.push_back(i);
    }
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(sigma.begin(), sigma.end(), default_random_engine(seed));
    size_t count = 0;
    for (size_t i : sigma) {
        if (count % 100 == 0) cout << "Num of node processed: " << count << endl;
        count++;
        vector<size_t> result = GreedySearch(graph, s, allNodes[i], L);
        RobustPrune(graph, i, result, alpha, R);
        set<size_t> neighbors = graph.getNeighbors(allNodes[i]);
        for (size_t j : neighbors) {
            set<size_t> unionV = graph.getNode(j).outEdge;
            vector<size_t> unionVec;
            for (size_t i : unionV) {
                unionVec.push_back(i);
            }
            unionV.insert(i);
            if (unionV.size() > R) {
                RobustPrune(graph, j, unionVec, alpha, R);
            } else {
                graph.setEdge(j, unionV);
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
    if (!f) {cout << "Base file not open" << endl;}
    double each;
    for (size_t j = 0; j < TOTAL; j++) {
        //void* ptr = aligned_alloc(32, DIMENSION*8);
        //float* coord = new(ptr) float[DIMENSION];
        double* coord = new double[DIMENSION];
        for (size_t i = 0; i < DIMENSION; i++) {
            f >> each;
            coord[i] = each;
        }
        DataNode data = DataNode(coord);
        allNodes.push_back(data);
    }
    cout << "End of get nodes" << endl;
}

void getNodesGlove(vector<DataNode>& allNodes, const string& fileName, size_t dimension) {
    cout << "Getting nodes" << endl;
    fstream f;
    f.open(fileName);
    if (!f) {cout << "Base file not open" << endl;}
    double each;
    for (size_t j = 0; j < TOTAL; j++) {
        //void* ptr = aligned_alloc(32, DIMENSION*8);
        //float* coord = new(ptr) float[DIMENSION];
        string str;
        f >> str; 
        double* coord = new double[DIMENSION];
        for (size_t i = 0; i < DIMENSION; i++) {
            f >> each;
            coord[i] = each;
        }
        DataNode data = DataNode(coord);
        data.setWord(str);
        allNodes.push_back(data);
    }
    cout << "End of get nodes" << endl;
}
