#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;

const string base_path = "";
const string name = "";
const int num_nodes = 0;
const int num_training = 0;
const int num_queries = 0;
const int dim = 0;

void load_fvecs(const string& file, const string& type, float** nodes, int num, int dim) {
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
    if (num != f.tellg() / (dim * 4 + 4)) {
        cout << "Warning: requested number of " << type << " is different from number in file: "
            << num << " != " << f.tellg() / (dim * 4 + 4) << endl;
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

void calculate_stats(const string& type, float** nodes, int num, int dim, bool displayStats, bool exportStats, bool displayAggrStats) {
    // Calculate mean, median, and std of each dimension as well as min and max
    float* mean = new float[dim];
    float* median = new float[dim];
    float* std = new float[dim];

    float* min = new float[dim];
    float* max = new float[dim];

    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < num; j++) {
            sum += nodes[j][i];
        }
        mean[i] = sum / num;
    }

    for (int i = 0; i < dim; i++) {
        float* values = new float[num];
        for (int j = 0; j < num; j++) {
            values[j] = nodes[j][i];
        }
        sort(values, values + num);
        median[i] = values[num / 2];
        delete[] values;
    }

    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < num; j++) {
            sum += (nodes[j][i] - mean[i]) * (nodes[j][i] - mean[i]);
        }
        std[i] = sqrt(sum / num);
    }

    for (int i = 0; i < dim; i++) {
        float min_val = nodes[0][i];
        float max_val = nodes[0][i];
        for (int j = 1; j < num; j++) {
            if (nodes[j][i] < min_val) {
                min_val = nodes[j][i];
            }
            if (nodes[j][i] > max_val) {
                max_val = nodes[j][i];
            }
        }
        min[i] = min_val;
        max[i] = max_val;
    }

    if (displayStats) {
        cout << endl << "Mean: ";
        for (int i = 0; i < dim; i++) {
            cout << mean[i] << " ";
        }
        cout << endl;

        cout << endl << "Median: ";
        for (int i = 0; i < dim; i++) {
            cout << median[i] << " ";
        }
        cout << endl;

        cout << endl << "Std: ";
        for (int i = 0; i < dim; i++) {
            cout << std[i] << " ";
        }
        cout << endl;

        cout << endl << "Min: ";
        for (int i = 0; i < dim; i++) {
            cout << min[i] << " ";
        }
        cout << endl;

        cout << endl << "Max: ";
        for (int i = 0; i < dim; i++) {
            cout << max[i] << " ";
        }
        cout << endl;
    }

    if (exportStats) {
        ofstream f("exports/stats_" + type + ".txt");
        for (int i = 0; i < dim; i++) {
            f << mean[i] << " ";
        }
        f << endl;

        for (int i = 0; i < dim; i++) {
            f << median[i] << " ";
        }
        f << endl;

        for (int i = 0; i < dim; i++) {
            f << std[i] << " ";
        }
        f << endl;

        for (int i = 0; i < dim; i++) {
            f << min[i] << " ";
        }
        f << endl;

        for (int i = 0; i < dim; i++) {
            f << max[i] << " ";
        }
        f << endl;
        f.close();
    }

    if (displayAggrStats) {
        // Sort mean, median, std, min, and max and display the top 10 and bottom 10 values
        sort(mean, mean + dim);
        sort(median, median + dim);
        sort(std, std + dim);
        sort(min, min + dim);
        sort(max, max + dim);

        cout << endl << "Top 10 mean: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
            cout << mean[i] << " ";
        }

        cout << endl << "Bottom 10 mean: ";
        for (int i = 0; i < 10; i++) {
            cout << mean[i] << " ";
        }

        cout << endl << "Top 10 median: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
            cout << median[i] << " ";
        }

        cout << endl << "Bottom 10 median: ";
        for (int i = 0; i < 10; i++) {
            cout << median[i] << " ";
        }

        cout << endl << "Top 10 std: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
            cout << std[i] << " ";
        }

        cout << endl << "Bottom 10 std: ";
        for (int i = 0; i < 10; i++) {
            cout << std[i] << " ";
        }

        cout << endl << "Top 10 min: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
            cout << min[i] << " ";
        }

        cout << endl << "Bottom 10 min: ";
        for (int i = 0; i < 10; i++) {
            cout << min[i] << " ";
        }

        cout << endl << "Top 10 max: ";
        for (int i = dim - 1; i >= dim - 10; i--) {
            cout << max[i] << " ";
        }

        cout << endl << "Bottom 10 max: ";
        for (int i = 0; i < 10; i++) {
            cout << max[i] << " ";
        }

        cout << endl;
    }

    delete[] mean;
    delete[] median;
    delete[] std;
    delete[] min;
    delete[] max;
}

int main() {
    bool displayStats = false;
    bool exportStats = true;
    bool displayAggrStats = true;

    float** nodes = new float*[num_nodes];
    float** training_nodes = new float*[num_training];
    float** query_nodes = new float*[num_queries];

    load_fvecs(base_path + name + "/" + name + "_base.fvecs", "base", nodes, num_nodes, dim);
    load_fvecs(base_path + name + "/" + name + "_learn.fvecs", "learn", training_nodes, num_training, dim);
    load_fvecs(base_path + name + "/" + name + "_query.fvecs", "query", query_nodes, num_queries, dim);

    // Calculate stats
    cout << "Base nodes:" << endl;
    calculate_stats(name + "_base", nodes, num_nodes, dim, displayStats, exportStats, displayAggrStats);
    cout << "Training nodes:" << endl;
    calculate_stats(name + "_learn", training_nodes, num_training, dim, displayStats, exportStats, displayAggrStats);
    cout << "Queries:" << endl;
    calculate_stats(name + "_query", query_nodes, num_queries, dim, displayStats, exportStats, displayAggrStats);

    // Delete objects
    for (int i = 0; i < num_nodes; i++) {
        delete[] nodes[i];
    }
    delete[] nodes;

    for (int i = 0; i < num_training; i++) {
        delete[] training_nodes[i];
    }
    delete[] training_nodes;

    for (int i = 0; i < num_queries; i++) {
        delete[] query_nodes[i];
    }
    delete[] query_nodes;
    return 0;
}
