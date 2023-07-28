import sys
import numpy as np

def fvecs_to_vectors(filename):
    data = np.fromfile(filename, dtype=np.float32)
    dim = data.view(np.int32)[0]
    assert len(data) % (dim + 1) == 0
    num_vectors = len(data) // (dim + 1)

    vectors = []
    for i in range(num_vectors):
        vectors.append(data[(i * (dim + 1) + 1):(i + 1) * (dim + 1)])
    return vectors

def convert_base_file(filename):
    vectors = fvecs_to_vectors(filename)
    file = open(filename.split(".")[0] + ".txt", "w")
    for item in vectors:
        line = ""
        for num in item:
            line += str(num)
            line += ' '
        print(line, file=file)

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_vec_file.py <filename>")
        return
    convert_base_file(sys.argv[1])

if __name__ == '__main__':
    main()
