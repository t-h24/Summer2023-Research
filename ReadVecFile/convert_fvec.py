import sys
import numpy as np

def fvecs_to_vectors(filename):
    data = np.memmap(filename, dtype=np.int32, mode='r')
    dim = int(data[0])
    assert len(data) % (dim + 1) == 0
    num_vectors = len(data) // (dim + 1)

    # Extract the vectors
    vectors = np.zeros((num_vectors, dim), dtype=np.int32)
    for i in range(num_vectors):
        vectors[i] = data[(i * (dim + 1) + 1):(i + 1) * (dim + 1)]

    return vectors

def convert_base_file(filename):
    vectors = fvecs_to_vectors(filename)
    np.savetxt(filename.split(".")[0] + ".txt", vectors, fmt='%d')

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_fvec.py <filename>")
        return
    convert_base_file(sys.argv[1])

if __name__ == '__main__':
    main()
