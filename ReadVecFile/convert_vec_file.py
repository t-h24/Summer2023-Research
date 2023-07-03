import numpy

def convert_base_file(fvecs_file_name):

    fv = numpy.fromfile(fvecs_file_name, dtype="int32")
    dim = fv.view(numpy.int32)[0]

    matrix = []

    for i in range(len(fv)):
        print(i)
        if i % (dim+1) == 0:
            vec = []
            for j in range(1, dim+1):
                vec.append(fv[i+j])
            matrix.append(vec)

    newFileName = fvecs_file_name[0:fvecs_file_name.find('.')] + '.txt'
    f = open(newFileName, 'w')

    for vec in matrix:
        line = ""
        for num in vec:
            line += str(num)
            line += ' '
        print(line, file = f)

def main():
    convert_base_file("siftsmall_query.fvecs")

main()
