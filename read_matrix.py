import numpy as np

def read_matrix(filename):
    with open(filename, 'r') as f:
        matrix = np.array([[x for x in line.split()] for line in f])
    # Replace 'i' with 'j' for Python's complex number format
    string_matrix = np.char.replace(matrix, 'i', 'j')

    # Convert the string matrix to a complex numpy matrix
    matrix = string_matrix.astype(np.complex128)
    return matrix
