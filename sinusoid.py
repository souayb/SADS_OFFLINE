import numpy as np
from itertools import count 

def get_lm(cell, row, col, neg:bool=False):
    if neg:
        l1, l2 = -((cell % row)+1), (cell // row) % col
    else: 
        l1, l2 = (cell % row), (cell // row) % col
    return l1, l2
def face1_face2(arr1, arr2):
    result = np.zeros((arr1.shape[0] + arr1.shape[0], arr1.shape[1]), dtype=int)
    result[::2, :] = arr1
    result[1::2, :] = arr2
    return result

def sinusoid(data, matrix_shape):
    data = np.sort(data)
    rows, cols = matrix_shape
    matrix = np.zeros(matrix_shape, dtype=int)
    for cell in data:
        l1, l2 = get_lm(cell-1, rows, cols)
        if l2 % 2 != 0 and l2!=0:
            l1 = -(l1+1)
            # l1 += 1
        matrix[l1, l2] = cell
    return matrix
 


# test_matrix = np.array([[112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99,  98,  97],
#                         [ 96,  95,  94,  93,  92,  91,  90,  89,  88,  87,  86,  85,  84, 83,  82,  81],
#                         [ 80,  79,  78,  77,  76,  75,  74,  73,  72,  71,  70,  69,  68, 67,  66,  65],
#                         [ 64,  63,  62,  61,  60,  59,  58,  57,  56,  55,  54,  53,  52, 51,  50,  49],
#                         [ 48,  47,  46,  45,  44,  43,  42,  41,  40,  39,  38,  37,  36, 35,  34,  33],
#                         [ 32,  31,  30,  29,  28,  27,  26,  25,  24,  23,  22,  21,  20, 19,  18,  17],
#                         [ 16,  15,  14,  13,  12,  11,  10,   9,   8,   7,   6,   5,   4, 3,   2,   1]])
test_matrix = np.arange(1, 11)
test_matrix = test_matrix.flatten()
matrix_shape = (2, 5)
result_matrix = sinusoid(test_matrix, matrix_shape)
print(result_matrix)
add_matrix = face1_face2(result_matrix, result_matrix)

print(add_matrix)
