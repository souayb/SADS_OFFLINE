
import numpy as np


def get_lm(cell, row, col, neg:bool=False):
    if neg:
        l1, l2 = -((cell % row)+1), (cell // row) % col
    else: 
        l1, l2 = (cell % row), (cell // row) % col
    return l1, l2

def face1_face2(arr1, arr2, mask:bool=False):
    if mask:
        result = np.ones((arr1.shape[0] + arr1.shape[0], arr1.shape[1]), dtype=int)
    else:
        result = np.zeros((arr1.shape[0] + arr1.shape[0], arr1.shape[1]), dtype=int)
        
    result[::2, :] = arr1
    result[1::2, :] = arr2
    return result


def sinusoid_duplicate(data, data_dup,  matrix_shape, target:str='anomaly'):
 
    rows, cols = matrix_shape
    matrix = np.zeros(matrix_shape, dtype=int)
    matrix_annot = np.zeros(matrix_shape, dtype=int)
    matrix_mask = np.ones(matrix_shape, dtype=int)

    for _, row in data.iterrows():

        dd = data_dup[ (data_dup['Barcode']== row['Barcode']) & (data_dup['Face'] == row['Face']) & \
                        (data_dup['Cell'] == row['Cell']) & \
                        (data_dup['Point'] == row['Point'])]
        
        
        l1, l2 = get_lm(row['Cell'] -1, rows, cols)
        if l2 % 2 != 0 and l2!=0:
            l1 = -(l1+1)
            # l1 += 1
        if dd['size'].values > 1:
            matrix[l1, l2] = row[target]
            matrix_annot[l1, l2] = row['Cell']
            matrix_mask[l1, l2] =  False
    return matrix, matrix_annot, matrix_mask

def sinusoid(data, matrix_shape, target:str='anomaly'):
 
    rows, cols = matrix_shape
    matrix = np.zeros(matrix_shape, dtype=int)
    matrix_annot = np.zeros(matrix_shape, dtype=int)
    matrix_mask = np.ones(matrix_shape, dtype=int)

    for cell, result in data[['Cell', target]].to_numpy():
        l1, l2 = get_lm(cell-1, rows, cols)
        if l2 % 2 != 0 and l2!=0:
            l1 = -(l1+1)
            # l1 += 1
        matrix[l1, l2] = result
        matrix_annot[l1, l2] = cell
        matrix_mask[l1, l2] = False
    return matrix, matrix_annot, matrix_mask

def get_face(pack_data_non_dup):
    face_1_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==1)]
    face_1_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==2)]
    face_2_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==1)]
    face_2_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==2)]
    return face_1_df_1,face_1_df_2,face_2_df_1,face_2_df_2
