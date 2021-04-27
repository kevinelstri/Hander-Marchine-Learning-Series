import numpy as np

'''
Author:kevinelstri
Date:2021/04/27
Desc:Numpy矩阵运算
'''
#####################################################################################
# 创建矩阵对象,list,np.array,np.matrix区别
L = [[1, 2, 3], [4, 5, 6]]  # list
arr = np.array([[1, 2, 3], [4, 5, 6]])  # np.array
mat = np.matrix([[1, 2, 3], [4, 5, 6]])  # np.matrix

# list,array,matrix可以相互转换，可根据实际需要进行变换
L_arr = np.array(L)  # list to array
arr_L = list(arr)  # array to list
arr_mat = np.matrix(arr)  # array to matrix
mat_arr = np.array(mat)  # matrix to array
L_mat = np.matrix(L)  # list to matrix
mat_L = list(mat)  # matrix to list

# list,array,matrix形状(长度，行数，列数)
lenList = len(L)
arrShape = np.shape(arr)
matShape = np.shape(mat)

print(f'list长度：\n{lenList}\n array行列：\n{arrShape}\n matrix行列：\n{matShape}\n')
print(f'list:\n{L} \n array:\n{arr} \n matrix:\n{mat} \n')
print(f'list_to_array:\n{L_arr} \n array_to_list:\n{arr_L} \n ')
print(f'array_to_matrix:\n{arr_mat} \n matrix_array:\n{mat_arr} \n')
print(f'list_to_matrix:\n{L_mat} \n matrix_to_list:\n{mat_L}\n')
#####################################################################################
# 修改list,array,matrix元素值
# 注意：list中需要使用L[i][j], array和matrix中需要使用array[row,col],matrix[row,col]
L[1][2] = 10
arr[1, 2] = 10
mat[1, 2] = 20
print(f'newlist:\n{L}\n newarr:\n{arr}\n newmat:\n{mat}\n')
#####################################################################################
# 矩阵转置：list,array,matrix
print(list(zip(*L)))  # zip()用于转置list，注意转置之后的结果为元组list
L_T = [list(x) for x in list(zip(*L))]
arr_T = arr.T
mat_T = mat.T
print(f'list转置：\n{L_T}\n arr转置：\n{arr_T}\n matrix转置：\n{mat_T}\n')
#####################################################################################
# 矩阵加减运算
# 注意：list直接相加表示拼接
L_splic = L + L  # 两个list拼接
arr_sum = arr + arr
arr_sub = arr - arr
mat_sum = mat + mat
mat_sub = mat - mat
print(f'list拼接：\n{L_splic}\n array相加：\n{arr_sum}\n array相减：\n{arr_sub}\n matrix相加：\n{mat_sum}\n matrix相减：\n{mat_sub}')
#####################################################################################
# 矩阵乘法
# 乘以标量n
L_mul = L * 2  # list乘以标量n，表示list元素重复n次
arr_mul = arr * 2  # array乘以标量n，表示array中每个元素都乘以n
mat_mul = mat * 2  # matrix乘以标量n，表示matrix中每个元素都乘以n
print(f'list重复n次：\n{L_mul}\n arr乘以n：\n{arr_mul}\n mat乘以n：\n{mat_mul}\n')
# 矩阵乘积：矩阵中对应元素相乘
arr_two_mul = np.multiply(arr, arr)
mat_two_mul = np.multiply(mat, mat)
print(f'array矩阵乘法：\n{arr_two_mul}\nmatrix矩阵乘法：\n{mat_two_mul}\n')
# 矩阵乘法：按照矩阵行列规则进行乘积
# array dot
arr1 = np.arange(4).reshape((2, 2))
arr2 = np.arange(6).reshape((2, 3))
arr_dot_1 = np.dot(arr1, arr2)
arr_dot_2 = arr1.dot(arr2)
arr_dot_3 = np.matmul(arr1, arr2)
arr_dot_4 = arr1 @ arr2
print('array dot:')
print(arr_dot_1 == arr_dot_2)
print(arr_dot_2 == arr_dot_3)
print(arr_dot_3 == arr_dot_4)
# matrix dot
mat1 = np.matrix(arr1)
mat2 = np.matrix(arr2)
mat_dot_1 = np.dot(mat1, mat2)
mat_dot_2 = mat1.dot(mat2)
mat_dot_3 = np.matmul(mat1, mat2)
mat_dot_4 = mat1 @ mat2
mat_dot_5 = mat1 * mat2
print('matrix dot:')
print(mat_dot_1 == mat_dot_2)
print(mat_dot_2 == mat_dot_3)
print(mat_dot_3 == mat_dot_4)
print(mat_dot_4 == mat_dot_5)
#####################################################################################
# 幂运算 **
# 注意：array的幂运算是对每个元素进行幂运算
#      matrix的幂运算是多个matrix的矩阵乘积
arr = np.arange(4).reshape((2, 2))
arr_mi = arr ** 2
mat = np.matrix(arr)
mat_mi = mat ** 2
print(f'array幂运算：\n{arr_mi}\n matrix幂运算：\n{mat_mi}\n')
print(mat ** 2 == mat * mat)
print(mat ** 3 == mat * mat * mat)
#####################################################################################
# -1 幂运算
# 注意：array的-1幂运算是每个元素的-1次方，前提是array元素必须为float类型，若为int类型，则报错
#      matrix的-1幂运算是matrix矩阵的逆矩阵运算
# 注意区分以下两种array生成的区别及-1幂运算的不同(-1幂运算只支持float类型，不支持int类型)
arr_int = np.arange(1, 5).reshape((2, 2))  # 此时生成的array为int类型
arr_float = np.array(arr_int, dtype=float)
print(arr_float)
print(arr_float ** -1)

arr_auto_float = np.arange(1, 5, 0.1).reshape((5, 8))  # 此时生成的array为float类型
print(arr_auto_float ** -1)

# matrix的-1幂运算是matrix矩阵的逆矩阵运算
mat = np.matrix(arr)
print(mat ** -1)
print(mat ** -2)
print(mat ** -2 == mat ** -1 * mat ** -1)  # True
print(mat ** -3 == mat ** -1 * mat ** -1 * mat ** -1)  # True
#####################################################################################
# 逆矩阵 numpy.linalg.inv(),**-1,.I
# 注意：array的逆矩阵智能使用np.linalg.inv()
#      matrix的逆矩阵可以使用三种方式来获得
arr = np.array([[2, 5], [1, 3]])
arr_inv = np.linalg.inv(arr)
print(f'arr的逆矩阵：\n{arr_inv}\n')

mat = np.matrix(arr)
mat_inv_1 = mat ** -1
mat_inv_2 = np.linalg.inv(mat)
mat_inv_3 = mat.I
print(mat_inv_1 == mat_inv_2)
print(mat_inv_2 == mat_inv_3)
print(f'matrix的逆矩阵：\n{mat_inv_3}\n')

# 伪逆矩阵：对于逆矩阵不存在的矩阵，可以计算伪逆矩阵，伪逆矩阵的伪逆矩阵 = 原矩阵
arr_p = np.array([[0, 0], [1, 3]])
arr_pinv = np.linalg.pinv(arr_p)
print(f'伪逆矩阵：\n{arr_pinv}\n')
#####################################################################################
# 行列式：np.linalg.det
# 注意：行列式是一个值，矩阵是很多值的一个集合
arr = np.array([[1, 2], [3, 4]])
arr_det = np.linalg.det(arr)
print(f'行列式的值：\n{arr_det}\n')
