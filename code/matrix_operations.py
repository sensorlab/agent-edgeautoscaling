import ray

import numpy as np

ray.init("ray://10.1.33.199:10001")
print(ray.available_resources())


@ray.remote(num_cpus=4, resources={"node:10.1.245.2": 1.0})
def matrix_op(matrix_a, matrix_b):
    result_matrix = np.dot(matrix_a, matrix_b)
    determinant = np.linalg.det(result_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(result_matrix)
    inverse_matrix = np.linalg.inv(result_matrix)
    return inverse_matrix


matrix_size = 1000
matrix = np.random.rand(matrix_size, matrix_size)
ray.get(matrix_op.remote(matrix, matrix))

ray.shutdown()
