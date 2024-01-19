import numpy as np

from wrapper import execute_on_best_pod


def matrix_op(matrix_size):
    result_matrix = np.dot(np.random.rand(matrix_size, matrix_size), np.random.rand(matrix_size, matrix_size))
    determinant = np.linalg.det(result_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(result_matrix)
    inverse_matrix = np.linalg.inv(result_matrix)
    return inverse_matrix


if __name__ == '__main__':
    m_size = 1000
    execute_on_best_pod(matrix_op, m_size)
