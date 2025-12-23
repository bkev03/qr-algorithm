import numpy as np

def householder(vector):
    """
    A function that constructs H-"hat" matrix from a vector.
    :param vector: the vector which the H-"hat" matrix will be constructed from
    :return: H-"hat" matrix
    """

    a = np.atleast_2d(vector).transpose()
    b = np.atleast_2d(np.zeros(len(a))).transpose()
    b[0, 0] = np.linalg.norm(a)

    h = a - b

    h_hat = np.eye(a.shape[0]) - ((2 / (np.linalg.norm(h) ** 2)) * (h @ h.T))

    return h_hat

def make_hessenberg(matrix):
    """
    A function that makes an upper hessenberg matrix similar to the matrix given.
    :param matrix: a square matrix that needs to be transformed into a hessenberg matrix (with similarity transformations)
    :return: a hessenberg matrix similar to the matrix given
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise Exception('Matrix must be square')
    elif not isinstance(matrix, np.ndarray):
        raise Exception('Parameter "matrix" must be a numpy array')

    A = matrix
    for j in range(A.shape[1] - 2):
        H = np.eye(A.shape[0])
        vector = A[j + 1:, j]
        if np.linalg.norm(vector) == 0:
            continue
        H[j + 1:, j + 1:] = householder(vector)
        A = H @ A @ H

    return A


def qr_algorithm(matrix, iterations=True):
    """
    A function that performs the QR algorithm on the matrix and calculates the eigenvalues of it.
    :param matrix: a matrix
    :param iterations: a boolean parameter which says if "constant iterations" or "margin of error" approach should be used,
                       it defaults to "True"
    :return: the list of eigenvalues of the given matrix in descending order and the iterations needed
    """

    if not isinstance(matrix, np.ndarray):
        raise Exception('Parameter "matrix" must be a numpy array')

    A = make_hessenberg(matrix)
    if iterations:
        for i in range(500):    # arbitrary iteration number
            Q,R = np.linalg.qr(A)
            A = np.dot(R, Q)
        iter_eigvals = np.diag(A).tolist()
        iter_eigvals.sort(reverse=True)
        return iter_eigvals, 500
    else:
        error = 1e-3   # the (arbitrary) margin of error we tolerate (exclusive)
        iters = 0
        while True:
            Q, R = np.linalg.qr(A)
            A = np.dot(R, Q)
            sum = 0
            for j in range(0, A.shape[1] - 1):
                sum += abs(A[j + 1, j])
            iters += 1
            if sum < error or iters > 50000:
                err_eigvals = np.diag(A).tolist()
                err_eigvals.sort(reverse=True)
                return err_eigvals, iters

