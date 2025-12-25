from qr import *
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=1000, threshold=1000)

    # GENERATING MATRICES

    real_matrices = []
    int_matrices = []
    int_matrices_100_1000 = []

    # real matrices between [-100, 100], n = [100, 145] with step = 5
    n = 100
    for i in range(10):
        matrix = np.random.uniform(-50, 51, size=(n, n))
        matrix = matrix + np.transpose(matrix)
        real_matrices.append(matrix)
        n += 5

    # int matrices between [-100, 100], n = [100, 145] with step = 5
    n = 100
    for i in range(10):
        matrix = np.random.randint(-50, 51, size=(n, n))
        matrix = matrix + np.transpose(matrix)
        int_matrices.append(matrix)
        n += 5

    # int matrices between [100, 1000], n = [100, 145] with step = 5
    n = 100
    for i in range(10):
        matrix = np.random.randint(50, 501, size=(n, n))
        matrix = matrix + np.transpose(matrix)
        int_matrices_100_1000.append(matrix)
        n += 5

    ## DICTIONARIES FOR COUNTING THE ITERATIONS WHEN USING THE EPSILON SOLUTION

    iterations_REAL = dict()
    iterations_INT = dict()
    iterations_INT100_1000 = dict()

    #################################### TIME & ACCURACY TESTS ####################################

    ####### 1. REAL MATRICES [-100, 100]

    print("\n1. REAL MATRICES [-100, 100]:\n")

    matrix_sizes = []
    custom_times_epsilon = []
    custom_times_const_iter = []
    built_in_times = []
    vector_norms_epsilon = []
    vector_norms_const_iter = []
    for matrix in real_matrices:
        matrix_sizes.append(matrix.shape[0])

        built_in_start = time.time()
        built_in_eigvals = np.linalg.eigvals(matrix)
        built_in_elapsed = time.time() - built_in_start
        built_in_times.append(built_in_elapsed)
        built_in_eigvals = built_in_eigvals.tolist()
        built_in_eigvals.sort(reverse=True)
        print("Eigenvalues (Built-in):", built_in_eigvals)

        custom_start = time.time()
        custom_eigvals, iters = qr_algorithm(matrix, False)
        custom_elapsed = time.time() - custom_start
        iterations_REAL[matrix.shape[0]] = iters
        custom_times_epsilon.append(custom_elapsed)
        print("Eigenvalues (Custom QR - epsilon):", custom_eigvals)

        eps_norm = np.linalg.norm(np.array(built_in_eigvals) - np.array(custom_eigvals))
        vector_norms_epsilon.append(eps_norm)

        custom_start = time.time()
        custom_eigvals, iters = qr_algorithm(matrix, True)
        custom_elapsed = time.time() - custom_start
        custom_times_const_iter.append(custom_elapsed)
        print("Eigenvalues (Custom QR - 500 iterations):", custom_eigvals)

        const_iter_norms = np.linalg.norm(np.array(built_in_eigvals) - np.array(custom_eigvals))
        vector_norms_const_iter.append(const_iter_norms)

    # TIME: CUSTOM-EPSILON VS. BUILT-IN
    plt.plot(matrix_sizes, custom_times_epsilon, label = "QR_custom (epsilon)", marker="o", markerfacecolor="red")
    plt.plot(matrix_sizes, built_in_times, label = "Built-in algorithm", marker="o", markerfacecolor="red")
    plt.suptitle("Eigenvalue calculating time")
    plt.title("Real Matrices - [-100, 100]")
    plt.xlabel("Matrix sizes")
    plt.ylabel("Time elapsed in seconds")
    plt.legend()
    plt.show()

    # TIME: CUSTOM-500-ITERATIONS VS. BUILT-IN
    plt.plot(matrix_sizes, custom_times_const_iter, label="QR_custom (500 iterations)", marker="o", markerfacecolor="red")
    plt.plot(matrix_sizes, built_in_times, label="Built-in algorithm", marker="o", markerfacecolor="red")
    plt.suptitle("Eigenvalue calculating time")
    plt.title("Real Matrices - [-100, 100]")
    plt.xlabel("Matrix sizes")
    plt.ylabel("Time elapsed in seconds")
    plt.legend()
    plt.show()

    # ACCURACY: CUSTOM-EPSILON VS. BUILT-IN
    plt.bar(matrix_sizes, vector_norms_epsilon)
    plt.suptitle("Accuracy using 2nd vector norm")
    plt.title("Real Matrices - [-100, 100]")
    plt.xlabel("Matrix sizes")
    plt.ylabel("2nd vector norm of (built-in - epsilon-based)")
    plt.show()

    # ACCURACY: CUSTOM-500-ITERATIONS VS. BUILT-IN
    plt.bar(matrix_sizes, vector_norms_const_iter)
    plt.suptitle("Accuracy using 2nd vector norm")
    plt.title("Real Matrices - [-100, 100]")
    plt.xlabel("Matrix sizes")
    plt.ylabel("2nd vector norm of (built-in - 500-iterations-based)")
    plt.show()

    print("\n------------------------------------------------------------------------------------------\n")

    ####### 2. INT MATRICES [-100, 100]

    print("2. INT MATRICES [-100, 100]:\n")

    matrix_sizes = []
    custom_times_epsilon = []
    custom_times_const_iter = []
    built_in_times = []
    vector_norms_epsilon = []
    vector_norms_const_iter = []
    for matrix in int_matrices:
        matrix_sizes.append(matrix.shape[0])

        built_in_start = time.time()
        built_in_eigvals = np.linalg.eigvals(matrix)
        built_in_elapsed = time.time() - built_in_start
        built_in_times.append(built_in_elapsed)
        built_in_eigvals = built_in_eigvals.tolist()
        built_in_eigvals.sort(reverse=True)
        print("Eigenvalues (Built-in):", built_in_eigvals)

        custom_start = time.time()
        custom_eigvals, iters = qr_algorithm(matrix, False)
        custom_elapsed = time.time() - custom_start
        iterations_INT[matrix.shape[0]] = iters
        custom_times_epsilon.append(custom_elapsed)
        print("Eigenvalues (Custom QR - epsilon):", custom_eigvals)

        eps_norm = np.linalg.norm(np.array(built_in_eigvals) - np.array(custom_eigvals))
        vector_norms_epsilon.append(eps_norm)

        custom_start = time.time()
        custom_eigvals, iters = qr_algorithm(matrix, True)
        custom_elapsed = time.time() - custom_start
        custom_times_const_iter.append(custom_elapsed)
        print("Eigenvalues (Custom QR - 500 iterations):", custom_eigvals)

        const_iter_norms = np.linalg.norm(np.array(built_in_eigvals) - np.array(custom_eigvals))
        vector_norms_const_iter.append(const_iter_norms)

    # TIME: CUSTOM-EPSILON VS. BUILT-IN
    plt.plot(matrix_sizes, custom_times_epsilon, label="QR_custom (epsilon)", marker="o", markerfacecolor="red")
    plt.plot(matrix_sizes, built_in_times, label="Built-in algorithm", marker="o", markerfacecolor="red")
    plt.suptitle("Eigenvalue calculating time")
    plt.title("Integer Matrices - [-100, 100]")
    plt.xlabel("Matrix size")
    plt.ylabel("Time elapsed in seconds")
    plt.legend()
    plt.show()

    # TIME: CUSTOM-500-ITERATIONS VS. BUILT-IN
    plt.plot(matrix_sizes, custom_times_const_iter, label="QR_custom (500 iterations)", marker="o", markerfacecolor="red")
    plt.plot(matrix_sizes, built_in_times, label="Built-in algorithm", marker="o", markerfacecolor="red")
    plt.suptitle("Eigenvalue calculating time")
    plt.title("Integer Matrices - [-100, 100]")
    plt.xlabel("Matrix size")
    plt.ylabel("Time elapsed in seconds")
    plt.legend()
    plt.show()

    # ACCURACY: CUSTOM-EPSILON VS. BUILT-IN
    plt.bar(matrix_sizes, vector_norms_epsilon)
    plt.suptitle("Accuracy using 2nd vector norm")
    plt.title("Integer Matrices - [-100, 100]")
    plt.xlabel("Matrix size")
    plt.ylabel("2nd vector norm of (built-in - epsilon-based)")
    plt.show()

    # ACCURACY: CUSTOM-500-ITERATIONS VS. BUILT-IN
    plt.bar(matrix_sizes, vector_norms_const_iter)
    plt.suptitle("Accuracy using 2nd vector norm")
    plt.title("Integer Matrices - [-100, 100]")
    plt.xlabel("Matrix size")
    plt.ylabel("2nd vector norm of (built-in - 500-iterations-based)")
    plt.show()

    print("\n------------------------------------------------------------------------------------------\n")

    ####### 3. INT MATRICES [100, 1000]

    print("3. INT MATRICES [100, 1000]:\n")

    matrix_sizes = []
    custom_times_epsilon = []
    custom_times_const_iter = []
    built_in_times = []
    vector_norms_epsilon = []
    vector_norms_const_iter = []
    for matrix in int_matrices_100_1000:
        matrix_sizes.append(matrix.shape[0])

        built_in_start = time.time()
        built_in_eigvals = np.linalg.eigvals(matrix)
        built_in_elapsed = time.time() - built_in_start
        built_in_times.append(built_in_elapsed)
        built_in_eigvals = built_in_eigvals.tolist()
        built_in_eigvals.sort(reverse=True)
        print("Eigenvalues (Built-in):", built_in_eigvals)

        custom_start = time.time()
        custom_eigvals, iters = qr_algorithm(matrix, False)
        custom_elapsed = time.time() - custom_start
        iterations_INT100_1000[matrix.shape[0]] = iters
        custom_times_epsilon.append(custom_elapsed)
        print("Eigenvalues (Custom QR - epsilon):", custom_eigvals)

        eps_norm = np.linalg.norm(np.array(built_in_eigvals) - np.array(custom_eigvals))
        vector_norms_epsilon.append(eps_norm)

        custom_start = time.time()
        custom_eigvals, iters = qr_algorithm(matrix, True)
        custom_elapsed = time.time() - custom_start
        custom_times_const_iter.append(custom_elapsed)
        print("Eigenvalues (Custom QR - 500 iterations):", custom_eigvals)

        const_iter_norms = np.linalg.norm(np.array(built_in_eigvals) - np.array(custom_eigvals))
        vector_norms_const_iter.append(const_iter_norms)

    # TIME: CUSTOM-EPSILON VS. BUILT-IN
    plt.plot(matrix_sizes, custom_times_epsilon, label="QR_custom (epsilon)", marker="o", markerfacecolor="red")
    plt.plot(matrix_sizes, built_in_times, label="Built-in algorithm", marker="o", markerfacecolor="red")
    plt.suptitle("Eigenvalue calculating time")
    plt.title("Integer Matrices - [100, 1000]")
    plt.xlabel("Matrix size")
    plt.ylabel("Time elapsed in seconds")
    plt.legend()
    plt.show()

    # TIME: CUSTOM-500-ITERATIONS VS. BUILT-IN
    plt.plot(matrix_sizes, custom_times_const_iter, label="QR_custom (500 iterations)", marker="o", markerfacecolor="red")
    plt.plot(matrix_sizes, built_in_times, label="Built-in algorithm", marker="o", markerfacecolor="red")
    plt.suptitle("Eigenvalue calculating time")
    plt.title("Integer Matrices - [100, 1000]")
    plt.xlabel("Matrix size")
    plt.ylabel("Time elapsed in seconds")
    plt.legend()
    plt.show()

    # ACCURACY: CUSTOM-EPSILON VS. BUILT-IN
    plt.bar(matrix_sizes, vector_norms_epsilon)
    plt.suptitle("Accuracy using 2nd vector norm")
    plt.title("Integer Matrices - [100, 1000]")
    plt.xlabel("Matrix size")
    plt.ylabel("2nd vector norm of (built-in - epsilon-based)")
    plt.show()

    # ACCURACY: CUSTOM-500-ITERATIONS VS. BUILT-IN
    plt.bar(matrix_sizes, vector_norms_const_iter)
    plt.suptitle("Accuracy using 2nd vector norm")
    plt.title("Integer Matrices - [100, 1000]")
    plt.xlabel("Matrix size")
    plt.ylabel("2nd vector norm of (built-in - 500-iterations-based)")
    plt.show()

    #################################### ITERATIONS ####################################

    # PRINT ALL THE EPSILON ITERATIONS NEEDED:

    print("1. REAL MATRICES ITERATIONS [-100, 100] WITH EPSILON:\n")
    for key, value in iterations_REAL.items():
        print(f"{key}*{key}: {value} iterations needed")

    print("----------------------------------------------")

    print("2. INT MATRICES ITERATIONS [-100, 100] WITH EPSILON:\n")
    for key, value in iterations_INT.items():
        print(f"{key}*{key}: {value} iterations needed")

    print("----------------------------------------------")

    print("3. INT MATRICES ITERATIONS [100, 1000] WITH EPSILON:\n")
    for key, value in iterations_INT100_1000.items():
        print(f"{key}*{key}: {value} iterations needed")

