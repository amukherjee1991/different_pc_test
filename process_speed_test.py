import numpy as np
import timeit
import platform

def complex_math_operations():
    """Perform complex mathematical computations"""
    
    # Define matrix size
    size = 1000  # Change this for larger or smaller tests
    
    # Generate two random matrices
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # Matrix multiplication
    C = np.dot(A, B)
    
    # Compute determinant (costly operation)
    det_A = np.linalg.det(A)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    
    # Solve a system of linear equations Ax = B
    x = np.linalg.solve(A, np.random.rand(size))
    
    return det_A, eigenvalues, x

# Time the execution
execution_time = timeit.timeit(complex_math_operations, number=1)

# System information
system_info = {
    "OS": platform.system(),
    "OS Version": platform.version(),
    "Processor": platform.processor(),
    "Architecture": platform.architecture(),
    "Execution Time (seconds)": execution_time
}

# Print results
print("\nSystem Info and Execution Time:")
for key, value in system_info.items():
    print(f"{key}: {value}")
