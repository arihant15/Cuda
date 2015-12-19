# CUDA
CUDA Programming

Compilation Steps:

	Method 1: Run make command.
	
		eg: $ make

	Method 2: compile each code individually.

		$gcc -o matrixNorm -lm matrixNorm.c
		$nvcc -o matrixNormCuda matrixNormCuda.cu

	Note: Ignore warning while coming CUDA program

Execution steps:

	Sequential Code:
		$./matrixNorm <matrix_dimension> [random seed]
	CUDA Code:
		$./matrixNormCuda <matrix_dimension> [random seed]

Examples:

	Sequential Code:
		$./matrixNorm 4 3
	CUDA Code:
		$./matrixNormCuda 4 3