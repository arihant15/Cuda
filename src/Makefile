
CC := nvcc
GC := gcc

assignment4: $(OBJ)
	$(GC) -o matrixNorm -lm matrixNorm.c
	$(CC) -o matrixNormCuda matrixNormCuda.cu

q1:
	./matrixNorm 32

q2:
	./matrixNormCuda 32

clean:
	rm -rf matrixNorm matrixNormCuda matrixNormCuda1 *.o

