#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
 
void matrixMultiplyCPU(float *A, float *B, float *C, int N) { //column-major order
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < N; j++) { 
            float sum = 0.0f; 
            for (int k = 0; k < N; k++) { 
                sum += A[i * N + k] * B[k * N + j]; // matrix multiplication logic
            } 
            C[i * N + j] = sum; 
        } 
    } 
} 
     /*
argv (char **)
  |
  v
+------------+------------+------------+
| argv[0]    | argv[1]    | argv[2]    |
+------------+------------+------------+
     |             |            |
     v             v            v
 "./mm"          "512"        "hello"

 
    */
int main(int argc, char **argv) { 
    int N = (argc > 1) ? atoi(argv[1]) : 1024; // allow matrix size as input 
    size_t size = N * N * sizeof(float); 
 
    float *A = (float *)malloc(size); 
    float *B = (float *)malloc(size); 
    float *C = (float *)malloc(size); 
 
    for (int i = 0; i < N * N; i++) { 
        A[i] = rand() % 100 / 100.0f; 
        B[i] = rand() % 100 / 100.0f; 
    } 
 
    clock_t start = clock(); 
    matrixMultiplyCPU(A, B, C, N); 
    clock_t end = clock(); 
 
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC; 
    printf("CPU execution time (N=%d): %f seconds\n", N, elapsed); 
 
    free(A); free(B); free(C); 
    return 0; 
} 