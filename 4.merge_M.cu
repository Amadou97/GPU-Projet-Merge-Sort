#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctime>
#include <time.h>
#include <algorithm>

#define N 8
#define X 0
#define Y 1

#define NTPB 8
#define NB 1

void testCUDA(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void trie(int * M, int sizeM){

    int nb_iter = sizeM ;
    int SIZEA = 1;
    int SIZEB = 1;

    __syncthreads();

    do {
        nb_iter /= 2;

        for(int t = 0; t < nb_iter ; ++t){
            __shared__ int A[8];
            __shared__ int B[8];

            if( threadIdx.x == 0 ){
                for(int j = 0; j < SIZEA; j++){
                    A[j] = M[j + t*(SIZEA+SIZEB)];
                    B[j] = M[j + (t+1)*(SIZEA+SIZEB)];
                }
            }

            __syncthreads();

            int i = threadIdx.x;
            /*
            if(i == 0 ) {
                printf("\n Ai ");
                for(int j = 0; j < SIZEA; j++){
                    printf("\t %d " ,A[j]);
                }
                printf("\n Ai fin ");
            }*/

            if( i < SIZEA + SIZEB ){
                int k[2];
                int P[2];
                if (i > SIZEA) {
                    k[X] = i - SIZEA;
                    k[Y] = SIZEA;
                    P[X] = SIZEA;
                    P[Y] = i - SIZEA;
                } else {
                    k[X] = 0;
                    k[Y] = i;
                    P[X] = i;
                    P[Y] = 0;
                }
                
                while (1) {
                    int offset = (abs(k[Y] - P[Y]))/2;
                    int Q[2] = {k[X] + offset, k[Y] - offset};
                    
                    if (Q[Y] >= 0 && Q[X] <= SIZEB && (Q[Y] == SIZEA || Q[X] == 0 || A[Q[Y]] > B[Q[X]-1])) {
                        if (Q[X] == SIZEB || Q[Y] == 0 || A[Q[Y]-1] <= B[Q[X]]) {
                            if (Q[Y] < SIZEA && (Q[X] == SIZEB || A[Q[Y]] <= B[Q[X]]) ) {
                                M[i] = A[Q[Y]];
                            } else {
                                M[i] = B[Q[X]];
                            }
                            break ;
                        }	else {
                            k[X] = Q[X] + 1;
                            k[Y] = Q[Y] - 1;
                        }
                    } else {
                        P[X] = Q[X] - 1;
                        P[Y] = Q[Y] + 1;
                    }
                }
            }
        }

        
        if(threadIdx.x == 0 ) {
            printf("\n M ");
            for(int j = 0; j < sizeM; j++){
                printf("\t %d " ,M[j]);
            }
            printf("\n");
        }

        SIZEA *= 2;
        SIZEB *= 2;
        __syncthreads();

    } while (nb_iter > 1);

}


int main (void) {
    int *M;
    int *gpu_M;
    int size_M = N * sizeof(int) ;

    //allocation pour le device
    cudaMalloc(&gpu_M, size_M);

    M =(int *) malloc (size_M);
   
    srand(0);
    for (int i = 0; i < N; i++) {
        M[i] = rand() % 10;
    } 

    printf("\nM ={ ");
    for (int i = 0; i < N; i++) {
        printf("%d\t ", M[i]);
    } 
     printf("}\n");

    //copie des donnees vers le device
    cudaMemcpy (gpu_M , M, size_M ,cudaMemcpyHostToDevice);

    float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions

    /******************************************************************/
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

    trie<<<1, NTPB>>>(gpu_M, N);

    testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
			 start, stop));							// GPU timer instructions
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions
	printf("\n temps d'execution du trie : %f ms \n", TimeExec);
    /******************************************************************/

    //copie du resultat vers host
    cudaMemcpy (M, gpu_M, size_M, cudaMemcpyDeviceToHost);

    /* Affichage */
    printf("\nM = { ");
    for (int i = 0; i < N; i++) {
        printf("%d\t", M[i]);
    }
    printf("}\n");

    //Liberation de l'espace alloué
    free(M);
    cudaFree(gpu_M);
    
    return 0 ;
}