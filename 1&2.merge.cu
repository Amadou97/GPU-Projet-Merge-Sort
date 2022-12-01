#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctime>
#include <time.h>
#include <algorithm>

#define N1 128
#define N2 128
#define X 0
#define Y 1

#define NTPB 256
#define NB 1

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}


#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void merge(int * A, int * B, int * M, int SIZEA, int SIZEB){

    int i = threadIdx.x;

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


__global__ void merge_many_blocks(int *A, int *B, int * M, int SIZEAi, int SIZEBi){

    /* on suppose que les Ai ont la meme taille, les Bi aussi */

    int SIZEA = SIZEAi / gridDim.x; //tab_sizeA[blockIdx.x];
    int SIZEB = SIZEBi / gridDim.x; //tab_sizeB[blockIdx.x];

    __shared__ int Ai[1024];
    __shared__ int Bi[1024];

   
    if( threadIdx.x < SIZEA ){
        Ai[threadIdx.x] = A[threadIdx.x + SIZEA*blockIdx.x ];
    }

    if( threadIdx.x < SIZEB ){
        Bi[threadIdx.x] = A[threadIdx.x + SIZEB*blockIdx.x ];
    }

    int i = threadIdx.x;
    
    if(i < SIZEA + SIZEB) {

        int k[2];
		int P[2];
		if (i > SIZEA) {
			k[X] = i - SIZEA;
			k[Y] = SIZEA;
			P[X] = SIZEA;
			P[Y] = i - SIZEA;
		}
		else {
			k[X] = 0;
			k[Y] = i;
			P[X] = i;
			P[Y] = 0;
		}

        while (1) {
            int offset = (abs(k[Y] - P[Y]))/2;
			int Q[2] = {k[X] + offset, k[Y] - offset};

            if (Q[Y] >= 0 && Q[X] <= SIZEB && (Q[Y] == SIZEA || Q[X] == 0 || Ai[Q[Y]] > Bi[Q[X]-1])) {
			if (Q[X] == SIZEB || Q[Y] == 0 || Ai[Q[Y]-1] <= Bi[Q[X]]) {
				if (Q[Y] < SIZEA && (Q[X] == SIZEB || Ai[Q[Y]] <= Bi[Q[X]]) ) {
					M[i + (SIZEA + SIZEB)*blockIdx.x] = Ai[Q[Y]];
				} else {
					M[i + (SIZEA + SIZEB)*blockIdx.x] = Bi[Q[X]];
				}

                break;

                } else {
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

int main (void) {
    int *A, *B, *M;
    int *gpu_A, *gpu_B, *gpu_M;
    int size_A = N1 * sizeof(int);
    int size_B = N2 * sizeof(int);
    int size_M = (N1+N2) * sizeof(int) ;
    
    //MyTab *gpu_M;

    //allocation pour le device
    cudaMalloc(&gpu_A, size_A);
    cudaMalloc(&gpu_B, size_B);
    cudaMalloc(&gpu_M, size_M);

    //cudaMalloc(&gpu_M, size_M);

    A =(int *) malloc (size_A);
    B =(int *) malloc (size_B);
    M =(int *) malloc (size_M);
    
    //MyTab *M;
	//testCUDA(cudaHostAlloc(&M, sizeof(MyTab), cudaHostAllocDefault));

    
    for (int i = 0; i < N1; i++) {
        A[i] = i;
    } 

    for (int i = 0; i < N2; i++) {
        B[i] = i;
    } 

    //copie des donnees vers le device
    cudaMemcpy (gpu_A , A, size_A ,cudaMemcpyHostToDevice);
    cudaMemcpy (gpu_B , B, size_B ,cudaMemcpyHostToDevice);
    
	//testCUDA(cudaMemcpy(GPUTab, CPUTab, sizeof(MyTab), cudaMemcpyHostToDevice));


    float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions

    /******************************************************************/
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

    merge<<<1, NTPB>>>(gpu_A, gpu_B, gpu_M, N1, N2);

    testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
			 start, stop));							// GPU timer instructions
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions
	printf("\n temps d'exécution avec merge un seul block: %f ms \n", TimeExec);
    /******************************************************************/


    /******************Test avec plusieurs blocks**********************/
    int nb_block[7] = {2, 4, 8, 16, 32, 64};
    for (int i = 0; i < 6; i++){
        testCUDA(cudaEventCreate(&start));				// GPU timer instructions
        testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
        testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

        merge_many_blocks<<<nb_block[i], NTPB>>>(gpu_A, gpu_B, gpu_M, N1, N2);

        testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
        testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
        testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
                start, stop));							// GPU timer instructions
        testCUDA(cudaEventDestroy(start));				// GPU timer instructions
        testCUDA(cudaEventDestroy(stop));				// GPU timer instructions
        printf("\n temps d'exécution pour %d blocks: %f ms \n", nb_block[i], TimeExec);
    }
    /***********************end****************************************/


    //copie du resultat vers host
    cudaMemcpy (M, gpu_M, size_M, cudaMemcpyDeviceToHost);
    //cudaMemcpy(M, gpu_M, sizeof(MyTab), cudaMemcpyDeviceToHost);

    /* Affichage */
    printf(" \n A = { ");
    for (int i = 0; i < N1; i++) {
        printf(" %d\t", A[i]);
    }
    printf(" } \n");

    printf(" \n B = { ");
    for (int i = 0; i < N2; i++) {
        printf(" %d\t", B[i]);
    }
    printf(" } \n");


    printf(" \n M = { ");
    for (int i = 0; i < N1 + N2; i++) {
        printf(" %d\t", M[i]);
    }
    printf(" } \n");
    
    //Liberation de l'espace alloué
    free(A); free(B); free(M);
    
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_M);
    
    return 0 ;
}