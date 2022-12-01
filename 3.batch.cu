#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

#define X 0
#define Y 1 

__global__ void batch_merge(int * A, int * sizeA, int *B, int *sizeB, int *M, const int d, int N){
	const int tidx = threadIdx.x % d;					// indice du thread dans son groupe 
	const int Qt = (threadIdx.x - tidx) / d;			// indice du thread dans son bloc(tableau shared)
	const int gbx = Qt + blockIdx.x * (blockDim.x / d);	// Numero du tableau(Ai, Bi) par rapport au tableau global (A,B)

	const int sizeAi = sizeA[gbx];				// Taille du tableau Ai
	const int sizeBi = sizeB[gbx];				// Taille du tableau Bi

	__shared__ int Ablock[1024];				// Toutes les Ai d'un bloc
	__shared__ int Bblock[1024];				// Toutes les Bi d'un bloc

	Ablock[Qt * d + tidx] = A[gbx * d + tidx];		// recuperation des Ai par le block
	Bblock[Qt * d + tidx] = B[gbx * d + tidx];		// recuperation des Bi par le block

	__syncthreads();					

	if (gbx * d + tidx >= N * d){
		return;
	}

	int K[2];
	int P[2];

	if (tidx > sizeAi) {
		K[X] = tidx - sizeAi;
		K[Y] = sizeAi;
		P[X] = sizeAi;
		P[Y] = tidx - sizeAi;
	}
	else {
		K[X] = 0;
		K[Y] = tidx;
		P[X] = tidx;
		P[Y] = 0;
	}

	while (1) {
		int offset = (abs(K[Y] - P[Y]))/2;
		int Q[2] = {K[X] + offset, K[Y] - offset};

		if (Q[Y] >= 0 && Q[X] <= sizeBi && (Q[Y] == sizeAi || Q[X] == 0 || Ablock[Qt*d + Q[Y]] > Bblock[Qt*d + Q[X]-1])) {
			if (Q[X] == sizeBi || Q[Y] == 0 || Ablock[Qt*d + Q[Y]-1] <= Bblock[Qt*d + Q[X]]) {
				if (Q[Y] < sizeAi && (Q[X] == sizeBi || Ablock[Qt*d + Q[Y]] <= Bblock[Qt*d + Q[X]]) ) {
						M[gbx * d + tidx] = Ablock[Qt*d + Q[Y]];
				}
				else {
						M[gbx * d + tidx] = Bblock[Qt*d + Q[X]];
				}
				break ;
			}
			else {
				K[X] = Q[X] + 1;
				K[Y] = Q[Y] - 1;
			}
		}
		else {
			P[X] = Q[X] - 1;
			P[Y] = Q[Y] + 1 ;
		}
	}
}


int main (void) {
    int *A, *B, *M, *size_A, *size_B;
    int *gpu_A, *gpu_B, *gpu_M, *gpu_size_A, *gpu_size_B;
	//int numBlocks = 4;
	int N = 1000; // nombre de Ai et Bi
	int NTPB = 1024;
	int dtab[8] = {2, 4, 8, 16, 32, 64, 128, 256};
	const int d = dtab[2];
	int numBlocks = (NTPB - 1 + N * d) / NTPB; // Nombre de blocs variables
	
	/*
	taille max de d = 256 car si chaque block doit merger plusieurs A et B en meme temps
	Le plus petit merge effectue par un block est 2*Ai et 2*Bi == 1024 (qui est le max)
	2*256 + 2*256 == 1024
	*/

	
    //allocation pour le device
    cudaMalloc(&gpu_A, N*d* sizeof(int));
    cudaMalloc(&gpu_B, N*d* sizeof(int));
    cudaMalloc(&gpu_M, N*d* sizeof(int));
	cudaMalloc(&gpu_size_A, N* sizeof(int));
	cudaMalloc(&gpu_size_B, N* sizeof(int));

    //cudaMalloc(&gpu_M, size_M);

    A =(int *) malloc(N*d * sizeof(int));
    B =(int *) malloc(N*d * sizeof(int));
    M =(int *) malloc(N*d * sizeof(int));
	size_A =(int *) malloc(N * sizeof(int));
	size_B =(int *) malloc(N * sizeof(int));

	// Remplissage des tableaux Ai et Bi
	/*
	On cree les Ai et Bi , avec d = |Ai| + |Bi|
	Les Ai sont de manieres contigues dans A = A0 || A1 || .... || An
	Les Bi sont de manieres contigues dans B = B0 || B1 || .... || Bn
	On stocke les tailles de chaque Ai et Bi
	*/

	srand(0);

	for (int i = 0; i < N; i++){
		size_A[i] = rand() % d;
		size_B[i] = d - size_A[i];

		for (int j = 0; j < size_A[i]; j++){
			A[i*d+j] = j;
		}
		
		for (int j = 0; j < size_B[i]; j++){
			B[i*d+j] = j + 1;
		}
	}

	int affiche = rand() % N; //on affiche une indice au hasard
	printf("A[%d, d = %d]  = {", affiche, size_A[affiche]);
	for (int j = 0; j < size_A[affiche]; j++){
		printf("%d\t", A[affiche*d+j]);
	}
	printf("}\n");

	printf("B[%d, d = %d] = {", affiche, size_B[affiche]);
	for (int j = 0; j < size_B[affiche]; j++){
		printf("%d\t", B[affiche*d+j]);
	}
	printf("}\n");

    //copie des donnees vers le device
    cudaMemcpy (gpu_A , A, N*d * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy (gpu_B , B, N*d * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy (gpu_size_A , size_A, N * sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy (gpu_size_B , size_B, N * sizeof(int),cudaMemcpyHostToDevice);

    float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions

    /******************************************************************/
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

    batch_merge<<<numBlocks, NTPB>>>(gpu_A, gpu_size_A, gpu_B, gpu_size_B, gpu_M, d, N);

    testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
			 start, stop));							// GPU timer instructions
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions
	printf("\n temps d'exécution : %f ms \n", TimeExec);
    /******************************************************************/


    //copie du resultat vers host
    cudaMemcpy (M, gpu_M, N*d * sizeof(int), cudaMemcpyDeviceToHost);

    /* Affichage du resultat*/
	printf("M[%d, d==%d] = {\n", affiche, d);
	for (int j = 0; j < d; j++){
		printf("%d\t", M[affiche*d+j]);
	}
	printf("}\n");
	
    //Liberation de l'espace alloué
    free(A); free(B); free(M); free(size_A); free(size_B);
    cudaFree(gpu_A); cudaFree(gpu_B); cudaFree(gpu_M); cudaFree(gpu_size_A); cudaFree(gpu_size_B);
    
    return 0 ;
}