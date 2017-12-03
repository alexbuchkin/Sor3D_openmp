#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

int N = 2*2*2*2*2*2;
int n_threads = 4;

double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;

double *** A;

double totalTime = 0.0;

void relax();
void init();
void verify(); 

int main(int argc, char **argv)
{
	int it;
	
	if (argc == 3) {
		sscanf(argv[1], "%d", &N);
		sscanf(argv[2], "%d", &n_threads);
	}
	
	N += 2;

	init();

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		//printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}

	verify();
	
	free(A);
	
	printf("Total time: %.6lf\n", totalTime);
	
	return 0;
}


void init()
{ 
	omp_lock_t lock;
	omp_init_lock(&lock);
	
	//allocating matrix
	A = (double ***)malloc(N * sizeof(*A));
	for (i = 0; i < N; ++i) {
		A[i] = (double **)malloc(N * sizeof(*A[i]));
		for (j = 0; j < N; ++j) {
			A[i][j] = (double *)malloc(N * sizeof(*A[i][j]));
		}
	}
	
	double initTime = 0.0;
	
	#pragma omp parallel shared(A, initTime) num_threads(n_threads)
	{
		double begin = omp_get_wtime();
		
		#pragma omp for private(i, j, k)
		for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
		for(k=0; k<=N-1; k++)
		{
			if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) {
				A[i][j][k]= 0.0;
			}
			else A[i][j][k]= ( 4. + i + j + k) ;
		}
		
		//counting time
		begin = omp_get_wtime() - begin;
		omp_set_lock(&lock);
		initTime = Max(initTime, begin);
		omp_unset_lock(&lock);
	}
	
	totalTime += initTime;
	
	omp_destroy_lock(&lock);
} 


void relax()
{
	omp_lock_t lock;
	omp_init_lock(&lock);
	
	int wave = 2;
	//wave traversal by i and j
	for (wave = 2; wave <= 2*(N - 2); ++wave) {
		
		double iterTime = 0.0;
		
		#pragma omp parallel shared(A, eps, iterTime) num_threads(n_threads)
		{
			double begin = omp_get_wtime();
			double localEps = 0.0;
			
			#pragma omp for private(i, j, k)
			for (i = Max(1, wave - (N - 2)); i <= Min((N - 2), wave - 1); ++i) {
				j = wave - i;
				
				for (k = 1; k <= (N - 2); ++k) {
					double e = A[i][j][k];
					A[i][j][k] = (A[i-1][j][k] +
								  A[i+1][j][k] +
								  A[i][j-1][k] +
								  A[i][j+1][k] +
								  A[i][j][k-1] +
								  A[i][j][k+1]) / 6.0;
					localEps = Max(localEps, fabs(e - A[i][j][k]));
				}
			}
			
			//updating eps
			omp_set_lock(&lock);
			eps = Max(eps, localEps);
			omp_unset_lock(&lock);
			
			//counting time
			begin = omp_get_wtime() - begin;
			omp_set_lock(&lock);
			iterTime = Max(iterTime, begin);
			omp_unset_lock(&lock);
		}
		
		totalTime += iterTime;
	}
	
	omp_destroy_lock(&lock);
}


void verify()
{ 
	omp_lock_t lock;
	omp_init_lock(&lock);
	
	double s = 0.0;
	double verifyTime = 0.0;
	
	#pragma omp parallel shared(A, s, verifyTime) num_threads(n_threads)
	{
		double begin = omp_get_wtime();
		double localS = 0.0;
		
		#pragma omp for private(i, j, k)
		for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
		for(k=0; k<=N-1; k++)
		{
			localS += A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
		}
		
		omp_set_lock(&lock);
		s = s + localS;
		omp_unset_lock(&lock);
		
		//counting time
		begin = omp_get_wtime() - begin;
		omp_set_lock(&lock);
		verifyTime = Max(verifyTime, begin);
		omp_unset_lock(&lock);
	}
	
	totalTime += verifyTime;
	printf("  S = %f\n",s);
	
	omp_destroy_lock(&lock);

}

