
/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <cutil.h>
//#include <cuda.h>
//#include <cutil_inline.h>

// includes, kernels
#include <mul_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
void MatrixReset(Matrix M);
Matrix MatrixPadding(Matrix M);
void MatrixPadRemoving(Matrix M,Matrix M2);

void MatrixMulOnDeviceA1(const Matrix M, const Matrix N, Matrix P);
void MatrixMulOnDeviceA2(const Matrix M, const Matrix N, Matrix P);
void MatrixMulOnDeviceB(const Matrix M, const Matrix N, Matrix P);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Matrices for the program
	Matrix  M;
	Matrix  N;
	Matrix  P;

	if (argc==4) {
		HM = atoi(argv[1]); //height of Matrix M
		WM = atoi(argv[2]); //Width of Matrix M
		WN = atoi(argv[3]); //Width of Matrix N
		HN = WM;
		HP = HM;
		WP = WN;
	}

	// Number of elements in the solution matrix
	//  Assuming square matrices, so the sizes of M, N and P are equal
	unsigned int size_elements = WP * HP;
	
	srand(2012);
	
	// Check command line for input matrix files
	// No inputs provided
	// Allocate and initialize the matrices
	M  = AllocateMatrix(HM, WM, 1);
	N  = AllocateMatrix(HN, WN, 1);
	P  = AllocateMatrix(HP, WP, 0);
    	// compute the matrix multiplication on the CPU for comparison
    	Matrix reference = AllocateMatrix(HP, WP, 0);
	clock_t st=clock();
    	computeGold(reference.elements, M.elements, N.elements, HM, WM, WN);
        st = clock()-st;

	printf("CPU executation is %.4f\n",(double)st/CLOCKS_PER_SEC);

	MatrixMulOnDeviceA1(M, N, P);

    	bool res=true;

   	for (int i=0;i<size_elements;i++)
	if (fabs(reference.elements[i]-P.elements[i])>0.0001f) {
		res=false;    
		break;
	}



    	// check if the device result is equivalent to the expected solution
    	//CUTBoolean res = cutComparefe(reference.elements, P.elements,
	//				size_elements, 0.0001f);
    	printf("Part A1 Test %s\n", (true == res) ? "PASSED" : "FAILED");

	MatrixReset(P);
	MatrixMulOnDeviceA2(M, N, P);
    	// check if the device result is equivalent to the expected solution
    	//res = cutComparefe(reference.elements, P.elements,
	//				size_elements, 0.0001f);
    	//printf("Part A2 Test %s\n", (1 == res) ? "PASSED" : "FAILED");
	res = true;   	
	for (int i=0;i<size_elements;i++)
	if (fabs(reference.elements[i]-P.elements[i])>0.0001f) {
			res=false;    
			break;
	}
    	printf("Part A2 Test %s\n", (true == res) ? "PASSED" : "FAILED");


	MatrixReset(P);
	MatrixMulOnDeviceB(M, N, P);
    	// check if the device result is equivalent to the expected solution
    	//res = cutComparefe(reference.elements, P.elements,
	//				size_elements, 0.0001f);
    	//printf("Part B Test %s\n", (1 == res) ? "PASSED" : "FAILED");
	res = true;   	
	for (int i=0;i<size_elements;i++)
	if (fabs(reference.elements[i]-P.elements[i])>0.0001f) {
			res=false;    
			break;
	}
    	printf("Part B Test %s\n", (true == res) ? "PASSED" : "FAILED");

    
	// Free host matrices
	FreeMatrix(&M);
	FreeMatrix(&N);
	FreeMatrix(&P);
	return 0;
}

void MatrixMulOnDeviceA1(const Matrix M, const Matrix N, Matrix P)
{

	//Allocate device matrices


 	//setup kernel configuration


        clock_t st=clock();	

	//Copying data from host to device
	


	//Launch kernel function
	


	//Copying results back to host
	
	st = clock()-st;
	printf("GPU executation for A1 is %.4f\n",(double)st/CLOCKS_PER_SEC);

	// Free device matrices

}

void MatrixMulOnDeviceA2(const Matrix M, const Matrix N, Matrix P)
{
	//Matrix Padding


	//Allocate device matrices

 	
	//setup kernel configurartion


        clock_t st=clock();	
	//Copying data from host to device
	


	//Launch kernel function
	


	//Copying results back to host
	
	
	st = clock()-st;
	printf("GPU executation for A2 is %.4f\n",(double)st/CLOCKS_PER_SEC);


	//Removing padding


	// Free device matrices

}


// Allocate a device matrix of same size as M.


void MatrixMulOnDeviceB(const Matrix M, const Matrix N, Matrix P)
{

       //Allocate device matrices


        //setup kernel configuration


        clock_t st=clock();

        //Copying data from host to device



        //Launch kernel function



        //Copying results back to host

        
	st = clock()-st;
        printf("GPU executation for B is %.4f\n",(double)st/CLOCKS_PER_SEC);

        // Free device matrices

	
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//      If init == 0, initialize to all zeroes.  
//      If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    // don't allocate memory on option 2
    if(init == 2)
                return M;

        M.elements = (float*) malloc(size*sizeof(float));

        for(unsigned int i = 0; i < M.height * M.width; i++)
        {
                M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
        }
    return M;
}


Matrix MatrixPadding(Matrix M)
{
	int height = ((M.height-1)/BLOCKSIZE+1)*BLOCKSIZE;
	int width = ((M.width-1)/BLOCKSIZE+1)*BLOCKSIZE;
	Matrix M2 =AllocateMatrix(height,width,0);

	for (int i=0;i<M.height;i++)
		for (int j=0;j<M.width;j++)
			M2.elements[i*M2.width+j] = M.elements[i*M.width+j];
	return M2;
}

void MatrixPadRemoving(Matrix M, Matrix M2)
{

	for (int i=0;i<M.height;i++)
		for (int j=0;j<M.width;j++)
			M.elements[i*M.width+j] = M2.elements[i*M2.width+j];
}


// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size,
                                        cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size,
                                        cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}


void MatrixReset(Matrix M)
{
      for (int i=0;i<M.width*M.height;i++)
	M.elements[i] = 0;
}
