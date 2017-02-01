
/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

// includes, project
//include <cutil.h>

// includes, kernels
#include <simplematrixmul_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Matrices for the program
	Matrix  M;
	Matrix  N;
	Matrix  P;
	// Number of elements in the solution matrix
	//  Assuming square matrices, so the sizes of M, N and P are equal
	unsigned int size_elements = WP * HP;
	int errorM = 0, errorN = 0;

	srand(2012);

	// Check command line for input matrix files
	if(argc != 3 && argc != 4)
	{
		// No inputs provided
		// Allocate and initialize the matrices
		M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
		N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
		P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	}
	else
	{
		// Inputs provided
		// Allocate and read source matrices from disk
		M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
		N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
		P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
		errorM = ReadFile(&M, argv[1]);
		errorN = ReadFile(&N, argv[2]);
		// check for read errors
		if(errorM != size_elements || errorN != size_elements)
		{
			printf("Error reading input files %d, %d\n", errorM, errorN);
			return 1;
		}
	}
	/*
	//print M
	printf("Element in the M is:\n");
 for(int i = 0 ; i < size_elements ; i++)
 {
	 std::cout<<M.elements[i]<< " ";
 }
 printf("\nend\n");
 //print N
 printf("Element in the N is:\n");
for(int i = 0 ; i < size_elements ; i++)
{
	std::cout<<N.elements[i]<< " ";
}
printf("\nend\n");
*/
	// M * N on the device
    MatrixMulOnDevice(M, N, P);

    // compute the matrix multiplication on the CPU for comparison
    Matrix reference = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
    computeGold(reference.elements, M.elements, N.elements, HM, WM, WN);

    bool res=true;
		/*
		printf("Element in the matrix is:\n");
	 for(int i = 0 ; i < size_elements ; i++)
	 {
		 std::cout<<reference.elements[i]<< " ";
	 }
	 printf("\nend\n");

	 printf("Element in the GPU is:\n");
		for(int i = 0 ; i < size_elements ; i++)
		{
			std::cout<<P.elements[i]<< " ";
		}
	printf("\nend\n");
	*/
   for (int i=0;i<size_elements;i++)
	 {
		 //printf("P.elements is [%f] and reference is [%f]",P.elements[i],reference.elements[i]);
		 if (fabs(reference.elements[i]-P.elements[i])>0.0001f) {
			 res=false;
			 break;
		 }
	 }

// check if the device result is equivalent to the expected solution
    //CUTBoolean res = cutComparefe(reference.elements, P.elements,
//									size_elements, 0.0001f);
    printf("Test %s\n", (true == res) ? "PASSED" : "FAILED");

    // output result if output file is requested
    if(argc == 4)
    {
		WriteFile(P, argv[3]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}

	// Free host matrices
    free(M.elements);
    M.elements = NULL;
    free(N.elements);
    N.elements = NULL;
    free(P.elements);
    P.elements = NULL;
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	//Interface host call to the device kernel code and invoke the kernel
	Matrix Mdevice = AllocateDeviceMatrix(M);
	Matrix Ndevice = AllocateDeviceMatrix(N);
	Matrix Pdevice = AllocateDeviceMatrix(P);
	CopyToDeviceMatrix(Mdevice, M);
	CopyToDeviceMatrix(Ndevice, N);
	int threadsPerBlockDim = 8;
	int gridDimSize = (MATRIX_SIZE + threadsPerBlockDim - 1) / threadsPerBlockDim;
	dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
	dim3 gridSize (gridDimSize, gridDimSize);
	printf("Start runing the CUDA Kernel\n");
	MatrixMulKernel<<<gridSize,blockSize>>>(Mdevice, Ndevice, Pdevice);
	CopyFromDeviceMatrix(P, Pdevice);

	cudaError_t cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.
//	If init == 1, perform random initialization.
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	}
    return M;
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

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = MATRIX_SIZE*MATRIX_SIZE;
	unsigned int i=0;
	//cutReadFilef(file_name, &(M->elements), &data_read, true);
	FILE *fp=fopen(file_name,"rb");
	if (fp==NULL) return -1;
	while (!feof(fp)) {
		fread(&(M->elements[i]),sizeof(float),1,fp);
		i++;
	}
	fclose(fp);
	return i;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    //cutWriteFilef(file_name, M.elements, M.width*M.height,
                       //0.0001f);
	FILE *fp=fopen(file_name,"wb");
	if (fp==NULL) return;
	fwrite(M.elements,sizeof(float),M.width*M.height,fp);
	fclose(fp);
}
