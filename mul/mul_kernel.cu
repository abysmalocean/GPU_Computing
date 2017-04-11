
/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MUL_KERNEL
#define _MUL_KERNEL

#include <stdio.h>
#include "mul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification




__global__ void MatrixMulKernelA1(Matrix M, Matrix N, Matrix P)
{
        //Multiply the two matrices

}

__global__ void MatrixMulKernelA2(Matrix M, Matrix N, Matrix P)
{
        //Multiply the two matrices

}




__global__ void MatrixMulKernelB(Matrix M, Matrix N, Matrix P)
{
        //Multiply the two matrices

        __shared__ float Ms[bm][bk];
        __shared__ float Ns[bk][bn];

}

#endif 
