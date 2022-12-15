/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"
#include "par_ilu.h"
#include "seq_mv.hpp"

/*********************************************************************************/
/*                   nalu_hypre_ILUSolveDeviceLUIter                                  */
/*********************************************************************************/
/* Incomplete LU solve (GPU) using Jacobi iterative approach
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
*/

#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE)

NALU_HYPRE_Int
nalu_hypre_ILUSolveLJacobiIter(nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *input_local, nalu_hypre_Vector *work_local,
                          nalu_hypre_Vector *output_local, NALU_HYPRE_Int lower_jacobi_iters)
{
   NALU_HYPRE_Real              *input_data          = nalu_hypre_VectorData(input_local);
   NALU_HYPRE_Real              *work_data           = nalu_hypre_VectorData(work_local);
   NALU_HYPRE_Real              *output_data         = nalu_hypre_VectorData(output_local);
   NALU_HYPRE_Int               num_rows             = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int kk = 0;

   /* L solve - Forward solve ; u^{k+1} = f - Lu^k*/
   /* Jacobi iteration loop */

   /* Since the initial guess to the jacobi iteration is 0, the result of the first L SpMV is 0, so no need to compute
      However, we still need to compute the transformation */
   hypreDevice_ComplexAxpyn(work_data, num_rows, input_data, output_data, 0.0);

   /* Do the remaining iterations */
   for ( kk = 1; kk < lower_jacobi_iters; ++kk )
   {

      /* apply SpMV */
      nalu_hypre_CSRMatrixSpMVDevice(0, 1.0, A, output_local, 0.0, work_local, -2);

      /* transform */
      hypreDevice_ComplexAxpyn(work_data, num_rows, input_data, output_data, -1.0);
   }

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_ILUSolveUJacobiIter(nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *input_local, nalu_hypre_Vector *work_local,
                          nalu_hypre_Vector *output_local, nalu_hypre_Vector *diag_diag, NALU_HYPRE_Int upper_jacobi_iters)
{
   NALU_HYPRE_Real              *output_data         = nalu_hypre_VectorData(output_local);
   NALU_HYPRE_Real              *work_data           = nalu_hypre_VectorData(work_local);
   NALU_HYPRE_Real              *input_data          = nalu_hypre_VectorData(input_local);
   NALU_HYPRE_Real              *diag_diag_data      = nalu_hypre_VectorData(diag_diag);
   NALU_HYPRE_Int               num_rows             = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int kk = 0;

   /* U solve - Backward solve :  u^{k+1} = f - Uu^k */
   /* Jacobi iteration loop */

   /* Since the initial guess to the jacobi iteration is 0, the result of the first U SpMV is 0, so no need to compute
      However, we still need to compute the transformation */
   hypreDevice_zeqxmydd(num_rows, input_data, 0.0, work_data, output_data, diag_diag_data);

   /* Do the remaining iterations */
   for ( kk = 1; kk < upper_jacobi_iters; ++kk )
   {

      /* apply SpMV */
      nalu_hypre_CSRMatrixSpMVDevice(0, 1.0, A, output_local, 0.0, work_local, 2);

      /* transform */
      hypreDevice_zeqxmydd(num_rows, input_data, -1.0, work_data, output_data, diag_diag_data);
   }

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_ILUSolveLUJacobiIter(nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *work1_local,
                           nalu_hypre_Vector *work2_local, nalu_hypre_Vector *inout_local, nalu_hypre_Vector *diag_diag,
                           NALU_HYPRE_Int lower_jacobi_iters, NALU_HYPRE_Int upper_jacobi_iters, NALU_HYPRE_Int my_id)
{
   /* apply the iterative solve to L */
   nalu_hypre_ILUSolveLJacobiIter(A, inout_local, work1_local, work2_local, lower_jacobi_iters);

   /* apply the iterative solve to U */
   nalu_hypre_ILUSolveUJacobiIter(A, work2_local, work1_local, inout_local, diag_diag, upper_jacobi_iters);

   return nalu_hypre_error_flag;
}


/* Incomplete LU solve using jacobi iterations on GPU
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
*/
NALU_HYPRE_Int
nalu_hypre_ILUSolveDeviceLUIter(nalu_hypre_ParCSRMatrix *A, nalu_hypre_CSRMatrix *matLU_d,
                           nalu_hypre_ParVector *f,  nalu_hypre_ParVector *u, NALU_HYPRE_Int *perm,
                           NALU_HYPRE_Int n, nalu_hypre_ParVector *ftemp, nalu_hypre_ParVector *utemp,
                           nalu_hypre_Vector *xtemp_local, nalu_hypre_Vector **Adiag_diag,
                           NALU_HYPRE_Int lower_jacobi_iters, NALU_HYPRE_Int upper_jacobi_iters)
{
   /* Only solve when we have stuffs to be solved */
   if (n == 0)
   {
      return nalu_hypre_error_flag;
   }

   MPI_Comm             comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int my_id;
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   nalu_hypre_Vector            *utemp_local         = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real              *utemp_data          = nalu_hypre_VectorData(utemp_local);

   nalu_hypre_Vector            *ftemp_local         = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real              *ftemp_data          = nalu_hypre_VectorData(ftemp_local);

   NALU_HYPRE_Real              alpha;
   NALU_HYPRE_Real              beta;

   /* begin */
   alpha = -1.0;
   beta = 1.0;

   /* Grab the main diagonal from the diagonal block. Only do this once */
   if (!(*Adiag_diag))
   {
      /* storage for the diagonal */
      *Adiag_diag = nalu_hypre_SeqVectorCreate(n);
      nalu_hypre_SeqVectorInitialize(*Adiag_diag);
      /* extract with device kernel */
      nalu_hypre_CSRMatrixExtractDiagonalDevice(matLU_d, nalu_hypre_VectorData(*Adiag_diag), 2);
      //nalu_hypre_CSRMatrixGetMainDiag(matLU_d, *Adiag);
   }

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
   */

   /* compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* apply permutation */
   NALU_HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);

   /* apply the iterative solve to L and U */
   nalu_hypre_ILUSolveLUJacobiIter(matLU_d, ftemp_local, xtemp_local, utemp_local, *Adiag_diag,
                              lower_jacobi_iters, upper_jacobi_iters, my_id);

   /* apply reverse permutation */
   NALU_HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);

   /* Update solution */
   nalu_hypre_ParVectorAxpy(beta, ftemp, u);

   return nalu_hypre_error_flag;
}

#endif
