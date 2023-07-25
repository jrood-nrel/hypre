/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSolveLUDevice
 *
 * Incomplete LU solve (GPU)
 *
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
 *
 * TODO (VPM): Merge this function with nalu_hypre_ILUSolveLUIterDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveLUDevice(nalu_hypre_ParCSRMatrix  *A,
                       nalu_hypre_CSRMatrix     *matLU_d,
                       nalu_hypre_ParVector     *f,
                       nalu_hypre_ParVector     *u,
                       NALU_HYPRE_Int           *perm,
                       nalu_hypre_ParVector     *ftemp,
                       nalu_hypre_ParVector     *utemp)
{
   NALU_HYPRE_Int            num_rows      = nalu_hypre_ParCSRMatrixNumRows(A);

   nalu_hypre_Vector        *utemp_local   = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Complex       *utemp_data    = nalu_hypre_VectorData(utemp_local);
   nalu_hypre_Vector        *ftemp_local   = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Complex       *ftemp_data    = nalu_hypre_VectorData(ftemp_local);

   NALU_HYPRE_Complex        alpha = -1.0;
   NALU_HYPRE_Complex        beta  = 1.0;

   /* Sanity check */
   if (num_rows == 0)
   {
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   nalu_hypre_GpuProfilingPushRange("ILUSolve");

   /* Compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* Apply permutation */
   if (perm)
   {
      NALU_HYPRE_THRUST_CALL(gather, perm, perm + num_rows, ftemp_data, utemp_data);
   }
   else
   {
      nalu_hypre_TMemcpy(utemp_data, ftemp_data, NALU_HYPRE_Complex, num_rows,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* L solve - Forward solve */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matLU_d, NULL, utemp_local, ftemp_local);

   /* U solve - Backward substitution */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matLU_d, NULL, ftemp_local, utemp_local);

   /* Apply reverse permutation */
   if (perm)
   {
      NALU_HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + num_rows, perm, ftemp_data);
   }
   else
   {
      nalu_hypre_TMemcpy(ftemp_data, utemp_data, NALU_HYPRE_Complex, num_rows,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* Update solution */
   nalu_hypre_ParVectorAxpy(beta, ftemp, u);

   nalu_hypre_GpuProfilingPopRange();
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUApplyLowerJacIterDevice
 *
 * Incomplete L solve (Forward) of u^{k+1} = L^{-1}u^k on the GPU using the
 * Jacobi iterative approach.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUApplyLowerJacIterDevice(nalu_hypre_CSRMatrix *A,
                                 nalu_hypre_Vector    *input,
                                 nalu_hypre_Vector    *work,
                                 nalu_hypre_Vector    *output,
                                 NALU_HYPRE_Int        lower_jacobi_iters)
{
   NALU_HYPRE_Complex   *input_data  = nalu_hypre_VectorData(input);
   NALU_HYPRE_Complex   *work_data   = nalu_hypre_VectorData(work);
   NALU_HYPRE_Complex   *output_data = nalu_hypre_VectorData(output);
   NALU_HYPRE_Int        num_rows    = nalu_hypre_CSRMatrixNumRows(A);

   NALU_HYPRE_Int        kk = 0;

   /* Since the initial guess to the jacobi iteration is 0, the result of
      the first L SpMV is 0, so no need to compute.
      However, we still need to compute the transformation */
   hypreDevice_ComplexAxpyn(work_data, num_rows, input_data, output_data, 0.0);

   /* Do the remaining iterations */
   for (kk = 1; kk < lower_jacobi_iters; kk++)
   {
      /* Apply SpMV */
      nalu_hypre_CSRMatrixSpMVDevice(0, 1.0, A, output, 0.0, work, -2);

      /* Transform */
      hypreDevice_ComplexAxpyn(work_data, num_rows, input_data, output_data, -1.0);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUApplyUpperJacIterDevice
 *
 * Incomplete U solve (Backward) of u^{k+1} = U^{-1}u^k on the GPU using the
 * Jacobi iterative approach.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUApplyUpperJacIterDevice(nalu_hypre_CSRMatrix *A,
                                 nalu_hypre_Vector    *input,
                                 nalu_hypre_Vector    *work,
                                 nalu_hypre_Vector    *output,
                                 nalu_hypre_Vector    *diag,
                                 NALU_HYPRE_Int        upper_jacobi_iters)
{
   NALU_HYPRE_Complex   *output_data    = nalu_hypre_VectorData(output);
   NALU_HYPRE_Complex   *work_data      = nalu_hypre_VectorData(work);
   NALU_HYPRE_Complex   *input_data     = nalu_hypre_VectorData(input);
   NALU_HYPRE_Complex   *diag_data      = nalu_hypre_VectorData(diag);
   NALU_HYPRE_Int        num_rows       = nalu_hypre_CSRMatrixNumRows(A);

   NALU_HYPRE_Int        kk = 0;

   /* Since the initial guess to the jacobi iteration is 0,
      the result of the first U SpMV is 0, so no need to compute.
      However, we still need to compute the transformation */
   hypreDevice_zeqxmydd(num_rows, input_data, 0.0, work_data, output_data, diag_data);

   /* Do the remaining iterations */
   for (kk = 1; kk < upper_jacobi_iters; kk++)
   {
      /* apply SpMV */
      nalu_hypre_CSRMatrixSpMVDevice(0, 1.0, A, output, 0.0, work, 2);

      /* transform */
      hypreDevice_zeqxmydd(num_rows, input_data, -1.0, work_data, output_data, diag_data);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUApplyLowerUpperJacIterDevice
 *
 * Incomplete LU solve of u^{k+1} = U^{-1} L^{-1} u^k on the GPU using the
 * Jacobi iterative approach.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUApplyLowerUpperJacIterDevice(nalu_hypre_CSRMatrix *A,
                                      nalu_hypre_Vector    *work1,
                                      nalu_hypre_Vector    *work2,
                                      nalu_hypre_Vector    *inout,
                                      nalu_hypre_Vector    *diag,
                                      NALU_HYPRE_Int        lower_jacobi_iters,
                                      NALU_HYPRE_Int        upper_jacobi_iters)
{
   /* Apply the iterative solve to L */
   nalu_hypre_ILUApplyLowerJacIterDevice(A, inout, work1, work2, lower_jacobi_iters);

   /* Apply the iterative solve to U */
   nalu_hypre_ILUApplyUpperJacIterDevice(A, work2, work1, inout, diag, upper_jacobi_iters);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSolveLUIterDevice
 *
 * Incomplete LU solve using jacobi iterations on GPU.
 * L, D and U factors only have local scope (no off-diagonal processor terms).
 *
 * TODO (VPM): Merge this function with nalu_hypre_ILUSolveLUDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveLUIterDevice(nalu_hypre_ParCSRMatrix *A,
                           nalu_hypre_CSRMatrix    *matLU,
                           nalu_hypre_ParVector    *f,
                           nalu_hypre_ParVector    *u,
                           NALU_HYPRE_Int          *perm,
                           nalu_hypre_ParVector    *ftemp,
                           nalu_hypre_ParVector    *utemp,
                           nalu_hypre_ParVector    *xtemp,
                           nalu_hypre_Vector      **diag_ptr,
                           NALU_HYPRE_Int           lower_jacobi_iters,
                           NALU_HYPRE_Int           upper_jacobi_iters)
{
   NALU_HYPRE_Int        num_rows    = nalu_hypre_ParCSRMatrixNumRows(A);

   nalu_hypre_Vector    *diag        = *diag_ptr;
   nalu_hypre_Vector    *xtemp_local = nalu_hypre_ParVectorLocalVector(xtemp);
   nalu_hypre_Vector    *utemp_local = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Complex   *utemp_data  = nalu_hypre_VectorData(utemp_local);
   nalu_hypre_Vector    *ftemp_local = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Complex   *ftemp_data  = nalu_hypre_VectorData(ftemp_local);

   NALU_HYPRE_Complex    alpha = -1.0;
   NALU_HYPRE_Complex    beta  = 1.0;

   /* Sanity check */
   if (num_rows == 0)
   {
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   nalu_hypre_GpuProfilingPushRange("ILUSolveLUIter");

   /* Grab the main diagonal from the diagonal block. Only do this once */
   if (!diag)
   {
      /* Storage for the diagonal */
      diag = nalu_hypre_SeqVectorCreate(num_rows);
      nalu_hypre_SeqVectorInitialize(diag);

      /* extract with device kernel */
      nalu_hypre_CSRMatrixExtractDiagonalDevice(matLU, nalu_hypre_VectorData(diag), 2);

      /* Save output pointer */
      *diag_ptr = diag;
   }

   /* Compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* Apply permutation */
   if (perm)
   {
      NALU_HYPRE_THRUST_CALL(gather, perm, perm + num_rows, ftemp_data, utemp_data);
   }
   else
   {
      nalu_hypre_TMemcpy(utemp_data, ftemp_data, NALU_HYPRE_Complex, num_rows,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* Apply the iterative solve to L and U */
   nalu_hypre_ILUApplyLowerUpperJacIterDevice(matLU, ftemp_local, xtemp_local, utemp_local,
                                         diag, lower_jacobi_iters, upper_jacobi_iters);

   /* Apply reverse permutation */
   if (perm)
   {
      NALU_HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + num_rows, perm, ftemp_data);
   }
   else
   {
      nalu_hypre_TMemcpy(ftemp_data, utemp_data, NALU_HYPRE_Complex, num_rows,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* Update solution */
   nalu_hypre_ParVectorAxpy(beta, ftemp, u);

   nalu_hypre_GpuProfilingPopRange();
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUSchurGMRESMatvecDevice
 *
 * Slightly different, for this new matvec, the diagonal of the original
 * matrix is the LU factorization. Thus, the matvec is done in an different way
 *
 * |IS_1 E_12 E_13|
 * |E_21 IS_2 E_23| = S
 * |E_31 E_32 IS_3|
 *
 * |IS_1          |
 * |     IS_2     | = M
 * |          IS_3|
 *
 * Solve Sy = g is just M^{-1}S = M^{-1}g
 *
 * |      I       IS_1^{-1}E_12 IS_1^{-1}E_13|
 * |IS_2^{-1}E_21       I       IS_2^{-1}E_23| = M^{-1}S
 * |IS_3^{-1}E_31 IS_3^{-1}E_32       I      |
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUSchurGMRESMatvecDevice(void          *matvec_data,
                                   NALU_HYPRE_Complex  alpha,
                                   void          *ilu_vdata,
                                   void          *x,
                                   NALU_HYPRE_Complex  beta,
                                   void          *y)
{
   /* Get matrix information first */
   nalu_hypre_ParILUData    *ilu_data       = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix  *S              = nalu_hypre_ParILUDataMatS(ilu_data);
   nalu_hypre_CSRMatrix     *S_diag         = nalu_hypre_ParCSRMatrixDiag(S);

   /* Fist step, apply matvec on empty diagonal slot */
   NALU_HYPRE_Int            num_rows       = nalu_hypre_CSRMatrixNumRows(S_diag);
   NALU_HYPRE_Int            num_nonzeros   = nalu_hypre_CSRMatrixNumNonzeros(S_diag);

   nalu_hypre_ParVector     *xtemp          = nalu_hypre_ParILUDataXTemp(ilu_data);
   nalu_hypre_ParVector     *ytemp          = nalu_hypre_ParILUDataYTemp(ilu_data);
   nalu_hypre_Vector        *xtemp_local    = nalu_hypre_ParVectorLocalVector(xtemp);
   nalu_hypre_Vector        *ytemp_local    = nalu_hypre_ParVectorLocalVector(ytemp);

   /* Local variables */
   NALU_HYPRE_Complex        zero           = 0.0;
   NALU_HYPRE_Complex        one            = 1.0;

   /* Matvec with
    *         |  O  E_12 E_13|
    * alpha * |E_21   O  E_23|
    *         |E_31 E_32   O |
    * store in xtemp
    */

   /* RL: temp. set S_diag's nnz = 0 to skip the matvec
      (based on the assumption in seq_mv/csr_matvec impl.) */
   nalu_hypre_CSRMatrixNumRows(S_diag)     = 0;
   nalu_hypre_CSRMatrixNumNonzeros(S_diag) = 0;
   nalu_hypre_ParCSRMatrixMatvec(alpha, (nalu_hypre_ParCSRMatrix *) S, (nalu_hypre_ParVector *) x, zero, xtemp);
   nalu_hypre_CSRMatrixNumRows(S_diag)     = num_rows;
   nalu_hypre_CSRMatrixNumNonzeros(S_diag) = num_nonzeros;

   /* Compute U^{-1}*L^{-1}*(S_offd * x)
    * Or in other words, matvec with
    *         |      O       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * alpha * |IS_2^{-1}E_21       O       IS_2^{-1}E_23|
    *         |IS_3^{-1}E_31 IS_3^{-1}E_32       O      |
    * store in xtemp
    */

   /* L solve - Forward solve */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, S_diag, NULL, xtemp_local, ytemp_local);

   /* U solve - Backward substitution */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, S_diag, NULL, ytemp_local, xtemp_local);

   /* xtemp = xtemp + alpha*x */
   nalu_hypre_ParVectorAxpy(alpha, (nalu_hypre_ParVector *) x, xtemp);

   /* y = xtemp + beta*y */
   nalu_hypre_ParVectorAxpyz(one, xtemp, beta, (nalu_hypre_ParVector *) y, (nalu_hypre_ParVector *) y);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSolveSchurGMRESDevice
 *
 * Schur Complement solve with GMRES on schur complement
 *
 * ParCSRMatrix S is already built in ilu data sturcture, here directly use
 *  S, L, D and U factors only have local scope (no off-diag terms) so apart
 *  from the residual calculation (which uses A), the solves with the L and U
 *  factors are local.
 * S is the global Schur complement
 * schur_solver is a GMRES solver
 * schur_precond is the ILU preconditioner for GMRES
 * rhs and x are helper vectors for solving the Schur system
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveSchurGMRESDevice(nalu_hypre_ParCSRMatrix  *A,
                               nalu_hypre_ParVector     *f,
                               nalu_hypre_ParVector     *u,
                               NALU_HYPRE_Int           *perm,
                               NALU_HYPRE_Int            nLU,
                               nalu_hypre_ParCSRMatrix  *S,
                               nalu_hypre_ParVector     *ftemp,
                               nalu_hypre_ParVector     *utemp,
                               NALU_HYPRE_Solver         schur_solver,
                               NALU_HYPRE_Solver         schur_precond,
                               nalu_hypre_ParVector     *rhs,
                               nalu_hypre_ParVector     *x,
                               NALU_HYPRE_Int           *u_end,
                               nalu_hypre_CSRMatrix     *matBLU_d,
                               nalu_hypre_CSRMatrix     *matE_d,
                               nalu_hypre_CSRMatrix     *matF_d)
{
   /* If we don't have S block, just do one L solve and one U solve */
   if (!S)
   {
      return nalu_hypre_ILUSolveLUDevice(A, matBLU_d, f, u, perm, ftemp, utemp);
   }

   /* Data objects for temp vector */
   nalu_hypre_Vector      *utemp_local      = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real        *utemp_data       = nalu_hypre_VectorData(utemp_local);
   nalu_hypre_Vector      *ftemp_local      = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real        *ftemp_data       = nalu_hypre_VectorData(ftemp_local);
   nalu_hypre_Vector      *rhs_local        = nalu_hypre_ParVectorLocalVector(rhs);
   nalu_hypre_Vector      *x_local          = nalu_hypre_ParVectorLocalVector(x);
   NALU_HYPRE_Real        *x_data           = nalu_hypre_VectorData(x_local);

   /* Problem size */
   nalu_hypre_CSRMatrix   *matSLU_d         = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int          m                = nalu_hypre_CSRMatrixNumRows(matSLU_d);
   NALU_HYPRE_Int          n                = nLU + m;

   /* Local variables */
   NALU_HYPRE_Real         alpha            = -1.0;
   NALU_HYPRE_Real         beta             = 1.0;
   nalu_hypre_Vector      *ftemp_upper;
   nalu_hypre_Vector      *utemp_lower;

   /* Temporary vectors */
   ftemp_upper = nalu_hypre_SeqVectorCreate(nLU);
   utemp_lower = nalu_hypre_SeqVectorCreate(m);
   nalu_hypre_VectorOwnsData(ftemp_upper) = 0;
   nalu_hypre_VectorOwnsData(utemp_lower) = 0;
   nalu_hypre_VectorData(ftemp_upper) = ftemp_data;
   nalu_hypre_VectorData(utemp_lower) = utemp_data + nLU;
   nalu_hypre_SeqVectorInitialize(ftemp_upper);
   nalu_hypre_SeqVectorInitialize(utemp_lower);

   /* Compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */

   /* Apply permutation before we can start our solve */
   if (perm)
   {
      NALU_HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);
   }
   else
   {
      nalu_hypre_TMemcpy(utemp_data, ftemp_data, NALU_HYPRE_Complex, n,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* This solve won't touch data in utemp, thus, gi is still in utemp_lower */
   /* L solve - Forward solve */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL, utemp_local, ftemp_local);

   /* 2nd need to compute g'i = gi - Ei*UBi^{-1}*xi
    * Ei*UBi^{-1} is exactly the matE_d here
    * Now:  LBi^{-1}f_i is in ftemp_upper
    *       gi' is in utemp_lower
    */
   nalu_hypre_CSRMatrixMatvec(alpha, matE_d, ftemp_upper, beta, utemp_lower);

   /* 3rd need to solve global Schur Complement M^{-1}Sy = M^{-1}g'
    * for now only solve the local system
    * solve y put in u_temp lower
    * only solve whe S is not NULL
    */

   /* Setup vectors for solve
    * rhs = M^{-1}g'
    */

   /* L solve */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice_core('L', 1, matSLU_d, NULL, utemp_local,
                                                nLU, ftemp_local, nLU);

   /* U solve */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice_core('U', 0, matSLU_d, NULL, ftemp_local,
                                                nLU, rhs_local, 0);

   /* Solve with tricky initial guess */
   NALU_HYPRE_GMRESSolve(schur_solver,
                    (NALU_HYPRE_Matrix) schur_precond,
                    (NALU_HYPRE_Vector) rhs,
                    (NALU_HYPRE_Vector) x);

   /* 4th need to compute zi = xi - LBi^-1*yi
    * put zi in f_temp upper
    * only do this computation when nLU < n
    * U is unsorted, search is expensive when unnecessary
    */
   nalu_hypre_CSRMatrixMatvec(alpha, matF_d, x_local, beta, ftemp_upper);

   /* 5th need to solve UBi*ui = zi */
   /* put result in u_temp upper */
   /* U solve - Forward solve */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL, ftemp_local, utemp_local);

   /* Copy lower part solution into u_temp as well */
   nalu_hypre_TMemcpy(utemp_data + nLU, x_data, NALU_HYPRE_Real, m,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   /* Perm back */
   if (perm)
   {
      NALU_HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);
   }
   else
   {
      nalu_hypre_TMemcpy(ftemp_data, utemp_data, NALU_HYPRE_Complex, n,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* Done, now everything are in u_temp, update solution */
   nalu_hypre_ParVectorAxpy(beta, ftemp, u);

   /* Free memory */
   nalu_hypre_SeqVectorDestroy(ftemp_upper);
   nalu_hypre_SeqVectorDestroy(utemp_lower);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSolveSchurGMRESJacIterDevice
 *
 * Schur Complement solve with GMRES.
 *
 * ParCSRMatrix S is already built in the ilu data structure. S, L, D and U
 *  factors only have local scope (no off-diag terms). So apart from the
 *  residual calculation (which uses A), the solves with the L and U factors
 *  are local.
 * S: the global Schur complement
 * schur_solver: GMRES solver
 * schur_precond: ILU preconditioner for GMRES
 * rhs and x are helper vectors for solving the Schur system
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveSchurGMRESJacIterDevice(nalu_hypre_ParCSRMatrix *A,
                                      nalu_hypre_ParVector    *f,
                                      nalu_hypre_ParVector    *u,
                                      NALU_HYPRE_Int          *perm,
                                      NALU_HYPRE_Int           nLU,
                                      nalu_hypre_ParCSRMatrix *S,
                                      nalu_hypre_ParVector    *ftemp,
                                      nalu_hypre_ParVector    *utemp,
                                      NALU_HYPRE_Solver        schur_solver,
                                      NALU_HYPRE_Solver        schur_precond,
                                      nalu_hypre_ParVector    *rhs,
                                      nalu_hypre_ParVector    *x,
                                      NALU_HYPRE_Int          *u_end,
                                      nalu_hypre_CSRMatrix    *matBLU_d,
                                      nalu_hypre_CSRMatrix    *matE_d,
                                      nalu_hypre_CSRMatrix    *matF_d,
                                      nalu_hypre_ParVector    *ztemp,
                                      nalu_hypre_Vector      **Adiag_diag,
                                      nalu_hypre_Vector      **Sdiag_diag,
                                      NALU_HYPRE_Int           lower_jacobi_iters,
                                      NALU_HYPRE_Int           upper_jacobi_iters)
{
   /* If we don't have S block, just do one L solve and one U solve */
   if (!S)
   {
      return nalu_hypre_ILUSolveLUIterDevice(A, matBLU_d, f, u, perm,
                                        ftemp, utemp, ztemp, Adiag_diag,
                                        lower_jacobi_iters, upper_jacobi_iters);
   }

   /* Data objects for work vectors */
   nalu_hypre_Vector      *utemp_local = nalu_hypre_ParVectorLocalVector(utemp);
   nalu_hypre_Vector      *ftemp_local = nalu_hypre_ParVectorLocalVector(ftemp);
   nalu_hypre_Vector      *ztemp_local = nalu_hypre_ParVectorLocalVector(ztemp);
   nalu_hypre_Vector      *rhs_local   = nalu_hypre_ParVectorLocalVector(rhs);
   nalu_hypre_Vector      *x_local     = nalu_hypre_ParVectorLocalVector(x);

   NALU_HYPRE_Complex     *utemp_data  = nalu_hypre_VectorData(utemp_local);
   NALU_HYPRE_Complex     *ftemp_data  = nalu_hypre_VectorData(ftemp_local);
   NALU_HYPRE_Complex     *x_data      = nalu_hypre_VectorData(x_local);

   /* Problem size */
   nalu_hypre_CSRMatrix   *matSLU_d    = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int          m           = nalu_hypre_CSRMatrixNumRows(matSLU_d);
   NALU_HYPRE_Int          n           = nLU + m;

   /* Local variables */
   NALU_HYPRE_Complex      alpha = -1.0;
   NALU_HYPRE_Complex      beta  = 1.0;
   nalu_hypre_Vector      *ftemp_upper;
   nalu_hypre_Vector      *utemp_lower;
   nalu_hypre_Vector      *ftemp_shift;
   nalu_hypre_Vector      *utemp_shift;

   /* Set work vectors */
   ftemp_upper = nalu_hypre_SeqVectorCreate(nLU);
   utemp_lower = nalu_hypre_SeqVectorCreate(m);
   ftemp_shift = nalu_hypre_SeqVectorCreate(m);
   utemp_shift = nalu_hypre_SeqVectorCreate(m);

   nalu_hypre_VectorOwnsData(ftemp_upper) = 0;
   nalu_hypre_VectorOwnsData(utemp_lower) = 0;
   nalu_hypre_VectorOwnsData(ftemp_shift) = 0;
   nalu_hypre_VectorOwnsData(utemp_shift) = 0;

   nalu_hypre_VectorData(ftemp_upper) = ftemp_data;
   nalu_hypre_VectorData(utemp_lower) = utemp_data + nLU;
   nalu_hypre_VectorData(ftemp_shift) = ftemp_data + nLU;
   nalu_hypre_VectorData(utemp_shift) = utemp_data + nLU;

   nalu_hypre_SeqVectorInitialize(ftemp_upper);
   nalu_hypre_SeqVectorInitialize(utemp_lower);
   nalu_hypre_SeqVectorInitialize(ftemp_shift);
   nalu_hypre_SeqVectorInitialize(utemp_shift);

   /* Grab the main diagonal from the diagonal block. Only do this once */
   if (!(*Adiag_diag))
   {
      /* Storage for the diagonal */
      *Adiag_diag = nalu_hypre_SeqVectorCreate(n);
      nalu_hypre_SeqVectorInitialize(*Adiag_diag);

      /* Extract with device kernel */
      nalu_hypre_CSRMatrixExtractDiagonalDevice(matBLU_d, nalu_hypre_VectorData(*Adiag_diag), 2);
   }

   /* Compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */

   /* Apply permutation before we can start our solve */
   if (perm)
   {
      NALU_HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);
   }
   else
   {
      nalu_hypre_TMemcpy(utemp_data, ftemp_data, NALU_HYPRE_Complex, n,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   if (nLU > 0)
   {
      /* Apply the iterative solve to L */
      nalu_hypre_ILUApplyLowerJacIterDevice(matBLU_d, utemp_local, ztemp_local,
                                       ftemp_local, lower_jacobi_iters);

      /* 2nd need to compute g'i = gi - Ei*UBi^{-1}*xi
       * Ei*UBi^{-1} is exactly the matE_d here
       * Now:  LBi^{-1}f_i is in ftemp_upper
       *       gi' is in utemp_lower
       */
      nalu_hypre_CSRMatrixMatvec(alpha, matE_d, ftemp_upper, beta, utemp_lower);
   }

   /* 3rd need to solve global Schur Complement M^{-1}Sy = M^{-1}g'
    * for now only solve the local system
    * solve y put in u_temp lower
    * only solve whe S is not NULL
    */

   /* Setup vectors for solve
    * rhs = M^{-1}g'
    */
   if (m > 0)
   {
      /* Grab the main diagonal from the diagonal block. Only do this once */
      if (!(*Sdiag_diag))
      {
         /* Storage for the diagonal */
         *Sdiag_diag = nalu_hypre_SeqVectorCreate(m);
         nalu_hypre_SeqVectorInitialize(*Sdiag_diag);

         /* Extract with device kernel */
         nalu_hypre_CSRMatrixExtractDiagonalDevice(matSLU_d, nalu_hypre_VectorData(*Sdiag_diag), 2);
      }

      /* Apply the iterative solve to L */
      nalu_hypre_ILUApplyLowerJacIterDevice(matSLU_d, utemp_shift, rhs_local,
                                       ftemp_shift, lower_jacobi_iters);

      /* Apply the iterative solve to U */
      nalu_hypre_ILUApplyUpperJacIterDevice(matSLU_d, ftemp_shift, utemp_shift,
                                       rhs_local, *Sdiag_diag, upper_jacobi_iters);
   }

   /* Solve with tricky initial guess */
   NALU_HYPRE_GMRESSolve(schur_solver,
                    (NALU_HYPRE_Matrix) schur_precond,
                    (NALU_HYPRE_Vector) rhs,
                    (NALU_HYPRE_Vector) x);

   /* 4th need to compute zi = xi - LBi^-1*yi
    * put zi in f_temp upper
    * only do this computation when nLU < n
    * U is unsorted, search is expensive when unnecessary
    */
   if (nLU > 0)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, matF_d, x_local, beta, ftemp_upper);

      /* 5th need to solve UBi*ui = zi */
      /* put result in u_temp upper */

      /* Apply the iterative solve to U */
      nalu_hypre_ILUApplyUpperJacIterDevice(matBLU_d, ftemp_local, ztemp_local,
                                       utemp_local, *Adiag_diag, upper_jacobi_iters);
   }

   /* Copy lower part solution into u_temp as well */
   nalu_hypre_TMemcpy(utemp_data + nLU, x_data, NALU_HYPRE_Real, m,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   /* Perm back */
   if (perm)
   {
      NALU_HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);
   }
   else
   {
      nalu_hypre_TMemcpy(ftemp_data, utemp_data, NALU_HYPRE_Complex, n,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* Update solution */
   nalu_hypre_ParVectorAxpy(beta, ftemp, u);

   /* Free memory */
   nalu_hypre_SeqVectorDestroy(ftemp_shift);
   nalu_hypre_SeqVectorDestroy(utemp_shift);
   nalu_hypre_SeqVectorDestroy(ftemp_upper);
   nalu_hypre_SeqVectorDestroy(utemp_lower);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUSchurGMRESMatvecJacIterDevice
 *
 * Slightly different, for this new matvec, the diagonal of the original matrix
 * is the LU factorization. Thus, the matvec is done in an different way
 *
 * |IS_1 E_12 E_13|
 * |E_21 IS_2 E_23| = S
 * |E_31 E_32 IS_3|
 *
 * |IS_1          |
 * |     IS_2     | = M
 * |          IS_3|
 *
 * Solve Sy = g is just M^{-1}S = M^{-1}g
 *
 * |      I       IS_1^{-1}E_12 IS_1^{-1}E_13|
 * |IS_2^{-1}E_21       I       IS_2^{-1}E_23| = M^{-1}S
 * |IS_3^{-1}E_31 IS_3^{-1}E_32       I      |
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUSchurGMRESMatvecJacIterDevice(void          *matvec_data,
                                          NALU_HYPRE_Complex  alpha,
                                          void          *ilu_vdata,
                                          void          *x,
                                          NALU_HYPRE_Complex  beta,
                                          void          *y)
{
   /* get matrix information first */
   nalu_hypre_ParILUData    *ilu_data           = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix  *S                  = nalu_hypre_ParILUDataMatS(ilu_data);
   nalu_hypre_Vector        *Sdiag_diag         = nalu_hypre_ParILUDataSDiagDiag(ilu_data);
   NALU_HYPRE_Int            lower_jacobi_iters = nalu_hypre_ParILUDataLowerJacobiIters(ilu_data);
   NALU_HYPRE_Int            upper_jacobi_iters = nalu_hypre_ParILUDataUpperJacobiIters(ilu_data);

   /* fist step, apply matvec on empty diagonal slot */
   nalu_hypre_CSRMatrix     *S_diag            = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int            S_diag_n          = nalu_hypre_CSRMatrixNumRows(S_diag);
   NALU_HYPRE_Int            S_diag_nnz        = nalu_hypre_CSRMatrixNumNonzeros(S_diag);

   nalu_hypre_ParVector     *xtemp             = nalu_hypre_ParILUDataXTemp(ilu_data);
   nalu_hypre_Vector        *xtemp_local       = nalu_hypre_ParVectorLocalVector(xtemp);
   nalu_hypre_ParVector     *ytemp             = nalu_hypre_ParILUDataYTemp(ilu_data);
   nalu_hypre_Vector        *ytemp_local       = nalu_hypre_ParVectorLocalVector(ytemp);
   nalu_hypre_ParVector     *ztemp             = nalu_hypre_ParILUDataZTemp(ilu_data);
   nalu_hypre_Vector        *ztemp_local       = nalu_hypre_ParVectorLocalVector(ztemp);
   NALU_HYPRE_Real           zero              = 0.0;
   NALU_HYPRE_Real           one               = 1.0;

   /* Matvec with
    *         |  O  E_12 E_13|
    * alpha * |E_21   O  E_23|
    *         |E_31 E_32   O |
    * store in xtemp
    */

   /* RL: temp. set S_diag's nnz = 0 to skip the matvec
      (based on the assumption in seq_mv/csr_matvec impl.) */
   nalu_hypre_CSRMatrixNumRows(S_diag)     = 0;
   nalu_hypre_CSRMatrixNumNonzeros(S_diag) = 0;
   nalu_hypre_ParCSRMatrixMatvec(alpha, (nalu_hypre_ParCSRMatrix *) S, (nalu_hypre_ParVector *) x, zero, xtemp);
   nalu_hypre_CSRMatrixNumRows(S_diag)     = S_diag_n;
   nalu_hypre_CSRMatrixNumNonzeros(S_diag) = S_diag_nnz;

   /* Grab the main diagonal from the diagonal block. Only do this once */
   if (!Sdiag_diag)
   {
      /* Storage for the diagonal */
      Sdiag_diag = nalu_hypre_SeqVectorCreate(S_diag_n);
      nalu_hypre_SeqVectorInitialize(Sdiag_diag);

      /* Extract with device kernel */
      nalu_hypre_CSRMatrixExtractDiagonalDevice(S_diag, nalu_hypre_VectorData(Sdiag_diag), 2);

      /* Save Schur diagonal */
      nalu_hypre_ParILUDataSDiagDiag(ilu_data) = Sdiag_diag;
   }

   /* Compute U^{-1}*L^{-1}*(A_offd * x)
    * Or in another words, matvec with
    *         |      O       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * alpha * |IS_2^{-1}E_21       O       IS_2^{-1}E_23|
    *         |IS_3^{-1}E_31 IS_3^{-1}E_32       O      |
    * store in xtemp
    */
   if (S_diag_n)
   {
      /* apply the iterative solve to L and U */
      nalu_hypre_ILUApplyLowerUpperJacIterDevice(S_diag, ytemp_local, ztemp_local,
                                            xtemp_local, Sdiag_diag,
                                            lower_jacobi_iters, upper_jacobi_iters);
   }

   /* now add the original x onto it */
   nalu_hypre_ParVectorAxpy(alpha, (nalu_hypre_ParVector *) x, xtemp);

   /* y = xtemp + beta*y */
   nalu_hypre_ParVectorAxpyz(one, xtemp, beta, (nalu_hypre_ParVector *) y, (nalu_hypre_ParVector *) y);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ILUSolveRAPGMRESDevice
 *
 * Device solve with GMRES on schur complement, RAP style.
 *
 * See nalu_hypre_ILUSolveRAPGMRESHost for more comments
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveRAPGMRESDevice(nalu_hypre_ParCSRMatrix   *A,
                             nalu_hypre_ParVector      *f,
                             nalu_hypre_ParVector      *u,
                             NALU_HYPRE_Int            *perm,
                             NALU_HYPRE_Int             nLU,
                             nalu_hypre_ParCSRMatrix   *S,
                             nalu_hypre_ParVector      *ftemp,
                             nalu_hypre_ParVector      *utemp,
                             nalu_hypre_ParVector      *xtemp,
                             nalu_hypre_ParVector      *ytemp,
                             NALU_HYPRE_Solver          schur_solver,
                             NALU_HYPRE_Solver          schur_precond,
                             nalu_hypre_ParVector      *rhs,
                             nalu_hypre_ParVector      *x,
                             NALU_HYPRE_Int            *u_end,
                             nalu_hypre_ParCSRMatrix   *Aperm,
                             nalu_hypre_CSRMatrix      *matALU_d,
                             nalu_hypre_CSRMatrix      *matBLU_d,
                             nalu_hypre_CSRMatrix      *matE_d,
                             nalu_hypre_CSRMatrix      *matF_d,
                             NALU_HYPRE_Int             test_opt)
{
   /* If we don't have S block, just do one L/U solve */
   if (!S)
   {
      return nalu_hypre_ILUSolveLUDevice(A, matBLU_d, f, u, perm, ftemp, utemp);
   }

   /* data objects for vectors */
   nalu_hypre_Vector      *utemp_local = nalu_hypre_ParVectorLocalVector(utemp);
   nalu_hypre_Vector      *ftemp_local = nalu_hypre_ParVectorLocalVector(ftemp);
   nalu_hypre_Vector      *xtemp_local = nalu_hypre_ParVectorLocalVector(xtemp);
   nalu_hypre_Vector      *rhs_local   = nalu_hypre_ParVectorLocalVector(rhs);
   nalu_hypre_Vector      *x_local     = nalu_hypre_ParVectorLocalVector(x);

   NALU_HYPRE_Complex     *utemp_data  = nalu_hypre_VectorData(utemp_local);
   NALU_HYPRE_Complex     *ftemp_data  = nalu_hypre_VectorData(ftemp_local);
   NALU_HYPRE_Complex     *xtemp_data  = nalu_hypre_VectorData(xtemp_local);
   NALU_HYPRE_Complex     *rhs_data    = nalu_hypre_VectorData(rhs_local);
   NALU_HYPRE_Complex     *x_data      = nalu_hypre_VectorData(x_local);

   nalu_hypre_CSRMatrix   *matSLU_d    = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int          m           = nalu_hypre_CSRMatrixNumRows(matSLU_d);
   NALU_HYPRE_Int          n           = nLU + m;
   NALU_HYPRE_Real         one         = 1.0;
   NALU_HYPRE_Real         mone        = -1.0;
   NALU_HYPRE_Real         zero        = 0.0;

   /* Temporary vectors */
   nalu_hypre_Vector      *ftemp_upper;
   nalu_hypre_Vector      *utemp_lower;

   /* Create temporary vectors */
   ftemp_upper = nalu_hypre_SeqVectorCreate(nLU);
   utemp_lower = nalu_hypre_SeqVectorCreate(m);

   nalu_hypre_VectorOwnsData(ftemp_upper) = 0;
   nalu_hypre_VectorOwnsData(utemp_lower) = 0;
   nalu_hypre_VectorData(ftemp_upper)     = ftemp_data;
   nalu_hypre_VectorData(utemp_lower)     = utemp_data + nLU;

   nalu_hypre_SeqVectorInitialize(ftemp_upper);
   nalu_hypre_SeqVectorInitialize(utemp_lower);

   switch (test_opt)
   {
      case 1: case 3:
      {
         /* E and F */
         /* compute residual */
         nalu_hypre_ParCSRMatrixMatvecOutOfPlace(mone, A, u, one, f, utemp);

         /* apply permutation before we can start our solve
          * Au=f -> (PAQ)Q'u=Pf
          */
         if (perm)
         {
            NALU_HYPRE_THRUST_CALL(gather, perm, perm + n, utemp_data, ftemp_data);
         }
         else
         {
            nalu_hypre_TMemcpy(ftemp_data, utemp_data, NALU_HYPRE_Complex, n,
                          NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         }

         /* A-smoothing
          * x = [UA\(LA\(P*f_u))] fill to xtemp
          */

         /* L solve - Forward solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matALU_d, NULL,
                                                 ftemp_local, utemp_local);

         /* U solve - Backward solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matALU_d, NULL,
                                                 utemp_local, xtemp_local);

         /* residual, we should not touch xtemp for now
          * r = R*(f-PAQx)
          */
         nalu_hypre_ParCSRMatrixMatvec(mone, Aperm, xtemp, one, ftemp);

         /* with R is complex */
         /* copy partial data in */
         nalu_hypre_TMemcpy(rhs_data, ftemp_data + nLU, NALU_HYPRE_Real, m,
                       NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

         /* solve L^{-1} */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL,
                                                 ftemp_local, utemp_local);

         /* -U^{-1}L^{-1} */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                 utemp_local, ftemp_local);

         /* -EU^{-1}L^{-1} */
         nalu_hypre_CSRMatrixMatvec(mone, matE_d, ftemp_upper, one, rhs_local);

         /* Solve S */
         if (S)
         {
            /* if we have a schur complement */
            nalu_hypre_ParVectorSetConstantValues(x, 0.0);
            NALU_HYPRE_GMRESSolve(schur_solver,
                             (NALU_HYPRE_Matrix) schur_precond,
                             (NALU_HYPRE_Vector) rhs,
                             (NALU_HYPRE_Vector) x);

            /* u = xtemp + P*x */
            /* -Fx */
            nalu_hypre_CSRMatrixMatvec(mone, matF_d, x_local, zero, ftemp_upper);

            /* -L^{-1}Fx */
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL,
                                                    ftemp_local, utemp_local);

            /* -U{-1}L^{-1}Fx */
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                    utemp_local, ftemp_local);

            /* now copy data to y_lower */
            nalu_hypre_TMemcpy(ftemp_data + nLU, x_data, NALU_HYPRE_Real, m,
                          NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         }
         else
         {
            /* otherwise just apply triangular solves */
            /* L solve - Forward solve */
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matSLU_d, NULL, rhs_local, x_local);

            /* U solve - Backward solve */
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matSLU_d, NULL, x_local, rhs_local);

            /* u = xtemp + P*x */
            /* -Fx */
            nalu_hypre_CSRMatrixMatvec(mone, matF_d, rhs_local, zero, ftemp_upper);

            /* -L^{-1}Fx */
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL,
                                                    ftemp_local, utemp_local);

            /* -U{-1}L^{-1}Fx */
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                    utemp_local, ftemp_local);

            /* now copy data to y_lower */
            nalu_hypre_TMemcpy(ftemp_data + nLU, rhs_data, NALU_HYPRE_Real, m,
                          NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         }

         /* correction to the residual */
         nalu_hypre_ParVectorAxpy(one, ftemp, xtemp);

         /* perm back */
         if (perm)
         {
            NALU_HYPRE_THRUST_CALL(scatter, xtemp_data, xtemp_data + n, perm, ftemp_data);
         }
         else
         {
            nalu_hypre_TMemcpy(ftemp_data, xtemp_data, NALU_HYPRE_Complex, n,
                          NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         }
      }
      break;

   case 0: case 2: default:
      {
         /* EU^{-1} and L^{-1}F */
         /* compute residual */
         nalu_hypre_ParCSRMatrixMatvecOutOfPlace(mone, A, u, one, f, ftemp);

         /* apply permutation before we can start our solve
          * Au=f -> (PAQ)Q'u=Pf
          */
         if (perm)
         {
            NALU_HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);
         }
         else
         {
            nalu_hypre_TMemcpy(utemp_data, ftemp_data, NALU_HYPRE_Complex, n,
                          NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         }

         /* A-smoothing
          * x = [UA\(LA\(P*f_u))] fill to xtemp
          */

         /* L solve - Forward solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matALU_d, NULL,
                                                 utemp_local, ftemp_local);

         /* U solve - Backward solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matALU_d, NULL,
                                                 ftemp_local, xtemp_local);

         /* residual, we should not touch xtemp for now
          * r = R*(f-PAQx)
          */
         nalu_hypre_ParCSRMatrixMatvec(mone, Aperm, xtemp, one, utemp);

         /* with R is complex */
         /* copy partial data in */
         nalu_hypre_TMemcpy(rhs_data, utemp_data + nLU, NALU_HYPRE_Real, m,
                       NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

         /* solve L^{-1} */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL,
                                                 utemp_local, ftemp_local);

         /* -EU^{-1}L^{-1} */
         nalu_hypre_CSRMatrixMatvec(mone, matE_d, ftemp_upper, one, rhs_local);

         /* Solve S */
         if (S)
         {
            /* if we have a schur complement */
            nalu_hypre_ParVectorSetConstantValues(x, 0.0);
            NALU_HYPRE_GMRESSolve(schur_solver,
                             (NALU_HYPRE_Matrix) schur_precond,
                             (NALU_HYPRE_Vector) rhs,
                             (NALU_HYPRE_Vector) x);

            /* u = xtemp + P*x */
            /* -L^{-1}Fx */
            nalu_hypre_CSRMatrixMatvec(mone, matF_d, x_local, zero, ftemp_upper);

            /* -U{-1}L^{-1}Fx */
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                    ftemp_local, utemp_local);

            /* now copy data to y_lower */
            nalu_hypre_TMemcpy(utemp_data + nLU, x_data, NALU_HYPRE_Real, m,
                          NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         }
         else
         {
            /* otherwise just apply triangular solves */
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matSLU_d, NULL, rhs_local, x_local);
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matSLU_d, NULL, x_local, rhs_local);

            /* u = xtemp + P*x */
            /* -L^{-1}Fx */
            nalu_hypre_CSRMatrixMatvec(mone, matF_d, rhs_local, zero, ftemp_upper);

            /* -U{-1}L^{-1}Fx */
            nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                    ftemp_local, utemp_local);

            /* now copy data to y_lower */
            nalu_hypre_TMemcpy(utemp_data + nLU, rhs_data, NALU_HYPRE_Real, m,
                          NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         }

         /* Update xtemp */
         nalu_hypre_ParVectorAxpy(one, utemp, xtemp);

         /* perm back */
         if (perm)
         {
            NALU_HYPRE_THRUST_CALL(scatter, xtemp_data, xtemp_data + n, perm, ftemp_data);
         }
         else
         {
            nalu_hypre_TMemcpy(ftemp_data, xtemp_data, NALU_HYPRE_Complex, n,
                          NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         }
      }
      break;
   }

   /* Done, now everything are in u_temp, update solution */
   nalu_hypre_ParVectorAxpy(one, ftemp, u);

   /* Destroy temporary vectors */
   nalu_hypre_SeqVectorDestroy(ftemp_upper);
   nalu_hypre_SeqVectorDestroy(utemp_lower);

   return nalu_hypre_error_flag;
}

#endif /* defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */
