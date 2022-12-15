/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "multivector.h"
#include "_nalu_hypre_utilities.h"

#ifndef LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS
#define LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS

#ifdef __cplusplus
extern "C" {
#endif

#define PROBLEM_SIZE_TOO_SMALL                  1
#define WRONG_BLOCK_SIZE                  2
#define WRONG_CONSTRAINTS                               3
#define REQUESTED_ACCURACY_NOT_ACHIEVED         -1

typedef struct
{

   NALU_HYPRE_Real   absolute;
   NALU_HYPRE_Real   relative;

} lobpcg_Tolerance;

typedef struct
{

   /* these pointers should point to 2 functions providing standard lapack  functionality */
   NALU_HYPRE_Int   (*dpotrf) (const char *uplo, NALU_HYPRE_Int *n, NALU_HYPRE_Real *a, NALU_HYPRE_Int *
                          lda, NALU_HYPRE_Int *info);
   NALU_HYPRE_Int   (*dsygv) (NALU_HYPRE_Int *itype, char *jobz, char *uplo, NALU_HYPRE_Int *
                         n, NALU_HYPRE_Real *a, NALU_HYPRE_Int *lda, NALU_HYPRE_Real *b, NALU_HYPRE_Int *ldb,
                         NALU_HYPRE_Real *w, NALU_HYPRE_Real *work, NALU_HYPRE_Int *lwork, NALU_HYPRE_Int *info);

} lobpcg_BLASLAPACKFunctions;

NALU_HYPRE_Int
lobpcg_solve( mv_MultiVectorPtr blockVectorX,
              void* operatorAData,
              void (*operatorA)( void*, void*, void* ),
              void* operatorBData,
              void (*operatorB)( void*, void*, void* ),
              void* operatorTData,
              void (*operatorT)( void*, void*, void* ),
              mv_MultiVectorPtr blockVectorY,
              lobpcg_BLASLAPACKFunctions blap_fn,
              lobpcg_Tolerance tolerance,
              NALU_HYPRE_Int maxIterations,
              NALU_HYPRE_Int verbosityLevel,
              NALU_HYPRE_Int* iterationNumber,

              /* eigenvalues; "lambda_values" should point to array  containing <blocksize> doubles where <blocksi
              ze> is the width of multivector "blockVectorX" */
              NALU_HYPRE_Real * lambda_values,

              /* eigenvalues history; a pointer to the entries of the  <blocksize>-by-(<maxIterations>+1) matrix s
              tored
              in  fortran-style. (i.e. column-wise) The matrix may be  a submatrix of a larger matrix, see next
              argument; If you don't need eigenvalues history, provide NULL in this entry */
              NALU_HYPRE_Real * lambdaHistory_values,

              /* global height of the matrix (stored in fotran-style)  specified by previous argument */
              NALU_HYPRE_Int lambdaHistory_gh,

              /* residual norms; argument should point to array of <blocksize> doubles */
              NALU_HYPRE_Real * residualNorms_values,

              /* residual norms history; a pointer to the entries of the  <blocksize>-by-(<maxIterations>+1) matri
              x
              stored in  fortran-style. (i.e. column-wise) The matrix may be  a submatrix of a larger matrix, see
              next
              argument If you don't need residual norms history, provide NULL in this entry */
              NALU_HYPRE_Real * residualNormsHistory_values,

              /* global height of the matrix (stored in fotran-style)  specified by previous argument */
              NALU_HYPRE_Int residualNormsHistory_gh

            );

#ifdef __cplusplus
}
#endif

#endif /* LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS */
