/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_AME_HEADER
#define nalu_hypre_AME_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary space Maxwell Eigensolver
 *--------------------------------------------------------------------------*/
typedef struct
{
   /* The AMS preconditioner */
   nalu_hypre_AMSData *precond;

   /* The edge element mass matrix */
   nalu_hypre_ParCSRMatrix *M;

   /* Discrete gradient matrix with eliminated boundary */
   nalu_hypre_ParCSRMatrix *G;
   /* The Laplacian matrix G^t M G */
   nalu_hypre_ParCSRMatrix *A_G;
   /* AMG preconditioner for A_G */
   NALU_HYPRE_Solver B1_G;
   /* PCG-AMG solver for A_G */
   NALU_HYPRE_Solver B2_G;

   /* Eigensystem for A x = lambda M x, G^t x = 0 */
   NALU_HYPRE_Int block_size;
   void *eigenvectors;
   NALU_HYPRE_Real *eigenvalues;

   /* Eigensolver (LOBPCG) options */
   NALU_HYPRE_Int pcg_maxit;
   NALU_HYPRE_Int maxit;
   NALU_HYPRE_Real atol;
   NALU_HYPRE_Real rtol;
   NALU_HYPRE_Int print_level;

   /* Matrix-vector interface interpreter */
   void *interpreter;

   /* Temporary vectors */
   nalu_hypre_ParVector *t1, *t2, *t3;

} nalu_hypre_AMEData;

#include "_nalu_hypre_lapack.h"

#endif
