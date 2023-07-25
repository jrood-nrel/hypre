/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_MATVEC_FUNCTIONS
#define NALU_HYPRE_MATVEC_FUNCTIONS

typedef struct
{
   void*  (*MatvecCreate)     ( void *A, void *x );
   NALU_HYPRE_Int (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                void *x, NALU_HYPRE_Complex beta, void *y );
   NALU_HYPRE_Int (*MatvecDestroy) ( void *matvec_data );

   void*  (*MatMultiVecCreate)     ( void *A, void *x );
   NALU_HYPRE_Int (*MatMultiVec)        ( void *data, NALU_HYPRE_Complex alpha, void *A,
                                     void *x, NALU_HYPRE_Complex beta, void *y );
   NALU_HYPRE_Int (*MatMultiVecDestroy) ( void *data );

} NALU_HYPRE_MatvecFunctions;

#endif
