/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_sstruct_ls.h"


NALU_HYPRE_Int nalu_hypre_SStructKrylovCopyVector( void *x, void *y );

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovIdentitySetup( void *vdata,
                                  void *A,
                                  void *b,
                                  void *x )

{
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovIdentity( void *vdata,
                             void *A,
                             void *b,
                             void *x )

{
   return ( nalu_hypre_SStructKrylovCopyVector(b, x) );
}

