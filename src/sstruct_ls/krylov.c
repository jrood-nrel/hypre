/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_sstruct_ls.h"


NALU_HYPRE_Int hypre_SStructKrylovCopyVector( void *x, void *y );

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovIdentitySetup( void *vdata,
                                  void *A,
                                  void *b,
                                  void *x )

{
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovIdentity( void *vdata,
                             void *A,
                             void *b,
                             void *x )

{
   return ( hypre_SStructKrylovCopyVector(b, x) );
}

