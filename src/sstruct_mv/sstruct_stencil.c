/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructStencilRef
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructStencilRef( nalu_hypre_SStructStencil  *stencil,
                         nalu_hypre_SStructStencil **stencil_ref )
{
   nalu_hypre_SStructStencilRefCount(stencil) ++;
   *stencil_ref = stencil;

   return nalu_hypre_error_flag;
}
