/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for nalu_hypre_SStructStencil data structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_SSTRUCT_STENCIL_HEADER
#define nalu_hypre_SSTRUCT_STENCIL_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructStencil
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_SStructStencil_struct
{
   nalu_hypre_StructStencil  *sstencil;
   NALU_HYPRE_Int            *vars;

   NALU_HYPRE_Int             ref_count;

} nalu_hypre_SStructStencil;

/*--------------------------------------------------------------------------
 * Accessor functions for the nalu_hypre_SStructStencil structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructStencilSStencil(stencil)     ((stencil) -> sstencil)
#define nalu_hypre_SStructStencilVars(stencil)         ((stencil) -> vars)
#define nalu_hypre_SStructStencilVar(stencil, i)       ((stencil) -> vars[i])
#define nalu_hypre_SStructStencilRefCount(stencil)     ((stencil) -> ref_count)

#define nalu_hypre_SStructStencilShape(stencil) \
nalu_hypre_StructStencilShape( nalu_hypre_SStructStencilSStencil(stencil) )
#define nalu_hypre_SStructStencilSize(stencil) \
nalu_hypre_StructStencilSize( nalu_hypre_SStructStencilSStencil(stencil) )
#define nalu_hypre_SStructStencilNDim(stencil) \
nalu_hypre_StructStencilNDim( nalu_hypre_SStructStencilSStencil(stencil) )
#define nalu_hypre_SStructStencilEntry(stencil, i) \
nalu_hypre_StructStencilElement( nalu_hypre_SStructStencilSStencil(stencil), i )

#endif
