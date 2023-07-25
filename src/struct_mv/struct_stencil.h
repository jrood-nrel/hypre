/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for nalu_hypre_StructStencil data structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_STRUCT_STENCIL_HEADER
#define nalu_hypre_STRUCT_STENCIL_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_StructStencil
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_StructStencil_struct
{
   nalu_hypre_Index   *shape;   /* Description of a stencil's shape */
   NALU_HYPRE_Int      size;    /* Number of stencil coefficients */

   NALU_HYPRE_Int      ndim;    /* Number of dimensions */

   NALU_HYPRE_Int      ref_count;
} nalu_hypre_StructStencil;

/*--------------------------------------------------------------------------
 * Accessor functions for the nalu_hypre_StructStencil structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_StructStencilShape(stencil)      ((stencil) -> shape)
#define nalu_hypre_StructStencilSize(stencil)       ((stencil) -> size)
#define nalu_hypre_StructStencilNDim(stencil)       ((stencil) -> ndim)
#define nalu_hypre_StructStencilRefCount(stencil)   ((stencil) -> ref_count)
#define nalu_hypre_StructStencilElement(stencil, i) nalu_hypre_StructStencilShape(stencil)[i]

#endif
