/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructStencil interface
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructStencilCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructStencilCreate( NALU_HYPRE_Int            dim,
                           NALU_HYPRE_Int            size,
                           NALU_HYPRE_StructStencil *stencil )
{
   nalu_hypre_Index  *shape;

   shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  size, NALU_HYPRE_MEMORY_HOST);

   *stencil = nalu_hypre_StructStencilCreate(dim, size, shape);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructStencilSetElement
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructStencilSetElement( NALU_HYPRE_StructStencil  stencil,
                               NALU_HYPRE_Int            element_index,
                               NALU_HYPRE_Int           *offset )
{
   nalu_hypre_Index  *shape;
   NALU_HYPRE_Int     d;

   shape = nalu_hypre_StructStencilShape(stencil);
   nalu_hypre_SetIndex(shape[element_index], 0);
   for (d = 0; d < nalu_hypre_StructStencilNDim(stencil); d++)
   {
      nalu_hypre_IndexD(shape[element_index], d) = offset[d];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructStencilDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructStencilDestroy( NALU_HYPRE_StructStencil stencil )
{
   return ( nalu_hypre_StructStencilDestroy(stencil) );
}

