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

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructStencilCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructStencilCreate( NALU_HYPRE_Int            dim,
                           NALU_HYPRE_Int            size,
                           NALU_HYPRE_StructStencil *stencil )
{
   hypre_Index  *shape;

   shape = hypre_CTAlloc(hypre_Index,  size, NALU_HYPRE_MEMORY_HOST);

   *stencil = hypre_StructStencilCreate(dim, size, shape);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructStencilSetElement
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructStencilSetElement( NALU_HYPRE_StructStencil  stencil,
                               NALU_HYPRE_Int            element_index,
                               NALU_HYPRE_Int           *offset )
{
   hypre_Index  *shape;
   NALU_HYPRE_Int     d;

   shape = hypre_StructStencilShape(stencil);
   hypre_SetIndex(shape[element_index], 0);
   for (d = 0; d < hypre_StructStencilNDim(stencil); d++)
   {
      hypre_IndexD(shape[element_index], d) = offset[d];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructStencilDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructStencilDestroy( NALU_HYPRE_StructStencil stencil )
{
   return ( hypre_StructStencilDestroy(stencil) );
}

