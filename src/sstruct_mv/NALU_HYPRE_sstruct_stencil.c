/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructStencil interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilCreate( NALU_HYPRE_Int             ndim,
                            NALU_HYPRE_Int             size,
                            NALU_HYPRE_SStructStencil *stencil_ptr )
{
   hypre_SStructStencil  *stencil;
   hypre_StructStencil   *sstencil;
   NALU_HYPRE_Int             *vars;

   stencil = hypre_TAlloc(hypre_SStructStencil,  1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_StructStencilCreate(ndim, size, &sstencil);
   vars = hypre_CTAlloc(NALU_HYPRE_Int,  hypre_StructStencilSize(sstencil), NALU_HYPRE_MEMORY_HOST);

   hypre_SStructStencilSStencil(stencil) = sstencil;
   hypre_SStructStencilVars(stencil)     = vars;
   hypre_SStructStencilRefCount(stencil) = 1;

   *stencil_ptr = stencil;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilDestroy( NALU_HYPRE_SStructStencil stencil )
{
   if (stencil)
   {
      hypre_SStructStencilRefCount(stencil) --;
      if (hypre_SStructStencilRefCount(stencil) == 0)
      {
         NALU_HYPRE_StructStencilDestroy(hypre_SStructStencilSStencil(stencil));
         hypre_TFree(hypre_SStructStencilVars(stencil), NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(stencil, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilSetEntry( NALU_HYPRE_SStructStencil  stencil,
                              NALU_HYPRE_Int             entry,
                              NALU_HYPRE_Int            *offset,
                              NALU_HYPRE_Int             var )
{
   hypre_StructStencil  *sstencil = hypre_SStructStencilSStencil(stencil);

   NALU_HYPRE_StructStencilSetElement(sstencil, entry, offset);
   hypre_SStructStencilVar(stencil, entry) = var;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilPrint( FILE *file, NALU_HYPRE_SStructStencil stencil )
{
   NALU_HYPRE_Int    ndim  = hypre_SStructStencilNDim(stencil);
   NALU_HYPRE_Int   *vars  = hypre_SStructStencilVars(stencil);
   hypre_Index *shape = hypre_SStructStencilShape(stencil);
   NALU_HYPRE_Int    size  = hypre_SStructStencilSize(stencil);

   NALU_HYPRE_Int    i;

   hypre_fprintf(file, "StencilCreate: %d %d", ndim, size);
   for (i = 0; i < size; i++)
   {
      hypre_fprintf(file, "\nStencilSetEntry: %d %d ", i, vars[i]);
      hypre_IndexPrint(file, ndim, shape[i]);
   }
   hypre_fprintf(file, "\n");

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilRead( FILE *file, NALU_HYPRE_SStructStencil *stencil_ptr )
{
   NALU_HYPRE_SStructStencil    stencil;

   NALU_HYPRE_Int               var;
   hypre_Index             shape;
   NALU_HYPRE_Int               i, ndim;
   NALU_HYPRE_Int               entry, size;

   hypre_fscanf(file, "StencilCreate: %d %d", &ndim, &size);
   NALU_HYPRE_SStructStencilCreate(ndim, size, &stencil);

   for (i = 0; i < size; i++)
   {
      hypre_fscanf(file, "\nStencilSetEntry: %d %d ", &entry, &var);
      hypre_IndexRead(file, ndim, shape);

      NALU_HYPRE_SStructStencilSetEntry(stencil, entry, shape, var);
   }
   hypre_fscanf(file, "\n");

   *stencil_ptr = stencil;

   return hypre_error_flag;
}
