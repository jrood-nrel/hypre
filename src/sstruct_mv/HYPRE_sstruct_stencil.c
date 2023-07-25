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

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilCreate( NALU_HYPRE_Int             ndim,
                            NALU_HYPRE_Int             size,
                            NALU_HYPRE_SStructStencil *stencil_ptr )
{
   nalu_hypre_SStructStencil  *stencil;
   nalu_hypre_StructStencil   *sstencil;
   NALU_HYPRE_Int             *vars;

   stencil = nalu_hypre_TAlloc(nalu_hypre_SStructStencil,  1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_StructStencilCreate(ndim, size, &sstencil);
   vars = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_StructStencilSize(sstencil), NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructStencilSStencil(stencil) = sstencil;
   nalu_hypre_SStructStencilVars(stencil)     = vars;
   nalu_hypre_SStructStencilRefCount(stencil) = 1;

   *stencil_ptr = stencil;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilDestroy( NALU_HYPRE_SStructStencil stencil )
{
   if (stencil)
   {
      nalu_hypre_SStructStencilRefCount(stencil) --;
      if (nalu_hypre_SStructStencilRefCount(stencil) == 0)
      {
         NALU_HYPRE_StructStencilDestroy(nalu_hypre_SStructStencilSStencil(stencil));
         nalu_hypre_TFree(nalu_hypre_SStructStencilVars(stencil), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(stencil, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilSetEntry( NALU_HYPRE_SStructStencil  stencil,
                              NALU_HYPRE_Int             entry,
                              NALU_HYPRE_Int            *offset,
                              NALU_HYPRE_Int             var )
{
   nalu_hypre_StructStencil  *sstencil = nalu_hypre_SStructStencilSStencil(stencil);

   NALU_HYPRE_StructStencilSetElement(sstencil, entry, offset);
   nalu_hypre_SStructStencilVar(stencil, entry) = var;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilPrint( FILE *file, NALU_HYPRE_SStructStencil stencil )
{
   NALU_HYPRE_Int    ndim  = nalu_hypre_SStructStencilNDim(stencil);
   NALU_HYPRE_Int   *vars  = nalu_hypre_SStructStencilVars(stencil);
   nalu_hypre_Index *shape = nalu_hypre_SStructStencilShape(stencil);
   NALU_HYPRE_Int    size  = nalu_hypre_SStructStencilSize(stencil);

   NALU_HYPRE_Int    i;

   nalu_hypre_fprintf(file, "StencilCreate: %d %d", ndim, size);
   for (i = 0; i < size; i++)
   {
      nalu_hypre_fprintf(file, "\nStencilSetEntry: %d %d ", i, vars[i]);
      nalu_hypre_IndexPrint(file, ndim, shape[i]);
   }
   nalu_hypre_fprintf(file, "\n");

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructStencilRead( FILE *file, NALU_HYPRE_SStructStencil *stencil_ptr )
{
   NALU_HYPRE_SStructStencil    stencil;

   NALU_HYPRE_Int               var;
   nalu_hypre_Index             shape;
   NALU_HYPRE_Int               i, ndim;
   NALU_HYPRE_Int               entry, size;

   nalu_hypre_fscanf(file, "StencilCreate: %d %d", &ndim, &size);
   NALU_HYPRE_SStructStencilCreate(ndim, size, &stencil);

   for (i = 0; i < size; i++)
   {
      nalu_hypre_fscanf(file, "\nStencilSetEntry: %d %d ", &entry, &var);
      nalu_hypre_IndexRead(file, ndim, shape);

      NALU_HYPRE_SStructStencilSetEntry(stencil, entry, shape, var);
   }
   nalu_hypre_fscanf(file, "\n");

   *stencil_ptr = stencil;

   return nalu_hypre_error_flag;
}
