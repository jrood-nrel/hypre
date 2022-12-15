/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Constructors and destructors for stencil structure.
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_StructStencilCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_StructStencil *
nalu_hypre_StructStencilCreate( NALU_HYPRE_Int     dim,
                           NALU_HYPRE_Int     size,
                           nalu_hypre_Index  *shape )
{
   nalu_hypre_StructStencil   *stencil;

   stencil = nalu_hypre_TAlloc(nalu_hypre_StructStencil, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_StructStencilShape(stencil)    = shape;
   nalu_hypre_StructStencilSize(stencil)     = size;
   nalu_hypre_StructStencilNDim(stencil)      = dim;
   nalu_hypre_StructStencilRefCount(stencil) = 1;

   return stencil;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructStencilRef
 *--------------------------------------------------------------------------*/

nalu_hypre_StructStencil *
nalu_hypre_StructStencilRef( nalu_hypre_StructStencil *stencil )
{
   nalu_hypre_StructStencilRefCount(stencil) ++;

   return stencil;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructStencilDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructStencilDestroy( nalu_hypre_StructStencil *stencil )
{
   if (stencil)
   {
      nalu_hypre_StructStencilRefCount(stencil) --;
      if (nalu_hypre_StructStencilRefCount(stencil) == 0)
      {
         nalu_hypre_TFree(nalu_hypre_StructStencilShape(stencil), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(stencil, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructStencilElementRank
 *    Returns the rank of the `stencil_element' in `stencil'.
 *    If the element is not found, a -1 is returned.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructStencilElementRank( nalu_hypre_StructStencil *stencil,
                                nalu_hypre_Index          stencil_element )
{
   nalu_hypre_Index  *stencil_shape;
   NALU_HYPRE_Int     rank;
   NALU_HYPRE_Int     i, ndim;

   rank = -1;
   ndim = nalu_hypre_StructStencilNDim(stencil);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   for (i = 0; i < nalu_hypre_StructStencilSize(stencil); i++)
   {
      if (nalu_hypre_IndexesEqual(stencil_shape[i], stencil_element, ndim))
      {
         rank = i;
         break;
      }
   }

   return rank;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructStencilSymmetrize:
 *    Computes a new "symmetrized" stencil.
 *
 *    An integer array called `symm_elements' is also set up.  A non-negative
 *    value of `symm_elements[i]' indicates that the `i'th stencil element
 *    is a "symmetric element".  That is, this stencil element is the
 *    transpose element of an element that is not a "symmetric element".
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructStencilSymmetrize( nalu_hypre_StructStencil  *stencil,
                               nalu_hypre_StructStencil **symm_stencil_ptr,
                               NALU_HYPRE_Int           **symm_elements_ptr )
{
   nalu_hypre_Index          *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   NALU_HYPRE_Int             stencil_size  = nalu_hypre_StructStencilSize(stencil);

   nalu_hypre_StructStencil  *symm_stencil;
   nalu_hypre_Index          *symm_stencil_shape;
   NALU_HYPRE_Int             symm_stencil_size;
   NALU_HYPRE_Int            *symm_elements;

   NALU_HYPRE_Int             no_symmetric_stencil_element, symmetric;
   NALU_HYPRE_Int             i, j, d, ndim;

   /*------------------------------------------------------
    * Copy stencil elements into `symm_stencil_shape'
    *------------------------------------------------------*/

   ndim = nalu_hypre_StructStencilNDim(stencil);
   symm_stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  2 * stencil_size, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      nalu_hypre_CopyIndex(stencil_shape[i], symm_stencil_shape[i]);
   }

   /*------------------------------------------------------
    * Create symmetric stencil elements and `symm_elements'
    *------------------------------------------------------*/

   symm_elements = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2 * stencil_size, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < 2 * stencil_size; i++)
   {
      symm_elements[i] = -1;
   }

   symm_stencil_size = stencil_size;
   for (i = 0; i < stencil_size; i++)
   {
      if (symm_elements[i] < 0)
      {
         /* note: start at i to handle "center" element correctly */
         no_symmetric_stencil_element = 1;
         for (j = i; j < stencil_size; j++)
         {
            symmetric = 1;
            for (d = 0; d < ndim; d++)
            {
               if (nalu_hypre_IndexD(symm_stencil_shape[j], d) !=
                   -nalu_hypre_IndexD(symm_stencil_shape[i], d))
               {
                  symmetric = 0;
                  break;
               }
            }
            if (symmetric)
            {
               /* only "off-center" elements have symmetric entries */
               if (i != j)
               {
                  symm_elements[j] = i;
               }
               no_symmetric_stencil_element = 0;
            }
         }

         if (no_symmetric_stencil_element)
         {
            /* add symmetric stencil element to `symm_stencil' */
            for (d = 0; d < ndim; d++)
            {
               nalu_hypre_IndexD(symm_stencil_shape[symm_stencil_size], d) =
                  -nalu_hypre_IndexD(symm_stencil_shape[i], d);
            }

            symm_elements[symm_stencil_size] = i;
            symm_stencil_size++;
         }
      }
   }

   symm_stencil = nalu_hypre_StructStencilCreate(nalu_hypre_StructStencilNDim(stencil),
                                            symm_stencil_size,
                                            symm_stencil_shape);

   *symm_stencil_ptr  = symm_stencil;
   *symm_elements_ptr = symm_elements;

   return nalu_hypre_error_flag;
}

