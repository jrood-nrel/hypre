/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fac.h"

#define AbsStencilShape(stencil, abs_shape) \
{\
   NALU_HYPRE_Int ii,jj,kk;\
   ii = nalu_hypre_IndexX(stencil);\
   jj = nalu_hypre_IndexY(stencil);\
   kk = nalu_hypre_IndexZ(stencil);\
   abs_shape= nalu_hypre_abs(ii) + nalu_hypre_abs(jj) + nalu_hypre_abs(kk); \
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CFInterfaceExtents: Given a cgrid_box, a fgrid_box, and stencils,
 * find the extents of the C/F interface (interface nodes in the C box).
 * Boxes corresponding to stencil shifts are stored in the first stencil_size
 * boxes, and the union of these are appended to the end of the returned
 * box_array.
 *--------------------------------------------------------------------------*/
nalu_hypre_BoxArray *
nalu_hypre_CFInterfaceExtents( nalu_hypre_Box              *fgrid_box,
                          nalu_hypre_Box              *cgrid_box,
                          nalu_hypre_StructStencil    *stencils,
                          nalu_hypre_Index             rfactors )
{

   nalu_hypre_BoxArray        *stencil_box_extents;
   nalu_hypre_BoxArray        *union_boxes;
   nalu_hypre_Box             *cfine_box;
   nalu_hypre_Box             *box;

   nalu_hypre_Index            stencil_shape, cstart, zero_index, neg_index;
   NALU_HYPRE_Int              stencil_size;
   NALU_HYPRE_Int              abs_stencil;

   NALU_HYPRE_Int              ndim = nalu_hypre_StructStencilNDim(stencils);
   NALU_HYPRE_Int              i, j;

   nalu_hypre_ClearIndex(zero_index);
   nalu_hypre_ClearIndex(neg_index);
   for (i = 0; i < ndim; i++)
   {
      neg_index[i] = -1;
   }
   nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cgrid_box), cstart);

   stencil_size       = nalu_hypre_StructStencilSize(stencils);
   stencil_box_extents = nalu_hypre_BoxArrayCreate(stencil_size, ndim);
   union_boxes        = nalu_hypre_BoxArrayCreate(0, ndim);

   for (i = 0; i < stencil_size; i++)
   {
      nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i), stencil_shape);
      AbsStencilShape(stencil_shape, abs_stencil);

      if (abs_stencil)  /* only do if not the centre stencil */
      {
         cfine_box = nalu_hypre_CF_StenBox(fgrid_box, cgrid_box, stencil_shape, rfactors,
                                      ndim);

         if ( nalu_hypre_BoxVolume(cfine_box) )
         {
            nalu_hypre_AppendBox(cfine_box, union_boxes);
            nalu_hypre_CopyBox(cfine_box, nalu_hypre_BoxArrayBox(stencil_box_extents, i));
            for (j = 0; j < ndim; j++)
            {
               nalu_hypre_BoxIMin(cfine_box)[j] -=  cstart[j];
               nalu_hypre_BoxIMax(cfine_box)[j] -=  cstart[j];
            }
            nalu_hypre_CopyBox(cfine_box, nalu_hypre_BoxArrayBox(stencil_box_extents, i));
         }

         else
         {
            nalu_hypre_BoxSetExtents(nalu_hypre_BoxArrayBox(stencil_box_extents, i),
                                zero_index, neg_index);
         }

         nalu_hypre_BoxDestroy(cfine_box);
      }

      else /* centre */
      {
         nalu_hypre_BoxSetExtents(nalu_hypre_BoxArrayBox(stencil_box_extents, i),
                             zero_index, neg_index);
      }
   }

   /*--------------------------------------------------------------------------
    * Union the stencil_box_extents to get the full CF extents and append to
    * the end of the stencil_box_extents BoxArray. Then shift the unioned boxes
    * by cstart.
    *--------------------------------------------------------------------------*/
   if (nalu_hypre_BoxArraySize(union_boxes) > 1)
   {
      nalu_hypre_UnionBoxes(union_boxes);
   }

   nalu_hypre_ForBoxI(i, union_boxes)
   {
      nalu_hypre_AppendBox(nalu_hypre_BoxArrayBox(union_boxes, i), stencil_box_extents);
   }
   nalu_hypre_BoxArrayDestroy(union_boxes);

   for (i = stencil_size; i < nalu_hypre_BoxArraySize(stencil_box_extents); i++)
   {
      box = nalu_hypre_BoxArrayBox(stencil_box_extents, i);
      for (j = 0; j < ndim; j++)
      {
         nalu_hypre_BoxIMin(box)[j] -=  cstart[j];
         nalu_hypre_BoxIMax(box)[j] -=  cstart[j];
      }
   }

   return stencil_box_extents;
}

NALU_HYPRE_Int
nalu_hypre_CFInterfaceExtents2( nalu_hypre_Box              *fgrid_box,
                           nalu_hypre_Box              *cgrid_box,
                           nalu_hypre_StructStencil    *stencils,
                           nalu_hypre_Index             rfactors,
                           nalu_hypre_BoxArray         *cf_interface )
{

   nalu_hypre_BoxArray        *stencil_box_extents;
   nalu_hypre_BoxArray        *union_boxes;
   nalu_hypre_Box             *cfine_box;

   nalu_hypre_Index            stencil_shape, zero_index, neg_index;
   NALU_HYPRE_Int              stencil_size;
   NALU_HYPRE_Int              abs_stencil;

   NALU_HYPRE_Int              ndim = nalu_hypre_StructStencilNDim(stencils);

   NALU_HYPRE_Int              i;
   NALU_HYPRE_Int              ierr = 0;

   nalu_hypre_ClearIndex(zero_index);
   nalu_hypre_ClearIndex(neg_index);
   for (i = 0; i < ndim; i++)
   {
      neg_index[i] = -1;
   }

   stencil_size       = nalu_hypre_StructStencilSize(stencils);
   stencil_box_extents = nalu_hypre_BoxArrayCreate(stencil_size, ndim);
   union_boxes        = nalu_hypre_BoxArrayCreate(0, ndim);

   for (i = 0; i < stencil_size; i++)
   {
      nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i), stencil_shape);
      AbsStencilShape(stencil_shape, abs_stencil);

      if (abs_stencil)  /* only do if not the centre stencil */
      {
         cfine_box = nalu_hypre_CF_StenBox(fgrid_box, cgrid_box, stencil_shape,
                                      rfactors, ndim);

         if ( nalu_hypre_BoxVolume(cfine_box) )
         {
            nalu_hypre_AppendBox(cfine_box, union_boxes);
            nalu_hypre_CopyBox(cfine_box, nalu_hypre_BoxArrayBox(stencil_box_extents, i));
         }

         else
         {
            nalu_hypre_BoxSetExtents(nalu_hypre_BoxArrayBox(stencil_box_extents, i),
                                zero_index, neg_index);
         }

         nalu_hypre_BoxDestroy(cfine_box);
      }

      else /* centre */
      {
         nalu_hypre_BoxSetExtents(nalu_hypre_BoxArrayBox(stencil_box_extents, i),
                             zero_index, neg_index);
      }
   }

   /*--------------------------------------------------------------------------
    * Union the stencil_box_extents to get the full CF extents and append to
    * the end of the stencil_box_extents BoxArray.
    *--------------------------------------------------------------------------*/
   if (nalu_hypre_BoxArraySize(union_boxes) > 1)
   {
      nalu_hypre_UnionBoxes(union_boxes);
   }

   nalu_hypre_ForBoxI(i, union_boxes)
   {
      nalu_hypre_AppendBox(nalu_hypre_BoxArrayBox(union_boxes, i), stencil_box_extents);
   }
   nalu_hypre_AppendBoxArray(stencil_box_extents, cf_interface);

   nalu_hypre_BoxArrayDestroy(union_boxes);
   nalu_hypre_BoxArrayDestroy(stencil_box_extents);

   return ierr;
}
