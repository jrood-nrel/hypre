/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Projection routines.
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ProjectBox:
 *   Projects a box onto a strided index space that contains the
 *   index `index' and has stride `stride'.
 *
 *   Note: An "empty" projection is represented by a box with volume 0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ProjectBox( nalu_hypre_Box    *box,
                  nalu_hypre_Index   index,
                  nalu_hypre_Index   stride )
{
   NALU_HYPRE_Int  i, s, d, hl, hu, kl, ku, ndim = nalu_hypre_BoxNDim(box);

   /*------------------------------------------------------
    * project in all ndim dimensions
    *------------------------------------------------------*/

   for (d = 0; d < ndim; d++)
   {

      i = nalu_hypre_IndexD(index, d);
      s = nalu_hypre_IndexD(stride, d);

      hl = nalu_hypre_BoxIMinD(box, d) - i;
      hu = nalu_hypre_BoxIMaxD(box, d) - i;

      if ( hl <= 0 )
      {
         kl = (NALU_HYPRE_Int) (hl / s);
      }
      else
      {
         kl = (NALU_HYPRE_Int) ((hl + (s - 1)) / s);
      }

      if ( hu >= 0 )
      {
         ku = (NALU_HYPRE_Int) (hu / s);
      }
      else
      {
         ku = (NALU_HYPRE_Int) ((hu - (s - 1)) / s);
      }

      nalu_hypre_BoxIMinD(box, d) = i + kl * s;
      nalu_hypre_BoxIMaxD(box, d) = i + ku * s;

   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ProjectBoxArray:
 *
 *   Note: The dimensions of the modified box array are not changed.
 *   So, it is possible to have boxes with volume 0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ProjectBoxArray( nalu_hypre_BoxArray  *box_array,
                       nalu_hypre_Index      index,
                       nalu_hypre_Index      stride    )
{
   nalu_hypre_Box  *box;
   NALU_HYPRE_Int   i;

   nalu_hypre_ForBoxI(i, box_array)
   {
      box = nalu_hypre_BoxArrayBox(box_array, i);
      nalu_hypre_ProjectBox(box, index, stride);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ProjectBoxArrayArray:
 *
 *   Note: The dimensions of the modified box array-array are not changed.
 *   So, it is possible to have boxes with volume 0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ProjectBoxArrayArray( nalu_hypre_BoxArrayArray  *box_array_array,
                            nalu_hypre_Index           index,
                            nalu_hypre_Index           stride          )
{
   nalu_hypre_BoxArray  *box_array;
   nalu_hypre_Box       *box;
   NALU_HYPRE_Int        i, j;

   nalu_hypre_ForBoxArrayI(i, box_array_array)
   {
      box_array = nalu_hypre_BoxArrayArrayBoxArray(box_array_array, i);
      nalu_hypre_ForBoxI(j, box_array)
      {
         box = nalu_hypre_BoxArrayBox(box_array, j);
         nalu_hypre_ProjectBox(box, index, stride);
      }
   }

   return nalu_hypre_error_flag;
}

