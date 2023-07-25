/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured scale routine
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * nalu_hypre_StructScale
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructScale( NALU_HYPRE_Complex       alpha,
                   nalu_hypre_StructVector *y     )
{
   nalu_hypre_Box       *y_data_box;

   NALU_HYPRE_Complex   *yp;

   nalu_hypre_BoxArray  *boxes;
   nalu_hypre_Box       *box;
   nalu_hypre_Index      loop_size;
   nalu_hypre_IndexRef   start;
   nalu_hypre_Index      unit_stride;

   NALU_HYPRE_Int        i;

   nalu_hypre_SetIndex(unit_stride, 1);

   boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(y));
   nalu_hypre_ForBoxI(i, boxes)
   {
      box   = nalu_hypre_BoxArrayBox(boxes, i);
      start = nalu_hypre_BoxIMin(box);

      y_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);
      yp = nalu_hypre_StructVectorBoxData(y, i);

      nalu_hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp)
      nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(y), loop_size,
                          y_data_box, start, unit_stride, yi);
      {
         yp[yi] *= alpha;
      }
      nalu_hypre_BoxLoop1End(yi);
#undef DEVICE_VAR
   }

   return nalu_hypre_error_flag;
}
