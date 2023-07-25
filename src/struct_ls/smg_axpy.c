/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGAxpy( NALU_HYPRE_Real          alpha,
               nalu_hypre_StructVector *x,
               nalu_hypre_StructVector *y,
               nalu_hypre_Index         base_index,
               nalu_hypre_Index         base_stride )
{
   NALU_HYPRE_Int         ndim = nalu_hypre_StructVectorNDim(x);
   nalu_hypre_Box        *x_data_box;
   nalu_hypre_Box        *y_data_box;

   NALU_HYPRE_Real       *xp;
   NALU_HYPRE_Real       *yp;

   nalu_hypre_BoxArray   *boxes;
   nalu_hypre_Box        *box;
   nalu_hypre_Index       loop_size;
   nalu_hypre_IndexRef    start;

   NALU_HYPRE_Int         i;

   box = nalu_hypre_BoxCreate(ndim);
   boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(y));
   nalu_hypre_ForBoxI(i, boxes)
   {
      nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(boxes, i), box);
      nalu_hypre_ProjectBox(box, base_index, base_stride);
      start = nalu_hypre_BoxIMin(box);

      x_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
      y_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);

      xp = nalu_hypre_StructVectorBoxData(x, i);
      yp = nalu_hypre_StructVectorBoxData(y, i);

      nalu_hypre_BoxGetStrideSize(box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(yp,xp)
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, base_stride, xi,
                          y_data_box, start, base_stride, yi);
      {
         yp[yi] += alpha * xp[xi];
      }
      nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
   }
   nalu_hypre_BoxDestroy(box);

   return nalu_hypre_error_flag;
}
