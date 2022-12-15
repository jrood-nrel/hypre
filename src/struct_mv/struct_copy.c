/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured copy routine
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * nalu_hypre_StructCopy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructCopy( nalu_hypre_StructVector *x,
                  nalu_hypre_StructVector *y     )
{
   nalu_hypre_Box       *x_data_box;
   nalu_hypre_Box       *y_data_box;

   NALU_HYPRE_Complex   *xp;
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

      x_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
      y_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);

      xp = nalu_hypre_StructVectorBoxData(x, i);
      yp = nalu_hypre_StructVectorBoxData(y, i);

      nalu_hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp,xp)
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, unit_stride, xi,
                          y_data_box, start, unit_stride, yi);
      {
         yp[yi] = xp[xi];
      }
      nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructPartialCopy: copy only the components on a subset of the grid.
 * A BoxArrayArray of boxes are needed- for each box of x, only an array
 * of subboxes (i.e., a boxarray for each box of x) are copied.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructPartialCopy( nalu_hypre_StructVector  *x,
                         nalu_hypre_StructVector  *y,
                         nalu_hypre_BoxArrayArray *array_boxes )
{
   nalu_hypre_Box       *x_data_box;
   nalu_hypre_Box       *y_data_box;

   NALU_HYPRE_Complex   *xp;
   NALU_HYPRE_Complex   *yp;

   nalu_hypre_BoxArray  *boxes;
   nalu_hypre_Box       *box;
   nalu_hypre_Index      loop_size;
   nalu_hypre_IndexRef   start;
   nalu_hypre_Index      unit_stride;

   NALU_HYPRE_Int        i, j ;

   nalu_hypre_SetIndex(unit_stride, 1);

   nalu_hypre_ForBoxArrayI(i, array_boxes)
   {
      boxes = nalu_hypre_BoxArrayArrayBoxArray(array_boxes, i);

      x_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
      y_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);

      xp = nalu_hypre_StructVectorBoxData(x, i);
      yp = nalu_hypre_StructVectorBoxData(y, i);

      /* array of sub_boxes of box_i of the vector */
      nalu_hypre_ForBoxI(j, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, j);

         start = nalu_hypre_BoxIMin(box);
         nalu_hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp,xp)
         nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                             x_data_box, start, unit_stride, xi,
                             y_data_box, start, unit_stride, yi);
         {
            yp[yi] = xp[xi];
         }
         nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
      }
   }

   return nalu_hypre_error_flag;
}
