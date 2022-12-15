/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured inner product routine
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * nalu_hypre_StructInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_StructInnerProd( nalu_hypre_StructVector *x,
                       nalu_hypre_StructVector *y )
{
   NALU_HYPRE_Real       final_innerprod_result;
   NALU_HYPRE_Real       process_result;

   nalu_hypre_Box       *x_data_box;
   nalu_hypre_Box       *y_data_box;

   NALU_HYPRE_Complex   *xp;
   NALU_HYPRE_Complex   *yp;

   nalu_hypre_BoxArray  *boxes;
   nalu_hypre_Box       *box;
   nalu_hypre_Index      loop_size;
   nalu_hypre_IndexRef   start;
   nalu_hypre_Index      unit_stride;

   NALU_HYPRE_Int        ndim = nalu_hypre_StructVectorNDim(x);
   NALU_HYPRE_Int        i;

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   const NALU_HYPRE_Int  data_location = nalu_hypre_StructGridDataLocation(nalu_hypre_StructVectorGrid(y));
#endif

   NALU_HYPRE_Real       local_result = 0.0;

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

#if defined(NALU_HYPRE_USING_KOKKOS) || defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_Real box_sum = 0.0;
#elif defined(NALU_HYPRE_USING_RAJA)
      ReduceSum<nalu_hypre_raja_reduce_policy, NALU_HYPRE_Real> box_sum(0.0);
#elif defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      ReduceSum<NALU_HYPRE_Real> box_sum(0.0);
#else
      NALU_HYPRE_Real box_sum = 0.0;
#endif

#ifdef NALU_HYPRE_BOX_REDUCTION
#undef NALU_HYPRE_BOX_REDUCTION
#endif

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#define NALU_HYPRE_BOX_REDUCTION map(tofrom: box_sum) reduction(+:box_sum)
#else
#define NALU_HYPRE_BOX_REDUCTION reduction(+:box_sum)
#endif

#define DEVICE_VAR is_device_ptr(yp,xp)
      nalu_hypre_BoxLoop2ReductionBegin(ndim, loop_size,
                                   x_data_box, start, unit_stride, xi,
                                   y_data_box, start, unit_stride, yi,
                                   box_sum)
      {
         NALU_HYPRE_Real tmp = xp[xi] * nalu_hypre_conj(yp[yi]);
         box_sum += tmp;
      }
      nalu_hypre_BoxLoop2ReductionEnd(xi, yi, box_sum);

      local_result += (NALU_HYPRE_Real) box_sum;
   }

   process_result = (NALU_HYPRE_Real) local_result;

   nalu_hypre_MPI_Allreduce(&process_result, &final_innerprod_result, 1,
                       NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, nalu_hypre_StructVectorComm(x));

   nalu_hypre_IncFLOPCount(2 * nalu_hypre_StructVectorGlobalSize(x));

   return final_innerprod_result;
}
