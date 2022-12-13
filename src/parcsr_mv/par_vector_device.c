/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

NALU_HYPRE_Int
hypre_ParVectorGetValuesDevice(hypre_ParVector *vector,
                               NALU_HYPRE_Int        num_values,
                               NALU_HYPRE_BigInt    *indices,
                               NALU_HYPRE_BigInt     base,
                               NALU_HYPRE_Complex   *values)
{
   NALU_HYPRE_BigInt    first_index  = hypre_ParVectorFirstIndex(vector);
   NALU_HYPRE_BigInt    last_index   = hypre_ParVectorLastIndex(vector);
   hypre_Vector   *local_vector = hypre_ParVectorLocalVector(vector);

   NALU_HYPRE_Int       component    = hypre_VectorComponent(local_vector);
   NALU_HYPRE_Int       vecstride    = hypre_VectorVectorStride(local_vector);
   NALU_HYPRE_Int       idxstride    = hypre_VectorIndexStride(local_vector);
   NALU_HYPRE_Complex  *data         = hypre_VectorData(local_vector);
   NALU_HYPRE_Int       vecoffset    = component * vecstride;

   NALU_HYPRE_Int       ierr = 0;

   if (idxstride != 1)
   {
      hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "hypre_ParVectorGetValuesDevice not implemented for non-columnwise vector storage\n");
      return hypre_error_flag;
   }

   /* If indices == NULL, assume that num_values components
      are to be retrieved from block starting at vec_start */
   if (indices)
   {
      ierr = NALU_HYPRE_THRUST_CALL( count_if,
                                indices,
                                indices + num_values,
                                out_of_range<NALU_HYPRE_BigInt>(first_index + base, last_index + base) );
      if (ierr)
      {
         hypre_error_in_arg(3);
         hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Index out of range! -- hypre_ParVectorGetValues.");
         hypre_printf(" error: %d indices out of range! -- hypre_ParVectorGetValues\n", ierr);

         NALU_HYPRE_THRUST_CALL( gather_if,
                            thrust::make_transform_iterator(indices, _1 - base - first_index),
                            thrust::make_transform_iterator(indices, _1 - base - first_index) + num_values,
                            indices,
                            data + vecoffset,
                            values,
                            in_range<NALU_HYPRE_BigInt>(first_index + base, last_index + base) );
      }
      else
      {
         NALU_HYPRE_THRUST_CALL( gather,
                            thrust::make_transform_iterator(indices, _1 - base - first_index),
                            thrust::make_transform_iterator(indices, _1 - base - first_index) + num_values,
                            data + vecoffset,
                            values);
      }
   }
   else
   {
      if (num_values > hypre_VectorSize(local_vector))
      {
         hypre_error_in_arg(3);
         return hypre_error_flag;
      }

      hypre_TMemcpy(values, data + vecoffset, NALU_HYPRE_Complex, num_values,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   return hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
