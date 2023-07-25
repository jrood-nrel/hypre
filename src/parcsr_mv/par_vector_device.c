/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_mv.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

NALU_HYPRE_Int
nalu_hypre_ParVectorGetValuesDevice(nalu_hypre_ParVector *vector,
                               NALU_HYPRE_Int        num_values,
                               NALU_HYPRE_BigInt    *indices,
                               NALU_HYPRE_BigInt     base,
                               NALU_HYPRE_Complex   *values)
{
   NALU_HYPRE_BigInt    first_index  = nalu_hypre_ParVectorFirstIndex(vector);
   NALU_HYPRE_BigInt    last_index   = nalu_hypre_ParVectorLastIndex(vector);
   nalu_hypre_Vector   *local_vector = nalu_hypre_ParVectorLocalVector(vector);

   NALU_HYPRE_Int       component    = nalu_hypre_VectorComponent(local_vector);
   NALU_HYPRE_Int       vecstride    = nalu_hypre_VectorVectorStride(local_vector);
   NALU_HYPRE_Int       idxstride    = nalu_hypre_VectorIndexStride(local_vector);
   NALU_HYPRE_Complex  *data         = nalu_hypre_VectorData(local_vector);
   NALU_HYPRE_Int       vecoffset    = component * vecstride;

   NALU_HYPRE_Int       ierr = 0;

   if (idxstride != 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "nalu_hypre_ParVectorGetValuesDevice not implemented for non-columnwise vector storage\n");
      return nalu_hypre_error_flag;
   }

   /* If indices == NULL, assume that num_values components
      are to be retrieved from block starting at vec_start */
   if (indices)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      ierr = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                indices,
                                indices + num_values,
                                out_of_range<NALU_HYPRE_BigInt>(first_index + base, last_index + base) );
#else
      ierr = NALU_HYPRE_THRUST_CALL( count_if,
                                indices,
                                indices + num_values,
                                out_of_range<NALU_HYPRE_BigInt>(first_index + base, last_index + base) );
#endif
      if (ierr)
      {
         nalu_hypre_error_in_arg(3);
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Index out of range! -- nalu_hypre_ParVectorGetValues.");
         nalu_hypre_printf(" error: %d indices out of range! -- nalu_hypre_ParVectorGetValues\n", ierr);

#if defined(NALU_HYPRE_USING_SYCL)
         /* /1* WM: todo - why can't I combine transform iterator and gather? *1/ */
         /* NALU_HYPRE_ONEDPL_CALL( std::transform, */
         /*                    indices, */
         /*                    indices + num_values, */
         /*                    indices, */
         /*                    [base, first_index] (const auto & x) {return x - base - first_index;} ); */
         /* hypreSycl_gather_if( indices, */
         /*                      indices+ num_values, */
         /*                      indices, */
         /*                      data + vecoffset, */
         /*                      values, */
         /*                      in_range<NALU_HYPRE_BigInt>(first_index + base, last_index + base) ); */
         /* } */
         /* else */
         /* { */
         /* /1* WM: todo - why can't I combine transform iterator and gather? *1/ */
         /* NALU_HYPRE_ONEDPL_CALL( std::transform, */
         /*                    indices, */
         /*                    indices + num_values, */
         /*                    indices, */
         /*                    [base, first_index] (const auto & x) {return x - base - first_index;} ); */
         /* hypreSycl_gather( indices, */
         /*                   indices+ num_values, */
         /*                   data + vecoffset, */
         /*                   values); */
         auto trans_it = oneapi::dpl::make_transform_iterator(indices, [base,
         first_index] (const auto & x) {return x - base - first_index;} );
         hypreSycl_gather_if( trans_it,
                              trans_it + num_values,
                              indices,
                              data + vecoffset,
                              values,
                              in_range<NALU_HYPRE_BigInt>(first_index + base, last_index + base) );
      }
      else
      {
         auto trans_it = oneapi::dpl::make_transform_iterator(indices, [base,
         first_index] (const auto & x) {return x - base - first_index;} );
         hypreSycl_gather( trans_it,
                           trans_it + num_values,
                           data + vecoffset,
                           values);
#else
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
#endif
      }
   }
   else
   {
      if (num_values > nalu_hypre_VectorSize(local_vector))
      {
         nalu_hypre_error_in_arg(3);
         return nalu_hypre_error_flag;
      }

      nalu_hypre_TMemcpy(values, data + vecoffset, NALU_HYPRE_Complex, num_values,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   return nalu_hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_GPU)
