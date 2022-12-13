/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(NALU_HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetConstantValuesDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqVectorSetConstantValuesDevice( hypre_Vector *v,
                                        NALU_HYPRE_Complex value )
{
   NALU_HYPRE_Complex *vector_data = hypre_VectorData(v);
   NALU_HYPRE_Int      num_vectors = hypre_VectorNumVectors(v);
   NALU_HYPRE_Int      size        = hypre_VectorSize(v);
   NALU_HYPRE_Int      total_size  = size * num_vectors;

   //hypre_SeqVectorPrefetch(v, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   hypreDevice_ComplexFilln( vector_data, total_size, value );

#elif defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::fill_n, vector_data, total_size, value );

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(vector_data)
   for (i = 0; i < total_size; i++)
   {
      vector_data[i] = value;
   }
#endif

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorScaleDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqVectorScaleDevice( NALU_HYPRE_Complex alpha,
                            hypre_Vector *y )
{
   NALU_HYPRE_Complex *y_data      = hypre_VectorData(y);
   NALU_HYPRE_Int      num_vectors = hypre_VectorNumVectors(y);
   NALU_HYPRE_Int      size        = hypre_VectorSize(y);
   NALU_HYPRE_Int      total_size  = size * num_vectors;

   //hypre_SeqVectorPrefetch(y, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#if defined(NALU_HYPRE_USING_CUBLAS)
   NALU_HYPRE_CUBLAS_CALL( hypre_cublas_scal(hypre_HandleCublasHandle(hypre_handle()),
                                        total_size, &alpha, y_data, 1) );
#else
   hypreDevice_ComplexScalen( y_data, total_size, y_data, alpha );
#endif

#elif defined(NALU_HYPRE_USING_SYCL)
#if defined(NALU_HYPRE_USING_ONEMKLBLAS)
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::blas::scal(*hypre_HandleComputeStream(hypre_handle()),
                                              total_size, alpha,
                                              y_data, 1).wait() );
#else
   NALU_HYPRE_ONEDPL_CALL( std::transform, y_data, y_data + total_size,
                      y_data, [alpha](NALU_HYPRE_Complex y) -> NALU_HYPRE_Complex { return alpha * y; } );
#endif

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data)
   for (i = 0; i < total_size; i++)
   {
      y_data[i] *= alpha;
   }
#endif

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpyDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqVectorAxpyDevice( NALU_HYPRE_Complex alpha,
                           hypre_Vector *x,
                           hypre_Vector *y )
{
   NALU_HYPRE_Complex *x_data      = hypre_VectorData(x);
   NALU_HYPRE_Complex *y_data      = hypre_VectorData(y);
   NALU_HYPRE_Int      num_vectors = hypre_VectorNumVectors(x);
   NALU_HYPRE_Int      size        = hypre_VectorSize(x);
   NALU_HYPRE_Int      total_size  = size * num_vectors;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#if defined(NALU_HYPRE_USING_CUBLAS)
   NALU_HYPRE_CUBLAS_CALL( hypre_cublas_axpy(hypre_HandleCublasHandle(hypre_handle()),
                                        total_size, &alpha, x_data, 1,
                                        y_data, 1) );
#else
   hypreDevice_ComplexAxpyn(x_data, total_size, y_data, y_data, alpha);
#endif

#elif defined(NALU_HYPRE_USING_SYCL)
#if defined(NALU_HYPRE_USING_ONEMKLBLAS)
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::blas::axpy(*hypre_HandleComputeStream(hypre_handle()),
                                              total_size, alpha,
                                              x_data, 1, y_data, 1).wait() );
#else
   NALU_HYPRE_ONEDPL_CALL( std::transform, x_data, x_data + total_size, y_data, y_data,
                      [alpha](NALU_HYPRE_Complex x, NALU_HYPRE_Complex y) -> NALU_HYPRE_Complex { return alpha * x + y; } );
#endif

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data, x_data)
   for (i = 0; i < total_size; i++)
   {
      y_data[i] += alpha * x_data[i];
   }
#endif

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorElmdivpyDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqVectorElmdivpyDevice( hypre_Vector *x,
                               hypre_Vector *b,
                               hypre_Vector *y,
                               NALU_HYPRE_Int    *marker,
                               NALU_HYPRE_Int     marker_val )
{
   NALU_HYPRE_Complex  *x_data        = hypre_VectorData(x);
   NALU_HYPRE_Complex  *b_data        = hypre_VectorData(b);
   NALU_HYPRE_Complex  *y_data        = hypre_VectorData(y);
   NALU_HYPRE_Int       num_vectors_x = hypre_VectorNumVectors(x);
   NALU_HYPRE_Int       num_vectors_y = hypre_VectorNumVectors(y);
   NALU_HYPRE_Int       num_vectors_b = hypre_VectorNumVectors(b);
   NALU_HYPRE_Int       size          = hypre_VectorSize(b);

#if defined(NALU_HYPRE_USING_CUDA) ||\
    defined(NALU_HYPRE_USING_HIP)  ||\
    defined(NALU_HYPRE_USING_SYCL)
   if (num_vectors_b == 1)
   {
      if (num_vectors_x == 1)
      {
         if (marker)
         {
            hypreDevice_IVAXPYMarked(size, b_data, x_data, y_data, marker, marker_val);
         }
         else
         {
            hypreDevice_IVAXPY(size, b_data, x_data, y_data);
         }
      }
#if !defined(NALU_HYPRE_USING_SYCL)
      else if (num_vectors_x == num_vectors_y)
      {
         if (!marker)
         {
            hypreDevice_IVAMXPMY(num_vectors_x, size, b_data, x_data, y_data);
         }
         else
         {
            hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "marker != NULL not supported!\n");
         }
      }
      else
      {
         hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Unsupported combination of num_vectors!\n");
      }

#else
      else
      {
         hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "num_vectors_x != 1 not supported for SYCL!\n");
      }
#endif
   }
   else
   {
      hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "num_vectors_b != 1 not supported!\n");
   }
#endif

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProdDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
hypre_SeqVectorInnerProdDevice( hypre_Vector *x,
                                hypre_Vector *y )
{
   NALU_HYPRE_Complex *x_data      = hypre_VectorData(x);
   NALU_HYPRE_Complex *y_data      = hypre_VectorData(y);
   NALU_HYPRE_Int      num_vectors = hypre_VectorNumVectors(x);
   NALU_HYPRE_Int      size        = hypre_VectorSize(x);
   NALU_HYPRE_Int      total_size  = size * num_vectors;

   NALU_HYPRE_Real     result = 0.0;

   //hypre_SeqVectorPrefetch(x, NALU_HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(y, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#if defined(NALU_HYPRE_USING_CUBLAS)
   NALU_HYPRE_CUBLAS_CALL( hypre_cublas_dot(hypre_HandleCublasHandle(hypre_handle()), total_size,
                                       x_data, 1, y_data, 1, &result) );
#else
   result = NALU_HYPRE_THRUST_CALL( inner_product, x_data, x_data + total_size, y_data, 0.0 );
#endif

#elif defined(NALU_HYPRE_USING_SYCL)
#if defined(NALU_HYPRE_USING_ONEMKLBLAS)
   NALU_HYPRE_Real *result_dev = hypre_CTAlloc(NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::blas::dot(*hypre_HandleComputeStream(hypre_handle()),
                                             total_size, x_data, 1,
                                             y_data, 1, result_dev).wait() );
   hypre_TMemcpy(&result, result_dev, NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(result_dev, NALU_HYPRE_MEMORY_DEVICE);
#else
   result = NALU_HYPRE_ONEDPL_CALL( std::transform_reduce, x_data, x_data + total_size, y_data, 0.0 );
#endif

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) reduction(+:result) is_device_ptr(y_data, x_data) map(result)
   for (i = 0; i < total_size; i++)
   {
      result += hypre_conj(y_data[i]) * x_data[i];
   }
#endif

   hypre_SyncComputeStream(hypre_handle());

   return result;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSumEltsDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Complex
hypre_SeqVectorSumEltsDevice( hypre_Vector *vector )
{
   NALU_HYPRE_Complex  *data        = hypre_VectorData(vector);
   NALU_HYPRE_Int       num_vectors = hypre_VectorNumVectors(vector);
   NALU_HYPRE_Int       size        = hypre_VectorSize(vector);
   NALU_HYPRE_Int       total_size  = size * num_vectors;
   NALU_HYPRE_Complex   sum = 0.0;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   sum = hypreDevice_ComplexReduceSum(total_size, data);

#elif defined(NALU_HYPRE_USING_SYCL)
   sum = NALU_HYPRE_ONEDPL_CALL( std::reduce, data, data + total_size );

#elif NALU_HYPRE_USING_DEVICE_OPENMP
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) reduction(+:sum) is_device_ptr(data) map(sum)
   for (i = 0; i < total_size; i++)
   {
      sum += data[i];
   }
#endif

   hypre_SyncComputeStream(hypre_handle());

   return sum;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorPrefetch
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqVectorPrefetch( hypre_Vector        *x,
                         NALU_HYPRE_MemoryLocation memory_location )
{
#if defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
   if (hypre_VectorMemoryLocation(x) != NALU_HYPRE_MEMORY_DEVICE)
   {
      /* hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC," Error! CUDA Prefetch with non-unified momory\n"); */
      return hypre_error_flag;
   }

   NALU_HYPRE_Complex  *x_data      = hypre_VectorData(x);
   NALU_HYPRE_Int       num_vectors = hypre_VectorNumVectors(x);
   NALU_HYPRE_Int       size        = hypre_VectorSize(x);
   NALU_HYPRE_Int       total_size  = size * num_vectors;

   if (total_size == 0)
   {
      return hypre_error_flag;
   }

   hypre_MemPrefetch(x_data, sizeof(NALU_HYPRE_Complex) * total_size, memory_location);
#endif

   return hypre_error_flag;
}

#endif
