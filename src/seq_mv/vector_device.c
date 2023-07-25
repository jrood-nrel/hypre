/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorSetConstantValuesDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorSetConstantValuesDevice( nalu_hypre_Vector *v,
                                        NALU_HYPRE_Complex value )
{
   NALU_HYPRE_Complex *vector_data = nalu_hypre_VectorData(v);
   NALU_HYPRE_Int      num_vectors = nalu_hypre_VectorNumVectors(v);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(v);
   NALU_HYPRE_Int      total_size  = size * num_vectors;

   //nalu_hypre_SeqVectorPrefetch(v, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_GPU)
   hypreDevice_ComplexFilln( vector_data, total_size, value );

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(vector_data)
   for (i = 0; i < total_size; i++)
   {
      vector_data[i] = value;
   }
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorScaleDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorScaleDevice( NALU_HYPRE_Complex alpha,
                            nalu_hypre_Vector *y )
{
   NALU_HYPRE_Complex *y_data      = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int      num_vectors = nalu_hypre_VectorNumVectors(y);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(y);
   NALU_HYPRE_Int      total_size  = size * num_vectors;

   nalu_hypre_GpuProfilingPushRange("SeqVectorScale");
   //nalu_hypre_SeqVectorPrefetch(y, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_GPU)

#if ( defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) ) && defined(NALU_HYPRE_USING_CUBLAS)
   NALU_HYPRE_CUBLAS_CALL( nalu_hypre_cublas_scal(nalu_hypre_HandleCublasHandle(nalu_hypre_handle()),
                                        total_size, &alpha, y_data, 1) );
#elif defined(NALU_HYPRE_USING_SYCL) && defined(NALU_HYPRE_USING_ONEMKLBLAS)
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::blas::scal(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                              total_size, alpha,
                                              y_data, 1).wait() );
#else
   hypreDevice_ComplexScalen( y_data, total_size, y_data, alpha );
#endif

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data)
   for (i = 0; i < total_size; i++)
   {
      y_data[i] *= alpha;
   }
#endif

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorAxpyDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorAxpyDevice( NALU_HYPRE_Complex alpha,
                           nalu_hypre_Vector *x,
                           nalu_hypre_Vector *y )
{
   NALU_HYPRE_Complex *x_data      = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex *y_data      = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int      num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int      total_size  = size * num_vectors;

#if defined(NALU_HYPRE_USING_GPU)

#if ( defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) ) && defined(NALU_HYPRE_USING_CUBLAS)
   NALU_HYPRE_CUBLAS_CALL( nalu_hypre_cublas_axpy(nalu_hypre_HandleCublasHandle(nalu_hypre_handle()),
                                        total_size, &alpha, x_data, 1,
                                        y_data, 1) );
#elif defined(NALU_HYPRE_USING_SYCL) && defined(NALU_HYPRE_USING_ONEMKLBLAS)
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::blas::axpy(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                              total_size, alpha,
                                              x_data, 1, y_data, 1).wait() );
#else
   hypreDevice_ComplexAxpyn(x_data, total_size, y_data, y_data, alpha);
#endif

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data, x_data)
   for (i = 0; i < total_size; i++)
   {
      y_data[i] += alpha * x_data[i];
   }
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorAxpyzDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorAxpyzDevice( NALU_HYPRE_Complex  alpha,
                            nalu_hypre_Vector  *x,
                            NALU_HYPRE_Complex  beta,
                            nalu_hypre_Vector  *y,
                            nalu_hypre_Vector  *z )
{
   NALU_HYPRE_Complex  *x_data      = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex  *y_data      = nalu_hypre_VectorData(y);
   NALU_HYPRE_Complex  *z_data      = nalu_hypre_VectorData(z);

   NALU_HYPRE_Int       num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int       size        = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int       total_size  = size * num_vectors;

#if defined(NALU_HYPRE_USING_GPU)
   hypreDevice_ComplexAxpyzn(total_size, x_data, y_data, z_data, alpha, beta);

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(z_data, y_data, x_data)
   for (i = 0; i < total_size; i++)
   {
      z_data[i] = alpha * x_data[i] + beta * y_data[i];
   }
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorElmdivpyDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorElmdivpyDevice( nalu_hypre_Vector *x,
                               nalu_hypre_Vector *b,
                               nalu_hypre_Vector *y,
                               NALU_HYPRE_Int    *marker,
                               NALU_HYPRE_Int     marker_val )
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_Complex  *x_data        = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex  *b_data        = nalu_hypre_VectorData(b);
   NALU_HYPRE_Complex  *y_data        = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int       num_vectors_x = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int       num_vectors_y = nalu_hypre_VectorNumVectors(y);
   NALU_HYPRE_Int       num_vectors_b = nalu_hypre_VectorNumVectors(b);
   NALU_HYPRE_Int       size          = nalu_hypre_VectorSize(b);

   nalu_hypre_GpuProfilingPushRange("SeqVectorElmdivpyDevice");
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
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "marker != NULL not supported!\n");
         }
      }
      else
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Unsupported combination of num_vectors!\n");
      }

#else
      else
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "num_vectors_x != 1 not supported for SYCL!\n");
      }
#endif
   }
   else
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "num_vectors_b != 1 not supported!\n");
   }

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());
   nalu_hypre_GpuProfilingPopRange();

#elif defined(NALU_HYPRE_USING_OPENMP)
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Not implemented for device OpenMP!\n");
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorInnerProdDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_SeqVectorInnerProdDevice( nalu_hypre_Vector *x,
                                nalu_hypre_Vector *y )
{
   NALU_HYPRE_Complex *x_data      = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex *y_data      = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int      num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int      total_size  = size * num_vectors;

   NALU_HYPRE_Real     result = 0.0;

   //nalu_hypre_SeqVectorPrefetch(x, NALU_HYPRE_MEMORY_DEVICE);
   //nalu_hypre_SeqVectorPrefetch(y, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_GPU)

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#if defined(NALU_HYPRE_USING_CUBLAS)
   NALU_HYPRE_CUBLAS_CALL( nalu_hypre_cublas_dot(nalu_hypre_HandleCublasHandle(nalu_hypre_handle()), total_size,
                                       x_data, 1, y_data, 1, &result) );
#else
   result = NALU_HYPRE_THRUST_CALL( inner_product, x_data, x_data + total_size, y_data, 0.0 );
#endif

#elif defined(NALU_HYPRE_USING_SYCL)
#if defined(NALU_HYPRE_USING_ONEMKLBLAS)
   NALU_HYPRE_Real *result_dev = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::blas::dot(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                             total_size, x_data, 1,
                                             y_data, 1, result_dev).wait() );
   nalu_hypre_TMemcpy(&result, result_dev, NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(result_dev, NALU_HYPRE_MEMORY_DEVICE);
#else
   result = NALU_HYPRE_ONEDPL_CALL( std::transform_reduce, x_data, x_data + total_size, y_data, 0.0 );
#endif
#endif

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) reduction(+:result) is_device_ptr(y_data, x_data) map(result)
   for (i = 0; i < total_size; i++)
   {
      result += nalu_hypre_conj(y_data[i]) * x_data[i];
   }
#endif

   return result;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorSumEltsDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Complex
nalu_hypre_SeqVectorSumEltsDevice( nalu_hypre_Vector *vector )
{
   NALU_HYPRE_Complex  *data        = nalu_hypre_VectorData(vector);
   NALU_HYPRE_Int       num_vectors = nalu_hypre_VectorNumVectors(vector);
   NALU_HYPRE_Int       size        = nalu_hypre_VectorSize(vector);
   NALU_HYPRE_Int       total_size  = size * num_vectors;
   NALU_HYPRE_Complex   sum = 0.0;

#if defined(NALU_HYPRE_USING_GPU)
   sum = hypreDevice_ComplexReduceSum(total_size, data);

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

#elif NALU_HYPRE_USING_DEVICE_OPENMP
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) reduction(+:sum) is_device_ptr(data) map(sum)
   for (i = 0; i < total_size; i++)
   {
      sum += data[i];
   }
#endif

   return sum;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorPrefetch
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorPrefetch( nalu_hypre_Vector        *x,
                         NALU_HYPRE_MemoryLocation memory_location )
{
#if defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
   if (nalu_hypre_VectorMemoryLocation(x) != NALU_HYPRE_MEMORY_DEVICE)
   {
      /* nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC," Error! CUDA Prefetch with non-unified momory\n"); */
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_Complex  *x_data      = nalu_hypre_VectorData(x);
   NALU_HYPRE_Int       num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int       size        = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int       total_size  = size * num_vectors;

   if (total_size == 0)
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MemPrefetch(x_data, sizeof(NALU_HYPRE_Complex) * total_size, memory_location);
#endif

   return nalu_hypre_error_flag;
}

#endif
