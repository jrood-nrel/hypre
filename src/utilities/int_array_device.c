/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_utilities.hpp"
#include "_nalu_hypre_onedpl.hpp"

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArraySetConstantValuesDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArraySetConstantValuesDevice( nalu_hypre_IntArray *v,
                                       NALU_HYPRE_Int       value )
{
   NALU_HYPRE_Int *array_data = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_Int  size       = nalu_hypre_IntArraySize(v);

#if defined(NALU_HYPRE_USING_GPU)
   hypreDevice_IntFilln( array_data, size, value );

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;
   #pragma omp target teams distribute parallel for private(i) is_device_ptr(array_data)
   for (i = 0; i < size; i++)
   {
      array_data[i] = value;
   }
#endif

   return nalu_hypre_error_flag;
}

#if defined(NALU_HYPRE_USING_GPU)
/*--------------------------------------------------------------------------
 * hypreGPUKernel_IntArrayInverseMapping
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_IntArrayInverseMapping( nalu_hypre_DeviceItem  &item,
                                       NALU_HYPRE_Int          size,
                                       NALU_HYPRE_Int         *v_data,
                                       NALU_HYPRE_Int         *w_data )
{
   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < size)
   {
      w_data[v_data[i]] = i;
   }
}
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayInverseMappingDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayInverseMappingDevice( nalu_hypre_IntArray  *v,
                                    nalu_hypre_IntArray  *w )
{
   NALU_HYPRE_Int   size    = nalu_hypre_IntArraySize(v);
   NALU_HYPRE_Int  *v_data  = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_Int  *w_data  = nalu_hypre_IntArrayData(w);

#if defined(NALU_HYPRE_USING_GPU)
   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(size, "thread", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_IntArrayInverseMapping, gDim, bDim, size, v_data, w_data );

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(v_data, w_data)
   for (i = 0; i < size; i++)
   {
      w_data[v_data[i]] = i;
   }
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayCountDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayCountDevice( nalu_hypre_IntArray *v,
                           NALU_HYPRE_Int       value,
                           NALU_HYPRE_Int      *num_values_ptr )
{
   NALU_HYPRE_Int  *array_data  = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_Int   size        = nalu_hypre_IntArraySize(v);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   *num_values_ptr = NALU_HYPRE_THRUST_CALL( count,
                                        array_data,
                                        array_data + size,
                                        value );

#elif defined(NALU_HYPRE_USING_SYCL)
   *num_values_ptr = NALU_HYPRE_ONEDPL_CALL( std::count,
                                        array_data,
                                        array_data + size,
                                        value );

#elif defined (NALU_HYPRE_USING_DEVICE_OPENMP)
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Function not implemented for Device OpenMP");
   *num_values_ptr = 0;
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayNegateDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayNegateDevice( nalu_hypre_IntArray *v )
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_THRUST_CALL( transform,
                      nalu_hypre_IntArrayData(v),
                      nalu_hypre_IntArrayData(v) + nalu_hypre_IntArraySize(v),
                      nalu_hypre_IntArrayData(v),
                      thrust::negate<NALU_HYPRE_Int>() );
#elif defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      nalu_hypre_IntArrayData(v),
                      nalu_hypre_IntArrayData(v) + nalu_hypre_IntArraySize(v),
                      nalu_hypre_IntArrayData(v),
                      std::negate<NALU_HYPRE_Int>() );
#else
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Not implemented yet!");
#endif

   return nalu_hypre_error_flag;
}

#endif
