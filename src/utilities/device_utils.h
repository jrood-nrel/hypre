/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_DEVICE_UTILS_H
#define NALU_HYPRE_DEVICE_UTILS_H

#if defined(NALU_HYPRE_USING_GPU)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                          cuda includes
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_CUDA)
using nalu_hypre_DeviceItem = void*;
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cusparse.h>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

#ifndef CUDA_VERSION
#error CUDA_VERSION Undefined!
#endif

#if CUDA_VERSION >= 11000
#define THRUST_IGNORE_DEPRECATED_CPP11
#define CUB_IGNORE_DEPRECATED_CPP11
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT
#endif

#define CUSPARSE_NEWAPI_VERSION 11000
#define CUSPARSE_NEWSPMM_VERSION 11401
#define CUDA_MALLOCASYNC_VERSION 11020
#define THRUST_CALL_BLOCKING 1

#if defined(NALU_HYPRE_USING_DEVICE_MALLOC_ASYNC)
#if CUDA_VERSION < CUDA_MALLOCASYNC_VERSION
#error cudaMalloc/FreeAsync needs CUDA 11.2
#endif
#endif
#endif // defined(NALU_HYPRE_USING_CUDA)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                          hip includes
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_HIP)
using nalu_hypre_DeviceItem = void*;
#include <hip/hip_runtime.h>

#if defined(NALU_HYPRE_USING_ROCSPARSE)
#include <rocsparse.h>
#endif

#if defined(NALU_HYPRE_USING_ROCRAND)
#include <rocrand/rocrand.h>
#endif
#endif // defined(NALU_HYPRE_USING_HIP)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                          thrust includes
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#include <thrust/execution_policy.h>
#if defined(NALU_HYPRE_USING_CUDA)
#include <thrust/system/cuda/execution_policy.h>
#elif defined(NALU_HYPRE_USING_HIP)
#include <thrust/system/hip/execution_policy.h>
#endif
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/adjacent_difference.h>
#include <thrust/inner_product.h>
#include <thrust/logical.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/remove.h>

using namespace thrust::placeholders;
#endif // defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                          sycl includes
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_SYCL)

#include <sycl/sycl.hpp>
#if defined(NALU_HYPRE_USING_ONEMKLSPARSE)
#include <oneapi/mkl/spblas.hpp>
#endif
#if defined(NALU_HYPRE_USING_ONEMKLBLAS)
#include <oneapi/mkl/blas.hpp>
#endif
#if defined(NALU_HYPRE_USING_ONEMKLRAND)
#include <oneapi/mkl/rng.hpp>
#endif

/* The following definitions facilitate code reuse and limits
 * if/def-ing when unifying cuda/hip code with sycl code */
using dim3 = sycl::range<1>;
using nalu_hypre_DeviceItem = sycl::nd_item<1>;
#define __global__
#define __host__
#define __device__
#define __forceinline__ __inline__ __attribute__((always_inline))

#endif // defined(NALU_HYPRE_USING_SYCL)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      device defined values
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define NALU_HYPRE_MAX_NTHREADS_BLOCK 1024

// NALU_HYPRE_WARP_BITSHIFT is just log2 of NALU_HYPRE_WARP_SIZE
#if defined(NALU_HYPRE_USING_CUDA)
#define NALU_HYPRE_WARP_SIZE       32
#define NALU_HYPRE_WARP_BITSHIFT   5
#elif defined(NALU_HYPRE_USING_HIP)
#define NALU_HYPRE_WARP_SIZE       64
#define NALU_HYPRE_WARP_BITSHIFT   6
#elif defined(NALU_HYPRE_USING_SYCL)
#define NALU_HYPRE_WARP_SIZE       16
#define NALU_HYPRE_WARP_BITSHIFT   4
#endif

#define NALU_HYPRE_WARP_FULL_MASK  0xFFFFFFFF
#define NALU_HYPRE_MAX_NUM_WARPS   (64 * 64 * 32)
#define NALU_HYPRE_FLT_LARGE       1e30
#define NALU_HYPRE_1D_BLOCK_SIZE   512
#define NALU_HYPRE_MAX_NUM_STREAMS 10
#define NALU_HYPRE_SPGEMM_MAX_NBIN 10

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *       macro for launching GPU kernels
 *       NOTE: IN HYPRE'S DEFAULT STREAM
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_DEBUG)
#define GPU_LAUNCH_SYNC { nalu_hypre_SyncComputeStream(nalu_hypre_handle()); nalu_hypre_GetDeviceLastError(); }
#else
#define GPU_LAUNCH_SYNC
#endif

/* cuda/hip version */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#define NALU_HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, shmem_size, ...)                                                        \
{                                                                                                                                   \
   if ( gridsize.x  == 0 || gridsize.y  == 0 || gridsize.z  == 0 ||                                                                 \
        blocksize.x == 0 || blocksize.y == 0 || blocksize.z == 0 )                                                                  \
   {                                                                                                                                \
      /* printf("Warning %s %d: Zero CUDA grid/block (%d %d %d) (%d %d %d)\n",                                                      \
                 __FILE__, __LINE__,                                                                                                \
                 gridsize.x, gridsize.y, gridsize.z, blocksize.x, blocksize.y, blocksize.z); */                                     \
   }                                                                                                                                \
   else                                                                                                                             \
   {                                                                                                                                \
      nalu_hypre_DeviceItem item = NULL;                                                                                                 \
      (kernel_name) <<< (gridsize), (blocksize), shmem_size, nalu_hypre_HandleComputeStream(nalu_hypre_handle()) >>> (item, __VA_ARGS__);     \
      GPU_LAUNCH_SYNC;                                                                                                              \
   }                                                                                                                                \
}

#define NALU_HYPRE_GPU_LAUNCH(kernel_name, gridsize, blocksize, ...) NALU_HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, 0, __VA_ARGS__)
#endif // defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/* sycl version */
#if defined(NALU_HYPRE_USING_SYCL)
#define NALU_HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, shmem_size, ...)                 \
{                                                                                            \
   if ( gridsize[0] == 0 || blocksize[0] == 0 )                                              \
   {                                                                                         \
     /* nalu_hypre_printf("Warning %s %d: Zero SYCL 1D launch parameters grid/block (%d) (%d)\n", \
                  __FILE__, __LINE__,                                                        \
                  gridsize[0], blocksize[0]); */                                             \
   }                                                                                         \
   else                                                                                      \
   {                                                                                         \
      nalu_hypre_HandleComputeStream(nalu_hypre_handle())->submit([&] (sycl::handler& cgh) {           \
         cgh.parallel_for(sycl::nd_range<1>(gridsize*blocksize, blocksize),                  \
            [=] (nalu_hypre_DeviceItem item) [[intel::reqd_sub_group_size(NALU_HYPRE_WARP_SIZE)]]      \
               { (kernel_name)(item, __VA_ARGS__);                                           \
         });                                                                                 \
      }).wait_and_throw();                                                                   \
   }                                                                                         \
}
#define NALU_HYPRE_GPU_DEBUG_LAUNCH(kernel_name, gridsize, blocksize, ...)                        \
{                                                                                            \
   if ( gridsize[0] == 0 || blocksize[0] == 0 )                                              \
   {                                                                                         \
     /* nalu_hypre_printf("Warning %s %d: Zero SYCL 1D launch parameters grid/block (%d) (%d)\n", \
                  __FILE__, __LINE__,                                                        \
                  gridsize[0], blocksize[0]); */                                             \
   }                                                                                         \
   else                                                                                      \
   {                                                                                         \
      nalu_hypre_HandleComputeStream(nalu_hypre_handle())->submit([&] (sycl::handler& cgh) {           \
         auto debug_stream = sycl::stream(4096, 1024, cgh);                                  \
         cgh.parallel_for(sycl::nd_range<1>(gridsize*blocksize, blocksize),                  \
            [=] (sycl::nd_item<1> item) [[intel::reqd_sub_group_size(NALU_HYPRE_WARP_SIZE)]]      \
               { (kernel_name)(item, debug_stream, __VA_ARGS__);                             \
         });                                                                                 \
      }).wait_and_throw();                                                                   \
   }                                                                                         \
}

#define NALU_HYPRE_GPU_LAUNCH(kernel_name, gridsize, blocksize, ...) NALU_HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, 0, __VA_ARGS__)
#endif // defined(NALU_HYPRE_USING_SYCL)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      macros for wrapping cuda/hip/sycl calls for error reporting
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_CUDA)
#define NALU_HYPRE_CUDA_CALL(call) do {                                                           \
   cudaError_t err = call;                                                                   \
   if (cudaSuccess != err) {                                                                 \
      printf("CUDA ERROR (code = %d, %s) at %s:%d\n", err, cudaGetErrorString(err),          \
                   __FILE__, __LINE__);                                                      \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)

#elif defined(NALU_HYPRE_USING_HIP)
#define NALU_HYPRE_HIP_CALL(call) do {                                                            \
   hipError_t err = call;                                                                    \
   if (hipSuccess != err) {                                                                  \
      printf("HIP ERROR (code = %d, %s) at %s:%d\n", err, hipGetErrorString(err),            \
                   __FILE__, __LINE__);                                                      \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)

#elif defined(NALU_HYPRE_USING_SYCL)
#define NALU_HYPRE_SYCL_CALL(call)                                                                \
   try                                                                                       \
   {                                                                                         \
      call;                                                                                  \
   }                                                                                         \
   catch (sycl::exception const &ex)                                                         \
   {                                                                                         \
      nalu_hypre_printf("SYCL ERROR (code = %s) at %s:%d\n", ex.what(),                           \
                     __FILE__, __LINE__);                                                    \
      assert(0); exit(1);                                                                    \
   }                                                                                         \
   catch(std::runtime_error const& ex)                                                       \
   {                                                                                         \
      nalu_hypre_printf("STD ERROR (code = %s) at %s:%d\n", ex.what(),                            \
                   __FILE__, __LINE__);                                                      \
      assert(0); exit(1);                                                                    \
   }
#endif

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      macros for wrapping vendor library calls for error reporting
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_COMPLEX) /* Double Complex */
/* TODO */
#elif defined(NALU_HYPRE_SINGLE) /* Single */
/* cublas */
#define nalu_hypre_cublas_scal                      cublasSscal
#define nalu_hypre_cublas_axpy                      cublasSaxpy
#define nalu_hypre_cublas_dot                       cublasSdot
/* cusparse */
#define nalu_hypre_cusparse_csru2csr_bufferSizeExt  cusparseScsru2csr_bufferSizeExt
#define nalu_hypre_cusparse_csru2csr                cusparseScsru2csr
#define nalu_hypre_cusparse_csrsv2_bufferSize       cusparseScsrsv2_bufferSize
#define nalu_hypre_cusparse_csrsv2_analysis         cusparseScsrsv2_analysis
#define nalu_hypre_cusparse_csrsv2_solve            cusparseScsrsv2_solve
#define nalu_hypre_cusparse_csrmv                   cusparseScsrmv
#define nalu_hypre_cusparse_csrgemm                 cusparseScsrgemm
#define nalu_hypre_cusparse_csr2csc                 cusparseScsr2csc
#define nalu_hypre_cusparse_csrilu02_bufferSize     cusparseScsrilu02_bufferSize
#define nalu_hypre_cusparse_csrilu02_analysis       cusparseScsrilu02_analysis
#define nalu_hypre_cusparse_csrilu02                cusparseScsrilu02
#define nalu_hypre_cusparse_csrsm2_bufferSizeExt    cusparseScsrsm2_bufferSizeExt
#define nalu_hypre_cusparse_csrsm2_analysis         cusparseScsrsm2_analysis
#define nalu_hypre_cusparse_csrsm2_solve            cusparseScsrsm2_solve
/* rocsparse */
#define nalu_hypre_rocsparse_csrsv_buffer_size      rocsparse_scsrsv_buffer_size
#define nalu_hypre_rocsparse_csrsv_analysis         rocsparse_scsrsv_analysis
#define nalu_hypre_rocsparse_csrsv_solve            rocsparse_scsrsv_solve
#define nalu_hypre_rocsparse_gthr                   rocsparse_sgthr
#define nalu_hypre_rocsparse_csrmv_analysis         rocsparse_scsrmv_analysis
#define nalu_hypre_rocsparse_csrmv                  rocsparse_scsrmv
#define nalu_hypre_rocsparse_csrgemm_buffer_size    rocsparse_scsrgemm_buffer_size
#define nalu_hypre_rocsparse_csrgemm                rocsparse_scsrgemm
#define nalu_hypre_rocsparse_csr2csc                rocsparse_scsr2csc
#elif defined(NALU_HYPRE_LONG_DOUBLE) /* Long Double */
/* ... */
#else /* Double */
/* cublas */
#define nalu_hypre_cublas_scal                      cublasDscal
#define nalu_hypre_cublas_axpy                      cublasDaxpy
#define nalu_hypre_cublas_dot                       cublasDdot
/* cusparse */
#define nalu_hypre_cusparse_csru2csr_bufferSizeExt  cusparseDcsru2csr_bufferSizeExt
#define nalu_hypre_cusparse_csru2csr                cusparseDcsru2csr
#define nalu_hypre_cusparse_csrsv2_bufferSize       cusparseDcsrsv2_bufferSize
#define nalu_hypre_cusparse_csrsv2_analysis         cusparseDcsrsv2_analysis
#define nalu_hypre_cusparse_csrsv2_solve            cusparseDcsrsv2_solve
#define nalu_hypre_cusparse_csrmv                   cusparseDcsrmv
#define nalu_hypre_cusparse_csrgemm                 cusparseDcsrgemm
#define nalu_hypre_cusparse_csr2csc                 cusparseDcsr2csc
#define nalu_hypre_cusparse_csrilu02_bufferSize     cusparseDcsrilu02_bufferSize
#define nalu_hypre_cusparse_csrilu02_analysis       cusparseDcsrilu02_analysis
#define nalu_hypre_cusparse_csrilu02                cusparseDcsrilu02
#define nalu_hypre_cusparse_csrsm2_bufferSizeExt    cusparseDcsrsm2_bufferSizeExt
#define nalu_hypre_cusparse_csrsm2_analysis         cusparseDcsrsm2_analysis
#define nalu_hypre_cusparse_csrsm2_solve            cusparseDcsrsm2_solve
/* rocsparse */
#define nalu_hypre_rocsparse_csrsv_buffer_size      rocsparse_dcsrsv_buffer_size
#define nalu_hypre_rocsparse_csrsv_analysis         rocsparse_dcsrsv_analysis
#define nalu_hypre_rocsparse_csrsv_solve            rocsparse_dcsrsv_solve
#define nalu_hypre_rocsparse_gthr                   rocsparse_dgthr
#define nalu_hypre_rocsparse_csrmv_analysis         rocsparse_dcsrmv_analysis
#define nalu_hypre_rocsparse_csrmv                  rocsparse_dcsrmv
#define nalu_hypre_rocsparse_csrgemm_buffer_size    rocsparse_dcsrgemm_buffer_size
#define nalu_hypre_rocsparse_csrgemm                rocsparse_dcsrgemm
#define nalu_hypre_rocsparse_csr2csc                rocsparse_dcsr2csc
#endif


#define NALU_HYPRE_CUBLAS_CALL(call) do {                                                         \
   cublasStatus_t err = call;                                                                \
   if (CUBLAS_STATUS_SUCCESS != err) {                                                       \
      printf("CUBLAS ERROR (code = %d, %d) at %s:%d\n",                                      \
            err, err == CUBLAS_STATUS_EXECUTION_FAILED, __FILE__, __LINE__);                 \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define NALU_HYPRE_CUSPARSE_CALL(call) do {                                                       \
   cusparseStatus_t err = call;                                                              \
   if (CUSPARSE_STATUS_SUCCESS != err) {                                                     \
      printf("CUSPARSE ERROR (code = %d, %s) at %s:%d\n",                                    \
            err, cusparseGetErrorString(err), __FILE__, __LINE__);                           \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define NALU_HYPRE_ROCSPARSE_CALL(call) do {                                                      \
   rocsparse_status err = call;                                                              \
   if (rocsparse_status_success != err) {                                                    \
      printf("rocSPARSE ERROR (code = %d) at %s:%d\n",                                       \
            err, __FILE__, __LINE__);                                                        \
      assert(0); exit(1);                                                                    \
   } } while(0)

#define NALU_HYPRE_CURAND_CALL(call) do {                                                         \
   curandStatus_t err = call;                                                                \
   if (CURAND_STATUS_SUCCESS != err) {                                                       \
      printf("CURAND ERROR (code = %d) at %s:%d\n", err, __FILE__, __LINE__);                \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define NALU_HYPRE_ROCRAND_CALL(call) do {                                                        \
   rocrand_status err = call;                                                                \
   if (ROCRAND_STATUS_SUCCESS != err) {                                                      \
      printf("ROCRAND ERROR (code = %d) at %s:%d\n", err, __FILE__, __LINE__);               \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define NALU_HYPRE_ONEMKL_CALL(call)                                                              \
   try                                                                                       \
   {                                                                                         \
      call;                                                                                  \
   }                                                                                         \
   catch (oneapi::mkl::exception const &ex)                                                  \
   {                                                                                         \
      nalu_hypre_printf("ONEMKL ERROR (code = %s) at %s:%d\n", ex.what(),                         \
                   __FILE__, __LINE__);                                                      \
      assert(0); exit(1);                                                                    \
   }

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      macros for wrapping thrust/oneDPL calls for error reporting
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/* RL: TODO Want macro NALU_HYPRE_THRUST_CALL to return value but I don't know how to do it right
 * The following one works OK for now */

#if defined(NALU_HYPRE_USING_CUDA)
#define NALU_HYPRE_THRUST_CALL(func_name, ...) \
   thrust::func_name(thrust::cuda::par(nalu_hypre_HandleDeviceAllocator(nalu_hypre_handle())).on(nalu_hypre_HandleComputeStream(nalu_hypre_handle())), __VA_ARGS__);
#elif defined(NALU_HYPRE_USING_HIP)
#define NALU_HYPRE_THRUST_CALL(func_name, ...) \
   thrust::func_name(thrust::hip::par(nalu_hypre_HandleDeviceAllocator(nalu_hypre_handle())).on(nalu_hypre_HandleComputeStream(nalu_hypre_handle())), __VA_ARGS__);
#elif defined(NALU_HYPRE_USING_SYCL)
#define NALU_HYPRE_ONEDPL_CALL(func_name, ...)                                                    \
  func_name(oneapi::dpl::execution::make_device_policy(                                      \
           *nalu_hypre_HandleComputeStream(nalu_hypre_handle())), __VA_ARGS__);
#endif

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      device info data structures
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct nalu_hypre_cub_CachingDeviceAllocator;
typedef struct nalu_hypre_cub_CachingDeviceAllocator nalu_hypre_cub_CachingDeviceAllocator;

struct nalu_hypre_DeviceData
{
#if defined(NALU_HYPRE_USING_CURAND)
   curandGenerator_t                 curand_generator;
#endif

#if defined(NALU_HYPRE_USING_ROCRAND)
   rocrand_generator                 curand_generator;
#endif

#if defined(NALU_HYPRE_USING_CUBLAS)
   cublasHandle_t                    cublas_handle;
#endif

#if defined(NALU_HYPRE_USING_CUSPARSE)
   cusparseHandle_t                  cusparse_handle;
#endif

#if defined(NALU_HYPRE_USING_ROCSPARSE)
   rocsparse_handle                  cusparse_handle;
#endif

#if defined(NALU_HYPRE_USING_CUDA_STREAMS)
#if defined(NALU_HYPRE_USING_CUDA)
   cudaStream_t                      streams[NALU_HYPRE_MAX_NUM_STREAMS];
#elif defined(NALU_HYPRE_USING_HIP)
   hipStream_t                       streams[NALU_HYPRE_MAX_NUM_STREAMS];
#elif defined(NALU_HYPRE_USING_SYCL)
   sycl::queue*                      streams[NALU_HYPRE_MAX_NUM_STREAMS] = {NULL};
#endif
#endif

#if defined(NALU_HYPRE_USING_DEVICE_POOL)
   nalu_hypre_uint                        cub_bin_growth;
   nalu_hypre_uint                        cub_min_bin;
   nalu_hypre_uint                        cub_max_bin;
   size_t                            cub_max_cached_bytes;
   nalu_hypre_cub_CachingDeviceAllocator *cub_dev_allocator;
   nalu_hypre_cub_CachingDeviceAllocator *cub_uvm_allocator;
#endif

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_device_allocator            device_allocator;
#endif
#if defined(NALU_HYPRE_USING_SYCL)
   sycl::device                     *device;
   NALU_HYPRE_Int                         device_max_work_group_size;
#else
   NALU_HYPRE_Int                         device;
#endif
   nalu_hypre_int                         device_max_shmem_per_block[2];
   /* by default, hypre puts GPU computations in this stream
    * Do not be confused with the default (null) stream */
   NALU_HYPRE_Int                         compute_stream_num;
   /* work space for hypre's device reducer */
   void                             *reduce_buffer;
   /* the device buffers needed to do MPI communication for struct comm */
   NALU_HYPRE_Complex*                    struct_comm_recv_buffer;
   NALU_HYPRE_Complex*                    struct_comm_send_buffer;
   NALU_HYPRE_Int                         struct_comm_recv_buffer_size;
   NALU_HYPRE_Int                         struct_comm_send_buffer_size;
   /* device spgemm options */
   NALU_HYPRE_Int                         spgemm_algorithm;
   NALU_HYPRE_Int                         spgemm_binned;
   NALU_HYPRE_Int                         spgemm_num_bin;
   /* the highest bins for symbl [0] and numer [1]
    * which are not necessary to be `spgemm_num_bin' due to shmem limit on GPUs */
   NALU_HYPRE_Int                         spgemm_highest_bin[2];
   /* for bin i: ([0][i], [2][i]) = (max #block to launch, block dimension) for symbl
    *            ([1][i], [3][i]) = (max #block to launch, block dimension) for numer */
   NALU_HYPRE_Int                         spgemm_block_num_dim[4][NALU_HYPRE_SPGEMM_MAX_NBIN + 1];
   NALU_HYPRE_Int                         spgemm_rownnz_estimate_method;
   NALU_HYPRE_Int                         spgemm_rownnz_estimate_nsamples;
   float                             spgemm_rownnz_estimate_mult_factor;
   /* cusparse */
   NALU_HYPRE_Int                         spmv_use_vendor;
   NALU_HYPRE_Int                         sptrans_use_vendor;
   NALU_HYPRE_Int                         spgemm_use_vendor;
   /* PMIS RNG */
   NALU_HYPRE_Int                         use_gpu_rand;
};

#define nalu_hypre_DeviceDataCubBinGrowth(data)                   ((data) -> cub_bin_growth)
#define nalu_hypre_DeviceDataCubMinBin(data)                      ((data) -> cub_min_bin)
#define nalu_hypre_DeviceDataCubMaxBin(data)                      ((data) -> cub_max_bin)
#define nalu_hypre_DeviceDataCubMaxCachedBytes(data)              ((data) -> cub_max_cached_bytes)
#define nalu_hypre_DeviceDataCubDevAllocator(data)                ((data) -> cub_dev_allocator)
#define nalu_hypre_DeviceDataCubUvmAllocator(data)                ((data) -> cub_uvm_allocator)
#define nalu_hypre_DeviceDataDevice(data)                         ((data) -> device)
#define nalu_hypre_DeviceDataDeviceMaxWorkGroupSize(data)         ((data) -> device_max_work_group_size)
#define nalu_hypre_DeviceDataDeviceMaxShmemPerBlock(data)         ((data) -> device_max_shmem_per_block)
#define nalu_hypre_DeviceDataComputeStreamNum(data)               ((data) -> compute_stream_num)
#define nalu_hypre_DeviceDataReduceBuffer(data)                   ((data) -> reduce_buffer)
#define nalu_hypre_DeviceDataStructCommRecvBuffer(data)           ((data) -> struct_comm_recv_buffer)
#define nalu_hypre_DeviceDataStructCommSendBuffer(data)           ((data) -> struct_comm_send_buffer)
#define nalu_hypre_DeviceDataStructCommRecvBufferSize(data)       ((data) -> struct_comm_recv_buffer_size)
#define nalu_hypre_DeviceDataStructCommSendBufferSize(data)       ((data) -> struct_comm_send_buffer_size)
#define nalu_hypre_DeviceDataSpgemmUseVendor(data)                ((data) -> spgemm_use_vendor)
#define nalu_hypre_DeviceDataSpMVUseVendor(data)                  ((data) -> spmv_use_vendor)
#define nalu_hypre_DeviceDataSpTransUseVendor(data)               ((data) -> sptrans_use_vendor)
#define nalu_hypre_DeviceDataSpgemmAlgorithm(data)                ((data) -> spgemm_algorithm)
#define nalu_hypre_DeviceDataSpgemmBinned(data)                   ((data) -> spgemm_binned)
#define nalu_hypre_DeviceDataSpgemmNumBin(data)                   ((data) -> spgemm_num_bin)
#define nalu_hypre_DeviceDataSpgemmHighestBin(data)               ((data) -> spgemm_highest_bin)
#define nalu_hypre_DeviceDataSpgemmBlockNumDim(data)              ((data) -> spgemm_block_num_dim)
#define nalu_hypre_DeviceDataSpgemmRownnzEstimateMethod(data)     ((data) -> spgemm_rownnz_estimate_method)
#define nalu_hypre_DeviceDataSpgemmRownnzEstimateNsamples(data)   ((data) -> spgemm_rownnz_estimate_nsamples)
#define nalu_hypre_DeviceDataSpgemmRownnzEstimateMultFactor(data) ((data) -> spgemm_rownnz_estimate_mult_factor)
#define nalu_hypre_DeviceDataDeviceAllocator(data)                ((data) -> device_allocator)
#define nalu_hypre_DeviceDataUseGpuRand(data)                     ((data) -> use_gpu_rand)

nalu_hypre_DeviceData*     nalu_hypre_DeviceDataCreate();
void                nalu_hypre_DeviceDataDestroy(nalu_hypre_DeviceData* data);

#if defined(NALU_HYPRE_USING_CURAND)
curandGenerator_t   nalu_hypre_DeviceDataCurandGenerator(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_ROCRAND)
rocrand_generator   nalu_hypre_DeviceDataCurandGenerator(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_CUBLAS)
cublasHandle_t      nalu_hypre_DeviceDataCublasHandle(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_CUSPARSE)
cusparseHandle_t    nalu_hypre_DeviceDataCusparseHandle(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_ROCSPARSE)
rocsparse_handle    nalu_hypre_DeviceDataCusparseHandle(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_CUDA)
cudaStream_t        nalu_hypre_DeviceDataStream(nalu_hypre_DeviceData *data, NALU_HYPRE_Int i);
cudaStream_t        nalu_hypre_DeviceDataComputeStream(nalu_hypre_DeviceData *data);
#elif defined(NALU_HYPRE_USING_HIP)
hipStream_t         nalu_hypre_DeviceDataStream(nalu_hypre_DeviceData *data, NALU_HYPRE_Int i);
hipStream_t         nalu_hypre_DeviceDataComputeStream(nalu_hypre_DeviceData *data);
#elif defined(NALU_HYPRE_USING_SYCL)
sycl::queue*        nalu_hypre_DeviceDataStream(nalu_hypre_DeviceData *data, NALU_HYPRE_Int i);
sycl::queue*        nalu_hypre_DeviceDataComputeStream(nalu_hypre_DeviceData *data);
#endif

// Data structure and accessor routines for Cuda Sparse Triangular Matrices
struct nalu_hypre_CsrsvData
{
#if defined(NALU_HYPRE_USING_CUSPARSE)
   csrsv2Info_t info_L;
   csrsv2Info_t info_U;
#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   rocsparse_mat_info info_L;
   rocsparse_mat_info info_U;
#elif defined(NALU_HYPRE_USING_ONEMKLSPARSE)
   /* WM: todo - placeholders */
   char info_L;
   char info_U;
#endif
   nalu_hypre_int    BufferSize;
   char        *Buffer;
};

#define nalu_hypre_CsrsvDataInfoL(data)      ((data) -> info_L)
#define nalu_hypre_CsrsvDataInfoU(data)      ((data) -> info_U)
#define nalu_hypre_CsrsvDataBufferSize(data) ((data) -> BufferSize)
#define nalu_hypre_CsrsvDataBuffer(data)     ((data) -> Buffer)

struct nalu_hypre_GpuMatData
{
#if defined(NALU_HYPRE_USING_CUSPARSE)
   cusparseMatDescr_t    mat_descr;
   char                 *spmv_buffer;
#endif

#if defined(NALU_HYPRE_USING_ROCSPARSE)
   rocsparse_mat_descr   mat_descr;
   rocsparse_mat_info    mat_info;
#endif

#if defined(NALU_HYPRE_USING_ONEMKLSPARSE)
   oneapi::mkl::sparse::matrix_handle_t mat_handle;
#endif
};

#define nalu_hypre_GpuMatDataMatDecsr(data)    ((data) -> mat_descr)
#define nalu_hypre_GpuMatDataMatInfo(data)     ((data) -> mat_info)
#define nalu_hypre_GpuMatDataMatHandle(data)   ((data) -> mat_handle)
#define nalu_hypre_GpuMatDataSpMVBuffer(data)  ((data) -> spmv_buffer)

#endif //#if defined(NALU_HYPRE_USING_GPU)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      generic device functions (cuda/hip/sycl)
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
template <typename T>
static __device__ __forceinline__
T read_only_load( const T *ptr )
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
   return __ldg( ptr );
#else
   return *ptr;
#endif
}
#endif // defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      cuda/hip functions
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/* return the number of threads in block */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_num_threads(nalu_hypre_DeviceItem &item)
{
   switch (dim)
   {
      case 1:
         return (blockDim.x);
      case 2:
         return (blockDim.x * blockDim.y);
      case 3:
         return (blockDim.x * blockDim.y * blockDim.z);
   }

   return -1;
}

/* return the flattened thread id in block */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_thread_id(nalu_hypre_DeviceItem &item)
{
   switch (dim)
   {
      case 1:
         return (threadIdx.x);
      case 2:
         return (threadIdx.y * blockDim.x + threadIdx.x);
      case 3:
         return (threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
                 threadIdx.x);
   }

   return -1;
}

/* return the number of warps in block */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_num_warps(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_num_threads<dim>(item) >> NALU_HYPRE_WARP_BITSHIFT;
}

/* return the warp id in block */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_warp_id(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_thread_id<dim>(item) >> NALU_HYPRE_WARP_BITSHIFT;
}

/* return the thread lane id in warp */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_lane_id(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_thread_id<dim>(item) & (NALU_HYPRE_WARP_SIZE - 1);
}

/* return the num of blocks in grid */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_cuda_get_num_blocks()
{
   switch (dim)
   {
      case 1:
         return (gridDim.x);
      case 2:
         return (gridDim.x * gridDim.y);
      case 3:
         return (gridDim.x * gridDim.y * gridDim.z);
   }

   return -1;
}

/* return the flattened block id in grid */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_cuda_get_block_id()
{
   switch (dim)
   {
      case 1:
         return (blockIdx.x);
      case 2:
         return (blockIdx.y * gridDim.x + blockIdx.x);
      case 3:
         return (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x +
                 blockIdx.x);
   }

   return -1;
}

/* return the number of threads in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_num_threads(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_cuda_get_num_blocks<gdim>() * nalu_hypre_gpu_get_num_threads<bdim>(item);
}

/* return the flattened thread id in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_thread_id(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_cuda_get_block_id<gdim>() * nalu_hypre_gpu_get_num_threads<bdim>(item) +
          nalu_hypre_gpu_get_thread_id<bdim>(item);
}

/* return the number of warps in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_num_warps(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_cuda_get_num_blocks<gdim>() * nalu_hypre_gpu_get_num_warps<bdim>(item);
}

/* return the flattened warp id in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_warp_id(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_cuda_get_block_id<gdim>() * nalu_hypre_gpu_get_num_warps<bdim>(item) +
          nalu_hypre_gpu_get_warp_id<bdim>(item);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __device__ __forceinline__
nalu_hypre_double atomicAdd(nalu_hypre_double* address, nalu_hypre_double val)
{
   nalu_hypre_ulonglongint* address_as_ull = (nalu_hypre_ulonglongint*) address;
   nalu_hypre_ulonglongint old = *address_as_ull, assumed;

   do
   {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(val +
                                           __longlong_as_double(assumed)));

      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
   }
   while (assumed != old);

   return __longlong_as_double(old);
}
#endif

// There are no *_sync functions in HIP
#if defined(NALU_HYPRE_USING_HIP) || (CUDA_VERSION < 9000)

template <typename T>
static __device__ __forceinline__
T __shfl_sync(unsigned mask, T val, nalu_hypre_int src_line, nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl(val, src_line, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_down_sync(unsigned mask, T val, unsigned delta, nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_down(val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_xor_sync(unsigned mask, T val, unsigned lanemask, nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_xor(val, lanemask, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_up_sync(unsigned mask, T val, unsigned delta, nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_up(val, delta, width);
}

static __device__ __forceinline__
void __syncwarp()
{
}

#endif // #if defined(NALU_HYPRE_USING_HIP) || (CUDA_VERSION < 9000)


// __any and __ballot were technically deprecated in CUDA 7 so we don't bother
// with these overloads for CUDA, just for HIP.
#if defined(NALU_HYPRE_USING_HIP)
static __device__ __forceinline__
nalu_hypre_int __any_sync(unsigned mask, nalu_hypre_int predicate)
{
   return __any(predicate);
}

static __device__ __forceinline__
nalu_hypre_int __ballot_sync(unsigned mask, nalu_hypre_int predicate)
{
   return __ballot(predicate);
}
#endif

/* exclusive prefix scan */
template <typename T>
static __device__ __forceinline__
T warp_prefix_sum(nalu_hypre_DeviceItem &item, nalu_hypre_int lane_id, T in, T &all_sum)
{
#pragma unroll
   for (nalu_hypre_int d = 2; d <= NALU_HYPRE_WARP_SIZE; d <<= 1)
   {
      T t = __shfl_up_sync(NALU_HYPRE_WARP_FULL_MASK, in, d >> 1);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, in, NALU_HYPRE_WARP_SIZE - 1);

   if (lane_id == NALU_HYPRE_WARP_SIZE - 1)
   {
      in = 0;
   }

#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      T t = __shfl_xor_sync(NALU_HYPRE_WARP_FULL_MASK, in, d);

      if ( (lane_id & (d - 1)) == (d - 1))
      {
         if ( (lane_id & ((d << 1) - 1)) == ((d << 1) - 1) )
         {
            in += t;
         }
         else
         {
            in = t;
         }
      }
   }
   return in;
}

static __device__ __forceinline__
nalu_hypre_int warp_any_sync(nalu_hypre_DeviceItem &item, unsigned mask, nalu_hypre_int predicate)
{
   return __any_sync(mask, predicate);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int src_line,
                    nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_sync(mask, val, src_line, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_up_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int delta,
                       nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_up_sync(mask, val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_down_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int delta,
                         nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_down_sync(mask, val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_xor_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int lane_mask,
                        nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_xor_sync(mask, val, lane_mask, width);
}

template <typename T>
static __device__ __forceinline__
T warp_reduce_sum(nalu_hypre_DeviceItem &item, T in)
{
#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, in, d);
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_allreduce_sum(nalu_hypre_DeviceItem &item, T in)
{
#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in += __shfl_xor_sync(NALU_HYPRE_WARP_FULL_MASK, in, d);
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_reduce_max(nalu_hypre_DeviceItem &item, T in)
{
#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = max(in, __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, in, d));
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_allreduce_max(nalu_hypre_DeviceItem &item, T in)
{
#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = max(in, __shfl_xor_sync(NALU_HYPRE_WARP_FULL_MASK, in, d));
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_reduce_min(nalu_hypre_DeviceItem &item, T in)
{
#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = min(in, __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, in, d));
   }
   return in;
}

template <typename T>
static __device__ __forceinline__
T warp_allreduce_min(nalu_hypre_DeviceItem &item, T in)
{
#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = min(in, __shfl_xor_sync(NALU_HYPRE_WARP_FULL_MASK, in, d));
   }
   return in;
}

static __device__ __forceinline__
nalu_hypre_int next_power_of_2(nalu_hypre_int n)
{
   if (n <= 0)
   {
      return 0;
   }

   /* if n is power of 2, return itself */
   if ( (n & (n - 1)) == 0 )
   {
      return n;
   }

   n |= (n >>  1);
   n |= (n >>  2);
   n |= (n >>  4);
   n |= (n >>  8);
   n |= (n >> 16);
   n ^= (n >>  1);
   n  = (n <<  1);

   return n;
}

template<typename T1, typename T2>
struct type_cast : public thrust::unary_function<T1, T2>
{
   __host__ __device__ T2 operator()(const T1 &x) const
   {
      return (T2) x;
   }
};

template<typename T>
struct absolute_value : public thrust::unary_function<T, T>
{
   __host__ __device__ T operator()(const T &x) const
   {
      return x < T(0) ? -x : x;
   }
};

template<typename T1, typename T2>
struct TupleComp2
{
   typedef thrust::tuple<T1, T2> Tuple;

   __host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
   {
      if (thrust::get<0>(t1) < thrust::get<0>(t2))
      {
         return true;
      }
      if (thrust::get<0>(t1) > thrust::get<0>(t2))
      {
         return false;
      }
      return nalu_hypre_abs(thrust::get<1>(t1)) > nalu_hypre_abs(thrust::get<1>(t2));
   }
};

template<typename T1, typename T2>
struct TupleComp3
{
   typedef thrust::tuple<T1, T2> Tuple;

   __host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
   {
      if (thrust::get<0>(t1) < thrust::get<0>(t2))
      {
         return true;
      }
      if (thrust::get<0>(t1) > thrust::get<0>(t2))
      {
         return false;
      }
      if (thrust::get<0>(t2) == thrust::get<1>(t2))
      {
         return false;
      }
      return thrust::get<0>(t1) == thrust::get<1>(t1) || thrust::get<1>(t1) < thrust::get<1>(t2);
   }
};

template<typename T>
struct is_negative : public thrust::unary_function<T, bool>
{
   __host__ __device__ bool operator()(const T &x)
   {
      return (x < 0);
   }
};

template<typename T>
struct is_positive : public thrust::unary_function<T, bool>
{
   __host__ __device__ bool operator()(const T &x)
   {
      return (x > 0);
   }
};

template<typename T>
struct is_nonnegative : public thrust::unary_function<T, bool>
{
   __host__ __device__ bool operator()(const T &x)
   {
      return (x >= 0);
   }
};

template<typename T>
struct in_range : public thrust::unary_function<T, bool>
{
   T low, up;

   in_range(T low_, T up_) { low = low_; up = up_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x >= low && x <= up);
   }
};

template<typename T>
struct out_of_range : public thrust::unary_function<T, bool>
{
   T low, up;

   out_of_range(T low_, T up_) { low = low_; up = up_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x < low || x > up);
   }
};

template<typename T>
struct less_than : public thrust::unary_function<T, bool>
{
   T val;

   less_than(T val_) { val = val_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x < val);
   }
};

template<typename T>
struct modulo : public thrust::unary_function<T, T>
{
   T val;

   modulo(T val_) { val = val_; }

   __host__ __device__ T operator()(const T &x)
   {
      return (x % val);
   }
};

template<typename T>
struct equal : public thrust::unary_function<T, bool>
{
   T val;

   equal(T val_) { val = val_; }

   __host__ __device__ bool operator()(const T &x)
   {
      return (x == val);
   }
};

struct print_functor
{
   __host__ __device__ void operator()(NALU_HYPRE_Real val)
   {
      printf("%f\n", val);
   }
};

#endif // defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      sycl functions
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_SYCL)

/* return the number of threads in block */
/* WM: todo - only supports bdim = gdim = 1 DOUBLE CHECK THIS */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_num_threads(nalu_hypre_DeviceItem &item)
{
   return static_cast<NALU_HYPRE_Int>(item.get_group().get_group_range().get(0));
}

/* return the flattened thread id in block */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_thread_id(nalu_hypre_DeviceItem &item)
{
   return static_cast<NALU_HYPRE_Int>(item.get_local_linear_id());
}

/* return the flattened thread id in grid */
/* WM: todo - only supports bdim = gdim = 1 */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_thread_id(nalu_hypre_DeviceItem &item)
{
   return static_cast<NALU_HYPRE_Int>(item.get_global_linear_id());
}

/* return the flattened warp id in nd_range */
/* WM: todo - only supports bdim = gdim = 1 */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_warp_id(nalu_hypre_DeviceItem &item)
{
   return item.get_group_linear_id() * item.get_sub_group().get_group_range().get(0) +
          item.get_sub_group().get_group_linear_id();
}

/* return the thread lane id in warp */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_lane_id(nalu_hypre_DeviceItem &item)
{
   return (nalu_hypre_int) item.get_sub_group().get_local_linear_id();
}

/* exclusive prefix scan */
template <typename T>
static __device__ __forceinline__
T warp_prefix_sum(nalu_hypre_DeviceItem &item, nalu_hypre_int lane_id, T in, T &all_sum)
{
#pragma unroll
   for (nalu_hypre_int d = 2; d <= NALU_HYPRE_WARP_SIZE; d <<= 1)
   {
      T t = item.get_sub_group().shuffle_up(in, d >> 1);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = item.get_sub_group().shuffle(in, NALU_HYPRE_WARP_SIZE - 1);

   if (lane_id == NALU_HYPRE_WARP_SIZE - 1)
   {
      in = 0;
   }

#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      T t = item.get_sub_group().shuffle_xor(in, d);

      if ( (lane_id & (d - 1)) == (d - 1))
      {
         if ( (lane_id & ((d << 1) - 1)) == ((d << 1) - 1) )
         {
            in += t;
         }
         else
         {
            in = t;
         }
      }
   }
   return in;
}

static __device__ __forceinline__
nalu_hypre_int warp_any_sync(nalu_hypre_DeviceItem &item, unsigned mask, nalu_hypre_int predicate)
{
   return sycl::any_of_group(item.get_sub_group(), predicate);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int src_line)
{
   /* WM: todo - try removing barrier with new implementation */
   item.get_sub_group().barrier();
   return sycl::group_broadcast(item.get_sub_group(), val, src_line);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int src_line,
                    nalu_hypre_int width)
{
   nalu_hypre_int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   nalu_hypre_int group_start = (lane_id / width) * width;
   nalu_hypre_int src_in_warp = group_start + src_line;
   return sycl::select_from_group(item.get_sub_group(), val, src_in_warp);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_up_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int delta)
{
   return sycl::shift_group_right(item.get_sub_group(), val, delta);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_up_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int delta,
                       nalu_hypre_int width)
{
   nalu_hypre_int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   nalu_hypre_int group_start = (lane_id / width) * width;
   nalu_hypre_int src_in_warp = sycl::max(group_start, lane_id - delta);
   return sycl::select_from_group(item.get_sub_group(), val, src_in_warp);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_down_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int delta)
{
   return sycl::shift_group_left(item.get_sub_group(), val, delta);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_down_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int delta,
                         nalu_hypre_int width)
{
   nalu_hypre_int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   nalu_hypre_int group_end = ((lane_id / width) + 1) * width - 1;
   nalu_hypre_int src_in_warp = sycl::min(group_end, lane_id + delta);
   return sycl::select_from_group(item.get_sub_group(), val, src_in_warp);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_xor_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int lane_mask)
{
   return sycl::permute_group_by_xor(item.get_sub_group(), val, lane_mask);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_xor_sync(nalu_hypre_DeviceItem &item, unsigned mask, T val, nalu_hypre_int lane_mask,
                        nalu_hypre_int width)
{
   nalu_hypre_int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   nalu_hypre_int group_end = ((lane_id / width) + 1) * width - 1;
   nalu_hypre_int src_in_warp = lane_id ^ lane_mask;
   src_in_warp = src_in_warp > group_end ? lane_id : src_in_warp;
   return sycl::select_from_group(item.get_sub_group(), val, src_in_warp);
}

template <typename T>
static __forceinline__
T warp_reduce_sum(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in += item.get_sub_group().shuffle_down(in, d);
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_allreduce_sum(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in += item.get_sub_group().shuffle_xor(in, d);
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_reduce_max(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = std::max(in, item.get_sub_group().shuffle_down(in, d));
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_allreduce_max(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = std::max(in, item.get_sub_group().shuffle_xor(in, d));
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_reduce_min(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = std::min(in, item.get_sub_group().shuffle_down(in, d));
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_allreduce_min(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in = std::min(in, item.get_sub_group().shuffle_xor(in, d));
   }
   return in;
}

template<typename T>
struct is_negative
{
   is_negative() {}

   constexpr bool operator()(const T &x) const { return (x < 0); }
};

template<typename T>
struct is_positive
{
   is_positive() {}

   constexpr bool operator()(const T &x) const { return (x > 0); }
};

template<typename T>
struct is_nonnegative
{
   is_nonnegative() {}

   constexpr bool operator()(const T &x) const { return (x >= 0); }
};

template<typename T>
struct in_range
{
   T low, high;
   in_range(T low_, T high_) { low = low_; high = high_; }

   constexpr bool operator()(const T &x) const { return (x >= low && x <= high); }
};

template<typename T>
struct out_of_range
{
   T low, high;
   out_of_range(T low_, T high_) { low = low_; high = high_; }

   constexpr bool operator()(const T &x) const { return (x < low || x > high); }
};

template<typename T>
struct less_than
{
   T val;
   less_than(T val_) { val = val_; }

   constexpr bool operator()(const T &x) const { return (x < val); }
};

template<typename T>
struct modulo
{
   T val;
   modulo(T val_) { val = val_; }

   constexpr T operator()(const T &x) const { return (x % val); }
};

template<typename T>
struct equal
{
   T val;
   equal(T val_) { val = val_; }

   constexpr bool operator()(const T &x) const { return (x == val); }
};

template<typename... T>
struct TupleComp2
{
   typedef std::tuple<T...> Tuple;
   bool operator()(const Tuple& t1, const Tuple& t2)
   {
      if (std::get<0>(t1) < std::get<0>(t2))
      {
         return true;
      }
      if (std::get<0>(t1) > std::get<0>(t2))
      {
         return false;
      }
      return nalu_hypre_abs(std::get<1>(t1)) > nalu_hypre_abs(std::get<1>(t2));
   }
};

template<typename... T>
struct TupleComp3
{
   typedef std::tuple<T...> Tuple;
   bool operator()(const Tuple& t1, const Tuple& t2)
   {
      if (std::get<0>(t1) < std::get<0>(t2))
      {
         return true;
      }
      if (std::get<0>(t1) > std::get<0>(t2))
      {
         return false;
      }
      if (std::get<0>(t2) == std::get<1>(t2))
      {
         return false;
      }
      return std::get<0>(t1) == std::get<1>(t1) || std::get<1>(t1) < std::get<1>(t2);
   }
};

#endif // #if defined(NALU_HYPRE_USING_SYCL)

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      end of functions defined here
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/* device_utils.c */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
dim3 nalu_hypre_GetDefaultDeviceBlockDimension();

dim3 nalu_hypre_GetDefaultDeviceGridDimension( NALU_HYPRE_Int n, const char *granularity, dim3 bDim );

template <typename T1, typename T2, typename T3> NALU_HYPRE_Int hypreDevice_StableSortByTupleKey(
   NALU_HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals, NALU_HYPRE_Int opt);

template <typename T1, typename T2, typename T3, typename T4> NALU_HYPRE_Int
hypreDevice_StableSortTupleByTupleKey(NALU_HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals1, T4 *vals2,
                                      NALU_HYPRE_Int opt);

template <typename T1, typename T2, typename T3> NALU_HYPRE_Int hypreDevice_ReduceByTupleKey(NALU_HYPRE_Int N,
                                                                                        T1 *keys1_in,  T2 *keys2_in,  T3 *vals_in, T1 *keys1_out, T2 *keys2_out, T3 *vals_out);

template <typename T>
NALU_HYPRE_Int hypreDevice_ScatterConstant(T *x, NALU_HYPRE_Int n, NALU_HYPRE_Int *map, T v);

NALU_HYPRE_Int hypreDevice_GenScatterAdd(NALU_HYPRE_Real *x, NALU_HYPRE_Int ny, NALU_HYPRE_Int *map, NALU_HYPRE_Real *y,
                                    char *work);

template <typename T>
NALU_HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum(NALU_HYPRE_Int nrows, NALU_HYPRE_Int nnz,
                                                    NALU_HYPRE_Int *d_row_ptr, T *d_row_num, T *d_row_ind);

#endif

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

NALU_HYPRE_Int hypreDevice_BigToSmallCopy(NALU_HYPRE_Int *tgt, const NALU_HYPRE_BigInt *src, NALU_HYPRE_Int size);

#if defined(NALU_HYPRE_USING_CUDA)
cudaError_t nalu_hypre_CachingMallocDevice(void **ptr, size_t nbytes);

cudaError_t nalu_hypre_CachingMallocManaged(void **ptr, size_t nbytes);

cudaError_t nalu_hypre_CachingFreeDevice(void *ptr);

cudaError_t nalu_hypre_CachingFreeManaged(void *ptr);
#endif

nalu_hypre_cub_CachingDeviceAllocator * nalu_hypre_DeviceDataCubCachingAllocatorCreate(nalu_hypre_uint bin_growth,
                                                                             nalu_hypre_uint min_bin, nalu_hypre_uint max_bin, size_t max_cached_bytes, bool skip_cleanup, bool debug,
                                                                             bool use_managed_memory);

void nalu_hypre_DeviceDataCubCachingAllocatorDestroy(nalu_hypre_DeviceData *data);

#endif // #if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

#if defined(NALU_HYPRE_USING_CUSPARSE)

cudaDataType nalu_hypre_HYPREComplexToCudaDataType();

cusparseIndexType_t nalu_hypre_HYPREIntToCusparseIndexType();

#endif // #if defined(NALU_HYPRE_USING_CUSPARSE)

#endif /* #ifndef NALU_HYPRE_CUDA_UTILS_H */
