/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_DEVICE_UTILS_H
#define NALU_HYPRE_DEVICE_UTILS_H

#if defined(NALU_HYPRE_USING_GPU)

/* Data types depending on GPU architecture */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_SYCL)
typedef nalu_hypre_uint          nalu_hypre_mask;
#define nalu_hypre_mask_one      1U

#elif defined(NALU_HYPRE_USING_HIP)
typedef nalu_hypre_ulonglongint  nalu_hypre_mask;
#define nalu_hypre_mask_one      1ULL

#endif

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                          cuda includes
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_CUDA)
using nalu_hypre_DeviceItem = void*;
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#if defined(NALU_HYPRE_USING_CURAND)
#include <curand.h>
#endif

#if defined(NALU_HYPRE_USING_CUBLAS)
#include <cublas_v2.h>
#endif

#if defined(NALU_HYPRE_USING_CUSPARSE)
#include <cusparse.h>
#endif

#if defined(NALU_HYPRE_USING_CUSOLVER)
#include <cusolverDn.h>
#endif

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

#ifndef CUSPARSE_VERSION
#if defined(CUSPARSE_VER_MAJOR) && defined(CUSPARSE_VER_MINOR) && defined(CUSPARSE_VER_PATCH)
#define CUSPARSE_VERSION (CUSPARSE_VER_MAJOR * 1000 + CUSPARSE_VER_MINOR *  100 + CUSPARSE_VER_PATCH)
#else
#define CUSPARSE_VERSION CUDA_VERSION
#endif
#endif

#define CUSPARSE_NEWAPI_VERSION 11000
#define CUSPARSE_NEWSPMM_VERSION 11401
#define CUDA_MALLOCASYNC_VERSION 11020
#define CUDA_THRUST_NOSYNC_VERSION 12000

#define CUSPARSE_SPSV_VERSION 11600
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
#define nalu_hypre_cusparseSpSVDescr         cusparseSpSVDescr_t
#define nalu_hypre_cusparseSpSV_createDescr  cusparseSpSV_createDescr
#define nalu_hypre_cusparseSpSV_destroyDescr cusparseSpSV_destroyDescr
#else
#define nalu_hypre_cusparseSpSVDescr         csrsv2Info_t
#define nalu_hypre_cusparseSpSV_createDescr  cusparseCreateCsrsv2Info
#define nalu_hypre_cusparseSpSV_destroyDescr cusparseDestroyCsrsv2Info
#endif

#define CUSPARSE_SPSM_VERSION 11600
#if CUSPARSE_VERSION >= CUSPARSE_SPSM_VERSION
#define nalu_hypre_cusparseSpSMDescr         cusparseSpSMDescr_t
#define nalu_hypre_cusparseSpSM_createDescr  cusparseSpSM_createDescr
#define nalu_hypre_cusparseSpSM_destroyDescr cusparseSpSM_destroyDescr
#else
#define nalu_hypre_cusparseSpSMDescr         csrsm2Info_t
#define nalu_hypre_cusparseSpSM_createDescr  cusparseCreateCsrsm2Info
#define nalu_hypre_cusparseSpSM_destroyDescr cusparseDestroyCsrsm2Info
#endif

#if defined(NALU_HYPRE_USING_DEVICE_MALLOC_ASYNC)
#if CUDA_VERSION < CUDA_MALLOCASYNC_VERSION
#error cudaMalloc/FreeAsync needs CUDA 11.2
#endif
#endif

#if defined(NALU_HYPRE_USING_THRUST_NOSYNC)
#if CUDA_VERSION < CUDA_THRUST_NOSYNC_VERSION
#error thrust::cuda::par_nosync needs CUDA 12
#endif
#define NALU_HYPRE_THRUST_EXECUTION thrust::cuda::par_nosync
#else
#define NALU_HYPRE_THRUST_EXECUTION thrust::cuda::par
#endif

#endif /* defined(NALU_HYPRE_USING_CUDA) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                          hip includes
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_HIP)

using nalu_hypre_DeviceItem = void*;
#include <hip/hip_runtime.h>

#if defined(NALU_HYPRE_USING_ROCBLAS)
#include <rocblas/rocblas.h>
#endif

#if defined(NALU_HYPRE_USING_ROCSPARSE)
#include <rocsparse/rocsparse.h>
#endif

#if defined(NALU_HYPRE_USING_ROCSOLVER)
#include <rocsolver/rocsolver.h>
#endif

#if defined(NALU_HYPRE_USING_ROCRAND)
#include <rocrand/rocrand.h>
#endif

#endif /* defined(NALU_HYPRE_USING_HIP) */

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
#endif /* defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */

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
#include <oneapi/mkl/lapack.hpp>
#endif
#if defined(NALU_HYPRE_USING_ONEMKLRAND)
#include <oneapi/mkl/rng.hpp>
#endif

/* The following definitions facilitate code reuse and limits
 * if/def-ing when unifying cuda/hip code with sycl code */
using dim3 = sycl::range<3>;
using nalu_hypre_DeviceItem = sycl::nd_item<3>;
#define __global__
#define __host__
#define __device__
#define __forceinline__ __inline__ __attribute__((always_inline))

#endif /* defined(NALU_HYPRE_USING_SYCL) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      device defined values
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define NALU_HYPRE_MAX_NTHREADS_BLOCK 1024

// NALU_HYPRE_WARP_BITSHIFT is just log2 of NALU_HYPRE_WARP_SIZE
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_SYCL)
#define NALU_HYPRE_WARP_SIZE       32
#define NALU_HYPRE_WARP_BITSHIFT   5
#define NALU_HYPRE_WARP_FULL_MASK  0xFFFFFFFF
#elif defined(NALU_HYPRE_USING_HIP)
#define NALU_HYPRE_WARP_SIZE       64
#define NALU_HYPRE_WARP_BITSHIFT   6
#define NALU_HYPRE_WARP_FULL_MASK  0xFFFFFFFFFFFFFFF
#endif

#define NALU_HYPRE_MAX_NUM_WARPS   (64 * 64 * 32)
#define NALU_HYPRE_FLT_LARGE       1e30
#define NALU_HYPRE_1D_BLOCK_SIZE   512
#define NALU_HYPRE_MAX_NUM_STREAMS 10
#define NALU_HYPRE_SPGEMM_MAX_NBIN 10

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *       macro for launching GPU kernels
 *       NOTE: IN NALU_HYPRE'S DEFAULT STREAM
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
#endif /* defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */

/* sycl version */
#if defined(NALU_HYPRE_USING_SYCL)

#define NALU_HYPRE_GPU_LAUNCH(kernel_name, gridsize, blocksize, ...)                              \
{                                                                                            \
   if ( gridsize[2] == 0 || blocksize[2] == 0 )                                              \
   {                                                                                         \
     /* nalu_hypre_printf("Warning %s %d: Zero SYCL 1D launch parameters grid/block (%d) (%d)\n", \
                  __FILE__, __LINE__,                                                        \
                  gridsize[0], blocksize[0]); */                                             \
   }                                                                                         \
   else                                                                                      \
   {                                                                                         \
      nalu_hypre_HandleComputeStream(nalu_hypre_handle())->submit([&] (sycl::handler& cgh) {           \
         cgh.parallel_for(sycl::nd_range<3>(gridsize*blocksize, blocksize),                  \
            [=] (nalu_hypre_DeviceItem item) [[intel::reqd_sub_group_size(NALU_HYPRE_WARP_SIZE)]]      \
               { (kernel_name)(item, __VA_ARGS__);                                           \
         });                                                                                 \
      }).wait_and_throw();                                                                   \
   }                                                                                         \
}

#define NALU_HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, shmem_size, ...)                 \
{                                                                                            \
   if ( gridsize[2] == 0 || blocksize[2] == 0 )                                              \
   {                                                                                         \
     /* nalu_hypre_printf("Warning %s %d: Zero SYCL 1D launch parameters grid/block (%d) (%d)\n", \
                  __FILE__, __LINE__,                                                        \
                  gridsize[0], blocksize[0]); */                                             \
   }                                                                                         \
   else                                                                                      \
   {                                                                                         \
      nalu_hypre_HandleComputeStream(nalu_hypre_handle())->submit([&] (sycl::handler& cgh) {           \
         sycl::range<1> shmem_range(shmem_size);                                             \
         sycl::accessor<char, 1, sycl::access_mode::read_write,                              \
            sycl::target::local> shmem_accessor(shmem_range, cgh);                           \
         cgh.parallel_for(sycl::nd_range<3>(gridsize*blocksize, blocksize),                  \
            [=] (nalu_hypre_DeviceItem item) [[intel::reqd_sub_group_size(NALU_HYPRE_WARP_SIZE)]]      \
               { (kernel_name)(item, shmem_accessor.get_pointer(), __VA_ARGS__);             \
         });                                                                                 \
      }).wait_and_throw();                                                                   \
   }                                                                                         \
}

#define NALU_HYPRE_GPU_DEBUG_LAUNCH(kernel_name, gridsize, blocksize, ...)                        \
{                                                                                            \
   if ( gridsize[2] == 0 || blocksize[2] == 0 )                                              \
   {                                                                                         \
     /* nalu_hypre_printf("Warning %s %d: Zero SYCL 1D launch parameters grid/block (%d) (%d)\n", \
                  __FILE__, __LINE__,                                                        \
                  gridsize[0], blocksize[0]); */                                             \
   }                                                                                         \
   else                                                                                      \
   {                                                                                         \
      nalu_hypre_HandleComputeStream(nalu_hypre_handle())->submit([&] (sycl::handler& cgh) {           \
         auto debug_stream = sycl::stream(4096, 1024, cgh);                                  \
         cgh.parallel_for(sycl::nd_range<3>(gridsize*blocksize, blocksize),                  \
            [=] (nalu_hypre_DeviceItem item) [[intel::reqd_sub_group_size(NALU_HYPRE_WARP_SIZE)]]      \
               { (kernel_name)(item, debug_stream, __VA_ARGS__);                             \
         });                                                                                 \
      }).wait_and_throw();                                                                   \
   }                                                                                         \
}

#define NALU_HYPRE_GPU_DEBUG_LAUNCH2(kernel_name, gridsize, blocksize, shmem_size, ...)           \
{                                                                                            \
   if ( gridsize[2] == 0 || blocksize[2] == 0 )                                              \
   {                                                                                         \
     /* nalu_hypre_printf("Warning %s %d: Zero SYCL 1D launch parameters grid/block (%d) (%d)\n", \
                  __FILE__, __LINE__,                                                        \
                  gridsize[0], blocksize[0]); */                                             \
   }                                                                                         \
   else                                                                                      \
   {                                                                                         \
      nalu_hypre_HandleComputeStream(nalu_hypre_handle())->submit([&] (sycl::handler& cgh) {           \
         auto debug_stream = sycl::stream(4096, 1024, cgh);                                  \
         sycl::range<1> shmem_range(shmem_size);                                             \
         sycl::accessor<char, 1, sycl::access_mode::read_write,                              \
            sycl::target::local> shmem_accessor(shmem_range, cgh);                           \
         cgh.parallel_for(sycl::nd_range<3>(gridsize*blocksize, blocksize),                  \
            [=] (nalu_hypre_DeviceItem item) [[intel::reqd_sub_group_size(NALU_HYPRE_WARP_SIZE)]]      \
               { (kernel_name)(item, debug_stream, shmem_accessor.get_pointer(), __VA_ARGS__);\
         });                                                                                 \
      }).wait_and_throw();                                                                   \
   }                                                                                         \
}
#endif /* defined(NALU_HYPRE_USING_SYCL) */

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
#error "GPU build does not support complex numbers!"

#elif defined(NALU_HYPRE_SINGLE) /* Single */
/* cuBLAS */
#define nalu_hypre_cublas_scal                      cublasSscal
#define nalu_hypre_cublas_axpy                      cublasSaxpy
#define nalu_hypre_cublas_dot                       cublasSdot
#define nalu_hypre_cublas_getrfBatched              cublasSgetrfBatched
#define nalu_hypre_cublas_getriBatched              cublasSgetriBatched

/* cuSPARSE */
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

/* rocSPARSE */
#define nalu_hypre_rocsparse_csrsv_buffer_size      rocsparse_scsrsv_buffer_size
#define nalu_hypre_rocsparse_csrsv_analysis         rocsparse_scsrsv_analysis
#define nalu_hypre_rocsparse_csrsv_solve            rocsparse_scsrsv_solve
#define nalu_hypre_rocsparse_gthr                   rocsparse_sgthr
#define nalu_hypre_rocsparse_csrmv_analysis         rocsparse_scsrmv_analysis
#define nalu_hypre_rocsparse_csrmv                  rocsparse_scsrmv
#define nalu_hypre_rocsparse_csrgemm_buffer_size    rocsparse_scsrgemm_buffer_size
#define nalu_hypre_rocsparse_csrgemm                rocsparse_scsrgemm
#define nalu_hypre_rocsparse_csr2csc                rocsparse_scsr2csc
#define nalu_hypre_rocsparse_csrilu0_buffer_size    rocsparse_scsrilu0_buffer_size
#define nalu_hypre_rocsparse_csrilu0_analysis       rocsparse_scsrilu0_analysis
#define nalu_hypre_rocsparse_csrilu0                rocsparse_scsrilu0

#elif defined(NALU_HYPRE_LONG_DOUBLE) /* Long Double */
#error "GPU build does not support Long Double numbers!"

#else /* Double */
/* cuBLAS */
#define nalu_hypre_cublas_scal                      cublasDscal
#define nalu_hypre_cublas_axpy                      cublasDaxpy
#define nalu_hypre_cublas_dot                       cublasDdot
#define nalu_hypre_cublas_getrfBatched              cublasDgetrfBatched
#define nalu_hypre_cublas_getriBatched              cublasDgetriBatched

/* cuSPARSE */
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

/* rocSPARSE */
#define nalu_hypre_rocsparse_csrsv_buffer_size      rocsparse_dcsrsv_buffer_size
#define nalu_hypre_rocsparse_csrsv_analysis         rocsparse_dcsrsv_analysis
#define nalu_hypre_rocsparse_csrsv_solve            rocsparse_dcsrsv_solve
#define nalu_hypre_rocsparse_gthr                   rocsparse_dgthr
#define nalu_hypre_rocsparse_csrmv_analysis         rocsparse_dcsrmv_analysis
#define nalu_hypre_rocsparse_csrmv                  rocsparse_dcsrmv
#define nalu_hypre_rocsparse_csrgemm_buffer_size    rocsparse_dcsrgemm_buffer_size
#define nalu_hypre_rocsparse_csrgemm                rocsparse_dcsrgemm
#define nalu_hypre_rocsparse_csr2csc                rocsparse_dcsr2csc
#define nalu_hypre_rocsparse_csrilu0_buffer_size    rocsparse_dcsrilu0_buffer_size
#define nalu_hypre_rocsparse_csrilu0_analysis       rocsparse_dcsrilu0_analysis
#define nalu_hypre_rocsparse_csrilu0                rocsparse_dcsrilu0
#endif

#define NALU_HYPRE_CUBLAS_CALL(call) do {                                                         \
   cublasStatus_t err = call;                                                                \
   if (CUBLAS_STATUS_SUCCESS != err) {                                                       \
      printf("CUBLAS ERROR (code = %d, %d) at %s:%d\n",                                      \
            err, err == CUBLAS_STATUS_EXECUTION_FAILED, __FILE__, __LINE__);                 \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define NALU_HYPRE_ROCBLAS_CALL(call) do {                                                        \
   rocblas_status err = call;                                                                \
   if (rocblas_status_success != err) {                                                      \
      printf("rocBLAS ERROR (code = %d, %s) at %s:%d\n",                                     \
             err, rocblas_status_to_string(err), __FILE__, __LINE__);                        \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)

#if CUSPARSE_VERSION >= 10300
#define NALU_HYPRE_CUSPARSE_CALL(call) do {                                                       \
   cusparseStatus_t err = call;                                                              \
   if (CUSPARSE_STATUS_SUCCESS != err) {                                                     \
      printf("CUSPARSE ERROR (code = %d, %s) at %s:%d\n",                                    \
            err, cusparseGetErrorString(err), __FILE__, __LINE__);                           \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)
#else
#define NALU_HYPRE_CUSPARSE_CALL(call) do {                                                       \
   cusparseStatus_t err = call;                                                              \
   if (CUSPARSE_STATUS_SUCCESS != err) {                                                     \
      printf("CUSPARSE ERROR (code = %d) at %s:%d\n",                                        \
            err, __FILE__, __LINE__);                                                        \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)
#endif

#define NALU_HYPRE_ROCSPARSE_CALL(call) do {                                                      \
   rocsparse_status err = call;                                                              \
   if (rocsparse_status_success != err) {                                                    \
      printf("rocSPARSE ERROR (code = %d) at %s:%d\n",                                       \
            err, __FILE__, __LINE__);                                                        \
      assert(0); exit(1);                                                                    \
   } } while(0)

#define NALU_HYPRE_CUSOLVER_CALL(call) do {                                                       \
   cusolverStatus_t err = call;                                                              \
   if (CUSOLVER_STATUS_SUCCESS != err) {                                                     \
      printf("cuSOLVER ERROR (code = %d) at %s:%d\n",                                        \
            err, __FILE__, __LINE__);                                                        \
      nalu_hypre_assert(0); exit(1);                                                              \
   } } while(0)

#define NALU_HYPRE_ROCSOLVER_CALL(call) do {                                                      \
   rocblas_status err = call;                                                                \
   if (rocblas_status_success != err) {                                                      \
      printf("rocSOLVER ERROR (code = %d, %s) at %s:%d\n",                                   \
             err, rocblas_status_to_string(err), __FILE__, __LINE__);                        \
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
   thrust::func_name(NALU_HYPRE_THRUST_EXECUTION(nalu_hypre_HandleDeviceAllocator(nalu_hypre_handle())).on(nalu_hypre_HandleComputeStream(nalu_hypre_handle())), __VA_ARGS__);
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

#if defined(NALU_HYPRE_USING_CUSOLVER)
typedef cusolverDnHandle_t vendorSolverHandle_t;
#elif defined(NALU_HYPRE_USING_ROCSOLVER)
typedef rocblas_handle     vendorSolverHandle_t;
#endif

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

#if defined(NALU_HYPRE_USING_CUSOLVER) || defined(NALU_HYPRE_USING_ROCSOLVER)
   vendorSolverHandle_t              vendor_solver_handle;
#endif

   /* TODO (VPM): Change to NALU_HYPRE_USING_GPU_STREAMS*/
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
void                  nalu_hypre_DeviceDataDestroy(nalu_hypre_DeviceData* data);

#if defined(NALU_HYPRE_USING_CURAND)
curandGenerator_t     nalu_hypre_DeviceDataCurandGenerator(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_ROCRAND)
rocrand_generator     nalu_hypre_DeviceDataCurandGenerator(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_CUBLAS)
cublasHandle_t        nalu_hypre_DeviceDataCublasHandle(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_CUSPARSE)
cusparseHandle_t      nalu_hypre_DeviceDataCusparseHandle(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_ROCSPARSE)
rocsparse_handle      nalu_hypre_DeviceDataCusparseHandle(nalu_hypre_DeviceData *data);
#endif

#if defined(NALU_HYPRE_USING_CUSOLVER) || defined(NALU_HYPRE_USING_ROCSOLVER)
vendorSolverHandle_t  nalu_hypre_DeviceDataVendorSolverHandle(nalu_hypre_DeviceData *data);
#endif

/* TODO (VPM): Create a deviceStream_t to encapsulate all stream types below */
#if defined(NALU_HYPRE_USING_CUDA)
cudaStream_t          nalu_hypre_DeviceDataStream(nalu_hypre_DeviceData *data, NALU_HYPRE_Int i);
cudaStream_t          nalu_hypre_DeviceDataComputeStream(nalu_hypre_DeviceData *data);
#elif defined(NALU_HYPRE_USING_HIP)
hipStream_t           nalu_hypre_DeviceDataStream(nalu_hypre_DeviceData *data, NALU_HYPRE_Int i);
hipStream_t           nalu_hypre_DeviceDataComputeStream(nalu_hypre_DeviceData *data);
#elif defined(NALU_HYPRE_USING_SYCL)
sycl::queue*          nalu_hypre_DeviceDataStream(nalu_hypre_DeviceData *data, NALU_HYPRE_Int i);
sycl::queue*          nalu_hypre_DeviceDataComputeStream(nalu_hypre_DeviceData *data);
#endif

/* Data structure and accessor routines for Sparse Triangular Matrices */
struct nalu_hypre_CsrsvData
{
#if defined(NALU_HYPRE_USING_CUSPARSE)
   nalu_hypre_cusparseSpSVDescr   info_L;
   nalu_hypre_cusparseSpSVDescr   info_U;
   cusparseSolvePolicy_t     analysis_policy;
   cusparseSolvePolicy_t     solve_policy;

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   rocsparse_mat_info        info_L;
   rocsparse_mat_info        info_U;
   rocsparse_analysis_policy analysis_policy;
   rocsparse_solve_policy    solve_policy;

#elif defined(NALU_HYPRE_USING_ONEMKLSPARSE)
   /* WM: todo - placeholders */
   char                      info_L;
   char                      info_U;
   char                      analysis_policy;
   char                      solve_policy;
#endif

#if defined(NALU_HYPRE_USING_CUSPARSE) && (CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION)
   size_t                    buffer_size_L;
   size_t                    buffer_size_U;
   char                     *buffer_L;
   char                     *buffer_U;
#else
   nalu_hypre_int                 buffer_size;
   char                     *buffer;
#endif

   /* Temporary array to save matrix values with modified diagonal */
   NALU_HYPRE_Complex            *mat_data;

   /* Flags for checking whether the analysis phase has been executed or not */
   NALU_HYPRE_Int                 analyzed_L;
   NALU_HYPRE_Int                 analyzed_U;
};

#define nalu_hypre_CsrsvDataInfoL(data)          ((data) -> info_L)
#define nalu_hypre_CsrsvDataInfoU(data)          ((data) -> info_U)
#define nalu_hypre_CsrsvDataAnalyzedL(data)      ((data) -> analyzed_L)
#define nalu_hypre_CsrsvDataAnalyzedU(data)      ((data) -> analyzed_U)
#define nalu_hypre_CsrsvDataSolvePolicy(data)    ((data) -> solve_policy)
#define nalu_hypre_CsrsvDataAnalysisPolicy(data) ((data) -> analysis_policy)
#if defined(NALU_HYPRE_USING_CUSPARSE) && (CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION)
#define nalu_hypre_CsrsvDataBufferSizeL(data)    ((data) -> buffer_size_L)
#define nalu_hypre_CsrsvDataBufferSizeU(data)    ((data) -> buffer_size_U)
#define nalu_hypre_CsrsvDataBufferL(data)        ((data) -> buffer_L)
#define nalu_hypre_CsrsvDataBufferU(data)        ((data) -> buffer_U)
#else
#define nalu_hypre_CsrsvDataBufferSize(data)     ((data) -> buffer_size)
#define nalu_hypre_CsrsvDataBuffer(data)         ((data) -> buffer)
#endif
#define nalu_hypre_CsrsvDataMatData(data)        ((data) -> mat_data)

struct nalu_hypre_GpuMatData
{
#if defined(NALU_HYPRE_USING_CUSPARSE)
   cusparseMatDescr_t                   mat_descr;
   char                                *spmv_buffer;

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   rocsparse_mat_descr                  mat_descr;
   rocsparse_mat_info                   mat_info;

#elif defined(NALU_HYPRE_USING_ONEMKLSPARSE)
   oneapi::mkl::sparse::matrix_handle_t mat_handle;
#endif
};

#define nalu_hypre_GpuMatDataMatDescr(data)    ((data) -> mat_descr)
#define nalu_hypre_GpuMatDataMatInfo(data)     ((data) -> mat_info)
#define nalu_hypre_GpuMatDataMatHandle(data)   ((data) -> mat_handle)
#define nalu_hypre_GpuMatDataSpMVBuffer(data)  ((data) -> spmv_buffer)

#endif /* if defined(NALU_HYPRE_USING_GPU) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *      generic device functions (cuda/hip/sycl)
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_GPU)
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

#endif // defined(NALU_HYPRE_USING_GPU)

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
nalu_hypre_int nalu_hypre_gpu_get_num_blocks()
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
nalu_hypre_int nalu_hypre_gpu_get_block_id(nalu_hypre_DeviceItem &item)
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
   return nalu_hypre_gpu_get_num_blocks<gdim>() * nalu_hypre_gpu_get_num_threads<bdim>(item);
}

/* return the flattened thread id in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_thread_id(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_block_id<gdim>(item) * nalu_hypre_gpu_get_num_threads<bdim>(item) +
          nalu_hypre_gpu_get_thread_id<bdim>(item);
}

/* return the number of warps in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_num_warps(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_num_blocks<gdim>() * nalu_hypre_gpu_get_num_warps<bdim>(item);
}

/* return the flattened warp id in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_warp_id(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_block_id<gdim>(item) * nalu_hypre_gpu_get_num_warps<bdim>(item) +
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
T __shfl_sync(nalu_hypre_mask mask, T val, nalu_hypre_int src_line, nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl(val, src_line, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_up_sync(nalu_hypre_mask mask, T val, nalu_hypre_uint delta, nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_up(val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_down_sync(nalu_hypre_mask mask, T val, nalu_hypre_uint delta, nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_down(val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T __shfl_xor_sync(nalu_hypre_mask mask, T val, nalu_hypre_int lanemask, nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_xor(val, lanemask, width);
}

static __device__ __forceinline__
void __syncwarp()
{
}

#endif // #if defined(NALU_HYPRE_USING_HIP) || (CUDA_VERSION < 9000)

static __device__ __forceinline__
nalu_hypre_mask nalu_hypre_ballot_sync(nalu_hypre_mask mask, nalu_hypre_int predicate)
{
#if defined(NALU_HYPRE_USING_CUDA)
   return __ballot_sync(mask, predicate);
#else
   return __ballot(predicate);
#endif
}

static __device__ __forceinline__
NALU_HYPRE_Int nalu_hypre_popc(nalu_hypre_mask mask)
{
#if defined(NALU_HYPRE_USING_CUDA)
   return (NALU_HYPRE_Int) __popc(mask);
#else
   return (NALU_HYPRE_Int) __popcll(mask);
#endif
}

static __device__ __forceinline__
NALU_HYPRE_Int nalu_hypre_ffs(nalu_hypre_mask mask)
{
#if defined(NALU_HYPRE_USING_CUDA)
   return (NALU_HYPRE_Int) __ffs(mask);
#else
   return (NALU_HYPRE_Int) __ffsll(mask);
#endif
}

/* Flip n-th bit of bitmask (0 becomes 1. 1 becomes 0) */
static __device__ __forceinline__
nalu_hypre_mask nalu_hypre_mask_flip_at(nalu_hypre_mask bitmask, nalu_hypre_int n)
{
   return bitmask ^ (nalu_hypre_mask_one << n);
}

#if defined(NALU_HYPRE_USING_HIP)
static __device__ __forceinline__
nalu_hypre_int __any_sync(unsigned mask, nalu_hypre_int predicate)
{
   return __any(predicate);
}
#endif

/* sync the thread block */
static __device__ __forceinline__
void block_sync(nalu_hypre_DeviceItem &item)
{
   __syncthreads();
}

/* sync the warp */
static __device__ __forceinline__
void warp_sync(nalu_hypre_DeviceItem &item)
{
   __syncwarp();
}

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
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
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
nalu_hypre_int warp_any_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, nalu_hypre_int predicate)
{
   return __any_sync(mask, predicate);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_int src_line,
                    nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_sync(mask, val, src_line, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_up_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_uint delta,
                       nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_up_sync(mask, val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_down_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_uint delta,
                         nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_down_sync(mask, val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_xor_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_int lane_mask,
                        nalu_hypre_int width = NALU_HYPRE_WARP_SIZE)
{
   return __shfl_xor_sync(mask, val, lane_mask, width);
}

template <typename T>
static __device__ __forceinline__
T warp_reduce_sum(nalu_hypre_DeviceItem &item, T in)
{
#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
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
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
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
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
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
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
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
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
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
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
   {
      in = min(in, __shfl_xor_sync(NALU_HYPRE_WARP_FULL_MASK, in, d));
   }
   return in;
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
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_num_threads(nalu_hypre_DeviceItem &item)
{
   return static_cast<nalu_hypre_int>(item.get_local_range().size());
}

/* return the flattened thread id in block */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_thread_id(nalu_hypre_DeviceItem &item)
{
   return static_cast<nalu_hypre_int>(item.get_local_linear_id());
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
   return static_cast<nalu_hypre_int>(item.get_sub_group().get_local_linear_id());
}

/* return the num of blocks in grid */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_num_blocks(nalu_hypre_DeviceItem &item)
{
   return static_cast<nalu_hypre_int>(item.get_group_range().size());
}

/* return the flattened block id in grid */
template <nalu_hypre_int dim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_block_id(nalu_hypre_DeviceItem &item)
{
   return item.get_group_linear_id();
}

/* return the number of threads in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_num_threads(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_num_blocks<gdim>(item) * nalu_hypre_gpu_get_num_threads<bdim>(item);
}

/* return the flattened thread id in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_thread_id(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_block_id<gdim>(item) * nalu_hypre_gpu_get_num_threads<bdim>(item) +
          nalu_hypre_gpu_get_thread_id<bdim>(item);
}

/* return the number of warps in grid */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_num_warps(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_num_blocks<gdim>(item) * nalu_hypre_gpu_get_num_warps<bdim>(item);
}

/* return the flattened warp id in nd_range */
template <nalu_hypre_int bdim, nalu_hypre_int gdim>
static __device__ __forceinline__
nalu_hypre_int nalu_hypre_gpu_get_grid_warp_id(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_block_id<gdim>(item) * nalu_hypre_gpu_get_num_warps<bdim>(item) +
          nalu_hypre_gpu_get_warp_id<bdim>(item);
}

/* sync the thread block */
static __device__ __forceinline__
void block_sync(nalu_hypre_DeviceItem &item)
{
   item.barrier();
}

/* sync the warp */
static __device__ __forceinline__
void warp_sync(nalu_hypre_DeviceItem &item)
{
   item.get_sub_group().barrier();
}

/* exclusive prefix scan */
template <typename T>
static __device__ __forceinline__
T warp_prefix_sum(nalu_hypre_DeviceItem &item, nalu_hypre_int lane_id, T in, T &all_sum)
{
#pragma unroll
   for (nalu_hypre_int d = 2; d <= NALU_HYPRE_WARP_SIZE; d <<= 1)
   {
      T t = sycl::shift_group_right(item.get_sub_group(), in, d >> 1);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = sycl::group_broadcast(item.get_sub_group(), in, NALU_HYPRE_WARP_SIZE - 1);

   if (lane_id == NALU_HYPRE_WARP_SIZE - 1)
   {
      in = 0;
   }

#pragma unroll
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
   {
      T t = sycl::permute_group_by_xor(item.get_sub_group(), in, d);

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
nalu_hypre_int warp_any_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, nalu_hypre_int predicate)
{
   return sycl::any_of_group(item.get_sub_group(), predicate);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_int src_line)
{
   /* WM: todo - I'm still getting bad results if I try to remove this barrier. Needs investigation. */
   item.get_sub_group().barrier();
   return sycl::group_broadcast(item.get_sub_group(), val, src_line);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_int src_line,
                    nalu_hypre_int width)
{
   nalu_hypre_int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   nalu_hypre_int group_start = (lane_id / width) * width;
   nalu_hypre_int src_in_warp = group_start + (src_line % width);
   return sycl::select_from_group(item.get_sub_group(), val, src_in_warp);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_up_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_uint delta)
{
   return sycl::shift_group_right(item.get_sub_group(), val, delta);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_up_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_uint delta,
                       nalu_hypre_int width)
{
   nalu_hypre_int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   nalu_hypre_int group_start = (lane_id / width) * width;
   nalu_hypre_int src_in_warp = lane_id - delta >= group_start ? lane_id - delta : lane_id;
   return sycl::select_from_group(item.get_sub_group(), val, src_in_warp);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_down_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_uint delta)
{
   return sycl::shift_group_left(item.get_sub_group(), val, delta);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_down_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_uint delta,
                         nalu_hypre_int width)
{
   nalu_hypre_int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   nalu_hypre_int group_end = ((lane_id / width) + 1) * width - 1;
   nalu_hypre_int src_in_warp = lane_id + delta <= group_end ? lane_id + delta : lane_id;
   return sycl::select_from_group(item.get_sub_group(), val, src_in_warp);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_xor_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_int lane_mask)
{
   return sycl::permute_group_by_xor(item.get_sub_group(), val, lane_mask);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_xor_sync(nalu_hypre_DeviceItem &item, nalu_hypre_mask mask, T val, nalu_hypre_int lane_mask,
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
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
   {
      in += sycl::shift_group_left(item.get_sub_group(), in, d);
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_allreduce_sum(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
   {
      in += sycl::permute_group_by_xor(item.get_sub_group(), in, d);
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_reduce_max(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
   {
      in = std::max(in, sycl::shift_group_left(item.get_sub_group(), in, d));
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_allreduce_max(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
   {
      in = std::max(in, sycl::permute_group_by_xor(item.get_sub_group(), in, d));
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_reduce_min(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
   {
      in = std::min(in, sycl::shift_group_left(item.get_sub_group(), in, d));
   }
   return in;
}

template <typename T>
static __forceinline__
T warp_allreduce_min(nalu_hypre_DeviceItem &item, T in)
{
   for (nalu_hypre_int d = NALU_HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
   {
      in = std::min(in, sycl::permute_group_by_xor(item.get_sub_group(), in, d));
   }
   return in;
}

template<typename T>
struct is_negative
{
   is_negative() {}

   constexpr bool operator()(const T &x = T()) const { return (x < 0); }
};

template<typename T>
struct is_positive
{
   is_positive() {}

   constexpr bool operator()(const T &x = T()) const { return (x > 0); }
};

template<typename T>
struct is_nonnegative
{
   is_nonnegative() {}

   constexpr bool operator()(const T &x = T()) const { return (x >= 0); }
};

template<typename T>
struct in_range
{
   T low, high;
   in_range(T low_ = T(), T high_ = T()) { low = low_; high = high_; }

   constexpr bool operator()(const T &x) const { return (x >= low && x <= high); }
};

template<typename T>
struct out_of_range
{
   T low, high;
   out_of_range(T low_ = T(), T high_ = T()) { low = low_; high = high_; }

   constexpr bool operator()(const T &x) const { return (x < low || x > high); }
};

template<typename T>
struct less_than
{
   T val;
   less_than(T val_ = T()) { val = val_; }

   constexpr bool operator()(const T &x) const { return (x < val); }
};

template<typename T>
struct modulo
{
   T val;
   modulo(T val_ = T()) { val = val_; }

   constexpr T operator()(const T &x) const { return (x % val); }
};

template<typename T>
struct equal
{
   T val;
   equal(T val_ = T()) { val = val_; }

   constexpr bool operator()(const T &x) const { return (x == val); }
};

template<typename T1, typename T2>
struct type_cast
{
   constexpr T2 operator()(const T1 &x = T1()) const { return (T2) x; }
};

template<typename T>
struct absolute_value
{
   constexpr T operator()(const T &x) const { return x < T(0) ? -x : x; }
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
#if defined(NALU_HYPRE_USING_GPU)
dim3 nalu_hypre_GetDefaultDeviceBlockDimension();

dim3 nalu_hypre_GetDefaultDeviceGridDimension( NALU_HYPRE_Int n, const char *granularity, dim3 bDim );

dim3 nalu_hypre_dim3(NALU_HYPRE_Int x);
dim3 nalu_hypre_dim3(NALU_HYPRE_Int x, NALU_HYPRE_Int y);
dim3 nalu_hypre_dim3(NALU_HYPRE_Int x, NALU_HYPRE_Int y, NALU_HYPRE_Int z);

template <typename T1, typename T2, typename T3>
NALU_HYPRE_Int hypreDevice_StableSortByTupleKey(NALU_HYPRE_Int N, T1 *keys1, T2 *keys2,
                                           T3 *vals, NALU_HYPRE_Int opt);

template <typename T1, typename T2, typename T3, typename T4>
NALU_HYPRE_Int hypreDevice_StableSortTupleByTupleKey(NALU_HYPRE_Int N, T1 *keys1, T2 *keys2,
                                                T3 *vals1, T4 *vals2, NALU_HYPRE_Int opt);

template <typename T1, typename T2, typename T3>
NALU_HYPRE_Int hypreDevice_ReduceByTupleKey(NALU_HYPRE_Int N, T1 *keys1_in, T2 *keys2_in,
                                       T3 *vals_in, T1 *keys1_out, T2 *keys2_out,
                                       T3 *vals_out);

template <typename T>
NALU_HYPRE_Int hypreDevice_ScatterConstant(T *x, NALU_HYPRE_Int n, NALU_HYPRE_Int *map, T v);

NALU_HYPRE_Int hypreDevice_GenScatterAdd(NALU_HYPRE_Real *x, NALU_HYPRE_Int ny, NALU_HYPRE_Int *map,
                                    NALU_HYPRE_Real *y, char *work);

template <typename T>
NALU_HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum(NALU_HYPRE_Int nrows, NALU_HYPRE_Int nnz,
                                                    NALU_HYPRE_Int *d_row_ptr, T *d_row_num, T *d_row_ind);

#endif

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

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

#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
cusparseIndexType_t nalu_hypre_HYPREIntToCusparseIndexType();
#endif

#endif // #if defined(NALU_HYPRE_USING_CUSPARSE)

#endif /* #ifndef NALU_HYPRE_CUDA_UTILS_H */
