/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* CUDA reducer class */

#ifndef NALU_HYPRE_CUDA_REDUCER_H
#define NALU_HYPRE_CUDA_REDUCER_H

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#if !defined(NALU_HYPRE_USING_RAJA) && !defined(NALU_HYPRE_USING_KOKKOS)

template<typename T> void OneBlockReduce(T *d_arr, NALU_HYPRE_Int N, T *h_out);

struct NALU_HYPRE_double4
{
   NALU_HYPRE_Real x, y, z, w;

   __host__ __device__
   NALU_HYPRE_double4() {}

   __host__ __device__
   NALU_HYPRE_double4(NALU_HYPRE_Real x1, NALU_HYPRE_Real x2, NALU_HYPRE_Real x3, NALU_HYPRE_Real x4)
   {
      x = x1;
      y = x2;
      z = x3;
      w = x4;
   }

   __host__ __device__
   void operator=(NALU_HYPRE_Real val)
   {
      x = y = z = w = val;
   }

   __host__ __device__
   void operator+=(NALU_HYPRE_double4 rhs)
   {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      w += rhs.w;
   }

};

struct NALU_HYPRE_double6
{
   NALU_HYPRE_Real x, y, z, w, u, v;

   __host__ __device__
   NALU_HYPRE_double6() {}

   __host__ __device__
   NALU_HYPRE_double6(NALU_HYPRE_Real x1, NALU_HYPRE_Real x2, NALU_HYPRE_Real x3, NALU_HYPRE_Real x4,
                 NALU_HYPRE_Real x5, NALU_HYPRE_Real x6)
   {
      x = x1;
      y = x2;
      z = x3;
      w = x4;
      u = x5;
      v = x6;
   }

   __host__ __device__
   void operator=(NALU_HYPRE_Real val)
   {
      x = y = z = w = u = v = val;
   }

   __host__ __device__
   void operator+=(NALU_HYPRE_double6 rhs)
   {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      w += rhs.w;
      u += rhs.u;
      v += rhs.v;
   }

};

/* reduction within a warp */
__inline__ __host__ __device__
NALU_HYPRE_Real warpReduceSum(NALU_HYPRE_Real val)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   for (NALU_HYPRE_Int offset = warpSize / 2; offset > 0; offset /= 2)
   {
      val += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val, offset);
   }
#endif
   return val;
}

__inline__ __host__ __device__
NALU_HYPRE_double4 warpReduceSum(NALU_HYPRE_double4 val)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   for (NALU_HYPRE_Int offset = warpSize / 2; offset > 0; offset /= 2)
   {
      val.x += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.x, offset);
      val.y += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.y, offset);
      val.z += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.z, offset);
      val.w += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.w, offset);
   }
#endif
   return val;
}

__inline__ __host__ __device__
NALU_HYPRE_double6 warpReduceSum(NALU_HYPRE_double6 val)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   for (NALU_HYPRE_Int offset = warpSize / 2; offset > 0; offset /= 2)
   {
      val.x += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.x, offset);
      val.y += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.y, offset);
      val.z += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.z, offset);
      val.w += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.w, offset);
      val.u += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.u, offset);
      val.v += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, val.v, offset);
   }
#endif
   return val;
}

/* reduction within a block */
template <typename T>
__inline__ __host__ __device__
T blockReduceSum(T val)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   // Shared mem for NALU_HYPRE_WARP_SIZE partial sums
   __shared__ T shared[NALU_HYPRE_WARP_SIZE];

   NALU_HYPRE_Int lane = threadIdx.x & (warpSize - 1);
   NALU_HYPRE_Int wid  = threadIdx.x >> NALU_HYPRE_WARP_BITSHIFT;

   // Each warp performs partial reduction
   val = warpReduceSum(val);

   // Write reduced value to shared memory
   if (lane == 0)
   {
      shared[wid] = val;
   }

   // Wait for all partial reductions
   __syncthreads();

   // read from shared memory only if that warp existed
   if (threadIdx.x < (blockDim.x >> NALU_HYPRE_WARP_BITSHIFT))
   {
      val = shared[lane];
   }
   else
   {
      val = 0.0;
   }

   // Final reduce within first warp
   if (wid == 0)
   {
      val = warpReduceSum(val);
   }

#endif
   return val;
}

template<typename T>
__global__ void
OneBlockReduceKernel(nalu_hypre_DeviceItem &item,
                     T                *arr,
                     NALU_HYPRE_Int         N)
{
   T sum;

   sum = 0.0;

   if (threadIdx.x < N)
   {
      sum = arr[threadIdx.x];
   }

   sum = blockReduceSum(sum);

   if (threadIdx.x == 0)
   {
      arr[0] = sum;
   }
}

/* Reducer class */
template <typename T>
struct ReduceSum
{
   using value_type = T;

   T init;                    /* initial value passed in */
   mutable T __thread_sum;    /* place to hold local sum of a thread,
                                 and partial sum of a block */
   T *d_buf;                  /* place to store partial sum within blocks
                                 in the 1st round, used in the 2nd round */
   NALU_HYPRE_Int nblocks;         /* number of blocks used in the 1st round */

   /* constructor
    * val is the initial value (added to the reduced sum) */
   __host__
   ReduceSum(T val)
   {
      init = val;
      __thread_sum = 0.0;
      nblocks = -1;
   }

   /* copy constructor */
   __host__ __device__
   ReduceSum(const ReduceSum<T>& other)
   {
      *this = other;
   }

   __host__ void
   Allocate2ndPhaseBuffer()
   {
      if (nalu_hypre_HandleReduceBuffer(nalu_hypre_handle()) == NULL)
      {
         /* allocate for the max size for reducing double6 type */
         nalu_hypre_HandleReduceBuffer(nalu_hypre_handle()) =
            nalu_hypre_TAlloc(NALU_HYPRE_double6, NALU_HYPRE_MAX_NTHREADS_BLOCK, NALU_HYPRE_MEMORY_DEVICE);
      }

      d_buf = (T*) nalu_hypre_HandleReduceBuffer(nalu_hypre_handle());
   }

   /* reduction within blocks */
   __host__ __device__
   void BlockReduce() const
   {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      __thread_sum = blockReduceSum(__thread_sum);
      if (threadIdx.x == 0)
      {
         d_buf[blockIdx.x] = __thread_sum;
      }
#endif
   }

   __host__ __device__
   void operator+=(T val) const
   {
      __thread_sum += val;
   }

   /* invoke the 2nd reduction at the time want the sum from the reducer */
   __host__
   operator T()
   {
      T val;

      const NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());
      const NALU_HYPRE_ExecutionPolicy exec_policy = nalu_hypre_GetExecPolicy1(memory_location);

      if (exec_policy == NALU_HYPRE_EXEC_HOST)
      {
         val = __thread_sum;
         val += init;
      }
      else
      {
         /* 2nd reduction with only *one* block */
         nalu_hypre_assert(nblocks >= 0 && nblocks <= NALU_HYPRE_MAX_NTHREADS_BLOCK);
         const dim3 gDim(1), bDim(NALU_HYPRE_MAX_NTHREADS_BLOCK);
         NALU_HYPRE_GPU_LAUNCH( OneBlockReduceKernel, gDim, bDim, d_buf, nblocks );
         nalu_hypre_TMemcpy(&val, d_buf, T, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
         val += init;
      }

      return val;
   }

   /* destructor */
   __host__ __device__
   ~ReduceSum<T>()
   {
   }
};

#endif /* #if !defined(NALU_HYPRE_USING_RAJA) && !defined(NALU_HYPRE_USING_KOKKOS) */
#endif /* #if defined(NALU_HYPRE_USING_CUDA)  || defined(NALU_HYPRE_USING_HIP) */
#endif /* #ifndef NALU_HYPRE_CUDA_REDUCER_H */
