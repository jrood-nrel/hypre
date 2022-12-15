/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef CSR_SPGEMM_DEVICE_H
#define CSR_SPGEMM_DEVICE_H

#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

#define COHEN_USE_SHMEM 0

static const char NALU_HYPRE_SPGEMM_HASH_TYPE = 'D';

/* bin settings                             0   1   2    3    4    5     6     7     8     9     10 */
constexpr NALU_HYPRE_Int SYMBL_HASH_SIZE[11] = { 0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
#if defined(NALU_HYPRE_USING_CUDA)
constexpr NALU_HYPRE_Int NUMER_HASH_SIZE[11] = { 0, 16, 32,  64, 128, 256,  512, 1024, 2048, 4096,  8192 };
#elif defined(NALU_HYPRE_USING_HIP)
constexpr NALU_HYPRE_Int NUMER_HASH_SIZE[11] = { 0,  8, 16,  32,  64, 128,  256,  512, 1024, 2048,  4096 };
#endif
constexpr NALU_HYPRE_Int T_GROUP_SIZE[11]    = { 0,  2,  4,   8,  16,  32,   64,  128,  256,  512,  1024 };

#if defined(NALU_HYPRE_USING_CUDA)
#define NALU_HYPRE_SPGEMM_DEFAULT_BIN 5
#elif defined(NALU_HYPRE_USING_HIP)
#define NALU_HYPRE_SPGEMM_DEFAULT_BIN 6
#endif

/* unroll factor in the kernels */
#if defined(NALU_HYPRE_USING_CUDA)
#define NALU_HYPRE_SPGEMM_NUMER_UNROLL 256
#define NALU_HYPRE_SPGEMM_SYMBL_UNROLL 512
#elif defined(NALU_HYPRE_USING_HIP)
#define NALU_HYPRE_SPGEMM_NUMER_UNROLL 256
#define NALU_HYPRE_SPGEMM_SYMBL_UNROLL 512
#endif

//#define NALU_HYPRE_SPGEMM_TIMING
//#define NALU_HYPRE_SPGEMM_PRINTF
//#define NALU_HYPRE_SPGEMM_NVTX

/* ----------------------------------------------------------------------------------------------- *
 * these are under the assumptions made in spgemm on block sizes: only use in csr_spgemm routines
 * where we assume CUDA block is 3D and blockDim.x * blockDim.y = GROUP_SIZE which is a multiple
 * of NALU_HYPRE_WARP_SIZE
 *------------------------------------------------------------------------------------------------ */

/* the number of groups in block */
static __device__ __forceinline__
nalu_hypre_int get_num_groups()
{
   return blockDim.z;
}

/* the group id in the block */
static __device__ __forceinline__
nalu_hypre_int get_group_id()
{
   return threadIdx.z;
}

/* the thread id (lane) in the group */
static __device__ __forceinline__
nalu_hypre_int get_group_lane_id(nalu_hypre_DeviceItem &item)
{
   return nalu_hypre_gpu_get_thread_id<2>(item);
}

/* the warp id in the group */
template <NALU_HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
nalu_hypre_int get_warp_in_group_id(nalu_hypre_DeviceItem &item)
{
   if (GROUP_SIZE <= NALU_HYPRE_WARP_SIZE)
   {
      return 0;
   }
   else
   {
      return nalu_hypre_gpu_get_warp_id<2>(item);
   }
}

/* group reads 2 values from ptr to v1 and v2
 * GROUP_SIZE must be >= 2
 */
template <NALU_HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_read(nalu_hypre_DeviceItem &item, const NALU_HYPRE_Int *ptr, bool valid_ptr, NALU_HYPRE_Int &v1,
                NALU_HYPRE_Int &v2)
{
   if (GROUP_SIZE >= NALU_HYPRE_WARP_SIZE)
   {
      /* lane = warp_lane
       * Note: use "2" since assume NALU_HYPRE_WARP_SIZE divides (blockDim.x * blockDim.y) */
      const NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<2>(item);

      if (lane < 2)
      {
         v1 = read_only_load(ptr + lane);
      }
      v2 = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, v1, 1);
      v1 = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, v1, 0);
   }
   else
   {
      /* lane = group_lane */
      const NALU_HYPRE_Int lane = get_group_lane_id(item);

      if (valid_ptr && lane < 2)
      {
         v1 = read_only_load(ptr + lane);
      }
      v2 = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, v1, 1, GROUP_SIZE);
      v1 = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, v1, 0, GROUP_SIZE);
   }
}

/* group reads a value from ptr to v1
 * GROUP_SIZE must be >= 2
 */
template <NALU_HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_read(nalu_hypre_DeviceItem &item, const NALU_HYPRE_Int *ptr, bool valid_ptr, NALU_HYPRE_Int &v1)
{
   if (GROUP_SIZE >= NALU_HYPRE_WARP_SIZE)
   {
      /* lane = warp_lane
       * Note: use "2" since assume NALU_HYPRE_WARP_SIZE divides (blockDim.x * blockDim.y) */
      const NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<2>(item);

      if (!lane)
      {
         v1 = read_only_load(ptr);
      }
      v1 = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, v1, 0);
   }
   else
   {
      /* lane = group_lane */
      const NALU_HYPRE_Int lane = get_group_lane_id(item);

      if (valid_ptr && !lane)
      {
         v1 = read_only_load(ptr);
      }
      v1 = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, v1, 0, GROUP_SIZE);
   }
}

template <typename T, NALU_HYPRE_Int NUM_GROUPS_PER_BLOCK, NALU_HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_reduce_sum(nalu_hypre_DeviceItem &item, T in)
{
#if defined(NALU_HYPRE_DEBUG)
   nalu_hypre_device_assert(GROUP_SIZE <= NALU_HYPRE_WARP_SIZE);
#endif

#pragma unroll
   for (nalu_hypre_int d = GROUP_SIZE / 2; d > 0; d >>= 1)
   {
      in += __shfl_down_sync(NALU_HYPRE_WARP_FULL_MASK, in, d);
   }

   return in;
}

/* s_WarpData[NUM_GROUPS_PER_BLOCK * GROUP_SIZE / NALU_HYPRE_WARP_SIZE] */
template <typename T, NALU_HYPRE_Int NUM_GROUPS_PER_BLOCK, NALU_HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_reduce_sum(nalu_hypre_DeviceItem &item, T in, volatile T *s_WarpData)
{
#if defined(NALU_HYPRE_DEBUG)
   nalu_hypre_device_assert(GROUP_SIZE > NALU_HYPRE_WARP_SIZE);
#endif

   T out = warp_reduce_sum(item, in);

   const NALU_HYPRE_Int warp_lane_id = nalu_hypre_gpu_get_lane_id<2>(item);
   const NALU_HYPRE_Int warp_id = nalu_hypre_gpu_get_warp_id<3>(item);

   if (warp_lane_id == 0)
   {
      s_WarpData[warp_id] = out;
   }

   __syncthreads();

   if (get_warp_in_group_id<GROUP_SIZE>(item) == 0)
   {
      const T a = warp_lane_id < GROUP_SIZE / NALU_HYPRE_WARP_SIZE ? s_WarpData[warp_id + warp_lane_id] : 0.0;
      out = warp_reduce_sum(item, a);
   }

   __syncthreads();

   return out;
}

/* GROUP_SIZE must <= NALU_HYPRE_WARP_SIZE */
template <typename T, NALU_HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_prefix_sum(nalu_hypre_int lane_id, T in, T &all_sum)
{
#pragma unroll
   for (nalu_hypre_int d = 2; d <= GROUP_SIZE; d <<= 1)
   {
      T t = __shfl_up_sync(NALU_HYPRE_WARP_FULL_MASK, in, d >> 1, GROUP_SIZE);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, in, GROUP_SIZE - 1, GROUP_SIZE);

   if (lane_id == GROUP_SIZE - 1)
   {
      in = 0;
   }

#pragma unroll
   for (nalu_hypre_int d = GROUP_SIZE >> 1; d > 0; d >>= 1)
   {
      T t = __shfl_xor_sync(NALU_HYPRE_WARP_FULL_MASK, in, d, GROUP_SIZE);

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

template <NALU_HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_sync()
{
   if (GROUP_SIZE <= NALU_HYPRE_WARP_SIZE)
   {
      __syncwarp();
   }
   else
   {
      __syncthreads();
   }
}

/* Hash functions */
static __device__ __forceinline__
NALU_HYPRE_Int Hash2Func(NALU_HYPRE_Int key)
{
   //return ( (key << 1) | 1 );
   //TODO: 6 --> should depend on hash1 size
   return ( (key >> 6) | 1 );
}

template <char HASHTYPE>
static __device__ __forceinline__
NALU_HYPRE_Int HashFunc(NALU_HYPRE_Int m, NALU_HYPRE_Int key, NALU_HYPRE_Int i, NALU_HYPRE_Int prev)
{
   NALU_HYPRE_Int hashval = 0;

   /* assume m is power of 2 */
   if (HASHTYPE == 'L')
   {
      //hashval = (key + i) % m;
      hashval = ( prev + 1 ) & (m - 1);
   }
   else if (HASHTYPE == 'Q')
   {
      //hashval = (key + (i + i*i)/2) & (m-1);
      hashval = ( prev + i ) & (m - 1);
   }
   else if (HASHTYPE == 'D')
   {
      //hashval = (key + i*Hash2Func(key) ) & (m - 1);
      hashval = ( prev + Hash2Func(key) ) & (m - 1);
   }

   return hashval;
}

template <NALU_HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE>
static __device__ __forceinline__
NALU_HYPRE_Int HashFunc(NALU_HYPRE_Int key, NALU_HYPRE_Int i, NALU_HYPRE_Int prev)
{
   NALU_HYPRE_Int hashval = 0;

   /* assume m is power of 2 */
   if (HASHTYPE == 'L')
   {
      //hashval = (key + i) % SHMEM_HASH_SIZE;
      hashval = ( prev + 1 ) & (SHMEM_HASH_SIZE - 1);
   }
   else if (HASHTYPE == 'Q')
   {
      //hashval = (key + (i + i*i)/2) & (SHMEM_HASH_SIZE-1);
      hashval = ( prev + i ) & (SHMEM_HASH_SIZE - 1);
   }
   else if (HASHTYPE == 'D')
   {
      //hashval = (key + i*Hash2Func(key) ) & (SHMEM_HASH_SIZE - 1);
      hashval = ( prev + Hash2Func(key) ) & (SHMEM_HASH_SIZE - 1);
   }

   return hashval;
}

template<typename T>
struct spgemm_bin_op : public thrust::unary_function<T, char>
{
   char s, t, u; /* s: base size of bins; t: lowest bin; u: highest bin */

   spgemm_bin_op(char s_, char t_, char u_) { s = s_; t = t_; u = u_; }

   __device__ char operator()(const T &x)
   {
      if (x <= 0)
      {
         return 0;
      }

      const T y = (x + s - 1) / s;

      if ( y <= (1 << (t - 1)) )
      {
         return t;
      }

      for (char i = t; i < u - 1; i++)
      {
         if (y <= (1 << i))
         {
            return i + 1;
         }
      }

      return u;
   }
};

void nalu_hypre_create_ija(NALU_HYPRE_Int m, NALU_HYPRE_Int *row_id, NALU_HYPRE_Int *d_c, NALU_HYPRE_Int *d_i,
                      NALU_HYPRE_Int **d_j, NALU_HYPRE_Complex **d_a, NALU_HYPRE_Int *nnz_ptr );

void nalu_hypre_create_ija(NALU_HYPRE_Int SHMEM_HASH_SIZE, NALU_HYPRE_Int m, NALU_HYPRE_Int *row_id, NALU_HYPRE_Int *d_c,
                      NALU_HYPRE_Int *d_i, NALU_HYPRE_Int **d_j, NALU_HYPRE_Complex **d_a, NALU_HYPRE_Int *nnz_ptr );

NALU_HYPRE_Int nalu_hypre_SpGemmCreateGlobalHashTable( NALU_HYPRE_Int num_rows, NALU_HYPRE_Int *row_id,
                                             NALU_HYPRE_Int num_ghash, NALU_HYPRE_Int *row_sizes, NALU_HYPRE_Int SHMEM_HASH_SIZE, NALU_HYPRE_Int **ghash_i_ptr,
                                             NALU_HYPRE_Int **ghash_j_ptr, NALU_HYPRE_Complex **ghash_a_ptr, NALU_HYPRE_Int *ghash_size_ptr);

NALU_HYPRE_Int hypreDevice_CSRSpGemmRownnzEstimate(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                              NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Int *d_rc,
                                              NALU_HYPRE_Int row_est_mtd);

NALU_HYPRE_Int hypreDevice_CSRSpGemmRownnzUpperbound(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                                NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Int in_rc,
                                                NALU_HYPRE_Int *d_rc, NALU_HYPRE_Int *rownnz_exact_ptr);

NALU_HYPRE_Int hypreDevice_CSRSpGemmRownnz(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n, NALU_HYPRE_Int nnzA,
                                      NALU_HYPRE_Int *d_ia,
                                      NALU_HYPRE_Int *d_ja, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Int in_rc, NALU_HYPRE_Int *d_rc);

NALU_HYPRE_Int hypreDevice_CSRSpGemmNumerWithRownnzUpperbound(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                                         NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_a, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb,
                                                         NALU_HYPRE_Complex *d_b, NALU_HYPRE_Int *d_rc, NALU_HYPRE_Int exact_rownnz, NALU_HYPRE_Int **d_ic_out,
                                                         NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_c_out, NALU_HYPRE_Int *nnzC);

NALU_HYPRE_Int nalu_hypre_SpGemmCreateBins( NALU_HYPRE_Int m, char s, char t, char u, NALU_HYPRE_Int *d_rc,
                                  bool d_rc_indice_in, NALU_HYPRE_Int *d_rc_indice, NALU_HYPRE_Int *h_bin_ptr );

template <NALU_HYPRE_Int BIN, NALU_HYPRE_Int SHMEM_HASH_SIZE, NALU_HYPRE_Int GROUP_SIZE, bool HAS_RIND>
NALU_HYPRE_Int nalu_hypre_spgemm_symbolic_rownnz( NALU_HYPRE_Int m, NALU_HYPRE_Int *row_ind, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                        bool need_ghash, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb,
                                        NALU_HYPRE_Int *d_rc, bool can_fail, char *d_rf );

template <NALU_HYPRE_Int BIN, NALU_HYPRE_Int SHMEM_HASH_SIZE, NALU_HYPRE_Int GROUP_SIZE, bool HAS_RIND>
NALU_HYPRE_Int nalu_hypre_spgemm_numerical_with_rownnz( NALU_HYPRE_Int m, NALU_HYPRE_Int *row_ind, NALU_HYPRE_Int k,
                                              NALU_HYPRE_Int n, bool need_ghash,
                                              NALU_HYPRE_Int exact_rownnz, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_a,
                                              NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Complex *d_b, NALU_HYPRE_Int *d_rc, NALU_HYPRE_Int *d_ic,
                                              NALU_HYPRE_Int *d_jc, NALU_HYPRE_Complex *d_c );

template <NALU_HYPRE_Int SHMEM_HASH_SIZE, NALU_HYPRE_Int GROUP_SIZE>
NALU_HYPRE_Int nalu_hypre_spgemm_symbolic_max_num_blocks( NALU_HYPRE_Int multiProcessorCount,
                                                NALU_HYPRE_Int *num_blocks_ptr, NALU_HYPRE_Int *block_size_ptr );

template <NALU_HYPRE_Int SHMEM_HASH_SIZE, NALU_HYPRE_Int GROUP_SIZE>
NALU_HYPRE_Int nalu_hypre_spgemm_numerical_max_num_blocks( NALU_HYPRE_Int multiProcessorCount,
                                                 NALU_HYPRE_Int *num_blocks_ptr, NALU_HYPRE_Int *block_size_ptr );

NALU_HYPRE_Int hypreDevice_CSRSpGemmBinnedGetBlockNumDim();

template <NALU_HYPRE_Int GROUP_SIZE>
NALU_HYPRE_Int hypreDevice_CSRSpGemmNumerPostCopy( NALU_HYPRE_Int m, NALU_HYPRE_Int *d_rc, NALU_HYPRE_Int *nnzC,
                                              NALU_HYPRE_Int **d_ic, NALU_HYPRE_Int **d_jc, NALU_HYPRE_Complex **d_c);

template <NALU_HYPRE_Int GROUP_SIZE>
static constexpr NALU_HYPRE_Int
nalu_hypre_spgemm_get_num_groups_per_block()
{
#if defined(NALU_HYPRE_USING_CUDA)
   return nalu_hypre_min(nalu_hypre_max(512 / GROUP_SIZE, 1), 64);
#elif defined(NALU_HYPRE_USING_HIP)
   return nalu_hypre_max(512 / GROUP_SIZE, 1);
#endif
}

#if defined(NALU_HYPRE_SPGEMM_PRINTF) || defined(NALU_HYPRE_SPGEMM_TIMING)
#define NALU_HYPRE_SPGEMM_PRINT(...) nalu_hypre_ParPrintf(nalu_hypre_MPI_COMM_WORLD, __VA_ARGS__)
#else
#define NALU_HYPRE_SPGEMM_PRINT(...)
#endif

#endif /* NALU_HYPRE_USING_CUDA || defined(NALU_HYPRE_USING_HIP) */
#endif /* #ifndef CSR_SPGEMM_DEVICE_H */

