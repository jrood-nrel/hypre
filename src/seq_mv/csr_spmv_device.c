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

#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) ||\
    defined(NALU_HYPRE_USING_HIP)  ||\
    defined(NALU_HYPRE_USING_SYCL)

#include "csr_spmv_device.h"

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatvecShuffleGT8
 *
 * Templated SpMV device kernel based of warp-shuffle reduction.
 * Uses groups of K threads per row.
 * Specialized function for num_vectors > 8
 *
 * Template parameters:
 *   1) K:  number of threads working on a single row. K = 2, 4, 8, 16, 32
 *   2) F:  fill-mode. See hypreDevice_CSRMatrixMatvec for supported values
 *   3) NV: number of vectors (> 1 for multi-component vectors)
 *   4) T:  data type of matrix/vector coefficients
 *--------------------------------------------------------------------------*/

template <NALU_HYPRE_Int F, NALU_HYPRE_Int K, NALU_HYPRE_Int NV, typename T>
__global__ void
hypreGPUKernel_CSRMatvecShuffleGT8(nalu_hypre_DeviceItem &item,
                                   NALU_HYPRE_Int         num_rows,
                                   NALU_HYPRE_Int         num_vectors,
                                   NALU_HYPRE_Int        *row_id,
                                   NALU_HYPRE_Int         idxstride_x,
                                   NALU_HYPRE_Int         idxstride_y,
                                   NALU_HYPRE_Int         vecstride_x,
                                   NALU_HYPRE_Int         vecstride_y,
                                   T                 alpha,
                                   NALU_HYPRE_Int        *d_ia,
                                   NALU_HYPRE_Int        *d_ja,
                                   T                *d_a,
                                   T                *d_x,
                                   T                 beta,
                                   T                *d_y )
{
#if defined (NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int        item_local_id = item.get_local_id(0);
   const NALU_HYPRE_Int  grid_ngroups  = item.get_group_range(0) * (NALU_HYPRE_SPMV_BLOCKDIM / K);
   NALU_HYPRE_Int        grid_group_id = (item.get_group(0) * NALU_HYPRE_SPMV_BLOCKDIM + item_local_id) / K;
   const NALU_HYPRE_Int  group_lane    = item_local_id & (K - 1);
#else
   const NALU_HYPRE_Int  grid_ngroups  = gridDim.x * (NALU_HYPRE_SPMV_BLOCKDIM / K);
   NALU_HYPRE_Int        grid_group_id = (blockIdx.x * NALU_HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
   const NALU_HYPRE_Int  group_lane    = threadIdx.x & (K - 1);
#endif
   T sum[64];

   for (; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, grid_group_id < num_rows);
        grid_group_id += grid_ngroups)
   {
      NALU_HYPRE_Int grid_row_id = -1, p = 0, q = 0;

      if (row_id)
      {
         if (grid_group_id < num_rows && group_lane == 0)
         {
            grid_row_id = read_only_load(&row_id[grid_group_id]);
         }
         grid_row_id = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, grid_row_id, 0, K);
      }
      else
      {
         grid_row_id = grid_group_id;
      }

      if (grid_group_id < num_rows && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_row_id + group_lane]);
      }
      q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1, K);
      p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0, K);

      for (NALU_HYPRE_Int i = 0; i < num_vectors; i++)
      {
         sum[i] = T(0.0);
      }

#pragma unroll 1
      for (p += group_lane; p < q; p += K * 2)
      {
         NALU_HYPRE_SPMV_ADD_SUM(p, num_vectors)
         if (p + K < q)
         {
            NALU_HYPRE_SPMV_ADD_SUM((p + K), num_vectors)
         }
      }

      // parallel reduction
      for (NALU_HYPRE_Int i = 0; i < num_vectors; i++)
      {
         for (NALU_HYPRE_Int d = K / 2; d > 0; d >>= 1)
         {
            sum[i] += warp_shuffle_down_sync(item, NALU_HYPRE_WARP_FULL_MASK, sum[i], d);
         }
      }

      if (grid_group_id < num_rows && group_lane == 0)
      {
         if (beta)
         {
            for (NALU_HYPRE_Int i = 0; i < num_vectors; i++)
            {
               d_y[grid_row_id * idxstride_y + i * vecstride_y] =
                  alpha * sum[i] +
                  beta * d_y[grid_row_id * idxstride_y + i * vecstride_y];
            }
         }
         else
         {
            for (NALU_HYPRE_Int i = 0; i < num_vectors; i++)
            {
               d_y[grid_row_id * idxstride_y + i * vecstride_y] = alpha * sum[i];
            }
         }
      }
   }
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatvecShuffle
 *
 * Templated SpMV device kernel based of warp-shuffle reduction.
 * Uses groups of K threads per row
 *
 * Template parameters:
 *   1) K:  number of threads working on a single row. K = 2, 4, 8, 16, 32
 *   2) F:  fill-mode. See hypreDevice_CSRMatrixMatvec for supported values
 *   3) NV: number of vectors (> 1 for multi-component vectors)
 *   4) T:  data type of matrix/vector coefficients
 *--------------------------------------------------------------------------*/

template <NALU_HYPRE_Int F, NALU_HYPRE_Int K, NALU_HYPRE_Int NV, typename T>
__global__ void
//__launch_bounds__(512, 1)
hypreGPUKernel_CSRMatvecShuffle(nalu_hypre_DeviceItem &item,
                                NALU_HYPRE_Int         num_rows,
                                NALU_HYPRE_Int         num_vectors,
                                NALU_HYPRE_Int        *row_id,
                                NALU_HYPRE_Int         idxstride_x,
                                NALU_HYPRE_Int         idxstride_y,
                                NALU_HYPRE_Int         vecstride_x,
                                NALU_HYPRE_Int         vecstride_y,
                                T                 alpha,
                                NALU_HYPRE_Int        *d_ia,
                                NALU_HYPRE_Int        *d_ja,
                                T                *d_a,
                                T                *d_x,
                                T                 beta,
                                T                *d_y )
{
#if defined (NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int        item_local_id = item.get_local_id(0);
   const NALU_HYPRE_Int  grid_ngroups  = item.get_group_range(0) * (NALU_HYPRE_SPMV_BLOCKDIM / K);
   NALU_HYPRE_Int        grid_group_id = (item.get_group(0) * NALU_HYPRE_SPMV_BLOCKDIM + item_local_id) / K;
   const NALU_HYPRE_Int  group_lane    = item_local_id & (K - 1);
#else
   const NALU_HYPRE_Int  grid_ngroups  = gridDim.x * (NALU_HYPRE_SPMV_BLOCKDIM / K);
   NALU_HYPRE_Int        grid_group_id = (blockIdx.x * NALU_HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
   const NALU_HYPRE_Int  group_lane    = threadIdx.x & (K - 1);
#endif

   for (; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, grid_group_id < num_rows);
        grid_group_id += grid_ngroups)
   {
      NALU_HYPRE_Int grid_row_id = -1, p = 0, q = 0;

      if (row_id)
      {
         if (grid_group_id < num_rows && group_lane == 0)
         {
            grid_row_id = read_only_load(&row_id[grid_group_id]);
         }
         grid_row_id = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, grid_row_id, 0, K);
      }
      else
      {
         grid_row_id = grid_group_id;
      }

      if (grid_group_id < num_rows && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_row_id + group_lane]);
      }
      q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1, K);
      p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0, K);

      T sum[NV] = {T(0)};
#if NALU_HYPRE_SPMV_VERSION == 1
#pragma unroll 1
      for (p += group_lane; p < q; p += K * 2)
      {
         NALU_HYPRE_SPMV_ADD_SUM(p, NV)
         if (p + K < q)
         {
            NALU_HYPRE_SPMV_ADD_SUM((p + K), NV)
         }
      }
#elif NALU_HYPRE_SPMV_VERSION == 2
#pragma unroll 1
      for (p += group_lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, p < q); p += K)
      {
         if (p < q)
         {
            NALU_HYPRE_SPMV_ADD_SUM(p, NV)
         }
      }
#else
#pragma unroll 1
      for (p += group_lane;  p < q; p += K)
      {
         NALU_HYPRE_SPMV_ADD_SUM(p, NV)
      }
#endif

      // parallel reduction
      for (NALU_HYPRE_Int i = 0; i < NV; i++)
      {
         for (NALU_HYPRE_Int d = K / 2; d > 0; d >>= 1)
         {
            sum[i] += warp_shuffle_down_sync(item, NALU_HYPRE_WARP_FULL_MASK, sum[i], d);
         }
      }

      if (grid_group_id < num_rows && group_lane == 0)
      {
         if (beta)
         {
            for (NALU_HYPRE_Int i = 0; i < NV; i++)
            {
               d_y[grid_row_id * idxstride_y + i * vecstride_y] =
                  alpha * sum[i] +
                  beta * d_y[grid_row_id * idxstride_y + i * vecstride_y];
            }
         }
         else
         {
            for (NALU_HYPRE_Int i = 0; i < NV; i++)
            {
               d_y[grid_row_id * idxstride_y + i * vecstride_y] = alpha * sum[i];
            }
         }
      }
   }
}

/*--------------------------------------------------------------------------
 * hypreDevice_CSRMatrixMatvec
 *
 * Templated host function for launching the device kernels for SpMV.
 *
 * The template parameter F is the fill-mode. Supported values:
 *    0: whole matrix
 *   -1: lower
 *    1: upper
 *   -2: strict lower
 *    2: strict upper
 * The template parameter T is the matrix/vector coefficient data type
 *--------------------------------------------------------------------------*/

template <NALU_HYPRE_Int F, typename T>
NALU_HYPRE_Int
hypreDevice_CSRMatrixMatvec( NALU_HYPRE_Int  num_vectors,
                             NALU_HYPRE_Int  num_rows,
                             NALU_HYPRE_Int *rowid,
                             NALU_HYPRE_Int  num_nonzeros,
                             NALU_HYPRE_Int  idxstride_x,
                             NALU_HYPRE_Int  idxstride_y,
                             NALU_HYPRE_Int  vecstride_x,
                             NALU_HYPRE_Int  vecstride_y,
                             T          alpha,
                             NALU_HYPRE_Int *d_ia,
                             NALU_HYPRE_Int *d_ja,
                             T         *d_a,
                             T         *d_x,
                             T          beta,
                             T         *d_y )
{
   if (num_vectors > 64)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "hypre's SpMV: (num_vectors > 64) not implemented");
      return nalu_hypre_error_flag;
   }

   const NALU_HYPRE_Int avg_rownnz = (num_nonzeros + num_rows - 1) / num_rows;

   static constexpr NALU_HYPRE_Int group_sizes[5] = {32, 16, 8, 4, 4};

   static constexpr NALU_HYPRE_Int unroll_depth[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

   static NALU_HYPRE_Int avg_rownnz_lower_bounds[5] = {64, 32, 16, 8, 0};

   static NALU_HYPRE_Int num_groups_per_block[5] = { NALU_HYPRE_SPMV_BLOCKDIM / group_sizes[0],
                                                NALU_HYPRE_SPMV_BLOCKDIM / group_sizes[1],
                                                NALU_HYPRE_SPMV_BLOCKDIM / group_sizes[2],
                                                NALU_HYPRE_SPMV_BLOCKDIM / group_sizes[3],
                                                NALU_HYPRE_SPMV_BLOCKDIM / group_sizes[4]
                                              };

   const dim3 bDim(NALU_HYPRE_SPMV_BLOCKDIM);

   /* Select execution path */
   switch (num_vectors)
   {
      case unroll_depth[1]:
         NALU_HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[1]);
         break;

      case unroll_depth[2]:
         NALU_HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[2]);
         break;

      case unroll_depth[3]:
         NALU_HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[3]);
         break;

      case unroll_depth[4]:
         NALU_HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[4]);
         break;

      case unroll_depth[5]:
         NALU_HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[5]);
         break;

      case unroll_depth[6]:
         NALU_HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[6]);
         break;

      case unroll_depth[7]:
         NALU_HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[7]);
         break;

      case unroll_depth[8]:
         NALU_HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[8]);
         break;

      default:
         NALU_HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffleGT8, unroll_depth[8]);
         break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSpMVDevice
 *
 * hypre's internal implementation of sparse matrix/vector multiplication
 * (SpMV) on GPUs.
 *
 * Computes:  y = alpha*op(B)*x + beta*y
 *
 * Supported cases:
 *   1) rownnz_B != NULL: y(rownnz_B) = alpha*op(B)*x + beta*y(rownnz_B)
 *
 *   2) op(B) = B (trans = 0) or B^T (trans = 1)
 *      op(B) = B^T: not recommended since it computes B^T at every call
 *
 *   3) multi-component vectors up to 64 components (1 <= num_vectors <= 64)
 *
 * Notes:
 *   1) if B has no numerical values, assume the values are all ones
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSpMVDevice( NALU_HYPRE_Int        trans,
                           NALU_HYPRE_Complex    alpha,
                           nalu_hypre_CSRMatrix *B,
                           nalu_hypre_Vector    *x,
                           NALU_HYPRE_Complex    beta,
                           nalu_hypre_Vector    *y,
                           NALU_HYPRE_Int        fill )
{
   /* Input data variables */
   NALU_HYPRE_Int        num_rows      = trans ? nalu_hypre_CSRMatrixNumCols(B) : nalu_hypre_CSRMatrixNumRows(B);
   NALU_HYPRE_Int        num_nonzeros  = nalu_hypre_CSRMatrixNumNonzeros(B);
   NALU_HYPRE_Int        num_vectors_x = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int        num_vectors_y = nalu_hypre_VectorNumVectors(y);
   NALU_HYPRE_Complex   *d_x           = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex   *d_y           = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int        idxstride_x   = nalu_hypre_VectorIndexStride(x);
   NALU_HYPRE_Int        vecstride_x   = nalu_hypre_VectorVectorStride(x);
   NALU_HYPRE_Int        idxstride_y   = nalu_hypre_VectorIndexStride(y);
   NALU_HYPRE_Int        vecstride_y   = nalu_hypre_VectorVectorStride(y);

   /* Matrix A variables */
   nalu_hypre_CSRMatrix *A = NULL;
   NALU_HYPRE_Int       *d_ia;
   NALU_HYPRE_Int       *d_ja;
   NALU_HYPRE_Complex   *d_a;
   NALU_HYPRE_Int       *d_rownnz_A = NULL;

   /* Sanity checks */
   if (num_vectors_x != num_vectors_y)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "num_vectors_x != num_vectors_y");
      return nalu_hypre_error_flag;
   }
   nalu_hypre_assert(num_rows > 0);

   /* Trivial case when alpha * op(B) * x = 0 */
   if (num_nonzeros <= 0 || alpha == 0.0)
   {
      nalu_hypre_SeqVectorScale(beta, y);
      return nalu_hypre_error_flag;
   }

   /* Select op(B) */
   if (trans)
   {
      nalu_hypre_CSRMatrixTransposeDevice(B, &A, nalu_hypre_CSRMatrixData(B) != NULL);
   }
   else
   {
      A = B;
   }

   /* Get matrix A info */
   d_ia = nalu_hypre_CSRMatrixI(A);
   d_ja = nalu_hypre_CSRMatrixJ(A);
   d_a  = nalu_hypre_CSRMatrixData(A);
   if (nalu_hypre_CSRMatrixRownnz(A))
   {
      num_rows   = nalu_hypre_CSRMatrixNumRownnz(A);
      d_rownnz_A = nalu_hypre_CSRMatrixRownnz(A);
   }

   /* Choose matrix fill mode */
   switch (fill)
   {
      case NALU_HYPRE_SPMV_FILL_STRICT_LOWER:
         /* Strict lower matrix */
         hypreDevice_CSRMatrixMatvec<NALU_HYPRE_SPMV_FILL_STRICT_LOWER>(num_vectors_x,
                                                                   num_rows,
                                                                   d_rownnz_A,
                                                                   num_nonzeros,
                                                                   idxstride_x,
                                                                   idxstride_y,
                                                                   vecstride_x,
                                                                   vecstride_y,
                                                                   alpha,
                                                                   d_ia,
                                                                   d_ja,
                                                                   d_a,
                                                                   d_x,
                                                                   beta,
                                                                   d_y);
         break;

      case NALU_HYPRE_SPMV_FILL_LOWER:
         /* Lower matrix */
         hypreDevice_CSRMatrixMatvec<NALU_HYPRE_SPMV_FILL_LOWER>(num_vectors_x,
                                                            num_rows,
                                                            d_rownnz_A,
                                                            num_nonzeros,
                                                            idxstride_x,
                                                            idxstride_y,
                                                            vecstride_x,
                                                            vecstride_y,
                                                            alpha,
                                                            d_ia,
                                                            d_ja,
                                                            d_a,
                                                            d_x,
                                                            beta,
                                                            d_y);
         break;

      case NALU_HYPRE_SPMV_FILL_WHOLE:
         /* Full matrix */
         hypreDevice_CSRMatrixMatvec<NALU_HYPRE_SPMV_FILL_WHOLE>(num_vectors_x,
                                                            num_rows,
                                                            d_rownnz_A,
                                                            num_nonzeros,
                                                            idxstride_x,
                                                            idxstride_y,
                                                            vecstride_x,
                                                            vecstride_y,
                                                            alpha,
                                                            d_ia,
                                                            d_ja,
                                                            d_a,
                                                            d_x,
                                                            beta,
                                                            d_y);
         break;

      case NALU_HYPRE_SPMV_FILL_UPPER:
         /* Upper matrix */
         hypreDevice_CSRMatrixMatvec<NALU_HYPRE_SPMV_FILL_UPPER>(num_vectors_x,
                                                            num_rows,
                                                            d_rownnz_A,
                                                            num_nonzeros,
                                                            idxstride_x,
                                                            idxstride_y,
                                                            vecstride_x,
                                                            vecstride_y,
                                                            alpha,
                                                            d_ia,
                                                            d_ja,
                                                            d_a,
                                                            d_x,
                                                            beta,
                                                            d_y);
         break;

      case NALU_HYPRE_SPMV_FILL_STRICT_UPPER:
         /* Strict upper matrix */
         hypreDevice_CSRMatrixMatvec<NALU_HYPRE_SPMV_FILL_STRICT_UPPER>(num_vectors_x,
                                                                   num_rows,
                                                                   d_rownnz_A,
                                                                   num_nonzeros,
                                                                   idxstride_x,
                                                                   idxstride_y,
                                                                   vecstride_x,
                                                                   vecstride_y,
                                                                   alpha,
                                                                   d_ia,
                                                                   d_ja,
                                                                   d_a,
                                                                   d_x,
                                                                   beta,
                                                                   d_y);
         break;

      default:
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Fill mode for SpMV unavailable!");
         return nalu_hypre_error_flag;
   }

   /* Free memory */
   if (trans)
   {
      nalu_hypre_CSRMatrixDestroy(A);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixIntSpMVDevice
 *
 * Sparse matrix/vector multiplication with integer data on GPUs
 *
 * Note: This function does not support multi-component vectors
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixIntSpMVDevice( NALU_HYPRE_Int  num_rows,
                              NALU_HYPRE_Int  num_nonzeros,
                              NALU_HYPRE_Int  alpha,
                              NALU_HYPRE_Int *d_ia,
                              NALU_HYPRE_Int *d_ja,
                              NALU_HYPRE_Int *d_a,
                              NALU_HYPRE_Int *d_x,
                              NALU_HYPRE_Int  beta,
                              NALU_HYPRE_Int *d_y )
{
   /* Additional input variables */
   NALU_HYPRE_Int        num_vectors = 1;
   NALU_HYPRE_Int        idxstride_x = 1;
   NALU_HYPRE_Int        vecstride_x = 1;
   NALU_HYPRE_Int        idxstride_y = 1;
   NALU_HYPRE_Int        vecstride_y = 1;
   NALU_HYPRE_Int       *d_rownnz    = NULL;

   hypreDevice_CSRMatrixMatvec<NALU_HYPRE_SPMV_FILL_WHOLE, NALU_HYPRE_Int>(num_vectors,
                                                                 num_rows,
                                                                 d_rownnz,
                                                                 num_nonzeros,
                                                                 idxstride_x,
                                                                 idxstride_y,
                                                                 vecstride_x,
                                                                 vecstride_y,
                                                                 alpha,
                                                                 d_ia,
                                                                 d_ja,
                                                                 d_a,
                                                                 d_x,
                                                                 beta,
                                                                 d_y);

   return nalu_hypre_error_flag;
}
#endif /* #if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL) */
