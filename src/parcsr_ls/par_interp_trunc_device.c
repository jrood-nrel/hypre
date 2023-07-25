/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

#define NALU_HYPRE_INTERPTRUNC_ALGORITHM_SWITCH 8

/* special case for max_elmts = 0, i.e. no max_elmts limit */
__global__ void
hypreGPUKernel_InterpTruncationPass0_v1( nalu_hypre_DeviceItem &item,
                                         NALU_HYPRE_Int   nrows,
                                         NALU_HYPRE_Real  trunc_factor,
                                         NALU_HYPRE_Int  *P_diag_i,
                                         NALU_HYPRE_Int  *P_diag_j,
                                         NALU_HYPRE_Real *P_diag_a,
                                         NALU_HYPRE_Int  *P_offd_i,
                                         NALU_HYPRE_Int  *P_offd_j,
                                         NALU_HYPRE_Real *P_offd_a,
                                         NALU_HYPRE_Int  *P_diag_i_new,
                                         NALU_HYPRE_Int  *P_offd_i_new )
{
   NALU_HYPRE_Real row_max = 0.0, row_sum = 0.0, row_scal = 0.0;

   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p_diag = 0, q_diag = 0, p_offd = 0, q_offd = 0;

   if (lane < 2)
   {
      p_diag = read_only_load(P_diag_i + row + lane);
      p_offd = read_only_load(P_offd_i + row + lane);
   }
   q_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 0);
   q_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 0);

   /* 1. compute row rowsum, rowmax */
   for (NALU_HYPRE_Int i = p_diag + lane; i < q_diag; i += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Real v = P_diag_a[i];
      row_sum += v;
      row_max = nalu_hypre_max(row_max, nalu_hypre_abs(v));
   }

   for (NALU_HYPRE_Int i = p_offd + lane; i < q_offd; i += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Real v = P_offd_a[i];
      row_sum += v;
      row_max = nalu_hypre_max(row_max, nalu_hypre_abs(v));
   }

   row_max = warp_allreduce_max(item, row_max) * trunc_factor;
   row_sum = warp_allreduce_sum(item, row_sum);

   NALU_HYPRE_Int cnt_diag = 0, cnt_offd = 0;

   /* 2. move wanted entries to the front and row scal */
   for (NALU_HYPRE_Int i = p_diag + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < q_diag);
        i += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Real v = 0.0;
      NALU_HYPRE_Int j = -1;

      if (i < q_diag)
      {
         v = P_diag_a[i];

         if (nalu_hypre_abs(v) >= row_max)
         {
            j = P_diag_j[i];
            row_scal += v;
         }
      }

      NALU_HYPRE_Int sum, pos;
      pos = warp_prefix_sum(item, lane, (NALU_HYPRE_Int) (j != -1), sum);

      if (j != -1)
      {
         P_diag_a[p_diag + cnt_diag + pos] = v;
         P_diag_j[p_diag + cnt_diag + pos] = j;
      }

      cnt_diag += sum;
   }

   for (NALU_HYPRE_Int i = p_offd + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < q_offd);
        i += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Real v = 0.0;
      NALU_HYPRE_Int j = -1;

      if (i < q_offd)
      {
         v = P_offd_a[i];

         if (nalu_hypre_abs(v) >= row_max)
         {
            j = P_offd_j[i];
            row_scal += v;
         }
      }

      NALU_HYPRE_Int sum, pos;
      pos = warp_prefix_sum(item, lane, (NALU_HYPRE_Int) (j != -1), sum);

      if (j != -1)
      {
         P_offd_a[p_offd + cnt_offd + pos] = v;
         P_offd_j[p_offd + cnt_offd + pos] = j;
      }

      cnt_offd += sum;
   }

   row_scal = warp_allreduce_sum(item, row_scal);

   if (row_scal)
   {
      row_scal = row_sum / row_scal;
   }
   else
   {
      row_scal = 1.0;
   }

   /* 3. scale the row */
   for (NALU_HYPRE_Int i = p_diag + lane; i < p_diag + cnt_diag; i += NALU_HYPRE_WARP_SIZE)
   {
      P_diag_a[i] *= row_scal;
   }

   for (NALU_HYPRE_Int i = p_offd + lane; i < p_offd + cnt_offd; i += NALU_HYPRE_WARP_SIZE)
   {
      P_offd_a[i] *= row_scal;
   }

   if (!lane)
   {
      P_diag_i_new[row] = cnt_diag;
      P_offd_i_new[row] = cnt_offd;
   }
}

static __device__ __forceinline__
void nalu_hypre_smallest_abs_val( NALU_HYPRE_Int   n,
                             NALU_HYPRE_Real *v,
                             NALU_HYPRE_Real &min_v,
                             NALU_HYPRE_Int  &min_j )
{
   min_v = nalu_hypre_abs(v[0]);
   min_j = 0;

   for (NALU_HYPRE_Int j = 1; j < n; j++)
   {
      const NALU_HYPRE_Real vj = nalu_hypre_abs(v[j]);
      if (vj < min_v)
      {
         min_v = vj;
         min_j = j;
      }
   }
}

/* TODO: using 1 thread per row, which can be suboptimal */
__global__ void
hypreGPUKernel_InterpTruncationPass1_v1( nalu_hypre_DeviceItem &item,
#if defined(NALU_HYPRE_USING_SYCL)
                                         char *shmem_ptr,
#endif
                                         NALU_HYPRE_Int   nrows,
                                         NALU_HYPRE_Real  trunc_factor,
                                         NALU_HYPRE_Int   max_elmts,
                                         NALU_HYPRE_Int  *P_diag_i,
                                         NALU_HYPRE_Int  *P_diag_j,
                                         NALU_HYPRE_Real *P_diag_a,
                                         NALU_HYPRE_Int  *P_offd_i,
                                         NALU_HYPRE_Int  *P_offd_j,
                                         NALU_HYPRE_Real *P_offd_a,
                                         NALU_HYPRE_Int  *P_diag_i_new,
                                         NALU_HYPRE_Int  *P_offd_i_new )
{
   const NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   const NALU_HYPRE_Int p_diag = read_only_load(P_diag_i + row);
   const NALU_HYPRE_Int q_diag = read_only_load(P_diag_i + row + 1);
   const NALU_HYPRE_Int p_offd = read_only_load(P_offd_i + row);
   const NALU_HYPRE_Int q_offd = read_only_load(P_offd_i + row + 1);

   /* 1. get row max and compute truncation threshold, and compute row_sum */
   NALU_HYPRE_Real row_max = 0.0, row_sum = 0.0;

   for (NALU_HYPRE_Int i = p_diag; i < q_diag; i++)
   {
      NALU_HYPRE_Real v = P_diag_a[i];
      row_sum += v;
      row_max = nalu_hypre_max(row_max, nalu_hypre_abs(v));
   }

   for (NALU_HYPRE_Int i = p_offd; i < q_offd; i++)
   {
      NALU_HYPRE_Real v = P_offd_a[i];
      row_sum += v;
      row_max = nalu_hypre_max(row_max, nalu_hypre_abs(v));
   }

   row_max *= trunc_factor;

   /* 2. save the largest max_elmts entries in sh_val/pos */
   const NALU_HYPRE_Int nt = nalu_hypre_gpu_get_num_threads<1>(item);
   const NALU_HYPRE_Int tid = nalu_hypre_gpu_get_thread_id<1>(item);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int *shared_mem = (NALU_HYPRE_Int*) shmem_ptr;
#else
   extern __shared__ NALU_HYPRE_Int shared_mem[];
#endif
   NALU_HYPRE_Int *sh_pos = &shared_mem[tid * max_elmts];
   NALU_HYPRE_Real *sh_val = &((NALU_HYPRE_Real *) &shared_mem[nt * max_elmts])[tid * max_elmts];
   NALU_HYPRE_Int cnt = 0;

   for (NALU_HYPRE_Int i = p_diag; i < q_diag; i++)
   {
      const NALU_HYPRE_Real v = P_diag_a[i];

      if (nalu_hypre_abs(v) < row_max) { continue; }

      if (cnt < max_elmts)
      {
         sh_val[cnt] = v;
         sh_pos[cnt ++] = i;
      }
      else
      {
         NALU_HYPRE_Real min_v;
         NALU_HYPRE_Int min_j;

         nalu_hypre_smallest_abs_val(max_elmts, sh_val, min_v, min_j);

         if (nalu_hypre_abs(v) > min_v)
         {
            sh_val[min_j] = v;
            sh_pos[min_j] = i;
         }
      }
   }

   for (NALU_HYPRE_Int i = p_offd; i < q_offd; i++)
   {
      const NALU_HYPRE_Real v = P_offd_a[i];

      if (nalu_hypre_abs(v) < row_max) { continue; }

      if (cnt < max_elmts)
      {
         sh_val[cnt] = v;
         sh_pos[cnt ++] = i + q_diag;
      }
      else
      {
         NALU_HYPRE_Real min_v;
         NALU_HYPRE_Int min_j;

         nalu_hypre_smallest_abs_val(max_elmts, sh_val, min_v, min_j);

         if (nalu_hypre_abs(v) > min_v)
         {
            sh_val[min_j] = v;
            sh_pos[min_j] = i + q_diag;
         }
      }
   }

   /* 3. load actual j and compute row_scal */
   NALU_HYPRE_Real row_scal = 0.0;

   for (NALU_HYPRE_Int i = 0; i < cnt; i++)
   {
      const NALU_HYPRE_Int j = sh_pos[i];

      if (j < q_diag)
      {
         sh_pos[i] = P_diag_j[j];
      }
      else
      {
         sh_pos[i] = -1 - P_offd_j[j - q_diag];
      }

      row_scal += sh_val[i];
   }

   if (row_scal)
   {
      row_scal = row_sum / row_scal;
   }
   else
   {
      row_scal = 1.0;
   }

   /* 4. write to P_diag_j and P_offd_j */
   NALU_HYPRE_Int cnt_diag = 0;
   for (NALU_HYPRE_Int i = 0; i < cnt; i++)
   {
      const NALU_HYPRE_Int j = sh_pos[i];

      if (j >= 0)
      {
         P_diag_j[p_diag + cnt_diag] = j;
         P_diag_a[p_diag + cnt_diag] = sh_val[i] * row_scal;
         cnt_diag ++;
      }
      else
      {
         P_offd_j[p_offd + i - cnt_diag] = -1 - j;
         P_offd_a[p_offd + i - cnt_diag] = sh_val[i] * row_scal;
      }
   }

   P_diag_i_new[row] = cnt_diag;
   P_offd_i_new[row] = cnt - cnt_diag;
}

/* using 1 warp per row */
__global__ void
hypreGPUKernel_InterpTruncationPass2_v1( nalu_hypre_DeviceItem &item,
                                         NALU_HYPRE_Int   nrows,
                                         NALU_HYPRE_Int  *P_diag_i,
                                         NALU_HYPRE_Int  *P_diag_j,
                                         NALU_HYPRE_Real *P_diag_a,
                                         NALU_HYPRE_Int  *P_offd_i,
                                         NALU_HYPRE_Int  *P_offd_j,
                                         NALU_HYPRE_Real *P_offd_a,
                                         NALU_HYPRE_Int  *P_diag_i_new,
                                         NALU_HYPRE_Int  *P_diag_j_new,
                                         NALU_HYPRE_Real *P_diag_a_new,
                                         NALU_HYPRE_Int  *P_offd_i_new,
                                         NALU_HYPRE_Int  *P_offd_j_new,
                                         NALU_HYPRE_Real *P_offd_a_new )
{
   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0, pnew = 0, qnew = 0, shift;

   if (lane < 2)
   {
      p = read_only_load(P_diag_i + i + lane);
      pnew = read_only_load(P_diag_i_new + i + lane);
   }
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);
   qnew = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pnew, 1);
   pnew = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pnew, 0);

   shift = p - pnew;
   for (NALU_HYPRE_Int k = pnew + lane; k < qnew; k += NALU_HYPRE_WARP_SIZE)
   {
      P_diag_j_new[k] = P_diag_j[k + shift];
      P_diag_a_new[k] = P_diag_a[k + shift];
   }

   if (lane < 2)
   {
      p = read_only_load(P_offd_i + i + lane);
      pnew = read_only_load(P_offd_i_new + i + lane);
   }
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);
   qnew = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pnew, 1);
   pnew = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pnew, 0);

   shift = p - pnew;
   for (NALU_HYPRE_Int k = pnew + lane; k < qnew; k += NALU_HYPRE_WARP_SIZE)
   {
      P_offd_j_new[k] = P_offd_j[k + shift];
      P_offd_a_new[k] = P_offd_a[k + shift];
   }
}

/* This is a "fast" version that works for small max_elmts values */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGInterpTruncationDevice_v1( nalu_hypre_ParCSRMatrix *P,
                                          NALU_HYPRE_Real          trunc_factor,
                                          NALU_HYPRE_Int           max_elmts )
{
   NALU_HYPRE_Int        nrows       = nalu_hypre_ParCSRMatrixNumRows(P);
   nalu_hypre_CSRMatrix *P_diag      = nalu_hypre_ParCSRMatrixDiag(P);
   NALU_HYPRE_Int       *P_diag_i    = nalu_hypre_CSRMatrixI(P_diag);
   NALU_HYPRE_Int       *P_diag_j    = nalu_hypre_CSRMatrixJ(P_diag);
   NALU_HYPRE_Real      *P_diag_a    = nalu_hypre_CSRMatrixData(P_diag);
   nalu_hypre_CSRMatrix *P_offd      = nalu_hypre_ParCSRMatrixOffd(P);
   NALU_HYPRE_Int       *P_offd_i    = nalu_hypre_CSRMatrixI(P_offd);
   NALU_HYPRE_Int       *P_offd_j    = nalu_hypre_CSRMatrixJ(P_offd);
   NALU_HYPRE_Real      *P_offd_a    = nalu_hypre_CSRMatrixData(P_offd);

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(P);

   NALU_HYPRE_Int *P_diag_i_new = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows + 1, memory_location);
   NALU_HYPRE_Int *P_offd_i_new = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows + 1, memory_location);

   /* truncate P, wanted entries are marked negative in P_diag/offd_j */
   if (max_elmts == 0)
   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_InterpTruncationPass0_v1,
                        gDim, bDim,
                        nrows, trunc_factor,
                        P_diag_i, P_diag_j, P_diag_a,
                        P_offd_i, P_offd_j, P_offd_a,
                        P_diag_i_new, P_offd_i_new);
   }
   else
   {
      dim3 bDim = nalu_hypre_dim3(256);
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nrows, "thread", bDim);
#if defined(NALU_HYPRE_USING_SYCL)
      size_t shmem_bytes = bDim.get(2) * max_elmts * (sizeof(NALU_HYPRE_Int) + sizeof(NALU_HYPRE_Real));
#else
      size_t shmem_bytes = bDim.x * max_elmts * (sizeof(NALU_HYPRE_Int) + sizeof(NALU_HYPRE_Real));
#endif
      NALU_HYPRE_GPU_LAUNCH2( hypreGPUKernel_InterpTruncationPass1_v1,
                         gDim, bDim, shmem_bytes,
                         nrows, trunc_factor, max_elmts,
                         P_diag_i, P_diag_j, P_diag_a,
                         P_offd_i, P_offd_j, P_offd_a,
                         P_diag_i_new, P_offd_i_new);
   }

   nalu_hypre_Memset(&P_diag_i_new[nrows], 0, sizeof(NALU_HYPRE_Int), memory_location);
   nalu_hypre_Memset(&P_offd_i_new[nrows], 0, sizeof(NALU_HYPRE_Int), memory_location);

   hypreDevice_IntegerExclusiveScan(nrows + 1, P_diag_i_new);
   hypreDevice_IntegerExclusiveScan(nrows + 1, P_offd_i_new);

   NALU_HYPRE_Int nnz_diag, nnz_offd;

   nalu_hypre_TMemcpy(&nnz_diag, &P_diag_i_new[nrows], NALU_HYPRE_Int, 1,
                 NALU_HYPRE_MEMORY_HOST, memory_location);
   nalu_hypre_TMemcpy(&nnz_offd, &P_offd_i_new[nrows], NALU_HYPRE_Int, 1,
                 NALU_HYPRE_MEMORY_HOST, memory_location);

   nalu_hypre_CSRMatrixNumNonzeros(P_diag) = nnz_diag;
   nalu_hypre_CSRMatrixNumNonzeros(P_offd) = nnz_offd;

   NALU_HYPRE_Int  *P_diag_j_new = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_diag, memory_location);
   NALU_HYPRE_Real *P_diag_a_new = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_diag, memory_location);
   NALU_HYPRE_Int  *P_offd_j_new = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_offd, memory_location);
   NALU_HYPRE_Real *P_offd_a_new = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_offd, memory_location);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_InterpTruncationPass2_v1,
                     gDim, bDim,
                     nrows,
                     P_diag_i, P_diag_j, P_diag_a,
                     P_offd_i, P_offd_j, P_offd_a,
                     P_diag_i_new, P_diag_j_new, P_diag_a_new,
                     P_offd_i_new, P_offd_j_new, P_offd_a_new );

   nalu_hypre_CSRMatrixI   (P_diag) = P_diag_i_new;
   nalu_hypre_CSRMatrixJ   (P_diag) = P_diag_j_new;
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_a_new;
   nalu_hypre_CSRMatrixI   (P_offd) = P_offd_i_new;
   nalu_hypre_CSRMatrixJ   (P_offd) = P_offd_j_new;
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_a_new;

   nalu_hypre_TFree(P_diag_i, memory_location);
   nalu_hypre_TFree(P_diag_j, memory_location);
   nalu_hypre_TFree(P_diag_a, memory_location);
   nalu_hypre_TFree(P_offd_i, memory_location);
   nalu_hypre_TFree(P_offd_j, memory_location);
   nalu_hypre_TFree(P_offd_a, memory_location);

   return nalu_hypre_error_flag;
}

__global__ void
hypreGPUKernel_InterpTruncation_v2( nalu_hypre_DeviceItem &item,
                                    NALU_HYPRE_Int   nrows,
                                    NALU_HYPRE_Real  trunc_factor,
                                    NALU_HYPRE_Int   max_elmts,
                                    NALU_HYPRE_Int  *P_i,
                                    NALU_HYPRE_Int  *P_j,
                                    NALU_HYPRE_Real *P_a)
{
   NALU_HYPRE_Real row_max = 0.0, row_sum = 0.0, row_scal = 0.0;
   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item), p = 0, q;

   /* 1. compute row max, rowsum */
   if (lane < 2)
   {
      p = read_only_load(P_i + row + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int i = p + lane; i < q; i += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Real v = read_only_load(&P_a[i]);
      row_max = nalu_hypre_max(row_max, nalu_hypre_abs(v));
      row_sum += v;
   }

   row_max = warp_allreduce_max(item, row_max) * trunc_factor;
   row_sum = warp_allreduce_sum(item, row_sum);

   /* 2. mark dropped entries by -1 in P_j, and compute row_scal */
   NALU_HYPRE_Int last_pos = -1;
   for (NALU_HYPRE_Int i = p + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < q); i += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int cond = 0, cond_prev;

      cond_prev = i == p + lane || warp_allreduce_min(item, cond);

      if (i < q)
      {
         NALU_HYPRE_Real v;
         cond = cond_prev && (max_elmts == 0 || i < p + max_elmts);
         if (cond)
         {
            v = read_only_load(&P_a[i]);
         }
         cond = cond && nalu_hypre_abs(v) >= row_max;

         if (cond)
         {
            last_pos = i;
            row_scal += v;
         }
         else
         {
            P_j[i] = -1;
         }
      }
   }

   row_scal = warp_allreduce_sum(item, row_scal);

   if (row_scal)
   {
      row_scal = row_sum / row_scal;
   }
   else
   {
      row_scal = 1.0;
   }

   /* 3. scale the row */
   for (NALU_HYPRE_Int i = p + lane; i <= last_pos; i += NALU_HYPRE_WARP_SIZE)
   {
      P_a[i] *= row_scal;
   }
}

/*------------------------------------------------------------------------------------
 * RL: To be consistent with the CPU version, max_elmts == 0 means no limit on rownnz
 * This is a generic version that works for all max_elmts values
 */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGInterpTruncationDevice_v2( nalu_hypre_ParCSRMatrix *P,
                                          NALU_HYPRE_Real          trunc_factor,
                                          NALU_HYPRE_Int           max_elmts )
{
   nalu_hypre_CSRMatrix *P_diag      = nalu_hypre_ParCSRMatrixDiag(P);
   NALU_HYPRE_Int       *P_diag_i    = nalu_hypre_CSRMatrixI(P_diag);
   NALU_HYPRE_Int       *P_diag_j    = nalu_hypre_CSRMatrixJ(P_diag);
   NALU_HYPRE_Real      *P_diag_a    = nalu_hypre_CSRMatrixData(P_diag);

   nalu_hypre_CSRMatrix *P_offd      = nalu_hypre_ParCSRMatrixOffd(P);
   NALU_HYPRE_Int       *P_offd_i    = nalu_hypre_CSRMatrixI(P_offd);
   NALU_HYPRE_Int       *P_offd_j    = nalu_hypre_CSRMatrixJ(P_offd);
   NALU_HYPRE_Real      *P_offd_a    = nalu_hypre_CSRMatrixData(P_offd);

   //NALU_HYPRE_Int        ncols       = nalu_hypre_CSRMatrixNumCols(P_diag);
   NALU_HYPRE_Int        nrows       = nalu_hypre_CSRMatrixNumRows(P_diag);
   NALU_HYPRE_Int        nnz_diag    = nalu_hypre_CSRMatrixNumNonzeros(P_diag);
   NALU_HYPRE_Int        nnz_offd    = nalu_hypre_CSRMatrixNumNonzeros(P_offd);
   NALU_HYPRE_Int        nnz_P       = nnz_diag + nnz_offd;
   NALU_HYPRE_Int       *P_i         = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_P,     NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int       *P_j         = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_P,     NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Real      *P_a         = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_P,     NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int       *P_rowptr    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int       *tmp_rowid   = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_P,     NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_Int        new_nnz_diag = 0, new_nnz_offd = 0;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(P);

   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnz_diag, P_diag_i, P_i);
   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnz_offd, P_offd_i, P_i + nnz_diag);

   nalu_hypre_TMemcpy(P_j, P_diag_j, NALU_HYPRE_Int, nnz_diag, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   /* offd col id := -2 - offd col id */
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL(std::transform, P_offd_j, P_offd_j + nnz_offd, P_j + nnz_diag,
   [] (const auto & x) {return -x - 2;} );
#else
   NALU_HYPRE_THRUST_CALL(transform, P_offd_j, P_offd_j + nnz_offd, P_j + nnz_diag, -_1 - 2);
#endif

   nalu_hypre_TMemcpy(P_a,            P_diag_a, NALU_HYPRE_Real, nnz_diag, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(P_a + nnz_diag, P_offd_a, NALU_HYPRE_Real, nnz_offd, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_DEVICE);

   /* sort rows based on (rowind, abs(P_a)) */
   hypreDevice_StableSortByTupleKey(nnz_P, P_i, P_a, P_j, 1);

   hypreDevice_CsrRowIndicesToPtrs_v2(nrows, nnz_P, P_i, P_rowptr);

   /* truncate P, unwanted entries are marked -1 in P_j */
   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_InterpTruncation_v2, gDim, bDim,
                     nrows, trunc_factor, max_elmts, P_rowptr, P_j, P_a );

   /* build new P_diag and P_offd */
   if (nnz_diag)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(P_i,       P_j,       P_a),
                                        oneapi::dpl::make_zip_iterator(P_i + nnz_P, P_j + nnz_P, P_a + nnz_P),
                                        P_j,
                                        oneapi::dpl::make_zip_iterator(tmp_rowid, P_diag_j,  P_diag_a),
                                        is_nonnegative<NALU_HYPRE_Int>() );
      new_nnz_diag = std::get<0>(new_end.base()) - tmp_rowid;
#else
      auto new_end = NALU_HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(P_i,       P_j,       P_a)),
                        thrust::make_zip_iterator(thrust::make_tuple(P_i + nnz_P, P_j + nnz_P, P_a + nnz_P)),
                        P_j,
                        thrust::make_zip_iterator(thrust::make_tuple(tmp_rowid, P_diag_j,  P_diag_a)),
                        is_nonnegative<NALU_HYPRE_Int>() );
      new_nnz_diag = thrust::get<0>(new_end.get_iterator_tuple()) - tmp_rowid;
#endif

      nalu_hypre_assert(new_nnz_diag <= nnz_diag);

      hypreDevice_CsrRowIndicesToPtrs_v2(nrows, new_nnz_diag, tmp_rowid, P_diag_i);
   }

   if (nnz_offd)
   {
      less_than<NALU_HYPRE_Int> pred(-1);
#if defined(NALU_HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(P_i,       P_j,       P_a),
                                        oneapi::dpl::make_zip_iterator(P_i + nnz_P, P_j + nnz_P, P_a + nnz_P),
                                        P_j,
                                        oneapi::dpl::make_zip_iterator(tmp_rowid, P_offd_j,  P_offd_a),
                                        pred );
      new_nnz_offd = std::get<0>(new_end.base()) - tmp_rowid;
#else
      auto new_end = NALU_HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(P_i,       P_j,       P_a)),
                        thrust::make_zip_iterator(thrust::make_tuple(P_i + nnz_P, P_j + nnz_P, P_a + nnz_P)),
                        P_j,
                        thrust::make_zip_iterator(thrust::make_tuple(tmp_rowid, P_offd_j,  P_offd_a)),
                        pred );
      new_nnz_offd = thrust::get<0>(new_end.get_iterator_tuple()) - tmp_rowid;
#endif

      nalu_hypre_assert(new_nnz_offd <= nnz_offd);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL(std::transform, P_offd_j, P_offd_j + new_nnz_offd, P_offd_j,
      [] (const auto & x) {return -x - 2;} );
#else
      NALU_HYPRE_THRUST_CALL(transform, P_offd_j, P_offd_j + new_nnz_offd, P_offd_j, -_1 - 2);
#endif

      hypreDevice_CsrRowIndicesToPtrs_v2(nrows, new_nnz_offd, tmp_rowid, P_offd_i);
   }

   nalu_hypre_CSRMatrixJ   (P_diag) = nalu_hypre_TReAlloc_v2(P_diag_j, NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_Int,
                                                   new_nnz_diag, memory_location);
   nalu_hypre_CSRMatrixData(P_diag) = nalu_hypre_TReAlloc_v2(P_diag_a, NALU_HYPRE_Real, nnz_diag, NALU_HYPRE_Real,
                                                   new_nnz_diag, memory_location);
   nalu_hypre_CSRMatrixJ   (P_offd) = nalu_hypre_TReAlloc_v2(P_offd_j, NALU_HYPRE_Int,  nnz_offd, NALU_HYPRE_Int,
                                                   new_nnz_offd, memory_location);
   nalu_hypre_CSRMatrixData(P_offd) = nalu_hypre_TReAlloc_v2(P_offd_a, NALU_HYPRE_Real, nnz_offd, NALU_HYPRE_Real,
                                                   new_nnz_offd, memory_location);
   nalu_hypre_CSRMatrixNumNonzeros(P_diag) = new_nnz_diag;
   nalu_hypre_CSRMatrixNumNonzeros(P_offd) = new_nnz_offd;

   nalu_hypre_TFree(P_i,       NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(P_j,       NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(P_a,       NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(P_rowptr,  NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(tmp_rowid, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGInterpTruncationDevice( nalu_hypre_ParCSRMatrix *P,
                                       NALU_HYPRE_Real          trunc_factor,
                                       NALU_HYPRE_Int           max_elmts )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_INTERP_TRUNC] -= nalu_hypre_MPI_Wtime();
#endif
   nalu_hypre_GpuProfilingPushRange("Interp-Truncation");

   if (max_elmts <= NALU_HYPRE_INTERPTRUNC_ALGORITHM_SWITCH)
   {
      nalu_hypre_BoomerAMGInterpTruncationDevice_v1(P, trunc_factor, max_elmts);
   }
   else
   {
      nalu_hypre_BoomerAMGInterpTruncationDevice_v2(P, trunc_factor, max_elmts);
   }

   nalu_hypre_GpuProfilingPopRange();

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_INTERP_TRUNC] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

#endif /* #if defined(NALU_HYPRE_USING_GPU) */
