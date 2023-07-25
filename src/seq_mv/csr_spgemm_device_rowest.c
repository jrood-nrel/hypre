/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                Row size estimations
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(NALU_HYPRE_USING_GPU)

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                       NAIVE
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
template <char type>
static __device__ __forceinline__
void nalu_hypre_rownnz_naive_rowi( nalu_hypre_DeviceItem &item,
                              NALU_HYPRE_Int  rowi,
                              NALU_HYPRE_Int  lane_id,
                              NALU_HYPRE_Int *ia,
                              NALU_HYPRE_Int *ja,
                              NALU_HYPRE_Int *ib,
                              NALU_HYPRE_Int &row_nnz_sum,
                              NALU_HYPRE_Int &row_nnz_max )
{
   /* load the start and end position of row i of A */
   NALU_HYPRE_Int j = -1;
   if (lane_id < 2)
   {
      j = read_only_load(ia + rowi + lane_id);
   }
   const NALU_HYPRE_Int istart = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 0);
   const NALU_HYPRE_Int iend   = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 1);

   row_nnz_sum = 0;
   row_nnz_max = 0;

   /* load column idx and values of row i of A */
   for (NALU_HYPRE_Int i = istart; i < iend; i += NALU_HYPRE_WARP_SIZE)
   {
      if (i + lane_id < iend)
      {
         NALU_HYPRE_Int colA = read_only_load(ja + i + lane_id);
         NALU_HYPRE_Int rowB_start = read_only_load(ib + colA);
         NALU_HYPRE_Int rowB_end   = read_only_load(ib + colA + 1);
         if (type == 'U' || type == 'B')
         {
            row_nnz_sum += rowB_end - rowB_start;
         }
         if (type == 'L' || type == 'B')
         {
#if defined(NALU_HYPRE_USING_SYCL)
            row_nnz_max = std::max(row_nnz_max, rowB_end - rowB_start);
#else
            row_nnz_max = max(row_nnz_max, rowB_end - rowB_start);
#endif
         }
      }
   }
}

template <char type, NALU_HYPRE_Int NUM_WARPS_PER_BLOCK>
__global__
void nalu_hypre_spgemm_rownnz_naive( nalu_hypre_DeviceItem &item,
                                NALU_HYPRE_Int  M,
                                NALU_HYPRE_Int  N,
                                NALU_HYPRE_Int *ia,
                                NALU_HYPRE_Int *ja,
                                NALU_HYPRE_Int *ib,
                                NALU_HYPRE_Int *jb,
                                NALU_HYPRE_Int *rcL,
                                NALU_HYPRE_Int *rcU )
{
#if defined(NALU_HYPRE_USING_SYCL)
   const NALU_HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * item.get_global_range(2);
   NALU_HYPRE_Int blockIdx_x = item.get_group(2);
#else
   const NALU_HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim.x;
   NALU_HYPRE_Int blockIdx_x = blockIdx.x;
#endif
   /* warp id inside the block */
   const NALU_HYPRE_Int warp_id = get_group_id(item);
   /* lane id inside the warp */
   volatile const NALU_HYPRE_Int lane_id = get_group_lane_id(item);

#if defined(NALU_HYPRE_USING_SYCL)
   nalu_hypre_device_assert(item.get_local_range(2) * item.get_local_range(1) == NALU_HYPRE_WARP_SIZE);
#else
   nalu_hypre_device_assert(blockDim.x * blockDim.y == NALU_HYPRE_WARP_SIZE);
#endif

   for (NALU_HYPRE_Int i = blockIdx_x * NUM_WARPS_PER_BLOCK + warp_id;
        i < M;
        i += num_warps)
   {
      NALU_HYPRE_Int jU, jL;

      nalu_hypre_rownnz_naive_rowi<type>(item, i, lane_id, ia, ja, ib, jU, jL);

      if (type == 'U' || type == 'B')
      {
         jU = warp_reduce_sum(item, jU);
#if defined(NALU_HYPRE_USING_SYCL)
         jU = sycl::min(jU, N);
#else
         jU = min(jU, N);
#endif
      }

      if (type == 'L' || type == 'B')
      {
         jL = warp_reduce_max(item, jL);
      }

      if (lane_id == 0)
      {
         if (type == 'L' || type == 'B')
         {
            rcL[i] = jL;
         }

         if (type == 'U' || type == 'B')
         {
            rcU[i] = jU;
         }
      }
   }
}

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                       COHEN
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
__global__
void nalu_hypre_expdistfromuniform( nalu_hypre_DeviceItem &item,
                               NALU_HYPRE_Int   n,
                               float      *x )
{
   const NALU_HYPRE_Int global_thread_id  = nalu_hypre_gpu_get_grid_thread_id<3, 1>(item);
   const NALU_HYPRE_Int total_num_threads = nalu_hypre_gpu_get_grid_num_threads<3, 1>(item);

#if defined(NALU_HYPRE_USING_SYCL)
   nalu_hypre_device_assert(item.get_local_range(2) * item.get_local_range(1) == NALU_HYPRE_WARP_SIZE);
#else
   nalu_hypre_device_assert(blockDim.x * blockDim.y == NALU_HYPRE_WARP_SIZE);
#endif

   for (NALU_HYPRE_Int i = global_thread_id; i < n; i += total_num_threads)
   {
      x[i] = -logf(x[i]);
   }
}

/* T = float: single precision should be enough */
template <typename T, NALU_HYPRE_Int NUM_WARPS_PER_BLOCK, NALU_HYPRE_Int SHMEM_SIZE_PER_WARP, NALU_HYPRE_Int layer>
__global__
void nalu_hypre_cohen_rowest_kernel( nalu_hypre_DeviceItem &item,
                                NALU_HYPRE_Int  nrow,
                                NALU_HYPRE_Int *rowptr,
                                NALU_HYPRE_Int *colidx,
                                T         *V_in,
                                T         *V_out,
                                NALU_HYPRE_Int *rc,
                                NALU_HYPRE_Int  nsamples,
                                NALU_HYPRE_Int *low,
                                NALU_HYPRE_Int *upp,
                                T          mult )
{
#if defined(NALU_HYPRE_USING_SYCL)
   const NALU_HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * item.get_global_range(2);
   NALU_HYPRE_Int blockIdx_x = item.get_group(2);
#else
   const NALU_HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim.x;
   NALU_HYPRE_Int blockIdx_x = blockIdx.x;
#endif
   /* warp id inside the block */
   const NALU_HYPRE_Int warp_id = get_group_id(item);
   /* lane id inside the warp */
   volatile NALU_HYPRE_Int lane_id = get_group_lane_id(item);
#if COHEN_USE_SHMEM
   __shared__ volatile NALU_HYPRE_Int s_col[NUM_WARPS_PER_BLOCK * SHMEM_SIZE_PER_WARP];
   volatile NALU_HYPRE_Int  *warp_s_col = s_col + warp_id * SHMEM_SIZE_PER_WARP;
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   nalu_hypre_device_assert(item.get_local_range(1)                           == NUM_WARPS_PER_BLOCK);
   nalu_hypre_device_assert(item.get_local_range(2) * item.get_local_range(1) == NALU_HYPRE_WARP_SIZE);
#else
   nalu_hypre_device_assert(blockDim.z              == NUM_WARPS_PER_BLOCK);
   nalu_hypre_device_assert(blockDim.x * blockDim.y == NALU_HYPRE_WARP_SIZE);
#endif
   nalu_hypre_device_assert(sizeof(T) == sizeof(float));

   for (NALU_HYPRE_Int i = blockIdx_x * NUM_WARPS_PER_BLOCK + warp_id;
        i < nrow;
        i += num_warps)
   {
      /* load the start and end position of row i */
      NALU_HYPRE_Int tmp = -1;
      if (lane_id < 2)
      {
         tmp = read_only_load(rowptr + i + lane_id);
      }
      const NALU_HYPRE_Int istart = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, tmp, 0);
      const NALU_HYPRE_Int iend   = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, tmp, 1);

      /* works on WARP_SIZE samples at a time */
      for (NALU_HYPRE_Int r = 0; r < nsamples; r += NALU_HYPRE_WARP_SIZE)
      {
         T vmin = NALU_HYPRE_FLT_LARGE;
         for (NALU_HYPRE_Int j = istart; j < iend; j += NALU_HYPRE_WARP_SIZE)
         {
            NALU_HYPRE_Int col = -1;
            const NALU_HYPRE_Int j1 = j + lane_id;
#if COHEN_USE_SHMEM
            const NALU_HYPRE_Int j2 = j1 - istart;
            if (r == 0)
            {
               if (j1 < iend)
               {
                  col = read_only_load(colidx + j1);
                  if (j2 < SHMEM_SIZE_PER_WARP)
                  {
                     warp_s_col[j2] = col;
                  }
               }

            }
            else
            {
               if (j1 < iend)
               {
                  if (j2 < SHMEM_SIZE_PER_WARP)
                  {
                     col = warp_s_col[j2];
                  }
                  else
                  {
                     col = read_only_load(colidx + j1);
                  }
               }
            }
#else
            if (j1 < iend)
            {
               col = read_only_load(colidx + j1);
            }
#endif

            for (NALU_HYPRE_Int k = 0; k < NALU_HYPRE_WARP_SIZE; k++)
            {
               NALU_HYPRE_Int colk = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, col, k);
               if (colk == -1)
               {
                  nalu_hypre_device_assert(j + NALU_HYPRE_WARP_SIZE >= iend);

                  break;
               }
               if (r + lane_id < nsamples)
               {
                  T val = read_only_load(V_in + r + lane_id + colk * nsamples);
#if defined(NALU_HYPRE_USING_SYCL)
                  vmin = sycl::min(vmin, val);
#else
                  vmin = min(vmin, val);
#endif
               }
            }
         }

         if (layer == 2)
         {
            if (r + lane_id < nsamples)
            {
               V_out[r + lane_id + i * nsamples] = vmin;
            }
         }
         else if (layer == 1)
         {
            if (r + lane_id >= nsamples)
            {
               vmin = 0.0;
            }

            /* partial sum along r */
            vmin = warp_reduce_sum(item, vmin);

            if (lane_id == 0)
            {
               if (r == 0)
               {
                  V_out[i] = vmin;
               }
               else
               {
                  V_out[i] += vmin;
               }
            }
         }
      } /* for (r = 0; ...) */

      if (layer == 1)
      {
         if (lane_id == 0)
         {
            /* estimated length of row i*/
            NALU_HYPRE_Int len = rintf( (nsamples - 1) / V_out[i] * mult );

            if (low)
            {
#if defined(NALU_HYPRE_USING_SYCL)
               len = std::max(low[i], len);
#else
               len = max(low[i], len);
#endif
            }
            if (upp)
            {
#if defined(NALU_HYPRE_USING_SYCL)
               len = std::min(upp[i], len);
#else
               len = min(upp[i], len);
#endif
            }
            if (rc)
            {
               rc[i] = len;
            }
         }
      }
   } /* for (i = ...) */
}

template <typename T, NALU_HYPRE_Int BDIMX, NALU_HYPRE_Int BDIMY, NALU_HYPRE_Int NUM_WARPS_PER_BLOCK, NALU_HYPRE_Int SHMEM_SIZE_PER_WARP>
void nalu_hypre_spgemm_rownnz_cohen( NALU_HYPRE_Int  M,
                                NALU_HYPRE_Int  K,
                                NALU_HYPRE_Int  N,
                                NALU_HYPRE_Int *d_ia,
                                NALU_HYPRE_Int *d_ja,
                                NALU_HYPRE_Int *d_ib,
                                NALU_HYPRE_Int *d_jb,
                                NALU_HYPRE_Int *d_low,
                                NALU_HYPRE_Int *d_upp,
                                NALU_HYPRE_Int *d_rc,
                                NALU_HYPRE_Int  nsamples,
                                T          mult_factor,
                                T         *work )
{
#if defined(NALU_HYPRE_USING_SYCL)
   dim3 bDim(NUM_WARPS_PER_BLOCK, BDIMY, BDIMX);
   nalu_hypre_assert(bDim.get(2) * bDim.get(1) == NALU_HYPRE_WARP_SIZE);
#else
   dim3 bDim(BDIMX, BDIMY, NUM_WARPS_PER_BLOCK);
   nalu_hypre_assert(bDim.x * bDim.y == NALU_HYPRE_WARP_SIZE);
#endif

   T *d_V1, *d_V2, *d_V3;

   d_V1 = work;
   d_V2 = d_V1 + nsamples * N;
   //d_V1 = nalu_hypre_TAlloc(T, nsamples*N, NALU_HYPRE_MEMORY_DEVICE);
   //d_V2 = nalu_hypre_TAlloc(T, nsamples*K, NALU_HYPRE_MEMORY_DEVICE);

#ifdef NALU_HYPRE_SPGEMM_TIMING
   NALU_HYPRE_Real t1, t2;
   t1 = nalu_hypre_MPI_Wtime();
#endif

   /* random V1: uniform --> exp */
   nalu_hypre_CurandUniformSingle(nsamples * N, d_V1, 0, 0, 0, 0);

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   t2 = nalu_hypre_MPI_Wtime() - t1;
   NALU_HYPRE_SPGEMM_PRINT("Curand time %f\n", t2);
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   dim3 gDim( 1, 1, (nsamples * N + bDim.get(0) * NALU_HYPRE_WARP_SIZE - 1) / (bDim.get(
                                                                             0) * NALU_HYPRE_WARP_SIZE) );
#else
   dim3 gDim( (nsamples * N + bDim.z * NALU_HYPRE_WARP_SIZE - 1) / (bDim.z * NALU_HYPRE_WARP_SIZE), 1, 1 );
#endif

   NALU_HYPRE_GPU_LAUNCH( nalu_hypre_expdistfromuniform, gDim, bDim, nsamples * N, d_V1 );

   /* step-1: layer 3-2 */
#if defined(NALU_HYPRE_USING_SYCL)
   gDim[2] = (K + bDim.get(0) - 1) / bDim.get(0);
#else
   gDim.x = (K + bDim.z - 1) / bDim.z;
#endif
   NALU_HYPRE_GPU_LAUNCH( (nalu_hypre_cohen_rowest_kernel<T, NUM_WARPS_PER_BLOCK, SHMEM_SIZE_PER_WARP, 2>),
                     gDim, bDim,
                     K, d_ib, d_jb, d_V1, d_V2, NULL, nsamples, NULL, NULL, -1.0);

   //nalu_hypre_TFree(d_V1, NALU_HYPRE_MEMORY_DEVICE);

   /* step-2: layer 2-1 */
   d_V3 = (T*) d_rc;

#if defined(NALU_HYPRE_USING_SYCL)
   gDim[2] = (M + bDim.get(0) - 1) / bDim.get(0);
#else
   gDim.x = (M + bDim.z - 1) / bDim.z;
#endif
   NALU_HYPRE_GPU_LAUNCH( (nalu_hypre_cohen_rowest_kernel<T, NUM_WARPS_PER_BLOCK, SHMEM_SIZE_PER_WARP, 1>),
                     gDim, bDim,
                     M, d_ia, d_ja, d_V2, d_V3, d_rc, nsamples, d_low, d_upp, mult_factor);

   /* done */
   //nalu_hypre_TFree(d_V2, NALU_HYPRE_MEMORY_DEVICE);
}


NALU_HYPRE_Int
hypreDevice_CSRSpGemmRownnzEstimate( NALU_HYPRE_Int  m,
                                     NALU_HYPRE_Int  k,
                                     NALU_HYPRE_Int  n,
                                     NALU_HYPRE_Int *d_ia,
                                     NALU_HYPRE_Int *d_ja,
                                     NALU_HYPRE_Int *d_ib,
                                     NALU_HYPRE_Int *d_jb,
                                     NALU_HYPRE_Int *d_rc,
                                     NALU_HYPRE_Int  row_est_mtd )
{
#ifdef NALU_HYPRE_SPGEMM_NVTX
   nalu_hypre_GpuProfilingPushRange("CSRSpGemmRowEstimate");
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPGEMM_ROWNNZ] -= nalu_hypre_MPI_Wtime();
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   NALU_HYPRE_Real t1 = nalu_hypre_MPI_Wtime();
#endif

   const NALU_HYPRE_Int num_warps_per_block =  16;
   const NALU_HYPRE_Int shmem_size_per_warp = 128;
   const NALU_HYPRE_Int BDIMX               =   2;
   const NALU_HYPRE_Int BDIMY               = NALU_HYPRE_WARP_SIZE / BDIMX;

#if defined(NALU_HYPRE_USING_SYCL)
   /* CUDA kernel configurations */
   dim3 bDim(num_warps_per_block, BDIMY, BDIMX);
   nalu_hypre_assert(bDim.get(2) * bDim.get(1) == NALU_HYPRE_WARP_SIZE);
   // for cases where one WARP works on a row
   dim3 gDim(1, 1, (m + bDim.get(0) - 1) / bDim.get(0));
#else
   /* CUDA kernel configurations */
   dim3 bDim(BDIMX, BDIMY, num_warps_per_block);
   nalu_hypre_assert(bDim.x * bDim.y == NALU_HYPRE_WARP_SIZE);
   // for cases where one WARP works on a row
   dim3 gDim( (m + bDim.z - 1) / bDim.z );
#endif

   size_t cohen_nsamples = nalu_hypre_HandleSpgemmRownnzEstimateNsamples(nalu_hypre_handle());
   float  cohen_mult     = nalu_hypre_HandleSpgemmRownnzEstimateMultFactor(nalu_hypre_handle());

   //nalu_hypre_printf("Cohen Nsamples %d, mult %f\n", cohen_nsamples, cohen_mult);

   if (row_est_mtd == 1)
   {
      /* naive overestimate */
      NALU_HYPRE_GPU_LAUNCH( (nalu_hypre_spgemm_rownnz_naive<'U', num_warps_per_block>), gDim, bDim,
                        m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, NULL, d_rc );
   }
   else if (row_est_mtd == 2)
   {
      /* naive underestimate */
      NALU_HYPRE_GPU_LAUNCH( (nalu_hypre_spgemm_rownnz_naive<'L', num_warps_per_block>), gDim, bDim,
                        m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, d_rc, NULL );
   }
   else if (row_est_mtd == 3)
   {
      /* [optional] first run naive estimate for naive lower and upper bounds,
                    which will be given to Cohen's alg as corrections */
      char *work_mem = nalu_hypre_TAlloc(char,
                                    cohen_nsamples * (n + k) * sizeof(float) + 2 * m * sizeof(NALU_HYPRE_Int),
                                    NALU_HYPRE_MEMORY_DEVICE);
      char *work_mem_saved = work_mem;

      //NALU_HYPRE_Int *d_low_upp = nalu_hypre_TAlloc(NALU_HYPRE_Int, 2 * m, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int *d_low_upp = (NALU_HYPRE_Int *) work_mem;
      work_mem += 2 * m * sizeof(NALU_HYPRE_Int);

      NALU_HYPRE_Int *d_low = d_low_upp;
      NALU_HYPRE_Int *d_upp = d_low_upp + m;

      NALU_HYPRE_GPU_LAUNCH( (nalu_hypre_spgemm_rownnz_naive<'B', num_warps_per_block>), gDim, bDim,
                        m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, d_low, d_upp );

      /* Cohen's algorithm, stochastic approach */
      nalu_hypre_spgemm_rownnz_cohen<float, BDIMX, BDIMY, num_warps_per_block, shmem_size_per_warp>
      (m, k, n, d_ia, d_ja, d_ib, d_jb, d_low, d_upp, d_rc, cohen_nsamples, cohen_mult,
       (float *)work_mem);

      //nalu_hypre_TFree(d_low_upp, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(work_mem_saved, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      char msg[256];
      nalu_hypre_sprintf(msg, "Unknown row nnz estimation method %d! \n", row_est_mtd);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, msg);
   }

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   NALU_HYPRE_Real t2 = nalu_hypre_MPI_Wtime() - t1;
   NALU_HYPRE_SPGEMM_PRINT("RownnzEst time %f\n", t2);
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPGEMM_ROWNNZ] += nalu_hypre_MPI_Wtime();
#endif

#ifdef NALU_HYPRE_SPGEMM_NVTX
   nalu_hypre_GpuProfilingPopRange();
#endif

   return nalu_hypre_error_flag;
}

#endif /* defined(NALU_HYPRE_USING_GPU) */

