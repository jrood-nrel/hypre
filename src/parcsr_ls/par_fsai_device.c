/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"
#include "par_fsai.h"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

#define mat_(ldim, k, i, j) mat_data[ldim * (ldim * k + i) + j]
#define rhs_(ldim, i, j)    rhs_data[ldim * i + j]
#define sol_(ldim, i, j)    sol_data[ldim * i + j]

#define NALU_HYPRE_THRUST_ZIP3(A, B, C) thrust::make_zip_iterator(thrust::make_tuple(A, B, C))

/*--------------------------------------------------------------------
 * hypreGPUKernel_FSAIExtractSubSystems
 *
 * Output:
 *   1) mat_data: dense matrix coefficients.
 *   2) rhs_data: right hand side coefficients.
 *   3) G_r: number of nonzero coefficients per row of the matrix G.
 *
 * TODO:
 *   1) Minimize intra-warp divergence.
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_FSAIExtractSubSystems( nalu_hypre_DeviceItem &item,
                                      NALU_HYPRE_Int         num_rows,
                                      NALU_HYPRE_Int        *A_i,
                                      NALU_HYPRE_Int        *A_j,
                                      NALU_HYPRE_Complex    *A_a,
                                      NALU_HYPRE_Int        *P_i,
                                      NALU_HYPRE_Int        *P_e,
                                      NALU_HYPRE_Int        *P_j,
                                      NALU_HYPRE_Int         ldim,
                                      NALU_HYPRE_Complex    *mat_data,
                                      NALU_HYPRE_Complex    *rhs_data,
                                      NALU_HYPRE_Int        *G_r )
{
   NALU_HYPRE_Int      lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int      i, j, jj, k;
   NALU_HYPRE_Int      pj, qj;
   NALU_HYPRE_Int      pk, qk;
   NALU_HYPRE_Int      A_col, P_col;
   NALU_HYPRE_Complex  val;
   nalu_hypre_mask     bitmask;

   /* Grid-stride loop over matrix rows */
   for (i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);
        i < num_rows;
        i += nalu_hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      /* Set identity matrix */
      for (j = lane; j < ldim; j += NALU_HYPRE_WARP_SIZE)
      {
         mat_(ldim, i, j, j) = 1.0;
      }

      if (lane == 0)
      {
         pj = read_only_load(P_i + i);
         qj = read_only_load(P_e + i);
      }
      qj = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, qj, 0, NALU_HYPRE_WARP_SIZE);
      pj = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pj, 0, NALU_HYPRE_WARP_SIZE);

      if (lane < 2)
      {
         pk = read_only_load(A_i + i + lane);
      }
      qk = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pk, 1, NALU_HYPRE_WARP_SIZE);
      pk = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pk, 0, NALU_HYPRE_WARP_SIZE);

      /* Set right hand side vector */
      for (j = pj; j < qj; j++)
      {
         if (lane == 0)
         {
            P_col = read_only_load(P_j + j);
         }
         P_col = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, P_col, 0, NALU_HYPRE_WARP_SIZE);

         for (k = pk + lane;
              warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, k < qk);
              k += NALU_HYPRE_WARP_SIZE)
         {
            if (k < qk)
            {
               A_col = read_only_load(A_j + k);
            }
            else
            {
               A_col = -1;
            }

            bitmask = nalu_hypre_ballot_sync(NALU_HYPRE_WARP_FULL_MASK, A_col == P_col);
            if (bitmask > 0)
            {
               if (lane == (nalu_hypre_ffs(bitmask) - 1))
               {
                  rhs_(ldim, i, j - pj) = - read_only_load(A_a + k);
               }
               break;
            }
         }
      }

      /* Loop over requested rows */
      for (j = pj; j < qj; j++)
      {
         if (lane < 2)
         {
            pk = read_only_load(A_i + P_j[j] + lane);
         }
         qk = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pk, 1, NALU_HYPRE_WARP_SIZE);
         pk = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pk, 0, NALU_HYPRE_WARP_SIZE);

         /* Visit only the lower triangular part */
         for (jj = pj; jj <= j; jj++)
         {
            if (lane == 0)
            {
               P_col = read_only_load(P_j + jj);
            }
            P_col = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, P_col, 0, NALU_HYPRE_WARP_SIZE);

            for (k = pk + lane;
                 warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, k < qk);
                 k += NALU_HYPRE_WARP_SIZE)
            {
               if (k < qk)
               {
                  A_col = read_only_load(A_j + k);
               }
               else
               {
                  A_col = -1;
               }

               bitmask = nalu_hypre_ballot_sync(NALU_HYPRE_WARP_FULL_MASK, A_col == P_col);
               if (bitmask > 0)
               {
                  if (lane == (nalu_hypre_ffs(bitmask) - 1))
                  {
                     val = read_only_load(A_a + k);
                     mat_(ldim, i, j - pj, jj - pj) = val;
                     mat_(ldim, i, jj - pj, j - pj) = val;
                  }
                  break;
               }
            }
         }
      }

      /* Set number of nonzero coefficients per row of G */
      if (lane == 0)
      {
         G_r[i] = qj - pj + 1;
      }
   } /* Grid-stride loop over matrix rows */
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_FSAIScaling
 *
 * TODO: unroll inner loop
 *       Use fma?
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_FSAIScaling( nalu_hypre_DeviceItem &item,
                            NALU_HYPRE_Int         num_rows,
                            NALU_HYPRE_Int         ldim,
                            NALU_HYPRE_Complex    *sol_data,
                            NALU_HYPRE_Complex    *rhs_data,
                            NALU_HYPRE_Complex    *scaling,
                            NALU_HYPRE_Int        *info )
{
   NALU_HYPRE_Int      i, j;
   NALU_HYPRE_Complex  val;

   /* Grid-stride loop over matrix rows */
   for (i = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += nalu_hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      val = scaling[i];
      for (j = 0; j < ldim; j++)
      {
         val += sol_(ldim, i, j) * rhs_(ldim, i, j);
      }

      if (val > 0)
      {
         scaling[i] = 1.0 / sqrt(val);
      }
      else
      {
         scaling[i] = 1.0 / sqrt(scaling[i]);
         info[i] = 1;
      }
   }
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_FSAIGatherEntries
 *
 * Output:
 *   1) G_j: column indices of G_diag
 *   2) G_a: coefficients of G_diag
 *
 * TODO:
 *   1) Use a (sub-)warp per row of G
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_FSAIGatherEntries( nalu_hypre_DeviceItem &item,
                                  NALU_HYPRE_Int         num_rows,
                                  NALU_HYPRE_Int         ldim,
                                  NALU_HYPRE_Complex    *sol_data,
                                  NALU_HYPRE_Complex    *scaling,
                                  NALU_HYPRE_Int        *K_i,
                                  NALU_HYPRE_Int        *K_e,
                                  NALU_HYPRE_Int        *K_j,
                                  NALU_HYPRE_Int        *G_i,
                                  NALU_HYPRE_Int        *G_j,
                                  NALU_HYPRE_Complex    *G_a )
{
   NALU_HYPRE_Int      i, j;
   NALU_HYPRE_Int      cnt, il;
   NALU_HYPRE_Int      col;
   NALU_HYPRE_Complex  val;

   /* Grid-stride loop over matrix rows */
   for (i = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += nalu_hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      /* Set scaling factor */
      val = scaling[i];

      /* Set diagonal coefficient */
      cnt = G_i[i];
      G_j[cnt] = i;
      G_a[cnt] = val;
      cnt++;

      /* Set off-diagonal coefficients */
      il = 0;
      for (j = K_i[i]; j < K_e[i]; j++)
      {
         col = K_j[j];

         G_j[cnt + il] = col;
         G_a[cnt + il] = sol_(ldim, i, il) * val;
         il++;
      }
   }
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_FSAITruncateCandidateOrdered
 *
 * Truncates the candidate pattern matrix (K). This function extracts
 * lower triangular portion of the matrix up to the largest
 * "max_nonzeros_row" coefficients in absolute value.
 *
 * Assumptions:
 *    1) columns are ordered with descreasing absolute coef. values
 *    2) max_nonzeros_row < warp_size.
 *
 * TODO:
 *    1) Perform truncation with COO matrix
 *    2) Use less than one warp per row when possible
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_FSAITruncateCandidateOrdered( nalu_hypre_DeviceItem &item,
                                             NALU_HYPRE_Int         max_nonzeros_row,
                                             NALU_HYPRE_Int         num_rows,
                                             NALU_HYPRE_Int        *K_i,
                                             NALU_HYPRE_Int        *K_j,
                                             NALU_HYPRE_Complex    *K_a )
{
   NALU_HYPRE_Int      lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int      p = 0;
   NALU_HYPRE_Int      q = 0;
   NALU_HYPRE_Int      i, j, k, kk, cnt;
   NALU_HYPRE_Int      col;
   nalu_hypre_mask     bitmask;
   NALU_HYPRE_Complex  val;
   NALU_HYPRE_Int      max_lane;
   NALU_HYPRE_Int      max_idx;
   NALU_HYPRE_Complex  max_val;
   NALU_HYPRE_Complex  warp_max_val;

   /* Grid-stride loop over matrix rows */
   for (i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);
        i < num_rows;
        i += nalu_hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      if (lane < 2)
      {
         p = read_only_load(K_i + i + lane);
      }
      q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1, NALU_HYPRE_WARP_SIZE);
      p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0, NALU_HYPRE_WARP_SIZE);

      k = 0;
      while (k < max_nonzeros_row)
      {
         /* Initialize variables */
         j = p + k + lane;
         max_val = 0.0;
         max_idx = -1;

         /* Find maximum val/col pair in each lane */
         if (j < q)
         {
            if (K_j[j] < i)
            {
               max_val = abs(K_a[j]);
               max_idx = j;
            }
         }

         for (j += NALU_HYPRE_WARP_SIZE; j < q; j += NALU_HYPRE_WARP_SIZE)
         {
            if (K_j[j] < i)
            {
               val = abs(K_a[j]);
               if (val > max_val)
               {
                  max_val = val;
                  max_idx = j;
               }
            }
         }

         /* Find maximum coefficient in absolute value in the warp */
         warp_max_val = warp_allreduce_max(item, max_val);

         /* Reorder col/val entries associated with warp_max_val */
         bitmask = nalu_hypre_ballot_sync(NALU_HYPRE_WARP_FULL_MASK, warp_max_val == max_val);
         if (warp_max_val > 0.0)
         {
            cnt = min(nalu_hypre_popc(bitmask), max_nonzeros_row - k);

            for (kk = 0; kk < cnt; kk++)
            {
               /* warp_sync(item); */
               max_lane = nalu_hypre_ffs(bitmask) - 1;
               if (lane == max_lane)
               {
                  col = K_j[p + k + kk];
                  val = K_a[p + k + kk];

                  K_j[p + k + kk] = K_j[max_idx];
                  K_a[p + k + kk] = max_val;

                  K_j[max_idx] = col;
                  K_a[max_idx] = val;
               }

               /* Update bitmask */
               bitmask = nalu_hypre_mask_flip_at(bitmask, max_lane);
            }

            /* Update number of nonzeros per row */
            k += cnt;
         }
         else
         {
            break;
         }
      }

      /* Exclude remaining columns */
      for (j = p + k + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
      {
         K_j[j] = -1;
      }
   }
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_FSAITruncateCandidateUnordered
 *
 * Truncates the candidate pattern matrix (K). This function extracts
 * lower triangular portion of the matrix up to the largest
 * "max_nonzeros_row" coefficients in absolute value.
 *
 * Assumptions:
 *    1) max_nonzeros_row < warp_size.
 *
 * TODO:
 *    1) Use less than one warp per row when possible
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_FSAITruncateCandidateUnordered( nalu_hypre_DeviceItem &item,
                                               NALU_HYPRE_Int         max_nonzeros_row,
                                               NALU_HYPRE_Int         num_rows,
                                               NALU_HYPRE_Int        *K_i,
                                               NALU_HYPRE_Int        *K_e,
                                               NALU_HYPRE_Int        *K_j,
                                               NALU_HYPRE_Complex    *K_a )
{
   NALU_HYPRE_Int      lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int      p = 0;
   NALU_HYPRE_Int      q = 0;
   NALU_HYPRE_Int      ee, e, i, j, k, kk, cnt;
   nalu_hypre_mask     bitmask;
   NALU_HYPRE_Complex  val;
   NALU_HYPRE_Int      max_lane;
   NALU_HYPRE_Int      max_idx;
   NALU_HYPRE_Int      max_col;
   NALU_HYPRE_Int      colK;
   NALU_HYPRE_Complex  valK;
   NALU_HYPRE_Complex  max_val;
   NALU_HYPRE_Complex  warp_max_val;

   /* Grid-stride loop over matrix rows */
   for (i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);
        i < num_rows;
        i += nalu_hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      if (lane < 2)
      {
         p = read_only_load(K_i + i + lane);
      }
      q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1, NALU_HYPRE_WARP_SIZE);
      p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0, NALU_HYPRE_WARP_SIZE);

      k = 0;
      while (k < max_nonzeros_row)
      {
         /* Initialize variables */
         j = p + k + lane;
         max_val = 0.0;
         max_idx = -1;

         /* Find maximum val/col pair in each lane */
         if (j < q)
         {
            if (K_j[j] < i)
            {
               max_val = abs(K_a[j]);
               max_idx = j;
            }
         }

         for (j += NALU_HYPRE_WARP_SIZE; j < q; j += NALU_HYPRE_WARP_SIZE)
         {
            if (K_j[j] < i)
            {
               val = abs(K_a[j]);
               if (val > max_val)
               {
                  max_val = val;
                  max_idx = j;
               }
            }
         }

         /* Find maximum coefficient in absolute value in the warp */
         warp_max_val = warp_allreduce_max(item, max_val);

         /* Reorder col/val entries associated with warp_max_val */
         bitmask = nalu_hypre_ballot_sync(NALU_HYPRE_WARP_FULL_MASK, warp_max_val == max_val);
         if (warp_max_val > 0.0)
         {
            cnt = min(nalu_hypre_popc(bitmask), max_nonzeros_row - k);

            for (kk = 0; kk < cnt; kk++)
            {
               /* warp_sync(item); */
               max_lane = nalu_hypre_ffs(bitmask) - 1;
               if (lane == max_lane)
               {
                  colK = K_j[p + k + kk];
                  valK = K_a[p + k + kk];
                  max_col = K_j[max_idx];

                  if (k + kk == 0)
                  {
                     K_j[p] = max_col;
                     K_a[p] = max_val;

                     K_j[max_idx] = colK;
                     K_a[max_idx] = valK;
                  }
                  else
                  {
                     if (max_col > K_j[p + k + kk - 1])
                     {
                        /* Insert from the right */
                        K_j[p + k + kk] = max_col;
                        K_a[p + k + kk] = max_val;

                        K_j[max_idx] = colK;
                        K_a[max_idx] = valK;
                     }
                     else if (max_col < K_j[p])
                     {
                        /* Insert from the left */
                        for (ee = k + kk; ee > 0; ee--)
                        {
                           K_j[p + ee] = K_j[p + ee - 1];
                           K_a[p + ee] = K_a[p + ee - 1];
                        }

                        K_j[p] = max_col;
                        K_a[p] = max_val;

                        if (max_idx > p + k + kk)
                        {
                           K_j[max_idx] = colK;
                           K_a[max_idx] = valK;
                        }
                     }
                     else
                     {
                        /* Insert in the middle */
                        for (e = k + kk - 1; e >= 0; e--)
                        {
                           if (K_j[p + e] < max_col)
                           {
                              for (ee = k + kk - 1; ee > e; ee--)
                              {
                                 K_j[p + ee + 1] = K_j[p + ee];
                                 K_a[p + ee + 1] = K_a[p + ee];
                              }

                              K_j[p + e + 1] = max_col;
                              K_a[p + e + 1] = max_val;

                              if (max_idx > p + k + kk)
                              {
                                 K_j[max_idx] = colK;
                                 K_a[max_idx] = valK;
                              }

                              break;
                           }
                        }
                     }
                  }
               }

               /* Update bitmask */
               bitmask = nalu_hypre_mask_flip_at(bitmask, max_lane);
            }

            /* Update number of nonzeros per row */
            k += cnt;
         }
         else
         {
            break;
         }
      }

      /* Set pointer to the end of this row */
      if (lane == 0)
      {
         K_e[i] = p + k;
      }
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAIExtractSubSystemsDevice
 *
 * TODO (VPM): This could be a nalu_hypre_CSRMatrix routine
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAIExtractSubSystemsDevice( NALU_HYPRE_Int       num_rows,
                                   NALU_HYPRE_Int       num_nonzeros,
                                   NALU_HYPRE_Int      *A_i,
                                   NALU_HYPRE_Int      *A_j,
                                   NALU_HYPRE_Complex  *A_a,
                                   NALU_HYPRE_Int      *P_i,
                                   NALU_HYPRE_Int      *P_e,
                                   NALU_HYPRE_Int      *P_j,
                                   NALU_HYPRE_Int       ldim,
                                   NALU_HYPRE_Complex  *mat_data,
                                   NALU_HYPRE_Complex  *rhs_data,
                                   NALU_HYPRE_Int      *G_r )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return nalu_hypre_error_flag;
   }

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_rows, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_FSAIExtractSubSystems, gDim, bDim, num_rows,
                     A_i, A_j, A_a, P_i, P_e, P_j, ldim, mat_data, rhs_data, G_r );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAIScalingDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAIScalingDevice( NALU_HYPRE_Int       num_rows,
                         NALU_HYPRE_Int       ldim,
                         NALU_HYPRE_Complex  *sol_data,
                         NALU_HYPRE_Complex  *rhs_data,
                         NALU_HYPRE_Complex  *scaling,
                         NALU_HYPRE_Int      *info )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return nalu_hypre_error_flag;
   }

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_FSAIScaling, gDim, bDim,
                     num_rows, ldim, sol_data, rhs_data, scaling, info );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAIGatherEntriesDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAIGatherEntriesDevice( NALU_HYPRE_Int       num_rows,
                               NALU_HYPRE_Int       ldim,
                               NALU_HYPRE_Complex  *sol_data,
                               NALU_HYPRE_Complex  *scaling,
                               NALU_HYPRE_Int      *K_i,
                               NALU_HYPRE_Int      *K_e,
                               NALU_HYPRE_Int      *K_j,
                               NALU_HYPRE_Int      *G_i,
                               NALU_HYPRE_Int      *G_j,
                               NALU_HYPRE_Complex  *G_a )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return nalu_hypre_error_flag;
   }

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_FSAIGatherEntries, gDim, bDim,
                     num_rows, ldim, sol_data, scaling, K_i, K_e, K_j, G_i, G_j, G_a );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAITruncateCandidateDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAITruncateCandidateDevice( nalu_hypre_CSRMatrix *matrix,
                                   NALU_HYPRE_Int      **matrix_e,
                                   NALU_HYPRE_Int        max_nonzeros_row )
{
   NALU_HYPRE_Int      num_rows  = nalu_hypre_CSRMatrixNumRows(matrix);
   NALU_HYPRE_Int     *mat_i     = nalu_hypre_CSRMatrixI(matrix);
   NALU_HYPRE_Int     *mat_j     = nalu_hypre_CSRMatrixJ(matrix);
   NALU_HYPRE_Complex *mat_a     = nalu_hypre_CSRMatrixData(matrix);

   NALU_HYPRE_Int     *mat_e;

   /* Sanity check */
   if (num_rows <= 0)
   {
      *matrix_e = NULL;
      return nalu_hypre_error_flag;
   }

   /*-----------------------------------------------------
    * Keep only the largest coefficients in absolute value
    *-----------------------------------------------------*/

   /* Allocate memory for row indices array */
   nalu_hypre_GpuProfilingPushRange("Storage1");
   mat_e = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_GpuProfilingPopRange();

   /* Mark unwanted entries with -1 */
   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_rows, "warp", bDim);

   nalu_hypre_GpuProfilingPushRange("TruncCand");
   NALU_HYPRE_GPU_LAUNCH(hypreGPUKernel_FSAITruncateCandidateUnordered, gDim, bDim,
                    max_nonzeros_row, num_rows, mat_i, mat_e, mat_j, mat_a );
   nalu_hypre_GpuProfilingPopRange();

   *matrix_e = mat_e;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAISetupStaticPowerDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAISetupStaticPowerDevice( void               *fsai_vdata,
                                  nalu_hypre_ParCSRMatrix *A,
                                  nalu_hypre_ParVector    *f,
                                  nalu_hypre_ParVector    *u )
{
   nalu_hypre_ParFSAIData      *fsai_data        = (nalu_hypre_ParFSAIData*) fsai_vdata;
   nalu_hypre_ParCSRMatrix     *G                = nalu_hypre_ParFSAIDataGmat(fsai_data);
   nalu_hypre_CSRMatrix        *G_diag           = nalu_hypre_ParCSRMatrixDiag(G);
   NALU_HYPRE_Int               local_solve_type = nalu_hypre_ParFSAIDataLocalSolveType(fsai_data);
   NALU_HYPRE_Int               max_nnz_row      = nalu_hypre_ParFSAIDataMaxNnzRow(fsai_data);
   NALU_HYPRE_Int               num_levels       = nalu_hypre_ParFSAIDataNumLevels(fsai_data);
   NALU_HYPRE_Real              threshold        = nalu_hypre_ParFSAIDataThreshold(fsai_data);

   nalu_hypre_CSRMatrix        *A_diag           = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int               num_rows         = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int               block_size       = max_nnz_row * max_nnz_row;
   NALU_HYPRE_Int               num_nonzeros_G;

   nalu_hypre_ParCSRMatrix     *Atilde;
   nalu_hypre_ParCSRMatrix     *B;
   nalu_hypre_ParCSRMatrix     *Ktilde;
   nalu_hypre_CSRMatrix        *K_diag;
   NALU_HYPRE_Int              *K_e = NULL;
   NALU_HYPRE_Int               i;

   /* Local linear solve data */
#if defined (NALU_HYPRE_USING_MAGMA)
    magma_queue_t          queue     = nalu_hypre_HandleMagmaQueue(nalu_hypre_handle());
#endif

#if defined (NALU_HYPRE_USING_CUSOLVER) || defined (NALU_HYPRE_USING_ROCSOLVER)
    vendorSolverHandle_t   vs_handle = nalu_hypre_HandleVendorSolverHandle(nalu_hypre_handle());
#endif

   /* TODO: Move to fsai_data? */
   NALU_HYPRE_Complex          *scaling;
   NALU_HYPRE_Int              *info;
   NALU_HYPRE_Int              *h_info;

   /* Error code array for FSAI */
   info   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_DEVICE);
   h_info = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------
    *  Sanity checks
    *-----------------------------------------------------*/

   /* Check local linear solve algorithm */
   if (local_solve_type == 1)
   {
#if !(defined (NALU_HYPRE_USING_CUSOLVER) || defined(NALU_HYPRE_USING_ROCSOLVER))
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "local_solve_type == 1 requires cuSOLVER (CUDA) or rocSOLVER (HIP)\n");
      return nalu_hypre_error_flag;
#endif
   }
   else if (local_solve_type == 2)
   {
#if !defined (NALU_HYPRE_USING_MAGMA)
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "local_solve_type == 2 requires MAGMA\n");
      return nalu_hypre_error_flag;
#endif
   }
   else
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Unknown local linear solve type!\n");
      return nalu_hypre_error_flag;
   }

   /*-----------------------------------------------------
    *  Compute candidate pattern
    *-----------------------------------------------------*/

   nalu_hypre_GpuProfilingPushRange("CandPat");

   /* Compute filtered version of A */
   Atilde = nalu_hypre_ParCSRMatrixClone(A, 1);

   /* Pre-filter to reduce SpGEMM cost */
   if (num_levels > 1)
   {
      nalu_hypre_ParCSRMatrixDropSmallEntriesDevice(Atilde, threshold, 2);
   }

   /* TODO: Check if Atilde is diagonal */

   /* Compute power pattern */
   switch (num_levels)
   {
      case 1:
         Ktilde = Atilde;
         break;

      case 2:
         Ktilde = nalu_hypre_ParCSRMatMatDevice(Atilde, Atilde);
         break;

      case 3:
         /* First pass */
         B = nalu_hypre_ParCSRMatMatDevice(Atilde, Atilde);

         /* Second pass */
         Ktilde = nalu_hypre_ParCSRMatMatDevice(Atilde, B);
         nalu_hypre_ParCSRMatrixDestroy(B);
         break;

      case 4:
         /* First pass */
         B = nalu_hypre_ParCSRMatMatDevice(Atilde, Atilde);
         nalu_hypre_ParCSRMatrixDropSmallEntriesDevice(B, threshold, 2);

         /* Second pass */
         Ktilde = nalu_hypre_ParCSRMatMatDevice(B, B);
         nalu_hypre_ParCSRMatrixDestroy(B);
         break;

      default:
         Ktilde = nalu_hypre_ParCSRMatrixClone(Atilde, 1);
         for (i = 1; i < num_levels; i++)
         {
            /* Compute temporary matrix */
            B = nalu_hypre_ParCSRMatMatDevice(Atilde, Ktilde);

            /* Update resulting matrix */
            nalu_hypre_ParCSRMatrixDestroy(Ktilde);
            Ktilde = nalu_hypre_ParCSRMatrixClone(B, 1);
         }
   }

   nalu_hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Filter candidate pattern
    *-----------------------------------------------------*/

   nalu_hypre_GpuProfilingPushRange("FilterPat");

#if defined (DEBUG_FSAI)
   {
      nalu_hypre_ParCSRMatrixPrintIJ(Ktilde, 0, 0, "FSAI.out.H.ij");
   }
#endif

   /* Set pattern matrix diagonal matrix */
   K_diag = nalu_hypre_ParCSRMatrixDiag(Ktilde);

   /* Filter candidate pattern */
   nalu_hypre_FSAITruncateCandidateDevice(K_diag, &K_e, max_nnz_row);

#if defined (DEBUG_FSAI)
   {
      nalu_hypre_ParCSRMatrixPrintIJ(Ktilde, 0, 0, "FSAI.out.K.ij");
   }
#endif

   nalu_hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Preprocess input matrix
    *-----------------------------------------------------*/

   nalu_hypre_GpuProfilingPushRange("PreProcessA");

   /* TODO: implement faster diagonal extraction (use "i == A_j[A_i[i]]")*/
   scaling = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_rows, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_CSRMatrixExtractDiagonalDevice(A_diag, scaling, 0);

   nalu_hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Extract local linear systems
    *-----------------------------------------------------*/

   /* Allocate storage */
   nalu_hypre_GpuProfilingPushRange("Storage1");
   NALU_HYPRE_Complex  *mat_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,
                                            block_size * num_rows,
                                            NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex  *rhs_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, max_nnz_row * num_rows,
                                            NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex  *sol_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, max_nnz_row * num_rows,
                                            NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_GpuProfilingPopRange();

   /* Gather dense linear subsystems */
   nalu_hypre_GpuProfilingPushRange("ExtractLS");
   nalu_hypre_FSAIExtractSubSystemsDevice(num_rows,
                                     nalu_hypre_CSRMatrixNumNonzeros(A_diag),
                                     nalu_hypre_CSRMatrixI(A_diag),
                                     nalu_hypre_CSRMatrixJ(A_diag),
                                     nalu_hypre_CSRMatrixData(A_diag),
                                     nalu_hypre_CSRMatrixI(K_diag),
                                     K_e,
                                     nalu_hypre_CSRMatrixJ(K_diag),
                                     max_nnz_row,
                                     mat_data,
                                     rhs_data,
                                     nalu_hypre_CSRMatrixI(G_diag) + 1);
   nalu_hypre_GpuProfilingPopRange();

   /* Copy rhs to solution vector */
   nalu_hypre_GpuProfilingPushRange("CopyRHS");
   nalu_hypre_TMemcpy(sol_data, rhs_data, NALU_HYPRE_Complex, max_nnz_row * num_rows,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_GpuProfilingPopRange();

   /* Build array of pointers */
   nalu_hypre_GpuProfilingPushRange("Storage2");
   NALU_HYPRE_Complex **sol_aop = nalu_hypre_TAlloc(NALU_HYPRE_Complex *, num_rows, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex **mat_aop = nalu_hypre_TAlloc(NALU_HYPRE_Complex *, num_rows, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_GpuProfilingPopRange();

   nalu_hypre_GpuProfilingPushRange("FormAOP");
   hypreDevice_ComplexArrayToArrayOfPtrs(num_rows, block_size, mat_data, mat_aop);
   hypreDevice_ComplexArrayToArrayOfPtrs(num_rows, max_nnz_row, sol_data, sol_aop);
   nalu_hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Solve local linear systems
    *-----------------------------------------------------*/

   nalu_hypre_GpuProfilingPushRange("BatchedSolve");
   if (num_rows)
   {
      nalu_hypre_GpuProfilingPushRange("Factorization");

      if (local_solve_type == 1)
      {
#if defined (NALU_HYPRE_USING_CUSOLVER)
         NALU_HYPRE_CUSOLVER_CALL(cusolverDnDpotrfBatched(vs_handle,
                                                     CUBLAS_FILL_MODE_LOWER,
                                                     max_nnz_row,
                                                     mat_aop,
                                                     max_nnz_row,
                                                     info,
                                                     num_rows));

#elif defined (NALU_HYPRE_USING_ROCSOLVER)
         NALU_HYPRE_ROCSOLVER_CALL(rocsolver_dpotrf_batched(vs_handle,
                                                       rocblas_fill_lower,
                                                       max_nnz_row,
                                                       mat_aop,
                                                       max_nnz_row,
                                                       info,
                                                       num_rows));
#endif
      }
      else if (local_solve_type == 2)
      {
#if defined (NALU_HYPRE_USING_MAGMA)
         NALU_HYPRE_MAGMA_CALL(magma_dpotrf_batched(MagmaLower,
                                               max_nnz_row,
                                               mat_aop,
                                               max_nnz_row,
                                               info,
                                               num_rows,
                                               queue));
#endif
      }
      nalu_hypre_GpuProfilingPopRange(); /* Factorization */

#if defined (NALU_HYPRE_DEBUG)
      nalu_hypre_TMemcpy(h_info, info, NALU_HYPRE_Int, num_rows,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      for (NALU_HYPRE_Int k = 0; k < num_rows; k++)
      {
         if (h_info[k] != 0)
         {
            nalu_hypre_printf("Cholesky factorization failed at system #%d, subrow %d\n",
                         k, h_info[k]);
         }
      }
#endif

      nalu_hypre_GpuProfilingPushRange("Solve");

      if (local_solve_type == 1)
      {
#if defined (NALU_HYPRE_USING_CUSOLVER)
         NALU_HYPRE_CUSOLVER_CALL(cusolverDnDpotrsBatched(vs_handle,
                                                     CUBLAS_FILL_MODE_LOWER,
                                                     max_nnz_row,
                                                     1,
                                                     mat_aop,
                                                     max_nnz_row,
                                                     sol_aop,
                                                     max_nnz_row,
                                                     info,
                                                     num_rows));
#elif defined (NALU_HYPRE_USING_ROCSOLVER)
         NALU_HYPRE_ROCSOLVER_CALL(rocsolver_dpotrs_batched(vs_handle,
                                                       rocblas_fill_lower,
                                                       max_nnz_row,
                                                       1,
                                                       mat_aop,
                                                       max_nnz_row,
                                                       sol_aop,
                                                       max_nnz_row,
                                                       num_rows));
#endif
      }
      else if (local_solve_type == 2)
      {
#if defined (NALU_HYPRE_USING_MAGMA)
         NALU_HYPRE_MAGMA_CALL(magma_dpotrs_batched(MagmaLower,
                                               max_nnz_row,
                                               1,
                                               mat_aop,
                                               max_nnz_row,
                                               sol_aop,
                                               max_nnz_row,
                                               num_rows,
                                               queue));
#endif
      }
      nalu_hypre_GpuProfilingPopRange(); /* Solve */

#if defined (NALU_HYPRE_DEBUG)
      nalu_hypre_TMemcpy(h_info, info, NALU_HYPRE_Int, num_rows,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      for (NALU_HYPRE_Int k = 0; k < num_rows; k++)
      {
         if (h_info[k] != 0)
         {
            nalu_hypre_printf("Cholesky solution failed at system #%d with code %d\n",
                         k, h_info[k]);
         }
      }
#endif
   }
   nalu_hypre_GpuProfilingPopRange(); /* BatchedSolve */

   /*-----------------------------------------------------
    *  Finalize construction of the triangular factor
    *-----------------------------------------------------*/

   nalu_hypre_GpuProfilingPushRange("BuildFSAI");

   /* Update scaling factor */
   nalu_hypre_FSAIScalingDevice(num_rows, max_nnz_row, sol_data, rhs_data, scaling, info);

   /* Compute the row pointer G_i */
   hypreDevice_IntegerInclusiveScan(num_rows + 1, nalu_hypre_CSRMatrixI(G_diag));

   /* Get the actual number of nonzero coefficients of G_diag */
   nalu_hypre_TMemcpy(&num_nonzeros_G, nalu_hypre_CSRMatrixI(G_diag) + num_rows,
                 NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

   /* Update the nonzero count of matrix G */
   nalu_hypre_CSRMatrixNumNonzeros(G_diag) = num_nonzeros_G;

   /* Set column indices and coefficients of G */
   nalu_hypre_FSAIGatherEntriesDevice(num_rows,
                                 max_nnz_row,
                                 sol_data,
                                 scaling,
                                 nalu_hypre_CSRMatrixI(K_diag),
                                 K_e,
                                 nalu_hypre_CSRMatrixJ(K_diag),
                                 nalu_hypre_CSRMatrixI(G_diag),
                                 nalu_hypre_CSRMatrixJ(G_diag),
                                 nalu_hypre_CSRMatrixData(G_diag));

   nalu_hypre_GpuProfilingPopRange();
   /* TODO: Reallocate memory for G_j/G_a? */

   /*-----------------------------------------------------
    *  Free memory
    *-----------------------------------------------------*/

   nalu_hypre_ParCSRMatrixDestroy(Ktilde);
   if (num_levels > 1)
   {
      nalu_hypre_ParCSRMatrixDestroy(Atilde);
   }

   /* TODO: can we free some of these earlier? */
   nalu_hypre_TFree(K_e, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(rhs_data, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(sol_data, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(mat_data, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(sol_aop, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(mat_aop, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(scaling, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(info, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(h_info, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

#endif /* if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */
#if defined(NALU_HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAISetupDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAISetupDevice( void               *fsai_vdata,
                       nalu_hypre_ParCSRMatrix *A,
                       nalu_hypre_ParVector    *f,
                       nalu_hypre_ParVector    *u )
{
   nalu_hypre_ParFSAIData       *fsai_data     = (nalu_hypre_ParFSAIData*) fsai_vdata;
   NALU_HYPRE_Int                algo_type     = nalu_hypre_ParFSAIDataAlgoType(fsai_data);
   nalu_hypre_ParCSRMatrix      *G             = nalu_hypre_ParFSAIDataGmat(fsai_data);
   nalu_hypre_ParCSRMatrix      *h_A;

   nalu_hypre_GpuProfilingPushRange("FSAISetup");

   if (algo_type == 1 || algo_type == 2)
   {
      /* Initialize matrix G on host */
      nalu_hypre_ParCSRMatrixInitialize_v2(G, NALU_HYPRE_MEMORY_HOST);

      /* Clone input matrix on host */
      h_A = nalu_hypre_ParCSRMatrixClone_v2(A, 1, NALU_HYPRE_MEMORY_HOST);

      /* Compute FSAI factor on host */
      switch (algo_type)
      {
         case 2:
            nalu_hypre_FSAISetupOMPDyn(fsai_vdata, h_A, f, u);
            break;

         default:
            nalu_hypre_FSAISetupNative(fsai_vdata, h_A, f, u);
            break;
      }

      /* Move FSAI factor G to device */
      nalu_hypre_ParCSRMatrixMigrate(G, NALU_HYPRE_MEMORY_DEVICE);

      /* Destroy temporary data on host */
      NALU_HYPRE_ParCSRMatrixDestroy(h_A);
   }
   else
   {
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      /* Initialize matrix G on device */
      nalu_hypre_ParCSRMatrixInitialize_v2(G, NALU_HYPRE_MEMORY_DEVICE);

      if (algo_type == 3)
      {
         nalu_hypre_FSAISetupStaticPowerDevice(fsai_vdata, A, f, u);
      }
#else
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Device FSAI not implemented for SYCL!\n");
#endif
   }

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

#endif /* if defined(NALU_HYPRE_USING_GPU) */
