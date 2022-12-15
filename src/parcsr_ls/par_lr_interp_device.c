/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_ls.h"
#include "aux_interp.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)

#define MAX_C_CONNECTIONS 100
#define HAVE_COMMON_C 1

//-----------------------------------------------------------------------
// S_*_j is the special j-array from device SoC
// -1: weak, -2: diag, >=0 (== A_diag_j) : strong
// add weak and the diagonal entries of F-rows
__global__
void hypreGPUKernel_compute_weak_rowsums( nalu_hypre_DeviceItem    &item,
                                          NALU_HYPRE_Int      nr_of_rows,
                                          bool           has_offd,
                                          NALU_HYPRE_Int     *CF_marker,
                                          NALU_HYPRE_Int     *A_diag_i,
                                          NALU_HYPRE_Complex *A_diag_a,
                                          NALU_HYPRE_Int     *Soc_diag_j,
                                          NALU_HYPRE_Int     *A_offd_i,
                                          NALU_HYPRE_Complex *A_offd_a,
                                          NALU_HYPRE_Int     *Soc_offd_j,
                                          NALU_HYPRE_Real    *rs,
                                          NALU_HYPRE_Int      flag)
{
   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int ib = 0, ie;

   if (lane == 0)
   {
      ib = read_only_load(CF_marker + row);
   }
   ib = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib, 0);

   if (ib >= flag)
   {
      return;
   }

   if (lane < 2)
   {
      ib = read_only_load(A_diag_i + row + lane);
   }
   ie = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib, 1);
   ib = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib, 0);

   NALU_HYPRE_Complex rl = 0.0;

   for (NALU_HYPRE_Int i = ib + lane; i < ie; i += NALU_HYPRE_WARP_SIZE)
   {
      rl += read_only_load(&A_diag_a[i]) * (read_only_load(&Soc_diag_j[i]) < 0);
   }

   if (has_offd)
   {
      if (lane < 2)
      {
         ib = read_only_load(A_offd_i + row + lane);
      }
      ie = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib, 1);
      ib = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib, 0);

      for (NALU_HYPRE_Int i = ib + lane; i < ie; i += NALU_HYPRE_WARP_SIZE)
      {
         rl += read_only_load(&A_offd_a[i]) * (read_only_load(&Soc_offd_j[i]) < 0);
      }
   }

   rl = warp_reduce_sum(item, rl);

   if (lane == 0)
   {
      rs[row] = rl;
   }
}

//-----------------------------------------------------------------------
__global__
void hypreGPUKernel_compute_aff_afc( nalu_hypre_DeviceItem    &item,
                                     NALU_HYPRE_Int      nr_of_rows,
                                     NALU_HYPRE_Int     *AFF_diag_i,
                                     NALU_HYPRE_Int     *AFF_diag_j,
                                     NALU_HYPRE_Complex *AFF_diag_data,
                                     NALU_HYPRE_Int     *AFF_offd_i,
                                     NALU_HYPRE_Complex *AFF_offd_data,
                                     NALU_HYPRE_Int     *AFC_diag_i,
                                     NALU_HYPRE_Complex *AFC_diag_data,
                                     NALU_HYPRE_Int     *AFC_offd_i,
                                     NALU_HYPRE_Complex *AFC_offd_data,
                                     NALU_HYPRE_Complex *rsW,
                                     NALU_HYPRE_Complex *rsFC )
{
   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0, q;

   NALU_HYPRE_Complex iscale = 0.0, beta = 0.0;

   if (lane == 0)
   {
      iscale = -1.0 / read_only_load(&rsW[row]);
      beta = read_only_load(&rsFC[row]);
   }
   iscale = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, iscale, 0);
   beta   = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, beta,   0);

   // AFF
   /* Diag part */
   if (lane < 2)
   {
      p = read_only_load(AFF_diag_i + row + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   // do not assume diag is the first element of row
   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      if (read_only_load(&AFF_diag_j[j]) == row)
      {
         AFF_diag_data[j] = beta * iscale;
      }
      else
      {
         AFF_diag_data[j] *= iscale;
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p = read_only_load(AFF_offd_i + row + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      AFF_offd_data[j] *= iscale;
   }

   if (beta != 0.0)
   {
      beta = 1.0 / beta;
   }

   // AFC
   if (lane < 2)
   {
      p = read_only_load(AFC_diag_i + row + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   /* Diag part */
   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      AFC_diag_data[j] *= beta;
   }

   /* offd part */
   if (lane < 2)
   {
      p = read_only_load(AFC_offd_i + row + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      AFC_offd_data[j] *= beta;
   }
}


//-----------------------------------------------------------------------
void
hypreDevice_extendWtoP( NALU_HYPRE_Int      P_nr_of_rows,
                        NALU_HYPRE_Int      W_nr_of_rows,
                        NALU_HYPRE_Int      W_nr_of_cols,
                        NALU_HYPRE_Int     *CF_marker,
                        NALU_HYPRE_Int      W_diag_nnz,
                        NALU_HYPRE_Int     *W_diag_i,
                        NALU_HYPRE_Int     *W_diag_j,
                        NALU_HYPRE_Complex *W_diag_data,
                        NALU_HYPRE_Int     *P_diag_i,
                        NALU_HYPRE_Int     *P_diag_j,
                        NALU_HYPRE_Complex *P_diag_data,
                        NALU_HYPRE_Int     *W_offd_i,
                        NALU_HYPRE_Int     *P_offd_i )
{
   // row index shift P --> W
   NALU_HYPRE_Int *PWoffset = nalu_hypre_TAlloc(NALU_HYPRE_Int, P_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      CF_marker,
                      &CF_marker[P_nr_of_rows],
                      PWoffset,
                      is_nonnegative<NALU_HYPRE_Int>() );
#else
   NALU_HYPRE_THRUST_CALL( transform,
                      CF_marker,
                      &CF_marker[P_nr_of_rows],
                      PWoffset,
                      is_nonnegative<NALU_HYPRE_Int>() );
#endif

   nalu_hypre_Memset(PWoffset + P_nr_of_rows, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_IntegerExclusiveScan(P_nr_of_rows + 1, PWoffset);

   // map F+C to (next) F
   NALU_HYPRE_Int *map2F = nalu_hypre_TAlloc(NALU_HYPRE_Int, P_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(P_nr_of_rows + 1),
                      PWoffset,
                      map2F,
                      std::minus<NALU_HYPRE_Int>() );
#else
   NALU_HYPRE_THRUST_CALL( transform,
                      thrust::counting_iterator<NALU_HYPRE_Int>(0),
                      thrust::counting_iterator<NALU_HYPRE_Int>(P_nr_of_rows + 1),
                      PWoffset,
                      map2F,
                      thrust::minus<NALU_HYPRE_Int>() );
#endif

   // P_diag_i
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( map2F,
                     map2F + P_nr_of_rows + 1,
                     W_diag_i,
                     P_diag_i );

   hypreDevice_IntAxpyn( P_diag_i, P_nr_of_rows + 1, PWoffset, P_diag_i, 1 );

   // P_offd_i
   if (W_offd_i && P_offd_i)
   {
      hypreSycl_gather( map2F,
                        map2F + P_nr_of_rows + 1,
                        W_offd_i,
                        P_offd_i );
   }
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      map2F,
                      map2F + P_nr_of_rows + 1,
                      W_diag_i,
                      P_diag_i );

   hypreDevice_IntAxpyn( P_diag_i, P_nr_of_rows + 1, PWoffset, P_diag_i, 1 );

   // P_offd_i
   if (W_offd_i && P_offd_i)
   {
      NALU_HYPRE_THRUST_CALL( gather,
                         map2F,
                         map2F + P_nr_of_rows + 1,
                         W_offd_i,
                         P_offd_i );
   }
#endif

   nalu_hypre_TFree(map2F, NALU_HYPRE_MEMORY_DEVICE);

   // row index shift W --> P
   NALU_HYPRE_Int *WPoffset = nalu_hypre_TAlloc(NALU_HYPRE_Int, W_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int *new_end = hypreSycl_copy_if( PWoffset,
                                           PWoffset + P_nr_of_rows,
                                           CF_marker,
                                           WPoffset,
                                           is_negative<NALU_HYPRE_Int>() );
#else
   NALU_HYPRE_Int *new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                           PWoffset,
                                           PWoffset + P_nr_of_rows,
                                           CF_marker,
                                           WPoffset,
                                           is_negative<NALU_HYPRE_Int>() );
#endif
   nalu_hypre_assert(new_end - WPoffset == W_nr_of_rows);

   nalu_hypre_TFree(PWoffset, NALU_HYPRE_MEMORY_DEVICE);

   // elements shift
   NALU_HYPRE_Int *shift = hypreDevice_CsrRowPtrsToIndices(W_nr_of_rows, W_diag_nnz, W_diag_i);
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( shift,
                     shift + W_diag_nnz,
                     WPoffset,
                     shift);
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      shift,
                      shift + W_diag_nnz,
                      WPoffset,
                      shift);
#endif

   nalu_hypre_TFree(WPoffset, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      shift,
                      shift + W_diag_nnz,
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(0),
                      shift,
                      std::plus<NALU_HYPRE_Int>() );

   // P_diag_j and P_diag_data
   if (W_diag_j && W_diag_data)
   {
      hypreSycl_scatter( oneapi::dpl::make_zip_iterator(W_diag_j, W_diag_data),
                         oneapi::dpl::make_zip_iterator(W_diag_j, W_diag_data) + W_diag_nnz,
                         shift,
                         oneapi::dpl::make_zip_iterator(P_diag_j, P_diag_data) );
   }
#else
   NALU_HYPRE_THRUST_CALL( transform,
                      shift,
                      shift + W_diag_nnz,
                      thrust::counting_iterator<NALU_HYPRE_Int>(0),
                      shift,
                      thrust::plus<NALU_HYPRE_Int>() );

   // P_diag_j and P_diag_data
   if (W_diag_j && W_diag_data)
   {
      NALU_HYPRE_THRUST_CALL( scatter,
                         thrust::make_zip_iterator(thrust::make_tuple(W_diag_j, W_diag_data)),
                         thrust::make_zip_iterator(thrust::make_tuple(W_diag_j, W_diag_data)) + W_diag_nnz,
                         shift,
                         thrust::make_zip_iterator(thrust::make_tuple(P_diag_j, P_diag_data)) );
   }
#endif
   nalu_hypre_TFree(shift, NALU_HYPRE_MEMORY_DEVICE);

   // fill the gap
   NALU_HYPRE_Int *PC_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, W_nr_of_cols, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   new_end = hypreSycl_copy_if( P_diag_i,
                                P_diag_i + P_nr_of_rows,
                                CF_marker,
                                PC_i,
                                is_nonnegative<NALU_HYPRE_Int>() );
#else
   new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                P_diag_i,
                                P_diag_i + P_nr_of_rows,
                                CF_marker,
                                PC_i,
                                is_nonnegative<NALU_HYPRE_Int>() );
#endif

   nalu_hypre_assert(new_end - PC_i == W_nr_of_cols);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( copy,
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(W_nr_of_cols),
                      oneapi::dpl::make_permutation_iterator(P_diag_j, PC_i) );
#else
   NALU_HYPRE_THRUST_CALL( scatter,
                      thrust::counting_iterator<NALU_HYPRE_Int>(0),
                      thrust::counting_iterator<NALU_HYPRE_Int>(W_nr_of_cols),
                      PC_i,
                      P_diag_j );
#endif

   hypreDevice_ScatterConstant(P_diag_data, W_nr_of_cols, PC_i, (NALU_HYPRE_Complex) 1.0);

   nalu_hypre_TFree(PC_i, NALU_HYPRE_MEMORY_DEVICE);
}

//-----------------------------------------------------------------------
// For Ext+i Interp, scale AFF from the left and the right
__global__
void hypreGPUKernel_compute_twiaff_w( nalu_hypre_DeviceItem    &item,
                                      NALU_HYPRE_Int      nr_of_rows,
                                      NALU_HYPRE_BigInt   first_index,
                                      NALU_HYPRE_Int     *AFF_diag_i,
                                      NALU_HYPRE_Int     *AFF_diag_j,
                                      NALU_HYPRE_Complex *AFF_diag_data,
                                      NALU_HYPRE_Complex *AFF_diag_data_old,
                                      NALU_HYPRE_Int     *AFF_offd_i,
                                      NALU_HYPRE_Int     *AFF_offd_j,
                                      NALU_HYPRE_Complex *AFF_offd_data,
                                      NALU_HYPRE_Int     *AFF_ext_i,
                                      NALU_HYPRE_BigInt  *AFF_ext_j,
                                      NALU_HYPRE_Complex *AFF_ext_data,
                                      NALU_HYPRE_Complex *rsW,
                                      NALU_HYPRE_Complex *rsFC,
                                      NALU_HYPRE_Complex *rsFC_offd )
{
   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);

   NALU_HYPRE_Int ib_diag = 0, ie_diag, ib_offd = 0, ie_offd;

   // diag
   if (lane < 2)
   {
      ib_diag = read_only_load(AFF_diag_i + row + lane);
   }
   ie_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_diag, 1);
   ib_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_diag, 0);

   NALU_HYPRE_Complex theta_i = 0.0;

   // do not assume diag is the first element of row
   // entire warp works on each j
   for (NALU_HYPRE_Int indj = ib_diag; indj < ie_diag; indj++)
   {
      NALU_HYPRE_Int j = 0;

      if (lane == 0)
      {
         j = read_only_load(&AFF_diag_j[indj]);
      }
      j = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 0);

      if (j == row)
      {
         if (lane == 0)
         {
            AFF_diag_data[indj] = 1.0;
         }

         continue;
      }

      NALU_HYPRE_Int kb = 0, ke;

      // find if there exists entry (j, row) in row j of diag
      if (lane < 2)
      {
         kb = read_only_load(AFF_diag_i + j + lane);
      }
      ke = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, kb, 1);
      kb = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, kb, 0);

      NALU_HYPRE_Int kmatch = -1;
      for (NALU_HYPRE_Int indk = kb + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, indk < ke);
           indk += NALU_HYPRE_WARP_SIZE)
      {
         if (indk < ke && row == read_only_load(&AFF_diag_j[indk]))
         {
            kmatch = indk;
         }

         if (warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, kmatch >= 0))
         {
            break;
         }
      }
      kmatch = warp_reduce_max(item, kmatch);

      if (lane == 0)
      {
         NALU_HYPRE_Complex vji = kmatch >= 0 ? read_only_load(&AFF_diag_data_old[kmatch]) : 0.0;
         NALU_HYPRE_Complex rsj = read_only_load(&rsFC[j]) + vji;
         if (rsj)
         {
            NALU_HYPRE_Complex vij = read_only_load(&AFF_diag_data_old[indj]) / rsj;
            AFF_diag_data[indj] = vij;
            theta_i += vji * vij;
         }
         else
         {
            AFF_diag_data[indj] = 0.0;
            theta_i += read_only_load(&AFF_diag_data_old[indj]);
         }
      }
   }

   // offd
   if (lane < 2)
   {
      ib_offd = read_only_load(AFF_offd_i + row + lane);
   }
   ie_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_offd, 1);
   ib_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_offd, 0);

   for (NALU_HYPRE_Int indj = ib_offd; indj < ie_offd; indj++)
   {
      NALU_HYPRE_Int j = 0;

      if (lane == 0)
      {
         j = read_only_load(&AFF_offd_j[indj]);
      }
      j = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 0);

      NALU_HYPRE_Int kb = 0, ke;

      if (lane < 2)
      {
         kb = read_only_load(AFF_ext_i + j + lane);
      }
      ke = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, kb, 1);
      kb = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, kb, 0);

      NALU_HYPRE_Int kmatch = -1;
      for (NALU_HYPRE_Int indk = kb + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, indk < ke);
           indk += NALU_HYPRE_WARP_SIZE)
      {
         if (indk < ke && row + first_index == read_only_load(&AFF_ext_j[indk]))
         {
            kmatch = indk;
         }

         if (warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, kmatch >= 0))
         {
            break;
         }
      }
      kmatch = warp_reduce_max(item, kmatch);

      if (lane == 0)
      {
         NALU_HYPRE_Complex vji = kmatch >= 0 ? read_only_load(&AFF_ext_data[kmatch]) : 0.0;
         NALU_HYPRE_Complex rsj = read_only_load(&rsFC_offd[j]) + vji;
         if (rsj)
         {
            NALU_HYPRE_Complex vij = read_only_load(&AFF_offd_data[indj]) / rsj;
            AFF_offd_data[indj] = vij;
            theta_i += vji * vij;
         }
         else
         {
            AFF_offd_data[indj] = 0.0;
            theta_i += read_only_load(&AFF_offd_data[indj]);
         }
      }
   }

   // scale row
   if (lane == 0)
   {
      theta_i += read_only_load(rsW + row);
      theta_i = theta_i ? -1.0 / theta_i : -1.0;
   }
   theta_i = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, theta_i, 0);

   for (NALU_HYPRE_Int j = ib_diag + lane; j < ie_diag; j += NALU_HYPRE_WARP_SIZE)
   {
      AFF_diag_data[j] *= theta_i;
   }

   for (NALU_HYPRE_Int j = ib_offd + lane; j < ie_offd; j += NALU_HYPRE_WARP_SIZE)
   {
      AFF_offd_data[j] *= theta_i;
   }
}


//-----------------------------------------------------------------------
__global__
void hypreGPUKernel_compute_aff_afc_epe( nalu_hypre_DeviceItem    &item,
                                         NALU_HYPRE_Int      nr_of_rows,
                                         NALU_HYPRE_Int     *AFF_diag_i,
                                         NALU_HYPRE_Int     *AFF_diag_j,
                                         NALU_HYPRE_Complex *AFF_diag_data,
                                         NALU_HYPRE_Int     *AFF_offd_i,
                                         NALU_HYPRE_Int     *AFF_offd_j,
                                         NALU_HYPRE_Complex *AFF_offd_data,
                                         NALU_HYPRE_Int     *AFC_diag_i,
                                         NALU_HYPRE_Complex *AFC_diag_data,
                                         NALU_HYPRE_Int     *AFC_offd_i,
                                         NALU_HYPRE_Complex *AFC_offd_data,
                                         NALU_HYPRE_Complex *rsW,
                                         NALU_HYPRE_Complex *dlam,
                                         NALU_HYPRE_Complex *dtmp,
                                         NALU_HYPRE_Complex *dtmp_offd )
{
   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int pd = 0, qd, po = 0, qo, xd = 0, yd, xo = 0, yo;

   NALU_HYPRE_Complex theta = 0.0, value = 0.0;
   NALU_HYPRE_Complex dtau_i = 0.0;

   if (lane < 2)
   {
      pd = read_only_load(AFF_diag_i + row + lane);
      po = read_only_load(AFF_offd_i + row + lane);
      xd = read_only_load(AFC_diag_i + row + lane);
      xo = read_only_load(AFC_offd_i + row + lane);
   }

   qd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pd, 1);
   pd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pd, 0);
   qo = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, po, 1);
   po = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, po, 0);
   yd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, xd, 1);
   xd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, xd, 0);
   yo = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, xo, 1);
   xo = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, xo, 0);

   /* D_\tau */
   /* do not assume the first element is the diagonal */
   for (NALU_HYPRE_Int j = pd + lane; j < qd; j += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int index = read_only_load(&AFF_diag_j[j]);
      if (index != row)
      {
         dtau_i += AFF_diag_data[j] * read_only_load(&dtmp[index]);
      }
   }

   for (NALU_HYPRE_Int j = po + lane; j < qo; j += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int index = read_only_load(&AFF_offd_j[j]);
      dtau_i += AFF_offd_data[j] * read_only_load(&dtmp_offd[index]);
   }

   dtau_i = warp_reduce_sum(item, dtau_i);

   if (lane == 0)
   {
      value = read_only_load(&rsW[row]) + dtau_i;
      value = value != 0.0 ? -1.0 / value : 0.0;

      theta = read_only_load(&dlam[row]);
   }

   value = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, value, 0);
   theta = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, theta, 0);

   /* AFF Diag part */
   // do not assume diag is the first element of row
   for (NALU_HYPRE_Int j = pd + lane; j < qd; j += NALU_HYPRE_WARP_SIZE)
   {
      if (read_only_load(&AFF_diag_j[j]) == row)
      {
         AFF_diag_data[j] = theta * value;
      }
      else
      {
         AFF_diag_data[j] *= value;
      }
   }

   /* AFF offd part */
   for (NALU_HYPRE_Int j = po + lane; j < qo; j += NALU_HYPRE_WARP_SIZE)
   {
      AFF_offd_data[j] *= value;
   }

   theta = theta != 0.0 ? 1.0 / theta : 0.0;

   /* AFC Diag part */
   for (NALU_HYPRE_Int j = xd + lane; j < yd; j += NALU_HYPRE_WARP_SIZE)
   {
      AFC_diag_data[j] *= theta;
   }

   /* AFC offd part */
   for (NALU_HYPRE_Int j = xo + lane; j < yo; j += NALU_HYPRE_WARP_SIZE)
   {
      AFC_offd_data[j] *= theta;
   }
}

//-----------------------------------------------------------------------
// For Ext+e Interp, compute D_lambda and D_tmp = D_mu / D_lambda
__global__
void hypreGPUKernel_compute_dlam_dtmp( nalu_hypre_DeviceItem    &item,
                                       NALU_HYPRE_Int      nr_of_rows,
                                       NALU_HYPRE_Int     *AFF_diag_i,
                                       NALU_HYPRE_Int     *AFF_diag_j,
                                       NALU_HYPRE_Complex *AFF_diag_data,
                                       NALU_HYPRE_Int     *AFF_offd_i,
                                       NALU_HYPRE_Complex *AFF_offd_data,
                                       NALU_HYPRE_Complex *rsFC,
                                       NALU_HYPRE_Complex *dlam,
                                       NALU_HYPRE_Complex *dtmp )
{
   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p_diag = 0, p_offd = 0, q_diag, q_offd;

   if (lane < 2)
   {
      p_diag = read_only_load(AFF_diag_i + row + lane);
   }
   q_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 0);

   NALU_HYPRE_Complex row_sum = 0.0;
   NALU_HYPRE_Int find_diag = 0;

   /* do not assume the first element is the diagonal */
   for (NALU_HYPRE_Int j = p_diag + lane; j < q_diag; j += NALU_HYPRE_WARP_SIZE)
   {
      if (read_only_load(&AFF_diag_j[j]) == row)
      {
         find_diag ++;
      }
      else
      {
         row_sum += read_only_load(&AFF_diag_data[j]);
      }
   }

   if (lane < 2)
   {
      p_offd = read_only_load(AFF_offd_i + row + lane);
   }
   q_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (NALU_HYPRE_Int j = p_offd + lane; j < q_offd; j += NALU_HYPRE_WARP_SIZE)
   {
      row_sum += read_only_load(&AFF_offd_data[j]);
   }

   row_sum = warp_reduce_sum(item, row_sum);
   find_diag = warp_reduce_sum(item, find_diag);

   if (lane == 0)
   {
      NALU_HYPRE_Int num = q_diag - p_diag + q_offd - p_offd - find_diag;
      NALU_HYPRE_Complex mu = num > 0 ? row_sum / ((NALU_HYPRE_Complex) num) : 0.0;
      /* lambda = beta + mu */
      NALU_HYPRE_Complex lam = read_only_load(&rsFC[row]) + mu;
      dlam[row] = lam;
      dtmp[row] = lam != 0.0 ? mu / lam : 0.0;
   }
}

/*---------------------------------------------------------------------
 * Extended Interpolation in the form of Mat-Mat
 *---------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildExtInterpDevice(nalu_hypre_ParCSRMatrix  *A,
                                    NALU_HYPRE_Int           *CF_marker,
                                    nalu_hypre_ParCSRMatrix  *S,
                                    NALU_HYPRE_BigInt        *num_cpts_global,
                                    NALU_HYPRE_Int            num_functions,
                                    NALU_HYPRE_Int           *dof_func,
                                    NALU_HYPRE_Int            debug_flag,
                                    NALU_HYPRE_Real           trunc_factor,
                                    NALU_HYPRE_Int            max_elmts,
                                    nalu_hypre_ParCSRMatrix **P_ptr)
{
   NALU_HYPRE_Int           A_nr_of_rows = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_CSRMatrix    *A_diag       = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   nalu_hypre_CSRMatrix    *A_offd       = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data  = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i     = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int           A_offd_nnz   = nalu_hypre_CSRMatrixNumNonzeros(A_offd);

   nalu_hypre_ParCSRMatrix *AFF, *AFC;
   nalu_hypre_ParCSRMatrix *W, *P;
   NALU_HYPRE_Int           W_nr_of_rows, P_diag_nnz;
   NALU_HYPRE_Complex      *rsFC, *rsWA, *rsW;
   NALU_HYPRE_Int          *P_diag_i, *P_diag_j, *P_offd_i;
   NALU_HYPRE_Complex      *P_diag_data;

   nalu_hypre_BoomerAMGMakeSocFromSDevice(A, S);

   NALU_HYPRE_Int          *Soc_diag_j   = nalu_hypre_ParCSRMatrixSocDiagJ(S);
   NALU_HYPRE_Int          *Soc_offd_j   = nalu_hypre_ParCSRMatrixSocOffdJ(S);

   /* 0. Find row sums of weak elements */
   /* row sum of A-weak + Diag(A), i.e., (D_gamma + D_alpha) in the notes, only for F-pts */
   rsWA = nalu_hypre_TAlloc(NALU_HYPRE_Complex, A_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(A_nr_of_rows, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_of_rows,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     0 );

   // AFF AFC
   nalu_hypre_GpuProfilingPushRange("Extract Submatrix");
   nalu_hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, &AFC, &AFF);
   nalu_hypre_GpuProfilingPopRange();

   W_nr_of_rows = nalu_hypre_ParCSRMatrixNumRows(AFF);
   nalu_hypre_assert(A_nr_of_rows == W_nr_of_rows + nalu_hypre_ParCSRMatrixNumCols(AFC));

   rsW = nalu_hypre_TAlloc(NALU_HYPRE_Complex, W_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<NALU_HYPRE_Int>() );
#else
   NALU_HYPRE_Complex *new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<NALU_HYPRE_Int>() );
#endif
   nalu_hypre_assert(new_end - rsW == W_nr_of_rows);
   nalu_hypre_TFree(rsWA, NALU_HYPRE_MEMORY_DEVICE);

   /* row sum of AFC, i.e., D_beta */
   rsFC = nalu_hypre_TAlloc(NALU_HYPRE_Complex, W_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixDiag(AFC), NULL, NULL, rsFC, 0, 1.0, "set");
   nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixOffd(AFC), NULL, NULL, rsFC, 0, 1.0, "add");

   /* 5. Form matrix ~{A_FF}, (return twAFF in AFF data structure ) */
   /* 6. Form matrix ~{A_FC}, (return twAFC in AFC data structure) */
   nalu_hypre_GpuProfilingPushRange("Compute interp matrix");
   gDim = nalu_hypre_GetDefaultDeviceGridDimension(W_nr_of_rows, "warp", bDim);
   NALU_HYPRE_Int *AFF_diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Int *AFF_diag_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Complex *AFF_diag_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Int *AFF_offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(AFF));
   NALU_HYPRE_Complex *AFF_offd_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(AFF));
   NALU_HYPRE_Int *AFC_diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(AFC));
   NALU_HYPRE_Complex *AFC_diag_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(AFC));
   NALU_HYPRE_Int *AFC_offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(AFC));
   NALU_HYPRE_Complex *AFC_offd_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(AFC));
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_aff_afc,
                     gDim, bDim,
                     W_nr_of_rows,
                     AFF_diag_i,
                     AFF_diag_j,
                     AFF_diag_a,
                     AFF_offd_i,
                     AFF_offd_a,
                     AFC_diag_i,
                     AFC_diag_a,
                     AFC_offd_i,
                     AFC_offd_a,
                     rsW,
                     rsFC );
   nalu_hypre_TFree(rsW,  NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(rsFC, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_GpuProfilingPopRange();

   /* 7. Perform matrix-matrix multiplication */
   nalu_hypre_GpuProfilingPushRange("Matrix-matrix mult");
   W = nalu_hypre_ParCSRMatMatDevice(AFF, AFC);
   nalu_hypre_GpuProfilingPopRange();

   nalu_hypre_ParCSRMatrixDestroy(AFF);
   nalu_hypre_ParCSRMatrixDestroy(AFC);

   /* 8. Construct P from matrix product W */
   P_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(W)) +
                nalu_hypre_ParCSRMatrixNumCols(W);

   P_diag_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_offd_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_GpuProfilingPushRange("Extend matrix");
   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           nalu_hypre_ParCSRMatrixNumCols(W),
                           CF_marker,
                           nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(W)),
                           nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(W)),
                           nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(W)),
                           nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );
   nalu_hypre_GpuProfilingPopRange();

   // final P
   P = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(W),
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(W),
                                nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(W)));

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(P))    = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(W));
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(P)) = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(W));
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(W))    = NULL;
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(W)) = NULL;

   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(P)) = NALU_HYPRE_MEMORY_DEVICE;
   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(P)) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_ParCSRMatrixDeviceColMapOffd(P) = nalu_hypre_ParCSRMatrixDeviceColMapOffd(W);
   nalu_hypre_ParCSRMatrixColMapOffd(P)       = nalu_hypre_ParCSRMatrixColMapOffd(W);
   nalu_hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   nalu_hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   nalu_hypre_ParCSRMatrixNumNonzeros(P)  = nalu_hypre_ParCSRMatrixNumNonzeros(W) +
                                       nalu_hypre_ParCSRMatrixGlobalNumCols(W);
   nalu_hypre_ParCSRMatrixDNumNonzeros(P) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(P);

   nalu_hypre_GpuProfilingPushRange("Truncation");
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts );
   }
   nalu_hypre_GpuProfilingPopRange();

   nalu_hypre_MatvecCommPkgCreate(P);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<NALU_HYPRE_Int>(-3), -1);
#else
   NALU_HYPRE_THRUST_CALL( replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<NALU_HYPRE_Int>(-3), -1);
#endif

   *P_ptr = P;

   /* 9. Free memory */
   nalu_hypre_ParCSRMatrixDestroy(W);

   return nalu_hypre_error_flag;
}

/*-----------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildExtPIInterpDevice( nalu_hypre_ParCSRMatrix  *A,
                                       NALU_HYPRE_Int           *CF_marker,
                                       nalu_hypre_ParCSRMatrix  *S,
                                       NALU_HYPRE_BigInt        *num_cpts_global,
                                       NALU_HYPRE_Int            num_functions,
                                       NALU_HYPRE_Int           *dof_func,
                                       NALU_HYPRE_Int            debug_flag,
                                       NALU_HYPRE_Real           trunc_factor,
                                       NALU_HYPRE_Int            max_elmts,
                                       nalu_hypre_ParCSRMatrix **P_ptr)
{
   NALU_HYPRE_Int           A_nr_of_rows = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_CSRMatrix    *A_diag       = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   nalu_hypre_CSRMatrix    *A_offd       = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data  = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i     = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int           A_offd_nnz   = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
   nalu_hypre_CSRMatrix    *AFF_ext = NULL;
   nalu_hypre_ParCSRMatrix *AFF, *AFC;
   nalu_hypre_ParCSRMatrix *W, *P;
   NALU_HYPRE_Int           W_nr_of_rows, P_diag_nnz;
   NALU_HYPRE_Complex      *rsFC, *rsFC_offd, *rsWA, *rsW;
   NALU_HYPRE_Int          *P_diag_i, *P_diag_j, *P_offd_i, num_procs;
   NALU_HYPRE_Complex      *P_diag_data;

   nalu_hypre_BoomerAMGMakeSocFromSDevice(A, S);

   NALU_HYPRE_Int          *Soc_diag_j   = nalu_hypre_ParCSRMatrixSocDiagJ(S);
   NALU_HYPRE_Int          *Soc_offd_j   = nalu_hypre_ParCSRMatrixSocOffdJ(S);

   nalu_hypre_MPI_Comm_size(nalu_hypre_ParCSRMatrixComm(A), &num_procs);

   /* 0.Find row sums of weak elements */
   /* row sum of A-weak + Diag(A), i.e., (D_gamma + D_alpha) in the notes, only for F-pts */
   rsWA = nalu_hypre_TAlloc(NALU_HYPRE_Complex, A_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(A_nr_of_rows, "warp",   bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_of_rows,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     0 );

   // AFF AFC
   nalu_hypre_GpuProfilingPushRange("Extract Submatrix");
   nalu_hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, &AFC, &AFF);
   nalu_hypre_GpuProfilingPopRange();

   W_nr_of_rows  = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(AFF));
   nalu_hypre_assert(A_nr_of_rows == W_nr_of_rows + nalu_hypre_ParCSRMatrixNumCols(AFC));

   rsW = nalu_hypre_TAlloc(NALU_HYPRE_Complex, W_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<NALU_HYPRE_Int>() );
#else
   NALU_HYPRE_Complex *new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<NALU_HYPRE_Int>() );
#endif
   nalu_hypre_assert(new_end - rsW == W_nr_of_rows);
   nalu_hypre_TFree(rsWA, NALU_HYPRE_MEMORY_DEVICE);

   /* row sum of AFC, i.e., D_beta */
   rsFC = nalu_hypre_TAlloc(NALU_HYPRE_Complex, W_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixDiag(AFC), NULL, NULL, rsFC, 0, 1.0, "set");
   nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixOffd(AFC), NULL, NULL, rsFC, 0, 1.0, "add");

   /* collect off-processor rsFC */
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(AFF);
   nalu_hypre_ParCSRCommHandle *comm_handle;
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(AFF);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(AFF);
   }
   rsFC_offd = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(AFF)),
                            NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int num_elmts_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   NALU_HYPRE_Complex *send_buf = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_elmts_send, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     rsFC,
                     send_buf );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      rsFC,
                      send_buf );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && THRUST_CALL_BLOCKING == 0
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, send_buf,
                                                 NALU_HYPRE_MEMORY_DEVICE, rsFC_offd);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   nalu_hypre_TFree(send_buf, NALU_HYPRE_MEMORY_DEVICE);

   /* offd rows of AFF */
   if (num_procs > 1)
   {
      AFF_ext = nalu_hypre_ParCSRMatrixExtractBExtDevice(AFF, AFF, 1);
   }

   /* 5. Form matrix ~{A_FF}, (return twAFF in AFF data structure ) */
   NALU_HYPRE_Complex *AFF_diag_data_old = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                                                   nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(AFF)),
                                                   NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy( AFF_diag_data_old,
                  nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(AFF)),
                  NALU_HYPRE_Complex,
                  nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(AFF)),
                  NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_GpuProfilingPushRange("Compute interp matrix");
   gDim = nalu_hypre_GetDefaultDeviceGridDimension(W_nr_of_rows, "warp", bDim);
   NALU_HYPRE_BigInt AFF_first_row_idx = nalu_hypre_ParCSRMatrixFirstRowIndex(AFF);
   NALU_HYPRE_Int *AFF_diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Int *AFF_diag_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Complex *AFF_diag_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Int *AFF_offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(AFF));
   NALU_HYPRE_Int *AFF_offd_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(AFF));
   NALU_HYPRE_Complex *AFF_offd_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(AFF));
   NALU_HYPRE_Int *AFF_ext_i = NULL;
   NALU_HYPRE_BigInt *AFF_ext_bigj = NULL;
   NALU_HYPRE_Complex *AFF_ext_a = NULL;
   if (AFF_ext)
   {
      AFF_ext_i = nalu_hypre_CSRMatrixI(AFF_ext);
      AFF_ext_bigj = nalu_hypre_CSRMatrixBigJ(AFF_ext);
      AFF_ext_a = nalu_hypre_CSRMatrixData(AFF_ext);
   }
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_twiaff_w,
                     gDim, bDim,
                     W_nr_of_rows,
                     AFF_first_row_idx,
                     AFF_diag_i,
                     AFF_diag_j,
                     AFF_diag_a,
                     AFF_diag_data_old,
                     AFF_offd_i,
                     AFF_offd_j,
                     AFF_offd_a,
                     AFF_ext_i,
                     AFF_ext_bigj,
                     AFF_ext_a,
                     rsW,
                     rsFC,
                     rsFC_offd );
   nalu_hypre_TFree(rsW,               NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(rsFC,              NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(rsFC_offd,         NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(AFF_diag_data_old, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_CSRMatrixDestroy(AFF_ext);
   nalu_hypre_GpuProfilingPopRange();

   /* 7. Perform matrix-matrix multiplication */
   nalu_hypre_GpuProfilingPushRange("Matrix-matrix mult");
   W = nalu_hypre_ParCSRMatMatDevice(AFF, AFC);
   nalu_hypre_GpuProfilingPopRange();

   nalu_hypre_ParCSRMatrixDestroy(AFF);
   nalu_hypre_ParCSRMatrixDestroy(AFC);

   /* 8. Construct P from matrix product W */
   P_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(W)) +
                nalu_hypre_ParCSRMatrixNumCols(W);

   P_diag_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_offd_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_GpuProfilingPushRange("Extend matrix");
   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           nalu_hypre_ParCSRMatrixNumCols(W),
                           CF_marker,
                           nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(W)),
                           nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(W)),
                           nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(W)),
                           nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );
   nalu_hypre_GpuProfilingPopRange();

   // final P
   P = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(W),
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(W),
                                nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(W)));

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(P))    = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(W));
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(P)) = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(W));
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(W))    = NULL;
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(W)) = NULL;

   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(P)) = NALU_HYPRE_MEMORY_DEVICE;
   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(P)) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_ParCSRMatrixDeviceColMapOffd(P) = nalu_hypre_ParCSRMatrixDeviceColMapOffd(W);
   nalu_hypre_ParCSRMatrixColMapOffd(P)       = nalu_hypre_ParCSRMatrixColMapOffd(W);
   nalu_hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   nalu_hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   nalu_hypre_ParCSRMatrixNumNonzeros(P)  = nalu_hypre_ParCSRMatrixNumNonzeros(W) +
                                       nalu_hypre_ParCSRMatrixGlobalNumCols(W);
   nalu_hypre_ParCSRMatrixDNumNonzeros(P) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(P);

   nalu_hypre_GpuProfilingPushRange("Truncation");
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts );
   }
   nalu_hypre_GpuProfilingPopRange();

   nalu_hypre_MatvecCommPkgCreate(P);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<NALU_HYPRE_Int>(-3), -1);
#else
   NALU_HYPRE_THRUST_CALL( replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<NALU_HYPRE_Int>(-3), -1);
#endif

   *P_ptr = P;

   /* 9. Free memory */
   nalu_hypre_ParCSRMatrixDestroy(W);

   return nalu_hypre_error_flag;
}

/*---------------------------------------------------------------------
 * Extended+e Interpolation in the form of Mat-Mat
 *---------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildExtPEInterpDevice(nalu_hypre_ParCSRMatrix  *A,
                                      NALU_HYPRE_Int           *CF_marker,
                                      nalu_hypre_ParCSRMatrix  *S,
                                      NALU_HYPRE_BigInt        *num_cpts_global,
                                      NALU_HYPRE_Int            num_functions,
                                      NALU_HYPRE_Int           *dof_func,
                                      NALU_HYPRE_Int            debug_flag,
                                      NALU_HYPRE_Real           trunc_factor,
                                      NALU_HYPRE_Int            max_elmts,
                                      nalu_hypre_ParCSRMatrix **P_ptr)
{
   NALU_HYPRE_Int           A_nr_of_rows = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_CSRMatrix    *A_diag       = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   nalu_hypre_CSRMatrix    *A_offd       = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data  = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i     = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int           A_offd_nnz   = nalu_hypre_CSRMatrixNumNonzeros(A_offd);

   nalu_hypre_BoomerAMGMakeSocFromSDevice(A, S);

   NALU_HYPRE_Int          *Soc_diag_j   = nalu_hypre_ParCSRMatrixSocDiagJ(S);
   NALU_HYPRE_Int          *Soc_offd_j   = nalu_hypre_ParCSRMatrixSocOffdJ(S);
   nalu_hypre_ParCSRMatrix *AFF, *AFC;
   nalu_hypre_ParCSRMatrix *W, *P;
   NALU_HYPRE_Int           W_nr_of_rows, P_diag_nnz;
   NALU_HYPRE_Complex      *dlam, *dtmp, *dtmp_offd, *rsFC, *rsWA, *rsW;
   NALU_HYPRE_Int          *P_diag_i, *P_diag_j, *P_offd_i;
   NALU_HYPRE_Complex      *P_diag_data;

   /* 0. Find row sums of weak elements */
   /* row sum of A-weak + Diag(A), i.e., (D_gamma + D_FF) in the notes, only for F-pts */
   rsWA = nalu_hypre_TAlloc(NALU_HYPRE_Complex, A_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(A_nr_of_rows, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_of_rows,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     0 );

   // AFF AFC
   nalu_hypre_GpuProfilingPushRange("Extract Submatrix");
   nalu_hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, &AFC, &AFF);
   nalu_hypre_GpuProfilingPopRange();

   W_nr_of_rows = nalu_hypre_ParCSRMatrixNumRows(AFF);
   nalu_hypre_assert(A_nr_of_rows == W_nr_of_rows + nalu_hypre_ParCSRMatrixNumCols(AFC));

   rsW = nalu_hypre_TAlloc(NALU_HYPRE_Complex, W_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<NALU_HYPRE_Int>() );
#else
   NALU_HYPRE_Complex *new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker,
                                               rsW,
                                               is_negative<NALU_HYPRE_Int>() );
#endif
   nalu_hypre_assert(new_end - rsW == W_nr_of_rows);
   nalu_hypre_TFree(rsWA, NALU_HYPRE_MEMORY_DEVICE);

   /* row sum of AFC, i.e., D_beta */
   rsFC = nalu_hypre_TAlloc(NALU_HYPRE_Complex, W_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixDiag(AFC), NULL, NULL, rsFC, 0, 1.0, "set");
   nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixOffd(AFC), NULL, NULL, rsFC, 0, 1.0, "add");

   /* Generate D_lambda in the paper: D_beta + (row sum of AFF without diagonal elements / row_nnz) */
   /* Generate D_tmp, i.e., D_mu / D_lambda */
   dlam = nalu_hypre_TAlloc(NALU_HYPRE_Complex, W_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);
   dtmp = nalu_hypre_TAlloc(NALU_HYPRE_Complex, W_nr_of_rows, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_GpuProfilingPushRange("Compute D_tmp");
   gDim = nalu_hypre_GetDefaultDeviceGridDimension(W_nr_of_rows, "warp", bDim);
   NALU_HYPRE_Int *AFF_diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Int *AFF_diag_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Complex *AFF_diag_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Int *AFF_offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(AFF));
   NALU_HYPRE_Int *AFF_offd_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(AFF));
   NALU_HYPRE_Complex *AFF_offd_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(AFF));
   NALU_HYPRE_Int *AFC_diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(AFC));
   NALU_HYPRE_Complex *AFC_diag_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(AFC));
   NALU_HYPRE_Int *AFC_offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(AFC));
   NALU_HYPRE_Complex *AFC_offd_a = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(AFC));
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_dlam_dtmp,
                     gDim, bDim,
                     W_nr_of_rows,
                     AFF_diag_i,
                     AFF_diag_j,
                     AFF_diag_a,
                     AFF_offd_i,
                     AFF_offd_a,
                     rsFC,
                     dlam,
                     dtmp );

   /* collect off-processor dtmp */
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(AFF);
   nalu_hypre_ParCSRCommHandle *comm_handle;
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(AFF);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(AFF);
   }
   dtmp_offd = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(AFF)),
                            NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int num_elmts_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   NALU_HYPRE_Complex *send_buf = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_elmts_send, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     dtmp,
                     send_buf );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      dtmp,
                      send_buf );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && THRUST_CALL_BLOCKING == 0
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, send_buf,
                                                 NALU_HYPRE_MEMORY_DEVICE, dtmp_offd);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   nalu_hypre_TFree(send_buf, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_GpuProfilingPopRange();

   /* 4. Form D_tau */
   /* 5. Form matrix ~{A_FF}, (return twAFF in AFF data structure ) */
   /* 6. Form matrix ~{A_FC}, (return twAFC in AFC data structure) */
   nalu_hypre_GpuProfilingPushRange("Compute interp matrix");
   gDim = nalu_hypre_GetDefaultDeviceGridDimension(W_nr_of_rows, "warp", bDim);
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_aff_afc_epe,
                     gDim, bDim,
                     W_nr_of_rows,
                     AFF_diag_i,
                     AFF_diag_j,
                     AFF_diag_a,
                     AFF_offd_i,
                     AFF_offd_j,
                     AFF_offd_a,
                     AFC_diag_i,
                     AFC_diag_a,
                     AFC_offd_i,
                     AFC_offd_a,
                     rsW,
                     dlam,
                     dtmp,
                     dtmp_offd );
   nalu_hypre_TFree(rsW,  NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(rsFC, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(dlam, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(dtmp, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(dtmp_offd, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_GpuProfilingPopRange();

   /* 7. Perform matrix-matrix multiplication */
   nalu_hypre_GpuProfilingPushRange("Matrix-matrix mult");
   W = nalu_hypre_ParCSRMatMatDevice(AFF, AFC);
   nalu_hypre_GpuProfilingPopRange();

   nalu_hypre_ParCSRMatrixDestroy(AFF);
   nalu_hypre_ParCSRMatrixDestroy(AFC);

   /* 8. Construct P from matrix product W */
   P_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(W)) +
                nalu_hypre_ParCSRMatrixNumCols(W);

   P_diag_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_offd_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_GpuProfilingPushRange("Extend matrix");
   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           nalu_hypre_ParCSRMatrixNumCols(W),
                           CF_marker,
                           nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(W)),
                           nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(W)),
                           nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(W)),
                           nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );
   nalu_hypre_GpuProfilingPopRange();

   // final P
   P = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(W),
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(W),
                                nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(W)));

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(P))    = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(W));
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(P)) = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(W));
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(W))    = NULL;
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(W)) = NULL;

   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(P)) = NALU_HYPRE_MEMORY_DEVICE;
   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(P)) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_ParCSRMatrixDeviceColMapOffd(P) = nalu_hypre_ParCSRMatrixDeviceColMapOffd(W);
   nalu_hypre_ParCSRMatrixColMapOffd(P)       = nalu_hypre_ParCSRMatrixColMapOffd(W);
   nalu_hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   nalu_hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   nalu_hypre_ParCSRMatrixNumNonzeros(P)  = nalu_hypre_ParCSRMatrixNumNonzeros(W) +
                                       nalu_hypre_ParCSRMatrixGlobalNumCols(W);
   nalu_hypre_ParCSRMatrixDNumNonzeros(P) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(P);

   nalu_hypre_GpuProfilingPushRange("Truncation");
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts );
   }
   nalu_hypre_GpuProfilingPopRange();

   nalu_hypre_MatvecCommPkgCreate(P);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<NALU_HYPRE_Int>(-3), -1);
#else
   NALU_HYPRE_THRUST_CALL( replace_if, CF_marker, CF_marker + A_nr_of_rows, equal<NALU_HYPRE_Int>(-3), -1);
#endif

   *P_ptr = P;

   /* 9. Free memory */
   nalu_hypre_ParCSRMatrixDestroy(W);

   return nalu_hypre_error_flag;
}

#endif // defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
