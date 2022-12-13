/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)

#if defined(NALU_HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_compute_weak_rowsums( hypre_DeviceItem &item, NALU_HYPRE_Int nr_of_rows,
                                                     bool has_offd,
                                                     NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Complex *A_diag_a, NALU_HYPRE_Int *S_diag_j,
                                                     NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Complex *A_offd_a, NALU_HYPRE_Int *S_offd_j, NALU_HYPRE_Real *rs, NALU_HYPRE_Int flag );

__global__ void hypreGPUKernel_MMInterpScaleAFF( hypre_DeviceItem &item, NALU_HYPRE_Int AFF_nrows,
                                                 NALU_HYPRE_Int *AFF_diag_i,
                                                 NALU_HYPRE_Int *AFF_diag_j, NALU_HYPRE_Complex *AFF_diag_a, NALU_HYPRE_Int *AFF_offd_i, NALU_HYPRE_Int *AFF_offd_j,
                                                 NALU_HYPRE_Complex *AFF_offd_a, NALU_HYPRE_Complex *beta_diag, NALU_HYPRE_Complex *beta_offd, NALU_HYPRE_Int *F2_to_F,
                                                 NALU_HYPRE_Real *rsW );

#if defined(NALU_HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_compute_dlam_dtmp( hypre_DeviceItem &item, NALU_HYPRE_Int nr_of_rows,
                                                  NALU_HYPRE_Int *AFF_diag_i,
                                                  NALU_HYPRE_Int *AFF_diag_j, NALU_HYPRE_Complex *AFF_diag_data, NALU_HYPRE_Int *AFF_offd_i,
                                                  NALU_HYPRE_Complex *AFF_offd_data, NALU_HYPRE_Complex *rsFC, NALU_HYPRE_Complex *dlam, NALU_HYPRE_Complex *dtmp );

__global__ void hypreGPUKernel_MMPEInterpScaleAFF( hypre_DeviceItem &item, NALU_HYPRE_Int AFF_nrows,
                                                   NALU_HYPRE_Int *AFF_diag_i,
                                                   NALU_HYPRE_Int *AFF_diag_j, NALU_HYPRE_Complex *AFF_diag_a, NALU_HYPRE_Int *AFF_offd_i, NALU_HYPRE_Int *AFF_offd_j,
                                                   NALU_HYPRE_Complex *AFF_offd_a, NALU_HYPRE_Complex *tmp_diag, NALU_HYPRE_Complex *tmp_offd,
                                                   NALU_HYPRE_Complex *lam_diag, NALU_HYPRE_Complex *lam_offd, NALU_HYPRE_Int *F2_to_F, NALU_HYPRE_Real *rsW );

void hypreDevice_extendWtoP( NALU_HYPRE_Int P_nr_of_rows, NALU_HYPRE_Int W_nr_of_rows, NALU_HYPRE_Int W_nr_of_cols,
                             NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int W_diag_nnz, NALU_HYPRE_Int *W_diag_i, NALU_HYPRE_Int *W_diag_j,
                             NALU_HYPRE_Complex *W_diag_data, NALU_HYPRE_Int *P_diag_i, NALU_HYPRE_Int *P_diag_j, NALU_HYPRE_Complex *P_diag_data,
                             NALU_HYPRE_Int *W_offd_i, NALU_HYPRE_Int *P_offd_i );

/*--------------------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_BoomerAMGBuildModPartialExtInterpDevice( hypre_ParCSRMatrix  *A,
                                               NALU_HYPRE_Int           *CF_marker,
                                               hypre_ParCSRMatrix  *S,
                                               NALU_HYPRE_BigInt        *num_cpts_global,     /* C2 */
                                               NALU_HYPRE_BigInt        *num_old_cpts_global, /* C2 + F2 */
                                               NALU_HYPRE_Int            debug_flag,
                                               NALU_HYPRE_Real           trunc_factor,
                                               NALU_HYPRE_Int            max_elmts,
                                               hypre_ParCSRMatrix **P_ptr )
{
   NALU_HYPRE_Int           A_nr_local   = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag       = hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i     = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd       = hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i     = hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int           A_offd_nnz   = hypre_CSRMatrixNumNonzeros(A_offd);
   NALU_HYPRE_Complex      *Dbeta, *Dbeta_offd, *rsWA, *rsW;
   hypre_ParCSRMatrix *As_F2F, *As_FC, *W, *P;

   hypre_BoomerAMGMakeSocFromSDevice(A, S);

   NALU_HYPRE_Int          *Soc_diag_j   = hypre_ParCSRMatrixSocDiagJ(S);
   NALU_HYPRE_Int          *Soc_offd_j   = hypre_ParCSRMatrixSocOffdJ(S);

   /* As_F2F = As_{F2, F}, As_FC = As_{F, C2} */
   hypre_ParCSRMatrixGenerateFFFC3Device(A, CF_marker, num_cpts_global, S, &As_FC, &As_F2F);

   NALU_HYPRE_Int AFC_nr_local = hypre_ParCSRMatrixNumRows(As_FC);
   NALU_HYPRE_Int AF2F_nr_local = hypre_ParCSRMatrixNumRows(As_F2F);

   /* row sum of AFC, i.e., D_beta */
   Dbeta = hypre_TAlloc(NALU_HYPRE_Complex, AFC_nr_local, NALU_HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(As_FC), NULL, NULL, Dbeta, 0, 1.0, "set");
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(As_FC), NULL, NULL, Dbeta, 0, 1.0, "add");

   /* collect off-processor D_beta */
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(As_F2F);
   hypre_ParCSRCommHandle *comm_handle;
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(As_F2F);
      comm_pkg = hypre_ParCSRMatrixCommPkg(As_F2F);
   }
   Dbeta_offd = hypre_TAlloc(NALU_HYPRE_Complex, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(As_F2F)),
                             NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int num_elmts_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   NALU_HYPRE_Complex *send_buf = hypre_TAlloc(NALU_HYPRE_Complex, num_elmts_send, NALU_HYPRE_MEMORY_DEVICE);
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     Dbeta,
                     send_buf );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      Dbeta,
                      send_buf );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && THRUST_CALL_BLOCKING == 0
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   hypre_ForceSyncComputeStream(hypre_handle());
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, send_buf,
                                                 NALU_HYPRE_MEMORY_DEVICE, Dbeta_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(send_buf, NALU_HYPRE_MEMORY_DEVICE);

   /* weak row sum and diagonal, i.e., DF2F2 + Dgamma */
   rsWA = hypre_TAlloc(NALU_HYPRE_Complex, A_nr_local, NALU_HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(A_nr_local, "warp", bDim);

   /* only for rows corresponding to F2 (notice flag == -1) */
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_local,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     -1 );

   rsW = hypre_TAlloc(NALU_HYPRE_Complex, AF2F_nr_local, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_local,
                                               CF_marker,
                                               rsW,
                                               equal<NALU_HYPRE_Int>(-2) );
#else
   NALU_HYPRE_Complex *new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_local,
                                               CF_marker,
                                               rsW,
                                               equal<NALU_HYPRE_Int>(-2) );
#endif

   hypre_assert(new_end - rsW == AF2F_nr_local);

   hypre_TFree(rsWA, NALU_HYPRE_MEMORY_DEVICE);

   /* map from F2 to F */
   NALU_HYPRE_Int *map_to_F = hypre_TAlloc(NALU_HYPRE_Int, A_nr_local, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,              is_negative<NALU_HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + A_nr_local, is_negative<NALU_HYPRE_Int>()),
                      map_to_F,
                      NALU_HYPRE_Int(0) );/* *MUST* pass init value since input and output types diff. */
#else
   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,              is_negative<NALU_HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + A_nr_local, is_negative<NALU_HYPRE_Int>()),
                      map_to_F,
                      NALU_HYPRE_Int(0) );/* *MUST* pass init value since input and output types diff. */
#endif

   NALU_HYPRE_Int *map_F2_to_F = hypre_TAlloc(NALU_HYPRE_Int, AF2F_nr_local, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int *tmp_end = hypreSycl_copy_if( map_to_F,
                                           map_to_F + A_nr_local,
                                           CF_marker,
                                           map_F2_to_F,
                                           equal<NALU_HYPRE_Int>(-2) );
#else
   NALU_HYPRE_Int *tmp_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                           map_to_F,
                                           map_to_F + A_nr_local,
                                           CF_marker,
                                           map_F2_to_F,
                                           equal<NALU_HYPRE_Int>(-2) );
#endif

   hypre_assert(tmp_end - map_F2_to_F == AF2F_nr_local);

   hypre_TFree(map_to_F, NALU_HYPRE_MEMORY_DEVICE);

   /* add to rsW those in AF2F that correspond to Dbeta == 0
    * diagnoally scale As_F2F (from both sides) and replace the diagonal */
   gDim = hypre_GetDefaultDeviceGridDimension(AF2F_nr_local, "warp", bDim);

   NALU_HYPRE_Int *As_F2F_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(As_F2F));
   NALU_HYPRE_Int *As_F2F_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(As_F2F));
   NALU_HYPRE_Complex *As_F2F_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(As_F2F));
   NALU_HYPRE_Int *As_F2F_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(As_F2F));
   NALU_HYPRE_Int *As_F2F_offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(As_F2F));
   NALU_HYPRE_Complex *As_F2F_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(As_F2F));
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_MMInterpScaleAFF,
                     gDim, bDim,
                     AF2F_nr_local,
                     As_F2F_diag_i,
                     As_F2F_diag_j,
                     As_F2F_diag_data,
                     As_F2F_offd_i,
                     As_F2F_offd_j,
                     As_F2F_offd_data,
                     Dbeta,
                     Dbeta_offd,
                     map_F2_to_F,
                     rsW );

   hypre_TFree(Dbeta, NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(Dbeta_offd, NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(map_F2_to_F, NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsW, NALU_HYPRE_MEMORY_DEVICE);

   /* Perform matrix-matrix multiplication */
   W = hypre_ParCSRMatMatDevice(As_F2F, As_FC);

   hypre_ParCSRMatrixDestroy(As_F2F);
   hypre_ParCSRMatrixDestroy(As_FC);

   /* Construct P from matrix product W */
   NALU_HYPRE_Int     *P_diag_i, *P_diag_j, *P_offd_i;
   NALU_HYPRE_Complex *P_diag_data;
   NALU_HYPRE_Int      P_nr_local = A_nr_local - (AFC_nr_local - AF2F_nr_local);
   NALU_HYPRE_Int      P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)) +
                               hypre_ParCSRMatrixNumCols(W);

   hypre_assert(P_nr_local == hypre_ParCSRMatrixNumRows(W) + hypre_ParCSRMatrixNumCols(W));

   P_diag_i    = hypre_TAlloc(NALU_HYPRE_Int,     P_nr_local + 1, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(NALU_HYPRE_Int,     P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(NALU_HYPRE_Complex, P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(NALU_HYPRE_Int,     P_nr_local + 1, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_Int *C2F2_marker = hypre_TAlloc(NALU_HYPRE_Int, P_nr_local, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   tmp_end = hypreSycl_copy_if( CF_marker,
                                CF_marker + A_nr_local,
                                CF_marker,
                                C2F2_marker,
                                out_of_range<NALU_HYPRE_Int>(-1, 0) /* -2 or 1 */ );
#else
   tmp_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                CF_marker,
                                CF_marker + A_nr_local,
                                CF_marker,
                                C2F2_marker,
                                out_of_range<NALU_HYPRE_Int>(-1, 0) /* -2 or 1 */ );
#endif

   hypre_assert(tmp_end - C2F2_marker == P_nr_local);

   hypreDevice_extendWtoP( P_nr_local,
                           AF2F_nr_local,
                           hypre_ParCSRMatrixNumCols(W),
                           C2F2_marker,
                           hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );

   hypre_TFree(C2F2_marker, NALU_HYPRE_MEMORY_DEVICE);

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(W) + hypre_ParCSRMatrixGlobalNumCols(W),
                                hypre_ParCSRMatrixGlobalNumCols(W),
                                num_old_cpts_global,
                                num_cpts_global,
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W)));

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W))    = NULL;
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W)) = NULL;

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = NALU_HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = NALU_HYPRE_MEMORY_DEVICE;

   hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   hypre_ParCSRMatrixNumNonzeros(P)  = hypre_ParCSRMatrixNumNonzeros(W) +
                                       hypre_ParCSRMatrixGlobalNumCols(W);
   hypre_ParCSRMatrixDNumNonzeros(P) = (NALU_HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_ParCSRMatrixDestroy(W);

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts );
   }

   *P_ptr = P;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_BoomerAMGBuildModPartialExtPEInterpDevice( hypre_ParCSRMatrix  *A,
                                                 NALU_HYPRE_Int           *CF_marker,
                                                 hypre_ParCSRMatrix  *S,
                                                 NALU_HYPRE_BigInt        *num_cpts_global,     /* C2 */
                                                 NALU_HYPRE_BigInt        *num_old_cpts_global, /* C2 + F2 */
                                                 NALU_HYPRE_Int            debug_flag,
                                                 NALU_HYPRE_Real           trunc_factor,
                                                 NALU_HYPRE_Int            max_elmts,
                                                 hypre_ParCSRMatrix **P_ptr )
{
   NALU_HYPRE_Int           A_nr_local   = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag       = hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i     = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd       = hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i     = hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int           A_offd_nnz   = hypre_CSRMatrixNumNonzeros(A_offd);
   NALU_HYPRE_Complex      *Dbeta, *rsWA, *rsW, *dlam, *dlam_offd, *dtmp, *dtmp_offd;
   hypre_ParCSRMatrix *As_F2F, *As_FF, *As_FC, *W, *P;

   hypre_BoomerAMGMakeSocFromSDevice(A, S);

   NALU_HYPRE_Int          *Soc_diag_j   = hypre_ParCSRMatrixSocDiagJ(S);
   NALU_HYPRE_Int          *Soc_offd_j   = hypre_ParCSRMatrixSocOffdJ(S);

   /* As_F2F = As_{F2, F}, As_FC = As_{F, C2} */
   hypre_ParCSRMatrixGenerateFFFC3Device(A, CF_marker, num_cpts_global, S, &As_FC, &As_F2F);

   NALU_HYPRE_Int AFC_nr_local = hypre_ParCSRMatrixNumRows(As_FC);
   NALU_HYPRE_Int AF2F_nr_local = hypre_ParCSRMatrixNumRows(As_F2F);

   /* row sum of AFC, i.e., D_beta */
   Dbeta = hypre_TAlloc(NALU_HYPRE_Complex, AFC_nr_local, NALU_HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(As_FC), NULL, NULL, Dbeta, 0, 1.0, "set");
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(As_FC), NULL, NULL, Dbeta, 0, 1.0, "add");

   /* As_FF = As_{F,F} */
   hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, NULL, &As_FF);

   hypre_assert(AFC_nr_local == hypre_ParCSRMatrixNumRows(As_FF));

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(AFC_nr_local, "warp", bDim);

   /* Generate D_lambda in the paper: D_beta + (row sum of AFF without diagonal elements / row_nnz) */
   /* Generate D_tmp, i.e., D_mu / D_lambda */
   dlam = hypre_TAlloc(NALU_HYPRE_Complex, AFC_nr_local, NALU_HYPRE_MEMORY_DEVICE);
   dtmp = hypre_TAlloc(NALU_HYPRE_Complex, AFC_nr_local, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_Int *As_FF_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(As_FF));
   NALU_HYPRE_Int *As_FF_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(As_FF));
   NALU_HYPRE_Complex *As_FF_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(As_FF));
   NALU_HYPRE_Int *As_FF_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(As_FF));
   NALU_HYPRE_Complex *As_FF_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(As_FF));
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_dlam_dtmp,
                     gDim, bDim,
                     AFC_nr_local,
                     As_FF_diag_i,
                     As_FF_diag_j,
                     As_FF_diag_data,
                     As_FF_offd_i,
                     As_FF_offd_data,
                     Dbeta,
                     dlam,
                     dtmp );

   hypre_ParCSRMatrixDestroy(As_FF);
   hypre_TFree(Dbeta, NALU_HYPRE_MEMORY_DEVICE);

   /* collect off-processor dtmp and dlam */
   dtmp_offd = hypre_TAlloc(NALU_HYPRE_Complex, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(As_F2F)),
                            NALU_HYPRE_MEMORY_DEVICE);
   dlam_offd = hypre_TAlloc(NALU_HYPRE_Complex, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(As_F2F)),
                            NALU_HYPRE_MEMORY_DEVICE);

   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(As_F2F);
   hypre_ParCSRCommHandle *comm_handle;
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(As_F2F);
      comm_pkg = hypre_ParCSRMatrixCommPkg(As_F2F);
   }
   NALU_HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int num_elmts_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   NALU_HYPRE_Complex *send_buf = hypre_TAlloc(NALU_HYPRE_Complex, num_elmts_send, NALU_HYPRE_MEMORY_DEVICE);
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     dtmp,
                     send_buf );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      dtmp,
                      send_buf );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && THRUST_CALL_BLOCKING == 0
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   hypre_ForceSyncComputeStream(hypre_handle());
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, send_buf,
                                                 NALU_HYPRE_MEMORY_DEVICE, dtmp_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     dlam,
                     send_buf );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      dlam,
                      send_buf );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && THRUST_CALL_BLOCKING == 0
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   hypre_ForceSyncComputeStream(hypre_handle());
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, send_buf,
                                                 NALU_HYPRE_MEMORY_DEVICE, dlam_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   hypre_TFree(send_buf, NALU_HYPRE_MEMORY_DEVICE);

   /* weak row sum and diagonal, i.e., DFF + Dgamma */
   rsWA = hypre_TAlloc(NALU_HYPRE_Complex, A_nr_local, NALU_HYPRE_MEMORY_DEVICE);

   gDim = hypre_GetDefaultDeviceGridDimension(A_nr_local, "warp", bDim);

   /* only for rows corresponding to F2 (notice flag == -1) */
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_local,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     -1 );

   rsW = hypre_TAlloc(NALU_HYPRE_Complex, AF2F_nr_local, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_local,
                                               CF_marker,
                                               rsW,
                                               equal<NALU_HYPRE_Int>(-2) );
#else
   NALU_HYPRE_Complex *new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_local,
                                               CF_marker,
                                               rsW,
                                               equal<NALU_HYPRE_Int>(-2) );
#endif

   hypre_assert(new_end - rsW == AF2F_nr_local);

   hypre_TFree(rsWA, NALU_HYPRE_MEMORY_DEVICE);

   /* map from F2 to F */
   NALU_HYPRE_Int *map_to_F = hypre_TAlloc(NALU_HYPRE_Int, A_nr_local, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,              is_negative<NALU_HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + A_nr_local, is_negative<NALU_HYPRE_Int>()),
                      map_to_F,
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#else
   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,              is_negative<NALU_HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + A_nr_local, is_negative<NALU_HYPRE_Int>()),
                      map_to_F,
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#endif
   NALU_HYPRE_Int *map_F2_to_F = hypre_TAlloc(NALU_HYPRE_Int, AF2F_nr_local, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int *tmp_end = hypreSycl_copy_if( map_to_F,
                                           map_to_F + A_nr_local,
                                           CF_marker,
                                           map_F2_to_F,
                                           equal<NALU_HYPRE_Int>(-2) );
#else
   NALU_HYPRE_Int *tmp_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                           map_to_F,
                                           map_to_F + A_nr_local,
                                           CF_marker,
                                           map_F2_to_F,
                                           equal<NALU_HYPRE_Int>(-2) );
#endif

   hypre_assert(tmp_end - map_F2_to_F == AF2F_nr_local);

   hypre_TFree(map_to_F, NALU_HYPRE_MEMORY_DEVICE);

   /* add to rsW those in AFF that correspond to lam == 0
    * diagnoally scale As_F2F (from both sides) and replace the diagonal */
   gDim = hypre_GetDefaultDeviceGridDimension(AF2F_nr_local, "warp", bDim);

   NALU_HYPRE_Int *As_F2F_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(As_F2F));
   NALU_HYPRE_Int *As_F2F_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(As_F2F));
   NALU_HYPRE_Complex *As_F2F_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(As_F2F));
   NALU_HYPRE_Int *As_F2F_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(As_F2F));
   NALU_HYPRE_Int *As_F2F_offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(As_F2F));
   NALU_HYPRE_Complex *As_F2F_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(As_F2F));
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_MMPEInterpScaleAFF,
                     gDim, bDim,
                     AF2F_nr_local,
                     As_F2F_diag_i,
                     As_F2F_diag_j,
                     As_F2F_diag_data,
                     As_F2F_offd_i,
                     As_F2F_offd_j,
                     As_F2F_offd_data,
                     dtmp,
                     dtmp_offd,
                     dlam,
                     dlam_offd,
                     map_F2_to_F,
                     rsW );

   hypre_TFree(dlam,        NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(dlam_offd,   NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(dtmp,        NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(dtmp_offd,   NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(map_F2_to_F, NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsW,         NALU_HYPRE_MEMORY_DEVICE);

   /* Perform matrix-matrix multiplication */
   W = hypre_ParCSRMatMatDevice(As_F2F, As_FC);

   hypre_ParCSRMatrixDestroy(As_F2F);
   hypre_ParCSRMatrixDestroy(As_FC);

   /* Construct P from matrix product W */
   NALU_HYPRE_Int     *P_diag_i, *P_diag_j, *P_offd_i;
   NALU_HYPRE_Complex *P_diag_data;
   NALU_HYPRE_Int      P_nr_local = A_nr_local - (AFC_nr_local - AF2F_nr_local);
   NALU_HYPRE_Int      P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)) +
                               hypre_ParCSRMatrixNumCols(W);

   hypre_assert(P_nr_local == hypre_ParCSRMatrixNumRows(W) + hypre_ParCSRMatrixNumCols(W));

   P_diag_i    = hypre_TAlloc(NALU_HYPRE_Int,     P_nr_local + 1, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(NALU_HYPRE_Int,     P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(NALU_HYPRE_Complex, P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(NALU_HYPRE_Int,     P_nr_local + 1, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_Int *C2F2_marker = hypre_TAlloc(NALU_HYPRE_Int, P_nr_local, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   tmp_end = hypreSycl_copy_if( CF_marker,
                                CF_marker + A_nr_local,
                                CF_marker,
                                C2F2_marker,
                                out_of_range<NALU_HYPRE_Int>(-1, 0) /* -2 or 1 */ );
#else
   tmp_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                CF_marker,
                                CF_marker + A_nr_local,
                                CF_marker,
                                C2F2_marker,
                                out_of_range<NALU_HYPRE_Int>(-1, 0) /* -2 or 1 */ );
#endif

   hypre_assert(tmp_end - C2F2_marker == P_nr_local);

   hypreDevice_extendWtoP( P_nr_local,
                           AF2F_nr_local,
                           hypre_ParCSRMatrixNumCols(W),
                           C2F2_marker,
                           hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );

   hypre_TFree(C2F2_marker, NALU_HYPRE_MEMORY_DEVICE);

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(W) + hypre_ParCSRMatrixGlobalNumCols(W),
                                hypre_ParCSRMatrixGlobalNumCols(W),
                                num_old_cpts_global,
                                num_cpts_global,
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W)));

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W))    = NULL;
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W)) = NULL;

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = NALU_HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = NALU_HYPRE_MEMORY_DEVICE;

   hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   hypre_ParCSRMatrixNumNonzeros(P)  = hypre_ParCSRMatrixNumNonzeros(W) +
                                       hypre_ParCSRMatrixGlobalNumCols(W);
   hypre_ParCSRMatrixDNumNonzeros(P) = (NALU_HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_ParCSRMatrixDestroy(W);

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts );
   }

   *P_ptr = P;

   return hypre_error_flag;
}

//-----------------------------------------------------------------------
__global__
void hypreGPUKernel_MMInterpScaleAFF( hypre_DeviceItem    &item,
                                      NALU_HYPRE_Int      AFF_nrows,
                                      NALU_HYPRE_Int     *AFF_diag_i,
                                      NALU_HYPRE_Int     *AFF_diag_j,
                                      NALU_HYPRE_Complex *AFF_diag_a,
                                      NALU_HYPRE_Int     *AFF_offd_i,
                                      NALU_HYPRE_Int     *AFF_offd_j,
                                      NALU_HYPRE_Complex *AFF_offd_a,
                                      NALU_HYPRE_Complex *beta_diag,
                                      NALU_HYPRE_Complex *beta_offd,
                                      NALU_HYPRE_Int     *F2_to_F,
                                      NALU_HYPRE_Real    *rsW )
{
   NALU_HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= AFF_nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int ib_diag = 0, ie_diag;
   NALU_HYPRE_Int rowF = 0;

   if (lane == 0)
   {
      rowF = read_only_load(&F2_to_F[row]);
   }
   rowF = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, rowF, 0);

   if (lane < 2)
   {
      ib_diag = read_only_load(AFF_diag_i + row + lane);
   }
   ie_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_diag, 1);
   ib_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_diag, 0);

   NALU_HYPRE_Complex rl = 0.0;

   for (NALU_HYPRE_Int i = ib_diag + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < ie_diag);
        i += NALU_HYPRE_WARP_SIZE)
   {
      if (i < ie_diag)
      {
         NALU_HYPRE_Int j = read_only_load(&AFF_diag_j[i]);

         if (j == rowF)
         {
            /* diagonal */
            AFF_diag_a[i] = 1.0;
         }
         else
         {
            /* off-diagonal */
            NALU_HYPRE_Complex beta = read_only_load(&beta_diag[j]);
            NALU_HYPRE_Complex val = AFF_diag_a[i];

            if (beta == 0.0)
            {
               rl += val;
               AFF_diag_a[i] = 0.0;
            }
            else
            {
               AFF_diag_a[i] = val / beta;
            }
         }
      }
   }

   NALU_HYPRE_Int ib_offd = 0, ie_offd;

   if (lane < 2)
   {
      ib_offd = read_only_load(AFF_offd_i + row + lane);
   }
   ie_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_offd, 1);
   ib_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_offd, 0);

   for (NALU_HYPRE_Int i = ib_offd + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < ie_offd);
        i += NALU_HYPRE_WARP_SIZE)
   {
      if (i < ie_offd)
      {
         NALU_HYPRE_Int j = read_only_load(&AFF_offd_j[i]);
         NALU_HYPRE_Complex beta = read_only_load(&beta_offd[j]);
         NALU_HYPRE_Complex val = AFF_offd_a[i];

         if (beta == 0.0)
         {
            rl += val;
            AFF_offd_a[i] = 0.0;
         }
         else
         {
            AFF_offd_a[i] = val / beta;
         }
      }
   }

   rl = warp_reduce_sum(item, rl);

   if (lane == 0)
   {
      rl += read_only_load(&rsW[row]);
      rl = rl == 0.0 ? 0.0 : -1.0 / rl;
   }

   rl = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, rl, 0);

   for (NALU_HYPRE_Int i = ib_diag + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < ie_diag);
        i += NALU_HYPRE_WARP_SIZE)
   {
      if (i < ie_diag)
      {
         AFF_diag_a[i] *= rl;
      }
   }

   for (NALU_HYPRE_Int i = ib_offd + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < ie_offd);
        i += NALU_HYPRE_WARP_SIZE)
   {
      if (i < ie_offd)
      {
         AFF_offd_a[i] *= rl;
      }
   }
}

//-----------------------------------------------------------------------
__global__
void hypreGPUKernel_MMPEInterpScaleAFF( hypre_DeviceItem    &item,
                                        NALU_HYPRE_Int      AFF_nrows,
                                        NALU_HYPRE_Int     *AFF_diag_i,
                                        NALU_HYPRE_Int     *AFF_diag_j,
                                        NALU_HYPRE_Complex *AFF_diag_a,
                                        NALU_HYPRE_Int     *AFF_offd_i,
                                        NALU_HYPRE_Int     *AFF_offd_j,
                                        NALU_HYPRE_Complex *AFF_offd_a,
                                        NALU_HYPRE_Complex *tmp_diag,
                                        NALU_HYPRE_Complex *tmp_offd,
                                        NALU_HYPRE_Complex *lam_diag,
                                        NALU_HYPRE_Complex *lam_offd,
                                        NALU_HYPRE_Int     *F2_to_F,
                                        NALU_HYPRE_Real    *rsW )
{
   NALU_HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= AFF_nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int ib_diag = 0, ie_diag;
   NALU_HYPRE_Int rowF = 0;

   if (lane == 0)
   {
      rowF = read_only_load(&F2_to_F[row]);
   }
   rowF = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, rowF, 0);

   if (lane < 2)
   {
      ib_diag = read_only_load(AFF_diag_i + row + lane);
   }
   ie_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_diag, 1);
   ib_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_diag, 0);

   NALU_HYPRE_Complex rl = 0.0;

   for (NALU_HYPRE_Int i = ib_diag + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < ie_diag);
        i += NALU_HYPRE_WARP_SIZE)
   {
      if (i < ie_diag)
      {
         NALU_HYPRE_Int j = read_only_load(&AFF_diag_j[i]);

         if (j == rowF)
         {
            /* diagonal */
            AFF_diag_a[i] = 1.0;
         }
         else
         {
            /* off-diagonal */
            NALU_HYPRE_Complex lam = read_only_load(&lam_diag[j]);
            NALU_HYPRE_Complex val = AFF_diag_a[i];

            if (lam == 0.0)
            {
               rl += val;
               AFF_diag_a[i] = 0.0;
            }
            else
            {
               rl += val * read_only_load(&tmp_diag[j]);
               AFF_diag_a[i] = val / lam;
            }
         }
      }
   }

   NALU_HYPRE_Int ib_offd = 0, ie_offd;

   if (lane < 2)
   {
      ib_offd = read_only_load(AFF_offd_i + row + lane);
   }
   ie_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_offd, 1);
   ib_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, ib_offd, 0);

   for (NALU_HYPRE_Int i = ib_offd + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < ie_offd);
        i += NALU_HYPRE_WARP_SIZE)
   {
      if (i < ie_offd)
      {
         NALU_HYPRE_Int j = read_only_load(&AFF_offd_j[i]);
         NALU_HYPRE_Complex lam = read_only_load(&lam_offd[j]);
         NALU_HYPRE_Complex val = AFF_offd_a[i];

         if (lam == 0.0)
         {
            rl += val;
            AFF_offd_a[i] = 0.0;
         }
         else
         {
            rl += val * read_only_load(&tmp_offd[j]);
            AFF_offd_a[i] = val / lam;
         }
      }
   }

   rl = warp_reduce_sum(item, rl);

   if (lane == 0)
   {
      rl += read_only_load(&rsW[row]);
      rl = rl == 0.0 ? 0.0 : -1.0 / rl;
   }

   rl = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, rl, 0);

   for (NALU_HYPRE_Int i = ib_diag + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < ie_diag);
        i += NALU_HYPRE_WARP_SIZE)
   {
      if (i < ie_diag)
      {
         AFF_diag_a[i] *= rl;
      }
   }

   for (NALU_HYPRE_Int i = ib_offd + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, i < ie_offd);
        i += NALU_HYPRE_WARP_SIZE)
   {
      if (i < ie_offd)
      {
         AFF_offd_a[i] *= rl;
      }
   }
}

#endif /* #if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL) */
