/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

/* TODO (VPM): Rename to hypreGPUKernel_. Also, do we need these prototypes? */

__global__ void nalu_hypre_BoomerAMGBuildDirInterp_getnnz( nalu_hypre_DeviceItem &item, NALU_HYPRE_Int nr_of_rows,
                                                      NALU_HYPRE_Int *S_diag_i,
                                                      NALU_HYPRE_Int *S_diag_j, NALU_HYPRE_Int *S_offd_i, NALU_HYPRE_Int *S_offd_j, NALU_HYPRE_Int *CF_marker,
                                                      NALU_HYPRE_Int *CF_marker_offd, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd,
                                                      NALU_HYPRE_Int *P_diag_i, NALU_HYPRE_Int *P_offd_i);

__global__ void nalu_hypre_BoomerAMGBuildDirInterp_getcoef( nalu_hypre_DeviceItem &item, NALU_HYPRE_Int nr_of_rows,
                                                       NALU_HYPRE_Int *A_diag_i,
                                                       NALU_HYPRE_Int *A_diag_j, NALU_HYPRE_Real *A_diag_data, NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j,
                                                       NALU_HYPRE_Real *A_offd_data, NALU_HYPRE_Int *Soc_diag_j, NALU_HYPRE_Int *Soc_offd_j, NALU_HYPRE_Int *CF_marker,
                                                       NALU_HYPRE_Int *CF_marker_offd, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd,
                                                       NALU_HYPRE_Int *P_diag_i, NALU_HYPRE_Int *P_diag_j, NALU_HYPRE_Real *P_diag_data, NALU_HYPRE_Int *P_offd_i,
                                                       NALU_HYPRE_Int *P_offd_j, NALU_HYPRE_Real *P_offd_data, NALU_HYPRE_Int *fine_to_coarse );

__global__ void nalu_hypre_BoomerAMGBuildDirInterp_getcoef_v2( nalu_hypre_DeviceItem &item,
                                                          NALU_HYPRE_Int nr_of_rows,
                                                          NALU_HYPRE_Int *A_diag_i,
                                                          NALU_HYPRE_Int *A_diag_j, NALU_HYPRE_Real *A_diag_data, NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j,
                                                          NALU_HYPRE_Real *A_offd_data, NALU_HYPRE_Int *Soc_diag_j, NALU_HYPRE_Int *Soc_offd_j, NALU_HYPRE_Int *CF_marker,
                                                          NALU_HYPRE_Int *CF_marker_offd, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd,
                                                          NALU_HYPRE_Int *P_diag_i, NALU_HYPRE_Int *P_diag_j, NALU_HYPRE_Real *P_diag_data, NALU_HYPRE_Int *P_offd_i,
                                                          NALU_HYPRE_Int *P_offd_j, NALU_HYPRE_Real *P_offd_data, NALU_HYPRE_Int *fine_to_coarse );

__global__ void
nalu_hypre_BoomerAMGBuildInterpOnePnt_getnnz( nalu_hypre_DeviceItem &item, NALU_HYPRE_Int nr_of_rows,
                                         NALU_HYPRE_Int *A_diag_i,
                                         NALU_HYPRE_Int *A_strong_diag_j, NALU_HYPRE_Complex *A_diag_a, NALU_HYPRE_Int *A_offd_i,
                                         NALU_HYPRE_Int *A_strong_offd_j, NALU_HYPRE_Complex *A_offd_a, NALU_HYPRE_Int *CF_marker,
                                         NALU_HYPRE_Int *CF_marker_offd, NALU_HYPRE_Int *diag_compress_marker, NALU_HYPRE_Int *offd_compress_marker,
                                         NALU_HYPRE_Int *P_diag_i, NALU_HYPRE_Int *P_diag_j, NALU_HYPRE_Int *P_offd_i, NALU_HYPRE_Int *P_offd_j);

/*---------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBuildDirInterp
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildDirInterpDevice( nalu_hypre_ParCSRMatrix   *A,
                                     NALU_HYPRE_Int            *CF_marker,
                                     nalu_hypre_ParCSRMatrix   *S,
                                     NALU_HYPRE_BigInt         *num_cpts_global,
                                     NALU_HYPRE_Int             num_functions,
                                     NALU_HYPRE_Int            *dof_func,
                                     NALU_HYPRE_Int             debug_flag,
                                     NALU_HYPRE_Real            trunc_factor,
                                     NALU_HYPRE_Int             max_elmts,
                                     NALU_HYPRE_Int             interp_type,
                                     nalu_hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm                comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;

   nalu_hypre_CSRMatrix *A_diag      = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i    = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j    = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd      = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i    = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j    = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int        num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   NALU_HYPRE_Int        n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_BoomerAMGMakeSocFromSDevice(A, S);

   nalu_hypre_CSRMatrix *S_diag   = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd   = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   NALU_HYPRE_Int       *Soc_diag_j = nalu_hypre_ParCSRMatrixSocDiagJ(S);
   NALU_HYPRE_Int       *Soc_offd_j = nalu_hypre_ParCSRMatrixSocOffdJ(S);

   NALU_HYPRE_Int       *CF_marker_offd = NULL;
   NALU_HYPRE_Int       *dof_func_offd = NULL;

   nalu_hypre_ParCSRMatrix *P;
   nalu_hypre_CSRMatrix *P_diag;
   nalu_hypre_CSRMatrix *P_offd;
   NALU_HYPRE_Real      *P_diag_data;
   NALU_HYPRE_Int       *P_diag_i;
   NALU_HYPRE_Int       *P_diag_j;
   NALU_HYPRE_Real      *P_offd_data;
   NALU_HYPRE_Int       *P_offd_i;
   NALU_HYPRE_Int       *P_offd_j;
   NALU_HYPRE_Int        P_diag_size, P_offd_size;

   NALU_HYPRE_Int       *fine_to_coarse_d;
   NALU_HYPRE_Int       *fine_to_coarse_h;
   NALU_HYPRE_BigInt     total_global_cpts;

   NALU_HYPRE_Int        my_id;
   NALU_HYPRE_Int        num_procs;
   NALU_HYPRE_Int        num_sends;
   NALU_HYPRE_Int       *int_buf_data;

   NALU_HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (my_id == (num_procs - 1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   nalu_hypre_MPI_Bcast( &total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }
   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds();
   }

   /* 1. Communicate CF_marker to/from other processors */
   if (num_cols_A_offd)
   {
      CF_marker_offd = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                               NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                                                       num_sends),
                     CF_marker,
                     int_buf_data );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                            num_sends),
                      CF_marker,
                      int_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, int_buf_data,
                                                 NALU_HYPRE_MEMORY_DEVICE, CF_marker_offd);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   if (num_functions > 1)
   {
      /* 2. Communicate dof_func to/from other processors */
      if (num_cols_A_offd > 0)
      {
         dof_func_offd = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
      }

#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                                                          num_sends),
                        dof_func,
                        int_buf_data );
#else
      NALU_HYPRE_THRUST_CALL( gather,
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                               num_sends),
                         dof_func,
                         int_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, int_buf_data,
                                                    NALU_HYPRE_MEMORY_DEVICE, dof_func_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n", my_id, wall_time);
      fflush(NULL);
   }

   /* 3. Figure out the size of the interpolation matrix, P, i.e., compute P_diag_i and P_offd_i */
   /*    Also, compute fine_to_coarse array: When i is a coarse point, fine_to_coarse[i] will hold a  */
   /*    corresponding coarse point index in the range 0..n_coarse-1 */
   P_diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fine + 1, memory_location);
   P_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fine + 1, memory_location);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(n_fine, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH( nalu_hypre_BoomerAMGBuildDirInterp_getnnz, gDim, bDim,
                     n_fine, S_diag_i, S_diag_j, S_offd_i, S_offd_j,
                     CF_marker, CF_marker_offd, num_functions,
                     dof_func, dof_func_offd, P_diag_i, P_offd_i);

   /* The scans will transform P_diag_i and P_offd_i to the CSR I-vectors */
   nalu_hypre_Memset(P_diag_i + n_fine, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_Memset(P_offd_i + n_fine, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_IntegerExclusiveScan(n_fine + 1, P_diag_i);
   hypreDevice_IntegerExclusiveScan(n_fine + 1, P_offd_i);

   fine_to_coarse_d = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);
   /* The scan will make fine_to_coarse[i] for i a coarse point hold a
    * coarse point index in the range from 0 to n_coarse-1 */
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,          is_nonnegative<NALU_HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_fine, is_nonnegative<NALU_HYPRE_Int>()),
                      fine_to_coarse_d,
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#else
   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,          is_nonnegative<NALU_HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_fine, is_nonnegative<NALU_HYPRE_Int>()),
                      fine_to_coarse_d,
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#endif

   /* 4. Compute the CSR arrays P_diag_j, P_diag_data, P_offd_j, and P_offd_data */
   /*    P_diag_i and P_offd_i are now known, first allocate the remaining CSR arrays of P */
   nalu_hypre_TMemcpy(&P_diag_size, &P_diag_i[n_fine], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, memory_location);
   nalu_hypre_TMemcpy(&P_offd_size, &P_offd_i[n_fine], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, memory_location);

   P_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  P_diag_size, memory_location);
   P_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, P_diag_size, memory_location);

   P_offd_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  P_offd_size, memory_location);
   P_offd_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, P_offd_size, memory_location);

   if (interp_type == 3)
   {
      NALU_HYPRE_GPU_LAUNCH( nalu_hypre_BoomerAMGBuildDirInterp_getcoef, gDim, bDim,
                        n_fine, A_diag_i, A_diag_j, A_diag_data,
                        A_offd_i, A_offd_j, A_offd_data,
                        Soc_diag_j,
                        Soc_offd_j,
                        CF_marker, CF_marker_offd,
                        num_functions, dof_func, dof_func_offd,
                        P_diag_i, P_diag_j, P_diag_data,
                        P_offd_i, P_offd_j, P_offd_data,
                        fine_to_coarse_d );
   }
   else
   {
      NALU_HYPRE_GPU_LAUNCH( nalu_hypre_BoomerAMGBuildDirInterp_getcoef_v2, gDim, bDim,
                        n_fine, A_diag_i, A_diag_j, A_diag_data,
                        A_offd_i, A_offd_j, A_offd_data,
                        Soc_diag_j,
                        Soc_offd_j,
                        CF_marker, CF_marker_offd,
                        num_functions, dof_func, dof_func_offd,
                        P_diag_i, P_diag_j, P_diag_data,
                        P_offd_i, P_offd_j, P_offd_data,
                        fine_to_coarse_d );
   }

   /* !!!! Free them here */
   /*
   nalu_hypre_TFree(nalu_hypre_ParCSRMatrixSocDiagJ(S), NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(nalu_hypre_ParCSRMatrixSocOffdJ(S), NALU_HYPRE_MEMORY_DEVICE);
   */

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL(std::replace, CF_marker, CF_marker + n_fine, -3, -1);
#else
   NALU_HYPRE_THRUST_CALL(replace, CF_marker, CF_marker + n_fine, -3, -1);
#endif

   /* 5. Construct the result as a ParCSRMatrix. At this point, P's column indices */
   /*    are defined with A's enumeration of columns */

   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                num_cols_A_offd,
                                P_diag_size,
                                P_offd_size);

   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_data;
   nalu_hypre_CSRMatrixI(P_diag)    = P_diag_i;
   nalu_hypre_CSRMatrixJ(P_diag)    = P_diag_j;

   P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_data;
   nalu_hypre_CSRMatrixI(P_offd)    = P_offd_i;
   nalu_hypre_CSRMatrixJ(P_offd)    = P_offd_j;

   nalu_hypre_CSRMatrixMemoryLocation(P_diag) = memory_location;
   nalu_hypre_CSRMatrixMemoryLocation(P_offd) = memory_location;

   /* 6. Compress P, removing coefficients smaller than trunc_factor * Max, and */
   /*    make sure no row has more than max_elmts elements */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts);
   }

   /* 7. Translate P_offd's column indices from the values inherited from A_offd to a 0,1,2,3,... enumeration, */
   /*    and construct the col_map array that translates these into the global 0..c-1 enumeration */

   /* Array P_marker has length equal to the number of A's offd columns+1, and will */
   /* store a translation code from A_offd's local column numbers to P_offd's local column numbers */
   NALU_HYPRE_Int *P_colids;
   NALU_HYPRE_Int *P_colids_h = NULL;

   nalu_hypre_CSRMatrixCompressColumnsDevice(P_offd, NULL, &P_colids, NULL);
   P_colids_h = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_CSRMatrixNumCols(P_offd), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(P_colids_h, P_colids, NALU_HYPRE_Int, nalu_hypre_CSRMatrixNumCols(P_offd),
                 NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(P_colids, NALU_HYPRE_MEMORY_DEVICE);

   /* 8. P_offd_j now has a 0,1,2,3... local column index enumeration. */
   /*    tmp_map_offd contains the index mapping from P's offd local columns to A's offd local columns.*/
   /*    Below routine is in parcsr_ls/par_rap_communication.c. It sets col_map_offd in P, */
   /*    comm_pkg in P, and perhaps more members of P ??? */

   fine_to_coarse_h = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(fine_to_coarse_h, fine_to_coarse_d, NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST,
                 NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_ParCSRMatrixColMapOffd(P) = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nalu_hypre_CSRMatrixNumCols(P_offd),
                                                   NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_GetCommPkgRTFromCommPkgA(P, A, fine_to_coarse_h, P_colids_h);

   *P_ptr = P;

   nalu_hypre_TFree(CF_marker_offd,   NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(dof_func_offd,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(int_buf_data,     NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(fine_to_coarse_d, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(fine_to_coarse_h, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(P_colids_h,       NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}


/*-----------------------------------------------------------------------*/
__global__ void
nalu_hypre_BoomerAMGBuildDirInterp_getnnz( nalu_hypre_DeviceItem &item,
                                      NALU_HYPRE_Int  nr_of_rows,
                                      NALU_HYPRE_Int *S_diag_i,
                                      NALU_HYPRE_Int *S_diag_j,
                                      NALU_HYPRE_Int *S_offd_i,
                                      NALU_HYPRE_Int *S_offd_j,
                                      NALU_HYPRE_Int *CF_marker,
                                      NALU_HYPRE_Int *CF_marker_offd,
                                      NALU_HYPRE_Int  num_functions,
                                      NALU_HYPRE_Int *dof_func,
                                      NALU_HYPRE_Int *dof_func_offd,
                                      NALU_HYPRE_Int *P_diag_i,
                                      NALU_HYPRE_Int *P_offd_i)
{
   /*-----------------------------------------------------------------------*/
   /* Determine size of interpolation matrix, P

      If A is of size m x m, then P will be of size m x c where c is the
      number of coarse points.

      It is assumed that S have the same global column enumeration as A

      Input: nr_of_rows         - Number of rows in matrix (local in processor)
             S_diag_i, S_diag_j - CSR representation of S_diag
             S_offd_i, S_offd_j - CSR representation of S_offd
             num_function  - Number of degrees of freedom per grid point
             dof_func      - vector of length nr_of_rows, indicating the degree of freedom of vector element.
             dof_func_offd - vector over ncols of A_offd, indicating the degree of freedom.

      Output: P_diag_i       - Vector where P_diag_i[i] holds the number of non-zero elements of P_diag on row i.
              P_offd_i       - Vector where P_offd_i[i] holds the number of non-zero elements of P_offd on row i.
              fine_to_coarse - Vector of length nr_of_rows.
                               fine_to_coarse[i] is set to 1 if i is a coarse pt.
                               Eventually, fine_to_coarse[j] will map A's column j
                               to a re-enumerated column index in matrix P.
    */
   /*-----------------------------------------------------------------------*/

   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int p = 0, q = 0, dof_func_i = 0;
   NALU_HYPRE_Int jPd = 0, jPo = 0;
   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);

   if (lane == 0)
   {
      p = read_only_load(CF_marker + i);
   }
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   /*--------------------------------------------------------------------
    *  If i is a C-point, interpolation is the identity.
    *--------------------------------------------------------------------*/
   if (p >= 0)
   {
      if (lane == 0)
      {
         P_diag_i[i] = 1;
         P_offd_i[i] = 0;
      }
      return;
   }

   /*--------------------------------------------------------------------
    *  If i is an F-point, interpolation is from the C-points that
    *  strongly influence i.
    *--------------------------------------------------------------------*/
   if (num_functions > 1 && dof_func != NULL)
   {
      if (lane == 0)
      {
         dof_func_i = read_only_load(&dof_func[i]);
      }
      dof_func_i = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, dof_func_i, 0);
   }

   /* diag part */
   if (lane < 2)
   {
      p = read_only_load(S_diag_i + i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q); j += NALU_HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         const NALU_HYPRE_Int col = read_only_load(&S_diag_j[j]);
         if ( read_only_load(&CF_marker[col]) > 0 && (num_functions == 1 ||
                                                      read_only_load(&dof_func[col]) == dof_func_i) )
         {
            jPd++;
         }
      }
   }
   jPd = warp_reduce_sum(item, jPd);

   /* offd part */
   if (lane < 2)
   {
      p = read_only_load(S_offd_i + i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q); j += NALU_HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         const NALU_HYPRE_Int tmp = read_only_load(&S_offd_j[j]);
         const NALU_HYPRE_Int col = tmp;
         if ( read_only_load(&CF_marker_offd[col]) > 0 && (num_functions == 1 ||
                                                           read_only_load(&dof_func_offd[col]) == dof_func_i) )
         {
            jPo++;
         }
      }
   }
   jPo = warp_reduce_sum(item, jPo);

   if (lane == 0)
   {
      P_diag_i[i] = jPd;
      P_offd_i[i] = jPo;
   }
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
__global__ void
nalu_hypre_BoomerAMGBuildDirInterp_getcoef( nalu_hypre_DeviceItem &item,
                                       NALU_HYPRE_Int   nr_of_rows,
                                       NALU_HYPRE_Int  *A_diag_i,
                                       NALU_HYPRE_Int  *A_diag_j,
                                       NALU_HYPRE_Real *A_diag_data,
                                       NALU_HYPRE_Int  *A_offd_i,
                                       NALU_HYPRE_Int  *A_offd_j,
                                       NALU_HYPRE_Real *A_offd_data,
                                       NALU_HYPRE_Int  *Soc_diag_j,
                                       NALU_HYPRE_Int  *Soc_offd_j,
                                       NALU_HYPRE_Int  *CF_marker,
                                       NALU_HYPRE_Int  *CF_marker_offd,
                                       NALU_HYPRE_Int   num_functions,
                                       NALU_HYPRE_Int  *dof_func,
                                       NALU_HYPRE_Int  *dof_func_offd,
                                       NALU_HYPRE_Int  *P_diag_i,
                                       NALU_HYPRE_Int  *P_diag_j,
                                       NALU_HYPRE_Real *P_diag_data,
                                       NALU_HYPRE_Int  *P_offd_i,
                                       NALU_HYPRE_Int  *P_offd_j,
                                       NALU_HYPRE_Real *P_offd_data,
                                       NALU_HYPRE_Int  *fine_to_coarse )
{
   /*-----------------------------------------------------------------------*/
   /* Compute interpolation matrix, P

      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_i, A_diag_j, A_diag_data - CSR representation of A_diag
             A_offd_i, A_offd_j, A_offd_data - CSR representation of A_offd
             S_diag_i, S_diag_j - CSR representation of S_diag
             S_offd_i, S_offd_j - CSR representation of S_offd
             CF_marker          - Coarse/Fine flags for indices (rows) in this processor
             CF_marker_offd     - Coarse/Fine flags for indices (rows) not in this processor
             num_function  - Number of degrees of freedom per grid point
             dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
             dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom
             fine_to_coarse - Vector of length nr_of_rows-1.

      Output: P_diag_j         - Column indices in CSR representation of P_diag
              P_diag_data      - Matrix elements in CSR representation of P_diag
              P_offd_j         - Column indices in CSR representation of P_offd
              P_offd_data      - Matrix elements in CSR representation of P_diag
   */
   /*-----------------------------------------------------------------------*/

   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);

   NALU_HYPRE_Int k = 0, dof_func_i = 0;

   if (lane == 0)
   {
      k = read_only_load(CF_marker + i);
   }
   k = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, k, 0);

   /*--------------------------------------------------------------------
    *  If i is a C-point, interpolation is the identity.
    *--------------------------------------------------------------------*/
   if (k > 0)
   {
      if (lane == 0)
      {
         const NALU_HYPRE_Int ind = read_only_load(&P_diag_i[i]);
         P_diag_j[ind]       = read_only_load(&fine_to_coarse[i]);
         P_diag_data[ind]    = 1.0;
      }

      return;
   }

   /*--------------------------------------------------------------------
    *  Point is f-point, use direct interpolation
    *--------------------------------------------------------------------*/
   if (num_functions > 1 && dof_func != NULL)
   {
      if (lane == 0)
      {
         dof_func_i = read_only_load(&dof_func[i]);
      }
      dof_func_i = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, dof_func_i, 0);
   }

   NALU_HYPRE_Real diagonal = 0.0, sum_N_pos = 0.0, sum_N_neg = 0.0, sum_P_pos = 0.0, sum_P_neg = 0.0;

   /* diag part */
   NALU_HYPRE_Int p_diag_A = 0, q_diag_A, p_diag_P = 0, q_diag_P;
   if (lane < 2)
   {
      p_diag_A = read_only_load(A_diag_i + i + lane);
      p_diag_P = read_only_load(P_diag_i + i + lane);
   }
   q_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 1);
   p_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 0);
   q_diag_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_P, 1);
   p_diag_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_P, 0);

   k = p_diag_P;
   for (NALU_HYPRE_Int j = p_diag_A + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_diag_A);
        j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int col, sum, pos;
      NALU_HYPRE_Int is_SC = 0; /* if is a Strong-C */
      NALU_HYPRE_Complex val;

      if (j < q_diag_A)
      {
         col = read_only_load(&A_diag_j[j]);

         if (i == col)
         {
            diagonal = read_only_load(&A_diag_data[j]);
         }
         else if ( num_functions == 1 || read_only_load(&dof_func[col]) == dof_func_i )
         {
            val = read_only_load(&A_diag_data[j]);

            if (val > 0.0)
            {
               sum_N_pos += val;
            }
            else
            {
               sum_N_neg += val;
            }

            is_SC = read_only_load(&Soc_diag_j[j]) > -1 && read_only_load(&CF_marker[col]) > 0;

            if (is_SC)
            {
               if (val > 0.0)
               {
                  sum_P_pos += val;
               }
               else
               {
                  sum_P_neg += val;
               }
            }
         }
      }

      pos = warp_prefix_sum(item, lane, is_SC, sum);

      if (is_SC)
      {
         P_diag_data[k + pos] = val;
         P_diag_j[k + pos] = read_only_load(&fine_to_coarse[col]);
      }
      k += sum;
   }

   nalu_hypre_device_assert(k == q_diag_P);

   /* offd part */
   NALU_HYPRE_Int p_offd_A = 0, q_offd_A, p_offd_P = 0, q_offd_P;
   if (lane < 2)
   {
      p_offd_A = read_only_load(A_offd_i + i + lane);
      p_offd_P = read_only_load(P_offd_i + i + lane);
   }
   q_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 1);
   p_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 0);
   q_offd_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_P, 1);
   p_offd_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_P, 0);

   k = p_offd_P;
   for (NALU_HYPRE_Int j = p_offd_A + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_offd_A);
        j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int col, sum, pos;
      NALU_HYPRE_Int is_SC = 0; /* if is a Strong-C */
      NALU_HYPRE_Complex val;

      if (j < q_offd_A)
      {
         col = read_only_load(&A_offd_j[j]);

         if ( num_functions == 1 || read_only_load(&dof_func_offd[col]) == dof_func_i )
         {
            val = read_only_load(&A_offd_data[j]);

            if (val > 0.0)
            {
               sum_N_pos += val;
            }
            else
            {
               sum_N_neg += val;
            }

            is_SC = read_only_load(&Soc_offd_j[j]) > -1 && read_only_load(&CF_marker_offd[col]) > 0;

            if (is_SC)
            {
               if (val > 0.0)
               {
                  sum_P_pos += val;
               }
               else
               {
                  sum_P_neg += val;
               }
            }
         }
      }

      pos = warp_prefix_sum(item, lane, is_SC, sum);

      if (is_SC)
      {
         P_offd_data[k + pos] = val;
         P_offd_j[k + pos] = col;
      }
      k += sum;
   }

   nalu_hypre_device_assert(k == q_offd_P);

   diagonal  = warp_allreduce_sum(item, diagonal);
   sum_N_pos = warp_allreduce_sum(item, sum_N_pos);
   sum_N_neg = warp_allreduce_sum(item, sum_N_neg);
   sum_P_pos = warp_allreduce_sum(item, sum_P_pos);
   sum_P_neg = warp_allreduce_sum(item, sum_P_neg);

   NALU_HYPRE_Complex alfa = 1.0, beta = 1.0;

   if (sum_P_neg)
   {
      alfa = sum_N_neg / (sum_P_neg * diagonal);
   }

   if (sum_P_pos)
   {
      beta = sum_N_pos / (sum_P_pos * diagonal);
   }

   for (NALU_HYPRE_Int j = p_diag_P + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_diag_P);
        j += NALU_HYPRE_WARP_SIZE)
   {
      /* if (P_diag_data[j] > 0.0)
            P_diag_data[j] *= -beta;
         else
            P_diag_data[j] *= -alfa; */
      if (j < q_diag_P)
      {
         P_diag_data[j] *= (P_diag_data[j] > 0.0) * (alfa - beta) - alfa;
      }
   }

   for (NALU_HYPRE_Int j = p_offd_P + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_offd_P);
        j += NALU_HYPRE_WARP_SIZE)
   {
      /* if (P_offd_data[indp]> 0)
            P_offd_data[indp] *= -beta;
         else
            P_offd_data[indp] *= -alfa; */
      if (j < q_offd_P)
      {
         P_offd_data[j] *= (P_offd_data[j] > 0.0) * (alfa - beta) - alfa;
      }
   }
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
__global__ void
nalu_hypre_BoomerAMGBuildDirInterp_getcoef_v2( nalu_hypre_DeviceItem &item,
                                          NALU_HYPRE_Int   nr_of_rows,
                                          NALU_HYPRE_Int  *A_diag_i,
                                          NALU_HYPRE_Int  *A_diag_j,
                                          NALU_HYPRE_Real *A_diag_data,
                                          NALU_HYPRE_Int  *A_offd_i,
                                          NALU_HYPRE_Int  *A_offd_j,
                                          NALU_HYPRE_Real *A_offd_data,
                                          NALU_HYPRE_Int  *Soc_diag_j,
                                          NALU_HYPRE_Int  *Soc_offd_j,
                                          NALU_HYPRE_Int  *CF_marker,
                                          NALU_HYPRE_Int  *CF_marker_offd,
                                          NALU_HYPRE_Int   num_functions,
                                          NALU_HYPRE_Int  *dof_func,
                                          NALU_HYPRE_Int  *dof_func_offd,
                                          NALU_HYPRE_Int  *P_diag_i,
                                          NALU_HYPRE_Int  *P_diag_j,
                                          NALU_HYPRE_Real *P_diag_data,
                                          NALU_HYPRE_Int  *P_offd_i,
                                          NALU_HYPRE_Int  *P_offd_j,
                                          NALU_HYPRE_Real *P_offd_data,
                                          NALU_HYPRE_Int  *fine_to_coarse )
{
   /*-----------------------------------------------------------------------*/
   /* Compute interpolation matrix, P

      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_i, A_diag_j, A_diag_data - CSR representation of A_diag
             A_offd_i, A_offd_j, A_offd_data - CSR representation of A_offd
             S_diag_i, S_diag_j - CSR representation of S_diag
             S_offd_i, S_offd_j - CSR representation of S_offd
             CF_marker          - Coarse/Fine flags for indices (rows) in this processor
             CF_marker_offd     - Coarse/Fine flags for indices (rows) not in this processor
             num_function  - Number of degrees of freedom per grid point
             dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
             dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom
             fine_to_coarse - Vector of length nr_of_rows-1.

      Output: P_diag_j         - Column indices in CSR representation of P_diag
              P_diag_data      - Matrix elements in CSR representation of P_diag
              P_offd_j         - Column indices in CSR representation of P_offd
              P_offd_data      - Matrix elements in CSR representation of P_diag
   */
   /*-----------------------------------------------------------------------*/

   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);

   NALU_HYPRE_Int k = 0, dof_func_i = 0;

   if (lane == 0)
   {
      k = read_only_load(CF_marker + i);
   }
   k = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, k, 0);

   /*--------------------------------------------------------------------
    *  If i is a C-point, interpolation is the identity.
    *--------------------------------------------------------------------*/
   if (k > 0)
   {
      if (lane == 0)
      {
         const NALU_HYPRE_Int ind = read_only_load(&P_diag_i[i]);
         P_diag_j[ind]       = read_only_load(&fine_to_coarse[i]);
         P_diag_data[ind]    = 1.0;
      }

      return;
   }

   /*--------------------------------------------------------------------
    *  Point is f-point, use direct interpolation
    *--------------------------------------------------------------------*/
   if (num_functions > 1 && dof_func != NULL)
   {
      if (lane == 0)
      {
         dof_func_i = read_only_load(&dof_func[i]);
      }
      dof_func_i = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, dof_func_i, 0);
   }

   NALU_HYPRE_Real diagonal = 0.0, sum_F = 0.0;

   /* diag part */
   NALU_HYPRE_Int p_diag_A = 0, q_diag_A, p_diag_P = 0, q_diag_P;
   if (lane < 2)
   {
      p_diag_A = read_only_load(A_diag_i + i + lane);
      p_diag_P = read_only_load(P_diag_i + i + lane);
   }
   q_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 1);
   p_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 0);
   q_diag_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_P, 1);
   p_diag_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_P, 0);

   k = p_diag_P;
   for (NALU_HYPRE_Int j = p_diag_A + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_diag_A);
        j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int col, sum, pos;
      NALU_HYPRE_Int is_SC = 0; /* if is a Strong-C */
      NALU_HYPRE_Complex val;

      if (j < q_diag_A)
      {
         col = read_only_load(&A_diag_j[j]);

         if (i == col)
         {
            diagonal = read_only_load(&A_diag_data[j]);
         }
         else if ( num_functions == 1 || read_only_load(&dof_func[col]) == dof_func_i )
         {
            val = read_only_load(&A_diag_data[j]);
            if (read_only_load(&Soc_diag_j[j]) > -1)
            {
               if (read_only_load(&CF_marker[col]) > 0)
               {
                  is_SC = 1;
               }
               else
               {
                  sum_F += val;
               }
            }
            else
            {
               diagonal += val;
            }
         }
      }

      pos = warp_prefix_sum(item, lane, is_SC, sum);

      if (is_SC)
      {
         P_diag_data[k + pos] = val;
         P_diag_j[k + pos] = read_only_load(&fine_to_coarse[col]);
      }
      k += sum;
   }

   nalu_hypre_device_assert(k == q_diag_P);

   /* offd part */
   NALU_HYPRE_Int p_offd_A = 0, q_offd_A, p_offd_P = 0, q_offd_P;
   if (lane < 2)
   {
      p_offd_A = read_only_load(A_offd_i + i + lane);
      p_offd_P = read_only_load(P_offd_i + i + lane);
   }
   q_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 1);
   p_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 0);
   q_offd_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_P, 1);
   p_offd_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_P, 0);

   k = p_offd_P;
   for (NALU_HYPRE_Int j = p_offd_A + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_offd_A);
        j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int col, sum, pos;
      NALU_HYPRE_Int is_SC = 0; /* if is a Strong-C */
      NALU_HYPRE_Complex val;

      if (j < q_offd_A)
      {
         col = read_only_load(&A_offd_j[j]);

         if ( num_functions == 1 || read_only_load(&dof_func_offd[col]) == dof_func_i )
         {
            val = read_only_load(&A_offd_data[j]);
            if (read_only_load(&Soc_offd_j[j]) > -1)
            {
               if (read_only_load(&CF_marker_offd[col]) > 0)
               {
                  is_SC = 1;
               }
               else
               {
                  sum_F += val;
               }
            }
            else
            {
               diagonal += val;
            }
         }
      }

      pos = warp_prefix_sum(item, lane, is_SC, sum);

      if (is_SC)
      {
         P_offd_data[k + pos] = val;
         P_offd_j[k + pos] = col;
      }
      k += sum;
   }

   nalu_hypre_device_assert(k == q_offd_P);

   diagonal  = warp_allreduce_sum(item, diagonal);
   sum_F     = warp_allreduce_sum(item, sum_F);

   NALU_HYPRE_Complex beta = sum_F / (q_diag_P - p_diag_P + q_offd_P - p_offd_P);

   for (NALU_HYPRE_Int j = p_diag_P + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_diag_P);
        j += NALU_HYPRE_WARP_SIZE)
   {
      /* if (P_diag_data[j] > 0.0)
            P_diag_data[j] *= -beta;
         else
            P_diag_data[j] *= -alfa; */
      if (j < q_diag_P)
      {
         P_diag_data[j] = -(P_diag_data[j] + beta) / diagonal;
      }
   }

   for (NALU_HYPRE_Int j = p_offd_P + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_offd_P);
        j += NALU_HYPRE_WARP_SIZE)
   {
      /* if (P_offd_data[indp]> 0)
            P_offd_data[indp] *= -beta;
         else
            P_offd_data[indp] *= -alfa; */
      if (j < q_offd_P)
      {
         P_offd_data[j] = -(P_offd_data[j] + beta) / diagonal;
      }
   }
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildInterpOnePntDevice( nalu_hypre_ParCSRMatrix  *A,
                                        NALU_HYPRE_Int           *CF_marker,
                                        nalu_hypre_ParCSRMatrix  *S,
                                        NALU_HYPRE_BigInt        *num_cpts_global,
                                        NALU_HYPRE_Int            num_functions,
                                        NALU_HYPRE_Int           *dof_func,
                                        NALU_HYPRE_Int            debug_flag,
                                        nalu_hypre_ParCSRMatrix **P_ptr)
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   nalu_hypre_CSRMatrix         *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int               *A_diag_i        = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int               *A_strong_diag_j = nalu_hypre_ParCSRMatrixSocDiagJ(S);
   NALU_HYPRE_Complex           *A_diag_a        = nalu_hypre_CSRMatrixData(A_diag);

   nalu_hypre_CSRMatrix         *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int               *A_offd_i        = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int               *A_strong_offd_j = nalu_hypre_ParCSRMatrixSocOffdJ(S);
   NALU_HYPRE_Complex           *A_offd_a        = nalu_hypre_CSRMatrixData(A_offd);

   NALU_HYPRE_Int                num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   /* Interpolation matrix P */
   nalu_hypre_ParCSRMatrix      *P;
   /* csr's */
   nalu_hypre_CSRMatrix    *P_diag;
   nalu_hypre_CSRMatrix    *P_offd;
   /* arrays */
   NALU_HYPRE_Real         *P_diag_data;
   NALU_HYPRE_Int          *P_diag_i;
   NALU_HYPRE_Int          *P_diag_j;
   NALU_HYPRE_Int          *P_diag_j_temp;
   NALU_HYPRE_Int          *P_diag_j_temp_compressed;
   NALU_HYPRE_Real         *P_offd_data;
   NALU_HYPRE_Int          *P_offd_i;
   NALU_HYPRE_Int          *P_offd_j;
   NALU_HYPRE_Int          *P_offd_j_temp;
   NALU_HYPRE_Int          *P_offd_j_temp_compressed;
   NALU_HYPRE_Int           num_cols_P_offd;
   NALU_HYPRE_BigInt       *col_map_offd_P = NULL;
   NALU_HYPRE_BigInt       *col_map_offd_P_device = NULL;
   /* CF marker off-diag part */
   NALU_HYPRE_Int          *CF_marker_offd = NULL;
   /* nnz */
   NALU_HYPRE_Int           nnz_diag, nnz_offd;
   /* local size */
   NALU_HYPRE_Int           n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   /* fine to coarse mapping: diag part and offd part */
   NALU_HYPRE_Int          *fine_to_coarse;
   NALU_HYPRE_BigInt       *fine_to_coarse_offd = NULL;
   NALU_HYPRE_BigInt        total_global_cpts, my_first_cpt;
   NALU_HYPRE_Int           my_id, num_procs;
   NALU_HYPRE_Int           num_sends;
   NALU_HYPRE_Int          *int_buf_data = NULL;
   NALU_HYPRE_BigInt       *big_int_buf_data = NULL;
   //NALU_HYPRE_Int col_start = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   //NALU_HYPRE_Int col_end   = col_start + n_fine;
   /* arrays for compressing P_diag and P_offd col indices and data */
   NALU_HYPRE_Int          *diag_compress_marker;
   NALU_HYPRE_Int          *offd_compress_marker;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /* fine to coarse mapping */
   fine_to_coarse = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,          is_nonnegative<NALU_HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_fine, is_nonnegative<NALU_HYPRE_Int>()),
                      fine_to_coarse,
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#else
   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,          is_nonnegative<NALU_HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_fine, is_nonnegative<NALU_HYPRE_Int>()),
                      fine_to_coarse,
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#endif

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   if (num_cols_A_offd)
   {
      CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
   }
   /* if CommPkg of A is not present, create it */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }
   /* number of sends to do (number of procs) */
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* send buffer, of size send_map_starts[num_sends]),
    * i.e., number of entries to send */
   int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                NALU_HYPRE_MEMORY_DEVICE);

   /* copy CF markers of elements to send to buffer */
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                     nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     CF_marker,
                     int_buf_data );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                      nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                      CF_marker,
                      int_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   /* create a handle to start communication. 11: for integer */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, int_buf_data,
                                                 NALU_HYPRE_MEMORY_DEVICE, CF_marker_offd);
   /* destroy the handle to finish communication */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_DEVICE);

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping,
    *  and find the most strongly influencing C-pt for each F-pt
    *-----------------------------------------------------------------------*/

   P_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, NALU_HYPRE_MEMORY_DEVICE);

   diag_compress_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);
   offd_compress_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);

   /* Overallocate here and compress later */
   P_diag_j_temp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_j_temp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(n_fine, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH( nalu_hypre_BoomerAMGBuildInterpOnePnt_getnnz, gDim, bDim,
                     n_fine, A_diag_i, A_strong_diag_j, A_diag_a, A_offd_i, A_strong_offd_j,
                     A_offd_a, CF_marker, CF_marker_offd, diag_compress_marker,
                     offd_compress_marker, P_diag_i, P_diag_j_temp, P_offd_i, P_offd_j_temp);

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/
   fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
   big_int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                    NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                     nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     fine_to_coarse,
                     big_int_buf_data );
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      big_int_buf_data,
                      big_int_buf_data + nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                      big_int_buf_data,
   [my_first_cpt = my_first_cpt] (const auto & x) { return x + my_first_cpt; } );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                      nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                      fine_to_coarse,
                      big_int_buf_data );
   NALU_HYPRE_THRUST_CALL( transform,
                      big_int_buf_data,
                      big_int_buf_data + nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                      thrust::make_constant_iterator(my_first_cpt),
                      big_int_buf_data,
                      thrust::plus<NALU_HYPRE_BigInt>() );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure big_int_buf_data is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, big_int_buf_data,
                                                 NALU_HYPRE_MEMORY_DEVICE, fine_to_coarse_offd);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   nalu_hypre_TFree(big_int_buf_data, NALU_HYPRE_MEMORY_DEVICE);

   /*-----------------------------------------------------------------------
    *  Fill values and finish setting up P.
    *-----------------------------------------------------------------------*/

   /* scan P_diag_i (which has number of nonzeros in each row) to get row indices */
   hypreDevice_IntegerExclusiveScan(n_fine + 1, P_diag_i);
   hypreDevice_IntegerExclusiveScan(n_fine + 1, P_offd_i);

   /* get the number of nonzeros and allocate column index and data arrays */
   nalu_hypre_TMemcpy(&nnz_diag, &P_diag_i[n_fine], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(&nnz_offd, &P_offd_i[n_fine], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

   P_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_diag, NALU_HYPRE_MEMORY_DEVICE);


   P_offd_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_offd, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_offd, NALU_HYPRE_MEMORY_DEVICE);

   /* set data values to 1.0 */
   hypreDevice_ComplexFilln( P_diag_data, nnz_diag, 1.0 );
   hypreDevice_ComplexFilln( P_offd_data, nnz_offd, 1.0 );

   /* compress temporary column indices */
   P_diag_j_temp_compressed = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_diag, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_j_temp_compressed = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_offd, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_copy_if( P_diag_j_temp,
                      P_diag_j_temp + n_fine,
                      diag_compress_marker,
                      P_diag_j_temp_compressed,
                      equal<NALU_HYPRE_Int>(1) );
   hypreSycl_copy_if( P_offd_j_temp,
                      P_offd_j_temp + n_fine,
                      offd_compress_marker,
                      P_offd_j_temp_compressed,
                      equal<NALU_HYPRE_Int>(1) );

   /* map the diag column indices */
   hypreSycl_gather( P_diag_j_temp_compressed,
                     P_diag_j_temp_compressed + nnz_diag,
                     fine_to_coarse,
                     P_diag_j );
#else
   NALU_HYPRE_THRUST_CALL( copy_if,
                      P_diag_j_temp,
                      P_diag_j_temp + n_fine,
                      diag_compress_marker,
                      P_diag_j_temp_compressed,
                      equal<NALU_HYPRE_Int>(1) );
   NALU_HYPRE_THRUST_CALL( copy_if,
                      P_offd_j_temp,
                      P_offd_j_temp + n_fine,
                      offd_compress_marker,
                      P_offd_j_temp_compressed,
                      equal<NALU_HYPRE_Int>(1) );

   /* map the diag column indices */
   NALU_HYPRE_THRUST_CALL( gather,
                      P_diag_j_temp_compressed,
                      P_diag_j_temp_compressed + nnz_diag,
                      fine_to_coarse,
                      P_diag_j );
#endif

   nalu_hypre_TFree(P_diag_j_temp_compressed, NALU_HYPRE_MEMORY_DEVICE);

   /* mark the offd indices for P as a subset of offd indices of A */
   NALU_HYPRE_Int *mark_P_offd_idx = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
   // note that scatter is usually not safe if the same index appears more than once in the map,
   // but here we are just scattering constant values, so this is safe
#if defined(NALU_HYPRE_USING_SYCL)
   auto perm_iter = oneapi::dpl::make_permutation_iterator(mark_P_offd_idx, P_offd_j_temp_compressed);
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      perm_iter,
                      perm_iter + nnz_offd,
                      perm_iter,
   [] (const auto & x) { return 1; } );
   num_cols_P_offd = NALU_HYPRE_ONEDPL_CALL(std::reduce, mark_P_offd_idx,
                                       mark_P_offd_idx + num_cols_A_offd);
#else
   NALU_HYPRE_THRUST_CALL( scatter,
                      thrust::make_constant_iterator(1),
                      thrust::make_constant_iterator(1) + nnz_offd,
                      P_offd_j_temp_compressed,
                      mark_P_offd_idx );
   num_cols_P_offd = NALU_HYPRE_THRUST_CALL(reduce, mark_P_offd_idx, mark_P_offd_idx + num_cols_A_offd);
#endif

   /* get a mapping from P offd indices to A offd indices */
   /* offd_map_P_to_A[ P offd idx ] = A offd idx */
   NALU_HYPRE_Int *offd_map_P_to_A = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_P_offd, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   oneapi::dpl::counting_iterator<NALU_HYPRE_Int> count(0);
   hypreSycl_copy_if( count,
                      count + num_cols_A_offd,
                      mark_P_offd_idx,
                      offd_map_P_to_A,
                      equal<NALU_HYPRE_Int>(1) );
#else
   NALU_HYPRE_THRUST_CALL( copy_if,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(num_cols_A_offd),
                      mark_P_offd_idx,
                      offd_map_P_to_A,
                      equal<NALU_HYPRE_Int>(1) );
#endif
   nalu_hypre_TFree(mark_P_offd_idx, NALU_HYPRE_MEMORY_DEVICE);

   /* also get an inverse mapping from A offd indices to P offd indices */
   /* offd_map_A_to_P[ A offd idx ] = -1 if not a P idx, else P offd idx */
   NALU_HYPRE_Int *offd_map_A_to_P = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
   hypreDevice_IntFilln( offd_map_A_to_P, num_cols_A_offd, -1 );

#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_scatter( count,
                      count + num_cols_P_offd,
                      offd_map_P_to_A,
                      offd_map_A_to_P );

   /* use inverse mapping above to map P_offd_j */
   hypreSycl_gather( P_offd_j_temp_compressed,
                     P_offd_j_temp_compressed + nnz_offd,
                     offd_map_A_to_P,
                     P_offd_j );
#else
   NALU_HYPRE_THRUST_CALL( scatter,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(num_cols_P_offd),
                      offd_map_P_to_A,
                      offd_map_A_to_P );

   /* use inverse mapping above to map P_offd_j */
   NALU_HYPRE_THRUST_CALL( gather,
                      P_offd_j_temp_compressed,
                      P_offd_j_temp_compressed + nnz_offd,
                      offd_map_A_to_P,
                      P_offd_j );
#endif
   nalu_hypre_TFree(P_offd_j_temp_compressed, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(offd_map_A_to_P, NALU_HYPRE_MEMORY_DEVICE);

   /* setup col_map_offd for P */
   col_map_offd_P_device = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_P_offd, NALU_HYPRE_MEMORY_DEVICE);
   col_map_offd_P = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_P_offd, NALU_HYPRE_MEMORY_HOST);
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( offd_map_P_to_A,
                     offd_map_P_to_A + num_cols_P_offd,
                     fine_to_coarse_offd,
                     col_map_offd_P_device);
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      offd_map_P_to_A,
                      offd_map_P_to_A + num_cols_P_offd,
                      fine_to_coarse_offd,
                      col_map_offd_P_device);
#endif
   nalu_hypre_TMemcpy(col_map_offd_P, col_map_offd_P_device, NALU_HYPRE_BigInt, num_cols_P_offd,
                 NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(offd_map_P_to_A, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(col_map_offd_P_device, NALU_HYPRE_MEMORY_DEVICE);

   /* Now, we should have everything of Parcsr matrix P */
   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumCols(A), /* global num of rows */
                                total_global_cpts, /* global num of cols */
                                nalu_hypre_ParCSRMatrixColStarts(A), /* row_starts */
                                num_cpts_global, /* col_starts */
                                num_cols_P_offd, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_data;
   nalu_hypre_CSRMatrixI(P_diag)    = P_diag_i;
   nalu_hypre_CSRMatrixJ(P_diag)    = P_diag_j;

   P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_data;
   nalu_hypre_CSRMatrixI(P_offd)    = P_offd_i;
   nalu_hypre_CSRMatrixJ(P_offd)    = P_offd_j;

   nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;

   /* create CommPkg of P */
   nalu_hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   /* free workspace */
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(diag_compress_marker, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(offd_compress_marker, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(P_diag_j_temp, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(P_offd_j_temp, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

/*-----------------------------------------------------------------------*/
__global__ void
nalu_hypre_BoomerAMGBuildInterpOnePnt_getnnz( nalu_hypre_DeviceItem    &item,
                                         NALU_HYPRE_Int      nr_of_rows,
                                         NALU_HYPRE_Int     *A_diag_i,
                                         NALU_HYPRE_Int     *A_strong_diag_j,
                                         NALU_HYPRE_Complex *A_diag_a,
                                         NALU_HYPRE_Int     *A_offd_i,
                                         NALU_HYPRE_Int     *A_strong_offd_j,
                                         NALU_HYPRE_Complex *A_offd_a,
                                         NALU_HYPRE_Int     *CF_marker,
                                         NALU_HYPRE_Int     *CF_marker_offd,
                                         NALU_HYPRE_Int     *diag_compress_marker,
                                         NALU_HYPRE_Int     *offd_compress_marker,
                                         NALU_HYPRE_Int     *P_diag_i,
                                         NALU_HYPRE_Int     *P_diag_j,
                                         NALU_HYPRE_Int     *P_offd_i,
                                         NALU_HYPRE_Int     *P_offd_j)
{
   /*-----------------------------------------------------------------------*/
   /* Determine size of interpolation matrix, P

      If A is of size m x m, then P will be of size m x c where c is the
      number of coarse points.

      It is assumed that S have the same global column enumeration as A

      Input: nr_of_rows                  - Number of rows in matrix (local in processor)
             A_diag_i, A_strong_diag_j,  - Arrays associated with ParCSRMatrix A
             A_diag_a, A_offd_i,           where the column indices are taken from S
             A_strong_offd_j, A_offd_a     and mark weak connections with negative indices
             CF_maker                    - coarse/fine marker for on-processor points
             CF_maker_offd               - coarse/fine marker for off-processor connections

      Output: P_diag_i             - Vector where P_diag_i[i] holds the number of non-zero elements of P_diag on row i (will be 1).
              P_diag_i             - Vector where P_diag_j[i] holds a temporary, uncompressed column indices for P_diag.
              P_offd_i             - Vector where P_offd_i[i] holds the number of non-zero elements of P_offd on row i (will be 1).
              P_offd_i             - Vector where P_offd_j[i] holds a temporary, uncompressed column indices for P_offd.
              diag_compress_marker - Array of 0s and 1s used to compress P_diag col indices and data.
              offd_compress_marker - Array of 0s and 1s used to compress P_offd col indices and data.
    */
   /*-----------------------------------------------------------------------*/

   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int p = 0, q;
   NALU_HYPRE_Int max_j_diag = -1, max_j_offd = -1;
   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Real max_diag = -1.0, max_offd = -1.0;
   NALU_HYPRE_Real warp_max_diag = -1.0, warp_max_offd = -1.0;

   if (lane == 0)
   {
      p = read_only_load(CF_marker + i);
   }
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   /*--------------------------------------------------------------------
    *  If i is a C-point, interpolation is the identity.
    *--------------------------------------------------------------------*/
   if (p >= 0)
   {
      if (lane == 0)
      {
         P_diag_i[i] = 1;
         P_diag_j[i] = i;
         diag_compress_marker[i] = 1;
      }
      return;
   }

   /*--------------------------------------------------------------------
    *  If i is an F-point, find strongest connected C-point,
    *  which could be in diag or offd.
    *--------------------------------------------------------------------*/

   /* diag part */
   if (lane < 2)
   {
      p = read_only_load(A_diag_i + i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      /* column indices are negative for weak connections */
      const NALU_HYPRE_Int col = read_only_load(&A_strong_diag_j[j]);
      if (col >= 0)
      {
         const NALU_HYPRE_Complex val = nalu_hypre_abs( read_only_load(&A_diag_a[j]) );
         if ( read_only_load(&CF_marker[col]) > 0 && val > max_diag )
         {
            max_diag = val;
            max_j_diag = col;
         }
      }
   }
   warp_max_diag = warp_allreduce_max(item, max_diag);

   /* offd part */
   if (lane < 2)
   {
      p = read_only_load(A_offd_i + i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int col = read_only_load(&A_strong_offd_j[j]);
      /* column indices are negative for weak connections */
      if (col >= 0)
      {
         const NALU_HYPRE_Complex val = nalu_hypre_abs( read_only_load(&A_offd_a[j]) );
         if ( read_only_load(&CF_marker_offd[col]) > 0 && val > max_offd )
         {
            max_offd = val;
            max_j_offd = col;
         }
      }
   }
   warp_max_offd = warp_allreduce_max(item, max_offd);

   /*--------------------------------------------------------------------
    *  If no max found, then there is no strongly connected C-point,
    *  and this will be a zero row
    *--------------------------------------------------------------------*/

   if (warp_max_offd < 0 && warp_max_diag < 0)
   {
      return;
   }

   /*--------------------------------------------------------------------
    *  Otherwise, find the column index in either diag or offd
    *--------------------------------------------------------------------*/

   if (warp_max_offd > warp_max_diag)
   {
      if (warp_max_offd != max_offd)
      {
         max_j_offd = -1;
      }
      max_j_offd = warp_reduce_max(item, max_j_offd);
      if (lane == 0)
      {
         P_offd_i[i] = 1;
         P_offd_j[i] = max_j_offd;
         offd_compress_marker[i] = 1;
      }
   }
   else
   {
      if (warp_max_diag != max_diag)
      {
         max_j_diag = -1;
      }
      max_j_diag = warp_reduce_max(item, max_j_diag);
      if (lane == 0)
      {
         P_diag_i[i] = 1;
         P_diag_j[i] = max_j_diag;
         diag_compress_marker[i] = 1;
      }
   }
}

#endif // defined(NALU_HYPRE_USING_GPU)
