/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

__global__ void nalu_hypre_BoomerAMGBuildRestrNeumannAIR_assembleRdiag( nalu_hypre_DeviceItem &item,
                                                                   NALU_HYPRE_Int nr_of_rows,
                                                                   NALU_HYPRE_Int *Fmap, NALU_HYPRE_Int *Cmap, NALU_HYPRE_Int *Z_diag_i, NALU_HYPRE_Int *Z_diag_j, NALU_HYPRE_Complex *Z_diag_a,
                                                                   NALU_HYPRE_Int *R_diag_i, NALU_HYPRE_Int *R_diag_j, NALU_HYPRE_Complex *R_diag_a);

/*---------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBuildRestrNeumannAIR
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildRestrNeumannAIRDevice( nalu_hypre_ParCSRMatrix   *A,
                                           NALU_HYPRE_Int            *CF_marker,
                                           NALU_HYPRE_BigInt         *num_cpts_global,
                                           NALU_HYPRE_Int             num_functions,
                                           NALU_HYPRE_Int            *dof_func,
                                           NALU_HYPRE_Int             NeumannDeg,
                                           NALU_HYPRE_Real            strong_thresholdR,
                                           NALU_HYPRE_Real            filter_thresholdR,
                                           NALU_HYPRE_Int             debug_flag,
                                           nalu_hypre_ParCSRMatrix  **R_ptr)
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   nalu_hypre_CSRMatrix *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);

   /* Restriction matrix R and CSR's */
   nalu_hypre_ParCSRMatrix *R;
   nalu_hypre_CSRMatrix *R_diag;

   /* arrays */
   NALU_HYPRE_Complex      *R_diag_a;
   NALU_HYPRE_Int          *R_diag_i;
   NALU_HYPRE_Int          *R_diag_j;
   NALU_HYPRE_BigInt       *col_map_offd_R;
   NALU_HYPRE_Int           num_cols_offd_R;
   NALU_HYPRE_Int           my_id, num_procs;
   NALU_HYPRE_BigInt        total_global_cpts;
   NALU_HYPRE_Int           nnz_diag, nnz_offd;
   NALU_HYPRE_BigInt       *send_buf_i;
   NALU_HYPRE_Int           i;

   /* local size */
   NALU_HYPRE_Int n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt col_start = nalu_hypre_ParCSRMatrixFirstRowIndex(A);

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* global number of C points and my start position */
   if (my_id == (num_procs - 1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /* get AFF and ACF */
   nalu_hypre_ParCSRMatrix *AFF, *ACF, *Dinv, *N, *X, *X2, *Z, *Z2;
   if (strong_thresholdR > 0)
   {
      nalu_hypre_ParCSRMatrix *S;
      nalu_hypre_BoomerAMGCreateSabs(A,
                                strong_thresholdR,
                                0.9,
                                num_functions,
                                dof_func,
                                &S);
      nalu_hypre_ParCSRMatrixGenerateFFCFDevice(A, CF_marker, num_cpts_global, S, &ACF, &AFF);
      nalu_hypre_ParCSRMatrixDestroy(S);
   }
   else
   {
      nalu_hypre_ParCSRMatrixGenerateFFCFDevice(A, CF_marker, num_cpts_global, NULL, &ACF, &AFF);
   }

   NALU_HYPRE_Int        n_fpts = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(AFF));
   NALU_HYPRE_Int        n_cpts = n_fine - n_fpts;
   nalu_hypre_assert(n_cpts == nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(ACF)));

   /* maps from F-pts and C-pts to all points */
   NALU_HYPRE_Int       *Fmap = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fpts, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int       *Cmap = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_cpts, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_THRUST_CALL( copy_if,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(n_fine),
                      CF_marker,
                      Fmap,
                      is_negative<NALU_HYPRE_Int>());
   NALU_HYPRE_THRUST_CALL( copy_if,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(n_fine),
                      CF_marker,
                      Cmap,
                      is_positive<NALU_HYPRE_Int>());

   /* setup Dinv = 1/(diagonal of AFF) */
   Dinv = nalu_hypre_ParCSRMatrixCreate(comm,
                                   nalu_hypre_ParCSRMatrixGlobalNumRows(AFF),
                                   nalu_hypre_ParCSRMatrixGlobalNumCols(AFF),
                                   nalu_hypre_ParCSRMatrixRowStarts(AFF),
                                   nalu_hypre_ParCSRMatrixColStarts(AFF),
                                   0,
                                   nalu_hypre_ParCSRMatrixNumRows(AFF),
                                   0);
   nalu_hypre_ParCSRMatrixAssumedPartition(Dinv) = nalu_hypre_ParCSRMatrixAssumedPartition(AFF);
   nalu_hypre_ParCSRMatrixOwnsAssumedPartition(Dinv) = 0;
   nalu_hypre_ParCSRMatrixInitialize(Dinv);
   nalu_hypre_CSRMatrix *Dinv_diag = nalu_hypre_ParCSRMatrixDiag(Dinv);
   NALU_HYPRE_THRUST_CALL( copy,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(nalu_hypre_CSRMatrixNumRows(Dinv_diag) + 1),
                      nalu_hypre_CSRMatrixI(Dinv_diag) );
   NALU_HYPRE_THRUST_CALL( copy,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(nalu_hypre_CSRMatrixNumRows(Dinv_diag)),
                      nalu_hypre_CSRMatrixJ(Dinv_diag) );
   nalu_hypre_CSRMatrixExtractDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(AFF), nalu_hypre_CSRMatrixData(Dinv_diag),
                                        2);

   /* N = I - D^{-1}*A_FF */
   if (NeumannDeg >= 1)
   {
      N = nalu_hypre_ParCSRMatMat(Dinv, AFF);

      nalu_hypre_CSRMatrixRemoveDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(N));

      NALU_HYPRE_THRUST_CALL( transform,
                         nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(N)),
                         nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(N)) + nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(
                                                                                                        N)),
                         nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(N)),
                         thrust::negate<NALU_HYPRE_Complex>() );
      NALU_HYPRE_THRUST_CALL( transform,
                         nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(N)),
                         nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(N)) + nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(
                                                                                                        N)),
                         nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(N)),
                         thrust::negate<NALU_HYPRE_Complex>() );
   }

   /* Z = Acf * (I + N + N^2 + ... + N^k) * D^{-1} */
   if (NeumannDeg < 1)
   {
      Z = ACF;
   }
   else if (NeumannDeg == 1)
   {
      X = nalu_hypre_ParCSRMatMat(ACF, N);
      nalu_hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      nalu_hypre_ParCSRMatrixDestroy(X);
   }
   else
   {
      X = nalu_hypre_ParCSRMatMat(N, N);
      nalu_hypre_ParCSRMatrixAdd(1.0, N, 1.0, X, &Z);
      for (i = 2; i < NeumannDeg; i++)
      {
         X2 = nalu_hypre_ParCSRMatMat(X, N);
         nalu_hypre_ParCSRMatrixAdd(1.0, Z, 1.0, X2, &Z2);
         nalu_hypre_ParCSRMatrixDestroy(X);
         nalu_hypre_ParCSRMatrixDestroy(Z);
         Z = Z2;
         X = X2;
      }
      nalu_hypre_ParCSRMatrixDestroy(X);
      X = nalu_hypre_ParCSRMatMat(ACF, Z);
      nalu_hypre_ParCSRMatrixDestroy(Z);
      nalu_hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      nalu_hypre_ParCSRMatrixDestroy(X);
   }

   X = Z;
   Z = nalu_hypre_ParCSRMatMat(X, Dinv);

   nalu_hypre_ParCSRMatrixDestroy(X);
   nalu_hypre_ParCSRMatrixDestroy(Dinv);
   nalu_hypre_ParCSRMatrixDestroy(AFF);
   if (NeumannDeg >= 1)
   {
      nalu_hypre_ParCSRMatrixDestroy(ACF);
      nalu_hypre_ParCSRMatrixDestroy(N);
   }

   nalu_hypre_CSRMatrix *Z_diag = nalu_hypre_ParCSRMatrixDiag(Z);
   nalu_hypre_CSRMatrix *Z_offd = nalu_hypre_ParCSRMatrixOffd(Z);
   NALU_HYPRE_Complex   *Z_diag_a = nalu_hypre_CSRMatrixData(Z_diag);
   NALU_HYPRE_Int       *Z_diag_i = nalu_hypre_CSRMatrixI(Z_diag);
   NALU_HYPRE_Int       *Z_diag_j = nalu_hypre_CSRMatrixJ(Z_diag);
   NALU_HYPRE_Int        num_cols_offd_Z = nalu_hypre_CSRMatrixNumCols(Z_offd);
   NALU_HYPRE_Int        nnz_diag_Z = nalu_hypre_CSRMatrixNumNonzeros(Z_diag);
   NALU_HYPRE_BigInt    *Fmap_offd_global = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_Z,
                                                    NALU_HYPRE_MEMORY_DEVICE);

   /* send and recv Fmap (wrt Z): global */
   if (num_procs > 1)
   {
      nalu_hypre_MatvecCommPkgCreate(Z);

      nalu_hypre_ParCSRCommPkg *comm_pkg_Z = nalu_hypre_ParCSRMatrixCommPkg(Z);
      NALU_HYPRE_Int num_sends_Z = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_Z);
      NALU_HYPRE_Int num_elems_send_Z = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg_Z, num_sends_Z);
      send_buf_i = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_elems_send_Z, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg_Z);
      NALU_HYPRE_THRUST_CALL( gather,
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg_Z),
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg_Z) +
                         nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg_Z, num_sends_Z),
                         Fmap,
                         send_buf_i );
      NALU_HYPRE_THRUST_CALL( transform,
                         send_buf_i,
                         send_buf_i + num_elems_send_Z,
                         thrust::make_constant_iterator(col_start),
                         send_buf_i,
                         thrust::plus<NALU_HYPRE_BigInt>() );

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && THRUST_CALL_BLOCKING == 0
      /* RL: make sure send_buf_i is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(21, comm_pkg_Z, NALU_HYPRE_MEMORY_DEVICE, send_buf_i,
                                                    NALU_HYPRE_MEMORY_DEVICE, Fmap_offd_global);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      nalu_hypre_TFree(send_buf_i, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* Assemble R = [-Z I] */
   nnz_diag = nnz_diag_Z + n_cpts;
   nnz_offd = nalu_hypre_CSRMatrixNumNonzeros(Z_offd);

   /* allocate arrays for R diag */
   R_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_cpts + 1, NALU_HYPRE_MEMORY_DEVICE);
   R_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_DEVICE);
   R_diag_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nnz_diag, NALU_HYPRE_MEMORY_DEVICE);

   /* setup R row indices (just Z row indices plus one extra entry for each C-pt)*/
   NALU_HYPRE_THRUST_CALL( transform,
                      Z_diag_i,
                      Z_diag_i + n_cpts + 1,
                      thrust::make_counting_iterator(0),
                      R_diag_i,
                      thrust::plus<NALU_HYPRE_Int>() );

   /* assemble the diagonal part of R from Z */
   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(n_fine, "warp", bDim);
   NALU_HYPRE_GPU_LAUNCH( nalu_hypre_BoomerAMGBuildRestrNeumannAIR_assembleRdiag, gDim, bDim,
                     n_cpts, Fmap, Cmap, Z_diag_i, Z_diag_j, Z_diag_a, R_diag_i, R_diag_j, R_diag_a);

   num_cols_offd_R = num_cols_offd_Z;
   col_map_offd_R = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_Z, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(col_map_offd_R, Fmap_offd_global, NALU_HYPRE_BigInt, num_cols_offd_Z, NALU_HYPRE_MEMORY_HOST,
                 NALU_HYPRE_MEMORY_DEVICE);

   /* Now, we should have everything of Parcsr matrix R */
   R = nalu_hypre_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                nalu_hypre_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = nalu_hypre_ParCSRMatrixDiag(R);
   nalu_hypre_CSRMatrixData(R_diag) = R_diag_a;
   nalu_hypre_CSRMatrixI(R_diag)    = R_diag_i;
   nalu_hypre_CSRMatrixJ(R_diag)    = R_diag_j;

   /* R_offd is simply a clone of -Z_offd */
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(R));
   nalu_hypre_ParCSRMatrixOffd(R) = nalu_hypre_CSRMatrixClone(Z_offd, 1);
   NALU_HYPRE_THRUST_CALL( transform,
                      nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(R)),
                      nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(R)) + nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(
                                                                                                     R)),
                      nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(R)),
                      thrust::negate<NALU_HYPRE_Complex>() );

   nalu_hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   nalu_hypre_ParCSRMatrixAssumedPartition(R) = nalu_hypre_ParCSRMatrixAssumedPartition(A);
   nalu_hypre_ParCSRMatrixOwnsAssumedPartition(R) = 0;
   nalu_hypre_MatvecCommPkgCreate(R);

   /* Filter small entries from R */
   if (filter_thresholdR > 0)
   {
      nalu_hypre_ParCSRMatrixDropSmallEntries(R, filter_thresholdR, -1);
   }

   *R_ptr = R;

   nalu_hypre_ParCSRMatrixDestroy(Z);
   nalu_hypre_TFree(Fmap, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(Cmap, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(Fmap_offd_global, NALU_HYPRE_MEMORY_DEVICE);

   return 0;
}

/*-----------------------------------------------------------------------*/
__global__ void
nalu_hypre_BoomerAMGBuildRestrNeumannAIR_assembleRdiag( nalu_hypre_DeviceItem    &item,
                                                   NALU_HYPRE_Int      nr_of_rows,
                                                   NALU_HYPRE_Int     *Fmap,
                                                   NALU_HYPRE_Int     *Cmap,
                                                   NALU_HYPRE_Int     *Z_diag_i,
                                                   NALU_HYPRE_Int     *Z_diag_j,
                                                   NALU_HYPRE_Complex *Z_diag_a,
                                                   NALU_HYPRE_Int     *R_diag_i,
                                                   NALU_HYPRE_Int     *R_diag_j,
                                                   NALU_HYPRE_Complex *R_diag_a)
{
   /*-----------------------------------------------------------------------*/
   /* Assemble diag part of R = [-Z I]

      Input: nr_of_rows - Number of rows in matrix (local in processor)
             CSR represetnation of Z diag, assuming column indices of Z are
             already mapped appropriately

      Output: CSR representation of R diag
    */
   /*-----------------------------------------------------------------------*/

   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int p = 0, q, pZ = 0;
   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);

   /* diag part */
   if (lane < 2)
   {
      p = read_only_load(R_diag_i + i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);
   if (lane < 1)
   {
      pZ = read_only_load(Z_diag_i + i + lane);
   }
   pZ = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pZ, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      if (j == q - 1)
      {
         R_diag_j[j] = Cmap[i];
         R_diag_a[j] = 1.0;
      }
      else
      {
         NALU_HYPRE_Int jZ = pZ + (j - p);
         R_diag_j[j] = Fmap[ Z_diag_j[jZ] ];
         R_diag_a[j] = -Z_diag_a[jZ];
      }
   }
}


struct setTo1minus1 : public thrust::unary_function<NALU_HYPRE_Int, NALU_HYPRE_Int>
{
   __host__ __device__ NALU_HYPRE_Int operator()(const NALU_HYPRE_Int &x) const
   {
      return x > 0 ? 1 : -1;
   }
};

NALU_HYPRE_Int
nalu_hypre_BoomerAMGCFMarkerTo1minus1Device( NALU_HYPRE_Int *CF_marker,
                                        NALU_HYPRE_Int size )
{
   NALU_HYPRE_THRUST_CALL( transform,
                      CF_marker,
                      CF_marker + size,
                      CF_marker,
                      setTo1minus1() );

   return nalu_hypre_error_flag;
}

#endif // defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
