/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Two-grid system solver
 *
 *****************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "seq_mv/seq_mv.h"
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined (NALU_HYPRE_USING_GPU)

template<typename T>
#if defined(NALU_HYPRE_USING_SYCL)
struct functor
#else
struct functor : public thrust::binary_function<T, T, T>
#endif
{
   T scale;

   functor(T scale_) { scale = scale_; }

   __host__ __device__
   T operator()(const T &x, const T &y) const
   {
      return x + scale * (y - nalu_hypre_abs(x));
   }
};

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRBuildPFromWpDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRBuildPFromWpDevice( nalu_hypre_ParCSRMatrix   *A,
                             nalu_hypre_ParCSRMatrix   *Wp,
                             NALU_HYPRE_Int            *CF_marker,
                             nalu_hypre_ParCSRMatrix  **P_ptr)
{
   /* Wp info */
   nalu_hypre_CSRMatrix     *Wp_diag = nalu_hypre_ParCSRMatrixDiag(Wp);
   nalu_hypre_CSRMatrix     *Wp_offd = nalu_hypre_ParCSRMatrixOffd(Wp);

   /* Local variables */
   nalu_hypre_ParCSRMatrix  *P;
   nalu_hypre_CSRMatrix     *P_diag;
   nalu_hypre_CSRMatrix     *P_offd;
   NALU_HYPRE_Int            P_diag_nnz;

   nalu_hypre_GpuProfilingPushRange("MGRBuildPFromWp");

   /* Set local variables */
   P_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(Wp_diag) +
                nalu_hypre_CSRMatrixNumCols(Wp_diag);

   /* Create interpolation matrix */
   P = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(Wp),
                                nalu_hypre_ParCSRMatrixRowStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(Wp),
                                nalu_hypre_CSRMatrixNumCols(Wp_offd),
                                P_diag_nnz,
                                nalu_hypre_CSRMatrixNumNonzeros(Wp_offd));

   /* Initialize interpolation matrix */
   nalu_hypre_ParCSRMatrixInitialize_v2(P, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_ParCSRMatrixDNumNonzeros(P) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(P);
   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   P_offd = nalu_hypre_ParCSRMatrixOffd(P);

   /* Copy contents from W to P and set identity matrix for the mapping between coarse points */
   hypreDevice_extendWtoP(nalu_hypre_ParCSRMatrixNumRows(A),
                          nalu_hypre_ParCSRMatrixNumRows(Wp),
                          nalu_hypre_CSRMatrixNumCols(Wp_diag),
                          CF_marker,
                          nalu_hypre_CSRMatrixNumNonzeros(Wp_diag),
                          nalu_hypre_CSRMatrixI(Wp_diag),
                          nalu_hypre_CSRMatrixJ(Wp_diag),
                          nalu_hypre_CSRMatrixData(Wp_diag),
                          nalu_hypre_CSRMatrixI(P_diag),
                          nalu_hypre_CSRMatrixJ(P_diag),
                          nalu_hypre_CSRMatrixData(P_diag),
                          nalu_hypre_CSRMatrixI(Wp_offd),
                          nalu_hypre_CSRMatrixI(P_offd));

   /* Swap some pointers to avoid data copies */
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(P))    = nalu_hypre_CSRMatrixJ(Wp_offd);
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(P)) = nalu_hypre_CSRMatrixData(Wp_offd);
   nalu_hypre_CSRMatrixJ(Wp_offd)    = NULL;
   nalu_hypre_CSRMatrixData(Wp_offd) = NULL;
   /* nalu_hypre_ParCSRMatrixDeviceColMapOffd(P)    = nalu_hypre_ParCSRMatrixDeviceColMapOffd(Wp); */
   /* nalu_hypre_ParCSRMatrixColMapOffd(P)          = nalu_hypre_ParCSRMatrixColMapOffd(Wp); */
   /* nalu_hypre_ParCSRMatrixDeviceColMapOffd(Wp)   = NULL; */
   /* nalu_hypre_ParCSRMatrixColMapOffd(Wp)         = NULL; */

   /* Create communication package */
   nalu_hypre_MatvecCommPkgCreate(P);

   /* Set output pointer to the interpolation matrix */
   *P_ptr = P;

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRBuildPDevice
 *
 * TODO: make use of nalu_hypre_MGRBuildPFromWpDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRBuildPDevice(nalu_hypre_ParCSRMatrix  *A,
                      NALU_HYPRE_Int           *CF_marker,
                      NALU_HYPRE_BigInt        *num_cpts_global,
                      NALU_HYPRE_Int            method,
                      nalu_hypre_ParCSRMatrix **P_ptr)
{
   MPI_Comm            comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int           num_procs, my_id;
   NALU_HYPRE_Int           A_nr_of_rows = nalu_hypre_ParCSRMatrixNumRows(A);

   nalu_hypre_ParCSRMatrix *A_FF = NULL, *A_FC = NULL, *P = NULL;
   nalu_hypre_CSRMatrix    *W_diag = NULL, *W_offd = NULL;
   NALU_HYPRE_Int           W_nr_of_rows, P_diag_nnz, nfpoints;
   NALU_HYPRE_Int          *P_diag_i = NULL, *P_diag_j = NULL, *P_offd_i = NULL;
   NALU_HYPRE_Complex      *P_diag_data = NULL, *diag = NULL, *diag1 = NULL;
   NALU_HYPRE_BigInt        nC_global;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_GpuProfilingPushRange("MGRBuildP");

#if defined(NALU_HYPRE_USING_SYCL)
   nfpoints = NALU_HYPRE_ONEDPL_CALL(std::count,
                                CF_marker,
                                CF_marker + A_nr_of_rows,
                                -1);
#else
   nfpoints = NALU_HYPRE_THRUST_CALL(count,
                                CF_marker,
                                CF_marker + A_nr_of_rows,
                                -1);
#endif

   if (method > 0)
   {
      nalu_hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, NULL, &A_FC, &A_FF);
      diag = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nfpoints, NALU_HYPRE_MEMORY_DEVICE);
      if (method == 1)
      {
         // extract diag inverse sqrt
         // nalu_hypre_CSRMatrixExtractDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(A_FF), diag, 3);

         // L1-Jacobi-type interpolation
         NALU_HYPRE_Complex scal = 1.0;

         diag1 = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nfpoints, NALU_HYPRE_MEMORY_DEVICE);
         nalu_hypre_CSRMatrixExtractDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(A_FF), diag, 0);

         nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixDiag(A_FF), NULL, NULL,
                                            diag1, 1, 1.0, "set");
         nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixDiag(A_FC), NULL, NULL,
                                            diag1, 1, 1.0, "add");
         nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixOffd(A_FF), NULL, NULL,
                                            diag1, 1, 1.0, "add");
         nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixOffd(A_FC), NULL, NULL,
                                            diag1, 1, 1.0, "add");

#if defined(NALU_HYPRE_USING_SYCL)
         NALU_HYPRE_ONEDPL_CALL(std::transform,
                           diag,
                           diag + nfpoints,
                           diag1,
                           diag,
                           functor<NALU_HYPRE_Complex>(scal));

         NALU_HYPRE_ONEDPL_CALL(std::transform,
                           diag,
                           diag + nfpoints,
                           diag,
         [] (auto x) { return 1.0 / x; });
#else
         NALU_HYPRE_THRUST_CALL(transform,
                           diag,
                           diag + nfpoints,
                           diag1,
                           diag,
                           functor<NALU_HYPRE_Complex>(scal));

         NALU_HYPRE_THRUST_CALL(transform,
                           diag,
                           diag + nfpoints,
                           diag,
                           1.0 / _1);
#endif

         nalu_hypre_TFree(diag1, NALU_HYPRE_MEMORY_DEVICE);
      }
      else if (method == 2)
      {
         // extract diag inverse
         nalu_hypre_CSRMatrixExtractDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(A_FF), diag, 2);
      }

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( transform, diag, diag + nfpoints, diag, std::negate<NALU_HYPRE_Complex>() );
#else
      NALU_HYPRE_THRUST_CALL( transform, diag, diag + nfpoints, diag, thrust::negate<NALU_HYPRE_Complex>() );
#endif

      nalu_hypre_Vector *D_FF_inv = nalu_hypre_SeqVectorCreate(nfpoints);
      nalu_hypre_VectorData(D_FF_inv) = diag;
      nalu_hypre_SeqVectorInitialize_v2(D_FF_inv, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixDiagScaleDevice(nalu_hypre_ParCSRMatrixDiag(A_FC), D_FF_inv, NULL);
      nalu_hypre_CSRMatrixDiagScaleDevice(nalu_hypre_ParCSRMatrixOffd(A_FC), D_FF_inv, NULL);
      nalu_hypre_SeqVectorDestroy(D_FF_inv);
      W_diag = nalu_hypre_ParCSRMatrixDiag(A_FC);
      W_offd = nalu_hypre_ParCSRMatrixOffd(A_FC);
      nC_global = nalu_hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      W_diag = nalu_hypre_CSRMatrixCreate(nfpoints, A_nr_of_rows - nfpoints, 0);
      W_offd = nalu_hypre_CSRMatrixCreate(nfpoints, 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(W_diag, 0, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixInitialize_v2(W_offd, 0, NALU_HYPRE_MEMORY_DEVICE);

      if (my_id == (num_procs - 1))
      {
         nC_global = num_cpts_global[1];
      }
      nalu_hypre_MPI_Bcast(&nC_global, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }

   W_nr_of_rows = nalu_hypre_CSRMatrixNumRows(W_diag);

   /* Construct P from matrix product W_diag */
   P_diag_nnz  = nalu_hypre_CSRMatrixNumNonzeros(W_diag) + nalu_hypre_CSRMatrixNumCols(W_diag);
   P_diag_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, P_diag_nnz,     NALU_HYPRE_MEMORY_DEVICE);
   P_offd_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           nalu_hypre_CSRMatrixNumCols(W_diag),
                           CF_marker,
                           nalu_hypre_CSRMatrixNumNonzeros(W_diag),
                           nalu_hypre_CSRMatrixI(W_diag),
                           nalu_hypre_CSRMatrixJ(W_diag),
                           nalu_hypre_CSRMatrixData(W_diag),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           nalu_hypre_CSRMatrixI(W_offd),
                           P_offd_i );

   // final P
   P = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nC_global,
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                nalu_hypre_CSRMatrixNumCols(W_offd),
                                P_diag_nnz,
                                nalu_hypre_CSRMatrixNumNonzeros(W_offd) );

   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(P)) = NALU_HYPRE_MEMORY_DEVICE;
   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(P)) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(P))    = nalu_hypre_CSRMatrixJ(W_offd);
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(P)) = nalu_hypre_CSRMatrixData(W_offd);
   nalu_hypre_CSRMatrixJ(W_offd)    = NULL;
   nalu_hypre_CSRMatrixData(W_offd) = NULL;

   if (method > 0)
   {
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(P)    = nalu_hypre_ParCSRMatrixDeviceColMapOffd(A_FC);
      nalu_hypre_ParCSRMatrixColMapOffd(P)          = nalu_hypre_ParCSRMatrixColMapOffd(A_FC);
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(A_FC) = NULL;
      nalu_hypre_ParCSRMatrixColMapOffd(A_FC)       = NULL;
      nalu_hypre_ParCSRMatrixNumNonzeros(P)         = nalu_hypre_ParCSRMatrixNumNonzeros(A_FC) +
                                                 nalu_hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      nalu_hypre_ParCSRMatrixNumNonzeros(P) = nC_global;
   }
   nalu_hypre_ParCSRMatrixDNumNonzeros(P) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(P);

   nalu_hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   if (A_FF)
   {
      nalu_hypre_ParCSRMatrixDestroy(A_FF);
   }
   if (A_FC)
   {
      nalu_hypre_ParCSRMatrixDestroy(A_FC);
   }

   if (method <= 0)
   {
      nalu_hypre_CSRMatrixDestroy(W_diag);
      nalu_hypre_CSRMatrixDestroy(W_offd);
   }

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRRelaxL1JacobiDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRRelaxL1JacobiDevice( nalu_hypre_ParCSRMatrix *A,
                              nalu_hypre_ParVector    *f,
                              NALU_HYPRE_Int          *CF_marker,
                              NALU_HYPRE_Int           relax_points,
                              NALU_HYPRE_Real          relax_weight,
                              NALU_HYPRE_Real         *l1_norms,
                              nalu_hypre_ParVector    *u,
                              nalu_hypre_ParVector    *Vtemp )
{
   nalu_hypre_BoomerAMGRelax(A, f, CF_marker, 18,
                        relax_points, relax_weight, 1.0,
                        l1_norms, u, Vtemp, NULL);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatrixExtractBlockDiag
 *
 * Fills vector diag with the block diagonals from the input matrix.
 * This function uses column-major storage for diag.
 *
 * TODOs:
 *    1) Move this to csr_matop_device.c
 *    2) Use sub-warps?
 *    3) blk_size as template arg.
 *    4) Choose diag storage between row and column-major?
 *    5) Should we build flat arrays, arrays of pointers, or allow both?
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMatrixExtractBlockDiag( nalu_hypre_DeviceItem  &item,
                                          NALU_HYPRE_Int          blk_size,
                                          NALU_HYPRE_Int          num_rows,
                                          NALU_HYPRE_Int         *A_i,
                                          NALU_HYPRE_Int         *A_j,
                                          NALU_HYPRE_Complex     *A_a,
                                          NALU_HYPRE_Int         *B_i,
                                          NALU_HYPRE_Int         *B_j,
                                          NALU_HYPRE_Complex     *B_a )
{
   NALU_HYPRE_Int   lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int   bs2  = blk_size * blk_size;
   NALU_HYPRE_Int   bidx;
   NALU_HYPRE_Int   lidx;
   NALU_HYPRE_Int   i, ii, j, pj, qj;
   NALU_HYPRE_Int   col;

   /* Grid-stride loop over block matrix rows */
   for (bidx = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);
        bidx < num_rows / blk_size;
        bidx += nalu_hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      ii = bidx * blk_size;

      /* Set output row pointer and column indices */
      for (i = lane; i < blk_size; i += NALU_HYPRE_WARP_SIZE)
      {
         B_i[ii + i + 1] = (ii + i + 1) * blk_size;
      }

      /* Set output column indices (row major) */
      for (j = lane; j < bs2; j += NALU_HYPRE_WARP_SIZE)
      {
         B_j[ii * blk_size + j] = ii + j % blk_size;
      }

      /* TODO: unroll this loop */
      for (lidx = 0; lidx < blk_size; lidx++)
      {
         i = ii + lidx;

         if (lane < 2)
         {
            pj = read_only_load(A_i + i + lane);
         }
         qj = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pj, 1);
         pj = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pj, 0);

         /* Loop over columns */
         for (j = pj + lane; j < qj; j += NALU_HYPRE_WARP_SIZE)
         {
            col = read_only_load(A_j + j);

            if ((col >= ii) &&
                (col <  ii + blk_size) &&
                (fabs(A_a[j]) > NALU_HYPRE_REAL_MIN))
            {
               /* batch offset + column offset + row offset */
               B_a[ii * blk_size + (col - ii) * blk_size + lidx] = A_a[j];
            }
         }
      } /* Local block loop */
   } /* Grid-stride loop */
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatrixExtractBlockDiagMarked
 *
 * Fills vector diag with the block diagonals from the input matrix.
 * This function uses column-major storage for diag.
 *
 * TODOs:
 *    1) Move this to csr_matop_device.c
 *    2) Use sub-warps?
 *    3) blk_size as template arg.
 *    4) Choose diag storage between row and column-major?
 *    5) Should we build flat arrays, arrays of pointers, or allow both?
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMatrixExtractBlockDiagMarked( nalu_hypre_DeviceItem  &item,
                                                NALU_HYPRE_Int          blk_size,
                                                NALU_HYPRE_Int          num_rows,
                                                NALU_HYPRE_Int          marker_val,
                                                NALU_HYPRE_Int         *marker,
                                                NALU_HYPRE_Int         *marker_indices,
                                                NALU_HYPRE_Int         *A_i,
                                                NALU_HYPRE_Int         *A_j,
                                                NALU_HYPRE_Complex     *A_a,
                                                NALU_HYPRE_Int         *B_i,
                                                NALU_HYPRE_Int         *B_j,
                                                NALU_HYPRE_Complex     *B_a )
{
   NALU_HYPRE_Int   lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int   bidx;
   NALU_HYPRE_Int   lidx;
   NALU_HYPRE_Int   i, ii, j, pj, qj, k;
   NALU_HYPRE_Int   col;

   /* Grid-stride loop over block matrix rows */
   for (bidx = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);
        bidx < num_rows / blk_size;
        bidx += nalu_hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      /* TODO: unroll this loop */
      for (lidx = 0; lidx < blk_size; lidx++)
      {
         ii = bidx * blk_size;
         i  = ii + lidx;

         if (marker[i] == marker_val)
         {
            if (lane < 2)
            {
               pj = read_only_load(A_i + i + lane);
            }
            qj = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pj, 1);
            pj = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, pj, 0);

            /* Loop over columns */
            for (j = pj + lane; j < qj; j += NALU_HYPRE_WARP_SIZE)
            {
               k = read_only_load(A_j + j);
               col = A_j[k];

               if (marker[col] == marker_val)
               {
                  if ((col >= ii) &&
                      (col <  ii + blk_size) &&
                      (fabs(A_a[k]) > NALU_HYPRE_REAL_MIN))
                  {
                     /* batch offset + column offset + row offset */
                     B_a[marker_indices[ii] * blk_size + (col - ii) * blk_size + lidx] = A_a[k];
                  }
               }
            }
         } /* row check */
      } /* Local block loop */
   } /* Grid-stride loop */
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_ComplexMatrixBatchedTranspose
 *
 * Transposes a group of dense matrices. Assigns one warp per block (batch).
 * Naive implementation.
 *
 * TODOs (VPM):
 *    1) Move to proper file.
 *    2) Use template argument for other data types
 *    3) Implement in-place transpose.
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_ComplexMatrixBatchedTranspose( nalu_hypre_DeviceItem  &item,
                                              NALU_HYPRE_Int          num_blocks,
                                              NALU_HYPRE_Int          block_size,
                                              NALU_HYPRE_Complex     *A_data,
                                              NALU_HYPRE_Complex     *B_data )
{
   NALU_HYPRE_Int   lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int   bs2  = block_size * block_size;
   NALU_HYPRE_Int   bidx, lidx;

   /* Grid-stride loop over block matrix rows */
   for (bidx = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);
        bidx < num_blocks;
        bidx += nalu_hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      for (lidx = lane; lidx < bs2; lidx += NALU_HYPRE_WARP_SIZE)
      {
         B_data[bidx * bs2 + lidx] =
            A_data[bidx * bs2 + (lidx / block_size + (lidx % block_size) * block_size)];
      }
   } /* Grid-stride loop */
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixExtractBlockDiagDevice
 *
 * TODOs (VPM):
 *   1) Allow other local solver choices. Design an interface for that.
 *   2) Move this to par_csr_matop_device.c
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixExtractBlockDiagDevice( nalu_hypre_ParCSRMatrix   *A,
                                          NALU_HYPRE_Int             blk_size,
                                          NALU_HYPRE_Int             num_points,
                                          NALU_HYPRE_Int             point_type,
                                          NALU_HYPRE_Int            *CF_marker,
                                          NALU_HYPRE_Int             diag_size,
                                          NALU_HYPRE_Int             diag_type,
                                          NALU_HYPRE_Int            *B_diag_i,
                                          NALU_HYPRE_Int            *B_diag_j,
                                          NALU_HYPRE_Complex        *B_diag_data )
{
   /* Matrix variables */
   NALU_HYPRE_BigInt          num_rows_A   = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   nalu_hypre_CSRMatrix      *A_diag       = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int             num_rows     = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int            *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int            *A_diag_j     = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex        *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);

   /* Local LS variables */
#if defined(NALU_HYPRE_USING_ONEMKLBLAS)
   std::int64_t         *pivots;
   std::int64_t          work_sizes[2];
   std::int64_t          work_size;
   NALU_HYPRE_Complex        *scratchpad;
#else
   NALU_HYPRE_Int            *pivots;
   NALU_HYPRE_Complex       **tmpdiag_aop;
   NALU_HYPRE_Int            *infos;
#endif
   NALU_HYPRE_Int            *blk_row_indices;
   NALU_HYPRE_Complex        *tmpdiag;
   NALU_HYPRE_Complex       **diag_aop;

   /* Local variables */
   NALU_HYPRE_Int             bs2 = blk_size * blk_size;
   NALU_HYPRE_Int             num_blocks;
   NALU_HYPRE_Int             bdiag_size;

   /* Additional variables for debugging */
#if NALU_HYPRE_DEBUG
   NALU_HYPRE_Int            *h_infos;
   NALU_HYPRE_Int             k, myid;

   nalu_hypre_MPI_Comm_rank(nalu_hypre_ParCSRMatrixComm(A), &myid);
#endif

   /*-----------------------------------------------------------------
    * Sanity checks
    *-----------------------------------------------------------------*/

   if (blk_size < 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Invalid block size!");

      return nalu_hypre_error_flag;
   }

   if ((num_rows_A > 0) && (num_rows_A < blk_size))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Input matrix is smaller than block size!");

      return nalu_hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    * Initial
    *-----------------------------------------------------------------*/

   nalu_hypre_GpuProfilingPushRange("ParCSRMatrixExtractBlockDiag");

   /* Count the number of points matching point_type in CF_marker */
   if (CF_marker)
   {
      /* Compute block row indices */
      blk_row_indices = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_DEVICE);
      hypreDevice_IntFilln(blk_row_indices, (size_t) num_rows, 1);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL(oneapi::dpl::exclusive_scan_by_segment,
                        CF_marker,
                        CF_marker + num_rows,
                        blk_row_indices,
                        blk_row_indices);
#else
      NALU_HYPRE_THRUST_CALL(exclusive_scan_by_key,
                        CF_marker,
                        CF_marker + num_rows,
                        blk_row_indices,
                        blk_row_indices);
#endif
   }
   else
   {
      blk_row_indices = NULL;
   }

   /* Compute block info */
   num_blocks = (num_points - 1) / blk_size + 1;
   bdiag_size = num_blocks * bs2;

   if (num_points % blk_size)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "TODO! num_points % blk_size != 0");
      nalu_hypre_GpuProfilingPopRange();

      return nalu_hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    * Extract diagonal sub-blocks (pattern and coefficients)
    *-----------------------------------------------------------------*/
   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_rows / blk_size, "warp", bDim);

      if (CF_marker)
      {
         NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixExtractBlockDiagMarked, gDim, bDim,
                           blk_size, num_rows, point_type, CF_marker, blk_row_indices,
                           A_diag_i, A_diag_j, A_diag_data,
                           B_diag_i, B_diag_j, B_diag_data );
      }
      else
      {
         NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixExtractBlockDiag, gDim, bDim,
                           blk_size, num_rows,
                           A_diag_i, A_diag_j, A_diag_data,
                           B_diag_i, B_diag_j, B_diag_data );
      }
   }

   /*-----------------------------------------------------------------
    * Invert diagonal sub-blocks
    *-----------------------------------------------------------------*/

   if (diag_type == 1)
   {
      NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "InvertDiagSubBlocks");

      /* Memory allocation */
      tmpdiag     = nalu_hypre_TAlloc(NALU_HYPRE_Complex, bdiag_size, NALU_HYPRE_MEMORY_DEVICE);
      diag_aop    = nalu_hypre_TAlloc(NALU_HYPRE_Complex *, num_rows, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_ONEMKLBLAS)
      pivots      = nalu_hypre_CTAlloc(std::int64_t, num_rows * blk_size, NALU_HYPRE_MEMORY_DEVICE);
#else
      pivots      = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows * blk_size, NALU_HYPRE_MEMORY_DEVICE);
      tmpdiag_aop = nalu_hypre_TAlloc(NALU_HYPRE_Complex *, num_rows, NALU_HYPRE_MEMORY_DEVICE);
      infos       = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_DEVICE);
#if defined (NALU_HYPRE_DEBUG)
      h_infos     = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_rows, NALU_HYPRE_MEMORY_HOST);
#endif

      /* Memory copy */
      nalu_hypre_TMemcpy(tmpdiag, B_diag_data, NALU_HYPRE_Complex, bdiag_size,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

      /* Set work array of pointers */
      hypreDevice_ComplexArrayToArrayOfPtrs(num_rows, bs2, tmpdiag, tmpdiag_aop);
#endif

      /* Set array of pointers */
      hypreDevice_ComplexArrayToArrayOfPtrs(num_rows, bs2, B_diag_data, diag_aop);

      /* Compute LU factorization */
#if defined(NALU_HYPRE_USING_CUBLAS)
      NALU_HYPRE_CUBLAS_CALL(nalu_hypre_cublas_getrfBatched(nalu_hypre_HandleCublasHandle(nalu_hypre_handle()),
                                                  blk_size,
                                                  tmpdiag_aop,
                                                  blk_size,
                                                  pivots,
                                                  infos,
                                                  num_blocks));
#elif defined(NALU_HYPRE_USING_ROCSOLVER)
      NALU_HYPRE_ROCSOLVER_CALL(rocsolver_dgetrf_batched(nalu_hypre_HandleVendorSolverHandle(nalu_hypre_handle()),
                                                    blk_size,
                                                    blk_size,
                                                    tmpdiag_aop,
                                                    blk_size,
                                                    pivots,
                                                    blk_size,
                                                    infos,
                                                    num_blocks));

#elif defined(NALU_HYPRE_USING_ONEMKLBLAS)
      NALU_HYPRE_ONEMKL_CALL( work_sizes[0] =
                            oneapi::mkl::lapack::getrf_batch_scratchpad_size<NALU_HYPRE_Complex>( *nalu_hypre_HandleComputeStream(
                                                                                                nalu_hypre_handle()),
                                                                                             blk_size, // std::int64_t m,
                                                                                             blk_size, // std::int64_t n,
                                                                                             blk_size, // std::int64_t lda,
                                                                                             bs2, // std::int64_t stride_a,
                                                                                             blk_size, // std::int64_t stride_ipiv,
                                                                                             num_blocks ) ); // std::int64_t batch_size

      NALU_HYPRE_ONEMKL_CALL( work_sizes[1] =
                            oneapi::mkl::lapack::getri_batch_scratchpad_size<NALU_HYPRE_Complex>( *nalu_hypre_HandleComputeStream(
                                                                                                nalu_hypre_handle()),
                                                                                             (std::int64_t) blk_size, // std::int64_t n,
                                                                                             (std::int64_t) blk_size, // std::int64_t lda,
                                                                                             (std::int64_t) bs2, // std::int64_t stride_a,
                                                                                             (std::int64_t) blk_size, // std::int64_t stride_ipiv,
                                                                                             (std::int64_t) num_blocks // std::int64_t batch_size
                                                                                           ) );
      work_size  = nalu_hypre_max(work_sizes[0], work_sizes[1]);
      scratchpad = nalu_hypre_TAlloc(NALU_HYPRE_Complex, work_size, NALU_HYPRE_MEMORY_DEVICE);

      NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::lapack::getrf_batch( *nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                           (std::int64_t) blk_size, // std::int64_t m,
                                                           (std::int64_t) blk_size, // std::int64_t n,
                                                           *diag_aop, // T *a,
                                                           (std::int64_t) blk_size, // std::int64_t lda,
                                                           (std::int64_t) bs2, // std::int64_t stride_a,
                                                           pivots, // std::int64_t *ipiv,
                                                           (std::int64_t) blk_size, // std::int64_t stride_ipiv,
                                                           (std::int64_t) num_blocks, // std::int64_t batch_size,
                                                           scratchpad, // T *scratchpad,
                                                           (std::int64_t) work_size // std::int64_t scratchpad_size,
                                                         ).wait() ); // const std::vector<cl::sycl::event> &events = {} ) );
#else
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Block inversion not available!");
      return nalu_hypre_error_flag;
#endif

#if defined (NALU_HYPRE_DEBUG) && !defined(NALU_HYPRE_USING_ONEMKLBLAS)
      nalu_hypre_TMemcpy(h_infos, infos, NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      for (k = 0; k < num_rows; k++)
      {
         if (h_infos[k] != 0)
         {
            if (h_infos[k] < 0)
            {
               nalu_hypre_printf("[%d]: LU fact. failed at system %d, parameter %d ",
                            myid, k, h_infos[k]);
            }
            else
            {
               nalu_hypre_printf("[%d]: Singular U(%d, %d) at system %d",
                            myid, h_infos[k], h_infos[k], k);
            }
         }
      }
#endif

      /* Compute sub-blocks inverses */
#if defined(NALU_HYPRE_USING_CUBLAS)
      NALU_HYPRE_CUBLAS_CALL(nalu_hypre_cublas_getriBatched(nalu_hypre_HandleCublasHandle(nalu_hypre_handle()),
                                                  blk_size,
                                                  (const NALU_HYPRE_Real **) tmpdiag_aop,
                                                  blk_size,
                                                  pivots,
                                                  diag_aop,
                                                  blk_size,
                                                  infos,
                                                  num_blocks));
#elif defined(NALU_HYPRE_USING_ROCSOLVER)
      NALU_HYPRE_ROCSOLVER_CALL(rocsolver_dgetri_batched(nalu_hypre_HandleVendorSolverHandle(nalu_hypre_handle()),
                                                    blk_size,
                                                    tmpdiag_aop,
                                                    blk_size,
                                                    pivots,
                                                    blk_size,
                                                    infos,
                                                    num_blocks));
#elif defined(NALU_HYPRE_USING_ONEMKLBLAS)
      NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::lapack::getri_batch( *nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                           (std::int64_t) blk_size, // std::int64_t n,
                                                           *diag_aop, // T *a,
                                                           (std::int64_t) blk_size, // std::int64_t lda,
                                                           (std::int64_t) bs2, // std::int64_t stride_a,
                                                           pivots, // std::int64_t *ipiv,
                                                           (std::int64_t) blk_size, // std::int64_t stride_ipiv,
                                                           (std::int64_t) num_blocks, // std::int64_t batch_size,
                                                           scratchpad, // T *scratchpad,
                                                           work_size // std::int64_t scratchpad_size
                                                         ).wait() );
#else
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Block inversion not available!");
      return nalu_hypre_error_flag;
#endif

      /* Free memory */
      nalu_hypre_TFree(diag_aop, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(pivots, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_ONEMKLBLAS)
      nalu_hypre_TFree(scratchpad, NALU_HYPRE_MEMORY_DEVICE);
#else
      nalu_hypre_TFree(tmpdiag_aop, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(infos, NALU_HYPRE_MEMORY_DEVICE);
#if defined (NALU_HYPRE_DEBUG)
      nalu_hypre_TFree(h_infos, NALU_HYPRE_MEMORY_HOST);
#endif
#endif

      /* Transpose data to row-major format */
      {
         dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
         dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_blocks, "warp", bDim);

         /* Memory copy */
         nalu_hypre_TMemcpy(tmpdiag, B_diag_data, NALU_HYPRE_Complex, bdiag_size,
                       NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

         NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_ComplexMatrixBatchedTranspose, gDim, bDim,
                           num_blocks, blk_size, tmpdiag, B_diag_data );
      }

      /* Free memory */
      nalu_hypre_TFree(tmpdiag, NALU_HYPRE_MEMORY_DEVICE);

      NALU_HYPRE_ANNOTATE_REGION_END("%s", "InvertDiagSubBlocks");
   }

   /* Free memory */
   nalu_hypre_TFree(blk_row_indices, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixBlockDiagMatrixDevice
 *
 * TODO: Move this to par_csr_matop_device.c (VPM)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixBlockDiagMatrixDevice( nalu_hypre_ParCSRMatrix  *A,
                                         NALU_HYPRE_Int            blk_size,
                                         NALU_HYPRE_Int            point_type,
                                         NALU_HYPRE_Int           *CF_marker,
                                         NALU_HYPRE_Int            diag_type,
                                         nalu_hypre_ParCSRMatrix **B_ptr )
{
   /* Input matrix info */
   MPI_Comm              comm            = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_BigInt         *row_starts_A    = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_BigInt          num_rows_A      = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   nalu_hypre_CSRMatrix      *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int             A_diag_num_rows = nalu_hypre_CSRMatrixNumRows(A_diag);

   /* Global block matrix info */
   nalu_hypre_ParCSRMatrix   *par_B;
   NALU_HYPRE_BigInt          num_rows_B;
   NALU_HYPRE_BigInt          row_starts_B[2];

   /* Diagonal block matrix info */
   nalu_hypre_CSRMatrix      *B_diag;
   NALU_HYPRE_Int             B_diag_num_rows;
   NALU_HYPRE_Int             B_diag_size;
   NALU_HYPRE_Int            *B_diag_i;
   NALU_HYPRE_Int            *B_diag_j;
   NALU_HYPRE_Complex        *B_diag_data;

   /* Local variables */
   NALU_HYPRE_BigInt          num_rows_big;
   NALU_HYPRE_BigInt          scan_recv;
   NALU_HYPRE_Int             num_procs, my_id;
   NALU_HYPRE_Int             num_blocks;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /*-----------------------------------------------------------------
    * Count the number of points matching point_type in CF_marker
    *-----------------------------------------------------------------*/

   if (!CF_marker)
   {
      B_diag_num_rows = A_diag_num_rows;
   }
   else
   {
#if defined(NALU_HYPRE_USING_SYCL)
      B_diag_num_rows = NALU_HYPRE_ONEDPL_CALL( std::count,
                                           CF_marker,
                                           CF_marker + A_diag_num_rows,
                                           point_type );
#else
      B_diag_num_rows = NALU_HYPRE_THRUST_CALL( count,
                                           CF_marker,
                                           CF_marker + A_diag_num_rows,
                                           point_type );
#endif
   }
   num_blocks  = 1 + (B_diag_num_rows - 1) / blk_size;
   B_diag_size = blk_size * (blk_size * num_blocks);

   /*-----------------------------------------------------------------
    * Compute global number of rows and partitionings
    *-----------------------------------------------------------------*/

   if (CF_marker)
   {
      num_rows_big = (NALU_HYPRE_BigInt) B_diag_num_rows;
      nalu_hypre_MPI_Scan(&num_rows_big, &scan_recv, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

      /* first point in my range */
      row_starts_B[0] = scan_recv - num_rows_big;

      /* first point in next proc's range */
      row_starts_B[1] = scan_recv;
      if (my_id == (num_procs - 1))
      {
         num_rows_B = row_starts_B[1];
      }
      nalu_hypre_MPI_Bcast(&num_rows_B, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      row_starts_B[0] = row_starts_A[0];
      row_starts_B[1] = row_starts_A[1];
      num_rows_B = num_rows_A;
   }

   /* Create matrix B */
   par_B = nalu_hypre_ParCSRMatrixCreate(comm,
                                    num_rows_B,
                                    num_rows_B,
                                    row_starts_B,
                                    row_starts_B,
                                    0,
                                    B_diag_size,
                                    0);
   nalu_hypre_ParCSRMatrixInitialize_v2(par_B, NALU_HYPRE_MEMORY_DEVICE);
   B_diag      = nalu_hypre_ParCSRMatrixDiag(par_B);
   B_diag_i    = nalu_hypre_CSRMatrixI(B_diag);
   B_diag_j    = nalu_hypre_CSRMatrixJ(B_diag);
   B_diag_data = nalu_hypre_CSRMatrixData(B_diag);

   /*-----------------------------------------------------------------------
    * Extract coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_ParCSRMatrixExtractBlockDiagDevice(A, blk_size, B_diag_num_rows,
                                            point_type, CF_marker,
                                            B_diag_size, diag_type,
                                            B_diag_i, B_diag_j, B_diag_data);

   /* Set output pointer */
   *B_ptr = par_B;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRComputeNonGalerkinCGDevice
 *
 * Available methods:
 *   1: inv(A_FF) approximated by its (block) diagonal inverse
 *   2: CPR-like approx. with inv(A_FF) approx. by its diagonal inverse
 *   3: CPR-like approx. with inv(A_FF) approx. by its block diagonal inverse
 *   4: inv(A_FF) approximated by sparse approximate inverse
 *
 * TODO (VPM): Can we have a single function that works for host and device?
 *             inv(A_FF)*A_FC might have been computed before. Reuse it!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRComputeNonGalerkinCGDevice(nalu_hypre_ParCSRMatrix    *A_FF,
                                    nalu_hypre_ParCSRMatrix    *A_FC,
                                    nalu_hypre_ParCSRMatrix    *A_CF,
                                    nalu_hypre_ParCSRMatrix    *A_CC,
                                    NALU_HYPRE_Int              blk_size,
                                    NALU_HYPRE_Int              method,
                                    NALU_HYPRE_Complex          threshold,
                                    nalu_hypre_ParCSRMatrix   **A_H_ptr)
{
   /* Local variables */
   nalu_hypre_ParCSRMatrix   *A_H;
   nalu_hypre_ParCSRMatrix   *A_Hc;
   nalu_hypre_ParCSRMatrix   *A_CF_trunc;
   nalu_hypre_ParCSRMatrix   *Wp;
   NALU_HYPRE_Complex         alpha = -1.0;

   nalu_hypre_GpuProfilingPushRange("MGRComputeNonGalerkinCG");

   /* Truncate A_CF according to the method */
   if (method == 2 || method == 3)
   {
      nalu_hypre_MGRTruncateAcfCPRDevice(A_CF, &A_CF_trunc);
   }
   else
   {
      A_CF_trunc = A_CF;
   }

   /* Compute Wp */
   if (method == 1 || method == 2)
   {
      nalu_hypre_Vector         *D_FF_inv;
      NALU_HYPRE_Complex        *data;

      /* Create vector to store A_FF's diagonal inverse  */
      D_FF_inv = nalu_hypre_SeqVectorCreate(nalu_hypre_ParCSRMatrixNumRows(A_FF));
      nalu_hypre_SeqVectorInitialize_v2(D_FF_inv, NALU_HYPRE_MEMORY_DEVICE);
      data = nalu_hypre_VectorData(D_FF_inv);

      /* Compute the inverse of A_FF and compute its inverse */
      nalu_hypre_CSRMatrixExtractDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(A_FF), data, 2);

      /* Compute D_FF_inv*A_FC */
      Wp = nalu_hypre_ParCSRMatrixClone(A_FC, 1);
      nalu_hypre_CSRMatrixDiagScaleDevice(nalu_hypre_ParCSRMatrixDiag(Wp), D_FF_inv, NULL);
      nalu_hypre_CSRMatrixDiagScaleDevice(nalu_hypre_ParCSRMatrixOffd(Wp), D_FF_inv, NULL);

      /* Free memory */
      nalu_hypre_SeqVectorDestroy(D_FF_inv);
   }
   else if (method == 3)
   {
      nalu_hypre_ParCSRMatrix  *B_FF_inv;

      /* Compute the block diagonal inverse of A_FF */
      nalu_hypre_ParCSRMatrixBlockDiagMatrixDevice(A_FF, blk_size, -1, NULL, 1, &B_FF_inv);

      /* Compute Wp = A_FF_inv * A_FC */
      Wp = nalu_hypre_ParCSRMatMat(B_FF_inv, A_FC);

      /* Free memory */
      nalu_hypre_ParCSRMatrixDestroy(B_FF_inv);
   }
   else
   {
      /* Use approximate inverse for ideal interploation */
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error: feature not implemented yet!");
      nalu_hypre_GpuProfilingPopRange();

      return nalu_hypre_error_flag;
   }

   /* Compute A_Hc (the correction for A_H) */
   A_Hc = nalu_hypre_ParCSRMatMat(A_CF_trunc, Wp);

   /* Drop small entries from A_Hc */
   nalu_hypre_ParCSRMatrixDropSmallEntriesDevice(A_Hc, threshold, -1);

   /* Coarse grid (Schur complement) computation */
   nalu_hypre_ParCSRMatrixAdd(1.0, A_CC, alpha, A_Hc, &A_H);

   /* Free memory */
   nalu_hypre_ParCSRMatrixDestroy(A_Hc);
   nalu_hypre_ParCSRMatrixDestroy(Wp);
   if (method == 2 || method == 3)
   {
      nalu_hypre_ParCSRMatrixDestroy(A_CF_trunc);
   }

   /* Set output pointer to coarse grid matrix */
   *A_H_ptr = A_H;

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

#endif
