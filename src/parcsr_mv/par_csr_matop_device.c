/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

/* return B = [Adiag, Aoffd] */
#if 1
__global__ void
hypreGPUKernel_ConcatDiagAndOffd( nalu_hypre_DeviceItem &item,
                                  NALU_HYPRE_Int  nrows,    NALU_HYPRE_Int  diag_ncol,
                                  NALU_HYPRE_Int *d_diag_i, NALU_HYPRE_Int *d_diag_j, NALU_HYPRE_Complex *d_diag_a,
                                  NALU_HYPRE_Int *d_offd_i, NALU_HYPRE_Int *d_offd_j, NALU_HYPRE_Complex *d_offd_a,
                                  NALU_HYPRE_Int *cols_offd_map,
                                  NALU_HYPRE_Int *d_ib,     NALU_HYPRE_Int *d_jb,     NALU_HYPRE_Complex *d_ab)
{
   const NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   /* lane id inside the warp */
   const NALU_HYPRE_Int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int i, j = 0, k = 0, p, istart, iend, bstart;

   /* diag part */
   if (lane_id < 2)
   {
      j = read_only_load(d_diag_i + row + lane_id);
   }
   if (lane_id == 0)
   {
      k = read_only_load(d_ib + row);
   }
   istart = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 1);
   bstart = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, k, 0);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += NALU_HYPRE_WARP_SIZE)
   {
      d_jb[p + i] = read_only_load(d_diag_j + i);
      d_ab[p + i] = read_only_load(d_diag_a + i);
   }

   /* offd part */
   if (lane_id < 2)
   {
      j = read_only_load(d_offd_i + row + lane_id);
   }
   bstart += iend - istart;
   istart = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 1);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int t = read_only_load(d_offd_j + i);
      d_jb[p + i] = (cols_offd_map ? read_only_load(&cols_offd_map[t]) : t) + diag_ncol;
      d_ab[p + i] = read_only_load(d_offd_a + i);
   }
}

nalu_hypre_CSRMatrix*
nalu_hypre_ConcatDiagAndOffdDevice(nalu_hypre_ParCSRMatrix *A)
{
   nalu_hypre_GpuProfilingPushRange("ConcatDiagAndOffdDevice");

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);

   nalu_hypre_CSRMatrix *B = nalu_hypre_CSRMatrixCreate( nalu_hypre_CSRMatrixNumRows(A_diag),
                                               nalu_hypre_CSRMatrixNumCols(A_diag) + nalu_hypre_CSRMatrixNumCols(A_offd),
                                               nalu_hypre_CSRMatrixNumNonzeros(A_diag) + nalu_hypre_CSRMatrixNumNonzeros(A_offd) );

   nalu_hypre_CSRMatrixInitialize_v2(B, 0, NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(nalu_hypre_CSRMatrixNumRows(B), NULL, nalu_hypre_CSRMatrixI(A_diag),
                         nalu_hypre_CSRMatrixI(A_offd), nalu_hypre_CSRMatrixI(B));

   hypreDevice_IntegerExclusiveScan(nalu_hypre_CSRMatrixNumRows(B) + 1, nalu_hypre_CSRMatrixI(B));

   const dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   const dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nalu_hypre_CSRMatrixNumRows(A_diag), "warp", bDim);

   NALU_HYPRE_Int  nrows = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int  diag_ncol = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int *d_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int *d_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex *d_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int *d_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int *d_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Complex *d_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int *cols_offd_map = NULL;
   NALU_HYPRE_Int *d_ib = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int *d_jb = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Complex *d_ab = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_ConcatDiagAndOffd,
                     gDim, bDim,
                     nrows,
                     diag_ncol,
                     d_diag_i,
                     d_diag_j,
                     d_diag_a,
                     d_offd_i,
                     d_offd_j,
                     d_offd_a,
                     cols_offd_map,
                     d_ib,
                     d_jb,
                     d_ab );

   nalu_hypre_GpuProfilingPopRange();

   return B;
}
#else
nalu_hypre_CSRMatrix*
nalu_hypre_ConcatDiagAndOffdDevice(nalu_hypre_ParCSRMatrix *A)
{
   nalu_hypre_CSRMatrix *A_diag     = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int       *A_diag_i   = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j   = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex   *A_diag_a   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int        A_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
   nalu_hypre_CSRMatrix *A_offd     = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int       *A_offd_i   = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j   = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Complex   *A_offd_a   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int        A_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_offd);

   nalu_hypre_CSRMatrix *B;
   NALU_HYPRE_Int        B_nrows = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int        B_ncols = nalu_hypre_CSRMatrixNumCols(A_diag) + nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Int        B_nnz   = A_diag_nnz + A_offd_nnz;
   NALU_HYPRE_Int       *B_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_nnz, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int       *B_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_nnz, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex   *B_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, B_nnz, NALU_HYPRE_MEMORY_DEVICE);

   // Adiag
   NALU_HYPRE_Int *A_diag_ii = hypreDevice_CsrRowPtrsToIndices(B_nrows, A_diag_nnz, A_diag_i);
   NALU_HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                      A_diag_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_j, B_a)) );
   nalu_hypre_TFree(A_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

   // Aoffd
   NALU_HYPRE_Int *A_offd_ii = hypreDevice_CsrRowPtrsToIndices(B_nrows, A_offd_nnz, A_offd_i);
   NALU_HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, A_offd_a)),
                      A_offd_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_a)) + A_diag_nnz );
   nalu_hypre_TFree(A_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_THRUST_CALL( transform,
                      A_offd_j,
                      A_offd_j + A_offd_nnz,
                      thrust::make_constant_iterator(nalu_hypre_CSRMatrixNumCols(A_diag)),
                      B_j + A_diag_nnz,
                      thrust::plus<NALU_HYPRE_Int>() );

   // B
   NALU_HYPRE_THRUST_CALL( stable_sort_by_key,
                      B_ii,
                      B_ii + B_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_j, B_a)) );

   NALU_HYPRE_Int *B_i = hypreDevice_CsrRowIndicesToPtrs(B_nrows, B_nnz, B_ii);
   nalu_hypre_TFree(B_ii, NALU_HYPRE_MEMORY_DEVICE);

   B = nalu_hypre_CSRMatrixCreate(B_nrows, B_ncols, B_nnz);
   nalu_hypre_CSRMatrixI(B) = B_i;
   nalu_hypre_CSRMatrixJ(B) = B_j;
   nalu_hypre_CSRMatrixData(B) = B_a;
   nalu_hypre_CSRMatrixMemoryLocation(B) = NALU_HYPRE_MEMORY_DEVICE;

   return B;
}
#endif

/* return B = [Adiag, Aoffd; E] */
#if 1
NALU_HYPRE_Int
nalu_hypre_ConcatDiagOffdAndExtDevice(nalu_hypre_ParCSRMatrix *A,
                                 nalu_hypre_CSRMatrix    *E,
                                 nalu_hypre_CSRMatrix   **B_ptr,
                                 NALU_HYPRE_Int          *num_cols_offd_ptr,
                                 NALU_HYPRE_BigInt      **cols_map_offd_ptr)
{
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_CSRMatrix *E_diag, *E_offd, *B;
   NALU_HYPRE_Int       *cols_offd_map, num_cols_offd;
   NALU_HYPRE_BigInt    *cols_map_offd;

   nalu_hypre_CSRMatrixSplitDevice(E, nalu_hypre_ParCSRMatrixFirstColDiag(A), nalu_hypre_ParCSRMatrixLastColDiag(A),
                              nalu_hypre_CSRMatrixNumCols(A_offd), nalu_hypre_ParCSRMatrixDeviceColMapOffd(A),
                              &cols_offd_map, &num_cols_offd, &cols_map_offd, &E_diag, &E_offd);

   B = nalu_hypre_CSRMatrixCreate(nalu_hypre_ParCSRMatrixNumRows(A) + nalu_hypre_CSRMatrixNumRows(E),
                             nalu_hypre_ParCSRMatrixNumCols(A) + num_cols_offd,
                             nalu_hypre_CSRMatrixNumNonzeros(A_diag) + nalu_hypre_CSRMatrixNumNonzeros(A_offd) +
                             nalu_hypre_CSRMatrixNumNonzeros(E));

   nalu_hypre_CSRMatrixInitialize_v2(B, 0, NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(nalu_hypre_ParCSRMatrixNumRows(A), NULL, nalu_hypre_CSRMatrixI(A_diag),
                         nalu_hypre_CSRMatrixI(A_offd), nalu_hypre_CSRMatrixI(B));
   hypreDevice_IntegerExclusiveScan(nalu_hypre_ParCSRMatrixNumRows(A) + 1, nalu_hypre_CSRMatrixI(B));

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nalu_hypre_ParCSRMatrixNumRows(A), "warp", bDim);

   NALU_HYPRE_Int  nrows = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int  diag_ncol = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int *d_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int *d_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex *d_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int *d_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int *d_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Complex *d_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int *d_ib = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int *d_jb = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Complex *d_ab = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_ConcatDiagAndOffd,
                     gDim, bDim,
                     nrows,
                     diag_ncol,
                     d_diag_i,
                     d_diag_j,
                     d_diag_a,
                     d_offd_i,
                     d_offd_j,
                     d_offd_a,
                     cols_offd_map,
                     d_ib,
                     d_jb,
                     d_ab );

   nalu_hypre_TFree(cols_offd_map, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TMemcpy(nalu_hypre_CSRMatrixI(B) + nalu_hypre_ParCSRMatrixNumRows(A) + 1, nalu_hypre_CSRMatrixI(E) + 1,
                 NALU_HYPRE_Int, nalu_hypre_CSRMatrixNumRows(E),
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#ifdef NALU_HYPRE_USING_SYCL
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      nalu_hypre_CSRMatrixI(B) + nalu_hypre_ParCSRMatrixNumRows(A) + 1,
                      nalu_hypre_CSRMatrixI(B) + nalu_hypre_ParCSRMatrixNumRows(A) + nalu_hypre_CSRMatrixNumRows(E) + 1,
                      nalu_hypre_CSRMatrixI(B) + nalu_hypre_ParCSRMatrixNumRows(A) + 1,
                      [const_val = nalu_hypre_CSRMatrixNumNonzeros(A_diag) + nalu_hypre_CSRMatrixNumNonzeros(A_offd)] (
   const auto & x) {return x + const_val;} );
#else
   NALU_HYPRE_THRUST_CALL( transform,
                      nalu_hypre_CSRMatrixI(B) + nalu_hypre_ParCSRMatrixNumRows(A) + 1,
                      nalu_hypre_CSRMatrixI(B) + nalu_hypre_ParCSRMatrixNumRows(A) + nalu_hypre_CSRMatrixNumRows(E) + 1,
                      thrust::make_constant_iterator(nalu_hypre_CSRMatrixNumNonzeros(A_diag) + nalu_hypre_CSRMatrixNumNonzeros(
                                                        A_offd)),
                      nalu_hypre_CSRMatrixI(B) + nalu_hypre_ParCSRMatrixNumRows(A) + 1,
                      thrust::plus<NALU_HYPRE_Int>() );
#endif

   gDim = nalu_hypre_GetDefaultDeviceGridDimension(nalu_hypre_CSRMatrixNumRows(E), "warp", bDim);

   nalu_hypre_assert(nalu_hypre_CSRMatrixNumCols(E_diag) == nalu_hypre_CSRMatrixNumCols(A_diag));

   nrows = nalu_hypre_CSRMatrixNumRows(E_diag);
   diag_ncol = nalu_hypre_CSRMatrixNumCols(E_diag);
   d_diag_i = nalu_hypre_CSRMatrixI(E_diag);
   d_diag_j = nalu_hypre_CSRMatrixJ(E_diag);
   d_diag_a = nalu_hypre_CSRMatrixData(E_diag);
   d_offd_i = nalu_hypre_CSRMatrixI(E_offd);
   d_offd_j = nalu_hypre_CSRMatrixJ(E_offd);
   d_offd_a = nalu_hypre_CSRMatrixData(E_offd);
   cols_offd_map = NULL;
   d_ib = nalu_hypre_CSRMatrixI(B) + nalu_hypre_ParCSRMatrixNumRows(A);
   d_jb = nalu_hypre_CSRMatrixJ(B);
   d_ab = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_ConcatDiagAndOffd,
                     gDim, bDim,
                     nrows,
                     diag_ncol,
                     d_diag_i,
                     d_diag_j,
                     d_diag_a,
                     d_offd_i,
                     d_offd_j,
                     d_offd_a,
                     cols_offd_map,
                     d_ib,
                     d_jb,
                     d_ab );

   nalu_hypre_CSRMatrixDestroy(E_diag);
   nalu_hypre_CSRMatrixDestroy(E_offd);

   *B_ptr = B;
   *num_cols_offd_ptr = num_cols_offd;
   *cols_map_offd_ptr = cols_map_offd;

   return nalu_hypre_error_flag;
}
#else
NALU_HYPRE_Int
nalu_hypre_ConcatDiagOffdAndExtDevice(nalu_hypre_ParCSRMatrix *A,
                                 nalu_hypre_CSRMatrix    *E,
                                 nalu_hypre_CSRMatrix   **B_ptr,
                                 NALU_HYPRE_Int          *num_cols_offd_ptr,
                                 NALU_HYPRE_BigInt      **cols_map_offd_ptr)
{
   nalu_hypre_CSRMatrix *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int        A_nrows         = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int        A_ncols         = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int       *A_diag_i        = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j        = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex   *A_diag_a        = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int        A_diag_nnz      = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
   nalu_hypre_CSRMatrix *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int       *A_offd_i        = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j        = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Complex   *A_offd_a        = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int        A_offd_nnz      = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
   NALU_HYPRE_BigInt     first_col_A     = nalu_hypre_ParCSRMatrixFirstColDiag(A);
   NALU_HYPRE_BigInt     last_col_A      = nalu_hypre_ParCSRMatrixLastColDiag(A);
   NALU_HYPRE_Int        num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt    *col_map_offd_A  = nalu_hypre_ParCSRMatrixDeviceColMapOffd(A);

   NALU_HYPRE_Int       *E_i     = nalu_hypre_CSRMatrixI(E);
   NALU_HYPRE_BigInt    *E_bigj  = nalu_hypre_CSRMatrixBigJ(E);
   NALU_HYPRE_Complex   *E_a     = nalu_hypre_CSRMatrixData(E);
   NALU_HYPRE_Int        E_nrows = nalu_hypre_CSRMatrixNumRows(E);
   NALU_HYPRE_Int        E_nnz   = nalu_hypre_CSRMatrixNumNonzeros(E);
   NALU_HYPRE_Int        E_diag_nnz, E_offd_nnz;

   nalu_hypre_CSRMatrix *B;
   NALU_HYPRE_Int        B_nnz   = A_diag_nnz + A_offd_nnz + E_nnz;
   NALU_HYPRE_Int       *B_ii    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_nnz, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int       *B_j     = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_nnz, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex   *B_a     = nalu_hypre_TAlloc(NALU_HYPRE_Complex, B_nnz, NALU_HYPRE_MEMORY_DEVICE);

   // E
   nalu_hypre_CSRMatrixSplitDevice_core(0, E_nrows, E_nnz, NULL, E_bigj, NULL, NULL, first_col_A,
                                   last_col_A, num_cols_offd_A,
                                   NULL, NULL, NULL, NULL, &E_diag_nnz, NULL, NULL, NULL, NULL, &E_offd_nnz,
                                   NULL, NULL, NULL, NULL);

   NALU_HYPRE_Int    *cols_offd_map, num_cols_offd;
   NALU_HYPRE_BigInt *cols_map_offd;
   NALU_HYPRE_Int *E_ii = hypreDevice_CsrRowPtrsToIndices(E_nrows, E_nnz, E_i);

   nalu_hypre_CSRMatrixSplitDevice_core(1,
                                   E_nrows, E_nnz, E_ii, E_bigj, E_a, NULL,
                                   first_col_A, last_col_A, num_cols_offd_A, col_map_offd_A,
                                   &cols_offd_map, &num_cols_offd, &cols_map_offd,
                                   &E_diag_nnz,
                                   B_ii + A_diag_nnz + A_offd_nnz,
                                   B_j  + A_diag_nnz + A_offd_nnz,
                                   B_a  + A_diag_nnz + A_offd_nnz,
                                   NULL,
                                   &E_offd_nnz,
                                   B_ii + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   B_j  + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   B_a  + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   NULL);
   nalu_hypre_TFree(E_ii, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_THRUST_CALL( transform,
                      B_ii + A_diag_nnz + A_offd_nnz,
                      B_ii + B_nnz,
                      thrust::make_constant_iterator(A_nrows),
                      B_ii + A_diag_nnz + A_offd_nnz,
                      thrust::plus<NALU_HYPRE_Int>() );

   // Adiag
   NALU_HYPRE_Int *A_diag_ii = hypreDevice_CsrRowPtrsToIndices(A_nrows, A_diag_nnz, A_diag_i);
   NALU_HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                      A_diag_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_j, B_a)) );
   nalu_hypre_TFree(A_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

   // Aoffd
   NALU_HYPRE_Int *A_offd_ii = hypreDevice_CsrRowPtrsToIndices(A_nrows, A_offd_nnz, A_offd_i);
   NALU_HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, A_offd_a)),
                      A_offd_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_a)) + A_diag_nnz );
   nalu_hypre_TFree(A_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_THRUST_CALL( gather,
                      A_offd_j,
                      A_offd_j + A_offd_nnz,
                      cols_offd_map,
                      B_j + A_diag_nnz);

   nalu_hypre_TFree(cols_offd_map, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_THRUST_CALL( transform,
                      B_j + A_diag_nnz,
                      B_j + A_diag_nnz + A_offd_nnz,
                      thrust::make_constant_iterator(A_ncols),
                      B_j + A_diag_nnz,
                      thrust::plus<NALU_HYPRE_Int>() );

   NALU_HYPRE_THRUST_CALL( transform,
                      B_j + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                      B_j + B_nnz,
                      thrust::make_constant_iterator(A_ncols),
                      B_j + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                      thrust::plus<NALU_HYPRE_Int>() );

   // B
   NALU_HYPRE_THRUST_CALL( stable_sort_by_key,
                      B_ii,
                      B_ii + B_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_j, B_a)) );

   NALU_HYPRE_Int *B_i = hypreDevice_CsrRowIndicesToPtrs(A_nrows + E_nrows, B_nnz, B_ii);
   nalu_hypre_TFree(B_ii, NALU_HYPRE_MEMORY_DEVICE);

   B = nalu_hypre_CSRMatrixCreate(A_nrows + E_nrows, A_ncols + num_cols_offd, B_nnz);
   nalu_hypre_CSRMatrixI(B) = B_i;
   nalu_hypre_CSRMatrixJ(B) = B_j;
   nalu_hypre_CSRMatrixData(B) = B_a;
   nalu_hypre_CSRMatrixMemoryLocation(B) = NALU_HYPRE_MEMORY_DEVICE;

   *B_ptr = B;
   *num_cols_offd_ptr = num_cols_offd;
   *cols_map_offd_ptr = cols_map_offd;

   return nalu_hypre_error_flag;
}
#endif

/* The input B_ext is a BigJ matrix, so is the output */
/* RL: TODO FIX the num of columns of the output (from B_ext 'big' num cols) */
NALU_HYPRE_Int
nalu_hypre_ExchangeExternalRowsDeviceInit( nalu_hypre_CSRMatrix      *B_ext,
                                      nalu_hypre_ParCSRCommPkg  *comm_pkg_A,
                                      NALU_HYPRE_Int             want_data,
                                      void                **request_ptr)
{
   MPI_Comm   comm             = nalu_hypre_ParCSRCommPkgComm(comm_pkg_A);
   NALU_HYPRE_Int  num_recvs        = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   NALU_HYPRE_Int *recv_procs       = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   NALU_HYPRE_Int *recv_vec_starts  = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   NALU_HYPRE_Int  num_sends        = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   NALU_HYPRE_Int *send_procs       = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
   NALU_HYPRE_Int *send_map_starts  = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);

   NALU_HYPRE_Int  num_elmts_send   = send_map_starts[num_sends];
   NALU_HYPRE_Int  num_elmts_recv   = recv_vec_starts[num_recvs];

   NALU_HYPRE_Int     *B_ext_i_d      = nalu_hypre_CSRMatrixI(B_ext);
   NALU_HYPRE_BigInt  *B_ext_j_d      = nalu_hypre_CSRMatrixBigJ(B_ext);
   NALU_HYPRE_Complex *B_ext_a_d      = nalu_hypre_CSRMatrixData(B_ext);
   NALU_HYPRE_Int      B_ext_ncols    = nalu_hypre_CSRMatrixNumCols(B_ext);
   NALU_HYPRE_Int      B_ext_nrows    = nalu_hypre_CSRMatrixNumRows(B_ext);
   NALU_HYPRE_Int      B_ext_nnz      = nalu_hypre_CSRMatrixNumNonzeros(B_ext);
   NALU_HYPRE_Int     *B_ext_rownnz_d = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_ext_nrows + 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *B_ext_rownnz_h = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_ext_nrows,     NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int     *B_ext_i_h      = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_ext_nrows + 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_assert(num_elmts_recv == B_ext_nrows);

   /* output matrix */
   nalu_hypre_CSRMatrix *B_int_d;
   NALU_HYPRE_Int        B_int_nrows = num_elmts_send;
   NALU_HYPRE_Int        B_int_ncols = B_ext_ncols;
   NALU_HYPRE_Int       *B_int_i_h   = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_int_nrows + 1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int       *B_int_i_d   = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_int_nrows + 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_BigInt    *B_int_j_d   = NULL;
   NALU_HYPRE_Complex   *B_int_a_d   = NULL;
   NALU_HYPRE_Int        B_int_nnz;

   nalu_hypre_ParCSRCommHandle *comm_handle, *comm_handle_j, *comm_handle_a;
   nalu_hypre_ParCSRCommPkg    *comm_pkg_j = NULL;

   NALU_HYPRE_Int *jdata_recv_vec_starts;
   NALU_HYPRE_Int *jdata_send_map_starts;

   NALU_HYPRE_Int i;
   NALU_HYPRE_Int num_procs, my_id;
   void    **vrequest;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   jdata_send_map_starts = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_sends + 1, NALU_HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * B_ext_rownnz contains the number of elements of row j
    * (to be determined through send_map_elmnts on the receiving end)
    *--------------------------------------------------------------------------*/
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL(std::adjacent_difference, B_ext_i_d, B_ext_i_d + B_ext_nrows + 1, B_ext_rownnz_d);
#else
   NALU_HYPRE_THRUST_CALL(adjacent_difference, B_ext_i_d, B_ext_i_d + B_ext_nrows + 1, B_ext_rownnz_d);
#endif
   nalu_hypre_TMemcpy(B_ext_rownnz_h, B_ext_rownnz_d + 1, NALU_HYPRE_Int, B_ext_nrows,
                 NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

   /*--------------------------------------------------------------------------
    * initialize communication: send/recv the row nnz
    * (note the use of comm_pkg_A, mode 12, as in transpose matvec
    *--------------------------------------------------------------------------*/
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(12, comm_pkg_A, B_ext_rownnz_h, B_int_i_h + 1);

   jdata_recv_vec_starts = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_recvs + 1, NALU_HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts[0] = 0;

   B_ext_i_h[0] = 0;
   nalu_hypre_TMemcpy(B_ext_i_h + 1, B_ext_rownnz_h, NALU_HYPRE_Int, B_ext_nrows, NALU_HYPRE_MEMORY_HOST,
                 NALU_HYPRE_MEMORY_HOST);
   for (i = 1; i <= B_ext_nrows; i++)
   {
      B_ext_i_h[i] += B_ext_i_h[i - 1];
   }

   nalu_hypre_assert(B_ext_i_h[B_ext_nrows] == B_ext_nnz);

   for (i = 1; i <= num_recvs; i++)
   {
      jdata_recv_vec_starts[i] = B_ext_i_h[recv_vec_starts[i]];
   }

   /* Create the communication package - note the order of send/recv is reversed */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_sends, send_procs, jdata_send_map_starts,
                                    num_recvs, recv_procs, jdata_recv_vec_starts,
                                    NULL,
                                    &comm_pkg_j);

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /*--------------------------------------------------------------------------
    * compute B_int: row nnz to row ptrs
    *--------------------------------------------------------------------------*/
   B_int_i_h[0] = 0;
   for (i = 1; i <= B_int_nrows; i++)
   {
      B_int_i_h[i] += B_int_i_h[i - 1];
   }

   B_int_nnz = B_int_i_h[B_int_nrows];

   B_int_j_d = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, B_int_nnz, NALU_HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      B_int_a_d = nalu_hypre_TAlloc(NALU_HYPRE_Complex, B_int_nnz, NALU_HYPRE_MEMORY_DEVICE);
   }

   for (i = 0; i <= num_sends; i++)
   {
      jdata_send_map_starts[i] = B_int_i_h[send_map_starts[i]];
   }

   /* RL: assume B_ext_a_d and B_ext_j_d are ready at input */
   /* send/recv CSR rows */
   if (want_data)
   {
      comm_handle_a = nalu_hypre_ParCSRCommHandleCreate_v2( 1, comm_pkg_j,
                                                       NALU_HYPRE_MEMORY_DEVICE, B_ext_a_d,
                                                       NALU_HYPRE_MEMORY_DEVICE, B_int_a_d );
   }
   else
   {
      comm_handle_a = NULL;
   }

   comm_handle_j = nalu_hypre_ParCSRCommHandleCreate_v2(21, comm_pkg_j,
                                                   NALU_HYPRE_MEMORY_DEVICE, B_ext_j_d,
                                                   NALU_HYPRE_MEMORY_DEVICE, B_int_j_d );

   nalu_hypre_TMemcpy(B_int_i_d, B_int_i_h, NALU_HYPRE_Int, B_int_nrows + 1, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_HOST);

   /* create CSR: on device */
   B_int_d = nalu_hypre_CSRMatrixCreate(B_int_nrows, B_int_ncols, B_int_nnz);
   nalu_hypre_CSRMatrixI(B_int_d)    = B_int_i_d;
   nalu_hypre_CSRMatrixBigJ(B_int_d) = B_int_j_d;
   nalu_hypre_CSRMatrixData(B_int_d) = B_int_a_d;
   nalu_hypre_CSRMatrixMemoryLocation(B_int_d) = NALU_HYPRE_MEMORY_DEVICE;

   /* output */
   vrequest = nalu_hypre_TAlloc(void *, 3, NALU_HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) B_int_d;

   *request_ptr = (void *) vrequest;

   /* free */
   nalu_hypre_TFree(B_ext_rownnz_d, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(B_ext_rownnz_h, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(B_ext_i_h,      NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(B_int_i_h,      NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_pkg_j, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

nalu_hypre_CSRMatrix*
nalu_hypre_ExchangeExternalRowsDeviceWait(void *vrequest)
{
   void **request = (void **) vrequest;

   nalu_hypre_ParCSRCommHandle *comm_handle_j = (nalu_hypre_ParCSRCommHandle *) request[0];
   nalu_hypre_ParCSRCommHandle *comm_handle_a = (nalu_hypre_ParCSRCommHandle *) request[1];
   nalu_hypre_CSRMatrix        *B_int_d       = (nalu_hypre_CSRMatrix *)        request[2];

   /* communication done */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_j);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_a);

   nalu_hypre_TFree(request, NALU_HYPRE_MEMORY_HOST);

   return B_int_d;
}

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixExtractBExtDeviceInit( nalu_hypre_ParCSRMatrix  *B,
                                         nalu_hypre_ParCSRMatrix  *A,
                                         NALU_HYPRE_Int            want_data,
                                         void               **request_ptr)
{
   nalu_hypre_assert( nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(B)) ==
                 nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(B)) );

   /*
   nalu_hypre_assert( nalu_hypre_GetActualMemLocation(
            nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(B))) == NALU_HYPRE_MEMORY_DEVICE );
   */

   if (!nalu_hypre_ParCSRMatrixCommPkg(A))
   {
      nalu_hypre_MatvecCommPkgCreate(A);
   }

   nalu_hypre_ParcsrGetExternalRowsDeviceInit(B,
                                         nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A)),
                                         nalu_hypre_ParCSRMatrixColMapOffd(A),
                                         nalu_hypre_ParCSRMatrixCommPkg(A),
                                         want_data,
                                         request_ptr);
   return nalu_hypre_error_flag;
}

nalu_hypre_CSRMatrix*
nalu_hypre_ParCSRMatrixExtractBExtDeviceWait(void *request)
{
   return nalu_hypre_ParcsrGetExternalRowsDeviceWait(request);
}

nalu_hypre_CSRMatrix*
nalu_hypre_ParCSRMatrixExtractBExtDevice( nalu_hypre_ParCSRMatrix *B,
                                     nalu_hypre_ParCSRMatrix *A,
                                     NALU_HYPRE_Int want_data )
{
   void *request;

   nalu_hypre_ParCSRMatrixExtractBExtDeviceInit(B, A, want_data, &request);
   return nalu_hypre_ParCSRMatrixExtractBExtDeviceWait(request);
}

NALU_HYPRE_Int
nalu_hypre_ParcsrGetExternalRowsDeviceInit( nalu_hypre_ParCSRMatrix   *A,
                                       NALU_HYPRE_Int             indices_len,
                                       NALU_HYPRE_BigInt         *indices,
                                       nalu_hypre_ParCSRCommPkg  *comm_pkg,
                                       NALU_HYPRE_Int             want_data,
                                       void                **request_ptr)
{
   NALU_HYPRE_Int      i, j;
   NALU_HYPRE_Int      num_sends, num_rows_send, num_nnz_send, num_recvs, num_rows_recv, num_nnz_recv;
   NALU_HYPRE_Int     *d_send_i, *send_i, *d_send_map, *d_recv_i, *recv_i;
   NALU_HYPRE_BigInt  *d_send_j, *d_recv_j;
   NALU_HYPRE_Int     *send_jstarts, *recv_jstarts;
   NALU_HYPRE_Complex *d_send_a = NULL, *d_recv_a = NULL;
   nalu_hypre_ParCSRCommPkg     *comm_pkg_j = NULL;
   nalu_hypre_ParCSRCommHandle  *comm_handle, *comm_handle_j, *comm_handle_a;
   /* NALU_HYPRE_Int global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A); */
   /* diag part of A */
   nalu_hypre_CSRMatrix *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex   *A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   /* NALU_HYPRE_Int local_num_rows  = nalu_hypre_CSRMatrixNumRows(A_diag); */
   /* off-diag part of A */
   nalu_hypre_CSRMatrix *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex   *A_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   /* NALU_HYPRE_Int       *row_starts      = nalu_hypre_ParCSRMatrixRowStarts(A); */
   /* NALU_HYPRE_Int        first_row       = nalu_hypre_ParCSRMatrixFirstRowIndex(A); */
   NALU_HYPRE_BigInt     first_col        = nalu_hypre_ParCSRMatrixFirstColDiag(A);
   NALU_HYPRE_BigInt    *col_map_offd_A   = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Int        num_cols_A_offd  = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt    *d_col_map_offd_A = nalu_hypre_ParCSRMatrixDeviceColMapOffd(A);

   MPI_Comm         comm  = nalu_hypre_ParCSRMatrixComm(A);

   NALU_HYPRE_Int        num_procs;
   NALU_HYPRE_Int        my_id;
   void           **vrequest;

   nalu_hypre_CSRMatrix *A_ext;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* number of sends (#procs) */
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* number of rows to send */
   num_rows_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   /* number of recvs (#procs) */
   num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   /* number of rows to recv */
   num_rows_recv = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);

   /* must be true if indices contains proper offd indices */
   nalu_hypre_assert(indices_len == num_rows_recv);

   /* send_i/recv_i:
    * the arrays to send and recv: we first send and recv the row lengths */
   d_send_i   = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_send + 1, NALU_HYPRE_MEMORY_DEVICE);
   d_send_map = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_send,     NALU_HYPRE_MEMORY_DEVICE);
   send_i     = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_send,     NALU_HYPRE_MEMORY_HOST);
   recv_i     = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_recv + 1, NALU_HYPRE_MEMORY_HOST);
   d_recv_i   = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_recv + 1, NALU_HYPRE_MEMORY_DEVICE);

   /* fill the send array with row lengths */
   nalu_hypre_TMemcpy(d_send_map, nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg), NALU_HYPRE_Int,
                 num_rows_send, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_Memset(d_send_i, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);
   hypreDevice_GetRowNnz(num_rows_send, d_send_map, A_diag_i, A_offd_i, d_send_i + 1);

   /* send array send_i out: deviceTohost first and MPI (async)
    * note the shift in recv_i by one */
   nalu_hypre_TMemcpy(send_i, d_send_i + 1, NALU_HYPRE_Int, num_rows_send, NALU_HYPRE_MEMORY_HOST,
                 NALU_HYPRE_MEMORY_DEVICE);

   comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, send_i, recv_i + 1);

   hypreDevice_IntegerInclusiveScan(num_rows_send + 1, d_send_i);

   /* total number of nnz to send */
   nalu_hypre_TMemcpy(&num_nnz_send, d_send_i + num_rows_send, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST,
                 NALU_HYPRE_MEMORY_DEVICE);

   /* prepare data to send out. overlap with the above commmunication */
   d_send_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_nnz_send, NALU_HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_send_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_nnz_send, NALU_HYPRE_MEMORY_DEVICE);
   }

   if (d_col_map_offd_A == NULL)
   {
      d_col_map_offd_A = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, NALU_HYPRE_BigInt, num_cols_A_offd,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(A) = d_col_map_offd_A;
   }

   /* job == 2, d_send_i is input that contains row ptrs (length num_rows_send) */
   hypreDevice_CopyParCSRRows(num_rows_send, d_send_map, 2, num_procs > 1,
                              first_col, d_col_map_offd_A,
                              A_diag_i, A_diag_j, A_diag_a,
                              A_offd_i, A_offd_j, A_offd_a,
                              d_send_i, d_send_j, d_send_a);

   /* pointers to each proc in send_j */
   send_jstarts = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_sends + 1, NALU_HYPRE_MEMORY_HOST);
   send_jstarts[0] = 0;
   for (i = 1; i <= num_sends; i++)
   {
      send_jstarts[i] = send_jstarts[i - 1];
      for ( j = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i - 1);
            j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            j++ )
      {
         send_jstarts[i] += send_i[j];
      }
   }
   nalu_hypre_assert(send_jstarts[num_sends] == num_nnz_send);

   /* finish the above communication: send_i/recv_i */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* adjust recv_i to ptrs */
   recv_i[0] = 0;
   for (i = 1; i <= num_rows_recv; i++)
   {
      recv_i[i] += recv_i[i - 1];
   }
   num_nnz_recv = recv_i[num_rows_recv];

   /* allocate device memory for j and a */
   d_recv_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_nnz_recv, NALU_HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_recv_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_nnz_recv, NALU_HYPRE_MEMORY_DEVICE);
   }

   recv_jstarts = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_recvs + 1, NALU_HYPRE_MEMORY_HOST);
   recv_jstarts[0] = 0;
   for (i = 1; i <= num_recvs; i++)
   {
      j = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      recv_jstarts[i] = recv_i[j];
   }

   /* ready to send and recv: create a communication package for data */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    recv_jstarts,
                                    num_sends,
                                    nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    send_jstarts,
                                    NULL,
                                    &comm_pkg_j);

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI)
   /* RL: make sure d_send_j/d_send_a is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   /* init communication */
   /* ja */
   comm_handle_j = nalu_hypre_ParCSRCommHandleCreate_v2(21, comm_pkg_j,
                                                   NALU_HYPRE_MEMORY_DEVICE, d_send_j,
                                                   NALU_HYPRE_MEMORY_DEVICE, d_recv_j);
   if (want_data)
   {
      /* a */
      comm_handle_a = nalu_hypre_ParCSRCommHandleCreate_v2(1, comm_pkg_j,
                                                      NALU_HYPRE_MEMORY_DEVICE, d_send_a,
                                                      NALU_HYPRE_MEMORY_DEVICE, d_recv_a);
   }
   else
   {
      comm_handle_a = NULL;
   }

   nalu_hypre_TMemcpy(d_recv_i, recv_i, NALU_HYPRE_Int, num_rows_recv + 1, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_HOST);

   /* create A_ext: on device */
   A_ext = nalu_hypre_CSRMatrixCreate(num_rows_recv, nalu_hypre_ParCSRMatrixGlobalNumCols(A), num_nnz_recv);
   nalu_hypre_CSRMatrixI   (A_ext) = d_recv_i;
   nalu_hypre_CSRMatrixBigJ(A_ext) = d_recv_j;
   nalu_hypre_CSRMatrixData(A_ext) = d_recv_a;
   nalu_hypre_CSRMatrixMemoryLocation(A_ext) = NALU_HYPRE_MEMORY_DEVICE;

   /* output */
   vrequest = nalu_hypre_TAlloc(void *, 3, NALU_HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) A_ext;

   *request_ptr = (void *) vrequest;

   /* free */
   nalu_hypre_TFree(send_i,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_i,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(d_send_i,   NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_send_map, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_pkg_j, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

nalu_hypre_CSRMatrix*
nalu_hypre_ParcsrGetExternalRowsDeviceWait(void *vrequest)
{
   void **request = (void **) vrequest;

   nalu_hypre_ParCSRCommHandle *comm_handle_j = (nalu_hypre_ParCSRCommHandle *) request[0];
   nalu_hypre_ParCSRCommHandle *comm_handle_a = (nalu_hypre_ParCSRCommHandle *) request[1];
   nalu_hypre_CSRMatrix        *A_ext         = (nalu_hypre_CSRMatrix *)        request[2];
   NALU_HYPRE_BigInt           *send_j        = comm_handle_j ? (NALU_HYPRE_BigInt *)
                                           nalu_hypre_ParCSRCommHandleSendData(comm_handle_j) : NULL;
   NALU_HYPRE_Complex          *send_a        = comm_handle_a ? (NALU_HYPRE_Complex *)
                                           nalu_hypre_ParCSRCommHandleSendData(comm_handle_a) : NULL;

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_j);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_a);

   nalu_hypre_TFree(send_j, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(send_a, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TFree(request, NALU_HYPRE_MEMORY_HOST);

   return A_ext;
}

NALU_HYPRE_Int
nalu_hypre_ParCSRCommPkgCreateMatrixE( nalu_hypre_ParCSRCommPkg  *comm_pkg,
                                  NALU_HYPRE_Int             num_cols )
{
   /* Input variables */
   NALU_HYPRE_Int        num_sends      = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int        num_elements   = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   NALU_HYPRE_Int        num_components = nalu_hypre_ParCSRCommPkgNumComponents(comm_pkg);
   NALU_HYPRE_Int       *send_map       = nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
   NALU_HYPRE_Int       *send_map_def;

   /* Local variables */
   nalu_hypre_CSRMatrix *matrix_E;
   NALU_HYPRE_Int       *e_i;
   NALU_HYPRE_Int       *e_ii;
   NALU_HYPRE_Int       *e_j;
   NALU_HYPRE_Int       *new_end;
   NALU_HYPRE_Int        nid;

   /* Update number of elements exchanged when communicating multivectors */
   num_elements /= num_components;

   /* Create matrix_E */
   matrix_E = nalu_hypre_CSRMatrixCreate(num_cols, num_elements, num_elements);
   nalu_hypre_CSRMatrixMemoryLocation(matrix_E) = NALU_HYPRE_MEMORY_DEVICE;

   /* Build default (original) send_map_elements array */
   if (num_components > 1)
   {
      send_map_def = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_elements, NALU_HYPRE_MEMORY_DEVICE);
      hypreDevice_IntStridedCopy(num_elements, num_components, send_map, send_map_def);
   }
   else
   {
      send_map_def = send_map;
   }

   /* Allocate arrays */
   e_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_elements, NALU_HYPRE_MEMORY_DEVICE);
   e_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_elements, NALU_HYPRE_MEMORY_DEVICE);

   /* Build e_ii and e_j */
   nalu_hypre_TMemcpy(e_ii, send_map_def, NALU_HYPRE_Int, num_elements,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_sequence(e_j, e_j + num_elements, 0);
   hypreSycl_stable_sort_by_key(e_ii, e_ii + num_elements, e_j);
#else
   NALU_HYPRE_THRUST_CALL(sequence, e_j, e_j + num_elements);
   NALU_HYPRE_THRUST_CALL(stable_sort_by_key, e_ii, e_ii + num_elements, e_j);
#endif

   /* Construct row pointers from row indices */
   e_i = hypreDevice_CsrRowIndicesToPtrs(num_cols, num_elements, e_ii);

   /* Find row indices with nonzero coefficients */
#if defined(NALU_HYPRE_USING_SYCL)
   new_end = NALU_HYPRE_ONEDPL_CALL(std::unique, e_ii, e_ii + num_elements);
#else
   new_end = NALU_HYPRE_THRUST_CALL(unique, e_ii, e_ii + num_elements);
#endif
   nid = new_end - e_ii;
   e_ii = nalu_hypre_TReAlloc_v2(e_ii, NALU_HYPRE_Int, num_elements,
                            NALU_HYPRE_Int, nid, NALU_HYPRE_MEMORY_DEVICE);

   /* Set matrix_E pointers */
   nalu_hypre_CSRMatrixI(matrix_E) = e_i;
   nalu_hypre_CSRMatrixJ(matrix_E) = e_j;
   nalu_hypre_CSRMatrixNumRownnz(matrix_E) = nid;
   nalu_hypre_CSRMatrixRownnz(matrix_E) = e_ii;

   /* Set matrix_E */
   nalu_hypre_ParCSRCommPkgMatrixE(comm_pkg) = matrix_E;

   /* Free memory */
   if (num_components > 1)
   {
      nalu_hypre_TFree(send_map_def, NALU_HYPRE_MEMORY_DEVICE);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixCompressOffdMapDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixCompressOffdMapDevice(nalu_hypre_ParCSRMatrix *A)
{
   nalu_hypre_ParCSRMatrixCopyColMapOffdToDevice(A);

   nalu_hypre_CSRMatrix *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int        num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt    *col_map_offd_A  = nalu_hypre_ParCSRMatrixDeviceColMapOffd(A);
   NALU_HYPRE_BigInt    *col_map_offd_A_new;
   NALU_HYPRE_Int        num_cols_A_offd_new;

   nalu_hypre_CSRMatrixCompressColumnsDevice(A_offd, col_map_offd_A, NULL, &col_map_offd_A_new);

   num_cols_A_offd_new = nalu_hypre_CSRMatrixNumCols(A_offd);

   if (num_cols_A_offd_new < num_cols_A_offd)
   {
      nalu_hypre_TFree(col_map_offd_A, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(A) = col_map_offd_A_new;

      nalu_hypre_ParCSRMatrixColMapOffd(A) = nalu_hypre_TReAlloc(nalu_hypre_ParCSRMatrixColMapOffd(A),
                                                       NALU_HYPRE_BigInt, num_cols_A_offd_new,
                                                       NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(A),
                    nalu_hypre_ParCSRMatrixDeviceColMapOffd(A),
                    NALU_HYPRE_BigInt, num_cols_A_offd_new,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   }

   return nalu_hypre_error_flag;
}

/* Get element-wise tolerances based on row norms for ParCSRMatrix
 * NOTE: Keep the diagonal, i.e. elmt_tol = 0.0 for diagonals
 * Output vectors have size nnz:
 *    elmt_tols_diag[j] = tol * (norm of row i) for j in [ A_diag_i[i] , A_diag_i[i+1] )
 *    elmt_tols_offd[j] = tol * (norm of row i) for j in [ A_offd_i[i] , A_offd_i[i+1] )
 * type == -1, infinity norm,
 *         1, 1-norm
 *         2, 2-norm
 */
template<NALU_HYPRE_Int type>
__global__ void
nalu_hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols( nalu_hypre_DeviceItem &item,
                                                      NALU_HYPRE_Int      nrows,
                                                      NALU_HYPRE_Real     tol,
                                                      NALU_HYPRE_Int     *A_diag_i,
                                                      NALU_HYPRE_Int     *A_diag_j,
                                                      NALU_HYPRE_Complex *A_diag_a,
                                                      NALU_HYPRE_Int     *A_offd_i,
                                                      NALU_HYPRE_Complex *A_offd_a,
                                                      NALU_HYPRE_Real     *elmt_tols_diag,
                                                      NALU_HYPRE_Real     *elmt_tols_offd)
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p_diag = 0, p_offd = 0, q_diag, q_offd;

   /* sum row norm over diag part */
   if (lane < 2)
   {
      p_diag = read_only_load(A_diag_i + row_i + lane);
   }
   q_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 0);

   NALU_HYPRE_Real row_norm_i = 0.0;

   for (NALU_HYPRE_Int j = p_diag + lane; j < q_diag; j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Complex val = A_diag_a[j];

      if (type == -1)
      {
         row_norm_i = nalu_hypre_max(row_norm_i, nalu_hypre_cabs(val));
      }
      else if (type == 1)
      {
         row_norm_i += nalu_hypre_cabs(val);
      }
      else if (type == 2)
      {
         row_norm_i += val * val;
      }
   }

   /* sum row norm over offd part */
   if (lane < 2)
   {
      p_offd = read_only_load(A_offd_i + row_i + lane);
   }
   q_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (NALU_HYPRE_Int j = p_offd + lane; j < q_offd; j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Complex val = A_offd_a[j];

      if (type == -1)
      {
         row_norm_i = nalu_hypre_max(row_norm_i, nalu_hypre_cabs(val));
      }
      else if (type == 1)
      {
         row_norm_i += nalu_hypre_cabs(val);
      }
      else if (type == 2)
      {
         row_norm_i += val * val;
      }
   }

   /* allreduce to get the row norm on all threads */
   if (type == -1)
   {
      row_norm_i = warp_allreduce_max(item, row_norm_i);
   }
   else
   {
      row_norm_i = warp_allreduce_sum(item, row_norm_i);
   }
   if (type == 2)
   {
      row_norm_i = nalu_hypre_sqrt(row_norm_i);
   }

   /* set elmt_tols_diag */
   for (NALU_HYPRE_Int j = p_diag + lane; j < q_diag; j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int col = A_diag_j[j];

      /* elmt_tol = 0.0 ensures diagonal will be kept */
      if (col == row_i)
      {
         elmt_tols_diag[j] = 0.0;
      }
      else
      {
         elmt_tols_diag[j] = tol * row_norm_i;
      }
   }

   /* set elmt_tols_offd */
   for (NALU_HYPRE_Int j = p_offd + lane; j < q_offd; j += NALU_HYPRE_WARP_SIZE)
   {
      elmt_tols_offd[j] = tol * row_norm_i;
   }

}

/* drop the entries that are not on the diagonal and smaller than:
 *    type 0: tol
 *    type 1: tol*(1-norm of row)
 *    type 2: tol*(2-norm of row)
 *    type -1: tol*(infinity norm of row) */
NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixDropSmallEntriesDevice( nalu_hypre_ParCSRMatrix *A,
                                          NALU_HYPRE_Complex       tol,
                                          NALU_HYPRE_Int           type)
{
   nalu_hypre_CSRMatrix *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int        num_cols_A_offd  = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt    *h_col_map_offd_A = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_BigInt    *col_map_offd_A = nalu_hypre_ParCSRMatrixDeviceColMapOffd(A);

   NALU_HYPRE_Real      *elmt_tols_diag = NULL;
   NALU_HYPRE_Real      *elmt_tols_offd = NULL;

   /* Exit if tolerance is zero */
   if (tol < NALU_HYPRE_REAL_MIN)
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_GpuProfilingPushRange("ParCSRMatrixDropSmallEntries");

   if (col_map_offd_A == NULL)
   {
      col_map_offd_A = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(col_map_offd_A, h_col_map_offd_A, NALU_HYPRE_BigInt, num_cols_A_offd,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(A) = col_map_offd_A;
   }

   /* get elmement-wise tolerances if needed */
   if (type != 0)
   {
      elmt_tols_diag = nalu_hypre_TAlloc(NALU_HYPRE_Real, nalu_hypre_CSRMatrixNumNonzeros(A_diag), NALU_HYPRE_MEMORY_DEVICE);
      elmt_tols_offd = nalu_hypre_TAlloc(NALU_HYPRE_Real, nalu_hypre_CSRMatrixNumNonzeros(A_offd), NALU_HYPRE_MEMORY_DEVICE);
   }

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nalu_hypre_CSRMatrixNumRows(A_diag), "warp", bDim);

   NALU_HYPRE_Int A_diag_nrows = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Complex *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   if (type == -1)
   {
      NALU_HYPRE_GPU_LAUNCH( nalu_hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols < -1 >, gDim, bDim,
                        A_diag_nrows, tol, A_diag_i,
                        A_diag_j, A_diag_data, A_offd_i,
                        A_offd_data, elmt_tols_diag, elmt_tols_offd);
   }
   if (type == 1)
   {
      NALU_HYPRE_GPU_LAUNCH( nalu_hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols<1>, gDim, bDim,
                        A_diag_nrows, tol, A_diag_i,
                        A_diag_j, A_diag_data, A_offd_i,
                        A_offd_data, elmt_tols_diag, elmt_tols_offd);
   }
   if (type == 2)
   {
      NALU_HYPRE_GPU_LAUNCH( nalu_hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols<2>, gDim, bDim,
                        A_diag_nrows, tol, A_diag_i,
                        A_diag_j, A_diag_data, A_offd_i,
                        A_offd_data, elmt_tols_diag, elmt_tols_offd);
   }

   /* drop entries from diag and offd CSR matrices */
   nalu_hypre_CSRMatrixDropSmallEntriesDevice(A_diag, tol, elmt_tols_diag);
   nalu_hypre_CSRMatrixDropSmallEntriesDevice(A_offd, tol, elmt_tols_offd);

   nalu_hypre_ParCSRMatrixSetNumNonzeros(A);
   nalu_hypre_ParCSRMatrixDNumNonzeros(A) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(A);

   /* squeeze out zero columns of A_offd */
   nalu_hypre_ParCSRMatrixCompressOffdMapDevice(A);

   if (type != 0)
   {
      nalu_hypre_TFree(elmt_tols_diag, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(elmt_tols_offd, NALU_HYPRE_MEMORY_DEVICE);
   }

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

nalu_hypre_CSRMatrix*
nalu_hypre_MergeDiagAndOffdDevice(nalu_hypre_ParCSRMatrix *A)
{
   MPI_Comm         comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex   *A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex   *A_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int        local_num_rows   = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt     glbal_num_cols   = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   NALU_HYPRE_BigInt     first_col        = nalu_hypre_ParCSRMatrixFirstColDiag(A);
   NALU_HYPRE_Int        num_cols_A_offd  = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt    *col_map_offd_A   = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_BigInt    *d_col_map_offd_A = nalu_hypre_ParCSRMatrixDeviceColMapOffd(A);

   nalu_hypre_CSRMatrix *B;
   NALU_HYPRE_Int        B_nrows = local_num_rows;
   NALU_HYPRE_BigInt     B_ncols = glbal_num_cols;
   NALU_HYPRE_Int       *B_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_nrows + 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_BigInt    *B_j;
   NALU_HYPRE_Complex   *B_a;
   NALU_HYPRE_Int        B_nnz;

   NALU_HYPRE_Int        num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   nalu_hypre_Memset(B_i, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(B_nrows, NULL, A_diag_i, A_offd_i, B_i + 1);

   hypreDevice_IntegerInclusiveScan(B_nrows + 1, B_i);

   /* total number of nnz */
   nalu_hypre_TMemcpy(&B_nnz, B_i + B_nrows, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

   B_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  B_nnz, NALU_HYPRE_MEMORY_DEVICE);
   B_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, B_nnz, NALU_HYPRE_MEMORY_DEVICE);

   if (d_col_map_offd_A == NULL)
   {
      d_col_map_offd_A = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, NALU_HYPRE_BigInt, num_cols_A_offd,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(A) = d_col_map_offd_A;
   }

   hypreDevice_CopyParCSRRows(B_nrows, NULL, 2, num_procs > 1, first_col, d_col_map_offd_A,
                              A_diag_i, A_diag_j, A_diag_a, A_offd_i, A_offd_j, A_offd_a,
                              B_i, B_j, B_a);

   /* output */
   B = nalu_hypre_CSRMatrixCreate(B_nrows, B_ncols, B_nnz);
   nalu_hypre_CSRMatrixI   (B) = B_i;
   nalu_hypre_CSRMatrixBigJ(B) = B_j;
   nalu_hypre_CSRMatrixData(B) = B_a;
   nalu_hypre_CSRMatrixMemoryLocation(B) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return B;
}

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGetRowDevice( nalu_hypre_ParCSRMatrix  *mat,
                                NALU_HYPRE_BigInt         row,
                                NALU_HYPRE_Int           *size,
                                NALU_HYPRE_BigInt       **col_ind,
                                NALU_HYPRE_Complex      **values )
{
   NALU_HYPRE_Int nrows, local_row;
   NALU_HYPRE_BigInt row_start, row_end;
   nalu_hypre_CSRMatrix *Aa;
   nalu_hypre_CSRMatrix *Ba;

   if (!mat)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   Aa = (nalu_hypre_CSRMatrix *) nalu_hypre_ParCSRMatrixDiag(mat);
   Ba = (nalu_hypre_CSRMatrix *) nalu_hypre_ParCSRMatrixOffd(mat);

   if (nalu_hypre_ParCSRMatrixGetrowactive(mat))
   {
      return (-1);
   }

   nalu_hypre_ParCSRMatrixGetrowactive(mat) = 1;

   row_start = nalu_hypre_ParCSRMatrixFirstRowIndex(mat);
   row_end = nalu_hypre_ParCSRMatrixLastRowIndex(mat) + 1;
   nrows = row_end - row_start;

   if (row < row_start || row >= row_end)
   {
      return (-1);
   }

   local_row = row - row_start;

   /* if buffer is not allocated and some information is requested, allocate buffer with the max row_nnz */
   if ( !nalu_hypre_ParCSRMatrixRowvalues(mat) && (col_ind || values) )
   {
      NALU_HYPRE_Int max_row_nnz;
      NALU_HYPRE_Int *row_nnz = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows, NALU_HYPRE_MEMORY_DEVICE);

      hypreDevice_GetRowNnz(nrows, NULL, nalu_hypre_CSRMatrixI(Aa), nalu_hypre_CSRMatrixI(Ba), row_nnz);

      nalu_hypre_TMemcpy(size, row_nnz + local_row, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      max_row_nnz = NALU_HYPRE_ONEDPL_CALL(std::reduce, row_nnz, row_nnz + nrows, 0,
                                      oneapi::dpl::maximum<NALU_HYPRE_Int>());
#else
      max_row_nnz = NALU_HYPRE_THRUST_CALL(reduce, row_nnz, row_nnz + nrows, 0, thrust::maximum<NALU_HYPRE_Int>());
#endif

      /*
            NALU_HYPRE_Int *max_row_nnz_d = NALU_HYPRE_THRUST_CALL(max_element, row_nnz, row_nnz + nrows);
            nalu_hypre_TMemcpy( &max_row_nnz, max_row_nnz_d,
                           NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE );
      */

      nalu_hypre_TFree(row_nnz, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRMatrixRowvalues(mat)  =
         (NALU_HYPRE_Complex *) nalu_hypre_TAlloc(NALU_HYPRE_Complex, max_row_nnz, nalu_hypre_ParCSRMatrixMemoryLocation(mat));
      nalu_hypre_ParCSRMatrixRowindices(mat) =
         (NALU_HYPRE_BigInt *)  nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  max_row_nnz, nalu_hypre_ParCSRMatrixMemoryLocation(mat));
   }
   else
   {
      NALU_HYPRE_Int *size_d = nalu_hypre_TAlloc(NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_DEVICE);
      hypreDevice_GetRowNnz(1, NULL, nalu_hypre_CSRMatrixI(Aa) + local_row, nalu_hypre_CSRMatrixI(Ba) + local_row,
                            size_d);
      nalu_hypre_TMemcpy(size, size_d, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(size_d, NALU_HYPRE_MEMORY_DEVICE);
   }

   if (col_ind || values)
   {
      if (nalu_hypre_ParCSRMatrixDeviceColMapOffd(mat) == NULL)
      {
         nalu_hypre_ParCSRMatrixDeviceColMapOffd(mat) =
            nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nalu_hypre_CSRMatrixNumCols(Ba), NALU_HYPRE_MEMORY_DEVICE);

         nalu_hypre_TMemcpy( nalu_hypre_ParCSRMatrixDeviceColMapOffd(mat),
                        nalu_hypre_ParCSRMatrixColMapOffd(mat),
                        NALU_HYPRE_BigInt,
                        nalu_hypre_CSRMatrixNumCols(Ba),
                        NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST );
      }

      hypreDevice_CopyParCSRRows( 1, NULL, -1, Ba != NULL,
                                  nalu_hypre_ParCSRMatrixFirstColDiag(mat),
                                  nalu_hypre_ParCSRMatrixDeviceColMapOffd(mat),
                                  nalu_hypre_CSRMatrixI(Aa) + local_row,
                                  nalu_hypre_CSRMatrixJ(Aa),
                                  nalu_hypre_CSRMatrixData(Aa),
                                  nalu_hypre_CSRMatrixI(Ba) + local_row,
                                  nalu_hypre_CSRMatrixJ(Ba),
                                  nalu_hypre_CSRMatrixData(Ba),
                                  NULL,
                                  nalu_hypre_ParCSRMatrixRowindices(mat),
                                  nalu_hypre_ParCSRMatrixRowvalues(mat) );
   }

   if (col_ind)
   {
      *col_ind = nalu_hypre_ParCSRMatrixRowindices(mat);
   }

   if (values)
   {
      *values = nalu_hypre_ParCSRMatrixRowvalues(mat);
   }

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixTransposeDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixTransposeDevice( nalu_hypre_ParCSRMatrix  *A,
                                   nalu_hypre_ParCSRMatrix **AT_ptr,
                                   NALU_HYPRE_Int            data )
{
   nalu_hypre_CSRMatrix    *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix    *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_CSRMatrix    *A_diagT;
   nalu_hypre_CSRMatrix    *AT_offd;
   NALU_HYPRE_Int           num_procs;
   NALU_HYPRE_Int           num_cols_offd_AT = 0;
   NALU_HYPRE_BigInt       *col_map_offd_AT = NULL;
   nalu_hypre_ParCSRMatrix *AT;

   nalu_hypre_MPI_Comm_size(nalu_hypre_ParCSRMatrixComm(A), &num_procs);

   if (num_procs > 1)
   {
      void *request;
      nalu_hypre_CSRMatrix *A_offdT, *Aext;
      NALU_HYPRE_Int *Aext_ii, *Aext_j, Aext_nnz;
      NALU_HYPRE_Complex *Aext_data;
      NALU_HYPRE_BigInt *tmp_bigj;

      nalu_hypre_CSRMatrixTranspose(A_offd, &A_offdT, data);
      nalu_hypre_CSRMatrixBigJ(A_offdT) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nalu_hypre_CSRMatrixNumNonzeros(A_offdT),
                                                  NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         nalu_hypre_CSRMatrixJ(A_offdT),
                         nalu_hypre_CSRMatrixJ(A_offdT) + nalu_hypre_CSRMatrixNumNonzeros(A_offdT),
                         nalu_hypre_CSRMatrixBigJ(A_offdT),
      [y = nalu_hypre_ParCSRMatrixFirstRowIndex(A)] (const auto & x) {return x + y;} );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         nalu_hypre_CSRMatrixJ(A_offdT),
                         nalu_hypre_CSRMatrixJ(A_offdT) + nalu_hypre_CSRMatrixNumNonzeros(A_offdT),
                         thrust::make_constant_iterator(nalu_hypre_ParCSRMatrixFirstRowIndex(A)),
                         nalu_hypre_CSRMatrixBigJ(A_offdT),
                         thrust::plus<NALU_HYPRE_BigInt>() );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure A_offdT is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      if (!nalu_hypre_ParCSRMatrixCommPkg(A))
      {
         nalu_hypre_MatvecCommPkgCreate(A);
      }

      nalu_hypre_ExchangeExternalRowsDeviceInit(A_offdT, nalu_hypre_ParCSRMatrixCommPkg(A), data, &request);

      nalu_hypre_CSRMatrixTranspose(A_diag, &A_diagT, data);

      Aext = nalu_hypre_ExchangeExternalRowsDeviceWait(request);

      nalu_hypre_CSRMatrixDestroy(A_offdT);

      // Aext contains offd of AT
      Aext_nnz = nalu_hypre_CSRMatrixNumNonzeros(Aext);
      Aext_ii = hypreDevice_CsrRowPtrsToIndices(nalu_hypre_CSRMatrixNumRows(Aext), Aext_nnz,
                                                nalu_hypre_CSRMatrixI(Aext));

      nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(nalu_hypre_ParCSRMatrixCommPkg(A));

#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_gather( Aext_ii,
                        Aext_ii + Aext_nnz,
                        nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(nalu_hypre_ParCSRMatrixCommPkg(A)),
                        Aext_ii );
#else
      NALU_HYPRE_THRUST_CALL( gather,
                         Aext_ii,
                         Aext_ii + Aext_nnz,
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(nalu_hypre_ParCSRMatrixCommPkg(A)),
                         Aext_ii );
#endif

      tmp_bigj = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, Aext_nnz, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(tmp_bigj, nalu_hypre_CSRMatrixBigJ(Aext), NALU_HYPRE_BigInt, Aext_nnz, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::sort,
                         tmp_bigj,
                         tmp_bigj + Aext_nnz );

      NALU_HYPRE_BigInt *new_end = NALU_HYPRE_ONEDPL_CALL( std::unique,
                                                 tmp_bigj,
                                                 tmp_bigj + Aext_nnz );
#else
      NALU_HYPRE_THRUST_CALL( sort,
                         tmp_bigj,
                         tmp_bigj + Aext_nnz );

      NALU_HYPRE_BigInt *new_end = NALU_HYPRE_THRUST_CALL( unique,
                                                 tmp_bigj,
                                                 tmp_bigj + Aext_nnz );
#endif

      num_cols_offd_AT = new_end - tmp_bigj;
      col_map_offd_AT = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_AT, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(col_map_offd_AT, tmp_bigj, NALU_HYPRE_BigInt, num_cols_offd_AT, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_TFree(tmp_bigj, NALU_HYPRE_MEMORY_DEVICE);

      Aext_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, Aext_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                         col_map_offd_AT,
                         col_map_offd_AT + num_cols_offd_AT,
                         nalu_hypre_CSRMatrixBigJ(Aext),
                         nalu_hypre_CSRMatrixBigJ(Aext) + Aext_nnz,
                         Aext_j );
#else
      NALU_HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_AT,
                         col_map_offd_AT + num_cols_offd_AT,
                         nalu_hypre_CSRMatrixBigJ(Aext),
                         nalu_hypre_CSRMatrixBigJ(Aext) + Aext_nnz,
                         Aext_j );
#endif

      Aext_data = nalu_hypre_CSRMatrixData(Aext);
      nalu_hypre_CSRMatrixData(Aext) = NULL;
      nalu_hypre_CSRMatrixDestroy(Aext);

      if (data)
      {
         hypreDevice_StableSortByTupleKey(Aext_nnz, Aext_ii, Aext_j, Aext_data, 0);
      }
      else
      {
#if defined(NALU_HYPRE_USING_SYCL)
         NALU_HYPRE_ONEDPL_CALL( std::stable_sort,
                            oneapi::dpl::make_zip_iterator(Aext_ii, Aext_j),
                            oneapi::dpl::make_zip_iterator(Aext_ii, Aext_j) + Aext_nnz,
         [] (const auto & x, const auto & y) {return std::get<0>(x) < std::get<0>(y);} );
#else
         NALU_HYPRE_THRUST_CALL( stable_sort,
                            thrust::make_zip_iterator(thrust::make_tuple(Aext_ii, Aext_j)),
                            thrust::make_zip_iterator(thrust::make_tuple(Aext_ii, Aext_j)) + Aext_nnz );
#endif
      }

      AT_offd = nalu_hypre_CSRMatrixCreate(nalu_hypre_ParCSRMatrixNumCols(A), num_cols_offd_AT, Aext_nnz);
      nalu_hypre_CSRMatrixJ(AT_offd) = Aext_j;
      nalu_hypre_CSRMatrixData(AT_offd) = Aext_data;
      nalu_hypre_CSRMatrixInitialize_v2(AT_offd, 0, NALU_HYPRE_MEMORY_DEVICE);
      hypreDevice_CsrRowIndicesToPtrs_v2(nalu_hypre_CSRMatrixNumRows(AT_offd), Aext_nnz, Aext_ii,
                                         nalu_hypre_CSRMatrixI(AT_offd));
      nalu_hypre_TFree(Aext_ii, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      nalu_hypre_CSRMatrixTransposeDevice(A_diag, &A_diagT, data);
      AT_offd = nalu_hypre_CSRMatrixCreate(nalu_hypre_ParCSRMatrixNumCols(A), 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(AT_offd, 0, NALU_HYPRE_MEMORY_DEVICE);
   }

   AT = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumCols(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixColStarts(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A),
                                 num_cols_offd_AT,
                                 nalu_hypre_CSRMatrixNumNonzeros(A_diagT),
                                 nalu_hypre_CSRMatrixNumNonzeros(AT_offd));

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(AT));
   nalu_hypre_ParCSRMatrixDiag(AT) = A_diagT;

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(AT));
   nalu_hypre_ParCSRMatrixOffd(AT) = AT_offd;

   if (num_cols_offd_AT)
   {
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(AT) = col_map_offd_AT;

      nalu_hypre_ParCSRMatrixColMapOffd(AT) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_AT, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(AT), col_map_offd_AT, NALU_HYPRE_BigInt, num_cols_offd_AT,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   }

   *AT_ptr = AT;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixAddDevice( NALU_HYPRE_Complex        alpha,
                             nalu_hypre_ParCSRMatrix  *A,
                             NALU_HYPRE_Complex        beta,
                             nalu_hypre_ParCSRMatrix  *B,
                             nalu_hypre_ParCSRMatrix **C_ptr )
{
   nalu_hypre_CSRMatrix *A_diag           = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *A_offd           = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_CSRMatrix *B_diag           = nalu_hypre_ParCSRMatrixDiag(B);
   nalu_hypre_CSRMatrix *B_offd           = nalu_hypre_ParCSRMatrixOffd(B);
   NALU_HYPRE_Int        num_cols_offd_A  = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Int        num_cols_offd_B  = nalu_hypre_CSRMatrixNumCols(B_offd);
   NALU_HYPRE_Int        num_cols_offd_C  = 0;
   NALU_HYPRE_BigInt    *d_col_map_offd_C = NULL;
   NALU_HYPRE_Int        num_procs;

   nalu_hypre_MPI_Comm_size(nalu_hypre_ParCSRMatrixComm(A), &num_procs);
   nalu_hypre_GpuProfilingPushRange("nalu_hypre_ParCSRMatrixAdd");

   nalu_hypre_CSRMatrix *C_diag = nalu_hypre_CSRMatrixAddDevice(alpha, A_diag, beta, B_diag);
   nalu_hypre_CSRMatrix *C_offd;

   //if (num_cols_offd_A || num_cols_offd_B)
   if (num_procs > 1)
   {
      nalu_hypre_ParCSRMatrixCopyColMapOffdToDevice(A);
      nalu_hypre_ParCSRMatrixCopyColMapOffdToDevice(B);

      NALU_HYPRE_BigInt *tmp = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_A + num_cols_offd_B,
                                       NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_TMemcpy(tmp,                   nalu_hypre_ParCSRMatrixDeviceColMapOffd(A), NALU_HYPRE_BigInt,
                    num_cols_offd_A, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(tmp + num_cols_offd_A, nalu_hypre_ParCSRMatrixDeviceColMapOffd(B), NALU_HYPRE_BigInt,
                    num_cols_offd_B, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::sort, tmp, tmp + num_cols_offd_A + num_cols_offd_B );
      NALU_HYPRE_BigInt *new_end = NALU_HYPRE_ONEDPL_CALL( std::unique, tmp,
                                                 tmp + num_cols_offd_A + num_cols_offd_B );
#else
      NALU_HYPRE_THRUST_CALL( sort, tmp, tmp + num_cols_offd_A + num_cols_offd_B );
      NALU_HYPRE_BigInt *new_end = NALU_HYPRE_THRUST_CALL( unique, tmp,
                                                 tmp + num_cols_offd_A + num_cols_offd_B );
#endif
      num_cols_offd_C = new_end - tmp;
      d_col_map_offd_C = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_C, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(d_col_map_offd_C, tmp, NALU_HYPRE_BigInt, num_cols_offd_C, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);

      /* reuse memory of tmp */
      NALU_HYPRE_Int *offd_A2C = (NALU_HYPRE_Int *) tmp;
      NALU_HYPRE_Int *offd_B2C = offd_A2C + num_cols_offd_A;
#if defined(NALU_HYPRE_USING_SYCL)
      /* WM: todo - getting an error when num_cols_offd_A is zero */
      if (num_cols_offd_A > 0)
      {
         NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                            d_col_map_offd_C,
                            d_col_map_offd_C + num_cols_offd_C,
                            nalu_hypre_ParCSRMatrixDeviceColMapOffd(A),
                            nalu_hypre_ParCSRMatrixDeviceColMapOffd(A) + num_cols_offd_A,
                            offd_A2C );
      }
      /* WM: todo - getting an error when num_cols_offd_B is zero */
      if (num_cols_offd_B > 0)
      {
         NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                            d_col_map_offd_C,
                            d_col_map_offd_C + num_cols_offd_C,
                            nalu_hypre_ParCSRMatrixDeviceColMapOffd(B),
                            nalu_hypre_ParCSRMatrixDeviceColMapOffd(B) + num_cols_offd_B,
                            offd_B2C );
      }
#else
      NALU_HYPRE_THRUST_CALL( lower_bound,
                         d_col_map_offd_C,
                         d_col_map_offd_C + num_cols_offd_C,
                         nalu_hypre_ParCSRMatrixDeviceColMapOffd(A),
                         nalu_hypre_ParCSRMatrixDeviceColMapOffd(A) + num_cols_offd_A,
                         offd_A2C );
      NALU_HYPRE_THRUST_CALL( lower_bound,
                         d_col_map_offd_C,
                         d_col_map_offd_C + num_cols_offd_C,
                         nalu_hypre_ParCSRMatrixDeviceColMapOffd(B),
                         nalu_hypre_ParCSRMatrixDeviceColMapOffd(B) + num_cols_offd_B,
                         offd_B2C );
#endif

      NALU_HYPRE_Int *C_offd_i, *C_offd_j, nnzC_offd;
      NALU_HYPRE_Complex *C_offd_a;

      hypreDevice_CSRSpAdd( nalu_hypre_CSRMatrixNumRows(A_offd),
                            nalu_hypre_CSRMatrixNumRows(B_offd),
                            num_cols_offd_C,
                            nalu_hypre_CSRMatrixNumNonzeros(A_offd),
                            nalu_hypre_CSRMatrixNumNonzeros(B_offd),
                            nalu_hypre_CSRMatrixI(A_offd),
                            nalu_hypre_CSRMatrixJ(A_offd),
                            alpha,
                            nalu_hypre_CSRMatrixData(A_offd),
                            offd_A2C,
                            nalu_hypre_CSRMatrixI(B_offd),
                            nalu_hypre_CSRMatrixJ(B_offd),
                            beta,
                            nalu_hypre_CSRMatrixData(B_offd),
                            offd_B2C,
                            NULL,
                            &nnzC_offd,
                            &C_offd_i,
                            &C_offd_j,
                            &C_offd_a );

      nalu_hypre_TFree(tmp, NALU_HYPRE_MEMORY_DEVICE);

      C_offd = nalu_hypre_CSRMatrixCreate(nalu_hypre_CSRMatrixNumRows(A_offd), num_cols_offd_C, nnzC_offd);
      nalu_hypre_CSRMatrixI(C_offd) = C_offd_i;
      nalu_hypre_CSRMatrixJ(C_offd) = C_offd_j;
      nalu_hypre_CSRMatrixData(C_offd) = C_offd_a;
      nalu_hypre_CSRMatrixMemoryLocation(C_offd) = NALU_HYPRE_MEMORY_DEVICE;
   }
   else
   {
      C_offd = nalu_hypre_CSRMatrixCreate(nalu_hypre_CSRMatrixNumRows(A_offd), 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* Create ParCSRMatrix C */
   nalu_hypre_ParCSRMatrix *C = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                                    nalu_hypre_ParCSRMatrixGlobalNumCols(A),
                                                    nalu_hypre_ParCSRMatrixRowStarts(A),
                                                    nalu_hypre_ParCSRMatrixColStarts(A),
                                                    num_cols_offd_C,
                                                    nalu_hypre_CSRMatrixNumNonzeros(C_diag),
                                                    nalu_hypre_CSRMatrixNumNonzeros(C_offd));

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(C));
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(C));
   nalu_hypre_ParCSRMatrixDiag(C) = C_diag;
   nalu_hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(C) = d_col_map_offd_C;

      nalu_hypre_ParCSRMatrixColMapOffd(C) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,
                                                     num_cols_offd_C,
                                                     NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(C), d_col_map_offd_C,
                    NALU_HYPRE_BigInt, num_cols_offd_C,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   }

   nalu_hypre_ParCSRMatrixSetNumNonzeros(C);
   nalu_hypre_ParCSRMatrixDNumNonzeros(C) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(C);

   /* create CommPkg of C */
   nalu_hypre_MatvecCommPkgCreate(C);

   *C_ptr = C;

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_GPU)

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixDiagScaleDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixDiagScaleDevice( nalu_hypre_ParCSRMatrix *par_A,
                                   nalu_hypre_ParVector    *par_ld,
                                   nalu_hypre_ParVector    *par_rd )
{
   /* Input variables */
   nalu_hypre_ParCSRCommPkg    *comm_pkg  = nalu_hypre_ParCSRMatrixCommPkg(par_A);
   nalu_hypre_ParCSRCommHandle *comm_handle;
   NALU_HYPRE_Int               num_sends;
   NALU_HYPRE_Int              *d_send_map_elmts;
   NALU_HYPRE_Int               send_map_num_elmts;

   nalu_hypre_CSRMatrix        *A_diag        = nalu_hypre_ParCSRMatrixDiag(par_A);
   nalu_hypre_CSRMatrix        *A_offd        = nalu_hypre_ParCSRMatrixOffd(par_A);
   NALU_HYPRE_Int               num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_Vector          *ld             = (par_ld) ? nalu_hypre_ParVectorLocalVector(par_ld) : NULL;
   nalu_hypre_Vector           *rd            = nalu_hypre_ParVectorLocalVector(par_rd);
   NALU_HYPRE_Complex          *rd_data       = nalu_hypre_VectorData(rd);

   /* Local variables */
   nalu_hypre_Vector           *rdbuf;
   NALU_HYPRE_Complex          *recv_rdbuf_data;
   NALU_HYPRE_Complex          *send_rdbuf_data;
   NALU_HYPRE_Int               sync_stream;

   /*---------------------------------------------------------------------
    * Setup communication info
    *--------------------------------------------------------------------*/

   nalu_hypre_GetSyncCudaCompute(&sync_stream);
   nalu_hypre_SetSyncCudaCompute(0);

   /* Create buffer vectors */
   rdbuf = nalu_hypre_SeqVectorCreate(num_cols_offd);

   /* If there exists no CommPkg for A, create it. */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(par_A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(par_A);
   }

   /* Communicate a single vector component */
   nalu_hypre_ParCSRCommPkgUpdateVecStarts(comm_pkg, par_rd);

   /* send_map_elmts on device */
   nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   /* Set variables */
   num_sends          = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   d_send_map_elmts   = nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
   send_map_num_elmts = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /*---------------------------------------------------------------------
    * Allocate/reuse receive data buffer
    *--------------------------------------------------------------------*/

   if (!nalu_hypre_ParCSRCommPkgTmpData(comm_pkg))
   {
      nalu_hypre_ParCSRCommPkgTmpData(comm_pkg) = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                                                          num_cols_offd,
                                                          NALU_HYPRE_MEMORY_DEVICE);
   }
   nalu_hypre_VectorData(rdbuf) = recv_rdbuf_data = nalu_hypre_ParCSRCommPkgTmpData(comm_pkg);
   nalu_hypre_SeqVectorSetDataOwner(rdbuf, 0);
   nalu_hypre_SeqVectorInitialize_v2(rdbuf, NALU_HYPRE_MEMORY_DEVICE);

   /*---------------------------------------------------------------------
    * Allocate/reuse send data buffer
    *--------------------------------------------------------------------*/

   if (!nalu_hypre_ParCSRCommPkgBufData(comm_pkg))
   {
      nalu_hypre_ParCSRCommPkgBufData(comm_pkg) = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                                                          send_map_num_elmts,
                                                          NALU_HYPRE_MEMORY_DEVICE);
   }
   send_rdbuf_data = nalu_hypre_ParCSRCommPkgBufData(comm_pkg);

   /*---------------------------------------------------------------------
    * Pack send data
    *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int  i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(send_rdbuf_data, rd_data, d_send_map_elmts)
   for (i = 0; i < send_map_num_elmts; i++)
   {
      send_rdbuf_data[i] = rd_data[d_send_map_elmts[i]];
   }
#else
#if defined(NALU_HYPRE_USING_SYCL)
   auto permuted_source = oneapi::dpl::make_permutation_iterator(rd_data,
                                                                 d_send_map_elmts);
   NALU_HYPRE_ONEDPL_CALL( std::copy,
                      permuted_source,
                      permuted_source + send_map_num_elmts,
                      send_rdbuf_data );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      d_send_map_elmts,
                      d_send_map_elmts + send_map_num_elmts,
                      rd_data,
                      send_rdbuf_data );
#endif
#endif


#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
   /* make sure send_rdbuf_data is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   /* A_diag = diag(ld) * A_diag * diag(rd) */
   nalu_hypre_CSRMatrixDiagScale(A_diag, ld, rd);

   /* Communication phase */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 NALU_HYPRE_MEMORY_DEVICE, send_rdbuf_data,
                                                 NALU_HYPRE_MEMORY_DEVICE, recv_rdbuf_data);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* A_offd = diag(ld) * A_offd * diag(rd) */
   nalu_hypre_CSRMatrixDiagScale(A_offd, ld, rdbuf);

#if defined(NALU_HYPRE_USING_GPU)
   /*---------------------------------------------------------------------
    * Synchronize calls
    *--------------------------------------------------------------------*/
   nalu_hypre_SetSyncCudaCompute(sync_stream);
   nalu_hypre_SyncComputeStream(nalu_hypre_handle());
#endif

   /* Free memory */
   nalu_hypre_SeqVectorDestroy(rdbuf);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRDiagScaleVectorDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRDiagScaleVectorDevice( nalu_hypre_ParCSRMatrix *par_A,
                                   nalu_hypre_ParVector    *par_y,
                                   nalu_hypre_ParVector    *par_x )
{
   /* Local Matrix and Vectors */
   nalu_hypre_CSRMatrix    *A_diag        = nalu_hypre_ParCSRMatrixDiag(par_A);
   nalu_hypre_Vector       *x             = nalu_hypre_ParVectorLocalVector(par_x);
   nalu_hypre_Vector       *y             = nalu_hypre_ParVectorLocalVector(par_y);

   /* Local vector x info */
   NALU_HYPRE_Complex      *x_data        = nalu_hypre_VectorData(x);
   NALU_HYPRE_Int           x_size        = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int           x_num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int           x_vecstride   = nalu_hypre_VectorVectorStride(x);

   /* Local vector y info */
   NALU_HYPRE_Complex      *y_data        = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int           y_size        = nalu_hypre_VectorSize(y);
   NALU_HYPRE_Int           y_num_vectors = nalu_hypre_VectorNumVectors(y);
   NALU_HYPRE_Int           y_vecstride   = nalu_hypre_VectorVectorStride(y);

   /* Local matrix A info */
   NALU_HYPRE_Int           num_rows      = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int          *A_i           = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Complex      *A_data        = nalu_hypre_CSRMatrixData(A_diag);

   /* Sanity checks */
   nalu_hypre_assert(x_vecstride == x_size);
   nalu_hypre_assert(y_vecstride == y_size);
   nalu_hypre_assert(x_num_vectors == y_num_vectors);

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(x_data,y_data,A_data,A_i)
   for (i = 0; i < num_rows; i++)
   {
      x_data[i] = y_data[i] / A_data[A_i[i]];
   }
#else
   hypreDevice_DiagScaleVector(x_num_vectors, num_rows, A_i, A_data, y_data, 0.0, x_data);
#endif // #if defined(NALU_HYPRE_USING_DEVICE_OPENMP)

   //nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
