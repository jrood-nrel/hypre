/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_mv.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvecOutOfPlaceDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMatvecOutOfPlaceDevice( NALU_HYPRE_Complex       alpha,
                                          nalu_hypre_ParCSRMatrix *A,
                                          nalu_hypre_ParVector    *x,
                                          NALU_HYPRE_Complex       beta,
                                          nalu_hypre_ParVector    *b,
                                          nalu_hypre_ParVector    *y )
{
   nalu_hypre_GpuProfilingPushRange("Matvec");

   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   NALU_HYPRE_Int               *d_send_map_elmts;
   NALU_HYPRE_Int                send_map_num_elmts;

   nalu_hypre_CSRMatrix         *diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix         *offd = nalu_hypre_ParCSRMatrixOffd(A);

   nalu_hypre_Vector            *x_local  = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector            *b_local  = nalu_hypre_ParVectorLocalVector(b);
   nalu_hypre_Vector            *y_local  = nalu_hypre_ParVectorLocalVector(y);
   nalu_hypre_Vector            *x_tmp;

   NALU_HYPRE_BigInt             num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt             num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   NALU_HYPRE_BigInt             x_size   = nalu_hypre_ParVectorGlobalSize(x);
   NALU_HYPRE_BigInt             b_size   = nalu_hypre_ParVectorGlobalSize(b);
   NALU_HYPRE_BigInt             y_size   = nalu_hypre_ParVectorGlobalSize(y);

   NALU_HYPRE_Int                num_cols_offd = nalu_hypre_CSRMatrixNumCols(offd);
   NALU_HYPRE_Int                num_recvs, num_sends;
   NALU_HYPRE_Int                ierr = 0;

   NALU_HYPRE_Int                idxstride    = nalu_hypre_VectorIndexStride(x_local);
   NALU_HYPRE_Int                num_vectors  = nalu_hypre_VectorNumVectors(x_local);
   NALU_HYPRE_Complex           *x_local_data = nalu_hypre_VectorData(x_local);
   NALU_HYPRE_Complex           *x_tmp_data;
   NALU_HYPRE_Complex           *x_buf_data;

   NALU_HYPRE_Int                sync_stream;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   nalu_hypre_GetSyncCudaCompute(&sync_stream);
   nalu_hypre_SetSyncCudaCompute(0);

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  ParMatvec returns ierr = 11 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 12 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 13 if both are true.
    *
    *  Because temporary vectors are often used in ParMatvec, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
   nalu_hypre_assert( idxstride > 0 );

   if (num_cols != x_size)
   {
      ierr = 11;
   }

   if (num_rows != y_size || num_rows != b_size)
   {
      ierr = 12;
   }

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
   {
      ierr = 13;
   }

   nalu_hypre_assert( nalu_hypre_VectorNumVectors(b_local) == num_vectors );
   nalu_hypre_assert( nalu_hypre_VectorNumVectors(y_local) == num_vectors );

   if (num_vectors == 1)
   {
      x_tmp = nalu_hypre_SeqVectorCreate(num_cols_offd);
   }
   else
   {
      nalu_hypre_assert(num_vectors > 1);
      x_tmp = nalu_hypre_SeqMultiVectorCreate(num_cols_offd, num_vectors);
      nalu_hypre_VectorMultiVecStorageMethod(x_tmp) = 1;
   }

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* Update send_map_starts, send_map_elmts, and recv_vec_starts when doing
      sparse matrix/multivector product  */
   nalu_hypre_ParCSRCommPkgUpdateVecStarts(comm_pkg, x);

   /* Copy send_map_elmts to the device if not already there */
   nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   /* Get information from the communication package*/
   num_recvs          = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   num_sends          = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   d_send_map_elmts   = nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
   send_map_num_elmts = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* Sanity checks */
   nalu_hypre_assert( num_cols_offd * num_vectors ==
                 nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) );
   nalu_hypre_assert( nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0) == 0 );
   nalu_hypre_assert( nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] -= nalu_hypre_MPI_Wtime();
#endif

   /*---------------------------------------------------------------------
    * Allocate or reuse receive data buffer for x_tmp
    *--------------------------------------------------------------------*/

   if (!nalu_hypre_ParCSRCommPkgTmpData(comm_pkg))
   {
      nalu_hypre_ParCSRCommPkgTmpData(comm_pkg) = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                                                          num_cols_offd * num_vectors,
                                                          NALU_HYPRE_MEMORY_DEVICE);
   }
   nalu_hypre_VectorData(x_tmp) = x_tmp_data = nalu_hypre_ParCSRCommPkgTmpData(comm_pkg);
   nalu_hypre_SeqVectorSetDataOwner(x_tmp, 0);
   nalu_hypre_SeqVectorInitialize_v2(x_tmp, NALU_HYPRE_MEMORY_DEVICE);

   /*---------------------------------------------------------------------
    * Allocate or reuse send data buffer
    *--------------------------------------------------------------------*/

   if (!nalu_hypre_ParCSRCommPkgBufData(comm_pkg))
   {
      nalu_hypre_ParCSRCommPkgBufData(comm_pkg) = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                                                          send_map_num_elmts,
                                                          NALU_HYPRE_MEMORY_DEVICE);
   }
   x_buf_data = nalu_hypre_ParCSRCommPkgBufData(comm_pkg);

   /* The assert is because this code has been tested for column-wise vector storage only. */
   nalu_hypre_assert(idxstride == 1);

   //nalu_hypre_SeqVectorPrefetch(x_local, NALU_HYPRE_MEMORY_DEVICE);

   /*---------------------------------------------------------------------
    * Pack send data
    *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int  i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(x_buf_data, x_local_data, d_send_map_elmts)
   for (i = 0; i < send_map_num_elmts; i++)
   {
      x_buf_data[i] = x_local_data[d_send_map_elmts[i]];
   }
#else
#if defined(NALU_HYPRE_USING_SYCL)
   auto permuted_source = oneapi::dpl::make_permutation_iterator(x_local_data,
                                                                 d_send_map_elmts);
   NALU_HYPRE_ONEDPL_CALL( std::copy,
                      permuted_source,
                      permuted_source + send_map_num_elmts,
                      x_buf_data );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      d_send_map_elmts,
                      d_send_map_elmts + send_map_num_elmts,
                      x_local_data,
                      x_buf_data );
#endif
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] += nalu_hypre_MPI_Wtime();
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure x_buf_data is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   /* when using GPUs, start local matvec first in order to overlap with communication */
   nalu_hypre_CSRMatrixMatvecOutOfPlace(alpha, diag, x_local, beta, b_local, y_local, 0);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] -= nalu_hypre_MPI_Wtime();
#endif

   /* Non-blocking communication starts */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 NALU_HYPRE_MEMORY_DEVICE, x_buf_data,
                                                 NALU_HYPRE_MEMORY_DEVICE, x_tmp_data);

   /* Non-blocking communication ends */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] += nalu_hypre_MPI_Wtime();
#endif

   /* computation offd part */
   if (num_cols_offd)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] -= nalu_hypre_MPI_Wtime();
#endif

   /*---------------------------------------------------------------------
    * Free memory
    *--------------------------------------------------------------------*/
   nalu_hypre_SeqVectorDestroy(x_tmp);

   /*---------------------------------------------------------------------
    * Synchronize calls
    *--------------------------------------------------------------------*/
   nalu_hypre_SetSyncCudaCompute(sync_stream);
   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   /*---------------------------------------------------------------------
    * Performance profiling
    *--------------------------------------------------------------------*/
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] += nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_ANNOTATE_FUNC_END;

   nalu_hypre_GpuProfilingPopRange();

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvecTDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMatvecTDevice( NALU_HYPRE_Complex       alpha,
                                 nalu_hypre_ParCSRMatrix *A,
                                 nalu_hypre_ParVector    *x,
                                 NALU_HYPRE_Complex       beta,
                                 nalu_hypre_ParVector    *y )
{
   nalu_hypre_GpuProfilingPushRange("MatvecT");

   nalu_hypre_ParCSRCommPkg     *comm_pkg      = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   NALU_HYPRE_Int                send_map_num_elmts;

   nalu_hypre_CSRMatrix         *diag          = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix         *offd          = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_CSRMatrix         *diagT         = nalu_hypre_ParCSRMatrixDiagT(A);
   nalu_hypre_CSRMatrix         *offdT         = nalu_hypre_ParCSRMatrixOffdT(A);

   nalu_hypre_Vector            *x_local       = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector            *y_local       = nalu_hypre_ParVectorLocalVector(y);
   nalu_hypre_Vector            *y_tmp;

   NALU_HYPRE_Int                num_cols_diag = nalu_hypre_CSRMatrixNumCols(diag);
   NALU_HYPRE_Int                num_cols_offd = nalu_hypre_CSRMatrixNumCols(offd);
   NALU_HYPRE_BigInt             num_rows      = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt             num_cols      = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   NALU_HYPRE_BigInt             x_size        = nalu_hypre_ParVectorGlobalSize(x);
   NALU_HYPRE_BigInt             y_size        = nalu_hypre_ParVectorGlobalSize(y);

   NALU_HYPRE_Complex           *y_tmp_data;
   NALU_HYPRE_Complex           *y_buf_data;
   NALU_HYPRE_Complex           *y_local_data  = nalu_hypre_VectorData(y_local);
   NALU_HYPRE_Int                idxstride     = nalu_hypre_VectorIndexStride(y_local);
   NALU_HYPRE_Int                num_vectors   = nalu_hypre_VectorNumVectors(y_local);
   NALU_HYPRE_Int                num_sends;
   NALU_HYPRE_Int                num_recvs;
   NALU_HYPRE_Int                ierr = 0;
   NALU_HYPRE_Int                sync_stream;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   nalu_hypre_GetSyncCudaCompute(&sync_stream);
   nalu_hypre_SetSyncCudaCompute(0);

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
   if (num_rows != x_size)
   {
      ierr = 1;
   }

   if (num_cols != y_size)
   {
      ierr = 2;
   }

   if (num_rows != x_size && num_cols != y_size)
   {
      ierr = 3;
   }

   nalu_hypre_assert( nalu_hypre_VectorNumVectors(x_local) == num_vectors );
   nalu_hypre_assert( nalu_hypre_VectorNumVectors(y_local) == num_vectors );

   if (num_vectors == 1)
   {
      y_tmp = nalu_hypre_SeqVectorCreate(num_cols_offd);
   }
   else
   {
      nalu_hypre_assert(num_vectors > 1);
      y_tmp = nalu_hypre_SeqMultiVectorCreate(num_cols_offd, num_vectors);
      nalu_hypre_VectorMultiVecStorageMethod(y_tmp) = 1;
   }

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* Update send_map_starts, send_map_elmts, and recv_vec_starts for SpMV with multivecs */
   nalu_hypre_ParCSRCommPkgUpdateVecStarts(comm_pkg, y);

   /* Update send_map_elmts on device */
   nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   /* Get information from the communication package*/
   num_recvs          = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   num_sends          = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_map_num_elmts = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* Sanity checks */
   nalu_hypre_assert( num_cols_offd * num_vectors ==
                 nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) );
   nalu_hypre_assert( nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0) == 0 );
   nalu_hypre_assert( nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] -= nalu_hypre_MPI_Wtime();
#endif

   /*---------------------------------------------------------------------
    * Allocate or reuse send data buffer for y_tmp
    *--------------------------------------------------------------------*/

   if (!nalu_hypre_ParCSRCommPkgTmpData(comm_pkg))
   {
      nalu_hypre_ParCSRCommPkgTmpData(comm_pkg) = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                                                          num_cols_offd * num_vectors,
                                                          NALU_HYPRE_MEMORY_DEVICE);
   }
   nalu_hypre_VectorData(y_tmp) = y_tmp_data = nalu_hypre_ParCSRCommPkgTmpData(comm_pkg);
   nalu_hypre_SeqVectorSetDataOwner(y_tmp, 0);
   nalu_hypre_SeqVectorInitialize_v2(y_tmp, NALU_HYPRE_MEMORY_DEVICE);

   /*---------------------------------------------------------------------
    * Allocate receive data buffer
    *--------------------------------------------------------------------*/

   if (!nalu_hypre_ParCSRCommPkgBufData(comm_pkg))
   {
      nalu_hypre_ParCSRCommPkgBufData(comm_pkg) = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                                                          send_map_num_elmts,
                                                          NALU_HYPRE_MEMORY_DEVICE);
   }
   y_buf_data = nalu_hypre_ParCSRCommPkgBufData(comm_pkg);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] += nalu_hypre_MPI_Wtime();
#endif

   /* Compute y_tmp = offd^T * x_local */
   if (num_cols_offd)
   {
      if (offdT)
      {
         // offdT is optional. Used only if it's present
         nalu_hypre_CSRMatrixMatvec(alpha, offdT, x_local, 0.0, y_tmp);
      }
      else
      {
         nalu_hypre_CSRMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);
      }
   }

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI)
   /* RL: make sure y_tmp is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   /* when using GPUs, start local matvec first in order to overlap with communication */
   if (diagT)
   {
      // diagT is optional. Used only if it's present.
      nalu_hypre_CSRMatrixMatvec(alpha, diagT, x_local, beta, y_local);
   }
   else
   {
      nalu_hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] -= nalu_hypre_MPI_Wtime();
#endif

   /* Non-blocking communication starts */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(2, comm_pkg,
                                                 NALU_HYPRE_MEMORY_DEVICE, y_tmp_data,
                                                 NALU_HYPRE_MEMORY_DEVICE, y_buf_data );

   /* Non-blocking communication ends */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] += nalu_hypre_MPI_Wtime();
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK]   -= nalu_hypre_MPI_Wtime();
#endif

   /* The assert is here because this code has been tested for column-wise vector storage only. */
   nalu_hypre_assert( idxstride == 1 );

   /*---------------------------------------------------------------------
    * Unpack receive data
    *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int  *d_send_map_elmts = nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
   NALU_HYPRE_Int   i, j;

   for (i = 0; i < num_sends; i++)
   {
      NALU_HYPRE_Int  start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      NALU_HYPRE_Int  end   = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);

      #pragma omp target teams distribute parallel for private(j) is_device_ptr(y_buf_data, y_local_data, d_send_map_elmts)
      for (j = start; j < end; j++)
      {
         y_local_data[d_send_map_elmts[j]] += y_buf_data[j];
      }
   }
#else
   /* Use SpMV to unpack data */
   nalu_hypre_ParCSRMatrixMatvecT_unpack(comm_pkg, num_cols_diag, y_buf_data, y_local_data);
#endif

   /*---------------------------------------------------------------------
    * Free memory
    *--------------------------------------------------------------------*/

   nalu_hypre_SeqVectorDestroy(y_tmp);

   /*---------------------------------------------------------------------
    * Synchronize when using GPUs
    *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SetSyncCudaCompute(sync_stream);
   nalu_hypre_SyncComputeStream(nalu_hypre_handle());
#endif

   /*---------------------------------------------------------------------
    * Performance profiling
    *--------------------------------------------------------------------*/

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] += nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_ANNOTATE_FUNC_END;

   nalu_hypre_GpuProfilingPopRange();

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvecT_unpack
 *
 * Computes on the device:
 *
 *   local_data[send_map_elmts] += recv_data
 *
 * with hypre's internal SpMV.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMatvecT_unpack( nalu_hypre_ParCSRCommPkg *comm_pkg,
                                  NALU_HYPRE_Int            num_cols,
                                  NALU_HYPRE_Complex       *recv_data,
                                  NALU_HYPRE_Complex       *local_data )
{
   /* Input variables */
   nalu_hypre_CSRMatrix  *matrix_E       = nalu_hypre_ParCSRCommPkgMatrixE(comm_pkg);
   NALU_HYPRE_Int         num_sends      = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int         num_elements   = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   NALU_HYPRE_Int         num_components = nalu_hypre_ParCSRCommPkgNumComponents(comm_pkg);

   /* Local variables */
   nalu_hypre_Vector      vec_x;
   nalu_hypre_Vector      vec_y;
   NALU_HYPRE_Int         trans = 0;
   NALU_HYPRE_Int         fill  = 0;
   NALU_HYPRE_Complex     alpha = 1.0;
   NALU_HYPRE_Complex     beta  = 1.0;

   if (num_elements == 0)
   {
      return nalu_hypre_error_flag;
   }

   /* Create matrix E if it not exists */
   if (!matrix_E)
   {
      nalu_hypre_ParCSRCommPkgCreateMatrixE(comm_pkg, num_cols);
      matrix_E = nalu_hypre_ParCSRCommPkgMatrixE(comm_pkg);
   }

   /* Set vector x */
   nalu_hypre_VectorData(&vec_x)                  = recv_data;
   nalu_hypre_VectorOwnsData(&vec_x)              = 0;
   nalu_hypre_VectorSize(&vec_x)                  = num_elements / num_components;
   nalu_hypre_VectorVectorStride(&vec_x)          = 1;
   nalu_hypre_VectorIndexStride(&vec_x)           = num_components;
   nalu_hypre_VectorNumVectors(&vec_x)            = num_components;
   nalu_hypre_VectorMultiVecStorageMethod(&vec_x) = 1;

   /* Set vector y */
   nalu_hypre_VectorData(&vec_y)                  = local_data;
   nalu_hypre_VectorOwnsData(&vec_y)              = 0;
   nalu_hypre_VectorSize(&vec_y)                  = num_cols;
   nalu_hypre_VectorVectorStride(&vec_y)          = num_cols;
   nalu_hypre_VectorIndexStride(&vec_y)           = 1;
   nalu_hypre_VectorNumVectors(&vec_y)            = num_components;
   nalu_hypre_VectorMultiVecStorageMethod(&vec_y) = 0;

   /* WM: todo - port nalu_hypre_CSRMatrixSpMVDevice() to sycl */
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Complex *data = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                                      nalu_hypre_CSRMatrixNumNonzeros(matrix_E),
                                      NALU_HYPRE_MEMORY_DEVICE);
   hypreDevice_ComplexFilln(data, nalu_hypre_CSRMatrixNumNonzeros(matrix_E), 1.0);
   nalu_hypre_CSRMatrixData(matrix_E) = data;

   nalu_hypre_CSRMatrixMatvecDevice(trans, alpha, matrix_E, &vec_x, beta, &vec_y, &vec_y, 0);
#else
   /* Compute y += E*x */
   nalu_hypre_CSRMatrixSpMVDevice(trans, alpha, matrix_E, &vec_x, beta, &vec_y, fill);
#endif

   return nalu_hypre_error_flag;
}

#endif /* #if defined(NALU_HYPRE_USING_GPU) */
