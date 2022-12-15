/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvecOutOfPlaceHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMatvecOutOfPlaceHost( NALU_HYPRE_Complex       alpha,
                                        nalu_hypre_ParCSRMatrix *A,
                                        nalu_hypre_ParVector    *x,
                                        NALU_HYPRE_Complex       beta,
                                        nalu_hypre_ParVector    *b,
                                        nalu_hypre_ParVector    *y )
{
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);

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

   NALU_HYPRE_Int                i;
   NALU_HYPRE_Int                idxstride    = nalu_hypre_VectorIndexStride(x_local);
   NALU_HYPRE_Int                num_vectors  = nalu_hypre_VectorNumVectors(x_local);
   NALU_HYPRE_Complex           *x_local_data = nalu_hypre_VectorData(x_local);
   NALU_HYPRE_Complex           *x_tmp_data;
   NALU_HYPRE_Complex           *x_buf_data;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

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

   num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

   nalu_hypre_assert( num_cols_offd * num_vectors ==
                 nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) );
   nalu_hypre_assert( nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0) == 0 );
   nalu_hypre_assert( nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] -= nalu_hypre_MPI_Wtime();
#endif

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_ParCSRPersistentCommHandle *persistent_comm_handle =
      nalu_hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
#else
   nalu_hypre_ParCSRCommHandle *comm_handle;
#endif

   /*---------------------------------------------------------------------
    * Allocate (during nalu_hypre_SeqVectorInitialize_v2) or retrieve
    * persistent receive data buffer for x_tmp (if persistent is enabled).
    *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_VectorData(x_tmp) = (NALU_HYPRE_Complex *)
                             nalu_hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);
   nalu_hypre_SeqVectorSetDataOwner(x_tmp, 0);
#endif

   nalu_hypre_SeqVectorInitialize_v2(x_tmp, NALU_HYPRE_MEMORY_HOST);
   x_tmp_data = nalu_hypre_VectorData(x_tmp);

   /*---------------------------------------------------------------------
    * Allocate data send buffer
    *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   x_buf_data = (NALU_HYPRE_Complex *) nalu_hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);

#else
   x_buf_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                             nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                             NALU_HYPRE_MEMORY_HOST);
#endif

   /* The assert is because this code has been tested for column-wise vector storage only. */
   nalu_hypre_assert(idxstride == 1);

   /*---------------------------------------------------------------------
    * Pack send data
    *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
        i < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
        i++)
   {
      x_buf_data[i] = x_local_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK]   += nalu_hypre_MPI_Wtime();
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] -= nalu_hypre_MPI_Wtime();
#endif

   /* Non-blocking communication starts */
#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
   nalu_hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle,
                                         NALU_HYPRE_MEMORY_HOST, x_buf_data);
#else
   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 NALU_HYPRE_MEMORY_HOST, x_buf_data,
                                                 NALU_HYPRE_MEMORY_HOST, x_tmp_data);
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] += nalu_hypre_MPI_Wtime();
#endif

   /* overlapped local computation */
   nalu_hypre_CSRMatrixMatvecOutOfPlace(alpha, diag, x_local, beta, b_local, y_local, 0);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] -= nalu_hypre_MPI_Wtime();
#endif

   /* Non-blocking communication ends */
#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
   nalu_hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle, NALU_HYPRE_MEMORY_HOST, x_tmp_data);
#else
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] += nalu_hypre_MPI_Wtime();
#endif

   /* computation offd part */
   if (num_cols_offd)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);
   }

   /*---------------------------------------------------------------------
    * Free memory
    *--------------------------------------------------------------------*/
   nalu_hypre_SeqVectorDestroy(x_tmp);

#if !defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_TFree(x_buf_data, NALU_HYPRE_MEMORY_HOST);
#endif

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvecOutOfPlace
 *
 * Performs y <- alpha * A * x + beta * b
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMatvecOutOfPlace( NALU_HYPRE_Complex       alpha,
                                    nalu_hypre_ParCSRMatrix *A,
                                    nalu_hypre_ParVector    *x,
                                    NALU_HYPRE_Complex       beta,
                                    nalu_hypre_ParVector    *b,
                                    nalu_hypre_ParVector    *y )
{
   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParVectorMemoryLocation(x) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = nalu_hypre_ParCSRMatrixMatvecOutOfPlaceDevice(alpha, A, x, beta, b, y);
   }
   else
#endif
   {
      ierr = nalu_hypre_ParCSRMatrixMatvecOutOfPlaceHost(alpha, A, x, beta, b, y);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvec
 *
 * Performs y <- alpha * A * x + beta * y
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMatvec( NALU_HYPRE_Complex       alpha,
                          nalu_hypre_ParCSRMatrix *A,
                          nalu_hypre_ParVector    *x,
                          NALU_HYPRE_Complex       beta,
                          nalu_hypre_ParVector    *y )
{
   return nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, x, beta, y, y);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvecTHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMatvecTHost( NALU_HYPRE_Complex       alpha,
                               nalu_hypre_ParCSRMatrix *A,
                               nalu_hypre_ParVector    *x,
                               NALU_HYPRE_Complex       beta,
                               nalu_hypre_ParVector    *y )
{
   nalu_hypre_ParCSRCommPkg     *comm_pkg      = nalu_hypre_ParCSRMatrixCommPkg(A);

   nalu_hypre_CSRMatrix         *diag          = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix         *offd          = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_CSRMatrix         *diagT         = nalu_hypre_ParCSRMatrixDiagT(A);
   nalu_hypre_CSRMatrix         *offdT         = nalu_hypre_ParCSRMatrixOffdT(A);

   nalu_hypre_Vector            *x_local       = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector            *y_local       = nalu_hypre_ParVectorLocalVector(y);
   nalu_hypre_Vector            *y_tmp;

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
   NALU_HYPRE_Int                i;
   NALU_HYPRE_Int                ierr = 0;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

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

   num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

   nalu_hypre_assert( num_cols_offd * num_vectors ==
                 nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) );
   nalu_hypre_assert( nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0) == 0 );
   nalu_hypre_assert( nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] -= nalu_hypre_MPI_Wtime();
#endif

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_ParCSRPersistentCommHandle *persistent_comm_handle =
      nalu_hypre_ParCSRCommPkgGetPersistentCommHandle(2, comm_pkg);
#else
   nalu_hypre_ParCSRCommHandle *comm_handle;
#endif

   /*---------------------------------------------------------------------
    * Allocate (during nalu_hypre_SeqVectorInitialize_v2) or retrieve
    * persistent send data buffer for y_tmp (if persistent is enabled).
    *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_VectorData(y_tmp) = (NALU_HYPRE_Complex *)
                             nalu_hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);
   nalu_hypre_SeqVectorSetDataOwner(y_tmp, 0);
#endif

   nalu_hypre_SeqVectorInitialize_v2(y_tmp, NALU_HYPRE_MEMORY_HOST);
   y_tmp_data = nalu_hypre_VectorData(y_tmp);

   /*---------------------------------------------------------------------
    * Allocate receive data buffer
    *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   y_buf_data = (NALU_HYPRE_Complex *) nalu_hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);

#else
   y_buf_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                             nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                             NALU_HYPRE_MEMORY_HOST);
#endif

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

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] -= nalu_hypre_MPI_Wtime();
#endif

   /* Non-blocking communication starts */
#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle, NALU_HYPRE_MEMORY_HOST, y_tmp_data);

#else
   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(2, comm_pkg,
                                                 NALU_HYPRE_MEMORY_HOST, y_tmp_data,
                                                 NALU_HYPRE_MEMORY_HOST, y_buf_data );
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] += nalu_hypre_MPI_Wtime();
#endif

   /* Overlapped local computation.
      diagT is optional. Used only if it's present. */
   if (diagT)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, diagT, x_local, beta, y_local);
   }
   else
   {
      nalu_hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] -= nalu_hypre_MPI_Wtime();
#endif

   /* Non-blocking communication ends */
#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle,
                                        NALU_HYPRE_MEMORY_HOST, y_buf_data);
#else
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] += nalu_hypre_MPI_Wtime();
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK]   -= nalu_hypre_MPI_Wtime();
#endif

   /* The assert is here because this code has been tested for column-wise vector storage only. */
   nalu_hypre_assert(idxstride == 1);

   /* unpack recv data on host, TODO OMP? */
   for (i = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
        i < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
        i ++)
   {
      y_local_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] += y_buf_data[i];
   }

   /*---------------------------------------------------------------------
    * Free memory
    *--------------------------------------------------------------------*/
   nalu_hypre_SeqVectorDestroy(y_tmp);

#if !defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_TFree(y_buf_data, NALU_HYPRE_MEMORY_HOST);
#endif

   NALU_HYPRE_ANNOTATE_FUNC_END;

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] += nalu_hypre_MPI_Wtime();
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvecT
 *
 * Performs y <- alpha * A^T * x + beta * y
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMatvecT( NALU_HYPRE_Complex       alpha,
                           nalu_hypre_ParCSRMatrix *A,
                           nalu_hypre_ParVector    *x,
                           NALU_HYPRE_Complex       beta,
                           nalu_hypre_ParVector    *y )
{
   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParVectorMemoryLocation(x) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = nalu_hypre_ParCSRMatrixMatvecTDevice(alpha, A, x, beta, y);
   }
   else
#endif
   {
      ierr = nalu_hypre_ParCSRMatrixMatvecTHost(alpha, A, x, beta, y);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMatvec_FF( NALU_HYPRE_Complex       alpha,
                             nalu_hypre_ParCSRMatrix *A,
                             nalu_hypre_ParVector    *x,
                             NALU_HYPRE_Complex       beta,
                             nalu_hypre_ParVector    *y,
                             NALU_HYPRE_Int          *CF_marker,
                             NALU_HYPRE_Int           fpt )
{
   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_CSRMatrix        *diag   = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix        *offd   = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_Vector           *x_local  = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector           *y_local  = nalu_hypre_ParVectorLocalVector(y);
   NALU_HYPRE_BigInt            num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt            num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(A);

   nalu_hypre_Vector      *x_tmp;
   NALU_HYPRE_BigInt       x_size = nalu_hypre_ParVectorGlobalSize(x);
   NALU_HYPRE_BigInt       y_size = nalu_hypre_ParVectorGlobalSize(y);
   NALU_HYPRE_Int          num_cols_offd = nalu_hypre_CSRMatrixNumCols(offd);
   NALU_HYPRE_Int          ierr = 0;
   NALU_HYPRE_Int          num_sends, i, j, index, start, num_procs;
   NALU_HYPRE_Int         *int_buf_data = NULL;
   NALU_HYPRE_Int         *CF_marker_offd = NULL;

   NALU_HYPRE_Complex     *x_tmp_data = NULL;
   NALU_HYPRE_Complex     *x_buf_data = NULL;
   NALU_HYPRE_Complex     *x_local_data = nalu_hypre_VectorData(x_local);
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

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if (num_cols != x_size)
   {
      ierr = 11;
   }

   if (num_rows != y_size)
   {
      ierr = 12;
   }

   if (num_cols != x_size && num_rows != y_size)
   {
      ierr = 13;
   }

   if (num_procs > 1)
   {
      if (num_cols_offd)
      {
         x_tmp = nalu_hypre_SeqVectorCreate( num_cols_offd );
         nalu_hypre_SeqVectorInitialize(x_tmp);
         x_tmp_data = nalu_hypre_VectorData(x_tmp);
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

      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      if (num_sends)
         x_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  nalu_hypre_ParCSRCommPkgSendMapStart
                                    (comm_pkg,  num_sends), NALU_HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            x_buf_data[index++]
               = x_local_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
      comm_handle =
         nalu_hypre_ParCSRCommHandleCreate ( 1, comm_pkg, x_buf_data, x_tmp_data );
   }
   nalu_hypre_CSRMatrixMatvec_FF( alpha, diag, x_local, beta, y_local, CF_marker,
                             CF_marker, fpt);

   if (num_procs > 1)
   {
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      if (num_sends)
         int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_ParCSRCommPkgSendMapStart
                                      (comm_pkg,  num_sends), NALU_HYPRE_MEMORY_HOST);
      if (num_cols_offd) { CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
      comm_handle =
         nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd );

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      if (num_cols_offd) nalu_hypre_CSRMatrixMatvec_FF( alpha, offd, x_tmp, 1.0, y_local,
                                                      CF_marker, CF_marker_offd, fpt);

      nalu_hypre_SeqVectorDestroy(x_tmp);
      x_tmp = NULL;
      nalu_hypre_TFree(x_buf_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}
