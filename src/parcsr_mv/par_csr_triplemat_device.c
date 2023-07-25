/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_mv.h"
#include "_nalu_hypre_utilities.hpp"

#define PARCSRGEMM_TIMING 0

#if defined(NALU_HYPRE_USING_GPU)

/* option == 1, T = NALU_HYPRE_BigInt
 * option == 2, T = NALU_HYPRE_Int,
 */
template<NALU_HYPRE_Int option, typename T>
#if defined(NALU_HYPRE_USING_SYCL)
struct RAP_functor
#else
struct RAP_functor : public thrust::unary_function<NALU_HYPRE_Int, T>
#endif
{
   NALU_HYPRE_Int num_col;
   T         first_col;
   T        *col_map;

   RAP_functor(NALU_HYPRE_Int num_col_, T first_col_, T *col_map_)
   {
      num_col   = num_col_;
      first_col = first_col_;
      col_map   = col_map_;
   }

   __host__ __device__
   T operator()(const NALU_HYPRE_Int x) const
   {
      if (x < num_col)
      {
         if (option == 1)
         {
            return x + first_col;
         }
         else
         {
            return x;
         }
      }

      if (option == 1)
      {
         return col_map[x - num_col];
      }
      else
      {
         return col_map[x - num_col] + num_col;
      }
   }
};

/* C = A * B */
nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatMatDevice( nalu_hypre_ParCSRMatrix  *A,
                          nalu_hypre_ParCSRMatrix  *B )
{
   nalu_hypre_ParCSRMatrix *C;
   nalu_hypre_CSRMatrix    *C_diag;
   nalu_hypre_CSRMatrix    *C_offd;
   NALU_HYPRE_Int           num_cols_offd_C = 0;
   NALU_HYPRE_BigInt       *col_map_offd_C = NULL;

   NALU_HYPRE_Int num_procs;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if ( nalu_hypre_ParCSRMatrixGlobalNumCols(A) != nalu_hypre_ParCSRMatrixGlobalNumRows(B) ||
        nalu_hypre_ParCSRMatrixNumCols(A)       != nalu_hypre_ParCSRMatrixNumRows(B) )
   {
      nalu_hypre_error_in_arg(1);
      nalu_hypre_printf(" Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

#if PARCSRGEMM_TIMING > 0
   NALU_HYPRE_Real ta, tb;
   ta = nalu_hypre_MPI_Wtime();
#endif

#if PARCSRGEMM_TIMING > 1
   NALU_HYPRE_Real t1, t2;
#endif

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/
   if (num_procs > 1)
   {
      void *request;
      nalu_hypre_CSRMatrix *Abar, *Bbar, *Cbar, *Bext;
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * nalu_hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      /* contains communication which should be explicitly included to allow for overlap */
      nalu_hypre_ParCSRMatrixExtractBExtDeviceInit(B, A, 1, &request);
#if PARCSRGEMM_TIMING > 1
      t2 = nalu_hypre_MPI_Wtime();
#endif
      Abar = nalu_hypre_ConcatDiagAndOffdDevice(A);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t2;
      nalu_hypre_ParPrintf(comm, "Time Concat %f\n", t2);
#endif
      Bext = nalu_hypre_ParCSRMatrixExtractBExtDeviceWait(request);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1 - t2;
      nalu_hypre_ParPrintf(comm, "Time Bext %f\n", t2);
      nalu_hypre_ParPrintf(comm, "Size Bext %d %d %d\n", nalu_hypre_CSRMatrixNumRows(Bext),
                      nalu_hypre_CSRMatrixNumCols(Bext), nalu_hypre_CSRMatrixNumNonzeros(Bext));
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      nalu_hypre_ConcatDiagOffdAndExtDevice(B, Bext, &Bbar, &num_cols_offd_C, &col_map_offd_C);
      nalu_hypre_CSRMatrixDestroy(Bext);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time Concat %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      Cbar = nalu_hypre_CSRMatrixMultiplyDevice(Abar, Bbar);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time SpGemm %f\n", t2);
#endif

      nalu_hypre_CSRMatrixDestroy(Abar);
      nalu_hypre_CSRMatrixDestroy(Bbar);

      nalu_hypre_assert(nalu_hypre_CSRMatrixNumRows(Cbar) == nalu_hypre_ParCSRMatrixNumRows(A));
      nalu_hypre_assert(nalu_hypre_CSRMatrixNumCols(Cbar) == nalu_hypre_ParCSRMatrixNumCols(B) + num_cols_offd_C);

      // split into diag and offd
#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      in_range<NALU_HYPRE_Int> pred(0, nalu_hypre_ParCSRMatrixNumCols(B) - 1);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_Int nnz_C_diag = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                                nalu_hypre_CSRMatrixJ(Cbar),
                                                nalu_hypre_CSRMatrixJ(Cbar) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                                                pred );
#else
      NALU_HYPRE_Int nnz_C_diag = NALU_HYPRE_THRUST_CALL( count_if,
                                                nalu_hypre_CSRMatrixJ(Cbar),
                                                nalu_hypre_CSRMatrixJ(Cbar) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                                                pred );
#endif
      NALU_HYPRE_Int nnz_C_offd = nalu_hypre_CSRMatrixNumNonzeros(Cbar) - nnz_C_diag;

      C_diag = nalu_hypre_CSRMatrixCreate(nalu_hypre_ParCSRMatrixNumRows(A), nalu_hypre_ParCSRMatrixNumCols(B),
                                     nnz_C_diag);
      nalu_hypre_CSRMatrixInitialize_v2(C_diag, 0, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int     *C_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_C_diag, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int     *C_diag_j = nalu_hypre_CSRMatrixJ(C_diag);
      NALU_HYPRE_Complex *C_diag_a = nalu_hypre_CSRMatrixData(C_diag);

      NALU_HYPRE_Int *Cbar_ii = hypreDevice_CsrRowPtrsToIndices(nalu_hypre_ParCSRMatrixNumRows(A),
                                                           nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                                                           nalu_hypre_CSRMatrixI(Cbar));

#if defined(NALU_HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(Cbar_ii, nalu_hypre_CSRMatrixJ(Cbar),
                                                                       nalu_hypre_CSRMatrixData(Cbar)),
                                        oneapi::dpl::make_zip_iterator(Cbar_ii, nalu_hypre_CSRMatrixJ(Cbar),
                                                                       nalu_hypre_CSRMatrixData(Cbar)) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                                        nalu_hypre_CSRMatrixJ(Cbar),
                                        oneapi::dpl::make_zip_iterator(C_diag_ii, C_diag_j, C_diag_a),
                                        pred );
      nalu_hypre_assert( std::get<0>(new_end.base()) == C_diag_ii + nnz_C_diag );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(Cbar_ii, nalu_hypre_CSRMatrixJ(Cbar),
                                                                     nalu_hypre_CSRMatrixData(Cbar))),
                        thrust::make_zip_iterator(thrust::make_tuple(Cbar_ii, nalu_hypre_CSRMatrixJ(Cbar),
                                                                     nalu_hypre_CSRMatrixData(Cbar))) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                        nalu_hypre_CSRMatrixJ(Cbar),
                        thrust::make_zip_iterator(thrust::make_tuple(C_diag_ii, C_diag_j, C_diag_a)),
                        pred );
      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == C_diag_ii + nnz_C_diag );
#endif
      hypreDevice_CsrRowIndicesToPtrs_v2(nalu_hypre_CSRMatrixNumRows(C_diag), nnz_C_diag, C_diag_ii,
                                         nalu_hypre_CSRMatrixI(C_diag));
      nalu_hypre_TFree(C_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

      C_offd = nalu_hypre_CSRMatrixCreate(nalu_hypre_ParCSRMatrixNumRows(A), num_cols_offd_C, nnz_C_offd);
      nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int     *C_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_C_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int     *C_offd_j = nalu_hypre_CSRMatrixJ(C_offd);
      NALU_HYPRE_Complex *C_offd_a = nalu_hypre_CSRMatrixData(C_offd);
#if defined(NALU_HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(Cbar_ii, nalu_hypre_CSRMatrixJ(Cbar),
                                                                  nalu_hypre_CSRMatrixData(Cbar)),
                                   oneapi::dpl::make_zip_iterator(Cbar_ii, nalu_hypre_CSRMatrixJ(Cbar),
                                                                  nalu_hypre_CSRMatrixData(Cbar)) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                                   nalu_hypre_CSRMatrixJ(Cbar),
                                   oneapi::dpl::make_zip_iterator(C_offd_ii, C_offd_j, C_offd_a),
                                   std::not_fn(pred) );
      nalu_hypre_assert( std::get<0>(new_end.base()) == C_offd_ii + nnz_C_offd );
#else
      new_end = NALU_HYPRE_THRUST_CALL(
                   copy_if,
                   thrust::make_zip_iterator(thrust::make_tuple(Cbar_ii, nalu_hypre_CSRMatrixJ(Cbar),
                                                                nalu_hypre_CSRMatrixData(Cbar))),
                   thrust::make_zip_iterator(thrust::make_tuple(Cbar_ii, nalu_hypre_CSRMatrixJ(Cbar),
                                                                nalu_hypre_CSRMatrixData(Cbar))) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                   nalu_hypre_CSRMatrixJ(Cbar),
                   thrust::make_zip_iterator(thrust::make_tuple(C_offd_ii, C_offd_j, C_offd_a)),
                   thrust::not1(pred) );
      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == C_offd_ii + nnz_C_offd );
#endif

      hypreDevice_CsrRowIndicesToPtrs_v2(nalu_hypre_CSRMatrixNumRows(C_offd), nnz_C_offd, C_offd_ii,
                                         nalu_hypre_CSRMatrixI(C_offd));
      nalu_hypre_TFree(C_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         C_offd_j,
                         C_offd_j + nnz_C_offd,
                         C_offd_j,
      [const_val = nalu_hypre_ParCSRMatrixNumCols(B)] (const auto & x) {return x - const_val;} );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         C_offd_j,
                         C_offd_j + nnz_C_offd,
                         thrust::make_constant_iterator(nalu_hypre_ParCSRMatrixNumCols(B)),
                         C_offd_j,
                         thrust::minus<NALU_HYPRE_Int>() );
#endif

      nalu_hypre_TFree(Cbar_ii, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixDestroy(Cbar);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time Split %f\n", t2);
#endif
   }
   else
   {
#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      C_diag = nalu_hypre_CSRMatrixMultiplyDevice(nalu_hypre_ParCSRMatrixDiag(A), nalu_hypre_ParCSRMatrixDiag(B));
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time SpGemm %f\n", t2);
#endif
      C_offd = nalu_hypre_CSRMatrixCreate(nalu_hypre_ParCSRMatrixNumRows(A), 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, NALU_HYPRE_MEMORY_DEVICE);
   }

   C = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(B),
                                nalu_hypre_ParCSRMatrixRowStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(B),
                                num_cols_offd_C,
                                nalu_hypre_CSRMatrixNumNonzeros(C_diag),
                                nalu_hypre_CSRMatrixNumNonzeros(C_offd));

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(C));
   nalu_hypre_ParCSRMatrixDiag(C) = C_diag;

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(C));
   nalu_hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;
   }

   nalu_hypre_ParCSRMatrixCopyColMapOffdToHost(C);

#if PARCSRGEMM_TIMING > 0
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   tb = nalu_hypre_MPI_Wtime() - ta;
   nalu_hypre_ParPrintf(comm, "Time nalu_hypre_ParCSRMatMatDevice %f\n", tb);
#endif

   return C;
}

/* C = A^T * B */
nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRTMatMatKTDevice( nalu_hypre_ParCSRMatrix  *A,
                             nalu_hypre_ParCSRMatrix  *B,
                             NALU_HYPRE_Int            keep_transpose)
{
   nalu_hypre_CSRMatrix *A_diag  = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *A_offd  = nalu_hypre_ParCSRMatrixOffd(A);

   nalu_hypre_ParCSRMatrix *C;
   nalu_hypre_CSRMatrix    *C_diag;
   nalu_hypre_CSRMatrix    *C_offd;
   NALU_HYPRE_Int           num_cols_offd_C = 0;
   NALU_HYPRE_BigInt       *col_map_offd_C = NULL;

   NALU_HYPRE_Int num_procs;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if (nalu_hypre_ParCSRMatrixGlobalNumRows(A) != nalu_hypre_ParCSRMatrixGlobalNumRows(B) ||
       nalu_hypre_ParCSRMatrixNumRows(A)       != nalu_hypre_ParCSRMatrixNumRows(B))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

#if PARCSRGEMM_TIMING > 0
   NALU_HYPRE_Real ta, tb;
   ta = nalu_hypre_MPI_Wtime();
#endif

#if PARCSRGEMM_TIMING > 1
   NALU_HYPRE_Real t1, t2;
#endif

   if (num_procs > 1)
   {
      void *request;
      nalu_hypre_CSRMatrix *Bbar, *AbarT, *Cbar, *AT_diag, *AT_offd, *Cint, *Cext;
      nalu_hypre_CSRMatrix *B_offd = nalu_hypre_ParCSRMatrixOffd(B);
      NALU_HYPRE_Int local_nnz_Cbar;

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      Bbar = nalu_hypre_ConcatDiagAndOffdDevice(B);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time Concat %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif

      if (nalu_hypre_ParCSRMatrixDiagT(A))
      {
         AT_diag = nalu_hypre_ParCSRMatrixDiagT(A);
      }
      else
      {
         nalu_hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);
      }

      if (nalu_hypre_ParCSRMatrixOffdT(A))
      {
         AT_offd = nalu_hypre_ParCSRMatrixOffdT(A);
      }
      else
      {
         nalu_hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);
      }

#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time Transpose %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      AbarT = nalu_hypre_CSRMatrixStack2Device(AT_diag, AT_offd);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time Stack %f\n", t2);
#endif

      if (!nalu_hypre_ParCSRMatrixDiagT(A))
      {
         if (keep_transpose)
         {
            nalu_hypre_ParCSRMatrixDiagT(A) = AT_diag;
         }
         else
         {
            nalu_hypre_CSRMatrixDestroy(AT_diag);
         }
      }

      if (!nalu_hypre_ParCSRMatrixOffdT(A))
      {
         if (keep_transpose)
         {
            nalu_hypre_ParCSRMatrixOffdT(A) = AT_offd;
         }
         else
         {
            nalu_hypre_CSRMatrixDestroy(AT_offd);
         }
      }

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      Cbar = nalu_hypre_CSRMatrixMultiplyDevice(AbarT, Bbar);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time SpGemm %f\n", t2);
#endif

      nalu_hypre_CSRMatrixDestroy(AbarT);
      nalu_hypre_CSRMatrixDestroy(Bbar);

      nalu_hypre_assert(nalu_hypre_CSRMatrixNumRows(Cbar) == nalu_hypre_ParCSRMatrixNumCols(A) + nalu_hypre_CSRMatrixNumCols(
                      A_offd));
      nalu_hypre_assert(nalu_hypre_CSRMatrixNumCols(Cbar) == nalu_hypre_ParCSRMatrixNumCols(B) + nalu_hypre_CSRMatrixNumCols(
                      B_offd));

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      nalu_hypre_TMemcpy(&local_nnz_Cbar, nalu_hypre_CSRMatrixI(Cbar) + nalu_hypre_ParCSRMatrixNumCols(A), NALU_HYPRE_Int, 1,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

      // Cint is the bottom part of Cbar
      Cint = nalu_hypre_CSRMatrixCreate(nalu_hypre_CSRMatrixNumCols(A_offd), nalu_hypre_CSRMatrixNumCols(Cbar),
                                   nalu_hypre_CSRMatrixNumNonzeros(Cbar) - local_nnz_Cbar);
      nalu_hypre_CSRMatrixMemoryLocation(Cint) = NALU_HYPRE_MEMORY_DEVICE;
      nalu_hypre_CSRMatrixOwnsData(Cint) = 0;

      nalu_hypre_CSRMatrixI(Cint) = nalu_hypre_CSRMatrixI(Cbar) + nalu_hypre_ParCSRMatrixNumCols(A);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         nalu_hypre_CSRMatrixI(Cint),
                         nalu_hypre_CSRMatrixI(Cint) + nalu_hypre_CSRMatrixNumRows(Cint) + 1,
                         nalu_hypre_CSRMatrixI(Cint),
      [const_val = local_nnz_Cbar] (const auto & x) {return x - const_val;} );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         nalu_hypre_CSRMatrixI(Cint),
                         nalu_hypre_CSRMatrixI(Cint) + nalu_hypre_CSRMatrixNumRows(Cint) + 1,
                         thrust::make_constant_iterator(local_nnz_Cbar),
                         nalu_hypre_CSRMatrixI(Cint),
                         thrust::minus<NALU_HYPRE_Int>() );
#endif

      // Change Cint into a BigJ matrix
      // RL: TODO FIX the 'big' num of columns to global size
      nalu_hypre_CSRMatrixBigJ(Cint) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nalu_hypre_CSRMatrixNumNonzeros(Cint),
                                               NALU_HYPRE_MEMORY_DEVICE);

      RAP_functor<1, NALU_HYPRE_BigInt> func1( nalu_hypre_ParCSRMatrixNumCols(B),
                                          nalu_hypre_ParCSRMatrixFirstColDiag(B),
                                          nalu_hypre_ParCSRMatrixDeviceColMapOffd(B) );
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         nalu_hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         nalu_hypre_CSRMatrixJ(Cbar) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                         nalu_hypre_CSRMatrixBigJ(Cint),
                         func1 );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         nalu_hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         nalu_hypre_CSRMatrixJ(Cbar) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                         nalu_hypre_CSRMatrixBigJ(Cint),
                         func1 );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure Cint is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      nalu_hypre_CSRMatrixData(Cint) = nalu_hypre_CSRMatrixData(Cbar) + local_nnz_Cbar;

#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time Cint %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      nalu_hypre_ExchangeExternalRowsDeviceInit(Cint, nalu_hypre_ParCSRMatrixCommPkg(A), 1, &request);
      Cext = nalu_hypre_ExchangeExternalRowsDeviceWait(request);

      nalu_hypre_TFree(nalu_hypre_CSRMatrixBigJ(Cint), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(Cint, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TMemcpy(nalu_hypre_CSRMatrixI(Cbar) + nalu_hypre_ParCSRMatrixNumCols(A), &local_nnz_Cbar, NALU_HYPRE_Int, 1,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time Cext %f\n", t2);
      nalu_hypre_ParPrintf(comm, "Size Cext %d %d %d\n", nalu_hypre_CSRMatrixNumRows(Cext),
                      nalu_hypre_CSRMatrixNumCols(Cext), nalu_hypre_CSRMatrixNumNonzeros(Cext));
#endif

      /* add Cext to local part of Cbar */
      nalu_hypre_ParCSRTMatMatPartialAddDevice(nalu_hypre_ParCSRMatrixCommPkg(A),
                                          nalu_hypre_ParCSRMatrixNumCols(A),
                                          nalu_hypre_ParCSRMatrixNumCols(B),
                                          nalu_hypre_ParCSRMatrixFirstColDiag(B),
                                          nalu_hypre_ParCSRMatrixLastColDiag(B),
                                          nalu_hypre_CSRMatrixNumCols(B_offd),
                                          nalu_hypre_ParCSRMatrixDeviceColMapOffd(B),
                                          local_nnz_Cbar,
                                          Cbar,
                                          Cext,
                                          &C_diag,
                                          &C_offd,
                                          &num_cols_offd_C,
                                          &col_map_offd_C);
   }
   else
   {
      nalu_hypre_CSRMatrix *AT_diag;
      nalu_hypre_CSRMatrix *B_diag = nalu_hypre_ParCSRMatrixDiag(B);
#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      nalu_hypre_CSRMatrixTransposeDevice(A_diag, &AT_diag, 1);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time Transpose %f\n", t2);
#endif
#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      C_diag = nalu_hypre_CSRMatrixMultiplyDevice(AT_diag, B_diag);
#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time SpGemm %f\n", t2);
#endif
      C_offd = nalu_hypre_CSRMatrixCreate(nalu_hypre_ParCSRMatrixNumCols(A), 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, NALU_HYPRE_MEMORY_DEVICE);
      if (keep_transpose)
      {
         nalu_hypre_ParCSRMatrixDiagT(A) = AT_diag;
      }
      else
      {
         nalu_hypre_CSRMatrixDestroy(AT_diag);
      }
   }

   /* Move the diagonal entry to the first of each row */
   nalu_hypre_CSRMatrixMoveDiagFirstDevice(C_diag);

   C = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(A),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(B),
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(B),
                                num_cols_offd_C,
                                nalu_hypre_CSRMatrixNumNonzeros(C_diag),
                                nalu_hypre_CSRMatrixNumNonzeros(C_offd));

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(C));
   nalu_hypre_ParCSRMatrixDiag(C) = C_diag;

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(C));
   nalu_hypre_ParCSRMatrixOffd(C) = C_offd;

   nalu_hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;

   nalu_hypre_ParCSRMatrixCompressOffdMapDevice(C);

   nalu_hypre_ParCSRMatrixCopyColMapOffdToHost(C);

   nalu_hypre_assert(!nalu_hypre_CSRMatrixCheckDiagFirstDevice(nalu_hypre_ParCSRMatrixDiag(C)));

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

#if PARCSRGEMM_TIMING > 0
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   tb = nalu_hypre_MPI_Wtime() - ta;
   nalu_hypre_ParPrintf(comm, "Time nalu_hypre_ParCSRTMatMatKTDevice %f\n", tb);
#endif

   return C;
}

/* C = R^{T} * A * P */
nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatrixRAPKTDevice( nalu_hypre_ParCSRMatrix *R,
                               nalu_hypre_ParCSRMatrix *A,
                               nalu_hypre_ParCSRMatrix *P,
                               NALU_HYPRE_Int           keep_transpose )
{
   MPI_Comm             comm   = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix     *R_diag = nalu_hypre_ParCSRMatrixDiag(R);
   nalu_hypre_CSRMatrix     *R_offd = nalu_hypre_ParCSRMatrixOffd(R);

   nalu_hypre_ParCSRMatrix  *C;
   nalu_hypre_CSRMatrix     *C_diag;
   nalu_hypre_CSRMatrix     *C_offd;
   NALU_HYPRE_Int            num_cols_offd_C = 0;
   NALU_HYPRE_BigInt        *col_map_offd_C = NULL;

   NALU_HYPRE_Int            num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if ( nalu_hypre_ParCSRMatrixGlobalNumRows(R) != nalu_hypre_ParCSRMatrixGlobalNumRows(A) ||
        nalu_hypre_ParCSRMatrixGlobalNumCols(A) != nalu_hypre_ParCSRMatrixGlobalNumRows(P) )
   {
      nalu_hypre_error_in_arg(1);
      nalu_hypre_printf(" Error! Incompatible matrix global dimensions!\n");
      return NULL;
   }

   if ( nalu_hypre_ParCSRMatrixNumRows(R) != nalu_hypre_ParCSRMatrixNumRows(A) ||
        nalu_hypre_ParCSRMatrixNumCols(A) != nalu_hypre_ParCSRMatrixNumRows(P) )
   {
      nalu_hypre_error_in_arg(1);
      nalu_hypre_printf(" Error! Incompatible matrix local dimensions!\n");
      return NULL;
   }

   if (num_procs > 1)
   {
      void *request;
      nalu_hypre_CSRMatrix *Abar, *RbarT, *Pext, *Pbar, *R_diagT, *R_offdT, *Cbar, *Cint, *Cext;
      NALU_HYPRE_Int num_cols_offd, local_nnz_Cbar;
      NALU_HYPRE_BigInt *col_map_offd;

      nalu_hypre_ParCSRMatrixExtractBExtDeviceInit(P, A, 1, &request);

      Abar = nalu_hypre_ConcatDiagAndOffdDevice(A);

      if (nalu_hypre_ParCSRMatrixDiagT(R))
      {
         R_diagT = nalu_hypre_ParCSRMatrixDiagT(R);
      }
      else
      {
         nalu_hypre_CSRMatrixTransposeDevice(R_diag, &R_diagT, 1);
      }

      if (nalu_hypre_ParCSRMatrixOffdT(R))
      {
         R_offdT = nalu_hypre_ParCSRMatrixOffdT(R);
      }
      else
      {
         nalu_hypre_CSRMatrixTransposeDevice(R_offd, &R_offdT, 1);
      }

      RbarT = nalu_hypre_CSRMatrixStack2Device(R_diagT, R_offdT);

      if (!nalu_hypre_ParCSRMatrixDiagT(R))
      {
         if (keep_transpose)
         {
            nalu_hypre_ParCSRMatrixDiagT(R) = R_diagT;
         }
         else
         {
            nalu_hypre_CSRMatrixDestroy(R_diagT);
         }
      }

      if (!nalu_hypre_ParCSRMatrixOffdT(R))
      {
         if (keep_transpose)
         {
            nalu_hypre_ParCSRMatrixOffdT(R) = R_offdT;
         }
         else
         {
            nalu_hypre_CSRMatrixDestroy(R_offdT);
         }
      }

      Pext = nalu_hypre_ParCSRMatrixExtractBExtDeviceWait(request);
      nalu_hypre_ConcatDiagOffdAndExtDevice(P, Pext, &Pbar, &num_cols_offd, &col_map_offd);
      nalu_hypre_CSRMatrixDestroy(Pext);

      Cbar = nalu_hypre_CSRMatrixTripleMultiplyDevice(RbarT, Abar, Pbar);

      nalu_hypre_CSRMatrixDestroy(RbarT);
      nalu_hypre_CSRMatrixDestroy(Abar);
      nalu_hypre_CSRMatrixDestroy(Pbar);

      nalu_hypre_assert(nalu_hypre_CSRMatrixNumRows(Cbar) ==
                   nalu_hypre_ParCSRMatrixNumCols(R) + nalu_hypre_CSRMatrixNumCols(R_offd));
      nalu_hypre_assert(nalu_hypre_CSRMatrixNumCols(Cbar) ==
                   nalu_hypre_ParCSRMatrixNumCols(P) + num_cols_offd);

      nalu_hypre_TMemcpy(&local_nnz_Cbar,
                    nalu_hypre_CSRMatrixI(Cbar) + nalu_hypre_ParCSRMatrixNumCols(R),
                    NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

      // Cint is the bottom part of Cbar
      Cint = nalu_hypre_CSRMatrixCreate(nalu_hypre_CSRMatrixNumCols(R_offd), nalu_hypre_CSRMatrixNumCols(Cbar),
                                   nalu_hypre_CSRMatrixNumNonzeros(Cbar) - local_nnz_Cbar);
      nalu_hypre_CSRMatrixMemoryLocation(Cint) = NALU_HYPRE_MEMORY_DEVICE;
      nalu_hypre_CSRMatrixOwnsData(Cint) = 0;

      nalu_hypre_CSRMatrixI(Cint) = nalu_hypre_CSRMatrixI(Cbar) + nalu_hypre_ParCSRMatrixNumCols(R);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         nalu_hypre_CSRMatrixI(Cint),
                         nalu_hypre_CSRMatrixI(Cint) + nalu_hypre_CSRMatrixNumRows(Cint) + 1,
                         nalu_hypre_CSRMatrixI(Cint),
      [const_val = local_nnz_Cbar] (const auto & x) {return x - const_val;} );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         nalu_hypre_CSRMatrixI(Cint),
                         nalu_hypre_CSRMatrixI(Cint) + nalu_hypre_CSRMatrixNumRows(Cint) + 1,
                         thrust::make_constant_iterator(local_nnz_Cbar),
                         nalu_hypre_CSRMatrixI(Cint),
                         thrust::minus<NALU_HYPRE_Int>() );
#endif

      // Change Cint into a BigJ matrix
      // RL: TODO FIX the 'big' num of columns to global size
      nalu_hypre_CSRMatrixBigJ(Cint) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,
                                               nalu_hypre_CSRMatrixNumNonzeros(Cint),
                                               NALU_HYPRE_MEMORY_DEVICE);

      RAP_functor<1, NALU_HYPRE_BigInt> func1(nalu_hypre_ParCSRMatrixNumCols(P),
                                         nalu_hypre_ParCSRMatrixFirstColDiag(P),
                                         col_map_offd);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         nalu_hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         nalu_hypre_CSRMatrixJ(Cbar) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                         nalu_hypre_CSRMatrixBigJ(Cint),
                         func1 );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         nalu_hypre_CSRMatrixJ(Cbar) + local_nnz_Cbar,
                         nalu_hypre_CSRMatrixJ(Cbar) + nalu_hypre_CSRMatrixNumNonzeros(Cbar),
                         nalu_hypre_CSRMatrixBigJ(Cint),
                         func1 );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure Cint is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      nalu_hypre_CSRMatrixData(Cint) = nalu_hypre_CSRMatrixData(Cbar) + local_nnz_Cbar;

      nalu_hypre_ExchangeExternalRowsDeviceInit(Cint, nalu_hypre_ParCSRMatrixCommPkg(R), 1, &request);
      Cext = nalu_hypre_ExchangeExternalRowsDeviceWait(request);

      nalu_hypre_TFree(nalu_hypre_CSRMatrixBigJ(Cint), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(Cint, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TMemcpy(nalu_hypre_CSRMatrixI(Cbar) + nalu_hypre_ParCSRMatrixNumCols(R),
                    &local_nnz_Cbar, NALU_HYPRE_Int, 1,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);

      /* add Cext to local part of Cbar */
      nalu_hypre_ParCSRTMatMatPartialAddDevice(nalu_hypre_ParCSRMatrixCommPkg(R),
                                          nalu_hypre_ParCSRMatrixNumCols(R),
                                          nalu_hypre_ParCSRMatrixNumCols(P),
                                          nalu_hypre_ParCSRMatrixFirstColDiag(P),
                                          nalu_hypre_ParCSRMatrixLastColDiag(P),
                                          num_cols_offd,
                                          col_map_offd,
                                          local_nnz_Cbar,
                                          Cbar,
                                          Cext,
                                          &C_diag,
                                          &C_offd,
                                          &num_cols_offd_C,
                                          &col_map_offd_C);

      nalu_hypre_TFree(col_map_offd, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      nalu_hypre_CSRMatrix *R_diagT;
      nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
      nalu_hypre_CSRMatrix *P_diag = nalu_hypre_ParCSRMatrixDiag(P);

      /* Recover or compute transpose of R_diag */
      if (nalu_hypre_ParCSRMatrixDiagT(R))
      {
         R_diagT = nalu_hypre_ParCSRMatrixDiagT(R);
      }
      else
      {
         nalu_hypre_CSRMatrixTransposeDevice(R_diag, &R_diagT, 1);
      }

      C_diag = nalu_hypre_CSRMatrixTripleMultiplyDevice(R_diagT, A_diag, P_diag);
      C_offd = nalu_hypre_CSRMatrixCreate(nalu_hypre_ParCSRMatrixNumCols(R), 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, NALU_HYPRE_MEMORY_DEVICE);

      /* Keep or destroy transpose of R_diag */
      if (!nalu_hypre_ParCSRMatrixDiagT(R))
      {
         if (keep_transpose)
         {
            nalu_hypre_ParCSRMatrixDiagT(R) = R_diagT;
         }
         else
         {
            nalu_hypre_CSRMatrixDestroy(R_diagT);
         }
      }
   }

   /* Move the diagonal entry to the first of each row */
   nalu_hypre_CSRMatrixMoveDiagFirstDevice(C_diag);

   C = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(R),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(P),
                                nalu_hypre_ParCSRMatrixColStarts(R),
                                nalu_hypre_ParCSRMatrixColStarts(P),
                                num_cols_offd_C,
                                nalu_hypre_CSRMatrixNumNonzeros(C_diag),
                                nalu_hypre_CSRMatrixNumNonzeros(C_offd));

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(C));
   nalu_hypre_ParCSRMatrixDiag(C) = C_diag;

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(C));
   nalu_hypre_ParCSRMatrixOffd(C) = C_offd;

   nalu_hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;

   nalu_hypre_ParCSRMatrixCompressOffdMapDevice(C);
   nalu_hypre_ParCSRMatrixCopyColMapOffdToHost(C);

   nalu_hypre_assert(!nalu_hypre_CSRMatrixCheckDiagFirstDevice(nalu_hypre_ParCSRMatrixDiag(C)));

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return C;
}

NALU_HYPRE_Int
nalu_hypre_ParCSRTMatMatPartialAddDevice( nalu_hypre_ParCSRCommPkg *comm_pkg,
                                     NALU_HYPRE_Int            num_rows,
                                     NALU_HYPRE_Int            num_cols,
                                     NALU_HYPRE_BigInt         first_col_diag,
                                     NALU_HYPRE_BigInt         last_col_diag,
                                     NALU_HYPRE_Int            num_cols_offd,
                                     NALU_HYPRE_BigInt        *col_map_offd,
                                     NALU_HYPRE_Int            local_nnz_Cbar,
                                     nalu_hypre_CSRMatrix     *Cbar,
                                     nalu_hypre_CSRMatrix     *Cext,
                                     nalu_hypre_CSRMatrix    **C_diag_ptr,
                                     nalu_hypre_CSRMatrix    **C_offd_ptr,
                                     NALU_HYPRE_Int           *num_cols_offd_C_ptr,
                                     NALU_HYPRE_BigInt       **col_map_offd_C_ptr )
{
#if PARCSRGEMM_TIMING > 1
   MPI_Comm comm = nalu_hypre_ParCSRCommPkgComm(comm_pkg);
   NALU_HYPRE_Real t1, t2;
   t1 = nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Int        Cext_nnz = nalu_hypre_CSRMatrixNumNonzeros(Cext);
   NALU_HYPRE_Int        num_cols_offd_C;
   NALU_HYPRE_BigInt    *col_map_offd_C;
   nalu_hypre_CSRMatrix *Cz;

   // local part of Cbar
   nalu_hypre_CSRMatrix *Cbar_local = nalu_hypre_CSRMatrixCreate(num_rows, nalu_hypre_CSRMatrixNumCols(Cbar),
                                                       local_nnz_Cbar);
   nalu_hypre_CSRMatrixI(Cbar_local) = nalu_hypre_CSRMatrixI(Cbar);
   nalu_hypre_CSRMatrixJ(Cbar_local) = nalu_hypre_CSRMatrixJ(Cbar);
   nalu_hypre_CSRMatrixData(Cbar_local) = nalu_hypre_CSRMatrixData(Cbar);
   nalu_hypre_CSRMatrixOwnsData(Cbar_local) = 0;
   nalu_hypre_CSRMatrixMemoryLocation(Cbar_local) = NALU_HYPRE_MEMORY_DEVICE;

   if (!Cext_nnz)
   {
      num_cols_offd_C = num_cols_offd;
      col_map_offd_C = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(col_map_offd_C, col_map_offd, NALU_HYPRE_BigInt, num_cols_offd,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
      Cz = Cbar_local;
   }
   else
   {
      in_range<NALU_HYPRE_BigInt> pred1(first_col_diag, last_col_diag);

      if (!nalu_hypre_CSRMatrixJ(Cext))
      {
         nalu_hypre_CSRMatrixJ(Cext) = nalu_hypre_TAlloc(NALU_HYPRE_Int, Cext_nnz, NALU_HYPRE_MEMORY_DEVICE);
      }

      NALU_HYPRE_BigInt *Cext_bigj = nalu_hypre_CSRMatrixBigJ(Cext);
      NALU_HYPRE_BigInt *big_work  = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, Cext_nnz, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int    *work      = nalu_hypre_TAlloc(NALU_HYPRE_Int, Cext_nnz, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int    *map_offd_to_C;

      // Convert Cext from BigJ to J
      // Cext offd
#if defined(NALU_HYPRE_USING_SYCL)
      auto off_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(oneapi::dpl::counting_iterator(0),
                                                                       Cext_bigj),
                                        oneapi::dpl::make_zip_iterator(oneapi::dpl::counting_iterator(0),
                                                                       Cext_bigj) + Cext_nnz,
                                        Cext_bigj,
                                        oneapi::dpl::make_zip_iterator(work, big_work),
                                        std::not_fn(pred1) );

      NALU_HYPRE_Int Cext_offd_nnz = std::get<0>(off_end.base()) - work;
#else
      auto off_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), Cext_bigj)),
                                        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0),
                                                                                     Cext_bigj)) + Cext_nnz,
                                        Cext_bigj,
                                        thrust::make_zip_iterator(thrust::make_tuple(work, big_work)),
                                        thrust::not1(pred1) );

      NALU_HYPRE_Int Cext_offd_nnz = thrust::get<0>(off_end.get_iterator_tuple()) - work;
#endif

      nalu_hypre_CSRMatrixMergeColMapOffd(num_cols_offd, col_map_offd, Cext_offd_nnz, big_work,
                                     &num_cols_offd_C, &col_map_offd_C, &map_offd_to_C);

#if defined(NALU_HYPRE_USING_SYCL)
      /* WM: onedpl lower_bound currently does not accept zero length values */
      if (Cext_offd_nnz > 0)
      {
         NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                            col_map_offd_C,
                            col_map_offd_C + num_cols_offd_C,
                            big_work,
                            big_work + Cext_offd_nnz,
                            oneapi::dpl::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work) );
      }

      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         oneapi::dpl::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work),
                         oneapi::dpl::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work) + Cext_offd_nnz,
                         oneapi::dpl::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work),
      [const_val = num_cols] (const auto & x) {return x + const_val;} );
#else
      NALU_HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_C,
                         col_map_offd_C + num_cols_offd_C,
                         big_work,
                         big_work + Cext_offd_nnz,
                         thrust::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work) );

      NALU_HYPRE_THRUST_CALL( transform,
                         thrust::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work),
                         thrust::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work) + Cext_offd_nnz,
                         thrust::make_constant_iterator(num_cols),
                         thrust::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work),
                         thrust::plus<NALU_HYPRE_Int>() );
#endif

      // Cext diag
#if defined(NALU_HYPRE_USING_SYCL)
      auto dia_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(oneapi::dpl::counting_iterator(0),
                                                                       Cext_bigj),
                                        oneapi::dpl::make_zip_iterator(oneapi::dpl::counting_iterator(0),
                                                                       Cext_bigj) + Cext_nnz,
                                        Cext_bigj,
                                        oneapi::dpl::make_zip_iterator(work, big_work),
                                        pred1 );

      NALU_HYPRE_Int Cext_diag_nnz = std::get<0>(dia_end.base()) - work;
#else
      auto dia_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), Cext_bigj)),
                                        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0),
                                                                                     Cext_bigj)) + Cext_nnz,
                                        Cext_bigj,
                                        thrust::make_zip_iterator(thrust::make_tuple(work, big_work)),
                                        pred1 );

      NALU_HYPRE_Int Cext_diag_nnz = thrust::get<0>(dia_end.get_iterator_tuple()) - work;
#endif

      nalu_hypre_assert(Cext_diag_nnz + Cext_offd_nnz == Cext_nnz);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         big_work,
                         big_work + Cext_diag_nnz,
                         oneapi::dpl::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work),
      [const_val = first_col_diag](const auto & x) {return x - const_val;} );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         big_work,
                         big_work + Cext_diag_nnz,
                         thrust::make_constant_iterator(first_col_diag),
                         thrust::make_permutation_iterator(nalu_hypre_CSRMatrixJ(Cext), work),
                         thrust::minus<NALU_HYPRE_BigInt>());
#endif

      nalu_hypre_CSRMatrixNumCols(Cext) = num_cols + num_cols_offd_C;

      // transform Cbar_local J index
      RAP_functor<2, NALU_HYPRE_Int> func2(num_cols, 0, map_offd_to_C);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         nalu_hypre_CSRMatrixJ(Cbar_local),
                         nalu_hypre_CSRMatrixJ(Cbar_local) + local_nnz_Cbar,
                         nalu_hypre_CSRMatrixJ(Cbar_local),
                         func2 );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         nalu_hypre_CSRMatrixJ(Cbar_local),
                         nalu_hypre_CSRMatrixJ(Cbar_local) + local_nnz_Cbar,
                         nalu_hypre_CSRMatrixJ(Cbar_local),
                         func2 );
#endif

      nalu_hypre_CSRMatrixNumCols(Cbar_local) = num_cols + num_cols_offd_C;

      nalu_hypre_TFree(big_work,      NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(work,          NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(map_offd_to_C, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(Cext_bigj,     NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixBigJ(Cext) = NULL;

#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time PartialAdd1 %f\n", t2);
#endif

#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif

      // IE = [I, E]
      nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

      NALU_HYPRE_Int  num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      NALU_HYPRE_Int  num_elemt = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      NALU_HYPRE_Int *send_map  = nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);

      nalu_hypre_CSRMatrix *IE = nalu_hypre_CSRMatrixCreate(num_rows, num_rows + num_elemt,
                                                  num_rows + num_elemt);
      nalu_hypre_CSRMatrixMemoryLocation(IE) = NALU_HYPRE_MEMORY_DEVICE;

      NALU_HYPRE_Int     *ie_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows + num_elemt, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int     *ie_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows + num_elemt, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Complex *ie_a  = NULL;

      if (nalu_hypre_HandleSpgemmUseVendor(nalu_hypre_handle()))
      {
         ie_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_rows + num_elemt, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
         NALU_HYPRE_ONEDPL_CALL(std::fill, ie_a, ie_a + num_rows + num_elemt, 1.0);
#else
         NALU_HYPRE_THRUST_CALL(fill, ie_a, ie_a + num_rows + num_elemt, 1.0);
#endif
      }

#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_sequence(ie_ii, ie_ii + num_rows, 0);
      NALU_HYPRE_ONEDPL_CALL( std::copy, send_map, send_map + num_elemt, ie_ii + num_rows);
      hypreSycl_sequence(ie_j, ie_j + num_rows + num_elemt, 0);
      auto zipped_begin = oneapi::dpl::make_zip_iterator(ie_ii, ie_j);
      NALU_HYPRE_ONEDPL_CALL( std::stable_sort, zipped_begin, zipped_begin + num_rows + num_elemt,
      [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); } );
#else
      NALU_HYPRE_THRUST_CALL( sequence, ie_ii, ie_ii + num_rows);
      NALU_HYPRE_THRUST_CALL( copy, send_map, send_map + num_elemt, ie_ii + num_rows);
      NALU_HYPRE_THRUST_CALL( sequence, ie_j, ie_j + num_rows + num_elemt);
      NALU_HYPRE_THRUST_CALL( stable_sort_by_key, ie_ii, ie_ii + num_rows + num_elemt, ie_j );
#endif

      NALU_HYPRE_Int *ie_i = hypreDevice_CsrRowIndicesToPtrs(num_rows, num_rows + num_elemt, ie_ii);
      nalu_hypre_TFree(ie_ii, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_CSRMatrixI(IE)    = ie_i;
      nalu_hypre_CSRMatrixJ(IE)    = ie_j;
      nalu_hypre_CSRMatrixData(IE) = ie_a;

      // CC = [Cbar_local; Cext]
      nalu_hypre_CSRMatrix *CC = nalu_hypre_CSRMatrixStack2Device(Cbar_local, Cext);
      nalu_hypre_CSRMatrixDestroy(Cbar);
      nalu_hypre_CSRMatrixDestroy(Cext);

#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time PartialAdd2 %f\n", t2);
#endif

      // Cz = IE * CC
#if PARCSRGEMM_TIMING > 1
      t1 = nalu_hypre_MPI_Wtime();
#endif
      Cz = nalu_hypre_CSRMatrixMultiplyDevice(IE, CC);

      nalu_hypre_CSRMatrixDestroy(IE);
      nalu_hypre_CSRMatrixDestroy(CC);

#if PARCSRGEMM_TIMING > 1
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
      t2 = nalu_hypre_MPI_Wtime() - t1;
      nalu_hypre_ParPrintf(comm, "Time PartialAdd-SpGemm %f\n", t2);
#endif
   }

#if PARCSRGEMM_TIMING > 1
   t1 = nalu_hypre_MPI_Wtime();
#endif

   // split into diag and offd
   NALU_HYPRE_Int local_nnz_C = nalu_hypre_CSRMatrixNumNonzeros(Cz);

   NALU_HYPRE_Int     *zmp_i = hypreDevice_CsrRowPtrsToIndices(num_rows, local_nnz_C, nalu_hypre_CSRMatrixI(Cz));
   NALU_HYPRE_Int     *zmp_j = nalu_hypre_CSRMatrixJ(Cz);
   NALU_HYPRE_Complex *zmp_a = nalu_hypre_CSRMatrixData(Cz);

   in_range<NALU_HYPRE_Int> pred(0, num_cols - 1);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int nnz_C_diag = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                             zmp_j,
                                             zmp_j + local_nnz_C,
                                             pred );
#else
   NALU_HYPRE_Int nnz_C_diag = NALU_HYPRE_THRUST_CALL( count_if,
                                             zmp_j,
                                             zmp_j + local_nnz_C,
                                             pred );
#endif
   NALU_HYPRE_Int nnz_C_offd = local_nnz_C - nnz_C_diag;

   // diag
   nalu_hypre_CSRMatrix *C_diag = nalu_hypre_CSRMatrixCreate(num_rows, num_cols, nnz_C_diag);
   nalu_hypre_CSRMatrixInitialize_v2(C_diag, 0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *C_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_C_diag, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *C_diag_j = nalu_hypre_CSRMatrixJ(C_diag);
   NALU_HYPRE_Complex *C_diag_a = nalu_hypre_CSRMatrixData(C_diag);

#if defined(NALU_HYPRE_USING_SYCL)
   auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(zmp_i, zmp_j, zmp_a),
                                     oneapi::dpl::make_zip_iterator(zmp_i, zmp_j, zmp_a) + local_nnz_C,
                                     zmp_j,
                                     oneapi::dpl::make_zip_iterator(C_diag_ii, C_diag_j, C_diag_a),
                                     pred );
   nalu_hypre_assert( std::get<0>(new_end.base()) == C_diag_ii + nnz_C_diag );
#else
   auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                     thrust::make_zip_iterator(thrust::make_tuple(zmp_i, zmp_j, zmp_a)),
                                     thrust::make_zip_iterator(thrust::make_tuple(zmp_i, zmp_j, zmp_a)) + local_nnz_C,
                                     zmp_j,
                                     thrust::make_zip_iterator(thrust::make_tuple(C_diag_ii, C_diag_j, C_diag_a)),
                                     pred );
   nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == C_diag_ii + nnz_C_diag );
#endif
   hypreDevice_CsrRowIndicesToPtrs_v2(nalu_hypre_CSRMatrixNumRows(C_diag), nnz_C_diag, C_diag_ii,
                                      nalu_hypre_CSRMatrixI(C_diag));
   nalu_hypre_TFree(C_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

   // offd
   nalu_hypre_CSRMatrix *C_offd = nalu_hypre_CSRMatrixCreate(num_rows, num_cols_offd_C, nnz_C_offd);
   nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *C_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_C_offd, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *C_offd_j = nalu_hypre_CSRMatrixJ(C_offd);
   NALU_HYPRE_Complex *C_offd_a = nalu_hypre_CSRMatrixData(C_offd);
#if defined(NALU_HYPRE_USING_SYCL)
   new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(zmp_i, zmp_j, zmp_a),
                                oneapi::dpl::make_zip_iterator(zmp_i, zmp_j, zmp_a) + local_nnz_C,
                                zmp_j,
                                oneapi::dpl::make_zip_iterator(C_offd_ii, C_offd_j, C_offd_a),
                                std::not_fn(pred) );
   nalu_hypre_assert( std::get<0>(new_end.base()) == C_offd_ii + nnz_C_offd );
#else
   new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                thrust::make_zip_iterator(thrust::make_tuple(zmp_i, zmp_j, zmp_a)),
                                thrust::make_zip_iterator(thrust::make_tuple(zmp_i, zmp_j, zmp_a)) + local_nnz_C,
                                zmp_j,
                                thrust::make_zip_iterator(thrust::make_tuple(C_offd_ii, C_offd_j, C_offd_a)),
                                thrust::not1(pred) );
   nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == C_offd_ii + nnz_C_offd );
#endif
   hypreDevice_CsrRowIndicesToPtrs_v2(nalu_hypre_CSRMatrixNumRows(C_offd), nnz_C_offd, C_offd_ii,
                                      nalu_hypre_CSRMatrixI(C_offd));
   nalu_hypre_TFree(C_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      C_offd_j,
                      C_offd_j + nnz_C_offd,
                      C_offd_j,
   [const_val = num_cols] (const auto & x) {return x - const_val;} );
#else
   NALU_HYPRE_THRUST_CALL( transform,
                      C_offd_j,
                      C_offd_j + nnz_C_offd,
                      thrust::make_constant_iterator(num_cols),
                      C_offd_j,
                      thrust::minus<NALU_HYPRE_Int>() );
#endif

   // free
   nalu_hypre_TFree(Cbar_local, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(zmp_i, NALU_HYPRE_MEMORY_DEVICE);

   if (!Cext_nnz)
   {
      nalu_hypre_CSRMatrixDestroy(Cbar);
      nalu_hypre_CSRMatrixDestroy(Cext);
   }
   else
   {
      nalu_hypre_CSRMatrixDestroy(Cz);
   }

#if PARCSRGEMM_TIMING > 1
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   t2 = nalu_hypre_MPI_Wtime() - t1;
   nalu_hypre_ParPrintf(comm, "Time Split %f\n", t2);
#endif

   // output
   *C_diag_ptr = C_diag;
   *C_offd_ptr = C_offd;
   *num_cols_offd_C_ptr = num_cols_offd_C;
   *col_map_offd_C_ptr = col_map_offd_C;

   return nalu_hypre_error_flag;
}

#endif /* #if defined(NALU_HYPRE_USING_GPU) */
