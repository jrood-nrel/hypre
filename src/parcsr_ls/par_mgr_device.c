/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Two-grid system solver
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "seq_mv/protos.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
template<typename T>
struct functor : public thrust::binary_function<T, T, T>
{
   T scale;

   functor(T scale_) { scale = scale_; }

   __host__ __device__
   T operator()(T &x, T &y) const
   {
      return x + scale * (y - nalu_hypre_abs(x));
   }
};

void hypreDevice_extendWtoP( NALU_HYPRE_Int P_nr_of_rows, NALU_HYPRE_Int W_nr_of_rows, NALU_HYPRE_Int W_nr_of_cols,
                             NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int W_diag_nnz, NALU_HYPRE_Int *W_diag_i, NALU_HYPRE_Int *W_diag_j,
                             NALU_HYPRE_Complex *W_diag_data, NALU_HYPRE_Int *P_diag_i, NALU_HYPRE_Int *P_diag_j, NALU_HYPRE_Complex *P_diag_data,
                             NALU_HYPRE_Int *W_offd_i, NALU_HYPRE_Int *P_offd_i );

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

   nfpoints = NALU_HYPRE_THRUST_CALL( count,
                                 CF_marker,
                                 CF_marker + A_nr_of_rows,
                                 -1);

   if (method > 0)
   {
      nalu_hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, NULL, &A_FC, &A_FF);
      diag = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nfpoints, NALU_HYPRE_MEMORY_DEVICE);
      if (method == 1)
      {
         // extract diag inverse sqrt
         //        nalu_hypre_CSRMatrixExtractDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(A_FF), diag, 3);

         // L1-Jacobi-type interpolation
         NALU_HYPRE_Complex scal = 1.0;
         diag1 = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nfpoints, NALU_HYPRE_MEMORY_DEVICE);
         nalu_hypre_CSRMatrixExtractDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(A_FF), diag, 0);
         nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixDiag(A_FF), NULL, NULL, diag1, 1, 1.0, "set");
         nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixDiag(A_FC), NULL, NULL, diag1, 1, 1.0, "add");
         nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixOffd(A_FF), NULL, NULL, diag1, 1, 1.0, "add");
         nalu_hypre_CSRMatrixComputeRowSumDevice(nalu_hypre_ParCSRMatrixOffd(A_FC), NULL, NULL, diag1, 1, 1.0, "add");

         NALU_HYPRE_THRUST_CALL( transform, diag, diag + nfpoints, diag1, diag, functor<NALU_HYPRE_Complex>(scal));
         NALU_HYPRE_THRUST_CALL( transform, diag, diag + nfpoints, diag, 1.0 / _1);

         nalu_hypre_TFree(diag1, NALU_HYPRE_MEMORY_DEVICE);
      }
      else if (method == 2)
      {
         // extract diag inverse
         nalu_hypre_CSRMatrixExtractDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(A_FF), diag, 2);
      }

      NALU_HYPRE_THRUST_CALL( transform, diag, diag + nfpoints, diag, thrust::negate<NALU_HYPRE_Complex>() );

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

   //nalu_hypre_NvtxPushRangeColor("Extend matrix", 4);
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
   //nalu_hypre_NvtxPopRange();

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
      nalu_hypre_ParCSRMatrixNumNonzeros(P)         = nalu_hypre_ParCSRMatrixNumNonzeros(
                                                    A_FC) + nalu_hypre_ParCSRMatrixGlobalNumCols(A_FC);
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

   return nalu_hypre_error_flag;
}

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
   nalu_hypre_BoomerAMGRelax(A, f, CF_marker, 18, relax_points, relax_weight, 1.0, l1_norms, u, Vtemp,
                        NULL);

   return nalu_hypre_error_flag;
}

#endif
