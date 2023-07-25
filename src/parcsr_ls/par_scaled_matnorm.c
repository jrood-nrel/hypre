/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * computes |D^-1/2 A D^-1/2 |_sup where D diagonal matrix
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixScaledNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixScaledNorm( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Real *scnorm)
{
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   nalu_hypre_ParCSRCommPkg  *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   MPI_Comm     comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix      *diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int      *diag_i = nalu_hypre_CSRMatrixI(diag);
   NALU_HYPRE_Int      *diag_j = nalu_hypre_CSRMatrixJ(diag);
   NALU_HYPRE_Real     *diag_data = nalu_hypre_CSRMatrixData(diag);
   nalu_hypre_CSRMatrix      *offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int      *offd_i = nalu_hypre_CSRMatrixI(offd);
   NALU_HYPRE_Int      *offd_j = nalu_hypre_CSRMatrixJ(offd);
   NALU_HYPRE_Real     *offd_data = nalu_hypre_CSRMatrixData(offd);
   NALU_HYPRE_BigInt       global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt           *row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_Int       num_rows = nalu_hypre_CSRMatrixNumRows(diag);

   nalu_hypre_ParVector      *dinvsqrt;
   NALU_HYPRE_Real     *dis_data;
   nalu_hypre_Vector         *dis_ext;
   NALU_HYPRE_Real     *dis_ext_data;
   nalu_hypre_Vector         *sum;
   NALU_HYPRE_Real     *sum_data;

   NALU_HYPRE_Int         num_cols_offd = nalu_hypre_CSRMatrixNumCols(offd);
   NALU_HYPRE_Int         num_sends, i, j, index, start;

   NALU_HYPRE_Real *d_buf_data;
   NALU_HYPRE_Real  mat_norm, max_row_sum;

   dinvsqrt = nalu_hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   nalu_hypre_ParVectorInitialize(dinvsqrt);
   dis_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(dinvsqrt));
   dis_ext = nalu_hypre_SeqVectorCreate(num_cols_offd);
   nalu_hypre_SeqVectorInitialize(dis_ext);
   dis_ext_data = nalu_hypre_VectorData(dis_ext);
   sum = nalu_hypre_SeqVectorCreate(num_rows);
   nalu_hypre_SeqVectorInitialize(sum);
   sum_data = nalu_hypre_VectorData(sum);

   /* generate dinvsqrt */
   for (i = 0; i < num_rows; i++)
   {
      dis_data[i] = 1.0 / nalu_hypre_sqrt(nalu_hypre_abs(diag_data[diag_i[i]]));
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
   d_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                           num_sends), NALU_HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         d_buf_data[index++]
            = dis_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   comm_handle = nalu_hypre_ParCSRCommHandleCreate( 1, comm_pkg, d_buf_data,
                                               dis_ext_data);

   for (i = 0; i < num_rows; i++)
   {
      for (j = diag_i[i]; j < diag_i[i + 1]; j++)
      {
         sum_data[i] += nalu_hypre_abs(diag_data[j]) * dis_data[i] * dis_data[diag_j[j]];
      }
   }
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   for (i = 0; i < num_rows; i++)
   {
      for (j = offd_i[i]; j < offd_i[i + 1]; j++)
      {
         sum_data[i] += nalu_hypre_abs(offd_data[j]) * dis_data[i] * dis_ext_data[offd_j[j]];
      }
   }

   max_row_sum = 0;
   for (i = 0; i < num_rows; i++)
   {
      if (max_row_sum < sum_data[i])
      {
         max_row_sum = sum_data[i];
      }
   }

   nalu_hypre_MPI_Allreduce(&max_row_sum, &mat_norm, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_MAX, comm);

   nalu_hypre_ParVectorDestroy(dinvsqrt);
   nalu_hypre_SeqVectorDestroy(sum);
   nalu_hypre_SeqVectorDestroy(dis_ext);
   nalu_hypre_TFree(d_buf_data, NALU_HYPRE_MEMORY_HOST);

   *scnorm = mat_norm;
   return 0;
}
