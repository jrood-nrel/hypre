/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_block_mv.h"

/*---------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBlockCreateNodalA
 *
 * This is the block version of creating a nodal norm matrix.
 *
 * Option: determine which type of "norm" (or other measurement) is used.
 *
 *   1 = frobenius
 *   2 = sum of abs. value of all elements
 *   3 = largest element (positive or negative)
 *   4 = 1-norm
 *   5 = inf - norm
 *   6 = sum of all elements
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBlockCreateNodalA(nalu_hypre_ParCSRBlockMatrix *A,
                                 NALU_HYPRE_Int                option,
                                 NALU_HYPRE_Int                diag_option,
                                 nalu_hypre_ParCSRMatrix     **AN_ptr)
{
   MPI_Comm                 comm         = nalu_hypre_ParCSRBlockMatrixComm(A);
   nalu_hypre_CSRBlockMatrix    *A_diag       = nalu_hypre_ParCSRBlockMatrixDiag(A);
   NALU_HYPRE_Int               *A_diag_i     = nalu_hypre_CSRBlockMatrixI(A_diag);
   NALU_HYPRE_Real              *A_diag_data  = nalu_hypre_CSRBlockMatrixData(A_diag);

   NALU_HYPRE_Int                block_size = nalu_hypre_CSRBlockMatrixBlockSize(A_diag);
   NALU_HYPRE_Int                bnnz = block_size * block_size;

   nalu_hypre_CSRBlockMatrix    *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int               *A_offd_i        = nalu_hypre_CSRBlockMatrixI(A_offd);
   NALU_HYPRE_Real              *A_offd_data     = nalu_hypre_CSRBlockMatrixData(A_offd);
   NALU_HYPRE_Int               *A_diag_j        = nalu_hypre_CSRBlockMatrixJ(A_diag);
   NALU_HYPRE_Int               *A_offd_j        = nalu_hypre_CSRBlockMatrixJ(A_offd);

   NALU_HYPRE_BigInt            *row_starts      = nalu_hypre_ParCSRBlockMatrixRowStarts(A);
   NALU_HYPRE_BigInt            *col_map_offd    = nalu_hypre_ParCSRBlockMatrixColMapOffd(A);
   NALU_HYPRE_Int                num_nonzeros_diag;
   NALU_HYPRE_Int                num_nonzeros_offd = 0;
   NALU_HYPRE_Int                num_cols_offd = 0;

   nalu_hypre_ParCSRMatrix *AN;
   nalu_hypre_CSRMatrix    *AN_diag;
   NALU_HYPRE_Int          *AN_diag_i;
   NALU_HYPRE_Int          *AN_diag_j = NULL;
   NALU_HYPRE_Real         *AN_diag_data = NULL;
   nalu_hypre_CSRMatrix    *AN_offd;
   NALU_HYPRE_Int          *AN_offd_i;
   NALU_HYPRE_Int          *AN_offd_j = NULL;
   NALU_HYPRE_Real         *AN_offd_data = NULL;
   NALU_HYPRE_BigInt       *col_map_offd_AN = NULL;

   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRBlockMatrixCommPkg(A);
   NALU_HYPRE_Int            num_sends;
   NALU_HYPRE_Int            num_recvs;
   NALU_HYPRE_Int           *send_procs;
   NALU_HYPRE_Int           *send_map_starts;
   NALU_HYPRE_Int           *send_map_elmts;
   NALU_HYPRE_Int           *recv_procs;
   NALU_HYPRE_Int           *recv_vec_starts;

   nalu_hypre_ParCSRCommPkg *comm_pkg_AN = NULL;
   NALU_HYPRE_Int           *send_procs_AN = NULL;
   NALU_HYPRE_Int           *send_map_starts_AN = NULL;
   NALU_HYPRE_Int           *send_map_elmts_AN = NULL;
   NALU_HYPRE_Int           *recv_procs_AN = NULL;
   NALU_HYPRE_Int           *recv_vec_starts_AN = NULL;

   NALU_HYPRE_Int            i;

   NALU_HYPRE_Int            num_procs;
   NALU_HYPRE_Int            cnt;
   NALU_HYPRE_Int            norm_type;

   NALU_HYPRE_BigInt         global_num_nodes;
   NALU_HYPRE_Int            num_nodes;

   NALU_HYPRE_Int            index, k;

   NALU_HYPRE_Real           tmp;
   NALU_HYPRE_Real           sum;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if (!comm_pkg)
   {
      nalu_hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRBlockMatrixCommPkg(A);
   }

   norm_type = nalu_hypre_abs(option);

   /* Set up the new matrix AN */

   global_num_nodes = nalu_hypre_ParCSRBlockMatrixGlobalNumRows(A);
   num_nodes = nalu_hypre_CSRBlockMatrixNumRows(A_diag);

   /* the diag part */

   num_nonzeros_diag = A_diag_i[num_nodes];
   AN_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nodes + 1, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i <= num_nodes; i++)
   {
      AN_diag_i[i] = A_diag_i[i];
   }

   AN_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nonzeros_diag, NALU_HYPRE_MEMORY_HOST);
   AN_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_nonzeros_diag, NALU_HYPRE_MEMORY_HOST);

   AN_diag = nalu_hypre_CSRMatrixCreate(num_nodes, num_nodes, num_nonzeros_diag);
   nalu_hypre_CSRMatrixI(AN_diag) = AN_diag_i;
   nalu_hypre_CSRMatrixJ(AN_diag) = AN_diag_j;
   nalu_hypre_CSRMatrixData(AN_diag) = AN_diag_data;

   for (i = 0; i < num_nonzeros_diag; i++)
   {
      AN_diag_j[i]  = A_diag_j[i];
      nalu_hypre_CSRBlockMatrixBlockNorm(norm_type, &A_diag_data[i * bnnz],
                                    &tmp, block_size);
      AN_diag_data[i] = tmp;
   }


   if (diag_option == 1)
   {
      /* make the diag entry the negative of the sum of off-diag entries (NEED
       * to get more below!)*/
      /* the diagonal is the first element listed in each row - */
      for (i = 0; i < num_nodes; i++)
      {
         index = AN_diag_i[i];
         sum = 0.0;
         for (k = AN_diag_i[i] + 1; k < AN_diag_i[i + 1]; k++)
         {
            sum += AN_diag_data[k];

         }

         AN_diag_data[index] = -sum;
      }
   }
   else if (diag_option == 2)
   {
      /*  make all diagonal entries negative */
      /* the diagonal is the first element listed in each row - */

      for (i = 0; i < num_nodes; i++)
      {
         index = AN_diag_i[i];
         AN_diag_data[index] = -AN_diag_data[index];
      }
   }

   /* copy the commpkg */
   if (comm_pkg)
   {
      num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_procs = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg);
      send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

      if (num_sends)
      {
         send_procs_AN = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends, NALU_HYPRE_MEMORY_HOST);
         send_map_elmts_AN = nalu_hypre_CTAlloc(NALU_HYPRE_Int, send_map_starts[num_sends],
                                           NALU_HYPRE_MEMORY_HOST);
      }
      send_map_starts_AN = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends + 1, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         send_procs_AN[i] = send_procs[i];
         send_map_starts_AN[i + 1] = send_map_starts[i + 1];
      }

      cnt = send_map_starts_AN[num_sends];
      for (i = 0; i < cnt; i++)
      {
         send_map_elmts_AN[i] = send_map_elmts[i];
      }

      recv_vec_starts_AN = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs + 1, NALU_HYPRE_MEMORY_HOST);
      if (num_recvs)
      {
         recv_procs_AN = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs, NALU_HYPRE_MEMORY_HOST);
      }
      for (i = 0; i < num_recvs; i++)
      {
         recv_procs_AN[i] = recv_procs[i];
         recv_vec_starts_AN[i + 1] = recv_vec_starts[i + 1];
      }

      /* Create communication package */
      nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                       num_recvs, recv_procs_AN, recv_vec_starts_AN,
                                       num_sends, send_procs_AN, send_map_starts_AN,
                                       send_map_elmts_AN,
                                       &comm_pkg_AN);
   }

   /* the off-diag part */

   num_cols_offd = nalu_hypre_CSRBlockMatrixNumCols(A_offd);
   col_map_offd_AN = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_offd; i++)
   {
      col_map_offd_AN[i] = col_map_offd[i];
   }

   num_nonzeros_offd = A_offd_i[num_nodes];
   AN_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nodes + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= num_nodes; i++)
   {
      AN_offd_i[i] = A_offd_i[i];
   }

   AN_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nonzeros_offd, NALU_HYPRE_MEMORY_HOST);
   AN_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_nonzeros_offd, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_nonzeros_offd; i++)
   {
      AN_offd_j[i]  = A_offd_j[i];
      nalu_hypre_CSRBlockMatrixBlockNorm(norm_type, &A_offd_data[i * bnnz],
                                    &tmp, block_size);
      AN_offd_data[i] = tmp;
   }

   AN_offd = nalu_hypre_CSRMatrixCreate(num_nodes, num_cols_offd, num_nonzeros_offd);

   nalu_hypre_CSRMatrixI(AN_offd) = AN_offd_i;
   nalu_hypre_CSRMatrixJ(AN_offd) = AN_offd_j;
   nalu_hypre_CSRMatrixData(AN_offd) = AN_offd_data;

   if (diag_option == 1)
   {
      /* make the diag entry the negative of the sum of off-diag entries (here
         we are adding the off_diag contribution)*/
      /* the diagonal is the first element listed in each row of AN_diag_data - */
      for (i = 0; i < num_nodes; i++)
      {
         sum = 0.0;
         for (k = AN_offd_i[i]; k < AN_offd_i[i + 1]; k++)
         {
            sum += AN_offd_data[k];
         }
         index = AN_diag_i[i];/* location of diag entry in data */
         AN_diag_data[index] -= sum; /* subtract from current value */
      }
   }

   /* now create AN */
   AN = nalu_hypre_ParCSRMatrixCreate(comm, global_num_nodes, global_num_nodes,
                                 row_starts, row_starts, num_cols_offd,
                                 num_nonzeros_diag, num_nonzeros_offd);

   /* we already created the diag and offd matrices - so we don't need the ones
      created above */
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(AN));
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(AN));
   nalu_hypre_ParCSRMatrixDiag(AN) = AN_diag;
   nalu_hypre_ParCSRMatrixOffd(AN) = AN_offd;

   nalu_hypre_CSRMatrixMemoryLocation(AN_diag) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixMemoryLocation(AN_offd) = NALU_HYPRE_MEMORY_HOST;

   nalu_hypre_ParCSRMatrixColMapOffd(AN) = col_map_offd_AN;
   nalu_hypre_ParCSRMatrixCommPkg(AN) = comm_pkg_AN;

   *AN_ptr = AN;

   return nalu_hypre_error_flag;
}
