/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILUDevice
 *
 * ILU(0), ILUK, ILUT setup on the device
 *
 * Arguments:
 *    A = input matrix
 *    perm_data  = permutation array indicating ordering of rows.
 *                 Could come from a CF_marker array or a reordering routine.
 *    qperm_data = permutation array indicating ordering of columns
 *    nI  = number of internal unknowns
 *    nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *          Schur complement is formed if nLU < n
 *
 * This function will form the global Schur Matrix if nLU < n
 *
 * TODO (VPM): Change this function's name
 *             Change type of ilu_type to char ("0", "K", "T")
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupILUDevice(NALU_HYPRE_Int               ilu_type,
                        nalu_hypre_ParCSRMatrix     *A,
                        NALU_HYPRE_Int               lfil,
                        NALU_HYPRE_Real             *tol,
                        NALU_HYPRE_Int              *perm_data,
                        NALU_HYPRE_Int              *qperm_data,
                        NALU_HYPRE_Int               n,
                        NALU_HYPRE_Int               nLU,
                        nalu_hypre_CSRMatrix       **BLUptr,
                        nalu_hypre_ParCSRMatrix    **matSptr,
                        nalu_hypre_CSRMatrix       **Eptr,
                        nalu_hypre_CSRMatrix       **Fptr,
                        NALU_HYPRE_Int               tri_solve)
{
   /* Input matrix data */
   MPI_Comm                 comm                = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_MemoryLocation     memory_location     = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   nalu_hypre_ParCSRMatrix      *matS                = NULL;
   nalu_hypre_CSRMatrix         *A_diag              = NULL;
   nalu_hypre_CSRMatrix         *A_offd              = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_CSRMatrix         *h_A_offd            = NULL;
   NALU_HYPRE_Int               *A_offd_i            = NULL;
   NALU_HYPRE_Int               *A_offd_j            = NULL;
   NALU_HYPRE_Real              *A_offd_data         = NULL;
   nalu_hypre_CSRMatrix         *SLU                 = NULL;

   /* Permutation arrays */
   NALU_HYPRE_Int               *rperm_data          = NULL;
   NALU_HYPRE_Int               *rqperm_data         = NULL;
   nalu_hypre_IntArray          *perm                = NULL;
   nalu_hypre_IntArray          *rperm               = NULL;
   nalu_hypre_IntArray          *qperm               = NULL;
   nalu_hypre_IntArray          *rqperm              = NULL;
   nalu_hypre_IntArray          *h_perm              = NULL;
   nalu_hypre_IntArray          *h_rperm             = NULL;

   /* Variables for matS */
   NALU_HYPRE_Int                m                   = n - nLU;
   NALU_HYPRE_Int                nI                  = nLU; //use default
   NALU_HYPRE_Int                e                   = 0;
   NALU_HYPRE_Int                m_e                 = m;
   NALU_HYPRE_Int               *S_diag_i            = NULL;
   nalu_hypre_CSRMatrix         *S_offd              = NULL;
   NALU_HYPRE_Int               *S_offd_i            = NULL;
   NALU_HYPRE_Int               *S_offd_j            = NULL;
   NALU_HYPRE_Real              *S_offd_data         = NULL;
   NALU_HYPRE_BigInt            *S_offd_colmap       = NULL;
   NALU_HYPRE_Int                S_offd_nnz;
   NALU_HYPRE_Int                S_offd_ncols;
   NALU_HYPRE_Int                S_diag_nnz;

   nalu_hypre_ParCSRMatrix      *Apq                 = NULL;
   nalu_hypre_ParCSRMatrix      *ALU                 = NULL;
   nalu_hypre_ParCSRMatrix      *parL                = NULL;
   nalu_hypre_ParCSRMatrix      *parU                = NULL;
   nalu_hypre_ParCSRMatrix      *parS                = NULL;
   NALU_HYPRE_Real              *parD                = NULL;
   NALU_HYPRE_Int               *uend                = NULL;

   /* Local variables */
   NALU_HYPRE_BigInt            *send_buf            = NULL;
   nalu_hypre_ParCSRCommPkg     *comm_pkg;
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   NALU_HYPRE_Int                num_sends, begin, end;
   NALU_HYPRE_BigInt             total_rows, col_starts[2];
   NALU_HYPRE_Int                i, j, k1, k2, k3, col;
   NALU_HYPRE_Int                my_id, num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* Build the inverse permutation arrays */
   if (perm_data && qperm_data)
   {
      /* Create arrays */
      perm   = nalu_hypre_IntArrayCreate(n);
      qperm  = nalu_hypre_IntArrayCreate(n);

      /* Set existing data */
      nalu_hypre_IntArrayData(perm)  = perm_data;
      nalu_hypre_IntArrayData(qperm) = qperm_data;

      /* Initialize arrays */
      nalu_hypre_IntArrayInitialize_v2(perm, memory_location);
      nalu_hypre_IntArrayInitialize_v2(qperm, memory_location);

      /* Compute inverse permutation arrays */
      nalu_hypre_IntArrayInverseMapping(perm, &rperm);
      nalu_hypre_IntArrayInverseMapping(qperm, &rqperm);

      rqperm_data = nalu_hypre_IntArrayData(rqperm);
   }

   /* Only call ILU when we really have a matrix on this processor */
   if (n > 0)
   {
      /*
       * Apply ILU factorization to the entire A_diag
       *
       * | L \ U (B) L^{-1}F  |
       * | EU^{-1}   L \ U (S)|
       *
       * Extract submatrix L_B U_B, L_S U_S, EU_B^{-1}, L_B^{-1}F
       * Note that in this function after ILU, all rows are sorted
       * in a way different than HYPRE. Diagonal is not listed in the front
       */

      if (ilu_type == 0)
      {
         /* Copy diagonal matrix into a new place with permutation
          * That is, A_diag = A_diag(perm,qperm); */
         nalu_hypre_CSRMatrixPermute(nalu_hypre_ParCSRMatrixDiag(A), perm_data, rqperm_data, &A_diag);

         /* Compute ILU0 on the device */
         nalu_hypre_CSRMatrixILU0(A_diag);

         nalu_hypre_ParILUExtractEBFC(A_diag, nLU, BLUptr, &SLU, Eptr, Fptr);
         nalu_hypre_CSRMatrixDestroy(A_diag);
      }
      else
      {
         nalu_hypre_ParILURAPReorder(A, perm_data, rqperm_data, &Apq);
         if (ilu_type == 1)
         {
            nalu_hypre_ILUSetupILUK(Apq, lfil, NULL, NULL, n, n, &parL, &parD, &parU, &parS, &uend);
         }
         else if (ilu_type == 2)
         {
            nalu_hypre_ILUSetupILUT(Apq, lfil, tol, NULL, NULL, n, n,
                               &parL, &parD, &parU, &parS, &uend);
         }

         nalu_hypre_ParCSRMatrixDestroy(Apq);
         nalu_hypre_TFree(uend, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParCSRMatrixDestroy(parS);

         nalu_hypre_ILUSetupLDUtoCusparse(parL, parD, parU, &ALU);

         nalu_hypre_ParCSRMatrixDestroy(parL);
         nalu_hypre_ParCSRMatrixDestroy(parU);
         nalu_hypre_TFree(parD, NALU_HYPRE_MEMORY_DEVICE);

         nalu_hypre_ParILUExtractEBFC(nalu_hypre_ParCSRMatrixDiag(ALU), nLU,
                                 BLUptr, &SLU, Eptr, Fptr);

         nalu_hypre_ParCSRMatrixDestroy(ALU);
      }
   }
   else
   {
      *BLUptr = NULL;
      *Eptr = NULL;
      *Fptr = NULL;
      SLU = NULL;
   }

   /* Compute total rows in Schur block */
   NALU_HYPRE_BigInt big_m = (NALU_HYPRE_BigInt) m;
   nalu_hypre_MPI_Allreduce(&big_m, &total_rows, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

   /* only form when total_rows > 0 */
   if (total_rows > 0)
   {
      /* now create S - need to get new column start */
      {
         NALU_HYPRE_BigInt global_start;
         nalu_hypre_MPI_Scan(&big_m, &global_start, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      if (!SLU)
      {
         SLU = nalu_hypre_CSRMatrixCreate(0, 0, 0);
         nalu_hypre_CSRMatrixInitialize(SLU);
      }

      S_diag_i = nalu_hypre_CSRMatrixI(SLU);
      nalu_hypre_TMemcpy(&S_diag_nnz, S_diag_i + m, NALU_HYPRE_Int, 1,
                    NALU_HYPRE_MEMORY_HOST, nalu_hypre_CSRMatrixMemoryLocation(SLU));

      /* Build ParCSRMatrix matS
       * For example when np == 3 the new matrix takes the following form
       * |IS_1 E_12 E_13|
       * |E_21 IS_2 E_22| = S
       * |E_31 E_32 IS_3|
       * In which IS_i is the cusparse ILU factorization of S_i in one matrix
       * */

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = nalu_hypre_CSRMatrixNumCols(A_offd);

      matS = nalu_hypre_ParCSRMatrixCreate(comm,
                                      total_rows,
                                      total_rows,
                                      col_starts,
                                      col_starts,
                                      S_offd_ncols,
                                      S_diag_nnz,
                                      S_offd_nnz);

      /* first put diagonal data in */
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(matS));
      nalu_hypre_ParCSRMatrixDiag(matS) = SLU;

      /* now start to construct offdiag of S */
      S_offd = nalu_hypre_ParCSRMatrixOffd(matS);
      nalu_hypre_CSRMatrixInitialize_v2(S_offd, 0, NALU_HYPRE_MEMORY_HOST);
      S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
      S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);
      S_offd_data = nalu_hypre_CSRMatrixData(S_offd);
      S_offd_colmap = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, S_offd_ncols, NALU_HYPRE_MEMORY_HOST);

      /* Set/Move A_offd to host */
      h_A_offd = (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE) ?
                 nalu_hypre_CSRMatrixClone_v2(A_offd, 1, NALU_HYPRE_MEMORY_HOST) : A_offd;
      A_offd_i    = nalu_hypre_CSRMatrixI(h_A_offd);
      A_offd_j    = nalu_hypre_CSRMatrixJ(h_A_offd);
      A_offd_data = nalu_hypre_CSRMatrixData(h_A_offd);

      /* Clone permutation arrays on the host */
      if (rperm && perm)
      {
         h_perm  = nalu_hypre_IntArrayCloneDeep_v2(perm, NALU_HYPRE_MEMORY_HOST);
         h_rperm = nalu_hypre_IntArrayCloneDeep_v2(rperm, NALU_HYPRE_MEMORY_HOST);

         perm_data  = nalu_hypre_IntArrayData(h_perm);
         rperm_data = nalu_hypre_IntArrayData(h_rperm);
      }

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = (perm_data) ? perm_data[i + nI] : i + nI;
         k1 = A_offd_i[col];
         k2 = A_offd_i[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i + 1 + e] = k3;
      }

      /* give I, J, DATA to S_offd */
      nalu_hypre_CSRMatrixI(S_offd) = S_offd_i;
      nalu_hypre_CSRMatrixJ(S_offd) = S_offd_j;
      nalu_hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);

      /* setup comm_pkg if not yet built */
      if (!comm_pkg)
      {
         nalu_hypre_MatvecCommPkgCreate(A);
         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
      }

      /* get total num of send */
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      end = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_buf = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, end - begin, NALU_HYPRE_MEMORY_HOST);

      /* copy new index into send_buf */
      for (i = 0; i < (end - begin); i++)
      {
         send_buf[i] = (rperm_data) ?
                       rperm_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i + begin)] -
                       nLU + col_starts[0] :
                       nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i + begin) -
                       nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      nalu_hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      nalu_hypre_ILUSortOffdColmap(matS);

      /* Move S_offd to final memory location */
      nalu_hypre_CSRMatrixMigrate(S_offd, memory_location);

      /* Free memory */
      nalu_hypre_TFree(send_buf, NALU_HYPRE_MEMORY_HOST);
      if (h_A_offd != A_offd)
      {
         nalu_hypre_CSRMatrixDestroy(h_A_offd);
      }
   } /* end of forming S */
   else
   {
      nalu_hypre_CSRMatrixDestroy(SLU);
   }

   /* Set output pointer */
   *matSptr = matS;

   /* Do not free perm_data/qperm_data */
   if (perm)
   {
      nalu_hypre_IntArrayData(perm)  = NULL;
   }
   if (qperm)
   {
      nalu_hypre_IntArrayData(qperm) = NULL;
   }

   /* Free memory */
   nalu_hypre_IntArrayDestroy(perm);
   nalu_hypre_IntArrayDestroy(qperm);
   nalu_hypre_IntArrayDestroy(rperm);
   nalu_hypre_IntArrayDestroy(rqperm);
   nalu_hypre_IntArrayDestroy(h_perm);
   nalu_hypre_IntArrayDestroy(h_rperm);

   return nalu_hypre_error_flag;
}

#endif /* defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */
