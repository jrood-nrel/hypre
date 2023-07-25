/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_amg.h"


#define USE_ALLTOALL 0

/* here we have the sequential setup and solve - called from the
 * parallel one - for the coarser levels */

NALU_HYPRE_Int
nalu_hypre_seqAMGSetup( nalu_hypre_ParAMGData *amg_data,
                   NALU_HYPRE_Int         p_level,
                   NALU_HYPRE_Int         coarse_threshold)
{

   /* Par Data Structure variables */
   nalu_hypre_ParCSRMatrix **Par_A_array = nalu_hypre_ParAMGDataAArray(amg_data);

   MPI_Comm      comm = nalu_hypre_ParCSRMatrixComm(Par_A_array[0]);
   MPI_Comm      new_comm, seq_comm;

   nalu_hypre_ParCSRMatrix   *A_seq = NULL;
   nalu_hypre_CSRMatrix  *A_seq_diag;
   nalu_hypre_CSRMatrix  *A_seq_offd;
   nalu_hypre_ParVector   *F_seq = NULL;
   nalu_hypre_ParVector   *U_seq = NULL;

   nalu_hypre_ParCSRMatrix *A;

   nalu_hypre_IntArray         **dof_func_array;
   NALU_HYPRE_Int                num_procs, my_id;

   NALU_HYPRE_Int                level;
   NALU_HYPRE_Int                redundant;
   NALU_HYPRE_Int                num_functions;

   NALU_HYPRE_Solver  coarse_solver;

   /* misc */
   dof_func_array = nalu_hypre_ParAMGDataDofFuncArray(amg_data);
   num_functions = nalu_hypre_ParAMGDataNumFunctions(amg_data);
   redundant = nalu_hypre_ParAMGDataRedundant(amg_data);

   /*MPI Stuff */
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /*initial */
   level = p_level;

   /* convert A at this level to sequential */
   A = Par_A_array[level];

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   {
      NALU_HYPRE_Real *A_seq_data = NULL;
      NALU_HYPRE_Int *A_seq_i = NULL;
      NALU_HYPRE_Int *A_seq_offd_i = NULL;
      NALU_HYPRE_Int *A_seq_j = NULL;
      NALU_HYPRE_Int *seq_dof_func = NULL;

      NALU_HYPRE_Real *A_tmp_data = NULL;
      NALU_HYPRE_Int *A_tmp_i = NULL;
      NALU_HYPRE_Int *A_tmp_j = NULL;

      NALU_HYPRE_Int *info = NULL;
      NALU_HYPRE_Int *displs = NULL;
      NALU_HYPRE_Int *displs2 = NULL;
      NALU_HYPRE_Int i, j, size, num_nonzeros, total_nnz, cnt;

      nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
      nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
      NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(A);
      NALU_HYPRE_Int *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
      NALU_HYPRE_Int *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
      NALU_HYPRE_Int *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
      NALU_HYPRE_Int *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
      NALU_HYPRE_Real *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
      NALU_HYPRE_Real *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
      NALU_HYPRE_Int num_rows = nalu_hypre_CSRMatrixNumRows(A_diag);
      NALU_HYPRE_BigInt first_row_index = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
      NALU_HYPRE_Int new_num_procs;
      NALU_HYPRE_BigInt  row_starts[2];

      nalu_hypre_GenerateSubComm(comm, num_rows, &new_comm);


      /*nalu_hypre_MPI_Group orig_group, new_group;
      NALU_HYPRE_Int *ranks, new_num_procs, *row_starts;

      info = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_MPI_Allgather(&num_rows, 1, NALU_HYPRE_MPI_INT, info, 1, NALU_HYPRE_MPI_INT, comm);

      ranks = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);

      new_num_procs = 0;
      for (i=0; i < num_procs; i++)
         if (info[i])
         {
            ranks[new_num_procs] = i;
            info[new_num_procs++] = info[i];
         }

      nalu_hypre_MPI_Comm_group(comm, &orig_group);
      nalu_hypre_MPI_Group_incl(orig_group, new_num_procs, ranks, &new_group);
      nalu_hypre_MPI_Comm_create(comm, new_group, &new_comm);
      nalu_hypre_MPI_Group_free(&new_group);
      nalu_hypre_MPI_Group_free(&orig_group); */

      if (num_rows)
      {
         nalu_hypre_ParAMGDataParticipate(amg_data) = 1;
         nalu_hypre_MPI_Comm_size(new_comm, &new_num_procs);
         nalu_hypre_MPI_Comm_rank(new_comm, &my_id);
         info = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  new_num_procs, NALU_HYPRE_MEMORY_HOST);

         if (redundant)
         {
            nalu_hypre_MPI_Allgather(&num_rows, 1, NALU_HYPRE_MPI_INT, info, 1, NALU_HYPRE_MPI_INT, new_comm);
         }
         else
         {
            nalu_hypre_MPI_Gather(&num_rows, 1, NALU_HYPRE_MPI_INT, info, 1, NALU_HYPRE_MPI_INT, 0, new_comm);
         }

         /* alloc space in seq data structure only for participating procs*/
         if (redundant || my_id == 0)
         {
            NALU_HYPRE_BoomerAMGCreate(&coarse_solver);
            NALU_HYPRE_BoomerAMGSetMaxRowSum(coarse_solver,
                                        nalu_hypre_ParAMGDataMaxRowSum(amg_data));
            NALU_HYPRE_BoomerAMGSetStrongThreshold(coarse_solver,
                                              nalu_hypre_ParAMGDataStrongThreshold(amg_data));
            NALU_HYPRE_BoomerAMGSetCoarsenType(coarse_solver,
                                          nalu_hypre_ParAMGDataCoarsenType(amg_data));
            NALU_HYPRE_BoomerAMGSetInterpType(coarse_solver,
                                         nalu_hypre_ParAMGDataInterpType(amg_data));
            NALU_HYPRE_BoomerAMGSetTruncFactor(coarse_solver,
                                          nalu_hypre_ParAMGDataTruncFactor(amg_data));
            NALU_HYPRE_BoomerAMGSetPMaxElmts(coarse_solver,
                                        nalu_hypre_ParAMGDataPMaxElmts(amg_data));
            if (nalu_hypre_ParAMGDataUserRelaxType(amg_data) > -1)
               NALU_HYPRE_BoomerAMGSetRelaxType(coarse_solver,
                                           nalu_hypre_ParAMGDataUserRelaxType(amg_data));
            NALU_HYPRE_BoomerAMGSetRelaxOrder(coarse_solver,
                                         nalu_hypre_ParAMGDataRelaxOrder(amg_data));
            NALU_HYPRE_BoomerAMGSetRelaxWt(coarse_solver,
                                      nalu_hypre_ParAMGDataUserRelaxWeight(amg_data));
            if (nalu_hypre_ParAMGDataUserNumSweeps(amg_data) > -1)
               NALU_HYPRE_BoomerAMGSetNumSweeps(coarse_solver,
                                           nalu_hypre_ParAMGDataUserNumSweeps(amg_data));
            NALU_HYPRE_BoomerAMGSetNumFunctions(coarse_solver,
                                           num_functions);
            NALU_HYPRE_BoomerAMGSetMaxIter(coarse_solver, 1);
            NALU_HYPRE_BoomerAMGSetTol(coarse_solver, 0);
         }

         /* Create CSR Matrix, will be Diag part of new matrix */
         A_tmp_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows + 1, NALU_HYPRE_MEMORY_HOST);

         A_tmp_i[0] = 0;
         for (i = 1; i < num_rows + 1; i++)
         {
            A_tmp_i[i] = A_diag_i[i] - A_diag_i[i - 1] + A_offd_i[i] - A_offd_i[i - 1];
         }

         num_nonzeros = A_offd_i[num_rows] + A_diag_i[num_rows];

         A_tmp_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nonzeros, NALU_HYPRE_MEMORY_HOST);
         A_tmp_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_nonzeros, NALU_HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i = 0; i < num_rows; i++)
         {
            for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
            {
               A_tmp_j[cnt] = A_diag_j[j] + (NALU_HYPRE_Int)first_row_index;
               A_tmp_data[cnt++] = A_diag_data[j];
            }
            for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
            {
               A_tmp_j[cnt] = (NALU_HYPRE_Int)col_map_offd[A_offd_j[j]];
               A_tmp_data[cnt++] = A_offd_data[j];
            }
         }

         displs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  new_num_procs + 1, NALU_HYPRE_MEMORY_HOST);
         displs[0] = 0;
         for (i = 1; i < new_num_procs + 1; i++)
         {
            displs[i] = displs[i - 1] + info[i - 1];
         }
         size = displs[new_num_procs];

         if (redundant || my_id == 0)
         {
            A_seq_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size + 1, memory_location);
            A_seq_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size + 1, memory_location);
            if (num_functions > 1) { seq_dof_func = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size, memory_location); }
         }

         if (redundant)
         {
            nalu_hypre_MPI_Allgatherv ( &A_tmp_i[1], num_rows, NALU_HYPRE_MPI_INT, &A_seq_i[1], info,
                                   displs, NALU_HYPRE_MPI_INT, new_comm );
            if (num_functions > 1)
            {
               nalu_hypre_MPI_Allgatherv ( nalu_hypre_IntArrayData(dof_func_array[level]), num_rows, NALU_HYPRE_MPI_INT,
                                      seq_dof_func, info, displs, NALU_HYPRE_MPI_INT, new_comm );
               NALU_HYPRE_BoomerAMGSetDofFunc(coarse_solver, seq_dof_func);
            }
         }
         else
         {
            if (A_seq_i)
               nalu_hypre_MPI_Gatherv ( &A_tmp_i[1], num_rows, NALU_HYPRE_MPI_INT, &A_seq_i[1], info,
                                   displs, NALU_HYPRE_MPI_INT, 0, new_comm );
            else
               nalu_hypre_MPI_Gatherv ( &A_tmp_i[1], num_rows, NALU_HYPRE_MPI_INT, A_seq_i, info,
                                   displs, NALU_HYPRE_MPI_INT, 0, new_comm );
            if (num_functions > 1)
            {
               nalu_hypre_MPI_Gatherv ( nalu_hypre_IntArrayData(dof_func_array[level]), num_rows, NALU_HYPRE_MPI_INT,
                                   seq_dof_func, info, displs, NALU_HYPRE_MPI_INT, 0, new_comm );
               if (my_id == 0) { NALU_HYPRE_BoomerAMGSetDofFunc(coarse_solver, seq_dof_func); }
            }
         }

         if (redundant || my_id == 0)
         {
            displs2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  new_num_procs + 1, NALU_HYPRE_MEMORY_HOST);

            A_seq_i[0] = 0;
            displs2[0] = 0;
            for (j = 1; j < displs[1]; j++)
            {
               A_seq_i[j] = A_seq_i[j] + A_seq_i[j - 1];
            }
            for (i = 1; i < new_num_procs; i++)
            {
               for (j = displs[i]; j < displs[i + 1]; j++)
               {
                  A_seq_i[j] = A_seq_i[j] + A_seq_i[j - 1];
               }
            }
            A_seq_i[size] = A_seq_i[size] + A_seq_i[size - 1];
            displs2[new_num_procs] = A_seq_i[size];
            for (i = 1; i < new_num_procs + 1; i++)
            {
               displs2[i] = A_seq_i[displs[i]];
               info[i - 1] = displs2[i] - displs2[i - 1];
            }

            total_nnz = displs2[new_num_procs];
            A_seq_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  total_nnz, memory_location);
            A_seq_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  total_nnz, memory_location);
         }
         if (redundant)
         {
            nalu_hypre_MPI_Allgatherv ( A_tmp_j, num_nonzeros, NALU_HYPRE_MPI_INT,
                                   A_seq_j, info, displs2,
                                   NALU_HYPRE_MPI_INT, new_comm );

            nalu_hypre_MPI_Allgatherv ( A_tmp_data, num_nonzeros, NALU_HYPRE_MPI_REAL,
                                   A_seq_data, info, displs2,
                                   NALU_HYPRE_MPI_REAL, new_comm );
         }
         else
         {
            nalu_hypre_MPI_Gatherv ( A_tmp_j, num_nonzeros, NALU_HYPRE_MPI_INT,
                                A_seq_j, info, displs2,
                                NALU_HYPRE_MPI_INT, 0, new_comm );

            nalu_hypre_MPI_Gatherv ( A_tmp_data, num_nonzeros, NALU_HYPRE_MPI_REAL,
                                A_seq_data, info, displs2,
                                NALU_HYPRE_MPI_REAL, 0, new_comm );
         }

         nalu_hypre_TFree(info, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(displs, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(A_tmp_i, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(A_tmp_j, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(A_tmp_data, NALU_HYPRE_MEMORY_HOST);

         if (redundant || my_id == 0)
         {
            nalu_hypre_TFree(displs2, NALU_HYPRE_MEMORY_HOST);

            row_starts[0] = 0;
            row_starts[1] = size;

            /* Create 1 proc communicator */
            seq_comm = nalu_hypre_MPI_COMM_SELF;

            A_seq = nalu_hypre_ParCSRMatrixCreate(seq_comm, size, size,
                                             row_starts, row_starts,
                                             0, total_nnz, 0);

            A_seq_diag = nalu_hypre_ParCSRMatrixDiag(A_seq);
            A_seq_offd = nalu_hypre_ParCSRMatrixOffd(A_seq);

            nalu_hypre_CSRMatrixData(A_seq_diag) = A_seq_data;
            nalu_hypre_CSRMatrixI(A_seq_diag) = A_seq_i;
            nalu_hypre_CSRMatrixJ(A_seq_diag) = A_seq_j;
            nalu_hypre_CSRMatrixI(A_seq_offd) = A_seq_offd_i;

            F_seq = nalu_hypre_ParVectorCreate(seq_comm, size, row_starts);
            U_seq = nalu_hypre_ParVectorCreate(seq_comm, size, row_starts);
            nalu_hypre_ParVectorInitialize(F_seq);
            nalu_hypre_ParVectorInitialize(U_seq);

            nalu_hypre_BoomerAMGSetup(coarse_solver, A_seq, F_seq, U_seq);

            nalu_hypre_ParAMGDataCoarseSolver(amg_data) = coarse_solver;
            nalu_hypre_ParAMGDataACoarse(amg_data) = A_seq;
            nalu_hypre_ParAMGDataFCoarse(amg_data) = F_seq;
            nalu_hypre_ParAMGDataUCoarse(amg_data) = U_seq;
         }
         nalu_hypre_ParAMGDataNewComm(amg_data) = new_comm;
      }
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_seqAMGCycle
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_seqAMGCycle( nalu_hypre_ParAMGData *amg_data,
                   NALU_HYPRE_Int p_level,
                   nalu_hypre_ParVector  **Par_F_array,
                   nalu_hypre_ParVector  **Par_U_array   )
{

   nalu_hypre_ParVector    *Aux_U;
   nalu_hypre_ParVector    *Aux_F;

   /* Local variables  */

   NALU_HYPRE_Int       Solve_err_flag = 0;

   NALU_HYPRE_Int n;
   NALU_HYPRE_Int i;

   nalu_hypre_Vector   *u_local;
   NALU_HYPRE_Real     *u_data;

   NALU_HYPRE_Int       first_index;

   /* Acquire seq data */
   MPI_Comm new_comm = nalu_hypre_ParAMGDataNewComm(amg_data);
   NALU_HYPRE_Solver coarse_solver = nalu_hypre_ParAMGDataCoarseSolver(amg_data);
   nalu_hypre_ParCSRMatrix *A_coarse = nalu_hypre_ParAMGDataACoarse(amg_data);
   nalu_hypre_ParVector *F_coarse = nalu_hypre_ParAMGDataFCoarse(amg_data);
   nalu_hypre_ParVector *U_coarse = nalu_hypre_ParAMGDataUCoarse(amg_data);
   NALU_HYPRE_Int redundant = nalu_hypre_ParAMGDataRedundant(amg_data);

   Aux_U = Par_U_array[p_level];
   Aux_F = Par_F_array[p_level];

   first_index = (NALU_HYPRE_Int)nalu_hypre_ParVectorFirstIndex(Aux_U);
   u_local = nalu_hypre_ParVectorLocalVector(Aux_U);
   u_data  = nalu_hypre_VectorData(u_local);
   n =  nalu_hypre_VectorSize(u_local);


   /*if (A_coarse)*/
   if (nalu_hypre_ParAMGDataParticipate(amg_data))
   {
      NALU_HYPRE_Real     *f_data;
      nalu_hypre_Vector   *f_local;
      nalu_hypre_Vector   *tmp_vec;

      NALU_HYPRE_Int nf;
      NALU_HYPRE_Int local_info;
      NALU_HYPRE_Real *recv_buf = NULL;
      NALU_HYPRE_Int *displs = NULL;
      NALU_HYPRE_Int *info = NULL;
      NALU_HYPRE_Int new_num_procs, my_id;

      nalu_hypre_MPI_Comm_size(new_comm, &new_num_procs);
      nalu_hypre_MPI_Comm_rank(new_comm, &my_id);

      f_local = nalu_hypre_ParVectorLocalVector(Aux_F);
      f_data = nalu_hypre_VectorData(f_local);
      nf =  nalu_hypre_VectorSize(f_local);

      /* first f */
      info = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  new_num_procs, NALU_HYPRE_MEMORY_HOST);
      local_info = nf;
      if (redundant)
      {
         nalu_hypre_MPI_Allgather(&local_info, 1, NALU_HYPRE_MPI_INT, info, 1, NALU_HYPRE_MPI_INT, new_comm);
      }
      else
      {
         nalu_hypre_MPI_Gather(&local_info, 1, NALU_HYPRE_MPI_INT, info, 1, NALU_HYPRE_MPI_INT, 0, new_comm);
      }

      if (redundant || my_id == 0)
      {
         displs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  new_num_procs + 1, NALU_HYPRE_MEMORY_HOST);
         displs[0] = 0;
         for (i = 1; i < new_num_procs + 1; i++)
         {
            displs[i] = displs[i - 1] + info[i - 1];
         }

         if (F_coarse)
         {
            tmp_vec =  nalu_hypre_ParVectorLocalVector(F_coarse);
            recv_buf = nalu_hypre_VectorData(tmp_vec);
         }
      }

      if (redundant)
         nalu_hypre_MPI_Allgatherv ( f_data, nf, NALU_HYPRE_MPI_REAL,
                                recv_buf, info, displs,
                                NALU_HYPRE_MPI_REAL, new_comm );
      else
         nalu_hypre_MPI_Gatherv ( f_data, nf, NALU_HYPRE_MPI_REAL,
                             recv_buf, info, displs,
                             NALU_HYPRE_MPI_REAL, 0, new_comm );

      if (redundant || my_id == 0)
      {
         tmp_vec =  nalu_hypre_ParVectorLocalVector(U_coarse);
         recv_buf = nalu_hypre_VectorData(tmp_vec);
      }

      /*then u */
      if (redundant)
      {
         nalu_hypre_MPI_Allgatherv ( u_data, n, NALU_HYPRE_MPI_REAL,
                                recv_buf, info, displs,
                                NALU_HYPRE_MPI_REAL, new_comm );
         nalu_hypre_TFree(displs, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(info, NALU_HYPRE_MEMORY_HOST);
      }
      else
         nalu_hypre_MPI_Gatherv ( u_data, n, NALU_HYPRE_MPI_REAL,
                             recv_buf, info, displs,
                             NALU_HYPRE_MPI_REAL, 0, new_comm );

      /* clean up */
      if (redundant || my_id == 0)
      {
         nalu_hypre_BoomerAMGSolve(coarse_solver, A_coarse, F_coarse, U_coarse);
      }

      /*copy my part of U to parallel vector */
      if (redundant)
      {
         NALU_HYPRE_Real *local_data;

         local_data =  nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(U_coarse));

         for (i = 0; i < n; i++)
         {
            u_data[i] = local_data[first_index + i];
         }
      }
      else
      {
         NALU_HYPRE_Real *local_data = NULL;

         if (my_id == 0)
         {
            local_data =  nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(U_coarse));
         }

         nalu_hypre_MPI_Scatterv ( local_data, info, displs, NALU_HYPRE_MPI_REAL,
                              u_data, n, NALU_HYPRE_MPI_REAL, 0, new_comm );
         /*if (my_id == 0)
            local_data =  nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(F_coarse));
            nalu_hypre_MPI_Scatterv ( local_data, info, displs, NALU_HYPRE_MPI_REAL,
                       f_data, n, NALU_HYPRE_MPI_REAL, 0, new_comm );*/
         if (my_id == 0) { nalu_hypre_TFree(displs, NALU_HYPRE_MEMORY_HOST); }
         nalu_hypre_TFree(info, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return (Solve_err_flag);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GenerateSubComm
 *
 * generate sub communicator, which contains no idle processors
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_GenerateSubComm(MPI_Comm   comm,
                      NALU_HYPRE_Int  participate,
                      MPI_Comm  *new_comm_ptr)
{
   MPI_Comm          new_comm;
   nalu_hypre_MPI_Group   orig_group, new_group;
   nalu_hypre_MPI_Op      nalu_hypre_MPI_MERGE;
   NALU_HYPRE_Int        *info, *ranks, new_num_procs, my_info, my_id, num_procs;
   NALU_HYPRE_Int        *list_len;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (participate)
   {
      my_info = 1;
   }
   else
   {
      my_info = 0;
   }

   nalu_hypre_MPI_Allreduce(&my_info, &new_num_procs, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm);

   if (new_num_procs == 0)
   {
      new_comm = nalu_hypre_MPI_COMM_NULL;
      *new_comm_ptr = new_comm;

      return nalu_hypre_error_flag;
   }

   ranks = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_procs + 2, NALU_HYPRE_MEMORY_HOST);

   if (new_num_procs == 1)
   {
      if (participate)
      {
         my_info = my_id;
      }
      nalu_hypre_MPI_Allreduce(&my_info, &ranks[2], 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm);
   }
   else
   {
      info = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_procs + 2, NALU_HYPRE_MEMORY_HOST);
      list_len = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST);

      if (participate)
      {
         info[0] = 1;
         info[1] = 1;
         info[2] = my_id;
      }
      else
      {
         info[0] = 0;
      }

      list_len[0] = new_num_procs + 2;

      nalu_hypre_MPI_Op_create((nalu_hypre_MPI_User_function *)nalu_hypre_merge_lists, 0, &nalu_hypre_MPI_MERGE);

      nalu_hypre_MPI_Allreduce(info, ranks, list_len[0], NALU_HYPRE_MPI_INT, nalu_hypre_MPI_MERGE, comm);

      nalu_hypre_MPI_Op_free (&nalu_hypre_MPI_MERGE);

      nalu_hypre_TFree(list_len, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(info, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_group(comm, &orig_group);
   nalu_hypre_MPI_Group_incl(orig_group, new_num_procs, &ranks[2], &new_group);
   nalu_hypre_MPI_Comm_create(comm, new_group, &new_comm);
   nalu_hypre_MPI_Group_free(&new_group);
   nalu_hypre_MPI_Group_free(&orig_group);

   nalu_hypre_TFree(ranks, NALU_HYPRE_MEMORY_HOST);

   *new_comm_ptr = new_comm;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_merge_lists
 *--------------------------------------------------------------------------*/

void
nalu_hypre_merge_lists(NALU_HYPRE_Int          *list1,
                  NALU_HYPRE_Int          *list2,
                  nalu_hypre_int          *np1,
                  nalu_hypre_MPI_Datatype *dptr)
{
   NALU_HYPRE_Int i, len1, len2, indx1, indx2;

   if (list1[0] == 0)
   {
      return;
   }
   else
   {
      list2[0] = 1;
      len1 = list1[1];
      len2 = list2[1];
      list2[1] = len1 + len2;
      if ((nalu_hypre_int)(list2[1]) > *np1 + 2) // RL:???
      {
         printf("segfault in MPI User function merge_list\n");
      }
      indx1 = len1 + 1;
      indx2 = len2 + 1;
      for (i = len1 + len2 + 1; i > 1; i--)
      {
         if (indx2 > 1 && indx1 > 1 && list1[indx1] > list2[indx2])
         {
            list2[i] = list1[indx1];
            indx1--;
         }
         else if (indx2 > 1)
         {
            list2[i] = list2[indx2];
            indx2--;
         }
         else if (indx1 > 1)
         {
            list2[i] = list1[indx1];
            indx1--;
         }
      }
   }
}
