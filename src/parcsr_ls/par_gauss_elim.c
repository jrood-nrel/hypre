/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "Common.h"
#include "_nalu_hypre_blas.h"
#include "_nalu_hypre_lapack.h"

/*-------------------------------------------------------------------------
 *
 *                      Gaussian Elimination
 *
 *------------------------------------------------------------------------ */

NALU_HYPRE_Int nalu_hypre_GaussElimSetup (nalu_hypre_ParAMGData *amg_data, NALU_HYPRE_Int level, NALU_HYPRE_Int relax_type)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_GS_ELIM_SETUP] -= nalu_hypre_MPI_Wtime();
#endif

   nalu_hypre_GpuProfilingPushRange("GaussElimSetup");

   /* Par Data Structure variables */
   nalu_hypre_ParCSRMatrix *A      = nalu_hypre_ParAMGDataAArray(amg_data)[level];
   nalu_hypre_CSRMatrix    *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix    *A_offd = nalu_hypre_ParCSRMatrixOffd(A);

   NALU_HYPRE_Int num_rows = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int global_num_rows = (NALU_HYPRE_Int) nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(A);
   MPI_Comm new_comm;

   /* Generate sub communicator: processes that have nonzero num_rows */
   nalu_hypre_GenerateSubComm(comm, num_rows, &new_comm);

   if (num_rows)
   {
      nalu_hypre_CSRMatrix *A_diag_host, *A_offd_host;
      if (nalu_hypre_GetActualMemLocation(nalu_hypre_CSRMatrixMemoryLocation(A_diag)) != nalu_hypre_MEMORY_HOST)
      {
         A_diag_host = nalu_hypre_CSRMatrixClone_v2(A_diag, 1, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         A_diag_host = A_diag;
      }
      if (nalu_hypre_GetActualMemLocation(nalu_hypre_CSRMatrixMemoryLocation(A_offd)) != nalu_hypre_MEMORY_HOST)
      {
         A_offd_host = nalu_hypre_CSRMatrixClone_v2(A_offd, 1, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         A_offd_host = A_offd;
      }

      NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(A);
      NALU_HYPRE_Int  *A_diag_i    = nalu_hypre_CSRMatrixI(A_diag_host);
      NALU_HYPRE_Int  *A_offd_i    = nalu_hypre_CSRMatrixI(A_offd_host);
      NALU_HYPRE_Int  *A_diag_j    = nalu_hypre_CSRMatrixJ(A_diag_host);
      NALU_HYPRE_Int  *A_offd_j    = nalu_hypre_CSRMatrixJ(A_offd_host);
      NALU_HYPRE_Real *A_diag_data = nalu_hypre_CSRMatrixData(A_diag_host);
      NALU_HYPRE_Real *A_offd_data = nalu_hypre_CSRMatrixData(A_offd_host);

      NALU_HYPRE_Real *A_mat, *A_mat_local;
      NALU_HYPRE_Int *comm_info, *info, *displs;
      NALU_HYPRE_Int *mat_info, *mat_displs;
      NALU_HYPRE_Int new_num_procs, A_mat_local_size, i, jj, column;
      NALU_HYPRE_BigInt first_row_index = nalu_hypre_ParCSRMatrixFirstRowIndex(A);

      nalu_hypre_MPI_Comm_size(new_comm, &new_num_procs);

      comm_info  = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 2 * new_num_procs + 1, NALU_HYPRE_MEMORY_HOST);
      mat_info   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_procs,     NALU_HYPRE_MEMORY_HOST);
      mat_displs = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_procs + 1,   NALU_HYPRE_MEMORY_HOST);
      info = &comm_info[0];
      displs = &comm_info[new_num_procs];

      nalu_hypre_MPI_Allgather(&num_rows, 1, NALU_HYPRE_MPI_INT, info, 1, NALU_HYPRE_MPI_INT, new_comm);

      displs[0] = 0;
      mat_displs[0] = 0;
      for (i = 0; i < new_num_procs; i++)
      {
         displs[i + 1] = displs[i] + info[i];
         mat_displs[i + 1] = global_num_rows * displs[i + 1];
         mat_info[i] = global_num_rows * info[i];
      }

      nalu_hypre_ParAMGDataBVec(amg_data) = nalu_hypre_CTAlloc(NALU_HYPRE_Real, global_num_rows, NALU_HYPRE_MEMORY_HOST);

      A_mat_local_size = global_num_rows * num_rows;
      A_mat_local = nalu_hypre_CTAlloc(NALU_HYPRE_Real, A_mat_local_size,                NALU_HYPRE_MEMORY_HOST);
      A_mat       = nalu_hypre_CTAlloc(NALU_HYPRE_Real, global_num_rows * global_num_rows, NALU_HYPRE_MEMORY_HOST);

      /* load local matrix into A_mat_local */
      for (i = 0; i < num_rows; i++)
      {
         for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
         {
            /* need col major */
            column = A_diag_j[jj] + first_row_index;
            A_mat_local[i * global_num_rows + column] = A_diag_data[jj];
         }
         for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
         {
            /* need col major */
            column = col_map_offd[A_offd_j[jj]];
            A_mat_local[i * global_num_rows + column] = A_offd_data[jj];
         }
      }

      nalu_hypre_MPI_Allgatherv(A_mat_local, A_mat_local_size, NALU_HYPRE_MPI_REAL, A_mat, mat_info,
                           mat_displs, NALU_HYPRE_MPI_REAL, new_comm);

      if (relax_type == 99)
      {
         NALU_HYPRE_Real *AT_mat = nalu_hypre_CTAlloc(NALU_HYPRE_Real, global_num_rows * global_num_rows,
                                            NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < global_num_rows; i++)
         {
            for (jj = 0; jj < global_num_rows; jj++)
            {
               AT_mat[i * global_num_rows + jj] = A_mat[i + jj * global_num_rows];
            }
         }
         nalu_hypre_ParAMGDataAMat(amg_data) = AT_mat;
         nalu_hypre_TFree(A_mat, NALU_HYPRE_MEMORY_HOST);
      }
      else if (relax_type == 9)
      {
         nalu_hypre_ParAMGDataAMat(amg_data) = A_mat;
      }
      else if (relax_type == 199)
      {
         NALU_HYPRE_Real *AT_mat = nalu_hypre_TAlloc(NALU_HYPRE_Real, global_num_rows * global_num_rows, NALU_HYPRE_MEMORY_HOST);
         NALU_HYPRE_Real *Ainv   = nalu_hypre_TAlloc(NALU_HYPRE_Real, num_rows * global_num_rows,        NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < global_num_rows; i++)
         {
            for (jj = 0; jj < global_num_rows; jj++)
            {
               AT_mat[i * global_num_rows + jj] = A_mat[i + jj * global_num_rows];
            }
         }
         NALU_HYPRE_Int *ipiv, info, query = -1, lwork;
         NALU_HYPRE_Real lwork_opt, *work;
         ipiv = nalu_hypre_TAlloc(NALU_HYPRE_Int, global_num_rows, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_dgetrf(&global_num_rows, &global_num_rows, AT_mat, &global_num_rows, ipiv, &info);
         nalu_hypre_assert(info == 0);
         nalu_hypre_dgetri(&global_num_rows, AT_mat, &global_num_rows, ipiv, &lwork_opt, &query, &info);
         nalu_hypre_assert(info == 0);
         lwork = (NALU_HYPRE_Int)lwork_opt;
         work = nalu_hypre_TAlloc(NALU_HYPRE_Real, lwork, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_dgetri(&global_num_rows, AT_mat, &global_num_rows, ipiv, work, &lwork, &info);
         nalu_hypre_assert(info == 0);

         for (i = 0; i < global_num_rows; i++)
         {
            for (jj = 0; jj < num_rows; jj++)
            {
               Ainv[i * num_rows + jj] = AT_mat[i * global_num_rows + jj + first_row_index];
            }
         }

         nalu_hypre_TFree(ipiv,   NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(A_mat,  NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(AT_mat, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(work,   NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_ParAMGDataAInv(amg_data) = Ainv;
      }

      nalu_hypre_ParAMGDataCommInfo(amg_data) = comm_info;
      nalu_hypre_ParAMGDataNewComm(amg_data)  = new_comm;

      nalu_hypre_TFree(mat_info,    NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(mat_displs,  NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(A_mat_local, NALU_HYPRE_MEMORY_HOST);

      if (A_diag_host != A_diag)
      {
         nalu_hypre_CSRMatrixDestroy(A_diag_host);
      }

      if (A_offd_host != A_offd)
      {
         nalu_hypre_CSRMatrixDestroy(A_offd_host);
      }
   }

   nalu_hypre_ParAMGDataGSSetup(amg_data) = 1;

   nalu_hypre_GpuProfilingPopRange();

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_GS_ELIM_SETUP] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

/* relax_type = 9, 99, 199, see par_relax.c for 19 and 98 */
NALU_HYPRE_Int nalu_hypre_GaussElimSolve (nalu_hypre_ParAMGData *amg_data, NALU_HYPRE_Int level, NALU_HYPRE_Int relax_type)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_GS_ELIM_SOLVE] -= nalu_hypre_MPI_Wtime();
#endif

   nalu_hypre_GpuProfilingPushRange("GaussElimSolve");

   nalu_hypre_ParCSRMatrix *A = nalu_hypre_ParAMGDataAArray(amg_data)[level];
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int n = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int error_flag = 0;

   if (nalu_hypre_ParAMGDataGSSetup(amg_data) == 0)
   {
      nalu_hypre_GaussElimSetup(amg_data, level, relax_type);
   }

   if (n)
   {
      MPI_Comm new_comm = nalu_hypre_ParAMGDataNewComm(amg_data);
      nalu_hypre_ParVector *f = nalu_hypre_ParAMGDataFArray(amg_data)[level];
      nalu_hypre_ParVector *u = nalu_hypre_ParAMGDataUArray(amg_data)[level];
      NALU_HYPRE_Real *b_vec  = nalu_hypre_ParAMGDataBVec(amg_data);
      NALU_HYPRE_Real *f_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(f));
      NALU_HYPRE_Real *u_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(u));
      NALU_HYPRE_Int *comm_info = nalu_hypre_ParAMGDataCommInfo(amg_data);
      NALU_HYPRE_Int *displs, *info;
      NALU_HYPRE_Int n_global = (NALU_HYPRE_Int) nalu_hypre_ParCSRMatrixGlobalNumRows(A);
      NALU_HYPRE_Int new_num_procs;
      NALU_HYPRE_Int first_row_index = (NALU_HYPRE_Int) nalu_hypre_ParCSRMatrixFirstRowIndex(A);
      NALU_HYPRE_Int one_i = 1;

      nalu_hypre_MPI_Comm_size(new_comm, &new_num_procs);
      info = &comm_info[0];
      displs = &comm_info[new_num_procs];

      NALU_HYPRE_Real *f_data_host, *u_data_host;

      if (nalu_hypre_GetActualMemLocation(nalu_hypre_ParVectorMemoryLocation(f)) != nalu_hypre_MEMORY_HOST)
      {
         f_data_host = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_TMemcpy(f_data_host, f_data, NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST,
                       nalu_hypre_ParVectorMemoryLocation(f));
      }
      else
      {
         f_data_host = f_data;
      }

      if (nalu_hypre_GetActualMemLocation(nalu_hypre_ParVectorMemoryLocation(u)) != nalu_hypre_MEMORY_HOST)
      {
         u_data_host = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         u_data_host = u_data;
      }

      nalu_hypre_MPI_Allgatherv (f_data_host, n, NALU_HYPRE_MPI_REAL, b_vec, info,
                            displs, NALU_HYPRE_MPI_REAL, new_comm);

      if (f_data_host != f_data)
      {
         nalu_hypre_TFree(f_data_host, NALU_HYPRE_MEMORY_HOST);
      }

      if (relax_type == 9 || relax_type == 99)
      {
         NALU_HYPRE_Real *A_mat = nalu_hypre_ParAMGDataAMat(amg_data);
         NALU_HYPRE_Real *A_tmp;
         NALU_HYPRE_Int   i, my_info;

         A_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n_global * n_global, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < n_global * n_global; i++)
         {
            A_tmp[i] = A_mat[i];
         }

         if (relax_type == 9)
         {
            nalu_hypre_gselim(A_tmp, b_vec, n_global, error_flag);
         }
         else if (relax_type == 99) /* use pivoting */
         {
            NALU_HYPRE_Int *piv = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_global, NALU_HYPRE_MEMORY_HOST);

            /* write over A with LU */
            nalu_hypre_dgetrf(&n_global, &n_global, A_tmp, &n_global, piv, &my_info);

            /* now b_vec = inv(A)*b_vec */
            nalu_hypre_dgetrs("N", &n_global, &one_i, A_tmp, &n_global, piv, b_vec, &n_global, &my_info);

            nalu_hypre_TFree(piv, NALU_HYPRE_MEMORY_HOST);
         }

         for (i = 0; i < n; i++)
         {
            u_data_host[i] = b_vec[first_row_index + i];
         }

         nalu_hypre_TFree(A_tmp, NALU_HYPRE_MEMORY_HOST);
      }
      else if (relax_type == 199)
      {
         NALU_HYPRE_Real *Ainv = nalu_hypre_ParAMGDataAInv(amg_data);

         char cN = 'N';
         NALU_HYPRE_Real one = 1.0, zero = 0.0;
         nalu_hypre_dgemv(&cN, &n, &n_global, &one, Ainv, &n, b_vec, &one_i, &zero, u_data_host, &one_i);
      }

      if (u_data_host != u_data)
      {
         nalu_hypre_TMemcpy(u_data, u_data_host, NALU_HYPRE_Real, n, nalu_hypre_ParVectorMemoryLocation(u),
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(u_data_host, NALU_HYPRE_MEMORY_HOST);
      }
   }

   if (error_flag)
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_GS_ELIM_SOLVE] += nalu_hypre_MPI_Wtime();
#endif

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}































#if 0
#include "NALU_HYPRE_config.h"
#ifndef NALU_HYPRE_SEQUENTIAL
#define NALU_HYPRE_SEQUENTIAL
#endif
#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_blas.h"

#if defined(NALU_HYPRE_USING_GPU)

#define BLOCK_SIZE 512

__global__ void
hypreGPUKernel_dgemv(nalu_hypre_DeviceItem &item,
                     NALU_HYPRE_Int   m,
                     NALU_HYPRE_Int   n,
                     NALU_HYPRE_Int   lda,
                     NALU_HYPRE_Real *a,
                     NALU_HYPRE_Real *x,
                     NALU_HYPRE_Real *y)
{
   __shared__ NALU_HYPRE_Real sh_x[BLOCK_SIZE];

   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);
   NALU_HYPRE_Int tid = nalu_hypre_gpu_get_thread_id<1>(item);

   NALU_HYPRE_Real y_row = 0.0;

   for (NALU_HYPRE_Int k = 0; k < n; k += BLOCK_SIZE)
   {
      if (k + tid < n)
      {
         sh_x[tid] = read_only_load(&x[k + tid]);
      }

      __syncthreads();

      if (row < m)
      {
#pragma unroll
         for (NALU_HYPRE_Int j = 0; j < BLOCK_SIZE; j++)
         {
            const NALU_HYPRE_Int col = k + j;
            if (col < n)
            {
               y_row += a[row + col * lda] * sh_x[j];
            }
         }
      }

      __syncthreads();
   }

   if (row < m)
   {
      y[row] = y_row;
   }
}

NALU_HYPRE_Int nalu_hypre_dgemv_device(NALU_HYPRE_Int m, NALU_HYPRE_Int n, NALU_HYPRE_Int lda, NALU_HYPRE_Real *a, NALU_HYPRE_Real *x,
                             NALU_HYPRE_Real *y)
{
   dim3 bDim(BLOCK_SIZE, 1, 1);
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(m, "thread", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_dgemv, gDim, bDim, m, n, lda, a, x, y );

   return nalu_hypre_error_flag;
}

#endif // defined(NALU_HYPRE_USING_GPU)
#endif
