/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "Common.h"
#include "_nalu_hypre_lapack.h"
#include "par_relax.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax( nalu_hypre_ParCSRMatrix *A,
                      nalu_hypre_ParVector    *f,
                      NALU_HYPRE_Int          *cf_marker,
                      NALU_HYPRE_Int           relax_type,
                      NALU_HYPRE_Int           relax_points,
                      NALU_HYPRE_Real          relax_weight,
                      NALU_HYPRE_Real          omega,
                      NALU_HYPRE_Real         *l1_norms,
                      nalu_hypre_ParVector    *u,
                      nalu_hypre_ParVector    *Vtemp,
                      nalu_hypre_ParVector    *Ztemp )
{
   NALU_HYPRE_Int relax_error = 0;

   /*---------------------------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type =  0 -> Jacobi or CF-Jacobi
    *     relax_type =  1 -> Gauss-Seidel <--- very slow, sequential
    *     relax_type =  2 -> Gauss_Seidel: interior points in parallel,
    *                                      boundary sequential
    *     relax_type =  3 -> hybrid: SOR-J mix off-processor, SOR on-processor
    *                               with outer relaxation parameters (forward solve)
    *     relax_type =  4 -> hybrid: SOR-J mix off-processor, SOR on-processor
    *                               with outer relaxation parameters (backward solve)
    *     relax_type =  5 -> hybrid: GS-J mix off-processor, chaotic GS on-node
    *     relax_type =  6 -> hybrid: SSOR-J mix off-processor, SSOR on-processor
    *                               with outer relaxation parameters
    *     relax_type =  7 -> Jacobi (uses Matvec), only needed in CGNR
    *                        [GPU-supported, CF supported with redundant computation]
    *     relax_type =  8 -> hybrid L1 Symm. Gauss-Seidel (SSOR)
    *     relax_type =  9 -> Direct solve, Gaussian elimination
    *     relax_type = 10 -> On-processor direct forward solve for matrices with
    *                        triangular structure (indices need not be ordered
    *                        triangular)
    *     relax_type = 11 -> Two Stage approximation to GS. Uses the strict lower
    *                        part of the diagonal matrix
    *     relax_type = 12 -> Two Stage approximation to GS. Uses the strict lower
    *                        part of the diagonal matrix and a second iteration
    *                        for additional error approximation
    *     relax_type = 13 -> hybrid L1 Gauss-Seidel forward solve
    *     relax_type = 14 -> hybrid L1 Gauss-Seidel backward solve
    *     relax_type = 15 -> CG
    *     relax_type = 16 -> Scaled Chebyshev
    *     relax_type = 17 -> FCF-Jacobi
    *     relax_type = 18 -> L1-Jacobi [GPU-supported through call to relax7Jacobi]
    *     relax_type = 19 -> Direct Solve, (old version)
    *     relax_type = 20 -> Kaczmarz
    *     relax_type = 21 -> the same as 8 except forcing serialization on CPU (#OMP-thread = 1)
    *     relax_type = 29 -> Direct solve: use Gaussian elimination & BLAS
    *                        (with pivoting) (old version)
    *     relax_type = 98 -> Direct solve, Gaussian elimination
    *     relax_type = 99 -> Direct solve, Gaussian elimination
    *     relax_type = 199-> Direct solve, Gaussian elimination
    *-------------------------------------------------------------------------------------*/

   switch (relax_type)
   {
      case 0: /* Weighted Jacobi */
         nalu_hypre_BoomerAMGRelax0WeightedJacobi(A, f, cf_marker, relax_points,
                                             relax_weight, u, Vtemp);
         break;

      case 1: /* Gauss-Seidel VERY SLOW */
         nalu_hypre_BoomerAMGRelax1GaussSeidel(A, f, cf_marker, relax_points, u);
         break;

      case 2: /* Gauss-Seidel: relax interior points in parallel, boundary sequentially */
         nalu_hypre_BoomerAMGRelax2GaussSeidel(A, f, cf_marker, relax_points, u);
         break;

      /* Hybrid: Jacobi off-processor, Gauss-Seidel on-processor (forward loop) */
      case 3:
         nalu_hypre_BoomerAMGRelax3HybridGaussSeidel(A, f, cf_marker, relax_points,
                                                relax_weight, omega, u, Vtemp,
                                                Ztemp);
         break;

      case 4: /* Hybrid: Jacobi off-processor, Gauss-Seidel/SOR on-processor (backward loop) */
         nalu_hypre_BoomerAMGRelax4HybridGaussSeidel(A, f, cf_marker, relax_points,
                                                relax_weight, omega, u, Vtemp,
                                                Ztemp);
         break;

      case 5: /* Hybrid: Jacobi off-processor, chaotic Gauss-Seidel on-processor */
         nalu_hypre_BoomerAMGRelax5ChaoticHybridGaussSeidel(A, f, cf_marker, relax_points, u);
         break;

      case 6: /* Hybrid: Jacobi off-processor, Symm. Gauss-Seidel/SSOR on-processor with outer relaxation parameter */
         nalu_hypre_BoomerAMGRelax6HybridSSOR(A, f, cf_marker, relax_points,
                                         relax_weight, omega, u, Vtemp,
                                         Ztemp);
         break;

      case 7: /* Jacobi (uses ParMatvec) */
         nalu_hypre_BoomerAMGRelax7Jacobi(A, f, cf_marker, relax_points,
                                     relax_weight, l1_norms, u, Vtemp);
         break;

      case 8: /* hybrid L1 Symm. Gauss-Seidel */
         nalu_hypre_BoomerAMGRelax8HybridL1SSOR(A, f, cf_marker, relax_points,
                                           relax_weight, omega, l1_norms, u,
                                           Vtemp, Ztemp);
         break;

      /* Hybrid: Jacobi off-processor, ordered Gauss-Seidel on-processor */
      case 10:
         nalu_hypre_BoomerAMGRelax10TopoOrderedGaussSeidel(A, f, cf_marker, relax_points,
                                                      relax_weight, omega, u,
                                                      Vtemp, Ztemp);
         break;

      case 11: /* Two Stage Gauss Seidel. Forward sweep only */
         nalu_hypre_BoomerAMGRelax11TwoStageGaussSeidel(A, f, cf_marker, relax_points,
                                                   relax_weight, omega, l1_norms, u,
                                                   Vtemp, Ztemp);
         break;

      case 12: /* Two Stage Gauss Seidel. Uses the diagonal matrix for the GS part */
         nalu_hypre_BoomerAMGRelax12TwoStageGaussSeidel(A, f, cf_marker, relax_points,
                                                   relax_weight, omega, l1_norms, u,
                                                   Vtemp, Ztemp);
         break;

      case 13: /* hybrid L1 Gauss-Seidel forward solve */
         nalu_hypre_BoomerAMGRelax13HybridL1GaussSeidel(A, f, cf_marker, relax_points,
                                                   relax_weight, omega, l1_norms, u,
                                                   Vtemp, Ztemp);
         break;

      case 14: /* hybrid L1 Gauss-Seidel backward solve */
         nalu_hypre_BoomerAMGRelax14HybridL1GaussSeidel(A, f, cf_marker, relax_points,
                                                   relax_weight, omega, l1_norms, u,
                                                   Vtemp, Ztemp);
         break;

      case 18: /* weighted L1 Jacobi */
         nalu_hypre_BoomerAMGRelax18WeightedL1Jacobi(A, f, cf_marker, relax_points,
                                                relax_weight, l1_norms, u,
                                                Vtemp);
         break;

      case 19: /* Direct solve: use gaussian elimination */
         relax_error = nalu_hypre_BoomerAMGRelax19GaussElim(A, f, u);
         break;

      case 20: /* Kaczmarz */
         nalu_hypre_BoomerAMGRelaxKaczmarz(A, f, omega, l1_norms, u);
         break;

      case 98: /* Direct solve: use gaussian elimination & BLAS (with pivoting) */
         relax_error = nalu_hypre_BoomerAMGRelax98GaussElimPivot(A, f, u);
         break;
   }

   nalu_hypre_ParVectorAllZeros(u) = 0;

   return relax_error;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelaxWeightedJacobi_core( nalu_hypre_ParCSRMatrix *A,
                                         nalu_hypre_ParVector    *f,
                                         NALU_HYPRE_Int          *cf_marker,
                                         NALU_HYPRE_Int           relax_points,
                                         NALU_HYPRE_Real          relax_weight,
                                         NALU_HYPRE_Real         *l1_norms,
                                         nalu_hypre_ParVector    *u,
                                         nalu_hypre_ParVector    *Vtemp,
                                         NALU_HYPRE_Int           Skip_diag )
{
   MPI_Comm             comm          = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix     *A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real          *A_diag_data   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int           *A_diag_i      = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int           *A_diag_j      = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix     *A_offd        = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int           *A_offd_i      = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real          *A_offd_data   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int           *A_offd_j      = nalu_hypre_CSRMatrixJ(A_offd);
   nalu_hypre_ParCSRCommPkg *comm_pkg      = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int            num_rows      = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int            num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   nalu_hypre_Vector        *u_local       = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex       *u_data        = nalu_hypre_VectorData(u_local);
   nalu_hypre_Vector        *f_local       = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Complex       *f_data        = nalu_hypre_VectorData(f_local);
   nalu_hypre_Vector        *Vtemp_local   = nalu_hypre_ParVectorLocalVector(Vtemp);
   NALU_HYPRE_Complex       *Vtemp_data    = nalu_hypre_VectorData(Vtemp_local);
   NALU_HYPRE_Complex       *v_ext_data    = NULL;
   NALU_HYPRE_Complex       *v_buf_data    = NULL;

   NALU_HYPRE_Complex        zero             = 0.0;
   NALU_HYPRE_Real           one_minus_weight = 1.0 - relax_weight;
   NALU_HYPRE_Complex        res;

   NALU_HYPRE_Int num_procs, my_id, i, j, ii, jj, index, num_sends, start;
   nalu_hypre_ParCSRCommHandle *comm_handle;

   /* Sanity check */
   if (nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Jacobi relaxation doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (num_procs > 1)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      v_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                 NALU_HYPRE_MEMORY_HOST);
      v_ext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            v_buf_data[index++] = u_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, v_ext_data);
   }

   /*-----------------------------------------------------------------
    * Copy current approximation into temporary vector.
    *-----------------------------------------------------------------*/
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_rows; i++)
   {
      Vtemp_data[i] = u_data[i];
   }

   if (num_procs > 1)
   {
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   /*-----------------------------------------------------------------
    * Relax all points.
    *-----------------------------------------------------------------*/
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,ii,jj,res) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_rows; i++)
   {
      const NALU_HYPRE_Complex di = l1_norms ? l1_norms[i] : A_diag_data[A_diag_i[i]];

      /*-----------------------------------------------------------
       * If i is of the right type ( C or F or All ) and diagonal is
       * nonzero, relax point i; otherwise, skip it.
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------*/
      if ( (relax_points == 0 || cf_marker[i] == relax_points) && di != zero )
      {
         res = f_data[i];
         for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
         {
            ii = A_diag_j[jj];
            res -= A_diag_data[jj] * Vtemp_data[ii];
         }
         for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
         {
            ii = A_offd_j[jj];
            res -= A_offd_data[jj] * v_ext_data[ii];
         }

         if (Skip_diag)
         {
            u_data[i] *= one_minus_weight;
            u_data[i] += relax_weight * res / di;
         }
         else
         {
            u_data[i] += relax_weight * res / di;
         }
      }
   }

   if (num_procs > 1)
   {
      nalu_hypre_TFree(v_ext_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(v_buf_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax0WeightedJacobi( nalu_hypre_ParCSRMatrix *A,
                                     nalu_hypre_ParVector    *f,
                                     NALU_HYPRE_Int          *cf_marker,
                                     NALU_HYPRE_Int           relax_points,
                                     NALU_HYPRE_Real          relax_weight,
                                     nalu_hypre_ParVector    *u,
                                     nalu_hypre_ParVector    *Vtemp )
{
   return nalu_hypre_BoomerAMGRelaxWeightedJacobi_core(A, f, cf_marker, relax_points, relax_weight, NULL, u,
                                                  Vtemp, 1);
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax18WeightedL1Jacobi( nalu_hypre_ParCSRMatrix *A,
                                        nalu_hypre_ParVector    *f,
                                        NALU_HYPRE_Int          *cf_marker,
                                        NALU_HYPRE_Int           relax_points,
                                        NALU_HYPRE_Real          relax_weight,
                                        NALU_HYPRE_Real         *l1_norms,
                                        nalu_hypre_ParVector    *u,
                                        nalu_hypre_ParVector    *Vtemp )
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParVectorMemoryLocation(f) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      // XXX GPU calls Relax7 XXX
      return nalu_hypre_BoomerAMGRelax7Jacobi(A, f, cf_marker, relax_points, relax_weight, l1_norms, u, Vtemp);
   }
   else
#endif
   {
      /* in the case of non-CF, use relax-7 which is faster */
      if (relax_points == 0)
      {
         return nalu_hypre_BoomerAMGRelax7Jacobi(A, f, cf_marker, relax_points, relax_weight, l1_norms, u, Vtemp);
      }
      else
      {
         return nalu_hypre_BoomerAMGRelaxWeightedJacobi_core(A, f, cf_marker, relax_points, relax_weight,
                                                        l1_norms, u, Vtemp, 0);
      }
   }
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax1GaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                  nalu_hypre_ParVector    *f,
                                  NALU_HYPRE_Int          *cf_marker,
                                  NALU_HYPRE_Int           relax_points,
                                  nalu_hypre_ParVector    *u )
{
   MPI_Comm             comm          = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix     *A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real          *A_diag_data   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int           *A_diag_i      = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int           *A_diag_j      = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix     *A_offd        = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int           *A_offd_i      = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real          *A_offd_data   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int           *A_offd_j      = nalu_hypre_CSRMatrixJ(A_offd);
   nalu_hypre_ParCSRCommPkg *comm_pkg      = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int            num_rows      = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int            num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   nalu_hypre_Vector        *u_local       = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex       *u_data        = nalu_hypre_VectorData(u_local);
   nalu_hypre_Vector        *f_local       = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Complex       *f_data        = nalu_hypre_VectorData(f_local);
   NALU_HYPRE_Complex       *v_ext_data    = NULL;
   NALU_HYPRE_Complex       *v_buf_data    = NULL;
   NALU_HYPRE_Complex        zero          = 0.0;
   NALU_HYPRE_Complex        res;

   NALU_HYPRE_Int num_procs, my_id, i, j, ii, jj, p, jr, ip, num_sends, num_recvs, vec_start, vec_len;
   nalu_hypre_MPI_Status *status;
   nalu_hypre_MPI_Request *requests;

   /* Sanity check */
   if (nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "GS (1) relaxation doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (num_procs > 1)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);

      v_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,
                                 nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                 NALU_HYPRE_MEMORY_HOST);
      v_ext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_recvs + num_sends, NALU_HYPRE_MEMORY_HOST);
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_recvs + num_sends, NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------
    * Relax all points.
    *-----------------------------------------------------------------*/
   for (p = 0; p < num_procs; p++)
   {
      jr = 0;
      if (p != my_id)
      {
         for (i = 0; i < num_sends; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            if (ip == p)
            {
               vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
               for (j = vec_start; j < vec_start + vec_len; j++)
               {
                  v_buf_data[j] = u_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
               }
               nalu_hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, NALU_HYPRE_MPI_COMPLEX, ip, 0,
                               comm, &requests[jr++]);
            }
         }
         nalu_hypre_MPI_Waitall(jr, requests, status);
         nalu_hypre_MPI_Barrier(comm);
      }
      else
      {
         if (num_procs > 1)
         {
            for (i = 0; i < num_recvs; i++)
            {
               ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
               vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
               vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
               nalu_hypre_MPI_Irecv(&v_ext_data[vec_start], vec_len, NALU_HYPRE_MPI_COMPLEX, ip, 0,
                               comm, &requests[jr++]);
            }
            nalu_hypre_MPI_Waitall(jr, requests, status);
         }

         for (i = 0; i < num_rows; i++)
         {
            /*-----------------------------------------------------------
             * If i is of the right type ( C or F ) and diagonal is
             * nonzero, relax point i; otherwise, skip it.
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------*/
            if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
            {
               res = f_data[i];
               for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
               {
                  ii = A_diag_j[jj];
                  res -= A_diag_data[jj] * u_data[ii];
               }
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * v_ext_data[ii];
               }
               u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
         }

         if (num_procs > 1)
         {
            nalu_hypre_MPI_Barrier(comm);
         }
      }
   }

   if (num_procs > 1)
   {
      nalu_hypre_TFree(v_ext_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(v_buf_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax2GaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                  nalu_hypre_ParVector    *f,
                                  NALU_HYPRE_Int          *cf_marker,
                                  NALU_HYPRE_Int           relax_points,
                                  nalu_hypre_ParVector    *u )
{
   MPI_Comm             comm          = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix     *A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real          *A_diag_data   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int           *A_diag_i      = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int           *A_diag_j      = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix     *A_offd        = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int           *A_offd_i      = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real          *A_offd_data   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int           *A_offd_j      = nalu_hypre_CSRMatrixJ(A_offd);
   nalu_hypre_ParCSRCommPkg *comm_pkg      = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int            num_rows      = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int            num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   nalu_hypre_Vector        *u_local       = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex       *u_data        = nalu_hypre_VectorData(u_local);
   nalu_hypre_Vector        *f_local       = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Complex       *f_data        = nalu_hypre_VectorData(f_local);
   NALU_HYPRE_Complex       *v_ext_data    = NULL;
   NALU_HYPRE_Complex       *v_buf_data    = NULL;
   NALU_HYPRE_Complex        zero          = 0.0;
   NALU_HYPRE_Complex        res;

   NALU_HYPRE_Int num_procs, my_id, i, j, ii, jj, p, jr, ip, num_sends, num_recvs, vec_start, vec_len;
   nalu_hypre_MPI_Status *status;
   nalu_hypre_MPI_Request *requests;

   /* Sanity check */
   if (nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "GS (2) relaxation doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (num_procs > 1)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);

      v_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,
                                 nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                 NALU_HYPRE_MEMORY_HOST);
      v_ext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      status  = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_recvs + num_sends, NALU_HYPRE_MEMORY_HOST);
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_recvs + num_sends, NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------
    * Relax interior points first
    *-----------------------------------------------------------------*/
   for (i = 0; i < num_rows; i++)
   {
      /*-----------------------------------------------------------
       * If i is of the right type ( C or F or All ) and diagonal is
       * nonzero, relax point i; otherwise, skip it.
       *-----------------------------------------------------------*/
      if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_offd_i[i + 1] - A_offd_i[i] == zero &&
           A_diag_data[A_diag_i[i]] != zero )
      {
         res = f_data[i];
         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
         }
         u_data[i] = res / A_diag_data[A_diag_i[i]];
      }
   }

   for (p = 0; p < num_procs; p++)
   {
      jr = 0;
      if (p != my_id)
      {
         for (i = 0; i < num_sends; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            if (ip == p)
            {
               vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
               for (j = vec_start; j < vec_start + vec_len; j++)
               {
                  v_buf_data[j] = u_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
               }
               nalu_hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, NALU_HYPRE_MPI_COMPLEX, ip, 0,
                               comm, &requests[jr++]);
            }
         }
         nalu_hypre_MPI_Waitall(jr, requests, status);
         nalu_hypre_MPI_Barrier(comm);
      }
      else
      {
         if (num_procs > 1)
         {
            for (i = 0; i < num_recvs; i++)
            {
               ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
               vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
               vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
               nalu_hypre_MPI_Irecv(&v_ext_data[vec_start], vec_len, NALU_HYPRE_MPI_COMPLEX, ip, 0,
                               comm, &requests[jr++]);
            }
            nalu_hypre_MPI_Waitall(jr, requests, status);
         }
         for (i = 0; i < num_rows; i++)
         {
            /*-----------------------------------------------------------
             * If i is of the right type ( C or F or All) and diagonal is
             * nonzero, relax point i; otherwise, skip it.
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------*/
            if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_offd_i[i + 1] - A_offd_i[i] != zero &&
                 A_diag_data[A_diag_i[i]] != zero)
            {
               res = f_data[i];
               for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
               {
                  ii = A_diag_j[jj];
                  res -= A_diag_data[jj] * u_data[ii];
               }
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * v_ext_data[ii];
               }
               u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
         }
         if (num_procs > 1)
         {
            nalu_hypre_MPI_Barrier(comm);
         }
      }
   }
   if (num_procs > 1)
   {
      nalu_hypre_TFree(v_ext_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(v_buf_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelaxHybridGaussSeidel_core( nalu_hypre_ParCSRMatrix *A,
                                            nalu_hypre_ParVector    *f,
                                            NALU_HYPRE_Int          *cf_marker,
                                            NALU_HYPRE_Int           relax_points,
                                            NALU_HYPRE_Real          relax_weight,
                                            NALU_HYPRE_Real          omega,
                                            NALU_HYPRE_Real         *l1_norms,
                                            nalu_hypre_ParVector    *u,
                                            nalu_hypre_ParVector    *Vtemp,
                                            nalu_hypre_ParVector    *Ztemp,
                                            NALU_HYPRE_Int           GS_order,
                                            NALU_HYPRE_Int           Symm,
                                            NALU_HYPRE_Int           Skip_diag,
                                            NALU_HYPRE_Int           forced_seq,
                                            NALU_HYPRE_Int           Topo_order )
{
   MPI_Comm             comm          = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix     *A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real          *A_diag_data   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int           *A_diag_i      = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int           *A_diag_j      = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix     *A_offd        = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int           *A_offd_i      = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real          *A_offd_data   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int           *A_offd_j      = nalu_hypre_CSRMatrixJ(A_offd);
   nalu_hypre_ParCSRCommPkg *comm_pkg      = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int            num_rows      = nalu_hypre_CSRMatrixNumRows(A_diag);
   nalu_hypre_Vector        *u_local       = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex       *u_data        = nalu_hypre_VectorData(u_local);
   nalu_hypre_Vector        *f_local       = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Complex       *f_data        = nalu_hypre_VectorData(f_local);
   nalu_hypre_Vector        *Vtemp_local   = Vtemp ? nalu_hypre_ParVectorLocalVector(Vtemp) : NULL;
   NALU_HYPRE_Complex       *Vtemp_data    = Vtemp_local ? nalu_hypre_VectorData(Vtemp_local) : NULL;
   /*
   nalu_hypre_Vector        *Ztemp_local   = NULL;
   NALU_HYPRE_Complex       *Ztemp_data    = NULL;
   */
   NALU_HYPRE_Complex       *v_ext_data    = NULL;
   NALU_HYPRE_Complex       *v_buf_data    = NULL;
   NALU_HYPRE_Int           *proc_ordering = NULL;

   const NALU_HYPRE_Real     one_minus_omega  = 1.0 - omega;
   NALU_HYPRE_Int            num_procs, my_id, num_threads, j, num_sends;

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   // JSP: persistent comm can be similarly used for other smoothers
   nalu_hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#else
   nalu_hypre_ParCSRCommHandle           *comm_handle;
   NALU_HYPRE_Int                         num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
#endif

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = forced_seq ? 1 : nalu_hypre_NumThreads();

   /* Sanity check */
   if (nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Hybrid GS relaxation doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   /* GS order: forward or backward */
   const NALU_HYPRE_Int gs_order = GS_order > 0 ? 1 : -1;
   /* for symmetric GS, a forward followed by a backward */
   const NALU_HYPRE_Int num_sweeps = Symm ? 2 : 1;
   /* if relax_weight and omega are both 1.0 */
   const NALU_HYPRE_Int non_scale = relax_weight == 1.0 && omega == 1.0;
   /* */
   const NALU_HYPRE_Real prod = 1.0 - relax_weight * omega;

   /*
   if (num_threads > 1)
   {
      Ztemp_local = nalu_hypre_ParVectorLocalVector(Ztemp);
      Ztemp_data  = nalu_hypre_VectorData(Ztemp_local);
   }
   */

   if (num_procs > 1)
   {
#ifdef NALU_HYPRE_PROFILE
      nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] -= nalu_hypre_MPI_Wtime();
#endif

      if (!comm_pkg)
      {
         nalu_hypre_MatvecCommPkgCreate(A);
         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
      }

      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
      persistent_comm_handle = nalu_hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
      v_buf_data = (NALU_HYPRE_Real *) nalu_hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);
      v_ext_data = (NALU_HYPRE_Real *) nalu_hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);
#else
      v_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,
                                 nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                 NALU_HYPRE_MEMORY_HOST);
      v_ext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_cols_offd, NALU_HYPRE_MEMORY_HOST);
#endif

      NALU_HYPRE_Int begin = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      NALU_HYPRE_Int end   = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
      for (j = begin; j < end; j++)
      {
         v_buf_data[j - begin] = u_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

#ifdef NALU_HYPRE_PROFILE
      nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_PACK_UNPACK] += nalu_hypre_MPI_Wtime();
      nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] -= nalu_hypre_MPI_Wtime();
#endif

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
      nalu_hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle, NALU_HYPRE_MEMORY_HOST, v_buf_data);
#else
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, v_ext_data);
#endif

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
      nalu_hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle, NALU_HYPRE_MEMORY_HOST, v_ext_data);
#else
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

#ifdef NALU_HYPRE_PROFILE
      nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_HALO_EXCHANGE] += nalu_hypre_MPI_Wtime();
#endif
   }

   if (Topo_order)
   {
      /* Check for ordering of matrix. If stored, get pointer, otherwise
       * compute ordering and point matrix variable to array.
       * Used in AIR
       */
      if (!nalu_hypre_ParCSRMatrixProcOrdering(A))
      {
         proc_ordering = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_topo_sort(A_diag_i, A_diag_j, A_diag_data, proc_ordering, num_rows);
         nalu_hypre_ParCSRMatrixProcOrdering(A) = proc_ordering;
      }
      else
      {
         proc_ordering = nalu_hypre_ParCSRMatrixProcOrdering(A);
      }
   }

   /*-----------------------------------------------------------------
    * Relax all points.
    *-----------------------------------------------------------------*/
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RELAX] -= nalu_hypre_MPI_Wtime();
#endif

   if ( (num_threads > 1 || !non_scale) && Vtemp_data )
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < num_rows; j++)
      {
         Vtemp_data[j] = u_data[j];
      }
   }

   if (num_threads > 1)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < num_threads; j++)
      {
         NALU_HYPRE_Int ns, ne, sweep;
         nalu_hypre_partition1D(num_rows, num_threads, j, &ns, &ne);

         for (sweep = 0; sweep < num_sweeps; sweep++)
         {
            const NALU_HYPRE_Int iorder = num_sweeps == 1 ? gs_order : sweep == 0 ? 1 : -1;
            const NALU_HYPRE_Int ibegin = iorder > 0 ? ns : ne - 1;
            const NALU_HYPRE_Int iend = iorder > 0 ? ne : ns - 1;

            if (non_scale)
            {
               nalu_hypre_HybridGaussSeidelNSThreads(A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_j, A_offd_data,
                                                f_data, cf_marker, relax_points, l1_norms, u_data, Vtemp_data, v_ext_data,
                                                ns, ne, ibegin, iend, iorder, Skip_diag);
            }
            else
            {
               nalu_hypre_HybridGaussSeidelThreads(A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_j, A_offd_data,
                                              f_data, cf_marker, relax_points, relax_weight, omega, one_minus_omega,
                                              prod, l1_norms, u_data, Vtemp_data, v_ext_data, ns, ne, ibegin, iend, iorder, Skip_diag);
            }
         } /* for (sweep = 0; sweep < num_sweeps; sweep++) */
      } /* for (j = 0; j < num_threads; j++) */
   }
   else /* if (num_threads > 1) */
   {
      NALU_HYPRE_Int sweep;
      for (sweep = 0; sweep < num_sweeps; sweep++)
      {
         const NALU_HYPRE_Int iorder = num_sweeps == 1 ? gs_order : sweep == 0 ? 1 : -1;
         const NALU_HYPRE_Int ibegin = iorder > 0 ? 0 : num_rows - 1;
         const NALU_HYPRE_Int iend = iorder > 0 ? num_rows : -1;

         if (Topo_order)
         {
            nalu_hypre_HybridGaussSeidelOrderedNS(A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_j, A_offd_data,
                                             f_data, cf_marker, relax_points, u_data, NULL, v_ext_data,
                                             ibegin, iend, iorder, proc_ordering);
         }
         else
         {
            if (non_scale)
            {
               nalu_hypre_HybridGaussSeidelNS(A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_j, A_offd_data,
                                         f_data, cf_marker, relax_points, l1_norms, u_data, Vtemp_data, v_ext_data,
                                         ibegin, iend, iorder, Skip_diag);
            }
            else
            {
               nalu_hypre_HybridGaussSeidel(A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_j, A_offd_data,
                                       f_data, cf_marker, relax_points, relax_weight, omega, one_minus_omega,
                                       prod, l1_norms, u_data, Vtemp_data, v_ext_data, ibegin, iend, iorder, Skip_diag);
            }
         }
      } /* for (sweep = 0; sweep < num_sweeps; sweep++) */
   } /* if (num_threads > 1) */

#ifndef NALU_HYPRE_USING_PERSISTENT_COMM
   if (num_procs > 1)
   {
      nalu_hypre_TFree(v_ext_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(v_buf_data, NALU_HYPRE_MEMORY_HOST);
   }
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RELAX] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

/* forward hybrid G-S */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax3HybridGaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                        nalu_hypre_ParVector    *f,
                                        NALU_HYPRE_Int          *cf_marker,
                                        NALU_HYPRE_Int           relax_points,
                                        NALU_HYPRE_Real          relax_weight,
                                        NALU_HYPRE_Real          omega,
                                        nalu_hypre_ParVector    *u,
                                        nalu_hypre_ParVector    *Vtemp,
                                        nalu_hypre_ParVector    *Ztemp )
{
   return nalu_hypre_BoomerAMGRelaxHybridSOR(A, f, cf_marker, relax_points, relax_weight,
                                        omega, NULL, u, Vtemp, Ztemp,
                                        1, 0, 1, 0);
}

/* backward hybrid G-S */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax4HybridGaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                        nalu_hypre_ParVector    *f,
                                        NALU_HYPRE_Int          *cf_marker,
                                        NALU_HYPRE_Int           relax_points,
                                        NALU_HYPRE_Real          relax_weight,
                                        NALU_HYPRE_Real          omega,
                                        nalu_hypre_ParVector    *u,
                                        nalu_hypre_ParVector    *Vtemp,
                                        nalu_hypre_ParVector    *Ztemp )
{
   return nalu_hypre_BoomerAMGRelaxHybridSOR(A, f, cf_marker, relax_points, relax_weight,
                                        omega, NULL, u, Vtemp, Ztemp,
                                        -1, 0, 1, 0);
}

/* chaotic forward G-S */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax5ChaoticHybridGaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                               nalu_hypre_ParVector    *f,
                                               NALU_HYPRE_Int          *cf_marker,
                                               NALU_HYPRE_Int           relax_points,
                                               nalu_hypre_ParVector    *u )
{
   MPI_Comm             comm          = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix     *A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real          *A_diag_data   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int           *A_diag_i      = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int           *A_diag_j      = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix     *A_offd        = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int           *A_offd_i      = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real          *A_offd_data   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int           *A_offd_j      = nalu_hypre_CSRMatrixJ(A_offd);
   nalu_hypre_ParCSRCommPkg *comm_pkg      = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int            num_rows      = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int            num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   nalu_hypre_Vector        *u_local       = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex       *u_data        = nalu_hypre_VectorData(u_local);
   nalu_hypre_Vector        *f_local       = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Complex       *f_data        = nalu_hypre_VectorData(f_local);
   NALU_HYPRE_Complex       *v_ext_data    = NULL;
   NALU_HYPRE_Complex       *v_buf_data    = NULL;

   NALU_HYPRE_Complex        zero             = 0.0;
   NALU_HYPRE_Complex        res;

   NALU_HYPRE_Int num_procs, my_id, i, j, ii, jj, index, num_sends, start;
   nalu_hypre_ParCSRCommHandle *comm_handle;

   /* Sanity check */
   if (nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Chaotic GS relaxation doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (num_procs > 1)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      v_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends),
                                 NALU_HYPRE_MEMORY_HOST);
      v_ext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            v_buf_data[index++] = u_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, v_ext_data);

      /*-----------------------------------------------------------------
       * Copy current approximation into temporary vector.
       *-----------------------------------------------------------------*/
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,ii,jj,res) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_rows; i++)
   {
      /*-----------------------------------------------------------
       * If i is of the right type ( C or F or All) and diagonal is
       * nonzero, relax point i; otherwise, skip it.
       * Relax only C or F points as determined by relax_points.
       *-----------------------------------------------------------*/
      if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
      {
         res = f_data[i];
         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
         }
         for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
         {
            ii = A_offd_j[jj];
            res -= A_offd_data[jj] * v_ext_data[ii];
         }
         u_data[i] = res / A_diag_data[A_diag_i[i]];
      }
   }

   if (num_procs > 1)
   {
      nalu_hypre_TFree(v_ext_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(v_buf_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/* symmetric hybrid SOR */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelaxHybridSOR( nalu_hypre_ParCSRMatrix *A,
                               nalu_hypre_ParVector    *f,
                               NALU_HYPRE_Int          *cf_marker,
                               NALU_HYPRE_Int           relax_points,
                               NALU_HYPRE_Real          relax_weight,
                               NALU_HYPRE_Real          omega,
                               NALU_HYPRE_Real         *l1_norms,
                               nalu_hypre_ParVector    *u,
                               nalu_hypre_ParVector    *Vtemp,
                               nalu_hypre_ParVector    *Ztemp,
                               NALU_HYPRE_Int           direction,
                               NALU_HYPRE_Int           symm,
                               NALU_HYPRE_Int           skip_diag,
                               NALU_HYPRE_Int           force_seq )
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParVectorMemoryLocation(f) );

   // TODO implement CF relax on GPUs
   if (relax_points != 0)
   {
      exec = NALU_HYPRE_EXEC_HOST;
   }

   if (nalu_hypre_HandleDeviceGSMethod(nalu_hypre_handle()) == 0)
   {
      exec = NALU_HYPRE_EXEC_HOST;
   }

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      return nalu_hypre_BoomerAMGRelaxHybridGaussSeidelDevice(A, f, cf_marker, relax_points, relax_weight,
                                                         omega, l1_norms, u, Vtemp, Ztemp,
                                                         direction, symm);
   }
   else
#endif
   {
      return nalu_hypre_BoomerAMGRelaxHybridGaussSeidel_core(A, f, cf_marker, relax_points, relax_weight,
                                                        omega, l1_norms, u, Vtemp, Ztemp,
                                                        direction, symm, skip_diag, force_seq, 0);
   }
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax6HybridSSOR( nalu_hypre_ParCSRMatrix *A,
                                 nalu_hypre_ParVector    *f,
                                 NALU_HYPRE_Int          *cf_marker,
                                 NALU_HYPRE_Int           relax_points,
                                 NALU_HYPRE_Real          relax_weight,
                                 NALU_HYPRE_Real          omega,
                                 nalu_hypre_ParVector    *u,
                                 nalu_hypre_ParVector    *Vtemp,
                                 nalu_hypre_ParVector    *Ztemp )
{
   return nalu_hypre_BoomerAMGRelaxHybridSOR(A, f, cf_marker, relax_points, relax_weight,
                                        omega, NULL, u, Vtemp, Ztemp, 1, 1, 1, 0);
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax7Jacobi( nalu_hypre_ParCSRMatrix *A,
                             nalu_hypre_ParVector    *f,
                             NALU_HYPRE_Int          *cf_marker,
                             NALU_HYPRE_Int           relax_points,
                             NALU_HYPRE_Real          relax_weight,
                             NALU_HYPRE_Real         *l1_norms,
                             nalu_hypre_ParVector    *u,
                             nalu_hypre_ParVector    *Vtemp )
{
   NALU_HYPRE_Int       num_rows = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_Vector    l1_norms_vec;
   nalu_hypre_ParVector l1_norms_parvec;

   nalu_hypre_GpuProfilingPushRange("Relax7Jacobi");

   nalu_hypre_VectorNumVectors(&l1_norms_vec) = 1;
   nalu_hypre_VectorMultiVecStorageMethod(&l1_norms_vec) = 0;
   nalu_hypre_VectorOwnsData(&l1_norms_vec) = 0;
   nalu_hypre_VectorData(&l1_norms_vec) = l1_norms;
   nalu_hypre_VectorSize(&l1_norms_vec) = num_rows;

   /* TODO XXX
    * The next line is NOT 100% correct, which should be the memory location of l1_norms instead of f
    * But how do I know it? As said, don't use raw pointers, don't use raw pointers!
    * It is fine normally since A, f, and l1_norms should live in the same memory space
    */
   nalu_hypre_VectorMemoryLocation(&l1_norms_vec) = nalu_hypre_ParVectorMemoryLocation(f);
   nalu_hypre_ParVectorLocalVector(&l1_norms_parvec) = &l1_norms_vec;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_Int sync_stream;
   nalu_hypre_GetSyncCudaCompute(&sync_stream);
   nalu_hypre_SetSyncCudaCompute(0);
#endif

   /*-----------------------------------------------------------------
    * Copy f into temporary vector.
    *-----------------------------------------------------------------*/
   nalu_hypre_ParVectorCopy(f, Vtemp);

   /*-----------------------------------------------------------------
    * Perform Matvec Vtemp = w * (f - Au)
    *-----------------------------------------------------------------*/
   if (nalu_hypre_ParVectorAllZeros(u))
   {
#if defined(NALU_HYPRE_DEBUG)
      nalu_hypre_assert(nalu_hypre_ParVectorInnerProd(u, u) == 0.0);
      /*nalu_hypre_ParPrintf(nalu_hypre_ParCSRMatrixComm(A), "A %d: skip a matvec\n", nalu_hypre_ParCSRMatrixGlobalNumRows(A));*/
#endif
      nalu_hypre_ParVectorScale(relax_weight, Vtemp);
   }
   else
   {
      nalu_hypre_ParCSRMatrixMatvec(-relax_weight, A, u, relax_weight, Vtemp);
   }

   /*-----------------------------------------------------------------
    * u += D^{-1} * Vtemp, where D_ii = ||A(i,:)||_1
    *-----------------------------------------------------------------*/
   if (relax_points)
   {
      nalu_hypre_ParVectorElmdivpyMarked(Vtemp, &l1_norms_parvec, u, cf_marker, relax_points);
   }
   else
   {
      nalu_hypre_ParVectorElmdivpy(Vtemp, &l1_norms_parvec, u);
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_SetSyncCudaCompute(sync_stream);
   nalu_hypre_SyncComputeStream(nalu_hypre_handle());
#endif

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/* symmetric l1 hybrid G-S */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax8HybridL1SSOR( nalu_hypre_ParCSRMatrix *A,
                                   nalu_hypre_ParVector    *f,
                                   NALU_HYPRE_Int          *cf_marker,
                                   NALU_HYPRE_Int           relax_points,
                                   NALU_HYPRE_Real          relax_weight,
                                   NALU_HYPRE_Real          omega,
                                   NALU_HYPRE_Real         *l1_norms,
                                   nalu_hypre_ParVector    *u,
                                   nalu_hypre_ParVector    *Vtemp,
                                   nalu_hypre_ParVector    *Ztemp )
{
   const NALU_HYPRE_Int skip_diag = relax_weight == 1.0 && omega == 1.0 ? 0 : 1;

   return nalu_hypre_BoomerAMGRelaxHybridSOR(A, f, cf_marker, relax_points, relax_weight,
                                        omega, l1_norms, u, Vtemp, Ztemp, 1, 1, skip_diag, 0);
}

/* forward hybrid topology ordered G-S */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax10TopoOrderedGaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                              nalu_hypre_ParVector    *f,
                                              NALU_HYPRE_Int          *cf_marker,
                                              NALU_HYPRE_Int           relax_points,
                                              NALU_HYPRE_Real          relax_weight,
                                              NALU_HYPRE_Real          omega,
                                              nalu_hypre_ParVector    *u,
                                              nalu_hypre_ParVector    *Vtemp,
                                              nalu_hypre_ParVector    *Ztemp )
{
   return nalu_hypre_BoomerAMGRelaxHybridGaussSeidel_core(A, f, cf_marker, relax_points, relax_weight,
                                                     omega, NULL, u, Vtemp, Ztemp,
                                                     1 /* forward */, 0 /* nonsymm */, 1 /* skip_diag */, 1, 1);
}

/* forward l1 hybrid G-S */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax13HybridL1GaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                           nalu_hypre_ParVector    *f,
                                           NALU_HYPRE_Int          *cf_marker,
                                           NALU_HYPRE_Int           relax_points,
                                           NALU_HYPRE_Real          relax_weight,
                                           NALU_HYPRE_Real          omega,
                                           NALU_HYPRE_Real         *l1_norms,
                                           nalu_hypre_ParVector    *u,
                                           nalu_hypre_ParVector    *Vtemp,
                                           nalu_hypre_ParVector    *Ztemp )
{
   const NALU_HYPRE_Int skip_diag = relax_weight == 1.0 && omega == 1.0 ? 0 : 1;

   return nalu_hypre_BoomerAMGRelaxHybridSOR(A, f, cf_marker, relax_points, relax_weight,
                                        omega, l1_norms, u, Vtemp, Ztemp,
                                        1,  0, skip_diag, 0);
}

/* backward l1 hybrid G-S */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax14HybridL1GaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                           nalu_hypre_ParVector    *f,
                                           NALU_HYPRE_Int          *cf_marker,
                                           NALU_HYPRE_Int           relax_points,
                                           NALU_HYPRE_Real          relax_weight,
                                           NALU_HYPRE_Real          omega,
                                           NALU_HYPRE_Real         *l1_norms,
                                           nalu_hypre_ParVector    *u,
                                           nalu_hypre_ParVector    *Vtemp,
                                           nalu_hypre_ParVector    *Ztemp )
{
   const NALU_HYPRE_Int skip_diag = relax_weight == 1.0 && omega == 1.0 ? 0 : 1;

   return nalu_hypre_BoomerAMGRelaxHybridSOR(A, f, cf_marker, relax_points, relax_weight,
                                        omega, l1_norms, u, Vtemp, Ztemp,
                                        -1, 0, skip_diag, 0);
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax19GaussElim( nalu_hypre_ParCSRMatrix *A,
                                 nalu_hypre_ParVector    *f,
                                 nalu_hypre_ParVector    *u )
{
   NALU_HYPRE_BigInt     global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt     first_ind       = nalu_hypre_ParVectorFirstIndex(u);
   NALU_HYPRE_Int        n_global        = (NALU_HYPRE_Int) global_num_rows;
   NALU_HYPRE_Int        first_index     = (NALU_HYPRE_Int) first_ind;
   NALU_HYPRE_Int        num_rows        = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_Vector    *u_local         = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex   *u_data          = nalu_hypre_VectorData(u_local);
   nalu_hypre_CSRMatrix *A_CSR;
   NALU_HYPRE_Int       *A_CSR_i;
   NALU_HYPRE_Int       *A_CSR_j;
   NALU_HYPRE_Real      *A_CSR_data;
   nalu_hypre_Vector    *f_vector;
   NALU_HYPRE_Real      *f_vector_data;
   NALU_HYPRE_Real      *A_mat;
   NALU_HYPRE_Real      *b_vec;
   NALU_HYPRE_Int        i, jj, column, relax_error = 0;

   /* Sanity check */
   if (nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Gauss Elim. relaxation doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    *  Generate CSR matrix from ParCSRMatrix A
    *-----------------------------------------------------------------*/

   /* all processors are needed for these routines */
   A_CSR = nalu_hypre_ParCSRMatrixToCSRMatrixAll(A);
   f_vector = nalu_hypre_ParVectorToVectorAll(f);
   if (num_rows)
   {
      A_CSR_i = nalu_hypre_CSRMatrixI(A_CSR);
      A_CSR_j = nalu_hypre_CSRMatrixJ(A_CSR);
      A_CSR_data = nalu_hypre_CSRMatrixData(A_CSR);
      f_vector_data = nalu_hypre_VectorData(f_vector);

      A_mat = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n_global * n_global, NALU_HYPRE_MEMORY_HOST);
      b_vec = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n_global, NALU_HYPRE_MEMORY_HOST);

      /*---------------------------------------------------------------
       *  Load CSR matrix into A_mat.
       *---------------------------------------------------------------*/
      for (i = 0; i < n_global; i++)
      {
         for (jj = A_CSR_i[i]; jj < A_CSR_i[i + 1]; jj++)
         {
            column = A_CSR_j[jj];
            A_mat[i * n_global + column] = A_CSR_data[jj];
         }
         b_vec[i] = f_vector_data[i];
      }

      nalu_hypre_gselim(A_mat, b_vec, n_global, relax_error);

      for (i = 0; i < num_rows; i++)
      {
         u_data[i] = b_vec[first_index + i];
      }

      nalu_hypre_TFree(A_mat, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(b_vec, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      nalu_hypre_SeqVectorDestroy(f_vector);
      f_vector = NULL;
   }
   else
   {
      nalu_hypre_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      nalu_hypre_SeqVectorDestroy(f_vector);
      f_vector = NULL;
   }

   return relax_error;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax98GaussElimPivot( nalu_hypre_ParCSRMatrix *A,
                                      nalu_hypre_ParVector    *f,
                                      nalu_hypre_ParVector    *u )
{
   NALU_HYPRE_BigInt     global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt     first_ind       = nalu_hypre_ParVectorFirstIndex(u);
   NALU_HYPRE_Int        n_global        = (NALU_HYPRE_Int) global_num_rows;
   NALU_HYPRE_Int        first_index     = (NALU_HYPRE_Int) first_ind;
   NALU_HYPRE_Int        num_rows        = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_Vector    *u_local         = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex   *u_data          = nalu_hypre_VectorData(u_local);
   nalu_hypre_CSRMatrix *A_CSR;
   NALU_HYPRE_Int       *A_CSR_i;
   NALU_HYPRE_Int       *A_CSR_j;
   NALU_HYPRE_Real      *A_CSR_data;
   nalu_hypre_Vector    *f_vector;
   NALU_HYPRE_Real      *f_vector_data;
   NALU_HYPRE_Real      *A_mat;
   NALU_HYPRE_Real      *b_vec;
   NALU_HYPRE_Int        i, jj, column, relax_error = 0;
   NALU_HYPRE_Int        info;
   NALU_HYPRE_Int        one_i = 1;
   NALU_HYPRE_Int       *piv;

   /* Sanity check */
   if (nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Gauss Elim. (98) relaxation doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    *  Generate CSR matrix from ParCSRMatrix A
    *-----------------------------------------------------------------*/

   /* all processors are needed for these routines */
   A_CSR = nalu_hypre_ParCSRMatrixToCSRMatrixAll(A);
   f_vector = nalu_hypre_ParVectorToVectorAll(f);
   if (num_rows)
   {
      A_CSR_i = nalu_hypre_CSRMatrixI(A_CSR);
      A_CSR_j = nalu_hypre_CSRMatrixJ(A_CSR);
      A_CSR_data = nalu_hypre_CSRMatrixData(A_CSR);
      f_vector_data = nalu_hypre_VectorData(f_vector);

      A_mat = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  n_global * n_global, NALU_HYPRE_MEMORY_HOST);
      b_vec = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  n_global, NALU_HYPRE_MEMORY_HOST);

      /*---------------------------------------------------------------
       *  Load CSR matrix into A_mat.
       *---------------------------------------------------------------*/
      for (i = 0; i < n_global; i++)
      {
         for (jj = A_CSR_i[i]; jj < A_CSR_i[i + 1]; jj++)
         {
            /* need col major */
            column = A_CSR_j[jj];
            A_mat[i + n_global * column] = A_CSR_data[jj];
         }
         b_vec[i] = f_vector_data[i];
      }

      piv = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_global, NALU_HYPRE_MEMORY_HOST);

      /* write over A with LU */
      nalu_hypre_dgetrf(&n_global, &n_global, A_mat, &n_global, piv, &info);

      /*now b_vec = inv(A)*b_vec  */
      nalu_hypre_dgetrs("N", &n_global, &one_i, A_mat, &n_global, piv, b_vec, &n_global, &info);

      nalu_hypre_TFree(piv, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_rows; i++)
      {
         u_data[i] = b_vec[first_index + i];
      }

      nalu_hypre_TFree(A_mat, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(b_vec, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      nalu_hypre_SeqVectorDestroy(f_vector);
      f_vector = NULL;
   }
   else
   {
      nalu_hypre_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      nalu_hypre_SeqVectorDestroy(f_vector);
      f_vector = NULL;
   }

   return relax_error;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelaxKaczmarz( nalu_hypre_ParCSRMatrix *A,
                              nalu_hypre_ParVector    *f,
                              NALU_HYPRE_Real          omega,
                              NALU_HYPRE_Real         *l1_norms,
                              nalu_hypre_ParVector    *u )
{
   MPI_Comm             comm          = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix     *A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real          *A_diag_data   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int           *A_diag_i      = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int           *A_diag_j      = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix     *A_offd        = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int           *A_offd_i      = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real          *A_offd_data   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int           *A_offd_j      = nalu_hypre_CSRMatrixJ(A_offd);
   nalu_hypre_ParCSRCommPkg *comm_pkg      = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int            num_rows      = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int            num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   nalu_hypre_Vector        *u_local       = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex       *u_data        = nalu_hypre_VectorData(u_local);
   nalu_hypre_Vector        *f_local       = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Complex       *f_data        = nalu_hypre_VectorData(f_local);
   NALU_HYPRE_Complex       *u_offd_data   = NULL;
   NALU_HYPRE_Complex       *u_buf_data    = NULL;
   NALU_HYPRE_Complex        res;

   NALU_HYPRE_Int num_procs, my_id, i, j, index, num_sends, start;
   nalu_hypre_ParCSRCommHandle *comm_handle;

   /* Sanity check */
   if (nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Kaczmarz relaxation doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (num_procs > 1)
   {
      if (!comm_pkg)
      {
         nalu_hypre_MatvecCommPkgCreate(A);
         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
      }

      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      u_buf_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                NALU_HYPRE_MEMORY_HOST);
      u_offd_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            u_buf_data[index++] = u_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, u_buf_data, u_offd_data);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      nalu_hypre_TFree(u_buf_data, NALU_HYPRE_MEMORY_HOST);
   }

   /* Forward local pass */
   for (i = 0; i < num_rows; i++)
   {
      res = f_data[i];
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         res -= A_diag_data[j] * u_data[A_diag_j[j]];
      }

      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         res -= A_offd_data[j] * u_offd_data[A_offd_j[j]];
      }

      res /= l1_norms[i];

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         u_data[A_diag_j[j]] += omega * res * A_diag_data[j];
      }
   }

   /* Backward local pass */
   for (i = num_rows - 1; i > -1; i--)
   {
      res = f_data[i];
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         res -= A_diag_data[j] * u_data[A_diag_j[j]];
      }

      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         res -= A_offd_data[j] * u_offd_data[A_offd_j[j]];
      }

      res /= l1_norms[i];

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         u_data[A_diag_j[j]] += omega * res * A_diag_data[j];
      }
   }

   nalu_hypre_TFree(u_offd_data, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelaxTwoStageGaussSeidelHost
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelaxTwoStageGaussSeidelHost( nalu_hypre_ParCSRMatrix *A,
                                             nalu_hypre_ParVector    *f,
                                             NALU_HYPRE_Real          relax_weight,
                                             NALU_HYPRE_Real          omega,
                                             nalu_hypre_ParVector    *u,
                                             nalu_hypre_ParVector    *Vtemp,
                                             NALU_HYPRE_Int           num_inner_iters )
{
   nalu_hypre_CSRMatrix *A_diag      = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int        num_rows    = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i    = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j    = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_Vector    *Vtemp_local = nalu_hypre_ParVectorLocalVector(Vtemp);
   NALU_HYPRE_Complex   *Vtemp_data  = nalu_hypre_VectorData(Vtemp_local);
   nalu_hypre_Vector    *u_local     = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex   *u_data      = nalu_hypre_VectorData(u_local);

   NALU_HYPRE_Complex    multiplier  = 1.0;
   NALU_HYPRE_Int        i, k, jj, ii;

   /* Sanity check */
   if (nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "2-stage GS relaxation (Host) doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   /* Need to check that EVERY diagonal is nonzero first. If any are, throw exception */
   for (i = 0; i < num_rows; i++)
   {
      if (A_diag_data[A_diag_i[i]] == 0.0)
      {
         nalu_hypre_error_in_arg(1);
      }
   }

   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-relax_weight, A, u, relax_weight, f, Vtemp);

   /* Run the smoother */
   for (i = 0; i < num_rows; i++)
   {
      // V = V/D
      Vtemp_data[i] /= A_diag_data[A_diag_i[i]];

      // u = u + m*v
      u_data[i] += multiplier * Vtemp_data[i];
   }

   // adjust for the alternating series
   multiplier *= -1.0;

   for (k = 0; k < num_inner_iters; ++k)
   {
      // By going from bottom to top, we can update Vtemp in place because
      // we're operating with the strict, lower triangular matrix
      for (i = num_rows - 1; i >= 0; i--) /* Run the smoother */
      {
         // spmv for the row first
         NALU_HYPRE_Complex res = 0.0;
         for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
         {
            ii = A_diag_j[jj];
            if (ii < i)
            {
               res += A_diag_data[jj] * Vtemp_data[ii];
            }
         }
         // diagonal scaling has to come after the spmv accumulation. It's a row scaling
         // not column
         Vtemp_data[i] = res / A_diag_data[A_diag_i[i]];
         u_data[i] += multiplier * Vtemp_data[i];
      }

      // adjust for the alternating series
      multiplier *= -1.0;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelax11TwoStageGaussSeidel
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax11TwoStageGaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                           nalu_hypre_ParVector    *f,
                                           NALU_HYPRE_Int          *cf_marker,
                                           NALU_HYPRE_Int           relax_points,
                                           NALU_HYPRE_Real          relax_weight,
                                           NALU_HYPRE_Real          omega,
                                           NALU_HYPRE_Real         *A_diag_diag,
                                           nalu_hypre_ParVector    *u,
                                           nalu_hypre_ParVector    *Vtemp,
                                           nalu_hypre_ParVector    *Ztemp )
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParVectorMemoryLocation(f) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_BoomerAMGRelaxTwoStageGaussSeidelDevice(A, f, relax_weight, omega,
                                                    A_diag_diag, u, Vtemp, Ztemp, 1);
   }
   else
#endif
   {
      nalu_hypre_BoomerAMGRelaxTwoStageGaussSeidelHost(A, f, relax_weight, omega, u, Vtemp, 1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelax12TwoStageGaussSeidel
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax12TwoStageGaussSeidel( nalu_hypre_ParCSRMatrix *A,
                                           nalu_hypre_ParVector    *f,
                                           NALU_HYPRE_Int          *cf_marker,
                                           NALU_HYPRE_Int           relax_points,
                                           NALU_HYPRE_Real          relax_weight,
                                           NALU_HYPRE_Real          omega,
                                           NALU_HYPRE_Real         *A_diag_diag,
                                           nalu_hypre_ParVector    *u,
                                           nalu_hypre_ParVector    *Vtemp,
                                           nalu_hypre_ParVector    *Ztemp )
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParVectorMemoryLocation(f) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_BoomerAMGRelaxTwoStageGaussSeidelDevice(A, f, relax_weight, omega,
                                                    A_diag_diag, u, Vtemp, Ztemp, 2);
   }
   else
#endif
   {
      nalu_hypre_BoomerAMGRelaxTwoStageGaussSeidelHost(A, f, relax_weight, omega, u, Vtemp, 2);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelaxComputeL1Norms
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelaxComputeL1Norms( nalu_hypre_ParCSRMatrix *A,
                                    NALU_HYPRE_Int           relax_type,
                                    NALU_HYPRE_Int           relax_order,
                                    NALU_HYPRE_Int           coarsest_lvl,
                                    nalu_hypre_IntArray     *CF_marker,
                                    NALU_HYPRE_Real        **l1_norms_data_ptr )
{
   NALU_HYPRE_Int     *CF_marker_data;
   NALU_HYPRE_Real    *l1_norms_data = NULL;

   /* Relax according to F/C points ordering? */
   CF_marker_data = (relax_order && CF_marker) ? nalu_hypre_IntArrayData(CF_marker) : NULL;

   /* Are we in the coarsest level? */
   CF_marker_data = (coarsest_lvl) ? NULL : CF_marker_data;

   if (relax_type == 18)
   {
      /* l1_norm = sum(|A_ij|)_j */
      nalu_hypre_ParCSRComputeL1Norms(A, 1, CF_marker_data, &l1_norms_data);
   }
   else if (relax_type == 8 || relax_type == 13 || relax_type == 14)
   {
      /* l1_norm = sum(|D_ij| + 0.5*|A_offd_ij|)_j */
      nalu_hypre_ParCSRComputeL1Norms(A, 4, CF_marker_data, &l1_norms_data);
   }
   else if (relax_type == 7 || relax_type == 11 || relax_type == 12)
   {
      /* l1_norm = |D_ii| */
      nalu_hypre_ParCSRComputeL1Norms(A, 5, NULL, &l1_norms_data);
   }

   *l1_norms_data_ptr = l1_norms_data;

   return nalu_hypre_error_flag;
}
