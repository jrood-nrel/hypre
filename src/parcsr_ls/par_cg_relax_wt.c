/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_amg.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGCycle
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGCGRelaxWt( void       *amg_vdata,
                          NALU_HYPRE_Int   level,
                          NALU_HYPRE_Int   num_cg_sweeps,
                          NALU_HYPRE_Real *rlx_wt_ptr)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) amg_vdata;

   MPI_Comm comm;
   NALU_HYPRE_Solver *smoother;
   /* Data Structure variables */

   /* nalu_hypre_ParCSRMatrix **A_array = nalu_hypre_ParAMGDataAArray(amg_data); */
   /* nalu_hypre_ParCSRMatrix **R_array = nalu_hypre_ParAMGDataRArray(amg_data); */
   nalu_hypre_ParCSRMatrix *A = nalu_hypre_ParAMGDataAArray(amg_data)[level];
   /* nalu_hypre_ParVector    **F_array = nalu_hypre_ParAMGDataFArray(amg_data); */
   /* nalu_hypre_ParVector    **U_array = nalu_hypre_ParAMGDataUArray(amg_data); */
   nalu_hypre_ParVector    *Utemp;
   nalu_hypre_ParVector    *Vtemp;
   nalu_hypre_ParVector    *Ptemp;
   nalu_hypre_ParVector    *Rtemp;
   nalu_hypre_ParVector    *Ztemp;
   nalu_hypre_ParVector    *Qtemp = NULL;

   NALU_HYPRE_Int    *CF_marker = nalu_hypre_IntArrayData(nalu_hypre_ParAMGDataCFMarkerArray(amg_data)[level]);
   NALU_HYPRE_Real   *Ptemp_data;
   NALU_HYPRE_Real   *Ztemp_data;

   /* NALU_HYPRE_Int     **unknown_map_array;
   NALU_HYPRE_Int     **point_map_array;
   NALU_HYPRE_Int     **v_at_point_array; */


   NALU_HYPRE_Int      *grid_relax_type;

   /* Local variables  */
   NALU_HYPRE_Int       Solve_err_flag;
   NALU_HYPRE_Int       i, j, jj;
   NALU_HYPRE_Int       num_sweeps;
   NALU_HYPRE_Int       relax_type;
   NALU_HYPRE_Int       local_size;
   NALU_HYPRE_Int       old_size;
   NALU_HYPRE_Int       my_id = 0;
   NALU_HYPRE_Int       smooth_type;
   NALU_HYPRE_Int       smooth_num_levels;
   NALU_HYPRE_Int       smooth_option = 0;
   NALU_HYPRE_Int       needQ = 0;

   nalu_hypre_Vector *l1_norms = NULL;

   NALU_HYPRE_Real    alpha;
   NALU_HYPRE_Real    beta;
   NALU_HYPRE_Real    gamma = 1.0;
   NALU_HYPRE_Real    gammaold;

   NALU_HYPRE_Real   *tridiag;
   NALU_HYPRE_Real   *trioffd;
   NALU_HYPRE_Real    alphinv, row_sum = 0;
   NALU_HYPRE_Real    max_row_sum = 0;
   NALU_HYPRE_Real    rlx_wt = 0;
   NALU_HYPRE_Real    rlx_wt_old = 0;
   NALU_HYPRE_Real    lambda_max, lambda_max_old;
   /* NALU_HYPRE_Real    lambda_min, lambda_min_old; */

#if 0
   NALU_HYPRE_Real   *D_mat;
   NALU_HYPRE_Real   *S_vec;
#endif

#if !defined(NALU_HYPRE_USING_CUDA) && !defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_Int num_threads = nalu_hypre_NumThreads();
#endif

   /* Acquire data and allocate storage */

   tridiag  = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_cg_sweeps + 1, NALU_HYPRE_MEMORY_HOST);
   trioffd  = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_cg_sweeps + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cg_sweeps + 1; i++)
   {
      tridiag[i] = 0;
      trioffd[i] = 0;
   }

   Vtemp = nalu_hypre_ParAMGDataVtemp(amg_data);

   Rtemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Rtemp);

   Ptemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Ptemp);

   Ztemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Ztemp);

   if (nalu_hypre_ParAMGDataL1Norms(amg_data) != NULL)
   {
      l1_norms = nalu_hypre_ParAMGDataL1Norms(amg_data)[level];
   }

#if !defined(NALU_HYPRE_USING_CUDA) && !defined(NALU_HYPRE_USING_HIP)
   if (num_threads > 1)
#endif
   {
      needQ = 1;
   }

   grid_relax_type     = nalu_hypre_ParAMGDataGridRelaxType(amg_data);
   smooth_type         = nalu_hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels   = nalu_hypre_ParAMGDataSmoothNumLevels(amg_data);

   /* Initialize */

   Solve_err_flag = 0;

   comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (smooth_num_levels > level)
   {
      smoother = nalu_hypre_ParAMGDataSmoother(amg_data);
      smooth_option = smooth_type;
      if (smooth_type > 6 && smooth_type < 10)
      {
         Utemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                       nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                       nalu_hypre_ParCSRMatrixRowStarts(A));
         nalu_hypre_ParVectorInitialize(Utemp);
      }
   }

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/

   relax_type = grid_relax_type[1];
   num_sweeps = 1;

   local_size = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   old_size = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(Vtemp));
   nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(Vtemp)) =
      nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   Ptemp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Ptemp));
   Ztemp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Ztemp));
   /* if (level == 0)
      nalu_hypre_ParVectorCopy(nalu_hypre_ParAMGDataFArray(amg_data)[0],Rtemp);
   else
   {
      nalu_hypre_ParVectorCopy(F_array[level-1],Vtemp);
      alpha = -1.0;
      beta = 1.0;
      nalu_hypre_ParCSRMatrixMatvec(alpha, A_array[level-1], U_array[level-1],
                         beta, Vtemp);
      alpha = 1.0;
      beta = 0.0;

      nalu_hypre_ParCSRMatrixMatvecT(alpha,R_array[level-1],Vtemp,
                          beta,F_array[level]);
      nalu_hypre_ParVectorCopy(F_array[level],Rtemp);
   } */

   nalu_hypre_ParVectorSetRandomValues(Rtemp, 5128);

   if (needQ)
   {
      Qtemp = nalu_hypre_ParMultiVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                         nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                         nalu_hypre_ParCSRMatrixRowStarts(A),
                                         needQ);
      nalu_hypre_ParVectorInitialize(Qtemp);
   }

   /*------------------------------------------------------------------
    * Do the relaxation num_sweeps times
    *-----------------------------------------------------------------*/

   for (jj = 0; jj < num_cg_sweeps; jj++)
   {
      nalu_hypre_ParVectorSetConstantValues(Ztemp, 0.0);

      for (j = 0; j < num_sweeps; j++)
      {
         if (smooth_option > 6)
         {
            nalu_hypre_ParVectorCopy(Rtemp, Vtemp);
            alpha = -1.0;
            beta = 1.0;
            nalu_hypre_ParCSRMatrixMatvec(alpha, A,
                                     Ztemp, beta, Vtemp);
            if (smooth_option == 8)
            {
               NALU_HYPRE_ParCSRParaSailsSolve(smoother[level],
                                          (NALU_HYPRE_ParCSRMatrix) A,
                                          (NALU_HYPRE_ParVector) Vtemp,
                                          (NALU_HYPRE_ParVector) Utemp);
            }
            else if (smooth_option == 7)
            {
               NALU_HYPRE_ParCSRPilutSolve(smoother[level],
                                      (NALU_HYPRE_ParCSRMatrix) A,
                                      (NALU_HYPRE_ParVector) Vtemp,
                                      (NALU_HYPRE_ParVector) Utemp);
               nalu_hypre_ParVectorAxpy(1.0, Utemp, Ztemp);
            }
            else if (smooth_option == 9)
            {
               NALU_HYPRE_EuclidSolve(smoother[level],
                                 (NALU_HYPRE_ParCSRMatrix) A,
                                 (NALU_HYPRE_ParVector) Vtemp,
                                 (NALU_HYPRE_ParVector) Utemp);
               nalu_hypre_ParVectorAxpy(1.0, Utemp, Ztemp);
            }
         }
         else if (smooth_option == 6)
         {
            NALU_HYPRE_SchwarzSolve(smoother[level],
                               (NALU_HYPRE_ParCSRMatrix) A,
                               (NALU_HYPRE_ParVector) Rtemp,
                               (NALU_HYPRE_ParVector) Ztemp);
         }
         else
         {
            Solve_err_flag = nalu_hypre_BoomerAMGRelax(A,
                                                  Rtemp,
                                                  CF_marker,
                                                  relax_type,
                                                  0,
                                                  1.0,
                                                  1.0,
                                                  l1_norms ? nalu_hypre_VectorData(l1_norms) : NULL,
                                                  Ztemp,
                                                  Vtemp,
                                                  Qtemp);
         }

         if (Solve_err_flag != 0)
         {
            nalu_hypre_ParVectorDestroy(Ptemp);
            nalu_hypre_TFree(tridiag, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(trioffd, NALU_HYPRE_MEMORY_HOST);
            return (Solve_err_flag);
         }
      }

      gammaold = gamma;
      gamma = nalu_hypre_ParVectorInnerProd(Rtemp, Ztemp);
      if (jj == 0)
      {
         nalu_hypre_ParVectorCopy(Ztemp, Ptemp);
         beta = 1.0;
      }
      else
      {
         beta = gamma / gammaold;
         for (i = 0; i < local_size; i++)
         {
            Ptemp_data[i] = Ztemp_data[i] + beta * Ptemp_data[i];
         }
      }
      nalu_hypre_ParCSRMatrixMatvec(1.0, A, Ptemp, 0.0, Vtemp);
      alpha = gamma / (nalu_hypre_ParVectorInnerProd(Ptemp, Vtemp) + 1.e-80);
      alphinv = 1.0 / (alpha + 1.e-80);
      tridiag[jj + 1] = alphinv;
      tridiag[jj] *= beta;
      tridiag[jj] += alphinv;
      trioffd[jj] *= sqrt(beta);
      trioffd[jj + 1] = -alphinv;
      row_sum = fabs(tridiag[jj]) + fabs(trioffd[jj]);
      if (row_sum > max_row_sum) { max_row_sum = row_sum; }
      if (jj > 0)
      {
         row_sum = fabs(tridiag[jj - 1]) + fabs(trioffd[jj - 1])
                   + fabs(trioffd[jj]);
         if (row_sum > max_row_sum) { max_row_sum = row_sum; }
         /* lambda_min_old = lambda_min; */
         lambda_max_old = lambda_max;
         rlx_wt_old = rlx_wt;
         nalu_hypre_Bisection(jj + 1, tridiag, trioffd, lambda_max_old,
                         max_row_sum, 1.e-3, jj + 1, &lambda_max);
         rlx_wt = 1.0 / lambda_max;
         /* nalu_hypre_Bisection(jj+1, tridiag, trioffd, 0.0, lambda_min_old,
            1.e-3, 1, &lambda_min);
         rlx_wt = 2.0/(lambda_min+lambda_max); */
         if (fabs(rlx_wt - rlx_wt_old) < 1.e-3 )
         {
            /* if (my_id == 0) nalu_hypre_printf (" cg sweeps : %d\n", (jj+1)); */
            break;
         }
      }
      else
      {
         /* lambda_min = tridiag[0]; */
         lambda_max = tridiag[0];
      }

      nalu_hypre_ParVectorAxpy(-alpha, Vtemp, Rtemp);
   }
   /*if (my_id == 0)
     nalu_hypre_printf (" lambda-min: %f  lambda-max: %f\n", lambda_min, lambda_max);

   rlx_wt = fabs(tridiag[0])+fabs(trioffd[1]);

   for (i=1; i < num_cg_sweeps-1; i++)
   {
      row_sum = fabs(tridiag[i]) + fabs(trioffd[i]) + fabs(trioffd[i+1]);
      if (row_sum > rlx_wt) rlx_wt = row_sum;
   }
   row_sum = fabs(tridiag[num_cg_sweeps-1]) + fabs(trioffd[num_cg_sweeps-1]);
   if (row_sum > rlx_wt) rlx_wt = row_sum;

   nalu_hypre_Bisection(num_cg_sweeps, tridiag, trioffd, 0.0, rlx_wt, 1.e-3, 1,
   &lambda_min);
   nalu_hypre_Bisection(num_cg_sweeps, tridiag, trioffd, 0.0, rlx_wt, 1.e-3,
   num_cg_sweeps, &lambda_max);
   */


   nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(Vtemp)) = old_size;

   nalu_hypre_ParVectorDestroy(Ztemp);
   nalu_hypre_ParVectorDestroy(Ptemp);
   nalu_hypre_ParVectorDestroy(Rtemp);

   if (Qtemp)
   {
      nalu_hypre_ParVectorDestroy(Qtemp);
   }

   nalu_hypre_TFree(tridiag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(trioffd, NALU_HYPRE_MEMORY_HOST);

   if (smooth_option > 6 && smooth_option < 10)
   {
      nalu_hypre_ParVectorDestroy(Utemp);
   }

   *rlx_wt_ptr = rlx_wt;

   return (Solve_err_flag);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_Bisection
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_Bisection(NALU_HYPRE_Int n, NALU_HYPRE_Real *diag, NALU_HYPRE_Real *offd,
                NALU_HYPRE_Real y, NALU_HYPRE_Real z,
                NALU_HYPRE_Real tol, NALU_HYPRE_Int k, NALU_HYPRE_Real *ev_ptr)
{
   NALU_HYPRE_Real x;
   NALU_HYPRE_Real eigen_value;
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int sign_change = 0;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Real p0, p1, p2;

   while (fabs(y - z) > tol * (fabs(y) + fabs(z)))
   {
      x = (y + z) / 2;

      sign_change = 0;
      p0 = 1;
      p1 = diag[0] - x;
      if (p0 * p1 <= 0) { sign_change++; }
      for (i = 1; i < n; i++)
      {
         p2 = (diag[i] - x) * p1 - offd[i] * offd[i] * p0;
         p0 = p1;
         p1 = p2;
         if (p0 * p1 <= 0) { sign_change++; }
      }

      if (sign_change >= k)
      {
         z = x;
      }
      else
      {
         y = x;
      }
   }

   eigen_value = (y + z) / 2;
   *ev_ptr = eigen_value;

   return ierr;
}
