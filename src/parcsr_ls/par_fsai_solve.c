/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

/******************************************************************************
 *
 * FSAI solve routine
 *
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------
 * nalu_hypre_FSAISolve
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAISolve( void               *fsai_vdata,
                 nalu_hypre_ParCSRMatrix *A,
                 nalu_hypre_ParVector    *b,
                 nalu_hypre_ParVector    *x )
{
   MPI_Comm             comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParFSAIData   *fsai_data   = (nalu_hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   nalu_hypre_ParCSRMatrix  *G           = nalu_hypre_ParFSAIDataGmat(fsai_data);
   nalu_hypre_ParCSRMatrix  *GT          = nalu_hypre_ParFSAIDataGTmat(fsai_data);
   nalu_hypre_ParVector     *z_work      = nalu_hypre_ParFSAIDataZWork(fsai_data);
   nalu_hypre_ParVector     *r_work      = nalu_hypre_ParFSAIDataRWork(fsai_data);
   NALU_HYPRE_Int            tol         = nalu_hypre_ParFSAIDataTolerance(fsai_data);
   NALU_HYPRE_Int            zero_guess  = nalu_hypre_ParFSAIDataZeroGuess(fsai_data);
   NALU_HYPRE_Int            max_iter    = nalu_hypre_ParFSAIDataMaxIterations(fsai_data);
   NALU_HYPRE_Int            print_level = nalu_hypre_ParFSAIDataPrintLevel(fsai_data);
   NALU_HYPRE_Int            logging     = nalu_hypre_ParFSAIDataLogging(fsai_data);
   NALU_HYPRE_Real           omega       = nalu_hypre_ParFSAIDataOmega(fsai_data);

   /* Local variables */
   NALU_HYPRE_Int            iter, my_id;
   NALU_HYPRE_Real           old_resnorm, resnorm, rel_resnorm;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /*-----------------------------------------------------------------
    * Preconditioned Richardson - Main solver loop
    * x(k+1) = x(k) + omega * (G^T*G) * (b - A*x(k))
    * ----------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1)
   {
      nalu_hypre_printf("\n\n FSAI SOLVER SOLUTION INFO:\n");
   }

   iter        = 0;
   rel_resnorm = 1.0;

   if (my_id == 0 && print_level > 1)
   {
      nalu_hypre_printf("                new         relative\n");
      nalu_hypre_printf("    iter #      res norm    res norm\n");
      nalu_hypre_printf("    --------    --------    --------\n");
   }

   if (max_iter > 0)
   {
      /* First iteration */
      if (zero_guess)
      {
         /* Compute: x(k+1) = omega*G^T*G*b */
         nalu_hypre_ParCSRMatrixMatvec(1.0, G, b, 0.0, z_work);
         nalu_hypre_ParCSRMatrixMatvec(omega, GT, z_work, 0.0, x);
      }
      else
      {
         /* Compute: x(k+1) = omega*G^T*G*(b - A*x(k)) + x(k) */
         nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, x, 1.0, b, r_work);
         nalu_hypre_ParCSRMatrixMatvec(1.0, G, r_work, 0.0, z_work);
         nalu_hypre_ParCSRMatrixMatvec(omega, GT, z_work, 1.0, x);
      }

      /* Update iteration count */
      iter++;
   }
   else
   {
      nalu_hypre_ParVectorCopy(b, x);
   }

   /* Apply remaining iterations */
   for (; iter < max_iter; iter++)
   {
      /* Update residual */
      nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, x, 1.0, b, r_work);

      if (tol > 0.0)
      {
         old_resnorm = resnorm;
         resnorm = nalu_hypre_ParVectorInnerProd(r_work, r_work);

         /* Compute rel_resnorm */
         rel_resnorm = resnorm / old_resnorm;

         if (my_id == 0 && print_level > 1)
         {
            nalu_hypre_printf("    %e          %e          %e\n", iter, resnorm, rel_resnorm);
         }

         /* Exit if convergence tolerance has been achieved */
         if (rel_resnorm >= tol)
         {
            break;
         }
      }

      /* Compute: x(k+1) = omega*G^T*G*r + x(k) */
      nalu_hypre_ParCSRMatrixMatvec(1.0, G, r_work, 0.0, z_work);
      nalu_hypre_ParCSRMatrixMatvec(omega, GT, z_work, 1.0, x);
   }

   if (logging > 1)
   {
      nalu_hypre_ParFSAIDataNumIterations(fsai_data) = iter;
      nalu_hypre_ParFSAIDataRelResNorm(fsai_data)    = rel_resnorm;
   }
   else
   {
      nalu_hypre_ParFSAIDataNumIterations(fsai_data) = 0;
      nalu_hypre_ParFSAIDataRelResNorm(fsai_data)    = 0.0;
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}
