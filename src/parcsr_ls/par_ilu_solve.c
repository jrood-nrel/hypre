/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ILU solve routine
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

/*--------------------------------------------------------------------
 * nalu_hypre_ILUSolve
 *
 * TODO (VPM): Change variable names of F_array and U_array
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolve( void               *ilu_vdata,
                nalu_hypre_ParCSRMatrix *A,
                nalu_hypre_ParVector    *f,
                nalu_hypre_ParVector    *u )
{
   MPI_Comm              comm               = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParILUData     *ilu_data           = (nalu_hypre_ParILUData*) ilu_vdata;

   /* Matrices */
   nalu_hypre_ParCSRMatrix   *matmL              = nalu_hypre_ParILUDataMatLModified(ilu_data);
   nalu_hypre_ParCSRMatrix   *matmU              = nalu_hypre_ParILUDataMatUModified(ilu_data);
   nalu_hypre_ParCSRMatrix   *matA               = nalu_hypre_ParILUDataMatA(ilu_data);
   nalu_hypre_ParCSRMatrix   *matL               = nalu_hypre_ParILUDataMatL(ilu_data);
   nalu_hypre_ParCSRMatrix   *matU               = nalu_hypre_ParILUDataMatU(ilu_data);
   nalu_hypre_ParCSRMatrix   *matS               = nalu_hypre_ParILUDataMatS(ilu_data);
   NALU_HYPRE_Real           *matD               = nalu_hypre_ParILUDataMatD(ilu_data);
   NALU_HYPRE_Real           *matmD              = nalu_hypre_ParILUDataMatDModified(ilu_data);

   /* Vectors */
   NALU_HYPRE_Int             ilu_type           = nalu_hypre_ParILUDataIluType(ilu_data);
   NALU_HYPRE_Int            *perm               = nalu_hypre_ParILUDataPerm(ilu_data);
   NALU_HYPRE_Int            *qperm              = nalu_hypre_ParILUDataQPerm(ilu_data);
   nalu_hypre_ParVector      *F_array            = nalu_hypre_ParILUDataF(ilu_data);
   nalu_hypre_ParVector      *U_array            = nalu_hypre_ParILUDataU(ilu_data);

   /* Device data */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_CSRMatrix      *matALU_d           = nalu_hypre_ParILUDataMatAILUDevice(ilu_data);
   nalu_hypre_CSRMatrix      *matBLU_d           = nalu_hypre_ParILUDataMatBILUDevice(ilu_data);
   nalu_hypre_CSRMatrix      *matE_d             = nalu_hypre_ParILUDataMatEDevice(ilu_data);
   nalu_hypre_CSRMatrix      *matF_d             = nalu_hypre_ParILUDataMatFDevice(ilu_data);
   nalu_hypre_ParCSRMatrix   *Aperm              = nalu_hypre_ParILUDataAperm(ilu_data);
   nalu_hypre_Vector         *Adiag_diag         = nalu_hypre_ParILUDataADiagDiag(ilu_data);
   nalu_hypre_Vector         *Sdiag_diag         = nalu_hypre_ParILUDataSDiagDiag(ilu_data);
   nalu_hypre_ParVector      *Ztemp              = nalu_hypre_ParILUDataZTemp(ilu_data);
   NALU_HYPRE_Int             test_opt           = nalu_hypre_ParILUDataTestOption(ilu_data);
#endif

   /* Solver settings */
   NALU_HYPRE_Real            tol                = nalu_hypre_ParILUDataTol(ilu_data);
   NALU_HYPRE_Int             logging            = nalu_hypre_ParILUDataLogging(ilu_data);
   NALU_HYPRE_Int             print_level        = nalu_hypre_ParILUDataPrintLevel(ilu_data);
   NALU_HYPRE_Int             max_iter           = nalu_hypre_ParILUDataMaxIter(ilu_data);
   NALU_HYPRE_Int             tri_solve          = nalu_hypre_ParILUDataTriSolve(ilu_data);
   NALU_HYPRE_Int             lower_jacobi_iters = nalu_hypre_ParILUDataLowerJacobiIters(ilu_data);
   NALU_HYPRE_Int             upper_jacobi_iters = nalu_hypre_ParILUDataUpperJacobiIters(ilu_data);
   NALU_HYPRE_Real           *norms              = nalu_hypre_ParILUDataRelResNorms(ilu_data);
   nalu_hypre_ParVector      *Ftemp              = nalu_hypre_ParILUDataFTemp(ilu_data);
   nalu_hypre_ParVector      *Utemp              = nalu_hypre_ParILUDataUTemp(ilu_data);
   nalu_hypre_ParVector      *Xtemp              = nalu_hypre_ParILUDataXTemp(ilu_data);
   nalu_hypre_ParVector      *Ytemp              = nalu_hypre_ParILUDataYTemp(ilu_data);
   NALU_HYPRE_Real           *fext               = nalu_hypre_ParILUDataFExt(ilu_data);
   NALU_HYPRE_Real           *uext               = nalu_hypre_ParILUDataUExt(ilu_data);
   nalu_hypre_ParVector      *residual;
   NALU_HYPRE_Real            alpha              = -1.0;
   NALU_HYPRE_Real            beta               = 1.0;
   NALU_HYPRE_Real            conv_factor        = 0.0;
   NALU_HYPRE_Real            resnorm            = 1.0;
   NALU_HYPRE_Real            init_resnorm       = 0.0;
   NALU_HYPRE_Real            rel_resnorm;
   NALU_HYPRE_Real            rhs_norm           = 0.0;
   NALU_HYPRE_Real            old_resnorm;
   NALU_HYPRE_Real            ieee_check         = 0.0;
   NALU_HYPRE_Real            operat_cmplxty     = nalu_hypre_ParILUDataOperatorComplexity(ilu_data);
   NALU_HYPRE_Int             Solve_err_flag;
   NALU_HYPRE_Int             iter, num_procs, my_id;

   /* problem size */
   NALU_HYPRE_Int             n                  = nalu_hypre_ParCSRMatrixNumRows(A);
   NALU_HYPRE_Int             nLU                = nalu_hypre_ParILUDataNLU(ilu_data);
   NALU_HYPRE_Int            *u_end              = nalu_hypre_ParILUDataUEnd(ilu_data);

   /* Schur system solve */
   NALU_HYPRE_Solver          schur_solver       = nalu_hypre_ParILUDataSchurSolver(ilu_data);
   NALU_HYPRE_Solver          schur_precond      = nalu_hypre_ParILUDataSchurPrecond(ilu_data);
   nalu_hypre_ParVector      *rhs                = nalu_hypre_ParILUDataRhs(ilu_data);
   nalu_hypre_ParVector      *x                  = nalu_hypre_ParILUDataX(ilu_data);

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParVectorMemoryLocation(f) );

   /* VPM: Placeholder check to avoid -Wunused-variable warning. TODO: remove this */
   if (exec != NALU_HYPRE_EXEC_DEVICE && exec != NALU_HYPRE_EXEC_HOST)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Need to run either on host or device!");
      return nalu_hypre_error_flag;
   }
#endif

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (logging > 1)
   {
      residual = nalu_hypre_ParILUDataResidual(ilu_data);
   }

   nalu_hypre_ParILUDataNumIterations(ilu_data) = 0;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1)
   {
      nalu_hypre_ILUWriteSolverParams(ilu_data);
   }

   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;

   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1 && tol > 0.)
   {
      nalu_hypre_printf("\n\n ILU SOLVER SOLUTION INFO:\n");
   }

   /*-----------------------------------------------------------------------
    *    Compute initial residual and print
    *-----------------------------------------------------------------------*/

   if (print_level > 1 || logging > 1 || tol > 0.)
   {
      if (logging > 1)
      {
         nalu_hypre_ParVectorCopy(f, residual);
         if (tol > 0.0)
         {
            nalu_hypre_ParCSRMatrixMatvec(alpha, A, u, beta, residual);
         }
         resnorm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(residual, residual));
      }
      else
      {
         nalu_hypre_ParVectorCopy(f, Ftemp);
         if (tol > 0.0)
         {
            nalu_hypre_ParCSRMatrixMatvec(alpha, A, u, beta, Ftemp);
         }
         resnorm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(Ftemp, Ftemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resnorm != 0.)
      {
         ieee_check = resnorm / resnorm; /* INF -> NaN conversion */
      }
      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (print_level > 0)
         {
            nalu_hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            nalu_hypre_printf("ERROR -- nalu_hypre_ILUSolve: INFs and/or NaNs detected in input.\n");
            nalu_hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            nalu_hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
         NALU_HYPRE_ANNOTATE_FUNC_END;

         return nalu_hypre_error_flag;
      }

      init_resnorm = resnorm;
      rhs_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(f, f));
      if (rhs_norm > NALU_HYPRE_REAL_EPSILON)
      {
         rel_resnorm = init_resnorm / rhs_norm;
      }
      else
      {
         /* rhs is zero, return a zero solution */
         nalu_hypre_ParVectorSetConstantValues(U_array, 0.0);
         if (logging > 0)
         {
            rel_resnorm = 0.0;
            nalu_hypre_ParILUDataFinalRelResidualNorm(ilu_data) = rel_resnorm;
         }
         NALU_HYPRE_ANNOTATE_FUNC_END;

         return nalu_hypre_error_flag;
      }
   }
   else
   {
      rel_resnorm = 1.;
   }

   if (my_id == 0 && print_level > 1)
   {
      nalu_hypre_printf("                                            relative\n");
      nalu_hypre_printf("               residual        factor       residual\n");
      nalu_hypre_printf("               --------        ------       --------\n");
      nalu_hypre_printf("    Initial    %e                 %e\n", init_resnorm,
                   rel_resnorm);
   }

   matA    = A;
   U_array = u;
   F_array = f;

   /************** Main Solver Loop - always do 1 iteration ************/
   iter = 0;

   while ((rel_resnorm >= tol || iter < 1) &&
          (iter < max_iter))
   {
      /* Do one solve on LU*e = r */
      switch (ilu_type)
      {
      case 0: case 1: default:
            /* TODO (VPM): Encapsulate host and device functions into a single one */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               /* Apply GPU-accelerated LU solve - BJ-ILU0 */
               if (tri_solve == 1)
               {
                  nalu_hypre_ILUSolveLUDevice(matA, matBLU_d, F_array, U_array, perm, Utemp, Ftemp);
               }
               else
               {
                  nalu_hypre_ILUSolveLUIterDevice(matA, matBLU_d, F_array, U_array, perm,
                                             Utemp, Ftemp, Ztemp, &Adiag_diag,
                                             lower_jacobi_iters, upper_jacobi_iters);

                  /* Assign this now, in case it was set in method above */
                  nalu_hypre_ParILUDataADiagDiag(ilu_data) = Adiag_diag;
               }
            }
            else
#endif
            {
               /* BJ - nalu_hypre_ilu */
               if (tri_solve == 1)
               {
                  nalu_hypre_ILUSolveLU(matA, F_array, U_array, perm, n,
                                   matL, matD, matU, Utemp, Ftemp);
               }
               else
               {
                  nalu_hypre_ILUSolveLUIter(matA, F_array, U_array, perm, n,
                                       matL, matD, matU, Utemp, Ftemp, Xtemp,
                                       lower_jacobi_iters, upper_jacobi_iters);
               }
            }
            break;

         case 10: case 11:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               /* Apply GPU-accelerated GMRES-ILU solve */
               if (tri_solve == 1)
               {
                  nalu_hypre_ILUSolveSchurGMRESDevice(matA, F_array, U_array, perm, nLU, matS,
                                                 Utemp, Ftemp, schur_solver, schur_precond,
                                                 rhs, x, u_end, matBLU_d, matE_d, matF_d);
               }
               else
               {
                  nalu_hypre_ILUSolveSchurGMRESJacIterDevice(matA, F_array, U_array, perm, nLU, matS,
                                                        Utemp, Ftemp, schur_solver, schur_precond,
                                                        rhs, x, u_end, matBLU_d, matE_d, matF_d,
                                                        Ztemp, &Adiag_diag, &Sdiag_diag,
                                                        lower_jacobi_iters, upper_jacobi_iters);

                  /* Assign this now, in case it was set in method above */
                  nalu_hypre_ParILUDataADiagDiag(ilu_data) = Adiag_diag;
                  nalu_hypre_ParILUDataSDiagDiag(ilu_data) = Sdiag_diag;
               }
            }
            else
#endif
            {
               nalu_hypre_ILUSolveSchurGMRES(matA, F_array, U_array, perm, perm, nLU,
                                        matL, matD, matU, matS, Utemp, Ftemp,
                                        schur_solver, schur_precond, rhs, x, u_end);
            }
            break;

         case 20: case 21:
#if defined(NALU_HYPRE_USING_GPU) && !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                                 "NSH+ILU solve on device runs requires unified memory!");
               return nalu_hypre_error_flag;
            }
#endif
            /* NSH+ILU */
            nalu_hypre_ILUSolveSchurNSH(matA, F_array, U_array, perm, nLU, matL, matD, matU, matS,
                                   Utemp, Ftemp, schur_solver, rhs, x, u_end);
            break;

         case 30: case 31:
#if defined(NALU_HYPRE_USING_GPU) && !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                                 "RAS+ILU solve on device runs requires unified memory!");
               return nalu_hypre_error_flag;
            }
#endif
            /* RAS */
            nalu_hypre_ILUSolveLURAS(matA, F_array, U_array, perm, matL, matD, matU,
                                Utemp, Utemp, fext, uext);
            break;

         case 40: case 41:
#if defined(NALU_HYPRE_USING_GPU) && !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                                 "ddPQ+GMRES+ILU solve on device runs requires unified memory!");
               return nalu_hypre_error_flag;
            }
#endif

            /* ddPQ + GMRES + nalu_hypre_ilu[k,t]() */
            nalu_hypre_ILUSolveSchurGMRES(matA, F_array, U_array, perm, qperm, nLU,
                                     matL, matD, matU, matS, Utemp, Ftemp,
                                     schur_solver, schur_precond, rhs, x, u_end);
            break;

         case 50:
            /* GMRES-RAP */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               nalu_hypre_ILUSolveRAPGMRESDevice(matA, F_array, U_array, perm, nLU, matS, Utemp, Ftemp,
                                            Xtemp, Ytemp, schur_solver, schur_precond, rhs, x,
                                            u_end, Aperm, matALU_d, matBLU_d, matE_d, matF_d,
                                            test_opt);
            }
            else
#endif
            {
               nalu_hypre_ILUSolveRAPGMRESHost(matA, F_array, U_array, perm, nLU, matL, matD, matU,
                                          matmL, matmD, matmU, Utemp, Ftemp, Xtemp, Ytemp,
                                          schur_solver, schur_precond, rhs, x, u_end);
            }
            break;
      }

      /*---------------------------------------------------------------
       *    Compute residual and residual norm
       *----------------------------------------------------------------*/

      if (print_level > 1 || logging > 1 || tol > 0.)
      {
         old_resnorm = resnorm;

         if (logging > 1)
         {
            nalu_hypre_ParVectorCopy(F_array, residual);
            nalu_hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, residual);
            resnorm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(residual, residual));
         }
         else
         {
            nalu_hypre_ParVectorCopy(F_array, Ftemp);
            nalu_hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, Ftemp);
            resnorm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(Ftemp, Ftemp));
         }

         if (old_resnorm)
         {
            conv_factor = resnorm / old_resnorm;
         }
         else
         {
            conv_factor = resnorm;
         }

         if (rhs_norm > NALU_HYPRE_REAL_EPSILON)
         {
            rel_resnorm = resnorm / rhs_norm;
         }
         else
         {
            rel_resnorm = resnorm;
         }

         norms[iter] = rel_resnorm;
      }

      ++iter;
      nalu_hypre_ParILUDataNumIterations(ilu_data) = iter;
      nalu_hypre_ParILUDataFinalRelResidualNorm(ilu_data) = rel_resnorm;

      if (my_id == 0 && print_level > 1)
      {
         nalu_hypre_printf("    ILUSolve %2d   %e    %f     %e \n", iter,
                      resnorm, conv_factor, rel_resnorm);
      }
   }

   /* check convergence within max_iter */
   if (iter == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      nalu_hypre_error(NALU_HYPRE_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Print closing statistics
    *    Add operator and grid complexity stats
    *-----------------------------------------------------------------------*/

   if (iter > 0 && init_resnorm)
   {
      conv_factor = nalu_hypre_pow((resnorm / init_resnorm), (1.0 / (NALU_HYPRE_Real) iter));
   }
   else
   {
      conv_factor = 1.;
   }

   if (print_level > 1)
   {
      /*** compute operator and grid complexity (fill factor) here ?? ***/
      if (my_id == 0)
      {
         if (Solve_err_flag == 1)
         {
            nalu_hypre_printf("\n\n==============================================");
            nalu_hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
            nalu_hypre_printf("      within the allowed %d iterations\n", max_iter);
            nalu_hypre_printf("==============================================");
         }
         nalu_hypre_printf("\n\n Average Convergence Factor = %f \n", conv_factor);
         nalu_hypre_printf("                operator = %f\n", operat_cmplxty);
      }
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ILUSolveSchurGMRES
 *
 * Schur Complement solve with GMRES on schur complement
 *
 * ParCSRMatrix S is already built in ilu data sturcture, here directly
 * use S, L, D and U factors only have local scope (no off-diag terms)
 * so apart from the residual calculation (which uses A), the solves
 * with the L and U factors are local.
 *
 * S is the global Schur complement
 * schur_solver is a GMRES solver
 * schur_precond is the ILU preconditioner for GMRES
 * rhs and x are helper vector for solving Schur system
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveSchurGMRES(nalu_hypre_ParCSRMatrix *A,
                         nalu_hypre_ParVector    *f,
                         nalu_hypre_ParVector    *u,
                         NALU_HYPRE_Int          *perm,
                         NALU_HYPRE_Int          *qperm,
                         NALU_HYPRE_Int           nLU,
                         nalu_hypre_ParCSRMatrix *L,
                         NALU_HYPRE_Real         *D,
                         nalu_hypre_ParCSRMatrix *U,
                         nalu_hypre_ParCSRMatrix *S,
                         nalu_hypre_ParVector    *ftemp,
                         nalu_hypre_ParVector    *utemp,
                         NALU_HYPRE_Solver        schur_solver,
                         NALU_HYPRE_Solver        schur_precond,
                         nalu_hypre_ParVector    *rhs,
                         nalu_hypre_ParVector    *x,
                         NALU_HYPRE_Int          *u_end)
{
   /* Data objects for L and U */
   nalu_hypre_CSRMatrix   *L_diag      = nalu_hypre_ParCSRMatrixDiag(L);
   NALU_HYPRE_Real        *L_diag_data = nalu_hypre_CSRMatrixData(L_diag);
   NALU_HYPRE_Int         *L_diag_i    = nalu_hypre_CSRMatrixI(L_diag);
   NALU_HYPRE_Int         *L_diag_j    = nalu_hypre_CSRMatrixJ(L_diag);
   nalu_hypre_CSRMatrix   *U_diag      = nalu_hypre_ParCSRMatrixDiag(U);
   NALU_HYPRE_Real        *U_diag_data = nalu_hypre_CSRMatrixData(U_diag);
   NALU_HYPRE_Int         *U_diag_i    = nalu_hypre_CSRMatrixI(U_diag);
   NALU_HYPRE_Int         *U_diag_j    = nalu_hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   nalu_hypre_Vector      *utemp_local = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real        *utemp_data  = nalu_hypre_VectorData(utemp_local);
   nalu_hypre_Vector      *ftemp_local = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real        *ftemp_data  = nalu_hypre_VectorData(ftemp_local);
   NALU_HYPRE_Real         alpha       = -1.0;
   NALU_HYPRE_Real         beta        = 1.0;
   NALU_HYPRE_Int          i, j, k1, k2, col;

   /* Problem size */
   NALU_HYPRE_Int          n           = nalu_hypre_CSRMatrixNumRows(L_diag);
   nalu_hypre_Vector      *rhs_local;
   NALU_HYPRE_Real        *rhs_data;
   nalu_hypre_Vector      *x_local;
   NALU_HYPRE_Real        *x_data;

   /* Compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */
   /* now update with L to solve */
   for (i = 0 ; i < nLU ; i ++)
   {
      utemp_data[qperm[i]] = ftemp_data[perm[i]];
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         utemp_data[qperm[i]] -= L_diag_data[j] * utemp_data[qperm[L_diag_j[j]]];
      }
   }

   /* 2nd need to compute g'i = gi - Ei*UBi^-1*xi
    * now put g'i into the f_temp lower
    */
   for (i = nLU ; i < n ; i ++)
   {
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = L_diag_j[j];
         ftemp_data[perm[i]] -= L_diag_data[j] * utemp_data[qperm[col]];
      }
   }

   /* 3rd need to solve global Schur Complement Sy = g'
    * for now only solve the local system
    * solve y put in u_temp lower
    * only solve whe S is not NULL
    */
   if (S)
   {
      /*initialize solution to zero for residual equation */
      nalu_hypre_ParVectorSetConstantValues(x, 0.0);

      /* setup vectors for solve */
      rhs_local   = nalu_hypre_ParVectorLocalVector(rhs);
      rhs_data    = nalu_hypre_VectorData(rhs_local);
      x_local     = nalu_hypre_ParVectorLocalVector(x);
      x_data      = nalu_hypre_VectorData(x_local);

      /* set rhs value */
      for (i = nLU ; i < n ; i ++)
      {
         rhs_data[i - nLU] = ftemp_data[perm[i]];
      }

      /* solve */
      NALU_HYPRE_GMRESSolve(schur_solver, (NALU_HYPRE_Matrix)S, (NALU_HYPRE_Vector)rhs, (NALU_HYPRE_Vector)x);

      /* copy value back to original */
      for (i = nLU ; i < n ; i ++)
      {
         utemp_data[qperm[i]] = x_data[i - nLU];
      }
   }

   /* 4th need to compute zi = xi - LBi^-1*Fi*yi
    * put zi in f_temp upper
    * only do this computation when nLU < n
    * U is unsorted, search is expensive when unnecessary
    */
   if (nLU < n)
   {
      for (i = 0 ; i < nLU ; i ++)
      {
         ftemp_data[perm[i]] = utemp_data[qperm[i]];
         k1 = u_end[i] ; k2 = U_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = U_diag_j[j];
            ftemp_data[perm[i]] -= U_diag_data[j] * utemp_data[qperm[col]];
         }
      }
      for (i = 0 ; i < nLU ; i ++)
      {
         utemp_data[qperm[i]] = ftemp_data[perm[i]];
      }
   }

   /* 5th need to solve UBi*ui = zi */
   /* put result in u_temp upper */
   for (i = nLU - 1 ; i >= 0 ; i --)
   {
      k1 = U_diag_i[i] ; k2 = u_end[i];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = U_diag_j[j];
         utemp_data[qperm[i]] -= U_diag_data[j] * utemp_data[qperm[col]];
      }
      utemp_data[qperm[i]] *= D[i];
   }

   /* done, now everything are in u_temp, update solution */
   nalu_hypre_ParVectorAxpy(beta, utemp, u);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ILUSolveSchurNSH
 *
 * Newton-Schulz-Hotelling solve
 *
 * ParCSRMatrix S is already built in ilu data sturcture
 *
 * S here is the INVERSE of Schur Complement
 * L, D and U factors only have local scope (no off-diag terms)
 *  so apart from the residual calculation (which uses A), the solves
 *  with the L and U factors are local.
 * S is the inverse global Schur complement
 * rhs and x are helper vector for solving Schur system
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveSchurNSH(nalu_hypre_ParCSRMatrix *A,
                       nalu_hypre_ParVector    *f,
                       nalu_hypre_ParVector    *u,
                       NALU_HYPRE_Int          *perm,
                       NALU_HYPRE_Int           nLU,
                       nalu_hypre_ParCSRMatrix *L,
                       NALU_HYPRE_Real         *D,
                       nalu_hypre_ParCSRMatrix *U,
                       nalu_hypre_ParCSRMatrix *S,
                       nalu_hypre_ParVector    *ftemp,
                       nalu_hypre_ParVector    *utemp,
                       NALU_HYPRE_Solver        schur_solver,
                       nalu_hypre_ParVector    *rhs,
                       nalu_hypre_ParVector    *x,
                       NALU_HYPRE_Int          *u_end)
{
   /* data objects for L and U */
   nalu_hypre_CSRMatrix   *L_diag      = nalu_hypre_ParCSRMatrixDiag(L);
   NALU_HYPRE_Real        *L_diag_data = nalu_hypre_CSRMatrixData(L_diag);
   NALU_HYPRE_Int         *L_diag_i    = nalu_hypre_CSRMatrixI(L_diag);
   NALU_HYPRE_Int         *L_diag_j    = nalu_hypre_CSRMatrixJ(L_diag);
   nalu_hypre_CSRMatrix   *U_diag      = nalu_hypre_ParCSRMatrixDiag(U);
   NALU_HYPRE_Real        *U_diag_data = nalu_hypre_CSRMatrixData(U_diag);
   NALU_HYPRE_Int         *U_diag_i    = nalu_hypre_CSRMatrixI(U_diag);
   NALU_HYPRE_Int         *U_diag_j    = nalu_hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   nalu_hypre_Vector      *utemp_local = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real        *utemp_data  = nalu_hypre_VectorData(utemp_local);
   nalu_hypre_Vector      *ftemp_local = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real        *ftemp_data  = nalu_hypre_VectorData(ftemp_local);
   NALU_HYPRE_Real         alpha       = -1.0;
   NALU_HYPRE_Real         beta        = 1.0;
   NALU_HYPRE_Int          i, j, k1, k2, col;

   /* problem size */
   NALU_HYPRE_Int         n = nalu_hypre_CSRMatrixNumRows(L_diag);

   /* other data objects for computation */
   nalu_hypre_Vector      *rhs_local;
   NALU_HYPRE_Real        *rhs_data;
   nalu_hypre_Vector      *x_local;
   NALU_HYPRE_Real        *x_data;

   /* compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */
   /* now update with L to solve */
   for (i = 0 ; i < nLU ; i ++)
   {
      utemp_data[perm[i]] = ftemp_data[perm[i]];
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         utemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[L_diag_j[j]]];
      }
   }

   /* 2nd need to compute g'i = gi - Ei*UBi^-1*xi
    * now put g'i into the f_temp lower
    */
   for (i = nLU ; i < n ; i ++)
   {
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = L_diag_j[j];
         ftemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[col]];
      }
   }

   /* 3rd need to solve global Schur Complement Sy = g'
    * for now only solve the local system
    * solve y put in u_temp lower
    * only solve when S is not NULL
    */
   if (S)
   {
      /* Initialize solution to zero for residual equation */
      nalu_hypre_ParVectorSetConstantValues(x, 0.0);

      /* Setup vectors for solve */
      rhs_local = nalu_hypre_ParVectorLocalVector(rhs);
      rhs_data  = nalu_hypre_VectorData(rhs_local);
      x_local   = nalu_hypre_ParVectorLocalVector(x);
      x_data    = nalu_hypre_VectorData(x_local);

      /* set rhs value */
      for (i = nLU ; i < n ; i ++)
      {
         rhs_data[i - nLU] = ftemp_data[perm[i]];
      }

      /* Solve Schur system with approx inverse
       * x = S*rhs
       */
      nalu_hypre_NSHSolve(schur_solver, S, rhs, x);

      /* copy value back to original */
      for (i = nLU ; i < n ; i ++)
      {
         utemp_data[perm[i]] = x_data[i - nLU];
      }
   }

   /* 4th need to compute zi = xi - LBi^-1*yi
    * put zi in f_temp upper
    * only do this computation when nLU < n
    * U is unsorted, search is expensive when unnecessary
    */
   if (nLU < n)
   {
      for (i = 0 ; i < nLU ; i ++)
      {
         ftemp_data[perm[i]] = utemp_data[perm[i]];
         k1 = u_end[i] ; k2 = U_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = U_diag_j[j];
            ftemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[col]];
         }
      }
      for (i = 0 ; i < nLU ; i ++)
      {
         utemp_data[perm[i]] = ftemp_data[perm[i]];
      }
   }

   /* 5th need to solve UBi*ui = zi */
   /* put result in u_temp upper */
   for (i = nLU - 1 ; i >= 0 ; i --)
   {
      k1 = U_diag_i[i] ; k2 = u_end[i];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = U_diag_j[j];
         utemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[col]];
      }
      utemp_data[perm[i]] *= D[i];
   }

   /* Done, now everything are in u_temp, update solution */
   nalu_hypre_ParVectorAxpy(beta, utemp, u);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ILUSolveLU
 *
 * Incomplete LU solve
 *
 * L, D and U factors only have local scope (no off-diagterms)
 *  so apart from the residual calculation (which uses A),
 *  the solves with the L and U factors are local.
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveLU(nalu_hypre_ParCSRMatrix *A,
                 nalu_hypre_ParVector    *f,
                 nalu_hypre_ParVector    *u,
                 NALU_HYPRE_Int          *perm,
                 NALU_HYPRE_Int           nLU,
                 nalu_hypre_ParCSRMatrix *L,
                 NALU_HYPRE_Real         *D,
                 nalu_hypre_ParCSRMatrix *U,
                 nalu_hypre_ParVector    *ftemp,
                 nalu_hypre_ParVector    *utemp)
{
   /* data objects for L and U */
   nalu_hypre_CSRMatrix *L_diag      = nalu_hypre_ParCSRMatrixDiag(L);
   NALU_HYPRE_Real      *L_diag_data = nalu_hypre_CSRMatrixData(L_diag);
   NALU_HYPRE_Int       *L_diag_i    = nalu_hypre_CSRMatrixI(L_diag);
   NALU_HYPRE_Int       *L_diag_j    = nalu_hypre_CSRMatrixJ(L_diag);
   nalu_hypre_CSRMatrix *U_diag      = nalu_hypre_ParCSRMatrixDiag(U);
   NALU_HYPRE_Real      *U_diag_data = nalu_hypre_CSRMatrixData(U_diag);
   NALU_HYPRE_Int       *U_diag_i    = nalu_hypre_CSRMatrixI(U_diag);
   NALU_HYPRE_Int       *U_diag_j    = nalu_hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   nalu_hypre_Vector    *utemp_local = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real      *utemp_data  = nalu_hypre_VectorData(utemp_local);
   nalu_hypre_Vector    *ftemp_local = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real      *ftemp_data  = nalu_hypre_VectorData(ftemp_local);
   NALU_HYPRE_Real       alpha       = -1.0;
   NALU_HYPRE_Real       beta        = 1.0;
   NALU_HYPRE_Int        i, j, k1, k2;

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
    */
   //nalu_hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* L solve - Forward solve */
   /* copy rhs to account for diagonal of L (which is identity) */
   for (i = 0; i < nLU; i++)
   {
      utemp_data[perm[i]] = ftemp_data[perm[i]];
   }

   /* Update with remaining (off-diagonal) entries of L */
   for ( i = 0; i < nLU; i++ )
   {
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1; j < k2; j++)
      {
         utemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[L_diag_j[j]]];
      }
   }

   /*-------------------- U solve - Backward substitution */
   for ( i = nLU - 1; i >= 0; i-- )
   {
      /* first update with the remaining (off-diagonal) entries of U */
      k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
      for (j = k1; j < k2; j++)
      {
         utemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[U_diag_j[j]]];
      }

      /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
      utemp_data[perm[i]] *= D[i];
   }

   /* Update solution */
   nalu_hypre_ParVectorAxpy(beta, utemp, u);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ILUSolveLUIter
 *
 * Iterative incomplete LU solve
 *
 * L, D and U factors only have local scope (no off-diag terms)
 *  so apart from the residual calculation (which uses A), the solves
 *  with the L and U factors are local.
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveLUIter(nalu_hypre_ParCSRMatrix *A,
                     nalu_hypre_ParVector    *f,
                     nalu_hypre_ParVector    *u,
                     NALU_HYPRE_Int          *perm,
                     NALU_HYPRE_Int           nLU,
                     nalu_hypre_ParCSRMatrix *L,
                     NALU_HYPRE_Real         *D,
                     nalu_hypre_ParCSRMatrix *U,
                     nalu_hypre_ParVector    *ftemp,
                     nalu_hypre_ParVector    *utemp,
                     nalu_hypre_ParVector    *xtemp,
                     NALU_HYPRE_Int           lower_jacobi_iters,
                     NALU_HYPRE_Int           upper_jacobi_iters)
{
   /* Data objects for L and U */
   nalu_hypre_CSRMatrix *L_diag      = nalu_hypre_ParCSRMatrixDiag(L);
   NALU_HYPRE_Real      *L_diag_data = nalu_hypre_CSRMatrixData(L_diag);
   NALU_HYPRE_Int       *L_diag_i    = nalu_hypre_CSRMatrixI(L_diag);
   NALU_HYPRE_Int       *L_diag_j    = nalu_hypre_CSRMatrixJ(L_diag);
   nalu_hypre_CSRMatrix *U_diag      = nalu_hypre_ParCSRMatrixDiag(U);
   NALU_HYPRE_Real      *U_diag_data = nalu_hypre_CSRMatrixData(U_diag);
   NALU_HYPRE_Int       *U_diag_i    = nalu_hypre_CSRMatrixI(U_diag);
   NALU_HYPRE_Int       *U_diag_j    = nalu_hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   nalu_hypre_Vector    *utemp_local = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real      *utemp_data  = nalu_hypre_VectorData(utemp_local);
   nalu_hypre_Vector    *ftemp_local = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real      *ftemp_data  = nalu_hypre_VectorData(ftemp_local);
   nalu_hypre_Vector    *xtemp_local = nalu_hypre_ParVectorLocalVector(xtemp);
   NALU_HYPRE_Real      *xtemp_data  = nalu_hypre_VectorData(xtemp_local);

   /* Local variables */
   NALU_HYPRE_Real       alpha       = -1.0;
   NALU_HYPRE_Real       beta        = 1.0;
   NALU_HYPRE_Real       sum;
   NALU_HYPRE_Int        i, j, k1, k2, kk;

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
    */
   //nalu_hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* L solve - Forward solve */
   /* copy rhs to account for diagonal of L (which is identity) */

   /* Initialize iteration to 0 */
   for ( i = 0; i < nLU; i++ )
   {
      utemp_data[perm[i]] = 0.0;
   }

   /* Jacobi iteration loop */
   for ( kk = 0; kk < lower_jacobi_iters; kk++ )
   {
      /* u^{k+1} = f - Lu^k */

      /* Do a SpMV with L and save the results in xtemp */
      for ( i = 0; i < nLU; i++ )
      {
         sum = 0.0;
         k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
         for (j = k1; j < k2; j++)
         {
            sum += L_diag_data[j] * utemp_data[perm[L_diag_j[j]]];
         }
         xtemp_data[i] = sum;
      }

      for ( i = 0; i < nLU; i++ )
      {
         utemp_data[perm[i]] = ftemp_data[perm[i]] - xtemp_data[i];
      }
   } /* end jacobi loop */

   /* Initialize iteration to 0 */
   for ( i = 0; i < nLU; i++ )
   {
      /* this should is doable without the permutation */
      //ftemp_data[perm[i]] = utemp_data[perm[i]];
      ftemp_data[perm[i]] = 0.0;
   }

   /* Jacobi iteration loop */
   for ( kk = 0; kk < upper_jacobi_iters; kk++ )
   {
      /* u^{k+1} = f - Uu^k */

      /* Do a SpMV with U and save the results in xtemp */
      for ( i = 0; i < nLU; ++i )
      {
         sum = 0.0;
         k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
         for (j = k1; j < k2; j++)
         {
            sum += U_diag_data[j] * ftemp_data[perm[U_diag_j[j]]];
         }
         xtemp_data[i] = sum;
      }

      for ( i = 0; i < nLU; ++i )
      {
         ftemp_data[perm[i]] = D[i] * (utemp_data[perm[i]] - xtemp_data[i]);
      }
   } /* end jacobi loop */

   /* Update solution */
   nalu_hypre_ParVectorAxpy(beta, ftemp, u);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ILUSolveLURAS
 *
 * Incomplete LU solve RAS
 *
 * L, D and U factors only have local scope (no off-diag terms)
 *  so apart from the residual calculation (which uses A), the solves
 *  with the L and U factors are local.
 * fext and uext are tempory arrays for external data
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveLURAS(nalu_hypre_ParCSRMatrix *A,
                    nalu_hypre_ParVector    *f,
                    nalu_hypre_ParVector    *u,
                    NALU_HYPRE_Int          *perm,
                    nalu_hypre_ParCSRMatrix *L,
                    NALU_HYPRE_Real         *D,
                    nalu_hypre_ParCSRMatrix *U,
                    nalu_hypre_ParVector    *ftemp,
                    nalu_hypre_ParVector    *utemp,
                    NALU_HYPRE_Real         *fext,
                    NALU_HYPRE_Real         *uext)
{
   /* Parallel info */
   nalu_hypre_ParCSRCommPkg        *comm_pkg;
   nalu_hypre_ParCSRCommHandle     *comm_handle;
   NALU_HYPRE_Int                   num_sends, begin, end;


   /* Data objects for L and U */
   nalu_hypre_CSRMatrix            *L_diag      = nalu_hypre_ParCSRMatrixDiag(L);
   NALU_HYPRE_Real                 *L_diag_data = nalu_hypre_CSRMatrixData(L_diag);
   NALU_HYPRE_Int                  *L_diag_i    = nalu_hypre_CSRMatrixI(L_diag);
   NALU_HYPRE_Int                  *L_diag_j    = nalu_hypre_CSRMatrixJ(L_diag);
   nalu_hypre_CSRMatrix            *U_diag      = nalu_hypre_ParCSRMatrixDiag(U);
   NALU_HYPRE_Real                 *U_diag_data = nalu_hypre_CSRMatrixData(U_diag);
   NALU_HYPRE_Int                  *U_diag_i    = nalu_hypre_CSRMatrixI(U_diag);
   NALU_HYPRE_Int                  *U_diag_j    = nalu_hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   NALU_HYPRE_Int                   n           = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixDiag(A));
   NALU_HYPRE_Int                   m           = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A));
   NALU_HYPRE_Int                   n_total     = m + n;
   nalu_hypre_Vector               *utemp_local = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real                 *utemp_data  = nalu_hypre_VectorData(utemp_local);
   nalu_hypre_Vector               *ftemp_local = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real                 *ftemp_data  = nalu_hypre_VectorData(ftemp_local);

   /* Local variables */
   NALU_HYPRE_Int                   idx, jcol, col;
   NALU_HYPRE_Int                   i, j, k1, k2;
   NALU_HYPRE_Real                  alpha = -1.0;
   NALU_HYPRE_Real                  beta  = 1.0;

   /* prepare for communication */
   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);

   /* setup if not yet built */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
    */
   //nalu_hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* communication to get external data */

   /* get total num of send */
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   begin     = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
   end       = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* copy new index into send_buf */
   for (i = begin ; i < end ; i ++)
   {
      /* all we need is just send out data, we don't need to worry about the
       *    permutation of offd part, actually we don't need to worry about
       *    permutation at all
       * borrow uext as send buffer .
       */
      uext[i - begin] = ftemp_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

   /* main communication */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, uext, fext);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* L solve - Forward solve */
   for ( i = 0 ; i < n_total ; i ++)
   {
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      if ( i < n )
      {
         /* diag part */
         utemp_data[perm[i]] = ftemp_data[perm[i]];
         for (j = k1; j < k2; j++)
         {
            col = L_diag_j[j];
            if ( col < n )
            {
               utemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[col]];
            }
            else
            {
               jcol = col - n;
               utemp_data[perm[i]] -= L_diag_data[j] * uext[jcol];
            }
         }
      }
      else
      {
         /* offd part */
         idx = i - n;
         uext[idx] = fext[idx];
         for (j = k1; j < k2; j++)
         {
            col = L_diag_j[j];
            if (col < n)
            {
               uext[idx] -= L_diag_data[j] * utemp_data[perm[col]];
            }
            else
            {
               jcol = col - n;
               uext[idx] -= L_diag_data[j] * uext[jcol];
            }
         }
      }
   }

   /*-------------------- U solve - Backward substitution */
   for ( i = n_total - 1; i >= 0; i-- )
   {
      /* first update with the remaining (off-diagonal) entries of U */
      k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
      if ( i < n )
      {
         /* diag part */
         for (j = k1; j < k2; j++)
         {
            col = U_diag_j[j];
            if ( col < n )
            {
               utemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[col]];
            }
            else
            {
               jcol = col - n;
               utemp_data[perm[i]] -= U_diag_data[j] * uext[jcol];
            }
         }
         /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
         utemp_data[perm[i]] *= D[i];
      }
      else
      {
         /* 2nd part of offd */
         idx = i - n;
         for (j = k1; j < k2; j++)
         {
            col = U_diag_j[j];
            if ( col < n )
            {
               uext[idx] -= U_diag_data[j] * utemp_data[perm[col]];
            }
            else
            {
               jcol = col - n;
               uext[idx] -= U_diag_data[j] * uext[jcol];
            }
         }
         /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
         uext[idx] *= D[i];
      }
   }

   /* Update solution */
   nalu_hypre_ParVectorAxpy(beta, utemp, u);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ILUSolveRAPGMRESHost
 *
 * Solve with GMRES on schur complement, RAP style.
 *
 * ParCSRMatrix S is already built in ilu data sturcture, here directly
 * use S, L, D and U factors only have local scope (no off-diag terms)
 * so apart from the residual calculation (which uses A), the solves
 * with the L and U factors are local.
 *
 * S is the global Schur complement
 * schur_solver is a GMRES solver
 * schur_precond is the ILU preconditioner for GMRES
 * rhs and x are helper vector for solving Schur system
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSolveRAPGMRESHost(nalu_hypre_ParCSRMatrix *A,
                           nalu_hypre_ParVector    *f,
                           nalu_hypre_ParVector    *u,
                           NALU_HYPRE_Int          *perm,
                           NALU_HYPRE_Int           nLU,
                           nalu_hypre_ParCSRMatrix *L,
                           NALU_HYPRE_Real         *D,
                           nalu_hypre_ParCSRMatrix *U,
                           nalu_hypre_ParCSRMatrix *mL,
                           NALU_HYPRE_Real         *mD,
                           nalu_hypre_ParCSRMatrix *mU,
                           nalu_hypre_ParVector    *ftemp,
                           nalu_hypre_ParVector    *utemp,
                           nalu_hypre_ParVector    *xtemp,
                           nalu_hypre_ParVector    *ytemp,
                           NALU_HYPRE_Solver        schur_solver,
                           NALU_HYPRE_Solver        schur_precond,
                           nalu_hypre_ParVector    *rhs,
                           nalu_hypre_ParVector    *x,
                           NALU_HYPRE_Int          *u_end)
{
   /* data objects for L and U */
   nalu_hypre_CSRMatrix   *L_diag       = nalu_hypre_ParCSRMatrixDiag(L);
   NALU_HYPRE_Real        *L_diag_data  = nalu_hypre_CSRMatrixData(L_diag);
   NALU_HYPRE_Int         *L_diag_i     = nalu_hypre_CSRMatrixI(L_diag);
   NALU_HYPRE_Int         *L_diag_j     = nalu_hypre_CSRMatrixJ(L_diag);

   nalu_hypre_CSRMatrix   *U_diag       = nalu_hypre_ParCSRMatrixDiag(U);
   NALU_HYPRE_Real        *U_diag_data  = nalu_hypre_CSRMatrixData(U_diag);
   NALU_HYPRE_Int         *U_diag_i     = nalu_hypre_CSRMatrixI(U_diag);
   NALU_HYPRE_Int         *U_diag_j     = nalu_hypre_CSRMatrixJ(U_diag);

   nalu_hypre_CSRMatrix   *mL_diag      = nalu_hypre_ParCSRMatrixDiag(mL);
   NALU_HYPRE_Real        *mL_diag_data = nalu_hypre_CSRMatrixData(mL_diag);
   NALU_HYPRE_Int         *mL_diag_i    = nalu_hypre_CSRMatrixI(mL_diag);
   NALU_HYPRE_Int         *mL_diag_j    = nalu_hypre_CSRMatrixJ(mL_diag);

   nalu_hypre_CSRMatrix   *mU_diag      = nalu_hypre_ParCSRMatrixDiag(mU);
   NALU_HYPRE_Real        *mU_diag_data = nalu_hypre_CSRMatrixData(mU_diag);
   NALU_HYPRE_Int         *mU_diag_i    = nalu_hypre_CSRMatrixI(mU_diag);
   NALU_HYPRE_Int         *mU_diag_j    = nalu_hypre_CSRMatrixJ(mU_diag);

   /* Vectors */
   nalu_hypre_Vector      *utemp_local  = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real        *utemp_data   = nalu_hypre_VectorData(utemp_local);
   nalu_hypre_Vector      *ftemp_local  = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real        *ftemp_data   = nalu_hypre_VectorData(ftemp_local);
   nalu_hypre_Vector      *xtemp_local  = NULL;
   NALU_HYPRE_Real        *xtemp_data   = NULL;
   nalu_hypre_Vector      *ytemp_local  = NULL;
   NALU_HYPRE_Real        *ytemp_data   = NULL;

   NALU_HYPRE_Real         alpha = -1.0;
   NALU_HYPRE_Real         beta  = 1.0;
   NALU_HYPRE_Int          i, j, k1, k2, col;

   /* problem size */
   NALU_HYPRE_Int          n = nalu_hypre_CSRMatrixNumRows(L_diag);
   NALU_HYPRE_Int          m = n - nLU;

   /* other data objects for computation */
   nalu_hypre_Vector      *rhs_local;
   NALU_HYPRE_Real        *rhs_data;
   nalu_hypre_Vector      *x_local;
   NALU_HYPRE_Real        *x_data;

   /* xtemp might be null when we have no Schur complement */
   if (xtemp)
   {
      xtemp_local = nalu_hypre_ParVectorLocalVector(xtemp);
      xtemp_data  = nalu_hypre_VectorData(xtemp_local);
      ytemp_local = nalu_hypre_ParVectorLocalVector(ytemp);
      ytemp_data  = nalu_hypre_VectorData(ytemp_local);
   }

   /* Setup vectors for solve */
   if (m > 0)
   {
      rhs_local   = nalu_hypre_ParVectorLocalVector(rhs);
      rhs_data    = nalu_hypre_VectorData(rhs_local);
      x_local     = nalu_hypre_ParVectorLocalVector(x);
      x_data      = nalu_hypre_VectorData(x_local);
   }

   /* only support RAP with partial factorized W and Z */

   /* compute residual */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* A-smoothing f_temp = [UA \ LA \ (f_temp[perm])] */
   /* permuted L solve */
   for (i = 0 ; i < n ; i ++)
   {
      utemp_data[i] = ftemp_data[perm[i]];
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = L_diag_j[j];
         utemp_data[i] -= L_diag_data[j] * utemp_data[col];
      }
   }

   if (!xtemp)
   {
      /* in this case, we don't have a Schur complement */
      /* U solve */
      for (i = n - 1 ; i >= 0 ; i --)
      {
         ftemp_data[perm[i]] = utemp_data[i];
         k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = U_diag_j[j];
            ftemp_data[perm[i]] -= U_diag_data[j] * ftemp_data[perm[col]];
         }
         ftemp_data[perm[i]] *= D[i];
      }

      nalu_hypre_ParVectorAxpy(beta, ftemp, u);

      return nalu_hypre_error_flag;
   }

   /* U solve */
   for (i = n - 1 ; i >= 0 ; i --)
   {
      xtemp_data[perm[i]] = utemp_data[i];
      k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = U_diag_j[j];
         xtemp_data[perm[i]] -= U_diag_data[j] * xtemp_data[perm[col]];
      }
      xtemp_data[perm[i]] *= D[i];
   }

   /* coarse-grid correction */
   /* now f_temp is the result of A-smoothing
    * rhs = R*(b - Ax)
    * */
   // utemp = (ftemp - A*xtemp)
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, xtemp, beta, ftemp, utemp);

   // R = [-L21 L\inv, I]
   if (m > 0)
   {
      /* first is L solve */
      for (i = 0 ; i < nLU ; i ++)
      {
         ytemp_data[i] = utemp_data[perm[i]];
         k1 = mL_diag_i[i] ; k2 = mL_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = mL_diag_j[j];
            ytemp_data[i] -= mL_diag_data[j] * ytemp_data[col];
         }
      }

      /* apply -W * ytemp on this, and take care of the I part */
      for (i = nLU ; i < n ; i ++)
      {
         rhs_data[i - nLU] = utemp_data[perm[i]];
         k1 = mL_diag_i[i] ; k2 = u_end[i];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = mL_diag_j[j];
            rhs_data[i - nLU] -= mL_diag_data[j] * ytemp_data[col];
         }
      }
   }

   /* now the rhs is ready */
   nalu_hypre_SeqVectorSetConstantValues(x_local, 0.0);
   NALU_HYPRE_GMRESSolve(schur_solver,
                    (NALU_HYPRE_Matrix) schur_precond,
                    (NALU_HYPRE_Vector) rhs,
                    (NALU_HYPRE_Vector) x);

   if (m > 0)
   {
      /*
      for(i = 0 ; i < m ; i ++)
      {
         x_data[i] = rhs_data[i];
         k1 = u_end[i+nLU] ; k2 = mL_diag_i[i+nLU+1];
         for(j = k1 ; j < k2 ; j ++)
         {
            col = mL_diag_j[j];
            x_data[i] -= mL_diag_data[j] * x_data[col-nLU];
         }
      }

      for(i = m-1 ; i >= 0 ; i --)
      {
         rhs_data[i] = x_data[i];
         k1 = mU_diag_i[i+nLU] ; k2 = mU_diag_i[i+1+nLU];
         for(j = k1 ; j < k2 ; j ++)
         {
            col = mU_diag_j[j];
            rhs_data[i] -= mU_diag_data[j] * rhs_data[col-nLU];
         }
         rhs_data[i] *= mD[i];
      }
      */

      /* after solve, update x = x + Pv
       * that is, xtemp = xtemp + P*x
       */
      /* first compute P*x
       * P = [ -U\inv U_12 ]
       *     [  I          ]
       */
      /* matvec */
      for (i = 0 ; i < nLU ; i ++)
      {
         ytemp_data[i] = 0.0;
         k1 = u_end[i] ; k2 = mU_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = mU_diag_j[j];
            ytemp_data[i] -= mU_diag_data[j] * x_data[col - nLU];
         }
      }
      /* U solve */
      for (i = nLU - 1 ; i >= 0 ; i --)
      {
         ftemp_data[perm[i]] = ytemp_data[i];
         k1 = mU_diag_i[i] ; k2 = u_end[i];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = mU_diag_j[j];
            ftemp_data[perm[i]] -= mU_diag_data[j] * ftemp_data[perm[col]];
         }
         ftemp_data[perm[i]] *= mD[i];
      }

      /* update with I */
      for (i = nLU ; i < n ; i ++)
      {
         ftemp_data[perm[i]] = x_data[i - nLU];
      }
      nalu_hypre_ParVectorAxpy(beta, ftemp, u);
   }

   nalu_hypre_ParVectorAxpy(beta, xtemp, u);

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * NSH functions.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------
 * nalu_hypre_NSHSolve
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSolve( void               *nsh_vdata,
                nalu_hypre_ParCSRMatrix *A,
                nalu_hypre_ParVector    *f,
                nalu_hypre_ParVector    *u )
{
   MPI_Comm              comm           = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParNSHData     *nsh_data       = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParCSRMatrix   *matA           = nalu_hypre_ParNSHDataMatA(nsh_data);
   nalu_hypre_ParCSRMatrix   *matM           = nalu_hypre_ParNSHDataMatM(nsh_data);
   nalu_hypre_ParVector      *F_array        = nalu_hypre_ParNSHDataF(nsh_data);
   nalu_hypre_ParVector      *U_array        = nalu_hypre_ParNSHDataU(nsh_data);

   NALU_HYPRE_Real            tol            = nalu_hypre_ParNSHDataTol(nsh_data);
   NALU_HYPRE_Int             logging        = nalu_hypre_ParNSHDataLogging(nsh_data);
   NALU_HYPRE_Int             print_level    = nalu_hypre_ParNSHDataPrintLevel(nsh_data);
   NALU_HYPRE_Int             max_iter       = nalu_hypre_ParNSHDataMaxIter(nsh_data);
   NALU_HYPRE_Real           *norms          = nalu_hypre_ParNSHDataRelResNorms(nsh_data);
   nalu_hypre_ParVector      *Ftemp          = nalu_hypre_ParNSHDataFTemp(nsh_data);
   nalu_hypre_ParVector      *Utemp          = nalu_hypre_ParNSHDataUTemp(nsh_data);
   nalu_hypre_ParVector      *residual;

   NALU_HYPRE_Real            alpha          = -1.0;
   NALU_HYPRE_Real            beta           = 1.0;
   NALU_HYPRE_Real            conv_factor    = 0.0;
   NALU_HYPRE_Real            resnorm        = 1.0;
   NALU_HYPRE_Real            init_resnorm   = 0.0;
   NALU_HYPRE_Real            rel_resnorm;
   NALU_HYPRE_Real            rhs_norm       = 0.0;
   NALU_HYPRE_Real            old_resnorm;
   NALU_HYPRE_Real            ieee_check     = 0.0;
   NALU_HYPRE_Real            operat_cmplxty = nalu_hypre_ParNSHDataOperatorComplexity(nsh_data);

   NALU_HYPRE_Int             iter, num_procs,  my_id;
   NALU_HYPRE_Int             Solve_err_flag;

   if (logging > 1)
   {
      residual = nalu_hypre_ParNSHDataResidual(nsh_data);
   }

   nalu_hypre_ParNSHDataNumIterations(nsh_data) = 0;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/
   if (my_id == 0 && print_level > 1)
   {
      nalu_hypre_NSHWriteSolverParams(nsh_data);
   }

   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;
   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1 && tol > 0.)
   {
      nalu_hypre_printf("\n\n Newton-Schulz-Hotelling SOLVER SOLUTION INFO:\n");
   }


   /*-----------------------------------------------------------------------
    *    Compute initial residual and print
    *-----------------------------------------------------------------------*/
   if (print_level > 1 || logging > 1 || tol > 0.)
   {
      if ( logging > 1 )
      {
         nalu_hypre_ParVectorCopy(f, residual );
         if (tol > 0.0)
         {
            nalu_hypre_ParCSRMatrixMatvec(alpha, A, u, beta, residual );
         }
         resnorm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd( residual, residual ));
      }
      else
      {
         nalu_hypre_ParVectorCopy(f, Ftemp);
         if (tol > 0.0)
         {
            nalu_hypre_ParCSRMatrixMatvec(alpha, A, u, beta, Ftemp);
         }
         resnorm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(Ftemp, Ftemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resnorm != 0.)
      {
         ieee_check = resnorm / resnorm; /* INF -> NaN conversion */
      }
      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (print_level > 0)
         {
            nalu_hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            nalu_hypre_printf("ERROR -- nalu_hypre_NSHSolve: INFs and/or NaNs detected in input.\n");
            nalu_hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            nalu_hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
         return nalu_hypre_error_flag;
      }

      init_resnorm = resnorm;
      rhs_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(f, f));
      if (rhs_norm > NALU_HYPRE_REAL_EPSILON)
      {
         rel_resnorm = init_resnorm / rhs_norm;
      }
      else
      {
         /* rhs is zero, return a zero solution */
         nalu_hypre_ParVectorSetConstantValues(U_array, 0.0);
         if (logging > 0)
         {
            rel_resnorm = 0.0;
            nalu_hypre_ParNSHDataFinalRelResidualNorm(nsh_data) = rel_resnorm;
         }
         return nalu_hypre_error_flag;
      }
   }
   else
   {
      rel_resnorm = 1.;
   }

   if (my_id == 0 && print_level > 1)
   {
      nalu_hypre_printf("                                            relative\n");
      nalu_hypre_printf("               residual        factor       residual\n");
      nalu_hypre_printf("               --------        ------       --------\n");
      nalu_hypre_printf("    Initial    %e                 %e\n", init_resnorm,
                   rel_resnorm);
   }

   matA = A;
   U_array = u;
   F_array = f;

   /************** Main Solver Loop - always do 1 iteration ************/
   iter = 0;

   while ((rel_resnorm >= tol || iter < 1) && iter < max_iter)
   {
      /* Do one solve on e = Mr */
      nalu_hypre_NSHSolveInverse(matA, f, u, matM, Utemp, Ftemp);

      /*---------------------------------------------------------------
       *    Compute residual and residual norm
       *----------------------------------------------------------------*/

      if (print_level > 1 || logging > 1 || tol > 0.)
      {
         old_resnorm = resnorm;

         if (logging > 1)
         {
            nalu_hypre_ParVectorCopy(F_array, residual);
            nalu_hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, residual );
            resnorm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd( residual, residual ));
         }
         else
         {
            nalu_hypre_ParVectorCopy(F_array, Ftemp);
            nalu_hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, Ftemp);
            resnorm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(Ftemp, Ftemp));
         }

         if (old_resnorm) { conv_factor = resnorm / old_resnorm; }
         else { conv_factor = resnorm; }
         if (rhs_norm > NALU_HYPRE_REAL_EPSILON)
         {
            rel_resnorm = resnorm / rhs_norm;
         }
         else
         {
            rel_resnorm = resnorm;
         }

         norms[iter] = rel_resnorm;
      }

      ++iter;
      nalu_hypre_ParNSHDataNumIterations(nsh_data) = iter;
      nalu_hypre_ParNSHDataFinalRelResidualNorm(nsh_data) = rel_resnorm;

      if (my_id == 0 && print_level > 1)
      {
         nalu_hypre_printf("    NSHSolve %2d   %e    %f     %e \n", iter,
                      resnorm, conv_factor, rel_resnorm);
      }
   }

   /* check convergence within max_iter */
   if (iter == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      nalu_hypre_error(NALU_HYPRE_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Print closing statistics
    *    Add operator and grid complexity stats
    *-----------------------------------------------------------------------*/

   if (iter > 0 && init_resnorm)
   {
      conv_factor = nalu_hypre_pow((resnorm / init_resnorm), (1.0 / (NALU_HYPRE_Real) iter));
   }
   else
   {
      conv_factor = 1.;
   }

   if (print_level > 1)
   {
      /*** compute operator and grid complexity (fill factor) here ?? ***/
      if (my_id == 0)
      {
         if (Solve_err_flag == 1)
         {
            nalu_hypre_printf("\n\n==============================================");
            nalu_hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
            nalu_hypre_printf("      within the allowed %d iterations\n", max_iter);
            nalu_hypre_printf("==============================================");
         }
         nalu_hypre_printf("\n\n Average Convergence Factor = %f \n", conv_factor);
         nalu_hypre_printf("                operator = %f\n", operat_cmplxty);
      }
   }
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_NSHSolveInverse
 *
 * Simply a matvec on residual with approximate inverse
 *
 * A: original matrix
 * f: rhs
 * u: solution
 * M: approximate inverse
 * ftemp, utemp: working vectors
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSolveInverse(nalu_hypre_ParCSRMatrix *A,
                      nalu_hypre_ParVector    *f,
                      nalu_hypre_ParVector    *u,
                      nalu_hypre_ParCSRMatrix *M,
                      nalu_hypre_ParVector    *ftemp,
                      nalu_hypre_ParVector    *utemp)
{
   NALU_HYPRE_Real  alpha = -1.0;
   NALU_HYPRE_Real  beta  = 1.0;
   NALU_HYPRE_Real  zero  = 0.0;

   /* r = f - Au */
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* e = Mr */
   nalu_hypre_ParCSRMatrixMatvec(beta, M, ftemp, zero, utemp);

   /* u = u + e */
   nalu_hypre_ParVectorAxpy(beta, utemp, u);

   return nalu_hypre_error_flag;
}
