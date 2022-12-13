/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Routines to set up preconditioners for use in test codes.
 * June 16, 2005
 *--------------------------------------------------------------------------*/
#include "hypre_test.h"


NALU_HYPRE_Int hypre_set_precond(NALU_HYPRE_Int matrix_id, NALU_HYPRE_Int solver_id, NALU_HYPRE_Int precond_id,
                            void *solver,
                            void *precond)
{
   hypre_set_precond_params(precond_id, precond);


   /************************************************************************
    * PARCSR MATRIX
    ***********************************************************************/
   if (matrix_id == NALU_HYPRE_PARCSR)
   {

      /************************************************************************
       *     PCG Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_PCG)
      {
         if (precond_id == NALU_HYPRE_BOOMERAMG)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_EUCLID)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_PARASAILS)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_SCHWARZ)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
      }

      /************************************************************************
       *     GMRES Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_GMRES)
      {
         if (precond_id == NALU_HYPRE_BOOMERAMG)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_EUCLID)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_PARASAILS)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_PILUT)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_SCHWARZ)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
      }

      /************************************************************************
       *     BiCGSTAB Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_BICGSTAB)
      {
         if (precond_id == NALU_HYPRE_BOOMERAMG)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_EUCLID)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_PILUT)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSolve,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
      }

      /************************************************************************
       *     CGNR Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_CGNR)
      {
         if (precond_id == NALU_HYPRE_BOOMERAMG)
         {
            NALU_HYPRE_CGNRSetPrecond( (NALU_HYPRE_Solver) solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolveT,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                  (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_CGNRSetPrecond( (NALU_HYPRE_Solver) solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                  (NALU_HYPRE_Solver) precond);
         }
      }

   }


   /************************************************************************
    * SSTRUCT MATRIX
    ***********************************************************************/
   if (matrix_id == NALU_HYPRE_SSTRUCT)
   {

      /************************************************************************
       *     PCG Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_PCG)
      {
         if (precond_id == NALU_HYPRE_SPLIT)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_SYSPFMG)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScale,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScaleSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
      }

      /************************************************************************
       * GMRES Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_GMRES)
      {
         if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScale,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScaleSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_SPLIT)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
      }

      /************************************************************************
       * BiCGSTAB Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_BICGSTAB)
      {
         if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScale,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScaleSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_SPLIT)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSolve,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
      }

      /************************************************************************
       * CGNR Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_CGNR)
      {
         if (precond_id == NALU_HYPRE_BOOMERAMG)
         {
            NALU_HYPRE_CGNRSetPrecond( (NALU_HYPRE_Solver) solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolveT,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                  (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_CGNRSetPrecond( (NALU_HYPRE_Solver) solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                  (NALU_HYPRE_Solver) precond);
         }
      }

   }

   /************************************************************************
    * STRUCT MATRIX
    ***********************************************************************/
   if (matrix_id == NALU_HYPRE_STRUCT)
   {

      /************************************************************************
       *     PCG Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_PCG)
      {
         if (precond_id == NALU_HYPRE_SMG)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_PFMG)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_SPARSEMSG)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_JACOBI)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScale,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScaleSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
      }

      /************************************************************************
       *     HYBRID Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_HYBRID)
      {
         if (precond_id == NALU_HYPRE_SMG)
         {
            NALU_HYPRE_StructHybridSetPrecond( (NALU_HYPRE_StructSolver) solver,
                                          (NALU_HYPRE_PtrToStructSolverFcn) NALU_HYPRE_StructSMGSolve,
                                          (NALU_HYPRE_PtrToStructSolverFcn) NALU_HYPRE_StructSMGSetup,
                                          (NALU_HYPRE_StructSolver) precond);
         }
         else if (precond_id == NALU_HYPRE_PFMG)
         {
            NALU_HYPRE_StructHybridSetPrecond( (NALU_HYPRE_StructSolver) solver,
                                          (NALU_HYPRE_PtrToStructSolverFcn) NALU_HYPRE_StructPFMGSolve,
                                          (NALU_HYPRE_PtrToStructSolverFcn) NALU_HYPRE_StructPFMGSetup,
                                          (NALU_HYPRE_StructSolver) precond);
         }
         else if (precond_id == NALU_HYPRE_SPARSEMSG)
         {
            NALU_HYPRE_StructHybridSetPrecond( (NALU_HYPRE_StructSolver) solver,
                                          (NALU_HYPRE_PtrToStructSolverFcn) NALU_HYPRE_StructSparseMSGSolve,
                                          (NALU_HYPRE_PtrToStructSolverFcn) NALU_HYPRE_StructSparseMSGSetup,
                                          (NALU_HYPRE_StructSolver) precond);
         }
      }

      /************************************************************************
       *     GMRES Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_GMRES)
      {
         if (precond_id == NALU_HYPRE_SMG)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_PFMG)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_SPARSEMSG)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_JACOBI)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScale,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScaleSetup,
                                   (NALU_HYPRE_Solver) precond);
         }
      }

      /************************************************************************
       *     BICGSTAB Solver
       ***********************************************************************/
      if (solver_id == NALU_HYPRE_BICGSTAB)
      {
         if (precond_id == NALU_HYPRE_SMG)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSolve,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_PFMG)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSolve,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_SPARSEMSG)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSolve,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_JACOBI)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSolve,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
         else if (precond_id == NALU_HYPRE_DIAGSCALE)
         {
            NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScale,
                                      (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScaleSetup,
                                      (NALU_HYPRE_Solver) precond);
         }
      }
   }
}


NALU_HYPRE_Int hypre_set_precond_params(NALU_HYPRE_Int precond_id, void *precond)
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int ierr;

   /* use BoomerAMG preconditioner */
   if (precond_id == NALU_HYPRE_BOOMERAMG)
   {
      NALU_HYPRE_BoomerAMGCreate(precond);
      NALU_HYPRE_BoomerAMGSetInterpType(precond, interp_type);
      NALU_HYPRE_BoomerAMGSetNumSamples(precond, gsmg_samples);
      NALU_HYPRE_BoomerAMGSetTol(precond, pc_tol);
      NALU_HYPRE_BoomerAMGSetCoarsenType(precond, (hybrid * coarsen_type));
      NALU_HYPRE_BoomerAMGSetMeasureType(precond, measure_type);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(precond, strong_threshold);
      NALU_HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor);
      NALU_HYPRE_BoomerAMGSetPrintLevel(precond, poutdat);
      NALU_HYPRE_BoomerAMGSetPrintFileName(precond, "driver.out.log");
      NALU_HYPRE_BoomerAMGSetMaxIter(precond, 1);
      NALU_HYPRE_BoomerAMGSetCycleType(precond, cycle_type);
      NALU_HYPRE_BoomerAMGSetNumGridSweeps(precond, num_grid_sweeps);
      NALU_HYPRE_BoomerAMGSetGridRelaxType(precond, grid_relax_type);
      NALU_HYPRE_BoomerAMGSetRelaxWeight(precond, relax_weight);
      NALU_HYPRE_BoomerAMGSetOmega(precond, omega);
      NALU_HYPRE_BoomerAMGSetSmoothType(precond, smooth_type);
      NALU_HYPRE_BoomerAMGSetSmoothNumLevels(precond, smooth_num_levels);
      NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(precond, smooth_num_sweeps);
      NALU_HYPRE_BoomerAMGSetGridRelaxPoints(precond, grid_relax_points);
      NALU_HYPRE_BoomerAMGSetMaxLevels(precond, max_levels);
      NALU_HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum);
      NALU_HYPRE_BoomerAMGSetNumFunctions(precond, num_functions);
      NALU_HYPRE_BoomerAMGSetVariant(precond, variant);
      NALU_HYPRE_BoomerAMGSetOverlap(precond, overlap);
      NALU_HYPRE_BoomerAMGSetDomainType(precond, domain_type);
      NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(precond, schwarz_rlx_weight);
      if (num_functions > 1)
      {
         NALU_HYPRE_BoomerAMGSetDofFunc(precond, dof_func);
      }
   }
   /* use DiagScale preconditioner */
   else if (precond_id == NALU_HYPRE_DIAGSCALE)
   {
      precond = NULL;
   }
   /* use ParaSails preconditioner */
   else if (precond_id == NALU_HYPRE_PARASAILS)
   {
      NALU_HYPRE_ParaSailsCreate(hypre_MPI_COMM_WORLD, precond);
      NALU_HYPRE_ParaSailsSetParams(precond, sai_threshold, max_levels);
      NALU_HYPRE_ParaSailsSetFilter(precond, sai_filter);
      NALU_HYPRE_ParaSailsSetLogging(precond, poutdat);
   }
   /* use Schwarz preconditioner */
   else if (precond_id == NALU_HYPRE_SCHWARZ)
   {
      NALU_HYPRE_SchwarzCreate(precond);
      NALU_HYPRE_SchwarzSetVariant(precond, variant);
      NALU_HYPRE_SchwarzSetOverlap(precond, overlap);
      NALU_HYPRE_SchwarzSetDomainType(precond, domain_type);
      NALU_HYPRE_SchwarzSetRelaxWeight(precond, schwarz_rlx_weight);
   }
   /* use GSMG as preconditioner */
   else if (precond_id == NALU_HYPRE_GSMG)
   {
      /* fine grid */
      num_grid_sweeps[0] = num_sweep;
      grid_relax_type[0] = relax_default;
      hypre_TFree(grid_relax_points[0], NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[0] = hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sweep; i++)
      {
         grid_relax_points[0][i] = 0;
      }

      /* down cycle */
      num_grid_sweeps[1] = num_sweep;
      grid_relax_type[1] = relax_default;
      hypre_TFree(grid_relax_points[1], NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[1] = hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sweep; i++)
      {
         grid_relax_points[1][i] = 0;
      }

      /* up cycle */
      num_grid_sweeps[2] = num_sweep;
      grid_relax_type[2] = relax_default;
      hypre_TFree(grid_relax_points[2], NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[2] = hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sweep; i++)
      {
         grid_relax_points[2][i] = 0;
      }

      /* coarsest grid */
      num_grid_sweeps[3] = 1;
      grid_relax_type[3] = 9;
      hypre_TFree(grid_relax_points[3], NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[3] = hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[3][0] = 0;

      NALU_HYPRE_BoomerAMGCreate(precond);
      NALU_HYPRE_BoomerAMGSetGSMG(precond, 4);
      NALU_HYPRE_BoomerAMGSetInterpType(precond, interp_type);
      NALU_HYPRE_BoomerAMGSetNumSamples(precond, gsmg_samples);
      NALU_HYPRE_BoomerAMGSetTol(precond, pc_tol);
      NALU_HYPRE_BoomerAMGSetCoarsenType(precond, (hybrid * coarsen_type));
      NALU_HYPRE_BoomerAMGSetMeasureType(precond, measure_type);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(precond, strong_threshold);
      NALU_HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor);
      NALU_HYPRE_BoomerAMGSetPrintLevel(precond, poutdat);
      NALU_HYPRE_BoomerAMGSetPrintFileName(precond, "driver.out.log");
      NALU_HYPRE_BoomerAMGSetMaxIter(precond, 1);
      NALU_HYPRE_BoomerAMGSetCycleType(precond, cycle_type);
      NALU_HYPRE_BoomerAMGSetNumGridSweeps(precond, num_grid_sweeps);
      NALU_HYPRE_BoomerAMGSetGridRelaxType(precond, grid_relax_type);
      NALU_HYPRE_BoomerAMGSetRelaxWeight(precond, relax_weight);
      NALU_HYPRE_BoomerAMGSetOmega(precond, omega);
      NALU_HYPRE_BoomerAMGSetSmoothType(precond, smooth_type);
      NALU_HYPRE_BoomerAMGSetSmoothNumLevels(precond, smooth_num_levels);
      NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(precond, smooth_num_sweeps);
      NALU_HYPRE_BoomerAMGSetVariant(precond, variant);
      NALU_HYPRE_BoomerAMGSetOverlap(precond, overlap);
      NALU_HYPRE_BoomerAMGSetDomainType(precond, domain_type);
      NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(precond, schwarz_rlx_weight);
      NALU_HYPRE_BoomerAMGSetGridRelaxPoints(precond, grid_relax_points);
      NALU_HYPRE_BoomerAMGSetMaxLevels(precond, max_levels);
      NALU_HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum);
      NALU_HYPRE_BoomerAMGSetNumFunctions(precond, num_functions);
      if (num_functions > 1)
      {
         NALU_HYPRE_BoomerAMGSetDofFunc(precond, dof_func);
      }
   }

   /* use PILUT as preconditioner */
   else if (precond_id == NALU_HYPRE_PILUT)
   {
      ierr = NALU_HYPRE_ParCSRPilutCreate( hypre_MPI_COMM_WORLD, precond );
   }
}


NALU_HYPRE_Int hypre_destroy_precond(NALU_HYPRE_Int precond_id, void *precond)
{

   if (precond_id == NALU_HYPRE_BICGSTAB)
   {
      NALU_HYPRE_BiCGSTABDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_BOOMERAMG)
   {
      NALU_HYPRE_BoomerAMGDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_CGNR)
   {
      NALU_HYPRE_CGNRDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_DIAGSCALE)
   {
      NALU_HYPRE_Destroy(precond);
   }

   else if (precond_id == NALU_HYPRE_EUCLID)
   {
      NALU_HYPRE_EuclidDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_GMRES)
   {
      NALU_HYPRE_GMRESDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_GSMG)
   {
      NALU_HYPRE_BoomerAMGDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_HYBRID)
   {
      NALU_HYPRE_HybridDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_JACOBI)
   {
      NALU_HYPRE_JacobiDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_PARASAILS)
   {
      NALU_HYPRE_ParaSailsDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_PCG)
   {
      NALU_HYPRE_PCGDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_PFMG)
   {
      NALU_HYPRE_PFMGDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_PILUT)
   {
      NALU_HYPRE_PilutDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_SCHWARZ)
   {
      NALU_HYPRE_SchwarzDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_SMG)
   {
      NALU_HYPRE_SMGDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_SPARSEMSG)
   {
      NALU_HYPRE_SparseMSGDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_SPLIT)
   {
      NALU_HYPRE_SplitDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_SPLITPFMG)
   {
      NALU_HYPRE_SplitDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_SPLITSMG)
   {
      NALU_HYPRE_SplitDestroy(precond);
   }

   else if (precond_id == NALU_HYPRE_SYSPFMG)
   {
      NALU_HYPRE_SysPFMGDestroy(precond);
   }
}
