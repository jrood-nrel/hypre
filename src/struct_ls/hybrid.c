/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   NALU_HYPRE_Real            tol;
   NALU_HYPRE_Real            cf_tol;
   NALU_HYPRE_Real            pcg_atolf;
   NALU_HYPRE_Int             dscg_max_its;
   NALU_HYPRE_Int             krylov_max_its;
   NALU_HYPRE_Int             two_norm;
   NALU_HYPRE_Int             stop_crit;
   NALU_HYPRE_Int             rel_change;
   NALU_HYPRE_Int             recompute_residual;
   NALU_HYPRE_Int             recompute_residual_p;
   NALU_HYPRE_Int             k_dim;
   NALU_HYPRE_Int             solver_type;

   NALU_HYPRE_Int             krylov_default;              /* boolean */
   NALU_HYPRE_Int           (*krylov_precond_solve)(void*, void*, void*, void*);
   NALU_HYPRE_Int           (*krylov_precond_setup)(void*, void*, void*, void*);
   void                 *krylov_precond;

   /* log info (always logged) */
   NALU_HYPRE_Int             dscg_num_its;
   NALU_HYPRE_Int             krylov_num_its;
   NALU_HYPRE_Real            final_rel_res_norm;
   NALU_HYPRE_Int             time_index;

   NALU_HYPRE_Int             print_level;
   /* additional information (place-holder currently used to print norms) */
   NALU_HYPRE_Int             logging;

} nalu_hypre_HybridData;

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_HybridCreate( MPI_Comm  comm )
{
   nalu_hypre_HybridData *hybrid_data;

   hybrid_data = nalu_hypre_CTAlloc(nalu_hypre_HybridData,  1, NALU_HYPRE_MEMORY_HOST);

   (hybrid_data -> comm)        = comm;
   (hybrid_data -> time_index)  = nalu_hypre_InitializeTiming("Hybrid");

   /* set defaults */
   (hybrid_data -> tol)               = 1.0e-06;
   (hybrid_data -> cf_tol)            = 0.90;
   (hybrid_data -> pcg_atolf)         = 0.0;
   (hybrid_data -> dscg_max_its)      = 1000;
   (hybrid_data -> krylov_max_its)    = 200;
   (hybrid_data -> two_norm)          = 0;
   (hybrid_data -> stop_crit)          = 0;
   (hybrid_data -> rel_change)        = 0;
   (hybrid_data -> solver_type)       = 1;
   (hybrid_data -> k_dim)             = 5;
   (hybrid_data -> krylov_default)       = 1;
   (hybrid_data -> krylov_precond_solve) = NULL;
   (hybrid_data -> krylov_precond_setup) = NULL;
   (hybrid_data -> krylov_precond)       = NULL;

   /* initialize */
   (hybrid_data -> dscg_num_its)      = 0;
   (hybrid_data -> krylov_num_its)    = 0;
   (hybrid_data -> logging)           = 0;
   (hybrid_data -> print_level)       = 0;

   return (void *) hybrid_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridDestroy( void  *hybrid_vdata )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *) hybrid_vdata;

   if (hybrid_data)
   {
      nalu_hypre_TFree(hybrid_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetTol( void       *hybrid_vdata,
                    NALU_HYPRE_Real  tol       )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> tol) = tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetConvergenceTol( void       *hybrid_vdata,
                               NALU_HYPRE_Real  cf_tol       )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> cf_tol) = cf_tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetDSCGMaxIter( void      *hybrid_vdata,
                            NALU_HYPRE_Int  dscg_max_its )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> dscg_max_its) = dscg_max_its;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetPCGMaxIter( void      *hybrid_vdata,
                           NALU_HYPRE_Int  krylov_max_its  )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> krylov_max_its) = krylov_max_its;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetPCGAbsoluteTolFactor( void       *hybrid_vdata,
                                     NALU_HYPRE_Real  pcg_atolf  )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> pcg_atolf) = pcg_atolf;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetTwoNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetTwoNorm( void      *hybrid_vdata,
                        NALU_HYPRE_Int  two_norm  )
{
   nalu_hypre_HybridData *hybrid_data = ( nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> two_norm) = two_norm;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetStopCrit( void      *hybrid_vdata,
                         NALU_HYPRE_Int  stop_crit  )
{
   nalu_hypre_HybridData *hybrid_data = ( nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> stop_crit) = stop_crit;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetRelChange( void      *hybrid_vdata,
                          NALU_HYPRE_Int  rel_change  )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> rel_change) = rel_change;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetSolverType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetSolverType( void      *hybrid_vdata,
                           NALU_HYPRE_Int  solver_type  )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> solver_type) = solver_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetRecomputeResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetRecomputeResidual( void      *hybrid_vdata,
                                  NALU_HYPRE_Int  recompute_residual )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> recompute_residual) = recompute_residual;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_HybridGetRecomputeResidual( void      *hybrid_vdata,
                                  NALU_HYPRE_Int *recompute_residual )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   *recompute_residual = (hybrid_data -> recompute_residual);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetRecomputeResidualP
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetRecomputeResidualP( void      *hybrid_vdata,
                                   NALU_HYPRE_Int  recompute_residual_p )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> recompute_residual_p) = recompute_residual_p;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_HybridGetRecomputeResidualP( void      *hybrid_vdata,
                                   NALU_HYPRE_Int *recompute_residual_p )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   *recompute_residual_p = (hybrid_data -> recompute_residual_p);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetKDim( void      *hybrid_vdata,
                     NALU_HYPRE_Int  k_dim  )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> k_dim) = k_dim;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetPrecond( void  *krylov_vdata,
                        NALU_HYPRE_Int  (*krylov_precond_solve)(void*, void*, void*, void*),
                        NALU_HYPRE_Int  (*krylov_precond_setup)(void*, void*, void*, void*),
                        void  *krylov_precond          )
{
   nalu_hypre_HybridData *krylov_data = (nalu_hypre_HybridData *)krylov_vdata;

   (krylov_data -> krylov_default)       = 0;
   (krylov_data -> krylov_precond_solve) = krylov_precond_solve;
   (krylov_data -> krylov_precond_setup) = krylov_precond_setup;
   (krylov_data -> krylov_precond)       = krylov_precond;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetLogging( void       *hybrid_vdata,
                        NALU_HYPRE_Int   logging  )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> logging) = logging;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetPrintLevel( void      *hybrid_vdata,
                           NALU_HYPRE_Int  print_level  )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> print_level) = print_level;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridGetNumIterations( void       *hybrid_vdata,
                              NALU_HYPRE_Int  *num_its      )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   *num_its = (hybrid_data -> dscg_num_its) + (hybrid_data -> krylov_num_its);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridGetDSCGNumIterations( void       *hybrid_vdata,
                                  NALU_HYPRE_Int  *dscg_num_its )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   *dscg_num_its = (hybrid_data -> dscg_num_its);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridGetPCGNumIterations( void       *hybrid_vdata,
                                 NALU_HYPRE_Int  *krylov_num_its  )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   *krylov_num_its = (hybrid_data -> krylov_num_its);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridGetFinalRelativeResidualNorm( void        *hybrid_vdata,
                                          NALU_HYPRE_Real  *final_rel_res_norm )
{
   nalu_hypre_HybridData *hybrid_data = (nalu_hypre_HybridData *)hybrid_vdata;

   *final_rel_res_norm = (hybrid_data -> final_rel_res_norm);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_HybridSetup( void               *hybrid_vdata,
                   nalu_hypre_StructMatrix *A,
                   nalu_hypre_StructVector *b,
                   nalu_hypre_StructVector *x            )
{
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_HybridSolve
 *--------------------------------------------------------------------------
 *
 * This solver is designed to solve Ax=b using a hybrid algorithm. First
 * the solver uses diagonally scaled conjugate gradients. If sufficient
 * progress is not made, the algorithm switches to preconditioned
 * conjugate gradients with user-specified preconditioner.
 *
 *--------------------------------------------------------------------------*/

/* Local helper function for creating default PCG solver */
void *
nalu_hypre_HybridSolveUsePCG( nalu_hypre_HybridData  *hybrid_data )
{
   void       *krylov_solver;
   NALU_HYPRE_Real  tol            = (hybrid_data -> tol);
   NALU_HYPRE_Real  pcg_atolf      = (hybrid_data -> pcg_atolf);
   NALU_HYPRE_Int   two_norm       = (hybrid_data -> two_norm);
   NALU_HYPRE_Int   stop_crit      = (hybrid_data -> stop_crit);
   NALU_HYPRE_Int   rel_change     = (hybrid_data -> rel_change);
   NALU_HYPRE_Int   recompute_residual   = (hybrid_data -> recompute_residual);
   NALU_HYPRE_Int   recompute_residual_p = (hybrid_data -> recompute_residual_p);
   NALU_HYPRE_Int   logging        = (hybrid_data -> logging);
   NALU_HYPRE_Int   print_level    = (hybrid_data -> print_level);

   nalu_hypre_PCGFunctions  *pcg_functions =
      nalu_hypre_PCGFunctionsCreate(
         nalu_hypre_StructKrylovCAlloc, nalu_hypre_StructKrylovFree,
         nalu_hypre_StructKrylovCommInfo,
         nalu_hypre_StructKrylovCreateVector,
         nalu_hypre_StructKrylovDestroyVector, nalu_hypre_StructKrylovMatvecCreate,
         nalu_hypre_StructKrylovMatvec, nalu_hypre_StructKrylovMatvecDestroy,
         nalu_hypre_StructKrylovInnerProd, nalu_hypre_StructKrylovCopyVector,
         nalu_hypre_StructKrylovClearVector,
         nalu_hypre_StructKrylovScaleVector, nalu_hypre_StructKrylovAxpy,
         nalu_hypre_StructKrylovIdentitySetup, nalu_hypre_StructKrylovIdentity );
   krylov_solver = nalu_hypre_PCGCreate( pcg_functions );

   nalu_hypre_PCGSetTol(krylov_solver, tol);
   nalu_hypre_PCGSetAbsoluteTolFactor(krylov_solver, pcg_atolf);
   nalu_hypre_PCGSetTwoNorm(krylov_solver, two_norm);
   nalu_hypre_PCGSetStopCrit(krylov_solver, stop_crit);
   nalu_hypre_PCGSetRelChange(krylov_solver, rel_change);
   nalu_hypre_PCGSetRecomputeResidual(krylov_solver, recompute_residual);
   nalu_hypre_PCGSetRecomputeResidualP(krylov_solver, recompute_residual_p);
   nalu_hypre_PCGSetPrintLevel(krylov_solver, print_level);
   nalu_hypre_PCGSetLogging(krylov_solver, logging);

   return krylov_solver;
}

/* Local helper function for setting up GMRES */
void *
nalu_hypre_HybridSolveUseGMRES( nalu_hypre_HybridData  *hybrid_data )
{
   void       *krylov_solver;
   NALU_HYPRE_Real  tol            = (hybrid_data -> tol);
   NALU_HYPRE_Int   stop_crit      = (hybrid_data -> stop_crit);
   NALU_HYPRE_Int   rel_change     = (hybrid_data -> rel_change);
   NALU_HYPRE_Int   logging        = (hybrid_data -> logging);
   NALU_HYPRE_Int   print_level    = (hybrid_data -> print_level);
   NALU_HYPRE_Int   k_dim          = (hybrid_data -> k_dim);

   nalu_hypre_GMRESFunctions  *gmres_functions =
      nalu_hypre_GMRESFunctionsCreate(
         nalu_hypre_StructKrylovCAlloc, nalu_hypre_StructKrylovFree,
         nalu_hypre_StructKrylovCommInfo,
         nalu_hypre_StructKrylovCreateVector,
         nalu_hypre_StructKrylovCreateVectorArray,
         nalu_hypre_StructKrylovDestroyVector, nalu_hypre_StructKrylovMatvecCreate,
         nalu_hypre_StructKrylovMatvec, nalu_hypre_StructKrylovMatvecDestroy,
         nalu_hypre_StructKrylovInnerProd, nalu_hypre_StructKrylovCopyVector,
         nalu_hypre_StructKrylovClearVector,
         nalu_hypre_StructKrylovScaleVector, nalu_hypre_StructKrylovAxpy,
         nalu_hypre_StructKrylovIdentitySetup, nalu_hypre_StructKrylovIdentity );
   krylov_solver = nalu_hypre_GMRESCreate( gmres_functions );

   nalu_hypre_GMRESSetTol(krylov_solver, tol);
   nalu_hypre_GMRESSetKDim(krylov_solver, k_dim);
   nalu_hypre_GMRESSetStopCrit(krylov_solver, stop_crit);
   nalu_hypre_GMRESSetRelChange(krylov_solver, rel_change);
   nalu_hypre_GMRESSetPrintLevel(krylov_solver, print_level);
   nalu_hypre_GMRESSetLogging(krylov_solver, logging);

   return krylov_solver;
}

/* Local helper function for setting up BiCGSTAB */
void *
nalu_hypre_HybridSolveUseBiCGSTAB( nalu_hypre_HybridData  *hybrid_data )
{
   void       *krylov_solver;
   NALU_HYPRE_Real  tol            = (hybrid_data -> tol);
   NALU_HYPRE_Int   stop_crit      = (hybrid_data -> stop_crit);
   NALU_HYPRE_Int   logging        = (hybrid_data -> logging);
   NALU_HYPRE_Int   print_level    = (hybrid_data -> print_level);

   nalu_hypre_BiCGSTABFunctions  *bicgstab_functions =
      nalu_hypre_BiCGSTABFunctionsCreate(
         nalu_hypre_StructKrylovCreateVector,
         nalu_hypre_StructKrylovDestroyVector, nalu_hypre_StructKrylovMatvecCreate,
         nalu_hypre_StructKrylovMatvec, nalu_hypre_StructKrylovMatvecDestroy,
         nalu_hypre_StructKrylovInnerProd, nalu_hypre_StructKrylovCopyVector,
         nalu_hypre_StructKrylovClearVector,
         nalu_hypre_StructKrylovScaleVector, nalu_hypre_StructKrylovAxpy,
         nalu_hypre_StructKrylovCommInfo,
         nalu_hypre_StructKrylovIdentitySetup, nalu_hypre_StructKrylovIdentity );
   krylov_solver = nalu_hypre_BiCGSTABCreate( bicgstab_functions );

   nalu_hypre_BiCGSTABSetTol(krylov_solver, tol);
   nalu_hypre_BiCGSTABSetStopCrit(krylov_solver, stop_crit);
   nalu_hypre_BiCGSTABSetPrintLevel(krylov_solver, print_level);
   nalu_hypre_BiCGSTABSetLogging(krylov_solver, logging);

   return krylov_solver;
}

NALU_HYPRE_Int
nalu_hypre_HybridSolve( void               *hybrid_vdata,
                   nalu_hypre_StructMatrix *A,
                   nalu_hypre_StructVector *b,
                   nalu_hypre_StructVector *x            )
{
   nalu_hypre_HybridData  *hybrid_data    = (nalu_hypre_HybridData *)hybrid_vdata;

   MPI_Comm           comm           = (hybrid_data -> comm);

   NALU_HYPRE_Real         cf_tol         = (hybrid_data -> cf_tol);
   NALU_HYPRE_Int          dscg_max_its   = (hybrid_data -> dscg_max_its);
   NALU_HYPRE_Int          krylov_max_its    = (hybrid_data -> krylov_max_its);
   NALU_HYPRE_Int          logging        = (hybrid_data -> logging);
   NALU_HYPRE_Int          solver_type    = (hybrid_data -> solver_type);

   NALU_HYPRE_Int          krylov_default = (hybrid_data -> krylov_default);
   NALU_HYPRE_Int        (*krylov_precond_solve)(void*, void*, void*, void*);
   NALU_HYPRE_Int        (*krylov_precond_setup)(void*, void*, void*, void*);
   void              *krylov_precond;
   void              *krylov_solver;

   NALU_HYPRE_Int          dscg_num_its;
   NALU_HYPRE_Int          krylov_num_its;
   NALU_HYPRE_Int          converged;

   NALU_HYPRE_Real         res_norm;
   NALU_HYPRE_Int          myid;

   if (solver_type == 1)
   {
      /*--------------------------------------------------------------------
       * Setup DSCG.
       *--------------------------------------------------------------------*/
      krylov_solver = nalu_hypre_HybridSolveUsePCG(hybrid_data);
      nalu_hypre_PCGSetMaxIter(krylov_solver, dscg_max_its);
      nalu_hypre_PCGSetConvergenceFactorTol(krylov_solver, cf_tol);

      krylov_precond = NULL;

      nalu_hypre_PCGSetPrecond((void*) krylov_solver,
                          (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_StructDiagScale,
                          (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_StructDiagScaleSetup,
                          (void*) krylov_precond);
      nalu_hypre_PCGSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Solve with DSCG.
       *--------------------------------------------------------------------*/
      nalu_hypre_PCGSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Get information for DSCG.
       *--------------------------------------------------------------------*/
      nalu_hypre_PCGGetNumIterations(krylov_solver, &dscg_num_its);
      (hybrid_data -> dscg_num_its) = dscg_num_its;
      nalu_hypre_PCGGetFinalRelativeResidualNorm(krylov_solver, &res_norm);

      /*--------------------------------------------------------------------
       * Get additional information from PCG if logging on for hybrid solver.
       * Currently used as debugging flag to print norms.
       *--------------------------------------------------------------------*/
      if ( logging > 1 )
      {
         nalu_hypre_MPI_Comm_rank(comm, &myid );
         nalu_hypre_PCGPrintLogging(krylov_solver, myid);
      }

      /*--------------------------------------------------------------------
       * check if converged.
       *--------------------------------------------------------------------*/
      nalu_hypre_PCGGetConverged(krylov_solver, &converged);
   }
   else if (solver_type == 2)
   {
      /*--------------------------------------------------------------------
       * Setup GMRES
       *--------------------------------------------------------------------*/
      krylov_solver = nalu_hypre_HybridSolveUseGMRES(hybrid_data);
      nalu_hypre_GMRESSetMaxIter(krylov_solver, dscg_max_its);
      nalu_hypre_GMRESSetConvergenceFactorTol(krylov_solver, cf_tol);

      krylov_precond = NULL;

      nalu_hypre_GMRESSetPrecond((void*) krylov_solver,
                            (NALU_HYPRE_Int (*)(void*, void*, void*, void*))NALU_HYPRE_StructDiagScale,
                            (NALU_HYPRE_Int (*)(void*, void*, void*, void*))NALU_HYPRE_StructDiagScaleSetup,
                            (void*) krylov_precond);
      nalu_hypre_GMRESSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Solve with GMRES
       *--------------------------------------------------------------------*/
      nalu_hypre_GMRESSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Get information for GMRES
       *--------------------------------------------------------------------*/
      nalu_hypre_GMRESGetNumIterations(krylov_solver, &dscg_num_its);
      (hybrid_data -> dscg_num_its) = dscg_num_its;
      nalu_hypre_GMRESGetFinalRelativeResidualNorm(krylov_solver, &res_norm);

      /*--------------------------------------------------------------------
       * check if converged.
       *--------------------------------------------------------------------*/
      nalu_hypre_GMRESGetConverged(krylov_solver, &converged);
   }

   else
   {
      /*--------------------------------------------------------------------
       * Setup BiCGSTAB
       *--------------------------------------------------------------------*/
      krylov_solver = nalu_hypre_HybridSolveUseBiCGSTAB(hybrid_data);
      nalu_hypre_BiCGSTABSetMaxIter(krylov_solver, dscg_max_its);
      nalu_hypre_BiCGSTABSetConvergenceFactorTol(krylov_solver, cf_tol);

      krylov_precond = NULL;

      nalu_hypre_BiCGSTABSetPrecond((void*) krylov_solver,
                               (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_StructDiagScale,
                               (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_StructDiagScaleSetup,
                               (void*) krylov_precond);
      nalu_hypre_BiCGSTABSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Solve with BiCGSTAB
       *--------------------------------------------------------------------*/
      nalu_hypre_BiCGSTABSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Get information for BiCGSTAB
       *--------------------------------------------------------------------*/
      nalu_hypre_BiCGSTABGetNumIterations(krylov_solver, &dscg_num_its);
      (hybrid_data -> dscg_num_its) = dscg_num_its;
      nalu_hypre_BiCGSTABGetFinalRelativeResidualNorm(krylov_solver, &res_norm);

      /*--------------------------------------------------------------------
       * check if converged.
       *--------------------------------------------------------------------*/
      nalu_hypre_BiCGSTABGetConverged(krylov_solver, &converged);
   }

   /*-----------------------------------------------------------------------
    * if converged, done...
    *-----------------------------------------------------------------------*/
   if ( converged )
   {
      (hybrid_data -> final_rel_res_norm) = res_norm;
      if (solver_type == 1)
      {
         nalu_hypre_PCGDestroy(krylov_solver);
      }
      else if (solver_type == 2)
      {
         nalu_hypre_GMRESDestroy(krylov_solver);
      }
      else
      {
         nalu_hypre_BiCGSTABDestroy(krylov_solver);
      }
   }

   /*-----------------------------------------------------------------------
    * ... otherwise, use solver+precond
    *-----------------------------------------------------------------------*/
   else
   {
      /*--------------------------------------------------------------------
       * Free up previous PCG solver structure and set up a new one.
       *--------------------------------------------------------------------*/
      if (solver_type == 1)
      {
         nalu_hypre_PCGDestroy(krylov_solver);

         krylov_solver = nalu_hypre_HybridSolveUsePCG(hybrid_data);
         nalu_hypre_PCGSetMaxIter(krylov_solver, krylov_max_its);
         nalu_hypre_PCGSetConvergenceFactorTol(krylov_solver, 0.0);
      }
      else if (solver_type == 2)
      {
         nalu_hypre_GMRESDestroy(krylov_solver);

         krylov_solver = nalu_hypre_HybridSolveUseGMRES(hybrid_data);
         nalu_hypre_GMRESSetMaxIter(krylov_solver, krylov_max_its);
         nalu_hypre_GMRESSetConvergenceFactorTol(krylov_solver, 0.0);
      }
      else
      {
         nalu_hypre_BiCGSTABDestroy(krylov_solver);

         krylov_solver = nalu_hypre_HybridSolveUseBiCGSTAB(hybrid_data);
         nalu_hypre_BiCGSTABSetMaxIter(krylov_solver, krylov_max_its);
         nalu_hypre_BiCGSTABSetConvergenceFactorTol(krylov_solver, 0.0);
      }

      /* Setup preconditioner */
      if (krylov_default)
      {
         krylov_precond = nalu_hypre_SMGCreate(comm);
         nalu_hypre_SMGSetMaxIter(krylov_precond, 1);
         nalu_hypre_SMGSetTol(krylov_precond, 0.0);
         nalu_hypre_SMGSetNumPreRelax(krylov_precond, 1);
         nalu_hypre_SMGSetNumPostRelax(krylov_precond, 1);
         nalu_hypre_SMGSetLogging(krylov_precond, 0);
         krylov_precond_solve = (NALU_HYPRE_Int (*)(void*, void*, void*, void*))nalu_hypre_SMGSolve;
         krylov_precond_setup = (NALU_HYPRE_Int (*)(void*, void*, void*, void*))nalu_hypre_SMGSetup;
      }
      else
      {
         krylov_precond       = (hybrid_data -> krylov_precond);
         krylov_precond_solve = (hybrid_data -> krylov_precond_solve);
         krylov_precond_setup = (hybrid_data -> krylov_precond_setup);
      }

      /* Complete setup of solver+precond */
      if (solver_type == 1)
      {
         nalu_hypre_PCGSetPrecond((void*) krylov_solver,
                             (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) krylov_precond_solve,
                             (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) krylov_precond_setup,
                             (void*) krylov_precond);
         nalu_hypre_PCGSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         nalu_hypre_PCGSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from PCG that is always logged in hybrid solver*/
         nalu_hypre_PCGGetNumIterations(krylov_solver, &krylov_num_its);
         (hybrid_data -> krylov_num_its)  = krylov_num_its;
         nalu_hypre_PCGGetFinalRelativeResidualNorm(krylov_solver, &res_norm);
         (hybrid_data -> final_rel_res_norm) = res_norm;

         /*-----------------------------------------------------------------
          * Get additional information from PCG if logging on for hybrid solver.
          * Currently used as debugging flag to print norms.
          *-----------------------------------------------------------------*/
         if ( logging > 1 )
         {
            nalu_hypre_MPI_Comm_rank(comm, &myid );
            nalu_hypre_PCGPrintLogging(krylov_solver, myid);
         }

         /* Free PCG and preconditioner */
         nalu_hypre_PCGDestroy(krylov_solver);
      }
      else if (solver_type == 2)
      {
         nalu_hypre_GMRESSetPrecond(krylov_solver,
                               krylov_precond_solve, krylov_precond_setup, krylov_precond);
         nalu_hypre_GMRESSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         nalu_hypre_GMRESSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from GMRES that is always logged in hybrid solver*/
         nalu_hypre_GMRESGetNumIterations(krylov_solver, &krylov_num_its);
         (hybrid_data -> krylov_num_its)  = krylov_num_its;
         nalu_hypre_GMRESGetFinalRelativeResidualNorm(krylov_solver, &res_norm);
         (hybrid_data -> final_rel_res_norm) = res_norm;

         /* Free GMRES and preconditioner */
         nalu_hypre_GMRESDestroy(krylov_solver);
      }
      else
      {
         nalu_hypre_BiCGSTABSetPrecond(krylov_solver, krylov_precond_solve,
                                  krylov_precond_setup, krylov_precond);
         nalu_hypre_BiCGSTABSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         nalu_hypre_BiCGSTABSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from BiCGSTAB that is always logged in hybrid solver*/
         nalu_hypre_BiCGSTABGetNumIterations(krylov_solver, &krylov_num_its);
         (hybrid_data -> krylov_num_its)  = krylov_num_its;
         nalu_hypre_BiCGSTABGetFinalRelativeResidualNorm(krylov_solver, &res_norm);
         (hybrid_data -> final_rel_res_norm) = res_norm;

         /* Free BiCGSTAB and preconditioner */
         nalu_hypre_BiCGSTABDestroy(krylov_solver);
      }

      if (krylov_default)
      {
         nalu_hypre_SMGDestroy(krylov_precond);
      }
   }

   return nalu_hypre_error_flag;
}

