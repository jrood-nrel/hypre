/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRPCG Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgcreate, NALU_HYPRE_PARCSRPCGCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgdestroy, NALU_HYPRE_PARCSRPCGDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetup, NALU_HYPRE_PARCSRPCGSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsolve, NALU_HYPRE_PARCSRPCGSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettol, NALU_HYPRE_PARCSRPCGSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetAbsoluteTol
 *-------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetatol, NALU_HYPRE_PARCSRPCGSETATOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetAbsoluteTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetmaxiter, NALU_HYPRE_PARCSRPCGSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetstopcrit, NALU_HYPRE_PARCSRPCGSETSTOPCRIT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *stop_crit,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetStopCrit(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettwonorm, NALU_HYPRE_PARCSRPCGSETTWONORM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *two_norm,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetTwoNorm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (two_norm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetrelchange, NALU_HYPRE_PARCSRPCGSETRELCHANGE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rel_change,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetRelChange(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetprecond, NALU_HYPRE_PARCSRPCGSETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Int *precond_id,
  hypre_F90_Obj *precond_solver,
  hypre_F90_Int *ierr            )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - do not set up a preconditioner
    * 1 - set up a ds preconditioner
    * 2 - set up an amg preconditioner
    * 3 - set up a pilut preconditioner
    * 4 - set up a ParaSails preconditioner
    * 5 - set up a Euclid preconditioner
    * 6 - set up a ILU preconditioner
    * 7 - set up a MGR preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRDiagScale,
                   NALU_HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_BoomerAMGSolve,
                   NALU_HYPRE_BoomerAMGSetup,
                   (NALU_HYPRE_Solver)       * precond_solver) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRPilutSolve,
                   NALU_HYPRE_ParCSRPilutSetup,
                   (NALU_HYPRE_Solver)       * precond_solver) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParaSailsSolve,
                   NALU_HYPRE_ParaSailsSetup,
                   (NALU_HYPRE_Solver)       * precond_solver) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_EuclidSolve,
                   NALU_HYPRE_EuclidSetup,
                   (NALU_HYPRE_Solver)       * precond_solver) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ILUSolve,
                   NALU_HYPRE_ILUSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_MGRSolve,
                   NALU_HYPRE_MGRSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetprecond, NALU_HYPRE_PARCSRPCGGETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *precond_solver_ptr,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGGetPrecond(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetprintlevel, NALU_HYPRE_PARCSRPCGSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetPrintLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetlogging, NALU_HYPRE_PARCSRPCGSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetnumiterations, NALU_HYPRE_PARCSRPCGGETNUMITERATIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetfinalrelative, NALU_HYPRE_PARCSRPCGGETFINALRELATIVE)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscalesetup, NALU_HYPRE_PARCSRDIAGSCALESETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *y,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRDiagScaleSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, y),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscale, NALU_HYPRE_PARCSRDIAGSCALE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *HA,
  hypre_F90_Obj *Hy,
  hypre_F90_Obj *Hx,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRDiagScale(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, HA),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, Hy),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, Hx)      ) );
}

#ifdef __cplusplus
}
#endif
