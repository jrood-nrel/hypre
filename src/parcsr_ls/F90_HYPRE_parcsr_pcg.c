/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRPCG Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgcreate, NALU_HYPRE_PARCSRPCGCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgdestroy, NALU_HYPRE_PARCSRPCGDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsetup, NALU_HYPRE_PARCSRPCGSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsolve, NALU_HYPRE_PARCSRPCGSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsettol, NALU_HYPRE_PARCSRPCGSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetAbsoluteTol
 *-------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsetatol, NALU_HYPRE_PARCSRPCGSETATOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetAbsoluteTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol)     ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsetmaxiter, NALU_HYPRE_PARCSRPCGSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsetstopcrit, NALU_HYPRE_PARCSRPCGSETSTOPCRIT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *stop_crit,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetStopCrit(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsettwonorm, NALU_HYPRE_PARCSRPCGSETTWONORM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *two_norm,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetTwoNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (two_norm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsetrelchange, NALU_HYPRE_PARCSRPCGSETRELCHANGE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rel_change,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsetprecond, NALU_HYPRE_PARCSRPCGSETPRECOND)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *precond_id,
  nalu_hypre_F90_Obj *precond_solver,
  nalu_hypre_F90_Int *ierr            )
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
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRDiagScale,
                   NALU_HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_BoomerAMGSolve,
                   NALU_HYPRE_BoomerAMGSetup,
                   (NALU_HYPRE_Solver)       * precond_solver) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRPilutSolve,
                   NALU_HYPRE_ParCSRPilutSetup,
                   (NALU_HYPRE_Solver)       * precond_solver) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParaSailsSolve,
                   NALU_HYPRE_ParaSailsSetup,
                   (NALU_HYPRE_Solver)       * precond_solver) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_EuclidSolve,
                   NALU_HYPRE_EuclidSetup,
                   (NALU_HYPRE_Solver)       * precond_solver) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ILUSolve,
                   NALU_HYPRE_ILUSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
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
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcggetprecond, NALU_HYPRE_PARCSRPCGGETPRECOND)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *precond_solver_ptr,
  nalu_hypre_F90_Int *ierr                )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGGetPrecond(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsetprintlevel, NALU_HYPRE_PARCSRPCGSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *level,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetPrintLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcgsetlogging, NALU_HYPRE_PARCSRPCGSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *level,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcggetnumiterations, NALU_HYPRE_PARCSRPCGGETNUMITERATIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr            )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrpcggetfinalrelative, NALU_HYPRE_PARCSRPCGGETFINALRELATIVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (norm)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrdiagscalesetup, NALU_HYPRE_PARCSRDIAGSCALESETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRDiagScaleSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, y),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrdiagscale, NALU_HYPRE_PARCSRDIAGSCALE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *HA,
  nalu_hypre_F90_Obj *Hy,
  nalu_hypre_F90_Obj *Hx,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRDiagScale(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, HA),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, Hy),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, Hx)      ) );
}

#ifdef __cplusplus
}
#endif
