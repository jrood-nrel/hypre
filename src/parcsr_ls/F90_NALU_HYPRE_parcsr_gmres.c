/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRGMRES Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmrescreate, NALU_HYPRE_PARCSRGMRESCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmresdestroy, NALU_HYPRE_PARCSRGMRESDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressetup, NALU_HYPRE_PARCSRGMRESSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressolve, NALU_HYPRE_PARCSRGMRESSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressetkdim, NALU_HYPRE_PARCSRGMRESSETKDIM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *kdim,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetKDim(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (kdim)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressettol, NALU_HYPRE_PARCSRGMRESSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressetabsolutetol, NALU_HYPRE_PARCSRGMRESSETABSOLUTETOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetAbsoluteTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressetminiter, NALU_HYPRE_PARCSRGMRESSETMINITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *min_iter,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetMinIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressetmaxiter, NALU_HYPRE_PARCSRGMRESSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressetstopcrit, NALU_HYPRE_PARCSRGMRESSETSTOPCRIT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *stop_crit,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetStopCrit(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressetprecond, NALU_HYPRE_PARCSRGMRESSETPRECOND)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *precond_id,
  nalu_hypre_F90_Obj *precond_solver,
  nalu_hypre_F90_Int *ierr          )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    *  0 - no preconditioner
    *  1 - set up a ds preconditioner
    *  2 - set up an amg preconditioner
    *  3 - set up a pilut preconditioner
    *  4 - set up a parasails preconditioner
    *  5 - set up a Euclid preconditioner
    *  6 - set up a ILU preconditioner
    *  7 - set up a MGR preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRDiagScale,
                   NALU_HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_BoomerAMGSolve,
                   NALU_HYPRE_BoomerAMGSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRPilutSolve,
                   NALU_HYPRE_ParCSRPilutSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRParaSailsSolve,
                   NALU_HYPRE_ParCSRParaSailsSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_EuclidSolve,
                   NALU_HYPRE_EuclidSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ILUSolve,
                   NALU_HYPRE_ILUSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
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
 * NALU_HYPRE_ParCSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmresgetprecond, NALU_HYPRE_PARCSRGMRESGETPRECOND)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *precond_solver_ptr,
  nalu_hypre_F90_Int *ierr                )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESGetPrecond(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressetlogging, NALU_HYPRE_PARCSRGMRESSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmressetprintlevel, NALU_HYPRE_PARCSRGMRESSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmresgetnumiteratio, NALU_HYPRE_PARCSRGMRESGETNUMITERATIO)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr            )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrgmresgetfinalrelati, NALU_HYPRE_PARCSRGMRESGETFINALRELATI)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
