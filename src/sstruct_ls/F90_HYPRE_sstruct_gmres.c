/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructGMRES interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmrescreate, NALU_HYPRE_SSTRUCTGMRESCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj  *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGMRESCreate(
              nalu_hypre_F90_PassComm(comm),
              nalu_hypre_F90_PassObjRef(NALU_HYPRE_SStructSolver, solver));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmresdestroy, NALU_HYPRE_SSTRUCTGMRESDESTROY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressetup, NALU_HYPRE_SSTRUCTGMRESSETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetup(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressolve, NALU_HYPRE_SSTRUCTGMRESSOLVE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSolve(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressetkdim, NALU_HYPRE_SSTRUCTGMRESSETKDIM)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *k_dim,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetKDim(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (k_dim) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressettol, NALU_HYPRE_SSTRUCTGMRESSETTOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressetabsolutetol, NALU_HYPRE_SSTRUCTGMRESSETABSOLUTETOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetAbsoluteTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressetminiter, NALU_HYPRE_SSTRUCTGMRESSETMINITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *min_iter,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetMinIter(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressetmaxiter, NALU_HYPRE_SSTRUCTGMRESSETMAXITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *max_iter,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetMaxIter(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressetstopcrit, NALU_HYPRE_SSTRUCTGMRESSETSTOPCRIT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *stop_crit,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetStopCrit(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressetprecond, NALU_HYPRE_SSTRUCTGMRESSETPRECOND)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *precond_id,
 nalu_hypre_F90_Obj *precond_solver,
 nalu_hypre_F90_Int *ierr)
/*------------------------------------------
 *    precond_id flags mean:
 *    2 - setup a split-solver preconditioner
 *    3 - setup a syspfmg preconditioner
 *    8 - setup a DiagScale preconditioner
 *    9 - no preconditioner setup
 *----------------------------------------*/

{
   if (*precond_id == 2)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructGMRESSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSplitSolve,
                  NALU_HYPRE_SStructSplitSetup,
                  nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 3)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructGMRESSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSysPFMGSolve,
                  NALU_HYPRE_SStructSysPFMGSetup,
                  nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 8)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructGMRESSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructDiagScale,
                  NALU_HYPRE_SStructDiagScaleSetup,
                  nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }
   else if (*precond_id == 9)
   {
      *ierr = 0;
   }

   else
   {
      *ierr = -1;
   }

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressetlogging, NALU_HYPRE_SSTRUCTGMRESSETLOGGING)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *logging,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetLogging(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmressetprintlevel, NALU_HYPRE_SSTRUCTGMRESSETPRINTLEVEL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetPrintLevel(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmresgetnumiterati, NALU_HYPRE_SSTRUCTGMRESGETNUMITERATI)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_iterations,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESGetNumIterations(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmresgetfinalrelat, NALU_HYPRE_SSTRUCTGMRESGETFINALRELAT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *norm,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassRealRef (norm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgmresgetresidual, NALU_HYPRE_SSTRUCTGMRESGETRESIDUAL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *residual,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESGetResidual(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               (void **)              *residual ) );
}

#ifdef __cplusplus
}
#endif
