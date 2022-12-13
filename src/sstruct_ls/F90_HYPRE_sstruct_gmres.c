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

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmrescreate, NALU_HYPRE_SSTRUCTGMRESCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj  *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_SStructGMRESCreate(
              hypre_F90_PassComm(comm),
              hypre_F90_PassObjRef(NALU_HYPRE_SStructSolver, solver));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresdestroy, NALU_HYPRE_SSTRUCTGMRESDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESDestroy(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetup, NALU_HYPRE_SSTRUCTGMRESSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetup(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressolve, NALU_HYPRE_SSTRUCTGMRESSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSolve(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetkdim, NALU_HYPRE_SSTRUCTGMRESSETKDIM)
(hypre_F90_Obj *solver,
 hypre_F90_Int *k_dim,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetKDim(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (k_dim) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressettol, NALU_HYPRE_SSTRUCTGMRESSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetTol(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetabsolutetol, NALU_HYPRE_SSTRUCTGMRESSETABSOLUTETOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetAbsoluteTol(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetminiter, NALU_HYPRE_SSTRUCTGMRESSETMINITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *min_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetMinIter(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetmaxiter, NALU_HYPRE_SSTRUCTGMRESSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetMaxIter(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetstopcrit, NALU_HYPRE_SSTRUCTGMRESSETSTOPCRIT)
(hypre_F90_Obj *solver,
 hypre_F90_Int *stop_crit,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetStopCrit(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetprecond, NALU_HYPRE_SSTRUCTGMRESSETPRECOND)
(hypre_F90_Obj *solver,
 hypre_F90_Int *precond_id,
 hypre_F90_Obj *precond_solver,
 hypre_F90_Int *ierr)
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
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_SStructGMRESSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSplitSolve,
                  NALU_HYPRE_SStructSplitSetup,
                  hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_SStructGMRESSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSysPFMGSolve,
                  NALU_HYPRE_SStructSysPFMGSetup,
                  hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_SStructGMRESSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructDiagScale,
                  NALU_HYPRE_SStructDiagScaleSetup,
                  hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
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
hypre_F90_IFACE(hypre_sstructgmressetlogging, NALU_HYPRE_SSTRUCTGMRESSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetLogging(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetprintlevel, NALU_HYPRE_SSTRUCTGMRESSETPRINTLEVEL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESSetPrintLevel(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetnumiterati, NALU_HYPRE_SSTRUCTGMRESGETNUMITERATI)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESGetNumIterations(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetfinalrelat, NALU_HYPRE_SSTRUCTGMRESGETFINALRELAT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassRealRef (norm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetresidual, NALU_HYPRE_SSTRUCTGMRESGETRESIDUAL)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *residual,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGMRESGetResidual(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               (void **)              *residual ) );
}

#ifdef __cplusplus
}
#endif
