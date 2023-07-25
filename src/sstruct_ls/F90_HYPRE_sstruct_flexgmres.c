/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructFlexGMRES interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmrescreate, NALU_HYPRE_SSTRUCTFLEXGMRESCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESCreate(
               nalu_hypre_F90_PassComm (comm),
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmresdestroy, NALU_HYPRE_SSTRUCTFLEXGMRESDESTROY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressetup, NALU_HYPRE_SSTRUCTFLEXGMRESSETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESSetup(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressolve, NALU_HYPRE_SSTRUCTFLEXGMRESSOLVE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESSolve(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressetkdim, NALU_HYPRE_SSTRUCTFLEXGMRESSETKDIM)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *k_dim,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESSetKDim(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (k_dim) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressettol, NALU_HYPRE_SSTRUCTFLEXGMRESSETTOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESSetTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressetabsolutetol, NALU_HYPRE_SSTRUCTFLEXGMRESSETABSOLUTETOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESSetAbsoluteTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressetminiter, NALU_HYPRE_SSTRUCTFLEXGMRESSETMINITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *min_iter,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESSetMinIter(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressetmaxiter, NALU_HYPRE_SSTRUCTFLEXGMRESSETMAXITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *max_iter,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESSetMaxIter(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (max_iter) ) );
}



/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressetprecond, NALU_HYPRE_SSTRUCTFLEXGMRESSETPRECOND)
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
              (NALU_HYPRE_SStructFlexGMRESSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSplitSolve,
                  NALU_HYPRE_SStructSplitSetup,
                  nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 3)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructFlexGMRESSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSysPFMGSolve,
                  NALU_HYPRE_SStructSysPFMGSetup,
                  nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 8)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructFlexGMRESSetPrecond(
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
 * NALU_HYPRE_SStructFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressetlogging, NALU_HYPRE_SSTRUCTFLEXGMRESSETLOGGING)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *logging,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESSetLogging(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmressetprintlevel, NALU_HYPRE_SSTRUCTFLEXGMRESSETPRINTLEVEL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESSetPrintLevel(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmresgetnumiterati, NALU_HYPRE_SSTRUCTFLEXGMRESGETNUMITERATI)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_iterations,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESGetNumIterations(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmresgetfinalrelat, NALU_HYPRE_SSTRUCTFLEXGMRESGETFINALRELAT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *norm,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassRealRef (norm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructflexgmresgetresidual, NALU_HYPRE_SSTRUCTFLEXGMRESGETRESIDUAL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *residual,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFlexGMRESGetResidual(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               (void **)              *residual ) );
}

#ifdef __cplusplus
}
#endif
