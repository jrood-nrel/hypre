/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructPCG interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgcreate, NALU_HYPRE_SSTRUCTPCGCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGCreate(
               nalu_hypre_F90_PassComm (comm),
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgdestroy, NALU_HYPRE_SSTRUCTPCGDESTROY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsetup, NALU_HYPRE_SSTRUCTPCGSETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGSetup(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsolve, NALU_HYPRE_SSTRUCTPCGSOLVE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGSolve(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsettol, NALU_HYPRE_SSTRUCTPCGSETTOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGSetTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsetabsolutetol, NALU_HYPRE_SSTRUCTPCGSETABSOLUTETOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGSetAbsoluteTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsetmaxiter, NALU_HYPRE_SSTRUCTPCGSETMAXITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *max_iter,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGSetMaxIter(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsettwonorm, NALU_HYPRE_SSTRUCTPCGSETTWONORM)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *two_norm,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGSetTwoNorm(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (two_norm) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsetrelchange, NALU_HYPRE_SSTRUCTPCGSETRELCHANGE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *rel_change,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGSetRelChange(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsetprecond, NALU_HYPRE_SSTRUCTPCGSETPRECOND)
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
              (NALU_HYPRE_SStructPCGSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSplitSolve,
                  NALU_HYPRE_SStructSplitSetup,
                  nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 3)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructPCGSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSysPFMGSolve,
                  NALU_HYPRE_SStructSysPFMGSetup,
                  nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 8)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructPCGSetPrecond(
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
 *  NALU_HYPRE_SStructPCGSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsetlogging, NALU_HYPRE_SSTRUCTPCGSETLOGGING)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *logging,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGSetLogging(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcgsetprintlevel, NALU_HYPRE_SSTRUCTPCGSETPRINTLEVEL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGSetPrintLevel(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcggetnumiteration, NALU_HYPRE_SSTRUCTPCGGETNUMITERATION)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_iterations,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGGetNumIterations(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcggetfinalrelativ, NALU_HYPRE_SSTRUCTPCGGETFINALRELATIV)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *norm,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGGetFinalRelativeResidualNorm(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassRealRef (norm) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPCGGetResidual
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpcggetresidual, NALU_HYPRE_SSTRUCTPCGGETRESIDUAL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *residual,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructPCGGetResidual(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               (void **)              *residual ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructDiagScaleSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructdiagscalesetup, NALU_HYPRE_SSTRUCTDIAGSCALESETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *y,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructDiagScaleSetup(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, y),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructDiagScale
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructdiagscale, NALU_HYPRE_SSTRUCTDIAGSCALE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *y,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructDiagScale(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, y),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)    ) );
}

#ifdef __cplusplus
}
#endif
