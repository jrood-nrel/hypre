/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructSplit solver interface
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitcreate, NALU_HYPRE_SSTRUCTSPLITCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *solver_ptr,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitCreate(
               hypre_F90_PassComm (comm),
               hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver_ptr) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitdestroy, NALU_HYPRE_SSTRUCTSPLITDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitDestroy(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetup, NALU_HYPRE_SSTRUCTSPLITSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitSetup(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsolve, NALU_HYPRE_SSTRUCTSPLITSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitSolve(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsettol, NALU_HYPRE_SSTRUCTSPLITSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitSetTol(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetmaxiter, NALU_HYPRE_SSTRUCTSPLITSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitSetMaxIter(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetzeroguess, NALU_HYPRE_SSTRUCTSPLITSETZEROGUESS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitSetZeroGuess(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetnonzerogue, NALU_HYPRE_SSTRUCTSPLITSETNONZEROGUE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitSetNonZeroGuess(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitSetStructSolver
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetstructsolv, NALU_HYPRE_SSTRUCTSPLITSETSTRUCTSOLV)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ssolver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitSetStructSolver(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (ssolver) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitgetnumiterati, NALU_HYPRE_SSTRUCTSPLITGETNUMITERATI)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitGetNumIterations(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSplitGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitgetfinalrelat, NALU_HYPRE_SSTRUCTSPLITGETFINALRELAT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSplitGetFinalRelativeResidualNorm(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
