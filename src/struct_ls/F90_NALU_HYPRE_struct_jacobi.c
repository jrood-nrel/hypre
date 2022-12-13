/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobicreate, NALU_HYPRE_STRUCTJACOBICREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobidestroy, NALU_HYPRE_STRUCTJACOBIDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiDestroy(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetup, NALU_HYPRE_STRUCTJACOBISETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetup(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisolve, NALU_HYPRE_STRUCTJACOBISOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSolve(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisettol, NALU_HYPRE_STRUCTJACOBISETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigettol, NALU_HYPRE_STRUCTJACOBIGETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetmaxiter, NALU_HYPRE_STRUCTJACOBISETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetmaxiter, NALU_HYPRE_STRUCTJACOBIGETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetzeroguess, NALU_HYPRE_STRUCTJACOBISETZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetzeroguess, NALU_HYPRE_STRUCTJACOBIGETZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *zeroguess,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetnonzerogue, NALU_HYPRE_STRUCTJACOBISETNONZEROGUE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetNonZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetnumiterati, NALU_HYPRE_STRUCTJACOBIGETNUMITERATI)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetfinalrelat, NALU_HYPRE_STRUCTJACOBIGETFINALRELAT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
