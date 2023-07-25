/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobicreate, NALU_HYPRE_STRUCTJACOBICREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobidestroy, NALU_HYPRE_STRUCTJACOBIDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobisetup, NALU_HYPRE_STRUCTJACOBISETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobisolve, NALU_HYPRE_STRUCTJACOBISOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobisettol, NALU_HYPRE_STRUCTJACOBISETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobigettol, NALU_HYPRE_STRUCTJACOBIGETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobisetmaxiter, NALU_HYPRE_STRUCTJACOBISETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobigetmaxiter, NALU_HYPRE_STRUCTJACOBIGETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobisetzeroguess, NALU_HYPRE_STRUCTJACOBISETZEROGUESS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobigetzeroguess, NALU_HYPRE_STRUCTJACOBIGETZEROGUESS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *zeroguess,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobisetnonzerogue, NALU_HYPRE_STRUCTJACOBISETNONZEROGUE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiSetNonZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobigetnumiterati, NALU_HYPRE_STRUCTJACOBIGETNUMITERATI)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructJacobiGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structjacobigetfinalrelat, NALU_HYPRE_STRUCTJACOBIGETFINALRELAT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructJacobiGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
