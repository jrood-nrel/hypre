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
 * NALU_HYPRE_StructSparseMSGCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgcreate, NALU_HYPRE_STRUCTSPARSEMSGCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgdestroy, NALU_HYPRE_STRUCTSPARSEMSGDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetup, NALU_HYPRE_STRUCTSPARSEMSGSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsolve, NALU_HYPRE_STRUCTSPARSEMSGSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsettol, NALU_HYPRE_STRUCTSPARSEMSGSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetmaxiter, NALU_HYPRE_STRUCTSPARSEMSGSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetJump
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetjump, NALU_HYPRE_STRUCTSPARSEMSGSETJUMP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *jump,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetJump(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (jump) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetrelchan, NALU_HYPRE_STRUCTSPARSEMSGSETRELCHAN)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rel_change,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetzerogue, NALU_HYPRE_STRUCTSPARSEMSGSETZEROGUE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetnonzero, NALU_HYPRE_STRUCTSPARSEMSGSETNONZERO)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetNonZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetrelaxty, NALU_HYPRE_STRUCTSPARSEMSGSETRELAXTY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_type,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetjacobiweigh, NALU_HYPRE_STRUCTSPARSEMSGSETJACOBIWEIGH)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *weight,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_StructSparseMSGSetJacobiWeight(
               nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
               nalu_hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetnumprer, NALU_HYPRE_STRUCTSPARSEMSGSETNUMPRER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_pre_relax,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetNumPreRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (num_pre_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetnumpost, NALU_HYPRE_STRUCTSPARSEMSGSETNUMPOST)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_post_relax,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetNumPostRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (num_post_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetnumfine, NALU_HYPRE_STRUCTSPARSEMSGSETNUMFINE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_fine_relax,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetNumFineRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (num_fine_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetlogging, NALU_HYPRE_STRUCTSPARSEMSGSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsgsetprintle, NALU_HYPRE_STRUCTSPARSEMSGSETPRINTLE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsggetnumiter, NALU_HYPRE_STRUCTSPARSEMSGGETNUMITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsparsemsggetfinalre, NALU_HYPRE_STRUCTSPARSEMSGGETFINALRE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
