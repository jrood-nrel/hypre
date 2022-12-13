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
 * NALU_HYPRE_StructSparseMSGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgcreate, NALU_HYPRE_STRUCTSPARSEMSGCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgdestroy, NALU_HYPRE_STRUCTSPARSEMSGDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGDestroy(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetup, NALU_HYPRE_STRUCTSPARSEMSGSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetup(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsolve, NALU_HYPRE_STRUCTSPARSEMSGSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSolve(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsettol, NALU_HYPRE_STRUCTSPARSEMSGSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetmaxiter, NALU_HYPRE_STRUCTSPARSEMSGSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetJump
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetjump, NALU_HYPRE_STRUCTSPARSEMSGSETJUMP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *jump,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetJump(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (jump) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetrelchan, NALU_HYPRE_STRUCTSPARSEMSGSETRELCHAN)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rel_change,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetRelChange(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetzerogue, NALU_HYPRE_STRUCTSPARSEMSGSETZEROGUE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnonzero, NALU_HYPRE_STRUCTSPARSEMSGSETNONZERO)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetNonZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetrelaxty, NALU_HYPRE_STRUCTSPARSEMSGSETRELAXTY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_structsparsemsgsetjacobiweigh, NALU_HYPRE_STRUCTSPARSEMSGSETJACOBIWEIGH)
(hypre_F90_Obj *solver,
 hypre_F90_Real *weight,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_StructSparseMSGSetJacobiWeight(
               hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
               hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumprer, NALU_HYPRE_STRUCTSPARSEMSGSETNUMPRER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_pre_relax,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetNumPreRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_pre_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumpost, NALU_HYPRE_STRUCTSPARSEMSGSETNUMPOST)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_post_relax,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetNumPostRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_post_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumfine, NALU_HYPRE_STRUCTSPARSEMSGSETNUMFINE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_fine_relax,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetNumFineRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_fine_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetlogging, NALU_HYPRE_STRUCTSPARSEMSGSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetprintle, NALU_HYPRE_STRUCTSPARSEMSGSETPRINTLE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsggetnumiter, NALU_HYPRE_STRUCTSPARSEMSGGETNUMITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsggetfinalre, NALU_HYPRE_STRUCTSPARSEMSGGETFINALRE)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
