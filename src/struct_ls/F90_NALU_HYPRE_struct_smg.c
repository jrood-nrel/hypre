/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructSMG Fortran interface
 *
 *****************************************************************************/

#include "_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgcreate, NALU_HYPRE_STRUCTSMGCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgdestroy, NALU_HYPRE_STRUCTSMGDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGDestroy(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetup, NALU_HYPRE_STRUCTSMGSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetup(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsolve, NALU_HYPRE_STRUCTSMGSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSolve(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetMemoryUse, NALU_HYPRE_StructSMGGetMemoryUse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmemoryuse, NALU_HYPRE_STRUCTSMGSETMEMORYUSE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *memory_use,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetMemoryUse(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (memory_use) ) );
}

void
hypre_F90_IFACE(hypre_structsmggetmemoryuse, NALU_HYPRE_STRUCTSMGGETMEMORYUSE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *memory_use,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetMemoryUse(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (memory_use) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetTol, NALU_HYPRE_StructSMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsettol, NALU_HYPRE_STRUCTSMGSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

void
hypre_F90_IFACE(hypre_structsmggettol, NALU_HYPRE_STRUCTSMGGETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetMaxIter, NALU_HYPRE_StructSMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmaxiter, NALU_HYPRE_STRUCTSMGSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

void
hypre_F90_IFACE(hypre_structsmggetmaxiter, NALU_HYPRE_STRUCTSMGGETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetRelChange, NALU_HYPRE_StructSMGGetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetrelchange, NALU_HYPRE_STRUCTSMGSETRELCHANGE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rel_change,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetRelChange(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (rel_change) ) );
}

void
hypre_F90_IFACE(hypre_structsmggetrelchange, NALU_HYPRE_STRUCTSMGGETRELCHANGE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rel_change,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetRelChange(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetZeroGuess, NALU_HYPRE_StructSMGGetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetzeroguess, NALU_HYPRE_STRUCTSMGSETZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

void
hypre_F90_IFACE(hypre_structsmggetzeroguess, NALU_HYPRE_STRUCTSMGGETZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *zeroguess,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnonzeroguess, NALU_HYPRE_STRUCTSMGSETNONZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetNonZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetNumPreRelax, NALU_HYPRE_StructSMGGetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumprerelax, NALU_HYPRE_STRUCTSMGSETNUMPRERELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_pre_relax,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetNumPreRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_pre_relax)) );
}

void
hypre_F90_IFACE(hypre_structsmggetnumprerelax, NALU_HYPRE_STRUCTSMGGETNUMPRERELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_pre_relax,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetNumPreRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_pre_relax)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetNumPostRelax, NALU_HYPRE_StructSMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumpostrelax, NALU_HYPRE_STRUCTSMGSETNUMPOSTRELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_post_relax,
  hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetNumPostRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_post_relax)) );
}

void
hypre_F90_IFACE(hypre_structsmggetnumpostrelax, NALU_HYPRE_STRUCTSMGGETNUMPOSTRELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_post_relax,
  hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetNumPostRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_post_relax)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetLogging, NALU_HYPRE_StructSMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetlogging, NALU_HYPRE_STRUCTSMGSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (logging)) );
}

void
hypre_F90_IFACE(hypre_structsmggetlogging, NALU_HYPRE_STRUCTSMGGETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetLogging(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (logging)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetPrintLevel, NALU_HYPRE_StructSMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetprintlevel, NALU_HYPRE_STRUCTSMGSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (print_level)) );
}

void
hypre_F90_IFACE(hypre_structsmggetprintlevel, NALU_HYPRE_STRUCTSMGGETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (print_level)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetnumiterations, NALU_HYPRE_STRUCTSMGGETNUMITERATIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetfinalrelative, NALU_HYPRE_STRUCTSMGGETFINALRELATIVE)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm) ) );
}
#ifdef __cplusplus
}
#endif
