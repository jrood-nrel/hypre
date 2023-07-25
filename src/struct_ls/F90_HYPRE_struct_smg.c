/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructSMG Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgcreate, NALU_HYPRE_STRUCTSMGCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgdestroy, NALU_HYPRE_STRUCTSMGDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetup, NALU_HYPRE_STRUCTSMGSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsolve, NALU_HYPRE_STRUCTSMGSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetMemoryUse, NALU_HYPRE_StructSMGGetMemoryUse
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetmemoryuse, NALU_HYPRE_STRUCTSMGSETMEMORYUSE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *memory_use,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetMemoryUse(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (memory_use) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetmemoryuse, NALU_HYPRE_STRUCTSMGGETMEMORYUSE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *memory_use,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetMemoryUse(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (memory_use) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetTol, NALU_HYPRE_StructSMGGetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsettol, NALU_HYPRE_STRUCTSMGSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (tol) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggettol, NALU_HYPRE_STRUCTSMGGETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetMaxIter, NALU_HYPRE_StructSMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetmaxiter, NALU_HYPRE_STRUCTSMGSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetmaxiter, NALU_HYPRE_STRUCTSMGGETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetRelChange, NALU_HYPRE_StructSMGGetRelChange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetrelchange, NALU_HYPRE_STRUCTSMGSETRELCHANGE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rel_change,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (rel_change) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetrelchange, NALU_HYPRE_STRUCTSMGGETRELCHANGE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rel_change,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetZeroGuess, NALU_HYPRE_StructSMGGetZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetzeroguess, NALU_HYPRE_STRUCTSMGSETZEROGUESS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetzeroguess, NALU_HYPRE_STRUCTSMGGETZEROGUESS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *zeroguess,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetnonzeroguess, NALU_HYPRE_STRUCTSMGSETNONZEROGUESS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetNonZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetNumPreRelax, NALU_HYPRE_StructSMGGetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetnumprerelax, NALU_HYPRE_STRUCTSMGSETNUMPRERELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_pre_relax,
  nalu_hypre_F90_Int *ierr         )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetNumPreRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (num_pre_relax)) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetnumprerelax, NALU_HYPRE_STRUCTSMGGETNUMPRERELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_pre_relax,
  nalu_hypre_F90_Int *ierr         )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetNumPreRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_pre_relax)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetNumPostRelax, NALU_HYPRE_StructSMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetnumpostrelax, NALU_HYPRE_STRUCTSMGSETNUMPOSTRELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_post_relax,
  nalu_hypre_F90_Int *ierr           )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetNumPostRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (num_post_relax)) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetnumpostrelax, NALU_HYPRE_STRUCTSMGGETNUMPOSTRELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_post_relax,
  nalu_hypre_F90_Int *ierr           )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetNumPostRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_post_relax)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetLogging, NALU_HYPRE_StructSMGGetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetlogging, NALU_HYPRE_STRUCTSMGSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (logging)) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetlogging, NALU_HYPRE_STRUCTSMGGETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (logging)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGSetPrintLevel, NALU_HYPRE_StructSMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmgsetprintlevel, NALU_HYPRE_STRUCTSMGSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (print_level)) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetprintlevel, NALU_HYPRE_STRUCTSMGGETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (print_level)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetnumiterations, NALU_HYPRE_STRUCTSMGGETNUMITERATIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr           )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsmggetfinalrelative, NALU_HYPRE_STRUCTSMGGETFINALRELATIVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (norm) ) );
}
#ifdef __cplusplus
}
#endif
