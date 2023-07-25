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
 * NALU_HYPRE_StructPFMGCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgcreate, NALU_HYPRE_STRUCTPFMGCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgdestroy, NALU_HYPRE_STRUCTPFMGDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetup, NALU_HYPRE_STRUCTPFMGSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsolve, NALU_HYPRE_STRUCTPFMGSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetTol, NALU_HYPRE_StructPFMGGetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsettol, NALU_HYPRE_STRUCTPFMGSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (tol)    ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggettol, NALU_HYPRE_STRUCTPFMGGETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (tol)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetMaxIter, NALU_HYPRE_StructPFMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetmaxiter, NALU_HYPRE_STRUCTPFMGSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (max_iter)  ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetmaxiter, NALU_HYPRE_STRUCTPFMGGETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (max_iter)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetMaxLevels, NALU_HYPRE_StructPFMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetmaxlevels, NALU_HYPRE_STRUCTPFMGSETMAXLEVELS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_levels,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetMaxLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (max_levels)  ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetmaxlevels, NALU_HYPRE_STRUCTPFMGGETMAXLEVELS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_levels,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetMaxLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (max_levels)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetRelChange, NALU_HYPRE_StructPFMGGetRelChange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetrelchange, NALU_HYPRE_STRUCTPFMGSETRELCHANGE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rel_change,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (rel_change)  ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetrelchange, NALU_HYPRE_STRUCTPFMGGETRELCHANGE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rel_change,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetZeroGuess, NALU_HYPRE_StructPFMGGetZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetzeroguess, NALU_HYPRE_STRUCTPFMGSETZEROGUESS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetzeroguess, NALU_HYPRE_STRUCTPFMGGETZEROGUESS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *zeroguess,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetnonzeroguess, NALU_HYPRE_STRUCTPFMGSETNONZEROGUESS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetNonZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetRelaxType, NALU_HYPRE_StructPFMGGetRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetrelaxtype, NALU_HYPRE_STRUCTPFMGSETRELAXTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_type,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (relax_type) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetrelaxtype, NALU_HYPRE_STRUCTPFMGGETRELAXTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_type,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetjacobiweigh, NALU_HYPRE_STRUCTPFMGSETJACOBIWEIGH)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *weight,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_StructPFMGSetJacobiWeight(
               nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
               nalu_hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetRAPType, NALU_HYPRE_StructPFMGSetRapType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetraptype, NALU_HYPRE_STRUCTPFMGSETRAPTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rap_type,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetRAPType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (rap_type) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetraptype, NALU_HYPRE_STRUCTPFMGGETRAPTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rap_type,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetRAPType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (rap_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetNumPreRelax, NALU_HYPRE_StructPFMGGetNumPreRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetnumprerelax, NALU_HYPRE_STRUCTPFMGSETNUMPRERELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_pre_relax,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetNumPreRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (num_pre_relax) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetnumprerelax, NALU_HYPRE_STRUCTPFMGGETNUMPRERELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_pre_relax,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetNumPreRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_pre_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetNumPostRelax, NALU_HYPRE_StructPFMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetnumpostrelax, NALU_HYPRE_STRUCTPFMGSETNUMPOSTRELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_post_relax,
  nalu_hypre_F90_Int *ierr           )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetNumPostRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (num_post_relax) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetnumpostrelax, NALU_HYPRE_STRUCTPFMGGETNUMPOSTRELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_post_relax,
  nalu_hypre_F90_Int *ierr           )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetNumPostRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_post_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetSkipRelax, NALU_HYPRE_StructPFMGGetSkipRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetskiprelax, NALU_HYPRE_STRUCTPFMGSETSKIPRELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *skip_relax,
  nalu_hypre_F90_Int *ierr           )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetSkipRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (skip_relax) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetskiprelax, NALU_HYPRE_STRUCTPFMGGETSKIPRELAX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *skip_relax,
  nalu_hypre_F90_Int *ierr           )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetSkipRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (skip_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetDxyz
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetdxyz, NALU_HYPRE_STRUCTPFMGSETDXYZ)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_RealArray *dxyz,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetDxyz(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealArray (dxyz)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetLogging, NALU_HYPRE_StructPFMGGetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetlogging, NALU_HYPRE_STRUCTPFMGSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetlogging, NALU_HYPRE_STRUCTPFMGGETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetPrintLevel, NALU_HYPRE_StructPFMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmgsetprintlevel, NALU_HYPRE_STRUCTPFMGSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetprintlevel, NALU_HYPRE_STRUCTPFMGGETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetnumiteration, NALU_HYPRE_STRUCTPFMGGETNUMITERATION)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr           )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpfmggetfinalrelativ, NALU_HYPRE_STRUCTPFMGGETFINALRELATIV)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (norm)   ) );
}

#ifdef __cplusplus
}
#endif
