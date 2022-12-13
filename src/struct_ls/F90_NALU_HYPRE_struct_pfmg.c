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
 * NALU_HYPRE_StructPFMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgcreate, NALU_HYPRE_STRUCTPFMGCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgdestroy, NALU_HYPRE_STRUCTPFMGDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGDestroy(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetup, NALU_HYPRE_STRUCTPFMGSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetup(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsolve, NALU_HYPRE_STRUCTPFMGSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSolve(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetTol, NALU_HYPRE_StructPFMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsettol, NALU_HYPRE_STRUCTPFMGSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol)    ) );
}

void
hypre_F90_IFACE(hypre_structpfmggettol, NALU_HYPRE_STRUCTPFMGGETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (tol)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetMaxIter, NALU_HYPRE_StructPFMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetmaxiter, NALU_HYPRE_STRUCTPFMGSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (max_iter)  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetmaxiter, NALU_HYPRE_STRUCTPFMGGETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (max_iter)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetMaxLevels, NALU_HYPRE_StructPFMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetmaxlevels, NALU_HYPRE_STRUCTPFMGSETMAXLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_levels,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetMaxLevels(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (max_levels)  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetmaxlevels, NALU_HYPRE_STRUCTPFMGGETMAXLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_levels,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetMaxLevels(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (max_levels)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetRelChange, NALU_HYPRE_StructPFMGGetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelchange, NALU_HYPRE_STRUCTPFMGSETRELCHANGE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rel_change,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetRelChange(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (rel_change)  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetrelchange, NALU_HYPRE_STRUCTPFMGGETRELCHANGE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rel_change,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetRelChange(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetZeroGuess, NALU_HYPRE_StructPFMGGetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetzeroguess, NALU_HYPRE_STRUCTPFMGSETZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetzeroguess, NALU_HYPRE_STRUCTPFMGGETZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *zeroguess,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnonzeroguess, NALU_HYPRE_STRUCTPFMGSETNONZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetNonZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetRelaxType, NALU_HYPRE_StructPFMGGetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelaxtype, NALU_HYPRE_STRUCTPFMGSETRELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (relax_type) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetrelaxtype, NALU_HYPRE_STRUCTPFMGGETRELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_structpfmgsetjacobiweigh, NALU_HYPRE_STRUCTPFMGSETJACOBIWEIGH)
(hypre_F90_Obj *solver,
 hypre_F90_Real *weight,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_StructPFMGSetJacobiWeight(
               hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
               hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetRAPType, NALU_HYPRE_StructPFMGSetRapType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetraptype, NALU_HYPRE_STRUCTPFMGSETRAPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rap_type,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetRAPType(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (rap_type) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetraptype, NALU_HYPRE_STRUCTPFMGGETRAPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rap_type,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetRAPType(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (rap_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetNumPreRelax, NALU_HYPRE_StructPFMGGetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumprerelax, NALU_HYPRE_STRUCTPFMGSETNUMPRERELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_pre_relax,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetNumPreRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_pre_relax) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetnumprerelax, NALU_HYPRE_STRUCTPFMGGETNUMPRERELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_pre_relax,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetNumPreRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_pre_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetNumPostRelax, NALU_HYPRE_StructPFMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumpostrelax, NALU_HYPRE_STRUCTPFMGSETNUMPOSTRELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_post_relax,
  hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetNumPostRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_post_relax) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetnumpostrelax, NALU_HYPRE_STRUCTPFMGGETNUMPOSTRELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_post_relax,
  hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetNumPostRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_post_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetSkipRelax, NALU_HYPRE_StructPFMGGetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetskiprelax, NALU_HYPRE_STRUCTPFMGSETSKIPRELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *skip_relax,
  hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetSkipRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (skip_relax) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetskiprelax, NALU_HYPRE_STRUCTPFMGGETSKIPRELAX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *skip_relax,
  hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetSkipRelax(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (skip_relax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetdxyz, NALU_HYPRE_STRUCTPFMGSETDXYZ)
( hypre_F90_Obj *solver,
  hypre_F90_RealArray *dxyz,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetDxyz(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealArray (dxyz)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetLogging, NALU_HYPRE_StructPFMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetlogging, NALU_HYPRE_STRUCTPFMGSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (logging) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetlogging, NALU_HYPRE_STRUCTPFMGGETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetLogging(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGSetPrintLevel, NALU_HYPRE_StructPFMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetprintlevel, NALU_HYPRE_STRUCTPFMGSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (print_level) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetprintlevel, NALU_HYPRE_STRUCTPFMGGETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmggetnumiteration, NALU_HYPRE_STRUCTPFMGGETNUMITERATION)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmggetfinalrelativ, NALU_HYPRE_STRUCTPFMGGETFINALRELATIV)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm)   ) );
}

#ifdef __cplusplus
}
#endif
