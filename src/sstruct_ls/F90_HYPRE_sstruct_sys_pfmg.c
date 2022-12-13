/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructSysPFMG interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgcreate, NALU_HYPRE_SSTRUCTSYSPFMGCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGCreate(
               hypre_F90_PassComm (comm),
               hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgdestroy, NALU_HYPRE_SSTRUCTSYSPFMGDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGDestroy(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetup, NALU_HYPRE_SSTRUCTSYSPFMGSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetup(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsolve, NALU_HYPRE_SSTRUCTSYSPFMGSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSolve(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsettol, NALU_HYPRE_SSTRUCTSYSPFMGSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetTol(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetmaxiter, NALU_HYPRE_SSTRUCTSYSPFMGSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetMaxIter(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (max_iter)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetrelchang, NALU_HYPRE_SSTRUCTSYSPFMGSETRELCHANG)
(hypre_F90_Obj *solver,
 hypre_F90_Int *rel_change,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetRelChange(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (rel_change)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetzerogues, NALU_HYPRE_SSTRUCTSYSPFMGSETZEROGUES)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetZeroGuess(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnonzerog, NALU_HYPRE_SSTRUCTSYSPFMGSETNONZEROG)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetNonZeroGuess(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetrelaxtyp, NALU_HYPRE_SSTRUCTSYSPFMGSETRELAXTYP)
(hypre_F90_Obj *solver,
 hypre_F90_Int *relax_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetRelaxType(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetJacobiWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetjacobiweigh, NALU_HYPRE_SSTRUCTSYSPFMGSETJACOBIWEIGH)
(hypre_F90_Obj *solver,
 hypre_F90_Real *weight,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetJacobiWeight(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnumprere, NALU_HYPRE_SSTRUCTSYSPFMGSETNUMPRERE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_pre_relax,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetNumPreRelax(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnumpostr, NALU_HYPRE_SSTRUCTSYSPFMGSETNUMPOSTR)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_post_relax,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetNumPostRelax(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (num_post_relax) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetskiprela, NALU_HYPRE_SSTRUCTSYSPFMGSETSKIPRELA)
(hypre_F90_Obj *solver,
 hypre_F90_Int *skip_relax,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetSkipRelax(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (skip_relax) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetdxyz, NALU_HYPRE_SSTRUCTSYSPFMGSETDXYZ)
(hypre_F90_Obj *solver,
 hypre_F90_RealArray *dxyz,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetDxyz(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassRealArray (dxyz)   ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetlogging, NALU_HYPRE_SSTRUCTSYSPFMGSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetLogging(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
  NALU_HYPRE_SStructSysPFMGSetPrintLevel
  *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetprintlev, NALU_HYPRE_SSTRUCTSYSPFMGSETPRINTLEV)
(hypre_F90_Obj *solver,
 hypre_F90_Int *print_level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGSetPrintLevel(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmggetnumitera, NALU_HYPRE_SSTRUCTSYSPFMGGETNUMITERA)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGGetNumIterations(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassIntRef (num_iterations) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmggetfinalrel, NALU_HYPRE_SSTRUCTSYSPFMGGETFINALREL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassRealRef (norm)   ));
}

#ifdef __cplusplus
}
#endif
