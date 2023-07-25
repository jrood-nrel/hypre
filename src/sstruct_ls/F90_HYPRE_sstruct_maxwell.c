/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructMaxwell interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellcreate, NALU_HYPRE_SSTRUCTMAXWELLCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMaxwellCreate(
               nalu_hypre_F90_PassComm (comm),
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwelldestroy, NALU_HYPRE_SSTRUCTMAXWELLDESTROY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMaxwellDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetup, NALU_HYPRE_SSTRUCTMAXWELLSETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsolve, NALU_HYPRE_SSTRUCTMAXWELLSOLVE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMaxwellSolve(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSolve2
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsolve2, NALU_HYPRE_SSTRUCTMAXWELLSOLVE2)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMaxwellSolve2(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MaxwellGrad
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_maxwellgrad, NALU_HYPRE_MAXWELLGRAD)
(nalu_hypre_F90_Obj *grid,
 nalu_hypre_F90_Obj *T,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MaxwellGrad(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, T) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetGrad
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetgrad, NALU_HYPRE_SSTRUCTMAXWELLSETGRAD)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *T,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetGrad(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, T) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetRfactors
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetrfactors, NALU_HYPRE_SSTRUCTMAXWELLSETRFACTORS)
(nalu_hypre_F90_Obj *solver,
 NALU_HYPRE_Int     (*rfactors)[3],
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetRfactors(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                rfactors[3] ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsettol, NALU_HYPRE_SSTRUCTMAXWELLSETTOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassReal (tol)    ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetconstant, NALU_HYPRE_SSTRUCTMAXWELLSETCONSTANT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *constant_coef,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetConstantCoef(
                (NALU_HYPRE_SStructSolver ) * solver,
                nalu_hypre_F90_PassInt (constant_coef)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetmaxiter, NALU_HYPRE_SSTRUCTMAXWELLSETMAXITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *max_iter,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (max_iter)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetRelChange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetrelchang, NALU_HYPRE_SSTRUCTMAXWELLSETRELCHANG)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *rel_change,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (rel_change)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetnumprere, NALU_HYPRE_SSTRUCTMAXWELLSETNUMPRERE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_pre_relax,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetNumPreRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetnumpostr, NALU_HYPRE_SSTRUCTMAXWELLSETNUMPOSTR)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_post_relax,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetNumPostRelax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (num_post_relax) ));

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetlogging, NALU_HYPRE_SSTRUCTMAXWELLSETLOGGING)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *logging,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (logging)));
}

/*--------------------------------------------------------------------------
  NALU_HYPRE_SStructMaxwellSetPrintLevel
  *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellsetprintlev, NALU_HYPRE_SSTRUCTMAXWELLSETPRINTLEV)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *print_level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellPrintLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellprintloggin, NALU_HYPRE_SSTRUCTMAXWELLPRINTLOGGIN)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *myid,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellPrintLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (myid)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellgetnumitera, NALU_HYPRE_SSTRUCTMAXWELLGETNUMITERA)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_iterations,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellgetfinalrel, NALU_HYPRE_SSTRUCTMAXWELLGETFINALREL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *norm,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassRealRef (norm)   ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellPhysBdy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellphysbdy, NALU_HYPRE_SSTRUCTMAXWELLPHYSBDY)
(nalu_hypre_F90_Obj *grid_l,
 nalu_hypre_F90_Int *num_levels,
 NALU_HYPRE_Int      (*rfactors)[3],
 NALU_HYPRE_Int      (***BdryRanks_ptr),
 NALU_HYPRE_Int      (**BdryRanksCnt_ptr),
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellPhysBdy(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructGrid, grid_l),
                nalu_hypre_F90_PassInt (num_levels),
                rfactors[3],
                BdryRanks_ptr,
                BdryRanksCnt_ptr ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellEliminateRowsCols
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwelleliminatero, NALU_HYPRE_SSTRUCTMAXWELLELIMINATERO)
(nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Int *nrows,
 nalu_hypre_F90_IntArray *rows,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellEliminateRowsCols(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassInt (nrows),
                nalu_hypre_F90_PassIntArray (rows) ));
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellZeroVector
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmaxwellzerovector, NALU_HYPRE_SSTRUCTMAXWELLZEROVECTOR)
(nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_IntArray *rows,
 nalu_hypre_F90_Int *nrows,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellZeroVector(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassIntArray (rows),
                nalu_hypre_F90_PassInt (nrows) ));
}

#ifdef __cplusplus
}
#endif
