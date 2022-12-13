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

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellcreate, NALU_HYPRE_SSTRUCTMAXWELLCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMaxwellCreate(
               hypre_F90_PassComm (comm),
               hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwelldestroy, NALU_HYPRE_SSTRUCTMAXWELLDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMaxwellDestroy(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetup, NALU_HYPRE_SSTRUCTMAXWELLSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetup(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsolve, NALU_HYPRE_SSTRUCTMAXWELLSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMaxwellSolve(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSolve2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsolve2, NALU_HYPRE_SSTRUCTMAXWELLSOLVE2)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMaxwellSolve2(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MaxwellGrad
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_maxwellgrad, NALU_HYPRE_MAXWELLGRAD)
(hypre_F90_Obj *grid,
 hypre_F90_Obj *T,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MaxwellGrad(
                hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
                hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, T) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetGrad
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetgrad, NALU_HYPRE_SSTRUCTMAXWELLSETGRAD)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *T,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetGrad(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, T) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetRfactors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetrfactors, NALU_HYPRE_SSTRUCTMAXWELLSETRFACTORS)
(hypre_F90_Obj *solver,
 NALU_HYPRE_Int     (*rfactors)[3],
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetRfactors(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                rfactors[3] ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsettol, NALU_HYPRE_SSTRUCTMAXWELLSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetTol(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassReal (tol)    ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetconstant, NALU_HYPRE_SSTRUCTMAXWELLSETCONSTANT)
(hypre_F90_Obj *solver,
 hypre_F90_Int *constant_coef,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetConstantCoef(
                (NALU_HYPRE_SStructSolver ) * solver,
                hypre_F90_PassInt (constant_coef)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetmaxiter, NALU_HYPRE_SSTRUCTMAXWELLSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (max_iter)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetrelchang, NALU_HYPRE_SSTRUCTMAXWELLSETRELCHANG)
(hypre_F90_Obj *solver,
 hypre_F90_Int *rel_change,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetRelChange(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (rel_change)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetnumprere, NALU_HYPRE_SSTRUCTMAXWELLSETNUMPRERE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_pre_relax,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetNumPreRelax(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetnumpostr, NALU_HYPRE_SSTRUCTMAXWELLSETNUMPOSTR)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_post_relax,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetNumPostRelax(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (num_post_relax) ));

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetlogging, NALU_HYPRE_SSTRUCTMAXWELLSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (logging)));
}

/*--------------------------------------------------------------------------
  NALU_HYPRE_SStructMaxwellSetPrintLevel
  *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetprintlev, NALU_HYPRE_SSTRUCTMAXWELLSETPRINTLEV)
(hypre_F90_Obj *solver,
 hypre_F90_Int *print_level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellPrintLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellprintloggin, NALU_HYPRE_SSTRUCTMAXWELLPRINTLOGGIN)
(hypre_F90_Obj *solver,
 hypre_F90_Int *myid,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellPrintLogging(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (myid)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellgetnumitera, NALU_HYPRE_SSTRUCTMAXWELLGETNUMITERA)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassIntRef (num_iterations) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellgetfinalrel, NALU_HYPRE_SSTRUCTMAXWELLGETFINALREL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassRealRef (norm)   ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellPhysBdy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellphysbdy, NALU_HYPRE_SSTRUCTMAXWELLPHYSBDY)
(hypre_F90_Obj *grid_l,
 hypre_F90_Int *num_levels,
 NALU_HYPRE_Int      (*rfactors)[3],
 NALU_HYPRE_Int      (***BdryRanks_ptr),
 NALU_HYPRE_Int      (**BdryRanksCnt_ptr),
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellPhysBdy(
                hypre_F90_PassObjRef (NALU_HYPRE_SStructGrid, grid_l),
                hypre_F90_PassInt (num_levels),
                rfactors[3],
                BdryRanks_ptr,
                BdryRanksCnt_ptr ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellEliminateRowsCols
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwelleliminatero, NALU_HYPRE_SSTRUCTMAXWELLELIMINATERO)
(hypre_F90_Obj *A,
 hypre_F90_Int *nrows,
 hypre_F90_IntArray *rows,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellEliminateRowsCols(
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassInt (nrows),
                hypre_F90_PassIntArray (rows) ));
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellZeroVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellzerovector, NALU_HYPRE_SSTRUCTMAXWELLZEROVECTOR)
(hypre_F90_Obj *b,
 hypre_F90_IntArray *rows,
 hypre_F90_Int *nrows,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructMaxwellZeroVector(
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassIntArray (rows),
                hypre_F90_PassInt (nrows) ));
}

#ifdef __cplusplus
}
#endif
