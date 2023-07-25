/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRHybrid Fortran Interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *    NALU_HYPRE_ParCSRHybridCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridcreate, NALU_HYPRE_PARCSRHYBRIDCREATE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridCreate(
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybriddestroy, NALU_HYPRE_PARCSRHYBRIDDESTROY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetup, NALU_HYPRE_PARCSRHYBRIDSETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetup(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsolve, NALU_HYPRE_PARCSRHYBRIDSOLVE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSolve(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsettol, NALU_HYPRE_PARCSRHYBRIDSETTOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (tol)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetconvergenc, NALU_HYPRE_PARCSRHYBRIDSETCONVERGENC)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *cf_tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetConvergenceTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (cf_tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetdscgmaxite, NALU_HYPRE_PARCSRHYBRIDSETDSCGMAXITE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *dscg_max_its,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetDSCGMaxIter(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (dscg_max_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetpcgmaxiter, NALU_HYPRE_PARCSRHYBRIDSETPCGMAXITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *pcg_max_its,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetPCGMaxIter(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (pcg_max_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetsolvertype, NALU_HYPRE_PARCSRHYBRIDSETSOLVERTYPE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *solver_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetSolverType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (solver_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetKDim
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetkdim, NALU_HYPRE_PARCSRHYBRIDSETKDIM)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *kdim,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetKDim(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (kdim)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsettwonorm, NALU_HYPRE_PARCSRHYBRIDSETTWONORM)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *two_norm,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetTwoNorm(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (two_norm)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetstopcrit, NALU_HYPRE_PARCSRHYBRIDSETSTOPCRIT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *stop_crit,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetStopCrit(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetrelchange, NALU_HYPRE_PARCSRHYBRIDSETRELCHANGE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *rel_change,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelChange(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetprecond, NALU_HYPRE_PARCSRHYBRIDSETPRECOND)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *precond_id,
 nalu_hypre_F90_Obj *precond_solver,
 nalu_hypre_F90_Int *ierr)
{
   /*----------------------------------------------------------------
    * precond_id definitions
    * 0 - no preconditioner
    * 1 - use diagscale preconditioner
    * 2 - use amg preconditioner
    * 3 - use pilut preconditioner
    * 4 - use parasails preconditioner
    * 5 - use Euclid preconditioner
    * 6 - use ILU preconditioner
    * 7 - use MGR preconditioner
    *---------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_ParCSRDiagScale,
                  NALU_HYPRE_ParCSRDiagScaleSetup,
                  NULL                      ));
   }
   else if (*precond_id == 2)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_BoomerAMGSolve,
                  NALU_HYPRE_BoomerAMGSetup,
                  (NALU_HYPRE_Solver)         * precond_solver ));
   }
   else if (*precond_id == 3)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_ParCSRPilutSolve,
                  NALU_HYPRE_ParCSRPilutSetup,
                  (NALU_HYPRE_Solver)          * precond_solver ));
   }
   else if (*precond_id == 4)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_ParCSRParaSailsSolve,
                  NALU_HYPRE_ParCSRParaSailsSetup,
                  (NALU_HYPRE_Solver)          * precond_solver ));
   }
   else if (*precond_id == 5)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_EuclidSolve,
                  NALU_HYPRE_EuclidSetup,
                  (NALU_HYPRE_Solver)          * precond_solver ));
   }
   else if (*precond_id == 6)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRHybridSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ILUSolve,
                   NALU_HYPRE_ILUSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRHybridSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_MGRSolve,
                   NALU_HYPRE_MGRSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetlogging, NALU_HYPRE_PARCSRHYBRIDSETLOGGING)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *logging,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetLogging(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (logging)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetprintlevel, NALU_HYPRE_PARCSRHYBRIDSETPRINTLEVEL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *print_level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetPrintLevel(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (print_level)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetstrongthre, NALU_HYPRE_PARCSRHYBRIDSETSTRONGTHRE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *strong_threshold,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetStrongThreshold(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (strong_threshold) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetmaxrowsum, NALU_HYPRE_PARCSRHYBRIDSETMAXROWSUM)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *max_row_sum,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetMaxRowSum(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (max_row_sum)   ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsettruncfacto, NALU_HYPRE_PARCSRHYBRIDSETTRUNCFACTO)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *trunc_factor,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetTruncFactor(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (trunc_factor) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPMaxElmts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetpmaxelmts, NALU_HYPRE_PARCSRHYBRIDSETPMAXELMTS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *p_max_elmts,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetPMaxElmts(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (p_max_elmts) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetmaxlevels, NALU_HYPRE_PARCSRHYBRIDSETMAXLEVELS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *max_levels,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetMaxLevels(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (max_levels)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMeasureType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetmeasuretyp, NALU_HYPRE_PARCSRHYBRIDSETMEASURETYP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *measure_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetMeasureType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (measure_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetcoarsentyp, NALU_HYPRE_PARCSRHYBRIDSETCOARSENTYP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *coarsen_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCoarsenType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (coarsen_type)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetInterpType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetinterptyp, NALU_HYPRE_PARCSRHYBRIDSETINTERPTYP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *interp_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCoarsenType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (interp_type)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCycleType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetcycletype, NALU_HYPRE_PARCSRHYBRIDSETCYCLETYPE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *cycle_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCycleType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (cycle_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetnumgridswe, NALU_HYPRE_PARCSRHYBRIDSETNUMGRIDSWE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_IntArray *num_grid_sweeps,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNumGridSweeps(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassIntArray (num_grid_sweeps) ));
}

/*------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetgridrelaxt, NALU_HYPRE_PARCSRHYBRIDSETGRIDRELAXT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_IntArray *grid_relax_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetGridRelaxType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassIntArray (grid_relax_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetgridrelaxp, NALU_HYPRE_PARCSRHYBRIDSETGRIDRELAXP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *grid_relax_points,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetGridRelaxPoints(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               (NALU_HYPRE_Int **)        grid_relax_points  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumSweeps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetnumsweeps, NALU_HYPRE_PARCSRHYBRIDSETNUMSWEEPS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_sweeps,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNumSweeps(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (num_sweeps)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetcyclenumsw, NALU_HYPRE_PARCSRHYBRIDSETCYCLENUMSW)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_sweeps,
 nalu_hypre_F90_Int *k,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCycleNumSweeps(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (num_sweeps),
               nalu_hypre_F90_PassInt (k) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetrelaxtype, NALU_HYPRE_PARCSRHYBRIDSETRELAXTYPE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *relax_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelaxType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetcyclerelax, NALU_HYPRE_PARCSRHYBRIDSETCYCLERELAX)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *relax_type,
 nalu_hypre_F90_Int *k,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCycleRelaxType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (relax_type),
               nalu_hypre_F90_PassInt (k) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetAggNumLevels
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetaggnumlev, NALU_HYPRE_PARCSRHYBRIDSETAGGNUMLEV)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *agg_nl,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetAggNumLevels(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (agg_nl) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumPaths
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetnumpaths, NALU_HYPRE_PARCSRHYBRIDSETNUMPATHS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_paths,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNumPaths(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (num_paths) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumFunctions
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetnumfunc, NALU_HYPRE_PARCSRHYBRIDSETNUMFUNC)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_fun,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNumFunctions(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (num_fun) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNodal
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetnodal, NALU_HYPRE_PARCSRHYBRIDSETNODAL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *nodal,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNodal(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (nodal) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetKeepTranspose
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetkeeptrans, NALU_HYPRE_PARCSRHYBRIDSETKEEPTRANS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *keepT,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetKeepTranspose(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (keepT) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetDofFunc
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetdoffunc, NALU_HYPRE_PARCSRHYBRIDSETDOFFUNC)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_IntArray *dof_func,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetDofFunc(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassIntArray (dof_func) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNonGalerkinTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetnongaltol, NALU_HYPRE_PARCSRHYBRIDSETNONGALTOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ng_num_tol,
 nalu_hypre_F90_RealArray *nongal_tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNonGalerkinTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (ng_num_tol),
               nalu_hypre_F90_PassRealArray (nongal_tol) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxOrder
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetrelaxorder, NALU_HYPRE_PARCSRHYBRIDSETRELAXORDER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *relax_order,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelaxOrder(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (relax_order) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetrelaxwt, NALU_HYPRE_PARCSRHYBRIDSETRELAXWT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *relax_wt,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelaxWt(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (relax_wt) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetlevelrelax, NALU_HYPRE_PARCSRHYBRIDSETLEVELRELAX)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *relax_wt,
 nalu_hypre_F90_Int *level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetLevelRelaxWt(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (relax_wt),
               nalu_hypre_F90_PassInt (level) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetOuterWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetouterwt, NALU_HYPRE_PARCSRHYBRIDSETOUTERWT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *outer_wt,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetOuterWt(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (outer_wt) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetlevelouter, NALU_HYPRE_PARCSRHYBRIDSETLEVELOUTER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *outer_wt,
 nalu_hypre_F90_Int *level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetLevelOuterWt(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (outer_wt),
               nalu_hypre_F90_PassInt (level) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetrelaxweigh, NALU_HYPRE_PARCSRHYBRIDSETRELAXWEIGH)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_RealArray *relax_weight,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelaxWeight(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassRealArray (relax_weight) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetOmega
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridsetomega, NALU_HYPRE_PARCSRHYBRIDSETOMEGA)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_RealArray *omega,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetOmega(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassRealArray (omega) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridgetnumiterati, NALU_HYPRE_PARCSRHYBRIDGETNUMITERATI)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_its,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridGetNumIterations(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassIntRef (num_its) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridgetdscgnumite, NALU_HYPRE_PARCSRHYBRIDGETDSCGNUMITE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *dscg_num_its,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridGetDSCGNumIterations(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassIntRef (dscg_num_its) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridgetpcgnumiter, NALU_HYPRE_PARCSRHYBRIDGETPCGNUMITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *pcg_num_its,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridGetPCGNumIterations(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassIntRef (pcg_num_its) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrhybridgetfinalrelat, NALU_HYPRE_PARCSRHYBRIDGETFINALRELAT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *norm,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassRealRef (norm) ));
}

#ifdef __cplusplus
}
#endif
