/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRHybrid Fortran Interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *    NALU_HYPRE_ParCSRHybridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridcreate, NALU_HYPRE_PARCSRHYBRIDCREATE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridCreate(
               hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybriddestroy, NALU_HYPRE_PARCSRHYBRIDDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridDestroy(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetup, NALU_HYPRE_PARCSRHYBRIDSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetup(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
               hypre_F90_PassObj (NALU_HYPRE_ParVector, x)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsolve, NALU_HYPRE_PARCSRHYBRIDSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSolve(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
               hypre_F90_PassObj (NALU_HYPRE_ParVector, x)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettol, NALU_HYPRE_PARCSRHYBRIDSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetTol(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (tol)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetconvergenc, NALU_HYPRE_PARCSRHYBRIDSETCONVERGENC)
(hypre_F90_Obj *solver,
 hypre_F90_Real *cf_tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetConvergenceTol(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (cf_tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetdscgmaxite, NALU_HYPRE_PARCSRHYBRIDSETDSCGMAXITE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *dscg_max_its,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetDSCGMaxIter(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (dscg_max_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetpcgmaxiter, NALU_HYPRE_PARCSRHYBRIDSETPCGMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *pcg_max_its,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetPCGMaxIter(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (pcg_max_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetsolvertype, NALU_HYPRE_PARCSRHYBRIDSETSOLVERTYPE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *solver_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetSolverType(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (solver_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetkdim, NALU_HYPRE_PARCSRHYBRIDSETKDIM)
(hypre_F90_Obj *solver,
 hypre_F90_Int *kdim,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetKDim(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (kdim)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettwonorm, NALU_HYPRE_PARCSRHYBRIDSETTWONORM)
(hypre_F90_Obj *solver,
 hypre_F90_Int *two_norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetTwoNorm(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (two_norm)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetstopcrit, NALU_HYPRE_PARCSRHYBRIDSETSTOPCRIT)
(hypre_F90_Obj *solver,
 hypre_F90_Int *stop_crit,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetStopCrit(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelchange, NALU_HYPRE_PARCSRHYBRIDSETRELCHANGE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *rel_change,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelChange(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetprecond, NALU_HYPRE_PARCSRHYBRIDSETPRECOND)
(hypre_F90_Obj *solver,
 hypre_F90_Int *precond_id,
 hypre_F90_Obj *precond_solver,
 hypre_F90_Int *ierr)
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
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_ParCSRDiagScale,
                  NALU_HYPRE_ParCSRDiagScaleSetup,
                  NULL                      ));
   }
   else if (*precond_id == 2)
   {
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_BoomerAMGSolve,
                  NALU_HYPRE_BoomerAMGSetup,
                  (NALU_HYPRE_Solver)         * precond_solver ));
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_ParCSRPilutSolve,
                  NALU_HYPRE_ParCSRPilutSetup,
                  (NALU_HYPRE_Solver)          * precond_solver ));
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_ParCSRParaSailsSolve,
                  NALU_HYPRE_ParCSRParaSailsSetup,
                  (NALU_HYPRE_Solver)          * precond_solver ));
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_ParCSRHybridSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                  NALU_HYPRE_EuclidSolve,
                  NALU_HYPRE_EuclidSetup,
                  (NALU_HYPRE_Solver)          * precond_solver ));
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRHybridSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ILUSolve,
                   NALU_HYPRE_ILUSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRHybridSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
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
hypre_F90_IFACE(hypre_parcsrhybridsetlogging, NALU_HYPRE_PARCSRHYBRIDSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetLogging(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (logging)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetprintlevel, NALU_HYPRE_PARCSRHYBRIDSETPRINTLEVEL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *print_level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetPrintLevel(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (print_level)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetstrongthre, NALU_HYPRE_PARCSRHYBRIDSETSTRONGTHRE)
(hypre_F90_Obj *solver,
 hypre_F90_Real *strong_threshold,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetStrongThreshold(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (strong_threshold) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmaxrowsum, NALU_HYPRE_PARCSRHYBRIDSETMAXROWSUM)
(hypre_F90_Obj *solver,
 hypre_F90_Real *max_row_sum,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetMaxRowSum(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (max_row_sum)   ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettruncfacto, NALU_HYPRE_PARCSRHYBRIDSETTRUNCFACTO)
(hypre_F90_Obj *solver,
 hypre_F90_Real *trunc_factor,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetTruncFactor(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (trunc_factor) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetpmaxelmts, NALU_HYPRE_PARCSRHYBRIDSETPMAXELMTS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *p_max_elmts,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetPMaxElmts(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (p_max_elmts) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmaxlevels, NALU_HYPRE_PARCSRHYBRIDSETMAXLEVELS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_levels,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetMaxLevels(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (max_levels)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMeasureType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmeasuretyp, NALU_HYPRE_PARCSRHYBRIDSETMEASURETYP)
(hypre_F90_Obj *solver,
 hypre_F90_Int *measure_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetMeasureType(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (measure_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcoarsentyp, NALU_HYPRE_PARCSRHYBRIDSETCOARSENTYP)
(hypre_F90_Obj *solver,
 hypre_F90_Int *coarsen_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCoarsenType(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (coarsen_type)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetinterptyp, NALU_HYPRE_PARCSRHYBRIDSETINTERPTYP)
(hypre_F90_Obj *solver,
 hypre_F90_Int *interp_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCoarsenType(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (interp_type)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcycletype, NALU_HYPRE_PARCSRHYBRIDSETCYCLETYPE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *cycle_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCycleType(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (cycle_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumgridswe, NALU_HYPRE_PARCSRHYBRIDSETNUMGRIDSWE)
(hypre_F90_Obj *solver,
 hypre_F90_IntArray *num_grid_sweeps,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNumGridSweeps(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassIntArray (num_grid_sweeps) ));
}

/*------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetgridrelaxt, NALU_HYPRE_PARCSRHYBRIDSETGRIDRELAXT)
(hypre_F90_Obj *solver,
 hypre_F90_IntArray *grid_relax_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetGridRelaxType(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassIntArray (grid_relax_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetgridrelaxp, NALU_HYPRE_PARCSRHYBRIDSETGRIDRELAXP)
(hypre_F90_Obj *solver,
 hypre_F90_Int *grid_relax_points,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetGridRelaxPoints(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               (NALU_HYPRE_Int **)        grid_relax_points  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumsweeps, NALU_HYPRE_PARCSRHYBRIDSETNUMSWEEPS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_sweeps,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNumSweeps(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (num_sweeps)  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcyclenumsw, NALU_HYPRE_PARCSRHYBRIDSETCYCLENUMSW)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_sweeps,
 hypre_F90_Int *k,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCycleNumSweeps(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (num_sweeps),
               hypre_F90_PassInt (k) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxtype, NALU_HYPRE_PARCSRHYBRIDSETRELAXTYPE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *relax_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelaxType(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcyclerelax, NALU_HYPRE_PARCSRHYBRIDSETCYCLERELAX)
(hypre_F90_Obj *solver,
 hypre_F90_Int *relax_type,
 hypre_F90_Int *k,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetCycleRelaxType(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (relax_type),
               hypre_F90_PassInt (k) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetAggNumLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetaggnumlev, NALU_HYPRE_PARCSRHYBRIDSETAGGNUMLEV)
(hypre_F90_Obj *solver,
 hypre_F90_Int *agg_nl,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetAggNumLevels(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (agg_nl) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumPaths
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumpaths, NALU_HYPRE_PARCSRHYBRIDSETNUMPATHS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_paths,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNumPaths(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (num_paths) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumFunctions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumfunc, NALU_HYPRE_PARCSRHYBRIDSETNUMFUNC)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_fun,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNumFunctions(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (num_fun) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNodal
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnodal, NALU_HYPRE_PARCSRHYBRIDSETNODAL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *nodal,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNodal(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (nodal) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetKeepTranspose
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetkeeptrans, NALU_HYPRE_PARCSRHYBRIDSETKEEPTRANS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *keepT,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetKeepTranspose(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (keepT) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetDofFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetdoffunc, NALU_HYPRE_PARCSRHYBRIDSETDOFFUNC)
(hypre_F90_Obj *solver,
 hypre_F90_IntArray *dof_func,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetDofFunc(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassIntArray (dof_func) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNonGalerkinTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnongaltol, NALU_HYPRE_PARCSRHYBRIDSETNONGALTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ng_num_tol,
 hypre_F90_RealArray *nongal_tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetNonGalerkinTol(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (ng_num_tol),
               hypre_F90_PassRealArray (nongal_tol) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxOrder
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxorder, NALU_HYPRE_PARCSRHYBRIDSETRELAXORDER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *relax_order,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelaxOrder(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (relax_order) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxwt, NALU_HYPRE_PARCSRHYBRIDSETRELAXWT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *relax_wt,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelaxWt(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (relax_wt) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetlevelrelax, NALU_HYPRE_PARCSRHYBRIDSETLEVELRELAX)
(hypre_F90_Obj *solver,
 hypre_F90_Real *relax_wt,
 hypre_F90_Int *level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetLevelRelaxWt(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (relax_wt),
               hypre_F90_PassInt (level) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetouterwt, NALU_HYPRE_PARCSRHYBRIDSETOUTERWT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *outer_wt,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetOuterWt(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (outer_wt) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetlevelouter, NALU_HYPRE_PARCSRHYBRIDSETLEVELOUTER)
(hypre_F90_Obj *solver,
 hypre_F90_Real *outer_wt,
 hypre_F90_Int *level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetLevelOuterWt(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (outer_wt),
               hypre_F90_PassInt (level) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxweigh, NALU_HYPRE_PARCSRHYBRIDSETRELAXWEIGH)
(hypre_F90_Obj *solver,
 hypre_F90_RealArray *relax_weight,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetRelaxWeight(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassRealArray (relax_weight) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetOmega
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetomega, NALU_HYPRE_PARCSRHYBRIDSETOMEGA)
(hypre_F90_Obj *solver,
 hypre_F90_RealArray *omega,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridSetOmega(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassRealArray (omega) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetnumiterati, NALU_HYPRE_PARCSRHYBRIDGETNUMITERATI)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_its,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridGetNumIterations(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassIntRef (num_its) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetdscgnumite, NALU_HYPRE_PARCSRHYBRIDGETDSCGNUMITE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *dscg_num_its,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridGetDSCGNumIterations(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassIntRef (dscg_num_its) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetpcgnumiter, NALU_HYPRE_PARCSRHYBRIDGETPCGNUMITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *pcg_num_its,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridGetPCGNumIterations(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassIntRef (pcg_num_its) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetfinalrelat, NALU_HYPRE_PARCSRHYBRIDGETFINALRELAT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassRealRef (norm) ));
}

#ifdef __cplusplus
}
#endif
