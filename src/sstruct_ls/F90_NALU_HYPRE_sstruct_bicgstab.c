/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructBiCGSTAB interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabcreate, NALU_HYPRE_SSTRUCTBICGSTABCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABCreate(
               hypre_F90_PassComm (comm),
               hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver) )) ;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabdestroy, NALU_HYPRE_SSTRUCTBICGSTABDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABDestroy(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetup, NALU_HYPRE_SSTRUCTBICGSTABSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetup(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsolve, NALU_HYPRE_SSTRUCTBICGSTABSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSolve(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsettol, NALU_HYPRE_SSTRUCTBICGSTABSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetTol(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol) ));
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetAnsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetabsolutetol, NALU_HYPRE_SSTRUCTBICGSTABSETABSOLUTETOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetAbsoluteTol(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetminiter, NALU_HYPRE_SSTRUCTBICGSTABSETMINITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *min_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetMinIter(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (min_iter) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetmaxiter, NALU_HYPRE_SSTRUCTBICGSTABSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetMaxIter(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (max_iter) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetstopcri, NALU_HYPRE_SSTRUCTBICGSTABSETSTOPCRI)
(hypre_F90_Obj *solver,
 hypre_F90_Int *stop_crit,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetStopCrit(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (stop_crit) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetprecond, NALU_HYPRE_SSTRUCTBICGSTABSETPRECOND)
(hypre_F90_Obj *solver,
 hypre_F90_Int *precond_id,
 hypre_F90_Obj *precond_solver,
 hypre_F90_Int *ierr)
/*------------------------------------------
 *    precond_id flags mean:
 *    2 - setup a split-solver preconditioner
 *    3 - setup a syspfmg preconditioner
 *    8 - setup a DiagScale preconditioner
 *    9 - no preconditioner setup
 *----------------------------------------*/

{
   if (*precond_id == 2)
   {
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_SStructBiCGSTABSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSplitSolve,
                  NALU_HYPRE_SStructSplitSetup,
                  hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_SStructBiCGSTABSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSysPFMGSolve,
                  NALU_HYPRE_SStructSysPFMGSetup,
                  hypre_F90_PassObj (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
              (NALU_HYPRE_SStructBiCGSTABSetPrecond(
                  hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructDiagScale,
                  NALU_HYPRE_SStructDiagScaleSetup,
                  hypre_F90_PassObj (NALU_HYPRE_SStructSolver, precond_solver)));
   }
   else if (*precond_id == 9)
   {
      *ierr = 0;
   }

   else
   {
      *ierr = -1;
   }

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetlogging, NALU_HYPRE_SSTRUCTBICGSTABSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetLogging(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetprintle, NALU_HYPRE_SSTRUCTBICGSTABSETPRINTLE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *print_level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetPrintLevel(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetnumiter, NALU_HYPRE_SSTRUCTBICGSTABGETNUMITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABGetNumIterations(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetfinalre, NALU_HYPRE_SSTRUCTBICGSTABGETFINALRE)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassRealRef (norm) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetresidua, NALU_HYPRE_SSTRUCTBICGSTABGETRESIDUA)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *residual,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABGetResidual(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               (void **)              *residual));
}

#ifdef __cplusplus
}
#endif
