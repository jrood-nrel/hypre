/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructBiCGSTAB interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabcreate, NALU_HYPRE_SSTRUCTBICGSTABCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABCreate(
               nalu_hypre_F90_PassComm (comm),
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver) )) ;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabdestroy, NALU_HYPRE_SSTRUCTBICGSTABDESTROY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsetup, NALU_HYPRE_SSTRUCTBICGSTABSETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetup(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsolve, NALU_HYPRE_SSTRUCTBICGSTABSOLVE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSolve(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsettol, NALU_HYPRE_SSTRUCTBICGSTABSETTOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassReal (tol) ));
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetAnsoluteTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsetabsolutetol, NALU_HYPRE_SSTRUCTBICGSTABSETABSOLUTETOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetAbsoluteTol(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassReal (tol) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsetminiter, NALU_HYPRE_SSTRUCTBICGSTABSETMINITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *min_iter,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetMinIter(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (min_iter) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsetmaxiter, NALU_HYPRE_SSTRUCTBICGSTABSETMAXITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *max_iter,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetMaxIter(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (max_iter) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsetstopcri, NALU_HYPRE_SSTRUCTBICGSTABSETSTOPCRI)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *stop_crit,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetStopCrit(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (stop_crit) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsetprecond, NALU_HYPRE_SSTRUCTBICGSTABSETPRECOND)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *precond_id,
 nalu_hypre_F90_Obj *precond_solver,
 nalu_hypre_F90_Int *ierr)
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
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructBiCGSTABSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSplitSolve,
                  NALU_HYPRE_SStructSplitSetup,
                  nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 3)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructBiCGSTABSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructSysPFMGSolve,
                  NALU_HYPRE_SStructSysPFMGSetup,
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 8)
   {
      *ierr = (nalu_hypre_F90_Int)
              (NALU_HYPRE_SStructBiCGSTABSetPrecond(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                  NALU_HYPRE_SStructDiagScale,
                  NALU_HYPRE_SStructDiagScaleSetup,
                  nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, precond_solver)));
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
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsetlogging, NALU_HYPRE_SSTRUCTBICGSTABSETLOGGING)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *logging,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetLogging(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabsetprintle, NALU_HYPRE_SSTRUCTBICGSTABSETPRINTLE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *print_level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABSetPrintLevel(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabgetnumiter, NALU_HYPRE_SSTRUCTBICGSTABGETNUMITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_iterations,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABGetNumIterations(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabgetfinalre, NALU_HYPRE_SSTRUCTBICGSTABGETFINALRE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *norm,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassRealRef (norm) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructbicgstabgetresidua, NALU_HYPRE_SSTRUCTBICGSTABGETRESIDUA)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *residual,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructBiCGSTABGetResidual(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               (void **)              *residual));
}

#ifdef __cplusplus
}
#endif
