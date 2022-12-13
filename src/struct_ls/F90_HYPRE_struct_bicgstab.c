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
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabcreate, NALU_HYPRE_STRUCTBICGSTABCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabdestroy, NALU_HYPRE_STRUCTBICGSTABDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABDestroy(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetup, NALU_HYPRE_STRUCTBICGSTABSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetup(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsolve, NALU_HYPRE_STRUCTBICGSTABSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSolve(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsettol, NALU_HYPRE_STRUCTBICGSTABSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetabstol, NALU_HYPRE_STRUCTBICGSTABSETABSTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetAbsoluteTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetmaxiter, NALU_HYPRE_STRUCTBICGSTABSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetprecond, NALU_HYPRE_STRUCTBICGSTABSETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Int *precond_id,
  hypre_F90_Obj *precond_solver,
  hypre_F90_Int *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    * 7 - setup a jacobi preconditioner
    * 8 - setup a ds preconditioner
    * 9 - dont setup a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_StructBiCGSTABSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructSMGSolve,
                   NALU_HYPRE_StructSMGSetup,
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_StructBiCGSTABSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructPFMGSolve,
                   NALU_HYPRE_StructPFMGSetup,
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_StructBiCGSTABSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructJacobiSolve,
                   NALU_HYPRE_StructJacobiSetup,
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_StructBiCGSTABSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructDiagScale,
                   NALU_HYPRE_StructDiagScaleSetup,
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
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
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetlogging, NALU_HYPRE_STRUCTBICGSTABSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetprintlev, NALU_HYPRE_STRUCTBICGSTABSETPRINTLEV)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetnumitera, NALU_HYPRE_STRUCTBICGSTABGETNUMITERA)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetresidual, NALU_HYPRE_STRUCTBICGSTABGETRESIDUAL)
( hypre_F90_Obj *solver,
  void *residual,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABGetResidual(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                (void **)          residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetfinalrel, NALU_HYPRE_STRUCTBICGSTABGETFINALREL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
