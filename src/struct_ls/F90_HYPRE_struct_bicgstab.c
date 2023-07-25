/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabcreate, NALU_HYPRE_STRUCTBICGSTABCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabdestroy, NALU_HYPRE_STRUCTBICGSTABDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabsetup, NALU_HYPRE_STRUCTBICGSTABSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabsolve, NALU_HYPRE_STRUCTBICGSTABSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabsettol, NALU_HYPRE_STRUCTBICGSTABSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabsetabstol, NALU_HYPRE_STRUCTBICGSTABSETABSTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetAbsoluteTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabsetmaxiter, NALU_HYPRE_STRUCTBICGSTABSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabsetprecond, NALU_HYPRE_STRUCTBICGSTABSETPRECOND)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *precond_id,
  nalu_hypre_F90_Obj *precond_solver,
  nalu_hypre_F90_Int *ierr           )
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
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructBiCGSTABSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructSMGSolve,
                   NALU_HYPRE_StructSMGSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructBiCGSTABSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructPFMGSolve,
                   NALU_HYPRE_StructPFMGSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructBiCGSTABSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructJacobiSolve,
                   NALU_HYPRE_StructJacobiSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructBiCGSTABSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructDiagScale,
                   NALU_HYPRE_StructDiagScaleSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
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
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabsetlogging, NALU_HYPRE_STRUCTBICGSTABSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabsetprintlev, NALU_HYPRE_STRUCTBICGSTABSETPRINTLEV)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabgetnumitera, NALU_HYPRE_STRUCTBICGSTABGETNUMITERA)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabgetresidual, NALU_HYPRE_STRUCTBICGSTABGETRESIDUAL)
( nalu_hypre_F90_Obj *solver,
  void *residual,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABGetResidual(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                (void **)          residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structbicgstabgetfinalrel, NALU_HYPRE_STRUCTBICGSTABGETFINALREL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
