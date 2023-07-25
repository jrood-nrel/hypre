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
nalu_hypre_F90_IFACE(nalu_hypre_structpcgcreate, NALU_HYPRE_STRUCTPCGCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgdestroy, NALU_HYPRE_STRUCTPCGDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsetup, NALU_HYPRE_STRUCTPCGSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsolve, NALU_HYPRE_STRUCTPCGSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsettol, NALU_HYPRE_STRUCTPCGSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsetabstol, NALU_HYPRE_STRUCTPCGSETABSTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGSetAbsoluteTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsetmaxiter, NALU_HYPRE_STRUCTPCGSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsettwonorm, NALU_HYPRE_STRUCTPCGSETTWONORM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *two_norm,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGSetTwoNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (two_norm) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsetrelchange, NALU_HYPRE_STRUCTPCGSETRELCHANGE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rel_change,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGSetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsetprecond, NALU_HYPRE_STRUCTPCGSETPRECOND)
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
              ( NALU_HYPRE_StructPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructSMGSolve,
                   NALU_HYPRE_StructSMGSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructPFMGSolve,
                   NALU_HYPRE_StructPFMGSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructPCGSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructJacobiSolve,
                   NALU_HYPRE_StructJacobiSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructPCGSetPrecond(
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
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsetlogging, NALU_HYPRE_STRUCTPCGSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcgsetprintlevel, NALU_HYPRE_STRUCTPCGSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcggetnumiterations, NALU_HYPRE_STRUCTPCGGETNUMITERATIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structpcggetfinalrelative, NALU_HYPRE_STRUCTPCGGETFINALRELATIVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructPCGGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (norm) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structdiagscalesetup, NALU_HYPRE_STRUCTDIAGSCALESETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructDiagScaleSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, y),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x)     ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structdiagscale, NALU_HYPRE_STRUCTDIAGSCALE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *HA,
  nalu_hypre_F90_Obj *Hy,
  nalu_hypre_F90_Obj *Hx,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructDiagScale(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, HA),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, Hy),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, Hx)     ) );
}

#ifdef __cplusplus
}
#endif
