/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRGMRES Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmrescreate, NALU_HYPRE_PARCSRGMRESCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresdestroy, NALU_HYPRE_PARCSRGMRESDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetup, NALU_HYPRE_PARCSRGMRESSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressolve, NALU_HYPRE_PARCSRGMRESSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetkdim, NALU_HYPRE_PARCSRGMRESSETKDIM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *kdim,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetKDim(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (kdim)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressettol, NALU_HYPRE_PARCSRGMRESSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetabsolutetol, NALU_HYPRE_PARCSRGMRESSETABSOLUTETOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetAbsoluteTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetminiter, NALU_HYPRE_PARCSRGMRESSETMINITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetMinIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetmaxiter, NALU_HYPRE_PARCSRGMRESSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetstopcrit, NALU_HYPRE_PARCSRGMRESSETSTOPCRIT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *stop_crit,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetStopCrit(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetprecond, NALU_HYPRE_PARCSRGMRESSETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Int *precond_id,
  hypre_F90_Obj *precond_solver,
  hypre_F90_Int *ierr          )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    *  0 - no preconditioner
    *  1 - set up a ds preconditioner
    *  2 - set up an amg preconditioner
    *  3 - set up a pilut preconditioner
    *  4 - set up a parasails preconditioner
    *  5 - set up a Euclid preconditioner
    *  6 - set up a ILU preconditioner
    *  7 - set up a MGR preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRDiagScale,
                   NALU_HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_BoomerAMGSolve,
                   NALU_HYPRE_BoomerAMGSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRPilutSolve,
                   NALU_HYPRE_ParCSRPilutSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRParaSailsSolve,
                   NALU_HYPRE_ParCSRParaSailsSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_EuclidSolve,
                   NALU_HYPRE_EuclidSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ILUSolve,
                   NALU_HYPRE_ILUSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRGMRESSetPrecond(
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
 * NALU_HYPRE_ParCSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetprecond, NALU_HYPRE_PARCSRGMRESGETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *precond_solver_ptr,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESGetPrecond(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetlogging, NALU_HYPRE_PARCSRGMRESSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetprintlevel, NALU_HYPRE_PARCSRGMRESSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetnumiteratio, NALU_HYPRE_PARCSRGMRESGETNUMITERATIO)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetfinalrelati, NALU_HYPRE_PARCSRGMRESGETFINALRELATI)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
