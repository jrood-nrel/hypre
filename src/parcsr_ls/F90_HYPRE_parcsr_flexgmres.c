/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRFlexGMRES Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmrescreate, NALU_HYPRE_PARCSRFLEXGMRESCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmresdestroy, NALU_HYPRE_PARCSRFLEXGMRESDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetup, NALU_HYPRE_PARCSRFLEXGMRESSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressolve, NALU_HYPRE_PARCSRFLEXGMRESSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetkdim, NALU_HYPRE_PARCSRFLEXGMRESSETKDIM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *kdim,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESSetKDim(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (kdim)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressettol, NALU_HYPRE_PARCSRFLEXGMRESSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESSetTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetabsolutetol, NALU_HYPRE_PARCSRFLEXGMRESSETABSOLUTETOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESSetAbsoluteTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetminiter, NALU_HYPRE_PARCSRFLEXGMRESSETMINITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESSetMinIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetmaxiter, NALU_HYPRE_PARCSRFLEXGMRESSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetprecond, NALU_HYPRE_PARCSRFLEXGMRESSETPRECOND)
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
              ( NALU_HYPRE_ParCSRFlexGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRDiagScale,
                   NALU_HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRFlexGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_BoomerAMGSolve,
                   NALU_HYPRE_BoomerAMGSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRFlexGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRPilutSolve,
                   NALU_HYPRE_ParCSRPilutSetup,
                   (NALU_HYPRE_Solver)      * precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRFlexGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRParaSailsSolve,
                   NALU_HYPRE_ParCSRParaSailsSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRFlexGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_EuclidSolve,
                   NALU_HYPRE_EuclidSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRFlexGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ILUSolve,
                   NALU_HYPRE_ILUSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRFlexGMRESSetPrecond(
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
 * NALU_HYPRE_ParCSRFlexGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmresgetprecond, NALU_HYPRE_PARCSRFLEXGMRESGETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *precond_solver_ptr,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESGetPrecond(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetlogging, NALU_HYPRE_PARCSRFLEXGMRESSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetprintlevel, NALU_HYPRE_PARCSRFLEXGMRESSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmresgetnumiteratio, NALU_HYPRE_PARCSRFLEXGMRESGETNUMITERATIO)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmresgetfinalrelati, NALU_HYPRE_PARCSRFLEXGMRESGETFINALRELATI)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
