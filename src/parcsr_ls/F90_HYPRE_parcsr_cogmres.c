/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRCOGMRES Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmrescreate, NALU_HYPRE_PARCSRCOGMRESCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresdestroy, NALU_HYPRE_PARCSRCOGMRESDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetup, NALU_HYPRE_PARCSRCOGMRESSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressolve, NALU_HYPRE_PARCSRCOGMRESSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetkdim, NALU_HYPRE_PARCSRCOGMRESSETKDIM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *kdim,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetKDim(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (kdim)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetUnroll
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetunroll, NALU_HYPRE_PARCSRCOGMRESSETUNROLL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *unroll,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetUnroll(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (unroll)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetCGS
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetcgs, NALU_HYPRE_PARCSRCOGMRESSETCGS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cgs,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetCGS(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (cgs)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressettol, NALU_HYPRE_PARCSRCOGMRESSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetabsolutet, NALU_HYPRE_PARCSRCOGMRESSETABSOLUTET)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetminiter, NALU_HYPRE_PARCSRCOGMRESSETMINITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetMinIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetmaxiter, NALU_HYPRE_PARCSRCOGMRESSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetprecond, NALU_HYPRE_PARCSRCOGMRESSETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Int *precond_id,
  hypre_F90_Obj *precond_solver,
  hypre_F90_Int *ierr          )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - no preconditioner
    * 1 - set up a ds preconditioner
    * 2 - set up an amg preconditioner
    * 3 - set up a pilut preconditioner
    * 4 - set up a parasails preconditioner
    * 5 - set up a Euclid preconditioner
    * 6 - set up a ILU preconditioner
    * 7 - set up a MGR preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRDiagScale,
                   NALU_HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_BoomerAMGSolve,
                   NALU_HYPRE_BoomerAMGSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRPilutSolve,
                   NALU_HYPRE_ParCSRPilutSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRParaSailsSolve,
                   NALU_HYPRE_ParCSRParaSailsSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_EuclidSolve,
                   NALU_HYPRE_EuclidSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ILUSolve,
                   NALU_HYPRE_ILUSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
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
 * NALU_HYPRE_ParCSRCOGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresgetprecond, NALU_HYPRE_PARCSRCOGMRESGETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *precond_solver_ptr,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESGetPrecond(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetlogging, NALU_HYPRE_PARCSRCOGMRESSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetprintleve, NALU_HYPRE_PARCSRCOGMRESSETPRINTLEVE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresgetnumiterat, NALU_HYPRE_PARCSRCOGMRESGETNUMITERAT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresgetfinalrela, NALU_HYPRE_PARCSRCOGMRESGETFINALRELA)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
