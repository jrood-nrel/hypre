/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRCOGMRES Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmrescreate, NALU_HYPRE_PARCSRCOGMRESCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmresdestroy, NALU_HYPRE_PARCSRCOGMRESDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetup, NALU_HYPRE_PARCSRCOGMRESSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressolve, NALU_HYPRE_PARCSRCOGMRESSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetkdim, NALU_HYPRE_PARCSRCOGMRESSETKDIM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *kdim,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetKDim(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (kdim)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetUnroll
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetunroll, NALU_HYPRE_PARCSRCOGMRESSETUNROLL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *unroll,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetUnroll(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (unroll)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetCGS
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetcgs, NALU_HYPRE_PARCSRCOGMRESSETCGS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *cgs,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetCGS(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (cgs)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressettol, NALU_HYPRE_PARCSRCOGMRESSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetabsolutet, NALU_HYPRE_PARCSRCOGMRESSETABSOLUTET)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetminiter, NALU_HYPRE_PARCSRCOGMRESSETMINITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *min_iter,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetMinIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetmaxiter, NALU_HYPRE_PARCSRCOGMRESSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetprecond, NALU_HYPRE_PARCSRCOGMRESSETPRECOND)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *precond_id,
  nalu_hypre_F90_Obj *precond_solver,
  nalu_hypre_F90_Int *ierr          )
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
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRDiagScale,
                   NALU_HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_BoomerAMGSolve,
                   NALU_HYPRE_BoomerAMGSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRPilutSolve,
                   NALU_HYPRE_ParCSRPilutSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ParCSRParaSailsSolve,
                   NALU_HYPRE_ParCSRParaSailsSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_EuclidSolve,
                   NALU_HYPRE_EuclidSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   NALU_HYPRE_ILUSolve,
                   NALU_HYPRE_ILUSetup,
                   (NALU_HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_ParCSRCOGMRESSetPrecond(
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
 * NALU_HYPRE_ParCSRCOGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmresgetprecond, NALU_HYPRE_PARCSRCOGMRESGETPRECOND)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *precond_solver_ptr,
  nalu_hypre_F90_Int *ierr                )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESGetPrecond(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetlogging, NALU_HYPRE_PARCSRCOGMRESSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmressetprintleve, NALU_HYPRE_PARCSRCOGMRESSETPRINTLEVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmresgetnumiterat, NALU_HYPRE_PARCSRCOGMRESGETNUMITERAT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr            )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrcogmresgetfinalrela, NALU_HYPRE_PARCSRCOGMRESGETFINALRELA)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
