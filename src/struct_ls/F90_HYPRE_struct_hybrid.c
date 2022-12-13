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
 * NALU_HYPRE_StructHybridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridcreate, NALU_HYPRE_STRUCTHYBRIDCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybriddestroy, NALU_HYPRE_STRUCTHYBRIDDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridDestroy(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetup, NALU_HYPRE_STRUCTHYBRIDSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetup(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsolve, NALU_HYPRE_STRUCTHYBRIDSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSolve(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettol, NALU_HYPRE_STRUCTHYBRIDSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetconvergenc, NALU_HYPRE_STRUCTHYBRIDSETCONVERGENC)
( hypre_F90_Obj *solver,
  hypre_F90_Real *cf_tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetConvergenceTol(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassReal (cf_tol)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetdscgmaxite, NALU_HYPRE_STRUCTHYBRIDSETDSCGMAXITE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *dscg_max_its,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetDSCGMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (dscg_max_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetpcgmaxiter, NALU_HYPRE_STRUCTHYBRIDSETPCGMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *pcg_max_its,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetPCGMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (pcg_max_its) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetpcgabsolut, NALU_HYPRE_STRUCTHYBRIDSETPCGABSOLUT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *pcg_atolf,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetPCGAbsoluteTolFactor(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassReal (pcg_atolf) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettwonorm, NALU_HYPRE_STRUCTHYBRIDSETTWONORM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *two_norm,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetTwoNorm(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (two_norm)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetstopcrit, NALU_HYPRE_STRUCTHYBRIDSETSTOPCRIT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *stop_crit,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetStopCrit(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (stop_crit)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetrelchange, NALU_HYPRE_STRUCTHYBRIDSETRELCHANGE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rel_change,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetRelChange(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetsolvertype, NALU_HYPRE_STRUCTHYBRIDSETSOLVERTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *solver_type,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetSolverType(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (solver_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetkdim, NALU_HYPRE_STRUCTHYBRIDSETKDIM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *k_dim,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_StructHybridSetKDim(
               hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
               hypre_F90_PassInt (k_dim) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetprecond, NALU_HYPRE_STRUCTHYBRIDSETPRECOND)
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
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_StructHybridSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructSMGSolve,
                   NALU_HYPRE_StructSMGSetup,
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_StructHybridSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructPFMGSolve,
                   NALU_HYPRE_StructPFMGSetup,
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_StructHybridSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructJacobiSolve,
                   NALU_HYPRE_StructJacobiSetup,
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_StructHybridSetPrecond(
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructDiagScale,
                   NALU_HYPRE_StructDiagScaleSetup,
                   hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetlogging, NALU_HYPRE_STRUCTHYBRIDSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (logging)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetprintlevel, NALU_HYPRE_STRUCTHYBRIDSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassInt (print_level)  ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetnumiterati, NALU_HYPRE_STRUCTHYBRIDGETNUMITERATI)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_its,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_its)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetdscgnumite, NALU_HYPRE_STRUCTHYBRIDGETDSCGNUMITE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *dscg_num_its,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridGetDSCGNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (dscg_num_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetpcgnumiter, NALU_HYPRE_STRUCTHYBRIDGETPCGNUMITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *pcg_num_its,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridGetPCGNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (pcg_num_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetfinalrelat, NALU_HYPRE_STRUCTHYBRIDGETFINALRELAT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructHybridGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
