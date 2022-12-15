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
 * NALU_HYPRE_StructHybridCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridcreate, NALU_HYPRE_STRUCTHYBRIDCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybriddestroy, NALU_HYPRE_STRUCTHYBRIDDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetup, NALU_HYPRE_STRUCTHYBRIDSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsolve, NALU_HYPRE_STRUCTHYBRIDSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsettol, NALU_HYPRE_STRUCTHYBRIDSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (tol)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetconvergenc, NALU_HYPRE_STRUCTHYBRIDSETCONVERGENC)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *cf_tol,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetConvergenceTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (cf_tol)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetdscgmaxite, NALU_HYPRE_STRUCTHYBRIDSETDSCGMAXITE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *dscg_max_its,
  nalu_hypre_F90_Int *ierr         )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetDSCGMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (dscg_max_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetpcgmaxiter, NALU_HYPRE_STRUCTHYBRIDSETPCGMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *pcg_max_its,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetPCGMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (pcg_max_its) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetpcgabsolut, NALU_HYPRE_STRUCTHYBRIDSETPCGABSOLUT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *pcg_atolf,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetPCGAbsoluteTolFactor(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassReal (pcg_atolf) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsettwonorm, NALU_HYPRE_STRUCTHYBRIDSETTWONORM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *two_norm,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetTwoNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (two_norm)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetstopcrit, NALU_HYPRE_STRUCTHYBRIDSETSTOPCRIT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *stop_crit,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetStopCrit(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (stop_crit)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetrelchange, NALU_HYPRE_STRUCTHYBRIDSETRELCHANGE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rel_change,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetsolvertype, NALU_HYPRE_STRUCTHYBRIDSETSOLVERTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *solver_type,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetSolverType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (solver_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetKDim
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetkdim, NALU_HYPRE_STRUCTHYBRIDSETKDIM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *k_dim,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_StructHybridSetKDim(
               nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
               nalu_hypre_F90_PassInt (k_dim) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetprecond, NALU_HYPRE_STRUCTHYBRIDSETPRECOND)
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
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructHybridSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructSMGSolve,
                   NALU_HYPRE_StructSMGSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructHybridSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructPFMGSolve,
                   NALU_HYPRE_StructPFMGSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructHybridSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructJacobiSolve,
                   NALU_HYPRE_StructJacobiSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_StructHybridSetPrecond(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                   NALU_HYPRE_StructDiagScale,
                   NALU_HYPRE_StructDiagScaleSetup,
                   nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, precond_solver)) );
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
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetlogging, NALU_HYPRE_STRUCTHYBRIDSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (logging)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridsetprintlevel, NALU_HYPRE_STRUCTHYBRIDSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassInt (print_level)  ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridgetnumiterati, NALU_HYPRE_STRUCTHYBRIDGETNUMITERATI)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_its,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_its)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridgetdscgnumite, NALU_HYPRE_STRUCTHYBRIDGETDSCGNUMITE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *dscg_num_its,
  nalu_hypre_F90_Int *ierr         )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridGetDSCGNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (dscg_num_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridgetpcgnumiter, NALU_HYPRE_STRUCTHYBRIDGETPCGNUMITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *pcg_num_its,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridGetPCGNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassIntRef (pcg_num_its) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structhybridgetfinalrelat, NALU_HYPRE_STRUCTHYBRIDGETFINALRELAT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *norm,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructHybridGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructSolver, solver),
                nalu_hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
