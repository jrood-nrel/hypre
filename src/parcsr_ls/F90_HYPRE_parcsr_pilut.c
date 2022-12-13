/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRPilut Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutcreate, NALU_HYPRE_PARCSRPILUTCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPilutCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutdestroy, NALU_HYPRE_PARCSRPILUTDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPilutDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutsetup, NALU_HYPRE_PARCSRPILUTSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPilutSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutsolve, NALU_HYPRE_PARCSRPILUTSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPilutSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutsetmaxiter, NALU_HYPRE_PARCSRPILUTSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPilutSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSetDropToleran
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutsetdroptoleran, NALU_HYPRE_PARCSRPILUTSETDROPTOLERAN)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPilutSetDropTolerance(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSetFacRowSize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutsetfacrowsize, NALU_HYPRE_PARCSRPILUTSETFACROWSIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *size,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRPilutSetFactorRowSize(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (size)    ) );
}

#ifdef __cplusplus
}
#endif
