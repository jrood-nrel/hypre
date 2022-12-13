/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_BlockTridiag Fortran interface
 *
 *****************************************************************************/

#include "block_tridiag.h"
#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagcreate, NALU_HYPRE_BLOCKTRIDIAGCREATE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_BlockTridiagCreate(
              hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagdestroy, NALU_HYPRE_BLOCKTRIDIAGDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_BlockTridiagDestroy(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetup, NALU_HYPRE_BLOCKTRIDIAGSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_BlockTridiagSetup(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
              hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
              hypre_F90_PassObj (NALU_HYPRE_ParVector, x));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsolve, NALU_HYPRE_BLOCKTRIDIAGSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_BlockTridiagSolve(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
              hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
              hypre_F90_PassObj (NALU_HYPRE_ParVector, x));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetindexset, NALU_HYPRE_BLOCKTRIDIAGSETINDEXSET)
(hypre_F90_Obj *solver,
 hypre_F90_Int *n,
 hypre_F90_IntArray *inds,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_BlockTridiagSetIndexSet(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (n),
              hypre_F90_PassIntArray (inds));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgstrengt, NALU_HYPRE_BLOCKTRIDIAGSETAMGSTRENGT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *thresh,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassReal (thresh));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgnumswee, NALU_HYPRE_BLOCKTRIDIAGSETAMGNUMSWEE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_sweeps,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_BlockTridiagSetAMGNumSweeps(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (num_sweeps));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgrelaxty, NALU_HYPRE_BLOCKTRIDIAGSETAMGRELAXTY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *relax_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_BlockTridiagSetAMGRelaxType(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (relax_type));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetprintlevel, NALU_HYPRE_BLOCKTRIDIAGSETPRINTLEVEL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *print_level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_BlockTridiagSetPrintLevel(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (print_level));
}

#ifdef __cplusplus
}
#endif
