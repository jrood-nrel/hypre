/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_BlockTridiag Fortran interface
 *
 *****************************************************************************/

#include "block_tridiag.h"
#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_blocktridiagcreate, NALU_HYPRE_BLOCKTRIDIAGCREATE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_BlockTridiagCreate(
              nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_blocktridiagdestroy, NALU_HYPRE_BLOCKTRIDIAGDESTROY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_BlockTridiagDestroy(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_blocktridiagsetup, NALU_HYPRE_BLOCKTRIDIAGSETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_BlockTridiagSetup(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_blocktridiagsolve, NALU_HYPRE_BLOCKTRIDIAGSOLVE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_BlockTridiagSolve(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_blocktridiagsetindexset, NALU_HYPRE_BLOCKTRIDIAGSETINDEXSET)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *n,
 nalu_hypre_F90_IntArray *inds,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_BlockTridiagSetIndexSet(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (n),
              nalu_hypre_F90_PassIntArray (inds));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_blocktridiagsetamgstrengt, NALU_HYPRE_BLOCKTRIDIAGSETAMGSTRENGT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *thresh,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassReal (thresh));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_blocktridiagsetamgnumswee, NALU_HYPRE_BLOCKTRIDIAGSETAMGNUMSWEE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_sweeps,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_BlockTridiagSetAMGNumSweeps(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (num_sweeps));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_blocktridiagsetamgrelaxty, NALU_HYPRE_BLOCKTRIDIAGSETAMGRELAXTY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *relax_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_BlockTridiagSetAMGRelaxType(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (relax_type));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_blocktridiagsetprintlevel, NALU_HYPRE_BLOCKTRIDIAGSETPRINTLEVEL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *print_level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_BlockTridiagSetPrintLevel(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (print_level));
}

#ifdef __cplusplus
}
#endif
