/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ILU Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilucreate, NALU_HYPRE_ILUCREATE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUCreate(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_iludestroy, NALU_HYPRE_ILUDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetup, NALU_HYPRE_ILUSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusolve, NALU_HYPRE_ILUSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetprintlevel, NALU_HYPRE_ILUSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetlogging, NALU_HYPRE_ILUSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetmaxiter, NALU_HYPRE_ILUSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusettol, NALU_HYPRE_ILUSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetDropThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetdropthreshold, NALU_HYPRE_ILUSETDROPTHRESHOLD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *threshold,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetDropThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (threshold)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetDropThresholdArray
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetdropthresholdarray, NALU_HYPRE_ILUSETDROPTHRESHOLDARRAY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_RealArray *threshold,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetDropThresholdArray(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealArray (threshold)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetNSHDropThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetnshdropthreshold, NALU_HYPRE_ILUSETNSHDROPTHRESHOLD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *threshold,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetNSHDropThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (threshold)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetSchurMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetschurmaxiter, NALU_HYPRE_ILUSETSCHURMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ss_max_iter,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (ss_max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetMaxNnzPerRow
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetmaxnnzperrow, NALU_HYPRE_ILUSETMAXNNZPERROW)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *nzmax,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetMaxNnzPerRow(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (nzmax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLevelOfFill
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetleveloffill, NALU_HYPRE_ILUSETLEVELOFFILL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *lfil,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetMaxNnzPerRow(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (lfil) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusettype, NALU_HYPRE_ILUSETTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ilu_type,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (ilu_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLocalReordering
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilusetlocalreordering, NALU_HYPRE_ILUSETLOCALREORDERING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ordering_type,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUSetType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (ordering_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilugetnumiterations, NALU_HYPRE_ILUGETNUMITERATIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUGetFinalRelResNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ilugetfinalrelresnorm, NALU_HYPRE_ILUGETFINALRELRESNORM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *res_norm,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ILUGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (res_norm) ) );
}


#ifdef __cplusplus
}
#endif
