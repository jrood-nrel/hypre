/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ILU Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilucreate, NALU_HYPRE_ILUCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUCreate(
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_iludestroy, NALU_HYPRE_ILUDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetup, NALU_HYPRE_ILUSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusolve, NALU_HYPRE_ILUSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetprintlevel, NALU_HYPRE_ILUSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetlogging, NALU_HYPRE_ILUSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetmaxiter, NALU_HYPRE_ILUSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusettol, NALU_HYPRE_ILUSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetDropThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetdropthreshold, NALU_HYPRE_ILUSETDROPTHRESHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *threshold,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetDropThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (threshold)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetDropThresholdArray
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetdropthresholdarray, NALU_HYPRE_ILUSETDROPTHRESHOLDARRAY)
( hypre_F90_Obj *solver,
  hypre_F90_RealArray *threshold,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetDropThresholdArray(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealArray (threshold)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetNSHDropThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetnshdropthreshold, NALU_HYPRE_ILUSETNSHDROPTHRESHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *threshold,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetNSHDropThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (threshold)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetSchurMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetschurmaxiter, NALU_HYPRE_ILUSETSCHURMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ss_max_iter,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (ss_max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetMaxNnzPerRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetmaxnnzperrow, NALU_HYPRE_ILUSETMAXNNZPERROW)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nzmax,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetMaxNnzPerRow(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (nzmax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLevelOfFill
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetleveloffill, NALU_HYPRE_ILUSETLEVELOFFILL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *lfil,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetMaxNnzPerRow(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (lfil) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusettype, NALU_HYPRE_ILUSETTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ilu_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (ilu_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLocalReordering
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetlocalreordering, NALU_HYPRE_ILUSETLOCALREORDERING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ordering_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUSetType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (ordering_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilugetnumiterations, NALU_HYPRE_ILUGETNUMITERATIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUGetFinalRelResNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilugetfinalrelresnorm, NALU_HYPRE_ILUGETFINALRELRESNORM)
( hypre_F90_Obj *solver,
  hypre_F90_Real *res_norm,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ILUGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (res_norm) ) );
}


#ifdef __cplusplus
}
#endif
