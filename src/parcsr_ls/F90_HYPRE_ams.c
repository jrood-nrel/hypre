/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_AMS Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amscreate, NALU_HYPRE_AMSCREATE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSCreate(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amsdestroy, NALU_HYPRE_AMSDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetup, NALU_HYPRE_AMSSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssolve, NALU_HYPRE_AMSSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetDimension
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetdimension, NALU_HYPRE_AMSSETDIMENSION)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *dim,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetDimension(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (dim) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetdiscretegradient, NALU_HYPRE_AMSSETDISCRETEGRADIENT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *G,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetDiscreteGradient(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, G) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetcoordinatevectors, NALU_HYPRE_AMSSETCOORDINATEVECTORS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Obj *z,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetCoordinateVectors(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, y),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, z) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetedgeconstantvectors, NALU_HYPRE_AMSSETEDGECONSTANTVECTORS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *Gx,
  nalu_hypre_F90_Obj *Gy,
  nalu_hypre_F90_Obj *Gz,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetEdgeConstantVectors(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, Gx),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, Gy),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, Gz) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetAlphaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetalphapoissonmatrix, NALU_HYPRE_AMSSETALPHAPOISSONMATRIX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A_alpha,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetAlphaPoissonMatrix(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A_alpha) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetBetaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetbetapoissonmatrix, NALU_HYPRE_AMSSETBETAPOISSONMATRIX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A_beta,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetBetaPoissonMatrix(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A_beta) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetmaxiter, NALU_HYPRE_AMSSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *maxiter,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (maxiter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssettol, NALU_HYPRE_AMSSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetCycleType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetcycletype, NALU_HYPRE_AMSSETCYCLETYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *cycle_type,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetCycleType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (cycle_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetprintlevel, NALU_HYPRE_AMSSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetsmoothingoptions, NALU_HYPRE_AMSSETSMOOTHINGOPTIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_type,
  nalu_hypre_F90_Int *relax_times,
  nalu_hypre_F90_Real *relax_weight,
  nalu_hypre_F90_Real *omega,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetSmoothingOptions(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (relax_type),
                nalu_hypre_F90_PassInt (relax_times),
                nalu_hypre_F90_PassReal (relax_weight),
                nalu_hypre_F90_PassReal (omega) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetalphaamgoptions, NALU_HYPRE_AMSSETALPHAAMGOPTIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *alpha_coarsen_type,
  nalu_hypre_F90_Int *alpha_agg_levels,
  nalu_hypre_F90_Int *alpha_relax_type,
  nalu_hypre_F90_Real *alpha_strength_threshold,
  nalu_hypre_F90_Int *alpha_interp_type,
  nalu_hypre_F90_Int *alpha_Pmax,
  nalu_hypre_F90_Int *ierr)

{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetAlphaAMGOptions(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (alpha_coarsen_type),
                nalu_hypre_F90_PassInt (alpha_agg_levels),
                nalu_hypre_F90_PassInt (alpha_relax_type),
                nalu_hypre_F90_PassReal (alpha_strength_threshold),
                nalu_hypre_F90_PassInt (alpha_interp_type),
                nalu_hypre_F90_PassInt (alpha_Pmax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetBetaAMGOptions
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amssetbetaamgoptions, NALU_HYPRE_AMSSETBETAAMGOPTIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *beta_coarsen_type,
  nalu_hypre_F90_Int *beta_agg_levels,
  nalu_hypre_F90_Int *beta_relax_type,
  nalu_hypre_F90_Real *beta_strength_threshold,
  nalu_hypre_F90_Int *beta_interp_type,
  nalu_hypre_F90_Int *beta_Pmax,
  nalu_hypre_F90_Int *ierr)

{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSSetBetaAMGOptions(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (beta_coarsen_type),
                nalu_hypre_F90_PassInt (beta_agg_levels),
                nalu_hypre_F90_PassInt (beta_relax_type),
                nalu_hypre_F90_PassReal (beta_strength_threshold),
                nalu_hypre_F90_PassInt (beta_interp_type),
                nalu_hypre_F90_PassInt (beta_Pmax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amsgetnumiterations, NALU_HYPRE_AMSGETNUMITERATIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amsgetfinalrelativeresidualnorm, NALU_HYPRE_AMSGETFINALRELATIVERESIDUALNORM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *rel_resid_norm,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (rel_resid_norm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSConstructDiscreteGradient
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_amsconstructdiscretegradient, NALU_HYPRE_AMSCONSTRUCTDISCRETEGRADIENT)
( nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *x_coord,
  nalu_hypre_F90_BigIntArray *edge_vertex,
  nalu_hypre_F90_Int *edge_orientation,
  nalu_hypre_F90_Obj *G,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_AMSConstructDiscreteGradient(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x_coord),
                nalu_hypre_F90_PassBigIntArray (edge_vertex),
                nalu_hypre_F90_PassInt (edge_orientation),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, G) ) );
}

#ifdef __cplusplus
}
#endif
