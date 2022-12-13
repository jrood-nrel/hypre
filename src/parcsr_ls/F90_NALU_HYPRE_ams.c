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

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amscreate, NALU_HYPRE_AMSCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSCreate(
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsdestroy, NALU_HYPRE_AMSDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetup, NALU_HYPRE_AMSSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssolve, NALU_HYPRE_AMSSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetDimension
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetdimension, NALU_HYPRE_AMSSETDIMENSION)
( hypre_F90_Obj *solver,
  hypre_F90_Int *dim,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetDimension(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (dim) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetdiscretegradient, NALU_HYPRE_AMSSETDISCRETEGRADIENT)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *G,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetDiscreteGradient(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, G) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetcoordinatevectors, NALU_HYPRE_AMSSETCOORDINATEVECTORS)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *x,
  hypre_F90_Obj *y,
  hypre_F90_Obj *z,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetCoordinateVectors(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, y),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, z) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetedgeconstantvectors, NALU_HYPRE_AMSSETEDGECONSTANTVECTORS)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *Gx,
  hypre_F90_Obj *Gy,
  hypre_F90_Obj *Gz,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetEdgeConstantVectors(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, Gx),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, Gy),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, Gz) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetAlphaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetalphapoissonmatrix, NALU_HYPRE_AMSSETALPHAPOISSONMATRIX)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A_alpha,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetAlphaPoissonMatrix(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A_alpha) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetBetaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetbetapoissonmatrix, NALU_HYPRE_AMSSETBETAPOISSONMATRIX)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A_beta,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetBetaPoissonMatrix(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A_beta) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetmaxiter, NALU_HYPRE_AMSSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *maxiter,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (maxiter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssettol, NALU_HYPRE_AMSSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetcycletype, NALU_HYPRE_AMSSETCYCLETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cycle_type,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetCycleType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (cycle_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetprintlevel, NALU_HYPRE_AMSSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetsmoothingoptions, NALU_HYPRE_AMSSETSMOOTHINGOPTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *relax_times,
  hypre_F90_Real *relax_weight,
  hypre_F90_Real *omega,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetSmoothingOptions(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_type),
                hypre_F90_PassInt (relax_times),
                hypre_F90_PassReal (relax_weight),
                hypre_F90_PassReal (omega) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetalphaamgoptions, NALU_HYPRE_AMSSETALPHAAMGOPTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *alpha_coarsen_type,
  hypre_F90_Int *alpha_agg_levels,
  hypre_F90_Int *alpha_relax_type,
  hypre_F90_Real *alpha_strength_threshold,
  hypre_F90_Int *alpha_interp_type,
  hypre_F90_Int *alpha_Pmax,
  hypre_F90_Int *ierr)

{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetAlphaAMGOptions(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (alpha_coarsen_type),
                hypre_F90_PassInt (alpha_agg_levels),
                hypre_F90_PassInt (alpha_relax_type),
                hypre_F90_PassReal (alpha_strength_threshold),
                hypre_F90_PassInt (alpha_interp_type),
                hypre_F90_PassInt (alpha_Pmax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetBetaAMGOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetbetaamgoptions, NALU_HYPRE_AMSSETBETAAMGOPTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *beta_coarsen_type,
  hypre_F90_Int *beta_agg_levels,
  hypre_F90_Int *beta_relax_type,
  hypre_F90_Real *beta_strength_threshold,
  hypre_F90_Int *beta_interp_type,
  hypre_F90_Int *beta_Pmax,
  hypre_F90_Int *ierr)

{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSSetBetaAMGOptions(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (beta_coarsen_type),
                hypre_F90_PassInt (beta_agg_levels),
                hypre_F90_PassInt (beta_relax_type),
                hypre_F90_PassReal (beta_strength_threshold),
                hypre_F90_PassInt (beta_interp_type),
                hypre_F90_PassInt (beta_Pmax) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsgetnumiterations, NALU_HYPRE_AMSGETNUMITERATIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsgetfinalrelativeresidualnorm, NALU_HYPRE_AMSGETFINALRELATIVERESIDUALNORM)
( hypre_F90_Obj *solver,
  hypre_F90_Real *rel_resid_norm,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (rel_resid_norm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSConstructDiscreteGradient
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsconstructdiscretegradient, NALU_HYPRE_AMSCONSTRUCTDISCRETEGRADIENT)
( hypre_F90_Obj *A,
  hypre_F90_Obj *x_coord,
  hypre_F90_BigIntArray *edge_vertex,
  hypre_F90_Int *edge_orientation,
  hypre_F90_Obj *G,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_AMSConstructDiscreteGradient(
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x_coord),
                hypre_F90_PassBigIntArray (edge_vertex),
                hypre_F90_PassInt (edge_orientation),
                hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, G) ) );
}

#ifdef __cplusplus
}
#endif
