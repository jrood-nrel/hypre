/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParAMG Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgcreate, NALU_HYPRE_BOOMERAMGCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGCreate(
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgdestroy, NALU_HYPRE_BOOMERAMGDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetup, NALU_HYPRE_BOOMERAMGSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsolve, NALU_HYPRE_BOOMERAMGSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSolveT
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsolvet, NALU_HYPRE_BOOMERAMGSOLVET)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSolveT(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRestriction
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrestriction, NALU_HYPRE_BOOMERAMGSETRESTRICTION)
( hypre_F90_Obj *solver,
  hypre_F90_Int *restr_par,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRestriction(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (restr_par) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxLevels, NALU_HYPRE_BoomerAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxlevels, NALU_HYPRE_BOOMERAMGSETMAXLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_levels,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxLevels(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_levels) ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetmaxlevels, NALU_HYPRE_BOOMERAMGGETMAXLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_levels,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMaxLevels(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (max_levels) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxCoarseSize, NALU_HYPRE_BoomerAMGGetMaxCoarseSize
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetmaxcoarsesize, NALU_HYPRE_BOOMERAMGSETMAXCOARSESIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_coarse_size,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxCoarseSize(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_coarse_size) ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetmaxcoarsesize, NALU_HYPRE_BOOMERAMGGETMAXCOARSESIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_coarse_size,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMaxCoarseSize(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (max_coarse_size) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMinCoarseSize, NALU_HYPRE_BoomerAMGGetMinCoarseSize
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetmincoarsesize, NALU_HYPRE_BOOMERAMGSETMINCOARSESIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_coarse_size,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMinCoarseSize(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (min_coarse_size) ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetmincoarsesize, NALU_HYPRE_BOOMERAMGGETMINCOARSESIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_coarse_size,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMinCoarseSize(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (min_coarse_size) ) );
}




/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetStrongThreshold, NALU_HYPRE_BoomerAMGGetStrongThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetstrongthrshld, NALU_HYPRE_BOOMERAMGSETSTRONGTHRSHLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *strong_threshold,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetStrongThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (strong_threshold) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetstrongthrshld, NALU_HYPRE_BOOMERAMGGETSTRONGTHRSHLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *strong_threshold,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetStrongThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (strong_threshold) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxRowSum, NALU_HYPRE_BoomerAMGGetMaxRowSum
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxrowsum, NALU_HYPRE_BOOMERAMGSETMAXROWSUM)
( hypre_F90_Obj *solver,
  hypre_F90_Real *max_row_sum,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxRowSum(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (max_row_sum) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmaxrowsum, NALU_HYPRE_BOOMERAMGGETMAXROWSUM)
( hypre_F90_Obj *solver,
  hypre_F90_Real *max_row_sum,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMaxRowSum(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (max_row_sum) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetTruncFactor, NALU_HYPRE_BoomerAMGGetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsettruncfactor, NALU_HYPRE_BOOMERAMGSETTRUNCFACTOR)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetTruncFactor(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (trunc_factor) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggettruncfactor, NALU_HYPRE_BOOMERAMGGETTRUNCFACTOR)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetTruncFactor(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetpmaxelmts, NALU_HYPRE_BOOMERAMGSETPMAXELMTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *p_max_elmts,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetPMaxElmts(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (p_max_elmts) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetpmaxelmts, NALU_HYPRE_BOOMERAMGGETPMAXELMTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *p_max_elmts,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetPMaxElmts(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (p_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold, NALU_HYPRE_BoomerAMGGetJacobiTruncThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetjacobitrunc, NALU_HYPRE_BOOMERAMGSETJACOBITRUNC)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (trunc_factor) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetjacobitrunc, NALU_HYPRE_BOOMERAMGGETJACOBITRUNC)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetJacobiTruncThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPostInterpType, NALU_HYPRE_BoomerAMGGetPostInterpType
 *  If >0, specifies something to do to improve a computed interpolation matrix.
 * defaults to 0, for nothing.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetpostinterp, NALU_HYPRE_BOOMERAMGSETPOSTINTERP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *type,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetPostInterpType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (type) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterptype, NALU_HYPRE_BOOMERAMGSETINTERPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *interp_type,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (interp_type) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSepWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetsepweight, NALU_HYPRE_BOOMERAMGSETSEPWEIGHT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *sep_weight,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSepWeight(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (sep_weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetminiter, NALU_HYPRE_BOOMERAMGSETMINITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMinIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxIter, NALU_HYPRE_BoomerAMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxiter, NALU_HYPRE_BOOMERAMGSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmaxiter, NALU_HYPRE_BOOMERAMGGETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCoarsenType, NALU_HYPRE_BoomerAMGGetCoarsenType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcoarsentype, NALU_HYPRE_BOOMERAMGSETCOARSENTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *coarsen_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCoarsenType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (coarsen_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcoarsentype, NALU_HYPRE_BOOMERAMGGETCOARSENTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *coarsen_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCoarsenType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (coarsen_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMeasureType, NALU_HYPRE_BoomerAMGGetMeasureType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmeasuretype, NALU_HYPRE_BOOMERAMGSETMEASURETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *measure_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMeasureType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (measure_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmeasuretype, NALU_HYPRE_BOOMERAMGGETMEASURETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *measure_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMeasureType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (measure_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOldDefault
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetolddefault, NALU_HYPRE_BOOMERAMGSETOLDDEFAULT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetOldDefault(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSetupType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsetuptype, NALU_HYPRE_BOOMERAMGSETSETUPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *setup_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSetupType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (setup_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCycleType, NALU_HYPRE_BoomerAMGGetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcycletype, NALU_HYPRE_BOOMERAMGSETCYCLETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cycle_type,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCycleType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (cycle_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcycletype, NALU_HYPRE_BOOMERAMGGETCYCLETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cycle_type,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCycleType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (cycle_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetTol, NALU_HYPRE_BoomerAMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsettol, NALU_HYPRE_BOOMERAMGSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

void
hypre_F90_IFACE(hypre_boomeramggettol, NALU_HYPRE_BOOMERAMGGETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumSweeps
 * DEPRECATED.  Use SetNumSweeps and SetCycleNumSweeps instead.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumgridsweeps, NALU_HYPRE_BOOMERAMGSETNUMGRIDSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_grid_sweeps,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumGridSweeps(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_grid_sweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumsweeps, NALU_HYPRE_BOOMERAMGSETNUMSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_sweeps,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumSweeps(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (num_sweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCycleNumSweeps, NALU_HYPRE_BoomerAMGGetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcyclenumsweeps, NALU_HYPRE_BOOMERAMGSETCYCLENUMSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_sweeps,
  hypre_F90_Int *k,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCycleNumSweeps(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (num_sweeps),
                hypre_F90_PassInt (k) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcyclenumsweeps, NALU_HYPRE_BOOMERAMGGETCYCLENUMSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_sweeps,
  hypre_F90_Int *k,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCycleNumSweeps(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_sweeps),
                hypre_F90_PassInt (k) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGInitGridRelaxation
 *
 * RDF: This is probably not a very useful Fortran routine because you can't do
 * anything with the pointers to arrays that are allocated.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramginitgridrelaxatn, NALU_HYPRE_BOOMERAMGINITGRIDRELAXATN)
( hypre_F90_Obj *num_grid_sweeps,
  hypre_F90_Obj *grid_relax_type,
  hypre_F90_Obj *grid_relax_points,
  hypre_F90_Int *coarsen_type,
  hypre_F90_Obj *relax_weights,
  hypre_F90_Int *max_levels,
  hypre_F90_Int *ierr               )
{
   *num_grid_sweeps   = (hypre_F90_Obj) hypre_CTAlloc(NALU_HYPRE_Int*,  1, NALU_HYPRE_MEMORY_HOST);
   *grid_relax_type   = (hypre_F90_Obj) hypre_CTAlloc(NALU_HYPRE_Int*,  1, NALU_HYPRE_MEMORY_HOST);
   *grid_relax_points = (hypre_F90_Obj) hypre_CTAlloc(NALU_HYPRE_Int**,  1, NALU_HYPRE_MEMORY_HOST);
   *relax_weights     = (hypre_F90_Obj) hypre_CTAlloc(NALU_HYPRE_Real*,  1, NALU_HYPRE_MEMORY_HOST);

   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGInitGridRelaxation(
                (NALU_HYPRE_Int **)    *num_grid_sweeps,
                (NALU_HYPRE_Int **)    *grid_relax_type,
                (NALU_HYPRE_Int ***)   *grid_relax_points,
                hypre_F90_PassInt (coarsen_type),
                (NALU_HYPRE_Real **) *relax_weights,
                hypre_F90_PassInt (max_levels)         ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGFinalizeGridRelaxation
 *
 * RDF: This is probably not a very useful Fortran routine because you can't do
 * anything with the pointers to arrays.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgfingridrelaxatn, NALU_HYPRE_BOOMERAMGFINGRIDRELAXATN)
( hypre_F90_Obj *num_grid_sweeps,
  hypre_F90_Obj *grid_relax_type,
  hypre_F90_Obj *grid_relax_points,
  hypre_F90_Obj *relax_weights,
  hypre_F90_Int *ierr               )
{
   char *ptr_num_grid_sweeps   = (char *) *num_grid_sweeps;
   char *ptr_grid_relax_type   = (char *) *grid_relax_type;
   char *ptr_grid_relax_points = (char *) *grid_relax_points;
   char *ptr_relax_weights     = (char *) *relax_weights;

   hypre_TFree(ptr_num_grid_sweeps, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(ptr_grid_relax_type, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(ptr_grid_relax_points, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(ptr_relax_weights, NALU_HYPRE_MEMORY_HOST);

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGridRelaxType
 * DEPRECATED.  Use SetRelaxType and SetCycleRelaxType instead.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgridrelaxtype, NALU_HYPRE_BOOMERAMGSETGRIDRELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_IntArray *grid_relax_type,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetGridRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntArray (grid_relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxtype, NALU_HYPRE_BOOMERAMGSETRELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCycleRelaxType, NALU_HYPRE_BoomerAMGGetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcyclerelaxtype, NALU_HYPRE_BOOMERAMGSETCYCLERELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *k,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCycleRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_type),
                hypre_F90_PassInt (k) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcyclerelaxtype, NALU_HYPRE_BOOMERAMGGETCYCLERELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *k,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCycleRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (relax_type),
                hypre_F90_PassInt (k)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxOrder
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxorder, NALU_HYPRE_BOOMERAMGSETRELAXORDER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_order,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRelaxOrder(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_order) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGridRelaxPoints
 * DEPRECATED.  There is no alternative function.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgridrelaxpnts, NALU_HYPRE_BOOMERAMGSETGRIDRELAXPNTS)
( hypre_F90_Obj *solver,
  NALU_HYPRE_Int      **grid_relax_points,
  hypre_F90_Int *ierr               )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetGridRelaxPoints(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                (NALU_HYPRE_Int **)        grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxWeight
 * DEPRECATED.  Use SetRelaxWt and SetLevelRelaxWt instead.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxweight, NALU_HYPRE_BOOMERAMGSETRELAXWEIGHT)
( hypre_F90_Obj *solver,
  hypre_F90_IntArray *relax_weights,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRelaxWeight(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealArray (relax_weights) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxwt, NALU_HYPRE_BOOMERAMGSETRELAXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *relax_weight,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRelaxWt(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (relax_weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevelrelaxwt, NALU_HYPRE_BOOMERAMGSETLEVELRELAXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *relax_weight,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLevelRelaxWt(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (relax_weight),
                hypre_F90_PassInt (level) ) );
}




/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetouterwt, NALU_HYPRE_BOOMERAMGSETOUTERWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *outer_wt,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetOuterWt(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (outer_wt) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevelouterwt, NALU_HYPRE_BOOMERAMGSETLEVELOUTERWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *outer_wt,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLevelOuterWt(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (outer_wt),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothType, NALU_HYPRE_BoomerAMGGetSmoothType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothtype, NALU_HYPRE_BOOMERAMGSETSMOOTHTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSmoothType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (smooth_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothtype, NALU_HYPRE_BOOMERAMGGETSMOOTHTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSmoothType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (smooth_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothNumLvls, NALU_HYPRE_BoomerAMGGetSmoothNumLvls
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothnumlvls, NALU_HYPRE_BOOMERAMGSETSMOOTHNUMLVLS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_num_levels,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSmoothNumLevels(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (smooth_num_levels) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothnumlvls, NALU_HYPRE_BOOMERAMGGETSMOOTHNUMLVLS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_num_levels,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSmoothNumLevels(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (smooth_num_levels) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothNumSwps, NALU_HYPRE_BoomerAMGGetSmoothNumSwps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothnumswps, NALU_HYPRE_BOOMERAMGSETSMOOTHNUMSWPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_num_sweeps,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (smooth_num_sweeps) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothnumswps, NALU_HYPRE_BOOMERAMGGETSMOOTHNUMSWPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_num_sweeps,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSmoothNumSweeps(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (smooth_num_sweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLogging, NALU_HYPRE_BoomerAMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlogging, NALU_HYPRE_BOOMERAMGSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetlogging, NALU_HYPRE_BOOMERAMGGETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPrintLevel, NALU_HYPRE_BoomerAMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetprintlevel, NALU_HYPRE_BOOMERAMGSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetprintlevel, NALU_HYPRE_BOOMERAMGGETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPrintFileName
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetprintfilename, NALU_HYPRE_BOOMERAMGSETPRINTFILENAME)
( hypre_F90_Obj *solver,
  char     *print_file_name,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetPrintFileName(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                (char *)        print_file_name ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDebugFlag, NALU_HYPRE_BoomerAMGGetDebugFlag
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdebugflag, NALU_HYPRE_BOOMERAMGSETDEBUGFLAG)
( hypre_F90_Obj *solver,
  hypre_F90_Int *debug_flag,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDebugFlag(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (debug_flag) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetdebugflag, NALU_HYPRE_BOOMERAMGGETDEBUGFLAG)
( hypre_F90_Obj *solver,
  hypre_F90_Int *debug_flag,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetDebugFlag(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (debug_flag) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetnumiterations, NALU_HYPRE_BOOMERAMGGETNUMITERATIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetCumNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetcumnumiterati, NALU_HYPRE_BOOMERAMGGETCUMNUMITERATI)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cum_num_iterations,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCumNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (cum_num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetresidual, NALU_HYPRE_BOOMERAMGGETRESIDUAL)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *residual,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetResidual(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObjRef (NALU_HYPRE_ParVector, residual)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetFinalRelativeResNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetfinalreltvres, NALU_HYPRE_BOOMERAMGGETFINALRELTVRES)
( hypre_F90_Obj *solver,
  hypre_F90_Real *rel_resid_norm,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (rel_resid_norm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetVariant, NALU_HYPRE_BoomerAMGGetVariant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetvariant, NALU_HYPRE_BOOMERAMGSETVARIANT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *variant,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetVariant(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (variant) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetvariant, NALU_HYPRE_BOOMERAMGGETVARIANT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *variant,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetVariant(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (variant) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOverlap, NALU_HYPRE_BoomerAMGGetOverlap
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetoverlap, NALU_HYPRE_BOOMERAMGSETOVERLAP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *overlap,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetOverlap(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (overlap) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetoverlap, NALU_HYPRE_BOOMERAMGGETOVERLAP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *overlap,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetOverlap(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (overlap) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDomainType, NALU_HYPRE_BoomerAMGGetDomainType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdomaintype, NALU_HYPRE_BOOMERAMGSETDOMAINTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *domain_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDomainType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (domain_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetdomaintype, NALU_HYPRE_BOOMERAMGGETDOMAINTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *domain_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetDomainType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (domain_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramgsetschwarznonsym, NALU_HYPRE_BOOMERAMGSETSCHWARZNONSYM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *schwarz_non_symm,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (schwarz_non_symm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSchwarzRlxWt, NALU_HYPRE_BoomerAMGGetSchwarzRlxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetschwarzrlxwt, NALU_HYPRE_BOOMERAMGSETSCHWARZRLXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *schwarz_rlx_weight,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (schwarz_rlx_weight)) );
}

void
hypre_F90_IFACE(hypre_boomeramggetschwarzrlxwt, NALU_HYPRE_BOOMERAMGGETSCHWARZRLXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *schwarz_rlx_weight,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSchwarzRlxWeight(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (schwarz_rlx_weight)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSym
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsym, NALU_HYPRE_BOOMERAMGSETSYM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *sym,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSym(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (sym) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevel, NALU_HYPRE_BOOMERAMGSETLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetthreshold, NALU_HYPRE_BOOMERAMGSETTHRESHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *threshold,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (threshold)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetFilter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetfilter, NALU_HYPRE_BOOMERAMGSETFILTER)
( hypre_F90_Obj *solver,
  hypre_F90_Real *filter,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetFilter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (filter)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDropTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdroptol, NALU_HYPRE_BOOMERAMGSETDROPTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *drop_tol,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDropTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (drop_tol)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxNzPerRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxnzperrow, NALU_HYPRE_BOOMERAMGSETMAXNZPERROW)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_nz_per_row,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxNzPerRow(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_nz_per_row) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuBJ
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteubj, NALU_HYPRE_BOOMERAMGSETEUBJ)
( hypre_F90_Obj *solver,
  hypre_F90_Int *eu_bj,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetEuBJ(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (eu_bj) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteulevel, NALU_HYPRE_BOOMERAMGSETEULEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *eu_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetEuLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (eu_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuSparseA
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteusparsea, NALU_HYPRE_BOOMERAMGSETEUSPARSEA)
( hypre_F90_Obj *solver,
  hypre_F90_Real *eu_sparse_a,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetEuSparseA(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (eu_sparse_a)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuclidFile
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteuclidfile, NALU_HYPRE_BOOMERAMGSETEUCLIDFILE)
( hypre_F90_Obj *solver,
  char     *euclidfile,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetEuclidFile(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                (char *)        euclidfile ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumFunctions, NALU_HYPRE_BoomerAMGGetNumFunctions
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnumfunctions, NALU_HYPRE_BOOMERAMGSETNUMFUNCTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_functions,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumFunctions(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (num_functions) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetnumfunctions, NALU_HYPRE_BOOMERAMGGETNUMFUNCTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_functions,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetNumFunctions(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_functions) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNodal
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnodal, NALU_HYPRE_BOOMERAMGSETNODAL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nodal,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNodal(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (nodal) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNodalDiag
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnodaldiag, NALU_HYPRE_BOOMERAMGSETNODALDIAG)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nodal_diag,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNodalDiag(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (nodal_diag) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDofFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdoffunc, NALU_HYPRE_BOOMERAMGSETDOFFUNC)
( hypre_F90_Obj *solver,
  hypre_F90_IntArray *dof_func,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDofFunc(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntArray (dof_func) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumPaths
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumpaths, NALU_HYPRE_BOOMERAMGSETNUMPATHS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_paths,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumPaths(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (num_paths) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggNumLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggnumlevels, NALU_HYPRE_BOOMERAMGSETAGGNUMLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *agg_num_levels,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggNumLevels(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (agg_num_levels) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggInterpType
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetagginterptype, NALU_HYPRE_BOOMERAMGSETAGGINTERPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *agg_interp_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggInterpType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (agg_interp_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggTruncFactor
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetaggtrfactor, NALU_HYPRE_BOOMERAMGSETAGGTRFACTOR)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggTruncFactor(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggP12TruncFactor
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetaggp12trfac, NALU_HYPRE_BOOMERAMGSETAGGP12TRFAC)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggpmaxelmts, NALU_HYPRE_BOOMERAMGSETAGGPMAXELMTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *p_max_elmts,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggPMaxElmts(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (p_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggP12MaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggp12maxelmt, NALU_HYPRE_BOOMERAMGSETAGGP12MAXELMT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *p_max_elmts,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (p_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterpvecs, NALU_HYPRE_BOOMERAMGSETINTERPVECS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_vectors,
  hypre_F90_Obj *interp_vectors,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpVectors(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (num_vectors),
                hypre_F90_PassObjRef (NALU_HYPRE_ParVector, interp_vectors) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecVariant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterpvecvar, NALU_HYPRE_BOOMERAMGSETINTERPVECVAR)
( hypre_F90_Obj *solver,
  hypre_F90_Int *var,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpVecVariant(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (var) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecQMax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterpvecqmx, NALU_HYPRE_BOOMERAMGSETINTERPVECQMX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *q_max,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpVecQMax(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (q_max) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterpvecqtr, NALU_HYPRE_BOOMERAMGSETINTERPVECQTR)
( hypre_F90_Obj *solver,
  hypre_F90_Real *q_trunc,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (q_trunc) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyOrder
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyorder, NALU_HYPRE_BOOMERAMGSETCHEBYORDER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cheby_order,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyOrder(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (cheby_order) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyFraction
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyfract, NALU_HYPRE_BOOMERAMGSETCHEBYFRACT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *cheby_fraction,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyFraction(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (cheby_fraction) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyscale, NALU_HYPRE_BOOMERAMGSETCHEBYSCALE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cheby_scale,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyScale(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (cheby_scale) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyVariant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyvariant, NALU_HYPRE_BOOMERAMGSETCHEBYVARIANT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cheby_variant,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyVariant(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (cheby_variant) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyEigEst
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyeigest, NALU_HYPRE_BOOMERAMGSETCHEBYEIGEST)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cheby_eig_est,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyEigEst(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (cheby_eig_est) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetKeepTranspose
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetkeeptransp, NALU_HYPRE_BOOMERAMGSETKEEPTRANSP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *keep_transpose,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetKeepTranspose(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (keep_transpose) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRAP2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrap2, NALU_HYPRE_BOOMERAMGSETRAP2)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rap2,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRAP2(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (rap2) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAdditive, NALU_HYPRE_BoomerAMGGetAdditive
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetadditive, NALU_HYPRE_BOOMERAMGSETADDITIVE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAdditive(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (add_lvl) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetadditive, NALU_HYPRE_BOOMERAMGGETADDITIVE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetAdditive(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (add_lvl) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMultAdditive, HYPRE BoomerAMGGetMultAdditive
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmultadd, NALU_HYPRE_BOOMERAMGSETMULTADD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMultAdditive(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (add_lvl) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmultadd, NALU_HYPRE_BOOMERAMGGETMULTADD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMultAdditive(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (add_lvl) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSimple, NALU_HYPRE_BoomerAMGGetSimple
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsimple, NALU_HYPRE_BOOMERAMGSETSIMPLE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSimple(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (add_lvl) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsimple, NALU_HYPRE_BOOMERAMGGETSIMPLE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSimple(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (add_lvl) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddLastLvl
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaddlastlvl, NALU_HYPRE_BOOMERAMGSETADDLASTLVL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_last_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAddLastLvl(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (add_last_lvl) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMultAddTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmultaddtrf, NALU_HYPRE_BOOMERAMGSETMULTADDTRF)
( hypre_F90_Obj *solver,
  hypre_F90_Real *add_tr,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (add_tr) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmultaddpmx, NALU_HYPRE_BOOMERAMGSETMULTADDPMX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_pmx,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (add_pmx) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaddrlxtype, NALU_HYPRE_BOOMERAMGSETADDRLXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_rlx_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAddRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (add_rlx_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaddrlxwt, NALU_HYPRE_BOOMERAMGSETADDRLXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *add_rlx_wt,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAddRelaxWt(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (add_rlx_wt) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSeqThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetseqthrshold, NALU_HYPRE_BOOMERAMGSETSEQTHRSHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *seq_th,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSeqThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (seq_th) ) );
}

#ifdef NALU_HYPRE_USING_DSUPERLU
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDSLUThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdsluthrshold, NALU_HYPRE_BOOMERAMGSETDSLUTHRSHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *dslu_th,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDSLUThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (dslu_th) ) );
}
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRedundant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetredundant, NALU_HYPRE_BOOMERAMGSETREDUNDANT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *redundant,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRedundant(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (redundant) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNonGalerkinTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnongaltol, NALU_HYPRE_BOOMERAMGSETNONGALTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *nongal_tol,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNonGalerkinTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (nongal_tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlvlnongaltol, NALU_HYPRE_BOOMERAMGSETLVLNONGALTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *nongal_tol,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (nongal_tol),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgsmg, NALU_HYPRE_BOOMERAMGSETGSMG)
( hypre_F90_Obj *solver,
  hypre_F90_Int *gsmg,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetGSMG(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (gsmg) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumsamples, NALU_HYPRE_BOOMERAMGSETNUMSAMPLES)
( hypre_F90_Obj *solver,
  hypre_F90_Int *gsmg,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumSamples(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (gsmg) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCGCIts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcgcits, NALU_HYPRE_BOOMERAMGSETCGCITS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *its,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCGCIts(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (its) ) );
}

#ifdef __cplusplus
}
#endif
