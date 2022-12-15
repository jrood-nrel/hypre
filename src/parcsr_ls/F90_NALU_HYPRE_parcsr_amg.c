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

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgcreate, NALU_HYPRE_BOOMERAMGCREATE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGCreate(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgdestroy, NALU_HYPRE_BOOMERAMGDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetup, NALU_HYPRE_BOOMERAMGSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsolve, NALU_HYPRE_BOOMERAMGSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSolveT
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsolvet, NALU_HYPRE_BOOMERAMGSOLVET)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSolveT(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRestriction
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetrestriction, NALU_HYPRE_BOOMERAMGSETRESTRICTION)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *restr_par,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRestriction(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (restr_par) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxLevels, NALU_HYPRE_BoomerAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmaxlevels, NALU_HYPRE_BOOMERAMGSETMAXLEVELS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_levels,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_levels) ) );
}



void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetmaxlevels, NALU_HYPRE_BOOMERAMGGETMAXLEVELS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_levels,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMaxLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (max_levels) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxCoarseSize, NALU_HYPRE_BoomerAMGGetMaxCoarseSize
 *--------------------------------------------------------------------------*/


void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmaxcoarsesize, NALU_HYPRE_BOOMERAMGSETMAXCOARSESIZE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_coarse_size,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxCoarseSize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_coarse_size) ) );
}



void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetmaxcoarsesize, NALU_HYPRE_BOOMERAMGGETMAXCOARSESIZE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_coarse_size,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMaxCoarseSize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (max_coarse_size) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMinCoarseSize, NALU_HYPRE_BoomerAMGGetMinCoarseSize
 *--------------------------------------------------------------------------*/


void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmincoarsesize, NALU_HYPRE_BOOMERAMGSETMINCOARSESIZE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *min_coarse_size,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMinCoarseSize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (min_coarse_size) ) );
}



void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetmincoarsesize, NALU_HYPRE_BOOMERAMGGETMINCOARSESIZE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *min_coarse_size,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMinCoarseSize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (min_coarse_size) ) );
}




/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetStrongThreshold, NALU_HYPRE_BoomerAMGGetStrongThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetstrongthrshld, NALU_HYPRE_BOOMERAMGSETSTRONGTHRSHLD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *strong_threshold,
  nalu_hypre_F90_Int *ierr              )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetStrongThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (strong_threshold) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetstrongthrshld, NALU_HYPRE_BOOMERAMGGETSTRONGTHRSHLD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *strong_threshold,
  nalu_hypre_F90_Int *ierr              )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetStrongThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (strong_threshold) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxRowSum, NALU_HYPRE_BoomerAMGGetMaxRowSum
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmaxrowsum, NALU_HYPRE_BOOMERAMGSETMAXROWSUM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *max_row_sum,
  nalu_hypre_F90_Int *ierr              )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxRowSum(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (max_row_sum) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetmaxrowsum, NALU_HYPRE_BOOMERAMGGETMAXROWSUM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *max_row_sum,
  nalu_hypre_F90_Int *ierr              )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMaxRowSum(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (max_row_sum) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetTruncFactor, NALU_HYPRE_BoomerAMGGetTruncFactor
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsettruncfactor, NALU_HYPRE_BOOMERAMGSETTRUNCFACTOR)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *trunc_factor,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetTruncFactor(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (trunc_factor) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggettruncfactor, NALU_HYPRE_BOOMERAMGGETTRUNCFACTOR)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *trunc_factor,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetTruncFactor(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPMaxElmts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetpmaxelmts, NALU_HYPRE_BOOMERAMGSETPMAXELMTS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *p_max_elmts,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetPMaxElmts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (p_max_elmts) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetpmaxelmts, NALU_HYPRE_BOOMERAMGGETPMAXELMTS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *p_max_elmts,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetPMaxElmts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (p_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold, NALU_HYPRE_BoomerAMGGetJacobiTruncThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetjacobitrunc, NALU_HYPRE_BOOMERAMGSETJACOBITRUNC)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *trunc_factor,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (trunc_factor) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetjacobitrunc, NALU_HYPRE_BOOMERAMGGETJACOBITRUNC)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *trunc_factor,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetJacobiTruncThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPostInterpType, NALU_HYPRE_BoomerAMGGetPostInterpType
 *  If >0, specifies something to do to improve a computed interpolation matrix.
 * defaults to 0, for nothing.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetpostinterp, NALU_HYPRE_BOOMERAMGSETPOSTINTERP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *type,
  nalu_hypre_F90_Int *ierr            )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetPostInterpType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (type) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetinterptype, NALU_HYPRE_BOOMERAMGSETINTERPTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *interp_type,
  nalu_hypre_F90_Int *ierr         )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (interp_type) ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSepWeight
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetsepweight, NALU_HYPRE_BOOMERAMGSETSEPWEIGHT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *sep_weight,
  nalu_hypre_F90_Int *ierr         )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSepWeight(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (sep_weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMinIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetminiter, NALU_HYPRE_BOOMERAMGSETMINITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *min_iter,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMinIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxIter, NALU_HYPRE_BoomerAMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmaxiter, NALU_HYPRE_BOOMERAMGSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetmaxiter, NALU_HYPRE_BOOMERAMGGETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCoarsenType, NALU_HYPRE_BoomerAMGGetCoarsenType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetcoarsentype, NALU_HYPRE_BOOMERAMGSETCOARSENTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *coarsen_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCoarsenType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (coarsen_type) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetcoarsentype, NALU_HYPRE_BOOMERAMGGETCOARSENTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *coarsen_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCoarsenType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (coarsen_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMeasureType, NALU_HYPRE_BoomerAMGGetMeasureType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmeasuretype, NALU_HYPRE_BOOMERAMGSETMEASURETYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *measure_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMeasureType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (measure_type) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetmeasuretype, NALU_HYPRE_BOOMERAMGGETMEASURETYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *measure_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMeasureType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (measure_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOldDefault
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetolddefault, NALU_HYPRE_BOOMERAMGSETOLDDEFAULT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetOldDefault(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSetupType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetsetuptype, NALU_HYPRE_BOOMERAMGSETSETUPTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *setup_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSetupType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (setup_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCycleType, NALU_HYPRE_BoomerAMGGetCycleType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetcycletype, NALU_HYPRE_BOOMERAMGSETCYCLETYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *cycle_type,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCycleType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (cycle_type) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetcycletype, NALU_HYPRE_BOOMERAMGGETCYCLETYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *cycle_type,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCycleType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (cycle_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetTol, NALU_HYPRE_BoomerAMGGetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsettol, NALU_HYPRE_BOOMERAMGSETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol)     ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggettol, NALU_HYPRE_BOOMERAMGGETTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (tol)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumSweeps
 * DEPRECATED.  Use SetNumSweeps and SetCycleNumSweeps instead.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetnumgridsweeps, NALU_HYPRE_BOOMERAMGSETNUMGRIDSWEEPS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_grid_sweeps,
  nalu_hypre_F90_Int *ierr             )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumGridSweeps(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_grid_sweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumSweeps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetnumsweeps, NALU_HYPRE_BOOMERAMGSETNUMSWEEPS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_sweeps,
  nalu_hypre_F90_Int *ierr             )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumSweeps(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (num_sweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCycleNumSweeps, NALU_HYPRE_BoomerAMGGetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetcyclenumsweeps, NALU_HYPRE_BOOMERAMGSETCYCLENUMSWEEPS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_sweeps,
  nalu_hypre_F90_Int *k,
  nalu_hypre_F90_Int *ierr             )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCycleNumSweeps(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (num_sweeps),
                nalu_hypre_F90_PassInt (k) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetcyclenumsweeps, NALU_HYPRE_BOOMERAMGGETCYCLENUMSWEEPS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_sweeps,
  nalu_hypre_F90_Int *k,
  nalu_hypre_F90_Int *ierr             )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCycleNumSweeps(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_sweeps),
                nalu_hypre_F90_PassInt (k) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGInitGridRelaxation
 *
 * RDF: This is probably not a very useful Fortran routine because you can't do
 * anything with the pointers to arrays that are allocated.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramginitgridrelaxatn, NALU_HYPRE_BOOMERAMGINITGRIDRELAXATN)
( nalu_hypre_F90_Obj *num_grid_sweeps,
  nalu_hypre_F90_Obj *grid_relax_type,
  nalu_hypre_F90_Obj *grid_relax_points,
  nalu_hypre_F90_Int *coarsen_type,
  nalu_hypre_F90_Obj *relax_weights,
  nalu_hypre_F90_Int *max_levels,
  nalu_hypre_F90_Int *ierr               )
{
   *num_grid_sweeps   = (nalu_hypre_F90_Obj) nalu_hypre_CTAlloc(NALU_HYPRE_Int*,  1, NALU_HYPRE_MEMORY_HOST);
   *grid_relax_type   = (nalu_hypre_F90_Obj) nalu_hypre_CTAlloc(NALU_HYPRE_Int*,  1, NALU_HYPRE_MEMORY_HOST);
   *grid_relax_points = (nalu_hypre_F90_Obj) nalu_hypre_CTAlloc(NALU_HYPRE_Int**,  1, NALU_HYPRE_MEMORY_HOST);
   *relax_weights     = (nalu_hypre_F90_Obj) nalu_hypre_CTAlloc(NALU_HYPRE_Real*,  1, NALU_HYPRE_MEMORY_HOST);

   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGInitGridRelaxation(
                (NALU_HYPRE_Int **)    *num_grid_sweeps,
                (NALU_HYPRE_Int **)    *grid_relax_type,
                (NALU_HYPRE_Int ***)   *grid_relax_points,
                nalu_hypre_F90_PassInt (coarsen_type),
                (NALU_HYPRE_Real **) *relax_weights,
                nalu_hypre_F90_PassInt (max_levels)         ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGFinalizeGridRelaxation
 *
 * RDF: This is probably not a very useful Fortran routine because you can't do
 * anything with the pointers to arrays.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgfingridrelaxatn, NALU_HYPRE_BOOMERAMGFINGRIDRELAXATN)
( nalu_hypre_F90_Obj *num_grid_sweeps,
  nalu_hypre_F90_Obj *grid_relax_type,
  nalu_hypre_F90_Obj *grid_relax_points,
  nalu_hypre_F90_Obj *relax_weights,
  nalu_hypre_F90_Int *ierr               )
{
   char *ptr_num_grid_sweeps   = (char *) *num_grid_sweeps;
   char *ptr_grid_relax_type   = (char *) *grid_relax_type;
   char *ptr_grid_relax_points = (char *) *grid_relax_points;
   char *ptr_relax_weights     = (char *) *relax_weights;

   nalu_hypre_TFree(ptr_num_grid_sweeps, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ptr_grid_relax_type, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ptr_grid_relax_points, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ptr_relax_weights, NALU_HYPRE_MEMORY_HOST);

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGridRelaxType
 * DEPRECATED.  Use SetRelaxType and SetCycleRelaxType instead.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetgridrelaxtype, NALU_HYPRE_BOOMERAMGSETGRIDRELAXTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_IntArray *grid_relax_type,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetGridRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntArray (grid_relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetrelaxtype, NALU_HYPRE_BOOMERAMGSETRELAXTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_type,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCycleRelaxType, NALU_HYPRE_BoomerAMGGetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetcyclerelaxtype, NALU_HYPRE_BOOMERAMGSETCYCLERELAXTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_type,
  nalu_hypre_F90_Int *k,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCycleRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (relax_type),
                nalu_hypre_F90_PassInt (k) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetcyclerelaxtype, NALU_HYPRE_BOOMERAMGGETCYCLERELAXTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_type,
  nalu_hypre_F90_Int *k,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCycleRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (relax_type),
                nalu_hypre_F90_PassInt (k)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxOrder
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetrelaxorder, NALU_HYPRE_BOOMERAMGSETRELAXORDER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_order,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRelaxOrder(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (relax_order) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGridRelaxPoints
 * DEPRECATED.  There is no alternative function.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetgridrelaxpnts, NALU_HYPRE_BOOMERAMGSETGRIDRELAXPNTS)
( nalu_hypre_F90_Obj *solver,
  NALU_HYPRE_Int      **grid_relax_points,
  nalu_hypre_F90_Int *ierr               )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetGridRelaxPoints(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                (NALU_HYPRE_Int **)        grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxWeight
 * DEPRECATED.  Use SetRelaxWt and SetLevelRelaxWt instead.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetrelaxweight, NALU_HYPRE_BOOMERAMGSETRELAXWEIGHT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_IntArray *relax_weights,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRelaxWeight(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealArray (relax_weights) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetrelaxwt, NALU_HYPRE_BOOMERAMGSETRELAXWT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *relax_weight,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRelaxWt(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (relax_weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetlevelrelaxwt, NALU_HYPRE_BOOMERAMGSETLEVELRELAXWT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *relax_weight,
  nalu_hypre_F90_Int *level,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLevelRelaxWt(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (relax_weight),
                nalu_hypre_F90_PassInt (level) ) );
}




/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOuterWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetouterwt, NALU_HYPRE_BOOMERAMGSETOUTERWT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *outer_wt,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetOuterWt(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (outer_wt) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetlevelouterwt, NALU_HYPRE_BOOMERAMGSETLEVELOUTERWT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *outer_wt,
  nalu_hypre_F90_Int *level,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLevelOuterWt(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (outer_wt),
                nalu_hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothType, NALU_HYPRE_BoomerAMGGetSmoothType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetsmoothtype, NALU_HYPRE_BOOMERAMGSETSMOOTHTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *smooth_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSmoothType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (smooth_type) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetsmoothtype, NALU_HYPRE_BOOMERAMGGETSMOOTHTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *smooth_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSmoothType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (smooth_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothNumLvls, NALU_HYPRE_BoomerAMGGetSmoothNumLvls
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetsmoothnumlvls, NALU_HYPRE_BOOMERAMGSETSMOOTHNUMLVLS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *smooth_num_levels,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSmoothNumLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (smooth_num_levels) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetsmoothnumlvls, NALU_HYPRE_BOOMERAMGGETSMOOTHNUMLVLS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *smooth_num_levels,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSmoothNumLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (smooth_num_levels) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothNumSwps, NALU_HYPRE_BoomerAMGGetSmoothNumSwps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetsmoothnumswps, NALU_HYPRE_BOOMERAMGSETSMOOTHNUMSWPS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *smooth_num_sweeps,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (smooth_num_sweeps) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetsmoothnumswps, NALU_HYPRE_BOOMERAMGGETSMOOTHNUMSWPS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *smooth_num_sweeps,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSmoothNumSweeps(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (smooth_num_sweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLogging, NALU_HYPRE_BoomerAMGGetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetlogging, NALU_HYPRE_BOOMERAMGSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetlogging, NALU_HYPRE_BOOMERAMGGETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPrintLevel, NALU_HYPRE_BoomerAMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetprintlevel, NALU_HYPRE_BOOMERAMGSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetprintlevel, NALU_HYPRE_BOOMERAMGGETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPrintFileName
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetprintfilename, NALU_HYPRE_BOOMERAMGSETPRINTFILENAME)
( nalu_hypre_F90_Obj *solver,
  char     *print_file_name,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetPrintFileName(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                (char *)        print_file_name ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDebugFlag, NALU_HYPRE_BoomerAMGGetDebugFlag
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetdebugflag, NALU_HYPRE_BOOMERAMGSETDEBUGFLAG)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *debug_flag,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDebugFlag(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (debug_flag) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetdebugflag, NALU_HYPRE_BOOMERAMGGETDEBUGFLAG)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *debug_flag,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetDebugFlag(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (debug_flag) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetnumiterations, NALU_HYPRE_BOOMERAMGGETNUMITERATIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_iterations,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetCumNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetcumnumiterati, NALU_HYPRE_BOOMERAMGGETCUMNUMITERATI)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *cum_num_iterations,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetCumNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (cum_num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetResidual
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetresidual, NALU_HYPRE_BOOMERAMGGETRESIDUAL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *residual,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetResidual(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParVector, residual)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetFinalRelativeResNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetfinalreltvres, NALU_HYPRE_BOOMERAMGGETFINALRELTVRES)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *rel_resid_norm,
  nalu_hypre_F90_Int *ierr            )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (rel_resid_norm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetVariant, NALU_HYPRE_BoomerAMGGetVariant
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetvariant, NALU_HYPRE_BOOMERAMGSETVARIANT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *variant,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetVariant(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (variant) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetvariant, NALU_HYPRE_BOOMERAMGGETVARIANT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *variant,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetVariant(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (variant) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOverlap, NALU_HYPRE_BoomerAMGGetOverlap
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetoverlap, NALU_HYPRE_BOOMERAMGSETOVERLAP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *overlap,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetOverlap(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (overlap) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetoverlap, NALU_HYPRE_BOOMERAMGGETOVERLAP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *overlap,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetOverlap(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (overlap) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDomainType, NALU_HYPRE_BoomerAMGGetDomainType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetdomaintype, NALU_HYPRE_BOOMERAMGSETDOMAINTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *domain_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDomainType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (domain_type) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetdomaintype, NALU_HYPRE_BOOMERAMGGETDOMAINTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *domain_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetDomainType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (domain_type) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetschwarznonsym, NALU_HYPRE_BOOMERAMGSETSCHWARZNONSYM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *schwarz_non_symm,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (schwarz_non_symm) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSchwarzRlxWt, NALU_HYPRE_BoomerAMGGetSchwarzRlxWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetschwarzrlxwt, NALU_HYPRE_BOOMERAMGSETSCHWARZRLXWT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *schwarz_rlx_weight,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (schwarz_rlx_weight)) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetschwarzrlxwt, NALU_HYPRE_BOOMERAMGGETSCHWARZRLXWT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *schwarz_rlx_weight,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSchwarzRlxWeight(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (schwarz_rlx_weight)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSym
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetsym, NALU_HYPRE_BOOMERAMGSETSYM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *sym,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSym(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (sym) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetlevel, NALU_HYPRE_BOOMERAMGSETLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *level,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetthreshold, NALU_HYPRE_BOOMERAMGSETTHRESHOLD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *threshold,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (threshold)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetFilter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetfilter, NALU_HYPRE_BOOMERAMGSETFILTER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *filter,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetFilter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (filter)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDropTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetdroptol, NALU_HYPRE_BOOMERAMGSETDROPTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *drop_tol,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDropTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (drop_tol)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxNzPerRow
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmaxnzperrow, NALU_HYPRE_BOOMERAMGSETMAXNZPERROW)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_nz_per_row,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMaxNzPerRow(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_nz_per_row) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuBJ
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgseteubj, NALU_HYPRE_BOOMERAMGSETEUBJ)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *eu_bj,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetEuBJ(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (eu_bj) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgseteulevel, NALU_HYPRE_BOOMERAMGSETEULEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *eu_level,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetEuLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (eu_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuSparseA
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgseteusparsea, NALU_HYPRE_BOOMERAMGSETEUSPARSEA)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *eu_sparse_a,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetEuSparseA(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (eu_sparse_a)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuclidFile
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgseteuclidfile, NALU_HYPRE_BOOMERAMGSETEUCLIDFILE)
( nalu_hypre_F90_Obj *solver,
  char     *euclidfile,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetEuclidFile(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                (char *)        euclidfile ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumFunctions, NALU_HYPRE_BoomerAMGGetNumFunctions
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetnumfunctions, NALU_HYPRE_BOOMERAMGSETNUMFUNCTIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_functions,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumFunctions(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (num_functions) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetnumfunctions, NALU_HYPRE_BOOMERAMGGETNUMFUNCTIONS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_functions,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetNumFunctions(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_functions) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNodal
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetnodal, NALU_HYPRE_BOOMERAMGSETNODAL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *nodal,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNodal(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (nodal) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNodalDiag
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetnodaldiag, NALU_HYPRE_BOOMERAMGSETNODALDIAG)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *nodal_diag,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNodalDiag(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (nodal_diag) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDofFunc
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetdoffunc, NALU_HYPRE_BOOMERAMGSETDOFFUNC)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_IntArray *dof_func,
  nalu_hypre_F90_Int *ierr             )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDofFunc(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntArray (dof_func) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumPaths
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetnumpaths, NALU_HYPRE_BOOMERAMGSETNUMPATHS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_paths,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumPaths(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (num_paths) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggNumLevels
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetaggnumlevels, NALU_HYPRE_BOOMERAMGSETAGGNUMLEVELS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *agg_num_levels,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggNumLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (agg_num_levels) ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggInterpType
 *--------------------------------------------------------------------------*/


void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetagginterptype, NALU_HYPRE_BOOMERAMGSETAGGINTERPTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *agg_interp_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggInterpType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (agg_interp_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggTruncFactor
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetaggtrfactor, NALU_HYPRE_BOOMERAMGSETAGGTRFACTOR)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *trunc_factor,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggTruncFactor(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggP12TruncFactor
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetaggp12trfac, NALU_HYPRE_BOOMERAMGSETAGGP12TRFAC)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *trunc_factor,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggPMaxElmts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetaggpmaxelmts, NALU_HYPRE_BOOMERAMGSETAGGPMAXELMTS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *p_max_elmts,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggPMaxElmts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (p_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggP12MaxElmts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetaggp12maxelmt, NALU_HYPRE_BOOMERAMGSETAGGP12MAXELMT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *p_max_elmts,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (p_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVectors
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetinterpvecs, NALU_HYPRE_BOOMERAMGSETINTERPVECS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *num_vectors,
  nalu_hypre_F90_Obj *interp_vectors,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpVectors(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (num_vectors),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParVector, interp_vectors) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecVariant
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetinterpvecvar, NALU_HYPRE_BOOMERAMGSETINTERPVECVAR)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *var,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpVecVariant(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (var) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecQMax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetinterpvecqmx, NALU_HYPRE_BOOMERAMGSETINTERPVECQMX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *q_max,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpVecQMax(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (q_max) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetinterpvecqtr, NALU_HYPRE_BOOMERAMGSETINTERPVECQTR)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *q_trunc,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (q_trunc) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyOrder
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetchebyorder, NALU_HYPRE_BOOMERAMGSETCHEBYORDER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *cheby_order,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyOrder(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (cheby_order) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyFraction
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetchebyfract, NALU_HYPRE_BOOMERAMGSETCHEBYFRACT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *cheby_fraction,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyFraction(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (cheby_fraction) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyScale
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetchebyscale, NALU_HYPRE_BOOMERAMGSETCHEBYSCALE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *cheby_scale,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyScale(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (cheby_scale) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyVariant
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetchebyvariant, NALU_HYPRE_BOOMERAMGSETCHEBYVARIANT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *cheby_variant,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyVariant(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (cheby_variant) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyEigEst
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetchebyeigest, NALU_HYPRE_BOOMERAMGSETCHEBYEIGEST)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *cheby_eig_est,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetChebyEigEst(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (cheby_eig_est) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetKeepTranspose
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetkeeptransp, NALU_HYPRE_BOOMERAMGSETKEEPTRANSP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *keep_transpose,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetKeepTranspose(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (keep_transpose) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRAP2
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetrap2, NALU_HYPRE_BOOMERAMGSETRAP2)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *rap2,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRAP2(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (rap2) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAdditive, NALU_HYPRE_BoomerAMGGetAdditive
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetadditive, NALU_HYPRE_BOOMERAMGSETADDITIVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *add_lvl,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAdditive(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (add_lvl) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetadditive, NALU_HYPRE_BOOMERAMGGETADDITIVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *add_lvl,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetAdditive(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (add_lvl) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMultAdditive, HYPRE BoomerAMGGetMultAdditive
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmultadd, NALU_HYPRE_BOOMERAMGSETMULTADD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *add_lvl,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMultAdditive(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (add_lvl) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetmultadd, NALU_HYPRE_BOOMERAMGGETMULTADD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *add_lvl,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetMultAdditive(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (add_lvl) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSimple, NALU_HYPRE_BoomerAMGGetSimple
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetsimple, NALU_HYPRE_BOOMERAMGSETSIMPLE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *add_lvl,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSimple(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (add_lvl) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramggetsimple, NALU_HYPRE_BOOMERAMGGETSIMPLE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *add_lvl,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGGetSimple(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (add_lvl) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddLastLvl
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetaddlastlvl, NALU_HYPRE_BOOMERAMGSETADDLASTLVL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *add_last_lvl,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAddLastLvl(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (add_last_lvl) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMultAddTruncFactor
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmultaddtrf, NALU_HYPRE_BOOMERAMGSETMULTADDTRF)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *add_tr,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (add_tr) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetmultaddpmx, NALU_HYPRE_BOOMERAMGSETMULTADDPMX)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *add_pmx,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (add_pmx) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetaddrlxtype, NALU_HYPRE_BOOMERAMGSETADDRLXTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *add_rlx_type,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAddRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (add_rlx_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddRelaxWt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetaddrlxwt, NALU_HYPRE_BOOMERAMGSETADDRLXWT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *add_rlx_wt,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetAddRelaxWt(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (add_rlx_wt) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSeqThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetseqthrshold, NALU_HYPRE_BOOMERAMGSETSEQTHRSHOLD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *seq_th,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetSeqThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (seq_th) ) );
}

#ifdef NALU_HYPRE_USING_DSUPERLU
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDSLUThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetdsluthrshold, NALU_HYPRE_BOOMERAMGSETDSLUTHRSHOLD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *dslu_th,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetDSLUThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (dslu_th) ) );
}
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRedundant
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetredundant, NALU_HYPRE_BOOMERAMGSETREDUNDANT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *redundant,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetRedundant(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (redundant) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNonGalerkinTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetnongaltol, NALU_HYPRE_BOOMERAMGSETNONGALTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *nongal_tol,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNonGalerkinTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (nongal_tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetlvlnongaltol, NALU_HYPRE_BOOMERAMGSETLVLNONGALTOL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *nongal_tol,
  nalu_hypre_F90_Int *level,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (nongal_tol),
                nalu_hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetgsmg, NALU_HYPRE_BOOMERAMGSETGSMG)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *gsmg,
  nalu_hypre_F90_Int *ierr            )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetGSMG(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (gsmg) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetnumsamples, NALU_HYPRE_BOOMERAMGSETNUMSAMPLES)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *gsmg,
  nalu_hypre_F90_Int *ierr            )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetNumSamples(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (gsmg) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCGCIts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_boomeramgsetcgcits, NALU_HYPRE_BOOMERAMGSETCGCITS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *its,
  nalu_hypre_F90_Int *ierr            )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_BoomerAMGSetCGCIts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (its) ) );
}

#ifdef __cplusplus
}
#endif
