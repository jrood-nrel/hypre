/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructFAC Routines
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfaccreate, NALU_HYPRE_SSTRUCTFACCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacdestroy2, NALU_HYPRE_SSTRUCTFACDESTROY2)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACDestroy2(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACAMR_RAP
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacamrrap, NALU_HYPRE_SSTRUCTFACAMRRAP)
(nalu_hypre_F90_Obj *A,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 nalu_hypre_F90_Obj *facA,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACAMR_RAP(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                rfactors,
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructMatrix, facA) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetup2
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetup2, NALU_HYPRE_SSTRUCTFACSETUP2)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetup2(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSolve3
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsolve3, NALU_HYPRE_SSTRUCTFACSOLVE3)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSolve3(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsettol, NALU_HYPRE_SSTRUCTFACSETTOL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassReal (tol) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetplevels, NALU_HYPRE_SSTRUCTFACSETPLEVELS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *nparts,
 nalu_hypre_F90_IntArray *plevels,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetPLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (nparts),
                nalu_hypre_F90_PassIntArray (plevels)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroCFSten
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfaczerocfsten, NALU_HYPRE_SSTRUCTFACZEROCFSTEN)
(nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *grid,
 nalu_hypre_F90_Int *part,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACZeroCFSten(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
                nalu_hypre_F90_PassInt (part),
                rfactors[NALU_HYPRE_MAXDIM] ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroFCSten
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfaczerofcsten, NALU_HYPRE_SSTRUCTFACZEROFCSTEN)
(nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *grid,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACZeroFCSten(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
                nalu_hypre_F90_PassInt (part) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroAMRMatrixData
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfaczeroamrmatrixdata, NALU_HYPRE_SSTRUCTFACZEROAMRMATRIXDATA)
(nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Int *part_crse,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACZeroAMRMatrixData(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                nalu_hypre_F90_PassInt (part_crse),
                rfactors[NALU_HYPRE_MAXDIM] ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroAMRVectorData
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfaczeroamrvectordata, NALU_HYPRE_SSTRUCTFACZEROAMRVECTORDATA)
(nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_IntArray *plevels,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACZeroAMRVectorData(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
                nalu_hypre_F90_PassIntArray (plevels),
                rfactors ));
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetprefinements, NALU_HYPRE_SSTRUCTFACSETPREFINEMENTS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *nparts,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetPRefinements(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (nparts),
                rfactors ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetmaxlevels, NALU_HYPRE_SSTRUCTFACSETMAXLEVELS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *max_levels,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetMaxLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (max_levels) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetmaxiter, NALU_HYPRE_SSTRUCTFACSETMAXITER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *max_iter,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (max_iter) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetrelchange, NALU_HYPRE_SSTRUCTFACSETRELCHANGE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *rel_change,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetRelChange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (rel_change) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetzeroguess, NALU_HYPRE_SSTRUCTFACSETZEROGUESS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetnonzeroguess, NALU_HYPRE_SSTRUCTFACSETNONZEROGUESS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetNonZeroGuess(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetrelaxtype, NALU_HYPRE_SSTRUCTFACSETRELAXTYPE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *relax_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetjacobiweigh, NALU_HYPRE_SSTRUCTFACSETJACOBIWEIGH)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *weight,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFACSetJacobiWeight( nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                                             nalu_hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetnumprerelax, NALU_HYPRE_SSTRUCTFACSETNUMPRERELAX)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_pre_relax,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetNumPreRelax( nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                                             nalu_hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetnumpostrelax, NALU_HYPRE_SSTRUCTFACSETNUMPOSTRELAX)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_post_relax,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFACSetNumPostRelax(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (num_post_relax) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetcoarsesolver, NALU_HYPRE_SSTRUCTFACSETCOARSESOLVER)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *csolver_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFACSetCoarseSolverType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (csolver_type)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacsetlogging, NALU_HYPRE_SSTRUCTFACSETLOGGING)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *logging,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructFACSetLogging(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               nalu_hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacgetnumiteration, NALU_HYPRE_SSTRUCTFACGETNUMITERATION)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_iterations,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructfacgetfinalrelativ, NALU_HYPRE_SSTRUCTFACGETFINALRELATIV)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *norm,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                nalu_hypre_F90_PassRealRef (norm) ));
}

#ifdef __cplusplus
}
#endif
