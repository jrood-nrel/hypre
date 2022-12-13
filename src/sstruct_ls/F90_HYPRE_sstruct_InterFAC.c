/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructFAC Routines
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaccreate, NALU_HYPRE_SSTRUCTFACCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacdestroy2, NALU_HYPRE_SSTRUCTFACDESTROY2)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACDestroy2(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACAMR_RAP
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacamrrap, NALU_HYPRE_SSTRUCTFACAMRRAP)
(hypre_F90_Obj *A,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 hypre_F90_Obj *facA,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACAMR_RAP(
                hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                rfactors,
                hypre_F90_PassObjRef (NALU_HYPRE_SStructMatrix, facA) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetup2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetup2, NALU_HYPRE_SSTRUCTFACSETUP2)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetup2(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSolve3
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsolve3, NALU_HYPRE_SSTRUCTFACSOLVE3)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSolve3(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
                hypre_F90_PassObj (NALU_HYPRE_SStructVector, x)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsettol, NALU_HYPRE_SSTRUCTFACSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetTol(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassReal (tol) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetplevels, NALU_HYPRE_SSTRUCTFACSETPLEVELS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *nparts,
 hypre_F90_IntArray *plevels,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetPLevels(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (nparts),
                hypre_F90_PassIntArray (plevels)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroCFSten
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczerocfsten, NALU_HYPRE_SSTRUCTFACZEROCFSTEN)
(hypre_F90_Obj *A,
 hypre_F90_Obj *grid,
 hypre_F90_Int *part,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACZeroCFSten(
                hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
                hypre_F90_PassInt (part),
                rfactors[NALU_HYPRE_MAXDIM] ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroFCSten
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczerofcsten, NALU_HYPRE_SSTRUCTFACZEROFCSTEN)
(hypre_F90_Obj *A,
 hypre_F90_Obj *grid,
 hypre_F90_Int *part,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACZeroFCSten(
                hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
                hypre_F90_PassInt (part) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroAMRMatrixData
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczeroamrmatrixdata, NALU_HYPRE_SSTRUCTFACZEROAMRMATRIXDATA)
(hypre_F90_Obj *A,
 hypre_F90_Int *part_crse,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACZeroAMRMatrixData(
                hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
                hypre_F90_PassInt (part_crse),
                rfactors[NALU_HYPRE_MAXDIM] ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroAMRVectorData
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczeroamrvectordata, NALU_HYPRE_SSTRUCTFACZEROAMRVECTORDATA)
(hypre_F90_Obj *b,
 hypre_F90_IntArray *plevels,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACZeroAMRVectorData(
                hypre_F90_PassObj (NALU_HYPRE_SStructVector, b),
                hypre_F90_PassIntArray (plevels),
                rfactors ));
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetprefinements, NALU_HYPRE_SSTRUCTFACSETPREFINEMENTS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *nparts,
 NALU_HYPRE_Int (*rfactors)[NALU_HYPRE_MAXDIM],
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetPRefinements(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (nparts),
                rfactors ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxlevels, NALU_HYPRE_SSTRUCTFACSETMAXLEVELS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_levels,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetMaxLevels(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (max_levels) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxiter, NALU_HYPRE_SSTRUCTFACSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (max_iter) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelchange, NALU_HYPRE_SSTRUCTFACSETRELCHANGE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *rel_change,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetRelChange(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (rel_change) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetzeroguess, NALU_HYPRE_SSTRUCTFACSETZEROGUESS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnonzeroguess, NALU_HYPRE_SSTRUCTFACSETNONZEROGUESS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetNonZeroGuess(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelaxtype, NALU_HYPRE_SSTRUCTFACSETRELAXTYPE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *relax_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_sstructfacsetjacobiweigh, NALU_HYPRE_SSTRUCTFACSETJACOBIWEIGH)
(hypre_F90_Obj *solver,
 hypre_F90_Real *weight,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructFACSetJacobiWeight( hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                                             hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumprerelax, NALU_HYPRE_SSTRUCTFACSETNUMPRERELAX)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_pre_relax,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACSetNumPreRelax( hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                                             hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumpostrelax, NALU_HYPRE_SSTRUCTFACSETNUMPOSTRELAX)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_post_relax,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructFACSetNumPostRelax(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (num_post_relax) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetcoarsesolver, NALU_HYPRE_SSTRUCTFACSETCOARSESOLVER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *csolver_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructFACSetCoarseSolverType(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (csolver_type)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetlogging, NALU_HYPRE_SSTRUCTFACSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructFACSetLogging(
               hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetnumiteration, NALU_HYPRE_SSTRUCTFACGETNUMITERATION)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassIntRef (num_iterations)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetfinalrelativ, NALU_HYPRE_SSTRUCTFACGETFINALRELATIV)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_SStructSolver, solver),
                hypre_F90_PassRealRef (norm) ));
}

#ifdef __cplusplus
}
#endif
