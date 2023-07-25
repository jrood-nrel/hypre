/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_Euclid Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidCreate - Return a Euclid "solver".
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidcreate, NALU_HYPRE_EUCLIDCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidCreate(
              nalu_hypre_F90_PassComm (comm),
              nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidDestroy - Destroy a Euclid object.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_eucliddestroy, NALU_HYPRE_EUCLIDDESTROY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidDestroy(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetup - Set up function for Euclid.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetup, NALU_HYPRE_EUCLIDSETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetup(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)   );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSolve - Solve function for Euclid.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsolve, NALU_HYPRE_EUCLIDSOLVE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSolve(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x)  );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetParams
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetparams, NALU_HYPRE_EUCLIDSETPARAMS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *argc,
 char **argv,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetParams(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (argc),
              (char **)       argv );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetParamsFromFile
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetparamsfromfile, NALU_HYPRE_EUCLIDSETPARAMSFROMFILE)
(nalu_hypre_F90_Obj *solver,
 char *filename,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetParamsFromFile(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              (char *)        filename );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetlevel, NALU_HYPRE_EUCLIDSETLEVEL)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *eu_level,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetLevel(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (eu_level) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetBJ
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetbj, NALU_HYPRE_EUCLIDSETBJ)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *bj,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetBJ(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (bj) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetStats
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetstats, NALU_HYPRE_EUCLIDSETSTATS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *eu_stats,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetStats(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (eu_stats) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetMem
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetmem, NALU_HYPRE_EUCLIDSETMEM)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *eu_mem,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetMem(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (eu_mem) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetSparseA
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetsparsea, NALU_HYPRE_EUCLIDSETSPARSEA)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *spa,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetSparseA(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassReal (spa) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetRowScale
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetrowscale, NALU_HYPRE_EUCLIDSETROWSCALE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *row_scale,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetRowScale(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassInt (row_scale) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetILUT *
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_euclidsetilut, NALU_HYPRE_EUCLIDSETILUT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *drop_tol,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_EuclidSetILUT(
              nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              nalu_hypre_F90_PassReal (drop_tol) );
}

#ifdef __cplusplus
}
#endif
