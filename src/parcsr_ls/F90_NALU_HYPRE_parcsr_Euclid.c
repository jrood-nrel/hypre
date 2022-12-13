/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_Euclid Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidCreate - Return a Euclid "solver".
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidcreate, NALU_HYPRE_EUCLIDCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidCreate(
              hypre_F90_PassComm (comm),
              hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidDestroy - Destroy a Euclid object.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_eucliddestroy, NALU_HYPRE_EUCLIDDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidDestroy(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetup - Set up function for Euclid.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetup, NALU_HYPRE_EUCLIDSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetup(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
              hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
              hypre_F90_PassObj (NALU_HYPRE_ParVector, x)   );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSolve - Solve function for Euclid.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsolve, NALU_HYPRE_EUCLIDSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSolve(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
              hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
              hypre_F90_PassObj (NALU_HYPRE_ParVector, x)  );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetParams
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetparams, NALU_HYPRE_EUCLIDSETPARAMS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *argc,
 char **argv,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetParams(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (argc),
              (char **)       argv );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetParamsFromFile
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetparamsfromfile, NALU_HYPRE_EUCLIDSETPARAMSFROMFILE)
(hypre_F90_Obj *solver,
 char *filename,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetParamsFromFile(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              (char *)        filename );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetlevel, NALU_HYPRE_EUCLIDSETLEVEL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *eu_level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetLevel(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (eu_level) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetBJ
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetbj, NALU_HYPRE_EUCLIDSETBJ)
(hypre_F90_Obj *solver,
 hypre_F90_Int *bj,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetBJ(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (bj) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetStats
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetstats, NALU_HYPRE_EUCLIDSETSTATS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *eu_stats,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetStats(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (eu_stats) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetMem
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetmem, NALU_HYPRE_EUCLIDSETMEM)
(hypre_F90_Obj *solver,
 hypre_F90_Int *eu_mem,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetMem(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (eu_mem) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetSparseA
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetsparsea, NALU_HYPRE_EUCLIDSETSPARSEA)
(hypre_F90_Obj *solver,
 hypre_F90_Real *spa,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetSparseA(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassReal (spa) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetRowScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetrowscale, NALU_HYPRE_EUCLIDSETROWSCALE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *row_scale,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetRowScale(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassInt (row_scale) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetILUT *
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetilut, NALU_HYPRE_EUCLIDSETILUT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *drop_tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_EuclidSetILUT(
              hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
              hypre_F90_PassReal (drop_tol) );
}

#ifdef __cplusplus
}
#endif
