/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_MGR Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrcreate, NALU_HYPRE_MGRCREATE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRCreate(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrdestroy, NALU_HYPRE_MGRDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetup, NALU_HYPRE_MGRSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsolve, NALU_HYPRE_MGRSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

#ifdef NALU_HYPRE_USING_DSUPERLU

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrdirectsolvercreate, NALU_HYPRE_MGRDIRECTSOLVERCREATE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRDirectSolverCreate(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrdirectsolverdestroy, NALU_HYPRE_MGRDIRECTSOLVERDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRDirectSolverDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrdirectsolversetup, NALU_HYPRE_MGRDIRECTSOLVERSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRDirectSolverSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrdirectsolversolve, NALU_HYPRE_MGRDIRECTSOLVERSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRDirectSolverSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCptsByCtgBlock
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetcptsbyctgblock, NALU_HYPRE_MGRSETCPTSBYCTGBLOCK)
( nalu_hypre_F90_Obj           *solver,
  nalu_hypre_F90_Int           *block_size,
  nalu_hypre_F90_Int           *max_num_levels,
  nalu_hypre_F90_BigIntArray   *idx_array,
  nalu_hypre_F90_IntArray      *block_num_coarse_points,
  nalu_hypre_F90_IntArrayArray *block_coarse_indexes,
  nalu_hypre_F90_Int           *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCpointsByContiguousBlock(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (block_size),
                nalu_hypre_F90_PassInt (max_num_levels),
                nalu_hypre_F90_PassBigIntArray (idx_array),
                nalu_hypre_F90_PassIntArray (block_num_coarse_points),
                nalu_hypre_F90_PassIntArrayArray (block_coarse_indexes) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCpointsByBlock
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetcpointsbyblock, NALU_HYPRE_MGRSETCPOINTSBYBLOCK)
( nalu_hypre_F90_Obj           *solver,
  nalu_hypre_F90_Int           *block_size,
  nalu_hypre_F90_Int           *max_num_levels,
  nalu_hypre_F90_IntArray      *block_num_coarse_points,
  nalu_hypre_F90_IntArrayArray *block_coarse_indexes,
  nalu_hypre_F90_Int           *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCpointsByBlock(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (block_size),
                nalu_hypre_F90_PassInt (max_num_levels),
                nalu_hypre_F90_PassIntArray (block_num_coarse_points),
                nalu_hypre_F90_PassIntArrayArray (block_coarse_indexes) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCptsByMarkerArray
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetcptsbymarkerarray, NALU_HYPRE_MGRSETCPTSBYMARKERARRAY)
( nalu_hypre_F90_Obj           *solver,
  nalu_hypre_F90_Int           *block_size,
  nalu_hypre_F90_Int           *max_num_levels,
  nalu_hypre_F90_IntArray      *num_block_coarse_points,
  nalu_hypre_F90_IntArrayArray *lvl_block_coarse_indexes,
  nalu_hypre_F90_IntArray      *point_marker_array,
  nalu_hypre_F90_Int           *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCpointsByPointMarkerArray(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (block_size),
                nalu_hypre_F90_PassInt (max_num_levels),
                nalu_hypre_F90_PassIntArray (num_block_coarse_points),
                nalu_hypre_F90_PassIntArrayArray (lvl_block_coarse_indexes),
                nalu_hypre_F90_PassIntArray (point_marker_array) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNonCptsToFpts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetnoncptstofpts, NALU_HYPRE_MGRSETNONCPTSTOFPTS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *nonCptToFptFlag,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetNonCpointsToFpoints(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (nonCptToFptFlag) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetFSolver
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetfsolver, NALU_HYPRE_MGRSETFSOLVER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *fsolver_id,
  nalu_hypre_F90_Obj *fsolver,
  nalu_hypre_F90_Int *ierr )
{
   /*------------------------------------------------------------
    * The fsolver_id flag means:
    *   0 - do not setup a F-solver.
    *   1 - BoomerAMG.
    *------------------------------------------------------------*/

   if (*fsolver_id == 0)
   {
      *ierr = 0;
   }
   else if (*fsolver_id == 1)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_MGRSetFSolver(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   (NALU_HYPRE_PtrToParSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                   (NALU_HYPRE_PtrToParSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                   (NALU_HYPRE_Solver) * fsolver) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRBuildAff
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrbuildaff, NALU_HYPRE_MGRBUILDAFF)
( nalu_hypre_F90_Obj      *A,
  nalu_hypre_F90_IntArray *CF_marker,
  nalu_hypre_F90_Int      *debug_flag,
  nalu_hypre_F90_Obj      *A_ff,
  nalu_hypre_F90_Int      *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRBuildAff(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassIntArray (CF_marker),
                nalu_hypre_F90_PassInt (debug_flag),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, A_ff) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCoarseSolver
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetcoarsesolver, NALU_HYPRE_MGRSETCOARSESOLVER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *csolver_id,
  nalu_hypre_F90_Obj *csolver,
  nalu_hypre_F90_Int *ierr )
{
   /*------------------------------------------------------------
    * The csolver_id flag means:
    *   0 - do not setup a coarse solver.
    *   1 - BoomerAMG.
    *------------------------------------------------------------*/

   if (*csolver_id == 0)
   {
      *ierr = 0;
   }
   else if (*csolver_id == 1)
   {
      *ierr = (nalu_hypre_F90_Int)
              ( NALU_HYPRE_MGRSetCoarseSolver(
                   nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                   (NALU_HYPRE_PtrToParSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                   (NALU_HYPRE_PtrToParSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                   (NALU_HYPRE_Solver) * csolver) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetMaxCoarseLevels
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetmaxcoarselevels, NALU_HYPRE_MGRSETMAXCOARSELEVELS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *maxlev,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetMaxCoarseLevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (maxlev) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetBlockSize
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetblocksize, NALU_HYPRE_MGRSETBLOCKSIZE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *bsize,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetBlockSize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (bsize) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetReservedCoarseNodes
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetreservedcoarsenodes, NALU_HYPRE_MGRSETRESERVEDCOARSENODES)
( nalu_hypre_F90_Obj         *solver,
  nalu_hypre_F90_Int         *reserved_coarse_size,
  nalu_hypre_F90_BigIntArray *reserved_coarse_indexes,
  nalu_hypre_F90_Int         *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetReservedCoarseNodes(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (reserved_coarse_size),
                nalu_hypre_F90_PassBigIntArray (reserved_coarse_indexes) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetReservedCptsLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetreservedcptslevel, NALU_HYPRE_MGRSETRESERVEDCPTSLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *level,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetReservedCpointsLevelToKeep(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetRestrictType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetrestricttype, NALU_HYPRE_MGRSETRESTRICTTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *restrict_type,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetRestrictType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (restrict_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelRestrictType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetlevelrestricttype, NALU_HYPRE_MGRSETLEVELRESTRICTTYPE)
( nalu_hypre_F90_Obj      *solver,
  nalu_hypre_F90_IntArray *restrict_type,
  nalu_hypre_F90_Int      *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLevelRestrictType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntArray (restrict_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetFRelaxMethod
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetfrelaxmethod, NALU_HYPRE_MGRSETFRELAXMETHOD)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_method,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetFRelaxMethod(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (relax_method) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelFRelaxMethod
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetlevelfrelaxmethod, NALU_HYPRE_MGRSETLEVELFRELAXMETHOD)
( nalu_hypre_F90_Obj      *solver,
  nalu_hypre_F90_IntArray *relax_method,
  nalu_hypre_F90_Int      *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLevelFRelaxMethod(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntArray (relax_method) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCoarseGridMethod
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetcoarsegridmethod, NALU_HYPRE_MGRSETCOARSEGRIDMETHOD)
( nalu_hypre_F90_Obj      *solver,
  nalu_hypre_F90_IntArray *cg_method,
  nalu_hypre_F90_Int      *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCoarseGridMethod(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntArray (cg_method) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelFRelaxNumFunc
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetlevelfrelaxnumfunc, NALU_HYPRE_MGRSETLEVELFRELAXNUMFUNC)
( nalu_hypre_F90_Obj      *solver,
  nalu_hypre_F90_IntArray *num_functions,
  nalu_hypre_F90_Int      *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLevelFRelaxNumFunctions(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntArray (num_functions) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetRelaxType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetrelaxtype, NALU_HYPRE_MGRSETRELAXTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *relax_type,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetRelaxType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNumRelaxSweeps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetnumrelaxsweeps, NALU_HYPRE_MGRSETNUMRELAXSWEEPS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *nsweeps,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetNumRelaxSweeps(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (nsweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetInterpType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetinterptype, NALU_HYPRE_MGRSETINTERPTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *interpType,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetInterpType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (interpType) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelInterpType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetlevelinterptype, NALU_HYPRE_MGRSETLEVELINTERPTYPE)
( nalu_hypre_F90_Obj      *solver,
  nalu_hypre_F90_IntArray *interpType,
  nalu_hypre_F90_Int      *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLevelInterpType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntArray (interpType) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNumInterpSweeps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetnuminterpsweeps, NALU_HYPRE_MGRSETNUMINTERPSWEEPS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *nsweeps,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetNumInterpSweeps(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (nsweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNumRestrictSweeps
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetnumrestrictsweeps, NALU_HYPRE_MGRSETNUMRESTRICTSWEEPS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *nsweeps,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetNumRestrictSweeps(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (nsweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCGridThreshold
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetcgridthreshold, NALU_HYPRE_MGRSETCGRIDTHRESHOLD)
( nalu_hypre_F90_Obj  *solver,
  nalu_hypre_F90_Real *threshold,
  nalu_hypre_F90_Int  *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetTruncateCoarseGridThreshold(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (threshold) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetFrelaxPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetfrelaxprintlevel, NALU_HYPRE_MGRSETFRELAXPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetFrelaxPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCgridPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetcgridprintlevel, NALU_HYPRE_MGRSETCGRIDPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCoarseGridPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetPrintLevel
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetprintlevel, NALU_HYPRE_MGRSETPRINTLEVEL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *print_level,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetPrintLevel(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetlogging, NALU_HYPRE_MGRSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetMaxIter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetmaxiter, NALU_HYPRE_MGRSETMAXITER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetMaxIter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetTol
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsettol, NALU_HYPRE_MGRSETTOL)
( nalu_hypre_F90_Obj  *solver,
  nalu_hypre_F90_Real *tol,
  nalu_hypre_F90_Int  *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetTol(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetMaxGlobalsmoothIt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetmaxglobalsmoothit, NALU_HYPRE_MGRSETMAXGLOBALSMOOTHIT)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *max_iter,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetMaxGlobalSmoothIters(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetGlobalsmoothType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetglobalsmoothtype, NALU_HYPRE_MGRSETGLOBALSMOOTHTYPE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *iter_type,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetGlobalSmoothType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (iter_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetPMaxElmts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrsetpmaxelmts, NALU_HYPRE_MGRSETPMAXELMTS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *P_max_elmts,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRSetPMaxElmts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (P_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRGetCoarseGridConvFac
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrgetcoarsegridconvfac, NALU_HYPRE_MGRGETCOARSEGRIDCONVFAC)
( nalu_hypre_F90_Obj  *solver,
  nalu_hypre_F90_Real *conv_factor,
  nalu_hypre_F90_Int  *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRGetCoarseGridConvergenceFactor(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (conv_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRGetNumIterations
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrgetnumiterations, NALU_HYPRE_MGRGETNUMITERATIONS)
( nalu_hypre_F90_Obj  *solver,
  nalu_hypre_F90_Int  *num_iterations,
  nalu_hypre_F90_Int  *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRGetNumIterations(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRGetFinalRelResNorm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_mgrgetfinalrelresnorm, NALU_HYPRE_MGRGETFINALRELRESNORM)
( nalu_hypre_F90_Obj   *solver,
  nalu_hypre_F90_Real  *res_norm,
  nalu_hypre_F90_Int   *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_MGRGetFinalRelativeResidualNorm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (res_norm) ) );
}

#ifdef __cplusplus
}
#endif
