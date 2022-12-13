/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_MGR Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrcreate, NALU_HYPRE_MGRCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRCreate(
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdestroy, NALU_HYPRE_MGRDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetup, NALU_HYPRE_MGRSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsolve, NALU_HYPRE_MGRSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

#ifdef NALU_HYPRE_USING_DSUPERLU

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdirectsolvercreate, NALU_HYPRE_MGRDIRECTSOLVERCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRDirectSolverCreate(
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdirectsolverdestroy, NALU_HYPRE_MGRDIRECTSOLVERDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRDirectSolverDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdirectsolversetup, NALU_HYPRE_MGRDIRECTSOLVERSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRDirectSolverSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdirectsolversolve, NALU_HYPRE_MGRDIRECTSOLVERSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRDirectSolverSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCptsByCtgBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcptsbyctgblock, NALU_HYPRE_MGRSETCPTSBYCTGBLOCK)
( hypre_F90_Obj           *solver,
  hypre_F90_Int           *block_size,
  hypre_F90_Int           *max_num_levels,
  hypre_F90_BigIntArray   *idx_array,
  hypre_F90_IntArray      *block_num_coarse_points,
  hypre_F90_IntArrayArray *block_coarse_indexes,
  hypre_F90_Int           *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCpointsByContiguousBlock(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (block_size),
                hypre_F90_PassInt (max_num_levels),
                hypre_F90_PassBigIntArray (idx_array),
                hypre_F90_PassIntArray (block_num_coarse_points),
                hypre_F90_PassIntArrayArray (block_coarse_indexes) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCpointsByBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcpointsbyblock, NALU_HYPRE_MGRSETCPOINTSBYBLOCK)
( hypre_F90_Obj           *solver,
  hypre_F90_Int           *block_size,
  hypre_F90_Int           *max_num_levels,
  hypre_F90_IntArray      *block_num_coarse_points,
  hypre_F90_IntArrayArray *block_coarse_indexes,
  hypre_F90_Int           *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCpointsByBlock(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (block_size),
                hypre_F90_PassInt (max_num_levels),
                hypre_F90_PassIntArray (block_num_coarse_points),
                hypre_F90_PassIntArrayArray (block_coarse_indexes) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCptsByMarkerArray
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcptsbymarkerarray, NALU_HYPRE_MGRSETCPTSBYMARKERARRAY)
( hypre_F90_Obj           *solver,
  hypre_F90_Int           *block_size,
  hypre_F90_Int           *max_num_levels,
  hypre_F90_IntArray      *num_block_coarse_points,
  hypre_F90_IntArrayArray *lvl_block_coarse_indexes,
  hypre_F90_IntArray      *point_marker_array,
  hypre_F90_Int           *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCpointsByPointMarkerArray(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (block_size),
                hypre_F90_PassInt (max_num_levels),
                hypre_F90_PassIntArray (num_block_coarse_points),
                hypre_F90_PassIntArrayArray (lvl_block_coarse_indexes),
                hypre_F90_PassIntArray (point_marker_array) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNonCptsToFpts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetnoncptstofpts, NALU_HYPRE_MGRSETNONCPTSTOFPTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nonCptToFptFlag,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetNonCpointsToFpoints(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (nonCptToFptFlag) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetFSolver
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetfsolver, NALU_HYPRE_MGRSETFSOLVER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *fsolver_id,
  hypre_F90_Obj *fsolver,
  hypre_F90_Int *ierr )
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
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_MGRSetFSolver(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
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
hypre_F90_IFACE(hypre_mgrbuildaff, NALU_HYPRE_MGRBUILDAFF)
( hypre_F90_Obj      *A,
  hypre_F90_IntArray *CF_marker,
  hypre_F90_Int      *debug_flag,
  hypre_F90_Obj      *A_ff,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRBuildAff(
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassIntArray (CF_marker),
                hypre_F90_PassInt (debug_flag),
                hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, A_ff) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCoarseSolver
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcoarsesolver, NALU_HYPRE_MGRSETCOARSESOLVER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *csolver_id,
  hypre_F90_Obj *csolver,
  hypre_F90_Int *ierr )
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
      *ierr = (hypre_F90_Int)
              ( NALU_HYPRE_MGRSetCoarseSolver(
                   hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
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
hypre_F90_IFACE(hypre_mgrsetmaxcoarselevels, NALU_HYPRE_MGRSETMAXCOARSELEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *maxlev,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetMaxCoarseLevels(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (maxlev) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetBlockSize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetblocksize, NALU_HYPRE_MGRSETBLOCKSIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *bsize,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetBlockSize(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (bsize) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetReservedCoarseNodes
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetreservedcoarsenodes, NALU_HYPRE_MGRSETRESERVEDCOARSENODES)
( hypre_F90_Obj         *solver,
  hypre_F90_Int         *reserved_coarse_size,
  hypre_F90_BigIntArray *reserved_coarse_indexes,
  hypre_F90_Int         *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetReservedCoarseNodes(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (reserved_coarse_size),
                hypre_F90_PassBigIntArray (reserved_coarse_indexes) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetReservedCptsLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetreservedcptslevel, NALU_HYPRE_MGRSETRESERVEDCPTSLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetReservedCpointsLevelToKeep(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetRestrictType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetrestricttype, NALU_HYPRE_MGRSETRESTRICTTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *restrict_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetRestrictType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (restrict_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelRestrictType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlevelrestricttype, NALU_HYPRE_MGRSETLEVELRESTRICTTYPE)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *restrict_type,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLevelRestrictType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntArray (restrict_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetFRelaxMethod
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetfrelaxmethod, NALU_HYPRE_MGRSETFRELAXMETHOD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_method,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetFRelaxMethod(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_method) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelFRelaxMethod
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlevelfrelaxmethod, NALU_HYPRE_MGRSETLEVELFRELAXMETHOD)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *relax_method,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLevelFRelaxMethod(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntArray (relax_method) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCoarseGridMethod
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcoarsegridmethod, NALU_HYPRE_MGRSETCOARSEGRIDMETHOD)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *cg_method,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCoarseGridMethod(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntArray (cg_method) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelFRelaxNumFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlevelfrelaxnumfunc, NALU_HYPRE_MGRSETLEVELFRELAXNUMFUNC)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *num_functions,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLevelFRelaxNumFunctions(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntArray (num_functions) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetrelaxtype, NALU_HYPRE_MGRSETRELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetRelaxType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNumRelaxSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetnumrelaxsweeps, NALU_HYPRE_MGRSETNUMRELAXSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nsweeps,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetNumRelaxSweeps(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (nsweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetinterptype, NALU_HYPRE_MGRSETINTERPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *interpType,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetInterpType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (interpType) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlevelinterptype, NALU_HYPRE_MGRSETLEVELINTERPTYPE)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *interpType,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLevelInterpType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntArray (interpType) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNumInterpSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetnuminterpsweeps, NALU_HYPRE_MGRSETNUMINTERPSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nsweeps,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetNumInterpSweeps(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (nsweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNumRestrictSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetnumrestrictsweeps, NALU_HYPRE_MGRSETNUMRESTRICTSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nsweeps,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetNumRestrictSweeps(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (nsweeps) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCGridThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcgridthreshold, NALU_HYPRE_MGRSETCGRIDTHRESHOLD)
( hypre_F90_Obj  *solver,
  hypre_F90_Real *threshold,
  hypre_F90_Int  *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetTruncateCoarseGridThreshold(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (threshold) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetFrelaxPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetfrelaxprintlevel, NALU_HYPRE_MGRSETFRELAXPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetFrelaxPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCgridPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcgridprintlevel, NALU_HYPRE_MGRSETCGRIDPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetCoarseGridPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetprintlevel, NALU_HYPRE_MGRSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetPrintLevel(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlogging, NALU_HYPRE_MGRSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetmaxiter, NALU_HYPRE_MGRSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetMaxIter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsettol, NALU_HYPRE_MGRSETTOL)
( hypre_F90_Obj  *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int  *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetTol(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetMaxGlobalsmoothIt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetmaxglobalsmoothit, NALU_HYPRE_MGRSETMAXGLOBALSMOOTHIT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetMaxGlobalSmoothIters(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetGlobalsmoothType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetglobalsmoothtype, NALU_HYPRE_MGRSETGLOBALSMOOTHTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *iter_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetGlobalSmoothType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (iter_type) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetpmaxelmts, NALU_HYPRE_MGRSETPMAXELMTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *P_max_elmts,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRSetPMaxElmts(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (P_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRGetCoarseGridConvFac
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrgetcoarsegridconvfac, NALU_HYPRE_MGRGETCOARSEGRIDCONVFAC)
( hypre_F90_Obj  *solver,
  hypre_F90_Real *conv_factor,
  hypre_F90_Int  *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRGetCoarseGridConvergenceFactor(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (conv_factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrgetnumiterations, NALU_HYPRE_MGRGETNUMITERATIONS)
( hypre_F90_Obj  *solver,
  hypre_F90_Int  *num_iterations,
  hypre_F90_Int  *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRGetNumIterations(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRGetFinalRelResNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrgetfinalrelresnorm, NALU_HYPRE_MGRGETFINALRELRESNORM)
( hypre_F90_Obj   *solver,
  hypre_F90_Real  *res_norm,
  hypre_F90_Int   *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_MGRGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (res_norm) ) );
}

#ifdef __cplusplus
}
#endif
