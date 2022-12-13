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

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACCreate( MPI_Comm comm, NALU_HYPRE_SStructSolver *solver )
{
   *solver = ( (NALU_HYPRE_SStructSolver) hypre_FACCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACDestroy2( NALU_HYPRE_SStructSolver solver )
{
   return ( hypre_FACDestroy2( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACAMR_RAP
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACAMR_RAP( NALU_HYPRE_SStructMatrix  A,
                         NALU_HYPRE_Int          (*rfactors)[NALU_HYPRE_MAXDIM],
                         NALU_HYPRE_SStructMatrix *fac_A )
{
   return ( hypre_AMR_RAP(A, rfactors, fac_A) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetup2
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetup2( NALU_HYPRE_SStructSolver  solver,
                        NALU_HYPRE_SStructMatrix  A,
                        NALU_HYPRE_SStructVector  b,
                        NALU_HYPRE_SStructVector  x )
{
   return ( hypre_FacSetup2( (void *) solver,
                             (hypre_SStructMatrix *)  A,
                             (hypre_SStructVector *)  b,
                             (hypre_SStructVector *)  x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSolve3
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSolve3(NALU_HYPRE_SStructSolver solver,
                       NALU_HYPRE_SStructMatrix A,
                       NALU_HYPRE_SStructVector b,
                       NALU_HYPRE_SStructVector x)
{
   return ( hypre_FACSolve3((void *) solver,
                            (hypre_SStructMatrix *)  A,
                            (hypre_SStructVector *)  b,
                            (hypre_SStructVector *)  x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetTol( NALU_HYPRE_SStructSolver solver,
                        NALU_HYPRE_Real         tol    )
{
   return ( hypre_FACSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetPLevels( NALU_HYPRE_SStructSolver  solver,
                            NALU_HYPRE_Int            nparts,
                            NALU_HYPRE_Int           *plevels)
{
   return ( hypre_FACSetPLevels( (void *) solver, nparts, plevels ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroCFSten
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACZeroCFSten( NALU_HYPRE_SStructMatrix  A,
                            NALU_HYPRE_SStructGrid    grid,
                            NALU_HYPRE_Int            part,
                            NALU_HYPRE_Int            rfactors[NALU_HYPRE_MAXDIM] )
{
   hypre_SStructPMatrix   *Af = hypre_SStructMatrixPMatrix(A, part);
   hypre_SStructPMatrix   *Ac = hypre_SStructMatrixPMatrix(A, part - 1);

   return ( hypre_FacZeroCFSten(Af, Ac, (hypre_SStructGrid *)grid,
                                part, rfactors) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroFCSten
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACZeroFCSten( NALU_HYPRE_SStructMatrix  A,
                            NALU_HYPRE_SStructGrid    grid,
                            NALU_HYPRE_Int            part )
{
   hypre_SStructPMatrix   *Af = hypre_SStructMatrixPMatrix(A, part);

   return ( hypre_FacZeroFCSten(Af, (hypre_SStructGrid *)grid,
                                part) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroAMRMatrixData
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACZeroAMRMatrixData( NALU_HYPRE_SStructMatrix  A,
                                   NALU_HYPRE_Int            part_crse,
                                   NALU_HYPRE_Int            rfactors[NALU_HYPRE_MAXDIM] )
{
   return ( hypre_ZeroAMRMatrixData(A, part_crse, rfactors) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroAMRVectorData
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACZeroAMRVectorData( NALU_HYPRE_SStructVector  b,
                                   NALU_HYPRE_Int           *plevels,
                                   NALU_HYPRE_Int          (*rfactors)[NALU_HYPRE_MAXDIM] )
{
   return ( hypre_ZeroAMRVectorData(b, plevels, rfactors) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetPRefinements( NALU_HYPRE_SStructSolver  solver,
                                 NALU_HYPRE_Int            nparts,
                                 NALU_HYPRE_Int          (*rfactors)[NALU_HYPRE_MAXDIM] )
{
   return ( hypre_FACSetPRefinements( (void *)         solver,
                                      nparts,
                                      rfactors ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetMaxLevels( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           max_levels  )
{
   return ( hypre_FACSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetMaxIter( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Int          max_iter  )
{
   return ( hypre_FACSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetRelChange( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int          rel_change  )
{
   return ( hypre_FACSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   return ( hypre_FACSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetNonZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   return ( hypre_FACSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetRelaxType( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int          relax_type )
{
   return ( hypre_FACSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetJacobiWeight
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetJacobiWeight( NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Real          weight)
{
   return ( hypre_FACSetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetNumPreRelax( NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int          num_pre_relax )
{
   return ( hypre_FACSetNumPreSmooth( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetNumPostRelax( NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int          num_post_relax )
{
   return ( hypre_FACSetNumPostSmooth( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetCoarseSolverType( NALU_HYPRE_SStructSolver solver,
                                     NALU_HYPRE_Int           csolver_type)
{
   return ( hypre_FACSetCoarseSolverType( (void *) solver, csolver_type) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetLogging( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Int          logging )
{
   return ( hypre_FACSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                  NALU_HYPRE_Int          *num_iterations )
{
   return ( hypre_FACGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                              NALU_HYPRE_Real         *norm   )
{
   return ( hypre_FACGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


