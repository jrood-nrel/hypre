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

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACCreate( MPI_Comm comm, NALU_HYPRE_SStructSolver *solver )
{
   *solver = ( (NALU_HYPRE_SStructSolver) nalu_hypre_FACCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACDestroy2( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_FACDestroy2( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACAMR_RAP
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACAMR_RAP( NALU_HYPRE_SStructMatrix  A,
                         NALU_HYPRE_Int          (*rfactors)[NALU_HYPRE_MAXDIM],
                         NALU_HYPRE_SStructMatrix *fac_A )
{
   return ( nalu_hypre_AMR_RAP(A, rfactors, fac_A) );
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
   return ( nalu_hypre_FacSetup2( (void *) solver,
                             (nalu_hypre_SStructMatrix *)  A,
                             (nalu_hypre_SStructVector *)  b,
                             (nalu_hypre_SStructVector *)  x ) );
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
   return ( nalu_hypre_FACSolve3((void *) solver,
                            (nalu_hypre_SStructMatrix *)  A,
                            (nalu_hypre_SStructVector *)  b,
                            (nalu_hypre_SStructVector *)  x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetTol( NALU_HYPRE_SStructSolver solver,
                        NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_FACSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetPLevels( NALU_HYPRE_SStructSolver  solver,
                            NALU_HYPRE_Int            nparts,
                            NALU_HYPRE_Int           *plevels)
{
   return ( nalu_hypre_FACSetPLevels( (void *) solver, nparts, plevels ) );
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
   nalu_hypre_SStructPMatrix   *Af = nalu_hypre_SStructMatrixPMatrix(A, part);
   nalu_hypre_SStructPMatrix   *Ac = nalu_hypre_SStructMatrixPMatrix(A, part - 1);

   return ( nalu_hypre_FacZeroCFSten(Af, Ac, (nalu_hypre_SStructGrid *)grid,
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
   nalu_hypre_SStructPMatrix   *Af = nalu_hypre_SStructMatrixPMatrix(A, part);

   return ( nalu_hypre_FacZeroFCSten(Af, (nalu_hypre_SStructGrid *)grid,
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
   return ( nalu_hypre_ZeroAMRMatrixData(A, part_crse, rfactors) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACZeroAMRVectorData
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACZeroAMRVectorData( NALU_HYPRE_SStructVector  b,
                                   NALU_HYPRE_Int           *plevels,
                                   NALU_HYPRE_Int          (*rfactors)[NALU_HYPRE_MAXDIM] )
{
   return ( nalu_hypre_ZeroAMRVectorData(b, plevels, rfactors) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetPRefinements( NALU_HYPRE_SStructSolver  solver,
                                 NALU_HYPRE_Int            nparts,
                                 NALU_HYPRE_Int          (*rfactors)[NALU_HYPRE_MAXDIM] )
{
   return ( nalu_hypre_FACSetPRefinements( (void *)         solver,
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
   return ( nalu_hypre_FACSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetMaxIter( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Int          max_iter  )
{
   return ( nalu_hypre_FACSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetRelChange( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int          rel_change  )
{
   return ( nalu_hypre_FACSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_FACSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetNonZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_FACSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetRelaxType( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int          relax_type )
{
   return ( nalu_hypre_FACSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetJacobiWeight
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetJacobiWeight( NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Real          weight)
{
   return ( nalu_hypre_FACSetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetNumPreRelax( NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int          num_pre_relax )
{
   return ( nalu_hypre_FACSetNumPreSmooth( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetNumPostRelax( NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int          num_post_relax )
{
   return ( nalu_hypre_FACSetNumPostSmooth( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetCoarseSolverType( NALU_HYPRE_SStructSolver solver,
                                     NALU_HYPRE_Int           csolver_type)
{
   return ( nalu_hypre_FACSetCoarseSolverType( (void *) solver, csolver_type) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetLogging( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Int          logging )
{
   return ( nalu_hypre_FACSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                  NALU_HYPRE_Int          *num_iterations )
{
   return ( nalu_hypre_FACGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                              NALU_HYPRE_Real         *norm   )
{
   return ( nalu_hypre_FACGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


