/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructMaxwell interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellCreate(MPI_Comm comm, NALU_HYPRE_SStructSolver *solver)
{
   *solver = ( (NALU_HYPRE_SStructSolver) hypre_MaxwellTVCreate(comm) );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellDestroy(NALU_HYPRE_SStructSolver solver)
{
   return ( hypre_MaxwellTVDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetup( NALU_HYPRE_SStructSolver  solver,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector b,
                           NALU_HYPRE_SStructVector x )
{
   return ( hypre_MaxwellTV_Setup( (void *) solver,
                                   (hypre_SStructMatrix *) A,
                                   (hypre_SStructVector *) b,
                                   (hypre_SStructVector *) x ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSolve( NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector b,
                           NALU_HYPRE_SStructVector x      )
{
   return ( hypre_MaxwellSolve( (void *) solver,
                                (hypre_SStructMatrix *) A,
                                (hypre_SStructVector *) b,
                                (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSolve2
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSolve2( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_SStructMatrix A,
                            NALU_HYPRE_SStructVector b,
                            NALU_HYPRE_SStructVector x      )
{
   return ( hypre_MaxwellSolve2( (void *) solver,
                                 (hypre_SStructMatrix *) A,
                                 (hypre_SStructVector *) b,
                                 (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MaxwellGrad
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_MaxwellGrad( NALU_HYPRE_SStructGrid   grid,
                   NALU_HYPRE_ParCSRMatrix *T )

{
   *T = ( (NALU_HYPRE_ParCSRMatrix) hypre_Maxwell_Grad( (hypre_SStructGrid *) grid));
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetGrad
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetGrad( NALU_HYPRE_SStructSolver  solver,
                             NALU_HYPRE_ParCSRMatrix   T )
{
   return ( hypre_MaxwellSetGrad( (void *)               solver,
                                  (hypre_ParCSRMatrix *) T) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetRfactors
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetRfactors( NALU_HYPRE_SStructSolver  solver,
                                 NALU_HYPRE_Int            rfactors[3] )
{
   return ( hypre_MaxwellSetRfactors( (void *)         solver,
                                      rfactors ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetTol
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetTol( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Real          tol    )
{
   return ( hypre_MaxwellSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetConstantCoef( NALU_HYPRE_SStructSolver solver,
                                     NALU_HYPRE_Int           constant_coef)
{
   return ( hypre_MaxwellSetConstantCoef( (void *) solver, constant_coef) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetMaxIter
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetMaxIter( NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int           max_iter  )
{
   return ( hypre_MaxwellSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetRelChange
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetRelChange( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int           rel_change  )
{
   return ( hypre_MaxwellSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetNumPreRelax
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetNumPreRelax( NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Int           num_pre_relax )
{
   return ( hypre_MaxwellSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetNumPostRelax( NALU_HYPRE_SStructSolver solver,
                                     NALU_HYPRE_Int           num_post_relax )
{
   return ( hypre_MaxwellSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellSetLogging
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetLogging( NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int           logging )
{
   return ( hypre_MaxwellSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
NALU_HYPRE_SStructMaxwellSetPrintLevel
*--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetPrintLevel( NALU_HYPRE_SStructSolver solver,
                                   NALU_HYPRE_Int           print_level )
{
   return ( hypre_MaxwellSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellPrintLogging
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellPrintLogging( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int           myid)
{
   return ( hypre_MaxwellPrintLogging( (void *) solver, myid) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellGetNumIterations
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                      NALU_HYPRE_Int           *num_iterations )
{
   return ( hypre_MaxwellGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                                  NALU_HYPRE_Real          *norm   )
{
   return ( hypre_MaxwellGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellPhysBdy
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellPhysBdy( NALU_HYPRE_SStructGrid  *grid_l,
                             NALU_HYPRE_Int           num_levels,
                             NALU_HYPRE_Int           rfactors[3],
                             NALU_HYPRE_Int        ***BdryRanks_ptr,
                             NALU_HYPRE_Int         **BdryRanksCnt_ptr )
{
   return ( hypre_Maxwell_PhysBdy( (hypre_SStructGrid  **) grid_l,
                                   num_levels,
                                   rfactors,
                                   BdryRanks_ptr,
                                   BdryRanksCnt_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellEliminateRowsCols
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellEliminateRowsCols( NALU_HYPRE_ParCSRMatrix  parA,
                                       NALU_HYPRE_Int           nrows,
                                       NALU_HYPRE_Int          *rows )
{
   return ( hypre_ParCSRMatrixEliminateRowsCols( (hypre_ParCSRMatrix *) parA,
                                                 nrows,
                                                 rows ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMaxwellZeroVector
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int NALU_HYPRE_SStructMaxwellZeroVector(NALU_HYPRE_ParVector  v,
                                         NALU_HYPRE_Int       *rows,
                                         NALU_HYPRE_Int        nrows)
{
   return ( hypre_ParVectorZeroBCValues( (hypre_ParVector *) v,
                                         rows,
                                         nrows ) );
}


