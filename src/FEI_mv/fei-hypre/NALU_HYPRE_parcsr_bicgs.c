/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"

#include "NALU_HYPRE_FEI.h"
#include "_nalu_hypre_FEI.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSCreate
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   *solver = (NALU_HYPRE_Solver) nalu_hypre_BiCGSCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSDestroy
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSDestroy( NALU_HYPRE_Solver solver )
{
   return( nalu_hypre_BiCGSDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSSetup
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( nalu_hypre_BiCGSSetup( (void *) solver, (void *) A, (void *) b,
                                 (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSSolve
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( nalu_hypre_BiCGSSolve( (void *) solver, (void *) A,
                                 (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSSetTol
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSSetTol( NALU_HYPRE_Solver solver, double tol    )
{
   return( nalu_hypre_BiCGSSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSSetMaxIter
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSSetMaxIter( NALU_HYPRE_Solver solver, int max_iter )
{
   return( nalu_hypre_BiCGSSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSetStopCrit
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit )
{
   return( nalu_hypre_BiCGSSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSSetPrecond
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void               *precond_data )
{
   return( nalu_hypre_BiCGSSetPrecond( (void *) solver,
								  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond,
								  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond_setup,
								  precond_data ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSSetLogging
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSSetLogging( NALU_HYPRE_Solver solver, int logging)
{
   return( nalu_hypre_BiCGSSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSetNumIterations
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSGetNumIterations(NALU_HYPRE_Solver solver,int *num_iterations)
{
   return( nalu_hypre_BiCGSGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( nalu_hypre_BiCGSGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

