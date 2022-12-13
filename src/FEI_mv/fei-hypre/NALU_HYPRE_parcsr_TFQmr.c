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

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"

#include "NALU_HYPRE_FEI.h"
#include "_hypre_FEI.h"

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRTFQmr interface
 *
 *****************************************************************************/

//extern void *hypre_TFQmrCreate();
//extern int hypre_TFQmrDestroy(void *);
//extern int hypre_TFQmrSetup(void *, void *, void *, void *);
//extern int hypre_TFQmrSolve(void *, void *, void *, void *);
//extern int hypre_TFQmrSetTol(void *, double);
//extern int hypre_TFQmrSetMaxIter(void *, int);
//extern int hypre_TFQmrSetStopCrit(void *, int);
//extern int hypre_TFQmrSetPrecond(void *, int (*precond)(void*,void*,void*,void*),
//                                 int (*precond_setup)(void*,void*,void*,void*), void *);
//extern int hypre_TFQmrSetLogging(void *, int);
//extern int hypre_TFQmrGetNumIterations(void *, int *);
//extern int hypre_TFQmrGetFinalRelativeResidualNorm(void *, double *);

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmrCreate
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   *solver = (NALU_HYPRE_Solver) hypre_TFQmrCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmrDestroy
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrDestroy( NALU_HYPRE_Solver solver )
{
   return( hypre_TFQmrDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmrSetup
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( hypre_TFQmrSetup( (void *) solver, (void *) A, (void *) b,
                                 (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmrSolve
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( hypre_TFQmrSolve( (void *) solver, (void *) A,
                                 (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmrSetTol
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrSetTol( NALU_HYPRE_Solver solver, double tol )
{
   return( hypre_TFQmrSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmrSetMaxIter
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrSetMaxIter( NALU_HYPRE_Solver solver, int max_iter )
{
   return( hypre_TFQmrSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmretStopCrit
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit )
{
   return( hypre_TFQmrSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmrSetPrecond
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void               *precond_data )
{
   return( hypre_TFQmrSetPrecond( (void *) solver,
								  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond,
								  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond_setup,
								  precond_data ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmrSetLogging
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrSetLogging( NALU_HYPRE_Solver solver, int logging)
{
   return( hypre_TFQmrSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmretNumIterations
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrGetNumIterations(NALU_HYPRE_Solver solver,int *num_iterations)
{
   return( hypre_TFQmrGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRTFQmrGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRTFQmrGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( hypre_TFQmrGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

