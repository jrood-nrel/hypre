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

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRLSICG interface
 *
 *****************************************************************************/

extern void *nalu_hypre_LSICGCreate();
extern int  nalu_hypre_LSICGDestroy(void *);
extern int  nalu_hypre_LSICGSetup(void *, void *, void *, void *);
extern int  nalu_hypre_LSICGSolve(void *, void  *, void  *, void  *);
extern int  nalu_hypre_LSICGSetTol(void *, double);
extern int  nalu_hypre_LSICGSetMaxIter(void *, int);
extern int  nalu_hypre_LSICGSetStopCrit(void *, double);
extern int  nalu_hypre_LSICGSetPrecond(void *, int (*precond)(void*,void*,void*,void*),
                                  int (*precond_setup)(void*,void*,void*,void*), void *);
extern int  nalu_hypre_LSICGSetLogging(void *, int);
extern int  nalu_hypre_LSICGGetNumIterations(void *,int *);
extern int nalu_hypre_LSICGGetFinalRelativeResidualNorm(void *, double *);

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGCreate
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   *solver = (NALU_HYPRE_Solver) nalu_hypre_LSICGCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGDestroy
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGDestroy( NALU_HYPRE_Solver solver )
{
   return( nalu_hypre_LSICGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGSetup
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( nalu_hypre_LSICGSetup( (void *) solver, (void *) A, (void *) b,
                                 (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGSolve
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( nalu_hypre_LSICGSolve( (void *) solver, (void *) A,
                                 (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGSetTol
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGSetTol( NALU_HYPRE_Solver solver, double tol    )
{
   return( nalu_hypre_LSICGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGSetMaxIter
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGSetMaxIter( NALU_HYPRE_Solver solver, int max_iter )
{
   return( nalu_hypre_LSICGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGetStopCrit
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit )
{
   return( nalu_hypre_LSICGSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGSetPrecond
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void *precond_data )
{
   return( nalu_hypre_LSICGSetPrecond( (void *) solver,
								  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond,
								  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond_setup,
								  precond_data ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGSetLogging
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGSetLogging( NALU_HYPRE_Solver solver, int logging)
{
   return( nalu_hypre_LSICGSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGetNumIterations
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGGetNumIterations(NALU_HYPRE_Solver solver,int *num_iterations)
{
   return( nalu_hypre_LSICGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLSICGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRLSICGGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( nalu_hypre_LSICGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

