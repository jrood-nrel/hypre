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
 * NALU_HYPRE_ParCSRSymQMR interface
 *
 *****************************************************************************/

extern void *nalu_hypre_SymQMRCreate();
extern int  nalu_hypre_SymQMRDestroy(void *);
extern int  nalu_hypre_SymQMRSetup(void *, void *, void *, void *);
extern int  nalu_hypre_SymQMRSolve(void *, void *, void *, void *);
extern int  nalu_hypre_SymQMRSetTol(void *, double);
extern int  nalu_hypre_SymQMRSetMaxIter(void *, int);
extern int  nalu_hypre_SymQMRSetStopCrit(void *, double);
extern int  nalu_hypre_SymQMRSetPrecond(void *, int (*precond)(void*,void*,void*,void*),
                                   int (*precond_setup)(void*,void*,void*,void*), void *);
extern int  nalu_hypre_SymQMRSetLogging(void *, int );
extern int  nalu_hypre_SymQMRGetNumIterations(void *, int *);
extern int  nalu_hypre_SymQMRGetFinalRelativeResidualNorm(void *, double *);

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRCreate
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   *solver = (NALU_HYPRE_Solver) nalu_hypre_SymQMRCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRDestroy
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRDestroy( NALU_HYPRE_Solver solver )
{
   return( nalu_hypre_SymQMRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRSetup
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                             NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( nalu_hypre_SymQMRSetup( (void *) solver, (void *) A, (void *) b,
                              (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRSolve
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( nalu_hypre_SymQMRSolve( (void *) solver, (void *) A,
                              (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRSetTol
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRSetTol( NALU_HYPRE_Solver solver, double tol    )
{
   return( nalu_hypre_SymQMRSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRSetMaxIter
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRSetMaxIter( NALU_HYPRE_Solver solver, int max_iter )
{
   return( nalu_hypre_SymQMRSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRSetStopCrit
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit )
{
   return( nalu_hypre_SymQMRSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRSetPrecond
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void                *precond_data )
{
   return( nalu_hypre_SymQMRSetPrecond( (void *) solver,
								   (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond,
								   (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond_setup,
								   precond_data ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRSetLogging
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRSetLogging( NALU_HYPRE_Solver solver, int logging)
{
   return( nalu_hypre_SymQMRSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRetNumIterations
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRGetNumIterations(NALU_HYPRE_Solver solver,int *num_iterations)
{
   return( nalu_hypre_SymQMRGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymQMRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRSymQMRGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( nalu_hypre_SymQMRGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

