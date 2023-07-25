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

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRBiCGSTABL interface
 *
 *****************************************************************************/

extern void *nalu_hypre_BiCGSTABLCreate();
extern int  nalu_hypre_BiCGSTABLDestroy(void *);
extern int  nalu_hypre_BiCGSTABLSetup(void *, void *, void *, void *);
extern int  nalu_hypre_BiCGSTABLSolve(void *, void *, void *, void *);
extern int  nalu_hypre_BiCGSTABLSetTol(void *, double);
extern int  nalu_hypre_BiCGSTABLSetSize(void *, int);
extern int  nalu_hypre_BiCGSTABLSetMaxIter(void *, int);
extern int  nalu_hypre_BiCGSTABLSetStopCrit(void *, double);
extern int  nalu_hypre_BiCGSTABLSetPrecond(void *, int (*precond)(void*,void*,void*,void*),
									  int (*precond_setup)(void*,void*,void*,void*), void *);
extern int  nalu_hypre_BiCGSTABLSetLogging(void *, int);
extern int  nalu_hypre_BiCGSTABLGetNumIterations(void *,int *);
extern int  nalu_hypre_BiCGSTABLGetFinalRelativeResidualNorm(void *, double *);

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLCreate
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   *solver = (NALU_HYPRE_Solver) nalu_hypre_BiCGSTABLCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLDestroy
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLDestroy( NALU_HYPRE_Solver solver )
{
   return( nalu_hypre_BiCGSTABLDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetup
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( nalu_hypre_BiCGSTABLSetup( (void *) solver, (void *) A, (void *) b,
                                 (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSolve
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( nalu_hypre_BiCGSTABLSolve( (void *) solver, (void *) A,
                                 (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetTol
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSetTol( NALU_HYPRE_Solver solver, double tol    )
{
   return( nalu_hypre_BiCGSTABLSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetSize
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSetSize( NALU_HYPRE_Solver solver, int size )
{
   return( nalu_hypre_BiCGSTABLSetSize( (void *) solver, size ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetMaxIter
 *--------------------------------------------------------------------------*/

int
NALU_HYPRE_ParCSRBiCGSTABLSetMaxIter( NALU_HYPRE_Solver solver, int max_iter )
{
   return( nalu_hypre_BiCGSTABLSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

int
NALU_HYPRE_ParCSRBiCGSTABLSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit )
{
   return( nalu_hypre_BiCGSTABLSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetPrecond
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void               *precond_data )
{
   return( nalu_hypre_BiCGSTABLSetPrecond( (void *) solver,
									  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond,
									  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond_setup,
									  precond_data ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetLogging
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSetLogging( NALU_HYPRE_Solver solver, int logging)
{
   return( nalu_hypre_BiCGSTABLSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLGetNumIterations(NALU_HYPRE_Solver solver,int *num_iterations)
{
   return( nalu_hypre_BiCGSTABLGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( nalu_hypre_BiCGSTABLGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

