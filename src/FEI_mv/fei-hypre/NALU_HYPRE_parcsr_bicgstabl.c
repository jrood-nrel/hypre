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
 * NALU_HYPRE_ParCSRBiCGSTABL interface
 *
 *****************************************************************************/

extern void *hypre_BiCGSTABLCreate();
extern int  hypre_BiCGSTABLDestroy(void *);
extern int  hypre_BiCGSTABLSetup(void *, void *, void *, void *);
extern int  hypre_BiCGSTABLSolve(void *, void *, void *, void *);
extern int  hypre_BiCGSTABLSetTol(void *, double);
extern int  hypre_BiCGSTABLSetSize(void *, int);
extern int  hypre_BiCGSTABLSetMaxIter(void *, int);
extern int  hypre_BiCGSTABLSetStopCrit(void *, double);
extern int  hypre_BiCGSTABLSetPrecond(void *, int (*precond)(void*,void*,void*,void*),
									  int (*precond_setup)(void*,void*,void*,void*), void *);
extern int  hypre_BiCGSTABLSetLogging(void *, int);
extern int  hypre_BiCGSTABLGetNumIterations(void *,int *);
extern int  hypre_BiCGSTABLGetFinalRelativeResidualNorm(void *, double *);

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLCreate
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   *solver = (NALU_HYPRE_Solver) hypre_BiCGSTABLCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLDestroy
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLDestroy( NALU_HYPRE_Solver solver )
{
   return( hypre_BiCGSTABLDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetup
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( hypre_BiCGSTABLSetup( (void *) solver, (void *) A, (void *) b,
                                 (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSolve
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( hypre_BiCGSTABLSolve( (void *) solver, (void *) A,
                                 (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetTol
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSetTol( NALU_HYPRE_Solver solver, double tol    )
{
   return( hypre_BiCGSTABLSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetSize
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSetSize( NALU_HYPRE_Solver solver, int size )
{
   return( hypre_BiCGSTABLSetSize( (void *) solver, size ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetMaxIter
 *--------------------------------------------------------------------------*/

int
NALU_HYPRE_ParCSRBiCGSTABLSetMaxIter( NALU_HYPRE_Solver solver, int max_iter )
{
   return( hypre_BiCGSTABLSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

int
NALU_HYPRE_ParCSRBiCGSTABLSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit )
{
   return( hypre_BiCGSTABLSetStopCrit( (void *) solver, stop_crit ) );
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
   return( hypre_BiCGSTABLSetPrecond( (void *) solver,
									  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond,
									  (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond_setup,
									  precond_data ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLSetLogging
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLSetLogging( NALU_HYPRE_Solver solver, int logging)
{
   return( hypre_BiCGSTABLSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLGetNumIterations(NALU_HYPRE_Solver solver,int *num_iterations)
{
   return( hypre_BiCGSTABLGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( hypre_BiCGSTABLGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

