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
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"
#include "NALU_HYPRE_parcsr_fgmres.h"

#include "NALU_HYPRE_FEI.h"
#include "_hypre_FEI.h"

//extern void *hypre_FGMRESCreate();
//extern int  hypre_FGMRESDestroy(void *);
//extern int  hypre_FGMRESSetup(void *, void *, void *, void *);
//extern int  hypre_FGMRESSolve(void *, void *, void *, void *);
//extern int  hypre_FGMRESSetKDim(void *, int);
//extern int  hypre_FGMRESSetTol(void *, double);
//extern int  hypre_FGMRESSetMaxIter(void *, int);
//extern int  hypre_FGMRESSetStopCrit(void *, double);
//extern int  hypre_FGMRESSetPrecond(void *, int (*precond)(void*,void*,void*,void*), 
//                                 int (*precond_setup)(void*,void*,void*,void*),void *precond_data);
//extern int  hypre_FGMRESGetPrecond(void *, NALU_HYPRE_Solver *);
//extern int  hypre_FGMRESSetLogging(void *, int);
//extern int  hypre_FGMRESGetNumIterations(void *, int *);
//extern int  hypre_FGMRESGetFinalRelativeResidualNorm(void *,double *);
//extern int  hypre_FGMRESUpdatePrecondTolerance(void *, int (*update_tol)(NALU_HYPRE_Solver,double));

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRFGMRES interface
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESCreate
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   *solver = (NALU_HYPRE_Solver) hypre_FGMRESCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESDestroy
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return( hypre_FGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESSetup
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                             NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( hypre_FGMRESSetup( (void *) solver, (void *) A, (void *) b,
                              (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESSolve
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                             NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x      )
{
   return( hypre_FGMRESSolve( (void *) solver, (void *) A,
                              (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESSetKDim
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESSetKDim( NALU_HYPRE_Solver solver, int dim    )
{
   return( hypre_FGMRESSetKDim( (void *) solver, dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESSetTol
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESSetTol( NALU_HYPRE_Solver solver, double tol    )
{
   return( hypre_FGMRESSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESSetMaxIter( NALU_HYPRE_Solver solver, int max_iter )
{
   return( hypre_FGMRESSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESetStopCrit
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit )
{
   return( hypre_FGMRESSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void *precond_data )
{
   return( hypre_FGMRESSetPrecond( (void *) solver,
                                   (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond,
								   (NALU_HYPRE_Int (*)(void*,void*,void*,void*))precond_setup,
								   precond_data ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESSetLogging
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESSetLogging( NALU_HYPRE_Solver solver, int logging)
{
   return( hypre_FGMRESSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESetNumIterations
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESGetNumIterations(NALU_HYPRE_Solver solver,int *num_iterations)
{
   return( hypre_FGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                    double *norm   )
{
   return( hypre_FGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFGMRESUpdatePrecondTolerance
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRFGMRESUpdatePrecondTolerance( NALU_HYPRE_Solver  solver,
          int (*update_tol)(NALU_HYPRE_Solver sol, double ) )
{
	return( hypre_FGMRESUpdatePrecondTolerance(solver,(int(*) (int*, double)) update_tol) );
}

