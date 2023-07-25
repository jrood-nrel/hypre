/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * pblas1.c
 *
 * This file contains functions that implement various distributed
 * level 1 BLAS routines
 *
 * Started 11/28/95
 * George
 *
 * $Id$
 *
 */

#include "_nalu_hypre_blas.h"
#include "DistributedMatrixPilutSolver.h"


/*************************************************************************
* This function computes the 2 norm of a vector. The result is returned
* at all the processors
**************************************************************************/
NALU_HYPRE_Real nalu_hypre_p_dnrm2(DataDistType *ddist, NALU_HYPRE_Real *x, nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int incx=1;
  NALU_HYPRE_Real sum;

  sum = nalu_hypre_dnrm2(&(ddist->ddist_lnrows), x, &incx);
  return nalu_hypre_sqrt(nalu_hypre_GlobalSESumDouble(sum*sum, pilut_comm));
}


/*************************************************************************
* This function computes the dot product of 2 vectors. 
* The result is returned at all the processors
**************************************************************************/
NALU_HYPRE_Real nalu_hypre_p_ddot(DataDistType *ddist, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y,
              nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int incx=1;

  return nalu_hypre_GlobalSESumDouble(nalu_hypre_ddot(&(ddist->ddist_lnrows), x, &incx, y, &incx), 
         pilut_comm );
}


/*************************************************************************
* This function performs y = alpha*x, where alpha resides on pe 0
**************************************************************************/
void nalu_hypre_p_daxy(DataDistType *ddist, NALU_HYPRE_Real alpha, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y)
{
  NALU_HYPRE_Int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    y[i] = alpha*x[i];
}


/*************************************************************************
* This function performs y = alpha*x+y, where alpha resides on pe 0
**************************************************************************/
void nalu_hypre_p_daxpy(DataDistType *ddist, NALU_HYPRE_Real alpha, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y)
{
  NALU_HYPRE_Int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    y[i] += alpha*x[i];
}



/*************************************************************************
* This function performs z = alpha*x+beta*y, where alpha resides on pe 0
**************************************************************************/
void nalu_hypre_p_daxbyz(DataDistType *ddist, NALU_HYPRE_Real alpha, NALU_HYPRE_Real *x, NALU_HYPRE_Real beta, 
              NALU_HYPRE_Real *y, NALU_HYPRE_Real *z)
{
  NALU_HYPRE_Int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    z[i] = alpha*x[i] + beta*y[i];
}

/*************************************************************************
* This function prints a vector
**************************************************************************/
NALU_HYPRE_Int nalu_hypre_p_vprintf(DataDistType *ddist, NALU_HYPRE_Real *x,
                    nalu_hypre_PilutSolverGlobals *globals )
{
  NALU_HYPRE_Int pe, i;

  for (pe=0; pe<npes; pe++) {
    if (mype == pe) {
      for (i=0; i<ddist->ddist_lnrows; i++)
        nalu_hypre_printf("%d:%f, ", ddist->ddist_rowdist[mype]+i, x[i]);
      if (pe == npes-1)
        nalu_hypre_printf("\n");
    }
    nalu_hypre_MPI_Barrier( pilut_comm );
  }
  fflush(stdout);
  nalu_hypre_MPI_Barrier( pilut_comm );

  return 0;
}
