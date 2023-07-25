/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * ilut.c
 *
 * This file contains the top level code for the parallel nalu_hypre_ILUT algorithms
 *
 * Started 11/29/95
 * George
 *
 * $Id$
 */

#include <math.h>
#include "./DistributedMatrixPilutSolver.h"

/*************************************************************************
* This function is the entry point of the nalu_hypre_ILUT factorization
**************************************************************************/
NALU_HYPRE_Int nalu_hypre_ILUT(DataDistType *ddist, NALU_HYPRE_DistributedMatrix matrix, FactorMatType *ldu,
          NALU_HYPRE_Int maxnz, NALU_HYPRE_Real tol, nalu_hypre_PilutSolverGlobals *globals )
{
  NALU_HYPRE_Int i, ierr;
  ReduceMatType rmat;
  NALU_HYPRE_Int dummy_row_ptr[2], size;
  NALU_HYPRE_Real *values;

#ifdef NALU_HYPRE_DEBUG
  NALU_HYPRE_Int logging = globals ? globals->logging : 0;
  if (logging)
  {
     nalu_hypre_printf("nalu_hypre_ILUT, maxnz = %d\n ", maxnz);
  }
#endif

  /* Allocate memory for ldu */
  if (ldu->lsrowptr) nalu_hypre_TFree(ldu->lsrowptr, NALU_HYPRE_MEMORY_HOST);
  ldu->lsrowptr = nalu_hypre_idx_malloc(ddist->ddist_lnrows, "nalu_hypre_ILUT: ldu->lsrowptr");

  if (ldu->lerowptr) nalu_hypre_TFree(ldu->lerowptr, NALU_HYPRE_MEMORY_HOST);
  ldu->lerowptr = nalu_hypre_idx_malloc(ddist->ddist_lnrows, "nalu_hypre_ILUT: ldu->lerowptr");

  if (ldu->lcolind) nalu_hypre_TFree(ldu->lcolind, NALU_HYPRE_MEMORY_HOST);
  ldu->lcolind  = nalu_hypre_idx_malloc_init(maxnz*ddist->ddist_lnrows, 0, "nalu_hypre_ILUT: ldu->lcolind");

  if (ldu->lvalues) nalu_hypre_TFree(ldu->lvalues, NALU_HYPRE_MEMORY_HOST);
  ldu->lvalues  =  nalu_hypre_fp_malloc_init(maxnz*ddist->ddist_lnrows, 0, "nalu_hypre_ILUT: ldu->lvalues");

  if (ldu->usrowptr) nalu_hypre_TFree(ldu->usrowptr, NALU_HYPRE_MEMORY_HOST);
  ldu->usrowptr = nalu_hypre_idx_malloc(ddist->ddist_lnrows, "nalu_hypre_ILUT: ldu->usrowptr");

  if (ldu->uerowptr) nalu_hypre_TFree(ldu->uerowptr, NALU_HYPRE_MEMORY_HOST);
  ldu->uerowptr = nalu_hypre_idx_malloc(ddist->ddist_lnrows, "nalu_hypre_ILUT: ldu->uerowptr");

  if (ldu->ucolind) nalu_hypre_TFree(ldu->ucolind, NALU_HYPRE_MEMORY_HOST);
  ldu->ucolind  = nalu_hypre_idx_malloc_init(maxnz*ddist->ddist_lnrows, 0, "nalu_hypre_ILUT: ldu->ucolind");

  if (ldu->uvalues) nalu_hypre_TFree(ldu->uvalues, NALU_HYPRE_MEMORY_HOST);
  ldu->uvalues  =  nalu_hypre_fp_malloc_init(maxnz*ddist->ddist_lnrows, 0.0, "nalu_hypre_ILUT: ldu->uvalues");

  if (ldu->dvalues) nalu_hypre_TFree(ldu->dvalues, NALU_HYPRE_MEMORY_HOST);
  ldu->dvalues = nalu_hypre_fp_malloc(ddist->ddist_lnrows, "nalu_hypre_ILUT: ldu->dvalues");

  if (ldu->nrm2s) nalu_hypre_TFree(ldu->nrm2s, NALU_HYPRE_MEMORY_HOST);
  ldu->nrm2s   = nalu_hypre_fp_malloc_init(ddist->ddist_lnrows, 0.0, "nalu_hypre_ILUT: ldu->nrm2s");

  if (ldu->perm) nalu_hypre_TFree(ldu->perm, NALU_HYPRE_MEMORY_HOST);
  ldu->perm  = nalu_hypre_idx_malloc_init(ddist->ddist_lnrows, 0, "nalu_hypre_ILUT: ldu->perm");

  if (ldu->iperm) nalu_hypre_TFree(ldu->iperm, NALU_HYPRE_MEMORY_HOST);
  ldu->iperm = nalu_hypre_idx_malloc_init(ddist->ddist_lnrows, 0, "nalu_hypre_ILUT: ldu->iperm");

  firstrow = ddist->ddist_rowdist[mype];

  dummy_row_ptr[ 0 ] = 0;

  /* Initialize ldu */
  for (i=0; i<ddist->ddist_lnrows; i++) {
    ldu->lsrowptr[i] =
      ldu->lerowptr[i] =
      ldu->usrowptr[i] =
      ldu->uerowptr[i] = maxnz*i;

    ierr = NALU_HYPRE_DistributedMatrixGetRow( matrix, firstrow+i, &size,
               NULL, &values);
    /* if (ierr) return(ierr);*/
    dummy_row_ptr[ 1 ] = size;
    nalu_hypre_ComputeAdd2Nrms( 1, dummy_row_ptr, values, &(ldu->nrm2s[i]) );
    ierr = NALU_HYPRE_DistributedMatrixRestoreRow( matrix, firstrow+i, &size,
               NULL, &values);
  }

  /* Factor the internal nodes first */
  nalu_hypre_MPI_Barrier( pilut_comm );

#ifdef NALU_HYPRE_TIMING
  {
   NALU_HYPRE_Int SerILUT_timer;

   SerILUT_timer = nalu_hypre_InitializeTiming( "Sequential nalu_hypre_ILUT done on each proc" );

   nalu_hypre_BeginTiming( SerILUT_timer );
#endif

  nalu_hypre_SerILUT(ddist, matrix, ldu, &rmat, maxnz, tol, globals);

  nalu_hypre_MPI_Barrier( pilut_comm );

#ifdef NALU_HYPRE_TIMING
   nalu_hypre_EndTiming( SerILUT_timer );
   /* nalu_hypre_FinalizeTiming( SerILUT_timer ); */
  }
#endif

  /* Factor the interface nodes */
#ifdef NALU_HYPRE_TIMING
  {
   NALU_HYPRE_Int ParILUT_timer;

   ParILUT_timer = nalu_hypre_InitializeTiming( "Parallel portion of nalu_hypre_ILUT factorization" );

   nalu_hypre_BeginTiming( ParILUT_timer );
#endif

  nalu_hypre_ParILUT(ddist, ldu, &rmat, maxnz, tol, globals);

  nalu_hypre_MPI_Barrier( pilut_comm );

#ifdef NALU_HYPRE_TIMING
   nalu_hypre_EndTiming( ParILUT_timer );
   /* nalu_hypre_FinalizeTiming( ParILUT_timer ); */
  }
#endif

  /*nalu_hypre_free_multi(rmat.rmat_rnz, rmat.rmat_rrowlen,
             rmat.rmat_rcolind, rmat.rmat_rvalues, -1);*/
  nalu_hypre_TFree(rmat.rmat_rnz, NALU_HYPRE_MEMORY_HOST);
  nalu_hypre_TFree(rmat.rmat_rrowlen, NALU_HYPRE_MEMORY_HOST);
  nalu_hypre_TFree(rmat.rmat_rcolind, NALU_HYPRE_MEMORY_HOST);
  nalu_hypre_TFree(rmat.rmat_rvalues, NALU_HYPRE_MEMORY_HOST);

  return( ierr );
}


/*************************************************************************
* This function computes the 2 norms of the rows and adds them into the
* nrm2s array ... Changed to "Add" by AJC, Dec 22 1997.
**************************************************************************/
void nalu_hypre_ComputeAdd2Nrms(NALU_HYPRE_Int num_rows, NALU_HYPRE_Int *rowptr, NALU_HYPRE_Real *values, NALU_HYPRE_Real *nrm2s)
{
  NALU_HYPRE_Int i, j, n;
  NALU_HYPRE_Real sum;

  for (i=0; i<num_rows; i++) {
    n = rowptr[i+1]-rowptr[i];
    /* sum = nalu_hypre_dnrm2(&n, values+rowptr[i], &incx);*/
    sum = 0.0;
    for (j=0; j<n; j++) sum += (values[rowptr[i]+j] * values[rowptr[i]+j]);
    sum = nalu_hypre_sqrt( sum );
    nrm2s[i] += sum;
  }
}
