/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * debug.c
 *
 * This file implements some debugging utilities.
 * I use checksums to compare entire arrays easily. Note that the
 * perm and iperm arrays always have the same checksum, even
 * though they are in a different order.
 *
 * Started 7/8/97
 * Mark
 *
 */

#undef NDEBUG

#include "./DistributedMatrixPilutSolver.h"

/*************************************************************************
* This function prints a message and file/line number
**************************************************************************/
void nalu_hypre_PrintLine(const char *str, nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int logging = globals ? globals->logging : 0;

  if (logging)
  {
     nalu_hypre_printf("PE %d ---- %-27s (%s, %d)\n",
           mype, str, __FILE__, __LINE__);
  }
  fflush(stdout);
}


/*************************************************************************
* This function exits if i is not in [low, up)
**************************************************************************/
void nalu_hypre_CheckBounds(NALU_HYPRE_Int low, NALU_HYPRE_Int i ,NALU_HYPRE_Int up, nalu_hypre_PilutSolverGlobals *globals)
{
  if ((i < low)  ||  (i >= up))
    nalu_hypre_errexit("PE %d Bad bound: %d <= %d < %d (%s %d)\n",
          mype, low, i, up, __FILE__, __LINE__ );
}

/*************************************************************************
* This function prints a checksum for an NALU_HYPRE_Int (NALU_HYPRE_Int) array
**************************************************************************/
nalu_hypre_longint nalu_hypre_IDX_Checksum(const NALU_HYPRE_Int *v, NALU_HYPRE_Int len, const char *msg, NALU_HYPRE_Int tag,
          nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int logging = globals ? globals->logging : 0;
  static NALU_HYPRE_Int numChk = 0;
  NALU_HYPRE_Int i;
  nalu_hypre_ulongint sum = 0;

  for (i=0; i<len; i++)
    sum += v[i] * i;

  if (logging)
  {
     nalu_hypre_printf("PE %d [i%3d] %15s/%3d chk: %16lx [len %4d]\n",
           mype, numChk, msg, tag, sum, len);
     fflush(stdout);
  }

  numChk++;

  return sum;
}

/*************************************************************************
* This function prints a checksum for an NALU_HYPRE_Int (NALU_HYPRE_Int) array
**************************************************************************/
nalu_hypre_longint nalu_hypre_INT_Checksum(const NALU_HYPRE_Int *v, NALU_HYPRE_Int len, const char *msg, NALU_HYPRE_Int tag,
          nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int logging = globals ? globals->logging : 0;
  static NALU_HYPRE_Int numChk = 0;
  NALU_HYPRE_Int i;
  nalu_hypre_ulongint sum = 0;

  for (i=0; i<len; i++)
    sum += v[i] * i;

  if (logging)
  {
     nalu_hypre_printf("PE %d [d%3d] %15s/%3d chk: %16lx [len %4d]\n",
           mype, numChk, msg, tag, sum, len);
     fflush(stdout);
  }

  numChk++;

  return sum;
}

/*************************************************************************
* This function prints a checksum for a float (NALU_HYPRE_Real) array
**************************************************************************/
nalu_hypre_longint nalu_hypre_FP_Checksum(const NALU_HYPRE_Real *v, NALU_HYPRE_Int len, const char *msg, NALU_HYPRE_Int tag,
          nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int logging = globals ? globals->logging : 0;
  static NALU_HYPRE_Int numChk = 0;
  NALU_HYPRE_Int i;
  nalu_hypre_ulongint sum = 0;
  NALU_HYPRE_Int *vv = (NALU_HYPRE_Int*)v;

  for (i=0; i<len; i++)
    sum += vv[i] * i;

  if (logging)
  {
     nalu_hypre_printf("PE %d [f%3d] %15s/%3d chk: %16lx [len %4d]\n",
           mype, numChk, msg, tag, sum, len);
     fflush(stdout);
  }

  numChk++;

  return sum;
}

/*************************************************************************
* This function prints checksums for each array of the rmat struct
**************************************************************************/
nalu_hypre_longint nalu_hypre_RMat_Checksum(const ReduceMatType *rmat,
          nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int logging = globals ? globals->logging : 0;
  NALU_HYPRE_Int i;
  static NALU_HYPRE_Int numChk = 0;

  /* for safety */
  if ( rmat          == NULL  ||
       rmat->rmat_rnz     == NULL  ||
       rmat->rmat_rrowlen == NULL  ||
       rmat->rmat_rcolind == NULL  ||
       rmat->rmat_rvalues == NULL ) {
     if (logging)
     {
        nalu_hypre_printf("PE %d [r%3d] rmat checksum -- not initializied\n",
              mype, numChk);
        fflush(stdout);
     }

    numChk++;
    return 0;
  }

  if (logging)
  {
     /* print ints */
     nalu_hypre_printf("PE %d [r%3d] rmat checksum -- ndone %d ntogo %d nlevel %d\n",
           mype, numChk, rmat->rmat_ndone, rmat->rmat_ntogo, rmat->rmat_nlevel);
     fflush(stdout);
  }

  /* print checksums for each array */
  nalu_hypre_IDX_Checksum(rmat->rmat_rnz,     rmat->rmat_ntogo, "rmat->rmat_rnz",     numChk,
      globals);
  nalu_hypre_IDX_Checksum(rmat->rmat_rrowlen, rmat->rmat_ntogo, "rmat->rmat_rrowlen", numChk,
      globals);

  for (i=0; i<rmat->rmat_ntogo; i++) {
    nalu_hypre_IDX_Checksum(rmat->rmat_rcolind[i], rmat->rmat_rrowlen[i], "rmat->rmat_rcolind", i,
      globals);
     nalu_hypre_FP_Checksum(rmat->rmat_rvalues[i], rmat->rmat_rrowlen[i], "rmat->rmat_rvalues", i,
      globals);
  }

  return 1;
}

/*************************************************************************
* This function prints checksums for some arrays of the LDU struct
**************************************************************************/
nalu_hypre_longint nalu_hypre_LDU_Checksum(const FactorMatType *ldu,
          nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int logging = globals ? globals->logging : 0;
  NALU_HYPRE_Int i, j;
  nalu_hypre_ulongint lisum=0, ldsum=0, uisum=0, udsum=0, dsum=0;
  static NALU_HYPRE_Int numChk = 0;

  if (ldu->lsrowptr == NULL  ||
      ldu->lerowptr == NULL  ||
      ldu->lcolind  == NULL  ||
      ldu->lvalues  == NULL  ||
      ldu->usrowptr == NULL  ||
      ldu->uerowptr == NULL  ||
      ldu->ucolind  == NULL  ||
      ldu->uvalues  == NULL  ||
      ldu->dvalues  == NULL  ||
      ldu->nrm2s    == NULL) {
    nalu_hypre_printf("PE %d [S%3d] LDU check -- not initializied\n",
          mype, numChk);
    fflush(stdout);
    return 0;
  }

  for (i=0; i<lnrows; i++) {
    for (j=ldu->lsrowptr[i]; j<ldu->lerowptr[i]; j++) {
      lisum += ldu->lcolind[j];
      ldsum += (nalu_hypre_longint)ldu->lvalues[j];
    }

    for (j=ldu->usrowptr[i]; j<ldu->uerowptr[i]; j++) {
      uisum += ldu->ucolind[j];
      udsum += (nalu_hypre_longint)ldu->uvalues[j];
    }

    if (ldu->usrowptr[i] < ldu->uerowptr[i])
      dsum += (nalu_hypre_longint)ldu->dvalues[i];
  }

  if (logging)
  {
     nalu_hypre_printf("PE %d [S%3d] LDU check [%16lx %16lx] [%16lx] [%16lx %16lx]\n",
           mype, numChk, lisum, ldsum, dsum, uisum, udsum);
     fflush(stdout);
  }

  nalu_hypre_FP_Checksum(ldu->nrm2s, lnrows, "2-norms", numChk,
      globals);

  return 1;
}


/*************************************************************************
* This function prints a vector on each processor
**************************************************************************/
void nalu_hypre_PrintVector(NALU_HYPRE_Int *v, NALU_HYPRE_Int n, char *msg,
          nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int logging = globals ? globals->logging : 0;
  NALU_HYPRE_Int i, penum;

  for (penum=0; penum<npes; penum++) {
    if (mype == penum) {
       if (logging)
       {
          nalu_hypre_printf("PE %d %s: ", mype, msg);

          for (i=0; i<n; i++)
             nalu_hypre_printf("%d ", v[i]);
          nalu_hypre_printf("\n");
       }
    }
    nalu_hypre_MPI_Barrier( pilut_comm );
  }
}
