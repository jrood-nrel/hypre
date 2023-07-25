/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * util.c
 *
 * This function contains various utility routines
 *
 * Started 9/28/95
 * George
 *
 * $Id$
 */

#include "ilu.h"
#include "DistributedMatrixPilutSolver.h"

/*************************************************************************
* This function finds the minimum value in the array removes it and
* returns it. It decreases the size of the array.
**************************************************************************/
NALU_HYPRE_Int nalu_hypre_ExtractMinLR( nalu_hypre_PilutSolverGlobals *globals )
{
  NALU_HYPRE_Int i, j=0 ;

  for (i=1; i<lastlr; i++) {
    if (nalu_hypre_lr[i] < nalu_hypre_lr[j])
      j = i;
  }
  i = nalu_hypre_lr[j];

  /* Remove it */
  lastlr-- ;
  if (j < lastlr) 
    nalu_hypre_lr[j] = nalu_hypre_lr[lastlr];

  return i;
}


/*************************************************************************
* This function sort an (idx,val) array in increasing idx values
**************************************************************************/
void nalu_hypre_IdxIncSort(NALU_HYPRE_Int n, NALU_HYPRE_Int *idx, NALU_HYPRE_Real *val)
{
  NALU_HYPRE_Int i, j, min;
  NALU_HYPRE_Real tmpval;
  NALU_HYPRE_Int tmpidx;

  for (i=0; i<n; i++) {
    min = i;
    for (j=i+1; j<n; j++) {
      if (idx[j] < idx[min])
        min = j;
    }

    if (min != i) {
      SWAP(idx[i], idx[min], tmpidx);
      SWAP(val[i], val[min], tmpval);
    }
  }
}



/*************************************************************************
* This function sort an (idx,val) array in decreasing abs val 
**************************************************************************/
void nalu_hypre_ValDecSort(NALU_HYPRE_Int n, NALU_HYPRE_Int *idx, NALU_HYPRE_Real *val)
{
  NALU_HYPRE_Int i, j, max;
  NALU_HYPRE_Int tmpidx;
  NALU_HYPRE_Real tmpval;

  for (i=0; i<n; i++) {
    max = i;
    for (j=i+1; j<n; j++) {
      if (nalu_hypre_abs(val[j]) > nalu_hypre_abs(val[max]))
        max = j;
    }

    if (max != i) {
      SWAP(idx[i], idx[max], tmpidx);
      SWAP(val[i], val[max], tmpval);
    }
  }
}





/*************************************************************************
* This function takes an (idx, val) array and compacts it so that every 
* entry with idx[] = -1, gets removed. It returns the new count
**************************************************************************/
NALU_HYPRE_Int nalu_hypre_CompactIdx(NALU_HYPRE_Int n, NALU_HYPRE_Int *idx, NALU_HYPRE_Real *val)
{
  NALU_HYPRE_Int i, j;

  j = n-1;
  for (i=0; i<n; i++) {
    if (idx[i] == -1) {
      while (j > i && idx[j] == -1)
        j--;
      if (j > i) {
        idx[i] = idx[j];
        val[i] = val[j];
        j--;
      }
      else {
        n = i;
        break;
      }
    }
    if (i == j) {
      n = i+1;
      break;
    }
  }

  return n;
}

/*************************************************************************
* This function prints an (idx, val) pair
**************************************************************************/
void nalu_hypre_PrintIdxVal(NALU_HYPRE_Int n, NALU_HYPRE_Int *idx, NALU_HYPRE_Real *val)
{
  NALU_HYPRE_Int i;

  nalu_hypre_printf("%3d ", n);
  for (i=0; i<n; i++) 
    nalu_hypre_printf("(%3d, %3.1e) ", idx[i], val[i]);
  nalu_hypre_printf("\n");

}



/*************************************************************************
* This function compares 2 KeyValueType variables for sorting in inc order
**************************************************************************/
NALU_HYPRE_Int nalu_hypre_DecKeyValueCmp(const void *v1, const void *v2)
{
  KeyValueType *n1, *n2;

  n1 = (KeyValueType *)v1;
  n2 = (KeyValueType *)v2;

  return n2->key - n1->key;

}


/*************************************************************************
* This function sorts an array of type KeyValueType in increasing order
**************************************************************************/
void nalu_hypre_SortKeyValueNodesDec(KeyValueType *nodes, NALU_HYPRE_Int n)
{
	nalu_hypre_tex_qsort((char *)nodes, (size_t)n, (size_t)sizeof(KeyValueType), (NALU_HYPRE_Int (*) (char*,char*))nalu_hypre_DecKeyValueCmp);
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
NALU_HYPRE_Int nalu_hypre_sasum(NALU_HYPRE_Int n, NALU_HYPRE_Int *x)
{
  NALU_HYPRE_Int sum = 0;
  NALU_HYPRE_Int i;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}


/*************************************************************************
* This function compares 2 ints for sorting in inc order
**************************************************************************/
static NALU_HYPRE_Int incshort(const void *v1, const void *v2)
{
  return (*((NALU_HYPRE_Int *)v1) - *((NALU_HYPRE_Int *)v2));
}

/*************************************************************************
* This function compares 2 ints for sorting in dec order
**************************************************************************/
static NALU_HYPRE_Int decshort(const void *v1, const void *v2)
{
  return (*((NALU_HYPRE_Int *)v2) - *((NALU_HYPRE_Int *)v1));
}

/*************************************************************************
* These functions sorts an array of XXX
**************************************************************************/
void nalu_hypre_sincsort(NALU_HYPRE_Int n, NALU_HYPRE_Int *a)
{
  nalu_hypre_tex_qsort((char *)a, (size_t)n, (size_t)sizeof(NALU_HYPRE_Int), (NALU_HYPRE_Int (*) (char*,char*))incshort);
}


void nalu_hypre_sdecsort(NALU_HYPRE_Int n, NALU_HYPRE_Int *a)
{
  nalu_hypre_tex_qsort((char *)a, (size_t)n, (size_t)sizeof(NALU_HYPRE_Int),(NALU_HYPRE_Int (*) (char*,char*)) decshort);
}




