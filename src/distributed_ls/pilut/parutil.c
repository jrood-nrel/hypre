/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * parutil.c
 *
 * This file contains utility functions
 *
 * Started 8/28/94
 * George
 *
 * $Id$
 *
 */

#include "ilu.h"
#include "DistributedMatrixPilutSolver.h"


/*************************************************************************
* This function prints an error message and exits
**************************************************************************/
void nalu_hypre_errexit(const char *f_str, ...)
{
  va_list argp;

  /*nalu_hypre_fprintf(stdout,"[%3d]", mype);*/

  va_start(argp, f_str);
  vfprintf(stdout, f_str, argp);
  va_end(argp);

  nalu_hypre_fprintf(stdout,"\n");
  fflush(stdout);

  abort();
}


/*************************************************************************
* This makes life easier by aborting all threads together, and printing
* some diagnostic with the PE.
**************************************************************************/
void nalu_hypre_my_abort( NALU_HYPRE_Int inSignal, nalu_hypre_PilutSolverGlobals *globals )
{
  nalu_hypre_printf( "PE %d caught sig %d\n", mype, inSignal );
  fflush(stdout);
  nalu_hypre_MPI_Abort( pilut_comm, inSignal );
}


/*************************************************************************
* The following function allocates an array of ints
**************************************************************************/
NALU_HYPRE_Int *nalu_hypre_idx_malloc(NALU_HYPRE_Int n,const char *msg)
{
  NALU_HYPRE_Int *ptr;

  if (n == 0)
    return NULL;

  ptr = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
  if (ptr == NULL) {
    nalu_hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(NALU_HYPRE_Int));
  }

  return ptr;

}


/*************************************************************************
* The follwoing function allocates an array of ints and initializes
**************************************************************************/
NALU_HYPRE_Int *nalu_hypre_idx_malloc_init(NALU_HYPRE_Int n, NALU_HYPRE_Int ival,const char *msg)
{
  NALU_HYPRE_Int *ptr;
  NALU_HYPRE_Int i;

  if (n == 0)
    return NULL;

  ptr = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
  if (ptr == NULL) {
    nalu_hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(NALU_HYPRE_Int));
  }

  for (i=0; i<n; i++)
    ptr[i] = ival;

  return ptr;
}


/*************************************************************************
* The following function allocates an array of floats
**************************************************************************/
NALU_HYPRE_Real *nalu_hypre_fp_malloc(NALU_HYPRE_Int n,const char *msg)
{
  NALU_HYPRE_Real *ptr;

  if (n == 0)
    return NULL;

  ptr = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST);
  if (ptr == NULL) {
    nalu_hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(NALU_HYPRE_Real));
  }

  return ptr;

}


/*************************************************************************
* The follwoing function allocates an array of floats and initializes
**************************************************************************/
NALU_HYPRE_Real *nalu_hypre_fp_malloc_init(NALU_HYPRE_Int n, NALU_HYPRE_Real ival,const char *msg)
{
  NALU_HYPRE_Real *ptr;
  NALU_HYPRE_Int i;

  if (n == 0)
    return NULL;

  ptr = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST);
  if (ptr == NULL) {
    nalu_hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(NALU_HYPRE_Real));
  }

  for (i=0; i<n; i++)
    ptr[i] = ival;

  return ptr;
}



/*************************************************************************
* This function is my wrapper around malloc.
**************************************************************************/
void *nalu_hypre_mymalloc(NALU_HYPRE_Int nbytes,const char *msg)
{
  void *ptr;

  if (nbytes == 0)
    return NULL;

  ptr = nalu_hypre_TAlloc(char, nbytes, NALU_HYPRE_MEMORY_HOST);
  if (ptr == NULL) {
    nalu_hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, nbytes);
  }

  return ptr;
}


/*************************************************************************
* This function is my wrapper around free, allows multiple pointers
**************************************************************************/
#if 0
void nalu_hypre_free_multi(void *ptr1,...)
{
   va_list plist;
   void *ptr;

   nalu_hypre_TFree(ptr1, NALU_HYPRE_MEMORY_HOST);

   va_start(plist, ptr1);

   while ( (ptr = va_arg(plist, void *)) != ((void *) -1) ) {
      nalu_hypre_TFree(ptr, NALU_HYPRE_MEMORY_HOST);
   }

   va_end(plist);
}
#endif

/*************************************************************************
* The following function copies an NALU_HYPRE_Int (NALU_HYPRE_Int) array
**************************************************************************/
void nalu_hypre_memcpy_int( NALU_HYPRE_Int *dest, const NALU_HYPRE_Int *src, size_t n )
{
   if (dest) nalu_hypre_TMemcpy(dest,  src, NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
}

/*************************************************************************
* The following function copies an NALU_HYPRE_Int (NALU_HYPRE_Int) array
**************************************************************************/
void nalu_hypre_memcpy_idx( NALU_HYPRE_Int *dest, const NALU_HYPRE_Int *src, size_t n )
{
   if (dest) nalu_hypre_TMemcpy(dest,  src, NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
}

/*************************************************************************
* The following function copies a floating point (NALU_HYPRE_Real) array.
* Note this assumes BLAS 1 routine nalu_hypre_dcopy. An alternative would be memcpy.
* There is a noticeable difference between this and just a for loop.
**************************************************************************/
void nalu_hypre_memcpy_fp( NALU_HYPRE_Real *dest, const NALU_HYPRE_Real *src, size_t n )
{
  NALU_HYPRE_Int i, ni = (NALU_HYPRE_Int) n;

  /*nalu_hypre_dcopy(&n, src, &inc, dest, &inc);*/
  for (i=0; i<ni; i++) dest[i] = src[i];
}

