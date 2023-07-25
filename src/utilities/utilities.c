/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_multmod
 *--------------------------------------------------------------------------*/

/* This function computes (a*b) % mod, which can avoid overflow in large value of (a*b) */
NALU_HYPRE_Int
nalu_hypre_multmod(NALU_HYPRE_Int a,
              NALU_HYPRE_Int b,
              NALU_HYPRE_Int mod)
{
   NALU_HYPRE_Int res = 0; // Initialize result
   a %= mod;
   while (b)
   {
      // If b is odd, add a with result
      if (b & 1)
      {
         res = (res + a) % mod;
      }
      // Here we assume that doing 2*a
      // doesn't cause overflow
      a = (2 * a) % mod;
      b >>= 1;  // b = b / 2
   }
   return res;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_partition1D
 *--------------------------------------------------------------------------*/
void
nalu_hypre_partition1D(NALU_HYPRE_Int  n, /* total number of elements */
                  NALU_HYPRE_Int  p, /* number of partitions */
                  NALU_HYPRE_Int  j, /* index of this partition */
                  NALU_HYPRE_Int *s, /* first element in this partition */
                  NALU_HYPRE_Int *e  /* past-the-end element */ )

{
   if (1 == p)
   {
      *s = 0;
      *e = n;
      return;
   }

   NALU_HYPRE_Int size = n / p;
   NALU_HYPRE_Int rest = n - size * p;
   if (j < rest)
   {
      *s = j * (size + 1);
      *e = (j + 1) * (size + 1);
   }
   else
   {
      *s = j * size + rest;
      *e = (j + 1) * size + rest;
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_strcpy
 *
 * Note: strcpy that allows overlapping in memory
 *--------------------------------------------------------------------------*/

char *
nalu_hypre_strcpy(char *destination, const char *source)
{
   size_t len = strlen(source);

   /* no overlapping */
   if (source > destination + len || destination > source + len)
   {
      return strcpy(destination, source);
   }
   else
   {
      /* +1: including the terminating null character */
      return ((char *) memmove(destination, source, len + 1));
   }
}
