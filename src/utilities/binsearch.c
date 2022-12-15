/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BinarySearch
 * performs a binary search for value on array list where list needs
 * to contain ordered nonnegative numbers
 * the routine returns the location of the value or -1
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_BinarySearch(NALU_HYPRE_Int *list, NALU_HYPRE_Int value, NALU_HYPRE_Int list_length)
{
   NALU_HYPRE_Int low, high, m;
   NALU_HYPRE_Int not_found = 1;

   low = 0;
   high = list_length - 1;
   while (not_found && low <= high)
   {
      m = (low + high) / 2;
      if (value < list[m])
      {
         high = m - 1;
      }
      else if (value > list[m])
      {
         low = m + 1;
      }
      else
      {
         not_found = 0;
         return m;
      }
   }
   return -1;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BigBinarySearch
 * performs a binary search for value on array list where list needs
 * to contain ordered nonnegative numbers
 * the routine returns the location of the value or -1
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_BigBinarySearch(NALU_HYPRE_BigInt *list, NALU_HYPRE_BigInt value, NALU_HYPRE_Int list_length)
{
   NALU_HYPRE_Int low, high, m;
   NALU_HYPRE_Int not_found = 1;

   low = 0;
   high = list_length - 1;
   while (not_found && low <= high)
   {
      m = low + (high - low) / 2;
      if (value < list[m])
      {
         high = m - 1;
      }
      else if (value > list[m])
      {
         low = m + 1;
      }
      else
      {
         not_found = 0;
         return m;
      }
   }
   return -1;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BinarySearch2
 * this one is a bit more robust:
 *   avoids overflow of m as can happen above when (low+high) overflows
 *   lets user specify high and low bounds for array (so a subset
     of array can be used)
 *  if not found, then spot returns where is should be inserted

 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_BinarySearch2(NALU_HYPRE_Int *list, NALU_HYPRE_Int value, NALU_HYPRE_Int low, NALU_HYPRE_Int high,
                              NALU_HYPRE_Int *spot)
{

   NALU_HYPRE_Int m;

   while (low <= high)
   {
      m = low + (high - low) / 2;

      if (value < list[m])
      {
         high = m - 1;
      }
      else if (value > list[m])
      {
         low = m + 1;
      }
      else
      {
         *spot = m;
         return m;
      }
   }

   /* not found (high = low-1) - so insert at low */
   *spot = low;

   return -1;
}

/*--------------------------------------------------------------------------
 * Equivalent to C++ std::lower_bound
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int *nalu_hypre_LowerBound( NALU_HYPRE_Int *first, NALU_HYPRE_Int *last, NALU_HYPRE_Int value )
{
   NALU_HYPRE_Int *it;
   NALU_HYPRE_Int count = last - first, step;

   while (count > 0)
   {
      it = first; step = count / 2; it += step;
      if (*it < value)
      {
         first = ++it;
         count -= step + 1;
      }
      else { count = step; }
   }
   return first;
}
/*--------------------------------------------------------------------------
 * Equivalent to C++ std::lower_bound
 *--------------------------------------------------------------------------*/

NALU_HYPRE_BigInt *nalu_hypre_BigLowerBound( NALU_HYPRE_BigInt *first, NALU_HYPRE_BigInt *last, NALU_HYPRE_BigInt value )
{
   NALU_HYPRE_BigInt *it;
   NALU_HYPRE_BigInt count = last - first, step;

   while (count > 0)
   {
      it = first; step = count / 2; it += step;
      if (*it < value)
      {
         first = ++it;
         count -= step + 1;
      }
      else { count = step; }
   }
   return first;
}
