/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"
#include <math.h>

/*--------------------------------------------------------------------------
 * nalu_hypre_DoubleQuickSplit
 * C version of the routine "qsplit" from SPARSKIT
 * Uses a quicksort-type algorithm to split data into
 * highest "NumberCut" values without completely sorting them.
 * Data is NALU_HYPRE_Real precision data.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_DoubleQuickSplit(NALU_HYPRE_Real *values, NALU_HYPRE_Int *indices,
                                 NALU_HYPRE_Int list_length, NALU_HYPRE_Int NumberKept )
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Real interchange_value;
   NALU_HYPRE_Real abskey;
   NALU_HYPRE_Int interchange_index;
   NALU_HYPRE_Int first, last;
   NALU_HYPRE_Int mid, j;
   NALU_HYPRE_Int done;

   first = 0;
   last = list_length - 1;

   if ( (NumberKept < first + 1) || (NumberKept > last + 1) )
   {
      return ( ierr );
   }

   /* Loop until the "midpoint" is NumberKept */
   done = 0;

   for ( ; !done; )
   {
      mid = first;
      abskey = nalu_hypre_abs( values[ mid ]);

      for ( j = first + 1; j <= last; j ++)
      {
         if ( nalu_hypre_abs( values[ j ]) > abskey )
         {
            mid ++;
            /* interchange values */
            interchange_value = values[ mid];
            interchange_index = indices[ mid];
            values[ mid] = values[ j];
            indices[ mid] = indices[ j];
            values[ j] = interchange_value;
            indices[ j] = interchange_index;
         }
      }

      /*  interchange the first and mid value */
      interchange_value = values[ mid];
      interchange_index = indices[ mid];
      values[ mid] = values[ first];
      indices[ mid] = indices[ first];
      values[ first] = interchange_value;
      indices[ first] = interchange_index;

      if ( mid + 1 == NumberKept )
      {
         done = 1;
         break;
      }
      if ( mid + 1 > NumberKept )
      {
         last = mid - 1;
      }
      else
      {
         first = mid + 1;
      }
   }

   return ( ierr );
}

