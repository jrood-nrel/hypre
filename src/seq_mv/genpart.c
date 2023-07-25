/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_GeneratePartitioning:
 * generates load balanced partitioning of a 1-d array
 *--------------------------------------------------------------------------*/
/* for multivectors, length should be the (global) length of a single vector.
 Thus each of the vectors of the multivector will get the same data distribution. */

NALU_HYPRE_Int
nalu_hypre_GeneratePartitioning(NALU_HYPRE_BigInt length, NALU_HYPRE_Int num_procs, NALU_HYPRE_BigInt **part_ptr)
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_BigInt *part;
   NALU_HYPRE_Int size, rest;
   NALU_HYPRE_Int i;

   part = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   size = (NALU_HYPRE_Int)(length / (NALU_HYPRE_BigInt)num_procs);
   rest = (NALU_HYPRE_Int)(length - (NALU_HYPRE_BigInt)(size * num_procs));
   part[0] = 0;
   for (i = 0; i < num_procs; i++)
   {
      part[i + 1] = part[i] + (NALU_HYPRE_BigInt)size;
      if (i < rest) { part[i + 1]++; }
   }

   *part_ptr = part;
   return ierr;
}


/* This function differs from the above in that it only returns
   the portion of the partition belonging to the individual process -
   to do this it requires the processor id as well AHB 6/05.

   This functions assumes that part is on the stack memory
   and has size equal to 2.
*/

NALU_HYPRE_Int
nalu_hypre_GenerateLocalPartitioning(NALU_HYPRE_BigInt   length,
                                NALU_HYPRE_Int      num_procs,
                                NALU_HYPRE_Int      myid,
                                NALU_HYPRE_BigInt  *part)
{
   NALU_HYPRE_Int  size, rest;

   size = (NALU_HYPRE_Int)(length / (NALU_HYPRE_BigInt)num_procs);
   rest = (NALU_HYPRE_Int)(length - (NALU_HYPRE_BigInt)(size * num_procs));

   /* first row I own */
   part[0] = (NALU_HYPRE_BigInt)(size * myid);
   part[0] += (NALU_HYPRE_BigInt)(nalu_hypre_min(myid, rest));

   /* last row I own */
   part[1] =  (NALU_HYPRE_BigInt)(size * (myid + 1));
   part[1] += (NALU_HYPRE_BigInt)(nalu_hypre_min(myid + 1, rest));
   part[1] = part[1] - 1;

   /* add 1 to last row since this is for "starts" vector */
   part[1] = part[1] + 1;

   return nalu_hypre_error_flag;
}
