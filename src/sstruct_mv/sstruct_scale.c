/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct scale routine
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructPScale( NALU_HYPRE_Complex         alpha,
                     hypre_SStructPVector *py )
{
   NALU_HYPRE_Int nvars = hypre_SStructPVectorNVars(py);
   NALU_HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructScale(alpha, hypre_SStructPVectorSVector(py, var));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructScale( NALU_HYPRE_Complex        alpha,
                    hypre_SStructVector *y )
{
   NALU_HYPRE_Int nparts = hypre_SStructVectorNParts(y);
   NALU_HYPRE_Int part;
   NALU_HYPRE_Int y_object_type = hypre_SStructVectorObjectType(y);

   if (y_object_type == NALU_HYPRE_SSTRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPScale(alpha, hypre_SStructVectorPVector(y, part));
      }
   }

   else if (y_object_type == NALU_HYPRE_PARCSR)
   {
      hypre_ParVector  *y_par;

      hypre_SStructVectorConvert(y, &y_par);
      hypre_ParVectorScale(alpha, y_par);
   }

   return hypre_error_flag;
}
