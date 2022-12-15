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

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPScale( NALU_HYPRE_Complex         alpha,
                     nalu_hypre_SStructPVector *py )
{
   NALU_HYPRE_Int nvars = nalu_hypre_SStructPVectorNVars(py);
   NALU_HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      nalu_hypre_StructScale(alpha, nalu_hypre_SStructPVectorSVector(py, var));
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructScale( NALU_HYPRE_Complex        alpha,
                    nalu_hypre_SStructVector *y )
{
   NALU_HYPRE_Int nparts = nalu_hypre_SStructVectorNParts(y);
   NALU_HYPRE_Int part;
   NALU_HYPRE_Int y_object_type = nalu_hypre_SStructVectorObjectType(y);

   if (y_object_type == NALU_HYPRE_SSTRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         nalu_hypre_SStructPScale(alpha, nalu_hypre_SStructVectorPVector(y, part));
      }
   }

   else if (y_object_type == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_ParVector  *y_par;

      nalu_hypre_SStructVectorConvert(y, &y_par);
      nalu_hypre_ParVectorScale(alpha, y_par);
   }

   return nalu_hypre_error_flag;
}
