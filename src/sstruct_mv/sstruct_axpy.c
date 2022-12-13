/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct axpy routine
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructPAxpy( NALU_HYPRE_Complex         alpha,
                    hypre_SStructPVector *px,
                    hypre_SStructPVector *py )
{
   NALU_HYPRE_Int nvars = hypre_SStructPVectorNVars(px);
   NALU_HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructAxpy(alpha,
                       hypre_SStructPVectorSVector(px, var),
                       hypre_SStructPVectorSVector(py, var));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructAxpy( NALU_HYPRE_Complex        alpha,
                   hypre_SStructVector *x,
                   hypre_SStructVector *y )
{
   NALU_HYPRE_Int nparts = hypre_SStructVectorNParts(x);
   NALU_HYPRE_Int part;

   NALU_HYPRE_Int    x_object_type = hypre_SStructVectorObjectType(x);
   NALU_HYPRE_Int    y_object_type = hypre_SStructVectorObjectType(y);

   if (x_object_type != y_object_type)
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (x_object_type == NALU_HYPRE_SSTRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPAxpy(alpha,
                            hypre_SStructVectorPVector(x, part),
                            hypre_SStructVectorPVector(y, part));
      }
   }

   else if (x_object_type == NALU_HYPRE_PARCSR)
   {
      hypre_ParVector  *x_par;
      hypre_ParVector  *y_par;

      hypre_SStructVectorConvert(x, &x_par);
      hypre_SStructVectorConvert(y, &y_par);

      hypre_ParVectorAxpy(alpha, x_par, y_par);
   }

   return hypre_error_flag;
}
