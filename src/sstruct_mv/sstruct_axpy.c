/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct axpy routine
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPAxpy( NALU_HYPRE_Complex         alpha,
                    nalu_hypre_SStructPVector *px,
                    nalu_hypre_SStructPVector *py )
{
   NALU_HYPRE_Int nvars = nalu_hypre_SStructPVectorNVars(px);
   NALU_HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      nalu_hypre_StructAxpy(alpha,
                       nalu_hypre_SStructPVectorSVector(px, var),
                       nalu_hypre_SStructPVectorSVector(py, var));
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructAxpy( NALU_HYPRE_Complex        alpha,
                   nalu_hypre_SStructVector *x,
                   nalu_hypre_SStructVector *y )
{
   NALU_HYPRE_Int nparts = nalu_hypre_SStructVectorNParts(x);
   NALU_HYPRE_Int part;

   NALU_HYPRE_Int    x_object_type = nalu_hypre_SStructVectorObjectType(x);
   NALU_HYPRE_Int    y_object_type = nalu_hypre_SStructVectorObjectType(y);

   if (x_object_type != y_object_type)
   {
      nalu_hypre_error_in_arg(2);
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   if (x_object_type == NALU_HYPRE_SSTRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         nalu_hypre_SStructPAxpy(alpha,
                            nalu_hypre_SStructVectorPVector(x, part),
                            nalu_hypre_SStructVectorPVector(y, part));
      }
   }

   else if (x_object_type == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_ParVector  *x_par;
      nalu_hypre_ParVector  *y_par;

      nalu_hypre_SStructVectorConvert(x, &x_par);
      nalu_hypre_SStructVectorConvert(y, &y_par);

      nalu_hypre_ParVectorAxpy(alpha, x_par, y_par);
   }

   return nalu_hypre_error_flag;
}
