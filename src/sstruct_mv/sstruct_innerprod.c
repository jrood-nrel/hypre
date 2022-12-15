/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct inner product routine
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPInnerProd( nalu_hypre_SStructPVector *px,
                         nalu_hypre_SStructPVector *py,
                         NALU_HYPRE_Real           *presult_ptr )
{
   NALU_HYPRE_Int    nvars = nalu_hypre_SStructPVectorNVars(px);
   NALU_HYPRE_Real   presult;
   NALU_HYPRE_Real   sresult;
   NALU_HYPRE_Int    var;

   presult = 0.0;
   for (var = 0; var < nvars; var++)
   {
      sresult = nalu_hypre_StructInnerProd(nalu_hypre_SStructPVectorSVector(px, var),
                                      nalu_hypre_SStructPVectorSVector(py, var));
      presult += sresult;
   }

   *presult_ptr = presult;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructInnerProd( nalu_hypre_SStructVector *x,
                        nalu_hypre_SStructVector *y,
                        NALU_HYPRE_Real          *result_ptr )
{
   NALU_HYPRE_Int    nparts = nalu_hypre_SStructVectorNParts(x);
   NALU_HYPRE_Real   result;
   NALU_HYPRE_Real   presult;
   NALU_HYPRE_Int    part;

   NALU_HYPRE_Int    x_object_type = nalu_hypre_SStructVectorObjectType(x);
   NALU_HYPRE_Int    y_object_type = nalu_hypre_SStructVectorObjectType(y);

   if (x_object_type != y_object_type)
   {
      nalu_hypre_error_in_arg(2);
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   result = 0.0;

   if ( (x_object_type == NALU_HYPRE_SSTRUCT) || (x_object_type == NALU_HYPRE_STRUCT) )
   {
      for (part = 0; part < nparts; part++)
      {
         nalu_hypre_SStructPInnerProd(nalu_hypre_SStructVectorPVector(x, part),
                                 nalu_hypre_SStructVectorPVector(y, part), &presult);
         result += presult;
      }
   }

   else if (x_object_type == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_ParVector  *x_par;
      nalu_hypre_ParVector  *y_par;

      nalu_hypre_SStructVectorConvert(x, &x_par);
      nalu_hypre_SStructVectorConvert(y, &y_par);

      result = nalu_hypre_ParVectorInnerProd(x_par, y_par);
   }

   *result_ptr = result;

   return nalu_hypre_error_flag;
}
