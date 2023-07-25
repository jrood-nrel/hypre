/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct copy routine
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPCopy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPCopy( nalu_hypre_SStructPVector *px,
                    nalu_hypre_SStructPVector *py )
{
   NALU_HYPRE_Int nvars = nalu_hypre_SStructPVectorNVars(px);
   NALU_HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      nalu_hypre_StructCopy(nalu_hypre_SStructPVectorSVector(px, var),
                       nalu_hypre_SStructPVectorSVector(py, var));
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPartialPCopy: Copy the components on only a subset of the
 * pgrid. For each box of an sgrid, an array of subboxes are copied.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPartialPCopy( nalu_hypre_SStructPVector *px,
                           nalu_hypre_SStructPVector *py,
                           nalu_hypre_BoxArrayArray **array_boxes )
{
   NALU_HYPRE_Int nvars = nalu_hypre_SStructPVectorNVars(px);
   nalu_hypre_BoxArrayArray  *boxes;
   NALU_HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      boxes = array_boxes[var];
      nalu_hypre_StructPartialCopy(nalu_hypre_SStructPVectorSVector(px, var),
                              nalu_hypre_SStructPVectorSVector(py, var),
                              boxes);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructCopy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructCopy( nalu_hypre_SStructVector *x,
                   nalu_hypre_SStructVector *y )
{
   NALU_HYPRE_Int nparts = nalu_hypre_SStructVectorNParts(x);
   NALU_HYPRE_Int part;

   NALU_HYPRE_Int x_object_type = nalu_hypre_SStructVectorObjectType(x);
   NALU_HYPRE_Int y_object_type = nalu_hypre_SStructVectorObjectType(y);

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
         nalu_hypre_SStructPCopy(nalu_hypre_SStructVectorPVector(x, part),
                            nalu_hypre_SStructVectorPVector(y, part));
      }
   }

   else if (x_object_type == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_ParVector  *x_par;
      nalu_hypre_ParVector  *y_par;

      nalu_hypre_SStructVectorConvert(x, &x_par);
      nalu_hypre_SStructVectorConvert(y, &y_par);

      nalu_hypre_ParVectorCopy(x_par, y_par);
   }

   return nalu_hypre_error_flag;
}
