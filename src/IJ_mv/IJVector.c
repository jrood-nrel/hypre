/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * nalu_hypre_IJVector interface
 *
 *****************************************************************************/

#include "./_nalu_hypre_IJ_mv.h"

#include "../NALU_HYPRE.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_IJVectorDistribute
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IJVectorDistribute( NALU_HYPRE_IJVector vector, const NALU_HYPRE_Int *vec_starts )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (vec == NULL)
   {
      nalu_hypre_printf("Vector variable is NULL -- nalu_hypre_IJVectorDistribute\n");
      exit(1);
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )

   {
      return ( nalu_hypre_IJVectorDistributePar(vec, vec_starts) );
   }

   else
   {
      nalu_hypre_printf("Unrecognized object type -- nalu_hypre_IJVectorDistribute\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IJVectorZeroValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IJVectorZeroValues( NALU_HYPRE_IJVector vector )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (vec == NULL)
   {
      nalu_hypre_printf("Vector variable is NULL -- nalu_hypre_IJVectorZeroValues\n");
      exit(1);
   }

   /*  if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PETSC )

      return( nalu_hypre_IJVectorZeroValuesPETSc(vec) );

   else if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_ISIS )

      return( nalu_hypre_IJVectorZeroValuesISIS(vec) );

   else */

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
      return ( nalu_hypre_IJVectorZeroValuesPar(vec) );
   }
   else
   {
      nalu_hypre_printf("Unrecognized object type -- nalu_hypre_IJVectorZeroValues\n");
      exit(1);
   }

   return -99;
}
