/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   *solver = ( (NALU_HYPRE_StructSolver) hypre_CyclicReductionCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( hypre_CyclicReductionDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedSetup( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( hypre_CyclicReductionSetup( (void *) solver,
                                        (hypre_StructMatrix *) A,
                                        (hypre_StructVector *) b,
                                        (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedSolve( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( hypre_CyclicReduction( (void *) solver,
                                   (hypre_StructMatrix *) A,
                                   (hypre_StructVector *) b,
                                   (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedSetTDim( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          tdim )
{
   return ( hypre_CyclicReductionSetCDir( (void *) solver, tdim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedSetBase( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          ndim,
                           NALU_HYPRE_Int         *base_index,
                           NALU_HYPRE_Int         *base_stride )
{
   hypre_Index  new_base_index;
   hypre_Index  new_base_stride;

   NALU_HYPRE_Int    d;

   hypre_SetIndex(new_base_index, 0);
   hypre_SetIndex(new_base_stride, 1);
   for (d = 0; d < ndim; d++)
   {
      hypre_IndexD(new_base_index, d)  = base_index[d];
      hypre_IndexD(new_base_stride, d) = base_stride[d];
   }

   return ( hypre_CyclicReductionSetBase( (void *) solver,
                                          new_base_index, new_base_stride ) );
}

