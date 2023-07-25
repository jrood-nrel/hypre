/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   *solver = ( (NALU_HYPRE_StructSolver) nalu_hypre_CyclicReductionCreate( comm ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_CyclicReductionDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedSetup( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_CyclicReductionSetup( (void *) solver,
                                        (nalu_hypre_StructMatrix *) A,
                                        (nalu_hypre_StructVector *) b,
                                        (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedSolve( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_CyclicReduction( (void *) solver,
                                   (nalu_hypre_StructMatrix *) A,
                                   (nalu_hypre_StructVector *) b,
                                   (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedSetTDim( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          tdim )
{
   return ( nalu_hypre_CyclicReductionSetCDir( (void *) solver, tdim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructCycRedSetBase( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          ndim,
                           NALU_HYPRE_Int         *base_index,
                           NALU_HYPRE_Int         *base_stride )
{
   nalu_hypre_Index  new_base_index;
   nalu_hypre_Index  new_base_stride;

   NALU_HYPRE_Int    d;

   nalu_hypre_SetIndex(new_base_index, 0);
   nalu_hypre_SetIndex(new_base_stride, 1);
   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_IndexD(new_base_index, d)  = base_index[d];
      nalu_hypre_IndexD(new_base_stride, d) = base_stride[d];
   }

   return ( nalu_hypre_CyclicReductionSetBase( (void *) solver,
                                          new_base_index, new_base_stride ) );
}

