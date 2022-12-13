/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixCreate( MPI_Comm             comm,
                          NALU_HYPRE_StructGrid     grid,
                          NALU_HYPRE_StructStencil  stencil,
                          NALU_HYPRE_StructMatrix  *matrix )
{
   *matrix = hypre_StructMatrixCreate(comm, grid, stencil);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixDestroy( NALU_HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixDestroy(matrix) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixInitialize( NALU_HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixInitialize(matrix) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixSetValues( NALU_HYPRE_StructMatrix  matrix,
                             NALU_HYPRE_Int          *grid_index,
                             NALU_HYPRE_Int           num_stencil_indices,
                             NALU_HYPRE_Int          *stencil_indices,
                             NALU_HYPRE_Complex      *values )
{
   hypre_Index  new_grid_index;
   NALU_HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructGridNDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructMatrixSetValues(matrix, new_grid_index,
                               num_stencil_indices, stencil_indices,
                               values, 0, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixGetValues( NALU_HYPRE_StructMatrix  matrix,
                             NALU_HYPRE_Int          *grid_index,
                             NALU_HYPRE_Int           num_stencil_indices,
                             NALU_HYPRE_Int          *stencil_indices,
                             NALU_HYPRE_Complex      *values )
{
   hypre_Index  new_grid_index;
   NALU_HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructGridNDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructMatrixSetValues(matrix, new_grid_index,
                               num_stencil_indices, stencil_indices,
                               values, -1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixSetBoxValues( NALU_HYPRE_StructMatrix  matrix,
                                NALU_HYPRE_Int          *ilower,
                                NALU_HYPRE_Int          *iupper,
                                NALU_HYPRE_Int           num_stencil_indices,
                                NALU_HYPRE_Int          *stencil_indices,
                                NALU_HYPRE_Complex      *values )
{
   NALU_HYPRE_StructMatrixSetBoxValues2(matrix, ilower, iupper, num_stencil_indices,
                                   stencil_indices, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixGetBoxValues( NALU_HYPRE_StructMatrix  matrix,
                                NALU_HYPRE_Int          *ilower,
                                NALU_HYPRE_Int          *iupper,
                                NALU_HYPRE_Int           num_stencil_indices,
                                NALU_HYPRE_Int          *stencil_indices,
                                NALU_HYPRE_Complex      *values )
{
   NALU_HYPRE_StructMatrixGetBoxValues2(matrix, ilower, iupper, num_stencil_indices,
                                   stencil_indices, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixSetBoxValues2( NALU_HYPRE_StructMatrix  matrix,
                                 NALU_HYPRE_Int          *ilower,
                                 NALU_HYPRE_Int          *iupper,
                                 NALU_HYPRE_Int           num_stencil_indices,
                                 NALU_HYPRE_Int          *stencil_indices,
                                 NALU_HYPRE_Int          *vilower,
                                 NALU_HYPRE_Int          *viupper,
                                 NALU_HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   NALU_HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));
   value_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));

   for (d = 0; d < hypre_StructMatrixNDim(matrix); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructMatrixSetBoxValues(matrix, set_box, value_box,
                                  num_stencil_indices, stencil_indices,
                                  values, 0, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixGetBoxValues2( NALU_HYPRE_StructMatrix  matrix,
                                 NALU_HYPRE_Int          *ilower,
                                 NALU_HYPRE_Int          *iupper,
                                 NALU_HYPRE_Int           num_stencil_indices,
                                 NALU_HYPRE_Int          *stencil_indices,
                                 NALU_HYPRE_Int          *vilower,
                                 NALU_HYPRE_Int          *viupper,
                                 NALU_HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   NALU_HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));
   value_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));

   for (d = 0; d < hypre_StructMatrixNDim(matrix); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructMatrixSetBoxValues(matrix, set_box, value_box,
                                  num_stencil_indices, stencil_indices,
                                  values, -1, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixSetConstantValues( NALU_HYPRE_StructMatrix matrix,
                                     NALU_HYPRE_Int          num_stencil_indices,
                                     NALU_HYPRE_Int         *stencil_indices,
                                     NALU_HYPRE_Complex     *values )
{
   return hypre_StructMatrixSetConstantValues(
             matrix, num_stencil_indices, stencil_indices, values, 0 );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixAddToValues( NALU_HYPRE_StructMatrix  matrix,
                               NALU_HYPRE_Int          *grid_index,
                               NALU_HYPRE_Int           num_stencil_indices,
                               NALU_HYPRE_Int          *stencil_indices,
                               NALU_HYPRE_Complex      *values )
{
   hypre_Index         new_grid_index;
   NALU_HYPRE_Int           d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructGridNDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructMatrixSetValues(matrix, new_grid_index,
                               num_stencil_indices, stencil_indices,
                               values, 1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixAddToBoxValues( NALU_HYPRE_StructMatrix  matrix,
                                  NALU_HYPRE_Int          *ilower,
                                  NALU_HYPRE_Int          *iupper,
                                  NALU_HYPRE_Int           num_stencil_indices,
                                  NALU_HYPRE_Int          *stencil_indices,
                                  NALU_HYPRE_Complex      *values )
{
   NALU_HYPRE_StructMatrixAddToBoxValues2(matrix, ilower, iupper, num_stencil_indices,
                                     stencil_indices, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixAddToBoxValues2( NALU_HYPRE_StructMatrix  matrix,
                                   NALU_HYPRE_Int          *ilower,
                                   NALU_HYPRE_Int          *iupper,
                                   NALU_HYPRE_Int           num_stencil_indices,
                                   NALU_HYPRE_Int          *stencil_indices,
                                   NALU_HYPRE_Int          *vilower,
                                   NALU_HYPRE_Int          *viupper,
                                   NALU_HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   NALU_HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));
   value_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));

   for (d = 0; d < hypre_StructMatrixNDim(matrix); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructMatrixSetBoxValues(matrix, set_box, value_box,
                                  num_stencil_indices, stencil_indices,
                                  values, 1, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixAddToConstantValues( NALU_HYPRE_StructMatrix matrix,
                                       NALU_HYPRE_Int          num_stencil_indices,
                                       NALU_HYPRE_Int         *stencil_indices,
                                       NALU_HYPRE_Complex     *values )
{
   return hypre_StructMatrixSetConstantValues(
             matrix, num_stencil_indices, stencil_indices, values, 1 );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixAssemble( NALU_HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixAssemble(matrix) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixSetNumGhost( NALU_HYPRE_StructMatrix  matrix,
                               NALU_HYPRE_Int          *num_ghost )
{
   return ( hypre_StructMatrixSetNumGhost(matrix, num_ghost) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixGetGrid( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_StructGrid *grid )
{
   *grid = hypre_StructMatrixGrid(matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixSetSymmetric( NALU_HYPRE_StructMatrix  matrix,
                                NALU_HYPRE_Int           symmetric )
{
   hypre_StructMatrixSymmetric(matrix) = symmetric;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Call this function to declare that certain stencil points are constant
 * throughout the mesh.
 * - nentries is the number of array entries
 * - Each NALU_HYPRE_Int entries[i] is an index into the shape array of the stencil of the
 * matrix.
 * In the present version, only three possibilites are recognized:
 * - no entries constant                 (constant_coefficient==0)
 * - all entries constant                (constant_coefficient==1)
 * - all but the diagonal entry constant (constant_coefficient==2)
 * If something else is attempted, this function will return a nonzero error.
 * In the present version, if this function is called more than once, only
 * the last call will take effect.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int  NALU_HYPRE_StructMatrixSetConstantEntries( NALU_HYPRE_StructMatrix  matrix,
                                                 NALU_HYPRE_Int           nentries,
                                                 NALU_HYPRE_Int          *entries )
{
   return hypre_StructMatrixSetConstantEntries( matrix, nentries, entries );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixPrint( const char         *filename,
                         NALU_HYPRE_StructMatrix  matrix,
                         NALU_HYPRE_Int           all )
{
   return ( hypre_StructMatrixPrint(filename, matrix, all) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixRead( MPI_Comm             comm,
                        const char          *filename,
                        NALU_HYPRE_Int           *num_ghost,
                        NALU_HYPRE_StructMatrix  *matrix )
{
   if (!matrix)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   *matrix = (NALU_HYPRE_StructMatrix) hypre_StructMatrixRead(comm, filename, num_ghost);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixMatvec( NALU_HYPRE_Complex      alpha,
                          NALU_HYPRE_StructMatrix A,
                          NALU_HYPRE_StructVector x,
                          NALU_HYPRE_Complex      beta,
                          NALU_HYPRE_StructVector y     )
{
   return ( hypre_StructMatvec( alpha, (hypre_StructMatrix *) A,
                                (hypre_StructVector *) x, beta,
                                (hypre_StructVector *) y) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructMatrixClearBoundary( NALU_HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixClearBoundary(matrix) );
}
