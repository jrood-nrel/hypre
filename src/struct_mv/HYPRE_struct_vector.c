/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorCreate( MPI_Comm             comm,
                          NALU_HYPRE_StructGrid     grid,
                          NALU_HYPRE_StructVector  *vector )
{
   *vector = hypre_StructVectorCreate(comm, grid);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorDestroy( NALU_HYPRE_StructVector struct_vector )
{
   return ( hypre_StructVectorDestroy(struct_vector) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorInitialize( NALU_HYPRE_StructVector vector )
{
   return ( hypre_StructVectorInitialize(vector) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorSetValues( NALU_HYPRE_StructVector  vector,
                             NALU_HYPRE_Int          *grid_index,
                             NALU_HYPRE_Complex       values )
{
   hypre_Index  new_grid_index;

   NALU_HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructVectorSetValues(vector, new_grid_index, &values, 0, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorSetBoxValues( NALU_HYPRE_StructVector  vector,
                                NALU_HYPRE_Int          *ilower,
                                NALU_HYPRE_Int          *iupper,
                                NALU_HYPRE_Complex      *values )
{
   NALU_HYPRE_StructVectorSetBoxValues2(vector, ilower, iupper, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorSetBoxValues2( NALU_HYPRE_StructVector  vector,
                                 NALU_HYPRE_Int          *ilower,
                                 NALU_HYPRE_Int          *iupper,
                                 NALU_HYPRE_Int          *vilower,
                                 NALU_HYPRE_Int          *viupper,
                                 NALU_HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   NALU_HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));
   value_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));

   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructVectorSetBoxValues(vector, set_box, value_box, values, 0, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorAddToValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorAddToValues( NALU_HYPRE_StructVector  vector,
                               NALU_HYPRE_Int          *grid_index,
                               NALU_HYPRE_Complex       values )
{
   hypre_Index  new_grid_index;

   NALU_HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructVectorSetValues(vector, new_grid_index, &values, 1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorAddToBoxValues( NALU_HYPRE_StructVector  vector,
                                  NALU_HYPRE_Int          *ilower,
                                  NALU_HYPRE_Int          *iupper,
                                  NALU_HYPRE_Complex      *values )
{
   NALU_HYPRE_StructVectorAddToBoxValues2(vector, ilower, iupper, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorAddToBoxValues2( NALU_HYPRE_StructVector  vector,
                                   NALU_HYPRE_Int          *ilower,
                                   NALU_HYPRE_Int          *iupper,
                                   NALU_HYPRE_Int          *vilower,
                                   NALU_HYPRE_Int          *viupper,
                                   NALU_HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   NALU_HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));
   value_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));

   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructVectorSetBoxValues(vector, set_box, value_box, values, 1, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorScaleValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorScaleValues( NALU_HYPRE_StructVector  vector,
                               NALU_HYPRE_Complex       factor )
{
   return hypre_StructVectorScaleValues( vector, factor );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorGetValues( NALU_HYPRE_StructVector  vector,
                             NALU_HYPRE_Int          *grid_index,
                             NALU_HYPRE_Complex      *values )
{
   hypre_Index  new_grid_index;

   NALU_HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructVectorSetValues(vector, new_grid_index, values, -1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorGetBoxValues( NALU_HYPRE_StructVector  vector,
                                NALU_HYPRE_Int          *ilower,
                                NALU_HYPRE_Int          *iupper,
                                NALU_HYPRE_Complex      *values )
{
   NALU_HYPRE_StructVectorGetBoxValues2(vector, ilower, iupper, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorGetBoxValues2( NALU_HYPRE_StructVector  vector,
                                 NALU_HYPRE_Int          *ilower,
                                 NALU_HYPRE_Int          *iupper,
                                 NALU_HYPRE_Int          *vilower,
                                 NALU_HYPRE_Int          *viupper,
                                 NALU_HYPRE_Complex      *values )
{
   hypre_Box          *set_box, *value_box;
   NALU_HYPRE_Int           d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));
   value_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));

   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructVectorSetBoxValues(vector, set_box, value_box, values, -1, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorAssemble( NALU_HYPRE_StructVector vector )
{
   return ( hypre_StructVectorAssemble(vector) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorPrint( const char         *filename,
                         NALU_HYPRE_StructVector  vector,
                         NALU_HYPRE_Int           all )
{
   return ( hypre_StructVectorPrint(filename, vector, all) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorRead( MPI_Comm             comm,
                        const char          *filename,
                        NALU_HYPRE_Int           *num_ghost,
                        NALU_HYPRE_StructVector  *vector )
{
   if (!vector)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   *vector = (NALU_HYPRE_StructVector) hypre_StructVectorRead(comm, filename, num_ghost);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorSetNumGhost( NALU_HYPRE_StructVector  vector,
                               NALU_HYPRE_Int          *num_ghost )
{
   return ( hypre_StructVectorSetNumGhost(vector, num_ghost) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorCopy
 * copies data from x to y
 * y has its own data array, so this is a deep copy in that sense.
 * The grid and other size information are not copied - they are
 * assumed to be consistent already.
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_StructVectorCopy( NALU_HYPRE_StructVector x, NALU_HYPRE_StructVector y )
{
   return ( hypre_StructVectorCopy( x, y ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorSetConstantValues( NALU_HYPRE_StructVector  vector,
                                     NALU_HYPRE_Complex       values )
{
   return ( hypre_StructVectorSetConstantValues(vector, values) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorGetMigrateCommPkg( NALU_HYPRE_StructVector  from_vector,
                                     NALU_HYPRE_StructVector  to_vector,
                                     NALU_HYPRE_CommPkg      *comm_pkg )
{
   *comm_pkg = hypre_StructVectorGetMigrateCommPkg(from_vector, to_vector);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorMigrate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorMigrate( NALU_HYPRE_CommPkg      comm_pkg,
                           NALU_HYPRE_StructVector from_vector,
                           NALU_HYPRE_StructVector to_vector )
{
   return ( hypre_StructVectorMigrate( comm_pkg, from_vector, to_vector) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CommPkgDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CommPkgDestroy( NALU_HYPRE_CommPkg comm_pkg )
{
   return ( hypre_CommPkgDestroy(comm_pkg) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorClone
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructVectorClone( NALU_HYPRE_StructVector x,
                         NALU_HYPRE_StructVector *y_ptr )
{
   *y_ptr = hypre_StructVectorClone(x);

   return hypre_error_flag;
}
