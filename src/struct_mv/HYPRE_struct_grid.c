/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructGrid interface
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructGridCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructGridCreate( MPI_Comm          comm,
                        NALU_HYPRE_Int         dim,
                        NALU_HYPRE_StructGrid *grid )
{
   nalu_hypre_StructGridCreate(comm, dim, grid);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructGridDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructGridDestroy( NALU_HYPRE_StructGrid grid )
{
   return ( nalu_hypre_StructGridDestroy(grid) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructGridSetExtents
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructGridSetExtents( NALU_HYPRE_StructGrid  grid,
                            NALU_HYPRE_Int        *ilower,
                            NALU_HYPRE_Int        *iupper )
{
   nalu_hypre_Index  new_ilower;
   nalu_hypre_Index  new_iupper;

   NALU_HYPRE_Int    d;

   nalu_hypre_SetIndex(new_ilower, 0);
   nalu_hypre_SetIndex(new_iupper, 0);
   for (d = 0; d < nalu_hypre_StructGridNDim((nalu_hypre_StructGrid *) grid); d++)
   {
      nalu_hypre_IndexD(new_ilower, d) = ilower[d];
      nalu_hypre_IndexD(new_iupper, d) = iupper[d];
   }

   return ( nalu_hypre_StructGridSetExtents(grid, new_ilower, new_iupper) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SetStructGridPeriodicity
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructGridSetPeriodic( NALU_HYPRE_StructGrid  grid,
                             NALU_HYPRE_Int        *periodic )
{
   nalu_hypre_Index  new_periodic;

   NALU_HYPRE_Int    d;

   nalu_hypre_SetIndex(new_periodic, 0);
   for (d = 0; d < nalu_hypre_StructGridNDim(grid); d++)
   {
      nalu_hypre_IndexD(new_periodic, d) = periodic[d];
   }

   return ( nalu_hypre_StructGridSetPeriodic(grid, new_periodic) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructGridAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructGridAssemble( NALU_HYPRE_StructGrid grid )
{
   return ( nalu_hypre_StructGridAssemble(grid) );
}

/*---------------------------------------------------------------------------
 * GEC0902
 * NALU_HYPRE_StructGridSetNumGhost
 * to set the numghost array inside the struct_grid_struct using an internal
 * function. This is just a wrapper.
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_StructGridSetNumGhost( NALU_HYPRE_StructGrid grid, NALU_HYPRE_Int *num_ghost )
{
   return ( nalu_hypre_StructGridSetNumGhost(grid, num_ghost) );
}

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int
NALU_HYPRE_StructGridSetDataLocation( NALU_HYPRE_StructGrid grid, NALU_HYPRE_MemoryLocation data_location )
{
   return ( nalu_hypre_StructGridSetDataLocation(grid, data_location) );
}
#endif
