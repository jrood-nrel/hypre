/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_StructGrid structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_STRUCT_GRID_HEADER
#define nalu_hypre_STRUCT_GRID_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGrid:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_StructGrid_struct
{
   MPI_Comm             comm;

   NALU_HYPRE_Int            ndim;         /* Number of grid dimensions */

   nalu_hypre_BoxArray      *boxes;        /* Array of boxes in this process */
   NALU_HYPRE_Int           *ids;          /* Unique IDs for boxes */
   nalu_hypre_Index          max_distance; /* Neighborhood size - in each dimension*/

   nalu_hypre_Box           *bounding_box; /* Bounding box around grid */

   NALU_HYPRE_Int            local_size;   /* Number of grid points locally */
   NALU_HYPRE_BigInt         global_size;  /* Total number of grid points */

   nalu_hypre_Index          periodic;     /* Indicates if grid is periodic */
   NALU_HYPRE_Int            num_periods;  /* number of box set periods */

   nalu_hypre_Index         *pshifts;      /* shifts of periodicity */


   NALU_HYPRE_Int            ref_count;


   NALU_HYPRE_Int            ghlocal_size; /* Number of vars in box including ghosts */
   NALU_HYPRE_Int            num_ghost[2 * NALU_HYPRE_MAXDIM]; /* ghost layer size */

   nalu_hypre_BoxManager    *boxman;
} nalu_hypre_StructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_StructGrid
 *--------------------------------------------------------------------------*/

#define nalu_hypre_StructGridComm(grid)          ((grid) -> comm)
#define nalu_hypre_StructGridNDim(grid)          ((grid) -> ndim)
#define nalu_hypre_StructGridBoxes(grid)         ((grid) -> boxes)
#define nalu_hypre_StructGridIDs(grid)           ((grid) -> ids)
#define nalu_hypre_StructGridMaxDistance(grid)   ((grid) -> max_distance)
#define nalu_hypre_StructGridBoundingBox(grid)   ((grid) -> bounding_box)
#define nalu_hypre_StructGridLocalSize(grid)     ((grid) -> local_size)
#define nalu_hypre_StructGridGlobalSize(grid)    ((grid) -> global_size)
#define nalu_hypre_StructGridPeriodic(grid)      ((grid) -> periodic)
#define nalu_hypre_StructGridNumPeriods(grid)    ((grid) -> num_periods)
#define nalu_hypre_StructGridPShifts(grid)       ((grid) -> pshifts)
#define nalu_hypre_StructGridPShift(grid, i)     ((grid) -> pshifts[i])
#define nalu_hypre_StructGridRefCount(grid)      ((grid) -> ref_count)
#define nalu_hypre_StructGridGhlocalSize(grid)   ((grid) -> ghlocal_size)
#define nalu_hypre_StructGridNumGhost(grid)      ((grid) -> num_ghost)
#define nalu_hypre_StructGridBoxMan(grid)        ((grid) -> boxman)

#define nalu_hypre_StructGridBox(grid, i)        (nalu_hypre_BoxArrayBox(nalu_hypre_StructGridBoxes(grid), i))
#define nalu_hypre_StructGridNumBoxes(grid)      (nalu_hypre_BoxArraySize(nalu_hypre_StructGridBoxes(grid)))

#define nalu_hypre_StructGridIDPeriod(grid)      nalu_hypre_BoxNeighborsIDPeriod(nalu_hypre_StructGridNeighbors(grid))
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#define nalu_hypre_StructGridDataLocation(grid)  ((grid) -> data_location)
#endif
/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ForStructGridBoxI(i, grid)    nalu_hypre_ForBoxI(i, nalu_hypre_StructGridBoxes(grid))

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#define NALU_HYPRE_MIN_GPU_SIZE                  (131072)
#define nalu_hypre_SetDeviceOn()                 nalu_hypre_HandleStructExecPolicy(nalu_hypre_handle()) = NALU_HYPRE_EXEC_DEVICE
#define nalu_hypre_SetDeviceOff()                nalu_hypre_HandleStructExecPolicy(nalu_hypre_handle()) = NALU_HYPRE_EXEC_HOST
#endif

#endif

