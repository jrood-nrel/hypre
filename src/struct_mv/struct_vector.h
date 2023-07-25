/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_StructVector structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_STRUCT_VECTOR_HEADER
#define nalu_hypre_STRUCT_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_StructVector:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_StructVector_struct
{
   MPI_Comm              comm;

   nalu_hypre_StructGrid     *grid;

   nalu_hypre_BoxArray       *data_space;

   NALU_HYPRE_MemoryLocation  memory_location;             /* memory location of data */
   NALU_HYPRE_Complex        *data;                        /* Pointer to vector data on device*/
   NALU_HYPRE_Int             data_alloced;                /* Boolean used for freeing data */
   NALU_HYPRE_Int             data_size;                   /* Size of vector data */
   NALU_HYPRE_Int            *data_indices;                /* num-boxes array of indices into
                                                         the data array.  data_indices[b]
                                                         is the starting index of vector
                                                         data corresponding to box b. */

   NALU_HYPRE_Int             num_ghost[2 * NALU_HYPRE_MAXDIM]; /* Num ghost layers in each
                                                       * direction */
   NALU_HYPRE_Int             bghost_not_clear;            /* Are boundary ghosts clear? */

   NALU_HYPRE_BigInt          global_size;                 /* Total number coefficients */

   NALU_HYPRE_Int             ref_count;

} nalu_hypre_StructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_StructVector
 *--------------------------------------------------------------------------*/

#define nalu_hypre_StructVectorComm(vector)           ((vector) -> comm)
#define nalu_hypre_StructVectorGrid(vector)           ((vector) -> grid)
#define nalu_hypre_StructVectorDataSpace(vector)      ((vector) -> data_space)
#define nalu_hypre_StructVectorMemoryLocation(vector) ((vector) -> memory_location)
#define nalu_hypre_StructVectorData(vector)           ((vector) -> data)
#define nalu_hypre_StructVectorDataAlloced(vector)    ((vector) -> data_alloced)
#define nalu_hypre_StructVectorDataSize(vector)       ((vector) -> data_size)
#define nalu_hypre_StructVectorDataIndices(vector)    ((vector) -> data_indices)
#define nalu_hypre_StructVectorNumGhost(vector)       ((vector) -> num_ghost)
#define nalu_hypre_StructVectorBGhostNotClear(vector) ((vector) -> bghost_not_clear)
#define nalu_hypre_StructVectorGlobalSize(vector)     ((vector) -> global_size)
#define nalu_hypre_StructVectorRefCount(vector)       ((vector) -> ref_count)

#define nalu_hypre_StructVectorNDim(vector) \
nalu_hypre_StructGridNDim(nalu_hypre_StructVectorGrid(vector))

#define nalu_hypre_StructVectorBox(vector, b) \
nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(vector), b)

#define nalu_hypre_StructVectorBoxData(vector, b) \
(nalu_hypre_StructVectorData(vector) + nalu_hypre_StructVectorDataIndices(vector)[b])

#define nalu_hypre_StructVectorBoxDataValue(vector, b, index) \
(nalu_hypre_StructVectorBoxData(vector, b) + \
 nalu_hypre_BoxIndexRank(nalu_hypre_StructVectorBox(vector, b), index))

#endif
