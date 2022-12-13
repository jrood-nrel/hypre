/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Parallel Vector data structure
 *
 *****************************************************************************/

#ifndef hypre_PAR_VECTOR_HEADER
#define hypre_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParVector
 *--------------------------------------------------------------------------*/

#ifndef NALU_HYPRE_PAR_VECTOR_STRUCT
#define NALU_HYPRE_PAR_VECTOR_STRUCT
#endif

typedef struct hypre_ParVector_struct
{
   MPI_Comm              comm;

   NALU_HYPRE_BigInt          global_size;
   NALU_HYPRE_BigInt          first_index;
   NALU_HYPRE_BigInt          last_index;
   NALU_HYPRE_BigInt          partitioning[2];
   /* stores actual length of data in local vector to allow memory
    * manipulations for temporary vectors*/
   NALU_HYPRE_Int             actual_local_size;
   hypre_Vector         *local_vector;

   /* Does the Vector create/destroy `data'? */
   NALU_HYPRE_Int             owns_data;
   /* If the vector is all zeros */
   NALU_HYPRE_Int             all_zeros;

   hypre_IJAssumedPart  *assumed_partition; /* only populated if this partition needed
                                              (for setting off-proc elements, for example)*/
} hypre_ParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_ParVectorComm(vector)             ((vector) -> comm)
#define hypre_ParVectorGlobalSize(vector)       ((vector) -> global_size)
#define hypre_ParVectorFirstIndex(vector)       ((vector) -> first_index)
#define hypre_ParVectorLastIndex(vector)        ((vector) -> last_index)
#define hypre_ParVectorPartitioning(vector)     ((vector) -> partitioning)
#define hypre_ParVectorActualLocalSize(vector)  ((vector) -> actual_local_size)
#define hypre_ParVectorLocalVector(vector)      ((vector) -> local_vector)
#define hypre_ParVectorOwnsData(vector)         ((vector) -> owns_data)
#define hypre_ParVectorAllZeros(vector)         ((vector) -> all_zeros)
#define hypre_ParVectorNumVectors(vector)       (hypre_VectorNumVectors(hypre_ParVectorLocalVector(vector)))

#define hypre_ParVectorAssumedPartition(vector) ((vector) -> assumed_partition)

static inline NALU_HYPRE_MemoryLocation
hypre_ParVectorMemoryLocation(hypre_ParVector *vector)
{
   return hypre_VectorMemoryLocation(hypre_ParVectorLocalVector(vector));
}

#endif
