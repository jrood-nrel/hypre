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

#ifndef nalu_hypre_PAR_VECTOR_HEADER
#define nalu_hypre_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVector
 *--------------------------------------------------------------------------*/

#ifndef NALU_HYPRE_PAR_VECTOR_STRUCT
#define NALU_HYPRE_PAR_VECTOR_STRUCT
#endif

typedef struct nalu_hypre_ParVector_struct
{
   MPI_Comm              comm;

   NALU_HYPRE_BigInt          global_size;
   NALU_HYPRE_BigInt          first_index;
   NALU_HYPRE_BigInt          last_index;
   NALU_HYPRE_BigInt          partitioning[2];
   /* stores actual length of data in local vector to allow memory
    * manipulations for temporary vectors*/
   NALU_HYPRE_Int             actual_local_size;
   nalu_hypre_Vector         *local_vector;

   /* Does the Vector create/destroy `data'? */
   NALU_HYPRE_Int             owns_data;
   /* If the vector is all zeros */
   NALU_HYPRE_Int             all_zeros;

   nalu_hypre_IJAssumedPart  *assumed_partition; /* only populated if this partition needed
                                              (for setting off-proc elements, for example)*/
} nalu_hypre_ParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ParVectorComm(vector)             ((vector) -> comm)
#define nalu_hypre_ParVectorGlobalSize(vector)       ((vector) -> global_size)
#define nalu_hypre_ParVectorFirstIndex(vector)       ((vector) -> first_index)
#define nalu_hypre_ParVectorLastIndex(vector)        ((vector) -> last_index)
#define nalu_hypre_ParVectorPartitioning(vector)     ((vector) -> partitioning)
#define nalu_hypre_ParVectorActualLocalSize(vector)  ((vector) -> actual_local_size)
#define nalu_hypre_ParVectorLocalVector(vector)      ((vector) -> local_vector)
#define nalu_hypre_ParVectorOwnsData(vector)         ((vector) -> owns_data)
#define nalu_hypre_ParVectorAllZeros(vector)         ((vector) -> all_zeros)
#define nalu_hypre_ParVectorNumVectors(vector)       (nalu_hypre_VectorNumVectors(nalu_hypre_ParVectorLocalVector(vector)))

#define nalu_hypre_ParVectorAssumedPartition(vector) ((vector) -> assumed_partition)

static inline NALU_HYPRE_MemoryLocation
nalu_hypre_ParVectorMemoryLocation(nalu_hypre_ParVector *vector)
{
   return nalu_hypre_VectorMemoryLocation(nalu_hypre_ParVectorLocalVector(vector));
}

#endif
