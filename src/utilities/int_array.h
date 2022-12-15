/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for nalu_hypre_IntArray struct for holding an array of integers
 *
 *****************************************************************************/

#ifndef nalu_hypre_INTARRAY_HEADER
#define nalu_hypre_INTARRAY_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArray
 *--------------------------------------------------------------------------*/

typedef struct
{
   /* pointer to data and size of data */
   NALU_HYPRE_Int            *data;
   NALU_HYPRE_Int             size;

   /* memory location of array data */
   NALU_HYPRE_MemoryLocation  memory_location;
} nalu_hypre_IntArray;

/*--------------------------------------------------------------------------
 * Accessor functions for the IntArray structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_IntArrayData(array)                  ((array) -> data)
#define nalu_hypre_IntArraySize(array)                  ((array) -> size)
#define nalu_hypre_IntArrayMemoryLocation(array)        ((array) -> memory_location)

#endif
