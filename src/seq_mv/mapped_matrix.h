/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Mapped Matrix data structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_MAPPED_MATRIX_HEADER
#define nalu_hypre_MAPPED_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Mapped Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   void               *matrix;
   NALU_HYPRE_Int         (*ColMap)(NALU_HYPRE_Int, void *);
   void               *MapData;

} nalu_hypre_MappedMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Mapped Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_MappedMatrixMatrix(matrix)           ((matrix) -> matrix)
#define nalu_hypre_MappedMatrixColMap(matrix)           ((matrix) -> ColMap)
#define nalu_hypre_MappedMatrixMapData(matrix)          ((matrix) -> MapData)

#define nalu_hypre_MappedMatrixColIndex(matrix,j) \
         (nalu_hypre_MappedMatrixColMap(matrix)(j,nalu_hypre_MappedMatrixMapData(matrix)))

#endif

