/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_IJ_MATRIX_HEADER
#define nalu_hypre_IJ_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_IJMatrix:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_IJMatrix_struct
{
   MPI_Comm      comm;

   NALU_HYPRE_BigInt  row_partitioning[2]; /* distribution of rows across processors */
   NALU_HYPRE_BigInt  col_partitioning[2]; /* distribution of columns */

   NALU_HYPRE_Int     object_type;         /* Indicates the type of "object" */
   void         *object;              /* Structure for storing local portion */
   void         *translator;          /* optional storage_type specific structure
                                         for holding additional local info */
   void         *assumed_part;        /* IJMatrix assumed partition */
   NALU_HYPRE_Int     assemble_flag;       /* indicates whether matrix has been
                                         assembled */

   NALU_HYPRE_BigInt  global_first_row;    /* these four data items are necessary */
   NALU_HYPRE_BigInt  global_first_col;    /* to be able to avoid using the global */
   NALU_HYPRE_BigInt  global_num_rows;     /* global partition */
   NALU_HYPRE_BigInt  global_num_cols;
   NALU_HYPRE_Int     omp_flag;
   NALU_HYPRE_Int     print_level;

} nalu_hypre_IJMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_IJMatrix
 *--------------------------------------------------------------------------*/

#define nalu_hypre_IJMatrixComm(matrix)             ((matrix) -> comm)
#define nalu_hypre_IJMatrixRowPartitioning(matrix)  ((matrix) -> row_partitioning)
#define nalu_hypre_IJMatrixColPartitioning(matrix)  ((matrix) -> col_partitioning)

#define nalu_hypre_IJMatrixObjectType(matrix)       ((matrix) -> object_type)
#define nalu_hypre_IJMatrixObject(matrix)           ((matrix) -> object)
#define nalu_hypre_IJMatrixTranslator(matrix)       ((matrix) -> translator)
#define nalu_hypre_IJMatrixAssumedPart(matrix)      ((matrix) -> assumed_part)

#define nalu_hypre_IJMatrixAssembleFlag(matrix)     ((matrix) -> assemble_flag)

#define nalu_hypre_IJMatrixGlobalFirstRow(matrix)   ((matrix) -> global_first_row)
#define nalu_hypre_IJMatrixGlobalFirstCol(matrix)   ((matrix) -> global_first_col)
#define nalu_hypre_IJMatrixGlobalNumRows(matrix)    ((matrix) -> global_num_rows)
#define nalu_hypre_IJMatrixGlobalNumCols(matrix)    ((matrix) -> global_num_cols)
#define nalu_hypre_IJMatrixOMPFlag(matrix)          ((matrix) -> omp_flag)
#define nalu_hypre_IJMatrixPrintLevel(matrix)       ((matrix) -> print_level)

static inline NALU_HYPRE_MemoryLocation
nalu_hypre_IJMatrixMemoryLocation(nalu_hypre_IJMatrix *matrix)
{
   if ( nalu_hypre_IJMatrixObject(matrix) && nalu_hypre_IJMatrixObjectType(matrix) == NALU_HYPRE_PARCSR)
   {
      return nalu_hypre_ParCSRMatrixMemoryLocation( (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(matrix) );
   }

   return NALU_HYPRE_MEMORY_UNDEFINED;
}

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/

#ifdef PETSC_AVAILABLE
/* IJMatrix_petsc.c */
NALU_HYPRE_Int
nalu_hypre_GetIJMatrixParCSRMatrix( NALU_HYPRE_IJMatrix IJmatrix, Mat *reference )
#endif

#ifdef ISIS_AVAILABLE
/* IJMatrix_isis.c */
NALU_HYPRE_Int
nalu_hypre_GetIJMatrixISISMatrix( NALU_HYPRE_IJMatrix IJmatrix, RowMatrix *reference )
#endif

#endif /* #ifndef nalu_hypre_IJ_MATRIX_HEADER */
