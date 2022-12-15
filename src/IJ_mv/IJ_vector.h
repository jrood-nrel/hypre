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

#ifndef nalu_hypre_IJ_VECTOR_HEADER
#define nalu_hypre_IJ_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_IJVector:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_IJVector_struct
{
   MPI_Comm      comm;
   NALU_HYPRE_BigInt  partitioning[2];   /* Indicates partitioning over tasks */
   NALU_HYPRE_Int     num_components;    /* Number of components of a multivector */
   NALU_HYPRE_Int     object_type;       /* Indicates the type of "local storage" */
   void         *object;            /* Structure for storing local portion */
   void         *translator;        /* Structure for storing off processor
                                       information */
   void         *assumed_part;      /* IJ Vector assumed partition */
   NALU_HYPRE_BigInt  global_first_row;  /* these for data items are necessary */
   NALU_HYPRE_BigInt  global_num_rows;   /* to be able to avoid using the global partition */
   NALU_HYPRE_Int     print_level;
} nalu_hypre_IJVector;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_IJVector
 *--------------------------------------------------------------------------*/

#define nalu_hypre_IJVectorComm(vector)            ((vector) -> comm)
#define nalu_hypre_IJVectorPartitioning(vector)    ((vector) -> partitioning)
#define nalu_hypre_IJVectorNumComponents(vector)   ((vector) -> num_components)
#define nalu_hypre_IJVectorObjectType(vector)      ((vector) -> object_type)
#define nalu_hypre_IJVectorObject(vector)          ((vector) -> object)
#define nalu_hypre_IJVectorTranslator(vector)      ((vector) -> translator)
#define nalu_hypre_IJVectorAssumedPart(vector)     ((vector) -> assumed_part)
#define nalu_hypre_IJVectorGlobalFirstRow(vector)  ((vector) -> global_first_row)
#define nalu_hypre_IJVectorGlobalNumRows(vector)   ((vector) -> global_num_rows)
#define nalu_hypre_IJVectorPrintLevel(vector)      ((vector) -> print_level)

static inline NALU_HYPRE_MemoryLocation
nalu_hypre_IJVectorMemoryLocation(nalu_hypre_IJVector *vector)
{
   if ( nalu_hypre_IJVectorObject(vector) && nalu_hypre_IJVectorObjectType(vector) == NALU_HYPRE_PARCSR)
   {
      return nalu_hypre_ParVectorMemoryLocation( (nalu_hypre_ParVector *) nalu_hypre_IJVectorObject(vector) );
   }

   return NALU_HYPRE_MEMORY_UNDEFINED;
}

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
/* #include "./internal_protos.h" */

#endif /* #ifndef nalu_hypre_IJ_VECTOR_HEADER */
