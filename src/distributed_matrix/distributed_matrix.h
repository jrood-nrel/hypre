/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_DistributedMatrix structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_DISTRIBUTED_MATRIX_HEADER
#define nalu_hypre_DISTRIBUTED_MATRIX_HEADER


#include "_nalu_hypre_utilities.h"


/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   NALU_HYPRE_BigInt M, N;                               /* number of rows and cols in matrix */

   void         *auxiliary_data;           /* Placeholder for implmentation specific
                                              data */

   void         *local_storage;            /* Structure for storing local portion */
   NALU_HYPRE_Int   	 local_storage_type;       /* Indicates the type of "local storage" */
   void         *translator;               /* optional storage_type specfic structure
                                              for holding additional local info */
#ifdef NALU_HYPRE_TIMING
   NALU_HYPRE_Int     GetRow_timer;
#endif
} nalu_hypre_DistributedMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_DistributedMatrix
 *--------------------------------------------------------------------------*/

#define nalu_hypre_DistributedMatrixContext(matrix)      ((matrix) -> context)
#define nalu_hypre_DistributedMatrixM(matrix)      ((matrix) -> M)
#define nalu_hypre_DistributedMatrixN(matrix)      ((matrix) -> N)
#define nalu_hypre_DistributedMatrixAuxiliaryData(matrix)         ((matrix) -> auxiliary_data)

#define nalu_hypre_DistributedMatrixLocalStorageType(matrix)  ((matrix) -> local_storage_type)
#define nalu_hypre_DistributedMatrixTranslator(matrix)   ((matrix) -> translator)
#define nalu_hypre_DistributedMatrixLocalStorage(matrix)         ((matrix) -> local_storage)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
#include "NALU_HYPRE_distributed_matrix_mv.h"
#include "internal_protos.h"

#endif
