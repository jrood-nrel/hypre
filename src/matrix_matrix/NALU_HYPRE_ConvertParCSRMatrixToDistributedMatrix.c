/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Routine for building a DistributedMatrix from a ParCSRMatrix
 *
 *****************************************************************************/

#ifdef NALU_HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include <NALU_HYPRE_config.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"

/* Prototypes for DistributedMatrix */
#include "NALU_HYPRE_distributed_matrix_types.h"
#include "NALU_HYPRE_distributed_matrix_protos.h"

/* Matrix prototypes for ParCSR */
#include "NALU_HYPRE_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ConvertParCSRMatrixToDistributedMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ConvertParCSRMatrixToDistributedMatrix(
   NALU_HYPRE_ParCSRMatrix parcsr_matrix,
   NALU_HYPRE_DistributedMatrix *DistributedMatrix )
{
   MPI_Comm comm;
   NALU_HYPRE_BigInt M, N;

#ifdef NALU_HYPRE_TIMING
   NALU_HYPRE_Int           timer;
   timer = hypre_InitializeTiming( "ConvertParCSRMatrisToDistributedMatrix");
   hypre_BeginTiming( timer );
#endif


   if (!parcsr_matrix)
   {
      hypre_error(NALU_HYPRE_ERROR_ARG);
      return hypre_error_flag;
   }

   NALU_HYPRE_ParCSRMatrixGetComm( parcsr_matrix, &comm);

   NALU_HYPRE_DistributedMatrixCreate( comm, DistributedMatrix );

   NALU_HYPRE_DistributedMatrixSetLocalStorageType( *DistributedMatrix, NALU_HYPRE_PARCSR );

   NALU_HYPRE_DistributedMatrixInitialize( *DistributedMatrix );

   NALU_HYPRE_DistributedMatrixSetLocalStorage( *DistributedMatrix, parcsr_matrix );


   NALU_HYPRE_ParCSRMatrixGetDims( parcsr_matrix, &M, &N);
   NALU_HYPRE_DistributedMatrixSetDims( *DistributedMatrix, M, N);

   NALU_HYPRE_DistributedMatrixAssemble( *DistributedMatrix );

#ifdef NALU_HYPRE_TIMING
   hypre_EndTiming( timer );
   /* hypre_FinalizeTiming( timer ); */
#endif

   return hypre_error_flag;
}

