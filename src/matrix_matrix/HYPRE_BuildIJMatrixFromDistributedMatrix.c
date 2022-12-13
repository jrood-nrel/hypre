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

#include "general.h"

#include "HYPRE.h"
#include "NALU_HYPRE_utilities.h"

/* Prototypes for DistributedMatrix */
#include "NALU_HYPRE_distributed_matrix_types.h"
#include "NALU_HYPRE_distributed_matrix_protos.h"

/* Matrix prototypes for IJMatrix */
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"

/* Local routine prototypes */
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetLocalStorageType(NALU_HYPRE_IJMatrix ij_matrix,
                                            NALU_HYPRE_Int local_storage_type );

NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetLocalSize(NALU_HYPRE_IJMatrix ij_matrix,
                                     NALU_HYPRE_Int row, NALU_HYPRE_Int col );

NALU_HYPRE_Int NALU_HYPRE_IJMatrixInsertRow( NALU_HYPRE_IJMatrix ij_matrix,
                                   NALU_HYPRE_Int size, NALU_HYPRE_BigInt i, NALU_HYPRE_BigInt *col_ind,
                                   NALU_HYPRE_Real *values );

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BuildIJMatrixFromDistributedMatrix
 *--------------------------------------------------------------------------*/
/**
Builds an IJMatrix from a distributed matrix by pulling rows out of the
distributed_matrix and putting them into the IJMatrix. This routine does not
effect the distributed matrix. In essence, it makes a copy of the input matrix
in another format. NOTE: because this routine makes a copy and is not just
a simple conversion, it is memory-expensive and should only be used in
low-memory requirement situations (such as unit-testing code).
*/
NALU_HYPRE_Int
NALU_HYPRE_BuildIJMatrixFromDistributedMatrix(
   NALU_HYPRE_DistributedMatrix DistributedMatrix,
   NALU_HYPRE_IJMatrix *ij_matrix,
   NALU_HYPRE_Int local_storage_type )
{
   NALU_HYPRE_Int ierr;
   MPI_Comm comm;
   NALU_HYPRE_BigInt M, N;
   NALU_HYPRE_BigInt first_local_row, last_local_row;
   NALU_HYPRE_BigInt first_local_col, last_local_col;
   NALU_HYPRE_BigInt i;
   NALU_HYPRE_Int size;
   NALU_HYPRE_BigInt *col_ind;
   NALU_HYPRE_Real *values;



   if (!DistributedMatrix) { return (-1); }

   comm = NALU_HYPRE_DistributedMatrixGetContext( DistributedMatrix );
   ierr = NALU_HYPRE_DistributedMatrixGetDims( DistributedMatrix, &M, &N );

   ierr = NALU_HYPRE_DistributedMatrixGetLocalRange( DistributedMatrix,
                                                &first_local_row, &last_local_row,
                                                &first_local_col, &last_local_col );

   ierr = NALU_HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                first_local_col, last_local_col,
                                ij_matrix );

   ierr = NALU_HYPRE_IJMatrixSetLocalStorageType(
             *ij_matrix, local_storage_type );
   /* if(ierr) return(ierr); */

   ierr = NALU_HYPRE_IJMatrixSetLocalSize( *ij_matrix,
                                      last_local_row - first_local_row + 1,
                                      last_local_col - first_local_col + 1 );

   ierr = NALU_HYPRE_IJMatrixInitialize( *ij_matrix );
   /* if(ierr) return(ierr);*/

   /* Loop through all locally stored rows and insert them into ij_matrix */
   for (i = first_local_row; i <= last_local_row; i++)
   {
      ierr = NALU_HYPRE_DistributedMatrixGetRow( DistributedMatrix, i, &size, &col_ind, &values );
      /* if( ierr ) return(ierr);*/

      ierr = NALU_HYPRE_IJMatrixInsertRow( *ij_matrix, size, i, col_ind, values );
      /* if( ierr ) return(ierr);*/

      ierr = NALU_HYPRE_DistributedMatrixRestoreRow( DistributedMatrix, i, &size, &col_ind, &values );
      /* if( ierr ) return(ierr); */

   }

   ierr = NALU_HYPRE_IJMatrixAssemble( *ij_matrix );
   /* if(ierr) return(ierr); */

   return (ierr);
}

