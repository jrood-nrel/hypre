/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_DistributedMatrix class for par_csr storage scheme.
 *
 *****************************************************************************/

#include "./distributed_matrix.h"

#include "NALU_HYPRE_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixDestroyParCSR
 *   Internal routine for freeing a matrix stored in Parcsr form.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_DistributedMatrixDestroyParCSR( nalu_hypre_DistributedMatrix *distributed_matrix )
{

   return(0);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixInitializeParCSR
 *--------------------------------------------------------------------------*/

  /* matrix must be set before calling this function*/

NALU_HYPRE_Int
nalu_hypre_DistributedMatrixInitializeParCSR(nalu_hypre_DistributedMatrix *matrix)
{

   return 0;
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixPrintParCSR
 *   Internal routine for printing a matrix stored in Parcsr form.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_DistributedMatrixPrintParCSR( nalu_hypre_DistributedMatrix *matrix )
{
   NALU_HYPRE_Int  ierr=0;
   NALU_HYPRE_ParCSRMatrix Parcsr_matrix = (NALU_HYPRE_ParCSRMatrix) nalu_hypre_DistributedMatrixLocalStorage(matrix);

   NALU_HYPRE_ParCSRMatrixPrint( Parcsr_matrix, "STDOUT" );
   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetLocalRangeParCSR
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_DistributedMatrixGetLocalRangeParCSR( nalu_hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt *row_start,
                             NALU_HYPRE_BigInt *row_end,
                             NALU_HYPRE_BigInt *col_start,
                             NALU_HYPRE_BigInt *col_end )
{
   NALU_HYPRE_Int ierr=0;
   NALU_HYPRE_ParCSRMatrix Parcsr_matrix = (NALU_HYPRE_ParCSRMatrix) nalu_hypre_DistributedMatrixLocalStorage(matrix);

   if (!Parcsr_matrix) return(-1);


   ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange( Parcsr_matrix, row_start, row_end,
                                           col_start, col_end );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetRowParCSR
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_DistributedMatrixGetRowParCSR( nalu_hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_ParCSRMatrix Parcsr_matrix = (NALU_HYPRE_ParCSRMatrix) nalu_hypre_DistributedMatrixLocalStorage(matrix);

   if (!Parcsr_matrix) return(-1);

   ierr = NALU_HYPRE_ParCSRMatrixGetRow( Parcsr_matrix, row, size, col_ind, values);

   // RL: if NALU_HYPRE_ParCSRMatrixGetRow was on device, need the next line to guarantee it's done
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncComputeStream(nalu_hypre_handle());
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixRestoreRowParCSR
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_DistributedMatrixRestoreRowParCSR( nalu_hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr;
   NALU_HYPRE_ParCSRMatrix Parcsr_matrix = (NALU_HYPRE_ParCSRMatrix) nalu_hypre_DistributedMatrixLocalStorage(matrix);

   if (Parcsr_matrix == NULL) return(-1);

   ierr = NALU_HYPRE_ParCSRMatrixRestoreRow( Parcsr_matrix, row, size, col_ind, values);

   return(ierr);
}
