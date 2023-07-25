/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_DistributedMatrix structures
 *
 *****************************************************************************/

#ifndef NALU_HYPRE_DISTRIBUTED_MATRIX_MV_HEADER
#define NALU_HYPRE_DISTRIBUTED_MATRIX_MV_HEADER


typedef void *NALU_HYPRE_DistributedMatrix;

/* NALU_HYPRE_distributed_matrix.c */
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixCreate (MPI_Comm context, NALU_HYPRE_DistributedMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixDestroy (NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixLimitedDestroy (NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixInitialize (NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixAssemble (NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixSetLocalStorageType (NALU_HYPRE_DistributedMatrix matrix , NALU_HYPRE_Int type );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixGetLocalStorageType (NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixSetLocalStorage (NALU_HYPRE_DistributedMatrix matrix , void *LocalStorage );
void *NALU_HYPRE_DistributedMatrixGetLocalStorage (NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixSetTranslator (NALU_HYPRE_DistributedMatrix matrix , void *Translator );
void *NALU_HYPRE_DistributedMatrixGetTranslator (NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixSetAuxiliaryData (NALU_HYPRE_DistributedMatrix matrix , void *AuxiliaryData );
void *NALU_HYPRE_DistributedMatrixGetAuxiliaryData (NALU_HYPRE_DistributedMatrix matrix );
MPI_Comm NALU_HYPRE_DistributedMatrixGetContext (NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixGetDims (NALU_HYPRE_DistributedMatrix matrix , NALU_HYPRE_BigInt *M , NALU_HYPRE_BigInt *N );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixSetDims (NALU_HYPRE_DistributedMatrix matrix , NALU_HYPRE_BigInt M , NALU_HYPRE_BigInt N );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPrint (NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixGetLocalRange (NALU_HYPRE_DistributedMatrix matrix , NALU_HYPRE_BigInt *row_start , NALU_HYPRE_BigInt *row_end, NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixGetRow (NALU_HYPRE_DistributedMatrix matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixRestoreRow (NALU_HYPRE_DistributedMatrix matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );

#endif
