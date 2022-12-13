/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

/* distributed_matrix.c */
hypre_DistributedMatrix *hypre_DistributedMatrixCreate (MPI_Comm context );
NALU_HYPRE_Int hypre_DistributedMatrixDestroy (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixLimitedDestroy (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixInitialize (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixAssemble (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixSetLocalStorageType (hypre_DistributedMatrix *matrix , NALU_HYPRE_Int type );
NALU_HYPRE_Int hypre_DistributedMatrixGetLocalStorageType (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixSetLocalStorage (hypre_DistributedMatrix *matrix , void *local_storage );
void *hypre_DistributedMatrixGetLocalStorage (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixSetTranslator (hypre_DistributedMatrix *matrix , void *translator );
void *hypre_DistributedMatrixGetTranslator (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixSetAuxiliaryData (hypre_DistributedMatrix *matrix , void *auxiliary_data );
void *hypre_DistributedMatrixGetAuxiliaryData (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixPrint (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixGetLocalRange (hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt *row_start , NALU_HYPRE_BigInt *row_end, NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int hypre_DistributedMatrixGetRow (hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );
NALU_HYPRE_Int hypre_DistributedMatrixRestoreRow (hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );

/* distributed_matrix_ISIS.c */
NALU_HYPRE_Int hypre_InitializeDistributedMatrixISIS(hypre_DistributedMatrix *dm);
NALU_HYPRE_Int hypre_FreeDistributedMatrixISIS( hypre_DistributedMatrix *dm);
NALU_HYPRE_Int hypre_PrintDistributedMatrixISIS( hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_GetDistributedMatrixLocalRangeISIS( hypre_DistributedMatrix *dm, NALU_HYPRE_BigInt *start, NALU_HYPRE_BigInt *end );
NALU_HYPRE_Int hypre_GetDistributedMatrixRowISIS( hypre_DistributedMatrix *dm, NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Real **values );
NALU_HYPRE_Int hypre_RestoreDistributedMatrixRowISIS( hypre_DistributedMatrix *dm, NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Real **values );

/* distributed_matrix_PETSc.c */
NALU_HYPRE_Int hypre_DistributedMatrixDestroyPETSc (hypre_DistributedMatrix *distributed_matrix );
NALU_HYPRE_Int hypre_DistributedMatrixPrintPETSc (hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixGetLocalRangePETSc (hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt *start , NALU_HYPRE_BigInt *end );
NALU_HYPRE_Int hypre_DistributedMatrixGetRowPETSc (hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );
NALU_HYPRE_Int hypre_DistributedMatrixRestoreRowPETSc (hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );

/* distributed_matrix_parcsr.c */
NALU_HYPRE_Int hypre_DistributedMatrixDestroyParCSR ( hypre_DistributedMatrix *distributed_matrix );
NALU_HYPRE_Int hypre_DistributedMatrixInitializeParCSR ( hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixPrintParCSR ( hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int hypre_DistributedMatrixGetLocalRangeParCSR ( hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt *row_start , NALU_HYPRE_BigInt *row_end , NALU_HYPRE_BigInt *col_start , NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int hypre_DistributedMatrixGetRowParCSR ( hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );
NALU_HYPRE_Int hypre_DistributedMatrixRestoreRowParCSR ( hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );
