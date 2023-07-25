/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
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
nalu_hypre_DistributedMatrix *nalu_hypre_DistributedMatrixCreate (MPI_Comm context );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixDestroy (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixLimitedDestroy (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixInitialize (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixAssemble (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixSetLocalStorageType (nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_Int type );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixGetLocalStorageType (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixSetLocalStorage (nalu_hypre_DistributedMatrix *matrix , void *local_storage );
void *nalu_hypre_DistributedMatrixGetLocalStorage (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixSetTranslator (nalu_hypre_DistributedMatrix *matrix , void *translator );
void *nalu_hypre_DistributedMatrixGetTranslator (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixSetAuxiliaryData (nalu_hypre_DistributedMatrix *matrix , void *auxiliary_data );
void *nalu_hypre_DistributedMatrixGetAuxiliaryData (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixPrint (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixGetLocalRange (nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt *row_start , NALU_HYPRE_BigInt *row_end, NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixGetRow (nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixRestoreRow (nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );

/* distributed_matrix_ISIS.c */
NALU_HYPRE_Int nalu_hypre_InitializeDistributedMatrixISIS(nalu_hypre_DistributedMatrix *dm);
NALU_HYPRE_Int nalu_hypre_FreeDistributedMatrixISIS( nalu_hypre_DistributedMatrix *dm);
NALU_HYPRE_Int nalu_hypre_PrintDistributedMatrixISIS( nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_GetDistributedMatrixLocalRangeISIS( nalu_hypre_DistributedMatrix *dm, NALU_HYPRE_BigInt *start, NALU_HYPRE_BigInt *end );
NALU_HYPRE_Int nalu_hypre_GetDistributedMatrixRowISIS( nalu_hypre_DistributedMatrix *dm, NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Real **values );
NALU_HYPRE_Int nalu_hypre_RestoreDistributedMatrixRowISIS( nalu_hypre_DistributedMatrix *dm, NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Real **values );

/* distributed_matrix_PETSc.c */
NALU_HYPRE_Int nalu_hypre_DistributedMatrixDestroyPETSc (nalu_hypre_DistributedMatrix *distributed_matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixPrintPETSc (nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixGetLocalRangePETSc (nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt *start , NALU_HYPRE_BigInt *end );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixGetRowPETSc (nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixRestoreRowPETSc (nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );

/* distributed_matrix_parcsr.c */
NALU_HYPRE_Int nalu_hypre_DistributedMatrixDestroyParCSR ( nalu_hypre_DistributedMatrix *distributed_matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixInitializeParCSR ( nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixPrintParCSR ( nalu_hypre_DistributedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixGetLocalRangeParCSR ( nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt *row_start , NALU_HYPRE_BigInt *row_end , NALU_HYPRE_BigInt *col_start , NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixGetRowParCSR ( nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );
NALU_HYPRE_Int nalu_hypre_DistributedMatrixRestoreRowParCSR ( nalu_hypre_DistributedMatrix *matrix , NALU_HYPRE_BigInt row , NALU_HYPRE_Int *size , NALU_HYPRE_BigInt **col_ind , NALU_HYPRE_Real **values );
