/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* csr_matop.c */
NALU_HYPRE_Int nalu_hypre_CSRMatrixAddFirstPass ( NALU_HYPRE_Int firstrow, NALU_HYPRE_Int lastrow, NALU_HYPRE_Int *marker,
                                        NALU_HYPRE_Int *twspace, NALU_HYPRE_Int *map_A2C, NALU_HYPRE_Int *map_B2C, nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B,
                                        NALU_HYPRE_Int nnzrows_C, NALU_HYPRE_Int nrows_C, NALU_HYPRE_Int ncols_C, NALU_HYPRE_Int *rownnz_C,
                                        NALU_HYPRE_MemoryLocation memory_location_C, NALU_HYPRE_Int *C_i, nalu_hypre_CSRMatrix **C_ptr );
NALU_HYPRE_Int nalu_hypre_CSRMatrixAddSecondPass ( NALU_HYPRE_Int firstrow, NALU_HYPRE_Int lastrow, NALU_HYPRE_Int *marker,
                                         NALU_HYPRE_Int *twspace, NALU_HYPRE_Int *map_A2C, NALU_HYPRE_Int *map_B2C, NALU_HYPRE_Int *rownnz_C,
                                         NALU_HYPRE_Complex alpha, NALU_HYPRE_Complex beta, nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B,
                                         nalu_hypre_CSRMatrix *C);
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixAddHost ( NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                                          NALU_HYPRE_Complex beta, nalu_hypre_CSRMatrix *B );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixAdd ( NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A, NALU_HYPRE_Complex beta,
                                      nalu_hypre_CSRMatrix *B );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixBigAdd ( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixMultiplyHost ( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixMultiply ( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixDeleteZeros ( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_CSRMatrixTransposeHost ( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix **AT, NALU_HYPRE_Int data );
NALU_HYPRE_Int nalu_hypre_CSRMatrixTranspose ( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix **AT, NALU_HYPRE_Int data );
NALU_HYPRE_Int nalu_hypre_CSRMatrixReorder ( nalu_hypre_CSRMatrix *A );
NALU_HYPRE_Complex nalu_hypre_CSRMatrixSumElts ( nalu_hypre_CSRMatrix *A );
NALU_HYPRE_Real nalu_hypre_CSRMatrixFnorm( nalu_hypre_CSRMatrix *A );
NALU_HYPRE_Int nalu_hypre_CSRMatrixSplit(nalu_hypre_CSRMatrix *Bs_ext, NALU_HYPRE_BigInt first_col_diag_B,
                               NALU_HYPRE_BigInt last_col_diag_B, NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_BigInt *col_map_offd_B,
                               NALU_HYPRE_Int *num_cols_offd_C_ptr, NALU_HYPRE_BigInt **col_map_offd_C_ptr, nalu_hypre_CSRMatrix **Bext_diag_ptr,
                               nalu_hypre_CSRMatrix **Bext_offd_ptr);
nalu_hypre_CSRMatrix * nalu_hypre_CSRMatrixAddPartial( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B,
                                             NALU_HYPRE_Int *row_nums);
void nalu_hypre_CSRMatrixComputeRowSumHost( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int *CF_i, NALU_HYPRE_Int *CF_j,
                                       NALU_HYPRE_Complex *row_sum, NALU_HYPRE_Int type, NALU_HYPRE_Complex scal, const char *set_or_add);
void nalu_hypre_CSRMatrixComputeRowSum( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int *CF_i, NALU_HYPRE_Int *CF_j,
                                   NALU_HYPRE_Complex *row_sum, NALU_HYPRE_Int type, NALU_HYPRE_Complex scal, const char *set_or_add);
void nalu_hypre_CSRMatrixExtractDiagonal( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Complex *d, NALU_HYPRE_Int type);
void nalu_hypre_CSRMatrixExtractDiagonalHost( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Complex *d, NALU_HYPRE_Int type);
NALU_HYPRE_Int nalu_hypre_CSRMatrixScale(nalu_hypre_CSRMatrix *A, NALU_HYPRE_Complex scalar);
NALU_HYPRE_Int nalu_hypre_CSRMatrixSetConstantValues( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Complex value);
NALU_HYPRE_Int nalu_hypre_CSRMatrixDiagScale( nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *ld, nalu_hypre_Vector *rd);

/* csr_matop_device.c */
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixAddDevice ( NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                                            NALU_HYPRE_Complex beta, nalu_hypre_CSRMatrix *B );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixMultiplyDevice ( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixTripleMultiplyDevice ( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B,
                                                       nalu_hypre_CSRMatrix *C );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMergeColMapOffd( NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_BigInt *col_map_offd_B,
                                          NALU_HYPRE_Int B_ext_offd_nnz, NALU_HYPRE_BigInt *B_ext_offd_bigj, NALU_HYPRE_Int *num_cols_offd_C_ptr,
                                          NALU_HYPRE_BigInt **col_map_offd_C_ptr, NALU_HYPRE_Int **map_B_to_C_ptr );
NALU_HYPRE_Int nalu_hypre_CSRMatrixSplitDevice_core( NALU_HYPRE_Int job, NALU_HYPRE_Int num_rows, NALU_HYPRE_Int B_ext_nnz,
                                           NALU_HYPRE_Int *B_ext_ii, NALU_HYPRE_BigInt *B_ext_bigj, NALU_HYPRE_Complex *B_ext_data, char *B_ext_xata,
                                           NALU_HYPRE_BigInt first_col_diag_B, NALU_HYPRE_BigInt last_col_diag_B, NALU_HYPRE_Int num_cols_offd_B,
                                           NALU_HYPRE_BigInt *col_map_offd_B, NALU_HYPRE_Int **map_B_to_C_ptr, NALU_HYPRE_Int *num_cols_offd_C_ptr,
                                           NALU_HYPRE_BigInt **col_map_offd_C_ptr, NALU_HYPRE_Int *B_ext_diag_nnz_ptr, NALU_HYPRE_Int *B_ext_diag_ii,
                                           NALU_HYPRE_Int *B_ext_diag_j, NALU_HYPRE_Complex *B_ext_diag_data, char *B_ext_diag_xata,
                                           NALU_HYPRE_Int *B_ext_offd_nnz_ptr, NALU_HYPRE_Int *B_ext_offd_ii, NALU_HYPRE_Int *B_ext_offd_j,
                                           NALU_HYPRE_Complex *B_ext_offd_data, char *B_ext_offd_xata );
NALU_HYPRE_Int nalu_hypre_CSRMatrixSplitDevice(nalu_hypre_CSRMatrix *B_ext, NALU_HYPRE_BigInt first_col_diag_B,
                                     NALU_HYPRE_BigInt last_col_diag_B, NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_BigInt *col_map_offd_B,
                                     NALU_HYPRE_Int **map_B_to_C_ptr, NALU_HYPRE_Int *num_cols_offd_C_ptr, NALU_HYPRE_BigInt **col_map_offd_C_ptr,
                                     nalu_hypre_CSRMatrix **B_ext_diag_ptr, nalu_hypre_CSRMatrix **B_ext_offd_ptr);
NALU_HYPRE_Int nalu_hypre_CSRMatrixTransposeDevice ( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix **AT,
                                           NALU_HYPRE_Int data );
nalu_hypre_CSRMatrix* nalu_hypre_CSRMatrixAddPartialDevice( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B,
                                                  NALU_HYPRE_Int *row_nums);
NALU_HYPRE_Int nalu_hypre_CSRMatrixColNNzRealDevice( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real *colnnz);
NALU_HYPRE_Int nalu_hypre_CSRMatrixMoveDiagFirstDevice( nalu_hypre_CSRMatrix  *A );
NALU_HYPRE_Int nalu_hypre_CSRMatrixCheckDiagFirstDevice( nalu_hypre_CSRMatrix  *A );
NALU_HYPRE_Int nalu_hypre_CSRMatrixReplaceDiagDevice( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Complex *new_diag,
                                            NALU_HYPRE_Complex v, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_CSRMatrixComputeRowSumDevice( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int *CF_i, NALU_HYPRE_Int *CF_j,
                                              NALU_HYPRE_Complex *row_sum, NALU_HYPRE_Int type,
                                              NALU_HYPRE_Complex scal, const char *set_or_add );
NALU_HYPRE_Int nalu_hypre_CSRMatrixExtractDiagonalDevice( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Complex *d,
                                                NALU_HYPRE_Int type );
nalu_hypre_CSRMatrix* nalu_hypre_CSRMatrixStack2Device(nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B);
nalu_hypre_CSRMatrix* nalu_hypre_CSRMatrixIdentityDevice(NALU_HYPRE_Int n, NALU_HYPRE_Complex alp);
nalu_hypre_CSRMatrix* nalu_hypre_CSRMatrixDiagMatrixFromVectorDevice(NALU_HYPRE_Int n, NALU_HYPRE_Complex *v);
nalu_hypre_CSRMatrix* nalu_hypre_CSRMatrixDiagMatrixFromMatrixDevice(nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int type);
NALU_HYPRE_Int nalu_hypre_CSRMatrixRemoveDiagonalDevice(nalu_hypre_CSRMatrix *A);
NALU_HYPRE_Int nalu_hypre_CSRMatrixDropSmallEntriesDevice( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real tol,
                                                 NALU_HYPRE_Real *elmt_tols);
NALU_HYPRE_Int nalu_hypre_CSRMatrixPermuteDevice( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int *perm,
                                        NALU_HYPRE_Int *rqperm, nalu_hypre_CSRMatrix *B );
NALU_HYPRE_Int nalu_hypre_CSRMatrixSortRow(nalu_hypre_CSRMatrix *A);
NALU_HYPRE_Int nalu_hypre_CSRMatrixSortRowOutOfPlace(nalu_hypre_CSRMatrix *A);
NALU_HYPRE_Int nalu_hypre_CSRMatrixTriLowerUpperSolveDevice_core(char uplo, NALU_HYPRE_Int unit_diag,
                                                       nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real *l1_norms, nalu_hypre_Vector *f, NALU_HYPRE_Int offset_f, nalu_hypre_Vector *u,
                                                       NALU_HYPRE_Int offset_u);
NALU_HYPRE_Int nalu_hypre_CSRMatrixTriLowerUpperSolveDevice(char uplo, NALU_HYPRE_Int unit_diag,
                                                  nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real *l1_norms, nalu_hypre_Vector *f, nalu_hypre_Vector *u );
NALU_HYPRE_Int nalu_hypre_CSRMatrixTriLowerUpperSolveRocsparse(char uplo, NALU_HYPRE_Int unit_diag,
                                                     nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real *l1_norms, NALU_HYPRE_Complex *f, NALU_HYPRE_Complex *u );
NALU_HYPRE_Int nalu_hypre_CSRMatrixTriLowerUpperSolveCusparse(char uplo, NALU_HYPRE_Int unit_diag,
                                                    nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real *l1_norms, NALU_HYPRE_Complex *f, NALU_HYPRE_Complex *u );
NALU_HYPRE_Int nalu_hypre_CSRMatrixIntersectPattern(nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B, NALU_HYPRE_Int *markA,
                                          NALU_HYPRE_Int diag_option);
NALU_HYPRE_Int nalu_hypre_CSRMatrixDiagScaleDevice( nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *ld, nalu_hypre_Vector *rd);
NALU_HYPRE_Int nalu_hypre_CSRMatrixCompressColumnsDevice(nalu_hypre_CSRMatrix *A, NALU_HYPRE_BigInt *col_map,
                                               NALU_HYPRE_Int **col_idx_new_ptr, NALU_HYPRE_BigInt **col_map_new_ptr);
NALU_HYPRE_Int nalu_hypre_CSRMatrixILU0(nalu_hypre_CSRMatrix *A);

/* csr_matrix.c */
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixCreate ( NALU_HYPRE_Int num_rows, NALU_HYPRE_Int num_cols,
                                         NALU_HYPRE_Int num_nonzeros );
NALU_HYPRE_Int nalu_hypre_CSRMatrixDestroy ( nalu_hypre_CSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_CSRMatrixInitialize_v2( nalu_hypre_CSRMatrix *matrix, NALU_HYPRE_Int bigInit,
                                        NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int nalu_hypre_CSRMatrixInitialize ( nalu_hypre_CSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_CSRMatrixBigInitialize ( nalu_hypre_CSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_CSRMatrixBigJtoJ ( nalu_hypre_CSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_CSRMatrixJtoBigJ ( nalu_hypre_CSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_CSRMatrixSetDataOwner ( nalu_hypre_CSRMatrix *matrix, NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int nalu_hypre_CSRMatrixSetPatternOnly( nalu_hypre_CSRMatrix *matrix, NALU_HYPRE_Int pattern_only );
NALU_HYPRE_Int nalu_hypre_CSRMatrixSetRownnz ( nalu_hypre_CSRMatrix *matrix );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixRead ( char *file_name );
NALU_HYPRE_Int nalu_hypre_CSRMatrixPrint ( nalu_hypre_CSRMatrix *matrix, const char *file_name );
NALU_HYPRE_Int nalu_hypre_CSRMatrixPrintIJ( nalu_hypre_CSRMatrix *matrix, NALU_HYPRE_Int base_i,
                                  NALU_HYPRE_Int base_j, char *filename );
NALU_HYPRE_Int nalu_hypre_CSRMatrixPrintHB ( nalu_hypre_CSRMatrix *matrix_input, char *file_name );
NALU_HYPRE_Int nalu_hypre_CSRMatrixPrintMM( nalu_hypre_CSRMatrix *matrix, NALU_HYPRE_Int basei, NALU_HYPRE_Int basej,
                                  NALU_HYPRE_Int trans, const char *file_name );
NALU_HYPRE_Int nalu_hypre_CSRMatrixCopy ( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B, NALU_HYPRE_Int copy_data );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMigrate( nalu_hypre_CSRMatrix *A, NALU_HYPRE_MemoryLocation memory_location );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixClone ( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int copy_data );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixClone_v2( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int copy_data,
                                          NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int nalu_hypre_CSRMatrixPermute( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int *perm,
                                  NALU_HYPRE_Int *rqperm, nalu_hypre_CSRMatrix **B_ptr );
nalu_hypre_CSRMatrix *nalu_hypre_CSRMatrixUnion( nalu_hypre_CSRMatrix *A,
                                       nalu_hypre_CSRMatrix *B,
                                       NALU_HYPRE_BigInt *col_map_offd_A,
                                       NALU_HYPRE_BigInt *col_map_offd_B,
                                       NALU_HYPRE_BigInt **col_map_offd_C );
NALU_HYPRE_Int nalu_hypre_CSRMatrixPrefetch( nalu_hypre_CSRMatrix *A, NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int nalu_hypre_CSRMatrixCheckSetNumNonzeros( nalu_hypre_CSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_CSRMatrixResize( nalu_hypre_CSRMatrix *matrix, NALU_HYPRE_Int new_num_rows,
                                 NALU_HYPRE_Int new_num_cols, NALU_HYPRE_Int new_num_nonzeros );

/* csr_matvec.c */
// y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end]
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvecOutOfPlace ( NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                                            nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *b, nalu_hypre_Vector *y, NALU_HYPRE_Int offset );
// y = alpha*A + beta*y
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvec ( NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *x,
                                  NALU_HYPRE_Complex beta, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvecT ( NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *x,
                                   NALU_HYPRE_Complex beta, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvec_FF ( NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *x,
                                     NALU_HYPRE_Complex beta, nalu_hypre_Vector *y, NALU_HYPRE_Int *CF_marker_x, NALU_HYPRE_Int *CF_marker_y,
                                     NALU_HYPRE_Int fpt );

/* csr_matvec_device.c */
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvecDevice(NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                                      nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *b, nalu_hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvecCusparseNewAPI( NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha,
                                               nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvecCusparseOldAPI( NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha,
                                               nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvecCusparse( NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha,
                                         nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvecOMPOffload (NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                                           nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvecRocsparse (NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                                          nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int nalu_hypre_CSRMatrixMatvecOnemklsparse (NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha,
                                             nalu_hypre_CSRMatrix *A,
                                             nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *y, NALU_HYPRE_Int offset );

/* genpart.c */
NALU_HYPRE_Int nalu_hypre_GeneratePartitioning ( NALU_HYPRE_BigInt length, NALU_HYPRE_Int num_procs,
                                       NALU_HYPRE_BigInt **part_ptr );
NALU_HYPRE_Int nalu_hypre_GenerateLocalPartitioning ( NALU_HYPRE_BigInt length, NALU_HYPRE_Int num_procs,
                                            NALU_HYPRE_Int myid, NALU_HYPRE_BigInt *part );

/* NALU_HYPRE_csr_matrix.c */
NALU_HYPRE_CSRMatrix NALU_HYPRE_CSRMatrixCreate ( NALU_HYPRE_Int num_rows, NALU_HYPRE_Int num_cols,
                                        NALU_HYPRE_Int *row_sizes );
NALU_HYPRE_Int NALU_HYPRE_CSRMatrixDestroy ( NALU_HYPRE_CSRMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_CSRMatrixInitialize ( NALU_HYPRE_CSRMatrix matrix );
NALU_HYPRE_CSRMatrix NALU_HYPRE_CSRMatrixRead ( char *file_name );
void NALU_HYPRE_CSRMatrixPrint ( NALU_HYPRE_CSRMatrix matrix, char *file_name );
NALU_HYPRE_Int NALU_HYPRE_CSRMatrixGetNumRows ( NALU_HYPRE_CSRMatrix matrix, NALU_HYPRE_Int *num_rows );

/* NALU_HYPRE_mapped_matrix.c */
NALU_HYPRE_MappedMatrix NALU_HYPRE_MappedMatrixCreate ( void );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixDestroy ( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixLimitedDestroy ( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixInitialize ( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixAssemble ( NALU_HYPRE_MappedMatrix matrix );
void NALU_HYPRE_MappedMatrixPrint ( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixGetColIndex ( NALU_HYPRE_MappedMatrix matrix, NALU_HYPRE_Int j );
void *NALU_HYPRE_MappedMatrixGetMatrix ( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixSetMatrix ( NALU_HYPRE_MappedMatrix matrix, void *matrix_data );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixSetColMap ( NALU_HYPRE_MappedMatrix matrix, NALU_HYPRE_Int (*ColMap )(NALU_HYPRE_Int,
                                                                                        void *));
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixSetMapData ( NALU_HYPRE_MappedMatrix matrix, void *MapData );

/* NALU_HYPRE_multiblock_matrix.c */
NALU_HYPRE_MultiblockMatrix NALU_HYPRE_MultiblockMatrixCreate ( void );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixDestroy ( NALU_HYPRE_MultiblockMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixLimitedDestroy ( NALU_HYPRE_MultiblockMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixInitialize ( NALU_HYPRE_MultiblockMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixAssemble ( NALU_HYPRE_MultiblockMatrix matrix );
void NALU_HYPRE_MultiblockMatrixPrint ( NALU_HYPRE_MultiblockMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixSetNumSubmatrices ( NALU_HYPRE_MultiblockMatrix matrix, NALU_HYPRE_Int n );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixSetSubmatrixType ( NALU_HYPRE_MultiblockMatrix matrix, NALU_HYPRE_Int j,
                                                   NALU_HYPRE_Int type );

/* NALU_HYPRE_vector.c */
NALU_HYPRE_Vector NALU_HYPRE_VectorCreate ( NALU_HYPRE_Int size );
NALU_HYPRE_Int NALU_HYPRE_VectorDestroy ( NALU_HYPRE_Vector vector );
NALU_HYPRE_Int NALU_HYPRE_VectorInitialize ( NALU_HYPRE_Vector vector );
NALU_HYPRE_Int NALU_HYPRE_VectorPrint ( NALU_HYPRE_Vector vector, char *file_name );
NALU_HYPRE_Vector NALU_HYPRE_VectorRead ( char *file_name );

/* mapped_matrix.c */
nalu_hypre_MappedMatrix *nalu_hypre_MappedMatrixCreate ( void );
NALU_HYPRE_Int nalu_hypre_MappedMatrixDestroy ( nalu_hypre_MappedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_MappedMatrixLimitedDestroy ( nalu_hypre_MappedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_MappedMatrixInitialize ( nalu_hypre_MappedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_MappedMatrixAssemble ( nalu_hypre_MappedMatrix *matrix );
void nalu_hypre_MappedMatrixPrint ( nalu_hypre_MappedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_MappedMatrixGetColIndex ( nalu_hypre_MappedMatrix *matrix, NALU_HYPRE_Int j );
void *nalu_hypre_MappedMatrixGetMatrix ( nalu_hypre_MappedMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_MappedMatrixSetMatrix ( nalu_hypre_MappedMatrix *matrix, void *matrix_data );
NALU_HYPRE_Int nalu_hypre_MappedMatrixSetColMap ( nalu_hypre_MappedMatrix *matrix, NALU_HYPRE_Int (*ColMap )(NALU_HYPRE_Int,
                                                                                         void *));
NALU_HYPRE_Int nalu_hypre_MappedMatrixSetMapData ( nalu_hypre_MappedMatrix *matrix, void *map_data );

/* multiblock_matrix.c */
nalu_hypre_MultiblockMatrix *nalu_hypre_MultiblockMatrixCreate ( void );
NALU_HYPRE_Int nalu_hypre_MultiblockMatrixDestroy ( nalu_hypre_MultiblockMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_MultiblockMatrixLimitedDestroy ( nalu_hypre_MultiblockMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_MultiblockMatrixInitialize ( nalu_hypre_MultiblockMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_MultiblockMatrixAssemble ( nalu_hypre_MultiblockMatrix *matrix );
void nalu_hypre_MultiblockMatrixPrint ( nalu_hypre_MultiblockMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_MultiblockMatrixSetNumSubmatrices ( nalu_hypre_MultiblockMatrix *matrix, NALU_HYPRE_Int n );
NALU_HYPRE_Int nalu_hypre_MultiblockMatrixSetSubmatrixType ( nalu_hypre_MultiblockMatrix *matrix, NALU_HYPRE_Int j,
                                                   NALU_HYPRE_Int type );
NALU_HYPRE_Int nalu_hypre_MultiblockMatrixSetSubmatrix ( nalu_hypre_MultiblockMatrix *matrix, NALU_HYPRE_Int j,
                                               void *submatrix );

/* vector.c */
nalu_hypre_Vector *nalu_hypre_SeqVectorCreate ( NALU_HYPRE_Int size );
nalu_hypre_Vector *nalu_hypre_SeqMultiVectorCreate ( NALU_HYPRE_Int size, NALU_HYPRE_Int num_vectors );
NALU_HYPRE_Int nalu_hypre_SeqVectorDestroy ( nalu_hypre_Vector *vector );
NALU_HYPRE_Int nalu_hypre_SeqVectorInitialize_v2( nalu_hypre_Vector *vector,
                                        NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int nalu_hypre_SeqVectorInitialize ( nalu_hypre_Vector *vector );
NALU_HYPRE_Int nalu_hypre_SeqVectorSetDataOwner ( nalu_hypre_Vector *vector, NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int nalu_hypre_SeqVectorSetSize ( nalu_hypre_Vector *vector, NALU_HYPRE_Int size );
NALU_HYPRE_Int nalu_hypre_SeqVectorResize ( nalu_hypre_Vector *vector, NALU_HYPRE_Int num_vectors_in );
nalu_hypre_Vector *nalu_hypre_SeqVectorRead ( char *file_name );
NALU_HYPRE_Int nalu_hypre_SeqVectorPrint ( nalu_hypre_Vector *vector, char *file_name );
NALU_HYPRE_Int nalu_hypre_SeqVectorSetConstantValues ( nalu_hypre_Vector *v, NALU_HYPRE_Complex value );
NALU_HYPRE_Int nalu_hypre_SeqVectorSetConstantValuesHost ( nalu_hypre_Vector *v, NALU_HYPRE_Complex value );
NALU_HYPRE_Int nalu_hypre_SeqVectorSetConstantValuesDevice ( nalu_hypre_Vector *v, NALU_HYPRE_Complex value );
NALU_HYPRE_Int nalu_hypre_SeqVectorSetRandomValues ( nalu_hypre_Vector *v, NALU_HYPRE_Int seed );
NALU_HYPRE_Int nalu_hypre_SeqVectorCopy ( nalu_hypre_Vector *x, nalu_hypre_Vector *y );
nalu_hypre_Vector *nalu_hypre_SeqVectorCloneDeep ( nalu_hypre_Vector *x );
nalu_hypre_Vector *nalu_hypre_SeqVectorCloneDeep_v2( nalu_hypre_Vector *x, NALU_HYPRE_MemoryLocation memory_location );
nalu_hypre_Vector *nalu_hypre_SeqVectorCloneShallow ( nalu_hypre_Vector *x );
NALU_HYPRE_Int nalu_hypre_SeqVectorScale( NALU_HYPRE_Complex alpha, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_SeqVectorScaleHost( NALU_HYPRE_Complex alpha, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_SeqVectorScaleDevice( NALU_HYPRE_Complex alpha, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_SeqVectorAxpy ( NALU_HYPRE_Complex alpha, nalu_hypre_Vector *x, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_SeqVectorAxpyHost ( NALU_HYPRE_Complex alpha, nalu_hypre_Vector *x, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_SeqVectorAxpyDevice ( NALU_HYPRE_Complex alpha, nalu_hypre_Vector *x, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_SeqVectorAxpyz ( NALU_HYPRE_Complex alpha, nalu_hypre_Vector *x,
                                 NALU_HYPRE_Complex beta, nalu_hypre_Vector *y,
                                 nalu_hypre_Vector *z );
NALU_HYPRE_Int nalu_hypre_SeqVectorAxpyzDevice ( NALU_HYPRE_Complex alpha, nalu_hypre_Vector *x,
                                       NALU_HYPRE_Complex beta, nalu_hypre_Vector *y,
                                       nalu_hypre_Vector *z );
NALU_HYPRE_Real nalu_hypre_SeqVectorInnerProd ( nalu_hypre_Vector *x, nalu_hypre_Vector *y );
NALU_HYPRE_Real nalu_hypre_SeqVectorInnerProdHost ( nalu_hypre_Vector *x, nalu_hypre_Vector *y );
NALU_HYPRE_Real nalu_hypre_SeqVectorInnerProdDevice ( nalu_hypre_Vector *x, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_SeqVectorMassInnerProd(nalu_hypre_Vector *x, nalu_hypre_Vector **y, NALU_HYPRE_Int k,
                                       NALU_HYPRE_Int unroll, NALU_HYPRE_Real *result);
NALU_HYPRE_Int nalu_hypre_SeqVectorMassInnerProd4(nalu_hypre_Vector *x, nalu_hypre_Vector **y, NALU_HYPRE_Int k,
                                        NALU_HYPRE_Real *result);
NALU_HYPRE_Int nalu_hypre_SeqVectorMassInnerProd8(nalu_hypre_Vector *x, nalu_hypre_Vector **y, NALU_HYPRE_Int k,
                                        NALU_HYPRE_Real *result);
NALU_HYPRE_Int nalu_hypre_SeqVectorMassDotpTwo(nalu_hypre_Vector *x, nalu_hypre_Vector *y, nalu_hypre_Vector **z,
                                     NALU_HYPRE_Int k, NALU_HYPRE_Int unroll,  NALU_HYPRE_Real *result_x, NALU_HYPRE_Real *result_y);
NALU_HYPRE_Int nalu_hypre_SeqVectorMassDotpTwo4(nalu_hypre_Vector *x, nalu_hypre_Vector *y, nalu_hypre_Vector **z,
                                      NALU_HYPRE_Int k, NALU_HYPRE_Real *result_x, NALU_HYPRE_Real *result_y);
NALU_HYPRE_Int nalu_hypre_SeqVectorMassDotpTwo8(nalu_hypre_Vector *x, nalu_hypre_Vector *y, nalu_hypre_Vector **z,
                                      NALU_HYPRE_Int k,  NALU_HYPRE_Real *result_x, NALU_HYPRE_Real *result_y);
NALU_HYPRE_Int nalu_hypre_SeqVectorMassAxpy(NALU_HYPRE_Complex *alpha, nalu_hypre_Vector **x, nalu_hypre_Vector *y,
                                  NALU_HYPRE_Int k, NALU_HYPRE_Int unroll);
NALU_HYPRE_Int nalu_hypre_SeqVectorMassAxpy4(NALU_HYPRE_Complex *alpha, nalu_hypre_Vector **x, nalu_hypre_Vector *y,
                                   NALU_HYPRE_Int k);
NALU_HYPRE_Int nalu_hypre_SeqVectorMassAxpy8(NALU_HYPRE_Complex *alpha, nalu_hypre_Vector **x, nalu_hypre_Vector *y,
                                   NALU_HYPRE_Int k);
NALU_HYPRE_Complex nalu_hypre_SeqVectorSumElts ( nalu_hypre_Vector *vector );
NALU_HYPRE_Complex nalu_hypre_SeqVectorSumEltsHost ( nalu_hypre_Vector *vector );
NALU_HYPRE_Complex nalu_hypre_SeqVectorSumEltsDevice ( nalu_hypre_Vector *vector );
NALU_HYPRE_Int nalu_hypre_SeqVectorPrefetch(nalu_hypre_Vector *x, NALU_HYPRE_MemoryLocation memory_location);
//NALU_HYPRE_Int nalu_hypre_SeqVectorMax( NALU_HYPRE_Complex alpha, nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *y );

NALU_HYPRE_Int hypreDevice_CSRSpAdd(NALU_HYPRE_Int ma, NALU_HYPRE_Int mb, NALU_HYPRE_Int n, NALU_HYPRE_Int nnzA,
                               NALU_HYPRE_Int nnzB, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex alpha, NALU_HYPRE_Complex *d_aa,
                               NALU_HYPRE_Int *d_ja_map, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Complex beta, NALU_HYPRE_Complex *d_ab,
                               NALU_HYPRE_Int *d_jb_map, NALU_HYPRE_Int *d_num_b, NALU_HYPRE_Int *nnzC_out, NALU_HYPRE_Int **d_ic_out,
                               NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_ac_out);

NALU_HYPRE_Int hypreDevice_CSRSpTrans(NALU_HYPRE_Int m, NALU_HYPRE_Int n, NALU_HYPRE_Int nnzA, NALU_HYPRE_Int *d_ia,
                                 NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_aa, NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out,
                                 NALU_HYPRE_Complex **d_ac_out, NALU_HYPRE_Int want_data);

NALU_HYPRE_Int hypreDevice_CSRSpTransCusparse(NALU_HYPRE_Int m, NALU_HYPRE_Int n, NALU_HYPRE_Int nnzA, NALU_HYPRE_Int *d_ia,
                                         NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_aa, NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out,
                                         NALU_HYPRE_Complex **d_ac_out, NALU_HYPRE_Int want_data);

NALU_HYPRE_Int hypreDevice_CSRSpTransRocsparse(NALU_HYPRE_Int m, NALU_HYPRE_Int n, NALU_HYPRE_Int nnzA, NALU_HYPRE_Int *d_ia,
                                          NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_aa, NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out,
                                          NALU_HYPRE_Complex **d_ac_out, NALU_HYPRE_Int want_data);

NALU_HYPRE_Int hypreDevice_CSRSpTransOnemklsparse(NALU_HYPRE_Int m, NALU_HYPRE_Int n, NALU_HYPRE_Int nnzA,
                                             NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_aa, NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out,
                                             NALU_HYPRE_Complex **d_ac_out, NALU_HYPRE_Int want_data);

NALU_HYPRE_Int hypreDevice_CSRSpGemm(nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B, nalu_hypre_CSRMatrix **C_ptr);

NALU_HYPRE_Int hypreDevice_CSRSpGemmCusparseGenericAPI(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                                  NALU_HYPRE_Int nnzA, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_a, NALU_HYPRE_Int nnzB,
                                                  NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Complex *d_b, NALU_HYPRE_Int *nnzC_out, NALU_HYPRE_Int **d_ic_out,
                                                  NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_c_out);

NALU_HYPRE_Int nalu_hypre_SeqVectorElmdivpy( nalu_hypre_Vector *x, nalu_hypre_Vector *b, nalu_hypre_Vector *y );
NALU_HYPRE_Int nalu_hypre_SeqVectorElmdivpyMarked( nalu_hypre_Vector *x, nalu_hypre_Vector *b, nalu_hypre_Vector *y,
                                         NALU_HYPRE_Int *marker, NALU_HYPRE_Int marker_val );
NALU_HYPRE_Int nalu_hypre_SeqVectorElmdivpyHost( nalu_hypre_Vector *x, nalu_hypre_Vector *b, nalu_hypre_Vector *y,
                                       NALU_HYPRE_Int *marker, NALU_HYPRE_Int marker_val );
NALU_HYPRE_Int nalu_hypre_SeqVectorElmdivpyDevice( nalu_hypre_Vector *x, nalu_hypre_Vector *b, nalu_hypre_Vector *y,
                                         NALU_HYPRE_Int *marker, NALU_HYPRE_Int marker_val );

NALU_HYPRE_Int nalu_hypre_CSRMatrixSpMVDevice( NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                                     nalu_hypre_Vector *x,
                                     NALU_HYPRE_Complex beta, nalu_hypre_Vector *y, NALU_HYPRE_Int fill );

NALU_HYPRE_Int nalu_hypre_CSRMatrixIntSpMVDevice( NALU_HYPRE_Int num_rows, NALU_HYPRE_Int num_nonzeros,
                                        NALU_HYPRE_Int alpha, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja,
                                        NALU_HYPRE_Int *d_a, NALU_HYPRE_Int *d_x, NALU_HYPRE_Int beta,
                                        NALU_HYPRE_Int *d_y );

#if defined(NALU_HYPRE_USING_CUSPARSE)  ||\
    defined(NALU_HYPRE_USING_ROCSPARSE) ||\
    defined(NALU_HYPRE_USING_ONEMKLSPARSE)
nalu_hypre_CsrsvData* nalu_hypre_CsrsvDataCreate();
NALU_HYPRE_Int nalu_hypre_CsrsvDataDestroy(nalu_hypre_CsrsvData *data);
nalu_hypre_GpuMatData* nalu_hypre_GpuMatDataCreate();
NALU_HYPRE_Int nalu_hypre_GPUMatDataSetCSRData(nalu_hypre_CSRMatrix *matrix);
NALU_HYPRE_Int nalu_hypre_GpuMatDataDestroy(nalu_hypre_GpuMatData *data);
nalu_hypre_GpuMatData* nalu_hypre_CSRMatrixGetGPUMatData(nalu_hypre_CSRMatrix *matrix);

#define nalu_hypre_CSRMatrixGPUMatDescr(matrix)       ( nalu_hypre_GpuMatDataMatDescr(nalu_hypre_CSRMatrixGetGPUMatData(matrix)) )
#define nalu_hypre_CSRMatrixGPUMatInfo(matrix)        ( nalu_hypre_GpuMatDataMatInfo (nalu_hypre_CSRMatrixGetGPUMatData(matrix)) )
#define nalu_hypre_CSRMatrixGPUMatHandle(matrix)      ( nalu_hypre_GpuMatDataMatHandle (nalu_hypre_CSRMatrixGetGPUMatData(matrix)) )
#define nalu_hypre_CSRMatrixGPUMatSpMVBuffer(matrix)  ( nalu_hypre_GpuMatDataSpMVBuffer (nalu_hypre_CSRMatrixGetGPUMatData(matrix)) )
#endif

NALU_HYPRE_Int nalu_hypre_CSRMatrixSpMVAnalysisDevice(nalu_hypre_CSRMatrix *matrix);
