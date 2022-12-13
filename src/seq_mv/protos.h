/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* csr_matop.c */
NALU_HYPRE_Int hypre_CSRMatrixAddFirstPass ( NALU_HYPRE_Int firstrow, NALU_HYPRE_Int lastrow, NALU_HYPRE_Int *marker,
                                        NALU_HYPRE_Int *twspace, NALU_HYPRE_Int *map_A2C, NALU_HYPRE_Int *map_B2C, hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                        NALU_HYPRE_Int nnzrows_C, NALU_HYPRE_Int nrows_C, NALU_HYPRE_Int ncols_C, NALU_HYPRE_Int *rownnz_C,
                                        NALU_HYPRE_MemoryLocation memory_location_C, NALU_HYPRE_Int *C_i, hypre_CSRMatrix **C_ptr );
NALU_HYPRE_Int hypre_CSRMatrixAddSecondPass ( NALU_HYPRE_Int firstrow, NALU_HYPRE_Int lastrow, NALU_HYPRE_Int *marker,
                                         NALU_HYPRE_Int *twspace, NALU_HYPRE_Int *map_A2C, NALU_HYPRE_Int *map_B2C, NALU_HYPRE_Int *rownnz_C,
                                         NALU_HYPRE_Complex alpha, NALU_HYPRE_Complex beta, hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                         hypre_CSRMatrix *C);
hypre_CSRMatrix *hypre_CSRMatrixAddHost ( NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                          NALU_HYPRE_Complex beta, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixAdd ( NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A, NALU_HYPRE_Complex beta,
                                      hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixBigAdd ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyHost ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiply ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixDeleteZeros ( hypre_CSRMatrix *A, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_CSRMatrixTransposeHost ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT, NALU_HYPRE_Int data );
NALU_HYPRE_Int hypre_CSRMatrixTranspose ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT, NALU_HYPRE_Int data );
NALU_HYPRE_Int hypre_CSRMatrixReorder ( hypre_CSRMatrix *A );
NALU_HYPRE_Complex hypre_CSRMatrixSumElts ( hypre_CSRMatrix *A );
NALU_HYPRE_Real hypre_CSRMatrixFnorm( hypre_CSRMatrix *A );
NALU_HYPRE_Int hypre_CSRMatrixSplit(hypre_CSRMatrix *Bs_ext, NALU_HYPRE_BigInt first_col_diag_B,
                               NALU_HYPRE_BigInt last_col_diag_B, NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_BigInt *col_map_offd_B,
                               NALU_HYPRE_Int *num_cols_offd_C_ptr, NALU_HYPRE_BigInt **col_map_offd_C_ptr, hypre_CSRMatrix **Bext_diag_ptr,
                               hypre_CSRMatrix **Bext_offd_ptr);
hypre_CSRMatrix * hypre_CSRMatrixAddPartial( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                             NALU_HYPRE_Int *row_nums);
void hypre_CSRMatrixComputeRowSumHost( hypre_CSRMatrix *A, NALU_HYPRE_Int *CF_i, NALU_HYPRE_Int *CF_j,
                                       NALU_HYPRE_Complex *row_sum, NALU_HYPRE_Int type, NALU_HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixComputeRowSum( hypre_CSRMatrix *A, NALU_HYPRE_Int *CF_i, NALU_HYPRE_Int *CF_j,
                                   NALU_HYPRE_Complex *row_sum, NALU_HYPRE_Int type, NALU_HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixExtractDiagonal( hypre_CSRMatrix *A, NALU_HYPRE_Complex *d, NALU_HYPRE_Int type);
void hypre_CSRMatrixExtractDiagonalHost( hypre_CSRMatrix *A, NALU_HYPRE_Complex *d, NALU_HYPRE_Int type);
NALU_HYPRE_Int hypre_CSRMatrixScale(hypre_CSRMatrix *A, NALU_HYPRE_Complex scalar);
NALU_HYPRE_Int hypre_CSRMatrixSetConstantValues( hypre_CSRMatrix *A, NALU_HYPRE_Complex value);
NALU_HYPRE_Int hypre_CSRMatrixDiagScale( hypre_CSRMatrix *A, hypre_Vector *ld, hypre_Vector *rd);

/* csr_matop_device.c */
hypre_CSRMatrix *hypre_CSRMatrixAddDevice ( NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                            NALU_HYPRE_Complex beta, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyDevice ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixTripleMultiplyDevice ( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                       hypre_CSRMatrix *C );
NALU_HYPRE_Int hypre_CSRMatrixMergeColMapOffd( NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_BigInt *col_map_offd_B,
                                          NALU_HYPRE_Int B_ext_offd_nnz, NALU_HYPRE_BigInt *B_ext_offd_bigj, NALU_HYPRE_Int *num_cols_offd_C_ptr,
                                          NALU_HYPRE_BigInt **col_map_offd_C_ptr, NALU_HYPRE_Int **map_B_to_C_ptr );
NALU_HYPRE_Int hypre_CSRMatrixSplitDevice_core( NALU_HYPRE_Int job, NALU_HYPRE_Int num_rows, NALU_HYPRE_Int B_ext_nnz,
                                           NALU_HYPRE_Int *B_ext_ii, NALU_HYPRE_BigInt *B_ext_bigj, NALU_HYPRE_Complex *B_ext_data, char *B_ext_xata,
                                           NALU_HYPRE_BigInt first_col_diag_B, NALU_HYPRE_BigInt last_col_diag_B, NALU_HYPRE_Int num_cols_offd_B,
                                           NALU_HYPRE_BigInt *col_map_offd_B, NALU_HYPRE_Int **map_B_to_C_ptr, NALU_HYPRE_Int *num_cols_offd_C_ptr,
                                           NALU_HYPRE_BigInt **col_map_offd_C_ptr, NALU_HYPRE_Int *B_ext_diag_nnz_ptr, NALU_HYPRE_Int *B_ext_diag_ii,
                                           NALU_HYPRE_Int *B_ext_diag_j, NALU_HYPRE_Complex *B_ext_diag_data, char *B_ext_diag_xata,
                                           NALU_HYPRE_Int *B_ext_offd_nnz_ptr, NALU_HYPRE_Int *B_ext_offd_ii, NALU_HYPRE_Int *B_ext_offd_j,
                                           NALU_HYPRE_Complex *B_ext_offd_data, char *B_ext_offd_xata );
NALU_HYPRE_Int hypre_CSRMatrixSplitDevice(hypre_CSRMatrix *B_ext, NALU_HYPRE_BigInt first_col_diag_B,
                                     NALU_HYPRE_BigInt last_col_diag_B, NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_BigInt *col_map_offd_B,
                                     NALU_HYPRE_Int **map_B_to_C_ptr, NALU_HYPRE_Int *num_cols_offd_C_ptr, NALU_HYPRE_BigInt **col_map_offd_C_ptr,
                                     hypre_CSRMatrix **B_ext_diag_ptr, hypre_CSRMatrix **B_ext_offd_ptr);
NALU_HYPRE_Int hypre_CSRMatrixTransposeDevice ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT,
                                           NALU_HYPRE_Int data );
hypre_CSRMatrix* hypre_CSRMatrixAddPartialDevice( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                  NALU_HYPRE_Int *row_nums);
NALU_HYPRE_Int hypre_CSRMatrixColNNzRealDevice( hypre_CSRMatrix *A, NALU_HYPRE_Real *colnnz);
NALU_HYPRE_Int hypre_CSRMatrixMoveDiagFirstDevice( hypre_CSRMatrix  *A );
NALU_HYPRE_Int hypre_CSRMatrixCheckDiagFirstDevice( hypre_CSRMatrix  *A );
NALU_HYPRE_Int hypre_CSRMatrixFixZeroDiagDevice( hypre_CSRMatrix *A, NALU_HYPRE_Complex v, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_CSRMatrixReplaceDiagDevice( hypre_CSRMatrix *A, NALU_HYPRE_Complex *new_diag,
                                            NALU_HYPRE_Complex v, NALU_HYPRE_Real tol );
void hypre_CSRMatrixComputeRowSumDevice( hypre_CSRMatrix *A, NALU_HYPRE_Int *CF_i, NALU_HYPRE_Int *CF_j,
                                         NALU_HYPRE_Complex *row_sum, NALU_HYPRE_Int type, NALU_HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixExtractDiagonalDevice( hypre_CSRMatrix *A, NALU_HYPRE_Complex *d, NALU_HYPRE_Int type);
hypre_CSRMatrix* hypre_CSRMatrixStack2Device(hypre_CSRMatrix *A, hypre_CSRMatrix *B);
hypre_CSRMatrix* hypre_CSRMatrixIdentityDevice(NALU_HYPRE_Int n, NALU_HYPRE_Complex alp);
hypre_CSRMatrix* hypre_CSRMatrixDiagMatrixFromVectorDevice(NALU_HYPRE_Int n, NALU_HYPRE_Complex *v);
hypre_CSRMatrix* hypre_CSRMatrixDiagMatrixFromMatrixDevice(hypre_CSRMatrix *A, NALU_HYPRE_Int type);
NALU_HYPRE_Int hypre_CSRMatrixRemoveDiagonalDevice(hypre_CSRMatrix *A);
NALU_HYPRE_Int hypre_CSRMatrixDropSmallEntriesDevice( hypre_CSRMatrix *A, NALU_HYPRE_Real tol,
                                                 NALU_HYPRE_Real *elmt_tols);
NALU_HYPRE_Int hypre_CSRMatrixSortRow(hypre_CSRMatrix *A);
NALU_HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveDevice(char uplo, hypre_CSRMatrix *A,
                                                  NALU_HYPRE_Real *l1_norms, hypre_Vector *f, hypre_Vector *u );
NALU_HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveRocsparse(char uplo, hypre_CSRMatrix *A,
                                                     NALU_HYPRE_Real *l1_norms, hypre_Vector *f, hypre_Vector *u );
NALU_HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveCusparse(char uplo, hypre_CSRMatrix *A,
                                                    NALU_HYPRE_Real *l1_norms, hypre_Vector *f, hypre_Vector *u );
NALU_HYPRE_Int hypre_CSRMatrixIntersectPattern(hypre_CSRMatrix *A, hypre_CSRMatrix *B, NALU_HYPRE_Int *markA,
                                          NALU_HYPRE_Int diag_option);
NALU_HYPRE_Int hypre_CSRMatrixDiagScaleDevice( hypre_CSRMatrix *A, hypre_Vector *ld, hypre_Vector *rd);

/* csr_matrix.c */
hypre_CSRMatrix *hypre_CSRMatrixCreate ( NALU_HYPRE_Int num_rows, NALU_HYPRE_Int num_cols,
                                         NALU_HYPRE_Int num_nonzeros );
NALU_HYPRE_Int hypre_CSRMatrixDestroy ( hypre_CSRMatrix *matrix );
NALU_HYPRE_Int hypre_CSRMatrixInitialize_v2( hypre_CSRMatrix *matrix, NALU_HYPRE_Int bigInit,
                                        NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int hypre_CSRMatrixInitialize ( hypre_CSRMatrix *matrix );
NALU_HYPRE_Int hypre_CSRMatrixBigInitialize ( hypre_CSRMatrix *matrix );
NALU_HYPRE_Int hypre_CSRMatrixBigJtoJ ( hypre_CSRMatrix *matrix );
NALU_HYPRE_Int hypre_CSRMatrixJtoBigJ ( hypre_CSRMatrix *matrix );
NALU_HYPRE_Int hypre_CSRMatrixSetDataOwner ( hypre_CSRMatrix *matrix, NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int hypre_CSRMatrixSetPatternOnly( hypre_CSRMatrix *matrix, NALU_HYPRE_Int pattern_only );
NALU_HYPRE_Int hypre_CSRMatrixSetRownnz ( hypre_CSRMatrix *matrix );
hypre_CSRMatrix *hypre_CSRMatrixRead ( char *file_name );
NALU_HYPRE_Int hypre_CSRMatrixPrint ( hypre_CSRMatrix *matrix, const char *file_name );
NALU_HYPRE_Int hypre_CSRMatrixPrintHB ( hypre_CSRMatrix *matrix_input, char *file_name );
NALU_HYPRE_Int hypre_CSRMatrixPrintMM( hypre_CSRMatrix *matrix, NALU_HYPRE_Int basei, NALU_HYPRE_Int basej,
                                  NALU_HYPRE_Int trans, const char *file_name );
NALU_HYPRE_Int hypre_CSRMatrixCopy ( hypre_CSRMatrix *A, hypre_CSRMatrix *B, NALU_HYPRE_Int copy_data );
hypre_CSRMatrix *hypre_CSRMatrixClone ( hypre_CSRMatrix *A, NALU_HYPRE_Int copy_data );
hypre_CSRMatrix *hypre_CSRMatrixClone_v2( hypre_CSRMatrix *A, NALU_HYPRE_Int copy_data,
                                          NALU_HYPRE_MemoryLocation memory_location );
hypre_CSRMatrix *hypre_CSRMatrixUnion ( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                        NALU_HYPRE_BigInt *col_map_offd_A, NALU_HYPRE_BigInt *col_map_offd_B, NALU_HYPRE_BigInt **col_map_offd_C );
NALU_HYPRE_Int hypre_CSRMatrixPrefetch( hypre_CSRMatrix *A, NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int hypre_CSRMatrixCheckSetNumNonzeros( hypre_CSRMatrix *matrix );
NALU_HYPRE_Int hypre_CSRMatrixResize( hypre_CSRMatrix *matrix, NALU_HYPRE_Int new_num_rows,
                                 NALU_HYPRE_Int new_num_cols, NALU_HYPRE_Int new_num_nonzeros );

/* csr_matvec.c */
// y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end]
NALU_HYPRE_Int hypre_CSRMatrixMatvecOutOfPlace ( NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                            hypre_Vector *x, NALU_HYPRE_Complex beta, hypre_Vector *b, hypre_Vector *y, NALU_HYPRE_Int offset );
// y = alpha*A + beta*y
NALU_HYPRE_Int hypre_CSRMatrixMatvec ( NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                  NALU_HYPRE_Complex beta, hypre_Vector *y );
NALU_HYPRE_Int hypre_CSRMatrixMatvecT ( NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                   NALU_HYPRE_Complex beta, hypre_Vector *y );
NALU_HYPRE_Int hypre_CSRMatrixMatvec_FF ( NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                     NALU_HYPRE_Complex beta, hypre_Vector *y, NALU_HYPRE_Int *CF_marker_x, NALU_HYPRE_Int *CF_marker_y,
                                     NALU_HYPRE_Int fpt );

/* csr_matvec_device.c */
NALU_HYPRE_Int hypre_CSRMatrixMatvecDevice(NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                      hypre_Vector *x, NALU_HYPRE_Complex beta, hypre_Vector *b, hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int hypre_CSRMatrixMatvecCusparseNewAPI( NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha,
                                               hypre_CSRMatrix *A, hypre_Vector *x, NALU_HYPRE_Complex beta, hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int hypre_CSRMatrixMatvecCusparseOldAPI( NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha,
                                               hypre_CSRMatrix *A, hypre_Vector *x, NALU_HYPRE_Complex beta, hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int hypre_CSRMatrixMatvecCusparse( NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha,
                                         hypre_CSRMatrix *A, hypre_Vector *x, NALU_HYPRE_Complex beta, hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int hypre_CSRMatrixMatvecOMPOffload (NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                           hypre_Vector *x, NALU_HYPRE_Complex beta, hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int hypre_CSRMatrixMatvecRocsparse (NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                          hypre_Vector *x, NALU_HYPRE_Complex beta, hypre_Vector *y, NALU_HYPRE_Int offset );
NALU_HYPRE_Int hypre_CSRMatrixMatvecOnemklsparse (NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha,
                                             hypre_CSRMatrix *A,
                                             hypre_Vector *x, NALU_HYPRE_Complex beta, hypre_Vector *y, NALU_HYPRE_Int offset );

/* genpart.c */
NALU_HYPRE_Int hypre_GeneratePartitioning ( NALU_HYPRE_BigInt length, NALU_HYPRE_Int num_procs,
                                       NALU_HYPRE_BigInt **part_ptr );
NALU_HYPRE_Int hypre_GenerateLocalPartitioning ( NALU_HYPRE_BigInt length, NALU_HYPRE_Int num_procs,
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
hypre_MappedMatrix *hypre_MappedMatrixCreate ( void );
NALU_HYPRE_Int hypre_MappedMatrixDestroy ( hypre_MappedMatrix *matrix );
NALU_HYPRE_Int hypre_MappedMatrixLimitedDestroy ( hypre_MappedMatrix *matrix );
NALU_HYPRE_Int hypre_MappedMatrixInitialize ( hypre_MappedMatrix *matrix );
NALU_HYPRE_Int hypre_MappedMatrixAssemble ( hypre_MappedMatrix *matrix );
void hypre_MappedMatrixPrint ( hypre_MappedMatrix *matrix );
NALU_HYPRE_Int hypre_MappedMatrixGetColIndex ( hypre_MappedMatrix *matrix, NALU_HYPRE_Int j );
void *hypre_MappedMatrixGetMatrix ( hypre_MappedMatrix *matrix );
NALU_HYPRE_Int hypre_MappedMatrixSetMatrix ( hypre_MappedMatrix *matrix, void *matrix_data );
NALU_HYPRE_Int hypre_MappedMatrixSetColMap ( hypre_MappedMatrix *matrix, NALU_HYPRE_Int (*ColMap )(NALU_HYPRE_Int,
                                                                                         void *));
NALU_HYPRE_Int hypre_MappedMatrixSetMapData ( hypre_MappedMatrix *matrix, void *map_data );

/* multiblock_matrix.c */
hypre_MultiblockMatrix *hypre_MultiblockMatrixCreate ( void );
NALU_HYPRE_Int hypre_MultiblockMatrixDestroy ( hypre_MultiblockMatrix *matrix );
NALU_HYPRE_Int hypre_MultiblockMatrixLimitedDestroy ( hypre_MultiblockMatrix *matrix );
NALU_HYPRE_Int hypre_MultiblockMatrixInitialize ( hypre_MultiblockMatrix *matrix );
NALU_HYPRE_Int hypre_MultiblockMatrixAssemble ( hypre_MultiblockMatrix *matrix );
void hypre_MultiblockMatrixPrint ( hypre_MultiblockMatrix *matrix );
NALU_HYPRE_Int hypre_MultiblockMatrixSetNumSubmatrices ( hypre_MultiblockMatrix *matrix, NALU_HYPRE_Int n );
NALU_HYPRE_Int hypre_MultiblockMatrixSetSubmatrixType ( hypre_MultiblockMatrix *matrix, NALU_HYPRE_Int j,
                                                   NALU_HYPRE_Int type );
NALU_HYPRE_Int hypre_MultiblockMatrixSetSubmatrix ( hypre_MultiblockMatrix *matrix, NALU_HYPRE_Int j,
                                               void *submatrix );

/* vector.c */
hypre_Vector *hypre_SeqVectorCreate ( NALU_HYPRE_Int size );
hypre_Vector *hypre_SeqMultiVectorCreate ( NALU_HYPRE_Int size, NALU_HYPRE_Int num_vectors );
NALU_HYPRE_Int hypre_SeqVectorDestroy ( hypre_Vector *vector );
NALU_HYPRE_Int hypre_SeqVectorInitialize_v2( hypre_Vector *vector,
                                        NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int hypre_SeqVectorInitialize ( hypre_Vector *vector );
NALU_HYPRE_Int hypre_SeqVectorSetDataOwner ( hypre_Vector *vector, NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int hypre_SeqVectorSetSize ( hypre_Vector *vector, NALU_HYPRE_Int size );
hypre_Vector *hypre_SeqVectorRead ( char *file_name );
NALU_HYPRE_Int hypre_SeqVectorPrint ( hypre_Vector *vector, char *file_name );
NALU_HYPRE_Int hypre_SeqVectorSetConstantValues ( hypre_Vector *v, NALU_HYPRE_Complex value );
NALU_HYPRE_Int hypre_SeqVectorSetConstantValuesHost ( hypre_Vector *v, NALU_HYPRE_Complex value );
NALU_HYPRE_Int hypre_SeqVectorSetConstantValuesDevice ( hypre_Vector *v, NALU_HYPRE_Complex value );
NALU_HYPRE_Int hypre_SeqVectorSetRandomValues ( hypre_Vector *v, NALU_HYPRE_Int seed );
NALU_HYPRE_Int hypre_SeqVectorCopy ( hypre_Vector *x, hypre_Vector *y );
hypre_Vector *hypre_SeqVectorCloneDeep ( hypre_Vector *x );
hypre_Vector *hypre_SeqVectorCloneDeep_v2( hypre_Vector *x, NALU_HYPRE_MemoryLocation memory_location );
hypre_Vector *hypre_SeqVectorCloneShallow ( hypre_Vector *x );
NALU_HYPRE_Int hypre_SeqVectorScale( NALU_HYPRE_Complex alpha, hypre_Vector *y );
NALU_HYPRE_Int hypre_SeqVectorScaleHost( NALU_HYPRE_Complex alpha, hypre_Vector *y );
NALU_HYPRE_Int hypre_SeqVectorScaleDevice( NALU_HYPRE_Complex alpha, hypre_Vector *y );
NALU_HYPRE_Int hypre_SeqVectorAxpy ( NALU_HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
NALU_HYPRE_Int hypre_SeqVectorAxpyHost ( NALU_HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
NALU_HYPRE_Int hypre_SeqVectorAxpyDevice ( NALU_HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
NALU_HYPRE_Real hypre_SeqVectorInnerProd ( hypre_Vector *x, hypre_Vector *y );
NALU_HYPRE_Real hypre_SeqVectorInnerProdHost ( hypre_Vector *x, hypre_Vector *y );
NALU_HYPRE_Real hypre_SeqVectorInnerProdDevice ( hypre_Vector *x, hypre_Vector *y );
NALU_HYPRE_Int hypre_SeqVectorMassInnerProd(hypre_Vector *x, hypre_Vector **y, NALU_HYPRE_Int k,
                                       NALU_HYPRE_Int unroll, NALU_HYPRE_Real *result);
NALU_HYPRE_Int hypre_SeqVectorMassInnerProd4(hypre_Vector *x, hypre_Vector **y, NALU_HYPRE_Int k,
                                        NALU_HYPRE_Real *result);
NALU_HYPRE_Int hypre_SeqVectorMassInnerProd8(hypre_Vector *x, hypre_Vector **y, NALU_HYPRE_Int k,
                                        NALU_HYPRE_Real *result);
NALU_HYPRE_Int hypre_SeqVectorMassDotpTwo(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                     NALU_HYPRE_Int k, NALU_HYPRE_Int unroll,  NALU_HYPRE_Real *result_x, NALU_HYPRE_Real *result_y);
NALU_HYPRE_Int hypre_SeqVectorMassDotpTwo4(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                      NALU_HYPRE_Int k, NALU_HYPRE_Real *result_x, NALU_HYPRE_Real *result_y);
NALU_HYPRE_Int hypre_SeqVectorMassDotpTwo8(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                      NALU_HYPRE_Int k,  NALU_HYPRE_Real *result_x, NALU_HYPRE_Real *result_y);
NALU_HYPRE_Int hypre_SeqVectorMassAxpy(NALU_HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                  NALU_HYPRE_Int k, NALU_HYPRE_Int unroll);
NALU_HYPRE_Int hypre_SeqVectorMassAxpy4(NALU_HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                   NALU_HYPRE_Int k);
NALU_HYPRE_Int hypre_SeqVectorMassAxpy8(NALU_HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                   NALU_HYPRE_Int k);
NALU_HYPRE_Complex hypre_SeqVectorSumElts ( hypre_Vector *vector );
NALU_HYPRE_Complex hypre_SeqVectorSumEltsHost ( hypre_Vector *vector );
NALU_HYPRE_Complex hypre_SeqVectorSumEltsDevice ( hypre_Vector *vector );
NALU_HYPRE_Int hypre_SeqVectorPrefetch(hypre_Vector *x, NALU_HYPRE_MemoryLocation memory_location);
//NALU_HYPRE_Int hypre_SeqVectorMax( NALU_HYPRE_Complex alpha, hypre_Vector *x, NALU_HYPRE_Complex beta, hypre_Vector *y );

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

NALU_HYPRE_Int hypreDevice_CSRSpGemm(hypre_CSRMatrix *A, hypre_CSRMatrix *B, hypre_CSRMatrix **C_ptr);

NALU_HYPRE_Int hypreDevice_CSRSpGemmCusparseGenericAPI(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                                  NALU_HYPRE_Int nnzA, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_a, NALU_HYPRE_Int nnzB,
                                                  NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Complex *d_b, NALU_HYPRE_Int *nnzC_out, NALU_HYPRE_Int **d_ic_out,
                                                  NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_c_out);

NALU_HYPRE_Int hypre_SeqVectorElmdivpy( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y );
NALU_HYPRE_Int hypre_SeqVectorElmdivpyMarked( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                         NALU_HYPRE_Int *marker, NALU_HYPRE_Int marker_val );
NALU_HYPRE_Int hypre_SeqVectorElmdivpyHost( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                       NALU_HYPRE_Int *marker, NALU_HYPRE_Int marker_val );
NALU_HYPRE_Int hypre_SeqVectorElmdivpyDevice( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                         NALU_HYPRE_Int *marker, NALU_HYPRE_Int marker_val );

NALU_HYPRE_Int hypre_CSRMatrixSpMVDevice( NALU_HYPRE_Int trans, NALU_HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                     hypre_Vector *x,
                                     NALU_HYPRE_Complex beta, hypre_Vector *y, NALU_HYPRE_Int fill );

NALU_HYPRE_Int hypre_CSRMatrixIntSpMVDevice( NALU_HYPRE_Int num_rows, NALU_HYPRE_Int num_nonzeros,
                                        NALU_HYPRE_Int alpha, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja,
                                        NALU_HYPRE_Int *d_a, NALU_HYPRE_Int *d_x, NALU_HYPRE_Int beta,
                                        NALU_HYPRE_Int *d_y );

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
hypre_CsrsvData* hypre_CsrsvDataCreate();
void hypre_CsrsvDataDestroy(hypre_CsrsvData *data);
hypre_GpuMatData* hypre_GpuMatDataCreate();
void hypre_GPUMatDataSetCSRData(hypre_GpuMatData *data, hypre_CSRMatrix *matrix);
void hypre_GpuMatDataDestroy(hypre_GpuMatData *data);
hypre_GpuMatData* hypre_CSRMatrixGetGPUMatData(hypre_CSRMatrix *matrix);
#define hypre_CSRMatrixGPUMatDescr(matrix)       ( hypre_GpuMatDataMatDecsr(hypre_CSRMatrixGetGPUMatData(matrix)) )
#define hypre_CSRMatrixGPUMatInfo(matrix)        ( hypre_GpuMatDataMatInfo (hypre_CSRMatrixGetGPUMatData(matrix)) )
#define hypre_CSRMatrixGPUMatHandle(matrix)      ( hypre_GpuMatDataMatHandle (hypre_CSRMatrixGetGPUMatData(matrix)) )
#define hypre_CSRMatrixGPUMatSpMVBuffer(matrix)  ( hypre_GpuMatDataSpMVBuffer (hypre_CSRMatrixGetGPUMatData(matrix)) )
#endif
void hypre_CSRMatrixGpuSpMVAnalysis(hypre_CSRMatrix *matrix);
