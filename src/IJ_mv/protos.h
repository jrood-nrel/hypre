/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* aux_parcsr_matrix.c */
NALU_HYPRE_Int nalu_hypre_AuxParCSRMatrixCreate ( nalu_hypre_AuxParCSRMatrix **aux_matrix,
                                        NALU_HYPRE_Int local_num_rows, NALU_HYPRE_Int local_num_cols, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int nalu_hypre_AuxParCSRMatrixDestroy ( nalu_hypre_AuxParCSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_AuxParCSRMatrixSetRownnz ( nalu_hypre_AuxParCSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_AuxParCSRMatrixInitialize ( nalu_hypre_AuxParCSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_AuxParCSRMatrixInitialize_v2( nalu_hypre_AuxParCSRMatrix *matrix,
                                              NALU_HYPRE_MemoryLocation memory_location );

/* aux_par_vector.c */
NALU_HYPRE_Int nalu_hypre_AuxParVectorCreate ( nalu_hypre_AuxParVector **aux_vector );
NALU_HYPRE_Int nalu_hypre_AuxParVectorDestroy ( nalu_hypre_AuxParVector *vector );
NALU_HYPRE_Int nalu_hypre_AuxParVectorInitialize ( nalu_hypre_AuxParVector *vector );
NALU_HYPRE_Int nalu_hypre_AuxParVectorInitialize_v2( nalu_hypre_AuxParVector *vector,
                                           NALU_HYPRE_MemoryLocation memory_location );

/* IJ_assumed_part.c */
NALU_HYPRE_Int nalu_hypre_IJMatrixCreateAssumedPartition ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJVectorCreateAssumedPartition ( nalu_hypre_IJVector *vector );

/* IJMatrix.c */
NALU_HYPRE_Int nalu_hypre_IJMatrixGetRowPartitioning ( NALU_HYPRE_IJMatrix matrix,
                                             NALU_HYPRE_BigInt **row_partitioning );
NALU_HYPRE_Int nalu_hypre_IJMatrixGetColPartitioning ( NALU_HYPRE_IJMatrix matrix,
                                             NALU_HYPRE_BigInt **col_partitioning );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetObject ( NALU_HYPRE_IJMatrix matrix, void *object );
NALU_HYPRE_Int nalu_hypre_IJMatrixRead( const char *filename, MPI_Comm comm, NALU_HYPRE_Int type,
                              NALU_HYPRE_IJMatrix *matrix_ptr, NALU_HYPRE_Int is_mm );

/* IJMatrix_isis.c */
NALU_HYPRE_Int nalu_hypre_IJMatrixSetLocalSizeISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int local_m,
                                           NALU_HYPRE_Int local_n );
NALU_HYPRE_Int nalu_hypre_IJMatrixCreateISIS ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetRowSizesISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetDiagRowSizesISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetOffDiagRowSizesISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int nalu_hypre_IJMatrixInitializeISIS ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixInsertBlockISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int m, NALU_HYPRE_Int n,
                                          NALU_HYPRE_Int *rows, NALU_HYPRE_Int *cols, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int nalu_hypre_IJMatrixAddToBlockISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int m, NALU_HYPRE_Int n,
                                         NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int nalu_hypre_IJMatrixInsertRowISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int n, NALU_HYPRE_BigInt row,
                                        NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int nalu_hypre_IJMatrixAddToRowISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int n, NALU_HYPRE_BigInt row,
                                       NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int nalu_hypre_IJMatrixAssembleISIS ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixDistributeISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_BigInt *row_starts,
                                         NALU_HYPRE_BigInt *col_starts );
NALU_HYPRE_Int nalu_hypre_IJMatrixApplyISIS ( nalu_hypre_IJMatrix *matrix, nalu_hypre_ParVector *x,
                                    nalu_hypre_ParVector *b );
NALU_HYPRE_Int nalu_hypre_IJMatrixDestroyISIS ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetTotalSizeISIS ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int size );

/* IJMatrix_parcsr.c */
NALU_HYPRE_Int nalu_hypre_IJMatrixCreateParCSR ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetRowSizesParCSR ( nalu_hypre_IJMatrix *matrix, const NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetDiagOffdSizesParCSR ( nalu_hypre_IJMatrix *matrix,
                                                 const NALU_HYPRE_Int *diag_sizes, const NALU_HYPRE_Int *offdiag_sizes );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetMaxOffProcElmtsParCSR ( nalu_hypre_IJMatrix *matrix,
                                                   NALU_HYPRE_Int max_off_proc_elmts );
NALU_HYPRE_Int nalu_hypre_IJMatrixInitializeParCSR ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixGetRowCountsParCSR ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                             NALU_HYPRE_BigInt *rows, NALU_HYPRE_Int *ncols );
NALU_HYPRE_Int nalu_hypre_IJMatrixGetValuesParCSR ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows, NALU_HYPRE_Int *ncols,
                                          NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetValuesParCSR ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows, NALU_HYPRE_Int *ncols,
                                          const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                          const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetAddValuesParCSRDevice ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                                   NALU_HYPRE_Int *ncols, const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                                   const NALU_HYPRE_Complex *values, const char *action );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetConstantValuesParCSR ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Complex value );
NALU_HYPRE_Int nalu_hypre_IJMatrixAddToValuesParCSR ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                            NALU_HYPRE_Int *ncols, const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                            const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_IJMatrixDestroyParCSR ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixTransposeParCSR ( nalu_hypre_IJMatrix  *matrix_A, nalu_hypre_IJMatrix *matrix_AT );
NALU_HYPRE_Int nalu_hypre_IJMatrixNormParCSR ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int nalu_hypre_IJMatrixAddParCSR ( NALU_HYPRE_Complex alpha, nalu_hypre_IJMatrix *matrix_A,
                                    NALU_HYPRE_Complex beta, nalu_hypre_IJMatrix *matrix_B, nalu_hypre_IJMatrix *matrix_C );
NALU_HYPRE_Int nalu_hypre_IJMatrixAssembleOffProcValsParCSR ( nalu_hypre_IJMatrix *matrix,
                                                    NALU_HYPRE_Int off_proc_i_indx, NALU_HYPRE_Int max_off_proc_elmts, NALU_HYPRE_Int current_num_elmts,
                                                    NALU_HYPRE_MemoryLocation memory_location, NALU_HYPRE_BigInt *off_proc_i, NALU_HYPRE_BigInt *off_proc_j,
                                                    NALU_HYPRE_Complex *off_proc_data );
NALU_HYPRE_Int nalu_hypre_FillResponseIJOffProcVals ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                            NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                            NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Int nalu_hypre_FindProc ( NALU_HYPRE_BigInt *list, NALU_HYPRE_BigInt value, NALU_HYPRE_Int list_length );
NALU_HYPRE_Int nalu_hypre_IJMatrixAssembleParCSR ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetValuesOMPParCSR ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                             NALU_HYPRE_Int *ncols, const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                             const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_IJMatrixAddToValuesOMPParCSR ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                               NALU_HYPRE_Int *ncols, const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                               const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_IJMatrixAssembleParCSRDevice(nalu_hypre_IJMatrix *matrix);
NALU_HYPRE_Int nalu_hypre_IJMatrixInitializeParCSR_v2(nalu_hypre_IJMatrix *matrix,
                                            NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int nalu_hypre_IJMatrixSetConstantValuesParCSRDevice( nalu_hypre_IJMatrix *matrix,
                                                       NALU_HYPRE_Complex value );

/* IJMatrix_petsc.c */
NALU_HYPRE_Int nalu_hypre_IJMatrixSetLocalSizePETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int local_m,
                                            NALU_HYPRE_Int local_n );
NALU_HYPRE_Int nalu_hypre_IJMatrixCreatePETSc ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetRowSizesPETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetDiagRowSizesPETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetOffDiagRowSizesPETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int nalu_hypre_IJMatrixInitializePETSc ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixInsertBlockPETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int m, NALU_HYPRE_Int n,
                                           NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int nalu_hypre_IJMatrixAddToBlockPETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int m, NALU_HYPRE_Int n,
                                          NALU_HYPRE_Int *rows, NALU_HYPRE_Int *cols, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int nalu_hypre_IJMatrixInsertRowPETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int n, NALU_HYPRE_BigInt row,
                                         NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int nalu_hypre_IJMatrixAddToRowPETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int n, NALU_HYPRE_BigInt row,
                                        NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int nalu_hypre_IJMatrixAssemblePETSc ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixDistributePETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_BigInt *row_starts,
                                          NALU_HYPRE_BigInt *col_starts );
NALU_HYPRE_Int nalu_hypre_IJMatrixApplyPETSc ( nalu_hypre_IJMatrix *matrix, nalu_hypre_ParVector *x,
                                     nalu_hypre_ParVector *b );
NALU_HYPRE_Int nalu_hypre_IJMatrixDestroyPETSc ( nalu_hypre_IJMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_IJMatrixSetTotalSizePETSc ( nalu_hypre_IJMatrix *matrix, NALU_HYPRE_Int size );

/* IJVector.c */
NALU_HYPRE_Int nalu_hypre_IJVectorDistribute ( NALU_HYPRE_IJVector vector, const NALU_HYPRE_Int *vec_starts );
NALU_HYPRE_Int nalu_hypre_IJVectorZeroValues ( NALU_HYPRE_IJVector vector );

/* IJVector_parcsr.c */
NALU_HYPRE_Int nalu_hypre_IJVectorCreatePar ( nalu_hypre_IJVector *vector, NALU_HYPRE_BigInt *IJpartitioning );
NALU_HYPRE_Int nalu_hypre_IJVectorDestroyPar ( nalu_hypre_IJVector *vector );
NALU_HYPRE_Int nalu_hypre_IJVectorInitializePar ( nalu_hypre_IJVector *vector );
NALU_HYPRE_Int nalu_hypre_IJVectorInitializePar_v2(nalu_hypre_IJVector *vector,
                                         NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int nalu_hypre_IJVectorSetMaxOffProcElmtsPar ( nalu_hypre_IJVector *vector,
                                                NALU_HYPRE_Int max_off_proc_elmts );
NALU_HYPRE_Int nalu_hypre_IJVectorDistributePar ( nalu_hypre_IJVector *vector, const NALU_HYPRE_Int *vec_starts );
NALU_HYPRE_Int nalu_hypre_IJVectorZeroValuesPar ( nalu_hypre_IJVector *vector );
NALU_HYPRE_Int nalu_hypre_IJVectorSetComponentPar ( nalu_hypre_IJVector *vector, NALU_HYPRE_Int component);
NALU_HYPRE_Int nalu_hypre_IJVectorSetValuesPar ( nalu_hypre_IJVector *vector, NALU_HYPRE_Int num_values,
                                       const NALU_HYPRE_BigInt *indices, const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_IJVectorAddToValuesPar ( nalu_hypre_IJVector *vector, NALU_HYPRE_Int num_values,
                                         const NALU_HYPRE_BigInt *indices, const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_IJVectorAssemblePar ( nalu_hypre_IJVector *vector );
NALU_HYPRE_Int nalu_hypre_IJVectorGetValuesPar ( nalu_hypre_IJVector *vector, NALU_HYPRE_Int num_values,
                                       const NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_IJVectorAssembleOffProcValsPar ( nalu_hypre_IJVector *vector,
                                                 NALU_HYPRE_Int max_off_proc_elmts, NALU_HYPRE_Int current_num_elmts, NALU_HYPRE_MemoryLocation memory_location,
                                                 NALU_HYPRE_BigInt *off_proc_i, NALU_HYPRE_Complex *off_proc_data );
NALU_HYPRE_Int nalu_hypre_IJVectorSetAddValuesParDevice(nalu_hypre_IJVector *vector, NALU_HYPRE_Int num_values,
                                              const NALU_HYPRE_BigInt *indices, const NALU_HYPRE_Complex *values, const char *action);
NALU_HYPRE_Int nalu_hypre_IJVectorAssembleParDevice(nalu_hypre_IJVector *vector);

NALU_HYPRE_Int nalu_hypre_IJVectorUpdateValuesDevice( nalu_hypre_IJVector *vector, NALU_HYPRE_Int num_values,
                                            const NALU_HYPRE_BigInt *indices, const NALU_HYPRE_Complex *values, NALU_HYPRE_Int action);

/* NALU_HYPRE_IJMatrix.c */
NALU_HYPRE_Int NALU_HYPRE_IJMatrixCreate ( MPI_Comm comm, NALU_HYPRE_BigInt ilower, NALU_HYPRE_BigInt iupper,
                                 NALU_HYPRE_BigInt jlower, NALU_HYPRE_BigInt jupper, NALU_HYPRE_IJMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixDestroy ( NALU_HYPRE_IJMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixInitialize ( NALU_HYPRE_IJMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetPrintLevel ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetValues ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Int nrows, NALU_HYPRE_Int *ncols,
                                    const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_BigInt *cols, const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetConstantValues ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Complex value );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixAddToValues ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Int nrows, NALU_HYPRE_Int *ncols,
                                      const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_BigInt *cols, const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixAssemble ( NALU_HYPRE_IJMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetRowCounts ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Int nrows, NALU_HYPRE_BigInt *rows,
                                       NALU_HYPRE_Int *ncols );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetValues ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Int nrows, NALU_HYPRE_Int *ncols,
                                    NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetObjectType ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Int type );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetObjectType ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Int *type );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetLocalRange ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_BigInt *ilower,
                                        NALU_HYPRE_BigInt *iupper, NALU_HYPRE_BigInt *jlower, NALU_HYPRE_BigInt *jupper );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetObject ( NALU_HYPRE_IJMatrix matrix, void **object );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetRowSizes ( NALU_HYPRE_IJMatrix matrix, const NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetDiagOffdSizes ( NALU_HYPRE_IJMatrix matrix, const NALU_HYPRE_Int *diag_sizes,
                                           const NALU_HYPRE_Int *offdiag_sizes );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetMaxOffProcElmts ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Int max_off_proc_elmts );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixRead ( const char *filename, MPI_Comm comm, NALU_HYPRE_Int type,
                               NALU_HYPRE_IJMatrix *matrix_ptr );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixReadMM( const char *filename, MPI_Comm comm, NALU_HYPRE_Int type,
                                NALU_HYPRE_IJMatrix *matrix_ptr );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixPrint ( NALU_HYPRE_IJMatrix matrix, const char *filename );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetOMPFlag ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Int omp_flag );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixTranspose ( NALU_HYPRE_IJMatrix  matrix_A, NALU_HYPRE_IJMatrix *matrix_AT );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixNorm ( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_IJMatrixAdd ( NALU_HYPRE_Complex alpha, NALU_HYPRE_IJMatrix matrix_A, NALU_HYPRE_Complex beta,
                              NALU_HYPRE_IJMatrix matrix_B, NALU_HYPRE_IJMatrix *matrix_C );

/* NALU_HYPRE_IJVector.c */
NALU_HYPRE_Int NALU_HYPRE_IJVectorCreate ( MPI_Comm comm, NALU_HYPRE_BigInt jlower, NALU_HYPRE_BigInt jupper,
                                 NALU_HYPRE_IJVector *vector );
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetNumComponents ( NALU_HYPRE_IJVector vector, NALU_HYPRE_Int num_components );
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetComponent ( NALU_HYPRE_IJVector vector, NALU_HYPRE_Int component );
NALU_HYPRE_Int NALU_HYPRE_IJVectorDestroy ( NALU_HYPRE_IJVector vector );
NALU_HYPRE_Int NALU_HYPRE_IJVectorInitialize ( NALU_HYPRE_IJVector vector );
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetPrintLevel ( NALU_HYPRE_IJVector vector, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetValues ( NALU_HYPRE_IJVector vector, NALU_HYPRE_Int nvalues,
                                    const NALU_HYPRE_BigInt *indices, const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_IJVectorAddToValues ( NALU_HYPRE_IJVector vector, NALU_HYPRE_Int nvalues,
                                      const NALU_HYPRE_BigInt *indices, const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_IJVectorAssemble ( NALU_HYPRE_IJVector vector );
NALU_HYPRE_Int NALU_HYPRE_IJVectorGetValues ( NALU_HYPRE_IJVector vector, NALU_HYPRE_Int nvalues,
                                    const NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetMaxOffProcElmts ( NALU_HYPRE_IJVector vector, NALU_HYPRE_Int max_off_proc_elmts );
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetObjectType ( NALU_HYPRE_IJVector vector, NALU_HYPRE_Int type );
NALU_HYPRE_Int NALU_HYPRE_IJVectorGetObjectType ( NALU_HYPRE_IJVector vector, NALU_HYPRE_Int *type );
NALU_HYPRE_Int NALU_HYPRE_IJVectorGetLocalRange ( NALU_HYPRE_IJVector vector, NALU_HYPRE_BigInt *jlower,
                                        NALU_HYPRE_BigInt *jupper );
NALU_HYPRE_Int NALU_HYPRE_IJVectorGetObject ( NALU_HYPRE_IJVector vector, void **object );
NALU_HYPRE_Int NALU_HYPRE_IJVectorRead ( const char *filename, MPI_Comm comm, NALU_HYPRE_Int type,
                               NALU_HYPRE_IJVector *vector_ptr );
NALU_HYPRE_Int NALU_HYPRE_IJVectorPrint ( NALU_HYPRE_IJVector vector, const char *filename );
NALU_HYPRE_Int NALU_HYPRE_IJVectorInnerProd ( NALU_HYPRE_IJVector x, NALU_HYPRE_IJVector y, NALU_HYPRE_Real *prod );
