/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* aux_parcsr_matrix.c */
NALU_HYPRE_Int hypre_AuxParCSRMatrixCreate ( hypre_AuxParCSRMatrix **aux_matrix,
                                        NALU_HYPRE_Int local_num_rows, NALU_HYPRE_Int local_num_cols, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int hypre_AuxParCSRMatrixDestroy ( hypre_AuxParCSRMatrix *matrix );
NALU_HYPRE_Int hypre_AuxParCSRMatrixSetRownnz ( hypre_AuxParCSRMatrix *matrix );
NALU_HYPRE_Int hypre_AuxParCSRMatrixInitialize ( hypre_AuxParCSRMatrix *matrix );
NALU_HYPRE_Int hypre_AuxParCSRMatrixInitialize_v2( hypre_AuxParCSRMatrix *matrix,
                                              NALU_HYPRE_MemoryLocation memory_location );

/* aux_par_vector.c */
NALU_HYPRE_Int hypre_AuxParVectorCreate ( hypre_AuxParVector **aux_vector );
NALU_HYPRE_Int hypre_AuxParVectorDestroy ( hypre_AuxParVector *vector );
NALU_HYPRE_Int hypre_AuxParVectorInitialize ( hypre_AuxParVector *vector );
NALU_HYPRE_Int hypre_AuxParVectorInitialize_v2( hypre_AuxParVector *vector,
                                           NALU_HYPRE_MemoryLocation memory_location );

/* IJ_assumed_part.c */
NALU_HYPRE_Int hypre_IJMatrixCreateAssumedPartition ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJVectorCreateAssumedPartition ( hypre_IJVector *vector );

/* IJMatrix.c */
NALU_HYPRE_Int hypre_IJMatrixGetRowPartitioning ( NALU_HYPRE_IJMatrix matrix,
                                             NALU_HYPRE_BigInt **row_partitioning );
NALU_HYPRE_Int hypre_IJMatrixGetColPartitioning ( NALU_HYPRE_IJMatrix matrix,
                                             NALU_HYPRE_BigInt **col_partitioning );
NALU_HYPRE_Int hypre_IJMatrixSetObject ( NALU_HYPRE_IJMatrix matrix, void *object );
NALU_HYPRE_Int hypre_IJMatrixRead( const char *filename, MPI_Comm comm, NALU_HYPRE_Int type,
                              NALU_HYPRE_IJMatrix *matrix_ptr, NALU_HYPRE_Int is_mm );

/* IJMatrix_isis.c */
NALU_HYPRE_Int hypre_IJMatrixSetLocalSizeISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_Int local_m,
                                           NALU_HYPRE_Int local_n );
NALU_HYPRE_Int hypre_IJMatrixCreateISIS ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixSetRowSizesISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int hypre_IJMatrixSetDiagRowSizesISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int hypre_IJMatrixSetOffDiagRowSizesISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int hypre_IJMatrixInitializeISIS ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixInsertBlockISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_Int m, NALU_HYPRE_Int n,
                                          NALU_HYPRE_Int *rows, NALU_HYPRE_Int *cols, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int hypre_IJMatrixAddToBlockISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_Int m, NALU_HYPRE_Int n,
                                         NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int hypre_IJMatrixInsertRowISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_Int n, NALU_HYPRE_BigInt row,
                                        NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int hypre_IJMatrixAddToRowISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_Int n, NALU_HYPRE_BigInt row,
                                       NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int hypre_IJMatrixAssembleISIS ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixDistributeISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_BigInt *row_starts,
                                         NALU_HYPRE_BigInt *col_starts );
NALU_HYPRE_Int hypre_IJMatrixApplyISIS ( hypre_IJMatrix *matrix, hypre_ParVector *x,
                                    hypre_ParVector *b );
NALU_HYPRE_Int hypre_IJMatrixDestroyISIS ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixSetTotalSizeISIS ( hypre_IJMatrix *matrix, NALU_HYPRE_Int size );

/* IJMatrix_parcsr.c */
NALU_HYPRE_Int hypre_IJMatrixCreateParCSR ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixSetRowSizesParCSR ( hypre_IJMatrix *matrix, const NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int hypre_IJMatrixSetDiagOffdSizesParCSR ( hypre_IJMatrix *matrix,
                                                 const NALU_HYPRE_Int *diag_sizes, const NALU_HYPRE_Int *offdiag_sizes );
NALU_HYPRE_Int hypre_IJMatrixSetMaxOffProcElmtsParCSR ( hypre_IJMatrix *matrix,
                                                   NALU_HYPRE_Int max_off_proc_elmts );
NALU_HYPRE_Int hypre_IJMatrixInitializeParCSR ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixGetRowCountsParCSR ( hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                             NALU_HYPRE_BigInt *rows, NALU_HYPRE_Int *ncols );
NALU_HYPRE_Int hypre_IJMatrixGetValuesParCSR ( hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows, NALU_HYPRE_Int *ncols,
                                          NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_IJMatrixSetValuesParCSR ( hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows, NALU_HYPRE_Int *ncols,
                                          const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                          const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_IJMatrixSetAddValuesParCSRDevice ( hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                                   NALU_HYPRE_Int *ncols, const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                                   const NALU_HYPRE_Complex *values, const char *action );
NALU_HYPRE_Int hypre_IJMatrixSetConstantValuesParCSR ( hypre_IJMatrix *matrix, NALU_HYPRE_Complex value );
NALU_HYPRE_Int hypre_IJMatrixAddToValuesParCSR ( hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                            NALU_HYPRE_Int *ncols, const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                            const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_IJMatrixDestroyParCSR ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixTransposeParCSR ( hypre_IJMatrix  *matrix_A, hypre_IJMatrix *matrix_AT );
NALU_HYPRE_Int hypre_IJMatrixNormParCSR ( hypre_IJMatrix *matrix, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int hypre_IJMatrixAddParCSR ( NALU_HYPRE_Complex alpha, hypre_IJMatrix *matrix_A,
                                    NALU_HYPRE_Complex beta, hypre_IJMatrix *matrix_B, hypre_IJMatrix *matrix_C );
NALU_HYPRE_Int hypre_IJMatrixAssembleOffProcValsParCSR ( hypre_IJMatrix *matrix,
                                                    NALU_HYPRE_Int off_proc_i_indx, NALU_HYPRE_Int max_off_proc_elmts, NALU_HYPRE_Int current_num_elmts,
                                                    NALU_HYPRE_MemoryLocation memory_location, NALU_HYPRE_BigInt *off_proc_i, NALU_HYPRE_BigInt *off_proc_j,
                                                    NALU_HYPRE_Complex *off_proc_data );
NALU_HYPRE_Int hypre_FillResponseIJOffProcVals ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                            NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                            NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Int hypre_FindProc ( NALU_HYPRE_BigInt *list, NALU_HYPRE_BigInt value, NALU_HYPRE_Int list_length );
NALU_HYPRE_Int hypre_IJMatrixAssembleParCSR ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixSetValuesOMPParCSR ( hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                             NALU_HYPRE_Int *ncols, const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                             const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_IJMatrixAddToValuesOMPParCSR ( hypre_IJMatrix *matrix, NALU_HYPRE_Int nrows,
                                               NALU_HYPRE_Int *ncols, const NALU_HYPRE_BigInt *rows, const NALU_HYPRE_Int *row_indexes, const NALU_HYPRE_BigInt *cols,
                                               const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_IJMatrixAssembleParCSRDevice(hypre_IJMatrix *matrix);
NALU_HYPRE_Int hypre_IJMatrixInitializeParCSR_v2(hypre_IJMatrix *matrix,
                                            NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int hypre_IJMatrixSetConstantValuesParCSRDevice( hypre_IJMatrix *matrix,
                                                       NALU_HYPRE_Complex value );

/* IJMatrix_petsc.c */
NALU_HYPRE_Int hypre_IJMatrixSetLocalSizePETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_Int local_m,
                                            NALU_HYPRE_Int local_n );
NALU_HYPRE_Int hypre_IJMatrixCreatePETSc ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixSetRowSizesPETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int hypre_IJMatrixSetDiagRowSizesPETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int hypre_IJMatrixSetOffDiagRowSizesPETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_Int *sizes );
NALU_HYPRE_Int hypre_IJMatrixInitializePETSc ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixInsertBlockPETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_Int m, NALU_HYPRE_Int n,
                                           NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int hypre_IJMatrixAddToBlockPETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_Int m, NALU_HYPRE_Int n,
                                          NALU_HYPRE_Int *rows, NALU_HYPRE_Int *cols, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int hypre_IJMatrixInsertRowPETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_Int n, NALU_HYPRE_BigInt row,
                                         NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int hypre_IJMatrixAddToRowPETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_Int n, NALU_HYPRE_BigInt row,
                                        NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *coeffs );
NALU_HYPRE_Int hypre_IJMatrixAssemblePETSc ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixDistributePETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_BigInt *row_starts,
                                          NALU_HYPRE_BigInt *col_starts );
NALU_HYPRE_Int hypre_IJMatrixApplyPETSc ( hypre_IJMatrix *matrix, hypre_ParVector *x,
                                     hypre_ParVector *b );
NALU_HYPRE_Int hypre_IJMatrixDestroyPETSc ( hypre_IJMatrix *matrix );
NALU_HYPRE_Int hypre_IJMatrixSetTotalSizePETSc ( hypre_IJMatrix *matrix, NALU_HYPRE_Int size );

/* IJVector.c */
NALU_HYPRE_Int hypre_IJVectorDistribute ( NALU_HYPRE_IJVector vector, const NALU_HYPRE_Int *vec_starts );
NALU_HYPRE_Int hypre_IJVectorZeroValues ( NALU_HYPRE_IJVector vector );

/* IJVector_parcsr.c */
NALU_HYPRE_Int hypre_IJVectorCreatePar ( hypre_IJVector *vector, NALU_HYPRE_BigInt *IJpartitioning );
NALU_HYPRE_Int hypre_IJVectorDestroyPar ( hypre_IJVector *vector );
NALU_HYPRE_Int hypre_IJVectorInitializePar ( hypre_IJVector *vector );
NALU_HYPRE_Int hypre_IJVectorInitializePar_v2(hypre_IJVector *vector,
                                         NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int hypre_IJVectorSetMaxOffProcElmtsPar ( hypre_IJVector *vector,
                                                NALU_HYPRE_Int max_off_proc_elmts );
NALU_HYPRE_Int hypre_IJVectorDistributePar ( hypre_IJVector *vector, const NALU_HYPRE_Int *vec_starts );
NALU_HYPRE_Int hypre_IJVectorZeroValuesPar ( hypre_IJVector *vector );
NALU_HYPRE_Int hypre_IJVectorSetComponentPar ( hypre_IJVector *vector, NALU_HYPRE_Int component);
NALU_HYPRE_Int hypre_IJVectorSetValuesPar ( hypre_IJVector *vector, NALU_HYPRE_Int num_values,
                                       const NALU_HYPRE_BigInt *indices, const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_IJVectorAddToValuesPar ( hypre_IJVector *vector, NALU_HYPRE_Int num_values,
                                         const NALU_HYPRE_BigInt *indices, const NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_IJVectorAssemblePar ( hypre_IJVector *vector );
NALU_HYPRE_Int hypre_IJVectorGetValuesPar ( hypre_IJVector *vector, NALU_HYPRE_Int num_values,
                                       const NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_IJVectorAssembleOffProcValsPar ( hypre_IJVector *vector,
                                                 NALU_HYPRE_Int max_off_proc_elmts, NALU_HYPRE_Int current_num_elmts, NALU_HYPRE_MemoryLocation memory_location,
                                                 NALU_HYPRE_BigInt *off_proc_i, NALU_HYPRE_Complex *off_proc_data );
NALU_HYPRE_Int hypre_IJVectorSetAddValuesParDevice(hypre_IJVector *vector, NALU_HYPRE_Int num_values,
                                              const NALU_HYPRE_BigInt *indices, const NALU_HYPRE_Complex *values, const char *action);
NALU_HYPRE_Int hypre_IJVectorAssembleParDevice(hypre_IJVector *vector);

NALU_HYPRE_Int hypre_IJVectorUpdateValuesDevice( hypre_IJVector *vector, NALU_HYPRE_Int num_values,
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
