/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* communicationT.c */
void nalu_hypre_RowsWithColumn_original ( NALU_HYPRE_Int *rowmin, NALU_HYPRE_Int *rowmax, NALU_HYPRE_BigInt column,
                                     nalu_hypre_ParCSRMatrix *A );
void nalu_hypre_RowsWithColumn ( NALU_HYPRE_Int *rowmin, NALU_HYPRE_Int *rowmax, NALU_HYPRE_BigInt column,
                            NALU_HYPRE_Int num_rows_diag, NALU_HYPRE_BigInt firstColDiag, NALU_HYPRE_BigInt *colMapOffd, NALU_HYPRE_Int *mat_i_diag,
                            NALU_HYPRE_Int *mat_j_diag, NALU_HYPRE_Int *mat_i_offd, NALU_HYPRE_Int *mat_j_offd );
void nalu_hypre_MatTCommPkgCreate_core ( MPI_Comm comm, NALU_HYPRE_BigInt *col_map_offd,
                                    NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *col_starts, NALU_HYPRE_Int num_rows_diag,
                                    NALU_HYPRE_Int num_cols_diag, NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_BigInt *row_starts,
                                    NALU_HYPRE_BigInt firstColDiag, NALU_HYPRE_BigInt *colMapOffd, NALU_HYPRE_Int *mat_i_diag, NALU_HYPRE_Int *mat_j_diag,
                                    NALU_HYPRE_Int *mat_i_offd, NALU_HYPRE_Int *mat_j_offd, NALU_HYPRE_Int data, NALU_HYPRE_Int *p_num_recvs,
                                    NALU_HYPRE_Int **p_recv_procs, NALU_HYPRE_Int **p_recv_vec_starts, NALU_HYPRE_Int *p_num_sends,
                                    NALU_HYPRE_Int **p_send_procs, NALU_HYPRE_Int **p_send_map_starts, NALU_HYPRE_Int **p_send_map_elmts );
NALU_HYPRE_Int nalu_hypre_MatTCommPkgCreate ( nalu_hypre_ParCSRMatrix *A );

/* driver_aat.c */

/* driver_boolaat.c */

/* driver_boolmatmul.c */

/* driver.c */

/* driver_matmul.c */

/* driver_mat_multivec.c */

/* driver_matvec.c */

/* driver_multivec.c */

/* NALU_HYPRE_parcsr_matrix.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_num_rows,
                                     NALU_HYPRE_BigInt global_num_cols, NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts,
                                     NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int num_nonzeros_diag, NALU_HYPRE_Int num_nonzeros_offd,
                                     NALU_HYPRE_ParCSRMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixDestroy ( NALU_HYPRE_ParCSRMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixInitialize ( NALU_HYPRE_ParCSRMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixBigInitialize ( NALU_HYPRE_ParCSRMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixRead ( MPI_Comm comm, const char *file_name,
                                   NALU_HYPRE_ParCSRMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixPrint ( NALU_HYPRE_ParCSRMatrix matrix, const char *file_name );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetComm ( NALU_HYPRE_ParCSRMatrix matrix, MPI_Comm *comm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetDims ( NALU_HYPRE_ParCSRMatrix matrix, NALU_HYPRE_BigInt *M, NALU_HYPRE_BigInt *N );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetRowPartitioning ( NALU_HYPRE_ParCSRMatrix matrix,
                                                 NALU_HYPRE_BigInt **row_partitioning_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetGlobalRowPartitioning ( NALU_HYPRE_ParCSRMatrix matrix,
                                                       NALU_HYPRE_Int all_procs, NALU_HYPRE_BigInt **row_partitioning_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetColPartitioning ( NALU_HYPRE_ParCSRMatrix matrix,
                                                 NALU_HYPRE_BigInt **col_partitioning_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetLocalRange ( NALU_HYPRE_ParCSRMatrix matrix, NALU_HYPRE_BigInt *row_start,
                                            NALU_HYPRE_BigInt *row_end, NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetRow ( NALU_HYPRE_ParCSRMatrix matrix, NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size,
                                     NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixRestoreRow ( NALU_HYPRE_ParCSRMatrix matrix, NALU_HYPRE_BigInt row,
                                         NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
NALU_HYPRE_Int NALU_HYPRE_CSRMatrixToParCSRMatrix ( MPI_Comm comm, NALU_HYPRE_CSRMatrix A_CSR,
                                          NALU_HYPRE_BigInt *row_partitioning, NALU_HYPRE_BigInt *col_partitioning, NALU_HYPRE_ParCSRMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning ( MPI_Comm comm, NALU_HYPRE_CSRMatrix A_CSR,
                                                              NALU_HYPRE_ParCSRMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixMatvec ( NALU_HYPRE_Complex alpha, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector x,
                                     NALU_HYPRE_Complex beta, NALU_HYPRE_ParVector y );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixMatvecT ( NALU_HYPRE_Complex alpha, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector x,
                                      NALU_HYPRE_Complex beta, NALU_HYPRE_ParVector y );

/* NALU_HYPRE_parcsr_vector.c */
NALU_HYPRE_Int NALU_HYPRE_ParVectorCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_size,
                                  NALU_HYPRE_BigInt *partitioning, NALU_HYPRE_ParVector *vector );
NALU_HYPRE_Int NALU_HYPRE_ParMultiVectorCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_size,
                                       NALU_HYPRE_BigInt *partitioning, NALU_HYPRE_Int number_vectors, NALU_HYPRE_ParVector *vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorDestroy ( NALU_HYPRE_ParVector vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorInitialize ( NALU_HYPRE_ParVector vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorRead ( MPI_Comm comm, const char *file_name, NALU_HYPRE_ParVector *vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorPrint ( NALU_HYPRE_ParVector vector, const char *file_name );
NALU_HYPRE_Int NALU_HYPRE_ParVectorSetConstantValues ( NALU_HYPRE_ParVector vector, NALU_HYPRE_Complex value );
NALU_HYPRE_Int NALU_HYPRE_ParVectorSetRandomValues ( NALU_HYPRE_ParVector vector, NALU_HYPRE_Int seed );
NALU_HYPRE_Int NALU_HYPRE_ParVectorCopy ( NALU_HYPRE_ParVector x, NALU_HYPRE_ParVector y );
NALU_HYPRE_ParVector NALU_HYPRE_ParVectorCloneShallow ( NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParVectorScale ( NALU_HYPRE_Complex value, NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParVectorAxpy ( NALU_HYPRE_Complex alpha, NALU_HYPRE_ParVector x, NALU_HYPRE_ParVector y );
NALU_HYPRE_Int NALU_HYPRE_ParVectorInnerProd ( NALU_HYPRE_ParVector x, NALU_HYPRE_ParVector y, NALU_HYPRE_Real *prod );
NALU_HYPRE_Int NALU_HYPRE_VectorToParVector ( MPI_Comm comm, NALU_HYPRE_Vector b, NALU_HYPRE_BigInt *partitioning,
                                    NALU_HYPRE_ParVector *vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorGetValues ( NALU_HYPRE_ParVector vector, NALU_HYPRE_Int num_values,
                                     NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *values);

/* gen_fffc.c */
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateFFFCHost( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                              NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S,
                                              nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                              nalu_hypre_ParCSRMatrix **A_FF_ptr ) ;
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateFFFC( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                          NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S,
                                          nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                          nalu_hypre_ParCSRMatrix **A_FF_ptr ) ;
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateFFFC3(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                          NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S, nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                          nalu_hypre_ParCSRMatrix **A_FF_ptr ) ;
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateFFFCD3(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                           NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S, nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                           nalu_hypre_ParCSRMatrix **A_FF_ptr, NALU_HYPRE_Real **D_lambda_ptr ) ;
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateFFFC3Device(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S, nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                                nalu_hypre_ParCSRMatrix **A_FF_ptr ) ;
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateCFDevice( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                              NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S, nalu_hypre_ParCSRMatrix **ACF_ptr) ;
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateCCDevice( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                              NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S, nalu_hypre_ParCSRMatrix **ACC_ptr) ;
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerate1DCFDevice( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S, nalu_hypre_ParCSRMatrix **ACX_ptr,
                                                nalu_hypre_ParCSRMatrix **AXC_ptr ) ;

/* new_commpkg.c */
NALU_HYPRE_Int nalu_hypre_PrintCommpkg ( nalu_hypre_ParCSRMatrix *A, const char *file_name );
NALU_HYPRE_Int nalu_hypre_ParCSRCommPkgCreateApart_core ( MPI_Comm comm, NALU_HYPRE_BigInt *col_map_off_d,
                                                NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_Int num_cols_off_d, NALU_HYPRE_BigInt global_num_cols,
                                                NALU_HYPRE_Int *p_num_recvs, NALU_HYPRE_Int **p_recv_procs, NALU_HYPRE_Int **p_recv_vec_starts,
                                                NALU_HYPRE_Int *p_num_sends, NALU_HYPRE_Int **p_send_procs, NALU_HYPRE_Int **p_send_map_starts,
                                                NALU_HYPRE_Int **p_send_map_elements, nalu_hypre_IJAssumedPart *apart );
NALU_HYPRE_Int nalu_hypre_ParCSRCommPkgCreateApart ( MPI_Comm  comm, NALU_HYPRE_BigInt *col_map_off_d,
                                           NALU_HYPRE_BigInt  first_col_diag, NALU_HYPRE_Int  num_cols_off_d, NALU_HYPRE_BigInt  global_num_cols,
                                           nalu_hypre_IJAssumedPart *apart, nalu_hypre_ParCSRCommPkg *comm_pkg );
NALU_HYPRE_Int nalu_hypre_NewCommPkgDestroy ( nalu_hypre_ParCSRMatrix *parcsr_A );
NALU_HYPRE_Int nalu_hypre_RangeFillResponseIJDetermineRecvProcs ( void *p_recv_contact_buf,
                                                        NALU_HYPRE_Int contact_size, NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                        NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Int nalu_hypre_FillResponseIJDetermineSendProcs ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                                   NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                   NALU_HYPRE_Int *response_message_size );

/* numbers.c */
nalu_hypre_NumbersNode *nalu_hypre_NumbersNewNode ( void );
void nalu_hypre_NumbersDeleteNode ( nalu_hypre_NumbersNode *node );
NALU_HYPRE_Int nalu_hypre_NumbersEnter ( nalu_hypre_NumbersNode *node, const NALU_HYPRE_Int n );
NALU_HYPRE_Int nalu_hypre_NumbersNEntered ( nalu_hypre_NumbersNode *node );
NALU_HYPRE_Int nalu_hypre_NumbersQuery ( nalu_hypre_NumbersNode *node, const NALU_HYPRE_Int n );
NALU_HYPRE_Int *nalu_hypre_NumbersArray ( nalu_hypre_NumbersNode *node );

/* parchord_to_parcsr.c */
void nalu_hypre_ParChordMatrix_RowStarts ( nalu_hypre_ParChordMatrix *Ac, MPI_Comm comm,
                                      NALU_HYPRE_BigInt **row_starts, NALU_HYPRE_BigInt *global_num_cols );
NALU_HYPRE_Int nalu_hypre_ParChordMatrixToParCSRMatrix ( nalu_hypre_ParChordMatrix *Ac, MPI_Comm comm,
                                               nalu_hypre_ParCSRMatrix **pAp );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixToParChordMatrix ( nalu_hypre_ParCSRMatrix *Ap, MPI_Comm comm,
                                               nalu_hypre_ParChordMatrix **pAc );

/* par_csr_aat.c */
void nalu_hypre_ParAat_RowSizes ( NALU_HYPRE_Int **C_diag_i, NALU_HYPRE_Int **C_offd_i, NALU_HYPRE_Int *B_marker,
                             NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j, NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j,
                             NALU_HYPRE_BigInt *A_col_map_offd, NALU_HYPRE_Int *A_ext_i, NALU_HYPRE_BigInt *A_ext_j,
                             NALU_HYPRE_BigInt *A_ext_row_map, NALU_HYPRE_Int *C_diag_size, NALU_HYPRE_Int *C_offd_size,
                             NALU_HYPRE_Int num_rows_diag_A, NALU_HYPRE_Int num_cols_offd_A, NALU_HYPRE_Int num_rows_A_ext,
                             NALU_HYPRE_BigInt first_col_diag_A, NALU_HYPRE_BigInt first_row_index_A );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRAAt ( nalu_hypre_ParCSRMatrix *A );
nalu_hypre_CSRMatrix *nalu_hypre_ParCSRMatrixExtractAExt ( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int data,
                                                 NALU_HYPRE_BigInt **pA_ext_row_map );

/* par_csr_assumed_part.c */
NALU_HYPRE_Int nalu_hypre_LocateAssumedPartition ( MPI_Comm comm, NALU_HYPRE_BigInt row_start,
                                         NALU_HYPRE_BigInt row_end, NALU_HYPRE_BigInt global_first_row, NALU_HYPRE_BigInt global_num_rows,
                                         nalu_hypre_IJAssumedPart *part, NALU_HYPRE_Int myid );
nalu_hypre_IJAssumedPart *nalu_hypre_AssumedPartitionCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_num,
                                                    NALU_HYPRE_BigInt start, NALU_HYPRE_BigInt end );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixCreateAssumedPartition ( nalu_hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_AssumedPartitionDestroy ( nalu_hypre_IJAssumedPart *apart );
NALU_HYPRE_Int nalu_hypre_GetAssumedPartitionProcFromRow ( MPI_Comm comm, NALU_HYPRE_BigInt row,
                                                 NALU_HYPRE_BigInt global_first_row, NALU_HYPRE_BigInt global_num_rows, NALU_HYPRE_Int *proc_id );
NALU_HYPRE_Int nalu_hypre_GetAssumedPartitionRowRange ( MPI_Comm comm, NALU_HYPRE_Int proc_id,
                                              NALU_HYPRE_BigInt global_first_row, NALU_HYPRE_BigInt global_num_rows, NALU_HYPRE_BigInt *row_start,
                                              NALU_HYPRE_BigInt *row_end );
NALU_HYPRE_Int nalu_hypre_ParVectorCreateAssumedPartition ( nalu_hypre_ParVector *vector );

/* par_csr_bool_matop.c */
nalu_hypre_ParCSRBooleanMatrix *nalu_hypre_ParBooleanMatmul ( nalu_hypre_ParCSRBooleanMatrix *A,
                                                    nalu_hypre_ParCSRBooleanMatrix *B );
nalu_hypre_CSRBooleanMatrix *nalu_hypre_ParCSRBooleanMatrixExtractBExt ( nalu_hypre_ParCSRBooleanMatrix *B,
                                                               nalu_hypre_ParCSRBooleanMatrix *A );
nalu_hypre_CSRBooleanMatrix *nalu_hypre_ParCSRBooleanMatrixExtractAExt ( nalu_hypre_ParCSRBooleanMatrix *A,
                                                               NALU_HYPRE_BigInt **pA_ext_row_map );
nalu_hypre_ParCSRBooleanMatrix *nalu_hypre_ParBooleanAAt ( nalu_hypre_ParCSRBooleanMatrix *A );
NALU_HYPRE_Int nalu_hypre_BooleanMatTCommPkgCreate ( nalu_hypre_ParCSRBooleanMatrix *A );
NALU_HYPRE_Int nalu_hypre_BooleanMatvecCommPkgCreate ( nalu_hypre_ParCSRBooleanMatrix *A );

/* par_csr_bool_matrix.c */
nalu_hypre_CSRBooleanMatrix *nalu_hypre_CSRBooleanMatrixCreate ( NALU_HYPRE_Int num_rows, NALU_HYPRE_Int num_cols,
                                                       NALU_HYPRE_Int num_nonzeros );
NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixDestroy ( nalu_hypre_CSRBooleanMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixInitialize ( nalu_hypre_CSRBooleanMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixBigInitialize ( nalu_hypre_CSRBooleanMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixSetDataOwner ( nalu_hypre_CSRBooleanMatrix *matrix,
                                               NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixSetBigDataOwner ( nalu_hypre_CSRBooleanMatrix *matrix,
                                                  NALU_HYPRE_Int owns_data );
nalu_hypre_CSRBooleanMatrix *nalu_hypre_CSRBooleanMatrixRead ( const char *file_name );
NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixPrint ( nalu_hypre_CSRBooleanMatrix *matrix, const char *file_name );
nalu_hypre_ParCSRBooleanMatrix *nalu_hypre_ParCSRBooleanMatrixCreate ( MPI_Comm comm,
                                                             NALU_HYPRE_BigInt global_num_rows, NALU_HYPRE_BigInt global_num_cols, NALU_HYPRE_BigInt *row_starts,
                                                             NALU_HYPRE_BigInt *col_starts, NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int num_nonzeros_diag,
                                                             NALU_HYPRE_Int num_nonzeros_offd );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixDestroy ( nalu_hypre_ParCSRBooleanMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixInitialize ( nalu_hypre_ParCSRBooleanMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixSetNNZ ( nalu_hypre_ParCSRBooleanMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixSetDataOwner ( nalu_hypre_ParCSRBooleanMatrix *matrix,
                                                  NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixSetRowStartsOwner ( nalu_hypre_ParCSRBooleanMatrix *matrix,
                                                       NALU_HYPRE_Int owns_row_starts );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixSetColStartsOwner ( nalu_hypre_ParCSRBooleanMatrix *matrix,
                                                       NALU_HYPRE_Int owns_col_starts );
nalu_hypre_ParCSRBooleanMatrix *nalu_hypre_ParCSRBooleanMatrixRead ( MPI_Comm comm, const char *file_name );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixPrint ( nalu_hypre_ParCSRBooleanMatrix *matrix,
                                           const char *file_name );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixPrintIJ ( nalu_hypre_ParCSRBooleanMatrix *matrix,
                                             const char *filename );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixGetLocalRange ( nalu_hypre_ParCSRBooleanMatrix *matrix,
                                                   NALU_HYPRE_BigInt *row_start, NALU_HYPRE_BigInt *row_end, NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixGetRow ( nalu_hypre_ParCSRBooleanMatrix *mat, NALU_HYPRE_BigInt row,
                                            NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind );
NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixRestoreRow ( nalu_hypre_ParCSRBooleanMatrix *matrix, NALU_HYPRE_BigInt row,
                                                NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind );
NALU_HYPRE_Int nalu_hypre_BuildCSRBooleanMatrixMPIDataType ( NALU_HYPRE_Int num_nonzeros, NALU_HYPRE_Int num_rows,
                                                   NALU_HYPRE_Int *a_i, NALU_HYPRE_Int *a_j, nalu_hypre_MPI_Datatype *csr_matrix_datatype );
nalu_hypre_ParCSRBooleanMatrix *nalu_hypre_CSRBooleanMatrixToParCSRBooleanMatrix ( MPI_Comm comm,
                                                                         nalu_hypre_CSRBooleanMatrix *A, NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts );
NALU_HYPRE_Int nalu_hypre_BooleanGenerateDiagAndOffd ( nalu_hypre_CSRBooleanMatrix *A,
                                             nalu_hypre_ParCSRBooleanMatrix *matrix, NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt last_col_diag );

/* par_csr_communication.c */
nalu_hypre_ParCSRCommHandle *nalu_hypre_ParCSRCommHandleCreate ( NALU_HYPRE_Int job, nalu_hypre_ParCSRCommPkg *comm_pkg,
                                                       void *send_data, void *recv_data );
nalu_hypre_ParCSRCommHandle *nalu_hypre_ParCSRCommHandleCreate_v2 ( NALU_HYPRE_Int job,
                                                          nalu_hypre_ParCSRCommPkg *comm_pkg,
                                                          NALU_HYPRE_MemoryLocation send_memory_location,
                                                          void *send_data_in,
                                                          NALU_HYPRE_MemoryLocation recv_memory_location,
                                                          void *recv_data_in );
NALU_HYPRE_Int nalu_hypre_ParCSRCommHandleDestroy ( nalu_hypre_ParCSRCommHandle *comm_handle );
void nalu_hypre_ParCSRCommPkgCreate_core ( MPI_Comm comm, NALU_HYPRE_BigInt *col_map_offd,
                                      NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *col_starts, NALU_HYPRE_Int num_cols_diag,
                                      NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int *p_num_recvs, NALU_HYPRE_Int **p_recv_procs,
                                      NALU_HYPRE_Int **p_recv_vec_starts, NALU_HYPRE_Int *p_num_sends, NALU_HYPRE_Int **p_send_procs,
                                      NALU_HYPRE_Int **p_send_map_starts, NALU_HYPRE_Int **p_send_map_elmts );
NALU_HYPRE_Int nalu_hypre_ParCSRCommPkgCreate(MPI_Comm comm, NALU_HYPRE_BigInt *col_map_offd,
                                    NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *col_starts,
                                    NALU_HYPRE_Int num_cols_diag, NALU_HYPRE_Int num_cols_offd,
                                    nalu_hypre_ParCSRCommPkg *comm_pkg);
NALU_HYPRE_Int nalu_hypre_ParCSRCommPkgCreateAndFill ( MPI_Comm comm, NALU_HYPRE_Int num_recvs,
                                             NALU_HYPRE_Int *recv_procs, NALU_HYPRE_Int *recv_vec_starts,
                                             NALU_HYPRE_Int num_sends, NALU_HYPRE_Int *send_procs,
                                             NALU_HYPRE_Int *send_map_starts, NALU_HYPRE_Int *send_map_elmts,
                                             nalu_hypre_ParCSRCommPkg **comm_pkg_ptr );
NALU_HYPRE_Int nalu_hypre_ParCSRCommPkgUpdateVecStarts ( nalu_hypre_ParCSRCommPkg *comm_pkg, nalu_hypre_ParVector *x );
NALU_HYPRE_Int nalu_hypre_MatvecCommPkgCreate ( nalu_hypre_ParCSRMatrix *A );
NALU_HYPRE_Int nalu_hypre_MatvecCommPkgDestroy ( nalu_hypre_ParCSRCommPkg *comm_pkg );
NALU_HYPRE_Int nalu_hypre_BuildCSRMatrixMPIDataType ( NALU_HYPRE_Int num_nonzeros, NALU_HYPRE_Int num_rows,
                                            NALU_HYPRE_Complex *a_data, NALU_HYPRE_Int *a_i, NALU_HYPRE_Int *a_j,
                                            nalu_hypre_MPI_Datatype *csr_matrix_datatype );
NALU_HYPRE_Int nalu_hypre_BuildCSRJDataType ( NALU_HYPRE_Int num_nonzeros, NALU_HYPRE_Complex *a_data, NALU_HYPRE_Int *a_j,
                                    nalu_hypre_MPI_Datatype *csr_jdata_datatype );
NALU_HYPRE_Int nalu_hypre_ParCSRFindExtendCommPkg(MPI_Comm comm, NALU_HYPRE_BigInt global_num_cols,
                                        NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_Int num_cols_diag, NALU_HYPRE_BigInt *col_starts,
                                        nalu_hypre_IJAssumedPart *apart, NALU_HYPRE_Int indices_len, NALU_HYPRE_BigInt *indices,
                                        nalu_hypre_ParCSRCommPkg **extend_comm_pkg);

/* par_csr_matop.c */
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixScale(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Complex scalar);
void nalu_hypre_ParMatmul_RowSizes ( NALU_HYPRE_MemoryLocation memory_location, NALU_HYPRE_Int **C_diag_i,
                                NALU_HYPRE_Int **C_offd_i, NALU_HYPRE_Int *rownnz_A, NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j,
                                NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j, NALU_HYPRE_Int *B_diag_i, NALU_HYPRE_Int *B_diag_j,
                                NALU_HYPRE_Int *B_offd_i, NALU_HYPRE_Int *B_offd_j, NALU_HYPRE_Int *B_ext_diag_i, NALU_HYPRE_Int *B_ext_diag_j,
                                NALU_HYPRE_Int *B_ext_offd_i, NALU_HYPRE_Int *B_ext_offd_j, NALU_HYPRE_Int *map_B_to_C, NALU_HYPRE_Int *C_diag_size,
                                NALU_HYPRE_Int *C_offd_size, NALU_HYPRE_Int num_rownnz_A, NALU_HYPRE_Int num_rows_diag_A,
                                NALU_HYPRE_Int num_cols_offd_A, NALU_HYPRE_Int  allsquare, NALU_HYPRE_Int num_cols_diag_B,
                                NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_Int num_cols_offd_C );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParMatmul ( nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix *B );
void nalu_hypre_ParCSRMatrixExtractBExt_Arrays ( NALU_HYPRE_Int **pB_ext_i, NALU_HYPRE_BigInt **pB_ext_j,
                                            NALU_HYPRE_Complex **pB_ext_data, NALU_HYPRE_BigInt **pB_ext_row_map, NALU_HYPRE_Int *num_nonzeros, NALU_HYPRE_Int data,
                                            NALU_HYPRE_Int find_row_map, MPI_Comm comm, nalu_hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int num_cols_B,
                                            NALU_HYPRE_Int num_recvs, NALU_HYPRE_Int num_sends, NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *row_starts,
                                            NALU_HYPRE_Int *recv_vec_starts, NALU_HYPRE_Int *send_map_starts, NALU_HYPRE_Int *send_map_elmts,
                                            NALU_HYPRE_Int *diag_i, NALU_HYPRE_Int *diag_j, NALU_HYPRE_Int *offd_i, NALU_HYPRE_Int *offd_j,
                                            NALU_HYPRE_BigInt *col_map_offd, NALU_HYPRE_Real *diag_data, NALU_HYPRE_Real *offd_data );
void nalu_hypre_ParCSRMatrixExtractBExt_Arrays_Overlap ( NALU_HYPRE_Int **pB_ext_i, NALU_HYPRE_BigInt **pB_ext_j,
                                                    NALU_HYPRE_Complex **pB_ext_data, NALU_HYPRE_BigInt **pB_ext_row_map, NALU_HYPRE_Int *num_nonzeros, NALU_HYPRE_Int data,
                                                    NALU_HYPRE_Int find_row_map, MPI_Comm comm, nalu_hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int num_cols_B,
                                                    NALU_HYPRE_Int num_recvs, NALU_HYPRE_Int num_sends, NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *row_starts,
                                                    NALU_HYPRE_Int *recv_vec_starts, NALU_HYPRE_Int *send_map_starts, NALU_HYPRE_Int *send_map_elmts,
                                                    NALU_HYPRE_Int *diag_i, NALU_HYPRE_Int *diag_j, NALU_HYPRE_Int *offd_i, NALU_HYPRE_Int *offd_j,
                                                    NALU_HYPRE_BigInt *col_map_offd, NALU_HYPRE_Real *diag_data, NALU_HYPRE_Real *offd_data,
                                                    nalu_hypre_ParCSRCommHandle **comm_handle_idx, nalu_hypre_ParCSRCommHandle **comm_handle_data,
                                                    NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *CF_marker_offd, NALU_HYPRE_Int skip_fine, NALU_HYPRE_Int skip_same_sign );
nalu_hypre_CSRMatrix *nalu_hypre_ParCSRMatrixExtractBExt ( nalu_hypre_ParCSRMatrix *B, nalu_hypre_ParCSRMatrix *A,
                                                 NALU_HYPRE_Int data );
nalu_hypre_CSRMatrix *nalu_hypre_ParCSRMatrixExtractBExt_Overlap ( nalu_hypre_ParCSRMatrix *B,
                                                         nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int data, nalu_hypre_ParCSRCommHandle **comm_handle_idx,
                                                         nalu_hypre_ParCSRCommHandle **comm_handle_data, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *CF_marker_offd,
                                                         NALU_HYPRE_Int skip_fine, NALU_HYPRE_Int skip_same_sign );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixExtractBExtDeviceInit( nalu_hypre_ParCSRMatrix *B, nalu_hypre_ParCSRMatrix *A,
                                                   NALU_HYPRE_Int want_data, void **request_ptr);
nalu_hypre_CSRMatrix* nalu_hypre_ParCSRMatrixExtractBExtDeviceWait(void *request);
nalu_hypre_CSRMatrix* nalu_hypre_ParCSRMatrixExtractBExtDevice( nalu_hypre_ParCSRMatrix *B, nalu_hypre_ParCSRMatrix *A,
                                                      NALU_HYPRE_Int want_data );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixLocalTranspose( nalu_hypre_ParCSRMatrix  *A );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixTranspose ( nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix **AT_ptr,
                                        NALU_HYPRE_Int data );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixTransposeHost ( nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix **AT_ptr,
                                            NALU_HYPRE_Int data );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixTransposeDevice ( nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix **AT_ptr,
                                              NALU_HYPRE_Int data );
void nalu_hypre_ParCSRMatrixGenSpanningTree ( nalu_hypre_ParCSRMatrix *G_csr, NALU_HYPRE_Int **indices,
                                         NALU_HYPRE_Int G_type );
void nalu_hypre_ParCSRMatrixExtractSubmatrices ( nalu_hypre_ParCSRMatrix *A_csr, NALU_HYPRE_Int *indices2,
                                            nalu_hypre_ParCSRMatrix ***submatrices );
void nalu_hypre_ParCSRMatrixExtractRowSubmatrices ( nalu_hypre_ParCSRMatrix *A_csr, NALU_HYPRE_Int *indices2,
                                               nalu_hypre_ParCSRMatrix ***submatrices );
NALU_HYPRE_Complex nalu_hypre_ParCSRMatrixLocalSumElts ( nalu_hypre_ParCSRMatrix *A );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixAminvDB ( nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix *B,
                                      NALU_HYPRE_Complex *d, nalu_hypre_ParCSRMatrix **C_ptr );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParTMatmul ( nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix *B );
NALU_HYPRE_Real nalu_hypre_ParCSRMatrixFnorm( nalu_hypre_ParCSRMatrix *A );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixInfNorm ( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int nalu_hypre_ExchangeExternalRowsInit( nalu_hypre_CSRMatrix *B_ext, nalu_hypre_ParCSRCommPkg *comm_pkg_A,
                                          void **request_ptr);
nalu_hypre_CSRMatrix* nalu_hypre_ExchangeExternalRowsWait(void *vequest);
NALU_HYPRE_Int nalu_hypre_ExchangeExternalRowsDeviceInit( nalu_hypre_CSRMatrix *B_ext,
                                                nalu_hypre_ParCSRCommPkg *comm_pkg_A, NALU_HYPRE_Int want_data, void **request_ptr);
nalu_hypre_CSRMatrix* nalu_hypre_ExchangeExternalRowsDeviceWait(void *vrequest);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateFFFCDevice( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S,
                                                nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                                nalu_hypre_ParCSRMatrix **A_FF_ptr );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateFFCFDevice( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S,
                                                nalu_hypre_ParCSRMatrix **A_CF_ptr,
                                                nalu_hypre_ParCSRMatrix **A_FF_ptr );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGenerateCCCFDevice( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, nalu_hypre_ParCSRMatrix *S,
                                                nalu_hypre_ParCSRMatrix **A_CF_ptr,
                                                nalu_hypre_ParCSRMatrix **A_CC_ptr );
nalu_hypre_CSRMatrix* nalu_hypre_ConcatDiagAndOffdDevice(nalu_hypre_ParCSRMatrix *A);
NALU_HYPRE_Int nalu_hypre_ConcatDiagOffdAndExtDevice(nalu_hypre_ParCSRMatrix *A, nalu_hypre_CSRMatrix *E,
                                           nalu_hypre_CSRMatrix **B_ptr, NALU_HYPRE_Int *num_cols_offd_ptr, NALU_HYPRE_BigInt **cols_map_offd_ptr);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGetRowDevice( nalu_hypre_ParCSRMatrix *mat, NALU_HYPRE_BigInt row,
                                          NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
NALU_HYPRE_Int nalu_hypre_ParCSRDiagScaleVector( nalu_hypre_ParCSRMatrix *par_A, nalu_hypre_ParVector *par_y,
                                       nalu_hypre_ParVector *par_x );
NALU_HYPRE_Int nalu_hypre_ParCSRDiagScaleVectorHost( nalu_hypre_ParCSRMatrix *par_A, nalu_hypre_ParVector *par_y,
                                           nalu_hypre_ParVector *par_x );
NALU_HYPRE_Int nalu_hypre_ParCSRDiagScaleVectorDevice( nalu_hypre_ParCSRMatrix *par_A, nalu_hypre_ParVector *par_y,
                                             nalu_hypre_ParVector *par_x );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixDropSmallEntries( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Real tol,
                                              NALU_HYPRE_Int type);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixDropSmallEntriesHost( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Real tol,
                                                  NALU_HYPRE_Int type);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixDropSmallEntriesDevice( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Complex tol,
                                                    NALU_HYPRE_Int type);

NALU_HYPRE_Int nalu_hypre_ParCSRCommPkgCreateMatrixE( nalu_hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int local_ncols );

#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
nalu_hypre_ParCSRPersistentCommHandle* nalu_hypre_ParCSRPersistentCommHandleCreate(NALU_HYPRE_Int job,
                                                                         nalu_hypre_ParCSRCommPkg *comm_pkg);
nalu_hypre_ParCSRPersistentCommHandle* nalu_hypre_ParCSRCommPkgGetPersistentCommHandle(NALU_HYPRE_Int job,
                                                                             nalu_hypre_ParCSRCommPkg *comm_pkg);
void nalu_hypre_ParCSRPersistentCommHandleDestroy(nalu_hypre_ParCSRPersistentCommHandle *comm_handle);
void nalu_hypre_ParCSRPersistentCommHandleStart(nalu_hypre_ParCSRPersistentCommHandle *comm_handle,
                                           NALU_HYPRE_MemoryLocation send_memory_location, void *send_data);
void nalu_hypre_ParCSRPersistentCommHandleWait(nalu_hypre_ParCSRPersistentCommHandle *comm_handle,
                                          NALU_HYPRE_MemoryLocation recv_memory_location, void *recv_data);
#endif

NALU_HYPRE_Int nalu_hypre_ParcsrGetExternalRowsInit( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int indices_len,
                                           NALU_HYPRE_BigInt *indices, nalu_hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int want_data, void **request_ptr);
nalu_hypre_CSRMatrix* nalu_hypre_ParcsrGetExternalRowsWait(void *vrequest);
NALU_HYPRE_Int nalu_hypre_ParcsrGetExternalRowsDeviceInit( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int indices_len,
                                                 NALU_HYPRE_BigInt *indices, nalu_hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int want_data, void **request_ptr);
nalu_hypre_CSRMatrix* nalu_hypre_ParcsrGetExternalRowsDeviceWait(void *vrequest);

NALU_HYPRE_Int nalu_hypre_ParvecBdiagInvScal( nalu_hypre_ParVector *b, NALU_HYPRE_Int blockSize, nalu_hypre_ParVector **bs,
                                    nalu_hypre_ParCSRMatrix *A);

NALU_HYPRE_Int nalu_hypre_ParcsrBdiagInvScal( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int blockSize,
                                    nalu_hypre_ParCSRMatrix **As);

NALU_HYPRE_Int nalu_hypre_ParCSRMatrixExtractSubmatrixFC( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, const char *job,
                                                nalu_hypre_ParCSRMatrix **B_ptr,
                                                NALU_HYPRE_Real strength_thresh);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixDiagScale( nalu_hypre_ParCSRMatrix *par_A, nalu_hypre_ParVector *par_ld,
                                       nalu_hypre_ParVector *par_rd );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixReorder ( nalu_hypre_ParCSRMatrix *A );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixAdd( NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Complex beta,
                                 nalu_hypre_ParCSRMatrix *B, nalu_hypre_ParCSRMatrix **Cout);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixAddHost( NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRMatrix *A,
                                     NALU_HYPRE_Complex beta, nalu_hypre_ParCSRMatrix *B,
                                     nalu_hypre_ParCSRMatrix **Cout);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixAddDevice( NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRMatrix *A,
                                       NALU_HYPRE_Complex beta, nalu_hypre_ParCSRMatrix *B,
                                       nalu_hypre_ParCSRMatrix **Cout);

/* par_csr_matop_device.c */
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixDiagScaleDevice ( nalu_hypre_ParCSRMatrix *par_A, nalu_hypre_ParVector *par_ld,
                                              nalu_hypre_ParVector *par_rd );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixCompressOffdMapDevice(nalu_hypre_ParCSRMatrix *A);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixCompressOffdMap(nalu_hypre_ParCSRMatrix *A);

/* par_csr_matop_marked.c */
void nalu_hypre_ParMatmul_RowSizes_Marked ( NALU_HYPRE_Int **C_diag_i, NALU_HYPRE_Int **C_offd_i,
                                       NALU_HYPRE_Int **B_marker, NALU_HYPRE_Int *A_diag_i,
                                       NALU_HYPRE_Int *A_diag_j, NALU_HYPRE_Int *A_offd_i,
                                       NALU_HYPRE_Int *A_offd_j, NALU_HYPRE_Int *B_diag_i,
                                       NALU_HYPRE_Int *B_diag_j, NALU_HYPRE_Int *B_offd_i,
                                       NALU_HYPRE_Int *B_offd_j, NALU_HYPRE_Int *B_ext_diag_i,
                                       NALU_HYPRE_Int *B_ext_diag_j, NALU_HYPRE_Int *B_ext_offd_i,
                                       NALU_HYPRE_Int *B_ext_offd_j, NALU_HYPRE_Int *map_B_to_C,
                                       NALU_HYPRE_Int *C_diag_size, NALU_HYPRE_Int *C_offd_size,
                                       NALU_HYPRE_Int num_rows_diag_A, NALU_HYPRE_Int num_cols_offd_A,
                                       NALU_HYPRE_Int allsquare, NALU_HYPRE_Int num_cols_diag_B,
                                       NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_Int num_cols_offd_C,
                                       NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *dof_func,
                                       NALU_HYPRE_Int *dof_func_offd );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParMatmul_FC ( nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix *P,
                                         NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *dof_func,
                                         NALU_HYPRE_Int *dof_func_offd );
void nalu_hypre_ParMatScaleDiagInv_F ( nalu_hypre_ParCSRMatrix *C, nalu_hypre_ParCSRMatrix *A,
                                  NALU_HYPRE_Complex weight, NALU_HYPRE_Int *CF_marker );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParMatMinus_F ( nalu_hypre_ParCSRMatrix *P, nalu_hypre_ParCSRMatrix *C,
                                          NALU_HYPRE_Int *CF_marker );
void nalu_hypre_ParCSRMatrixZero_F ( nalu_hypre_ParCSRMatrix *P, NALU_HYPRE_Int *CF_marker );
void nalu_hypre_ParCSRMatrixCopy_C ( nalu_hypre_ParCSRMatrix *P, nalu_hypre_ParCSRMatrix *C,
                                NALU_HYPRE_Int *CF_marker );
void nalu_hypre_ParCSRMatrixDropEntries ( nalu_hypre_ParCSRMatrix *C, nalu_hypre_ParCSRMatrix *P,
                                     NALU_HYPRE_Int *CF_marker );

/* par_csr_matrix.c */
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRMatrixCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_num_rows,
                                               NALU_HYPRE_BigInt global_num_cols, NALU_HYPRE_BigInt *row_starts_in, NALU_HYPRE_BigInt *col_starts_in,
                                               NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int num_nonzeros_diag, NALU_HYPRE_Int num_nonzeros_offd );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixDestroy ( nalu_hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixInitialize_v2( nalu_hypre_ParCSRMatrix *matrix,
                                           NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixInitialize ( nalu_hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixSetNumNonzeros ( nalu_hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixSetDNumNonzeros ( nalu_hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixSetNumRownnz ( nalu_hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixSetDataOwner ( nalu_hypre_ParCSRMatrix *matrix, NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixSetPatternOnly( nalu_hypre_ParCSRMatrix *matrix, NALU_HYPRE_Int pattern_only);
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRMatrixRead ( MPI_Comm comm, const char *file_name );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixPrint ( nalu_hypre_ParCSRMatrix *matrix, const char *file_name );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixPrintIJ ( const nalu_hypre_ParCSRMatrix *matrix, const NALU_HYPRE_Int base_i,
                                      const NALU_HYPRE_Int base_j, const char *filename );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixReadIJ ( MPI_Comm comm, const char *filename, NALU_HYPRE_Int *base_i_ptr,
                                     NALU_HYPRE_Int *base_j_ptr, nalu_hypre_ParCSRMatrix **matrix_ptr );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGetLocalRange ( nalu_hypre_ParCSRMatrix *matrix, NALU_HYPRE_BigInt *row_start,
                                            NALU_HYPRE_BigInt *row_end, NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixGetRow ( nalu_hypre_ParCSRMatrix *mat, NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size,
                                     NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixRestoreRow ( nalu_hypre_ParCSRMatrix *matrix, NALU_HYPRE_BigInt row,
                                         NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
nalu_hypre_ParCSRMatrix *nalu_hypre_CSRMatrixToParCSRMatrix ( MPI_Comm comm, nalu_hypre_CSRMatrix *A,
                                                    NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts );
NALU_HYPRE_Int GenerateDiagAndOffd ( nalu_hypre_CSRMatrix *A, nalu_hypre_ParCSRMatrix *matrix,
                                NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt last_col_diag );
nalu_hypre_CSRMatrix *nalu_hypre_MergeDiagAndOffd ( nalu_hypre_ParCSRMatrix *par_matrix );
nalu_hypre_CSRMatrix *nalu_hypre_MergeDiagAndOffdDevice ( nalu_hypre_ParCSRMatrix *par_matrix );
nalu_hypre_CSRMatrix *nalu_hypre_ParCSRMatrixToCSRMatrixAll ( nalu_hypre_ParCSRMatrix *par_matrix );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixCopy ( nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix *B,
                                   NALU_HYPRE_Int copy_data );
NALU_HYPRE_Int nalu_hypre_FillResponseParToCSRMatrix ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                             NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             NALU_HYPRE_Int *response_message_size );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRMatrixUnion ( nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix *B );
nalu_hypre_ParCSRMatrix* nalu_hypre_ParCSRMatrixClone ( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int copy_data );
#define nalu_hypre_ParCSRMatrixCompleteClone(A) nalu_hypre_ParCSRMatrixClone(A,0)
nalu_hypre_ParCSRMatrix* nalu_hypre_ParCSRMatrixClone_v2 ( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int copy_data,
                                                 NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixTruncate(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Real tol,
                                     NALU_HYPRE_Int max_row_elmts, NALU_HYPRE_Int rescale,
                                     NALU_HYPRE_Int nrm_type);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMigrate(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixSetConstantValues( nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Complex value );
void nalu_hypre_ParCSRMatrixCopyColMapOffdToDevice(nalu_hypre_ParCSRMatrix *A);
void nalu_hypre_ParCSRMatrixCopyColMapOffdToHost(nalu_hypre_ParCSRMatrix *A);

/* par_csr_matvec.c */
// y = alpha*A*x + beta*b
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMatvecOutOfPlace ( NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRMatrix *A,
                                               nalu_hypre_ParVector *x, NALU_HYPRE_Complex beta,
                                               nalu_hypre_ParVector *b, nalu_hypre_ParVector *y );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMatvecOutOfPlaceDevice ( NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRMatrix *A,
                                                     nalu_hypre_ParVector *x, NALU_HYPRE_Complex beta,
                                                     nalu_hypre_ParVector *b, nalu_hypre_ParVector *y );
// y = alpha*A*x + beta*y
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMatvec ( NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParVector *x,
                                     NALU_HYPRE_Complex beta, nalu_hypre_ParVector *y );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMatvecT ( NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRMatrix *A,
                                      nalu_hypre_ParVector *x, NALU_HYPRE_Complex beta, nalu_hypre_ParVector *y );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMatvecTDevice ( NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRMatrix *A,
                                            nalu_hypre_ParVector *x, NALU_HYPRE_Complex beta,
                                            nalu_hypre_ParVector *y );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMatvecT_unpack( nalu_hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int num_cols,
                                            NALU_HYPRE_Complex *recv_data, NALU_HYPRE_Complex *local_data );
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMatvec_FF ( NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRMatrix *A,
                                        nalu_hypre_ParVector *x, NALU_HYPRE_Complex beta, nalu_hypre_ParVector *y,
                                        NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int fpt );

/* par_csr_triplemat.c */
NALU_HYPRE_Int nalu_hypre_ParCSRTMatMatPartialAddDevice( nalu_hypre_ParCSRCommPkg *comm_pkg_A,
                                               NALU_HYPRE_Int num_cols_A, NALU_HYPRE_Int num_cols_B, NALU_HYPRE_BigInt first_col_diag_B,
                                               NALU_HYPRE_BigInt last_col_diag_B, NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_BigInt *col_map_offd_B,
                                               NALU_HYPRE_Int local_nnz_Cbar, nalu_hypre_CSRMatrix *Cbar, nalu_hypre_CSRMatrix *Cext,
                                               nalu_hypre_CSRMatrix **C_diag_ptr, nalu_hypre_CSRMatrix **C_offd_ptr, NALU_HYPRE_Int *num_cols_offd_C_ptr,
                                               NALU_HYPRE_BigInt **col_map_offd_C_ptr );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRMatMat( nalu_hypre_ParCSRMatrix  *A, nalu_hypre_ParCSRMatrix  *B );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRMatMatHost( nalu_hypre_ParCSRMatrix  *A, nalu_hypre_ParCSRMatrix  *B );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRMatMatDevice( nalu_hypre_ParCSRMatrix  *A, nalu_hypre_ParCSRMatrix  *B );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRTMatMatKTHost( nalu_hypre_ParCSRMatrix  *A, nalu_hypre_ParCSRMatrix  *B,
                                               NALU_HYPRE_Int keep_transpose);
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRTMatMatKTDevice( nalu_hypre_ParCSRMatrix  *A, nalu_hypre_ParCSRMatrix  *B,
                                                 NALU_HYPRE_Int keep_transpose);
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRTMatMatKT( nalu_hypre_ParCSRMatrix  *A, nalu_hypre_ParCSRMatrix  *B,
                                           NALU_HYPRE_Int keep_transpose);
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRTMatMat( nalu_hypre_ParCSRMatrix  *A, nalu_hypre_ParCSRMatrix  *B);
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRMatrixRAPKT( nalu_hypre_ParCSRMatrix *R, nalu_hypre_ParCSRMatrix  *A,
                                             nalu_hypre_ParCSRMatrix  *P, NALU_HYPRE_Int keepTranspose );
nalu_hypre_ParCSRMatrix *nalu_hypre_ParCSRMatrixRAP( nalu_hypre_ParCSRMatrix *R, nalu_hypre_ParCSRMatrix  *A,
                                           nalu_hypre_ParCSRMatrix  *P );
nalu_hypre_ParCSRMatrix* nalu_hypre_ParCSRMatrixRAPKTDevice( nalu_hypre_ParCSRMatrix *R, nalu_hypre_ParCSRMatrix *A,
                                                   nalu_hypre_ParCSRMatrix *P, NALU_HYPRE_Int keep_transpose );
nalu_hypre_ParCSRMatrix* nalu_hypre_ParCSRMatrixRAPKTHost( nalu_hypre_ParCSRMatrix *R, nalu_hypre_ParCSRMatrix *A,
                                                 nalu_hypre_ParCSRMatrix *P, NALU_HYPRE_Int keep_transpose );

/* par_make_system.c */
NALU_HYPRE_ParCSR_System_Problem *NALU_HYPRE_Generate2DSystem ( NALU_HYPRE_ParCSRMatrix H_L1,
                                                      NALU_HYPRE_ParCSRMatrix H_L2, NALU_HYPRE_ParVector H_b1, NALU_HYPRE_ParVector H_b2, NALU_HYPRE_ParVector H_x1,
                                                      NALU_HYPRE_ParVector H_x2, NALU_HYPRE_Complex *M_vals );
NALU_HYPRE_Int NALU_HYPRE_Destroy2DSystem ( NALU_HYPRE_ParCSR_System_Problem *sys_prob );

/* par_vector.c */
nalu_hypre_ParVector *nalu_hypre_ParVectorCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_size,
                                         NALU_HYPRE_BigInt *partitioning_in );
nalu_hypre_ParVector *nalu_hypre_ParMultiVectorCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_size,
                                              NALU_HYPRE_BigInt *partitioning, NALU_HYPRE_Int num_vectors );
NALU_HYPRE_Int nalu_hypre_ParVectorDestroy ( nalu_hypre_ParVector *vector );
NALU_HYPRE_Int nalu_hypre_ParVectorInitialize ( nalu_hypre_ParVector *vector );
NALU_HYPRE_Int nalu_hypre_ParVectorInitialize_v2( nalu_hypre_ParVector *vector,
                                        NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int nalu_hypre_ParVectorSetDataOwner ( nalu_hypre_ParVector *vector, NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int nalu_hypre_ParVectorSetLocalSize ( nalu_hypre_ParVector *vector, NALU_HYPRE_Int local_size );
NALU_HYPRE_Int nalu_hypre_ParVectorSetNumVectors ( nalu_hypre_ParVector *vector, NALU_HYPRE_Int num_vectors );
NALU_HYPRE_Int nalu_hypre_ParVectorSetComponent ( nalu_hypre_ParVector *vector, NALU_HYPRE_Int component );
NALU_HYPRE_Int nalu_hypre_ParVectorResize ( nalu_hypre_ParVector *vector, NALU_HYPRE_Int num_vectors );
nalu_hypre_ParVector *nalu_hypre_ParVectorRead ( MPI_Comm comm, const char *file_name );
NALU_HYPRE_Int nalu_hypre_ParVectorPrint ( nalu_hypre_ParVector *vector, const char *file_name );
NALU_HYPRE_Int nalu_hypre_ParVectorSetConstantValues ( nalu_hypre_ParVector *v, NALU_HYPRE_Complex value );
NALU_HYPRE_Int nalu_hypre_ParVectorSetZeros( nalu_hypre_ParVector *v );
NALU_HYPRE_Int nalu_hypre_ParVectorSetRandomValues ( nalu_hypre_ParVector *v, NALU_HYPRE_Int seed );
NALU_HYPRE_Int nalu_hypre_ParVectorCopy ( nalu_hypre_ParVector *x, nalu_hypre_ParVector *y );
nalu_hypre_ParVector *nalu_hypre_ParVectorCloneShallow ( nalu_hypre_ParVector *x );
nalu_hypre_ParVector *nalu_hypre_ParVectorCloneDeep_v2( nalu_hypre_ParVector *x,
                                              NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int nalu_hypre_ParVectorMigrate(nalu_hypre_ParVector *x, NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int nalu_hypre_ParVectorScale ( NALU_HYPRE_Complex alpha, nalu_hypre_ParVector *y );
NALU_HYPRE_Int nalu_hypre_ParVectorAxpy ( NALU_HYPRE_Complex alpha, nalu_hypre_ParVector *x, nalu_hypre_ParVector *y );
NALU_HYPRE_Int nalu_hypre_ParVectorAxpyz ( NALU_HYPRE_Complex alpha, nalu_hypre_ParVector *x,
                                 NALU_HYPRE_Complex beta, nalu_hypre_ParVector *y,
                                 nalu_hypre_ParVector *z );
NALU_HYPRE_Int nalu_hypre_ParVectorMassAxpy ( NALU_HYPRE_Complex *alpha, nalu_hypre_ParVector **x, nalu_hypre_ParVector *y,
                                    NALU_HYPRE_Int k, NALU_HYPRE_Int unroll);
NALU_HYPRE_Real nalu_hypre_ParVectorInnerProd ( nalu_hypre_ParVector *x, nalu_hypre_ParVector *y );
NALU_HYPRE_Int nalu_hypre_ParVectorMassInnerProd ( nalu_hypre_ParVector *x, nalu_hypre_ParVector **y, NALU_HYPRE_Int k,
                                         NALU_HYPRE_Int unroll, NALU_HYPRE_Real *prod );
NALU_HYPRE_Int nalu_hypre_ParVectorMassDotpTwo ( nalu_hypre_ParVector *x, nalu_hypre_ParVector *y, nalu_hypre_ParVector **z,
                                       NALU_HYPRE_Int k, NALU_HYPRE_Int unroll, NALU_HYPRE_Real *prod_x, NALU_HYPRE_Real *prod_y );
nalu_hypre_ParVector *nalu_hypre_VectorToParVector ( MPI_Comm comm, nalu_hypre_Vector *v,
                                           NALU_HYPRE_BigInt *vec_starts );
nalu_hypre_Vector *nalu_hypre_ParVectorToVectorAll ( nalu_hypre_ParVector *par_v );
NALU_HYPRE_Int nalu_hypre_ParVectorPrintIJ ( nalu_hypre_ParVector *vector, NALU_HYPRE_Int base_j,
                                   const char *filename );
NALU_HYPRE_Int nalu_hypre_ParVectorReadIJ ( MPI_Comm comm, const char *filename, NALU_HYPRE_Int *base_j_ptr,
                                  nalu_hypre_ParVector **vector_ptr );
NALU_HYPRE_Int nalu_hypre_FillResponseParToVectorAll ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                             NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Complex nalu_hypre_ParVectorLocalSumElts ( nalu_hypre_ParVector *vector );
NALU_HYPRE_Int nalu_hypre_ParVectorGetValues ( nalu_hypre_ParVector *vector, NALU_HYPRE_Int num_values,
                                     NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *values);
NALU_HYPRE_Int nalu_hypre_ParVectorGetValues2( nalu_hypre_ParVector *vector, NALU_HYPRE_Int num_values,
                                     NALU_HYPRE_BigInt *indices, NALU_HYPRE_BigInt base, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_ParVectorGetValuesHost(nalu_hypre_ParVector *vector, NALU_HYPRE_Int num_values,
                                       NALU_HYPRE_BigInt *indices, NALU_HYPRE_BigInt base, NALU_HYPRE_Complex *values);
NALU_HYPRE_Int nalu_hypre_ParVectorElmdivpy( nalu_hypre_ParVector *x, nalu_hypre_ParVector *b, nalu_hypre_ParVector *y );
NALU_HYPRE_Int nalu_hypre_ParVectorElmdivpyMarked( nalu_hypre_ParVector *x, nalu_hypre_ParVector *b,
                                         nalu_hypre_ParVector *y, NALU_HYPRE_Int *marker,
                                         NALU_HYPRE_Int marker_val );
/* par_vector_device.c */
NALU_HYPRE_Int nalu_hypre_ParVectorGetValuesDevice(nalu_hypre_ParVector *vector, NALU_HYPRE_Int num_values,
                                         NALU_HYPRE_BigInt *indices, NALU_HYPRE_BigInt base,
                                         NALU_HYPRE_Complex *values);
