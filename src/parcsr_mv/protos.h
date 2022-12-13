/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* communicationT.c */
void hypre_RowsWithColumn_original ( NALU_HYPRE_Int *rowmin, NALU_HYPRE_Int *rowmax, NALU_HYPRE_BigInt column,
                                     hypre_ParCSRMatrix *A );
void hypre_RowsWithColumn ( NALU_HYPRE_Int *rowmin, NALU_HYPRE_Int *rowmax, NALU_HYPRE_BigInt column,
                            NALU_HYPRE_Int num_rows_diag, NALU_HYPRE_BigInt firstColDiag, NALU_HYPRE_BigInt *colMapOffd, NALU_HYPRE_Int *mat_i_diag,
                            NALU_HYPRE_Int *mat_j_diag, NALU_HYPRE_Int *mat_i_offd, NALU_HYPRE_Int *mat_j_offd );
void hypre_MatTCommPkgCreate_core ( MPI_Comm comm, NALU_HYPRE_BigInt *col_map_offd,
                                    NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *col_starts, NALU_HYPRE_Int num_rows_diag,
                                    NALU_HYPRE_Int num_cols_diag, NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_BigInt *row_starts,
                                    NALU_HYPRE_BigInt firstColDiag, NALU_HYPRE_BigInt *colMapOffd, NALU_HYPRE_Int *mat_i_diag, NALU_HYPRE_Int *mat_j_diag,
                                    NALU_HYPRE_Int *mat_i_offd, NALU_HYPRE_Int *mat_j_offd, NALU_HYPRE_Int data, NALU_HYPRE_Int *p_num_recvs,
                                    NALU_HYPRE_Int **p_recv_procs, NALU_HYPRE_Int **p_recv_vec_starts, NALU_HYPRE_Int *p_num_sends,
                                    NALU_HYPRE_Int **p_send_procs, NALU_HYPRE_Int **p_send_map_starts, NALU_HYPRE_Int **p_send_map_elmts );
NALU_HYPRE_Int hypre_MatTCommPkgCreate ( hypre_ParCSRMatrix *A );

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
NALU_HYPRE_Int hypre_ParCSRMatrixTruncate(hypre_ParCSRMatrix *A, NALU_HYPRE_Real tol, NALU_HYPRE_Int max_row_elmts,
                                     NALU_HYPRE_Int rescale, NALU_HYPRE_Int nrm_type);
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
NALU_HYPRE_Int hypre_ParCSRMatrixGenerateFFFCHost( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                              NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                              hypre_ParCSRMatrix **A_FC_ptr,
                                              hypre_ParCSRMatrix **A_FF_ptr ) ;
NALU_HYPRE_Int hypre_ParCSRMatrixGenerateFFFC( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                          NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                          hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr ) ;
NALU_HYPRE_Int hypre_ParCSRMatrixGenerateFFFC3(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                          NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr ) ;
NALU_HYPRE_Int hypre_ParCSRMatrixGenerateFFFCD3(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                           NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                           hypre_ParCSRMatrix **A_FF_ptr, NALU_HYPRE_Real **D_lambda_ptr ) ;
NALU_HYPRE_Int hypre_ParCSRMatrixGenerateFFFC3Device(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                                hypre_ParCSRMatrix **A_FF_ptr ) ;
NALU_HYPRE_Int hypre_ParCSRMatrixGenerateCFDevice( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                              NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **ACF_ptr) ;
NALU_HYPRE_Int hypre_ParCSRMatrixGenerateCCDevice( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                              NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **ACC_ptr) ;
NALU_HYPRE_Int hypre_ParCSRMatrixGenerate1DCFDevice( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **ACX_ptr,
                                                hypre_ParCSRMatrix **AXC_ptr ) ;

/* new_commpkg.c */
NALU_HYPRE_Int hypre_PrintCommpkg ( hypre_ParCSRMatrix *A, const char *file_name );
NALU_HYPRE_Int hypre_ParCSRCommPkgCreateApart_core ( MPI_Comm comm, NALU_HYPRE_BigInt *col_map_off_d,
                                                NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_Int num_cols_off_d, NALU_HYPRE_BigInt global_num_cols,
                                                NALU_HYPRE_Int *p_num_recvs, NALU_HYPRE_Int **p_recv_procs, NALU_HYPRE_Int **p_recv_vec_starts,
                                                NALU_HYPRE_Int *p_num_sends, NALU_HYPRE_Int **p_send_procs, NALU_HYPRE_Int **p_send_map_starts,
                                                NALU_HYPRE_Int **p_send_map_elements, hypre_IJAssumedPart *apart );
NALU_HYPRE_Int hypre_ParCSRCommPkgCreateApart ( MPI_Comm  comm, NALU_HYPRE_BigInt *col_map_off_d,
                                           NALU_HYPRE_BigInt  first_col_diag, NALU_HYPRE_Int  num_cols_off_d, NALU_HYPRE_BigInt  global_num_cols,
                                           hypre_IJAssumedPart *apart, hypre_ParCSRCommPkg *comm_pkg );
NALU_HYPRE_Int hypre_NewCommPkgDestroy ( hypre_ParCSRMatrix *parcsr_A );
NALU_HYPRE_Int hypre_RangeFillResponseIJDetermineRecvProcs ( void *p_recv_contact_buf,
                                                        NALU_HYPRE_Int contact_size, NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                        NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Int hypre_FillResponseIJDetermineSendProcs ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                                   NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                   NALU_HYPRE_Int *response_message_size );

/* numbers.c */
hypre_NumbersNode *hypre_NumbersNewNode ( void );
void hypre_NumbersDeleteNode ( hypre_NumbersNode *node );
NALU_HYPRE_Int hypre_NumbersEnter ( hypre_NumbersNode *node, const NALU_HYPRE_Int n );
NALU_HYPRE_Int hypre_NumbersNEntered ( hypre_NumbersNode *node );
NALU_HYPRE_Int hypre_NumbersQuery ( hypre_NumbersNode *node, const NALU_HYPRE_Int n );
NALU_HYPRE_Int *hypre_NumbersArray ( hypre_NumbersNode *node );

/* parchord_to_parcsr.c */
void hypre_ParChordMatrix_RowStarts ( hypre_ParChordMatrix *Ac, MPI_Comm comm,
                                      NALU_HYPRE_BigInt **row_starts, NALU_HYPRE_BigInt *global_num_cols );
NALU_HYPRE_Int hypre_ParChordMatrixToParCSRMatrix ( hypre_ParChordMatrix *Ac, MPI_Comm comm,
                                               hypre_ParCSRMatrix **pAp );
NALU_HYPRE_Int hypre_ParCSRMatrixToParChordMatrix ( hypre_ParCSRMatrix *Ap, MPI_Comm comm,
                                               hypre_ParChordMatrix **pAc );

/* par_csr_aat.c */
void hypre_ParAat_RowSizes ( NALU_HYPRE_Int **C_diag_i, NALU_HYPRE_Int **C_offd_i, NALU_HYPRE_Int *B_marker,
                             NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j, NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j,
                             NALU_HYPRE_BigInt *A_col_map_offd, NALU_HYPRE_Int *A_ext_i, NALU_HYPRE_BigInt *A_ext_j,
                             NALU_HYPRE_BigInt *A_ext_row_map, NALU_HYPRE_Int *C_diag_size, NALU_HYPRE_Int *C_offd_size,
                             NALU_HYPRE_Int num_rows_diag_A, NALU_HYPRE_Int num_cols_offd_A, NALU_HYPRE_Int num_rows_A_ext,
                             NALU_HYPRE_BigInt first_col_diag_A, NALU_HYPRE_BigInt first_row_index_A );
hypre_ParCSRMatrix *hypre_ParCSRAAt ( hypre_ParCSRMatrix *A );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractAExt ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int data,
                                                 NALU_HYPRE_BigInt **pA_ext_row_map );

/* par_csr_assumed_part.c */
NALU_HYPRE_Int hypre_LocateAssumedPartition ( MPI_Comm comm, NALU_HYPRE_BigInt row_start,
                                         NALU_HYPRE_BigInt row_end, NALU_HYPRE_BigInt global_first_row, NALU_HYPRE_BigInt global_num_rows,
                                         hypre_IJAssumedPart *part, NALU_HYPRE_Int myid );
hypre_IJAssumedPart *hypre_AssumedPartitionCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_num,
                                                    NALU_HYPRE_BigInt start, NALU_HYPRE_BigInt end );
NALU_HYPRE_Int hypre_ParCSRMatrixCreateAssumedPartition ( hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int hypre_AssumedPartitionDestroy ( hypre_IJAssumedPart *apart );
NALU_HYPRE_Int hypre_GetAssumedPartitionProcFromRow ( MPI_Comm comm, NALU_HYPRE_BigInt row,
                                                 NALU_HYPRE_BigInt global_first_row, NALU_HYPRE_BigInt global_num_rows, NALU_HYPRE_Int *proc_id );
NALU_HYPRE_Int hypre_GetAssumedPartitionRowRange ( MPI_Comm comm, NALU_HYPRE_Int proc_id,
                                              NALU_HYPRE_BigInt global_first_row, NALU_HYPRE_BigInt global_num_rows, NALU_HYPRE_BigInt *row_start,
                                              NALU_HYPRE_BigInt *row_end );
NALU_HYPRE_Int hypre_ParVectorCreateAssumedPartition ( hypre_ParVector *vector );

/* par_csr_bool_matop.c */
hypre_ParCSRBooleanMatrix *hypre_ParBooleanMatmul ( hypre_ParCSRBooleanMatrix *A,
                                                    hypre_ParCSRBooleanMatrix *B );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractBExt ( hypre_ParCSRBooleanMatrix *B,
                                                               hypre_ParCSRBooleanMatrix *A );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractAExt ( hypre_ParCSRBooleanMatrix *A,
                                                               NALU_HYPRE_BigInt **pA_ext_row_map );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanAAt ( hypre_ParCSRBooleanMatrix *A );
NALU_HYPRE_Int hypre_BooleanMatTCommPkgCreate ( hypre_ParCSRBooleanMatrix *A );
NALU_HYPRE_Int hypre_BooleanMatvecCommPkgCreate ( hypre_ParCSRBooleanMatrix *A );

/* par_csr_bool_matrix.c */
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixCreate ( NALU_HYPRE_Int num_rows, NALU_HYPRE_Int num_cols,
                                                       NALU_HYPRE_Int num_nonzeros );
NALU_HYPRE_Int hypre_CSRBooleanMatrixDestroy ( hypre_CSRBooleanMatrix *matrix );
NALU_HYPRE_Int hypre_CSRBooleanMatrixInitialize ( hypre_CSRBooleanMatrix *matrix );
NALU_HYPRE_Int hypre_CSRBooleanMatrixBigInitialize ( hypre_CSRBooleanMatrix *matrix );
NALU_HYPRE_Int hypre_CSRBooleanMatrixSetDataOwner ( hypre_CSRBooleanMatrix *matrix,
                                               NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int hypre_CSRBooleanMatrixSetBigDataOwner ( hypre_CSRBooleanMatrix *matrix,
                                                  NALU_HYPRE_Int owns_data );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixRead ( const char *file_name );
NALU_HYPRE_Int hypre_CSRBooleanMatrixPrint ( hypre_CSRBooleanMatrix *matrix, const char *file_name );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixCreate ( MPI_Comm comm,
                                                             NALU_HYPRE_BigInt global_num_rows, NALU_HYPRE_BigInt global_num_cols, NALU_HYPRE_BigInt *row_starts,
                                                             NALU_HYPRE_BigInt *col_starts, NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int num_nonzeros_diag,
                                                             NALU_HYPRE_Int num_nonzeros_offd );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixDestroy ( hypre_ParCSRBooleanMatrix *matrix );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixInitialize ( hypre_ParCSRBooleanMatrix *matrix );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixSetNNZ ( hypre_ParCSRBooleanMatrix *matrix );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixSetDataOwner ( hypre_ParCSRBooleanMatrix *matrix,
                                                  NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixSetRowStartsOwner ( hypre_ParCSRBooleanMatrix *matrix,
                                                       NALU_HYPRE_Int owns_row_starts );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixSetColStartsOwner ( hypre_ParCSRBooleanMatrix *matrix,
                                                       NALU_HYPRE_Int owns_col_starts );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixRead ( MPI_Comm comm, const char *file_name );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixPrint ( hypre_ParCSRBooleanMatrix *matrix,
                                           const char *file_name );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixPrintIJ ( hypre_ParCSRBooleanMatrix *matrix,
                                             const char *filename );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixGetLocalRange ( hypre_ParCSRBooleanMatrix *matrix,
                                                   NALU_HYPRE_BigInt *row_start, NALU_HYPRE_BigInt *row_end, NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixGetRow ( hypre_ParCSRBooleanMatrix *mat, NALU_HYPRE_BigInt row,
                                            NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind );
NALU_HYPRE_Int hypre_ParCSRBooleanMatrixRestoreRow ( hypre_ParCSRBooleanMatrix *matrix, NALU_HYPRE_BigInt row,
                                                NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind );
NALU_HYPRE_Int hypre_BuildCSRBooleanMatrixMPIDataType ( NALU_HYPRE_Int num_nonzeros, NALU_HYPRE_Int num_rows,
                                                   NALU_HYPRE_Int *a_i, NALU_HYPRE_Int *a_j, hypre_MPI_Datatype *csr_matrix_datatype );
hypre_ParCSRBooleanMatrix *hypre_CSRBooleanMatrixToParCSRBooleanMatrix ( MPI_Comm comm,
                                                                         hypre_CSRBooleanMatrix *A, NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts );
NALU_HYPRE_Int hypre_BooleanGenerateDiagAndOffd ( hypre_CSRBooleanMatrix *A,
                                             hypre_ParCSRBooleanMatrix *matrix, NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt last_col_diag );

/* par_csr_communication.c */
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate ( NALU_HYPRE_Int job, hypre_ParCSRCommPkg *comm_pkg,
                                                       void *send_data, void *recv_data );
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate_v2 ( NALU_HYPRE_Int job,
                                                          hypre_ParCSRCommPkg *comm_pkg,
                                                          NALU_HYPRE_MemoryLocation send_memory_location,
                                                          void *send_data_in,
                                                          NALU_HYPRE_MemoryLocation recv_memory_location,
                                                          void *recv_data_in );
NALU_HYPRE_Int hypre_ParCSRCommHandleDestroy ( hypre_ParCSRCommHandle *comm_handle );
void hypre_ParCSRCommPkgCreate_core ( MPI_Comm comm, NALU_HYPRE_BigInt *col_map_offd,
                                      NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *col_starts, NALU_HYPRE_Int num_cols_diag,
                                      NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int *p_num_recvs, NALU_HYPRE_Int **p_recv_procs,
                                      NALU_HYPRE_Int **p_recv_vec_starts, NALU_HYPRE_Int *p_num_sends, NALU_HYPRE_Int **p_send_procs,
                                      NALU_HYPRE_Int **p_send_map_starts, NALU_HYPRE_Int **p_send_map_elmts );
NALU_HYPRE_Int hypre_ParCSRCommPkgCreate(MPI_Comm comm, NALU_HYPRE_BigInt *col_map_offd,
                                    NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *col_starts,
                                    NALU_HYPRE_Int num_cols_diag, NALU_HYPRE_Int num_cols_offd,
                                    hypre_ParCSRCommPkg *comm_pkg);
NALU_HYPRE_Int hypre_ParCSRCommPkgCreateAndFill ( MPI_Comm comm, NALU_HYPRE_Int num_recvs,
                                             NALU_HYPRE_Int *recv_procs, NALU_HYPRE_Int *recv_vec_starts,
                                             NALU_HYPRE_Int num_sends, NALU_HYPRE_Int *send_procs,
                                             NALU_HYPRE_Int *send_map_starts, NALU_HYPRE_Int *send_map_elmts,
                                             hypre_ParCSRCommPkg **comm_pkg_ptr );
NALU_HYPRE_Int hypre_ParCSRCommPkgUpdateVecStarts ( hypre_ParCSRCommPkg *comm_pkg, hypre_ParVector *x );
NALU_HYPRE_Int hypre_MatvecCommPkgCreate ( hypre_ParCSRMatrix *A );
NALU_HYPRE_Int hypre_MatvecCommPkgDestroy ( hypre_ParCSRCommPkg *comm_pkg );
NALU_HYPRE_Int hypre_BuildCSRMatrixMPIDataType ( NALU_HYPRE_Int num_nonzeros, NALU_HYPRE_Int num_rows,
                                            NALU_HYPRE_Complex *a_data, NALU_HYPRE_Int *a_i, NALU_HYPRE_Int *a_j,
                                            hypre_MPI_Datatype *csr_matrix_datatype );
NALU_HYPRE_Int hypre_BuildCSRJDataType ( NALU_HYPRE_Int num_nonzeros, NALU_HYPRE_Complex *a_data, NALU_HYPRE_Int *a_j,
                                    hypre_MPI_Datatype *csr_jdata_datatype );
NALU_HYPRE_Int hypre_ParCSRFindExtendCommPkg(MPI_Comm comm, NALU_HYPRE_BigInt global_num_cols,
                                        NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_Int num_cols_diag, NALU_HYPRE_BigInt *col_starts,
                                        hypre_IJAssumedPart *apart, NALU_HYPRE_Int indices_len, NALU_HYPRE_BigInt *indices,
                                        hypre_ParCSRCommPkg **extend_comm_pkg);

/* par_csr_matop.c */
NALU_HYPRE_Int hypre_ParCSRMatrixScale(hypre_ParCSRMatrix *A, NALU_HYPRE_Complex scalar);
void hypre_ParMatmul_RowSizes ( NALU_HYPRE_MemoryLocation memory_location, NALU_HYPRE_Int **C_diag_i,
                                NALU_HYPRE_Int **C_offd_i, NALU_HYPRE_Int *rownnz_A, NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j,
                                NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j, NALU_HYPRE_Int *B_diag_i, NALU_HYPRE_Int *B_diag_j,
                                NALU_HYPRE_Int *B_offd_i, NALU_HYPRE_Int *B_offd_j, NALU_HYPRE_Int *B_ext_diag_i, NALU_HYPRE_Int *B_ext_diag_j,
                                NALU_HYPRE_Int *B_ext_offd_i, NALU_HYPRE_Int *B_ext_offd_j, NALU_HYPRE_Int *map_B_to_C, NALU_HYPRE_Int *C_diag_size,
                                NALU_HYPRE_Int *C_offd_size, NALU_HYPRE_Int num_rownnz_A, NALU_HYPRE_Int num_rows_diag_A,
                                NALU_HYPRE_Int num_cols_offd_A, NALU_HYPRE_Int  allsquare, NALU_HYPRE_Int num_cols_diag_B,
                                NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_Int num_cols_offd_C );
hypre_ParCSRMatrix *hypre_ParMatmul ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
void hypre_ParCSRMatrixExtractBExt_Arrays ( NALU_HYPRE_Int **pB_ext_i, NALU_HYPRE_BigInt **pB_ext_j,
                                            NALU_HYPRE_Complex **pB_ext_data, NALU_HYPRE_BigInt **pB_ext_row_map, NALU_HYPRE_Int *num_nonzeros, NALU_HYPRE_Int data,
                                            NALU_HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int num_cols_B,
                                            NALU_HYPRE_Int num_recvs, NALU_HYPRE_Int num_sends, NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *row_starts,
                                            NALU_HYPRE_Int *recv_vec_starts, NALU_HYPRE_Int *send_map_starts, NALU_HYPRE_Int *send_map_elmts,
                                            NALU_HYPRE_Int *diag_i, NALU_HYPRE_Int *diag_j, NALU_HYPRE_Int *offd_i, NALU_HYPRE_Int *offd_j,
                                            NALU_HYPRE_BigInt *col_map_offd, NALU_HYPRE_Real *diag_data, NALU_HYPRE_Real *offd_data );
void hypre_ParCSRMatrixExtractBExt_Arrays_Overlap ( NALU_HYPRE_Int **pB_ext_i, NALU_HYPRE_BigInt **pB_ext_j,
                                                    NALU_HYPRE_Complex **pB_ext_data, NALU_HYPRE_BigInt **pB_ext_row_map, NALU_HYPRE_Int *num_nonzeros, NALU_HYPRE_Int data,
                                                    NALU_HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int num_cols_B,
                                                    NALU_HYPRE_Int num_recvs, NALU_HYPRE_Int num_sends, NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt *row_starts,
                                                    NALU_HYPRE_Int *recv_vec_starts, NALU_HYPRE_Int *send_map_starts, NALU_HYPRE_Int *send_map_elmts,
                                                    NALU_HYPRE_Int *diag_i, NALU_HYPRE_Int *diag_j, NALU_HYPRE_Int *offd_i, NALU_HYPRE_Int *offd_j,
                                                    NALU_HYPRE_BigInt *col_map_offd, NALU_HYPRE_Real *diag_data, NALU_HYPRE_Real *offd_data,
                                                    hypre_ParCSRCommHandle **comm_handle_idx, hypre_ParCSRCommHandle **comm_handle_data,
                                                    NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *CF_marker_offd, NALU_HYPRE_Int skip_fine, NALU_HYPRE_Int skip_same_sign );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt ( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                                 NALU_HYPRE_Int data );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt_Overlap ( hypre_ParCSRMatrix *B,
                                                         hypre_ParCSRMatrix *A, NALU_HYPRE_Int data, hypre_ParCSRCommHandle **comm_handle_idx,
                                                         hypre_ParCSRCommHandle **comm_handle_data, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *CF_marker_offd,
                                                         NALU_HYPRE_Int skip_fine, NALU_HYPRE_Int skip_same_sign );
NALU_HYPRE_Int hypre_ParCSRMatrixExtractBExtDeviceInit( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                                   NALU_HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParCSRMatrixExtractBExtDeviceWait(void *request);
hypre_CSRMatrix* hypre_ParCSRMatrixExtractBExtDevice( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                                      NALU_HYPRE_Int want_data );
NALU_HYPRE_Int hypre_ParCSRMatrixLocalTranspose( hypre_ParCSRMatrix  *A );
NALU_HYPRE_Int hypre_ParCSRMatrixTranspose ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                        NALU_HYPRE_Int data );
NALU_HYPRE_Int hypre_ParCSRMatrixTransposeHost ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                            NALU_HYPRE_Int data );
NALU_HYPRE_Int hypre_ParCSRMatrixTransposeDevice ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                              NALU_HYPRE_Int data );
void hypre_ParCSRMatrixGenSpanningTree ( hypre_ParCSRMatrix *G_csr, NALU_HYPRE_Int **indices,
                                         NALU_HYPRE_Int G_type );
void hypre_ParCSRMatrixExtractSubmatrices ( hypre_ParCSRMatrix *A_csr, NALU_HYPRE_Int *indices2,
                                            hypre_ParCSRMatrix ***submatrices );
void hypre_ParCSRMatrixExtractRowSubmatrices ( hypre_ParCSRMatrix *A_csr, NALU_HYPRE_Int *indices2,
                                               hypre_ParCSRMatrix ***submatrices );
NALU_HYPRE_Complex hypre_ParCSRMatrixLocalSumElts ( hypre_ParCSRMatrix *A );
NALU_HYPRE_Int hypre_ParCSRMatrixAminvDB ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                      NALU_HYPRE_Complex *d, hypre_ParCSRMatrix **C_ptr );
hypre_ParCSRMatrix *hypre_ParTMatmul ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
NALU_HYPRE_Real hypre_ParCSRMatrixFnorm( hypre_ParCSRMatrix *A );
NALU_HYPRE_Int hypre_ParCSRMatrixInfNorm ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int hypre_ExchangeExternalRowsInit( hypre_CSRMatrix *B_ext, hypre_ParCSRCommPkg *comm_pkg_A,
                                          void **request_ptr);
hypre_CSRMatrix* hypre_ExchangeExternalRowsWait(void *vequest);
NALU_HYPRE_Int hypre_ExchangeExternalRowsDeviceInit( hypre_CSRMatrix *B_ext,
                                                hypre_ParCSRCommPkg *comm_pkg_A, NALU_HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ExchangeExternalRowsDeviceWait(void *vrequest);
NALU_HYPRE_Int hypre_ParCSRMatrixGenerateFFFCDevice( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                                hypre_ParCSRMatrix **A_FF_ptr );
NALU_HYPRE_Int hypre_ParCSRMatrixGenerateFFCFDevice( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_CF_ptr,
                                                hypre_ParCSRMatrix **A_FF_ptr );
hypre_CSRMatrix* hypre_ConcatDiagAndOffdDevice(hypre_ParCSRMatrix *A);
NALU_HYPRE_Int hypre_ConcatDiagOffdAndExtDevice(hypre_ParCSRMatrix *A, hypre_CSRMatrix *E,
                                           hypre_CSRMatrix **B_ptr, NALU_HYPRE_Int *num_cols_offd_ptr, NALU_HYPRE_BigInt **cols_map_offd_ptr);
NALU_HYPRE_Int hypre_ParCSRMatrixGetRowDevice( hypre_ParCSRMatrix *mat, NALU_HYPRE_BigInt row,
                                          NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
NALU_HYPRE_Int hypre_ParCSRDiagScaleVector( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                       hypre_ParVector *par_x );
NALU_HYPRE_Int hypre_ParCSRDiagScaleVectorHost( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                           hypre_ParVector *par_x );
NALU_HYPRE_Int hypre_ParCSRDiagScaleVectorDevice( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                             hypre_ParVector *par_x );
NALU_HYPRE_Int hypre_ParCSRMatrixDropSmallEntries( hypre_ParCSRMatrix *A, NALU_HYPRE_Real tol,
                                              NALU_HYPRE_Int type);
NALU_HYPRE_Int hypre_ParCSRMatrixDropSmallEntriesHost( hypre_ParCSRMatrix *A, NALU_HYPRE_Real tol,
                                                  NALU_HYPRE_Int type);
NALU_HYPRE_Int hypre_ParCSRMatrixDropSmallEntriesDevice( hypre_ParCSRMatrix *A, NALU_HYPRE_Complex tol,
                                                    NALU_HYPRE_Int type);

NALU_HYPRE_Int hypre_ParCSRCommPkgCreateMatrixE( hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int local_ncols );

#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
hypre_ParCSRPersistentCommHandle* hypre_ParCSRPersistentCommHandleCreate(NALU_HYPRE_Int job,
                                                                         hypre_ParCSRCommPkg *comm_pkg);
hypre_ParCSRPersistentCommHandle* hypre_ParCSRCommPkgGetPersistentCommHandle(NALU_HYPRE_Int job,
                                                                             hypre_ParCSRCommPkg *comm_pkg);
void hypre_ParCSRPersistentCommHandleDestroy(hypre_ParCSRPersistentCommHandle *comm_handle);
void hypre_ParCSRPersistentCommHandleStart(hypre_ParCSRPersistentCommHandle *comm_handle,
                                           NALU_HYPRE_MemoryLocation send_memory_location, void *send_data);
void hypre_ParCSRPersistentCommHandleWait(hypre_ParCSRPersistentCommHandle *comm_handle,
                                          NALU_HYPRE_MemoryLocation recv_memory_location, void *recv_data);
#endif

NALU_HYPRE_Int hypre_ParcsrGetExternalRowsInit( hypre_ParCSRMatrix *A, NALU_HYPRE_Int indices_len,
                                           NALU_HYPRE_BigInt *indices, hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParcsrGetExternalRowsWait(void *vrequest);
NALU_HYPRE_Int hypre_ParcsrGetExternalRowsDeviceInit( hypre_ParCSRMatrix *A, NALU_HYPRE_Int indices_len,
                                                 NALU_HYPRE_BigInt *indices, hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParcsrGetExternalRowsDeviceWait(void *vrequest);

NALU_HYPRE_Int hypre_ParvecBdiagInvScal( hypre_ParVector *b, NALU_HYPRE_Int blockSize, hypre_ParVector **bs,
                                    hypre_ParCSRMatrix *A);

NALU_HYPRE_Int hypre_ParcsrBdiagInvScal( hypre_ParCSRMatrix *A, NALU_HYPRE_Int blockSize,
                                    hypre_ParCSRMatrix **As);

NALU_HYPRE_Int hypre_ParCSRMatrixExtractSubmatrixFC( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                NALU_HYPRE_BigInt *cpts_starts, const char *job, hypre_ParCSRMatrix **B_ptr, NALU_HYPRE_Real strength_thresh);
NALU_HYPRE_Int hypre_ParCSRMatrixReorder ( hypre_ParCSRMatrix *A );
NALU_HYPRE_Int hypre_ParCSRMatrixAdd( NALU_HYPRE_Complex alpha, hypre_ParCSRMatrix *A, NALU_HYPRE_Complex beta,
                                 hypre_ParCSRMatrix *B, hypre_ParCSRMatrix **Cout);
NALU_HYPRE_Int hypre_ParCSRMatrixAddHost( NALU_HYPRE_Complex alpha, hypre_ParCSRMatrix *A, NALU_HYPRE_Complex beta,
                                     hypre_ParCSRMatrix *B, hypre_ParCSRMatrix **Cout);
NALU_HYPRE_Int hypre_ParCSRMatrixAddDevice( NALU_HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                       NALU_HYPRE_Complex beta, hypre_ParCSRMatrix *B, hypre_ParCSRMatrix **Cout);

/* par_csr_matop_marked.c */
void hypre_ParMatmul_RowSizes_Marked ( NALU_HYPRE_Int **C_diag_i, NALU_HYPRE_Int **C_offd_i,
                                       NALU_HYPRE_Int **B_marker, NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j, NALU_HYPRE_Int *A_offd_i,
                                       NALU_HYPRE_Int *A_offd_j, NALU_HYPRE_Int *B_diag_i, NALU_HYPRE_Int *B_diag_j, NALU_HYPRE_Int *B_offd_i,
                                       NALU_HYPRE_Int *B_offd_j, NALU_HYPRE_Int *B_ext_diag_i, NALU_HYPRE_Int *B_ext_diag_j, NALU_HYPRE_Int *B_ext_offd_i,
                                       NALU_HYPRE_Int *B_ext_offd_j, NALU_HYPRE_Int *map_B_to_C, NALU_HYPRE_Int *C_diag_size, NALU_HYPRE_Int *C_offd_size,
                                       NALU_HYPRE_Int num_rows_diag_A, NALU_HYPRE_Int num_cols_offd_A, NALU_HYPRE_Int allsquare,
                                       NALU_HYPRE_Int num_cols_diag_B, NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_Int num_cols_offd_C,
                                       NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd );
hypre_ParCSRMatrix *hypre_ParMatmul_FC ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                         NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd );
void hypre_ParMatScaleDiagInv_F ( hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *A,
                                  NALU_HYPRE_Complex weight, NALU_HYPRE_Int *CF_marker );
hypre_ParCSRMatrix *hypre_ParMatMinus_F ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                          NALU_HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixZero_F ( hypre_ParCSRMatrix *P, NALU_HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixCopy_C ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                NALU_HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixDropEntries ( hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *P,
                                     NALU_HYPRE_Int *CF_marker );

/* par_csr_matrix.c */
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_num_rows,
                                               NALU_HYPRE_BigInt global_num_cols, NALU_HYPRE_BigInt *row_starts_in, NALU_HYPRE_BigInt *col_starts_in,
                                               NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int num_nonzeros_diag, NALU_HYPRE_Int num_nonzeros_offd );
NALU_HYPRE_Int hypre_ParCSRMatrixDestroy ( hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int hypre_ParCSRMatrixInitialize_v2( hypre_ParCSRMatrix *matrix,
                                           NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int hypre_ParCSRMatrixInitialize ( hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int hypre_ParCSRMatrixSetNumNonzeros ( hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int hypre_ParCSRMatrixSetDNumNonzeros ( hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int hypre_ParCSRMatrixSetNumRownnz ( hypre_ParCSRMatrix *matrix );
NALU_HYPRE_Int hypre_ParCSRMatrixSetDataOwner ( hypre_ParCSRMatrix *matrix, NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int hypre_ParCSRMatrixSetPatternOnly( hypre_ParCSRMatrix *matrix, NALU_HYPRE_Int pattern_only);
hypre_ParCSRMatrix *hypre_ParCSRMatrixRead ( MPI_Comm comm, const char *file_name );
NALU_HYPRE_Int hypre_ParCSRMatrixPrint ( hypre_ParCSRMatrix *matrix, const char *file_name );
NALU_HYPRE_Int hypre_ParCSRMatrixPrintIJ ( const hypre_ParCSRMatrix *matrix, const NALU_HYPRE_Int base_i,
                                      const NALU_HYPRE_Int base_j, const char *filename );
NALU_HYPRE_Int hypre_ParCSRMatrixReadIJ ( MPI_Comm comm, const char *filename, NALU_HYPRE_Int *base_i_ptr,
                                     NALU_HYPRE_Int *base_j_ptr, hypre_ParCSRMatrix **matrix_ptr );
NALU_HYPRE_Int hypre_ParCSRMatrixGetLocalRange ( hypre_ParCSRMatrix *matrix, NALU_HYPRE_BigInt *row_start,
                                            NALU_HYPRE_BigInt *row_end, NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int hypre_ParCSRMatrixGetRow ( hypre_ParCSRMatrix *mat, NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size,
                                     NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
NALU_HYPRE_Int hypre_ParCSRMatrixRestoreRow ( hypre_ParCSRMatrix *matrix, NALU_HYPRE_BigInt row,
                                         NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix ( MPI_Comm comm, hypre_CSRMatrix *A,
                                                    NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts );
NALU_HYPRE_Int GenerateDiagAndOffd ( hypre_CSRMatrix *A, hypre_ParCSRMatrix *matrix,
                                NALU_HYPRE_BigInt first_col_diag, NALU_HYPRE_BigInt last_col_diag );
hypre_CSRMatrix *hypre_MergeDiagAndOffd ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_MergeDiagAndOffdDevice ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll ( hypre_ParCSRMatrix *par_matrix );
NALU_HYPRE_Int hypre_ParCSRMatrixCopy ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                   NALU_HYPRE_Int copy_data );
NALU_HYPRE_Int hypre_FillResponseParToCSRMatrix ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                             NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             NALU_HYPRE_Int *response_message_size );
hypre_ParCSRMatrix *hypre_ParCSRMatrixUnion ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int copy_data );
#define hypre_ParCSRMatrixCompleteClone(A) hypre_ParCSRMatrixClone(A,0)
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone_v2 ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int copy_data,
                                                 NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int hypre_ParCSRMatrixMigrate(hypre_ParCSRMatrix *A, NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int hypre_ParCSRMatrixSetConstantValues( hypre_ParCSRMatrix *A, NALU_HYPRE_Complex value );
void hypre_ParCSRMatrixCopyColMapOffdToDevice(hypre_ParCSRMatrix *A);

/* par_csr_matvec.c */
// y = alpha*A*x + beta*b
NALU_HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlace ( NALU_HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                               hypre_ParVector *x, NALU_HYPRE_Complex beta,
                                               hypre_ParVector *b, hypre_ParVector *y );
NALU_HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlaceDevice ( NALU_HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                                     hypre_ParVector *x, NALU_HYPRE_Complex beta,
                                                     hypre_ParVector *b, hypre_ParVector *y );
// y = alpha*A*x + beta*y
NALU_HYPRE_Int hypre_ParCSRMatrixMatvec ( NALU_HYPRE_Complex alpha, hypre_ParCSRMatrix *A, hypre_ParVector *x,
                                     NALU_HYPRE_Complex beta, hypre_ParVector *y );
NALU_HYPRE_Int hypre_ParCSRMatrixMatvecT ( NALU_HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                      hypre_ParVector *x, NALU_HYPRE_Complex beta, hypre_ParVector *y );
NALU_HYPRE_Int hypre_ParCSRMatrixMatvecTDevice ( NALU_HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                            hypre_ParVector *x, NALU_HYPRE_Complex beta,
                                            hypre_ParVector *y );
NALU_HYPRE_Int hypre_ParCSRMatrixMatvecT_unpack( hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int num_cols,
                                            NALU_HYPRE_Complex *recv_data, NALU_HYPRE_Complex *local_data );
NALU_HYPRE_Int hypre_ParCSRMatrixMatvec_FF ( NALU_HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                        hypre_ParVector *x, NALU_HYPRE_Complex beta, hypre_ParVector *y,
                                        NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int fpt );

/* par_csr_triplemat.c */
NALU_HYPRE_Int hypre_ParCSRTMatMatPartialAddDevice( hypre_ParCSRCommPkg *comm_pkg_A,
                                               NALU_HYPRE_Int num_cols_A, NALU_HYPRE_Int num_cols_B, NALU_HYPRE_BigInt first_col_diag_B,
                                               NALU_HYPRE_BigInt last_col_diag_B, NALU_HYPRE_Int num_cols_offd_B, NALU_HYPRE_BigInt *col_map_offd_B,
                                               NALU_HYPRE_Int local_nnz_Cbar, hypre_CSRMatrix *Cbar, hypre_CSRMatrix *Cext,
                                               hypre_CSRMatrix **C_diag_ptr, hypre_CSRMatrix **C_offd_ptr, NALU_HYPRE_Int *num_cols_offd_C_ptr,
                                               NALU_HYPRE_BigInt **col_map_offd_C_ptr );
hypre_ParCSRMatrix *hypre_ParCSRMatMat( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMatHost( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMatDevice( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKTHost( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                               NALU_HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKTDevice( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                                 NALU_HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKT( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                           NALU_HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMat( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B);
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAPKT( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                             hypre_ParCSRMatrix  *P, NALU_HYPRE_Int keepTranspose );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAP( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                           hypre_ParCSRMatrix  *P );
hypre_ParCSRMatrix* hypre_ParCSRMatrixRAPKTDevice( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix *A,
                                                   hypre_ParCSRMatrix *P, NALU_HYPRE_Int keep_transpose );
hypre_ParCSRMatrix* hypre_ParCSRMatrixRAPKTHost( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, NALU_HYPRE_Int keep_transpose );

/* par_make_system.c */
NALU_HYPRE_ParCSR_System_Problem *NALU_HYPRE_Generate2DSystem ( NALU_HYPRE_ParCSRMatrix H_L1,
                                                      NALU_HYPRE_ParCSRMatrix H_L2, NALU_HYPRE_ParVector H_b1, NALU_HYPRE_ParVector H_b2, NALU_HYPRE_ParVector H_x1,
                                                      NALU_HYPRE_ParVector H_x2, NALU_HYPRE_Complex *M_vals );
NALU_HYPRE_Int NALU_HYPRE_Destroy2DSystem ( NALU_HYPRE_ParCSR_System_Problem *sys_prob );

/* par_vector.c */
hypre_ParVector *hypre_ParVectorCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_size,
                                         NALU_HYPRE_BigInt *partitioning_in );
hypre_ParVector *hypre_ParMultiVectorCreate ( MPI_Comm comm, NALU_HYPRE_BigInt global_size,
                                              NALU_HYPRE_BigInt *partitioning, NALU_HYPRE_Int num_vectors );
NALU_HYPRE_Int hypre_ParVectorDestroy ( hypre_ParVector *vector );
NALU_HYPRE_Int hypre_ParVectorInitialize ( hypre_ParVector *vector );
NALU_HYPRE_Int hypre_ParVectorInitialize_v2( hypre_ParVector *vector,
                                        NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int hypre_ParVectorSetDataOwner ( hypre_ParVector *vector, NALU_HYPRE_Int owns_data );
NALU_HYPRE_Int hypre_ParVectorSetLocalSize ( hypre_ParVector *vector, NALU_HYPRE_Int local_size );
NALU_HYPRE_Int hypre_ParVectorSetNumVectors ( hypre_ParVector *vector, NALU_HYPRE_Int num_vectors );
NALU_HYPRE_Int hypre_ParVectorSetComponent ( hypre_ParVector *vector, NALU_HYPRE_Int component );
hypre_ParVector *hypre_ParVectorRead ( MPI_Comm comm, const char *file_name );
NALU_HYPRE_Int hypre_ParVectorPrint ( hypre_ParVector *vector, const char *file_name );
NALU_HYPRE_Int hypre_ParVectorSetConstantValues ( hypre_ParVector *v, NALU_HYPRE_Complex value );
NALU_HYPRE_Int hypre_ParVectorSetZeros( hypre_ParVector *v );
NALU_HYPRE_Int hypre_ParVectorSetRandomValues ( hypre_ParVector *v, NALU_HYPRE_Int seed );
NALU_HYPRE_Int hypre_ParVectorCopy ( hypre_ParVector *x, hypre_ParVector *y );
hypre_ParVector *hypre_ParVectorCloneShallow ( hypre_ParVector *x );
hypre_ParVector *hypre_ParVectorCloneDeep_v2( hypre_ParVector *x,
                                              NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int hypre_ParVectorMigrate(hypre_ParVector *x, NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int hypre_ParVectorScale ( NALU_HYPRE_Complex alpha, hypre_ParVector *y );
NALU_HYPRE_Int hypre_ParVectorAxpy ( NALU_HYPRE_Complex alpha, hypre_ParVector *x, hypre_ParVector *y );
NALU_HYPRE_Int hypre_ParVectorMassAxpy ( NALU_HYPRE_Complex *alpha, hypre_ParVector **x, hypre_ParVector *y,
                                    NALU_HYPRE_Int k, NALU_HYPRE_Int unroll);
NALU_HYPRE_Real hypre_ParVectorInnerProd ( hypre_ParVector *x, hypre_ParVector *y );
NALU_HYPRE_Int hypre_ParVectorMassInnerProd ( hypre_ParVector *x, hypre_ParVector **y, NALU_HYPRE_Int k,
                                         NALU_HYPRE_Int unroll, NALU_HYPRE_Real *prod );
NALU_HYPRE_Int hypre_ParVectorMassDotpTwo ( hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector **z,
                                       NALU_HYPRE_Int k, NALU_HYPRE_Int unroll, NALU_HYPRE_Real *prod_x, NALU_HYPRE_Real *prod_y );
hypre_ParVector *hypre_VectorToParVector ( MPI_Comm comm, hypre_Vector *v,
                                           NALU_HYPRE_BigInt *vec_starts );
hypre_Vector *hypre_ParVectorToVectorAll ( hypre_ParVector *par_v );
NALU_HYPRE_Int hypre_ParVectorPrintIJ ( hypre_ParVector *vector, NALU_HYPRE_Int base_j,
                                   const char *filename );
NALU_HYPRE_Int hypre_ParVectorReadIJ ( MPI_Comm comm, const char *filename, NALU_HYPRE_Int *base_j_ptr,
                                  hypre_ParVector **vector_ptr );
NALU_HYPRE_Int hypre_FillResponseParToVectorAll ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                             NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Complex hypre_ParVectorLocalSumElts ( hypre_ParVector *vector );
NALU_HYPRE_Int hypre_ParVectorGetValues ( hypre_ParVector *vector, NALU_HYPRE_Int num_values,
                                     NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *values);
NALU_HYPRE_Int hypre_ParVectorGetValues2( hypre_ParVector *vector, NALU_HYPRE_Int num_values,
                                     NALU_HYPRE_BigInt *indices, NALU_HYPRE_BigInt base, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_ParVectorGetValuesHost(hypre_ParVector *vector, NALU_HYPRE_Int num_values,
                                       NALU_HYPRE_BigInt *indices, NALU_HYPRE_BigInt base, NALU_HYPRE_Complex *values);
NALU_HYPRE_Int hypre_ParVectorElmdivpy( hypre_ParVector *x, hypre_ParVector *b, hypre_ParVector *y );
NALU_HYPRE_Int hypre_ParVectorElmdivpyMarked( hypre_ParVector *x, hypre_ParVector *b,
                                         hypre_ParVector *y, NALU_HYPRE_Int *marker,
                                         NALU_HYPRE_Int marker_val );
/* par_vector_device.c */
NALU_HYPRE_Int hypre_ParVectorGetValuesDevice(hypre_ParVector *vector, NALU_HYPRE_Int num_values,
                                         NALU_HYPRE_BigInt *indices, NALU_HYPRE_BigInt base,
                                         NALU_HYPRE_Complex *values);
