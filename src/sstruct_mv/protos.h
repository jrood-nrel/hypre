/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* NALU_HYPRE_sstruct_graph.c */
NALU_HYPRE_Int NALU_HYPRE_SStructGraphCreate ( MPI_Comm comm, NALU_HYPRE_SStructGrid grid,
                                     NALU_HYPRE_SStructGraph *graph_ptr );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphDestroy ( NALU_HYPRE_SStructGraph graph );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphSetDomainGrid ( NALU_HYPRE_SStructGraph graph,
                                            NALU_HYPRE_SStructGrid domain_grid );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphSetStencil ( NALU_HYPRE_SStructGraph graph, NALU_HYPRE_Int part, NALU_HYPRE_Int var,
                                         NALU_HYPRE_SStructStencil stencil );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphSetFEM ( NALU_HYPRE_SStructGraph graph, NALU_HYPRE_Int part );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphSetFEMSparsity ( NALU_HYPRE_SStructGraph graph, NALU_HYPRE_Int part,
                                             NALU_HYPRE_Int nsparse, NALU_HYPRE_Int *sparsity );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphAddEntries ( NALU_HYPRE_SStructGraph graph, NALU_HYPRE_Int part, NALU_HYPRE_Int *index,
                                         NALU_HYPRE_Int var, NALU_HYPRE_Int to_part, NALU_HYPRE_Int *to_index, NALU_HYPRE_Int to_var );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphAssemble ( NALU_HYPRE_SStructGraph graph );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphSetObjectType ( NALU_HYPRE_SStructGraph graph, NALU_HYPRE_Int type );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphPrint ( FILE *file, NALU_HYPRE_SStructGraph graph );
NALU_HYPRE_Int NALU_HYPRE_SStructGraphRead ( FILE *file, NALU_HYPRE_SStructGrid grid,
                                   NALU_HYPRE_SStructStencil **stencils, NALU_HYPRE_SStructGraph *graph_ptr );

/* NALU_HYPRE_sstruct_grid.c */
NALU_HYPRE_Int NALU_HYPRE_SStructGridCreate ( MPI_Comm comm, NALU_HYPRE_Int ndim, NALU_HYPRE_Int nparts,
                                    NALU_HYPRE_SStructGrid *grid_ptr );
NALU_HYPRE_Int NALU_HYPRE_SStructGridDestroy ( NALU_HYPRE_SStructGrid grid );
NALU_HYPRE_Int NALU_HYPRE_SStructGridSetExtents ( NALU_HYPRE_SStructGrid grid, NALU_HYPRE_Int part, NALU_HYPRE_Int *ilower,
                                        NALU_HYPRE_Int *iupper );
NALU_HYPRE_Int NALU_HYPRE_SStructGridSetVariables ( NALU_HYPRE_SStructGrid grid, NALU_HYPRE_Int part, NALU_HYPRE_Int nvars,
                                          NALU_HYPRE_SStructVariable *vartypes );
NALU_HYPRE_Int NALU_HYPRE_SStructGridAddVariables ( NALU_HYPRE_SStructGrid grid, NALU_HYPRE_Int part, NALU_HYPRE_Int *index,
                                          NALU_HYPRE_Int nvars, NALU_HYPRE_SStructVariable *vartypes );
NALU_HYPRE_Int NALU_HYPRE_SStructGridSetFEMOrdering ( NALU_HYPRE_SStructGrid grid, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *ordering );
NALU_HYPRE_Int NALU_HYPRE_SStructGridSetNeighborPart ( NALU_HYPRE_SStructGrid grid, NALU_HYPRE_Int part,
                                             NALU_HYPRE_Int *ilower, NALU_HYPRE_Int *iupper, NALU_HYPRE_Int nbor_part, NALU_HYPRE_Int *nbor_ilower,
                                             NALU_HYPRE_Int *nbor_iupper, NALU_HYPRE_Int *index_map, NALU_HYPRE_Int *index_dir );
NALU_HYPRE_Int NALU_HYPRE_SStructGridSetSharedPart ( NALU_HYPRE_SStructGrid grid, NALU_HYPRE_Int part,
                                           NALU_HYPRE_Int *ilower, NALU_HYPRE_Int *iupper, NALU_HYPRE_Int *offset, NALU_HYPRE_Int shared_part,
                                           NALU_HYPRE_Int *shared_ilower, NALU_HYPRE_Int *shared_iupper, NALU_HYPRE_Int *shared_offset, NALU_HYPRE_Int *index_map,
                                           NALU_HYPRE_Int *index_dir );
NALU_HYPRE_Int NALU_HYPRE_SStructGridAddUnstructuredPart ( NALU_HYPRE_SStructGrid grid, NALU_HYPRE_Int ilower,
                                                 NALU_HYPRE_Int iupper );
NALU_HYPRE_Int NALU_HYPRE_SStructGridAssemble ( NALU_HYPRE_SStructGrid grid );
NALU_HYPRE_Int NALU_HYPRE_SStructGridSetPeriodic ( NALU_HYPRE_SStructGrid grid, NALU_HYPRE_Int part,
                                         NALU_HYPRE_Int *periodic );
NALU_HYPRE_Int NALU_HYPRE_SStructGridSetNumGhost ( NALU_HYPRE_SStructGrid grid, NALU_HYPRE_Int *num_ghost );

/* NALU_HYPRE_sstruct_matrix.c */
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixCreate ( MPI_Comm comm, NALU_HYPRE_SStructGraph graph,
                                      NALU_HYPRE_SStructMatrix *matrix_ptr );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixDestroy ( NALU_HYPRE_SStructMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixInitialize ( NALU_HYPRE_SStructMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixSetValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                         NALU_HYPRE_Int *index, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixAddToValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                           NALU_HYPRE_Int *index, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixAddFEMValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *index, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixGetValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                         NALU_HYPRE_Int *index, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixGetFEMValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *index, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixSetBoxValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *ilower, NALU_HYPRE_Int *iupper, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries,
                                            NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixAddToBoxValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                              NALU_HYPRE_Int *ilower, NALU_HYPRE_Int *iupper, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries,
                                              NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixGetBoxValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *ilower, NALU_HYPRE_Int *iupper, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries,
                                            NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixAssemble ( NALU_HYPRE_SStructMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixSetSymmetric ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int var, NALU_HYPRE_Int to_var, NALU_HYPRE_Int symmetric );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixSetNSSymmetric ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int symmetric );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixSetObjectType ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int type );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixGetObject ( NALU_HYPRE_SStructMatrix matrix, void **object );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixPrint ( const char *filename, NALU_HYPRE_SStructMatrix matrix,
                                     NALU_HYPRE_Int all );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixRead ( MPI_Comm comm, const char *filename,
                                    NALU_HYPRE_SStructMatrix *matrix_ptr );
NALU_HYPRE_Int NALU_HYPRE_SStructMatrixMatvec ( NALU_HYPRE_Complex alpha, NALU_HYPRE_SStructMatrix A,
                                      NALU_HYPRE_SStructVector x, NALU_HYPRE_Complex beta, NALU_HYPRE_SStructVector y );

/* NALU_HYPRE_sstruct_stencil.c */
NALU_HYPRE_Int NALU_HYPRE_SStructStencilCreate ( NALU_HYPRE_Int ndim, NALU_HYPRE_Int size,
                                       NALU_HYPRE_SStructStencil *stencil_ptr );
NALU_HYPRE_Int NALU_HYPRE_SStructStencilDestroy ( NALU_HYPRE_SStructStencil stencil );
NALU_HYPRE_Int NALU_HYPRE_SStructStencilSetEntry ( NALU_HYPRE_SStructStencil stencil, NALU_HYPRE_Int entry,
                                         NALU_HYPRE_Int *offset, NALU_HYPRE_Int var );
NALU_HYPRE_Int NALU_HYPRE_SStructStencilPrint ( FILE *file, NALU_HYPRE_SStructStencil stencil );
NALU_HYPRE_Int NALU_HYPRE_SStructStencilRead ( FILE *file, NALU_HYPRE_SStructStencil *stencil_ptr );


/* NALU_HYPRE_sstruct_vector.c */
NALU_HYPRE_Int NALU_HYPRE_SStructVectorCreate ( MPI_Comm comm, NALU_HYPRE_SStructGrid grid,
                                      NALU_HYPRE_SStructVector *vector_ptr );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorDestroy ( NALU_HYPRE_SStructVector vector );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorInitialize ( NALU_HYPRE_SStructVector vector );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorSetValues ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Int part,
                                         NALU_HYPRE_Int *index, NALU_HYPRE_Int var, NALU_HYPRE_Complex *value );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorAddToValues ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Int part,
                                           NALU_HYPRE_Int *index, NALU_HYPRE_Int var, NALU_HYPRE_Complex *value );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorAddFEMValues ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *index, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorGetValues ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Int part,
                                         NALU_HYPRE_Int *index, NALU_HYPRE_Int var, NALU_HYPRE_Complex *value );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorGetFEMValues ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *index, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorSetBoxValues ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *ilower, NALU_HYPRE_Int *iupper, NALU_HYPRE_Int var, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorAddToBoxValues ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Int part,
                                              NALU_HYPRE_Int *ilower, NALU_HYPRE_Int *iupper, NALU_HYPRE_Int var, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorGetBoxValues ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *ilower, NALU_HYPRE_Int *iupper, NALU_HYPRE_Int var, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorAssemble ( NALU_HYPRE_SStructVector vector );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorGather ( NALU_HYPRE_SStructVector vector );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorSetConstantValues ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Complex value );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorSetObjectType ( NALU_HYPRE_SStructVector vector, NALU_HYPRE_Int type );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorGetObject ( NALU_HYPRE_SStructVector vector, void **object );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorPrint ( const char *filename, NALU_HYPRE_SStructVector vector,
                                     NALU_HYPRE_Int all );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorRead ( MPI_Comm comm, const char *filename,
                                    NALU_HYPRE_SStructVector *vector );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorCopy ( NALU_HYPRE_SStructVector x, NALU_HYPRE_SStructVector y );
NALU_HYPRE_Int NALU_HYPRE_SStructVectorScale ( NALU_HYPRE_Complex alpha, NALU_HYPRE_SStructVector y );
NALU_HYPRE_Int NALU_HYPRE_SStructInnerProd ( NALU_HYPRE_SStructVector x, NALU_HYPRE_SStructVector y,
                                   NALU_HYPRE_Real *result );
NALU_HYPRE_Int NALU_HYPRE_SStructAxpy ( NALU_HYPRE_Complex alpha, NALU_HYPRE_SStructVector x, NALU_HYPRE_SStructVector y );

/* sstruct_axpy.c */
NALU_HYPRE_Int hypre_SStructPAxpy ( NALU_HYPRE_Complex alpha, hypre_SStructPVector *px,
                               hypre_SStructPVector *py );
NALU_HYPRE_Int hypre_SStructAxpy ( NALU_HYPRE_Complex alpha, hypre_SStructVector *x, hypre_SStructVector *y );

/* sstruct_copy.c */
NALU_HYPRE_Int hypre_SStructPCopy ( hypre_SStructPVector *px, hypre_SStructPVector *py );
NALU_HYPRE_Int hypre_SStructPartialPCopy ( hypre_SStructPVector *px, hypre_SStructPVector *py,
                                      hypre_BoxArrayArray **array_boxes );
NALU_HYPRE_Int hypre_SStructCopy ( hypre_SStructVector *x, hypre_SStructVector *y );

/* sstruct_graph.c */
NALU_HYPRE_Int hypre_SStructGraphRef ( hypre_SStructGraph *graph, hypre_SStructGraph **graph_ref );
NALU_HYPRE_Int hypre_SStructGraphGetUVEntryRank( hypre_SStructGraph *graph, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int var, hypre_Index index, NALU_HYPRE_BigInt *rank );
NALU_HYPRE_Int hypre_SStructGraphFindBoxEndpt ( hypre_SStructGraph *graph, NALU_HYPRE_Int part, NALU_HYPRE_Int var,
                                           NALU_HYPRE_Int proc, NALU_HYPRE_Int endpt, NALU_HYPRE_Int boxi );
NALU_HYPRE_Int hypre_SStructGraphFindSGridEndpts ( hypre_SStructGraph *graph, NALU_HYPRE_Int part,
                                              NALU_HYPRE_Int var, NALU_HYPRE_Int proc, NALU_HYPRE_Int endpt, NALU_HYPRE_Int *endpts );

/* sstruct_grid.c */
NALU_HYPRE_Int hypre_SStructVariableGetOffset ( NALU_HYPRE_SStructVariable vartype, NALU_HYPRE_Int ndim,
                                           hypre_Index varoffset );
NALU_HYPRE_Int hypre_SStructPGridCreate ( MPI_Comm comm, NALU_HYPRE_Int ndim,
                                     hypre_SStructPGrid **pgrid_ptr );
NALU_HYPRE_Int hypre_SStructPGridDestroy ( hypre_SStructPGrid *pgrid );
NALU_HYPRE_Int hypre_SStructPGridSetExtents ( hypre_SStructPGrid *pgrid, hypre_Index ilower,
                                         hypre_Index iupper );
NALU_HYPRE_Int hypre_SStructPGridSetCellSGrid ( hypre_SStructPGrid *pgrid,
                                           hypre_StructGrid *cell_sgrid );
NALU_HYPRE_Int hypre_SStructPGridSetVariables ( hypre_SStructPGrid *pgrid, NALU_HYPRE_Int nvars,
                                           NALU_HYPRE_SStructVariable *vartypes );
NALU_HYPRE_Int hypre_SStructPGridSetPNeighbor ( hypre_SStructPGrid *pgrid, hypre_Box *pneighbor_box,
                                           hypre_Index pnbor_offset );
NALU_HYPRE_Int hypre_SStructPGridAssemble ( hypre_SStructPGrid *pgrid );
NALU_HYPRE_Int hypre_SStructPGridGetMaxBoxSize ( hypre_SStructPGrid *pgrid );
NALU_HYPRE_Int hypre_SStructGridRef ( hypre_SStructGrid *grid, hypre_SStructGrid **grid_ref );
NALU_HYPRE_Int hypre_SStructGridAssembleBoxManagers ( hypre_SStructGrid *grid );
NALU_HYPRE_Int hypre_SStructGridAssembleNborBoxManagers ( hypre_SStructGrid *grid );
NALU_HYPRE_Int hypre_SStructGridCreateCommInfo ( hypre_SStructGrid *grid );
NALU_HYPRE_Int hypre_SStructGridFindBoxManEntry ( hypre_SStructGrid *grid, NALU_HYPRE_Int part,
                                             hypre_Index index, NALU_HYPRE_Int var, hypre_BoxManEntry **entry_ptr );
NALU_HYPRE_Int hypre_SStructGridFindNborBoxManEntry ( hypre_SStructGrid *grid, NALU_HYPRE_Int part,
                                                 hypre_Index index, NALU_HYPRE_Int var, hypre_BoxManEntry **entry_ptr );
NALU_HYPRE_Int hypre_SStructGridBoxProcFindBoxManEntry ( hypre_SStructGrid *grid, NALU_HYPRE_Int part,
                                                    NALU_HYPRE_Int var, NALU_HYPRE_Int box, NALU_HYPRE_Int proc, hypre_BoxManEntry **entry_ptr );
NALU_HYPRE_Int hypre_SStructBoxManEntryGetCSRstrides ( hypre_BoxManEntry *entry, hypre_Index strides );
NALU_HYPRE_Int hypre_SStructBoxManEntryGetGhstrides ( hypre_BoxManEntry *entry, hypre_Index strides );
NALU_HYPRE_Int hypre_SStructBoxManEntryGetGlobalCSRank ( hypre_BoxManEntry *entry, hypre_Index index,
                                                    NALU_HYPRE_BigInt *rank_ptr );
NALU_HYPRE_Int hypre_SStructBoxManEntryGetGlobalGhrank ( hypre_BoxManEntry *entry, hypre_Index index,
                                                    NALU_HYPRE_BigInt *rank_ptr );
NALU_HYPRE_Int hypre_SStructBoxManEntryGetProcess ( hypre_BoxManEntry *entry, NALU_HYPRE_Int *proc_ptr );
NALU_HYPRE_Int hypre_SStructBoxManEntryGetBoxnum ( hypre_BoxManEntry *entry, NALU_HYPRE_Int *id_ptr );
NALU_HYPRE_Int hypre_SStructBoxManEntryGetPart ( hypre_BoxManEntry *entry, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *part_ptr );
NALU_HYPRE_Int hypre_SStructIndexToNborIndex( hypre_Index index, hypre_Index root, hypre_Index nbor_root,
                                         hypre_Index coord, hypre_Index dir, NALU_HYPRE_Int ndim, hypre_Index nbor_index );
NALU_HYPRE_Int hypre_SStructBoxToNborBox ( hypre_Box *box, hypre_Index root, hypre_Index nbor_root,
                                      hypre_Index coord, hypre_Index dir );
NALU_HYPRE_Int hypre_SStructNborIndexToIndex( hypre_Index nbor_index, hypre_Index root,
                                         hypre_Index nbor_root, hypre_Index coord, hypre_Index dir, NALU_HYPRE_Int ndim, hypre_Index index );
NALU_HYPRE_Int hypre_SStructNborBoxToBox ( hypre_Box *nbor_box, hypre_Index root, hypre_Index nbor_root,
                                      hypre_Index coord, hypre_Index dir );
NALU_HYPRE_Int hypre_SStructVarToNborVar ( hypre_SStructGrid *grid, NALU_HYPRE_Int part, NALU_HYPRE_Int var,
                                      NALU_HYPRE_Int *coord, NALU_HYPRE_Int *nbor_var_ptr );
NALU_HYPRE_Int hypre_SStructGridSetNumGhost ( hypre_SStructGrid *grid, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int hypre_SStructBoxManEntryGetGlobalRank ( hypre_BoxManEntry *entry, hypre_Index index,
                                                  NALU_HYPRE_BigInt *rank_ptr, NALU_HYPRE_Int type );
NALU_HYPRE_Int hypre_SStructBoxManEntryGetStrides ( hypre_BoxManEntry *entry, hypre_Index strides,
                                               NALU_HYPRE_Int type );
NALU_HYPRE_Int hypre_SStructBoxNumMap ( hypre_SStructGrid *grid, NALU_HYPRE_Int part, NALU_HYPRE_Int boxnum,
                                   NALU_HYPRE_Int **num_varboxes_ptr, NALU_HYPRE_Int ***map_ptr );
NALU_HYPRE_Int hypre_SStructCellGridBoxNumMap ( hypre_SStructGrid *grid, NALU_HYPRE_Int part,
                                           NALU_HYPRE_Int ***num_varboxes_ptr, NALU_HYPRE_Int ****map_ptr );
NALU_HYPRE_Int hypre_SStructCellBoxToVarBox ( hypre_Box *box, hypre_Index offset, hypre_Index varoffset,
                                         NALU_HYPRE_Int *valid );
NALU_HYPRE_Int hypre_SStructGridIntersect ( hypre_SStructGrid *grid, NALU_HYPRE_Int part, NALU_HYPRE_Int var,
                                       hypre_Box *box, NALU_HYPRE_Int action, hypre_BoxManEntry ***entries_ptr, NALU_HYPRE_Int *nentries_ptr );
NALU_HYPRE_Int hypre_SStructGridGetMaxBoxSize ( hypre_SStructGrid *grid );
NALU_HYPRE_Int hypre_SStructGridPrint ( FILE *file, hypre_SStructGrid *grid );
NALU_HYPRE_Int hypre_SStructGridRead ( MPI_Comm comm, FILE *file, hypre_SStructGrid **grid_ptr );

/* sstruct_innerprod.c */
NALU_HYPRE_Int hypre_SStructPInnerProd ( hypre_SStructPVector *px, hypre_SStructPVector *py,
                                    NALU_HYPRE_Real *presult_ptr );
NALU_HYPRE_Int hypre_SStructInnerProd ( hypre_SStructVector *x, hypre_SStructVector *y,
                                   NALU_HYPRE_Real *result_ptr );

/* sstruct_matrix.c */
NALU_HYPRE_Int hypre_SStructPMatrixRef ( hypre_SStructPMatrix *matrix,
                                    hypre_SStructPMatrix **matrix_ref );
NALU_HYPRE_Int hypre_SStructPMatrixCreate ( MPI_Comm comm, hypre_SStructPGrid *pgrid,
                                       hypre_SStructStencil **stencils, hypre_SStructPMatrix **pmatrix_ptr );
NALU_HYPRE_Int hypre_SStructPMatrixDestroy ( hypre_SStructPMatrix *pmatrix );
NALU_HYPRE_Int hypre_SStructPMatrixInitialize ( hypre_SStructPMatrix *pmatrix );
NALU_HYPRE_Int hypre_SStructPMatrixSetValues ( hypre_SStructPMatrix *pmatrix, hypre_Index index,
                                          NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_SStructPMatrixSetBoxValues( hypre_SStructPMatrix *pmatrix, hypre_Box *set_box,
                                            NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, hypre_Box *value_box, NALU_HYPRE_Complex *values,
                                            NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_SStructPMatrixAccumulate ( hypre_SStructPMatrix *pmatrix );
NALU_HYPRE_Int hypre_SStructPMatrixAssemble ( hypre_SStructPMatrix *pmatrix );
NALU_HYPRE_Int hypre_SStructPMatrixSetSymmetric ( hypre_SStructPMatrix *pmatrix, NALU_HYPRE_Int var,
                                             NALU_HYPRE_Int to_var, NALU_HYPRE_Int symmetric );
NALU_HYPRE_Int hypre_SStructPMatrixPrint ( const char *filename, hypre_SStructPMatrix *pmatrix,
                                      NALU_HYPRE_Int all );
NALU_HYPRE_Int hypre_SStructUMatrixInitialize ( hypre_SStructMatrix *matrix );
NALU_HYPRE_Int hypre_SStructUMatrixSetValues ( hypre_SStructMatrix *matrix, NALU_HYPRE_Int part,
                                          hypre_Index index, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Complex *values,
                                          NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_SStructUMatrixSetBoxValues( hypre_SStructMatrix *matrix, NALU_HYPRE_Int part,
                                            hypre_Box *set_box, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, hypre_Box *value_box,
                                            NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_SStructUMatrixAssemble ( hypre_SStructMatrix *matrix );
NALU_HYPRE_Int hypre_SStructMatrixRef ( hypre_SStructMatrix *matrix, hypre_SStructMatrix **matrix_ref );
NALU_HYPRE_Int hypre_SStructMatrixSplitEntries ( hypre_SStructMatrix *matrix, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Int *nSentries_ptr,
                                            NALU_HYPRE_Int **Sentries_ptr, NALU_HYPRE_Int *nUentries_ptr, NALU_HYPRE_Int **Uentries_ptr );
NALU_HYPRE_Int hypre_SStructMatrixSetValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                         NALU_HYPRE_Int *index, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Complex *values,
                                         NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_SStructMatrixSetBoxValues( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                           hypre_Box *set_box, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, hypre_Box *value_box,
                                           NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_SStructMatrixSetInterPartValues( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                                 hypre_Box *set_box, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, hypre_Box *value_box,
                                                 NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_MemoryLocation hypre_SStructMatrixMemoryLocation(hypre_SStructMatrix *matrix);

/* sstruct_matvec.c */
NALU_HYPRE_Int hypre_SStructPMatvecCreate ( void **pmatvec_vdata_ptr );
NALU_HYPRE_Int hypre_SStructPMatvecSetup ( void *pmatvec_vdata, hypre_SStructPMatrix *pA,
                                      hypre_SStructPVector *px );
NALU_HYPRE_Int hypre_SStructPMatvecCompute ( void *pmatvec_vdata, NALU_HYPRE_Complex alpha,
                                        hypre_SStructPMatrix *pA, hypre_SStructPVector *px, NALU_HYPRE_Complex beta, hypre_SStructPVector *py );
NALU_HYPRE_Int hypre_SStructPMatvecDestroy ( void *pmatvec_vdata );
NALU_HYPRE_Int hypre_SStructPMatvec ( NALU_HYPRE_Complex alpha, hypre_SStructPMatrix *pA,
                                 hypre_SStructPVector *px, NALU_HYPRE_Complex beta, hypre_SStructPVector *py );
NALU_HYPRE_Int hypre_SStructMatvecCreate ( void **matvec_vdata_ptr );
NALU_HYPRE_Int hypre_SStructMatvecSetup ( void *matvec_vdata, hypre_SStructMatrix *A,
                                     hypre_SStructVector *x );
NALU_HYPRE_Int hypre_SStructMatvecCompute ( void *matvec_vdata, NALU_HYPRE_Complex alpha,
                                       hypre_SStructMatrix *A, hypre_SStructVector *x, NALU_HYPRE_Complex beta, hypre_SStructVector *y );
NALU_HYPRE_Int hypre_SStructMatvecDestroy ( void *matvec_vdata );
NALU_HYPRE_Int hypre_SStructMatvec ( NALU_HYPRE_Complex alpha, hypre_SStructMatrix *A, hypre_SStructVector *x,
                                NALU_HYPRE_Complex beta, hypre_SStructVector *y );

/* sstruct_scale.c */
NALU_HYPRE_Int hypre_SStructPScale ( NALU_HYPRE_Complex alpha, hypre_SStructPVector *py );
NALU_HYPRE_Int hypre_SStructScale ( NALU_HYPRE_Complex alpha, hypre_SStructVector *y );

/* sstruct_stencil.c */
NALU_HYPRE_Int hypre_SStructStencilRef ( hypre_SStructStencil *stencil,
                                    hypre_SStructStencil **stencil_ref );

/* sstruct_vector.c */
NALU_HYPRE_Int hypre_SStructPVectorRef ( hypre_SStructPVector *vector,
                                    hypre_SStructPVector **vector_ref );
NALU_HYPRE_Int hypre_SStructPVectorCreate ( MPI_Comm comm, hypre_SStructPGrid *pgrid,
                                       hypre_SStructPVector **pvector_ptr );
NALU_HYPRE_Int hypre_SStructPVectorDestroy ( hypre_SStructPVector *pvector );
NALU_HYPRE_Int hypre_SStructPVectorInitialize ( hypre_SStructPVector *pvector );
NALU_HYPRE_Int hypre_SStructPVectorSetValues ( hypre_SStructPVector *pvector, hypre_Index index,
                                          NALU_HYPRE_Int var, NALU_HYPRE_Complex *value, NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_SStructPVectorSetBoxValues( hypre_SStructPVector *pvector, hypre_Box *set_box,
                                            NALU_HYPRE_Int var, hypre_Box *value_box, NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_SStructPVectorAccumulate ( hypre_SStructPVector *pvector );
NALU_HYPRE_Int hypre_SStructPVectorAssemble ( hypre_SStructPVector *pvector );
NALU_HYPRE_Int hypre_SStructPVectorGather ( hypre_SStructPVector *pvector );
NALU_HYPRE_Int hypre_SStructPVectorGetValues ( hypre_SStructPVector *pvector, hypre_Index index,
                                          NALU_HYPRE_Int var, NALU_HYPRE_Complex *value );
NALU_HYPRE_Int hypre_SStructPVectorGetBoxValues( hypre_SStructPVector *pvector, hypre_Box *set_box,
                                            NALU_HYPRE_Int var, hypre_Box *value_box, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int hypre_SStructPVectorSetConstantValues ( hypre_SStructPVector *pvector,
                                                  NALU_HYPRE_Complex value );
NALU_HYPRE_Int hypre_SStructPVectorPrint ( const char *filename, hypre_SStructPVector *pvector,
                                      NALU_HYPRE_Int all );
NALU_HYPRE_Int hypre_SStructVectorRef ( hypre_SStructVector *vector, hypre_SStructVector **vector_ref );
NALU_HYPRE_Int hypre_SStructVectorSetConstantValues ( hypre_SStructVector *vector, NALU_HYPRE_Complex value );
NALU_HYPRE_Int hypre_SStructVectorConvert ( hypre_SStructVector *vector,
                                       hypre_ParVector **parvector_ptr );
NALU_HYPRE_Int hypre_SStructVectorParConvert ( hypre_SStructVector *vector,
                                          hypre_ParVector **parvector_ptr );
NALU_HYPRE_Int hypre_SStructVectorRestore ( hypre_SStructVector *vector, hypre_ParVector *parvector );
NALU_HYPRE_Int hypre_SStructVectorParRestore ( hypre_SStructVector *vector, hypre_ParVector *parvector );
NALU_HYPRE_Int hypre_SStructPVectorInitializeShell ( hypre_SStructPVector *pvector );
NALU_HYPRE_Int hypre_SStructVectorInitializeShell ( hypre_SStructVector *vector );
NALU_HYPRE_Int hypre_SStructVectorClearGhostValues ( hypre_SStructVector *vector );
NALU_HYPRE_MemoryLocation hypre_SStructVectorMemoryLocation(hypre_SStructVector *vector);

