/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
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
NALU_HYPRE_Int nalu_hypre_SStructPAxpy ( NALU_HYPRE_Complex alpha, nalu_hypre_SStructPVector *px,
                               nalu_hypre_SStructPVector *py );
NALU_HYPRE_Int nalu_hypre_SStructAxpy ( NALU_HYPRE_Complex alpha, nalu_hypre_SStructVector *x, nalu_hypre_SStructVector *y );

/* sstruct_copy.c */
NALU_HYPRE_Int nalu_hypre_SStructPCopy ( nalu_hypre_SStructPVector *px, nalu_hypre_SStructPVector *py );
NALU_HYPRE_Int nalu_hypre_SStructPartialPCopy ( nalu_hypre_SStructPVector *px, nalu_hypre_SStructPVector *py,
                                      nalu_hypre_BoxArrayArray **array_boxes );
NALU_HYPRE_Int nalu_hypre_SStructCopy ( nalu_hypre_SStructVector *x, nalu_hypre_SStructVector *y );

/* sstruct_graph.c */
NALU_HYPRE_Int nalu_hypre_SStructGraphRef ( nalu_hypre_SStructGraph *graph, nalu_hypre_SStructGraph **graph_ref );
NALU_HYPRE_Int nalu_hypre_SStructGraphGetUVEntryRank( nalu_hypre_SStructGraph *graph, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int var, nalu_hypre_Index index, NALU_HYPRE_BigInt *rank );
NALU_HYPRE_Int nalu_hypre_SStructGraphFindBoxEndpt ( nalu_hypre_SStructGraph *graph, NALU_HYPRE_Int part, NALU_HYPRE_Int var,
                                           NALU_HYPRE_Int proc, NALU_HYPRE_Int endpt, NALU_HYPRE_Int boxi );
NALU_HYPRE_Int nalu_hypre_SStructGraphFindSGridEndpts ( nalu_hypre_SStructGraph *graph, NALU_HYPRE_Int part,
                                              NALU_HYPRE_Int var, NALU_HYPRE_Int proc, NALU_HYPRE_Int endpt, NALU_HYPRE_Int *endpts );

/* sstruct_grid.c */
NALU_HYPRE_Int nalu_hypre_SStructVariableGetOffset ( NALU_HYPRE_SStructVariable vartype, NALU_HYPRE_Int ndim,
                                           nalu_hypre_Index varoffset );
NALU_HYPRE_Int nalu_hypre_SStructPGridCreate ( MPI_Comm comm, NALU_HYPRE_Int ndim,
                                     nalu_hypre_SStructPGrid **pgrid_ptr );
NALU_HYPRE_Int nalu_hypre_SStructPGridDestroy ( nalu_hypre_SStructPGrid *pgrid );
NALU_HYPRE_Int nalu_hypre_SStructPGridSetExtents ( nalu_hypre_SStructPGrid *pgrid, nalu_hypre_Index ilower,
                                         nalu_hypre_Index iupper );
NALU_HYPRE_Int nalu_hypre_SStructPGridSetCellSGrid ( nalu_hypre_SStructPGrid *pgrid,
                                           nalu_hypre_StructGrid *cell_sgrid );
NALU_HYPRE_Int nalu_hypre_SStructPGridSetVariables ( nalu_hypre_SStructPGrid *pgrid, NALU_HYPRE_Int nvars,
                                           NALU_HYPRE_SStructVariable *vartypes );
NALU_HYPRE_Int nalu_hypre_SStructPGridSetPNeighbor ( nalu_hypre_SStructPGrid *pgrid, nalu_hypre_Box *pneighbor_box,
                                           nalu_hypre_Index pnbor_offset );
NALU_HYPRE_Int nalu_hypre_SStructPGridAssemble ( nalu_hypre_SStructPGrid *pgrid );
NALU_HYPRE_Int nalu_hypre_SStructPGridGetMaxBoxSize ( nalu_hypre_SStructPGrid *pgrid );
NALU_HYPRE_Int nalu_hypre_SStructGridRef ( nalu_hypre_SStructGrid *grid, nalu_hypre_SStructGrid **grid_ref );
NALU_HYPRE_Int nalu_hypre_SStructGridAssembleBoxManagers ( nalu_hypre_SStructGrid *grid );
NALU_HYPRE_Int nalu_hypre_SStructGridAssembleNborBoxManagers ( nalu_hypre_SStructGrid *grid );
NALU_HYPRE_Int nalu_hypre_SStructGridCreateCommInfo ( nalu_hypre_SStructGrid *grid );
NALU_HYPRE_Int nalu_hypre_SStructGridFindBoxManEntry ( nalu_hypre_SStructGrid *grid, NALU_HYPRE_Int part,
                                             nalu_hypre_Index index, NALU_HYPRE_Int var, nalu_hypre_BoxManEntry **entry_ptr );
NALU_HYPRE_Int nalu_hypre_SStructGridFindNborBoxManEntry ( nalu_hypre_SStructGrid *grid, NALU_HYPRE_Int part,
                                                 nalu_hypre_Index index, NALU_HYPRE_Int var, nalu_hypre_BoxManEntry **entry_ptr );
NALU_HYPRE_Int nalu_hypre_SStructGridBoxProcFindBoxManEntry ( nalu_hypre_SStructGrid *grid, NALU_HYPRE_Int part,
                                                    NALU_HYPRE_Int var, NALU_HYPRE_Int box, NALU_HYPRE_Int proc, nalu_hypre_BoxManEntry **entry_ptr );
NALU_HYPRE_Int nalu_hypre_SStructBoxManEntryGetCSRstrides ( nalu_hypre_BoxManEntry *entry, nalu_hypre_Index strides );
NALU_HYPRE_Int nalu_hypre_SStructBoxManEntryGetGhstrides ( nalu_hypre_BoxManEntry *entry, nalu_hypre_Index strides );
NALU_HYPRE_Int nalu_hypre_SStructBoxManEntryGetGlobalCSRank ( nalu_hypre_BoxManEntry *entry, nalu_hypre_Index index,
                                                    NALU_HYPRE_BigInt *rank_ptr );
NALU_HYPRE_Int nalu_hypre_SStructBoxManEntryGetGlobalGhrank ( nalu_hypre_BoxManEntry *entry, nalu_hypre_Index index,
                                                    NALU_HYPRE_BigInt *rank_ptr );
NALU_HYPRE_Int nalu_hypre_SStructBoxManEntryGetProcess ( nalu_hypre_BoxManEntry *entry, NALU_HYPRE_Int *proc_ptr );
NALU_HYPRE_Int nalu_hypre_SStructBoxManEntryGetBoxnum ( nalu_hypre_BoxManEntry *entry, NALU_HYPRE_Int *id_ptr );
NALU_HYPRE_Int nalu_hypre_SStructBoxManEntryGetPart ( nalu_hypre_BoxManEntry *entry, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int *part_ptr );
NALU_HYPRE_Int nalu_hypre_SStructIndexToNborIndex( nalu_hypre_Index index, nalu_hypre_Index root, nalu_hypre_Index nbor_root,
                                         nalu_hypre_Index coord, nalu_hypre_Index dir, NALU_HYPRE_Int ndim, nalu_hypre_Index nbor_index );
NALU_HYPRE_Int nalu_hypre_SStructBoxToNborBox ( nalu_hypre_Box *box, nalu_hypre_Index root, nalu_hypre_Index nbor_root,
                                      nalu_hypre_Index coord, nalu_hypre_Index dir );
NALU_HYPRE_Int nalu_hypre_SStructNborIndexToIndex( nalu_hypre_Index nbor_index, nalu_hypre_Index root,
                                         nalu_hypre_Index nbor_root, nalu_hypre_Index coord, nalu_hypre_Index dir, NALU_HYPRE_Int ndim, nalu_hypre_Index index );
NALU_HYPRE_Int nalu_hypre_SStructNborBoxToBox ( nalu_hypre_Box *nbor_box, nalu_hypre_Index root, nalu_hypre_Index nbor_root,
                                      nalu_hypre_Index coord, nalu_hypre_Index dir );
NALU_HYPRE_Int nalu_hypre_SStructVarToNborVar ( nalu_hypre_SStructGrid *grid, NALU_HYPRE_Int part, NALU_HYPRE_Int var,
                                      NALU_HYPRE_Int *coord, NALU_HYPRE_Int *nbor_var_ptr );
NALU_HYPRE_Int nalu_hypre_SStructGridSetNumGhost ( nalu_hypre_SStructGrid *grid, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int nalu_hypre_SStructBoxManEntryGetGlobalRank ( nalu_hypre_BoxManEntry *entry, nalu_hypre_Index index,
                                                  NALU_HYPRE_BigInt *rank_ptr, NALU_HYPRE_Int type );
NALU_HYPRE_Int nalu_hypre_SStructBoxManEntryGetStrides ( nalu_hypre_BoxManEntry *entry, nalu_hypre_Index strides,
                                               NALU_HYPRE_Int type );
NALU_HYPRE_Int nalu_hypre_SStructBoxNumMap ( nalu_hypre_SStructGrid *grid, NALU_HYPRE_Int part, NALU_HYPRE_Int boxnum,
                                   NALU_HYPRE_Int **num_varboxes_ptr, NALU_HYPRE_Int ***map_ptr );
NALU_HYPRE_Int nalu_hypre_SStructCellGridBoxNumMap ( nalu_hypre_SStructGrid *grid, NALU_HYPRE_Int part,
                                           NALU_HYPRE_Int ***num_varboxes_ptr, NALU_HYPRE_Int ****map_ptr );
NALU_HYPRE_Int nalu_hypre_SStructCellBoxToVarBox ( nalu_hypre_Box *box, nalu_hypre_Index offset, nalu_hypre_Index varoffset,
                                         NALU_HYPRE_Int *valid );
NALU_HYPRE_Int nalu_hypre_SStructGridIntersect ( nalu_hypre_SStructGrid *grid, NALU_HYPRE_Int part, NALU_HYPRE_Int var,
                                       nalu_hypre_Box *box, NALU_HYPRE_Int action, nalu_hypre_BoxManEntry ***entries_ptr, NALU_HYPRE_Int *nentries_ptr );
NALU_HYPRE_Int nalu_hypre_SStructGridGetMaxBoxSize ( nalu_hypre_SStructGrid *grid );
NALU_HYPRE_Int nalu_hypre_SStructGridPrint ( FILE *file, nalu_hypre_SStructGrid *grid );
NALU_HYPRE_Int nalu_hypre_SStructGridRead ( MPI_Comm comm, FILE *file, nalu_hypre_SStructGrid **grid_ptr );

/* sstruct_innerprod.c */
NALU_HYPRE_Int nalu_hypre_SStructPInnerProd ( nalu_hypre_SStructPVector *px, nalu_hypre_SStructPVector *py,
                                    NALU_HYPRE_Real *presult_ptr );
NALU_HYPRE_Int nalu_hypre_SStructInnerProd ( nalu_hypre_SStructVector *x, nalu_hypre_SStructVector *y,
                                   NALU_HYPRE_Real *result_ptr );

/* sstruct_matrix.c */
NALU_HYPRE_Int nalu_hypre_SStructPMatrixRef ( nalu_hypre_SStructPMatrix *matrix,
                                    nalu_hypre_SStructPMatrix **matrix_ref );
NALU_HYPRE_Int nalu_hypre_SStructPMatrixCreate ( MPI_Comm comm, nalu_hypre_SStructPGrid *pgrid,
                                       nalu_hypre_SStructStencil **stencils, nalu_hypre_SStructPMatrix **pmatrix_ptr );
NALU_HYPRE_Int nalu_hypre_SStructPMatrixDestroy ( nalu_hypre_SStructPMatrix *pmatrix );
NALU_HYPRE_Int nalu_hypre_SStructPMatrixInitialize ( nalu_hypre_SStructPMatrix *pmatrix );
NALU_HYPRE_Int nalu_hypre_SStructPMatrixSetValues ( nalu_hypre_SStructPMatrix *pmatrix, nalu_hypre_Index index,
                                          NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_SStructPMatrixSetBoxValues( nalu_hypre_SStructPMatrix *pmatrix, nalu_hypre_Box *set_box,
                                            NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, nalu_hypre_Box *value_box, NALU_HYPRE_Complex *values,
                                            NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_SStructPMatrixAccumulate ( nalu_hypre_SStructPMatrix *pmatrix );
NALU_HYPRE_Int nalu_hypre_SStructPMatrixAssemble ( nalu_hypre_SStructPMatrix *pmatrix );
NALU_HYPRE_Int nalu_hypre_SStructPMatrixSetSymmetric ( nalu_hypre_SStructPMatrix *pmatrix, NALU_HYPRE_Int var,
                                             NALU_HYPRE_Int to_var, NALU_HYPRE_Int symmetric );
NALU_HYPRE_Int nalu_hypre_SStructPMatrixPrint ( const char *filename, nalu_hypre_SStructPMatrix *pmatrix,
                                      NALU_HYPRE_Int all );
NALU_HYPRE_Int nalu_hypre_SStructUMatrixInitialize ( nalu_hypre_SStructMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_SStructUMatrixSetValues ( nalu_hypre_SStructMatrix *matrix, NALU_HYPRE_Int part,
                                          nalu_hypre_Index index, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Complex *values,
                                          NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_SStructUMatrixSetBoxValues( nalu_hypre_SStructMatrix *matrix, NALU_HYPRE_Int part,
                                            nalu_hypre_Box *set_box, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, nalu_hypre_Box *value_box,
                                            NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_SStructUMatrixAssemble ( nalu_hypre_SStructMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_SStructMatrixRef ( nalu_hypre_SStructMatrix *matrix, nalu_hypre_SStructMatrix **matrix_ref );
NALU_HYPRE_Int nalu_hypre_SStructMatrixSplitEntries ( nalu_hypre_SStructMatrix *matrix, NALU_HYPRE_Int part,
                                            NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Int *nSentries_ptr,
                                            NALU_HYPRE_Int **Sentries_ptr, NALU_HYPRE_Int *nUentries_ptr, NALU_HYPRE_Int **Uentries_ptr );
NALU_HYPRE_Int nalu_hypre_SStructMatrixSetValues ( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                         NALU_HYPRE_Int *index, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, NALU_HYPRE_Complex *values,
                                         NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_SStructMatrixSetBoxValues( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                           nalu_hypre_Box *set_box, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, nalu_hypre_Box *value_box,
                                           NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_SStructMatrixSetInterPartValues( NALU_HYPRE_SStructMatrix matrix, NALU_HYPRE_Int part,
                                                 nalu_hypre_Box *set_box, NALU_HYPRE_Int var, NALU_HYPRE_Int nentries, NALU_HYPRE_Int *entries, nalu_hypre_Box *value_box,
                                                 NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_MemoryLocation nalu_hypre_SStructMatrixMemoryLocation(nalu_hypre_SStructMatrix *matrix);

/* sstruct_matvec.c */
NALU_HYPRE_Int nalu_hypre_SStructPMatvecCreate ( void **pmatvec_vdata_ptr );
NALU_HYPRE_Int nalu_hypre_SStructPMatvecSetup ( void *pmatvec_vdata, nalu_hypre_SStructPMatrix *pA,
                                      nalu_hypre_SStructPVector *px );
NALU_HYPRE_Int nalu_hypre_SStructPMatvecCompute ( void *pmatvec_vdata, NALU_HYPRE_Complex alpha,
                                        nalu_hypre_SStructPMatrix *pA, nalu_hypre_SStructPVector *px, NALU_HYPRE_Complex beta, nalu_hypre_SStructPVector *py );
NALU_HYPRE_Int nalu_hypre_SStructPMatvecDestroy ( void *pmatvec_vdata );
NALU_HYPRE_Int nalu_hypre_SStructPMatvec ( NALU_HYPRE_Complex alpha, nalu_hypre_SStructPMatrix *pA,
                                 nalu_hypre_SStructPVector *px, NALU_HYPRE_Complex beta, nalu_hypre_SStructPVector *py );
NALU_HYPRE_Int nalu_hypre_SStructMatvecCreate ( void **matvec_vdata_ptr );
NALU_HYPRE_Int nalu_hypre_SStructMatvecSetup ( void *matvec_vdata, nalu_hypre_SStructMatrix *A,
                                     nalu_hypre_SStructVector *x );
NALU_HYPRE_Int nalu_hypre_SStructMatvecCompute ( void *matvec_vdata, NALU_HYPRE_Complex alpha,
                                       nalu_hypre_SStructMatrix *A, nalu_hypre_SStructVector *x, NALU_HYPRE_Complex beta, nalu_hypre_SStructVector *y );
NALU_HYPRE_Int nalu_hypre_SStructMatvecDestroy ( void *matvec_vdata );
NALU_HYPRE_Int nalu_hypre_SStructMatvec ( NALU_HYPRE_Complex alpha, nalu_hypre_SStructMatrix *A, nalu_hypre_SStructVector *x,
                                NALU_HYPRE_Complex beta, nalu_hypre_SStructVector *y );

/* sstruct_scale.c */
NALU_HYPRE_Int nalu_hypre_SStructPScale ( NALU_HYPRE_Complex alpha, nalu_hypre_SStructPVector *py );
NALU_HYPRE_Int nalu_hypre_SStructScale ( NALU_HYPRE_Complex alpha, nalu_hypre_SStructVector *y );

/* sstruct_stencil.c */
NALU_HYPRE_Int nalu_hypre_SStructStencilRef ( nalu_hypre_SStructStencil *stencil,
                                    nalu_hypre_SStructStencil **stencil_ref );

/* sstruct_vector.c */
NALU_HYPRE_Int nalu_hypre_SStructPVectorRef ( nalu_hypre_SStructPVector *vector,
                                    nalu_hypre_SStructPVector **vector_ref );
NALU_HYPRE_Int nalu_hypre_SStructPVectorCreate ( MPI_Comm comm, nalu_hypre_SStructPGrid *pgrid,
                                       nalu_hypre_SStructPVector **pvector_ptr );
NALU_HYPRE_Int nalu_hypre_SStructPVectorDestroy ( nalu_hypre_SStructPVector *pvector );
NALU_HYPRE_Int nalu_hypre_SStructPVectorInitialize ( nalu_hypre_SStructPVector *pvector );
NALU_HYPRE_Int nalu_hypre_SStructPVectorSetValues ( nalu_hypre_SStructPVector *pvector, nalu_hypre_Index index,
                                          NALU_HYPRE_Int var, NALU_HYPRE_Complex *value, NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_SStructPVectorSetBoxValues( nalu_hypre_SStructPVector *pvector, nalu_hypre_Box *set_box,
                                            NALU_HYPRE_Int var, nalu_hypre_Box *value_box, NALU_HYPRE_Complex *values, NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_SStructPVectorAccumulate ( nalu_hypre_SStructPVector *pvector );
NALU_HYPRE_Int nalu_hypre_SStructPVectorAssemble ( nalu_hypre_SStructPVector *pvector );
NALU_HYPRE_Int nalu_hypre_SStructPVectorGather ( nalu_hypre_SStructPVector *pvector );
NALU_HYPRE_Int nalu_hypre_SStructPVectorGetValues ( nalu_hypre_SStructPVector *pvector, nalu_hypre_Index index,
                                          NALU_HYPRE_Int var, NALU_HYPRE_Complex *value );
NALU_HYPRE_Int nalu_hypre_SStructPVectorGetBoxValues( nalu_hypre_SStructPVector *pvector, nalu_hypre_Box *set_box,
                                            NALU_HYPRE_Int var, nalu_hypre_Box *value_box, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int nalu_hypre_SStructPVectorSetConstantValues ( nalu_hypre_SStructPVector *pvector,
                                                  NALU_HYPRE_Complex value );
NALU_HYPRE_Int nalu_hypre_SStructPVectorPrint ( const char *filename, nalu_hypre_SStructPVector *pvector,
                                      NALU_HYPRE_Int all );
NALU_HYPRE_Int nalu_hypre_SStructVectorRef ( nalu_hypre_SStructVector *vector, nalu_hypre_SStructVector **vector_ref );
NALU_HYPRE_Int nalu_hypre_SStructVectorSetConstantValues ( nalu_hypre_SStructVector *vector, NALU_HYPRE_Complex value );
NALU_HYPRE_Int nalu_hypre_SStructVectorConvert ( nalu_hypre_SStructVector *vector,
                                       nalu_hypre_ParVector **parvector_ptr );
NALU_HYPRE_Int nalu_hypre_SStructVectorParConvert ( nalu_hypre_SStructVector *vector,
                                          nalu_hypre_ParVector **parvector_ptr );
NALU_HYPRE_Int nalu_hypre_SStructVectorRestore ( nalu_hypre_SStructVector *vector, nalu_hypre_ParVector *parvector );
NALU_HYPRE_Int nalu_hypre_SStructVectorParRestore ( nalu_hypre_SStructVector *vector, nalu_hypre_ParVector *parvector );
NALU_HYPRE_Int nalu_hypre_SStructPVectorInitializeShell ( nalu_hypre_SStructPVector *pvector );
NALU_HYPRE_Int nalu_hypre_SStructVectorInitializeShell ( nalu_hypre_SStructVector *vector );
NALU_HYPRE_Int nalu_hypre_SStructVectorClearGhostValues ( nalu_hypre_SStructVector *vector );
NALU_HYPRE_MemoryLocation nalu_hypre_SStructVectorMemoryLocation(nalu_hypre_SStructVector *vector);

