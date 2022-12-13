/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* assumed_part.c */
NALU_HYPRE_Int hypre_APSubdivideRegion ( hypre_Box *region, NALU_HYPRE_Int dim, NALU_HYPRE_Int level,
                                    hypre_BoxArray *box_array, NALU_HYPRE_Int *num_new_boxes );
NALU_HYPRE_Int hypre_APFindMyBoxesInRegions ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         NALU_HYPRE_Int **p_count_array, NALU_HYPRE_Real **p_vol_array );
NALU_HYPRE_Int hypre_APGetAllBoxesInRegions ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         NALU_HYPRE_Int **p_count_array, NALU_HYPRE_Real **p_vol_array, MPI_Comm comm );
NALU_HYPRE_Int hypre_APShrinkRegions ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                  MPI_Comm comm );
NALU_HYPRE_Int hypre_APPruneRegions ( hypre_BoxArray *region_array, NALU_HYPRE_Int **p_count_array,
                                 NALU_HYPRE_Real **p_vol_array );
NALU_HYPRE_Int hypre_APRefineRegionsByVol ( hypre_BoxArray *region_array, NALU_HYPRE_Real *vol_array,
                                       NALU_HYPRE_Int max_regions, NALU_HYPRE_Real gamma, NALU_HYPRE_Int dim, NALU_HYPRE_Int *return_code, MPI_Comm comm );
NALU_HYPRE_Int hypre_StructAssumedPartitionCreate ( NALU_HYPRE_Int dim, hypre_Box *bounding_box,
                                               NALU_HYPRE_Real global_boxes_size, NALU_HYPRE_Int global_num_boxes, hypre_BoxArray *local_boxes,
                                               NALU_HYPRE_Int *local_boxnums, NALU_HYPRE_Int max_regions, NALU_HYPRE_Int max_refinements, NALU_HYPRE_Real gamma,
                                               MPI_Comm comm, hypre_StructAssumedPart **p_assumed_partition );
NALU_HYPRE_Int hypre_StructAssumedPartitionDestroy ( hypre_StructAssumedPart *assumed_part );
NALU_HYPRE_Int hypre_APFillResponseStructAssumedPart ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                                  NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                  NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Int hypre_StructAssumedPartitionGetRegionsFromProc ( hypre_StructAssumedPart *assumed_part,
                                                           NALU_HYPRE_Int proc_id, hypre_BoxArray *assumed_regions );
NALU_HYPRE_Int hypre_StructAssumedPartitionGetProcsFromBox ( hypre_StructAssumedPart *assumed_part,
                                                        hypre_Box *box, NALU_HYPRE_Int *num_proc_array, NALU_HYPRE_Int *size_alloc_proc_array,
                                                        NALU_HYPRE_Int **p_proc_array );

/* box_algebra.c */
NALU_HYPRE_Int hypre_IntersectBoxes ( hypre_Box *box1, hypre_Box *box2, hypre_Box *ibox );
NALU_HYPRE_Int hypre_SubtractBoxes ( hypre_Box *box1, hypre_Box *box2, hypre_BoxArray *box_array );
NALU_HYPRE_Int hypre_SubtractBoxArrays ( hypre_BoxArray *box_array1, hypre_BoxArray *box_array2,
                                    hypre_BoxArray *tmp_box_array );
NALU_HYPRE_Int hypre_UnionBoxes ( hypre_BoxArray *boxes );
NALU_HYPRE_Int hypre_MinUnionBoxes ( hypre_BoxArray *boxes );

/* box_boundary.c */
NALU_HYPRE_Int hypre_BoxBoundaryIntersect ( hypre_Box *box, hypre_StructGrid *grid, NALU_HYPRE_Int d,
                                       NALU_HYPRE_Int dir, hypre_BoxArray *boundary );
NALU_HYPRE_Int hypre_BoxBoundaryG ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundary );
NALU_HYPRE_Int hypre_BoxBoundaryDG ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundarym,
                                hypre_BoxArray *boundaryp, NALU_HYPRE_Int d );
NALU_HYPRE_Int hypre_GeneralBoxBoundaryIntersect( hypre_Box *box, hypre_StructGrid *grid,
                                             hypre_Index stencil_element, hypre_BoxArray *boundary );

/* box.c */
NALU_HYPRE_Int hypre_SetIndex ( hypre_Index index, NALU_HYPRE_Int val );
NALU_HYPRE_Int hypre_CopyIndex( hypre_Index in_index, hypre_Index out_index );
NALU_HYPRE_Int hypre_CopyToCleanIndex( hypre_Index in_index, NALU_HYPRE_Int ndim, hypre_Index out_index );
NALU_HYPRE_Int hypre_IndexEqual ( hypre_Index index, NALU_HYPRE_Int val, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int hypre_IndexMin( hypre_Index index, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int hypre_IndexMax( hypre_Index index, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int hypre_AddIndexes ( hypre_Index index1, hypre_Index index2, NALU_HYPRE_Int ndim,
                             hypre_Index result );
NALU_HYPRE_Int hypre_SubtractIndexes ( hypre_Index index1, hypre_Index index2, NALU_HYPRE_Int ndim,
                                  hypre_Index result );
NALU_HYPRE_Int hypre_IndexesEqual ( hypre_Index index1, hypre_Index index2, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int hypre_IndexPrint ( FILE *file, NALU_HYPRE_Int ndim, hypre_Index index );
NALU_HYPRE_Int hypre_IndexRead ( FILE *file, NALU_HYPRE_Int ndim, hypre_Index index );
hypre_Box *hypre_BoxCreate ( NALU_HYPRE_Int ndim );
NALU_HYPRE_Int hypre_BoxDestroy ( hypre_Box *box );
NALU_HYPRE_Int hypre_BoxInit( hypre_Box *box, NALU_HYPRE_Int  ndim );
NALU_HYPRE_Int hypre_BoxSetExtents ( hypre_Box *box, hypre_Index imin, hypre_Index imax );
NALU_HYPRE_Int hypre_CopyBox( hypre_Box *box1, hypre_Box *box2 );
hypre_Box *hypre_BoxDuplicate ( hypre_Box *box );
NALU_HYPRE_Int hypre_BoxVolume( hypre_Box *box );
NALU_HYPRE_Real hypre_doubleBoxVolume( hypre_Box *box );
NALU_HYPRE_Int hypre_IndexInBox ( hypre_Index index, hypre_Box *box );
NALU_HYPRE_Int hypre_BoxGetSize ( hypre_Box *box, hypre_Index size );
NALU_HYPRE_Int hypre_BoxGetStrideSize ( hypre_Box *box, hypre_Index stride, hypre_Index size );
NALU_HYPRE_Int hypre_BoxGetStrideVolume ( hypre_Box *box, hypre_Index stride, NALU_HYPRE_Int *volume_ptr );
NALU_HYPRE_Int hypre_BoxIndexRank( hypre_Box *box, hypre_Index index );
NALU_HYPRE_Int hypre_BoxRankIndex( hypre_Box *box, NALU_HYPRE_Int rank, hypre_Index index );
NALU_HYPRE_Int hypre_BoxOffsetDistance( hypre_Box *box, hypre_Index index );
NALU_HYPRE_Int hypre_BoxShiftPos( hypre_Box *box, hypre_Index shift );
NALU_HYPRE_Int hypre_BoxShiftNeg( hypre_Box *box, hypre_Index shift );
NALU_HYPRE_Int hypre_BoxGrowByIndex( hypre_Box *box, hypre_Index  index );
NALU_HYPRE_Int hypre_BoxGrowByValue( hypre_Box *box, NALU_HYPRE_Int val );
NALU_HYPRE_Int hypre_BoxGrowByArray ( hypre_Box *box, NALU_HYPRE_Int *array );
NALU_HYPRE_Int hypre_BoxPrint ( FILE *file, hypre_Box *box );
NALU_HYPRE_Int hypre_BoxRead ( FILE *file, NALU_HYPRE_Int ndim, hypre_Box **box_ptr );
hypre_BoxArray *hypre_BoxArrayCreate ( NALU_HYPRE_Int size, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int hypre_BoxArrayDestroy ( hypre_BoxArray *box_array );
NALU_HYPRE_Int hypre_BoxArraySetSize ( hypre_BoxArray *box_array, NALU_HYPRE_Int size );
hypre_BoxArray *hypre_BoxArrayDuplicate ( hypre_BoxArray *box_array );
NALU_HYPRE_Int hypre_AppendBox ( hypre_Box *box, hypre_BoxArray *box_array );
NALU_HYPRE_Int hypre_DeleteBox ( hypre_BoxArray *box_array, NALU_HYPRE_Int index );
NALU_HYPRE_Int hypre_DeleteMultipleBoxes ( hypre_BoxArray *box_array, NALU_HYPRE_Int *indices,
                                      NALU_HYPRE_Int num );
NALU_HYPRE_Int hypre_AppendBoxArray ( hypre_BoxArray *box_array_0, hypre_BoxArray *box_array_1 );
hypre_BoxArrayArray *hypre_BoxArrayArrayCreate ( NALU_HYPRE_Int size, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int hypre_BoxArrayArrayDestroy ( hypre_BoxArrayArray *box_array_array );
hypre_BoxArrayArray *hypre_BoxArrayArrayDuplicate ( hypre_BoxArrayArray *box_array_array );

/* box_manager.c */
NALU_HYPRE_Int hypre_BoxManEntryGetInfo ( hypre_BoxManEntry *entry, void **info_ptr );
NALU_HYPRE_Int hypre_BoxManEntryGetExtents ( hypre_BoxManEntry *entry, hypre_Index imin,
                                        hypre_Index imax );
NALU_HYPRE_Int hypre_BoxManEntryCopy ( hypre_BoxManEntry *fromentry, hypre_BoxManEntry *toentry );
NALU_HYPRE_Int hypre_BoxManSetAllGlobalKnown ( hypre_BoxManager *manager, NALU_HYPRE_Int known );
NALU_HYPRE_Int hypre_BoxManGetAllGlobalKnown ( hypre_BoxManager *manager, NALU_HYPRE_Int *known );
NALU_HYPRE_Int hypre_BoxManSetIsEntriesSort ( hypre_BoxManager *manager, NALU_HYPRE_Int is_sort );
NALU_HYPRE_Int hypre_BoxManGetIsEntriesSort ( hypre_BoxManager *manager, NALU_HYPRE_Int *is_sort );
NALU_HYPRE_Int hypre_BoxManGetGlobalIsGatherCalled ( hypre_BoxManager *manager, MPI_Comm comm,
                                                NALU_HYPRE_Int *is_gather );
NALU_HYPRE_Int hypre_BoxManGetAssumedPartition ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart **assumed_partition );
NALU_HYPRE_Int hypre_BoxManSetAssumedPartition ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart *assumed_partition );
NALU_HYPRE_Int hypre_BoxManSetBoundingBox ( hypre_BoxManager *manager, hypre_Box *bounding_box );
NALU_HYPRE_Int hypre_BoxManSetNumGhost ( hypre_BoxManager *manager, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int hypre_BoxManDeleteMultipleEntriesAndInfo ( hypre_BoxManager *manager, NALU_HYPRE_Int *indices,
                                                     NALU_HYPRE_Int num );
NALU_HYPRE_Int hypre_BoxManCreate ( NALU_HYPRE_Int max_nentries, NALU_HYPRE_Int info_size, NALU_HYPRE_Int dim,
                               hypre_Box *bounding_box, MPI_Comm comm, hypre_BoxManager **manager_ptr );
NALU_HYPRE_Int hypre_BoxManIncSize ( hypre_BoxManager *manager, NALU_HYPRE_Int inc_size );
NALU_HYPRE_Int hypre_BoxManDestroy ( hypre_BoxManager *manager );
NALU_HYPRE_Int hypre_BoxManAddEntry ( hypre_BoxManager *manager, hypre_Index imin, hypre_Index imax,
                                 NALU_HYPRE_Int proc_id, NALU_HYPRE_Int box_id, void *info );
NALU_HYPRE_Int hypre_BoxManGetEntry ( hypre_BoxManager *manager, NALU_HYPRE_Int proc, NALU_HYPRE_Int id,
                                 hypre_BoxManEntry **entry_ptr );
NALU_HYPRE_Int hypre_BoxManGetAllEntries ( hypre_BoxManager *manager, NALU_HYPRE_Int *num_entries,
                                      hypre_BoxManEntry **entries );
NALU_HYPRE_Int hypre_BoxManGetAllEntriesBoxes ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
NALU_HYPRE_Int hypre_BoxManGetLocalEntriesBoxes ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
NALU_HYPRE_Int hypre_BoxManGetAllEntriesBoxesProc ( hypre_BoxManager *manager, hypre_BoxArray *boxes,
                                               NALU_HYPRE_Int **procs_ptr );
NALU_HYPRE_Int hypre_BoxManGatherEntries ( hypre_BoxManager *manager, hypre_Index imin,
                                      hypre_Index imax );
NALU_HYPRE_Int hypre_BoxManAssemble ( hypre_BoxManager *manager );
NALU_HYPRE_Int hypre_BoxManIntersect ( hypre_BoxManager *manager, hypre_Index ilower, hypre_Index iupper,
                                  hypre_BoxManEntry ***entries_ptr, NALU_HYPRE_Int *nentries_ptr );
NALU_HYPRE_Int hypre_FillResponseBoxManAssemble1 ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                              NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Int hypre_FillResponseBoxManAssemble2 ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                              NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              NALU_HYPRE_Int *response_message_size );

/* communication_info.c */
NALU_HYPRE_Int hypre_CommInfoCreate ( hypre_BoxArrayArray *send_boxes, hypre_BoxArrayArray *recv_boxes,
                                 NALU_HYPRE_Int **send_procs, NALU_HYPRE_Int **recv_procs, NALU_HYPRE_Int **send_rboxnums,
                                 NALU_HYPRE_Int **recv_rboxnums, hypre_BoxArrayArray *send_rboxes, hypre_BoxArrayArray *recv_rboxes,
                                 NALU_HYPRE_Int boxes_match, hypre_CommInfo **comm_info_ptr );
NALU_HYPRE_Int hypre_CommInfoSetTransforms ( hypre_CommInfo *comm_info, NALU_HYPRE_Int num_transforms,
                                        hypre_Index *coords, hypre_Index *dirs, NALU_HYPRE_Int **send_transforms, NALU_HYPRE_Int **recv_transforms );
NALU_HYPRE_Int hypre_CommInfoGetTransforms ( hypre_CommInfo *comm_info, NALU_HYPRE_Int *num_transforms,
                                        hypre_Index **coords, hypre_Index **dirs );
NALU_HYPRE_Int hypre_CommInfoProjectSend ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
NALU_HYPRE_Int hypre_CommInfoProjectRecv ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
NALU_HYPRE_Int hypre_CommInfoDestroy ( hypre_CommInfo *comm_info );
NALU_HYPRE_Int hypre_CreateCommInfoFromStencil ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                            hypre_CommInfo **comm_info_ptr );
NALU_HYPRE_Int hypre_CreateCommInfoFromNumGhost ( hypre_StructGrid *grid, NALU_HYPRE_Int *num_ghost,
                                             hypre_CommInfo **comm_info_ptr );
NALU_HYPRE_Int hypre_CreateCommInfoFromGrids ( hypre_StructGrid *from_grid, hypre_StructGrid *to_grid,
                                          hypre_CommInfo **comm_info_ptr );

/* computation.c */
NALU_HYPRE_Int hypre_ComputeInfoCreate ( hypre_CommInfo *comm_info, hypre_BoxArrayArray *indt_boxes,
                                    hypre_BoxArrayArray *dept_boxes, hypre_ComputeInfo **compute_info_ptr );
NALU_HYPRE_Int hypre_ComputeInfoProjectSend ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
NALU_HYPRE_Int hypre_ComputeInfoProjectRecv ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
NALU_HYPRE_Int hypre_ComputeInfoProjectComp ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
NALU_HYPRE_Int hypre_ComputeInfoDestroy ( hypre_ComputeInfo *compute_info );
NALU_HYPRE_Int hypre_CreateComputeInfo ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                    hypre_ComputeInfo **compute_info_ptr );
NALU_HYPRE_Int hypre_ComputePkgCreate ( hypre_ComputeInfo *compute_info, hypre_BoxArray *data_space,
                                   NALU_HYPRE_Int num_values, hypre_StructGrid *grid, hypre_ComputePkg **compute_pkg_ptr );
NALU_HYPRE_Int hypre_ComputePkgDestroy ( hypre_ComputePkg *compute_pkg );
NALU_HYPRE_Int hypre_InitializeIndtComputations ( hypre_ComputePkg *compute_pkg, NALU_HYPRE_Complex *data,
                                             hypre_CommHandle **comm_handle_ptr );
NALU_HYPRE_Int hypre_FinalizeIndtComputations ( hypre_CommHandle *comm_handle );

/* NALU_HYPRE_struct_grid.c */
NALU_HYPRE_Int NALU_HYPRE_StructGridCreate ( MPI_Comm comm, NALU_HYPRE_Int dim, NALU_HYPRE_StructGrid *grid );
NALU_HYPRE_Int NALU_HYPRE_StructGridDestroy ( NALU_HYPRE_StructGrid grid );
NALU_HYPRE_Int NALU_HYPRE_StructGridSetExtents ( NALU_HYPRE_StructGrid grid, NALU_HYPRE_Int *ilower,
                                       NALU_HYPRE_Int *iupper );
NALU_HYPRE_Int NALU_HYPRE_StructGridSetPeriodic ( NALU_HYPRE_StructGrid grid, NALU_HYPRE_Int *periodic );
NALU_HYPRE_Int NALU_HYPRE_StructGridAssemble ( NALU_HYPRE_StructGrid grid );
NALU_HYPRE_Int NALU_HYPRE_StructGridSetNumGhost ( NALU_HYPRE_StructGrid grid, NALU_HYPRE_Int *num_ghost );

/* NALU_HYPRE_struct_matrix.c */
NALU_HYPRE_Int NALU_HYPRE_StructMatrixCreate ( MPI_Comm comm, NALU_HYPRE_StructGrid grid,
                                     NALU_HYPRE_StructStencil stencil, NALU_HYPRE_StructMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixDestroy ( NALU_HYPRE_StructMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixInitialize ( NALU_HYPRE_StructMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetValues ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_Int *grid_index,
                                        NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixGetValues ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_Int *grid_index,
                                        NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetBoxValues ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_Int *ilower,
                                           NALU_HYPRE_Int *iupper, NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices,
                                           NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixGetBoxValues ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_Int *ilower,
                                           NALU_HYPRE_Int *iupper, NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices,
                                           NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetConstantValues ( NALU_HYPRE_StructMatrix matrix,
                                                NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixAddToValues ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_Int *grid_index,
                                          NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixAddToBoxValues ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_Int *ilower,
                                             NALU_HYPRE_Int *iupper, NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices,
                                             NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixAddToConstantValues ( NALU_HYPRE_StructMatrix matrix,
                                                  NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixAssemble ( NALU_HYPRE_StructMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetNumGhost ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixGetGrid ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_StructGrid *grid );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetSymmetric ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_Int symmetric );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetConstantEntries ( NALU_HYPRE_StructMatrix matrix, NALU_HYPRE_Int nentries,
                                                 NALU_HYPRE_Int *entries );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixPrint ( const char *filename, NALU_HYPRE_StructMatrix matrix,
                                    NALU_HYPRE_Int all );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixMatvec ( NALU_HYPRE_Complex alpha, NALU_HYPRE_StructMatrix A,
                                     NALU_HYPRE_StructVector x, NALU_HYPRE_Complex beta, NALU_HYPRE_StructVector y );
NALU_HYPRE_Int NALU_HYPRE_StructMatrixClearBoundary( NALU_HYPRE_StructMatrix matrix );

/* NALU_HYPRE_struct_stencil.c */
NALU_HYPRE_Int NALU_HYPRE_StructStencilCreate ( NALU_HYPRE_Int dim, NALU_HYPRE_Int size, NALU_HYPRE_StructStencil *stencil );
NALU_HYPRE_Int NALU_HYPRE_StructStencilSetElement ( NALU_HYPRE_StructStencil stencil, NALU_HYPRE_Int element_index,
                                          NALU_HYPRE_Int *offset );
NALU_HYPRE_Int NALU_HYPRE_StructStencilDestroy ( NALU_HYPRE_StructStencil stencil );

/* NALU_HYPRE_struct_vector.c */
NALU_HYPRE_Int NALU_HYPRE_StructVectorCreate ( MPI_Comm comm, NALU_HYPRE_StructGrid grid,
                                     NALU_HYPRE_StructVector *vector );
NALU_HYPRE_Int NALU_HYPRE_StructVectorDestroy ( NALU_HYPRE_StructVector struct_vector );
NALU_HYPRE_Int NALU_HYPRE_StructVectorInitialize ( NALU_HYPRE_StructVector vector );
NALU_HYPRE_Int NALU_HYPRE_StructVectorSetValues ( NALU_HYPRE_StructVector vector, NALU_HYPRE_Int *grid_index,
                                        NALU_HYPRE_Complex values );
NALU_HYPRE_Int NALU_HYPRE_StructVectorSetBoxValues ( NALU_HYPRE_StructVector vector, NALU_HYPRE_Int *ilower,
                                           NALU_HYPRE_Int *iupper, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructVectorAddToValues ( NALU_HYPRE_StructVector vector, NALU_HYPRE_Int *grid_index,
                                          NALU_HYPRE_Complex values );
NALU_HYPRE_Int NALU_HYPRE_StructVectorAddToBoxValues ( NALU_HYPRE_StructVector vector, NALU_HYPRE_Int *ilower,
                                             NALU_HYPRE_Int *iupper, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructVectorScaleValues ( NALU_HYPRE_StructVector vector, NALU_HYPRE_Complex factor );
NALU_HYPRE_Int NALU_HYPRE_StructVectorGetValues ( NALU_HYPRE_StructVector vector, NALU_HYPRE_Int *grid_index,
                                        NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructVectorGetBoxValues ( NALU_HYPRE_StructVector vector, NALU_HYPRE_Int *ilower,
                                           NALU_HYPRE_Int *iupper, NALU_HYPRE_Complex *values );
NALU_HYPRE_Int NALU_HYPRE_StructVectorAssemble ( NALU_HYPRE_StructVector vector );
NALU_HYPRE_Int hypre_StructVectorPrintData ( FILE *file, hypre_StructVector *vector, NALU_HYPRE_Int all );
NALU_HYPRE_Int hypre_StructVectorReadData ( FILE *file, hypre_StructVector *vector );
NALU_HYPRE_Int NALU_HYPRE_StructVectorPrint ( const char *filename, NALU_HYPRE_StructVector vector,
                                    NALU_HYPRE_Int all );
NALU_HYPRE_Int NALU_HYPRE_StructVectorRead ( MPI_Comm comm, const char *filename,
                                   NALU_HYPRE_Int *num_ghost, NALU_HYPRE_StructVector *vector );
NALU_HYPRE_Int NALU_HYPRE_StructVectorSetNumGhost ( NALU_HYPRE_StructVector vector, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int NALU_HYPRE_StructVectorCopy ( NALU_HYPRE_StructVector x, NALU_HYPRE_StructVector y );
NALU_HYPRE_Int NALU_HYPRE_StructVectorSetConstantValues ( NALU_HYPRE_StructVector vector, NALU_HYPRE_Complex values );
NALU_HYPRE_Int NALU_HYPRE_StructVectorGetMigrateCommPkg ( NALU_HYPRE_StructVector from_vector,
                                                NALU_HYPRE_StructVector to_vector, NALU_HYPRE_CommPkg *comm_pkg );
NALU_HYPRE_Int NALU_HYPRE_StructVectorMigrate ( NALU_HYPRE_CommPkg comm_pkg, NALU_HYPRE_StructVector from_vector,
                                      NALU_HYPRE_StructVector to_vector );
NALU_HYPRE_Int NALU_HYPRE_CommPkgDestroy ( NALU_HYPRE_CommPkg comm_pkg );

/* project.c */
NALU_HYPRE_Int hypre_ProjectBox ( hypre_Box *box, hypre_Index index, hypre_Index stride );
NALU_HYPRE_Int hypre_ProjectBoxArray ( hypre_BoxArray *box_array, hypre_Index index,
                                  hypre_Index stride );
NALU_HYPRE_Int hypre_ProjectBoxArrayArray ( hypre_BoxArrayArray *box_array_array, hypre_Index index,
                                       hypre_Index stride );

/* struct_axpy.c */
NALU_HYPRE_Int hypre_StructAxpy ( NALU_HYPRE_Complex alpha, hypre_StructVector *x, hypre_StructVector *y );

/* struct_communication.c */
NALU_HYPRE_Int hypre_CommPkgCreate ( hypre_CommInfo *comm_info, hypre_BoxArray *send_data_space,
                                hypre_BoxArray *recv_data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Int **orders, NALU_HYPRE_Int reverse,
                                MPI_Comm comm, hypre_CommPkg **comm_pkg_ptr );
NALU_HYPRE_Int hypre_CommTypeSetEntries ( hypre_CommType *comm_type, NALU_HYPRE_Int *boxnums,
                                     hypre_Box *boxes, hypre_Index stride, hypre_Index coord, hypre_Index dir, NALU_HYPRE_Int *order,
                                     hypre_BoxArray *data_space, NALU_HYPRE_Int *data_offsets );
NALU_HYPRE_Int hypre_CommTypeSetEntry ( hypre_Box *box, hypre_Index stride, hypre_Index coord,
                                   hypre_Index dir, NALU_HYPRE_Int *order, hypre_Box *data_box, NALU_HYPRE_Int data_box_offset,
                                   hypre_CommEntryType *comm_entry );
NALU_HYPRE_Int hypre_InitializeCommunication ( hypre_CommPkg *comm_pkg, NALU_HYPRE_Complex *send_data,
                                          NALU_HYPRE_Complex *recv_data, NALU_HYPRE_Int action, NALU_HYPRE_Int tag, hypre_CommHandle **comm_handle_ptr );
NALU_HYPRE_Int hypre_FinalizeCommunication ( hypre_CommHandle *comm_handle );
NALU_HYPRE_Int hypre_ExchangeLocalData ( hypre_CommPkg *comm_pkg, NALU_HYPRE_Complex *send_data,
                                    NALU_HYPRE_Complex *recv_data, NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_CommPkgDestroy ( hypre_CommPkg *comm_pkg );

/* struct_copy.c */
NALU_HYPRE_Int hypre_StructCopy ( hypre_StructVector *x, hypre_StructVector *y );
NALU_HYPRE_Int hypre_StructPartialCopy ( hypre_StructVector *x, hypre_StructVector *y,
                                    hypre_BoxArrayArray *array_boxes );

/* struct_grid.c */
NALU_HYPRE_Int hypre_StructGridCreate ( MPI_Comm comm, NALU_HYPRE_Int dim, hypre_StructGrid **grid_ptr );
NALU_HYPRE_Int hypre_StructGridRef ( hypre_StructGrid *grid, hypre_StructGrid **grid_ref );
NALU_HYPRE_Int hypre_StructGridDestroy ( hypre_StructGrid *grid );
NALU_HYPRE_Int hypre_StructGridSetPeriodic ( hypre_StructGrid *grid, hypre_Index periodic );
NALU_HYPRE_Int hypre_StructGridSetExtents ( hypre_StructGrid *grid, hypre_Index ilower,
                                       hypre_Index iupper );
NALU_HYPRE_Int hypre_StructGridSetBoxes ( hypre_StructGrid *grid, hypre_BoxArray *boxes );
NALU_HYPRE_Int hypre_StructGridSetBoundingBox ( hypre_StructGrid *grid, hypre_Box *new_bb );
NALU_HYPRE_Int hypre_StructGridSetIDs ( hypre_StructGrid *grid, NALU_HYPRE_Int *ids );
NALU_HYPRE_Int hypre_StructGridSetBoxManager ( hypre_StructGrid *grid, hypre_BoxManager *boxman );
NALU_HYPRE_Int hypre_StructGridSetMaxDistance ( hypre_StructGrid *grid, hypre_Index dist );
NALU_HYPRE_Int hypre_StructGridAssemble ( hypre_StructGrid *grid );
NALU_HYPRE_Int hypre_GatherAllBoxes ( MPI_Comm comm, hypre_BoxArray *boxes, NALU_HYPRE_Int dim,
                                 hypre_BoxArray **all_boxes_ptr, NALU_HYPRE_Int **all_procs_ptr, NALU_HYPRE_Int *first_local_ptr );
NALU_HYPRE_Int hypre_ComputeBoxnums ( hypre_BoxArray *boxes, NALU_HYPRE_Int *procs, NALU_HYPRE_Int **boxnums_ptr );
NALU_HYPRE_Int hypre_StructGridPrint ( FILE *file, hypre_StructGrid *grid );
NALU_HYPRE_Int hypre_StructGridRead ( MPI_Comm comm, FILE *file, hypre_StructGrid **grid_ptr );
NALU_HYPRE_Int hypre_StructGridSetNumGhost ( hypre_StructGrid *grid, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int hypre_StructGridGetMaxBoxSize ( hypre_StructGrid *grid );
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int hypre_StructGridSetDataLocation( NALU_HYPRE_StructGrid grid,
                                           NALU_HYPRE_MemoryLocation data_location );
#endif
/* struct_innerprod.c */
NALU_HYPRE_Real hypre_StructInnerProd ( hypre_StructVector *x, hypre_StructVector *y );

/* struct_io.c */
NALU_HYPRE_Int hypre_PrintBoxArrayData ( FILE *file, hypre_BoxArray *box_array,
                                    hypre_BoxArray *data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Int dim, NALU_HYPRE_Complex *data );
NALU_HYPRE_Int hypre_PrintCCVDBoxArrayData ( FILE *file, hypre_BoxArray *box_array,
                                        hypre_BoxArray *data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Int center_rank, NALU_HYPRE_Int stencil_size,
                                        NALU_HYPRE_Int *symm_elements, NALU_HYPRE_Int dim, NALU_HYPRE_Complex *data );
NALU_HYPRE_Int hypre_PrintCCBoxArrayData ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Complex *data );
NALU_HYPRE_Int hypre_ReadBoxArrayData ( FILE *file, hypre_BoxArray *box_array,
                                   hypre_BoxArray *data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Int dim, NALU_HYPRE_Complex *data );
NALU_HYPRE_Int hypre_ReadBoxArrayData_CC ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, NALU_HYPRE_Int stencil_size, NALU_HYPRE_Int real_stencil_size,
                                      NALU_HYPRE_Int constant_coefficient, NALU_HYPRE_Int dim, NALU_HYPRE_Complex *data );

/* struct_matrix.c */
NALU_HYPRE_Complex *hypre_StructMatrixExtractPointerByIndex ( hypre_StructMatrix *matrix, NALU_HYPRE_Int b,
                                                         hypre_Index index );
hypre_StructMatrix *hypre_StructMatrixCreate ( MPI_Comm comm, hypre_StructGrid *grid,
                                               hypre_StructStencil *user_stencil );
hypre_StructMatrix *hypre_StructMatrixRef ( hypre_StructMatrix *matrix );
NALU_HYPRE_Int hypre_StructMatrixDestroy ( hypre_StructMatrix *matrix );
NALU_HYPRE_Int hypre_StructMatrixInitializeShell ( hypre_StructMatrix *matrix );
NALU_HYPRE_Int hypre_StructMatrixInitializeData ( hypre_StructMatrix *matrix, NALU_HYPRE_Complex *data,
                                             NALU_HYPRE_Complex *data_const);
NALU_HYPRE_Int hypre_StructMatrixInitialize ( hypre_StructMatrix *matrix );
NALU_HYPRE_Int hypre_StructMatrixSetValues ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                        NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Complex *values, NALU_HYPRE_Int action,
                                        NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int hypre_StructMatrixSetBoxValues ( hypre_StructMatrix *matrix, hypre_Box *set_box,
                                           hypre_Box *value_box, NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices,
                                           NALU_HYPRE_Complex *values, NALU_HYPRE_Int action, NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int hypre_StructMatrixSetConstantValues ( hypre_StructMatrix *matrix,
                                                NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Complex *values,
                                                NALU_HYPRE_Int action );
NALU_HYPRE_Int hypre_StructMatrixClearValues ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                          NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int hypre_StructMatrixClearBoxValues ( hypre_StructMatrix *matrix, hypre_Box *clear_box,
                                             NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int hypre_StructMatrixAssemble ( hypre_StructMatrix *matrix );
NALU_HYPRE_Int hypre_StructMatrixSetNumGhost ( hypre_StructMatrix *matrix, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int hypre_StructMatrixSetConstantCoefficient ( hypre_StructMatrix *matrix,
                                                     NALU_HYPRE_Int constant_coefficient );
NALU_HYPRE_Int hypre_StructMatrixSetConstantEntries ( hypre_StructMatrix *matrix, NALU_HYPRE_Int nentries,
                                                 NALU_HYPRE_Int *entries );
NALU_HYPRE_Int hypre_StructMatrixClearGhostValues ( hypre_StructMatrix *matrix );
NALU_HYPRE_Int hypre_StructMatrixPrintData ( FILE *file, hypre_StructMatrix *matrix, NALU_HYPRE_Int all );
NALU_HYPRE_Int hypre_StructMatrixReadData ( FILE *file, hypre_StructMatrix *matrix );
NALU_HYPRE_Int hypre_StructMatrixPrint ( const char *filename, hypre_StructMatrix *matrix,
                                    NALU_HYPRE_Int all );
hypre_StructMatrix *hypre_StructMatrixRead ( MPI_Comm comm, const char *filename,
                                             NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int hypre_StructMatrixMigrate ( hypre_StructMatrix *from_matrix,
                                      hypre_StructMatrix *to_matrix );
NALU_HYPRE_Int hypre_StructMatrixClearBoundary( hypre_StructMatrix *matrix);

/* struct_matrix_mask.c */
hypre_StructMatrix *hypre_StructMatrixCreateMask ( hypre_StructMatrix *matrix,
                                                   NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices );

/* struct_matvec.c */
void *hypre_StructMatvecCreate ( void );
NALU_HYPRE_Int hypre_StructMatvecSetup ( void *matvec_vdata, hypre_StructMatrix *A,
                                    hypre_StructVector *x );
NALU_HYPRE_Int hypre_StructMatvecCompute ( void *matvec_vdata, NALU_HYPRE_Complex alpha,
                                      hypre_StructMatrix *A, hypre_StructVector *x, NALU_HYPRE_Complex beta, hypre_StructVector *y );
NALU_HYPRE_Int hypre_StructMatvecCC0 ( NALU_HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
NALU_HYPRE_Int hypre_StructMatvecCC1 ( NALU_HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
NALU_HYPRE_Int hypre_StructMatvecCC2 ( NALU_HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
NALU_HYPRE_Int hypre_StructMatvecDestroy ( void *matvec_vdata );
NALU_HYPRE_Int hypre_StructMatvec ( NALU_HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                               NALU_HYPRE_Complex beta, hypre_StructVector *y );

/* struct_scale.c */
NALU_HYPRE_Int hypre_StructScale ( NALU_HYPRE_Complex alpha, hypre_StructVector *y );

/* struct_stencil.c */
hypre_StructStencil *hypre_StructStencilCreate ( NALU_HYPRE_Int dim, NALU_HYPRE_Int size,
                                                 hypre_Index *shape );
hypre_StructStencil *hypre_StructStencilRef ( hypre_StructStencil *stencil );
NALU_HYPRE_Int hypre_StructStencilDestroy ( hypre_StructStencil *stencil );
NALU_HYPRE_Int hypre_StructStencilElementRank ( hypre_StructStencil *stencil,
                                           hypre_Index stencil_element );
NALU_HYPRE_Int hypre_StructStencilSymmetrize ( hypre_StructStencil *stencil,
                                          hypre_StructStencil **symm_stencil_ptr, NALU_HYPRE_Int **symm_elements_ptr );

/* struct_vector.c */
hypre_StructVector *hypre_StructVectorCreate ( MPI_Comm comm, hypre_StructGrid *grid );
hypre_StructVector *hypre_StructVectorRef ( hypre_StructVector *vector );
NALU_HYPRE_Int hypre_StructVectorDestroy ( hypre_StructVector *vector );
NALU_HYPRE_Int hypre_StructVectorInitializeShell ( hypre_StructVector *vector );
NALU_HYPRE_Int hypre_StructVectorInitializeData ( hypre_StructVector *vector, NALU_HYPRE_Complex *data);
NALU_HYPRE_Int hypre_StructVectorInitialize ( hypre_StructVector *vector );
NALU_HYPRE_Int hypre_StructVectorSetValues ( hypre_StructVector *vector, hypre_Index grid_index,
                                        NALU_HYPRE_Complex *values, NALU_HYPRE_Int action, NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int hypre_StructVectorSetBoxValues ( hypre_StructVector *vector, hypre_Box *set_box,
                                           hypre_Box *value_box, NALU_HYPRE_Complex *values, NALU_HYPRE_Int action, NALU_HYPRE_Int boxnum,
                                           NALU_HYPRE_Int outside );
NALU_HYPRE_Int hypre_StructVectorClearValues ( hypre_StructVector *vector, hypre_Index grid_index,
                                          NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int hypre_StructVectorClearBoxValues ( hypre_StructVector *vector, hypre_Box *clear_box,
                                             NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int hypre_StructVectorClearAllValues ( hypre_StructVector *vector );
NALU_HYPRE_Int hypre_StructVectorSetNumGhost ( hypre_StructVector *vector, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int hypre_StructVectorSetDataSize(hypre_StructVector *vector, NALU_HYPRE_Int *data_size,
                                        NALU_HYPRE_Int *data_host_size);
NALU_HYPRE_Int hypre_StructVectorAssemble ( hypre_StructVector *vector );
NALU_HYPRE_Int hypre_StructVectorCopy ( hypre_StructVector *x, hypre_StructVector *y );
NALU_HYPRE_Int hypre_StructVectorSetConstantValues ( hypre_StructVector *vector, NALU_HYPRE_Complex values );
NALU_HYPRE_Int hypre_StructVectorSetFunctionValues ( hypre_StructVector *vector,
                                                NALU_HYPRE_Complex (*fcn )());
NALU_HYPRE_Int hypre_StructVectorClearGhostValues ( hypre_StructVector *vector );
NALU_HYPRE_Int hypre_StructVectorClearBoundGhostValues ( hypre_StructVector *vector, NALU_HYPRE_Int force );
NALU_HYPRE_Int hypre_StructVectorScaleValues ( hypre_StructVector *vector, NALU_HYPRE_Complex factor );
hypre_CommPkg *hypre_StructVectorGetMigrateCommPkg ( hypre_StructVector *from_vector,
                                                     hypre_StructVector *to_vector );
NALU_HYPRE_Int hypre_StructVectorMigrate ( hypre_CommPkg *comm_pkg, hypre_StructVector *from_vector,
                                      hypre_StructVector *to_vector );
NALU_HYPRE_Int hypre_StructVectorPrint ( const char *filename, hypre_StructVector *vector,
                                    NALU_HYPRE_Int all );
hypre_StructVector *hypre_StructVectorRead ( MPI_Comm comm, const char *filename,
                                             NALU_HYPRE_Int *num_ghost );
hypre_StructVector *hypre_StructVectorClone ( hypre_StructVector *vector );
