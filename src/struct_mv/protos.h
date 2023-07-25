/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* assumed_part.c */
NALU_HYPRE_Int nalu_hypre_APSubdivideRegion ( nalu_hypre_Box *region, NALU_HYPRE_Int dim, NALU_HYPRE_Int level,
                                    nalu_hypre_BoxArray *box_array, NALU_HYPRE_Int *num_new_boxes );
NALU_HYPRE_Int nalu_hypre_APFindMyBoxesInRegions ( nalu_hypre_BoxArray *region_array, nalu_hypre_BoxArray *my_box_array,
                                         NALU_HYPRE_Int **p_count_array, NALU_HYPRE_Real **p_vol_array );
NALU_HYPRE_Int nalu_hypre_APGetAllBoxesInRegions ( nalu_hypre_BoxArray *region_array, nalu_hypre_BoxArray *my_box_array,
                                         NALU_HYPRE_Int **p_count_array, NALU_HYPRE_Real **p_vol_array, MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_APShrinkRegions ( nalu_hypre_BoxArray *region_array, nalu_hypre_BoxArray *my_box_array,
                                  MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_APPruneRegions ( nalu_hypre_BoxArray *region_array, NALU_HYPRE_Int **p_count_array,
                                 NALU_HYPRE_Real **p_vol_array );
NALU_HYPRE_Int nalu_hypre_APRefineRegionsByVol ( nalu_hypre_BoxArray *region_array, NALU_HYPRE_Real *vol_array,
                                       NALU_HYPRE_Int max_regions, NALU_HYPRE_Real gamma, NALU_HYPRE_Int dim, NALU_HYPRE_Int *return_code, MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_StructAssumedPartitionCreate ( NALU_HYPRE_Int dim, nalu_hypre_Box *bounding_box,
                                               NALU_HYPRE_Real global_boxes_size, NALU_HYPRE_Int global_num_boxes, nalu_hypre_BoxArray *local_boxes,
                                               NALU_HYPRE_Int *local_boxnums, NALU_HYPRE_Int max_regions, NALU_HYPRE_Int max_refinements, NALU_HYPRE_Real gamma,
                                               MPI_Comm comm, nalu_hypre_StructAssumedPart **p_assumed_partition );
NALU_HYPRE_Int nalu_hypre_StructAssumedPartitionDestroy ( nalu_hypre_StructAssumedPart *assumed_part );
NALU_HYPRE_Int nalu_hypre_APFillResponseStructAssumedPart ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                                  NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                  NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Int nalu_hypre_StructAssumedPartitionGetRegionsFromProc ( nalu_hypre_StructAssumedPart *assumed_part,
                                                           NALU_HYPRE_Int proc_id, nalu_hypre_BoxArray *assumed_regions );
NALU_HYPRE_Int nalu_hypre_StructAssumedPartitionGetProcsFromBox ( nalu_hypre_StructAssumedPart *assumed_part,
                                                        nalu_hypre_Box *box, NALU_HYPRE_Int *num_proc_array, NALU_HYPRE_Int *size_alloc_proc_array,
                                                        NALU_HYPRE_Int **p_proc_array );

/* box_algebra.c */
NALU_HYPRE_Int nalu_hypre_IntersectBoxes ( nalu_hypre_Box *box1, nalu_hypre_Box *box2, nalu_hypre_Box *ibox );
NALU_HYPRE_Int nalu_hypre_SubtractBoxes ( nalu_hypre_Box *box1, nalu_hypre_Box *box2, nalu_hypre_BoxArray *box_array );
NALU_HYPRE_Int nalu_hypre_SubtractBoxArrays ( nalu_hypre_BoxArray *box_array1, nalu_hypre_BoxArray *box_array2,
                                    nalu_hypre_BoxArray *tmp_box_array );
NALU_HYPRE_Int nalu_hypre_UnionBoxes ( nalu_hypre_BoxArray *boxes );
NALU_HYPRE_Int nalu_hypre_MinUnionBoxes ( nalu_hypre_BoxArray *boxes );

/* box_boundary.c */
NALU_HYPRE_Int nalu_hypre_BoxBoundaryIntersect ( nalu_hypre_Box *box, nalu_hypre_StructGrid *grid, NALU_HYPRE_Int d,
                                       NALU_HYPRE_Int dir, nalu_hypre_BoxArray *boundary );
NALU_HYPRE_Int nalu_hypre_BoxBoundaryG ( nalu_hypre_Box *box, nalu_hypre_StructGrid *g, nalu_hypre_BoxArray *boundary );
NALU_HYPRE_Int nalu_hypre_BoxBoundaryDG ( nalu_hypre_Box *box, nalu_hypre_StructGrid *g, nalu_hypre_BoxArray *boundarym,
                                nalu_hypre_BoxArray *boundaryp, NALU_HYPRE_Int d );
NALU_HYPRE_Int nalu_hypre_GeneralBoxBoundaryIntersect( nalu_hypre_Box *box, nalu_hypre_StructGrid *grid,
                                             nalu_hypre_Index stencil_element, nalu_hypre_BoxArray *boundary );

/* box.c */
NALU_HYPRE_Int nalu_hypre_SetIndex ( nalu_hypre_Index index, NALU_HYPRE_Int val );
NALU_HYPRE_Int nalu_hypre_CopyIndex( nalu_hypre_Index in_index, nalu_hypre_Index out_index );
NALU_HYPRE_Int nalu_hypre_CopyToCleanIndex( nalu_hypre_Index in_index, NALU_HYPRE_Int ndim, nalu_hypre_Index out_index );
NALU_HYPRE_Int nalu_hypre_IndexEqual ( nalu_hypre_Index index, NALU_HYPRE_Int val, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int nalu_hypre_IndexMin( nalu_hypre_Index index, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int nalu_hypre_IndexMax( nalu_hypre_Index index, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int nalu_hypre_AddIndexes ( nalu_hypre_Index index1, nalu_hypre_Index index2, NALU_HYPRE_Int ndim,
                             nalu_hypre_Index result );
NALU_HYPRE_Int nalu_hypre_SubtractIndexes ( nalu_hypre_Index index1, nalu_hypre_Index index2, NALU_HYPRE_Int ndim,
                                  nalu_hypre_Index result );
NALU_HYPRE_Int nalu_hypre_IndexesEqual ( nalu_hypre_Index index1, nalu_hypre_Index index2, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int nalu_hypre_IndexPrint ( FILE *file, NALU_HYPRE_Int ndim, nalu_hypre_Index index );
NALU_HYPRE_Int nalu_hypre_IndexRead ( FILE *file, NALU_HYPRE_Int ndim, nalu_hypre_Index index );
nalu_hypre_Box *nalu_hypre_BoxCreate ( NALU_HYPRE_Int ndim );
NALU_HYPRE_Int nalu_hypre_BoxDestroy ( nalu_hypre_Box *box );
NALU_HYPRE_Int nalu_hypre_BoxInit( nalu_hypre_Box *box, NALU_HYPRE_Int  ndim );
NALU_HYPRE_Int nalu_hypre_BoxSetExtents ( nalu_hypre_Box *box, nalu_hypre_Index imin, nalu_hypre_Index imax );
NALU_HYPRE_Int nalu_hypre_CopyBox( nalu_hypre_Box *box1, nalu_hypre_Box *box2 );
nalu_hypre_Box *nalu_hypre_BoxDuplicate ( nalu_hypre_Box *box );
NALU_HYPRE_Int nalu_hypre_BoxVolume( nalu_hypre_Box *box );
NALU_HYPRE_Real nalu_hypre_doubleBoxVolume( nalu_hypre_Box *box );
NALU_HYPRE_Int nalu_hypre_IndexInBox ( nalu_hypre_Index index, nalu_hypre_Box *box );
NALU_HYPRE_Int nalu_hypre_BoxGetSize ( nalu_hypre_Box *box, nalu_hypre_Index size );
NALU_HYPRE_Int nalu_hypre_BoxGetStrideSize ( nalu_hypre_Box *box, nalu_hypre_Index stride, nalu_hypre_Index size );
NALU_HYPRE_Int nalu_hypre_BoxGetStrideVolume ( nalu_hypre_Box *box, nalu_hypre_Index stride, NALU_HYPRE_Int *volume_ptr );
NALU_HYPRE_Int nalu_hypre_BoxIndexRank( nalu_hypre_Box *box, nalu_hypre_Index index );
NALU_HYPRE_Int nalu_hypre_BoxRankIndex( nalu_hypre_Box *box, NALU_HYPRE_Int rank, nalu_hypre_Index index );
NALU_HYPRE_Int nalu_hypre_BoxOffsetDistance( nalu_hypre_Box *box, nalu_hypre_Index index );
NALU_HYPRE_Int nalu_hypre_BoxShiftPos( nalu_hypre_Box *box, nalu_hypre_Index shift );
NALU_HYPRE_Int nalu_hypre_BoxShiftNeg( nalu_hypre_Box *box, nalu_hypre_Index shift );
NALU_HYPRE_Int nalu_hypre_BoxGrowByIndex( nalu_hypre_Box *box, nalu_hypre_Index  index );
NALU_HYPRE_Int nalu_hypre_BoxGrowByValue( nalu_hypre_Box *box, NALU_HYPRE_Int val );
NALU_HYPRE_Int nalu_hypre_BoxGrowByArray ( nalu_hypre_Box *box, NALU_HYPRE_Int *array );
NALU_HYPRE_Int nalu_hypre_BoxPrint ( FILE *file, nalu_hypre_Box *box );
NALU_HYPRE_Int nalu_hypre_BoxRead ( FILE *file, NALU_HYPRE_Int ndim, nalu_hypre_Box **box_ptr );
nalu_hypre_BoxArray *nalu_hypre_BoxArrayCreate ( NALU_HYPRE_Int size, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int nalu_hypre_BoxArrayDestroy ( nalu_hypre_BoxArray *box_array );
NALU_HYPRE_Int nalu_hypre_BoxArraySetSize ( nalu_hypre_BoxArray *box_array, NALU_HYPRE_Int size );
nalu_hypre_BoxArray *nalu_hypre_BoxArrayDuplicate ( nalu_hypre_BoxArray *box_array );
NALU_HYPRE_Int nalu_hypre_AppendBox ( nalu_hypre_Box *box, nalu_hypre_BoxArray *box_array );
NALU_HYPRE_Int nalu_hypre_DeleteBox ( nalu_hypre_BoxArray *box_array, NALU_HYPRE_Int index );
NALU_HYPRE_Int nalu_hypre_DeleteMultipleBoxes ( nalu_hypre_BoxArray *box_array, NALU_HYPRE_Int *indices,
                                      NALU_HYPRE_Int num );
NALU_HYPRE_Int nalu_hypre_AppendBoxArray ( nalu_hypre_BoxArray *box_array_0, nalu_hypre_BoxArray *box_array_1 );
nalu_hypre_BoxArrayArray *nalu_hypre_BoxArrayArrayCreate ( NALU_HYPRE_Int size, NALU_HYPRE_Int ndim );
NALU_HYPRE_Int nalu_hypre_BoxArrayArrayDestroy ( nalu_hypre_BoxArrayArray *box_array_array );
nalu_hypre_BoxArrayArray *nalu_hypre_BoxArrayArrayDuplicate ( nalu_hypre_BoxArrayArray *box_array_array );

/* box_manager.c */
NALU_HYPRE_Int nalu_hypre_BoxManEntryGetInfo ( nalu_hypre_BoxManEntry *entry, void **info_ptr );
NALU_HYPRE_Int nalu_hypre_BoxManEntryGetExtents ( nalu_hypre_BoxManEntry *entry, nalu_hypre_Index imin,
                                        nalu_hypre_Index imax );
NALU_HYPRE_Int nalu_hypre_BoxManEntryCopy ( nalu_hypre_BoxManEntry *fromentry, nalu_hypre_BoxManEntry *toentry );
NALU_HYPRE_Int nalu_hypre_BoxManSetAllGlobalKnown ( nalu_hypre_BoxManager *manager, NALU_HYPRE_Int known );
NALU_HYPRE_Int nalu_hypre_BoxManGetAllGlobalKnown ( nalu_hypre_BoxManager *manager, NALU_HYPRE_Int *known );
NALU_HYPRE_Int nalu_hypre_BoxManSetIsEntriesSort ( nalu_hypre_BoxManager *manager, NALU_HYPRE_Int is_sort );
NALU_HYPRE_Int nalu_hypre_BoxManGetIsEntriesSort ( nalu_hypre_BoxManager *manager, NALU_HYPRE_Int *is_sort );
NALU_HYPRE_Int nalu_hypre_BoxManGetGlobalIsGatherCalled ( nalu_hypre_BoxManager *manager, MPI_Comm comm,
                                                NALU_HYPRE_Int *is_gather );
NALU_HYPRE_Int nalu_hypre_BoxManGetAssumedPartition ( nalu_hypre_BoxManager *manager,
                                            nalu_hypre_StructAssumedPart **assumed_partition );
NALU_HYPRE_Int nalu_hypre_BoxManSetAssumedPartition ( nalu_hypre_BoxManager *manager,
                                            nalu_hypre_StructAssumedPart *assumed_partition );
NALU_HYPRE_Int nalu_hypre_BoxManSetBoundingBox ( nalu_hypre_BoxManager *manager, nalu_hypre_Box *bounding_box );
NALU_HYPRE_Int nalu_hypre_BoxManSetNumGhost ( nalu_hypre_BoxManager *manager, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int nalu_hypre_BoxManDeleteMultipleEntriesAndInfo ( nalu_hypre_BoxManager *manager, NALU_HYPRE_Int *indices,
                                                     NALU_HYPRE_Int num );
NALU_HYPRE_Int nalu_hypre_BoxManCreate ( NALU_HYPRE_Int max_nentries, NALU_HYPRE_Int info_size, NALU_HYPRE_Int dim,
                               nalu_hypre_Box *bounding_box, MPI_Comm comm, nalu_hypre_BoxManager **manager_ptr );
NALU_HYPRE_Int nalu_hypre_BoxManIncSize ( nalu_hypre_BoxManager *manager, NALU_HYPRE_Int inc_size );
NALU_HYPRE_Int nalu_hypre_BoxManDestroy ( nalu_hypre_BoxManager *manager );
NALU_HYPRE_Int nalu_hypre_BoxManAddEntry ( nalu_hypre_BoxManager *manager, nalu_hypre_Index imin, nalu_hypre_Index imax,
                                 NALU_HYPRE_Int proc_id, NALU_HYPRE_Int box_id, void *info );
NALU_HYPRE_Int nalu_hypre_BoxManGetEntry ( nalu_hypre_BoxManager *manager, NALU_HYPRE_Int proc, NALU_HYPRE_Int id,
                                 nalu_hypre_BoxManEntry **entry_ptr );
NALU_HYPRE_Int nalu_hypre_BoxManGetAllEntries ( nalu_hypre_BoxManager *manager, NALU_HYPRE_Int *num_entries,
                                      nalu_hypre_BoxManEntry **entries );
NALU_HYPRE_Int nalu_hypre_BoxManGetAllEntriesBoxes ( nalu_hypre_BoxManager *manager, nalu_hypre_BoxArray *boxes );
NALU_HYPRE_Int nalu_hypre_BoxManGetLocalEntriesBoxes ( nalu_hypre_BoxManager *manager, nalu_hypre_BoxArray *boxes );
NALU_HYPRE_Int nalu_hypre_BoxManGetAllEntriesBoxesProc ( nalu_hypre_BoxManager *manager, nalu_hypre_BoxArray *boxes,
                                               NALU_HYPRE_Int **procs_ptr );
NALU_HYPRE_Int nalu_hypre_BoxManGatherEntries ( nalu_hypre_BoxManager *manager, nalu_hypre_Index imin,
                                      nalu_hypre_Index imax );
NALU_HYPRE_Int nalu_hypre_BoxManAssemble ( nalu_hypre_BoxManager *manager );
NALU_HYPRE_Int nalu_hypre_BoxManIntersect ( nalu_hypre_BoxManager *manager, nalu_hypre_Index ilower, nalu_hypre_Index iupper,
                                  nalu_hypre_BoxManEntry ***entries_ptr, NALU_HYPRE_Int *nentries_ptr );
NALU_HYPRE_Int nalu_hypre_FillResponseBoxManAssemble1 ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                              NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              NALU_HYPRE_Int *response_message_size );
NALU_HYPRE_Int nalu_hypre_FillResponseBoxManAssemble2 ( void *p_recv_contact_buf, NALU_HYPRE_Int contact_size,
                                              NALU_HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              NALU_HYPRE_Int *response_message_size );

/* communication_info.c */
NALU_HYPRE_Int nalu_hypre_CommInfoCreate ( nalu_hypre_BoxArrayArray *send_boxes, nalu_hypre_BoxArrayArray *recv_boxes,
                                 NALU_HYPRE_Int **send_procs, NALU_HYPRE_Int **recv_procs, NALU_HYPRE_Int **send_rboxnums,
                                 NALU_HYPRE_Int **recv_rboxnums, nalu_hypre_BoxArrayArray *send_rboxes, nalu_hypre_BoxArrayArray *recv_rboxes,
                                 NALU_HYPRE_Int boxes_match, nalu_hypre_CommInfo **comm_info_ptr );
NALU_HYPRE_Int nalu_hypre_CommInfoSetTransforms ( nalu_hypre_CommInfo *comm_info, NALU_HYPRE_Int num_transforms,
                                        nalu_hypre_Index *coords, nalu_hypre_Index *dirs, NALU_HYPRE_Int **send_transforms, NALU_HYPRE_Int **recv_transforms );
NALU_HYPRE_Int nalu_hypre_CommInfoGetTransforms ( nalu_hypre_CommInfo *comm_info, NALU_HYPRE_Int *num_transforms,
                                        nalu_hypre_Index **coords, nalu_hypre_Index **dirs );
NALU_HYPRE_Int nalu_hypre_CommInfoProjectSend ( nalu_hypre_CommInfo *comm_info, nalu_hypre_Index index,
                                      nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_CommInfoProjectRecv ( nalu_hypre_CommInfo *comm_info, nalu_hypre_Index index,
                                      nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_CommInfoDestroy ( nalu_hypre_CommInfo *comm_info );
NALU_HYPRE_Int nalu_hypre_CreateCommInfoFromStencil ( nalu_hypre_StructGrid *grid, nalu_hypre_StructStencil *stencil,
                                            nalu_hypre_CommInfo **comm_info_ptr );
NALU_HYPRE_Int nalu_hypre_CreateCommInfoFromNumGhost ( nalu_hypre_StructGrid *grid, NALU_HYPRE_Int *num_ghost,
                                             nalu_hypre_CommInfo **comm_info_ptr );
NALU_HYPRE_Int nalu_hypre_CreateCommInfoFromGrids ( nalu_hypre_StructGrid *from_grid, nalu_hypre_StructGrid *to_grid,
                                          nalu_hypre_CommInfo **comm_info_ptr );

/* computation.c */
NALU_HYPRE_Int nalu_hypre_ComputeInfoCreate ( nalu_hypre_CommInfo *comm_info, nalu_hypre_BoxArrayArray *indt_boxes,
                                    nalu_hypre_BoxArrayArray *dept_boxes, nalu_hypre_ComputeInfo **compute_info_ptr );
NALU_HYPRE_Int nalu_hypre_ComputeInfoProjectSend ( nalu_hypre_ComputeInfo *compute_info, nalu_hypre_Index index,
                                         nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_ComputeInfoProjectRecv ( nalu_hypre_ComputeInfo *compute_info, nalu_hypre_Index index,
                                         nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_ComputeInfoProjectComp ( nalu_hypre_ComputeInfo *compute_info, nalu_hypre_Index index,
                                         nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_ComputeInfoDestroy ( nalu_hypre_ComputeInfo *compute_info );
NALU_HYPRE_Int nalu_hypre_CreateComputeInfo ( nalu_hypre_StructGrid *grid, nalu_hypre_StructStencil *stencil,
                                    nalu_hypre_ComputeInfo **compute_info_ptr );
NALU_HYPRE_Int nalu_hypre_ComputePkgCreate ( nalu_hypre_ComputeInfo *compute_info, nalu_hypre_BoxArray *data_space,
                                   NALU_HYPRE_Int num_values, nalu_hypre_StructGrid *grid, nalu_hypre_ComputePkg **compute_pkg_ptr );
NALU_HYPRE_Int nalu_hypre_ComputePkgDestroy ( nalu_hypre_ComputePkg *compute_pkg );
NALU_HYPRE_Int nalu_hypre_InitializeIndtComputations ( nalu_hypre_ComputePkg *compute_pkg, NALU_HYPRE_Complex *data,
                                             nalu_hypre_CommHandle **comm_handle_ptr );
NALU_HYPRE_Int nalu_hypre_FinalizeIndtComputations ( nalu_hypre_CommHandle *comm_handle );

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
NALU_HYPRE_Int nalu_hypre_StructVectorPrintData ( FILE *file, nalu_hypre_StructVector *vector, NALU_HYPRE_Int all );
NALU_HYPRE_Int nalu_hypre_StructVectorReadData ( FILE *file, nalu_hypre_StructVector *vector );
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
NALU_HYPRE_Int nalu_hypre_ProjectBox ( nalu_hypre_Box *box, nalu_hypre_Index index, nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_ProjectBoxArray ( nalu_hypre_BoxArray *box_array, nalu_hypre_Index index,
                                  nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_ProjectBoxArrayArray ( nalu_hypre_BoxArrayArray *box_array_array, nalu_hypre_Index index,
                                       nalu_hypre_Index stride );

/* struct_axpy.c */
NALU_HYPRE_Int nalu_hypre_StructAxpy ( NALU_HYPRE_Complex alpha, nalu_hypre_StructVector *x, nalu_hypre_StructVector *y );

/* struct_communication.c */
NALU_HYPRE_Int nalu_hypre_CommPkgCreate ( nalu_hypre_CommInfo *comm_info, nalu_hypre_BoxArray *send_data_space,
                                nalu_hypre_BoxArray *recv_data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Int **orders, NALU_HYPRE_Int reverse,
                                MPI_Comm comm, nalu_hypre_CommPkg **comm_pkg_ptr );
NALU_HYPRE_Int nalu_hypre_CommTypeSetEntries ( nalu_hypre_CommType *comm_type, NALU_HYPRE_Int *boxnums,
                                     nalu_hypre_Box *boxes, nalu_hypre_Index stride, nalu_hypre_Index coord, nalu_hypre_Index dir, NALU_HYPRE_Int *order,
                                     nalu_hypre_BoxArray *data_space, NALU_HYPRE_Int *data_offsets );
NALU_HYPRE_Int nalu_hypre_CommTypeSetEntry ( nalu_hypre_Box *box, nalu_hypre_Index stride, nalu_hypre_Index coord,
                                   nalu_hypre_Index dir, NALU_HYPRE_Int *order, nalu_hypre_Box *data_box, NALU_HYPRE_Int data_box_offset,
                                   nalu_hypre_CommEntryType *comm_entry );
NALU_HYPRE_Int nalu_hypre_InitializeCommunication ( nalu_hypre_CommPkg *comm_pkg, NALU_HYPRE_Complex *send_data,
                                          NALU_HYPRE_Complex *recv_data, NALU_HYPRE_Int action, NALU_HYPRE_Int tag, nalu_hypre_CommHandle **comm_handle_ptr );
NALU_HYPRE_Int nalu_hypre_FinalizeCommunication ( nalu_hypre_CommHandle *comm_handle );
NALU_HYPRE_Int nalu_hypre_ExchangeLocalData ( nalu_hypre_CommPkg *comm_pkg, NALU_HYPRE_Complex *send_data,
                                    NALU_HYPRE_Complex *recv_data, NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_CommPkgDestroy ( nalu_hypre_CommPkg *comm_pkg );

/* struct_copy.c */
NALU_HYPRE_Int nalu_hypre_StructCopy ( nalu_hypre_StructVector *x, nalu_hypre_StructVector *y );
NALU_HYPRE_Int nalu_hypre_StructPartialCopy ( nalu_hypre_StructVector *x, nalu_hypre_StructVector *y,
                                    nalu_hypre_BoxArrayArray *array_boxes );

/* struct_grid.c */
NALU_HYPRE_Int nalu_hypre_StructGridCreate ( MPI_Comm comm, NALU_HYPRE_Int dim, nalu_hypre_StructGrid **grid_ptr );
NALU_HYPRE_Int nalu_hypre_StructGridRef ( nalu_hypre_StructGrid *grid, nalu_hypre_StructGrid **grid_ref );
NALU_HYPRE_Int nalu_hypre_StructGridDestroy ( nalu_hypre_StructGrid *grid );
NALU_HYPRE_Int nalu_hypre_StructGridSetPeriodic ( nalu_hypre_StructGrid *grid, nalu_hypre_Index periodic );
NALU_HYPRE_Int nalu_hypre_StructGridSetExtents ( nalu_hypre_StructGrid *grid, nalu_hypre_Index ilower,
                                       nalu_hypre_Index iupper );
NALU_HYPRE_Int nalu_hypre_StructGridSetBoxes ( nalu_hypre_StructGrid *grid, nalu_hypre_BoxArray *boxes );
NALU_HYPRE_Int nalu_hypre_StructGridSetBoundingBox ( nalu_hypre_StructGrid *grid, nalu_hypre_Box *new_bb );
NALU_HYPRE_Int nalu_hypre_StructGridSetIDs ( nalu_hypre_StructGrid *grid, NALU_HYPRE_Int *ids );
NALU_HYPRE_Int nalu_hypre_StructGridSetBoxManager ( nalu_hypre_StructGrid *grid, nalu_hypre_BoxManager *boxman );
NALU_HYPRE_Int nalu_hypre_StructGridSetMaxDistance ( nalu_hypre_StructGrid *grid, nalu_hypre_Index dist );
NALU_HYPRE_Int nalu_hypre_StructGridAssemble ( nalu_hypre_StructGrid *grid );
NALU_HYPRE_Int nalu_hypre_GatherAllBoxes ( MPI_Comm comm, nalu_hypre_BoxArray *boxes, NALU_HYPRE_Int dim,
                                 nalu_hypre_BoxArray **all_boxes_ptr, NALU_HYPRE_Int **all_procs_ptr, NALU_HYPRE_Int *first_local_ptr );
NALU_HYPRE_Int nalu_hypre_ComputeBoxnums ( nalu_hypre_BoxArray *boxes, NALU_HYPRE_Int *procs, NALU_HYPRE_Int **boxnums_ptr );
NALU_HYPRE_Int nalu_hypre_StructGridPrint ( FILE *file, nalu_hypre_StructGrid *grid );
NALU_HYPRE_Int nalu_hypre_StructGridRead ( MPI_Comm comm, FILE *file, nalu_hypre_StructGrid **grid_ptr );
NALU_HYPRE_Int nalu_hypre_StructGridSetNumGhost ( nalu_hypre_StructGrid *grid, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int nalu_hypre_StructGridGetMaxBoxSize ( nalu_hypre_StructGrid *grid );
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int nalu_hypre_StructGridSetDataLocation( NALU_HYPRE_StructGrid grid,
                                           NALU_HYPRE_MemoryLocation data_location );
#endif
/* struct_innerprod.c */
NALU_HYPRE_Real nalu_hypre_StructInnerProd ( nalu_hypre_StructVector *x, nalu_hypre_StructVector *y );

/* struct_io.c */
NALU_HYPRE_Int nalu_hypre_PrintBoxArrayData ( FILE *file, nalu_hypre_BoxArray *box_array,
                                    nalu_hypre_BoxArray *data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Int dim, NALU_HYPRE_Complex *data );
NALU_HYPRE_Int nalu_hypre_PrintCCVDBoxArrayData ( FILE *file, nalu_hypre_BoxArray *box_array,
                                        nalu_hypre_BoxArray *data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Int center_rank, NALU_HYPRE_Int stencil_size,
                                        NALU_HYPRE_Int *symm_elements, NALU_HYPRE_Int dim, NALU_HYPRE_Complex *data );
NALU_HYPRE_Int nalu_hypre_PrintCCBoxArrayData ( FILE *file, nalu_hypre_BoxArray *box_array,
                                      nalu_hypre_BoxArray *data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Complex *data );
NALU_HYPRE_Int nalu_hypre_ReadBoxArrayData ( FILE *file, nalu_hypre_BoxArray *box_array,
                                   nalu_hypre_BoxArray *data_space, NALU_HYPRE_Int num_values, NALU_HYPRE_Int dim, NALU_HYPRE_Complex *data );
NALU_HYPRE_Int nalu_hypre_ReadBoxArrayData_CC ( FILE *file, nalu_hypre_BoxArray *box_array,
                                      nalu_hypre_BoxArray *data_space, NALU_HYPRE_Int stencil_size, NALU_HYPRE_Int real_stencil_size,
                                      NALU_HYPRE_Int constant_coefficient, NALU_HYPRE_Int dim, NALU_HYPRE_Complex *data );

/* struct_matrix.c */
NALU_HYPRE_Complex *nalu_hypre_StructMatrixExtractPointerByIndex ( nalu_hypre_StructMatrix *matrix, NALU_HYPRE_Int b,
                                                         nalu_hypre_Index index );
nalu_hypre_StructMatrix *nalu_hypre_StructMatrixCreate ( MPI_Comm comm, nalu_hypre_StructGrid *grid,
                                               nalu_hypre_StructStencil *user_stencil );
nalu_hypre_StructMatrix *nalu_hypre_StructMatrixRef ( nalu_hypre_StructMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_StructMatrixDestroy ( nalu_hypre_StructMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_StructMatrixInitializeShell ( nalu_hypre_StructMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_StructMatrixInitializeData ( nalu_hypre_StructMatrix *matrix, NALU_HYPRE_Complex *data,
                                             NALU_HYPRE_Complex *data_const);
NALU_HYPRE_Int nalu_hypre_StructMatrixInitialize ( nalu_hypre_StructMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_StructMatrixSetValues ( nalu_hypre_StructMatrix *matrix, nalu_hypre_Index grid_index,
                                        NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Complex *values, NALU_HYPRE_Int action,
                                        NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int nalu_hypre_StructMatrixSetBoxValues ( nalu_hypre_StructMatrix *matrix, nalu_hypre_Box *set_box,
                                           nalu_hypre_Box *value_box, NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices,
                                           NALU_HYPRE_Complex *values, NALU_HYPRE_Int action, NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int nalu_hypre_StructMatrixSetConstantValues ( nalu_hypre_StructMatrix *matrix,
                                                NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Complex *values,
                                                NALU_HYPRE_Int action );
NALU_HYPRE_Int nalu_hypre_StructMatrixClearValues ( nalu_hypre_StructMatrix *matrix, nalu_hypre_Index grid_index,
                                          NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int nalu_hypre_StructMatrixClearBoxValues ( nalu_hypre_StructMatrix *matrix, nalu_hypre_Box *clear_box,
                                             NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices, NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int nalu_hypre_StructMatrixAssemble ( nalu_hypre_StructMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_StructMatrixSetNumGhost ( nalu_hypre_StructMatrix *matrix, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int nalu_hypre_StructMatrixSetConstantCoefficient ( nalu_hypre_StructMatrix *matrix,
                                                     NALU_HYPRE_Int constant_coefficient );
NALU_HYPRE_Int nalu_hypre_StructMatrixSetConstantEntries ( nalu_hypre_StructMatrix *matrix, NALU_HYPRE_Int nentries,
                                                 NALU_HYPRE_Int *entries );
NALU_HYPRE_Int nalu_hypre_StructMatrixClearGhostValues ( nalu_hypre_StructMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_StructMatrixPrintData ( FILE *file, nalu_hypre_StructMatrix *matrix, NALU_HYPRE_Int all );
NALU_HYPRE_Int nalu_hypre_StructMatrixReadData ( FILE *file, nalu_hypre_StructMatrix *matrix );
NALU_HYPRE_Int nalu_hypre_StructMatrixPrint ( const char *filename, nalu_hypre_StructMatrix *matrix,
                                    NALU_HYPRE_Int all );
nalu_hypre_StructMatrix *nalu_hypre_StructMatrixRead ( MPI_Comm comm, const char *filename,
                                             NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int nalu_hypre_StructMatrixMigrate ( nalu_hypre_StructMatrix *from_matrix,
                                      nalu_hypre_StructMatrix *to_matrix );
NALU_HYPRE_Int nalu_hypre_StructMatrixClearBoundary( nalu_hypre_StructMatrix *matrix);

/* struct_matrix_mask.c */
nalu_hypre_StructMatrix *nalu_hypre_StructMatrixCreateMask ( nalu_hypre_StructMatrix *matrix,
                                                   NALU_HYPRE_Int num_stencil_indices, NALU_HYPRE_Int *stencil_indices );

/* struct_matvec.c */
void *nalu_hypre_StructMatvecCreate ( void );
NALU_HYPRE_Int nalu_hypre_StructMatvecSetup ( void *matvec_vdata, nalu_hypre_StructMatrix *A,
                                    nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_StructMatvecCompute ( void *matvec_vdata, NALU_HYPRE_Complex alpha,
                                      nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *x, NALU_HYPRE_Complex beta, nalu_hypre_StructVector *y );
NALU_HYPRE_Int nalu_hypre_StructMatvecCC0 ( NALU_HYPRE_Complex alpha, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *x,
                                  nalu_hypre_StructVector *y, nalu_hypre_BoxArrayArray *compute_box_aa, nalu_hypre_IndexRef stride );
NALU_HYPRE_Int nalu_hypre_StructMatvecCC1 ( NALU_HYPRE_Complex alpha, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *x,
                                  nalu_hypre_StructVector *y, nalu_hypre_BoxArrayArray *compute_box_aa, nalu_hypre_IndexRef stride );
NALU_HYPRE_Int nalu_hypre_StructMatvecCC2 ( NALU_HYPRE_Complex alpha, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *x,
                                  nalu_hypre_StructVector *y, nalu_hypre_BoxArrayArray *compute_box_aa, nalu_hypre_IndexRef stride );
NALU_HYPRE_Int nalu_hypre_StructMatvecDestroy ( void *matvec_vdata );
NALU_HYPRE_Int nalu_hypre_StructMatvec ( NALU_HYPRE_Complex alpha, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *x,
                               NALU_HYPRE_Complex beta, nalu_hypre_StructVector *y );

/* struct_scale.c */
NALU_HYPRE_Int nalu_hypre_StructScale ( NALU_HYPRE_Complex alpha, nalu_hypre_StructVector *y );

/* struct_stencil.c */
nalu_hypre_StructStencil *nalu_hypre_StructStencilCreate ( NALU_HYPRE_Int dim, NALU_HYPRE_Int size,
                                                 nalu_hypre_Index *shape );
nalu_hypre_StructStencil *nalu_hypre_StructStencilRef ( nalu_hypre_StructStencil *stencil );
NALU_HYPRE_Int nalu_hypre_StructStencilDestroy ( nalu_hypre_StructStencil *stencil );
NALU_HYPRE_Int nalu_hypre_StructStencilElementRank ( nalu_hypre_StructStencil *stencil,
                                           nalu_hypre_Index stencil_element );
NALU_HYPRE_Int nalu_hypre_StructStencilSymmetrize ( nalu_hypre_StructStencil *stencil,
                                          nalu_hypre_StructStencil **symm_stencil_ptr, NALU_HYPRE_Int **symm_elements_ptr );

/* struct_vector.c */
nalu_hypre_StructVector *nalu_hypre_StructVectorCreate ( MPI_Comm comm, nalu_hypre_StructGrid *grid );
nalu_hypre_StructVector *nalu_hypre_StructVectorRef ( nalu_hypre_StructVector *vector );
NALU_HYPRE_Int nalu_hypre_StructVectorDestroy ( nalu_hypre_StructVector *vector );
NALU_HYPRE_Int nalu_hypre_StructVectorInitializeShell ( nalu_hypre_StructVector *vector );
NALU_HYPRE_Int nalu_hypre_StructVectorInitializeData ( nalu_hypre_StructVector *vector, NALU_HYPRE_Complex *data);
NALU_HYPRE_Int nalu_hypre_StructVectorInitialize ( nalu_hypre_StructVector *vector );
NALU_HYPRE_Int nalu_hypre_StructVectorSetValues ( nalu_hypre_StructVector *vector, nalu_hypre_Index grid_index,
                                        NALU_HYPRE_Complex *values, NALU_HYPRE_Int action, NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int nalu_hypre_StructVectorSetBoxValues ( nalu_hypre_StructVector *vector, nalu_hypre_Box *set_box,
                                           nalu_hypre_Box *value_box, NALU_HYPRE_Complex *values, NALU_HYPRE_Int action, NALU_HYPRE_Int boxnum,
                                           NALU_HYPRE_Int outside );
NALU_HYPRE_Int nalu_hypre_StructVectorClearValues ( nalu_hypre_StructVector *vector, nalu_hypre_Index grid_index,
                                          NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int nalu_hypre_StructVectorClearBoxValues ( nalu_hypre_StructVector *vector, nalu_hypre_Box *clear_box,
                                             NALU_HYPRE_Int boxnum, NALU_HYPRE_Int outside );
NALU_HYPRE_Int nalu_hypre_StructVectorClearAllValues ( nalu_hypre_StructVector *vector );
NALU_HYPRE_Int nalu_hypre_StructVectorSetNumGhost ( nalu_hypre_StructVector *vector, NALU_HYPRE_Int *num_ghost );
NALU_HYPRE_Int nalu_hypre_StructVectorSetDataSize(nalu_hypre_StructVector *vector, NALU_HYPRE_Int *data_size,
                                        NALU_HYPRE_Int *data_host_size);
NALU_HYPRE_Int nalu_hypre_StructVectorAssemble ( nalu_hypre_StructVector *vector );
NALU_HYPRE_Int nalu_hypre_StructVectorCopy ( nalu_hypre_StructVector *x, nalu_hypre_StructVector *y );
NALU_HYPRE_Int nalu_hypre_StructVectorSetConstantValues ( nalu_hypre_StructVector *vector, NALU_HYPRE_Complex values );
NALU_HYPRE_Int nalu_hypre_StructVectorSetFunctionValues ( nalu_hypre_StructVector *vector,
                                                NALU_HYPRE_Complex (*fcn )( NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Int ));
NALU_HYPRE_Int nalu_hypre_StructVectorClearGhostValues ( nalu_hypre_StructVector *vector );
NALU_HYPRE_Int nalu_hypre_StructVectorClearBoundGhostValues ( nalu_hypre_StructVector *vector, NALU_HYPRE_Int force );
NALU_HYPRE_Int nalu_hypre_StructVectorScaleValues ( nalu_hypre_StructVector *vector, NALU_HYPRE_Complex factor );
nalu_hypre_CommPkg *nalu_hypre_StructVectorGetMigrateCommPkg ( nalu_hypre_StructVector *from_vector,
                                                     nalu_hypre_StructVector *to_vector );
NALU_HYPRE_Int nalu_hypre_StructVectorMigrate ( nalu_hypre_CommPkg *comm_pkg, nalu_hypre_StructVector *from_vector,
                                      nalu_hypre_StructVector *to_vector );
NALU_HYPRE_Int nalu_hypre_StructVectorPrint ( const char *filename, nalu_hypre_StructVector *vector,
                                    NALU_HYPRE_Int all );
nalu_hypre_StructVector *nalu_hypre_StructVectorRead ( MPI_Comm comm, const char *filename,
                                             NALU_HYPRE_Int *num_ghost );
nalu_hypre_StructVector *nalu_hypre_StructVectorClone ( nalu_hypre_StructVector *vector );
