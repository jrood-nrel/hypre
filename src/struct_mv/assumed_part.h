/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the struct assumed partition
 *
 *****************************************************************************/

#ifndef nalu_hypre_ASSUMED_PART_HEADER
#define nalu_hypre_ASSUMED_PART_HEADER

typedef struct
{
   /* the entries will be the same for all procs */
   NALU_HYPRE_Int           ndim;             /* number of dimensions */
   nalu_hypre_BoxArray     *regions;          /* areas of the grid with boxes */
   NALU_HYPRE_Int           num_regions;      /* how many regions */
   NALU_HYPRE_Int          *proc_partitions;  /* proc ids assigned to each region
                                            (this is size num_regions +1) */
   nalu_hypre_Index        *divisions;        /* number of proc divisions in each
                                            direction for each region */
   /* these entries are specific to each proc */
   nalu_hypre_BoxArray     *my_partition;        /* my portion of grid (at most 2) */
   nalu_hypre_BoxArray     *my_partition_boxes;  /* boxes in my portion */
   NALU_HYPRE_Int          *my_partition_proc_ids;
   NALU_HYPRE_Int          *my_partition_boxnums;
   NALU_HYPRE_Int           my_partition_ids_size;
   NALU_HYPRE_Int           my_partition_ids_alloc;
   NALU_HYPRE_Int           my_partition_num_distinct_procs;

} nalu_hypre_StructAssumedPart;


/*Accessor macros */

#define nalu_hypre_StructAssumedPartNDim(apart) ((apart)->ndim)
#define nalu_hypre_StructAssumedPartRegions(apart) ((apart)->regions)
#define nalu_hypre_StructAssumedPartNumRegions(apart) ((apart)->num_regions)
#define nalu_hypre_StructAssumedPartDivisions(apart) ((apart)->divisions)
#define nalu_hypre_StructAssumedPartDivision(apart, i) ((apart)->divisions[i])
#define nalu_hypre_StructAssumedPartProcPartitions(apart) ((apart)->proc_partitions)
#define nalu_hypre_StructAssumedPartProcPartition(apart, i) ((apart)->proc_partitions[i])
#define nalu_hypre_StructAssumedPartMyPartition(apart) ((apart)->my_partition)
#define nalu_hypre_StructAssumedPartMyPartitionBoxes(apart) ((apart)->my_partition_boxes)
#define nalu_hypre_StructAssumedPartMyPartitionProcIds(apart) ((apart)->my_partition_proc_ids)
#define nalu_hypre_StructAssumedPartMyPartitionIdsSize(apart) ((apart)->my_partition_ids_size)
#define nalu_hypre_StructAssumedPartMyPartitionIdsAlloc(apart) ((apart)->my_partition_ids_alloc)
#define nalu_hypre_StructAssumedPartMyPartitionNumDistinctProcs(apart) ((apart)->my_partition_num_distinct_procs)
#define nalu_hypre_StructAssumedPartMyPartitionBoxnums(apart) ((apart)->my_partition_boxnums)

#define nalu_hypre_StructAssumedPartMyPartitionProcId(apart, i) ((apart)->my_partition_proc_ids[i])
#define nalu_hypre_StructAssumedPartMyPartitionBoxnum(apart, i) ((apart)->my_partition_boxnums[i])
#endif
