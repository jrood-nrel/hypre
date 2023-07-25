/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for computation
 *
 *****************************************************************************/

#ifndef nalu_hypre_COMPUTATION_HEADER
#define nalu_hypre_COMPUTATION_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_ComputeInfo:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_ComputeInfo_struct
{
   nalu_hypre_CommInfo        *comm_info;

   nalu_hypre_BoxArrayArray   *indt_boxes;
   nalu_hypre_BoxArrayArray   *dept_boxes;
   nalu_hypre_Index            stride;

} nalu_hypre_ComputeInfo;

/*--------------------------------------------------------------------------
 * nalu_hypre_ComputePkg:
 *   Structure containing information for doing computations.
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_ComputePkg_struct
{
   nalu_hypre_CommPkg         *comm_pkg;

   nalu_hypre_BoxArrayArray   *indt_boxes;
   nalu_hypre_BoxArrayArray   *dept_boxes;
   nalu_hypre_Index            stride;

   nalu_hypre_StructGrid      *grid;
   nalu_hypre_BoxArray        *data_space;
   NALU_HYPRE_Int              num_values;

} nalu_hypre_ComputePkg;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_ComputeInfo
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ComputeInfoCommInfo(info)     (info -> comm_info)
#define nalu_hypre_ComputeInfoIndtBoxes(info)    (info -> indt_boxes)
#define nalu_hypre_ComputeInfoDeptBoxes(info)    (info -> dept_boxes)
#define nalu_hypre_ComputeInfoStride(info)       (info -> stride)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_ComputePkg
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ComputePkgCommPkg(compute_pkg)      (compute_pkg -> comm_pkg)

#define nalu_hypre_ComputePkgIndtBoxes(compute_pkg)    (compute_pkg -> indt_boxes)
#define nalu_hypre_ComputePkgDeptBoxes(compute_pkg)    (compute_pkg -> dept_boxes)
#define nalu_hypre_ComputePkgStride(compute_pkg)       (compute_pkg -> stride)

#define nalu_hypre_ComputePkgGrid(compute_pkg)         (compute_pkg -> grid)
#define nalu_hypre_ComputePkgDataSpace(compute_pkg)    (compute_pkg -> data_space)
#define nalu_hypre_ComputePkgNumValues(compute_pkg)    (compute_pkg -> num_values)

#endif
