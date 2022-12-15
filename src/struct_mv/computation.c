/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ComputeInfoCreate( nalu_hypre_CommInfo       *comm_info,
                         nalu_hypre_BoxArrayArray  *indt_boxes,
                         nalu_hypre_BoxArrayArray  *dept_boxes,
                         nalu_hypre_ComputeInfo   **compute_info_ptr )
{
   nalu_hypre_ComputeInfo  *compute_info;

   compute_info = nalu_hypre_TAlloc(nalu_hypre_ComputeInfo,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ComputeInfoCommInfo(compute_info)  = comm_info;
   nalu_hypre_ComputeInfoIndtBoxes(compute_info) = indt_boxes;
   nalu_hypre_ComputeInfoDeptBoxes(compute_info) = dept_boxes;

   nalu_hypre_SetIndex(nalu_hypre_ComputeInfoStride(compute_info), 1);

   *compute_info_ptr = compute_info;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ComputeInfoProjectSend( nalu_hypre_ComputeInfo  *compute_info,
                              nalu_hypre_Index         index,
                              nalu_hypre_Index         stride )
{
   nalu_hypre_CommInfoProjectSend(nalu_hypre_ComputeInfoCommInfo(compute_info),
                             index, stride);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ComputeInfoProjectRecv( nalu_hypre_ComputeInfo  *compute_info,
                              nalu_hypre_Index         index,
                              nalu_hypre_Index         stride )
{
   nalu_hypre_CommInfoProjectRecv(nalu_hypre_ComputeInfoCommInfo(compute_info),
                             index, stride);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ComputeInfoProjectComp( nalu_hypre_ComputeInfo  *compute_info,
                              nalu_hypre_Index         index,
                              nalu_hypre_Index         stride )
{
   nalu_hypre_ProjectBoxArrayArray(nalu_hypre_ComputeInfoIndtBoxes(compute_info),
                              index, stride);
   nalu_hypre_ProjectBoxArrayArray(nalu_hypre_ComputeInfoDeptBoxes(compute_info),
                              index, stride);
   nalu_hypre_CopyIndex(stride, nalu_hypre_ComputeInfoStride(compute_info));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ComputeInfoDestroy( nalu_hypre_ComputeInfo  *compute_info )
{
   nalu_hypre_TFree(compute_info, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return descriptions of communications and computations patterns for
 * a given grid-stencil computation.  If HYPRE\_OVERLAP\_COMM\_COMP is
 * defined, then the patterns are computed to allow for overlapping
 * communications and computations.  The default is no overlap.
 *
 * Note: This routine assumes that the grid boxes do not overlap.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CreateComputeInfo( nalu_hypre_StructGrid      *grid,
                         nalu_hypre_StructStencil   *stencil,
                         nalu_hypre_ComputeInfo    **compute_info_ptr )
{
   NALU_HYPRE_Int                ndim = nalu_hypre_StructGridNDim(grid);
   nalu_hypre_CommInfo          *comm_info;
   nalu_hypre_BoxArrayArray     *indt_boxes;
   nalu_hypre_BoxArrayArray     *dept_boxes;

   nalu_hypre_BoxArray          *boxes;

   nalu_hypre_BoxArray          *cbox_array;
   nalu_hypre_Box               *cbox;

   NALU_HYPRE_Int                i;

#ifdef NALU_HYPRE_OVERLAP_COMM_COMP
   nalu_hypre_Box               *rembox;
   nalu_hypre_Index             *stencil_shape;
   nalu_hypre_Index              lborder, rborder;
   NALU_HYPRE_Int                cbox_array_size;
   NALU_HYPRE_Int                s, d;
#endif

   /*------------------------------------------------------
    * Extract needed grid info
    *------------------------------------------------------*/

   boxes = nalu_hypre_StructGridBoxes(grid);

   /*------------------------------------------------------
    * Get communication info
    *------------------------------------------------------*/

   nalu_hypre_CreateCommInfoFromStencil(grid, stencil, &comm_info);

#ifdef NALU_HYPRE_OVERLAP_COMM_COMP

   /*------------------------------------------------------
    * Compute border info
    *------------------------------------------------------*/

   nalu_hypre_SetIndex(lborder, 0);
   nalu_hypre_SetIndex(rborder, 0);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   for (s = 0; s < nalu_hypre_StructStencilSize(stencil); s++)
   {
      for (d = 0; d < ndim; d++)
      {
         i = nalu_hypre_IndexD(stencil_shape[s], d);
         if (i < 0)
         {
            lborder[d] = nalu_hypre_max(lborder[d], -i);
         }
         else if (i > 0)
         {
            rborder[d] = nalu_hypre_max(rborder[d], i);
         }
      }
   }

   /*------------------------------------------------------
    * Set up the dependent boxes
    *------------------------------------------------------*/

   dept_boxes = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(boxes), ndim);

   rembox = nalu_hypre_BoxCreate(nalu_hypre_StructGridNDim(grid));
   nalu_hypre_ForBoxI(i, boxes)
   {
      cbox_array = nalu_hypre_BoxArrayArrayBoxArray(dept_boxes, i);
      nalu_hypre_BoxArraySetSize(cbox_array, 2 * ndim);

      nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(boxes, i), rembox);
      cbox_array_size = 0;
      for (d = 0; d < ndim; d++)
      {
         if ( (nalu_hypre_BoxVolume(rembox)) && lborder[d] )
         {
            cbox = nalu_hypre_BoxArrayBox(cbox_array, cbox_array_size);
            nalu_hypre_CopyBox(rembox, cbox);
            nalu_hypre_BoxIMaxD(cbox, d) =
               nalu_hypre_BoxIMinD(cbox, d) + lborder[d] - 1;
            nalu_hypre_BoxIMinD(rembox, d) =
               nalu_hypre_BoxIMinD(cbox, d) + lborder[d];
            cbox_array_size++;
         }
         if ( (nalu_hypre_BoxVolume(rembox)) && rborder[d] )
         {
            cbox = nalu_hypre_BoxArrayBox(cbox_array, cbox_array_size);
            nalu_hypre_CopyBox(rembox, cbox);
            nalu_hypre_BoxIMinD(cbox, d) =
               nalu_hypre_BoxIMaxD(cbox, d) - rborder[d] + 1;
            nalu_hypre_BoxIMaxD(rembox, d) =
               nalu_hypre_BoxIMaxD(cbox, d) - rborder[d];
            cbox_array_size++;
         }
      }
      nalu_hypre_BoxArraySetSize(cbox_array, cbox_array_size);
   }
   nalu_hypre_BoxDestroy(rembox);

   /*------------------------------------------------------
    * Set up the independent boxes
    *------------------------------------------------------*/

   indt_boxes = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(boxes), ndim);

   nalu_hypre_ForBoxI(i, boxes)
   {
      cbox_array = nalu_hypre_BoxArrayArrayBoxArray(indt_boxes, i);
      nalu_hypre_BoxArraySetSize(cbox_array, 1);
      cbox = nalu_hypre_BoxArrayBox(cbox_array, 0);
      nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(boxes, i), cbox);

      for (d = 0; d < ndim; d++)
      {
         if ( lborder[d] )
         {
            nalu_hypre_BoxIMinD(cbox, d) += lborder[d];
         }
         if ( rborder[d] )
         {
            nalu_hypre_BoxIMaxD(cbox, d) -= rborder[d];
         }
      }
   }

#else

   /*------------------------------------------------------
    * Set up the independent boxes
    *------------------------------------------------------*/

   indt_boxes = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(boxes), ndim);

   /*------------------------------------------------------
    * Set up the dependent boxes
    *------------------------------------------------------*/

   dept_boxes = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(boxes), ndim);

   nalu_hypre_ForBoxI(i, boxes)
   {
      cbox_array = nalu_hypre_BoxArrayArrayBoxArray(dept_boxes, i);
      nalu_hypre_BoxArraySetSize(cbox_array, 1);
      cbox = nalu_hypre_BoxArrayBox(cbox_array, 0);
      nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(boxes, i), cbox);
   }

#endif

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   nalu_hypre_ComputeInfoCreate(comm_info, indt_boxes, dept_boxes,
                           compute_info_ptr);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Create a computation package from a grid-based description of a
 * communication-computation pattern.
 *
 * Note: The input boxes and processes are destroyed.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ComputePkgCreate( nalu_hypre_ComputeInfo     *compute_info,
                        nalu_hypre_BoxArray        *data_space,
                        NALU_HYPRE_Int              num_values,
                        nalu_hypre_StructGrid      *grid,
                        nalu_hypre_ComputePkg     **compute_pkg_ptr )
{
   nalu_hypre_ComputePkg  *compute_pkg;
   nalu_hypre_CommPkg     *comm_pkg;

   compute_pkg = nalu_hypre_CTAlloc(nalu_hypre_ComputePkg,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_CommPkgCreate(nalu_hypre_ComputeInfoCommInfo(compute_info),
                       data_space, data_space, num_values, NULL, 0,
                       nalu_hypre_StructGridComm(grid), &comm_pkg);
   nalu_hypre_CommInfoDestroy(nalu_hypre_ComputeInfoCommInfo(compute_info));
   nalu_hypre_ComputePkgCommPkg(compute_pkg) = comm_pkg;

   nalu_hypre_ComputePkgIndtBoxes(compute_pkg) =
      nalu_hypre_ComputeInfoIndtBoxes(compute_info);
   nalu_hypre_ComputePkgDeptBoxes(compute_pkg) =
      nalu_hypre_ComputeInfoDeptBoxes(compute_info);
   nalu_hypre_CopyIndex(nalu_hypre_ComputeInfoStride(compute_info),
                   nalu_hypre_ComputePkgStride(compute_pkg));

   nalu_hypre_StructGridRef(grid, &nalu_hypre_ComputePkgGrid(compute_pkg));
   nalu_hypre_ComputePkgDataSpace(compute_pkg) = data_space;
   nalu_hypre_ComputePkgNumValues(compute_pkg) = num_values;

   nalu_hypre_ComputeInfoDestroy(compute_info);

   *compute_pkg_ptr = compute_pkg;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Destroy a computation package.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ComputePkgDestroy( nalu_hypre_ComputePkg *compute_pkg )
{
   if (compute_pkg)
   {
      nalu_hypre_CommPkgDestroy(nalu_hypre_ComputePkgCommPkg(compute_pkg));

      nalu_hypre_BoxArrayArrayDestroy(nalu_hypre_ComputePkgIndtBoxes(compute_pkg));
      nalu_hypre_BoxArrayArrayDestroy(nalu_hypre_ComputePkgDeptBoxes(compute_pkg));

      nalu_hypre_StructGridDestroy(nalu_hypre_ComputePkgGrid(compute_pkg));

      nalu_hypre_TFree(compute_pkg, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Initialize a non-blocking communication exchange.  The independent
 * computations may be done after a call to this routine, to allow for
 * overlap of communications and computations.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_InitializeIndtComputations( nalu_hypre_ComputePkg  *compute_pkg,
                                  NALU_HYPRE_Complex     *data,
                                  nalu_hypre_CommHandle **comm_handle_ptr )
{
   nalu_hypre_CommPkg *comm_pkg = nalu_hypre_ComputePkgCommPkg(compute_pkg);

   nalu_hypre_InitializeCommunication(comm_pkg, data, data, 0, 0, comm_handle_ptr);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Finalize a communication exchange.  The dependent computations may
 * be done after a call to this routine.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FinalizeIndtComputations( nalu_hypre_CommHandle *comm_handle )
{
   nalu_hypre_FinalizeCommunication(comm_handle );

   return nalu_hypre_error_flag;
}
