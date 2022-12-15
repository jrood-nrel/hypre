/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NOTE: The following routines are currently only used as follows in hypre, and
 * also appear in '_nalu_hypre_struct_mv.h':
 *
 * nalu_hypre_BoxBoundaryG
 * struct_mv/box_boundary.c
 * struct_mv/struct_vector.c
 * sstruct_ls/maxwell_grad.c
 * sstruct_ls/maxwell_TV_setup.c
 *
 * nalu_hypre_BoxBoundaryDG
 * struct_mv/box_boundary.c
 * sstruct_ls/maxwell_grad.c
 * sstruct_ls/maxwell_PNedelec_bdy.c
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * Intersect a surface of 'box' with the physical boundary.  The surface is
 * given by (d,dir), where 'dir' is a direction (+-1) in dimension 'd'.
 *
 * The result will be returned in the box array 'boundary'.  Any boxes already
 * in 'boundary' will be overwritten.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxBoundaryIntersect( nalu_hypre_Box *box,
                            nalu_hypre_StructGrid *grid,
                            NALU_HYPRE_Int d,
                            NALU_HYPRE_Int dir,
                            nalu_hypre_BoxArray *boundary )
{
   NALU_HYPRE_Int           ndim = nalu_hypre_BoxNDim(box);
   nalu_hypre_BoxManager   *boxman;
   nalu_hypre_BoxManEntry **entries;
   nalu_hypre_BoxArray     *int_boxes, *tmp_boxes;
   nalu_hypre_Box          *bbox, *ibox;
   NALU_HYPRE_Int           nentries, i;

   /* set bbox to the box surface of interest */
   nalu_hypre_BoxArraySetSize(boundary, 1);
   bbox = nalu_hypre_BoxArrayBox(boundary, 0);
   nalu_hypre_CopyBox(box, bbox);
   if (dir > 0)
   {
      nalu_hypre_BoxIMinD(bbox, d) = nalu_hypre_BoxIMaxD(bbox, d);
   }
   else if (dir < 0)
   {
      nalu_hypre_BoxIMaxD(bbox, d) = nalu_hypre_BoxIMinD(bbox, d);
   }

   /* temporarily shift bbox in direction dir and intersect with the grid */
   nalu_hypre_BoxIMinD(bbox, d) += dir;
   nalu_hypre_BoxIMaxD(bbox, d) += dir;
   boxman = nalu_hypre_StructGridBoxMan(grid);
   nalu_hypre_BoxManIntersect(boxman, nalu_hypre_BoxIMin(bbox), nalu_hypre_BoxIMax(bbox),
                         &entries, &nentries);
   nalu_hypre_BoxIMinD(bbox, d) -= dir;
   nalu_hypre_BoxIMaxD(bbox, d) -= dir;

   /* shift intersected boxes in direction -dir and subtract from bbox */
   int_boxes  = nalu_hypre_BoxArrayCreate(nentries, ndim);
   tmp_boxes  = nalu_hypre_BoxArrayCreate(0, ndim);
   for (i = 0; i < nentries; i++)
   {
      ibox = nalu_hypre_BoxArrayBox(int_boxes, i);
      nalu_hypre_BoxManEntryGetExtents(
         entries[i], nalu_hypre_BoxIMin(ibox), nalu_hypre_BoxIMax(ibox));
      nalu_hypre_BoxIMinD(ibox, d) -= dir;
      nalu_hypre_BoxIMaxD(ibox, d) -= dir;
   }
   nalu_hypre_SubtractBoxArrays(boundary, int_boxes, tmp_boxes);

   nalu_hypre_BoxArrayDestroy(int_boxes);
   nalu_hypre_BoxArrayDestroy(tmp_boxes);
   nalu_hypre_TFree(entries, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Find the parts of the given box which lie on a (physical) boundary of grid g.
 * Stick them into the user-provided box array boundary.  Any input contents of
 * this box array will get overwritten.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxBoundaryG( nalu_hypre_Box *box,
                    nalu_hypre_StructGrid *g,
                    nalu_hypre_BoxArray *boundary )
{
   NALU_HYPRE_Int       ndim = nalu_hypre_BoxNDim(box);
   nalu_hypre_BoxArray *boundary_d;
   NALU_HYPRE_Int       d;

   boundary_d = nalu_hypre_BoxArrayCreate(0, ndim);
   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxBoundaryIntersect(box, g, d, -1, boundary_d);
      nalu_hypre_AppendBoxArray(boundary_d, boundary);
      nalu_hypre_BoxBoundaryIntersect(box, g, d,  1, boundary_d);
      nalu_hypre_AppendBoxArray(boundary_d, boundary);
   }
   nalu_hypre_BoxArrayDestroy(boundary_d);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Find the parts of the given box which lie on a (physical) boundary of grid g,
 * only in the (unsigned) direction of d (d=0,1,2).  Stick them into the
 * user-provided box arrays boundarym (minus direction) and boundaryp (plus
 * direction).  Any input contents of these box arrays will get overwritten.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxBoundaryDG( nalu_hypre_Box *box,
                     nalu_hypre_StructGrid *g,
                     nalu_hypre_BoxArray *boundarym,
                     nalu_hypre_BoxArray *boundaryp,
                     NALU_HYPRE_Int d )
{
   nalu_hypre_BoxBoundaryIntersect(box, g, d, -1, boundarym);
   nalu_hypre_BoxBoundaryIntersect(box, g, d,  1, boundaryp);

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * Intersect a surface of 'box' with the physical boundary.  A stencil element
 * indicates in which direction the surface should be determined.
 *
 * The result will be returned in the box array 'boundary'.  Any boxes already
 * in 'boundary' will be overwritten.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_GeneralBoxBoundaryIntersect( nalu_hypre_Box *box,
                                   nalu_hypre_StructGrid *grid,
                                   nalu_hypre_Index stencil_element,
                                   nalu_hypre_BoxArray *boundary )
{
   nalu_hypre_BoxManager   *boxman;
   nalu_hypre_BoxManEntry **entries;
   nalu_hypre_BoxArray     *int_boxes, *tmp_boxes;
   nalu_hypre_Box          *bbox, *ibox;
   NALU_HYPRE_Int           nentries, i, j;
   NALU_HYPRE_Int          *dd;
   NALU_HYPRE_Int           ndim;

   ndim = nalu_hypre_StructGridNDim(grid);
   dd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  ndim, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < ndim; i++)
   {
      dd[i] = nalu_hypre_IndexD(stencil_element, i);
   }

   /* set bbox to the box surface of interest */
   nalu_hypre_BoxArraySetSize(boundary, 1);
   bbox = nalu_hypre_BoxArrayBox(boundary, 0);
   nalu_hypre_CopyBox(box, bbox);

   /* temporarily shift bbox in direction dir and intersect with the grid */
   for (i = 0; i < ndim; i++)
   {
      nalu_hypre_BoxIMinD(bbox, i) += dd[i];
      nalu_hypre_BoxIMaxD(bbox, i) += dd[i];
   }

   boxman = nalu_hypre_StructGridBoxMan(grid);
   nalu_hypre_BoxManIntersect(boxman, nalu_hypre_BoxIMin(bbox), nalu_hypre_BoxIMax(bbox),
                         &entries, &nentries);
   for (i = 0; i < ndim; i++)
   {
      nalu_hypre_BoxIMinD(bbox, i) -= dd[i];
      nalu_hypre_BoxIMaxD(bbox, i) -= dd[i];
   }

   /* shift intersected boxes in direction -dir and subtract from bbox */
   int_boxes  = nalu_hypre_BoxArrayCreate(nentries, ndim);
   tmp_boxes  = nalu_hypre_BoxArrayCreate(0, ndim);
   for (i = 0; i < nentries; i++)
   {
      ibox = nalu_hypre_BoxArrayBox(int_boxes, i);
      nalu_hypre_BoxManEntryGetExtents(
         entries[i], nalu_hypre_BoxIMin(ibox), nalu_hypre_BoxIMax(ibox));
      for (j = 0; j < ndim; j++)
      {
         nalu_hypre_BoxIMinD(ibox, j) -= dd[j];
         nalu_hypre_BoxIMaxD(ibox, j) -= dd[j];
      }
   }
   nalu_hypre_SubtractBoxArrays(boundary, int_boxes, tmp_boxes);

   nalu_hypre_BoxArrayDestroy(int_boxes);
   nalu_hypre_BoxArrayDestroy(tmp_boxes);
   nalu_hypre_TFree(entries, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dd, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

