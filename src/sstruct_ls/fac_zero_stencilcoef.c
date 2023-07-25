/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "fac.h"

#define AbsStencilShape(stencil, abs_shape)                     \
   {                                                            \
      NALU_HYPRE_Int ii,jj,kk;                                       \
      ii = nalu_hypre_IndexX(stencil);                               \
      jj = nalu_hypre_IndexY(stencil);                               \
      kk = nalu_hypre_IndexZ(stencil);                               \
      abs_shape= nalu_hypre_abs(ii) + nalu_hypre_abs(jj) + nalu_hypre_abs(kk); \
   }

/*--------------------------------------------------------------------------
 * nalu_hypre_FacZeroCFSten: Zeroes the coarse stencil coefficients that reach
 * into an underlying coarsened refinement box.
 * Algo: For each cbox
 *       {
 *          1) refine cbox and expand by one in each direction
 *          2) boxman_intersect with the fboxman
 *                3) loop over intersection boxes to see if stencil
 *                   reaches over.
 *       }
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FacZeroCFSten( nalu_hypre_SStructPMatrix *Af,
                     nalu_hypre_SStructPMatrix *Ac,
                     nalu_hypre_SStructGrid    *grid,
                     NALU_HYPRE_Int             fine_part,
                     nalu_hypre_Index           rfactors )
{
   nalu_hypre_BoxManager      *fboxman;
   nalu_hypre_BoxManEntry    **boxman_entries;
   NALU_HYPRE_Int              nboxman_entries;

   nalu_hypre_SStructPGrid    *p_cgrid;

   nalu_hypre_Box              fgrid_box;
   nalu_hypre_StructGrid      *cgrid;
   nalu_hypre_BoxArray        *cgrid_boxes;
   nalu_hypre_Box             *cgrid_box;
   nalu_hypre_Box              scaled_box;

   nalu_hypre_Box             *shift_ibox;

   nalu_hypre_StructMatrix    *smatrix;

   nalu_hypre_StructStencil   *stencils;
   NALU_HYPRE_Int              stencil_size;

   nalu_hypre_Index            refine_factors, upper_shift;
   nalu_hypre_Index            stride;
   nalu_hypre_Index            stencil_shape;
   nalu_hypre_Index            zero_index, ilower, iupper;

   NALU_HYPRE_Int              nvars, var1, var2;
   NALU_HYPRE_Int              ndim;

   nalu_hypre_Box             *ac_dbox;
   NALU_HYPRE_Real            *ac_ptr;
   nalu_hypre_Index            loop_size;

   NALU_HYPRE_Int              ci, i, j;

   NALU_HYPRE_Int              abs_shape;

   NALU_HYPRE_Int              ierr = 0;

   p_cgrid  = nalu_hypre_SStructPMatrixPGrid(Ac);
   nvars    = nalu_hypre_SStructPMatrixNVars(Ac);
   ndim     = nalu_hypre_SStructPGridNDim(p_cgrid);

   nalu_hypre_BoxInit(&fgrid_box, ndim);
   nalu_hypre_BoxInit(&scaled_box, ndim);

   nalu_hypre_ClearIndex(zero_index);
   nalu_hypre_ClearIndex(stride);
   nalu_hypre_ClearIndex(upper_shift);
   for (i = 0; i < ndim; i++)
   {
      stride[i] = 1;
      upper_shift[i] = rfactors[i] - 1;
   }

   nalu_hypre_CopyIndex(rfactors, refine_factors);
   if (ndim < 3)
   {
      for (i = ndim; i < 3; i++)
      {
         refine_factors[i] = 1;
      }
   }

   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(Ac), var1);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

      fboxman = nalu_hypre_SStructGridBoxManager(grid, fine_part, var1);

      /*------------------------------------------------------------------
       * For each parent coarse box find all fboxes that may be connected
       * through a stencil entry- refine this box, expand it by one
       * in each direction, and boxman_intersect with fboxman
       *------------------------------------------------------------------*/
      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

         nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(cgrid_box), zero_index,
                                     refine_factors, nalu_hypre_BoxIMin(&scaled_box));
         nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(cgrid_box), upper_shift,
                                     refine_factors, nalu_hypre_BoxIMax(&scaled_box));

         nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&scaled_box), stride, 3,
                               nalu_hypre_BoxIMin(&scaled_box));
         nalu_hypre_AddIndexes(nalu_hypre_BoxIMax(&scaled_box), stride, 3,
                          nalu_hypre_BoxIMax(&scaled_box));

         nalu_hypre_BoxManIntersect(fboxman, nalu_hypre_BoxIMin(&scaled_box),
                               nalu_hypre_BoxIMax(&scaled_box), &boxman_entries,
                               &nboxman_entries);

         for (var2 = 0; var2 < nvars; var2++)
         {
            stencils =  nalu_hypre_SStructPMatrixSStencil(Ac, var1, var2);

            if (stencils != NULL)
            {
               stencil_size = nalu_hypre_StructStencilSize(stencils);
               smatrix     = nalu_hypre_SStructPMatrixSMatrix(Ac, var1, var2);
               ac_dbox     = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(smatrix),
                                               ci);

               /*---------------------------------------------------------
                * Find the stencil coefficients that must be zeroed off.
                * Loop over all possible boxes.
                *---------------------------------------------------------*/
               for (i = 0; i < stencil_size; i++)
               {
                  nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i),
                                  stencil_shape);
                  AbsStencilShape(stencil_shape, abs_shape);

                  if (abs_shape)   /* non-centre stencils are zeroed */
                  {
                     /* look for connecting fboxes that must be zeroed. */
                     for (j = 0; j < nboxman_entries; j++)
                     {
                        nalu_hypre_BoxManEntryGetExtents(boxman_entries[j], ilower, iupper);
                        nalu_hypre_BoxSetExtents(&fgrid_box, ilower, iupper);

                        shift_ibox = nalu_hypre_CF_StenBox(&fgrid_box, cgrid_box, stencil_shape,
                                                      refine_factors, ndim);

                        if ( nalu_hypre_BoxVolume(shift_ibox) )
                        {
                           ac_ptr = nalu_hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                                            ci,
                                                                            stencil_shape);
                           nalu_hypre_BoxGetSize(shift_ibox, loop_size);

#define DEVICE_VAR is_device_ptr(ac_ptr)
                           nalu_hypre_BoxLoop1Begin(ndim, loop_size,
                                               ac_dbox, nalu_hypre_BoxIMin(shift_ibox),
                                               stride, iac);
                           {
                              ac_ptr[iac] = 0.0;
                           }
                           nalu_hypre_BoxLoop1End(iac);
#undef DEVICE_VAR
                        }   /* if ( nalu_hypre_BoxVolume(shift_ibox) ) */

                        nalu_hypre_BoxDestroy(shift_ibox);

                     }  /* for (j= 0; j< nboxman_entries; j++) */
                  }     /* if (abs_shape)  */
               }        /* for (i= 0; i< stencil_size; i++) */
            }           /* if (stencils != NULL) */
         }              /* for (var2= 0; var2< nvars; var2++) */

         nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
      }   /* nalu_hypre_ForBoxI  ci */
   }      /* for (var1= 0; var1< nvars; var1++) */

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FacZeroFCSten: Zeroes the fine stencil coefficients that reach
 * into a coarse box.
 * Idea: zero off any stencil connection of a fine box that does not
 *       connect to a sibling box
 * Algo: For each fbox
 *       {
 *          1) expand by one in each direction so that sibling boxes can be
 *             reached
 *          2) boxman_intersect with the fboxman to get all fboxes including
 *             itself and the siblings
 *          3) loop over intersection boxes, shift them in the stencil
 *             direction (now we are off the fbox), and subtract any sibling
 *             extents. The remaining chunks (boxes of a box_array) are
 *             the desired but shifted extents.
 *          4) shift these shifted extents in the negative stencil direction
 *             to get back into fbox. Zero-off the matrix over these latter
 *             extents.
 *       }
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FacZeroFCSten( nalu_hypre_SStructPMatrix  *A,
                     nalu_hypre_SStructGrid     *grid,
                     NALU_HYPRE_Int              fine_part)
{
   MPI_Comm               comm =   nalu_hypre_SStructGridComm(grid);
   nalu_hypre_BoxManager      *fboxman;
   nalu_hypre_BoxManEntry    **boxman_entries;
   NALU_HYPRE_Int              nboxman_entries;

   nalu_hypre_SStructPGrid    *p_fgrid;
   nalu_hypre_StructGrid      *fgrid;
   nalu_hypre_BoxArray        *fgrid_boxes;
   nalu_hypre_Box             *fgrid_box;
   nalu_hypre_Box              scaled_box;


   nalu_hypre_BoxArray        *intersect_boxes, *tmp_box_array1, *tmp_box_array2;

   nalu_hypre_StructMatrix    *smatrix;

   nalu_hypre_StructStencil   *stencils;
   NALU_HYPRE_Int              stencil_size;

   nalu_hypre_Index            stride, ilower, iupper;
   nalu_hypre_Index            stencil_shape, shift_index;

   nalu_hypre_Box              shift_ibox;
   nalu_hypre_Box              intersect_box;
   nalu_hypre_Index            size_ibox;

   NALU_HYPRE_Int              nvars, var1, var2;
   NALU_HYPRE_Int              ndim;

   nalu_hypre_Box             *a_dbox;
   NALU_HYPRE_Real            *a_ptr;
   nalu_hypre_Index            loop_size;

   NALU_HYPRE_Int              fi, fj, i, j;
   NALU_HYPRE_Int              abs_shape;
   NALU_HYPRE_Int              myid, proc;
   NALU_HYPRE_Int              ierr = 0;

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   p_fgrid  = nalu_hypre_SStructPMatrixPGrid(A);
   nvars    = nalu_hypre_SStructPMatrixNVars(A);
   ndim     = nalu_hypre_SStructPGridNDim(p_fgrid);

   nalu_hypre_BoxInit(&scaled_box, ndim);
   nalu_hypre_BoxInit(&shift_ibox, ndim);
   nalu_hypre_BoxInit(&intersect_box, ndim);

   nalu_hypre_ClearIndex(stride);
   for (i = 0; i < ndim; i++)
   {
      stride[i] = 1;
   }

   tmp_box_array1 = nalu_hypre_BoxArrayCreate(1, ndim);

   for (var1 = 0; var1 < nvars; var1++)
   {
      fgrid      = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A), var1);
      fgrid_boxes = nalu_hypre_StructGridBoxes(fgrid);
      fboxman    = nalu_hypre_SStructGridBoxManager(grid, fine_part, var1);

      nalu_hypre_ForBoxI(fi, fgrid_boxes)
      {
         fgrid_box = nalu_hypre_BoxArrayBox(fgrid_boxes, fi);
         nalu_hypre_ClearIndex(size_ibox);
         for (i = 0; i < ndim; i++)
         {
            size_ibox[i] = nalu_hypre_BoxSizeD(fgrid_box, i) - 1;
         }

         /* expand fgrid_box & boxman_intersect with fboxman. */
         nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(fgrid_box), stride, 3,
                               nalu_hypre_BoxIMin(&scaled_box));
         nalu_hypre_AddIndexes(nalu_hypre_BoxIMax(fgrid_box), stride, 3,
                          nalu_hypre_BoxIMax(&scaled_box));

         nalu_hypre_BoxManIntersect(fboxman, nalu_hypre_BoxIMin(&scaled_box),
                               nalu_hypre_BoxIMax(&scaled_box), &boxman_entries,
                               &nboxman_entries);

         for (var2 = 0; var2 < nvars; var2++)
         {
            stencils =  nalu_hypre_SStructPMatrixSStencil(A, var1, var2);

            if (stencils != NULL)
            {
               stencil_size = nalu_hypre_StructStencilSize(stencils);
               smatrix     = nalu_hypre_SStructPMatrixSMatrix(A, var1, var2);
               a_dbox      = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(smatrix),
                                               fi);

               for (i = 0; i < stencil_size; i++)
               {
                  nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i),
                                  stencil_shape);
                  AbsStencilShape(stencil_shape, abs_shape);

                  if (abs_shape)   /* non-centre stencils are zeroed */
                  {
                     nalu_hypre_SetIndex3(shift_index,
                                     size_ibox[0]*stencil_shape[0],
                                     size_ibox[1]*stencil_shape[1],
                                     size_ibox[2]*stencil_shape[2]);
                     nalu_hypre_AddIndexes(shift_index, nalu_hypre_BoxIMin(fgrid_box), 3,
                                      nalu_hypre_BoxIMin(&shift_ibox));
                     nalu_hypre_AddIndexes(shift_index, nalu_hypre_BoxIMax(fgrid_box), 3,
                                      nalu_hypre_BoxIMax(&shift_ibox));
                     nalu_hypre_IntersectBoxes(&shift_ibox, fgrid_box, &shift_ibox);

                     nalu_hypre_SetIndex3(shift_index, -stencil_shape[0], -stencil_shape[1],
                                     -stencil_shape[2]);

                     /*-----------------------------------------------------------
                      * Check to see if the stencil does not couple to a sibling
                      * box. These boxes should be in boxman_entries. But do not
                      * subtract fgrid_box itself, which is also in boxman_entries.
                      *-----------------------------------------------------------*/
                     nalu_hypre_AddIndexes(stencil_shape, nalu_hypre_BoxIMin(&shift_ibox), 3,
                                      nalu_hypre_BoxIMin(&shift_ibox));
                     nalu_hypre_AddIndexes(stencil_shape, nalu_hypre_BoxIMax(&shift_ibox), 3,
                                      nalu_hypre_BoxIMax(&shift_ibox));

                     intersect_boxes =  nalu_hypre_BoxArrayCreate(1, ndim);
                     nalu_hypre_CopyBox(&shift_ibox, nalu_hypre_BoxArrayBox(intersect_boxes, 0));

                     for (j = 0; j < nboxman_entries; j++)
                     {
                        nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);
                        nalu_hypre_SStructBoxManEntryGetBoxnum(boxman_entries[j], &fj);

                        if ((proc != myid) || (fj != fi))
                        {
                           nalu_hypre_BoxManEntryGetExtents(boxman_entries[j], ilower, iupper);
                           nalu_hypre_BoxSetExtents(&scaled_box, ilower, iupper);

                           nalu_hypre_IntersectBoxes(&shift_ibox, &scaled_box, &intersect_box);

                           if ( nalu_hypre_BoxVolume(&intersect_box) )
                           {
                              nalu_hypre_CopyBox(&intersect_box,
                                            nalu_hypre_BoxArrayBox(tmp_box_array1, 0));

                              tmp_box_array2 = nalu_hypre_BoxArrayCreate(0, ndim);

                              nalu_hypre_SubtractBoxArrays(intersect_boxes,
                                                      tmp_box_array1,
                                                      tmp_box_array2);

                              nalu_hypre_BoxArrayDestroy(tmp_box_array2);
                           }
                        }
                     }   /* for (j= 0; j< nboxman_entries; j++) */

                     /*-----------------------------------------------------------
                      * intersect_boxes now has the shifted extents for the
                      * coefficients to be zeroed.
                      *-----------------------------------------------------------*/
                     a_ptr = nalu_hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                                     fi,
                                                                     stencil_shape);
                     nalu_hypre_ForBoxI(fj, intersect_boxes)
                     {
                        nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(intersect_boxes, fj), &intersect_box);

                        nalu_hypre_AddIndexes(shift_index, nalu_hypre_BoxIMin(&intersect_box), 3,
                                         nalu_hypre_BoxIMin(&intersect_box));
                        nalu_hypre_AddIndexes(shift_index, nalu_hypre_BoxIMax(&intersect_box), 3,
                                         nalu_hypre_BoxIMax(&intersect_box));

                        nalu_hypre_BoxGetSize(&intersect_box, loop_size);

#define DEVICE_VAR is_device_ptr(a_ptr)
                        nalu_hypre_BoxLoop1Begin(ndim, loop_size,
                                            a_dbox, nalu_hypre_BoxIMin(&intersect_box),
                                            stride, ia);
                        {
                           a_ptr[ia] = 0.0;
                        }
                        nalu_hypre_BoxLoop1End(ia);
#undef DEVICE_VAR

                     }  /* nalu_hypre_ForBoxI(fj, intersect_boxes) */

                     nalu_hypre_BoxArrayDestroy(intersect_boxes);

                  }  /* if (abs_shape) */
               }      /* for (i= 0; i< stencil_size; i++) */
            }         /* if (stencils != NULL) */
         }            /* for (var2= 0; var2< nvars; var2++) */

         nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
      }  /* nalu_hypre_ForBoxI(fi, fgrid_boxes) */
   }     /* for (var1= 0; var1< nvars; var1++) */

   nalu_hypre_BoxArrayDestroy(tmp_box_array1);

   return ierr;
}

