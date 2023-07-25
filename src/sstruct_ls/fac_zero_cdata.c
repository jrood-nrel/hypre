/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fac.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_FacZeroCData: Zeroes the data over the underlying coarse indices of
 * the refinement patches.
 *    Algo.:For each cbox
 *       {
 *          1) refine cbox and boxman_intersect with fboxman
 *          2) loop over intersection boxes
 *                3) coarsen and contract (only the coarse nodes on this
 *                   processor) and zero data.
 *       }
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FacZeroCData( void                 *fac_vdata,
                    nalu_hypre_SStructMatrix  *A )
{
   nalu_hypre_FACData         *fac_data      =  (nalu_hypre_FACData*)fac_vdata;

   nalu_hypre_SStructGrid     *grid;
   nalu_hypre_SStructPGrid    *p_cgrid;

   nalu_hypre_StructGrid      *cgrid;
   nalu_hypre_BoxArray        *cgrid_boxes;
   nalu_hypre_Box             *cgrid_box;

   nalu_hypre_BoxManager      *fboxman;
   nalu_hypre_BoxManEntry    **boxman_entries;
   NALU_HYPRE_Int              nboxman_entries;

   nalu_hypre_Box              scaled_box;
   nalu_hypre_Box              intersect_box;

   nalu_hypre_SStructPMatrix  *level_pmatrix;
   nalu_hypre_StructStencil   *stencils;
   NALU_HYPRE_Int              stencil_size;

   nalu_hypre_Index           *refine_factors;
   nalu_hypre_Index            temp_index;
   nalu_hypre_Index            ilower, iupper;

   NALU_HYPRE_Int              max_level     =  fac_data -> max_levels;
   NALU_HYPRE_Int             *level_to_part =  fac_data -> level_to_part;

   NALU_HYPRE_Int              ndim          =  nalu_hypre_SStructMatrixNDim(A);
   NALU_HYPRE_Int              part_crse     =  0;
   NALU_HYPRE_Int              part_fine     =  1;
   NALU_HYPRE_Int              level;
   NALU_HYPRE_Int              nvars, var;

   NALU_HYPRE_Int              ci, i, j, rem, intersect_size;

   NALU_HYPRE_Real            *values;

   NALU_HYPRE_Int              ierr = 0;

   nalu_hypre_BoxInit(&scaled_box, ndim);
   nalu_hypre_BoxInit(&intersect_box, ndim);

   for (level = max_level; level > 0; level--)
   {
      level_pmatrix = nalu_hypre_SStructMatrixPMatrix(fac_data -> A_level[level], part_crse);

      grid          = (fac_data -> grid_level[level]);
      refine_factors = &(fac_data -> refine_factors[level]);

      p_cgrid = nalu_hypre_SStructGridPGrid(grid, part_crse);
      nvars  = nalu_hypre_SStructPGridNVars(p_cgrid);

      for (var = 0; var < nvars; var++)
      {
         stencils    =  nalu_hypre_SStructPMatrixSStencil(level_pmatrix, var, var);
         stencil_size =  nalu_hypre_StructStencilSize(stencils);

         /*---------------------------------------------------------------------
          * For each variable, find the underlying boxes for each coarse box.
          *---------------------------------------------------------------------*/
         cgrid        = nalu_hypre_SStructPGridSGrid(p_cgrid, var);
         cgrid_boxes  = nalu_hypre_StructGridBoxes(cgrid);
         fboxman         = nalu_hypre_SStructGridBoxManager(grid, part_fine, var);

         nalu_hypre_ForBoxI(ci, cgrid_boxes)
         {
            cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

            nalu_hypre_ClearIndex(temp_index);
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(cgrid_box), temp_index,
                                        *refine_factors, nalu_hypre_BoxIMin(&scaled_box));
            for (i = 0; i < ndim; i++)
            {
               temp_index[i] = (*refine_factors)[i] - 1;
            }
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(cgrid_box), temp_index,
                                        *refine_factors, nalu_hypre_BoxIMax(&scaled_box));

            nalu_hypre_BoxManIntersect(fboxman, nalu_hypre_BoxIMin(&scaled_box),
                                  nalu_hypre_BoxIMax(&scaled_box), &boxman_entries,
                                  &nboxman_entries);

            for (i = 0; i < nboxman_entries; i++)
            {
               nalu_hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
               nalu_hypre_BoxSetExtents(&intersect_box, ilower, iupper);
               nalu_hypre_IntersectBoxes(&intersect_box, &scaled_box, &intersect_box);

               /* adjust the box so that it is divisible by refine_factors */
               for (j = 0; j < ndim; j++)
               {
                  rem = nalu_hypre_BoxIMin(&intersect_box)[j] % (*refine_factors)[j];
                  if (rem)
                  {
                     nalu_hypre_BoxIMin(&intersect_box)[j] += (*refine_factors)[j] - rem;
                  }
               }

               nalu_hypre_ClearIndex(temp_index);
               nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(&intersect_box), temp_index,
                                           *refine_factors, nalu_hypre_BoxIMin(&intersect_box));
               nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(&intersect_box), temp_index,
                                           *refine_factors, nalu_hypre_BoxIMax(&intersect_box));

               intersect_size = nalu_hypre_BoxVolume(&intersect_box);
               if (intersect_size > 0)
               {
                  /*------------------------------------------------------------
                   * Coarse underlying box found. Now zero off.
                   *------------------------------------------------------------*/
                  values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  intersect_size, NALU_HYPRE_MEMORY_HOST);

                  for (j = 0; j < stencil_size; j++)
                  {
                     NALU_HYPRE_SStructMatrixSetBoxValues(fac_data -> A_level[level],
                                                     part_crse,
                                                     nalu_hypre_BoxIMin(&intersect_box),
                                                     nalu_hypre_BoxIMax(&intersect_box),
                                                     var, 1, &j, values);

                     NALU_HYPRE_SStructMatrixSetBoxValues(A,
                                                     level_to_part[level - 1],
                                                     nalu_hypre_BoxIMin(&intersect_box),
                                                     nalu_hypre_BoxIMax(&intersect_box),
                                                     var, 1, &j, values);
                  }

                  nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

               }  /* if (intersect_size > 0) */
            }     /* for (i= 0; i< nboxman_entries; i++) */

            nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);

         }   /* nalu_hypre_ForBoxI(ci, cgrid_boxes) */
      }      /* for (var= 0; var< nvars; var++) */
   }         /* for (level= max_level; level> 0; level--) */

   return ierr;
}

