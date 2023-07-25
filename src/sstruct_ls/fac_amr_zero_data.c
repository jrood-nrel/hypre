/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fac.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ZeroAMRVectorData: Zeroes the data over the underlying coarse
 * indices of the refinement patches.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ZeroAMRVectorData(nalu_hypre_SStructVector  *b,
                        NALU_HYPRE_Int            *plevels,
                        nalu_hypre_Index          *rfactors )
{
   nalu_hypre_SStructGrid     *grid =  nalu_hypre_SStructVectorGrid(b);
   nalu_hypre_SStructPGrid    *p_cgrid;

   nalu_hypre_StructGrid      *cgrid;
   nalu_hypre_BoxArray        *cgrid_boxes;
   nalu_hypre_Box             *cgrid_box;

   nalu_hypre_BoxManager      *fboxman;
   nalu_hypre_BoxManEntry    **boxman_entries;
   NALU_HYPRE_Int              nboxman_entries;

   nalu_hypre_Box              scaled_box;
   nalu_hypre_Box              intersect_box;

   NALU_HYPRE_Int              npart =  nalu_hypre_SStructVectorNParts(b);
   NALU_HYPRE_Int              ndim =  nalu_hypre_SStructVectorNDim(b);

   NALU_HYPRE_Int             *levels;

   nalu_hypre_Index           *refine_factors;
   nalu_hypre_Index            temp_index, ilower, iupper;

   NALU_HYPRE_Int              level;
   NALU_HYPRE_Int              nvars, var;

   NALU_HYPRE_Int              part, ci, rem, i, j, intersect_size;

   NALU_HYPRE_Real            *values1;

   NALU_HYPRE_Int              ierr = 0;

   nalu_hypre_BoxInit(&scaled_box, ndim);
   nalu_hypre_BoxInit(&intersect_box, ndim);

   levels        = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  npart, NALU_HYPRE_MEMORY_HOST);
   refine_factors = nalu_hypre_CTAlloc(nalu_hypre_Index,  npart, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < npart; part++)
   {
      levels[plevels[part]] = part;
      for (i = 0; i < ndim; i++)
      {
         refine_factors[plevels[part]][i] = rfactors[part][i];
      }
      for (i = ndim; i < 3; i++)
      {
         refine_factors[plevels[part]][i] = 1;
      }
   }

   nalu_hypre_ClearIndex(temp_index);

   for (level = npart - 1; level > 0; level--)
   {
      p_cgrid = nalu_hypre_SStructGridPGrid(grid, levels[level - 1]);
      nvars  = nalu_hypre_SStructPGridNVars(p_cgrid);

      for (var = 0; var < nvars; var++)
      {
         /*---------------------------------------------------------------------
          * For each variable, find the underlying boxes for each fine box.
          *---------------------------------------------------------------------*/
         cgrid      = nalu_hypre_SStructPGridSGrid(p_cgrid, var);
         cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
         fboxman    = nalu_hypre_SStructGridBoxManager(grid, levels[level], var);

         nalu_hypre_ForBoxI(ci, cgrid_boxes)
         {
            cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

            nalu_hypre_ClearIndex(temp_index);
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(cgrid_box), temp_index,
                                        refine_factors[level], nalu_hypre_BoxIMin(&scaled_box));
            for (i = 0; i < ndim; i++)
            {
               temp_index[i] = refine_factors[level][i] - 1;
            }
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(cgrid_box), temp_index,
                                        refine_factors[level], nalu_hypre_BoxIMax(&scaled_box));
            nalu_hypre_ClearIndex(temp_index);

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
                  rem = nalu_hypre_BoxIMin(&intersect_box)[j] % refine_factors[level][j];
                  if (rem)
                  {
                     nalu_hypre_BoxIMin(&intersect_box)[j] += refine_factors[level][j] - rem;
                  }
               }

               nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(&intersect_box), temp_index,
                                           refine_factors[level], nalu_hypre_BoxIMin(&intersect_box));
               nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(&intersect_box), temp_index,
                                           refine_factors[level], nalu_hypre_BoxIMax(&intersect_box));

               intersect_size = nalu_hypre_BoxVolume(&intersect_box);
               if (intersect_size > 0)
               {
                  /*------------------------------------------------------------
                   * Coarse underlying box found. Now zero off.
                   *------------------------------------------------------------*/
                  values1 = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  intersect_size, NALU_HYPRE_MEMORY_HOST);

                  NALU_HYPRE_SStructVectorSetBoxValues(b, levels[level - 1],
                                                  nalu_hypre_BoxIMin(&intersect_box),
                                                  nalu_hypre_BoxIMax(&intersect_box),
                                                  var, values1);
                  nalu_hypre_TFree(values1, NALU_HYPRE_MEMORY_HOST);

               }  /* if (intersect_size > 0) */
            }     /* for (i= 0; i< nboxman_entries; i++) */

            nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);

         }   /* nalu_hypre_ForBoxI(ci, cgrid_boxes) */
      }      /* for (var= 0; var< nvars; var++) */
   }         /* for (level= max_level; level> 0; level--) */

   nalu_hypre_TFree(levels, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(refine_factors, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_ZeroAMRMatrixData: Zeroes the data over the underlying coarse
 * indices of the refinement patches between two levels.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ZeroAMRMatrixData(nalu_hypre_SStructMatrix  *A,
                        NALU_HYPRE_Int             part_crse,
                        nalu_hypre_Index           rfactors )
{
   nalu_hypre_SStructGraph    *graph =  nalu_hypre_SStructMatrixGraph(A);
   nalu_hypre_SStructGrid     *grid =  nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int              ndim =  nalu_hypre_SStructMatrixNDim(A);

   nalu_hypre_SStructPGrid    *p_cgrid;

   nalu_hypre_StructGrid      *cgrid;
   nalu_hypre_BoxArray        *cgrid_boxes;
   nalu_hypre_Box             *cgrid_box;

   nalu_hypre_BoxManager      *fboxman;
   nalu_hypre_BoxManEntry    **boxman_entries;
   NALU_HYPRE_Int              nboxman_entries;

   nalu_hypre_Box              scaled_box;
   nalu_hypre_Box              intersect_box;

   nalu_hypre_SStructStencil  *stencils;
   NALU_HYPRE_Int              stencil_size;

   nalu_hypre_Index           *stencil_shape;
   nalu_hypre_Index            temp_index, ilower, iupper;

   NALU_HYPRE_Int              nvars, var;

   NALU_HYPRE_Int              ci, i, j, rem, intersect_size, rank;

   NALU_HYPRE_Real            *values1, *values2;

   NALU_HYPRE_Int              ierr = 0;

   nalu_hypre_BoxInit(&scaled_box, ndim);
   nalu_hypre_BoxInit(&intersect_box, ndim);

   p_cgrid = nalu_hypre_SStructGridPGrid(grid, part_crse);
   nvars  = nalu_hypre_SStructPGridNVars(p_cgrid);

   for (var = 0; var < nvars; var++)
   {
      stencils     =  nalu_hypre_SStructGraphStencil(graph, part_crse, var);
      stencil_size =  nalu_hypre_SStructStencilSize(stencils);
      stencil_shape = nalu_hypre_SStructStencilShape(stencils);

      /*---------------------------------------------------------------------
       * For each variable, find the underlying boxes for each fine box.
       *---------------------------------------------------------------------*/
      cgrid        = nalu_hypre_SStructPGridSGrid(p_cgrid, var);
      cgrid_boxes  = nalu_hypre_StructGridBoxes(cgrid);
      fboxman      = nalu_hypre_SStructGridBoxManager(grid, part_crse + 1, var);

      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

         nalu_hypre_ClearIndex(temp_index);
         nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(cgrid_box), temp_index,
                                     rfactors, nalu_hypre_BoxIMin(&scaled_box));
         for (i = 0; i < ndim; i++)
         {
            temp_index[i] =  rfactors[i] - 1;
         }
         nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(cgrid_box), temp_index,
                                     rfactors, nalu_hypre_BoxIMax(&scaled_box));
         nalu_hypre_ClearIndex(temp_index);

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
               rem = nalu_hypre_BoxIMin(&intersect_box)[j] % rfactors[j];
               if (rem)
               {
                  nalu_hypre_BoxIMin(&intersect_box)[j] += rfactors[j] - rem;
               }
            }

            nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(&intersect_box), temp_index,
                                        rfactors, nalu_hypre_BoxIMin(&intersect_box));
            nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(&intersect_box), temp_index,
                                        rfactors, nalu_hypre_BoxIMax(&intersect_box));

            intersect_size = nalu_hypre_BoxVolume(&intersect_box);
            if (intersect_size > 0)
            {
               /*------------------------------------------------------------
                * Coarse underlying box found. Now zero off.
                *------------------------------------------------------------*/
               values1 = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  intersect_size, NALU_HYPRE_MEMORY_HOST);
               values2 = nalu_hypre_TAlloc(NALU_HYPRE_Real,  intersect_size, NALU_HYPRE_MEMORY_HOST);
               for (j = 0; j < intersect_size; j++)
               {
                  values2[j] = 1.0;
               }

               for (j = 0; j < stencil_size; j++)
               {
                  rank = nalu_hypre_abs(nalu_hypre_IndexX(stencil_shape[j])) +
                         nalu_hypre_abs(nalu_hypre_IndexY(stencil_shape[j])) +
                         nalu_hypre_abs(nalu_hypre_IndexZ(stencil_shape[j]));

                  if (rank)
                  {
                     NALU_HYPRE_SStructMatrixSetBoxValues(A,
                                                     part_crse,
                                                     nalu_hypre_BoxIMin(&intersect_box),
                                                     nalu_hypre_BoxIMax(&intersect_box),
                                                     var, 1, &j, values1);
                  }
                  else
                  {
                     NALU_HYPRE_SStructMatrixSetBoxValues(A,
                                                     part_crse,
                                                     nalu_hypre_BoxIMin(&intersect_box),
                                                     nalu_hypre_BoxIMax(&intersect_box),
                                                     var, 1, &j, values2);
                  }
               }
               nalu_hypre_TFree(values1, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(values2, NALU_HYPRE_MEMORY_HOST);

            }   /* if (intersect_size > 0) */
         }      /* for (i= 0; i< nmap_entries; i++) */

         nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
      }   /* nalu_hypre_ForBoxI(ci, cgrid_boxes) */
   }      /* for (var= 0; var< nvars; var++) */

   return ierr;
}



