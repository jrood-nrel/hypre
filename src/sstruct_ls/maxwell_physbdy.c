/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Need to fix the way these variables are set and incremented in loops:
 *   cnt
 *
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * Finds the physical boundary boxes for all levels. Since the coarse grid's
 * boundary may not be on the physical bdry, we need to compare the coarse
 * grid to the finest level boundary boxes. All boxes of the coarse grids
 * must be checked, not just the bounding box.
 *    Algo:
 *         1) obtain boundary boxes for the finest grid
 *             i) mark the fboxes that have boundary elements.
 *         2) loop over coarse levels
 *             i) for a cbox that maps to a fbox that has boundary layers
 *                a) refine the cbox
 *                b) intersect with the cell boundary layers of the fbox
 *                c) coarsen the intersection
 *            ii) determine the var boxes
 *           iii) mark the coarse box
 *
 * Concerns: Checking an individual pgrid may give artificial physical
 * boundaries. Need to check if any other pgrid is adjacent to it.
 * We omit this case and assume only one part for now.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_Maxwell_PhysBdy( nalu_hypre_SStructGrid      **grid_l,
                       NALU_HYPRE_Int                num_levels,
                       nalu_hypre_Index              rfactors,
                       NALU_HYPRE_Int             ***BdryRanksl_ptr,
                       NALU_HYPRE_Int              **BdryRanksCntsl_ptr )
{

   MPI_Comm                comm = (grid_l[0]-> comm);

   NALU_HYPRE_Int             **BdryRanks_l;
   NALU_HYPRE_Int              *BdryRanksCnts_l;

   NALU_HYPRE_Int              *npts;
   NALU_HYPRE_BigInt           *ranks, *upper_rank, *lower_rank;
   nalu_hypre_BoxManEntry      *boxman_entry;

   nalu_hypre_SStructGrid      *grid;
   nalu_hypre_SStructPGrid     *pgrid;
   nalu_hypre_StructGrid       *cell_fgrid, *cell_cgrid, *sgrid;

   nalu_hypre_BoxArrayArray ****bdry;
   nalu_hypre_BoxArrayArray    *fbdry;
   nalu_hypre_BoxArrayArray    *cbdry;

   nalu_hypre_BoxArray         *box_array;
   nalu_hypre_BoxArray         *fboxes, *cboxes;

   nalu_hypre_Box              *fbox, *cbox;
   nalu_hypre_Box              *box, *contract_fbox, rbox;
   nalu_hypre_Box               intersect;

   NALU_HYPRE_Int             **cbox_mapping, **fbox_mapping;
   NALU_HYPRE_Int             **boxes_with_bdry;

   NALU_HYPRE_Int               ndim, nvars;
   NALU_HYPRE_Int               nboxes, nfboxes;
   NALU_HYPRE_Int               boxi;

   nalu_hypre_Index             zero_shift, upper_shift, lower_shift;
   nalu_hypre_Index             loop_size, start, index, lindex;

   NALU_HYPRE_Int               i, j, k, l, m, n, p;
   NALU_HYPRE_Int               d;
   NALU_HYPRE_Int               cnt;

   NALU_HYPRE_Int               part = 0; /* NOTE, ASSUMING ONE PART */
   NALU_HYPRE_Int               matrix_type = NALU_HYPRE_PARCSR;
   NALU_HYPRE_Int               myproc;

   NALU_HYPRE_Int               ierr = 0;

   nalu_hypre_MPI_Comm_rank(comm, &myproc);

   ndim = nalu_hypre_SStructGridNDim(grid_l[0]);
   nalu_hypre_SetIndex3(zero_shift, 0, 0, 0);

   nalu_hypre_BoxInit(&intersect, ndim);

   /* bounding global ranks of this processor & allocate boundary box markers. */
   upper_rank = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_levels, NALU_HYPRE_MEMORY_HOST);
   lower_rank = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_levels, NALU_HYPRE_MEMORY_HOST);

   boxes_with_bdry = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_levels; i++)
   {
      grid = grid_l[i];
      lower_rank[i] = nalu_hypre_SStructGridStartRank(grid);

      /* note we are assuming only one part */
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      sgrid = nalu_hypre_SStructPGridSGrid(pgrid, nvars - 1);
      box_array = nalu_hypre_StructGridBoxes(sgrid);
      box  = nalu_hypre_BoxArrayBox(box_array, nalu_hypre_BoxArraySize(box_array) - 1);

      nalu_hypre_SStructGridBoxProcFindBoxManEntry(grid, part, nvars - 1,
                                              nalu_hypre_BoxArraySize(box_array) - 1, myproc, &boxman_entry);
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(boxman_entry, nalu_hypre_BoxIMax(box),
                                              &upper_rank[i]);

      sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      box_array = nalu_hypre_StructGridBoxes(sgrid);
      boxes_with_bdry[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(box_array), NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------------------
    * construct box_number mapping between levels, and offset strides because of
    * projection coarsening. Note: from the way the coarse boxes are created and
    * numbered, to determine the coarse box that matches the fbox, we need to
    * only check the tail end of the list of cboxes. In fact, given fbox_i,
    * if it's coarsened extents do not interesect with the first coarse box of the
    * tail end, then this fbox vanishes in the coarsening.
    *   c/fbox_mapping gives the fine/coarse box mapping between two consecutive levels
    *   of the multilevel hierarchy.
    *-----------------------------------------------------------------------------*/
   if (num_levels > 1)
   {
      cbox_mapping = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_levels, NALU_HYPRE_MEMORY_HOST);
      fbox_mapping = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   }
   for (i = 0; i < (num_levels - 1); i++)
   {
      grid = grid_l[i];
      pgrid = nalu_hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_fgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      fboxes = nalu_hypre_StructGridBoxes(cell_fgrid);
      nfboxes = nalu_hypre_BoxArraySize(nalu_hypre_StructGridBoxes(cell_fgrid));
      fbox_mapping[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nfboxes, NALU_HYPRE_MEMORY_HOST);

      grid = grid_l[i + 1];
      pgrid = nalu_hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_cgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      cboxes = nalu_hypre_StructGridBoxes(cell_cgrid);
      nboxes = nalu_hypre_BoxArraySize(nalu_hypre_StructGridBoxes(cell_cgrid));

      cbox_mapping[i + 1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nboxes, NALU_HYPRE_MEMORY_HOST);

      /* assuming if i1 > i2 and (box j1) is coarsened from (box i1)
         and (box j2) from (box i2), then j1 > j2. */
      k = 0;
      nalu_hypre_ForBoxI(j, fboxes)
      {
         fbox = nalu_hypre_BoxArrayBox(fboxes, j);
         nalu_hypre_CopyBox(fbox, &rbox);
         nalu_hypre_ProjectBox(&rbox, zero_shift, rfactors);
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(&rbox), zero_shift,
                                     rfactors, nalu_hypre_BoxIMin(&rbox));
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(&rbox), zero_shift,
                                     rfactors, nalu_hypre_BoxIMax(&rbox));

         /* since the ordering of the cboxes was determined by the fbox
            ordering, we only have to check if the first cbox in the
            list intersects with rbox. If not, this fbox vanished in the
            coarsening. */
         cbox = nalu_hypre_BoxArrayBox(cboxes, k);
         nalu_hypre_IntersectBoxes(&rbox, cbox, &rbox);
         if (nalu_hypre_BoxVolume(&rbox))
         {
            cbox_mapping[i + 1][k] = j;
            fbox_mapping[i][j] = k;
            k++;
         }  /* if (nalu_hypre_BoxVolume(&rbox)) */
      }     /* nalu_hypre_ForBoxI(j, fboxes) */
   }        /* for (i= 0; i< (num_levels-1); i++) */

   bdry = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray ***,  num_levels, NALU_HYPRE_MEMORY_HOST);
   npts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_levels, NALU_HYPRE_MEMORY_HOST);

   /* finest level boundary determination */
   grid = grid_l[0];
   pgrid = nalu_hypre_SStructGridPGrid(grid, 0); /* assuming one part */
   nvars = nalu_hypre_SStructPGridNVars(pgrid);
   cell_fgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
   nboxes = nalu_hypre_BoxArraySize(nalu_hypre_StructGridBoxes(cell_fgrid));

   nalu_hypre_Maxwell_PNedelec_Bdy(cell_fgrid, pgrid, &bdry[0]);
   for (i = 0; i < nboxes; i++)
   {
      if (bdry[0][i])  /* boundary layers on box[i] */
      {
         for (j = 0; j < nvars; j++)
         {
            fbdry = bdry[0][i][j + 1]; /*(j+1) since j= 0 stores cell-centred boxes*/
            nalu_hypre_ForBoxArrayI(k, fbdry)
            {
               box_array = nalu_hypre_BoxArrayArrayBoxArray(fbdry, k);
               nalu_hypre_ForBoxI(p, box_array)
               {
                  box = nalu_hypre_BoxArrayBox(box_array, p);
                  npts[0] += nalu_hypre_BoxVolume(box);
               }
            }
         }  /* for (j= 0; j< nvars; j++) */

         boxes_with_bdry[0][i] = 1; /* mark this box as containing boundary layers */
      }  /* if (bdry[0][i]) */
   }
   nfboxes = nboxes;

   /* coarser levels */
   for (i = 1; i < num_levels; i++)
   {
      grid = grid_l[i - 1];
      pgrid = nalu_hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_fgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      fboxes = nalu_hypre_StructGridBoxes(cell_fgrid);

      grid = grid_l[i];
      pgrid = nalu_hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_cgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      cboxes = nalu_hypre_StructGridBoxes(cell_cgrid);
      nboxes = nalu_hypre_BoxArraySize(nalu_hypre_StructGridBoxes(cell_cgrid));

      bdry[i] = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray **,  nboxes, NALU_HYPRE_MEMORY_HOST);
      p = 2 * (ndim - 1);
      for (j = 0; j < nboxes; j++)
      {
         bdry[i][j] = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray *,  nvars + 1, NALU_HYPRE_MEMORY_HOST);

         /* cell grid boxarrayarray */
         bdry[i][j][0] = nalu_hypre_BoxArrayArrayCreate(2 * ndim, ndim);

         /* var grid boxarrayarrays */
         for (k = 0; k < nvars; k++)
         {
            bdry[i][j][k + 1] = nalu_hypre_BoxArrayArrayCreate(p, ndim);
         }
      }

      /* check if there are boundary points from the previous level */
      for (j = 0; j < nfboxes; j++)
      {
         /* see if the j box of level (i-1) has any boundary layers */
         if (boxes_with_bdry[i - 1][j])
         {
            boxi = fbox_mapping[i - 1][j];
            cbox = nalu_hypre_BoxArrayBox(cboxes, boxi);
            fbox = nalu_hypre_BoxArrayBox(fboxes, j);

            /* contract the fbox so that divisible in rfactor */
            contract_fbox = nalu_hypre_BoxContraction(fbox, cell_fgrid, rfactors);

            /* refine the cbox. Expand the refined cbox so that the complete
               chunk of the fine box that coarsened to it is included. This
               requires some offsets */
            nalu_hypre_ClearIndex(upper_shift);
            nalu_hypre_ClearIndex(lower_shift);
            for (k = 0; k < ndim; k++)
            {
               m = nalu_hypre_BoxIMin(contract_fbox)[k];
               p = m % rfactors[k];

               if (p > 0 && m > 0)
               {
                  upper_shift[k] = p - 1;
                  lower_shift[k] = p - rfactors[k];
               }
               else
               {
                  upper_shift[k] = rfactors[k] - p - 1;
                  lower_shift[k] = -p;
               }
            }
            nalu_hypre_BoxDestroy(contract_fbox);

            nalu_hypre_CopyBox(cbox, &rbox);
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(&rbox), zero_shift,
                                        rfactors, nalu_hypre_BoxIMin(&rbox));
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(&rbox), zero_shift,
                                        rfactors, nalu_hypre_BoxIMax(&rbox));

            nalu_hypre_AddIndexes(lower_shift, nalu_hypre_BoxIMin(&rbox), 3,
                             nalu_hypre_BoxIMin(&rbox));
            nalu_hypre_AddIndexes(upper_shift, nalu_hypre_BoxIMax(&rbox), 3,
                             nalu_hypre_BoxIMax(&rbox));

            /* Determine, if any, boundary layers for this rbox. Since the
               boundaries of the coarser levels may not be physical, we cannot
               use nalu_hypre_BoxBoundaryDG. But accomplished through intersecting
               with the finer level boundary boxes. */
            fbdry = bdry[i - 1][j][0]; /* cell-centred boundary layers of level (i-1) */
            cbdry = bdry[i][boxi][0]; /* cell-centred boundary layers of level i */

            /* fbdry is the cell-centred box_arrayarray. Contains an array of (2*ndim)
               boxarrays, one for each direction. */
            cnt = 0;
            nalu_hypre_ForBoxArrayI(l, fbdry)
            {
               /* determine which boundary side we are doing. Depending on the
                  boundary, when we coarsen the refined boundary layer, the
                  extents may need to be changed,
                  e.g., index[lower,j,k]= index[upper,j,k]. */
               switch (l)
               {
                  case 0:  /* lower x direction, x_upper= x_lower */
                  {
                     n = 1; /* n flags whether upper or lower to be replaced */
                     d = 0; /* x component */
                     break;
                  }
                  case 1:  /* upper x direction, x_lower= x_upper */
                  {
                     n = 0; /* n flags whether upper or lower to be replaced */
                     d = 0; /* x component */
                     break;
                  }
                  case 2:  /* lower y direction, y_upper= y_lower */
                  {
                     n = 1; /* n flags whether upper or lower to be replaced */
                     d = 1; /* y component */
                     break;
                  }
                  case 3:  /* upper y direction, y_lower= y_upper */
                  {
                     n = 0; /* n flags whether upper or lower to be replaced */
                     d = 1; /* y component */
                     break;
                  }
                  case 4:  /* lower z direction, z_lower= z_upper */
                  {
                     n = 1; /* n flags whether upper or lower to be replaced */
                     d = 2; /* z component */
                     break;
                  }
                  case 5:  /* upper z direction, z_upper= z_lower */
                  {
                     n = 0; /* n flags whether upper or lower to be replaced */
                     d = 2; /* z component */
                     break;
                  }
               }

               box_array = nalu_hypre_BoxArrayArrayBoxArray(fbdry, l);
               nalu_hypre_ForBoxI(p, box_array)
               {
                  nalu_hypre_IntersectBoxes(nalu_hypre_BoxArrayBox(box_array, p), &rbox,
                                       &intersect);
                  if (nalu_hypre_BoxVolume(&intersect))
                  {
                     /* coarsen the refined boundary box and append it to
                        boxarray nalu_hypre_BoxArrayArrayBoxArray(cbdry, l) */
                     nalu_hypre_ProjectBox(&intersect, zero_shift, rfactors);
                     nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(&intersect),
                                                 zero_shift, rfactors, nalu_hypre_BoxIMin(&intersect));
                     nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(&intersect),
                                                 zero_shift, rfactors, nalu_hypre_BoxIMax(&intersect));

                     /* the coarsened intersect box may be incorrect because
                        of the box projecting formulas. */
                     if (n) /* replace upper by lower */
                     {
                        nalu_hypre_BoxIMax(&intersect)[d] = nalu_hypre_BoxIMin(&intersect)[d];
                     }
                     else   /* replace lower by upper */
                     {
                        nalu_hypre_BoxIMin(&intersect)[d] = nalu_hypre_BoxIMax(&intersect)[d];
                     }

                     nalu_hypre_AppendBox(&intersect,
                                     nalu_hypre_BoxArrayArrayBoxArray(cbdry, l));
                     cnt++; /* counter to signal boundary layers for cbox boxi */
                  }   /* if (nalu_hypre_BoxVolume(&intersect)) */
               }      /* nalu_hypre_ForBoxI(p, box_array) */
            }         /* nalu_hypre_ForBoxArrayI(l, fbdry) */

            /* All the boundary box_arrayarrays have been checked for coarse boxi.
               Now get the variable boundary layers if any, count the number of
               boundary points, and appropriately mark boxi. */
            if (cnt)
            {
               nalu_hypre_Maxwell_VarBdy(pgrid, bdry[i][boxi]);

               for (p = 0; p < nvars; p++)
               {
                  cbdry = bdry[i][boxi][p + 1];
                  nalu_hypre_ForBoxArrayI(l, cbdry)
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cbdry, l);
                     nalu_hypre_ForBoxI(m, box_array)
                     {
                        cbox = nalu_hypre_BoxArrayBox(box_array, m);
                        npts[i] += nalu_hypre_BoxVolume(cbox);
                     }
                  }
               }

               boxes_with_bdry[i][boxi] = 1; /* mark as containing boundary */
            }

         }  /* if (boxes_with_bdry[i-1][j]) */
      }     /* for (j= 0; j< nfboxes; j++) */

      nfboxes = nboxes;
   }  /* for (i= 1; i< num_levels; i++) */

   /* de-allocate objects that are not needed anymore */
   for (i = 0; i < (num_levels - 1); i++)
   {
      if (fbox_mapping[i])
      {
         nalu_hypre_TFree(fbox_mapping[i], NALU_HYPRE_MEMORY_HOST);
      }
      if (cbox_mapping[i + 1])
      {
         nalu_hypre_TFree(cbox_mapping[i + 1], NALU_HYPRE_MEMORY_HOST);
      }

      grid = grid_l[i + 1];
      pgrid = nalu_hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_cgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      cboxes = nalu_hypre_StructGridBoxes(cell_cgrid);
      nboxes = nalu_hypre_BoxArraySize(nalu_hypre_StructGridBoxes(cell_cgrid));
   }
   if (num_levels > 1)
   {
      nalu_hypre_TFree(fbox_mapping, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cbox_mapping, NALU_HYPRE_MEMORY_HOST);
   }

   /* find the ranks for the boundary points */
   BdryRanks_l    = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   BdryRanksCnts_l = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_levels, NALU_HYPRE_MEMORY_HOST);

   /* loop over levels and extract boundary ranks. Only extract unique
      ranks */
   for (i = 0; i < num_levels; i++)
   {
      grid = grid_l[i];
      pgrid = nalu_hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_cgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      cboxes = nalu_hypre_StructGridBoxes(cell_cgrid);
      nboxes = nalu_hypre_BoxArraySize(nalu_hypre_StructGridBoxes(cell_cgrid));

      ranks = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  npts[i], NALU_HYPRE_MEMORY_HOST);
      cnt = 0;
      for (j = 0; j < nboxes; j++)
      {
         if (boxes_with_bdry[i][j])
         {
            for (k = 0; k < nvars; k++)
            {
               fbdry = bdry[i][j][k + 1];

               nalu_hypre_ForBoxArrayI(m, fbdry)
               {
                  box_array = nalu_hypre_BoxArrayArrayBoxArray(fbdry, m);
                  nalu_hypre_ForBoxI(p, box_array)
                  {
                     box = nalu_hypre_BoxArrayBox(box_array, p);
                     nalu_hypre_BoxGetSize(box, loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(box), start);

                     nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                        nalu_hypre_AddIndexes(index, start, 3, index);

                        nalu_hypre_SStructGridFindBoxManEntry(grid, part, index,
                                                         k, &boxman_entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index,
                                                              &ranks[cnt], matrix_type);
                        cnt++;

                     }
                     nalu_hypre_SerialBoxLoop0End();
                  }  /* nalu_hypre_ForBoxI(p, box_array) */
               }     /* nalu_hypre_ForBoxArrayI(m, fbdry) */

            }  /* for (k= 0; k< nvars; k++) */
         } /* if (boxes_with_bdry[i][j]) */

         for (k = 0; k < nvars; k++)
         {
            nalu_hypre_BoxArrayArrayDestroy(bdry[i][j][k + 1]);
         }
         nalu_hypre_BoxArrayArrayDestroy(bdry[i][j][0]);
         nalu_hypre_TFree(bdry[i][j], NALU_HYPRE_MEMORY_HOST);

      }  /* for (j= 0; j< nboxes; j++) */
      nalu_hypre_TFree(bdry[i], NALU_HYPRE_MEMORY_HOST);

      /* mark all ranks that are outside this processor to -1 */
      for (j = 0; j < cnt; j++)
      {
         if ( (ranks[j] < lower_rank[i]) || (ranks[j] > upper_rank[i]) )
         {
            ranks[j] = -1;
         }
      }

      /* sort the ranks & extract the unique ones */
      if (cnt)  /* recall that some may not have bdry pts */
      {
         nalu_hypre_BigQsort0(ranks, 0, cnt - 1);

         k = 0;
         if (ranks[0] < 0) /* remove the off-processor markers */
         {
            for (j = 1; j < cnt; j++)
            {
               if (ranks[j] > -1)
               {
                  k = j;
                  break;
               }
            }
         }

         l = 1;
         for (j = k + 1; j < cnt; j++)
         {
            if (ranks[j] != ranks[j - 1])
            {
               l++;
            }
         }
         BdryRanks_l[i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  l, NALU_HYPRE_MEMORY_HOST);
         BdryRanksCnts_l[i] = l;

         l = 0;
         BdryRanks_l[i][l] = ranks[k] - lower_rank[i];
         for (j = k + 1; j < cnt; j++)
         {
            if (ranks[j] != ranks[j - 1])
            {
               l++;
               BdryRanks_l[i][l] = ranks[j] - lower_rank[i]; /* store local ranks */
            }
         }
      }

      else /* set BdryRanks_l[i] to be null */
      {
         BdryRanks_l[i] = NULL;
         BdryRanksCnts_l[i] = 0;
      }

      nalu_hypre_TFree(ranks, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(boxes_with_bdry[i], NALU_HYPRE_MEMORY_HOST);

   }  /* for (i= 0; i< num_levels; i++) */

   nalu_hypre_TFree(boxes_with_bdry, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(lower_rank, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(upper_rank, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(bdry, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(npts, NALU_HYPRE_MEMORY_HOST);

   *BdryRanksl_ptr    = BdryRanks_l;
   *BdryRanksCntsl_ptr = BdryRanksCnts_l;

   return ierr;
}

/*-----------------------------------------------------------------------------
 * Determine the variable boundary layers using the cell-centred boundary
 * layers. The cell-centred boundary layers are located in bdry[0], a
 * nalu_hypre_BoxArrayArray of size 2*ndim, one array for the upper side and one
 * for the lower side, for each direction.
 *-----------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_Maxwell_VarBdy( nalu_hypre_SStructPGrid       *pgrid,
                      nalu_hypre_BoxArrayArray     **bdry )
{
   NALU_HYPRE_Int              ierr = 0;
   NALU_HYPRE_Int              nvars = nalu_hypre_SStructPGridNVars(pgrid);

   nalu_hypre_BoxArrayArray   *cell_bdry = bdry[0];
   nalu_hypre_BoxArray        *box_array, *box_array2;
   nalu_hypre_Box             *bdy_box, *shifted_box;

   NALU_HYPRE_SStructVariable *vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);
   nalu_hypre_Index            varoffset, ishift, jshift, kshift;
   nalu_hypre_Index            lower, upper;

   NALU_HYPRE_Int              ndim = nalu_hypre_SStructPGridNDim(pgrid);
   NALU_HYPRE_Int              i, k, t;

   nalu_hypre_SetIndex3(ishift, 1, 0, 0);
   nalu_hypre_SetIndex3(jshift, 0, 1, 0);
   nalu_hypre_SetIndex3(kshift, 0, 0, 1);

   shifted_box = nalu_hypre_BoxCreate(ndim);
   for (i = 0; i < nvars; i++)
   {
      t = vartypes[i];
      nalu_hypre_SStructVariableGetOffset(vartypes[i], ndim, varoffset);
      switch (t)
      {
         case 2: /* xface, boundary i= lower, upper */
         {
            /* boundary i= lower */
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 0);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 0);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, varoffset, 3, lower);
                  nalu_hypre_SubtractIndexes(upper, varoffset, 3, upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary i= upper */
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 1);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 1);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }
            break;
         }

         case 3: /* yface, boundary j= lower, upper */
         {
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 2);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 0);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, varoffset, 3, lower);
                  nalu_hypre_SubtractIndexes(upper, varoffset, 3, upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 3);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 1);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }
            break;
         }

         case 5: /* xedge, boundary z_faces & y_faces */
         {
            /* boundary k= lower zface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 4);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 0);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, varoffset, 3, lower);
                  nalu_hypre_SubtractIndexes(upper, kshift, 3, upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary k= upper zface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 5);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 1);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, jshift, 3, lower);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary j= lower yface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 2);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 2);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, varoffset, 3, lower);
                  nalu_hypre_SubtractIndexes(upper, jshift, 3, upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary j= upper yface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 3);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 3);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, kshift, 3, lower);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }
            break;
         }

         case 6: /* yedge, boundary z_faces & x_faces */
         {
            /* boundary k= lower zface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 4);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 0);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, varoffset, 3, lower);
                  nalu_hypre_SubtractIndexes(upper, kshift, 3, upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary k= upper zface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 5);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 1);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, ishift, 3, lower);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary i= lower xface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 0);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 2);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, varoffset, 3, lower);
                  nalu_hypre_SubtractIndexes(upper, ishift, 3, upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary i= upper xface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 1);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 3);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, kshift, 3, lower);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }
            break;
         }

         case 7: /* zedge, boundary y_faces & x_faces */
         {
            /* boundary j= lower yface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 2);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 0);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, varoffset, 3, lower);
                  nalu_hypre_SubtractIndexes(upper, jshift, 3, upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary j= upper yface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 3);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 1);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, ishift, 3, lower);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary i= lower xface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 0);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 2);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, varoffset, 3, lower);
                  nalu_hypre_SubtractIndexes(upper, ishift, 3, upper);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }

            /* boundary i= upper xface*/
            box_array = nalu_hypre_BoxArrayArrayBoxArray(cell_bdry, 1);
            if (nalu_hypre_BoxArraySize(box_array))
            {
               box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[i + 1], 3);
               nalu_hypre_ForBoxI(k, box_array)
               {
                  bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                  nalu_hypre_SubtractIndexes(lower, jshift, 3, lower);

                  nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                  nalu_hypre_AppendBox(shifted_box, box_array2);
               }
            }
            break;
         }

      }  /* switch(t) */
   }     /* for (i= 0; i< nvars; i++) */

   nalu_hypre_BoxDestroy(shifted_box);

   return ierr;
}

