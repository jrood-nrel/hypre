/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Are private static arrays a problem?
 *
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fac.h"

#define MapStencilRank(stencil, rank)           \
   {                                            \
      NALU_HYPRE_Int ii,jj,kk;                       \
      ii = nalu_hypre_IndexX(stencil);               \
      jj = nalu_hypre_IndexY(stencil);               \
      kk = nalu_hypre_IndexZ(stencil);               \
      if (ii==-1)                               \
         ii=2;                                  \
      if (jj==-1)                               \
         jj=2;                                  \
      if (kk==-1)                               \
         kk=2;                                  \
      rank = ii + 3*jj + 9*kk;                  \
   }

#define InverseMapStencilRank(rank, stencil)    \
   {                                            \
      NALU_HYPRE_Int ij,ii,jj,kk;                    \
      ij = (rank%9);                            \
      ii = (ij%3);                              \
      jj = (ij-ii)/3;                           \
      kk = (rank-3*jj-ii)/9;                    \
      if (ii==2)                                \
         ii= -1;                                \
      if (jj==2)                                \
         jj= -1;                                \
      if (kk==2)                                \
         kk= -1;                                \
      nalu_hypre_SetIndex3(stencil, ii, jj, kk);     \
   }


#define AbsStencilShape(stencil, abs_shape)                     \
   {                                                            \
      NALU_HYPRE_Int ii,jj,kk;                                       \
      ii = nalu_hypre_IndexX(stencil);                               \
      jj = nalu_hypre_IndexY(stencil);                               \
      kk = nalu_hypre_IndexZ(stencil);                               \
      abs_shape= nalu_hypre_abs(ii) + nalu_hypre_abs(jj) + nalu_hypre_abs(kk); \
   }

/*--------------------------------------------------------------------------
 * nalu_hypre_AMR_CFCoarsen: Coarsens the CF interface to get the stencils
 * reaching into a coarsened fbox. Also sets the centre coefficient of CF
 * interface nodes to have "preserved" row sum.
 *
 * On entry, fac_A already has all the coefficient values of the cgrid
 * chunks that are not underlying a fbox.  Note that A & fac_A have the
 * same grid & graph. Therefore, we will use A's grid & graph.
 *
 * ASSUMING ONLY LIKE-VARIABLES COUPLE THROUGH CF CONNECTIONS.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMR_CFCoarsen( nalu_hypre_SStructMatrix  *   A,
                     nalu_hypre_SStructMatrix  *   fac_A,
                     nalu_hypre_Index              refine_factors,
                     NALU_HYPRE_Int                level )

{
   MPI_Comm                comm       = nalu_hypre_SStructMatrixComm(A);
   nalu_hypre_SStructGraph     *graph      = nalu_hypre_SStructMatrixGraph(A);
   NALU_HYPRE_Int               graph_type = nalu_hypre_SStructGraphObjectType(graph);
   nalu_hypre_SStructGrid      *grid       = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int               nUventries = nalu_hypre_SStructGraphNUVEntries(graph);
   NALU_HYPRE_IJMatrix          ij_A       = nalu_hypre_SStructMatrixIJMatrix(A);
   NALU_HYPRE_Int               matrix_type = nalu_hypre_SStructMatrixObjectType(A);
   NALU_HYPRE_Int               ndim       = nalu_hypre_SStructMatrixNDim(A);

   nalu_hypre_SStructPMatrix   *A_pmatrix;
   nalu_hypre_StructMatrix     *smatrix_var;
   nalu_hypre_StructStencil    *stencils;
   NALU_HYPRE_Int               stencil_size;
   nalu_hypre_Index             stencil_shape_i;
   nalu_hypre_Index             loop_size;
   nalu_hypre_Box               refined_box;
   NALU_HYPRE_Real            **a_ptrs;
   nalu_hypre_Box              *A_dbox;

   NALU_HYPRE_Int               part_crse = level - 1;
   NALU_HYPRE_Int               part_fine = level;

   nalu_hypre_BoxManager       *fboxman;
   nalu_hypre_BoxManEntry     **boxman_entries, *boxman_entry;
   NALU_HYPRE_Int               nboxman_entries;
   nalu_hypre_Box               boxman_entry_box;

   nalu_hypre_BoxArrayArray  ***fgrid_cinterface_extents;

   nalu_hypre_StructGrid       *cgrid;
   nalu_hypre_BoxArray         *cgrid_boxes;
   nalu_hypre_Box              *cgrid_box;
   nalu_hypre_Index             node_extents;
   nalu_hypre_Index             stridec, stridef;

   nalu_hypre_BoxArrayArray    *cinterface_arrays;
   nalu_hypre_BoxArray         *cinterface_array;
   nalu_hypre_Box              *fgrid_cinterface;

   NALU_HYPRE_Int               centre;

   NALU_HYPRE_Int               ci, fi, boxi;
   NALU_HYPRE_Int               max_stencil_size = 27;
   NALU_HYPRE_Int               falseV = 0;
   NALU_HYPRE_Int               trueV = 1;
   NALU_HYPRE_Int               found;
   NALU_HYPRE_Int              *stencil_ranks, *rank_stencils;
   NALU_HYPRE_BigInt            rank, startrank;
   NALU_HYPRE_Real             *vals;

   NALU_HYPRE_Int               i, j;
   NALU_HYPRE_Int               nvars, var1;

   nalu_hypre_Index             lindex, zero_index;
   nalu_hypre_Index             index1, index2;
   nalu_hypre_Index             index_temp;

   nalu_hypre_SStructUVEntry   *Uventry;
   NALU_HYPRE_Int               nUentries, cnt1;
   NALU_HYPRE_Int               box_array_size;

   NALU_HYPRE_Int              *ncols;
   NALU_HYPRE_BigInt           *rows, *cols;

   NALU_HYPRE_Int              *temp1, *temp2;

   NALU_HYPRE_Int               myid;

   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_SetIndex3(zero_index, 0, 0, 0);

   nalu_hypre_BoxInit(&refined_box, ndim);
   nalu_hypre_BoxInit(&boxman_entry_box, ndim);

   /*--------------------------------------------------------------------------
    *  Task: Coarsen the CF interface connections of A into fac_A so that
    *  fac_A will have the stencil coefficients extending into a coarsened
    *  fbox. The centre coefficient is constructed to preserve the row sum.
    *--------------------------------------------------------------------------*/

   if (graph_type == NALU_HYPRE_SSTRUCT)
   {
      startrank   = nalu_hypre_SStructGridGhstartRank(grid);
   }
   if (graph_type == NALU_HYPRE_PARCSR)
   {
      startrank   = nalu_hypre_SStructGridStartRank(grid);
   }

   /*--------------------------------------------------------------------------
    * Fine grid strides by the refinement factors.
    *--------------------------------------------------------------------------*/
   nalu_hypre_SetIndex3(stridec, 1, 1, 1);
   for (i = 0; i < ndim; i++)
   {
      stridef[i] = refine_factors[i];
   }
   for (i = ndim; i < 3; i++)
   {
      stridef[i] = 1;
   }

   /*--------------------------------------------------------------------------
    *  Determine the c/f interface index boxes: fgrid_cinterface_extents.
    *  These are between fpart= level and cpart= (level-1). The
    *  fgrid_cinterface_extents are indexed by cboxes, but fboxes that
    *  abutt a given cbox must be considered. Moreover, for each fbox,
    *  we can have a c/f interface from a number of different stencil
    *  directions- i.e., we have a boxarrayarray for each cbox, each
    *  fbox leading to a boxarray.
    *
    *  Algo.: For each cbox:
    *    1) refine & stretch by a unit in each dimension.
    *    2) boxman_intersect with the fgrid boxman to get all fboxes contained
    *       or abutting this cbox.
    *    3) get the fgrid_cinterface_extents for each of these fboxes.
    *
    *  fgrid_cinterface_extents[var1][ci]
    *--------------------------------------------------------------------------*/
   A_pmatrix =  nalu_hypre_SStructMatrixPMatrix(fac_A, part_crse);
   nvars    =  nalu_hypre_SStructPMatrixNVars(A_pmatrix);

   fgrid_cinterface_extents = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray **,  nvars, NALU_HYPRE_MEMORY_HOST);
   for (var1 = 0; var1 < nvars; var1++)
   {
      fboxman = nalu_hypre_SStructGridBoxManager(grid, part_fine, var1);
      stencils = nalu_hypre_SStructPMatrixSStencil(A_pmatrix, var1, var1);

      cgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
      fgrid_cinterface_extents[var1] = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray *,
                                                    nalu_hypre_BoxArraySize(cgrid_boxes), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

         nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(cgrid_box), zero_index,
                                     refine_factors, nalu_hypre_BoxIMin(&refined_box));
         nalu_hypre_SetIndex3(index1, refine_factors[0] - 1, refine_factors[1] - 1,
                         refine_factors[2] - 1);
         nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(cgrid_box), index1,
                                     refine_factors, nalu_hypre_BoxIMax(&refined_box));

         /*------------------------------------------------------------------------
          * Stretch the refined_box so that a BoxManIntersect will get abutting
          * fboxes.
          *------------------------------------------------------------------------*/
         for (i = 0; i < ndim; i++)
         {
            nalu_hypre_BoxIMin(&refined_box)[i] -= 1;
            nalu_hypre_BoxIMax(&refined_box)[i] += 1;
         }

         nalu_hypre_BoxManIntersect(fboxman, nalu_hypre_BoxIMin(&refined_box),
                               nalu_hypre_BoxIMax(&refined_box), &boxman_entries,
                               &nboxman_entries);

         fgrid_cinterface_extents[var1][ci] = nalu_hypre_BoxArrayArrayCreate(nboxman_entries, ndim);

         /*------------------------------------------------------------------------
          * Get the  fgrid_cinterface_extents using var1-var1 stencil (only like-
          * variables couple).
          *------------------------------------------------------------------------*/
         if (stencils != NULL)
         {
            for (i = 0; i < nboxman_entries; i++)
            {
               nalu_hypre_BoxManEntryGetExtents(boxman_entries[i],
                                           nalu_hypre_BoxIMin(&boxman_entry_box),
                                           nalu_hypre_BoxIMax(&boxman_entry_box));
               nalu_hypre_CFInterfaceExtents2(&boxman_entry_box, cgrid_box, stencils, refine_factors,
                                         nalu_hypre_BoxArrayArrayBoxArray(fgrid_cinterface_extents[var1][ci], i) );
            }
         }
         nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);

      }  /* nalu_hypre_ForBoxI(ci, cgrid_boxes) */
   }     /* for (var1= 0; var1< nvars; var1++) */

   /*--------------------------------------------------------------------------
    *  STEP 1:
    *        ADJUST THE ENTRIES ALONG THE C/F BOXES SO THAT THE COARSENED
    *        C/F CONNECTION HAS THE APPROPRIATE ROW SUM.
    *        WE ARE ASSUMING ONLY LIKE VARIABLES COUPLE.
    *--------------------------------------------------------------------------*/
   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
      stencils =  nalu_hypre_SStructPMatrixSStencil(A_pmatrix, var1, var1);

      /*----------------------------------------------------------------------
       * Extract only where variables couple.
       *----------------------------------------------------------------------*/
      if (stencils != NULL)
      {
         stencil_size = nalu_hypre_StructStencilSize(stencils);

         /*------------------------------------------------------------------
          *  stencil_ranks[i]      =  rank of stencil entry i.
          *  rank_stencils[i]      =  stencil entry of rank i.
          *
          * These are needed in collapsing the unstructured connections to
          * a stencil connection.
          *------------------------------------------------------------------*/
         stencil_ranks = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
         rank_stencils = nalu_hypre_TAlloc(NALU_HYPRE_Int,  max_stencil_size, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < max_stencil_size; i++)
         {
            rank_stencils[i] = -1;
            if (i < stencil_size)
            {
               stencil_ranks[i] = -1;
            }
         }

         for (i = 0; i < stencil_size; i++)
         {
            nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i), stencil_shape_i);
            MapStencilRank(stencil_shape_i, j);
            stencil_ranks[i] = j;
            rank_stencils[stencil_ranks[i]] = i;
         }
         centre = rank_stencils[0];

         smatrix_var = nalu_hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var1);

         a_ptrs   = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  stencil_size, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ForBoxI(ci, cgrid_boxes)
         {
            cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

            cinterface_arrays = fgrid_cinterface_extents[var1][ci];
            A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(smatrix_var), ci);

            /*-----------------------------------------------------------------
             * Ptrs to the correct data location.
             *-----------------------------------------------------------------*/
            for (i = 0; i < stencil_size; i++)
            {
               nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i), stencil_shape_i);
               a_ptrs[i] = nalu_hypre_StructMatrixExtractPointerByIndex(smatrix_var,
                                                                   ci,
                                                                   stencil_shape_i);
            }

            /*-------------------------------------------------------------------
             * Loop over the c/f interface boxes and set the centre to be the row
             * sum. Coarsen the c/f connection and set the centre to preserve
             * the row sum of the composite operator along the c/f interface.
             *-------------------------------------------------------------------*/
            nalu_hypre_ForBoxArrayI(fi, cinterface_arrays)
            {
               cinterface_array = nalu_hypre_BoxArrayArrayBoxArray(cinterface_arrays, fi);
               box_array_size  = nalu_hypre_BoxArraySize(cinterface_array);
               for (boxi = stencil_size; boxi < box_array_size; boxi++)
               {
                  fgrid_cinterface = nalu_hypre_BoxArrayBox(cinterface_array, boxi);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fgrid_cinterface), node_extents);
                  nalu_hypre_BoxGetSize(fgrid_cinterface, loop_size);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            A_dbox, node_extents, stridec, iA);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     for (i = 0; i < stencil_size; i++)
                     {
                        if (i != centre)
                        {
                           a_ptrs[centre][iA] += a_ptrs[i][iA];
                        }
                     }

                     /*-----------------------------------------------------------------
                      * Search for unstructured connections for this coarse node. Need
                      * to compute the index of the node. We will "collapse" the
                      * unstructured connections to the appropriate stencil entry. Thus
                      * we need to serch for the stencil entry.
                      *-----------------------------------------------------------------*/
                     index_temp[0] = node_extents[0] + lindex[0];
                     index_temp[1] = node_extents[1] + lindex[1];
                     index_temp[2] = node_extents[2] + lindex[2];

                     nalu_hypre_SStructGridFindBoxManEntry(grid, part_crse, index_temp, var1,
                                                      &boxman_entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index_temp, &rank,
                                                           matrix_type);
                     if (nUventries > 0)
                     {
                        found = falseV;
                        if ((rank - startrank) >= nalu_hypre_SStructGraphIUVEntry(graph, 0) &&
                            (rank - startrank) <= nalu_hypre_SStructGraphIUVEntry(graph, nUventries - 1))
                        {
                           found = trueV;
                        }
                     }

                     /*-----------------------------------------------------------------
                      * The graph has Uventries only if (nUventries > 0). Therefore,
                      * check this. Only like variables contribute to the row sum.
                      *-----------------------------------------------------------------*/
                     if (nUventries > 0 && found == trueV)
                     {
                        Uventry = nalu_hypre_SStructGraphUVEntry(graph, rank - startrank);

                        if (Uventry != NULL)
                        {
                           nUentries = nalu_hypre_SStructUVEntryNUEntries(Uventry);

                           /*-----------------------------------------------------------
                            * extract only the connections to level part_fine and the
                            * correct variable.
                            *-----------------------------------------------------------*/
                           temp1 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nUentries, NALU_HYPRE_MEMORY_HOST);
                           cnt1 = 0;
                           for (i = 0; i < nUentries; i++)
                           {
                              if (nalu_hypre_SStructUVEntryToPart(Uventry, i) == part_fine
                                  &&  nalu_hypre_SStructUVEntryToVar(Uventry, i) == var1)
                              {
                                 temp1[cnt1++] = i;
                              }
                           }

                           ncols = nalu_hypre_TAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
                           rows = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  cnt1, NALU_HYPRE_MEMORY_HOST);
                           cols = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  cnt1, NALU_HYPRE_MEMORY_HOST);
                           temp2 = nalu_hypre_TAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
                           vals = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  cnt1, NALU_HYPRE_MEMORY_HOST);

                           for (i = 0; i < cnt1; i++)
                           {
                              ncols[i] = 1;
                              rows[i] = rank;
                              cols[i] = nalu_hypre_SStructUVEntryToRank(Uventry, temp1[i]);

                              /* determine the stencil connection pattern */
                              nalu_hypre_StructMapFineToCoarse(
                                 nalu_hypre_SStructUVEntryToIndex(Uventry, temp1[i]),
                                 zero_index, stridef, index2);
                              nalu_hypre_SubtractIndexes(index2, index_temp,
                                                    ndim, index1);
                              MapStencilRank(index1, temp2[i]);

                              /* zero off this stencil connection into the fbox */
                              if (temp2[i] < max_stencil_size)
                              {
                                 j = rank_stencils[temp2[i]];
                                 if (j >= 0)
                                 {
                                    a_ptrs[j][iA] = 0.0;
                                 }
                              }
                           }  /* for (i= 0; i< cnt1; i++) */

                           nalu_hypre_TFree(temp1, NALU_HYPRE_MEMORY_HOST);

                           NALU_HYPRE_IJMatrixGetValues(ij_A, cnt1, ncols, rows, cols, vals);
                           for (i = 0; i < cnt1; i++)
                           {
                              a_ptrs[centre][iA] += vals[i];
                           }

                           nalu_hypre_TFree(ncols, NALU_HYPRE_MEMORY_HOST);
                           nalu_hypre_TFree(rows, NALU_HYPRE_MEMORY_HOST);
                           nalu_hypre_TFree(cols, NALU_HYPRE_MEMORY_HOST);

                           /* compute the connection to the coarsened fine box */
                           for (i = 0; i < cnt1; i++)
                           {
                              if (temp2[i] < max_stencil_size)
                              {
                                 j = rank_stencils[temp2[i]];
                                 if (j >= 0)
                                 {
                                    a_ptrs[j][iA] += vals[i];
                                 }
                              }
                           }
                           nalu_hypre_TFree(vals, NALU_HYPRE_MEMORY_HOST);
                           nalu_hypre_TFree(temp2, NALU_HYPRE_MEMORY_HOST);

                           /* centre connection which preserves the row sum */
                           for (i = 0; i < stencil_size; i++)
                           {
                              if (i != centre)
                              {
                                 a_ptrs[centre][iA] -= a_ptrs[i][iA];
                              }
                           }

                        }   /* if (Uventry != NULL) */
                     }       /* if (nUventries > 0) */
                  }
                  nalu_hypre_SerialBoxLoop1End(iA);
               }  /* for (boxi= stencil_size; boxi< box_array_size; boxi++) */
            }     /* nalu_hypre_ForBoxArrayI(fi, cinterface_arrays) */
         }        /* nalu_hypre_ForBoxI(ci, cgrid_boxes) */

         nalu_hypre_TFree(a_ptrs, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(stencil_ranks, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(rank_stencils, NALU_HYPRE_MEMORY_HOST);
      }   /* if (stencils != NULL) */
   }      /* end var1 */


   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         nalu_hypre_BoxArrayArrayDestroy(fgrid_cinterface_extents[var1][ci]);
      }
      nalu_hypre_TFree(fgrid_cinterface_extents[var1], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(fgrid_cinterface_extents, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

