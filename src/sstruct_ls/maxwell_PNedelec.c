/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Need to fix the way these variables are set and incremented in loops:
 *   j, k (only where they are listed at the end of SMP_PRIVATE)
 *
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

nalu_hypre_IJMatrix *
nalu_hypre_Maxwell_PNedelec( nalu_hypre_SStructGrid    *fgrid_edge,
                        nalu_hypre_SStructGrid    *cgrid_edge,
                        nalu_hypre_Index           rfactor    )
{
   MPI_Comm               comm = (fgrid_edge->  comm);

   NALU_HYPRE_IJMatrix         edge_Edge;

   nalu_hypre_SStructPGrid    *p_cgrid, *p_fgrid;
   nalu_hypre_StructGrid      *var_cgrid,  *var_fgrid;
   nalu_hypre_BoxArray        *cboxes, *fboxes, *box_array;
   nalu_hypre_Box             *cbox, *fbox, *cellbox, *vbox, copy_box;

   nalu_hypre_BoxArray       **contract_fedgeBoxes;
   nalu_hypre_Index          **Edge_cstarts, **upper_shifts, **lower_shifts;
   NALU_HYPRE_Int            **cfbox_mapping, **fcbox_mapping;

   nalu_hypre_BoxManEntry     *entry;
   NALU_HYPRE_BigInt           rank, rank2;
   NALU_HYPRE_BigInt           start_rank1, start_rank2;

   NALU_HYPRE_Int              nedges;

   NALU_HYPRE_BigInt          *iedgeEdge;
   NALU_HYPRE_BigInt          *jedge_Edge;

   NALU_HYPRE_Real            *vals_edgeEdge;
   NALU_HYPRE_Real             fCedge_ratio;
   NALU_HYPRE_Int             *ncols_edgeEdge;

   nalu_hypre_Index            cindex;
   nalu_hypre_Index            findex;
   nalu_hypre_Index            var_index, *boxoffset, *suboffset;
   nalu_hypre_Index            loop_size, start, cstart, stride, hi_index, lindex;
   nalu_hypre_Index            ishift, jshift, kshift, zero_index, one_index;
   NALU_HYPRE_Int              n_boxoffsets;

   NALU_HYPRE_Int              nparts = nalu_hypre_SStructGridNParts(fgrid_edge);
   NALU_HYPRE_Int              ndim  = nalu_hypre_SStructGridNDim(fgrid_edge);

   NALU_HYPRE_SStructVariable *vartypes, *Edge_vartypes;
   nalu_hypre_Index           *varoffsets;
   NALU_HYPRE_Int             *vartype_map;
   NALU_HYPRE_Int              matrix_type = NALU_HYPRE_PARCSR;

   NALU_HYPRE_Int              nvars, Edge_nvars, part, var;
   NALU_HYPRE_Int              tot_vars = 8;

   NALU_HYPRE_Int              t, i, j, k, m, n, size;
   NALU_HYPRE_BigInt           l, p;

   NALU_HYPRE_BigInt           ilower, iupper;
   NALU_HYPRE_BigInt           jlower, jupper;
   NALU_HYPRE_BigInt         **lower_ranks, **upper_ranks;

   NALU_HYPRE_Int           ***n_CtoVbox, ****CtoVboxnums;
   NALU_HYPRE_Int             *num_vboxes, **vboxnums;

   NALU_HYPRE_Int              trueV = 1;
   NALU_HYPRE_Int              falseV = 0;
   NALU_HYPRE_Int              row_in;

   NALU_HYPRE_Int              myproc;

   nalu_hypre_BoxInit(&copy_box, ndim);

   nalu_hypre_MPI_Comm_rank(comm, &myproc);
   nalu_hypre_SetIndex3(ishift, 1, 0, 0);
   nalu_hypre_SetIndex3(jshift, 0, 1, 0);
   nalu_hypre_SetIndex3(kshift, 0, 0, 1);
   nalu_hypre_SetIndex3(zero_index, 0, 0, 0);
   nalu_hypre_SetIndex3(one_index, 0, 0, 0);
   for (i = 0; i < ndim; i++)
   {
      one_index[i] = 1;
   }

   /* set rfactor[2]= 1 if ndim=2. */
   if (ndim == 2)
   {
      rfactor[2] = 1;
   }

   /*-------------------------------------------------------------------
    * Find the coarse-fine connection pattern, i.e., the topology
    * needed to create the interpolation operators.
    * These connections are determined using the cell-centred grids.
    * Note that we are assuming the variable type enumeration
    * given in nalu_hypre_SStructVariable_enum.
    *
    * We consider both 2-d and 3-d cases. In 2-d, the edges are faces.
    * We will continue to call them edges, but use the face variable
    * enumeration.
    *-------------------------------------------------------------------*/
   varoffsets = nalu_hypre_CTAlloc(nalu_hypre_Index,  tot_vars, NALU_HYPRE_MEMORY_HOST);

   /* total of 8 variable types. Create a mapping between user enumeration
      to hypre enumeration. Only need for edge grids. */
   vartype_map = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  tot_vars, NALU_HYPRE_MEMORY_HOST);

   part = 0;
   p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
   nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
   vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

   for (i = 0; i < nvars; i++)
   {
      t = vartypes[i];
      nalu_hypre_SStructVariableGetOffset((nalu_hypre_SStructVariable) t,
                                     ndim, varoffsets[t]);
      switch (t)
      {
         case 2:
         {
            vartype_map[2] = i;
            break;
         }

         case 3:
         {
            vartype_map[3] = i;
            break;
         }

         case 5:
         {
            vartype_map[5] = i;
            break;
         }

         case 6:
         {
            vartype_map[6] = i;
            break;
         }

         case 7:
         {
            vartype_map[7] = i;
            break;
         }
      }
   }

   /* local sizes */
   nedges   = 0;
   for (part = 0; part < nparts; part++)
   {
      /* same for 2-d & 3-d, assuming that fgrid_edge= fgrid_face in input */
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);    /* edge fgrid */
      nvars   = nalu_hypre_SStructPGridNVars(p_fgrid);

      for (var = 0; var < nvars; var++)
      {
         var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, var);
         nedges  += nalu_hypre_StructGridLocalSize(var_fgrid);
      }
   }

   /*--------------------------------------------------------------------------
    *  Form mappings between the c & f box numbers. Note that a cbox
    *  can land inside only one fbox since the latter was contracted. Without
    *  the extraction, a cbox can land in more than 1 fboxes (e.g., cbox
    *  boundary extending into other fboxes).
    *--------------------------------------------------------------------------*/
   cfbox_mapping = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nparts, NALU_HYPRE_MEMORY_HOST);
   fcbox_mapping = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nparts; i++)
   {
      p_fgrid  = nalu_hypre_SStructGridPGrid(fgrid_edge, i);
      var_fgrid = nalu_hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
      j        = nalu_hypre_BoxArraySize(fboxes);
      fcbox_mapping[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  j, NALU_HYPRE_MEMORY_HOST);

      p_cgrid  = nalu_hypre_SStructGridPGrid(cgrid_edge, i);
      var_cgrid = nalu_hypre_SStructPGridCellSGrid(p_cgrid);
      cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
      j        = nalu_hypre_BoxArraySize(fboxes);
      cfbox_mapping[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  j, NALU_HYPRE_MEMORY_HOST);

      /* assuming if i1 > i2 and (box j1) is coarsened from (box i1)
         and (box j2) from (box i2), then j1 > j2. */
      k = 0;
      nalu_hypre_ForBoxI(j, fboxes)
      {
         fbox = nalu_hypre_BoxArrayBox(fboxes, j);
         nalu_hypre_CopyBox(fbox, &copy_box);
         nalu_hypre_ProjectBox(&copy_box, zero_index, rfactor);
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(&copy_box), zero_index,
                                     rfactor, nalu_hypre_BoxIMin(&copy_box));
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(&copy_box), zero_index,
                                     rfactor, nalu_hypre_BoxIMax(&copy_box));

         /* since the ordering of the cboxes was determined by the fbox
            ordering, we only have to check if the first cbox in the
            list intersects with copy_box. If not, this fbox vanished in the
            coarsening. Note that this gives you the correct interior cbox. */
         cbox = nalu_hypre_BoxArrayBox(cboxes, k);
         nalu_hypre_IntersectBoxes(&copy_box, cbox, &copy_box);
         if (nalu_hypre_BoxVolume(&copy_box))
         {
            cfbox_mapping[i][k] = j;
            fcbox_mapping[i][j] = k;
            k++;
         }  /* if (nalu_hypre_BoxVolume(&copy_box)) */
      }     /* nalu_hypre_ForBoxI(j, fboxes) */
   }        /* for (i= 0; i< nparts; i++) */

   /* variable rank bounds for this processor */
   n_CtoVbox   = nalu_hypre_TAlloc(NALU_HYPRE_Int **,  nparts, NALU_HYPRE_MEMORY_HOST);
   CtoVboxnums = nalu_hypre_TAlloc(NALU_HYPRE_Int ***,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_SStructCellGridBoxNumMap(fgrid_edge, part, &n_CtoVbox[part],
                                     &CtoVboxnums[part]);
   }

   /* variable rank bounds for this processor */
   lower_ranks = nalu_hypre_TAlloc(NALU_HYPRE_BigInt *,  nparts, NALU_HYPRE_MEMORY_HOST);
   upper_ranks = nalu_hypre_TAlloc(NALU_HYPRE_BigInt *,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      p_fgrid  = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
      Edge_nvars = nalu_hypre_SStructPGridNVars(p_fgrid);

      lower_ranks[part] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  Edge_nvars, NALU_HYPRE_MEMORY_HOST);
      upper_ranks[part] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  Edge_nvars, NALU_HYPRE_MEMORY_HOST);
      for (t = 0; t < Edge_nvars; t++)
      {
         var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, t);
         box_array = nalu_hypre_StructGridBoxes(var_fgrid);

         fbox     = nalu_hypre_BoxArrayBox(box_array, 0);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fbox), findex);
         nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &lower_ranks[part][t],
                                               matrix_type);

         fbox = nalu_hypre_BoxArrayBox(box_array, nalu_hypre_BoxArraySize(box_array) - 1);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(fbox), findex);
         nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &upper_ranks[part][t],
                                               matrix_type);
      }
   }

   /* CREATE IJ_MATRICES- need to find the size of each one. Notice that the row
      and col ranks of these matrices can be created using only grid information.
      Grab the first part, first variable, first box, and lower index (lower rank);
      Grab the last part, last variable, last box, and upper index (upper rank). */

   /* edge_Edge. Same for 2-d and 3-d. */
   /* lower rank */
   start_rank1 = nalu_hypre_SStructGridStartRank(fgrid_edge);
   start_rank2 = nalu_hypre_SStructGridStartRank(cgrid_edge);
   ilower     = start_rank1;
   jlower     = start_rank2;

   /* upper rank */
   part = nparts - 1;
   p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
   nvars   = nalu_hypre_SStructPGridNVars(p_fgrid);
   var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, nvars - 1);
   fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
   fbox    = nalu_hypre_BoxArrayBox(fboxes, nalu_hypre_BoxArraySize(fboxes) - 1);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, nvars - 1,
                                           nalu_hypre_BoxArraySize(fboxes) - 1, myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(fbox), &iupper);

   p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
   nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
   var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, nvars - 1);
   cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox    = nalu_hypre_BoxArrayBox(cboxes, nalu_hypre_BoxArraySize(cboxes) - 1);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, nvars - 1,
                                           nalu_hypre_BoxArraySize(cboxes) - 1, myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(cbox), &jupper);

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &edge_Edge);
   NALU_HYPRE_IJMatrixSetObjectType(edge_Edge, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize(edge_Edge);

   /*-----------------------------------------------------------------------
    * edge_Edge, the actual interpolation matrix.
    * For each fine edge row, we need to know if it is a edge,
    * boundary edge, or face edge. Knowing this allows us to determine the
    * structure and weights of the interpolation matrix.
    * We assume that a coarse edge interpolates only to fine edges in or on
    * an agglomerate. That is, fine edges with indices that do were
    * truncated do not get interpolated to.
    * Scheme: Loop over fine edge grid. For each fine edge ijk,
    *     1) map it to a fine cell with the fine edge at the lower end
    *        of the box,e.g. x_edge[ijk] -> cell[i,j+1,k+1].
    *     2) coarsen the fine cell to obtain a coarse cell. Determine the
    *        location of the fine edge with respect to the coarse edges
    *        of this cell. Coarsening needed only when determining the
    *        column rank.
    * Need to distinguish between 2-d and 3-d.
    *-----------------------------------------------------------------------*/

   /* count the row/col connections */
   iedgeEdge     = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nedges, NALU_HYPRE_MEMORY_HOST);
   ncols_edgeEdge = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nedges, NALU_HYPRE_MEMORY_HOST);

   /* get the contracted boxes */
   contract_fedgeBoxes = nalu_hypre_TAlloc(nalu_hypre_BoxArray *,  nparts, NALU_HYPRE_MEMORY_HOST);
   Edge_cstarts = nalu_hypre_TAlloc(nalu_hypre_Index *,  nparts, NALU_HYPRE_MEMORY_HOST);
   upper_shifts = nalu_hypre_TAlloc(nalu_hypre_Index *,  nparts, NALU_HYPRE_MEMORY_HOST);
   lower_shifts = nalu_hypre_TAlloc(nalu_hypre_Index *,  nparts, NALU_HYPRE_MEMORY_HOST);

   for (part = 0; part < nparts; part++)
   {
      p_fgrid  = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
      var_fgrid = nalu_hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);

      /* fill up the contracted box_array */
      contract_fedgeBoxes[part] = nalu_hypre_BoxArrayCreate(0, ndim);
      Edge_cstarts[part] = nalu_hypre_TAlloc(nalu_hypre_Index,  nalu_hypre_BoxArraySize(fboxes), NALU_HYPRE_MEMORY_HOST);
      upper_shifts[part] = nalu_hypre_TAlloc(nalu_hypre_Index,  nalu_hypre_BoxArraySize(fboxes), NALU_HYPRE_MEMORY_HOST);
      lower_shifts[part] = nalu_hypre_TAlloc(nalu_hypre_Index,  nalu_hypre_BoxArraySize(fboxes), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_ForBoxI(i, fboxes)
      {
         fbox = nalu_hypre_BoxArrayBox(fboxes, i);

         /* contract the fbox to correspond to the correct cbox */
         cbox = nalu_hypre_BoxContraction(fbox, var_fgrid, rfactor);
         nalu_hypre_AppendBox(cbox, contract_fedgeBoxes[part]);

         /* record the offset mapping between the coarse cell index and
            the fine cell index */
         nalu_hypre_ClearIndex(upper_shifts[part][i]);
         nalu_hypre_ClearIndex(lower_shifts[part][i]);
         for (k = 0; k < ndim; k++)
         {
            m = nalu_hypre_BoxIMin(cbox)[k];
            p = m % rfactor[k];
            if (p > 0 && m > 0)
            {
               upper_shifts[part][i][k] = p - 1;
               lower_shifts[part][i][k] = p - rfactor[k];
            }
            else
            {
               upper_shifts[part][i][k] = rfactor[k] - p - 1;
               lower_shifts[part][i][k] = -p;
            }
         }

         /* record the cstarts of the cbox */
         nalu_hypre_ProjectBox(cbox, zero_index, rfactor);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cbox), Edge_cstarts[part][i]);
         nalu_hypre_StructMapFineToCoarse(Edge_cstarts[part][i], zero_index, rfactor,
                                     Edge_cstarts[part][i]);

         nalu_hypre_BoxDestroy(cbox);
      }

   }  /* for (part= 0; part< nparts; part++) */

   /*-----------------------------------------------------------------------
    * loop first over the fedges aligning with the agglomerate coarse edges.
    * Will loop over the face & interior edges separately also.
    *-----------------------------------------------------------------------*/
   j = 0;
   for (part = 0; part < nparts; part++)
   {
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
      Edge_nvars = nalu_hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_fgrid);

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes = contract_fedgeBoxes[part];

      for (t = 0; t < Edge_nvars; t++)
      {
         var         = Edge_vartypes[t];
         var_fgrid   = nalu_hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array   = nalu_hypre_StructGridBoxes(var_fgrid);

         n_boxoffsets = ndim - 1;
         boxoffset   = nalu_hypre_CTAlloc(nalu_hypre_Index,  n_boxoffsets, NALU_HYPRE_MEMORY_HOST);
         suboffset   = nalu_hypre_CTAlloc(nalu_hypre_Index,  n_boxoffsets, NALU_HYPRE_MEMORY_HOST);
         switch (var)
         {
            case 2: /* 2-d: x_face (vertical edges), stride=[rfactor[0],1,1] */
            {
               nalu_hypre_SetIndex3(stride, rfactor[0], 1, 1);
               nalu_hypre_CopyIndex(varoffsets[2], var_index);

               /* boxoffset shrink in the i direction */
               nalu_hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               nalu_hypre_SetIndex3(suboffset[0], 1, 0, 0);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 1, 0, 0);
               break;
            }

            case 3: /* 2-d: y_face (horizontal edges), stride=[1,rfactor[1],1] */
            {
               nalu_hypre_SetIndex3(stride, 1, rfactor[1], 1);
               nalu_hypre_CopyIndex(varoffsets[3], var_index);

               /* boxoffset shrink in the j direction */
               nalu_hypre_SetIndex3(boxoffset[0], 0, rfactor[1] - 1, 0);
               nalu_hypre_SetIndex3(suboffset[0], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 0, 1, 0);
               break;
            }

            case 5: /* 3-d: x_edge, stride=[1,rfactor[1],rfactor[2]] */
            {
               nalu_hypre_SetIndex3(stride, 1, rfactor[1], rfactor[2]);
               nalu_hypre_CopyIndex(varoffsets[5], var_index);

               /* boxoffset shrink in the j & k directions */
               nalu_hypre_SetIndex3(boxoffset[0], 0, rfactor[1] - 1, 0);
               nalu_hypre_SetIndex3(boxoffset[1], 0, 0, rfactor[2] - 1);
               nalu_hypre_SetIndex3(suboffset[0], 0, 1, 0);
               nalu_hypre_SetIndex3(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 0, 1, 1);
               break;
            }

            case 6: /* 3-d: y_edge, stride=[rfactor[0],1,rfactor[2]] */
            {
               nalu_hypre_SetIndex3(stride, rfactor[0], 1, rfactor[2]);
               nalu_hypre_CopyIndex(varoffsets[6], var_index);

               /* boxoffset shrink in the i & k directions */
               nalu_hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               nalu_hypre_SetIndex3(boxoffset[1], 0, 0, rfactor[2] - 1);
               nalu_hypre_SetIndex3(suboffset[0], 1, 0, 0);
               nalu_hypre_SetIndex3(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 1, 0, 1);
               break;
            }

            case 7: /* 3-d: z_edge, stride=[rfactor[0],rfactor[1],1] */
            {
               nalu_hypre_SetIndex3(stride, rfactor[0], rfactor[1], 1);
               nalu_hypre_CopyIndex(varoffsets[7], var_index);

               /* boxoffset shrink in the i & j directions */
               nalu_hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               nalu_hypre_SetIndex3(boxoffset[1], 0, rfactor[1] - 1, 0);
               nalu_hypre_SetIndex3(suboffset[0], 1, 0, 0);
               nalu_hypre_SetIndex3(suboffset[1], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 1, 1, 0);
               break;
            }
         }

         nalu_hypre_ForBoxI(i, fboxes)
         {
            cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

            /* vboxes inside the i'th cellbox */
            num_vboxes = n_CtoVbox[part][i];
            vboxnums  = CtoVboxnums[part][i];

            /* adjust the project cellbox to the variable box */
            nalu_hypre_CopyBox(cellbox, &copy_box);

            /* the adjusted variable box may be bigger than the actually
               variable box- variables that are shared may lead to smaller
               variable boxes than the SubtractIndex produces. If the box
               has to be decreased, then we decrease it by (rfactor[j]-1)
               in the appropriate direction.
               Check the location of the shifted lower box index. */
            for (k = 0; k < n_boxoffsets; k++)
            {
               nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), suboffset[k], 3,
                                     findex);
               row_in = falseV;
               for (p = 0; p < num_vboxes[t]; p++)
               {
                  vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);

                  if (nalu_hypre_IndexInBox(findex, vbox))
                  {
                     nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                     row_in = trueV;
                     break;
                  }
               }
               /* not in any vbox */
               if (!row_in)
               {
                  nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[k], 3,
                                   nalu_hypre_BoxIMin(&copy_box));
               }
            }

            nalu_hypre_BoxGetSize(&copy_box, loop_size);
            nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, stride,
                                        loop_size);
            /* extend the loop_size so that upper boundary of the box are reached. */
            nalu_hypre_AddIndexes(loop_size, hi_index, 3, loop_size);

            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

            nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                      &copy_box, start, stride, m);
            {
               zypre_BoxLoopGetIndex(lindex);
               nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
               for (k = 0; k < 3; k++)
               {
                  findex[k] *= stride[k];
               }
               nalu_hypre_AddIndexes(findex, start, 3, findex);

               nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t, &entry);
               nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &p, matrix_type);

               /* still row p may be outside the processor- check to make sure in */
               if ( (p <= upper_ranks[part][t]) && (p >= lower_ranks[part][t]) )
               {
                  iedgeEdge[j] = p;
                  ncols_edgeEdge[j] = 1;
                  j++;
               }
            }
            nalu_hypre_SerialBoxLoop1End(m);

         }   /* nalu_hypre_ForBoxI */

         nalu_hypre_TFree(boxoffset, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(suboffset, NALU_HYPRE_MEMORY_HOST);
      }  /* for (t= 0; t< nvars; t++) */
   }     /* for (part= 0; part< nparts; part++) */

   /*-----------------------------------------------------------------------
    * Record the row ranks for the face edges. Only for 3-d.
    * Loop over the face edges.
    *-----------------------------------------------------------------------*/
   if (ndim == 3)
   {
      for (part = 0; part < nparts; part++)
      {
         p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
         Edge_nvars = nalu_hypre_SStructPGridNVars(p_fgrid);
         Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_fgrid);

         /* note that fboxes are the contracted CELL boxes. Will get the correct
            variable grid extents. */
         fboxes = contract_fedgeBoxes[part];

         /* may need to shrink a given box in some boxoffset directions */
         boxoffset = nalu_hypre_TAlloc(nalu_hypre_Index,  ndim, NALU_HYPRE_MEMORY_HOST);
         for (t = 0; t < ndim; t++)
         {
            nalu_hypre_ClearIndex(boxoffset[t]);
            nalu_hypre_IndexD(boxoffset[t], t) = rfactor[t] - 1;
         }

         for (t = 0; t < Edge_nvars; t++)
         {
            var      = Edge_vartypes[t];
            var_fgrid = nalu_hypre_SStructPGridVTSGrid(p_fgrid, var);
            box_array = nalu_hypre_StructGridBoxes(var_fgrid);

            /* to reduce comparison, take the switch outside of the loop */
            switch (var)
            {
               case 5:
               {
                  /* 3-d x_edge, can be Y or Z_Face */
                  nalu_hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     /* adjust the contracted cellbox to the variable box */
                     nalu_hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         x_edge-> Z_Face & Y_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), kshift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[2], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), jshift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z plane direction */
                     loop_size[2]++;
                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }
                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /************************************************************
                         * Loop over the Z_Face x_edges.
                         ************************************************************/
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);

                              /* still row l may be outside the processor */
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* Z_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }

                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);

                     /* Y_Face */
                     nalu_hypre_CopyBox(cellbox, &copy_box);
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), jshift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[1], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), kshift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);
                     loop_size[1]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }
                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /* Y_Face */
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* Y_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }

                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);
                  }  /* nalu_hypre_ForBoxI(i, fboxes) */

                  break;
               }

               case 6:
               {
                  /* 3-d y_edge, can be X or Z_Face */
                  nalu_hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     /* adjust the project cellbox to the variable box */
                     nalu_hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         y_edge-> X_Face & Z_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      ******************************************************/
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), kshift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[2], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), ishift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z_Face direction to
                        cover upper boundary Z_Faces. */
                     loop_size[2]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }

                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /* Z_Face */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* Z_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }

                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);

                     /* X_Face */
                     nalu_hypre_CopyBox(cellbox, &copy_box);

                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), ishift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[0], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), kshift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     loop_size[0]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }

                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /* X_Face */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* X_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }

                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);
                  }  /* nalu_hypre_ForBoxI(i, fboxes) */

                  break;
               }

               case 7:
               {
                  /* 3-d z_edge, can be interior, X or Y_Face, or Z_Edge */
                  nalu_hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     /* adjust the project cellbox to the variable box */
                     nalu_hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         z_edge-> X_Face & Y_Face:
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), ishift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[0], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), jshift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the X_Face direction */
                     loop_size[0]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }

                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /* X_Face */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* X_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }
                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);

                     /* Y_Face */
                     nalu_hypre_CopyBox(cellbox, &copy_box);
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), jshift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[1], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), ishift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     loop_size[1]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }

                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /* Y_Face */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* Y_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }
                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);
                  }  /* nalu_hypre_ForBoxI(i, fboxes) */

                  break;
               }

            }  /* switch */
         }     /* for (t= 0; t< Edge_nvars; t++) */

         nalu_hypre_TFree(boxoffset, NALU_HYPRE_MEMORY_HOST);
      }  /* for (part= 0; part< nparts; part++) */
   }     /* if (ndim == 3) */

   for (part = 0; part < nparts; part++)
   {
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
      Edge_nvars = nalu_hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_fgrid);

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes = contract_fedgeBoxes[part];

      for (t = 0; t < Edge_nvars; t++)
      {
         var      = Edge_vartypes[t];
         var_fgrid = nalu_hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array = nalu_hypre_StructGridBoxes(var_fgrid);

         /* to reduce comparison, take the switch outside of the loop */
         switch (var)
         {
            case 2:
            {
               /* 2-d x_face = x_edge, can be interior */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                  /* adjust the contract cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /*nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }
                     nalu_hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[0]; p++)
                     {
                        nalu_hypre_CopyIndex(findex, var_index);
                        var_index[0] += p;
                        for (n = 0; n < rfactor[1]; n++)
                        {
                           nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                 matrix_type);
                           iedgeEdge[j] = l;

                           /* lies interior of Face. Two coarse Edge connection. */
                           ncols_edgeEdge[j] = 2;
                           j++;

                           var_index[1]++;
                        }  /* for (n= 0; n< rfactor[1]; n++) */
                     }     /* for (p= 1; p< rfactor[0]; p++) */

                  }
                  nalu_hypre_SerialBoxLoop1End(m);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 3:
            {
               /* 2-d y_face = y_edge, can be interior */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /* nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }

                     nalu_hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[1]; p++)
                     {
                        nalu_hypre_CopyIndex(findex, var_index);
                        var_index[1] += p;
                        for (n = 0; n < rfactor[0]; n++)
                        {
                           nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                 matrix_type);
                           iedgeEdge[j] = l;

                           /* lies interior of Face. Two coarse Edge connection. */
                           ncols_edgeEdge[j] = 2;
                           j++;

                           var_index[0]++;
                        }  /* for (n= 0; n< rfactor[0]; n++) */
                     }     /* for (p= 1; p< rfactor[1]; p++) */

                  }
                  nalu_hypre_SerialBoxLoop1End(m);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 5:
            {
               /* 3-d x_edge, can be only interior */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /* nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }
                     nalu_hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[2]; p++)
                     {
                        nalu_hypre_CopyIndex(findex, var_index);
                        var_index[2] += p;
                        for (n = 1; n < rfactor[1]; n++)
                        {
                           var_index[1]++;
                           for (k = 0; k < rfactor[0]; k++)
                           {
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              iedgeEdge[j] = l;

                              /* Interior. Four coarse Edge connections. */
                              ncols_edgeEdge[j] = 4;
                              j++;

                              var_index[0]++;
                           }  /* for (k= 0; k< rfactor[0]; k++) */

                           /* reset var_index[0] to the initial index for next k loop */
                           var_index[0] -= rfactor[0];

                        }  /* for (n= 1; n< rfactor[1]; n++) */

                        /* reset var_index[1] to the initial index for next n loop */
                        var_index[1] -= (rfactor[1] - 1);
                     }  /* for (p= 1; p< rfactor[2]; p++) */

                  }
                  nalu_hypre_SerialBoxLoop1End(m);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 6:
            {
               /* 3-d y_edge, can be only interior */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /* nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }
                     nalu_hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[2]; p++)
                     {
                        nalu_hypre_CopyIndex(findex, var_index);
                        var_index[2] += p;
                        for (n = 1; n < rfactor[0]; n++)
                        {
                           var_index[0]++;
                           for (k = 0; k < rfactor[1]; k++)
                           {
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              iedgeEdge[j] = l;

                              /* Interior. Four coarse Edge connections. */
                              ncols_edgeEdge[j] = 4;
                              j++;

                              var_index[1]++;
                           }  /* for (k= 0; k< rfactor[1]; k++) */

                           /* reset var_index[1] to the initial index for next k loop */
                           var_index[1] -= rfactor[1];

                        }  /* for (n= 1; n< rfactor[0]; n++) */

                        /* reset var_index[0] to the initial index for next n loop */
                        var_index[0] -= (rfactor[0] - 1);
                     }  /* for (p= 1; p< rfactor[2]; p++) */

                  }
                  nalu_hypre_SerialBoxLoop1End(m);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */

               break;
            }

            case 7:
            {
               /* 3-d z_edge, can be only interior */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /* nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }
                     nalu_hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[1]; p++)
                     {
                        nalu_hypre_CopyIndex(findex, var_index);
                        var_index[1] += p;
                        for (n = 1; n < rfactor[0]; n++)
                        {
                           var_index[0]++;
                           for (k = 0; k < rfactor[2]; k++)
                           {
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              iedgeEdge[j] = l;

                              /* Interior. Four coarse Edge connections. */
                              ncols_edgeEdge[j] = 4;
                              j++;

                              var_index[2]++;
                           }  /* for (k= 0; k< rfactor[2]; k++) */

                           /* reset var_index[2] to the initial index for next k loop */
                           var_index[2] -= rfactor[2];

                        }  /* for (n= 1; n< rfactor[0]; n++) */

                        /* reset var_index[0] to the initial index for next n loop */
                        var_index[0] -= (rfactor[0] - 1);
                     }  /* for (p= 1; p< rfactor[1]; p++) */
                  }
                  nalu_hypre_SerialBoxLoop1End(m);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */
               break;
            }

         }  /* switch */
      }     /* for (t= 0; t< Edge_nvars; t++) */
   }        /* for (part= 0; part< nparts; part++) */

   k = 0;
   j = 0;
   for (i = 0; i < nedges; i++)
   {
      if (ncols_edgeEdge[i])
      {
         k += ncols_edgeEdge[i];
         j++;
      }
   }
   vals_edgeEdge = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  k, NALU_HYPRE_MEMORY_HOST);
   jedge_Edge    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  k, NALU_HYPRE_MEMORY_HOST);

   /* update nedges so that the true number of rows is set */
   size = j;

   /*********************************************************************
    * Fill up the edge_Edge interpolation matrix. Interpolation weights
    * are determined differently for each type of fine edges.
    *********************************************************************/

   /* loop over fedges aligning with the agglomerate coarse edges first. */
   k = 0;
   for (part = 0; part < nparts; part++)
   {
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
      Edge_nvars = nalu_hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_fgrid);
      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part); /* Edge grid */

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes = contract_fedgeBoxes[part];

      for (t = 0; t < Edge_nvars; t++)
      {
         var      = Edge_vartypes[t];
         var_fgrid = nalu_hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array = nalu_hypre_StructGridBoxes(var_fgrid);

         n_boxoffsets = ndim - 1;
         boxoffset   = nalu_hypre_CTAlloc(nalu_hypre_Index,  n_boxoffsets, NALU_HYPRE_MEMORY_HOST);
         suboffset   = nalu_hypre_CTAlloc(nalu_hypre_Index,  n_boxoffsets, NALU_HYPRE_MEMORY_HOST);
         switch (var)
         {
            case 2: /* 2-d: x_face (vertical edges), stride=[rfactor[0],1,1]
                       fCedge_ratio= 1.0/rfactor[1] */
            {
               nalu_hypre_SetIndex3(stride, rfactor[0], 1, 1);
               fCedge_ratio = 1.0 / rfactor[1];

               /* boxoffset shrink in the i direction */
               nalu_hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               nalu_hypre_SetIndex3(suboffset[0], 1, 0, 0);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 1, 0, 0);
               break;
            }

            case 3: /* 2-d: y_face (horizontal edges), stride=[1,rfactor[1],1]
                       fCedge_ratio= 1.0/rfactor[0] */
            {
               nalu_hypre_SetIndex3(stride, 1, rfactor[1], 1);
               fCedge_ratio = 1.0 / rfactor[0];

               /* boxoffset shrink in the j direction */
               nalu_hypre_SetIndex3(boxoffset[0], 0, rfactor[1] - 1, 0);
               nalu_hypre_SetIndex3(suboffset[0], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 0, 1, 0);
               break;
            }

            case 5: /* 3-d: x_edge, stride=[1,rfactor[1],rfactor[2]]
                       fCedge_ratio= 1.0/rfactor[0] */
            {
               nalu_hypre_SetIndex3(stride, 1, rfactor[1], rfactor[2]);
               fCedge_ratio = 1.0 / rfactor[0];

               /* boxoffset shrink in the j & k directions */
               nalu_hypre_SetIndex3(boxoffset[0], 0, rfactor[1] - 1, 0);
               nalu_hypre_SetIndex3(boxoffset[1], 0, 0, rfactor[2] - 1);
               nalu_hypre_SetIndex3(suboffset[0], 0, 1, 0);
               nalu_hypre_SetIndex3(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 0, 1, 1);
               break;
            }

            case 6: /* 3-d: y_edge, stride=[rfactor[0],1,rfactor[2]]
                       fCedge_ratio= 1.0/rfactor[1] */
            {
               nalu_hypre_SetIndex3(stride, rfactor[0], 1, rfactor[2]);
               fCedge_ratio = 1.0 / rfactor[1];

               /* boxoffset shrink in the i & k directions */
               nalu_hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               nalu_hypre_SetIndex3(boxoffset[1], 0, 0, rfactor[2] - 1);
               nalu_hypre_SetIndex3(suboffset[0], 1, 0, 0);
               nalu_hypre_SetIndex3(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 1, 0, 1);
               break;
            }
            case 7: /* 3-d: z_edge, stride=[rfactor[0],rfactor[1],1]
                       fCedge_ratio= 1.0/rfactor[2] */
            {
               nalu_hypre_SetIndex3(stride, rfactor[0], rfactor[1], 1);
               fCedge_ratio = 1.0 / rfactor[2];

               /* boxoffset shrink in the i & j directions */
               nalu_hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               nalu_hypre_SetIndex3(boxoffset[1], 0, rfactor[1] - 1, 0);
               nalu_hypre_SetIndex3(suboffset[0], 1, 0, 0);
               nalu_hypre_SetIndex3(suboffset[1], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               nalu_hypre_SetIndex3(hi_index, 1, 1, 0);
               break;
            }
         }

         nalu_hypre_ForBoxI(i, fboxes)
         {
            cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

            /* vboxes inside the i'th cellbox */
            num_vboxes = n_CtoVbox[part][i];
            vboxnums  = CtoVboxnums[part][i];

            nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

            /* adjust the contracted cellbox to the variable box.
               Note that some of the fboxes may be skipped because they
               vanish. */
            nalu_hypre_CopyBox(cellbox, &copy_box);

            for (j = 0; j < n_boxoffsets; j++)
            {
               nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), suboffset[j], 3,
                                     findex);
               row_in = falseV;
               for (p = 0; p < num_vboxes[t]; p++)
               {
                  vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);

                  if (nalu_hypre_IndexInBox(findex, vbox))
                  {
                     nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                     row_in = trueV;
                     break;
                  }
               }
               /* not in any vbox */
               if (!row_in)
               {
                  nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[j], 3,
                                   nalu_hypre_BoxIMin(&copy_box));

                  /* also modify cstart */
                  nalu_hypre_AddIndexes(boxoffset[j], one_index, 3, boxoffset[j]);
                  nalu_hypre_StructMapFineToCoarse(boxoffset[j], zero_index, rfactor,
                                              boxoffset[j]);
                  nalu_hypre_AddIndexes(cstart, boxoffset[j], 3, cstart);
               }
            }

            nalu_hypre_BoxGetSize(&copy_box, loop_size);
            nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, stride,
                                        loop_size);

            /* extend the loop_size so that upper boundary of the box are reached. */
            nalu_hypre_AddIndexes(loop_size, hi_index, 3, loop_size);

            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

            /* note that the correct cbox corresponding to this non-vanishing
               fbox is used. */
            nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                      &copy_box, start, stride, m);
            {
               zypre_BoxLoopGetIndex(lindex);
               nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
               for (j = 0; j < 3; j++)
               {
                  findex[j] *= stride[j];
               }

               /* make sure that we do have the fine row corresponding to findex */
               nalu_hypre_AddIndexes(findex, start, 3, findex);
               nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t, &entry);
               nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &p, matrix_type);

               /* still row p may be outside the processor- check to make sure in */
               if ( (p <= upper_ranks[part][t]) && (p >= lower_ranks[part][t]) )
               {
                  nalu_hypre_SubtractIndexes(findex, start, 3, findex);

                  /* determine where the edge lies- coarsening required. */
                  nalu_hypre_StructMapFineToCoarse(findex, zero_index, rfactor,
                                              cindex);
                  nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                  /* lies on coarse Edge. Coarse Edge connection:
                     var_index= cindex - subtract_index.*/
                  nalu_hypre_SubtractIndexes(cindex, varoffsets[var], 3, var_index);

                  nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                   t, &entry);
                  nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                        matrix_type);
                  jedge_Edge[k] = l;
                  vals_edgeEdge[k] = fCedge_ratio;

                  k++;
               }  /* if ((p <= upper_ranks[part][t]) && (p >= lower_ranks[part][t])) */
            }
            nalu_hypre_SerialBoxLoop1End(m);
         }   /* nalu_hypre_ForBoxI */

         nalu_hypre_TFree(boxoffset, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(suboffset, NALU_HYPRE_MEMORY_HOST);
      }  /* for (t= 0; t< nvars; t++) */
   }     /* for (part= 0; part< nparts; part++) */

   /* generate the face interpolation weights/info. Only for 3-d */
   if (ndim == 3)
   {
      for (part = 0; part < nparts; part++)
      {
         p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
         Edge_nvars = nalu_hypre_SStructPGridNVars(p_fgrid);
         Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_fgrid);
         p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part); /* Edge grid */

         /* note that fboxes are the contracted CELL boxes. Will get the correct
            variable grid extents. */
         fboxes = contract_fedgeBoxes[part];

         /* may need to shrink a given box in some boxoffset directions */
         boxoffset = nalu_hypre_TAlloc(nalu_hypre_Index,  ndim, NALU_HYPRE_MEMORY_HOST);
         for (t = 0; t < ndim; t++)
         {
            nalu_hypre_ClearIndex(boxoffset[t]);
            nalu_hypre_IndexD(boxoffset[t], t) = rfactor[t] - 1;
         }

         for (t = 0; t < Edge_nvars; t++)
         {
            var      = Edge_vartypes[t];
            var_fgrid =  nalu_hypre_SStructPGridVTSGrid(p_fgrid, var);
            box_array = nalu_hypre_StructGridBoxes(var_fgrid);

            switch (var)
            {
               case 5:
               {
                  /* 3-d x_edge, can be Y or Z_Face */
                  nalu_hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     /* adjust the project cellbox to the variable box */
                     nalu_hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         x_edge-> Z_Face & Y_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), kshift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[2], 3,
                                         nalu_hypre_BoxIMin(&copy_box));

                        /* modify cstart */
                        nalu_hypre_AddIndexes(cstart, kshift, 3, cstart);
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), jshift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z plane direction */
                     loop_size[2]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        nalu_hypre_CopyIndex(findex, cindex);
                        nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /******************************************************
                         * ranks for coarse edges. Fine edges of agglomerate
                         * connect to these coarse edges.
                         * Z_Face (i,j,k-1). Two like-var coarse Edge connections.
                         * x_Edge (i,j,k-1), (i,j-1,k-1)
                         ******************************************************/
                        nalu_hypre_SubtractIndexes(cindex, kshift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        nalu_hypre_SubtractIndexes(var_index, jshift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of x_edges making up the Z_Face */
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);

                              /* still row l may be outside the processor */
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (NALU_HYPRE_Real) n / (rfactor[1] * rfactor[0]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[0] * (1.0 - (NALU_HYPRE_Real) n / rfactor[1]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);

                     /* Y plane direction */
                     nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);
                     nalu_hypre_CopyBox(cellbox, &copy_box);
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), jshift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[1], 3,
                                         nalu_hypre_BoxIMin(&copy_box));

                        /* modify cstart */
                        nalu_hypre_AddIndexes(cstart, jshift, 3, cstart);
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), kshift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     loop_size[1]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        nalu_hypre_CopyIndex(findex, cindex);
                        nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /******************************************************
                         * Y_Face. Two coarse Edge connections.
                         * x_Edge (i,j-1,k), (i,j-1,k-1)
                         ******************************************************/
                        nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of x_edges making up the Y_Face */
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (NALU_HYPRE_Real) n / (rfactor[0] * rfactor[2]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[0] * (1.0 - (NALU_HYPRE_Real) n / rfactor[2]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);
                  }  /* nalu_hypre_ForBoxI(i, fboxes) */
                  break;
               }

               case 6:
               {
                  /* 3-d y_edge, can be X or Z_Face */
                  nalu_hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     /* adjust the project cellbox to the variable box */
                     nalu_hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         y_edge-> X_Face & Z_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      ******************************************************/

                     /* Z_Face */
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), kshift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[2], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                        /* modify cstart */
                        nalu_hypre_AddIndexes(cstart, kshift, 3, cstart);
                     }

                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), ishift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z plane direction */
                     loop_size[2]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        nalu_hypre_CopyIndex(findex, cindex);
                        nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /******************************************************
                         * ranks for coarse edges. Fine edges of agglomerate
                         * connect to these coarse edges.
                         * Z_Face (i,j,k-1). Two like-var coarse Edge connections.
                         * y_Edge (i,j,k-1), (i-1,j,k-1)
                         ******************************************************/
                        nalu_hypre_SubtractIndexes(cindex, kshift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        nalu_hypre_SubtractIndexes(var_index, ishift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of y_edges making up the Z_Face */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (NALU_HYPRE_Real) n / (rfactor[0] * rfactor[1]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[1] * (1.0 - (NALU_HYPRE_Real) n / rfactor[0]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);

                     /* X_Face */
                     nalu_hypre_CopyBox(cellbox, &copy_box);
                     nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), ishift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[0], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                        /* modify cstart */
                        nalu_hypre_AddIndexes(cstart, ishift, 3, cstart);
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), kshift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     loop_size[0]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        nalu_hypre_CopyIndex(findex, cindex);
                        nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        nalu_hypre_AddIndexes(findex, start, 3, findex);
                        /******************************************************
                         * X_Face. Two coarse Edge connections.
                         * y_Edge (i-1,j,k), (i-1,j,k-1)
                         ******************************************************/
                        nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of y_edges making up the X_Face */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (NALU_HYPRE_Real) n / (rfactor[1] * rfactor[2]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[1] * (1.0 - (NALU_HYPRE_Real) n / rfactor[2]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */

                     }
                     nalu_hypre_SerialBoxLoop1End(m);
                  }  /* nalu_hypre_ForBoxI(i, fboxes) */
                  break;
               }

               case 7:
               {
                  /* 3-d z_edge, can be X or Y_Face */
                  nalu_hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = nalu_hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     /* adjust the project cellbox to the variable box */
                     nalu_hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         z_edge-> X_Face & Y_Face:
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), ishift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[0], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                        /* modify cstart */
                        nalu_hypre_AddIndexes(cstart, ishift, 3, cstart);
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), jshift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the X plane direction */
                     loop_size[0]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        nalu_hypre_CopyIndex(findex, cindex);
                        nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        nalu_hypre_AddIndexes(findex, start, 3, findex);

                        /******************************************************
                         * ranks for coarse edges. Fine edges of agglomerate
                         * connect to these coarse edges.
                         * X_Face. Two coarse Edge connections.
                         * z_Edge (i-1,j,k), (i-1,j-1,k)
                         ******************************************************/
                        nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        nalu_hypre_SubtractIndexes(var_index, jshift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of z_edges making up the X_Face */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (NALU_HYPRE_Real) n / (rfactor[0] * rfactor[2]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[2] * (1.0 - (NALU_HYPRE_Real) n / rfactor[0]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);

                     /* Y plane */
                     nalu_hypre_CopyBox(cellbox, &copy_box);
                     nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), jshift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = nalu_hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (nalu_hypre_IndexInBox(findex, vbox))
                        {
                           nalu_hypre_CopyIndex(findex, nalu_hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&copy_box), boxoffset[1], 3,
                                         nalu_hypre_BoxIMin(&copy_box));
                        /* modify cstart */
                        nalu_hypre_AddIndexes(cstart, jshift, 3, cstart);
                     }
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), ishift, 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     loop_size[1]++;

                     nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        nalu_hypre_CopyIndex(findex, cindex);
                        nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        nalu_hypre_AddIndexes(findex, start, 3, findex);
                        /**********************************************************
                         * Y_Face (i,j-1,k). Two like-var coarse Edge connections.
                         * z_Edge (i,j-1,k), (i-1,j-1,k)
                         **********************************************************/
                        nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        nalu_hypre_SubtractIndexes(var_index, ishift, 3, var_index);
                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of y_edges making up the Y_Face */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (NALU_HYPRE_Real) n / (rfactor[0] * rfactor[2]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[2] * (1.0 - (NALU_HYPRE_Real) n / rfactor[0]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */

                     }
                     nalu_hypre_SerialBoxLoop1End(m);
                  }  /* nalu_hypre_ForBoxI(i, fboxes) */
                  break;
               }

            }  /* switch */
         }     /* for (t= 0; t< Edge_nvars; t++) */
         nalu_hypre_TFree(boxoffset, NALU_HYPRE_MEMORY_HOST);
      }  /* for (part= 0; part< nparts; part++) */
   }     /* if (ndim == 3) */

   /* generate the interior interpolation weights/info */
   for (part = 0; part < nparts; part++)
   {
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
      Edge_nvars = nalu_hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_fgrid);
      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part); /* Edge grid */

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes = contract_fedgeBoxes[part];

      for (t = 0; t < Edge_nvars; t++)
      {
         var      = Edge_vartypes[t];
         var_fgrid =  nalu_hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array = nalu_hypre_StructGridBoxes(var_fgrid);

         switch (var)
         {
            case 2:
            {
               /* 2-d x_face = x_edge, can be interior or on X_Edge */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);
                  nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /* nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[0]; p++)
                     {
                        for (n = 0; n < rfactor[1]; n++)
                        {
                           nalu_hypre_CopyIndex(findex, cindex);
                           nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                           /*interior of Face. Extract the two coarse Edge
                             (x_Edge ijk & (i-1,j,k)*/
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           vals_edgeEdge[k] = (NALU_HYPRE_Real) p / (rfactor[0] * rfactor[1]);
                           k++;

                           nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           vals_edgeEdge[k] = (NALU_HYPRE_Real) (rfactor[0] - p) / (rfactor[0] * rfactor[1]);
                           k++;
                        }  /* for (n= 0; n< rfactor[1]; n++) */
                     }     /* for (p= 1; p< rfactor[0]; p++) */

                  }
                  nalu_hypre_SerialBoxLoop1End(r);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 3:
            {
               /* 2-d y_face = y_edge, can be interior or on Y_Edge */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);
                  nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /* nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[1]; p++)
                     {
                        for (n = 0; n < rfactor[0]; n++)
                        {
                           nalu_hypre_CopyIndex(findex, cindex);
                           nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                           /*lies interior of Face. Extract the two coarse Edge
                             (y_Edge ijk & (i,j-1,k). */
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           vals_edgeEdge[k] = (NALU_HYPRE_Real) p / (rfactor[0] * rfactor[1]);
                           k++;

                           nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           vals_edgeEdge[k] = (NALU_HYPRE_Real) (rfactor[1] - p) / (rfactor[0] * rfactor[1]);
                           k++;
                        }  /* for (n= 0; n< rfactor[0]; n++) */
                     }     /* for (p= 1; p< rfactor[1]; p++) */

                  }
                  nalu_hypre_SerialBoxLoop1End(r);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 5:
            {
               /* 3-d x_edge, must be interior */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);
                  nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /*nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[2]; p++)
                     {
                        for (n = 1; n < rfactor[1]; n++)
                        {
                           for (m = 0; m < rfactor[0]; m++)
                           {
                              nalu_hypre_CopyIndex(findex, cindex);
                              nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                              /***********************************************
                               * Interior.
                               * x_Edge ijk, (i,j-1,k), (i,j-1,k-1), (i,j,k-1)
                               ***********************************************/
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) p * n /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);

                              k++;

                              nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) p * (rfactor[1] - n) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) (rfactor[1] - n) * (rfactor[2] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              nalu_hypre_AddIndexes(var_index, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) n * (rfactor[2] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;
                           }  /* for (m= 0; m< rfactor[0]; m++) */
                        }     /* for (n= 1; n< rfactor[1]; n++) */
                     }        /* for (p= 1; p< rfactor[2]; p++) */
                  }
                  nalu_hypre_SerialBoxLoop1End(r);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 6:
            {
               /* 3-d y_edge, must be interior */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);
                  nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /*nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[2]; p++)
                     {
                        for (n = 1; n < rfactor[0]; n++)
                        {
                           for (m = 0; m < rfactor[1]; m++)
                           {
                              nalu_hypre_CopyIndex(findex, cindex);
                              nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                              /***********************************************
                               * Interior.
                               * y_Edge ijk, (i-1,j,k), (i-1,j,k-1), (i,j,k-1)
                               ***********************************************/
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) p * n /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) p * (rfactor[0] - n) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) (rfactor[0] - n) * (rfactor[2] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              nalu_hypre_AddIndexes(var_index, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) n * (rfactor[2] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;
                           }  /* for (m= 0; m< rfactor[1]; m++) */
                        }     /* for (n= 1; n< rfactor[0]; n++) */
                     }        /* for (p= 1; p< rfactor[2]; p++) */

                  }
                  nalu_hypre_SerialBoxLoop1End(r);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 7:
            {
               /* 3-d z_edge, only the interior */
               nalu_hypre_ForBoxI(i, fboxes)
               {
                  cellbox = nalu_hypre_BoxArrayBox(fboxes, i);
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);
                  nalu_hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));
                  /*nalu_hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  nalu_hypre_BoxGetSize(&copy_box, loop_size);
                  nalu_hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                  nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[1]; p++)
                     {
                        for (n = 1; n < rfactor[0]; n++)
                        {
                           for (m = 0; m < rfactor[2]; m++)
                           {
                              nalu_hypre_CopyIndex(findex, cindex);
                              nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

                              /*************************************************
                               * Interior.
                               * z_Edge ijk, (i-1,j,k), (i-1,j-1,k), (i,j-1,k)
                               *************************************************/
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) n * p /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) p * (rfactor[0] - n) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) (rfactor[1] - p) * (rfactor[0] - n) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              nalu_hypre_AddIndexes(var_index, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (NALU_HYPRE_Real) n * (rfactor[1] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;
                           }  /* for (m= 0; m< rfactor[2]; m++) */
                        }     /* for (n= 1; n< rfactor[0]; n++) */
                     }        /* for (p= 1; p< rfactor[1]; p++) */
                  }
                  nalu_hypre_SerialBoxLoop1End(r);
               }  /* nalu_hypre_ForBoxI(i, fboxes) */
               break;
            }
         }  /* switch */
      }     /* for (t= 0; t< Edge_nvars; t++) */
   }        /* for (part= 0; part< nparts; part++) */

   NALU_HYPRE_IJMatrixSetValues(edge_Edge, size, ncols_edgeEdge,
                           (const NALU_HYPRE_BigInt*) iedgeEdge, (const NALU_HYPRE_BigInt*) jedge_Edge,
                           (const NALU_HYPRE_Real*) vals_edgeEdge);
   NALU_HYPRE_IJMatrixAssemble((NALU_HYPRE_IJMatrix) edge_Edge);

   nalu_hypre_TFree(ncols_edgeEdge, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(iedgeEdge, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jedge_Edge, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vals_edgeEdge, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(varoffsets, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vartype_map, NALU_HYPRE_MEMORY_HOST);

   /* n_CtoVbox[part][cellboxi][var]  & CtoVboxnums[part][cellboxi][var][nvboxes] */
   for (part = 0; part < nparts; part++)
   {
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
      Edge_nvars = nalu_hypre_SStructPGridNVars(p_fgrid);

      var_fgrid = nalu_hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
      nalu_hypre_ForBoxI(j, fboxes)
      {
         for (t = 0; t < Edge_nvars; t++)
         {
            nalu_hypre_TFree(CtoVboxnums[part][j][t], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(n_CtoVbox[part][j], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(CtoVboxnums[part][j], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(n_CtoVbox[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(CtoVboxnums[part], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(n_CtoVbox, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CtoVboxnums, NALU_HYPRE_MEMORY_HOST);

   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_BoxArrayDestroy(contract_fedgeBoxes[part]);
      nalu_hypre_TFree(Edge_cstarts[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(upper_shifts[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(lower_shifts[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cfbox_mapping[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fcbox_mapping[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(upper_ranks[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(lower_ranks[part], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(contract_fedgeBoxes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Edge_cstarts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(upper_shifts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(lower_shifts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cfbox_mapping, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fcbox_mapping, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(upper_ranks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(lower_ranks, NALU_HYPRE_MEMORY_HOST);

   return (nalu_hypre_IJMatrix *) edge_Edge;
}

