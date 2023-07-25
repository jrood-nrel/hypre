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
 *   nElements, nElements_iedges, nFaces, nFaces_iedges, nEdges, nEdges_iedges,
 *   nElements_Faces, nElements_Edges,
 *   j, l, k (these three only where they are listed at the end of SMP_PRIVATE)
 *
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_Maxwell_Interp.c
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_CreatePTopology(void **PTopology_vdata_ptr)
{
   nalu_hypre_PTopology   *PTopology;
   NALU_HYPRE_Int          ierr = 0;

   PTopology = nalu_hypre_CTAlloc(nalu_hypre_PTopology, 1, NALU_HYPRE_MEMORY_HOST);

   (PTopology ->  Face_iedge)   = NULL;
   (PTopology ->  Element_iedge) = NULL;
   (PTopology ->  Edge_iedge)   = NULL;

   (PTopology ->  Element_Face) = NULL;
   (PTopology ->  Element_Edge) = NULL;

   *PTopology_vdata_ptr = (void *) PTopology;

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_DestroyPTopology(void *PTopology_vdata)
{
   nalu_hypre_PTopology       *PTopology = (nalu_hypre_PTopology       *)PTopology_vdata;
   NALU_HYPRE_Int              ierr     = 0;

   if (PTopology)
   {
      if ( (PTopology -> Face_iedge) != NULL)
      {
         NALU_HYPRE_IJMatrixDestroy(PTopology -> Face_iedge);
      }
      NALU_HYPRE_IJMatrixDestroy(PTopology -> Element_iedge);
      NALU_HYPRE_IJMatrixDestroy(PTopology -> Edge_iedge);

      if ( (PTopology -> Element_Face) != NULL)
      {
         NALU_HYPRE_IJMatrixDestroy(PTopology -> Element_Face);
      }
      NALU_HYPRE_IJMatrixDestroy(PTopology -> Element_Edge);
   }
   nalu_hypre_TFree(PTopology, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

nalu_hypre_IJMatrix *
nalu_hypre_Maxwell_PTopology(  nalu_hypre_SStructGrid    *fgrid_edge,
                          nalu_hypre_SStructGrid    *cgrid_edge,
                          nalu_hypre_SStructGrid    *fgrid_face,
                          nalu_hypre_SStructGrid    *cgrid_face,
                          nalu_hypre_SStructGrid    *fgrid_element,
                          nalu_hypre_SStructGrid    *cgrid_element,
                          nalu_hypre_ParCSRMatrix   *Aee,
                          nalu_hypre_Index           rfactor,
                          void                 *PTopology_vdata)
{
   MPI_Comm               comm = (fgrid_element ->  comm);

   nalu_hypre_PTopology       *PTopology = (nalu_hypre_PTopology *) PTopology_vdata;

   nalu_hypre_IJMatrix        *Face_iedge;
   nalu_hypre_IJMatrix        *Element_iedge;
   nalu_hypre_IJMatrix        *Edge_iedge;

   nalu_hypre_IJMatrix        *Element_Face;
   nalu_hypre_IJMatrix        *Element_Edge;

   nalu_hypre_IJMatrix        *edge_Edge;

   nalu_hypre_SStructPGrid    *p_cgrid, *p_fgrid;
   nalu_hypre_StructGrid      *var_cgrid,  *var_fgrid;
   nalu_hypre_BoxArray        *cboxes, *fboxes, *box_array;
   nalu_hypre_Box             *cbox, *fbox, *cellbox, *vbox, copy_box;

   nalu_hypre_BoxArray       **contract_fedgeBoxes;
   nalu_hypre_Index          **Edge_cstarts, **upper_shifts, **lower_shifts;
   NALU_HYPRE_Int            **cfbox_mapping, **fcbox_mapping;

   nalu_hypre_BoxManEntry     *entry;
   NALU_HYPRE_BigInt           rank, rank2;
   NALU_HYPRE_BigInt           start_rank1;

   NALU_HYPRE_Int              nFaces, nEdges, nElements, nedges;
   NALU_HYPRE_Int              nxFaces, nyFaces, nzFaces;
   /* NALU_HYPRE_Int              nxEdges, nyEdges, nzEdges; */
   NALU_HYPRE_Int              n_xFace_iedges, n_yFace_iedges, n_zFace_iedges;
   NALU_HYPRE_Int              n_Cell_iedges;

   NALU_HYPRE_Int              nElements_iedges, nFaces_iedges, nEdges_iedges;
   NALU_HYPRE_Int              nElements_Faces, nElements_Edges;

   NALU_HYPRE_BigInt          *iFace, *iEdge;
   NALU_HYPRE_BigInt          *jFace_edge;
   NALU_HYPRE_BigInt          *jEdge_iedge;
   NALU_HYPRE_BigInt          *jElement_Face, *jedge_Edge;
   NALU_HYPRE_BigInt          *iElement, *jElement_Edge, *iedgeEdge, *jElement_edge;

   NALU_HYPRE_Real            *vals_ElementEdge, *vals_ElementFace, *vals_edgeEdge, *vals_Faceedge;
   NALU_HYPRE_Real            *vals_Elementedge, *vals_Edgeiedge;
   NALU_HYPRE_Int             *ncols_Elementedge, *ncols_Edgeiedge, *ncols_edgeEdge, *ncols_Faceedge;
   NALU_HYPRE_Int             *ncols_ElementFace, *ncols_ElementEdge;
   NALU_HYPRE_Int             *bdryedge_location;
   NALU_HYPRE_Real             fCedge_ratio;
   NALU_HYPRE_Real            *stencil_vals, *upper, *lower, *diag, *face_w1, *face_w2;
   NALU_HYPRE_Int             *off_proc_flag;

   nalu_hypre_Index            cindex;
   nalu_hypre_Index            findex;
   nalu_hypre_Index            var_index, cell_index, *boxoffset, *suboffset;
   nalu_hypre_Index            loop_size, start, cstart, stride, low_index, hi_index;
   nalu_hypre_Index            ishift, jshift, kshift, zero_index, one_index;
   nalu_hypre_Index            lindex;
   NALU_HYPRE_Int              n_boxoffsets;

   NALU_HYPRE_Int              nparts = nalu_hypre_SStructGridNParts(fgrid_element);
   NALU_HYPRE_Int              ndim  = nalu_hypre_SStructGridNDim(fgrid_element);

   NALU_HYPRE_SStructVariable *vartypes, *Edge_vartypes, *Face_vartypes;
   nalu_hypre_Index           *varoffsets;
   NALU_HYPRE_Int             *vartype_map;
   NALU_HYPRE_Int              matrix_type = NALU_HYPRE_PARCSR;

   NALU_HYPRE_Int              nvars, Face_nvars, Edge_nvars, part, var, box, fboxi;
   NALU_HYPRE_Int              tot_vars = 8;

   NALU_HYPRE_Int              t, i, j, k, l, m, n, p;

   NALU_HYPRE_BigInt           ilower, iupper;
   NALU_HYPRE_BigInt           jlower, jupper;
   NALU_HYPRE_BigInt         **flower_ranks, **fupper_ranks;
   NALU_HYPRE_BigInt         **clower_ranks, **cupper_ranks;
   NALU_HYPRE_Int           ***n_CtoVbox, ****CtoVboxnums;
   NALU_HYPRE_Int             *num_vboxes, **vboxnums;

   NALU_HYPRE_Int              size1;
   NALU_HYPRE_Int              trueV = 1;
   NALU_HYPRE_Int              falseV = 0;
   NALU_HYPRE_Int              row_in;

   NALU_HYPRE_Int              myproc;

   NALU_HYPRE_MemoryLocation   memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(Aee);

   nalu_hypre_BoxInit(&copy_box, ndim);

   nalu_hypre_MPI_Comm_rank(comm, &myproc);
   nalu_hypre_SetIndex3(ishift, 1, 0, 0);
   nalu_hypre_SetIndex3(jshift, 0, 1, 0);
   nalu_hypre_SetIndex3(kshift, 0, 0, 1);
   nalu_hypre_ClearIndex(zero_index);
   nalu_hypre_ClearIndex(one_index);
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
    * Face_iedge, Edge_iedge, Element_iedge, Element_Face, Element_Edge,
    * and edge_Edge connections are defined in terms of parcsr_matrices.
    * These connections are determined using the cell-centred grids.
    * Note that we are assuming the variable type enumeration
    * given in nalu_hypre_SStructVariable_enum.
    *
    * We consider both 2-d and 3-d cases. In 2-d, the edges are faces.
    * We will continue to call them edges, but use the face variable
    * enumeration.
    *-------------------------------------------------------------------*/
   varoffsets = nalu_hypre_CTAlloc(nalu_hypre_Index, tot_vars, NALU_HYPRE_MEMORY_HOST);

   /* total of 8 variable types. Create a mapping between user enumeration
      to hypre enumeration. Only need for face and edge grids. */
   vartype_map = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 8, NALU_HYPRE_MEMORY_HOST);

   part = 0;
   p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_face, part);   /* face cgrid */
   nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
   vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

   for (i = 0; i < nvars; i++)
   {
      t = vartypes[i];
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

         case 4:
         {
            vartype_map[4] = i;
            break;
         }
      }
   }

   if (ndim == 3)
   {
      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);   /* edge cgrid */
      nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
      vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

      for (i = 0; i < nvars; i++)
      {
         t = vartypes[i];
         switch (t)
         {
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
   }

   /* local sizes */
   nFaces   = 0;
   nEdges   = 0;
   nElements = 0;
   nedges   = 0;

   nxFaces  = 0;
   nyFaces  = 0;
   nzFaces  = 0;
   /* nxEdges  = 0; */
   /* nyEdges  = 0; */
   /* nzEdges  = 0; */

   for (part = 0; part < nparts; part++)
   {
      p_cgrid   = nalu_hypre_SStructGridPGrid(cgrid_element, part);  /* cell cgrid */
      var_cgrid = nalu_hypre_SStructPGridCellSGrid(p_cgrid) ;
      nElements += nalu_hypre_StructGridLocalSize(var_cgrid);

      t = 0;
      nalu_hypre_SStructVariableGetOffset((nalu_hypre_SStructVariable) t,
                                     ndim, varoffsets[0]);

      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_face, part);       /* face cgrid */
      nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
      vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

      for (var = 0; var < nvars; var++)
      {
         var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, var);
         t = vartypes[var];
         nFaces += nalu_hypre_StructGridLocalSize(var_cgrid);

         switch (t)
         {
            case 2:
               nxFaces += nalu_hypre_StructGridLocalSize(var_cgrid);
               break;
            case 3:
               nyFaces += nalu_hypre_StructGridLocalSize(var_cgrid);
               break;
            case 4:
               nzFaces += nalu_hypre_StructGridLocalSize(var_cgrid);
               break;
         }

         nalu_hypre_SStructVariableGetOffset((nalu_hypre_SStructVariable) t,
                                        ndim, varoffsets[t]);
      }

      /* 2-d vs 3-d case */
      if (ndim < 3)
      {
         nEdges = nFaces;
         /* nxEdges = nxFaces; */
         /* nyEdges = nyFaces; */
         /* nzEdges = 0; */
      }

      else
      {
         p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);    /* edge cgrid */
         nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
         vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

         for (var = 0; var < nvars; var++)
         {
            var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, var);
            t = vartypes[var];
            nEdges += nalu_hypre_StructGridLocalSize(var_cgrid);

            /* switch (t) */
            /* { */
            /*    case 5: */
            /*       nxEdges += nalu_hypre_StructGridLocalSize(var_cgrid); */
            /*       break; */
            /*    case 6: */
            /*       nyEdges += nalu_hypre_StructGridLocalSize(var_cgrid); */
            /*       break; */
            /*    case 7: */
            /*       nzEdges += nalu_hypre_StructGridLocalSize(var_cgrid); */
            /*       break; */
            /* } */

            nalu_hypre_SStructVariableGetOffset((nalu_hypre_SStructVariable) t,
                                           ndim, varoffsets[t]);
         }
      }

      /* same for 2-d & 3-d, assuming that fgrid_edge= fgrid_face in input */
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);    /* edge fgrid */
      nvars   = nalu_hypre_SStructPGridNVars(p_fgrid);
      vartypes = nalu_hypre_SStructPGridVarTypes(p_fgrid);

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
    *  boundary extending into other fboxes). These mappings are for the
    *  cell-centred boxes.
    *  Check: Other variable boxes should follow this mapping, by
    *  property of the variable-shifted indices? Can the cell-centred boundary
    *  indices of a box be non-cell-centred indices for another box?
    *
    *  Also determine contracted cell-centred fboxes.
    *--------------------------------------------------------------------------*/
   cfbox_mapping = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nparts, NALU_HYPRE_MEMORY_HOST);
   fcbox_mapping = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nparts, NALU_HYPRE_MEMORY_HOST);
   contract_fedgeBoxes = nalu_hypre_TAlloc(nalu_hypre_BoxArray *,  nparts, NALU_HYPRE_MEMORY_HOST);
   Edge_cstarts = nalu_hypre_TAlloc(nalu_hypre_Index *,  nparts, NALU_HYPRE_MEMORY_HOST);
   upper_shifts = nalu_hypre_TAlloc(nalu_hypre_Index *,  nparts, NALU_HYPRE_MEMORY_HOST);
   lower_shifts = nalu_hypre_TAlloc(nalu_hypre_Index *,  nparts, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < nparts; i++)
   {
      p_fgrid  = nalu_hypre_SStructGridPGrid(fgrid_element, i);
      var_fgrid = nalu_hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
      j        = nalu_hypre_BoxArraySize(fboxes);
      fcbox_mapping[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  j, NALU_HYPRE_MEMORY_HOST);

      p_cgrid  = nalu_hypre_SStructGridPGrid(cgrid_element, i);
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

      /* fill up the contracted box_array */
      contract_fedgeBoxes[i] = nalu_hypre_BoxArrayCreate(0, ndim);
      Edge_cstarts[i] = nalu_hypre_TAlloc(nalu_hypre_Index,  nalu_hypre_BoxArraySize(fboxes), NALU_HYPRE_MEMORY_HOST);
      upper_shifts[i] = nalu_hypre_TAlloc(nalu_hypre_Index,  nalu_hypre_BoxArraySize(fboxes), NALU_HYPRE_MEMORY_HOST);
      lower_shifts[i] = nalu_hypre_TAlloc(nalu_hypre_Index,  nalu_hypre_BoxArraySize(fboxes), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ForBoxI(j, fboxes)
      {
         fbox = nalu_hypre_BoxArrayBox(fboxes, j);

         /* contract the fbox to correspond to the correct cbox */
         cbox = nalu_hypre_BoxContraction(fbox, var_fgrid, rfactor);
         nalu_hypre_AppendBox(cbox, contract_fedgeBoxes[i]);

         /* record the offset mapping between the coarse cell index and
            the fine cell index */
         nalu_hypre_ClearIndex(upper_shifts[i][j]);
         nalu_hypre_ClearIndex(lower_shifts[i][j]);
         for (l = 0; l < ndim; l++)
         {
            m = nalu_hypre_BoxIMin(cbox)[l];
            p = m % rfactor[l];
            if (p > 0 && m > 0)
            {
               upper_shifts[i][j][l] = p - 1;
               lower_shifts[i][j][l] = p - rfactor[l];
            }
            else
            {
               upper_shifts[i][j][l] = rfactor[l] - p - 1;
               lower_shifts[i][j][l] = -p;
            }
         }

         /* record the cstarts of the cbox */
         nalu_hypre_ProjectBox(cbox, zero_index, rfactor);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cbox), Edge_cstarts[i][j]);
         nalu_hypre_StructMapFineToCoarse(Edge_cstarts[i][j], zero_index, rfactor,
                                     Edge_cstarts[i][j]);

         nalu_hypre_BoxDestroy(cbox);
      }
   }  /* for (i= 0; i< nparts; i++) */

   /* variable rank bounds for this processor */
   n_CtoVbox   = nalu_hypre_TAlloc(NALU_HYPRE_Int **,  nparts, NALU_HYPRE_MEMORY_HOST);
   CtoVboxnums = nalu_hypre_TAlloc(NALU_HYPRE_Int ***,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_SStructCellGridBoxNumMap(fgrid_edge, part, &n_CtoVbox[part],
                                     &CtoVboxnums[part]);
   }

   /* variable rank bounds for this processor */
   flower_ranks = nalu_hypre_TAlloc(NALU_HYPRE_BigInt *,  nparts, NALU_HYPRE_MEMORY_HOST);
   fupper_ranks = nalu_hypre_TAlloc(NALU_HYPRE_BigInt *,  nparts, NALU_HYPRE_MEMORY_HOST);

   clower_ranks = nalu_hypre_TAlloc(NALU_HYPRE_BigInt *,  nparts, NALU_HYPRE_MEMORY_HOST);
   cupper_ranks = nalu_hypre_TAlloc(NALU_HYPRE_BigInt *,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      flower_ranks[part] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  tot_vars, NALU_HYPRE_MEMORY_HOST);
      fupper_ranks[part] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  tot_vars, NALU_HYPRE_MEMORY_HOST);

      /* cell grid ranks */
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_element, part);
      var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, 0);
      box_array = nalu_hypre_StructGridBoxes(var_fgrid);

      fbox     = nalu_hypre_BoxArrayBox(box_array, 0);
      nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fbox), findex);
      nalu_hypre_SStructGridFindBoxManEntry(fgrid_element, part, findex, 0,
                                       &entry);
      nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &flower_ranks[part][0],
                                            matrix_type);

      fbox = nalu_hypre_BoxArrayBox(box_array, nalu_hypre_BoxArraySize(box_array) - 1);
      nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(fbox), findex);
      nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, 0,
                                       &entry);
      nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &fupper_ranks[part][0],
                                            matrix_type);

      clower_ranks[part] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  tot_vars, NALU_HYPRE_MEMORY_HOST);
      cupper_ranks[part] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  tot_vars, NALU_HYPRE_MEMORY_HOST);

      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_element, part);
      var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, 0);
      box_array = nalu_hypre_StructGridBoxes(var_cgrid);

      cbox     = nalu_hypre_BoxArrayBox(box_array, 0);
      nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cbox), cindex);
      nalu_hypre_SStructGridFindBoxManEntry(cgrid_element, part, cindex, 0,
                                       &entry);
      nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &clower_ranks[part][0],
                                            matrix_type);

      cbox = nalu_hypre_BoxArrayBox(box_array, nalu_hypre_BoxArraySize(box_array) - 1);
      nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(cbox), cindex);
      nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, 0,
                                       &entry);
      nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &cupper_ranks[part][0],
                                            matrix_type);

      /* face grid ranks */
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_face, part);
      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_face, part);
      nvars  = nalu_hypre_SStructPGridNVars(p_fgrid);
      vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

      for (i = 0; i < nvars; i++)
      {
         t = vartypes[i];
         var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, i);
         box_array = nalu_hypre_StructGridBoxes(var_fgrid);

         fbox     = nalu_hypre_BoxArrayBox(box_array, 0);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fbox), findex);
         nalu_hypre_SStructGridFindBoxManEntry(fgrid_face, part, findex, i,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &flower_ranks[part][t],
                                               matrix_type);

         fbox = nalu_hypre_BoxArrayBox(box_array, nalu_hypre_BoxArraySize(box_array) - 1);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(fbox), findex);
         nalu_hypre_SStructGridFindBoxManEntry(fgrid_face, part, findex, i,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &fupper_ranks[part][t],
                                               matrix_type);

         var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, i);
         box_array = nalu_hypre_StructGridBoxes(var_cgrid);
         cbox     = nalu_hypre_BoxArrayBox(box_array, 0);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cbox), cindex);
         nalu_hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, i,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &clower_ranks[part][t],
                                               matrix_type);

         cbox = nalu_hypre_BoxArrayBox(box_array, nalu_hypre_BoxArraySize(box_array) - 1);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(cbox), cindex);
         nalu_hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, i,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &cupper_ranks[part][t],
                                               matrix_type);
      }
      /* edge grid ranks */
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
      nvars  = nalu_hypre_SStructPGridNVars(p_fgrid);
      vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

      for (i = 0; i < nvars; i++)
      {
         t = vartypes[i];
         var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, i);
         box_array = nalu_hypre_StructGridBoxes(var_fgrid);

         fbox     = nalu_hypre_BoxArrayBox(box_array, 0);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fbox), findex);
         nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, i,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &flower_ranks[part][t],
                                               matrix_type);

         fbox = nalu_hypre_BoxArrayBox(box_array, nalu_hypre_BoxArraySize(box_array) - 1);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(fbox), findex);
         nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, i,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &fupper_ranks[part][t],
                                               matrix_type);

         var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, i);
         box_array = nalu_hypre_StructGridBoxes(var_cgrid);
         cbox     = nalu_hypre_BoxArrayBox(box_array, 0);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cbox), cindex);
         nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, i,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &clower_ranks[part][t],
                                               matrix_type);

         cbox = nalu_hypre_BoxArrayBox(box_array, nalu_hypre_BoxArraySize(box_array) - 1);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(cbox), cindex);
         nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, i,
                                          &entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &cupper_ranks[part][t],
                                               matrix_type);
      }
   }

   /* CREATE IJ_MATRICES- need to find the size of each one. Notice that the row
      and col ranks of these matrices can be created using only grid information.
      Grab the first part, first variable, first box, and lower index (lower rank);
      Grab the last part, last variable, last box, and upper index (upper rank). */

   /* Element_iedge- same for 2-d and 3-d */
   /* lower rank */
   part = 0;
   box = 0;
   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, 0, box, myproc, &entry);

   p_cgrid  = nalu_hypre_SStructGridPGrid(cgrid_element, part);
   var_cgrid = nalu_hypre_SStructPGridCellSGrid(p_cgrid) ;
   cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox     = nalu_hypre_BoxArrayBox(cboxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(cbox), &ilower);

   p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
   nalu_hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, 0, box, myproc, &entry);

   var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, 0);
   fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
   fbox     = nalu_hypre_BoxArrayBox(fboxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(fbox), &jlower);

   /* upper rank */
   part = nparts - 1;
   p_cgrid  = nalu_hypre_SStructGridPGrid(cgrid_element, part);
   var_cgrid = nalu_hypre_SStructPGridCellSGrid(p_cgrid) ;
   cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox     = nalu_hypre_BoxArrayBox(cboxes, nalu_hypre_BoxArraySize(cboxes) - 1);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, 0, nalu_hypre_BoxArraySize(cboxes) - 1,
                                           myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(cbox), &iupper);

   p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
   nvars   = nalu_hypre_SStructPGridNVars(p_fgrid);

   var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, nvars - 1);
   fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
   fbox     = nalu_hypre_BoxArrayBox(fboxes, nalu_hypre_BoxArraySize(fboxes) - 1);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, nvars - 1, nalu_hypre_BoxArraySize(fboxes) - 1,
                                           myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(fbox), &jupper);

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Element_iedge);
   NALU_HYPRE_IJMatrixSetObjectType(Element_iedge, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize(Element_iedge);

   /* Edge_iedge. Note that even though not all the iedges are involved (e.g.,
    * truncated edges are not), we use the ranks determined by the Edge/edge grids.
    * Same for 2-d and 3-d. */
   /* lower rank */
   part = 0;
   box = 0;
   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, 0, box, myproc, &entry);
   p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
   var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, 0);
   cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox    = nalu_hypre_BoxArrayBox(cboxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(cbox), &ilower);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, 0, box, myproc, &entry);
   p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
   var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, 0);
   fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
   fbox    = nalu_hypre_BoxArrayBox(fboxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(fbox), &jlower);

   /* upper rank */
   part = nparts - 1;
   p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
   nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
   var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, nvars - 1);
   cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox    = nalu_hypre_BoxArrayBox(cboxes, nalu_hypre_BoxArraySize(cboxes) - 1);
   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, nvars - 1,
                                           nalu_hypre_BoxArraySize(cboxes) - 1, myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(cbox), &iupper);

   p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
   nvars   = nalu_hypre_SStructPGridNVars(p_fgrid);
   var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, nvars - 1);
   fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
   fbox    = nalu_hypre_BoxArrayBox(fboxes, nalu_hypre_BoxArraySize(fboxes) - 1);
   nalu_hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, nvars - 1,
                                           nalu_hypre_BoxArraySize(fboxes) - 1, myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(fbox), &jupper);

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Edge_iedge);
   NALU_HYPRE_IJMatrixSetObjectType(Edge_iedge, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize(Edge_iedge);

   /* edge_Edge. Same for 2-d and 3-d. */
   /* lower rank */
   part = 0;
   box = 0;
   nalu_hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, 0, box, myproc, &entry);
   p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
   var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, 0);
   fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
   fbox    = nalu_hypre_BoxArrayBox(fboxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(fbox), &ilower);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, 0, box, myproc, &entry);
   p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
   var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, 0);
   cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox    = nalu_hypre_BoxArrayBox(cboxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(cbox), &jlower);

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

   /* Face_iedge. Only needed in 3-d. */
   if (ndim == 3)
   {
      /* lower rank */
      part = 0;
      box = 0;
      nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_face, part, 0, box, myproc, &entry);

      p_cgrid  = nalu_hypre_SStructGridPGrid(cgrid_face, part);
      var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, 0);
      cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
      cbox     = nalu_hypre_BoxArrayBox(cboxes, 0);
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(cbox), &ilower);

      nalu_hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, 0, box, myproc, &entry);
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
      var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, 0);
      fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
      fbox     = nalu_hypre_BoxArrayBox(fboxes, 0);
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(fbox), &jlower);

      /* upper rank */
      part = nparts - 1;
      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_face, part);
      nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
      var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, nvars - 1);
      cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
      cbox    = nalu_hypre_BoxArrayBox(cboxes, nalu_hypre_BoxArraySize(cboxes) - 1);

      nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_face, part, nvars - 1,
                                              nalu_hypre_BoxArraySize(cboxes) - 1, myproc, &entry);
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(cbox), &iupper);

      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
      nvars   = nalu_hypre_SStructPGridNVars(p_fgrid);

      var_fgrid = nalu_hypre_SStructPGridSGrid(p_fgrid, nvars - 1);
      fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);
      fbox     = nalu_hypre_BoxArrayBox(fboxes, nalu_hypre_BoxArraySize(fboxes) - 1);

      nalu_hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, nvars - 1,
                                              nalu_hypre_BoxArraySize(fboxes) - 1, myproc, &entry);
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(fbox), &jupper);

      NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Face_iedge);
      NALU_HYPRE_IJMatrixSetObjectType(Face_iedge, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJMatrixInitialize(Face_iedge);
   }

   /* Element_Face. Only for 3-d since Element_Edge= Element_Face in 2-d. */
   /* lower rank */
   if (ndim == 3)
   {
      part = 0;
      box = 0;
      nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, 0, box,
                                              myproc, &entry);

      p_cgrid  = nalu_hypre_SStructGridPGrid(cgrid_element, part);
      var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, 0);
      cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
      cbox     = nalu_hypre_BoxArrayBox(cboxes, 0);
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(cbox), &ilower);

      nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_face, part, 0, box,
                                              myproc, &entry);
      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_face, part);
      var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, 0);
      cboxes  = nalu_hypre_StructGridBoxes(var_cgrid);
      cbox    = nalu_hypre_BoxArrayBox(cboxes, 0);
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(cbox), &jlower);

      /* upper rank */
      part = nparts - 1;
      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_element, part);
      nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
      var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, nvars - 1);
      cboxes  = nalu_hypre_StructGridBoxes(var_cgrid);
      cbox    = nalu_hypre_BoxArrayBox(cboxes, nalu_hypre_BoxArraySize(cboxes) - 1);

      nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, nvars - 1,
                                              nalu_hypre_BoxArraySize(cboxes) - 1, myproc, &entry);
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(cbox), &iupper);

      p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_face, part);
      nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
      var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, nvars - 1);
      cboxes  = nalu_hypre_StructGridBoxes(var_cgrid);
      cbox    = nalu_hypre_BoxArrayBox(cboxes, nalu_hypre_BoxArraySize(cboxes) - 1);

      nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_face, part, nvars - 1,
                                              nalu_hypre_BoxArraySize(cboxes) - 1, myproc, &entry);
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(cbox), &jupper);

      NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Element_Face);
      NALU_HYPRE_IJMatrixSetObjectType(Element_Face, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJMatrixInitialize(Element_Face);
   }

   /* Element_Edge. Same for 2-d and 3-d. */
   /* lower rank */
   part = 0;
   box = 0;
   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, 0, box, myproc, &entry);

   p_cgrid  = nalu_hypre_SStructGridPGrid(cgrid_element, part);
   var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, 0);
   cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox     = nalu_hypre_BoxArrayBox(cboxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(cbox), &ilower);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, 0, box, myproc, &entry);
   p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
   var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, 0);
   cboxes  = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox    = nalu_hypre_BoxArrayBox(cboxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(cbox), &jlower);

   /* upper rank */
   part = nparts - 1;
   p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_element, part);
   nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
   var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, nvars - 1);
   cboxes  = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox    = nalu_hypre_BoxArrayBox(cboxes, nalu_hypre_BoxArraySize(cboxes) - 1);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, nvars - 1,
                                           nalu_hypre_BoxArraySize(cboxes) - 1, myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(cbox), &iupper);

   p_cgrid = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
   nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
   var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, nvars - 1);
   cboxes  = nalu_hypre_StructGridBoxes(var_cgrid);
   cbox    = nalu_hypre_BoxArrayBox(cboxes, nalu_hypre_BoxArraySize(cboxes) - 1);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, nvars - 1,
                                           nalu_hypre_BoxArraySize(cboxes) - 1, myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(cbox), &jupper);

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Element_Edge);
   NALU_HYPRE_IJMatrixSetObjectType(Element_Edge, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize(Element_Edge);

   /*------------------------------------------------------------------------------
    * fill up the parcsr matrices.
    *------------------------------------------------------------------------------*/
   /* count the number of connections, i.e., the columns
    * no. of interior edges per face, or no. of interior edges per cell.
    * Need to distinguish between 2 and 3-d. */
   if (ndim == 3)
   {
      n_xFace_iedges = (rfactor[1] - 1) * rfactor[2] + (rfactor[2] - 1) * rfactor[1];
      n_yFace_iedges = (rfactor[0] - 1) * rfactor[2] + (rfactor[2] - 1) * rfactor[0];
      n_zFace_iedges = (rfactor[1] - 1) * rfactor[0] + (rfactor[0] - 1) * rfactor[1];
      n_Cell_iedges = (rfactor[2] - 1) * n_zFace_iedges +
                      rfactor[2] * (rfactor[0] - 1) * (rfactor[1] - 1);

      nFaces_iedges = nxFaces * n_xFace_iedges + nyFaces * n_yFace_iedges +
                      nzFaces * n_zFace_iedges;
      nElements_iedges = nElements * n_Cell_iedges;
   }
   else
   {
      n_Cell_iedges = (rfactor[0] - 1) * rfactor[1] + (rfactor[1] - 1) * rfactor[0];
      nElements_iedges = nElements * n_Cell_iedges;
   }

   if (ndim == 3)
   {
      iFace = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nFaces, memory_location);
   }
   iEdge    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nEdges, memory_location);
   iElement = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nElements, memory_location);

   /* array structures needed for forming ij_matrices */

   /* Element_edge. Same for 2-d and 3-d. */
   ncols_Elementedge = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nElements, memory_location);
   for (i = 0; i < nElements; i++)
   {
      ncols_Elementedge[i] = n_Cell_iedges;
   }
   jElement_edge    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nElements_iedges, memory_location);
   vals_Elementedge = nalu_hypre_CTAlloc(NALU_HYPRE_Real,   nElements_iedges, memory_location);

   /*---------------------------------------------------------------------------
    * Fill up the row/column ranks of Element_edge. Will need to distinguish
    * between 2-d and 3-d.
    *      Loop over the coarse element grid
    *        a) Refine the coarse cell and grab the fine cells that will contain
    *           the fine edges.
    *           To obtain the correct coarse-to-fine cell index mapping, we
    *           map lindex to the fine cell grid and then adjust
    *           so that the final mapped fine cell is the one on the upper
    *           corner of the agglomerate. Will need to determine the fine box
    *           corresponding to the coarse box.
    *        b) loop map these fine cells and find the ranks of the fine edges.
    *---------------------------------------------------------------------------*/
   nElements = 0;
   nElements_iedges = 0;
   for (part = 0; part < nparts; part++)
   {
      if (ndim == 3)
      {
         p_cgrid      = nalu_hypre_SStructGridPGrid(cgrid_edge, part);  /* Edge grid */
         Edge_nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
         Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);
      }
      else if (ndim == 2) /* edge is a face in 2-d*/
      {
         p_cgrid      = nalu_hypre_SStructGridPGrid(cgrid_face, part);  /* Face grid */
         Face_nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
         Face_vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);
      }

      p_cgrid   = nalu_hypre_SStructGridPGrid(cgrid_element, part);  /* ccell grid */
      var_cgrid = nalu_hypre_SStructPGridCellSGrid(p_cgrid);
      cboxes    = nalu_hypre_StructGridBoxes(var_cgrid);

      p_fgrid   = nalu_hypre_SStructGridPGrid(fgrid_element, part);  /* fcell grid */
      var_fgrid = nalu_hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes    = nalu_hypre_StructGridBoxes(var_fgrid);

      nalu_hypre_ForBoxI(i, cboxes)
      {
         cbox = nalu_hypre_BoxArrayBox(cboxes, i);
         nalu_hypre_BoxGetSize(cbox, loop_size);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cbox), cstart);

         /* determine which fine box cbox has coarsened from. Obtained from
            cfbox_mapping. */
         fboxi = cfbox_mapping[part][i];
         fbox = nalu_hypre_BoxArrayBox(fboxes, fboxi);

         /**********************************************************************
          * determine the shift to get the correct c-to-f cell index map:
          *    d= nalu_hypre_BoxIMin(fbox)[j]%rfactor[j]*sign(nalu_hypre_BoxIMin(fbox)[j])
          *    stride[j]= d-1  if d>0
          *    stride[j]= rfactor[j]-1+d  if d<=0.
          * This is upper_shifts[part][fboxi].
          **********************************************************************/
         nalu_hypre_ClearIndex(stride);
         nalu_hypre_CopyIndex(upper_shifts[part][fboxi], stride);

         /* loop over each cell and find the row rank of Element_edge and then
            the column ranks of the connected fine edges. */
         nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
         {
            zypre_BoxLoopGetIndex(lindex);
            nalu_hypre_SetIndex3(cindex, lindex[0], lindex[1], lindex[2]);
            nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);

            /* refined cindex to get the correct upper fine index */
            nalu_hypre_StructMapCoarseToFine(cindex, zero_index, rfactor, findex);
            nalu_hypre_AddIndexes(findex, stride, 3, findex);

            /* Element(i,j,k) rank */
            nalu_hypre_SStructGridFindBoxManEntry(cgrid_element, part, cindex, 0, &entry);
            nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank, matrix_type);
            iElement[nElements] = rank;
            nElements++;

            /* Element_iedge columns: 3-d, x_edges, y_edges, and z_edges. */
            if (ndim == 3)
            {
               nalu_hypre_SetIndex3(low_index, findex[0] - rfactor[0] + 1,
                               findex[1] - rfactor[1] + 1,
                               findex[2] - rfactor[2] + 1);

               for (t = 0; t < Edge_nvars; t++)
               {
                  nalu_hypre_CopyIndex(findex, hi_index);
                  var = Edge_vartypes[t]; /* c & f edges enumerated the same */

                  /* determine looping extents over the refined cells that
                     will have fine edges. */
                  switch (var)
                  {
                     case 5:  /* x_edges */
                     {
                        hi_index[1] -= 1;
                        hi_index[2] -= 1;
                        break;
                     }
                     case 6:  /* y_edges */
                     {
                        hi_index[0] -= 1;
                        hi_index[2] -= 1;
                        break;
                     }
                     case 7:  /* z_edges */
                     {
                        hi_index[0] -= 1;
                        hi_index[1] -= 1;
                        break;
                     }
                  }   /* switch (var) */

                  /* column ranks. */
                  for (m = low_index[2]; m <= hi_index[2]; m++)
                  {
                     for (k = low_index[1]; k <= hi_index[1]; k++)
                     {
                        for (j = low_index[0]; j <= hi_index[0]; j++)
                        {
                           nalu_hypre_SetIndex3(var_index, j, k, m);
                           nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jElement_edge[nElements_iedges] = rank;
                           nElements_iedges++;
                        }  /* for (j= findex[0]; j<= hi_index[0]; j++) */
                     }     /* for (k= findex[1]; k<= hi_index[1]; k++) */
                  }        /* for (m= findex[2]; m<= hi_index[2]; m++) */
               }           /* for (t= 0; t< Edge_nvars; t++) */
            }              /* if (ndim == 3) */

            else if (ndim == 2) /* only x & y faces */
            {
               nalu_hypre_SetIndex3(low_index, findex[0] - rfactor[0] + 1,
                               findex[1] - rfactor[1] + 1,
                               findex[2]);

               for (t = 0; t < Face_nvars; t++)
               {
                  nalu_hypre_CopyIndex(findex, hi_index);
                  var = Face_vartypes[t]; /* c & f faces enumerated the same */

                  switch (var) /* note: hi_index computed differently in 2-d */
                  {
                     case 2:  /* x_faces */
                     {
                        hi_index[0] -= 1;
                        break;
                     }
                     case 3:  /* y_edges */
                     {
                        hi_index[1] -= 1;
                        break;
                     }
                  }   /* switch (var) */

                  /* column ranks. */
                  for (k = low_index[1]; k <= hi_index[1]; k++)
                  {
                     for (j = low_index[0]; j <= hi_index[0]; j++)
                     {
                        nalu_hypre_SetIndex3(var_index, j, k, findex[2]);
                        nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                         t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                              &rank, matrix_type);
                        jElement_edge[nElements_iedges] = rank;
                        nElements_iedges++;
                     }  /* for (j= findex[0]; j<= hi_index[0]; j++) */
                  }     /* for (k= findex[1]; k<= hi_index[1]; k++) */
               }        /* for (t= 0; t< Face_nvars; t++) */
            }           /* if (ndim == 2) */
         }
         nalu_hypre_SerialBoxLoop0End();
      }  /* nalu_hypre_ForBoxI(i, cboxes) */
   }     /* for (part= 0; part< nparts; part++) */

   NALU_HYPRE_IJMatrixSetValues(Element_iedge, nElements, ncols_Elementedge,
                           (const NALU_HYPRE_BigInt*) iElement, (const NALU_HYPRE_BigInt*) jElement_edge,
                           (const NALU_HYPRE_Real*) vals_Elementedge);
   NALU_HYPRE_IJMatrixAssemble((NALU_HYPRE_IJMatrix) Element_iedge);

   nalu_hypre_TFree(ncols_Elementedge, memory_location);
   nalu_hypre_TFree(jElement_edge, memory_location);
   nalu_hypre_TFree(vals_Elementedge, memory_location);

   /* Face_edge */
   /*------------------------------------------------------------------------------
    * Fill out Face_edge a row at a time. Since we have different Face types
    * so that the size of the cols change depending on what type the Face
    * is, we need to loop over the grids and take a count of the col elements.
    * Loop over the coarse face grids and add up the number of interior edges.
    * Will compute only for 3-d. In 2-d, these structures are obtained for
    * Edge_edge.
    *------------------------------------------------------------------------------*/
   if (ndim == 3)
   {
      ncols_Faceedge = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nFaces, memory_location);
      nFaces = 0;
      j = 0;
      for (part = 0; part < nparts; part++)
      {
         p_cgrid      = nalu_hypre_SStructGridPGrid(cgrid_face, part);  /* Face grid */
         Face_nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
         Face_vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

         p_fgrid   = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
         var_fgrid = nalu_hypre_SStructPGridCellSGrid(p_fgrid);
         fboxes    = nalu_hypre_StructGridBoxes(var_fgrid);

         for (t = 0; t < Face_nvars; t++)
         {
            var = Face_vartypes[t];
            var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, t);
            k = nalu_hypre_StructGridLocalSize(var_cgrid);

            switch (var)
            {
               case 2: /* x_Faces (i,j,k) then (i-1,j,k), contain y,z edges */
               {
                  for (i = 0; i < k; i++)
                  {
                     /* y_iedge connections to x_Face */
                     ncols_Faceedge[nFaces] = (rfactor[2] - 1) * rfactor[1];

                     /* z_iedge connections to x_Face */
                     ncols_Faceedge[nFaces] += rfactor[2] * (rfactor[1] - 1);

                     j += ncols_Faceedge[nFaces];
                     nFaces++;
                  }
                  break;
                  }   /* case 2 */

               case 3: /* y_Faces (i,j,k) then (i,j-1,k), contain x,z edges */
               {
                  for (i = 0; i < k; i++)
                  {
                     /* x_iedge connections to y_Face */
                     ncols_Faceedge[nFaces] = (rfactor[2] - 1) * rfactor[0];

                     /* z_iedge connections to y_Face */
                     ncols_Faceedge[nFaces] += rfactor[2] * (rfactor[0] - 1);

                     j += ncols_Faceedge[nFaces];
                     nFaces++;
                  }
                  break;
                  }   /* case 3 */

               case 4: /* z_Faces (i,j,k) then (i,j,k-1), contain x,y edges */
               {
                  for (i = 0; i < k; i++)
                  {
                     /* x_iedge connections to z_Face */
                     ncols_Faceedge[nFaces] = (rfactor[1] - 1) * rfactor[0];

                     /* y_iedge connections to z_Face */
                     ncols_Faceedge[nFaces] += rfactor[1] * (rfactor[0] - 1);

                     j += ncols_Faceedge[nFaces];
                     nFaces++;
                  }
                  break;
                  }   /* case 4 */

            } /* switch(var) */
         }    /* for (t= 0; t< Face_nvars; t++) */
      }       /* for (part= 0; part< nparts; part++) */

      jFace_edge    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, j, memory_location);
      vals_Faceedge = nalu_hypre_CTAlloc(NALU_HYPRE_Real, j, memory_location);
      for (i = 0; i < j; i++)
      {
         vals_Faceedge[i] = 1.0;
      }

      /*---------------------------------------------------------------------------
       * Fill up the row/column ranks of Face_edge.
       *      Loop over the coarse Cell grid
       *        a) for each Cell box, stretch to a Face box
       *        b) for each coarse face, if it is on the proc, map it to a
       *           coarse cell (add the variable offset).
       *        c) refine the coarse cell and grab the fine cells that will contain
       *           the fine edges. Refining requires a shifting dependent on the
       *           begining index of the fine box.
       *        d) map these fine cells to the fine edges.
       *---------------------------------------------------------------------------*/
      nFaces       = 0;
      nFaces_iedges = 0;
      for (part = 0; part < nparts; part++)
      {
         p_cgrid      = nalu_hypre_SStructGridPGrid(cgrid_face, part);  /* Face grid */
         Face_nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
         Face_vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

         for (t = 0; t < Face_nvars; t++)
         {
            var = Face_vartypes[t];
            var_cgrid = nalu_hypre_SStructPGridCellSGrid(p_cgrid);
            cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);

            /* to eliminate comparisons, take the switch outside of the loop. */
            switch (var)
            {
               case 2:  /* x_Faces-> y_iedges, z_iedges */
               {
                  nalu_hypre_ForBoxI(i, cboxes)
                  {
                     cbox = nalu_hypre_BoxArrayBox(cboxes, i);
                     nalu_hypre_CopyBox(cbox, &copy_box);
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     /* determine which fine box cbox has coarsened from */
                     fboxi = cfbox_mapping[part][i];
                     fbox = nalu_hypre_BoxArrayBox(fboxes, fboxi);

                     /**********************************************************
                      * determine the shift to get the correct c-to-f cell
                      * index map. This is upper_shifts[part][fboxi].
                      **********************************************************/
                     nalu_hypre_ClearIndex(stride);
                     nalu_hypre_CopyIndex(upper_shifts[part][fboxi], stride);

                     nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(cindex, lindex[0], lindex[1], lindex[2]);
                        nalu_hypre_AddIndexes(cindex, start, 3, cindex);

                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank, matrix_type);

                        /* check if rank on proc before continuing */
                        if ((rank <= cupper_ranks[part][var]) &&
                            (rank >= clower_ranks[part][var]))
                        {
                           iFace[nFaces] = rank;
                           nFaces++;

                           /* transform face index to cell index */
                           nalu_hypre_AddIndexes(cindex, varoffsets[var], 3, cell_index);

                           /* Refine the coarse cell to the upper fine index. The face will
                              be on the "lower end" fine cells, i.e., the slab box starting
                              with fine index cell_index. The fine edges will be on the
                              lower x of the fine cell, e.g., with fine cell (i,j,k),
                              y_iedge (i-1,j,k) & z_iedge (i-1,j,k). */
                           nalu_hypre_StructMapCoarseToFine(cell_index, zero_index,
                                                       rfactor, findex);
                           nalu_hypre_AddIndexes(findex, stride, 3, findex);

                           /* cell_index was refined to the upper fine index. Shift
                              back to the lower end, subtract (rfactor-1). */
                           for (j = 0; j < ndim; j++)
                           {
                              findex[j] -= rfactor[j] - 1;
                           }

                           /* y_iedges */
                           ilower = findex[0] - 1;
                           for (k = 0; k < rfactor[2] - 1; k++)
                           {
                              for (j = 0; j < rfactor[1]; j++)
                              {
                                 nalu_hypre_SetIndex3(var_index, ilower, j + findex[1], k + findex[2]);
                                 nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[6], &entry);
                                 nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges] = rank;
                                 nFaces_iedges++;
                              }
                           }

                           /* z_iedges */
                           for (k = 0; k < rfactor[2]; k++)
                           {
                              for (j = 0; j < rfactor[1] - 1; j++)
                              {
                                 nalu_hypre_SetIndex3(var_index, ilower, j + findex[1], k + findex[2]);
                                 nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[7], &entry);
                                 nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges] = rank;
                                 nFaces_iedges++;
                              }
                           }
                        }  /* if ((rank <= cupper_ranks[part][var]) &&
(rank >= clower_ranks[part][var])) */
                     }

                     nalu_hypre_SerialBoxLoop0End();
                  }  /* nalu_hypre_ForBoxI(i, cboxes) */
                  break;
                  }   /* case 2:  x_Faces-> y_iedges, z_iedges */

               case 3:  /* y_Faces-> x_iedges, z_iedges */
               {
                  nalu_hypre_ForBoxI(i, cboxes)
                  {
                     cbox = nalu_hypre_BoxArrayBox(cboxes, i);
                     nalu_hypre_CopyBox(cbox, &copy_box);
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     /* determine which fine box cbox has coarsened from */
                     fboxi = cfbox_mapping[part][i];
                     fbox = nalu_hypre_BoxArrayBox(fboxes, fboxi);

                     /**********************************************************
                      * determine the shift to get the correct c-to-f cell
                      * index map. This is upper_shifts[part][fboxi].
                      **********************************************************/
                     nalu_hypre_ClearIndex(stride);
                     nalu_hypre_CopyIndex(upper_shifts[part][fboxi], stride);

                     nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(cindex, lindex[0], lindex[1], lindex[2]);
                        nalu_hypre_AddIndexes(cindex, start, 3, cindex);

                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank, matrix_type);
                        /* check if rank on proc before continuing */
                        if ((rank <= cupper_ranks[part][var]) &&
                            (rank >= clower_ranks[part][var]))
                        {
                           iFace[nFaces] = rank;
                           nFaces++;

                           /* transform face index to cell index */
                           nalu_hypre_AddIndexes(cindex, varoffsets[var], 3, cell_index);

                           /* Refine the coarse cell to the upper fine index. The face will
                              be on the "lower end" fine cells, i.e., the slab box starting
                              with fine index cell_index. The fine edges will be on the
                              lower x of the fine cell, e.g., with fine cell (i,j,k),
                              y_iedge (i-1,j,k) & z_iedge (i-1,j,k). */
                           nalu_hypre_StructMapCoarseToFine(cell_index, zero_index,
                                                       rfactor, findex);
                           nalu_hypre_AddIndexes(findex, stride, 3, findex);

                           /* cell_index is refined to the upper fine index. Shift
                              back to the lower end, subtract (rfactor-1). */
                           for (j = 0; j < ndim; j++)
                           {
                              findex[j] -= rfactor[j] - 1;
                           }

                           /* x_iedges */
                           ilower = findex[1] - 1;
                           for (k = 0; k < rfactor[2] - 1; k++)
                           {
                              for (j = 0; j < rfactor[0]; j++)
                              {
                                 nalu_hypre_SetIndex3(var_index, j + findex[0], ilower, k + findex[2]);
                                 nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[5], &entry);
                                 nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges] = rank;
                                 nFaces_iedges++;
                              }
                           }

                           /* z_iedges */
                           for (k = 0; k < rfactor[2]; k++)
                           {
                              for (j = 0; j < rfactor[0] - 1; j++)
                              {
                                 nalu_hypre_SetIndex3(var_index, j + findex[0], ilower, k + findex[2]);
                                 nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[7], &entry);
                                 nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges] = rank;
                                 nFaces_iedges++;
                              }
                           }
                        }  /* if ((rank <= cupper_ranks[part][var]) &&
(rank >= clower_ranks[part][var])) */
                     }

                     nalu_hypre_SerialBoxLoop0End();
                  }  /* nalu_hypre_ForBoxI(i, cboxes) */
                  break;
                  }   /* case 3:  y_Faces-> x_iedges, z_iedges */

               case 4:  /* z_Faces-> x_iedges, y_iedges */
               {
                  nalu_hypre_ForBoxI(i, cboxes)
                  {
                     cbox = nalu_hypre_BoxArrayBox(cboxes, i);
                     nalu_hypre_CopyBox(cbox, &copy_box);
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                           nalu_hypre_BoxIMin(&copy_box));

                     nalu_hypre_BoxGetSize(&copy_box, loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

                     /* determine which fine box cbox has coarsened from */
                     fboxi = cfbox_mapping[part][i];
                     fbox = nalu_hypre_BoxArrayBox(fboxes, fboxi);

                     /**********************************************************
                      * determine the shift to get the correct c-to-f cell
                      * index map. This is upper_shifts[part][fboxi].
                      **********************************************************/
                     nalu_hypre_ClearIndex(stride);
                     nalu_hypre_CopyIndex(upper_shifts[part][fboxi], stride);

                     nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(cindex, lindex[0], lindex[1], lindex[2]);
                        nalu_hypre_AddIndexes(cindex, start, 3, cindex);

                        nalu_hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, t, &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank, matrix_type);

                        /* check if rank on proc before continuing */
                        if ((rank <= cupper_ranks[part][var]) &&
                            (rank >= clower_ranks[part][var]))
                        {
                           iFace[nFaces] = rank;
                           nFaces++;

                           /* transform face index to cell index */
                           nalu_hypre_AddIndexes(cindex, varoffsets[var], 3, cell_index);

                           /* Refine the coarse cell to the upper fine index. The face will
                              be on the "lower end" fine cells, i.e., the slab box starting
                              with fine index cell_index. The fine edges will be on the
                              lower x of the fine cell, e.g., with fine cell (i,j,k),
                              y_iedge (i-1,j,k) & z_iedge (i-1,j,k). */
                           nalu_hypre_StructMapCoarseToFine(cell_index, zero_index,
                                                       rfactor, findex);
                           nalu_hypre_AddIndexes(findex, stride, 3, findex);

                           /* cell_index is refined to the upper fine index. Shift
                              back to the lower end, subtract (rfactor-1). */
                           for (j = 0; j < ndim; j++)
                           {
                              findex[j] -= rfactor[j] - 1;
                           }

                           /* x_iedges */
                           ilower = findex[2] - 1;
                           for (k = 0; k < rfactor[1] - 1; k++)
                           {
                              for (j = 0; j < rfactor[0]; j++)
                              {
                                 nalu_hypre_SetIndex3(var_index, j + findex[0], k + findex[1], ilower);
                                 nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[5], &entry);
                                 nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges] = rank;
                                 nFaces_iedges++;
                              }
                           }

                           /* y_iedges */
                           for (k = 0; k < rfactor[1]; k++)
                           {
                              for (j = 0; j < rfactor[0] - 1; j++)
                              {
                                 nalu_hypre_SetIndex3(var_index, j + findex[0], k + findex[1], ilower);
                                 nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[6], &entry);
                                 nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges] = rank;
                                 nFaces_iedges++;
                              }
                           }
                        }  /* if ((rank <= cupper_ranks[part][var]) &&
(rank >= clower_ranks[part][var])) */
                     }
                     nalu_hypre_SerialBoxLoop0End();
                  }  /* nalu_hypre_ForBoxI(i, cboxes) */
                  break;
                  }   /* case 4:  z_Faces-> x_iedges, y_iedges */

            }   /* switch(var) */
         }      /* for (t= 0; t< Face_nvars; t++) */
      }         /* for (part= 0; part< nparts; part++) */

      NALU_HYPRE_IJMatrixSetValues(Face_iedge, nFaces, ncols_Faceedge,
                              (const NALU_HYPRE_BigInt*) iFace, (const NALU_HYPRE_BigInt*) jFace_edge,
                              (const NALU_HYPRE_Real*) vals_Faceedge);
      NALU_HYPRE_IJMatrixAssemble((NALU_HYPRE_IJMatrix) Face_iedge);

      nalu_hypre_TFree(ncols_Faceedge, memory_location);
      nalu_hypre_TFree(iFace, memory_location);
      nalu_hypre_TFree(jFace_edge, memory_location);
      nalu_hypre_TFree(vals_Faceedge, memory_location);
   }  /* if (ndim == 3) */

   /* Edge_edge */
   /*------------------------------------------------------------------------------
    * Count the Edge_edge connections. Will need to distinguish 2-d and 3-d.
    *------------------------------------------------------------------------------*/
   /* nEdges should be correct for 2-d & 3-d */
   ncols_Edgeiedge = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nEdges, memory_location);

   nEdges = 0;
   k = 0;
   for (part = 0; part < nparts; part++)
   {
      /* Edge grid. In 2-d this will be the face grid, which is assumed to be
         in cgrid_edge. */
      p_cgrid      = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
      Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);
      Edge_nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);

      for (t = 0; t < Edge_nvars; t++)
      {
         var = Edge_vartypes[t];
         var_cgrid = nalu_hypre_SStructPGridSGrid(p_cgrid, t);
         j = nalu_hypre_StructGridLocalSize(var_cgrid);

         switch (var)
         {
            case 2:    /* 2-d, x_Face */
            {
               m = rfactor[1];
               break;
            }

            case 3:    /* 2-d, y_Face */
            {
               m = rfactor[0];
               break;
            }

            case 5:    /* 3-d, x_Edge */
            {
               m = rfactor[0];
               break;
            }

            case 6:    /* 3-d, y_Edge */
            {
               m = rfactor[1];
               break;
            }

            case 7:    /* 3-d, z_Edge */
            {
               m = rfactor[2];
               break;
            }
         }

         for (i = nEdges; i < nEdges + j; i++) /*fill in the column size for Edge */
         {
            ncols_Edgeiedge[i] = m;
            k += m;
         }
         nEdges += j;

      }  /* for (t= 0; t< Edge_nvars; t++) */
   }     /* for (part= 0; part< nparts; part++) */

   jEdge_iedge    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, k, memory_location);
   vals_Edgeiedge = nalu_hypre_CTAlloc(NALU_HYPRE_Real, k, memory_location);
   for (i = 0; i < k; i++)
   {
      vals_Edgeiedge[i] = 1.0;
   }

   /*---------------------------------------------------------------------------
    * Fill up the row/column ranks of Edge_edge. Since a refinement of the
    * coarse edge index does not get the correct fine edge index, we need to
    * map it to the cell grid. Recall, all variable grids are gotten by coarsening
    * a cell centred grid.
    *      Loop over the coarse Cell grid
    *        a) for each Cell box, map to an Edge box
    *        b) for each coarse Edge on my proc , map it to a coarse cell
    *           (add the variable offset).
    *        c) refine the coarse cell and grab the fine cells that will contain
    *           the fine edges.
    *        d) map these fine cells to the fine edges.
    *---------------------------------------------------------------------------*/

   nEdges       = 0;
   nEdges_iedges = 0;
   for (part = 0; part < nparts; part++)
   {
      p_cgrid      = nalu_hypre_SStructGridPGrid(cgrid_edge, part);
      Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);
      Edge_nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);

      p_fgrid   = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
      var_fgrid = nalu_hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes    = nalu_hypre_StructGridBoxes(var_fgrid);

      for (t = 0; t < Edge_nvars; t++)
      {
         var = Edge_vartypes[t];
         var_cgrid = nalu_hypre_SStructPGridCellSGrid(p_cgrid);
         cboxes   = nalu_hypre_StructGridBoxes(var_cgrid);

         nalu_hypre_ForBoxI(i, cboxes)
         {
            cbox = nalu_hypre_BoxArrayBox(cboxes, i);

            /*-------------------------------------------------------------------
             * extract the variable box by offsetting with var_offset. Note that
             * this may lead to a bigger variable domain than is on this proc.
             * Off-proc Edges will be checked to eliminate this problem.
             *-------------------------------------------------------------------*/
            nalu_hypre_CopyBox(cbox, &copy_box);
            nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                  nalu_hypre_BoxIMin(&copy_box));
            nalu_hypre_BoxGetSize(&copy_box, loop_size);
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&copy_box), start);

            /* determine which fine box cbox has coarsened from */
            fboxi = cfbox_mapping[part][i];
            fbox = nalu_hypre_BoxArrayBox(fboxes, fboxi);

            /**********************************************************
             * determine the shift to get the correct c-to-f cell
             * index map. This is upper_shifts[part][fboxi].
             **********************************************************/
            nalu_hypre_ClearIndex(stride);
            nalu_hypre_CopyIndex(upper_shifts[part][fboxi], stride);

            nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
            {
               zypre_BoxLoopGetIndex(lindex);
               nalu_hypre_SetIndex3(cindex, lindex[0], lindex[1], lindex[2]);
               nalu_hypre_AddIndexes(cindex, start, 3, cindex);

               /* row rank */
               nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                &entry);
               nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                     matrix_type);

               /* check if rank on proc before continuing */
               if ((rank <= cupper_ranks[part][var]) &&
                   (rank >= clower_ranks[part][var]))
               {
                  iEdge[nEdges] = rank;
                  nEdges++;

                  nalu_hypre_AddIndexes(cindex, varoffsets[var], 3, cell_index);

                  /* refine cindex and then map back to variable index */
                  nalu_hypre_StructMapCoarseToFine(cell_index, zero_index, rfactor,
                                              findex);
                  nalu_hypre_AddIndexes(findex, stride, 3, findex);

                  /* cell_index is refined to the upper fine index. Shift
                     back to the lower end, subtract (rfactor-1). */
                  for (j = 0; j < ndim; j++)
                  {
                     findex[j] -= rfactor[j] - 1;
                  }

                  nalu_hypre_SubtractIndexes(findex, varoffsets[var], 3, var_index);

                  switch (var)
                  {
                     case 2:    /* 2-d, x_face */
                     {
                        for (m = 0; m < rfactor[1]; m++)
                        {
                           nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges] = rank;
                           nEdges_iedges++;

                           /* increment the x component to get the next one in the
                              refinement cell. */
                           var_index[1]++;
                        }
                        break;
                     }

                     case 3:    /* 2-d, y_face */
                     {
                        for (m = 0; m < rfactor[0]; m++)
                        {
                           nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges] = rank;
                           nEdges_iedges++;

                           /* increment the y component to get the next one in the
                              refinement cell. */
                           var_index[0]++;
                        }
                        break;
                     }

                     case 5:    /* 3-d, x_edge */
                     {
                        for (m = 0; m < rfactor[0]; m++)
                        {
                           nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges] = rank;
                           nEdges_iedges++;

                           /* increment the x component to get the next one in the
                              refinement cell. */
                           var_index[0]++;
                        }
                        break;
                     }

                     case 6:    /* 3-d, y_edge */
                     {
                        for (m = 0; m < rfactor[1]; m++)
                        {
                           nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges] = rank;
                           nEdges_iedges++;

                           /* increment the y component to get the next one in the
                              refinement cell. */
                           var_index[1]++;
                        }
                        break;
                     }

                     case 7:    /* 3-d, z_edge */
                     {
                        for (m = 0; m < rfactor[2]; m++)
                        {
                           nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges] = rank;
                           nEdges_iedges++;

                           /* increment the z component to get the next one in the
                              refinement cell. */
                           var_index[2]++;
                        }
                        break;
                     }
                  }  /* switch(var) */

               }   /* if ((rank <= cupper_ranks[part][var]) &&
                      (rank >= clower_ranks[part][var])) */
            }
            nalu_hypre_SerialBoxLoop0End();

         }  /* nalu_hypre_ForBoxI(i, cboxes) */
      }     /* for (t= 0; t< Edge_nvars; t++) */
   }        /* for (part= 0; part< nparts; part++) */

   NALU_HYPRE_IJMatrixSetValues(Edge_iedge, nEdges, ncols_Edgeiedge,
                           (const NALU_HYPRE_BigInt*) iEdge, (const NALU_HYPRE_BigInt*) jEdge_iedge,
                           (const NALU_HYPRE_Real*) vals_Edgeiedge);
   NALU_HYPRE_IJMatrixAssemble((NALU_HYPRE_IJMatrix) Edge_iedge);

   nalu_hypre_TFree(ncols_Edgeiedge, memory_location);
   nalu_hypre_TFree(iEdge, memory_location);
   nalu_hypre_TFree(jEdge_iedge, memory_location);
   nalu_hypre_TFree(vals_Edgeiedge, memory_location);

   /* Element_Face & Element_Edge. Element_Face only for 3-d. */
   if (ndim == 3)
   {
      ncols_ElementFace = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nElements, memory_location);
      j = 2 * ndim;
      for (i = 0; i < nElements; i++)
      {
         ncols_ElementFace[i] = j;  /* 3-dim -> 6  */
      }

      j *= nElements;
      jElement_Face    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, j, memory_location);
      vals_ElementFace = nalu_hypre_CTAlloc(NALU_HYPRE_Real, j, memory_location);
      for (i = 0; i < j; i++)
      {
         vals_ElementFace[i] = 1.0;
      }
   }

   ncols_ElementEdge = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nElements, memory_location);
   j = 2 * ndim;
   k = (ndim - 1) * j;
   for (i = 0; i < nElements; i++)
   {
      ncols_ElementEdge[i] = k;  /* 2-dim -> 4; 3-dim -> 12 */
   }

   k *= nElements;
   jElement_Edge   = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, k, memory_location);
   vals_ElementEdge = nalu_hypre_CTAlloc(NALU_HYPRE_Real, k, memory_location);
   for (i = 0; i < k; i++)
   {
      vals_ElementEdge[i] = 1.0;
   }

   /*---------------------------------------------------------------------------
    * Fill up the column ranks of ELement_Face and Element_Edge. Note that the
    * iElement has alrady been formed when filling Element_edge.
    *---------------------------------------------------------------------------*/
   nElements_Faces = 0;
   nElements_Edges = 0;
   for (part = 0; part < nparts; part++)
   {
      /* grab the nvars & vartypes for the face and edge variables */
      if (ndim == 3)
      {
         p_cgrid      = nalu_hypre_SStructGridPGrid(cgrid_face, part);
         Face_nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
         Face_vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);
      }

      p_cgrid      = nalu_hypre_SStructGridPGrid(cgrid_edge, part);  /* Edge grid */
      Edge_nvars   = nalu_hypre_SStructPGridNVars(p_cgrid);
      Edge_vartypes = nalu_hypre_SStructPGridVarTypes(p_cgrid);

      p_cgrid   = nalu_hypre_SStructGridPGrid(cgrid_element, part);  /* cell grid */
      var_cgrid = nalu_hypre_SStructPGridCellSGrid(p_cgrid);
      cboxes    = nalu_hypre_StructGridBoxes(var_cgrid);

      if (ndim == 3)
      {
         nalu_hypre_ForBoxI(i, cboxes)
         {
            cbox = nalu_hypre_BoxArrayBox(cboxes, i);
            nalu_hypre_BoxGetSize(cbox, loop_size);
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cbox), start);

            nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
            {
               zypre_BoxLoopGetIndex(lindex);
               nalu_hypre_SetIndex3(cindex, lindex[0], lindex[1], lindex[2]);
               nalu_hypre_AddIndexes(cindex, start, 3, cindex);

               /*-------------------------------------------------------------
                * jElement_Face: (i,j,k) then (i-1,j,k), (i,j-1,k), (i,j,k-1).
                *-------------------------------------------------------------*/
               for (t = 0; t < Face_nvars; t++)
               {
                  var = Face_vartypes[t];

                  nalu_hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, t,
                                                   &entry);
                  nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                        matrix_type);
                  jElement_Face[nElements_Faces] = rank;
                  nElements_Faces++;

                  nalu_hypre_SubtractIndexes(cindex, varoffsets[var], 3, var_index);
                  nalu_hypre_SStructGridFindBoxManEntry(cgrid_face, part, var_index, t,
                                                   &entry);
                  nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                        matrix_type);
                  jElement_Face[nElements_Faces] = rank;
                  nElements_Faces++;
               }

            }
            nalu_hypre_SerialBoxLoop0End();
         }  /* nalu_hypre_ForBoxI(i, cboxes) */
      }  /* if (ndim == 3) */

      /*-------------------------------------------------------------------
       * jElement_Edge:
       *    3-dim
       *       x_Edge: (i,j,k) then (i,j-1,k), (i,j-1,k-1), (i,j,k-1)
       *       y_Edge: (i,j,k) then (i-1,j,k), (i-1,j,k-1), (i,j,k-1)
       *       z_Edge: (i,j,k) then (i,j-1,k), (i-1,j-1,k), (i-1,j,k)
       *
       *    2-dim
       *       x_Edge or x_Face: (i,j) then (i-1,j)
       *       y_Edge or y_Face: (i,j) then (i,j-1)
       *-------------------------------------------------------------------*/
      nalu_hypre_ForBoxI(i, cboxes)
      {
         cbox = nalu_hypre_BoxArrayBox(cboxes, i);
         nalu_hypre_BoxGetSize(cbox, loop_size);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cbox), start);

         nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
         {
            zypre_BoxLoopGetIndex(lindex);
            nalu_hypre_SetIndex3(cindex, lindex[0], lindex[1], lindex[2]);
            nalu_hypre_AddIndexes(cindex, start, 3, cindex);

            for (t = 0; t < Edge_nvars; t++)
            {
               /* Edge (i,j,k) */
               var = Edge_vartypes[t];

               switch (var)
               {
                  case 2: /* x_Face= {(i,j), (i-1,j)} */
                  {
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;
                     break;
                  }

                  case 3: /* y_Face= {(i,j), (i,j-1)} */
                  {
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;
                     break;
                  }

                  case 5: /* "/" x_Edge={(i,j,k),(i,j-1,k),(i,j-1,k-1),(i,j,k-1)} */
                  {
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_AddIndexes(var_index, jshift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;
                     break;
                  }

                  case 6: /* "-" y_Edge={(i,j,k),(i-1,j,k),(i-1,j,k-1),(i,j,k-1)}*/
                  {
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_AddIndexes(var_index, ishift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;
                     break;
                  }

                  case 7: /* "|" z_Edge={(i,j,k),(i,j-1,k),(i-1,j-1,k),(i-1,j,k)}*/
                  {
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_SubtractIndexes(var_index, ishift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;

                     nalu_hypre_AddIndexes(var_index, jshift, 3, var_index);
                     nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges] = rank;
                     nElements_Edges++;
                     break;
                  }

               }   /* switch (var) */
            }      /* for (t= 0; t< Edge_nvars; t++) */
         }
         nalu_hypre_SerialBoxLoop0End();
      }  /* nalu_hypre_ForBoxI(i, cboxes) */
   }     /* for (part= 0; part< nparts; part++) */

   if (ndim == 3)
   {
      NALU_HYPRE_IJMatrixSetValues(Element_Face, nElements, ncols_ElementFace,
                              (const NALU_HYPRE_BigInt*) iElement, (const NALU_HYPRE_BigInt*) jElement_Face,
                              (const NALU_HYPRE_Real*) vals_ElementFace);
      NALU_HYPRE_IJMatrixAssemble((NALU_HYPRE_IJMatrix) Element_Face);

      nalu_hypre_TFree(ncols_ElementFace, memory_location);
      nalu_hypre_TFree(jElement_Face, memory_location);
      nalu_hypre_TFree(vals_ElementFace, memory_location);
   }  /* if (ndim == 3) */

   NALU_HYPRE_IJMatrixSetValues(Element_Edge, nElements, ncols_ElementEdge,
                           (const NALU_HYPRE_BigInt*) iElement, (const NALU_HYPRE_BigInt*) jElement_Edge,
                           (const NALU_HYPRE_Real*) vals_ElementEdge);
   NALU_HYPRE_IJMatrixAssemble((NALU_HYPRE_IJMatrix) Element_Edge);

   nalu_hypre_TFree(ncols_ElementEdge, memory_location);
   nalu_hypre_TFree(iElement, memory_location);
   nalu_hypre_TFree(jElement_Edge, memory_location);
   nalu_hypre_TFree(vals_ElementEdge, memory_location);

   /*-----------------------------------------------------------------------
    * edge_Edge, the actual interpolation matrix.
    * For each fine edge row, we need to know if it is a edge,
    * boundary edge, or face edge. Knowing this allows us to determine the
    * structure and weights of the interpolation matrix.
    *
    * Scheme:A.Loop over contracted boxes of fine edge grid.
    *          For each fine edge ijk,
    *     1) map it to a fine cell with the fine edge at the lower end
    *        of the box,e.g. x_edge[ijk] -> cell[i,j+1,k+1].
    *     2) coarsen the fine cell to obtain a coarse cell. Determine the
    *        location of the fine edge with respect to the coarse edges
    *        of this cell. Coarsening needed only when determining the
    *        column rank.
    *
    * Need to distinguish between 2-d and 3-d.
    *-----------------------------------------------------------------------*/

   /* count the row/col connections */
   iedgeEdge      = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nedges, memory_location);
   ncols_edgeEdge = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nedges, memory_location);

   /*-----------------------------------------------------------------------
    * loop first over the fedges aligning with the agglomerate coarse edges.
    * Will loop over the face & interior edges separately also.
    * Since the weights for these edges will be used to determine the
    * weights along the face edges, we need to retrieve these computed
    * weights from vals_edgeEdge. Done by keeping a pointer of size nedges
    * that points to the location of the weight:
    *          pointer[rank of edge]= index location where weight resides.
    *-----------------------------------------------------------------------*/
   j = 0;
   start_rank1 = nalu_hypre_SStructGridStartRank(fgrid_edge);
   bdryedge_location = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nedges, NALU_HYPRE_MEMORY_HOST);
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
         boxoffset   = nalu_hypre_CTAlloc(nalu_hypre_Index, n_boxoffsets, NALU_HYPRE_MEMORY_HOST);
         suboffset   = nalu_hypre_CTAlloc(nalu_hypre_Index, n_boxoffsets, NALU_HYPRE_MEMORY_HOST);
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
               nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &rank, matrix_type);
               /* still row p may be outside the processor- check to make sure in */
               if ((rank <= fupper_ranks[part][var]) && (rank >= flower_ranks[part][var]))
               {
                  iedgeEdge[j] = rank;
                  ncols_edgeEdge[j] = 1;
                  bdryedge_location[rank - start_rank1] = j;
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
    *
    * Loop over the face edges.
    * Since the weights for these edges will be used to determine the
    * weights along the face edges, we need to retrieve these computed
    * weights form vals_edgeEdge. Done by keeping a pointer of size nedges
    * that points to the location of the weight:
    *          pointer[rank of edge]= index location where weight resides.
    *-----------------------------------------------------------------------*/
   if (ndim == 3)
   {
      l = j;
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
            var_fgrid =  nalu_hypre_SStructPGridVTSGrid(p_fgrid, var);
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
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              /* still row rank may be outside the processor */
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j] = rank;

                                 /* Z_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank - start_rank1] = l;
                                 l += 2; /* two weight values */
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

                        /************************************************************
                         * Loop over the Y_Face x_edges.
                         ************************************************************/
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j] = rank;

                                 /* Y_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank - start_rank1] = l;
                                 l += 2;
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

                     /* reset and then increase the loop_size by one in the Z_Face direction */
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
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j] = rank;

                                 /* Z_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank - start_rank1] = l;
                                 l += 2;
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

                        /*****************************************************
                         * Loop over the X_Face y_edges.
                         *****************************************************/
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j] = rank;
                                 /* X_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank - start_rank1] = l;
                                 l += 2;
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

                        /******************************************************
                         * Loop over the X_Face z_edges.
                         ******************************************************/
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j] = rank;

                                 /* X_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank - start_rank1] = l;
                                 l += 2;
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
                        /****************************************************
                         * Loop over the Y_Face z_edges.
                         ****************************************************/
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j] = rank;

                                 /* Y_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank - start_rank1] = l;
                                 l += 2;
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
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);

                  /* adjust the contracted cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));

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
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           iedgeEdge[j] = rank;

                           /* lies interior of Face. Four coarse Edge connection. */
                           ncols_edgeEdge[j] = 4;
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
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));

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
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           iedgeEdge[j] = rank;

                           /* lies interior of Face. Four coarse Edge connection. */
                           ncols_edgeEdge[j] = 4;
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
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);

                  /* adjust the project cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));

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
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              iedgeEdge[j] = rank;

                              /* Interior. Twelve coarse Edge connections. */
                              ncols_edgeEdge[j] = 12;
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
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);

                  /* adjust the contract cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));

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
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              iedgeEdge[j] = rank;

                              /* Interior. Twelve coarse Edge connections. */
                              ncols_edgeEdge[j] = 12;
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
                  vbox   = nalu_hypre_BoxArrayBox(box_array, i);

                  /* adjust the contracted cellbox to the variable box */
                  nalu_hypre_CopyBox(cellbox, &copy_box);
                  nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        nalu_hypre_BoxIMin(&copy_box));

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
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              iedgeEdge[j] = rank;

                              /* Interior. Twelve coarse Edge connections. */
                              ncols_edgeEdge[j] = 12;
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
   vals_edgeEdge = nalu_hypre_CTAlloc(NALU_HYPRE_Real, k, memory_location);
   jedge_Edge    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, k, memory_location);
   size1         = j;

   /*********************************************************************
    * Fill up the edge_Edge interpolation matrix. Interpolation weights
    * are determined differently for each type of fine edges.
    *
    * fedge_on_CEdge: use geometric interpolation, i.e., length of
    * edge ratio.
    *
    * fedge_on_agglomerate_face: box mg approach. Collapse the like
    * variable stencil connections of the given face. Weighted linear
    * interpolation of the fedge_on_CEdge values.
    *
    * fedge_in_agglomerate_interior: amge.
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
               NALU_HYPRE_BigInt big_j;
               zypre_BoxLoopGetIndex(lindex);
               nalu_hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
               for (j = 0; j < 3; j++)
               {
                  findex[j] *= stride[j];
               }

               nalu_hypre_AddIndexes(findex, start, 3, findex);
               nalu_hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t, &entry);
               nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &big_j, matrix_type);

               /* still row p may be outside the processor- check to make sure in */
               if ((big_j <= fupper_ranks[part][var]) && (big_j >= flower_ranks[part][var]))
               {
                  nalu_hypre_SubtractIndexes(findex, start, 3, findex);

                  /* determine where the edge lies- coarsening required. */
                  nalu_hypre_StructMapFineToCoarse(findex, zero_index, rfactor,
                                              cindex);
                  nalu_hypre_AddIndexes(cindex, cstart, 3, cindex);
                  nalu_hypre_AddIndexes(findex, start, 3, findex);

                  /* lies on coarse Edge. Coarse Edge connection:
                     var_index= cindex - subtract_index.*/
                  nalu_hypre_SubtractIndexes(cindex, varoffsets[var], 3, var_index);

                  nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                   t, &entry);
                  nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                        matrix_type);
                  jedge_Edge[k] = rank;
                  vals_edgeEdge[k] = fCedge_ratio;

                  k++;
               }
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
      /* Allocate memory to arrays for the tridiagonal system & solutions.
         Take the maximum size needed. */
      i = rfactor[0] - 1;
      for (j = 1; j < ndim; j++)
      {
         if (i < (rfactor[j] - 1))
         {
            i = rfactor[j] - 1;
         }
      }
      upper = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  i, NALU_HYPRE_MEMORY_HOST);
      lower = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  i, NALU_HYPRE_MEMORY_HOST);
      diag = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  i, NALU_HYPRE_MEMORY_HOST);
      face_w1 = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  i, NALU_HYPRE_MEMORY_HOST);
      face_w2 = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  i, NALU_HYPRE_MEMORY_HOST);
      off_proc_flag = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  i + 1, NALU_HYPRE_MEMORY_HOST);

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
                  fCedge_ratio = 1.0 / rfactor[0];
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

                        /* loop over the strips of x_edges making up the Z_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n = 0; n < rfactor[1] - 1; n++)
                           {
                              face_w1[n] = 0.0;
                              face_w2[n] = 0.0;
                           }

                           /******************************************************
                            * grab the already computed lower-end edge weight.
                            * These are bdry agglomerate wgts that are pre-determined
                            * so that no communication is needed.
                            ******************************************************/

                           /* lower-end and upper-end edge weights */
                           face_w1[0] = fCedge_ratio;
                           face_w2[rfactor[1] - 2] = fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * x_edge for Z_Face: collapse_dir= 2, stencil_dir= 1
                            ******************************************************/
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              off_proc_flag[n] =
                                 nalu_hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                2,
                                                                1,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n - 1] = stencil_vals[0];
                              diag[n - 1] = stencil_vals[1];
                              upper[n - 1] = stencil_vals[2];
                              nalu_hypre_TFree(stencil_vals, NALU_HYPRE_MEMORY_HOST);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0] *= -lower[0];
                           face_w2[rfactor[1] - 2] *= -upper[rfactor[1] - 2];
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[1] - 1);
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[1] - 1);

                           /* place weights into vals_edgeEdge */
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = face_w1[n - 1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = face_w2[n - 1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);

                     /* Y_Face */
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

                        /* loop over the strips of x_edges making up the Y_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n = 0; n < rfactor[2] - 1; n++)
                           {
                              face_w1[n] = 0.0;
                              face_w2[n] = 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0] = fCedge_ratio;
                           face_w2[rfactor[2] - 2] = fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * x_edge for Y_Face: collapse_dir= 1, stencil_dir= 2
                            ******************************************************/
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              off_proc_flag[n] =
                                 nalu_hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                1,
                                                                2,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n - 1] = stencil_vals[0];
                              diag[n - 1] = stencil_vals[1];
                              upper[n - 1] = stencil_vals[2];
                              nalu_hypre_TFree(stencil_vals, NALU_HYPRE_MEMORY_HOST);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0] *= -lower[0];
                           face_w2[rfactor[2] - 2] *= -upper[rfactor[2] - 2];
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[2] - 1);
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[2] - 1);

                           /* place weights into vals_edgeEdge */
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = face_w1[n - 1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = face_w2[n - 1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[0]; p++) */

                     }
                     nalu_hypre_SerialBoxLoop1End(m);
                  }  /* nalu_hypre_ForBoxI(i, fboxes) */
                  break;
               }

               case 6:
               {
                  /* 3-d y_edge, can be X or Z_Face */
                  fCedge_ratio = 1.0 / rfactor[1];
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

                        /* loop over the strips of y_edges making up the Z_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n = 0; n < rfactor[0] - 1; n++)
                           {
                              face_w1[n] = 0.0;
                              face_w2[n] = 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0] = fCedge_ratio;
                           face_w2[rfactor[0] - 2] = fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * y_edge for Z_Face: collapse_dir= 2, stencil_dir= 0
                            ******************************************************/
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              off_proc_flag[n] =
                                 nalu_hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                2,
                                                                0,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n - 1] = stencil_vals[0];
                              diag[n - 1] = stencil_vals[1];
                              upper[n - 1] = stencil_vals[2];
                              nalu_hypre_TFree(stencil_vals, NALU_HYPRE_MEMORY_HOST);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0] *= -lower[0];
                           face_w2[rfactor[0] - 2] *= -upper[rfactor[0] - 2];
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[0] - 1);
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[0] - 1);

                           /* place weights into vals_edgeEdge */
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = face_w1[n - 1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = face_w2[n - 1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[1]; p++) */
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

                        /* loop over the strips of y_edges making up the X_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n = 0; n < rfactor[2] - 1; n++)
                           {
                              face_w1[n] = 0.0;
                              face_w2[n] = 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0] = fCedge_ratio;
                           face_w2[rfactor[0] - 2] = fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * y_edge for X_Face: collapse_dir= 0, stencil_dir= 2
                            ******************************************************/
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              off_proc_flag[n] =
                                 nalu_hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                0,
                                                                2,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n - 1] = stencil_vals[0];
                              diag[n - 1] = stencil_vals[1];
                              upper[n - 1] = stencil_vals[2];
                              nalu_hypre_TFree(stencil_vals, NALU_HYPRE_MEMORY_HOST);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0] *= -lower[0];
                           face_w2[rfactor[2] - 2] *= -upper[rfactor[2] - 2];
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[2] - 1);
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[2] - 1);

                           /* place weights into vals_edgeEdge */
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = face_w1[n - 1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = face_w2[n - 1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[1]; p++) */

                     }
                     nalu_hypre_SerialBoxLoop1End(m);
                  }  /* nalu_hypre_ForBoxI(i, fboxes) */
                  break;
               }

               case 7:
               {
                  /* 3-d z_edge, can be X or Y_Face */
                  fCedge_ratio = 1.0 / rfactor[2];
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

                        /* loop over the strips of z_edges making up the X_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n = 0; n < rfactor[1] - 1; n++)
                           {
                              face_w1[n] = 0.0;
                              face_w2[n] = 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0] = fCedge_ratio;
                           face_w2[rfactor[1] - 2] = fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * z_edge for X_Face: collapse_dir= 0, stencil_dir= 1
                            ******************************************************/
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              off_proc_flag[n] =
                                 nalu_hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                0,
                                                                1,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n - 1] = stencil_vals[0];
                              diag[n - 1] = stencil_vals[1];
                              upper[n - 1] = stencil_vals[2];
                              nalu_hypre_TFree(stencil_vals, NALU_HYPRE_MEMORY_HOST);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0] *= -lower[0];
                           face_w2[rfactor[1] - 2] *= -upper[rfactor[1] - 2];
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[1] - 1);
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[1] - 1);

                           /* place weights into vals_edgeEdge */
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = face_w1[n - 1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = face_w2[n - 1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     nalu_hypre_SerialBoxLoop1End(m);

                     /* Y_Face */
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

                        /* loop over the strips of y_edges making up the Y_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n = 0; n < rfactor[0] - 1; n++)
                           {
                              face_w1[n] = 0.0;
                              face_w2[n] = 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0] = fCedge_ratio;
                           face_w2[rfactor[0] - 2] = fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * z_edge for Y_Face: collapse_dir= 1, stencil_dir= 0
                            ******************************************************/
                           nalu_hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              off_proc_flag[n] =
                                 nalu_hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                1,
                                                                0,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n - 1] = stencil_vals[0];
                              diag[n - 1] = stencil_vals[1];
                              upper[n - 1] = stencil_vals[2];
                              nalu_hypre_TFree(stencil_vals, NALU_HYPRE_MEMORY_HOST);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0] *= -lower[0];
                           face_w2[rfactor[0] - 2] *= -upper[rfactor[0] - 2];
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[0] - 1);
                           nalu_hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[0] - 1);

                           /* place weights into vals_edgeEdge */
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = face_w1[n - 1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = face_w2[n - 1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[2]; p++) */

                     }
                     nalu_hypre_SerialBoxLoop1End(m);
                  }  /* nalu_hypre_ForBoxI(i, fboxes) */
                  break;
               }

            }  /* switch */
         }     /* for (t= 0; t< Edge_nvars; t++) */

         nalu_hypre_TFree(boxoffset, NALU_HYPRE_MEMORY_HOST);
      }  /* for (part= 0; part< nparts; part++) */

      nalu_hypre_TFree(upper, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(lower, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(diag, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(face_w1, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(face_w2, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(off_proc_flag, NALU_HYPRE_MEMORY_HOST);
   }  /* if (ndim == 3) */

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

                           /*interior of Face. Extract the four coarse Edge
                             (x_Edge ijk & (i-1,j,k) and y_Edge ijk & (i,j-1,k)
                             column ranks. No weights determined. */
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           k++;

                           nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           k++;

                           /* y_Edges */
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            vartype_map[3], &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           k++;

                           nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            vartype_map[3], &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
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

                           /*lies interior of Face. Extract the four coarse Edge
                             (y_Edge ijk & (i,j-1,k) and x_Edge ijk & (i-1,j,k)
                             column ranks. No weights determined. */
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           k++;

                           nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            t, &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           k++;

                           /* x_Edges */
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            vartype_map[2], &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           k++;

                           nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                           nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            vartype_map[2], &entry);
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
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
                               * y_Edge ijk, (i-1,j,k), (i-1,j,k-1), (i,j,k-1)
                               * z_Edge ijk, (i-1,j,k), (i-1,j-1,k), (i,j-1,k)
                               *
                               * vals_edgeEdge's are not set.
                               ***********************************************/
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_AddIndexes(var_index, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              /* y_Edge */
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[6], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_AddIndexes(var_index, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              /* z_Edge */
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[7], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_AddIndexes(var_index, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
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
                               * z_Edge ijk, (i-1,j,k), (i-1,j-1,k), (i,j-1,k)
                               * x_Edge ijk, (i,j-1,k), (i,j-1,k-1), (i,j,k-1)
                               *
                               * vals_edgeEdge's are not set.
                               ***********************************************/
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_AddIndexes(var_index, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              /* z_Edge */
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[7], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_AddIndexes(var_index, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              /* x_Edge */
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[5], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_AddIndexes(var_index, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
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
                               * x_Edge ijk, (i,j-1,k), (i,j-1,k-1), (i,j,k-1)
                               * y_Edge ijk, (i-1,j,k), (i-1,j,k-1), (i,j,k-1)
                               *
                               * vals_edgeEdge's are not set.
                               *************************************************/
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_AddIndexes(var_index, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              /* x_Edge */
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[5], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_AddIndexes(var_index, jshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              /* y_Edge */
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[6], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              k++;

                              nalu_hypre_AddIndexes(var_index, ishift, 3, var_index);
                              nalu_hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
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
   nalu_hypre_TFree(bdryedge_location, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_IJMatrixSetValues(edge_Edge, size1, ncols_edgeEdge,
                           (const NALU_HYPRE_BigInt*) iedgeEdge, (const NALU_HYPRE_BigInt*) jedge_Edge,
                           (const NALU_HYPRE_Real*) vals_edgeEdge);
   NALU_HYPRE_IJMatrixAssemble((NALU_HYPRE_IJMatrix) edge_Edge);

   nalu_hypre_TFree(ncols_edgeEdge, memory_location);
   nalu_hypre_TFree(iedgeEdge, memory_location);
   nalu_hypre_TFree(jedge_Edge, memory_location);
   nalu_hypre_TFree(vals_edgeEdge, memory_location);

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
      p_fgrid = nalu_hypre_SStructGridPGrid(fgrid_edge, part);
      var_fgrid = nalu_hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = nalu_hypre_StructGridBoxes(var_fgrid);

      nalu_hypre_BoxArrayDestroy(contract_fedgeBoxes[part]);
      nalu_hypre_TFree(Edge_cstarts[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(upper_shifts[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(lower_shifts[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cfbox_mapping[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fcbox_mapping[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fupper_ranks[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(flower_ranks[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cupper_ranks[part], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(clower_ranks[part], NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(contract_fedgeBoxes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Edge_cstarts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(upper_shifts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(lower_shifts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cfbox_mapping, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fcbox_mapping, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fupper_ranks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(flower_ranks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cupper_ranks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(clower_ranks, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(varoffsets, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vartype_map, NALU_HYPRE_MEMORY_HOST);

   if (ndim > 2)
   {
      (PTopology ->  Face_iedge)   = Face_iedge;
      (PTopology ->  Element_Face) = Element_Face;
   }
   (PTopology ->  Element_iedge) = Element_iedge;
   (PTopology ->  Edge_iedge)   = Edge_iedge;
   (PTopology ->  Element_Edge) = Element_Edge;

   return edge_Edge;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CollapseStencilToStencil: Collapses 3d stencil shape & values to
 * a 2d 3-point stencil: collapsed_vals= [ldiag diag udiag].
 * Algo:
 *    1) Given the collapsing direction & the collapsed stencil pattern,
 *       group the ranks into three collapsed sets: diag_ranks, ldiag_ranks,
 *       udiag_ranks.
 *    2) concatenate these sets, marking the set location
 *    3) qsort the concatenated set and the col_inds
 *    4) search compare the two sorted arrays to compute the collapsed vals.
 *
 *  Example, suppose collapsing to y_edges. Then the new_stencil pattern
 *    is [n c s]^t and we need to collapse in the x direction to get this
 *    3-pt stencil: collapse_dir= 0 & new_stencil_dir= 1.
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_CollapseStencilToStencil(nalu_hypre_ParCSRMatrix     *Aee,
                               nalu_hypre_SStructGrid      *grid,
                               NALU_HYPRE_Int               part,
                               NALU_HYPRE_Int               var,
                               nalu_hypre_Index             pt_location,
                               NALU_HYPRE_Int               collapse_dir,
                               NALU_HYPRE_Int               new_stencil_dir,
                               NALU_HYPRE_Real            **collapsed_vals_ptr)
{
   NALU_HYPRE_Int                ierr = 0;

   NALU_HYPRE_Int                matrix_type = NALU_HYPRE_PARCSR;
   NALU_HYPRE_BigInt             start_rank = nalu_hypre_ParCSRMatrixFirstRowIndex(Aee);
   NALU_HYPRE_BigInt             end_rank   = nalu_hypre_ParCSRMatrixLastRowIndex(Aee);

   nalu_hypre_BoxManEntry       *entry;

   NALU_HYPRE_BigInt            *ranks;
   NALU_HYPRE_Int               *marker;     /* marker to record the rank groups */
   NALU_HYPRE_Int                max_ranksize = 9;

   NALU_HYPRE_Real              *collapsed_vals;

   nalu_hypre_Index              index1, index2;

   NALU_HYPRE_Int                size;
   NALU_HYPRE_BigInt            *col_inds, *col_inds2;
   NALU_HYPRE_Real              *values;
   NALU_HYPRE_BigInt             rank, row_rank;
   NALU_HYPRE_Int               *swap_inds;

   NALU_HYPRE_Int                i, j, m, centre, found;
   NALU_HYPRE_Int                getrow_ierr;
   NALU_HYPRE_Int                cnt;

   /* create the collapsed stencil coefficients. Three components. */
   collapsed_vals = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  3, NALU_HYPRE_MEMORY_HOST);

   /* check if the row corresponding to pt_location is on this proc. If
      not, return an identity row. THIS SHOULD BE CORRECTED IN THE FUTURE
      TO GIVE SOMETHING MORE REASONABLE. */
   nalu_hypre_SStructGridFindBoxManEntry(grid, part, pt_location, var, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, pt_location, &rank, matrix_type);
   if (rank < start_rank || rank > end_rank)
   {
      collapsed_vals[1] = 1.0;
      *collapsed_vals_ptr = collapsed_vals;
      ierr = 1;
      return ierr;
   }

   /* Extract the ranks of the collapsed stencil pattern. Since only like-var
      collapsing, we assume that max stencil size is 9. This agrees with the
      assumed pattern surrounding pt_location. Concatenating done. */
   ranks = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  max_ranksize, NALU_HYPRE_MEMORY_HOST);
   marker = nalu_hypre_TAlloc(NALU_HYPRE_Int,  max_ranksize, NALU_HYPRE_MEMORY_HOST);

   cnt = 0;
   centre = 0;
   for (j = -1; j <= 1; j++)
   {
      nalu_hypre_CopyIndex(pt_location, index1);
      index1[new_stencil_dir] += j;

      for (i = -1; i <= 1; i++)
      {
         nalu_hypre_CopyIndex(index1, index2);
         index2[collapse_dir] += i;

         nalu_hypre_SStructGridFindBoxManEntry(grid, part, index2, var, &entry);
         if (entry)
         {
            nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index2, &rank, matrix_type);
            ranks[cnt] = rank;
            marker[cnt] = j + 1;

            /* mark centre component- entry!=NULL always */
            if ( (!i) && (!j) )
            {
               centre = cnt;
            }
            cnt++;
         }
      }
   }

   /* Grab the row corresponding to index pt_location. rank located in location
      centre of ranks, i.e., rank for index2= pt_location. Mark location of values,
      which will record the original location of values after the sorting. */
   row_rank = ranks[centre];
   getrow_ierr = NALU_HYPRE_ParCSRMatrixGetRow((NALU_HYPRE_ParCSRMatrix) Aee, row_rank,
                                          &size, &col_inds, &values);
   if (getrow_ierr < 0)
   {
      nalu_hypre_printf("offproc collapsing problem");
   }

   swap_inds = nalu_hypre_TAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
   col_inds2 = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  size, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < size; i++)
   {
      swap_inds[i] = i;
      col_inds2[i] = col_inds[i];
   }

   /* qsort ranks & col_inds */
   nalu_hypre_BigQsortbi(ranks, marker, 0, cnt - 1);
   nalu_hypre_BigQsortbi(col_inds2, swap_inds, 0, size - 1);

   /* search for values to collapse */
   m = 0;
   for (i = 0; i < cnt; i++)
   {
      found = 0;
      while (!found)
      {
         if (ranks[i] != col_inds2[m])
         {
            m++;
         }
         else
         {
            collapsed_vals[marker[i]] += values[swap_inds[m]];
            m++;
            break; /* break out of while loop */
         }
      }  /* while (!found) */
   }  /* for (i= 0; i< cnt; i++) */

   NALU_HYPRE_ParCSRMatrixRestoreRow((NALU_HYPRE_ParCSRMatrix) Aee, row_rank, &size,
                                &col_inds, &values);

   nalu_hypre_TFree(col_inds2, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ranks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(swap_inds, NALU_HYPRE_MEMORY_HOST);

   *collapsed_vals_ptr = collapsed_vals;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_TriDiagSolve: Direct tridiagonal solve
 *------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_TriDiagSolve(NALU_HYPRE_Real *diag,
                   NALU_HYPRE_Real *upper,
                   NALU_HYPRE_Real *lower,
                   NALU_HYPRE_Real *rhs,
                   NALU_HYPRE_Int   size)
{
   NALU_HYPRE_Int       ierr = 0;

   NALU_HYPRE_Int       i, size1;
   NALU_HYPRE_Real     *copy_diag;
   NALU_HYPRE_Real      multiplier;

   size1 = size - 1;

   /* copy diag so that the matrix is not modified */
   copy_diag = nalu_hypre_TAlloc(NALU_HYPRE_Real,  size, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < size; i++)
   {
      copy_diag[i] = diag[i];
   }

   /* forward substitution */
   for (i = 1; i < size; i++)
   {
      multiplier = -lower[i] / copy_diag[i - 1];
      copy_diag[i] += multiplier * upper[i - 1];
      rhs[i] += multiplier * rhs[i - 1];
   }

   /* backward substitution */
   rhs[size1] /= copy_diag[size1];
   for (i = size1 - 1; i >= 0; i--)
   {
      rhs[i] = (rhs[i] - upper[i] * rhs[i + 1]) / copy_diag[i];
   }

   nalu_hypre_TFree(copy_diag, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}
