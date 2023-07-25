/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "fac.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_FacSetup2: Constructs the level composite structures.
 * Each consists only of two levels, the refinement patches and the
 * coarse parent base grids.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FacSetup2( void                 *fac_vdata,
                 nalu_hypre_SStructMatrix  *A_in,
                 nalu_hypre_SStructVector  *b,
                 nalu_hypre_SStructVector  *x )
{
   nalu_hypre_FACData          *fac_data      =  (nalu_hypre_FACData*)fac_vdata;

   NALU_HYPRE_Int              *plevels       = (fac_data-> plevels);
   nalu_hypre_Index            *rfactors      = (fac_data-> prefinements);

   MPI_Comm                comm;
   NALU_HYPRE_Int               ndim;
   NALU_HYPRE_Int               npart;
   NALU_HYPRE_Int               nparts_level  =  2;
   NALU_HYPRE_Int               part_crse     =  0;
   NALU_HYPRE_Int               part_fine     =  1;
   nalu_hypre_SStructPMatrix   *A_pmatrix;
   nalu_hypre_StructMatrix     *A_smatrix;
   nalu_hypre_Box              *A_smatrix_dbox;

   nalu_hypre_SStructGrid     **grid_level;
   nalu_hypre_SStructGraph    **graph_level;
   NALU_HYPRE_Int               part, level;
   NALU_HYPRE_Int               nvars;

   nalu_hypre_SStructGraph     *graph;
   nalu_hypre_SStructGrid      *grid;
   nalu_hypre_SStructPGrid     *pgrid;
   nalu_hypre_StructGrid       *sgrid;
   nalu_hypre_BoxArray         *sgrid_boxes;
   nalu_hypre_Box              *sgrid_box;
   nalu_hypre_SStructStencil   *stencils;
   nalu_hypre_BoxArray         *iboxarray;

   nalu_hypre_Index            *refine_factors;
   nalu_hypre_IndexRef          box_start;
   nalu_hypre_IndexRef          box_end;

   nalu_hypre_SStructUVEntry  **Uventries;
   NALU_HYPRE_Int               nUventries;
   NALU_HYPRE_Int              *iUventries;
   nalu_hypre_SStructUVEntry   *Uventry;
   nalu_hypre_SStructUEntry    *Uentry;
   nalu_hypre_Index             index, to_index, stride;
   NALU_HYPRE_Int               var, to_var, to_part, level_part, level_topart;
   NALU_HYPRE_Int               var1, var2;
   NALU_HYPRE_Int               i, j, k, nUentries;
   NALU_HYPRE_BigInt            row_coord, to_rank;
   nalu_hypre_BoxManEntry      *boxman_entry;

   nalu_hypre_SStructMatrix    *A_rap;
   nalu_hypre_SStructMatrix   **A_level;
   nalu_hypre_SStructVector   **b_level;
   nalu_hypre_SStructVector   **x_level;
   nalu_hypre_SStructVector   **r_level;
   nalu_hypre_SStructVector   **e_level;
   nalu_hypre_SStructPVector  **tx_level;
   nalu_hypre_SStructVector    *tx;

   void                  **matvec_data_level;
   void                  **pmatvec_data_level;
   void                   *matvec_data;
   void                  **relax_data_level;
   void                  **interp_data_level;
   void                  **restrict_data_level;


   /* coarsest grid solver */
   NALU_HYPRE_Int               csolver_type       = (fac_data-> csolver_type);
   NALU_HYPRE_SStructSolver     crse_solver = NULL;
   NALU_HYPRE_SStructSolver     crse_precond = NULL;

   NALU_HYPRE_Int               max_level        =  nalu_hypre_FACDataMaxLevels(fac_data);
   NALU_HYPRE_Int               relax_type       =  fac_data -> relax_type;
   NALU_HYPRE_Int               usr_jacobi_weight =  fac_data -> usr_jacobi_weight;
   NALU_HYPRE_Real              jacobi_weight    =  fac_data -> jacobi_weight;
   NALU_HYPRE_Int              *levels;
   NALU_HYPRE_Int              *part_to_level;

   NALU_HYPRE_Int               box, box_volume;
   NALU_HYPRE_Int               max_box_volume;
   NALU_HYPRE_Int               stencil_size;
   nalu_hypre_Index             stencil_shape_i, loop_size;
   NALU_HYPRE_Int              *stencil_vars;
   NALU_HYPRE_Real             *values;
   NALU_HYPRE_Real             *A_smatrix_value;

   NALU_HYPRE_Int              *nrows;
   NALU_HYPRE_Int             **ncols;
   NALU_HYPRE_BigInt          **rows;
   NALU_HYPRE_BigInt          **cols;
   NALU_HYPRE_Int              *cnt;
   NALU_HYPRE_Real             *vals;

   NALU_HYPRE_BigInt           *level_rows;
   NALU_HYPRE_BigInt           *level_cols;
   NALU_HYPRE_Int               level_cnt;

   NALU_HYPRE_IJMatrix          ij_A;
   NALU_HYPRE_Int               matrix_type;

   NALU_HYPRE_Int               max_cycles;

   NALU_HYPRE_Int               ierr = 0;
   /*nalu_hypre_SStructMatrix *nested_A;

     nested_A= nalu_hypre_TAlloc(nalu_hypre_SStructMatrix ,  1, NALU_HYPRE_MEMORY_HOST);
     nested_A= nalu_hypre_CoarsenAMROp(fac_vdata, A);*/

   /* generate the composite operator with the computed coarse-grid operators */
   nalu_hypre_AMR_RAP(A_in, rfactors, &A_rap);
   (fac_data -> A_rap) = A_rap;

   comm = nalu_hypre_SStructMatrixComm(A_rap);
   ndim = nalu_hypre_SStructMatrixNDim(A_rap);
   npart = nalu_hypre_SStructMatrixNParts(A_rap);
   graph = nalu_hypre_SStructMatrixGraph(A_rap);
   grid = nalu_hypre_SStructGraphGrid(graph);
   ij_A = nalu_hypre_SStructMatrixIJMatrix(A_rap);
   matrix_type = nalu_hypre_SStructMatrixObjectType(A_rap);

   /*--------------------------------------------------------------------------
    * logging arrays.
    *--------------------------------------------------------------------------*/
   if ((fac_data -> logging) > 0)
   {
      max_cycles = (fac_data -> max_cycles);
      (fac_data -> norms)    = nalu_hypre_TAlloc(NALU_HYPRE_Real,  max_cycles, NALU_HYPRE_MEMORY_HOST);
      (fac_data -> rel_norms) = nalu_hypre_TAlloc(NALU_HYPRE_Real,  max_cycles, NALU_HYPRE_MEMORY_HOST);
   }

   /*--------------------------------------------------------------------------
    * Extract the amr/sstruct level/part structure and refinement factors.
    *--------------------------------------------------------------------------*/
   levels        = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  npart, NALU_HYPRE_MEMORY_HOST);
   part_to_level = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  npart, NALU_HYPRE_MEMORY_HOST);
   refine_factors = nalu_hypre_CTAlloc(nalu_hypre_Index,  npart, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < npart; part++)
   {
      part_to_level[part]  = plevels[part];
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
   (fac_data -> level_to_part) = levels;
   (fac_data -> part_to_level) = part_to_level;
   (fac_data -> refine_factors) = refine_factors;

   /*--------------------------------------------------------------------------
    * Create the level SStructGrids using the original composite grid.
    *--------------------------------------------------------------------------*/
   grid_level = nalu_hypre_TAlloc(nalu_hypre_SStructGrid *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   for (level = max_level; level >= 0; level--)
   {
      NALU_HYPRE_SStructGridCreate(comm, ndim, nparts_level, &grid_level[level]);
   }

   for (level = max_level; level >= 0; level--)
   {
      /*--------------------------------------------------------------------------
       * Create the fine part of the finest level SStructGrids using the original
       * composite grid.
       *--------------------------------------------------------------------------*/
      if (level == max_level)
      {
         pgrid = nalu_hypre_SStructGridPGrid(grid, levels[level]);
         iboxarray = nalu_hypre_SStructPGridCellIBoxArray(pgrid);
         for (box = 0; box < nalu_hypre_BoxArraySize(iboxarray); box++)
         {
            NALU_HYPRE_SStructGridSetExtents(grid_level[level], part_fine,
                                        nalu_hypre_BoxIMin( nalu_hypre_BoxArrayBox(iboxarray, box) ),
                                        nalu_hypre_BoxIMax( nalu_hypre_BoxArrayBox(iboxarray, box) ));
         }

         NALU_HYPRE_SStructGridSetVariables( grid_level[level], part_fine,
                                        nalu_hypre_SStructPGridNVars(pgrid),
                                        nalu_hypre_SStructPGridVarTypes(pgrid) );

         /*-----------------------------------------------------------------------
          * Create the coarsest level grid if A has only 1 level
          *-----------------------------------------------------------------------*/
         if (level == 0)
         {
            for (box = 0; box < nalu_hypre_BoxArraySize(iboxarray); box++)
            {
               NALU_HYPRE_SStructGridSetExtents(grid_level[level], part_crse,
                                           nalu_hypre_BoxIMin( nalu_hypre_BoxArrayBox(iboxarray, box) ),
                                           nalu_hypre_BoxIMax( nalu_hypre_BoxArrayBox(iboxarray, box) ));
            }

            NALU_HYPRE_SStructGridSetVariables( grid_level[level], part_crse,
                                           nalu_hypre_SStructPGridNVars(pgrid),
                                           nalu_hypre_SStructPGridVarTypes(pgrid) );
         }
      }

      /*--------------------------------------------------------------------------
       * Create the coarse part of level SStructGrids using the original composite
       * grid, the coarsest part SStructGrid, and the fine part if level < max_level.
       *--------------------------------------------------------------------------*/
      if (level > 0)
      {
         pgrid = nalu_hypre_SStructGridPGrid(grid, levels[level - 1]);
         iboxarray = nalu_hypre_SStructPGridCellIBoxArray(pgrid);
         for (box = 0; box < nalu_hypre_BoxArraySize(iboxarray); box++)
         {
            NALU_HYPRE_SStructGridSetExtents(grid_level[level], part_crse,
                                        nalu_hypre_BoxIMin( nalu_hypre_BoxArrayBox(iboxarray, box) ),
                                        nalu_hypre_BoxIMax( nalu_hypre_BoxArrayBox(iboxarray, box) ));

            NALU_HYPRE_SStructGridSetExtents(grid_level[level - 1], part_fine,
                                        nalu_hypre_BoxIMin( nalu_hypre_BoxArrayBox(iboxarray, box) ),
                                        nalu_hypre_BoxIMax( nalu_hypre_BoxArrayBox(iboxarray, box) ));


            if (level == 1)
            {
               NALU_HYPRE_SStructGridSetExtents(grid_level[level - 1], part_crse,
                                           nalu_hypre_BoxIMin( nalu_hypre_BoxArrayBox(iboxarray, box) ),
                                           nalu_hypre_BoxIMax( nalu_hypre_BoxArrayBox(iboxarray, box) ));
            }
         }

         NALU_HYPRE_SStructGridSetVariables( grid_level[level], part_crse,
                                        nalu_hypre_SStructPGridNVars(pgrid),
                                        nalu_hypre_SStructPGridVarTypes(pgrid) );

         NALU_HYPRE_SStructGridSetVariables( grid_level[level - 1], part_fine,
                                        nalu_hypre_SStructPGridNVars(pgrid),
                                        nalu_hypre_SStructPGridVarTypes(pgrid) );

         /* coarsest SStructGrid */
         if (level == 1)
         {
            NALU_HYPRE_SStructGridSetVariables( grid_level[level - 1], part_crse,
                                           nalu_hypre_SStructPGridNVars(pgrid),
                                           nalu_hypre_SStructPGridVarTypes(pgrid) );
         }
      }

      NALU_HYPRE_SStructGridAssemble(grid_level[level]);
   }

   (fac_data -> grid_level) = grid_level;

   /*-----------------------------------------------------------
    * Set up the graph. Create only the structured components
    * first.
    *-----------------------------------------------------------*/
   graph_level = nalu_hypre_TAlloc(nalu_hypre_SStructGraph *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   for (level = max_level; level >= 0; level--)
   {
      NALU_HYPRE_SStructGraphCreate(comm, grid_level[level], &graph_level[level]);
   }

   for (level = max_level; level >= 0; level--)
   {
      /*-----------------------------------------------------------------------
       * Create the fine part of the finest level structured graph connection.
       *-----------------------------------------------------------------------*/
      if (level == max_level)
      {
         pgrid = nalu_hypre_SStructGridPGrid(grid, levels[level]);
         nvars = nalu_hypre_SStructPGridNVars(pgrid);
         for (var1 = 0; var1 < nvars; var1++)
         {
            stencils = nalu_hypre_SStructGraphStencil(graph, levels[level], var1);
            NALU_HYPRE_SStructGraphSetStencil(graph_level[level], part_fine, var1, stencils);

            if (level == 0)
            {
               NALU_HYPRE_SStructGraphSetStencil(graph_level[level], part_crse, var1, stencils);
            }
         }
      }

      /*--------------------------------------------------------------------------
       * Create the coarse part of the graph_level using the graph of A, and the
       * and the fine part if level < max_level.
       *--------------------------------------------------------------------------*/
      if (level > 0)
      {
         pgrid = nalu_hypre_SStructGridPGrid(grid, levels[level - 1]);
         nvars = nalu_hypre_SStructPGridNVars(pgrid);

         for (var1 = 0; var1 < nvars; var1++)
         {
            stencils = nalu_hypre_SStructGraphStencil(graph, levels[level - 1], var1);
            NALU_HYPRE_SStructGraphSetStencil(graph_level[level], part_crse, var1, stencils );
            NALU_HYPRE_SStructGraphSetStencil(graph_level[level - 1], part_fine, var1, stencils );

            if (level == 1)
            {
               NALU_HYPRE_SStructGraphSetStencil(graph_level[level - 1], part_crse, var1, stencils );
            }

         }
      }
   }

   /*-----------------------------------------------------------
    * Extract the non-stencil graph structure: assuming only like
    * variables connect. Also count the number of unstructured
    * connections per part.
    *
    * THE COARSEST COMPOSITE MATRIX DOES NOT HAVE ANY NON-STENCIL
    * CONNECTIONS.
    *-----------------------------------------------------------*/
   Uventries =  nalu_hypre_SStructGraphUVEntries(graph);
   nUventries =  nalu_hypre_SStructGraphNUVEntries(graph);
   iUventries =  nalu_hypre_SStructGraphIUVEntries(graph);

   nrows     =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nUventries; i++)
   {
      Uventry =  Uventries[iUventries[i]];

      part     =  nalu_hypre_SStructUVEntryPart(Uventry);
      nalu_hypre_CopyIndex(nalu_hypre_SStructUVEntryIndex(Uventry), index);
      var      =  nalu_hypre_SStructUVEntryVar(Uventry);
      nUentries =  nalu_hypre_SStructUVEntryNUEntries(Uventry);

      for (k = 0; k < nUentries; k++)
      {
         Uentry  =  nalu_hypre_SStructUVEntryUEntry(Uventry, k);

         to_part =  nalu_hypre_SStructUEntryToPart(Uentry);
         nalu_hypre_CopyIndex(nalu_hypre_SStructUEntryToIndex(Uentry), to_index);
         to_var  =  nalu_hypre_SStructUEntryToVar(Uentry);

         if ( part_to_level[part] >= part_to_level[to_part] )
         {
            level        = part_to_level[part];
            level_part   = part_fine;
            level_topart = part_crse;
         }
         else
         {
            level        = part_to_level[to_part];
            level_part   = part_crse;
            level_topart = part_fine;
         }
         nrows[level]++;

         NALU_HYPRE_SStructGraphAddEntries(graph_level[level], level_part, index,
                                      var, level_topart, to_index, to_var);
      }
   }

   for (level = 0; level <= max_level; level++)
   {
      NALU_HYPRE_SStructGraphAssemble(graph_level[level]);
   }

   (fac_data -> graph_level) = graph_level;

   /*---------------------------------------------------------------
    * Create the level SStruct_Vectors, and temporary global
    * sstuct_vector.
    *---------------------------------------------------------------*/
   b_level = nalu_hypre_TAlloc(nalu_hypre_SStructVector *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   x_level = nalu_hypre_TAlloc(nalu_hypre_SStructVector *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   r_level = nalu_hypre_TAlloc(nalu_hypre_SStructVector *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   e_level = nalu_hypre_TAlloc(nalu_hypre_SStructVector *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);

   tx_level = nalu_hypre_TAlloc(nalu_hypre_SStructPVector *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);

   for (level = 0; level <= max_level; level++)
   {
      NALU_HYPRE_SStructVectorCreate(comm, grid_level[level], &b_level[level]);
      NALU_HYPRE_SStructVectorInitialize(b_level[level]);
      NALU_HYPRE_SStructVectorAssemble(b_level[level]);

      NALU_HYPRE_SStructVectorCreate(comm, grid_level[level], &x_level[level]);
      NALU_HYPRE_SStructVectorInitialize(x_level[level]);
      NALU_HYPRE_SStructVectorAssemble(x_level[level]);

      NALU_HYPRE_SStructVectorCreate(comm, grid_level[level], &r_level[level]);
      NALU_HYPRE_SStructVectorInitialize(r_level[level]);
      NALU_HYPRE_SStructVectorAssemble(r_level[level]);

      NALU_HYPRE_SStructVectorCreate(comm, grid_level[level], &e_level[level]);
      NALU_HYPRE_SStructVectorInitialize(e_level[level]);
      NALU_HYPRE_SStructVectorAssemble(e_level[level]);

      /* temporary vector for fine patch relaxation */
      nalu_hypre_SStructPVectorCreate(comm,
                                 nalu_hypre_SStructGridPGrid(grid_level[level], part_fine),
                                 &tx_level[level]);
      nalu_hypre_SStructPVectorInitialize(tx_level[level]);
      nalu_hypre_SStructPVectorAssemble(tx_level[level]);

   }

   /* temp SStructVectors */
   NALU_HYPRE_SStructVectorCreate(comm, grid, &tx);
   NALU_HYPRE_SStructVectorInitialize(tx);
   NALU_HYPRE_SStructVectorAssemble(tx);

   (fac_data -> b_level) = b_level;
   (fac_data -> x_level) = x_level;
   (fac_data -> r_level) = r_level;
   (fac_data -> e_level) = e_level;
   (fac_data -> tx_level) = tx_level;
   (fac_data -> tx)      = tx;

   /*-----------------------------------------------------------
    * Set up the level composite sstruct_matrices.
    *-----------------------------------------------------------*/

   A_level = nalu_hypre_TAlloc(nalu_hypre_SStructMatrix *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SetIndex3(stride, 1, 1, 1);
   for (level = 0; level <= max_level; level++)
   {
      NALU_HYPRE_SStructMatrixCreate(comm, graph_level[level], &A_level[level]);
      NALU_HYPRE_SStructMatrixInitialize(A_level[level]);

      max_box_volume = 0;
      pgrid = nalu_hypre_SStructGridPGrid(grid, levels[level]);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);

      for (var1 = 0; var1 < nvars; var1++)
      {
         sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var1);
         sgrid_boxes = nalu_hypre_StructGridBoxes(sgrid);

         nalu_hypre_ForBoxI(i, sgrid_boxes)
         {
            sgrid_box = nalu_hypre_BoxArrayBox(sgrid_boxes, i);
            box_volume = nalu_hypre_BoxVolume(sgrid_box);

            max_box_volume = nalu_hypre_max(max_box_volume, box_volume);
         }
      }

      values   = nalu_hypre_TAlloc(NALU_HYPRE_Real,  max_box_volume, NALU_HYPRE_MEMORY_HOST);
      A_pmatrix = nalu_hypre_SStructMatrixPMatrix(A_rap, levels[level]);

      /*-----------------------------------------------------------
       * extract stencil values for all fine levels.
       *-----------------------------------------------------------*/
      for (var1 = 0; var1 < nvars; var1++)
      {
         sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var1);
         sgrid_boxes = nalu_hypre_StructGridBoxes(sgrid);

         stencils = nalu_hypre_SStructGraphStencil(graph, levels[level], var1);
         stencil_size = nalu_hypre_SStructStencilSize(stencils);
         stencil_vars = nalu_hypre_SStructStencilVars(stencils);

         for (i = 0; i < stencil_size; i++)
         {
            var2 = stencil_vars[i];
            A_smatrix = nalu_hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var2);
            nalu_hypre_CopyIndex(nalu_hypre_SStructStencilEntry(stencils, i), stencil_shape_i);

            nalu_hypre_ForBoxI(j, sgrid_boxes)
            {
               sgrid_box =  nalu_hypre_BoxArrayBox(sgrid_boxes, j);
               box_start =  nalu_hypre_BoxIMin(sgrid_box);
               box_end  =  nalu_hypre_BoxIMax(sgrid_box);

               A_smatrix_dbox =  nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A_smatrix), j);
               A_smatrix_value =
                  nalu_hypre_StructMatrixExtractPointerByIndex(A_smatrix, j, stencil_shape_i);

               nalu_hypre_BoxGetSize(sgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(values,A_smatrix_value)
               nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                   sgrid_box, box_start, stride, k,
                                   A_smatrix_dbox, box_start, stride, iA);
               {
                  values[k] = A_smatrix_value[iA];
               }
               nalu_hypre_BoxLoop2End(k, iA);
#undef DEVICE_VAR

               NALU_HYPRE_SStructMatrixSetBoxValues(A_level[level], part_fine, box_start, box_end,
                                               var1, 1, &i, values);
            }   /* nalu_hypre_ForBoxI */
         }      /* for i */
      }         /* for var1 */
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------
       *  Extract the coarse part
       *-----------------------------------------------------------*/
      if (level > 0)
      {
         max_box_volume = 0;
         pgrid = nalu_hypre_SStructGridPGrid(grid, levels[level - 1]);
         nvars = nalu_hypre_SStructPGridNVars(pgrid);

         for (var1 = 0; var1 < nvars; var1++)
         {
            sgrid      = nalu_hypre_SStructPGridSGrid( pgrid, var1 );
            sgrid_boxes = nalu_hypre_StructGridBoxes(sgrid);

            nalu_hypre_ForBoxI( i, sgrid_boxes )
            {
               sgrid_box = nalu_hypre_BoxArrayBox(sgrid_boxes, i);
               box_volume = nalu_hypre_BoxVolume(sgrid_box);

               max_box_volume = nalu_hypre_max(max_box_volume, box_volume );
            }
         }

         values   = nalu_hypre_TAlloc(NALU_HYPRE_Real,  max_box_volume, NALU_HYPRE_MEMORY_HOST);
         A_pmatrix = nalu_hypre_SStructMatrixPMatrix(A_rap, levels[level - 1]);

         /*-----------------------------------------------------------
          * extract stencil values
          *-----------------------------------------------------------*/
         for (var1 = 0; var1 < nvars; var1++)
         {
            sgrid      = nalu_hypre_SStructPGridSGrid(pgrid, var1);
            sgrid_boxes = nalu_hypre_StructGridBoxes(sgrid);

            stencils = nalu_hypre_SStructGraphStencil(graph, levels[level - 1], var1);
            stencil_size = nalu_hypre_SStructStencilSize(stencils);
            stencil_vars = nalu_hypre_SStructStencilVars(stencils);

            for (i = 0; i < stencil_size; i++)
            {
               var2 = stencil_vars[i];
               A_smatrix = nalu_hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var2);
               nalu_hypre_CopyIndex(nalu_hypre_SStructStencilEntry(stencils, i), stencil_shape_i);

               nalu_hypre_ForBoxI( j, sgrid_boxes )
               {
                  sgrid_box =  nalu_hypre_BoxArrayBox(sgrid_boxes, j);
                  box_start =  nalu_hypre_BoxIMin(sgrid_box);
                  box_end  =  nalu_hypre_BoxIMax(sgrid_box);

                  A_smatrix_dbox =  nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A_smatrix), j);
                  A_smatrix_value =
                     nalu_hypre_StructMatrixExtractPointerByIndex(A_smatrix, j, stencil_shape_i);

                  nalu_hypre_BoxGetSize(sgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(values,A_smatrix_value)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      sgrid_box, box_start, stride, k,
                                      A_smatrix_dbox, box_start, stride, iA);
                  {
                     values[k] = A_smatrix_value[iA];
                  }
                  nalu_hypre_BoxLoop2End(k, iA);
#undef DEVICE_VAR

                  NALU_HYPRE_SStructMatrixSetBoxValues(A_level[level], part_crse, box_start, box_end,
                                                  var1, 1, &i, values);
               }  /* nalu_hypre_ForBoxI */
            }     /* for i */
         }        /* for var1 */
         nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
      }            /* if level > 0 */
   }               /* for level */

   /*-----------------------------------------------------------
    * extract the non-stencil values for all but the coarsest
    * level sstruct_matrix. Use the NALU_HYPRE_IJMatrixGetValues
    * for each level of A.
    *-----------------------------------------------------------*/

   Uventries =  nalu_hypre_SStructGraphUVEntries(graph);
   nUventries =  nalu_hypre_SStructGraphNUVEntries(graph);
   iUventries =  nalu_hypre_SStructGraphIUVEntries(graph);

   /*-----------------------------------------------------------
    * Allocate memory for arguments of NALU_HYPRE_IJMatrixGetValues.
    *-----------------------------------------------------------*/
   ncols =  nalu_hypre_TAlloc(NALU_HYPRE_Int *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   rows  =  nalu_hypre_TAlloc(NALU_HYPRE_BigInt *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   cols  =  nalu_hypre_TAlloc(NALU_HYPRE_BigInt *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   cnt   =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_level + 1, NALU_HYPRE_MEMORY_HOST);

   ncols[0] = NULL;
   rows[0] = NULL;
   cols[0] = NULL;
   for (level = 1; level <= max_level; level++)
   {
      ncols[level] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nrows[level], NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < nrows[level]; i++)
      {
         ncols[level][i] = 1;
      }
      rows[level] = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  nrows[level], NALU_HYPRE_MEMORY_HOST);
      cols[level] = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  nrows[level], NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < nUventries; i++)
   {
      Uventry  =  Uventries[iUventries[i]];

      part     =  nalu_hypre_SStructUVEntryPart(Uventry);
      nalu_hypre_CopyIndex(nalu_hypre_SStructUVEntryIndex(Uventry), index);
      var      =  nalu_hypre_SStructUVEntryVar(Uventry);

      nalu_hypre_SStructGridFindBoxManEntry(grid, part, index, var, &boxman_entry);
      nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &row_coord,
                                            matrix_type);

      nUentries =  nalu_hypre_SStructUVEntryNUEntries(Uventry);
      for (k = 0; k < nUentries; k++)
      {
         to_part =  nalu_hypre_SStructUVEntryToPart(Uventry, k);
         to_rank =  nalu_hypre_SStructUVEntryToRank(Uventry, k);

         /*-----------------------------------------------------------
          *  store the row & col indices in the correct level.
          *-----------------------------------------------------------*/
         level   = nalu_hypre_max( part_to_level[part], part_to_level[to_part] );
         rows[level][ cnt[level] ] = row_coord;
         cols[level][ cnt[level]++ ] = to_rank;
      }
   }
   nalu_hypre_TFree(cnt, NALU_HYPRE_MEMORY_HOST);

   for (level = 1; level <= max_level; level++)
   {

      vals      = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  nrows[level], NALU_HYPRE_MEMORY_HOST);
      level_rows = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  nrows[level], NALU_HYPRE_MEMORY_HOST);
      level_cols = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  nrows[level], NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_IJMatrixGetValues(ij_A, nrows[level], ncols[level], rows[level],
                              cols[level], vals);

      Uventries =  nalu_hypre_SStructGraphUVEntries(graph_level[level]);
      /*-----------------------------------------------------------
       * Find the rows & cols of the level ij_matrices where the
       * extracted data must be placed. Note that because the
       * order in which the NALU_HYPRE_SStructGraphAddEntries in the
       * graph_level's is the same order in which rows[level] &
       * cols[level] were formed, the coefficients in val are
       * in the correct order.
       *-----------------------------------------------------------*/

      level_cnt = 0;
      for (i = 0; i < nalu_hypre_SStructGraphNUVEntries(graph_level[level]); i++)
      {
         j      =  nalu_hypre_SStructGraphIUVEntry(graph_level[level], i);
         Uventry =  Uventries[j];

         part     =  nalu_hypre_SStructUVEntryPart(Uventry);
         nalu_hypre_CopyIndex(nalu_hypre_SStructUVEntryIndex(Uventry), index);
         var      =  nalu_hypre_SStructUVEntryVar(Uventry);

         nalu_hypre_SStructGridFindBoxManEntry(grid_level[level], part, index, var, &boxman_entry);
         nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &row_coord, matrix_type);

         nUentries =  nalu_hypre_SStructUVEntryNUEntries(Uventry);
         for (k = 0; k < nUentries; k++)
         {
            to_rank =  nalu_hypre_SStructUVEntryToRank(Uventry, k);

            level_rows[level_cnt]  = row_coord;
            level_cols[level_cnt++] = to_rank;
         }
      }

      /*-----------------------------------------------------------
       * Place the extracted ij coefficients into the level ij
       * matrices.
       *-----------------------------------------------------------*/
      NALU_HYPRE_IJMatrixSetValues( nalu_hypre_SStructMatrixIJMatrix(A_level[level]),
                               nrows[level], ncols[level], (const NALU_HYPRE_BigInt *) level_rows,
                               (const NALU_HYPRE_BigInt *) level_cols, (const NALU_HYPRE_Real *) vals );

      nalu_hypre_TFree(ncols[level], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(rows[level], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cols[level], NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(vals, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(level_rows, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(level_cols, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(ncols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(rows, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nrows, NALU_HYPRE_MEMORY_HOST);

   /*---------------------------------------------------------------
    * Construct the fine grid (part 1) SStruct_PMatrix for all
    * levels except for max_level. This involves coarsening the
    * finer level SStruct_Matrix. Coarsening involves interpolation,
    * matvec, and restriction (to obtain the "row-sum").
    *---------------------------------------------------------------*/
   matvec_data_level  = nalu_hypre_TAlloc(void *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   pmatvec_data_level = nalu_hypre_TAlloc(void *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   interp_data_level  = nalu_hypre_TAlloc(void *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   restrict_data_level = nalu_hypre_TAlloc(void *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);
   for (level = 0; level <= max_level; level++)
   {
      if (level < max_level)
      {
         nalu_hypre_FacSemiInterpCreate2(&interp_data_level[level]);
         nalu_hypre_FacSemiInterpSetup2(interp_data_level[level],
                                   x_level[level + 1],
                                   nalu_hypre_SStructVectorPVector(x_level[level], part_fine),
                                   refine_factors[level + 1]);
      }
      else
      {
         interp_data_level[level] = NULL;
      }

      if (level > 0)
      {
         nalu_hypre_FacSemiRestrictCreate2(&restrict_data_level[level]);

         nalu_hypre_FacSemiRestrictSetup2(restrict_data_level[level],
                                     x_level[level], part_crse, part_fine,
                                     nalu_hypre_SStructVectorPVector(x_level[level - 1], part_fine),
                                     refine_factors[level]);
      }
      else
      {
         restrict_data_level[level] = NULL;
      }
   }

   for (level = max_level; level > 0; level--)
   {

      /*  nalu_hypre_FacZeroCFSten(nalu_hypre_SStructMatrixPMatrix(A_level[level], part_fine),
          nalu_hypre_SStructMatrixPMatrix(A_level[level], part_crse),
          grid_level[level],
          part_fine,
          refine_factors[level]);
          nalu_hypre_FacZeroFCSten(nalu_hypre_SStructMatrixPMatrix(A_level[level], part_fine),
          grid_level[level],
          part_fine);
      */

      nalu_hypre_ZeroAMRMatrixData(A_level[level], part_crse, refine_factors[level]);


      NALU_HYPRE_SStructMatrixAssemble(A_level[level]);
      /*------------------------------------------------------------
       * create data structures that are needed for coarsening
       -------------------------------------------------------------*/
      nalu_hypre_SStructMatvecCreate(&matvec_data_level[level]);
      nalu_hypre_SStructMatvecSetup(matvec_data_level[level],
                               A_level[level],
                               x_level[level]);

      nalu_hypre_SStructPMatvecCreate(&pmatvec_data_level[level]);
      nalu_hypre_SStructPMatvecSetup(pmatvec_data_level[level],
                                nalu_hypre_SStructMatrixPMatrix(A_level[level], part_fine),
                                nalu_hypre_SStructVectorPVector(x_level[level], part_fine));
   }

   /*---------------------------------------------------------------
    * To avoid memory leaks, we cannot reference the coarsest level
    * SStructPMatrix. We need only copy the stuctured coefs.
    *---------------------------------------------------------------*/
   pgrid = nalu_hypre_SStructGridPGrid(grid_level[0], part_fine);
   nvars = nalu_hypre_SStructPGridNVars(pgrid);
   A_pmatrix = nalu_hypre_SStructMatrixPMatrix(A_level[0], part_fine);
   for (var1 = 0; var1 < nvars; var1++)
   {
      sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var1);
      sgrid_boxes = nalu_hypre_StructGridBoxes(sgrid);

      max_box_volume = 0;
      nalu_hypre_ForBoxI(i, sgrid_boxes)
      {
         sgrid_box = nalu_hypre_BoxArrayBox(sgrid_boxes, i);
         box_volume = nalu_hypre_BoxVolume(sgrid_box);

         max_box_volume = nalu_hypre_max(max_box_volume, box_volume);
      }

      values   = nalu_hypre_TAlloc(NALU_HYPRE_Real,  max_box_volume, NALU_HYPRE_MEMORY_HOST);

      stencils = nalu_hypre_SStructGraphStencil(graph_level[0], part_fine, var1);
      stencil_size = nalu_hypre_SStructStencilSize(stencils);
      stencil_vars = nalu_hypre_SStructStencilVars(stencils);

      for (i = 0; i < stencil_size; i++)
      {
         var2 = stencil_vars[i];
         A_smatrix = nalu_hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var2);
         nalu_hypre_CopyIndex(nalu_hypre_SStructStencilEntry(stencils, i), stencil_shape_i);
         nalu_hypre_ForBoxI(j, sgrid_boxes)
         {
            sgrid_box =  nalu_hypre_BoxArrayBox(sgrid_boxes, j);
            box_start =  nalu_hypre_BoxIMin(sgrid_box);
            box_end  =  nalu_hypre_BoxIMax(sgrid_box);

            A_smatrix_dbox =  nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A_smatrix), j);
            A_smatrix_value =
               nalu_hypre_StructMatrixExtractPointerByIndex(A_smatrix, j, stencil_shape_i);

            nalu_hypre_BoxGetSize(sgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(values,A_smatrix_value)
            nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                sgrid_box, box_start, stride, k,
                                A_smatrix_dbox, box_start, stride, iA);
            {
               values[k] = A_smatrix_value[iA];
            }
            nalu_hypre_BoxLoop2End(k, iA);
#undef DEVICE_VAR

            NALU_HYPRE_SStructMatrixSetBoxValues(A_level[0], part_crse, box_start, box_end,
                                            var1, 1, &i, values);
         }   /* nalu_hypre_ForBoxI */
      }      /* for i */

      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
   }         /* for var1 */

   NALU_HYPRE_SStructMatrixAssemble(A_level[0]);

   nalu_hypre_SStructMatvecCreate(&matvec_data_level[0]);
   nalu_hypre_SStructMatvecSetup(matvec_data_level[0],
                            A_level[0],
                            x_level[0]);

   nalu_hypre_SStructPMatvecCreate(&pmatvec_data_level[0]);
   nalu_hypre_SStructPMatvecSetup(pmatvec_data_level[0],
                             nalu_hypre_SStructMatrixPMatrix(A_level[0], part_fine),
                             nalu_hypre_SStructVectorPVector(x_level[0], part_fine));

   nalu_hypre_SStructMatvecCreate(&matvec_data);
   nalu_hypre_SStructMatvecSetup(matvec_data, A_rap, x);

   /*NALU_HYPRE_SStructVectorPrint("sstruct.out.b_l", b_level[max_level], 0);*/
   /*NALU_HYPRE_SStructMatrixPrint("sstruct.out.A_l",  A_level[max_level-2], 0);*/
   (fac_data -> A_level)             = A_level;
   (fac_data -> matvec_data_level)   = matvec_data_level;
   (fac_data -> pmatvec_data_level)  = pmatvec_data_level;
   (fac_data -> matvec_data)         = matvec_data;
   (fac_data -> interp_data_level)   = interp_data_level;
   (fac_data -> restrict_data_level) = restrict_data_level;

   /*---------------------------------------------------------------
    * Create the fine patch relax_data structure.
    *---------------------------------------------------------------*/
   relax_data_level   = nalu_hypre_TAlloc(void *,  max_level + 1, NALU_HYPRE_MEMORY_HOST);

   for (level = 0; level <= max_level; level++)
   {
      relax_data_level[level] =  nalu_hypre_SysPFMGRelaxCreate(comm);
      nalu_hypre_SysPFMGRelaxSetTol(relax_data_level[level], 0.0);
      nalu_hypre_SysPFMGRelaxSetType(relax_data_level[level], relax_type);
      if (usr_jacobi_weight)
      {
         nalu_hypre_SysPFMGRelaxSetJacobiWeight(relax_data_level[level], jacobi_weight);
      }
      nalu_hypre_SysPFMGRelaxSetTempVec(relax_data_level[level], tx_level[level]);
      nalu_hypre_SysPFMGRelaxSetup(relax_data_level[level],
                              nalu_hypre_SStructMatrixPMatrix(A_level[level], part_fine),
                              nalu_hypre_SStructVectorPVector(b_level[level], part_fine),
                              nalu_hypre_SStructVectorPVector(x_level[level], part_fine));
   }
   (fac_data -> relax_data_level)    = relax_data_level;


   /*---------------------------------------------------------------
    * Create the coarsest composite level preconditioned solver.
    *  csolver_type=   1      multigrid-pcg
    *  csolver_type=   2      multigrid
    *---------------------------------------------------------------*/
   if (csolver_type == 1)
   {
      NALU_HYPRE_SStructPCGCreate(comm, &crse_solver);
      NALU_HYPRE_PCGSetMaxIter((NALU_HYPRE_Solver) crse_solver, 1);
      NALU_HYPRE_PCGSetTol((NALU_HYPRE_Solver) crse_solver, 1.0e-6);
      NALU_HYPRE_PCGSetTwoNorm((NALU_HYPRE_Solver) crse_solver, 1);

      /* use SysPFMG solver as preconditioner */
      NALU_HYPRE_SStructSysPFMGCreate(comm, &crse_precond);
      NALU_HYPRE_SStructSysPFMGSetMaxIter(crse_precond, 1);
      NALU_HYPRE_SStructSysPFMGSetTol(crse_precond, 0.0);
      NALU_HYPRE_SStructSysPFMGSetZeroGuess(crse_precond);
      /* weighted Jacobi = 1; red-black GS = 2 */
      NALU_HYPRE_SStructSysPFMGSetRelaxType(crse_precond, 3);
      if (usr_jacobi_weight)
      {
         NALU_HYPRE_SStructFACSetJacobiWeight(crse_precond, jacobi_weight);
      }
      NALU_HYPRE_SStructSysPFMGSetNumPreRelax(crse_precond, 1);
      NALU_HYPRE_SStructSysPFMGSetNumPostRelax(crse_precond, 1);
      NALU_HYPRE_PCGSetPrecond((NALU_HYPRE_Solver) crse_solver,
                          (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSolve,
                          (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSetup,
                          (NALU_HYPRE_Solver) crse_precond);

      NALU_HYPRE_PCGSetup((NALU_HYPRE_Solver) crse_solver,
                     (NALU_HYPRE_Matrix) A_level[0],
                     (NALU_HYPRE_Vector) b_level[0],
                     (NALU_HYPRE_Vector) x_level[0]);
   }

   else if (csolver_type == 2)
   {
      crse_precond = NULL;

      NALU_HYPRE_SStructSysPFMGCreate(comm, &crse_solver);
      NALU_HYPRE_SStructSysPFMGSetMaxIter(crse_solver, 1);
      NALU_HYPRE_SStructSysPFMGSetTol(crse_solver, 1.0e-6);
      NALU_HYPRE_SStructSysPFMGSetZeroGuess(crse_solver);
      /* weighted Jacobi = 1; red-black GS = 2 */
      NALU_HYPRE_SStructSysPFMGSetRelaxType(crse_solver, relax_type);
      if (usr_jacobi_weight)
      {
         NALU_HYPRE_SStructFACSetJacobiWeight(crse_precond, jacobi_weight);
      }
      NALU_HYPRE_SStructSysPFMGSetNumPreRelax(crse_solver, 1);
      NALU_HYPRE_SStructSysPFMGSetNumPostRelax(crse_solver, 1);
      NALU_HYPRE_SStructSysPFMGSetup(crse_solver, A_level[0], b_level[0], x_level[0]);
   }

   (fac_data -> csolver)  = crse_solver;
   (fac_data -> cprecond) = crse_precond;

   nalu_hypre_FacZeroCData(fac_vdata, A_rap);

   return ierr;
}

