/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "smg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetup( void               *smg_vdata,
                nalu_hypre_StructMatrix *A,
                nalu_hypre_StructVector *b,
                nalu_hypre_StructVector *x )
{
   nalu_hypre_SMGData        *smg_data = (nalu_hypre_SMGData *) smg_vdata;

   MPI_Comm              comm = (smg_data -> comm);
   nalu_hypre_IndexRef        base_index  = (smg_data -> base_index);
   nalu_hypre_IndexRef        base_stride = (smg_data -> base_stride);

   NALU_HYPRE_Int             n_pre   = (smg_data -> num_pre_relax);
   NALU_HYPRE_Int             n_post  = (smg_data -> num_post_relax);

   NALU_HYPRE_Int             max_iter;
   NALU_HYPRE_Int             max_levels;

   NALU_HYPRE_Int             num_levels;

   NALU_HYPRE_Int             cdir;

   nalu_hypre_Index           bindex;
   nalu_hypre_Index           bstride;
   nalu_hypre_Index           cindex;
   nalu_hypre_Index           findex;
   nalu_hypre_Index           stride;

   nalu_hypre_StructGrid    **grid_l;
   nalu_hypre_StructGrid    **PT_grid_l;

   NALU_HYPRE_Real           *data;
   NALU_HYPRE_Real           *data_const;
   NALU_HYPRE_Int             data_size = 0;
   NALU_HYPRE_Int             data_size_const = 0;

   nalu_hypre_StructMatrix  **A_l;
   nalu_hypre_StructMatrix  **PT_l;
   nalu_hypre_StructMatrix  **R_l;
   nalu_hypre_StructVector  **b_l;
   nalu_hypre_StructVector  **x_l;

   /* temp vectors */
   nalu_hypre_StructVector  **tb_l;
   nalu_hypre_StructVector  **tx_l;
   nalu_hypre_StructVector  **r_l;
   nalu_hypre_StructVector  **e_l;
   NALU_HYPRE_Real           *b_data;
   NALU_HYPRE_Real           *x_data;
   NALU_HYPRE_Int             b_data_alloced;
   NALU_HYPRE_Int             x_data_alloced;

   void                **relax_data_l;
   void                **residual_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   nalu_hypre_StructGrid     *grid;
   nalu_hypre_Box            *cbox;
   NALU_HYPRE_Int             i, l;

   NALU_HYPRE_Int             b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   NALU_HYPRE_Int             x_num_ghost[]  = {0, 0, 0, 0, 0, 0};

#if DEBUG
   char                  filename[255];
#endif

   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_StructMatrixMemoryLocation(A);

   /*-----------------------------------------------------
    * Set up coarsening direction
    *-----------------------------------------------------*/

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   cdir = nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A)) - 1;
   (smg_data -> cdir) = cdir;

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid = nalu_hypre_StructMatrixGrid(A);

   /* Compute a new max_levels value based on the grid */
   cbox = nalu_hypre_BoxDuplicate(nalu_hypre_StructGridBoundingBox(grid));
   max_levels = nalu_hypre_Log2(nalu_hypre_BoxSizeD(cbox, cdir)) + 2;
   if ((smg_data -> max_levels) > 0)
   {
      max_levels = nalu_hypre_min(max_levels, (smg_data -> max_levels));
   }
   (smg_data -> max_levels) = max_levels;

   grid_l = nalu_hypre_TAlloc(nalu_hypre_StructGrid *,  max_levels, NALU_HYPRE_MEMORY_HOST);
   PT_grid_l = nalu_hypre_TAlloc(nalu_hypre_StructGrid *,  max_levels, NALU_HYPRE_MEMORY_HOST);
   PT_grid_l[0] = NULL;
   nalu_hypre_StructGridRef(grid, &grid_l[0]);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   data_location = nalu_hypre_StructGridDataLocation(grid);
   if (data_location != NALU_HYPRE_MEMORY_HOST)
   {
      num_level_GPU = max_levels;
   }
   else
   {
      num_level_GPU = 0;
      device_level  = 0;
   }
   if (nalu_hypre_StructGridNDim(grid) != nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A)))
   {
      device_level = num_level_GPU;
   }
#endif
   for (l = 0; ; l++)
   {
      /* set cindex and stride */
      nalu_hypre_SMGSetCIndex(base_index, base_stride, l, cdir, cindex);
      nalu_hypre_SMGSetStride(base_index, base_stride, l, cdir, stride);

      /* check to see if we should coarsen */
      if ( ( nalu_hypre_BoxIMinD(cbox, cdir) == nalu_hypre_BoxIMaxD(cbox, cdir) ) ||
           (l == (max_levels - 1)) )
      {
         /* stop coarsening */
         break;
      }

      /* coarsen cbox */
      nalu_hypre_ProjectBox(cbox, cindex, stride);
      nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(cbox), cindex, stride,
                                  nalu_hypre_BoxIMin(cbox));
      nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(cbox), cindex, stride,
                                  nalu_hypre_BoxIMax(cbox));

      /* build the interpolation grid */
      nalu_hypre_StructCoarsen(grid_l[l], cindex, stride, 0, &PT_grid_l[l + 1]);

      /* build the coarse grid */
      nalu_hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l + 1]);
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      nalu_hypre_StructGridDataLocation(PT_grid_l[l + 1]) = data_location;
      if (device_level == -1 && num_level_GPU > 0)
      {
         max_box_size = nalu_hypre_StructGridGetMaxBoxSize(grid_l[l + 1]);
         if (max_box_size < NALU_HYPRE_MIN_GPU_SIZE)
         {
            num_level_GPU = l + 1;
            data_location = NALU_HYPRE_MEMORY_HOST;
            device_level  = num_level_GPU;
            //printf("num_level_GPU = %d,device_level = %d\n",num_level_GPU,device_level);
         }
      }
      else if (l + 1 == device_level)
      {
         num_level_GPU = l + 1;
         data_location = NALU_HYPRE_MEMORY_HOST;
      }

      nalu_hypre_StructGridDataLocation(grid_l[l + 1]) = data_location;
#endif
   }
   num_levels = l + 1;

   /* free up some things */
   nalu_hypre_BoxDestroy(cbox);

   (smg_data -> num_levels) = num_levels;
   (smg_data -> grid_l)     = grid_l;
   (smg_data -> PT_grid_l)  = PT_grid_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   PT_l = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_levels - 1, NALU_HYPRE_MEMORY_HOST);
   R_l  = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_levels - 1, NALU_HYPRE_MEMORY_HOST);
   b_l  = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   x_l  = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   tb_l = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   tx_l = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   r_l  = tx_l;
   e_l  = tx_l;

   A_l[0] = nalu_hypre_StructMatrixRef(A);
   b_l[0] = nalu_hypre_StructVectorRef(b);
   x_l[0] = nalu_hypre_StructVectorRef(x);

   for (i = 0; i <= cdir; i++)
   {
      x_num_ghost[2 * i]     = 1;
      x_num_ghost[2 * i + 1] = 1;
   }

   tb_l[0] = nalu_hypre_StructVectorCreate(comm, grid_l[0]);
   nalu_hypre_StructVectorSetNumGhost(tb_l[0], nalu_hypre_StructVectorNumGhost(b));
   nalu_hypre_StructVectorInitializeShell(tb_l[0]);
   nalu_hypre_StructVectorSetDataSize(tb_l[0], &data_size, &data_size_const);

   tx_l[0] = nalu_hypre_StructVectorCreate(comm, grid_l[0]);
   nalu_hypre_StructVectorSetNumGhost(tx_l[0], nalu_hypre_StructVectorNumGhost(x));
   nalu_hypre_StructVectorInitializeShell(tx_l[0]);
   nalu_hypre_StructVectorSetDataSize(tx_l[0], &data_size, &data_size_const);

   for (l = 0; l < (num_levels - 1); l++)
   {
      PT_l[l]  = nalu_hypre_SMGCreateInterpOp(A_l[l], PT_grid_l[l + 1], cdir);

      nalu_hypre_StructMatrixInitializeShell(PT_l[l]);
      data_size += nalu_hypre_StructMatrixDataSize(PT_l[l]);
      data_size_const += nalu_hypre_StructMatrixDataConstSize(PT_l[l]);

      if (nalu_hypre_StructMatrixSymmetric(A))
      {
         R_l[l] = PT_l[l];
      }
      else
      {
         R_l[l] = PT_l[l];
#if 0
         /* Allow R != PT for non symmetric case */
         /* NOTE: Need to create a non-pruned grid for this to work */
         R_l[l]   = nalu_hypre_SMGCreateRestrictOp(A_l[l], grid_l[l + 1], cdir);
         nalu_hypre_StructMatrixInitializeShell(R_l[l]);
         data_size += nalu_hypre_StructMatrixDataSize(R_l[l]);
         data_size_const += nalu_hypre_StructMatrixDataConstSize(R_l[l]);
#endif
      }

      A_l[l + 1] = nalu_hypre_SMGCreateRAPOp(R_l[l], A_l[l], PT_l[l], grid_l[l + 1]);
      nalu_hypre_StructMatrixInitializeShell(A_l[l + 1]);
      data_size += nalu_hypre_StructMatrixDataSize(A_l[l + 1]);
      data_size_const += nalu_hypre_StructMatrixDataConstSize(A_l[l + 1]);

      b_l[l + 1] = nalu_hypre_StructVectorCreate(comm, grid_l[l + 1]);
      nalu_hypre_StructVectorSetNumGhost(b_l[l + 1], b_num_ghost);
      nalu_hypre_StructVectorInitializeShell(b_l[l + 1]);
      nalu_hypre_StructVectorSetDataSize(b_l[l + 1], &data_size, &data_size_const);

      x_l[l + 1] = nalu_hypre_StructVectorCreate(comm, grid_l[l + 1]);
      nalu_hypre_StructVectorSetNumGhost(x_l[l + 1], x_num_ghost);
      nalu_hypre_StructVectorInitializeShell(x_l[l + 1]);
      nalu_hypre_StructVectorSetDataSize(x_l[l + 1], &data_size, &data_size_const);

      tb_l[l + 1] = nalu_hypre_StructVectorCreate(comm, grid_l[l + 1]);
      nalu_hypre_StructVectorSetNumGhost(tb_l[l + 1], nalu_hypre_StructVectorNumGhost(b));
      nalu_hypre_StructVectorInitializeShell(tb_l[l + 1]);

      tx_l[l + 1] = nalu_hypre_StructVectorCreate(comm, grid_l[l + 1]);
      nalu_hypre_StructVectorSetNumGhost(tx_l[l + 1], nalu_hypre_StructVectorNumGhost(x));
      nalu_hypre_StructVectorInitializeShell(tx_l[l + 1]);
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (l + 1 == num_level_GPU)
      {
         nalu_hypre_StructVectorSetDataSize(tb_l[l + 1], &data_size, &data_size_const);
         nalu_hypre_StructVectorSetDataSize(tx_l[l + 1], &data_size, &data_size_const);
      }
#endif
   }

   data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, data_size, memory_location);
   data_const = nalu_hypre_CTAlloc(NALU_HYPRE_Real, data_size_const, NALU_HYPRE_MEMORY_HOST);

   (smg_data -> memory_location) = memory_location;
   (smg_data -> data) = data;
   (smg_data -> data_const) = data_const;

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   //if (nalu_hypre_StructGridNDim(grid) == nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A)))
   //   printf("num_level_GPU = %d,device_level = %d / %d\n",num_level_GPU,device_level,num_levels);
   data_location = nalu_hypre_StructGridDataLocation(grid_l[0]);
   if (data_location != NALU_HYPRE_MEMORY_HOST)
   {
      nalu_hypre_StructVectorInitializeData(tb_l[0], data);
      nalu_hypre_StructVectorAssemble(tb_l[0]);
      data += nalu_hypre_StructVectorDataSize(tb_l[0]);
      nalu_hypre_StructVectorInitializeData(tx_l[0], data);
      nalu_hypre_StructVectorAssemble(tx_l[0]);
      data += nalu_hypre_StructVectorDataSize(tx_l[0]);
      //printf("smg_setup: Alloc tx_l[0] on GPU\n");
   }
   else
   {
      nalu_hypre_StructVectorInitializeData(tb_l[0], data_const);
      nalu_hypre_StructVectorAssemble(tb_l[0]);
      data_const += nalu_hypre_StructVectorDataSize(tb_l[0]);
      nalu_hypre_StructVectorInitializeData(tx_l[0], data_const);
      nalu_hypre_StructVectorAssemble(tx_l[0]);
      data_const += nalu_hypre_StructVectorDataSize(tx_l[0]);
      //printf("smg_setup: Alloc tx_l[0] on CPU\n");
   }
#else
   nalu_hypre_StructVectorInitializeData(tb_l[0], data);
   nalu_hypre_StructVectorAssemble(tb_l[0]);
   data += nalu_hypre_StructVectorDataSize(tb_l[0]);

   nalu_hypre_StructVectorInitializeData(tx_l[0], data);
   nalu_hypre_StructVectorAssemble(tx_l[0]);
   data += nalu_hypre_StructVectorDataSize(tx_l[0]);
#endif
   for (l = 0; l < (num_levels - 1); l++)
   {
      nalu_hypre_StructMatrixInitializeData(PT_l[l], data, data_const);
      data += nalu_hypre_StructMatrixDataSize(PT_l[l]);
      data_const += nalu_hypre_StructMatrixDataConstSize(PT_l[l]);

#if 0
      /* Allow R != PT for non symmetric case */
      if (!nalu_hypre_StructMatrixSymmetric(A))
      {
         nalu_hypre_StructMatrixInitializeData(R_l[l], data, data_const);
         data += nalu_hypre_StructMatrixDataSize(R_l[l]);
         data_const += nalu_hypre_StructMatrixDataConstSize(R_l[l]);
      }
#endif

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (l + 1 == num_level_GPU)
      {
         data_location = NALU_HYPRE_MEMORY_HOST;
      }
#endif

      nalu_hypre_StructMatrixInitializeData(A_l[l + 1], data, data_const);
      data += nalu_hypre_StructMatrixDataSize(A_l[l + 1]);
      data_const += nalu_hypre_StructMatrixDataConstSize(A_l[l + 1]);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (data_location != NALU_HYPRE_MEMORY_HOST)
      {
         nalu_hypre_StructVectorInitializeData(b_l[l + 1], data);
         nalu_hypre_StructVectorAssemble(b_l[l + 1]);
         data += nalu_hypre_StructVectorDataSize(b_l[l + 1]);

         nalu_hypre_StructVectorInitializeData(x_l[l + 1], data);
         nalu_hypre_StructVectorAssemble(x_l[l + 1]);
         data += nalu_hypre_StructVectorDataSize(x_l[l + 1]);
         nalu_hypre_StructVectorInitializeData(tb_l[l + 1],
                                          nalu_hypre_StructVectorData(tb_l[0]));
         nalu_hypre_StructVectorAssemble(tb_l[l + 1]);

         nalu_hypre_StructVectorInitializeData(tx_l[l + 1],
                                          nalu_hypre_StructVectorData(tx_l[0]));
         nalu_hypre_StructVectorAssemble(tx_l[l + 1]);
         //printf("\n Alloc x_l,b_l[%d] on GPU\n",l+1);
      }
      else
      {
         nalu_hypre_StructVectorInitializeData(b_l[l + 1], data_const);
         nalu_hypre_StructVectorAssemble(b_l[l + 1]);
         data_const += nalu_hypre_StructVectorDataSize(b_l[l + 1]);

         nalu_hypre_StructVectorInitializeData(x_l[l + 1], data_const);
         nalu_hypre_StructVectorAssemble(x_l[l + 1]);
         data_const += nalu_hypre_StructVectorDataSize(x_l[l + 1]);
         if (l + 1 == num_level_GPU)
         {
            nalu_hypre_StructVectorInitializeData(tb_l[l + 1], data_const);
            nalu_hypre_StructVectorAssemble(tb_l[l + 1]);
            data_const += nalu_hypre_StructVectorDataSize(tb_l[l + 1]);
            nalu_hypre_StructVectorInitializeData(tx_l[l + 1], data_const);
            nalu_hypre_StructVectorAssemble(tx_l[l + 1]);
            data_const += nalu_hypre_StructVectorDataSize(tx_l[l + 1]);
         }
         else
         {
            nalu_hypre_StructVectorInitializeData(tb_l[l + 1],
                                             nalu_hypre_StructVectorData(tb_l[num_level_GPU]));
            nalu_hypre_StructVectorAssemble(tb_l[l + 1]);

            nalu_hypre_StructVectorInitializeData(tx_l[l + 1],
                                             nalu_hypre_StructVectorData(tx_l[num_level_GPU]));
            nalu_hypre_StructVectorAssemble(tx_l[l + 1]);
         }
         //printf("\n Alloc x_l,b_l[%d] on CPU\n",l+1);
      }
#else

      nalu_hypre_StructVectorInitializeData(b_l[l + 1], data);
      nalu_hypre_StructVectorAssemble(b_l[l + 1]);
      data += nalu_hypre_StructVectorDataSize(b_l[l + 1]);

      nalu_hypre_StructVectorInitializeData(x_l[l + 1], data);
      nalu_hypre_StructVectorAssemble(x_l[l + 1]);
      data += nalu_hypre_StructVectorDataSize(x_l[l + 1]);

      nalu_hypre_StructVectorInitializeData(tb_l[l + 1],
                                       nalu_hypre_StructVectorData(tb_l[0]));
      nalu_hypre_StructVectorAssemble(tb_l[l + 1]);

      nalu_hypre_StructVectorInitializeData(tx_l[l + 1],
                                       nalu_hypre_StructVectorData(tx_l[0]));
      nalu_hypre_StructVectorAssemble(tx_l[l + 1]);
#endif
   }

   (smg_data -> A_l)  = A_l;
   (smg_data -> PT_l) = PT_l;
   (smg_data -> R_l)  = R_l;
   (smg_data -> b_l)  = b_l;
   (smg_data -> x_l)  = x_l;
   (smg_data -> tb_l) = tb_l;
   (smg_data -> tx_l) = tx_l;
   (smg_data -> r_l)  = r_l;
   (smg_data -> e_l)  = e_l;

   /*-----------------------------------------------------
    * Set up multigrid operators and call setup routines
    *
    * Note: The routine that sets up interpolation uses
    * the same relaxation routines used in the solve
    * phase of the algorithm.  To do this, the data for
    * the fine-grid unknown and right-hand-side vectors
    * is temporarily changed to temporary data.
    *-----------------------------------------------------*/

   relax_data_l    = nalu_hypre_TAlloc(void *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   residual_data_l = nalu_hypre_TAlloc(void *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   restrict_data_l = nalu_hypre_TAlloc(void *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   interp_data_l   = nalu_hypre_TAlloc(void *,  num_levels, NALU_HYPRE_MEMORY_HOST);

   /* temporarily set the data for x_l[0] and b_l[0] to temp data */
   b_data = nalu_hypre_StructVectorData(b_l[0]);
   b_data_alloced = nalu_hypre_StructVectorDataAlloced(b_l[0]);
   x_data = nalu_hypre_StructVectorData(x_l[0]);
   x_data_alloced = nalu_hypre_StructVectorDataAlloced(x_l[0]);
   nalu_hypre_StructVectorInitializeData(b_l[0], nalu_hypre_StructVectorData(tb_l[0]));
   nalu_hypre_StructVectorInitializeData(x_l[0], nalu_hypre_StructVectorData(tx_l[0]));
   nalu_hypre_StructVectorAssemble(b_l[0]);
   nalu_hypre_StructVectorAssemble(x_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (l == num_level_GPU)
      {
         nalu_hypre_SetDeviceOff();
      }
#endif

      nalu_hypre_SMGSetBIndex(base_index, base_stride, l, bindex);
      nalu_hypre_SMGSetBStride(base_index, base_stride, l, bstride);
      nalu_hypre_SMGSetCIndex(base_index, base_stride, l, cdir, cindex);
      nalu_hypre_SMGSetFIndex(base_index, base_stride, l, cdir, findex);
      nalu_hypre_SMGSetStride(base_index, base_stride, l, cdir, stride);

      /* set up relaxation */
      relax_data_l[l] = nalu_hypre_SMGRelaxCreate(comm);
      nalu_hypre_SMGRelaxSetBase(relax_data_l[l], bindex, bstride);
      nalu_hypre_SMGRelaxSetMemoryUse(relax_data_l[l], (smg_data -> memory_use));
      nalu_hypre_SMGRelaxSetTol(relax_data_l[l], 0.0);
      nalu_hypre_SMGRelaxSetNumSpaces(relax_data_l[l], 2);
      nalu_hypre_SMGRelaxSetSpace(relax_data_l[l], 0,
                             nalu_hypre_IndexD(cindex, cdir),
                             nalu_hypre_IndexD(stride, cdir));
      nalu_hypre_SMGRelaxSetSpace(relax_data_l[l], 1,
                             nalu_hypre_IndexD(findex, cdir),
                             nalu_hypre_IndexD(stride, cdir));
      nalu_hypre_SMGRelaxSetTempVec(relax_data_l[l], tb_l[l]);
      nalu_hypre_SMGRelaxSetNumPreRelax( relax_data_l[l], n_pre);
      nalu_hypre_SMGRelaxSetNumPostRelax( relax_data_l[l], n_post);
      //nalu_hypre_SMGRelaxSetMaxLevel( relax_data_l[l], l+6);
      nalu_hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      nalu_hypre_SMGSetupInterpOp(relax_data_l[l], A_l[l], b_l[l], x_l[l],
                             PT_l[l], cdir, cindex, findex, stride);

      /* (re)set relaxation parameters */
      nalu_hypre_SMGRelaxSetNumPreSpaces(relax_data_l[l], 0);
      nalu_hypre_SMGRelaxSetNumRegSpaces(relax_data_l[l], 2);
      nalu_hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      /* set up the residual routine */
      residual_data_l[l] = nalu_hypre_SMGResidualCreate();
      nalu_hypre_SMGResidualSetBase(residual_data_l[l], bindex, bstride);
      nalu_hypre_SMGResidualSetup(residual_data_l[l],
                             A_l[l], x_l[l], b_l[l], r_l[l]);

      /* set up the interpolation routine */
      interp_data_l[l] = nalu_hypre_SemiInterpCreate();
      nalu_hypre_SemiInterpSetup(interp_data_l[l], PT_l[l], 1, x_l[l + 1], e_l[l],
                            cindex, findex, stride);

      /* set up the restriction operator */
#if 0
      /* Allow R != PT for non symmetric case */
      if (!nalu_hypre_StructMatrixSymmetric(A))
         nalu_hypre_SMGSetupRestrictOp(A_l[l], R_l[l], tx_l[l], cdir,
                                  cindex, stride);
#endif
      /* set up the restriction routine */
      restrict_data_l[l] = nalu_hypre_SemiRestrictCreate();
      nalu_hypre_SemiRestrictSetup(restrict_data_l[l], R_l[l], 0, r_l[l], b_l[l + 1],
                              cindex, findex, stride);

      /* set up the coarse grid operator */
      nalu_hypre_SMGSetupRAPOp(R_l[l], A_l[l], PT_l[l], A_l[l + 1],
                          cindex, stride);
   }

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   if (l == num_level_GPU)
   {
      nalu_hypre_SetDeviceOff();
   }
#endif

   nalu_hypre_SMGSetBIndex(base_index, base_stride, l, bindex);
   nalu_hypre_SMGSetBStride(base_index, base_stride, l, bstride);
   relax_data_l[l] = nalu_hypre_SMGRelaxCreate(comm);
   nalu_hypre_SMGRelaxSetBase(relax_data_l[l], bindex, bstride);
   nalu_hypre_SMGRelaxSetTol(relax_data_l[l], 0.0);
   nalu_hypre_SMGRelaxSetMaxIter(relax_data_l[l], 1);
   nalu_hypre_SMGRelaxSetTempVec(relax_data_l[l], tb_l[l]);
   nalu_hypre_SMGRelaxSetNumPreRelax( relax_data_l[l], n_pre);
   nalu_hypre_SMGRelaxSetNumPostRelax( relax_data_l[l], n_post);
   nalu_hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

   /* set up the residual routine in case of a single grid level */
   if ( l == 0 )
   {
      residual_data_l[l] = nalu_hypre_SMGResidualCreate();
      nalu_hypre_SMGResidualSetBase(residual_data_l[l], bindex, bstride);
      nalu_hypre_SMGResidualSetup(residual_data_l[l],
                             A_l[l], x_l[l], b_l[l], r_l[l]);
   }

   /* set the data for x_l[0] and b_l[0] the way they were */
   nalu_hypre_StructVectorInitializeData(b_l[0], b_data);
   nalu_hypre_StructVectorDataAlloced(b_l[0]) = b_data_alloced;
   nalu_hypre_StructVectorInitializeData(x_l[0], x_data);
   nalu_hypre_StructVectorDataAlloced(x_l[0]) = x_data_alloced;
   nalu_hypre_StructVectorAssemble(b_l[0]);
   nalu_hypre_StructVectorAssemble(x_l[0]);

   (smg_data -> relax_data_l)      = relax_data_l;
   (smg_data -> residual_data_l)   = residual_data_l;
   (smg_data -> restrict_data_l)   = restrict_data_l;
   (smg_data -> interp_data_l)     = interp_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((smg_data -> logging) > 0)
   {
      max_iter = (smg_data -> max_iter);
      (smg_data -> norms)     = nalu_hypre_TAlloc(NALU_HYPRE_Real,  max_iter, NALU_HYPRE_MEMORY_HOST);
      (smg_data -> rel_norms) = nalu_hypre_TAlloc(NALU_HYPRE_Real,  max_iter, NALU_HYPRE_MEMORY_HOST);
   }

#if DEBUG
   if (nalu_hypre_StructGridNDim(grid_l[0]) == 3)
   {
      for (l = 0; l < (num_levels - 1); l++)
      {
         nalu_hypre_sprintf(filename, "zout_A.%02d", l);
         nalu_hypre_StructMatrixPrint(filename, A_l[l], 0);
         nalu_hypre_sprintf(filename, "zout_PT.%02d", l);
         nalu_hypre_StructMatrixPrint(filename, PT_l[l], 0);
      }
      nalu_hypre_sprintf(filename, "zout_A.%02d", l);
      nalu_hypre_StructMatrixPrint(filename, A_l[l], 0);
   }
#endif

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}
