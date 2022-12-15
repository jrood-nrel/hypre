/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "sparse_msg.h"

#define DEBUG 0

#define GRID 0

#define nalu_hypre_SparseMSGSetCIndex(cdir, cindex)  \
   {                                            \
      nalu_hypre_SetIndex3(cindex, 0, 0, 0);         \
      nalu_hypre_IndexD(cindex, cdir) = 0;           \
   }

#define nalu_hypre_SparseMSGSetFIndex(cdir, findex)  \
   {                                            \
      nalu_hypre_SetIndex3(findex, 0, 0, 0);         \
      nalu_hypre_IndexD(findex, cdir) = 1;           \
   }

#define nalu_hypre_SparseMSGSetStride(cdir, stride)  \
   {                                            \
      nalu_hypre_SetIndex3(stride, 1, 1, 1);         \
      nalu_hypre_IndexD(stride, cdir) = 2;           \
   }

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetup( void               *smsg_vdata,
                      nalu_hypre_StructMatrix *A,
                      nalu_hypre_StructVector *b,
                      nalu_hypre_StructVector *x          )
{
   nalu_hypre_SparseMSGData  *smsg_data = (nalu_hypre_SparseMSGData *) smsg_vdata;

   MPI_Comm              comm = (smsg_data -> comm);

   NALU_HYPRE_Int             max_iter;
   NALU_HYPRE_Int             jump       = (smsg_data -> jump);
   NALU_HYPRE_Int             relax_type = (smsg_data -> relax_type);
   NALU_HYPRE_Int             usr_jacobi_weight = (smsg_data -> usr_jacobi_weight);
   NALU_HYPRE_Real            jacobi_weight    = (smsg_data -> jacobi_weight);
   NALU_HYPRE_Int            *num_grids  = (smsg_data -> num_grids);
   NALU_HYPRE_Int             num_all_grids;
   NALU_HYPRE_Int             num_levels;

   nalu_hypre_StructGrid    **grid_a;
   nalu_hypre_StructGrid    **Px_grid_a;
   nalu_hypre_StructGrid    **Py_grid_a;
   nalu_hypre_StructGrid    **Pz_grid_a;

   NALU_HYPRE_Real           *data;
   NALU_HYPRE_Real           *tdata;
   NALU_HYPRE_Int             data_size = 0;
   nalu_hypre_StructMatrix  **A_a;
   nalu_hypre_StructMatrix  **Px_a;
   nalu_hypre_StructMatrix  **Py_a;
   nalu_hypre_StructMatrix  **Pz_a;
   nalu_hypre_StructMatrix  **RTx_a;
   nalu_hypre_StructMatrix  **RTy_a;
   nalu_hypre_StructMatrix  **RTz_a;
   nalu_hypre_StructVector  **b_a;
   nalu_hypre_StructVector  **x_a;

   /* temp vectors */
   nalu_hypre_StructVector  **t_a;
   nalu_hypre_StructVector  **r_a;
   nalu_hypre_StructVector  **e_a;

   nalu_hypre_StructVector  **visitx_a;
   nalu_hypre_StructVector  **visity_a;
   nalu_hypre_StructVector  **visitz_a;
   NALU_HYPRE_Int            *grid_on;

   void                **relax_a;
   void                **matvec_a;
   void                **restrictx_a;
   void                **restricty_a;
   void                **restrictz_a;
   void                **interpx_a;
   void                **interpy_a;
   void                **interpz_a;

   nalu_hypre_Index           cindex;
   nalu_hypre_Index           findex;
   nalu_hypre_Index           stride;
   nalu_hypre_Index           stridePR;

   nalu_hypre_StructGrid     *grid;
   NALU_HYPRE_Int             dim;
   nalu_hypre_Box            *cbox;

   NALU_HYPRE_Int             d, l, lx, ly, lz;
   NALU_HYPRE_Int             fi, ci;

   NALU_HYPRE_Int             b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   NALU_HYPRE_Int             x_num_ghost[]  = {1, 1, 1, 1, 1, 1};

   NALU_HYPRE_Int             ierr = 0;
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_StructMatrixMemoryLocation(A);
#if DEBUG
   char                  filename[255];
#endif


   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid  = nalu_hypre_StructMatrixGrid(A);
   dim   = nalu_hypre_StructGridNDim(grid);

   /* Determine num_grids[] and num_levels */
   num_levels = 1;
   cbox = nalu_hypre_BoxDuplicate(nalu_hypre_StructGridBoundingBox(grid));
   for (d = 0; d < dim; d++)
   {
      while ( nalu_hypre_BoxIMaxD(cbox, d) > nalu_hypre_BoxIMinD(cbox, d) )
      {
         /* set cindex, findex, and stride */
         nalu_hypre_SparseMSGSetCIndex(d, cindex);
         nalu_hypre_SparseMSGSetFIndex(d, findex);
         nalu_hypre_SparseMSGSetStride(d, stride);

         /* coarsen cbox */
         nalu_hypre_ProjectBox(cbox, cindex, stride);
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(cbox),
                                     cindex, stride, nalu_hypre_BoxIMin(cbox));
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(cbox),
                                     cindex, stride, nalu_hypre_BoxIMax(cbox));

         /* increment level counters */
         num_grids[d]++;
         num_levels++;
      }
   }

#if 0
   /* Restrict the semicoarsening to a particular direction */
   num_grids[1] = 1;
   num_grids[2] = 1;
   num_levels = num_grids[0];
#endif

   /* Compute the num_all_grids based on num_grids[] */
   num_all_grids = num_grids[0] * num_grids[1] * num_grids[2];

   /* Store some variables and clean up */
   nalu_hypre_BoxDestroy(cbox);

   (smsg_data -> num_all_grids) = num_all_grids;
   (smsg_data -> num_levels)    = num_levels;

   grid_a = nalu_hypre_TAlloc(nalu_hypre_StructGrid *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructGridRef(grid, &grid_a[0]);
   Px_grid_a = nalu_hypre_TAlloc(nalu_hypre_StructGrid *,  num_grids[0], NALU_HYPRE_MEMORY_HOST);
   Py_grid_a = nalu_hypre_TAlloc(nalu_hypre_StructGrid *,  num_grids[1], NALU_HYPRE_MEMORY_HOST);
   Pz_grid_a = nalu_hypre_TAlloc(nalu_hypre_StructGrid *,  num_grids[2], NALU_HYPRE_MEMORY_HOST);
   Px_grid_a[0] = NULL;
   Py_grid_a[0] = NULL;
   Pz_grid_a[0] = NULL;

   /*-----------------------------------------
    * Compute coarse grids
    *-----------------------------------------*/

   if (num_levels > 1)
   {
      /* coarsen in x direction */
      nalu_hypre_SparseMSGSetCIndex(0, cindex);
      nalu_hypre_SparseMSGSetStride(0, stride);
      for (lx = 0; lx < num_grids[0] - 1; lx++)
      {
         nalu_hypre_SparseMSGMapIndex(lx,   0, 0, num_grids, fi);
         nalu_hypre_SparseMSGMapIndex(lx + 1, 0, 0, num_grids, ci);
         nalu_hypre_StructCoarsen(grid_a[fi], cindex, stride, 1,
                             &grid_a[ci]);
      }

      /* coarsen in y direction */
      nalu_hypre_SparseMSGSetCIndex(1, cindex);
      nalu_hypre_SparseMSGSetStride(1, stride);
      for (ly = 0; ly < num_grids[1] - 1; ly++)
      {
         for (lx = 0; lx < num_grids[0]; lx++)
         {
            nalu_hypre_SparseMSGMapIndex(lx, ly,   0, num_grids, fi);
            nalu_hypre_SparseMSGMapIndex(lx, ly + 1, 0, num_grids, ci);
            nalu_hypre_StructCoarsen(grid_a[fi], cindex, stride, 1,
                                &grid_a[ci]);
         }
      }

      /* coarsen in z direction */
      nalu_hypre_SparseMSGSetCIndex(2, cindex);
      nalu_hypre_SparseMSGSetStride(2, stride);
      for (lz = 0; lz < num_grids[2] - 1; lz++)
      {
         for (ly = 0; ly < num_grids[1]; ly++)
         {
            for (lx = 0; lx < num_grids[0]; lx++)
            {
               nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);
               nalu_hypre_SparseMSGMapIndex(lx, ly, lz + 1, num_grids, ci);
               nalu_hypre_StructCoarsen(grid_a[fi], cindex, stride, 1,
                                   &grid_a[ci]);
            }
         }
      }
   }

   /*-----------------------------------------
    * Compute interpolation grids
    *-----------------------------------------*/

   if (num_levels > 1)
   {
      /* coarsen in x direction */
      nalu_hypre_SparseMSGSetFIndex(0, findex);
      nalu_hypre_SparseMSGSetStride(0, stride);
      for (lx = 0; lx < num_grids[0] - 1; lx++)
      {
         nalu_hypre_SparseMSGMapIndex(lx, 0, 0, num_grids, fi);
         nalu_hypre_StructCoarsen(grid_a[fi], findex, stride, 1,
                             &Px_grid_a[lx + 1]);
      }

      /* coarsen in y direction */
      nalu_hypre_SparseMSGSetFIndex(1, findex);
      nalu_hypre_SparseMSGSetStride(1, stride);
      for (ly = 0; ly < num_grids[1] - 1; ly++)
      {
         nalu_hypre_SparseMSGMapIndex(0, ly, 0, num_grids, fi);
         nalu_hypre_StructCoarsen(grid_a[fi], findex, stride, 1,
                             &Py_grid_a[ly + 1]);
      }

      /* coarsen in z direction */
      nalu_hypre_SparseMSGSetFIndex(2, findex);
      nalu_hypre_SparseMSGSetStride(2, stride);
      for (lz = 0; lz < num_grids[2] - 1; lz++)
      {
         nalu_hypre_SparseMSGMapIndex(0, 0, lz, num_grids, fi);
         nalu_hypre_StructCoarsen(grid_a[fi], findex, stride, 1,
                             &Pz_grid_a[lz + 1]);
      }
   }

   (smsg_data -> grid_array)    = grid_a;
   (smsg_data -> Px_grid_array) = Px_grid_a;
   (smsg_data -> Py_grid_array) = Py_grid_a;
   (smsg_data -> Pz_grid_array) = Pz_grid_a;

   /*------------------------------------------------------
    *  Compute P, R, and A operators
    *  Compute visit arrays and turn grids off if possible
    *
    *  Note: this is ordered to conserve memory
    *-----------------------------------------------------*/

   A_a   = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   Px_a  = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_grids[0] - 1, NALU_HYPRE_MEMORY_HOST);
   Py_a  = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_grids[1] - 1, NALU_HYPRE_MEMORY_HOST);
   Pz_a  = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_grids[2] - 1, NALU_HYPRE_MEMORY_HOST);
   RTx_a = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_grids[0] - 1, NALU_HYPRE_MEMORY_HOST);
   RTy_a = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_grids[1] - 1, NALU_HYPRE_MEMORY_HOST);
   RTz_a = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_grids[2] - 1, NALU_HYPRE_MEMORY_HOST);

   visitx_a = nalu_hypre_CTAlloc(nalu_hypre_StructVector *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   visity_a = nalu_hypre_CTAlloc(nalu_hypre_StructVector *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   visitz_a = nalu_hypre_CTAlloc(nalu_hypre_StructVector *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   grid_on  = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_all_grids, NALU_HYPRE_MEMORY_HOST);


   A_a[0] = nalu_hypre_StructMatrixRef(A);

   for (lz = 0; lz < num_grids[2]; lz++)
   {
      for (ly = 0; ly < num_grids[1]; ly++)
      {
         for (lx = 0; lx < num_grids[0]; lx++)
         {
            nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);

            /*-------------------------------
             * create visit arrays
             *-------------------------------*/

            /* RDF */
#if 0
            l = lx + ly + lz;
            if ((l >= 1) && (l <= jump))
            {
               visitx_a[fi] = visitx_a[0];
               visity_a[fi] = visity_a[0];
               visitz_a[fi] = visitz_a[0];
            }
            else
#endif
               /* RDF */
            {
               visitx_a[fi] = nalu_hypre_StructVectorCreate(comm, grid_a[fi]);
               visity_a[fi] = nalu_hypre_StructVectorCreate(comm, grid_a[fi]);
               visitz_a[fi] = nalu_hypre_StructVectorCreate(comm, grid_a[fi]);
               nalu_hypre_StructVectorSetNumGhost(visitx_a[fi], b_num_ghost);
               nalu_hypre_StructVectorSetNumGhost(visity_a[fi], b_num_ghost);
               nalu_hypre_StructVectorSetNumGhost(visitz_a[fi], b_num_ghost);
               nalu_hypre_StructVectorInitialize(visitx_a[fi]);
               nalu_hypre_StructVectorInitialize(visity_a[fi]);
               nalu_hypre_StructVectorInitialize(visitz_a[fi]);
            }
            nalu_hypre_SparseMSGFilterSetup(A_a[fi], num_grids, lx, ly, lz, jump,
                                       visitx_a[fi],
                                       visity_a[fi],
                                       visitz_a[fi]);
#if GRID
            vx_dot_vx = nalu_hypre_StructInnerProd(visitx_a[fi], visitx_a[fi]);
            vy_dot_vy = nalu_hypre_StructInnerProd(visity_a[fi], visity_a[fi]);
            vz_dot_vz = nalu_hypre_StructInnerProd(visitz_a[fi], visitz_a[fi]);
#else
            /* turn all grids on */
            grid_on[fi] = 1;
#endif

            /*-------------------------------
             * compute Px, RTx, and A
             *-------------------------------*/

            if (lx < (num_grids[0] - 1))
            {
               nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);
               nalu_hypre_SparseMSGMapIndex((lx + 1), ly, lz, num_grids, ci);

               nalu_hypre_SparseMSGSetCIndex(0, cindex);
               nalu_hypre_SparseMSGSetFIndex(0, findex);
               nalu_hypre_SparseMSGSetStride(0, stride);

               /* compute x-transfer operator */
               if ((lz == 0) && (ly == 0))
               {
                  Px_a[lx] = nalu_hypre_PFMGCreateInterpOp(A_a[fi],
                                                      Px_grid_a[lx + 1], 0, 0);
                  nalu_hypre_StructMatrixInitialize(Px_a[lx]);
                  nalu_hypre_PFMGSetupInterpOp(A_a[fi], 0, findex, stride,
                                          Px_a[lx], 0);
                  RTx_a[lx] = Px_a[lx];
               }

               /* compute coarse-operator with Px */
               A_a[ci] =
                  nalu_hypre_SparseMSGCreateRAPOp(RTx_a[lx], A_a[fi], Px_a[lx],
                                             grid_a[ci], 0);
               nalu_hypre_StructMatrixInitialize(A_a[ci]);
               nalu_hypre_SetIndex3(stridePR, 1, pow(2, ly), pow(2, lz));
               nalu_hypre_SparseMSGSetupRAPOp(RTx_a[lx], A_a[fi], Px_a[lx],
                                         0, cindex, stride, stridePR, A_a[ci]);
            }
         }

         /* RDF */
#if 0
         /* free up some coarse-operators to conserve memory */
         for (lx = 1; lx <= nalu_hypre_min((jump - ly - lz), (num_grids[0] - 1)); lx++)
         {
            nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);
            nalu_hypre_StructMatrixDestroy(A_a[fi]);
            A_a[fi] = NULL;
         }
#endif
         /* RDF */

         /*-------------------------------
          * compute Py, RTy, and A
          *-------------------------------*/

         if (ly < (num_grids[1] - 1))
         {
            nalu_hypre_SparseMSGMapIndex(0, ly, lz, num_grids, fi);
            nalu_hypre_SparseMSGMapIndex(0, (ly + 1), lz, num_grids, ci);

            nalu_hypre_SparseMSGSetCIndex(1, cindex);
            nalu_hypre_SparseMSGSetFIndex(1, findex);
            nalu_hypre_SparseMSGSetStride(1, stride);

            /* compute y-transfer operators */
            if (lz == 0)
            {
               Py_a[ly] = nalu_hypre_PFMGCreateInterpOp(A_a[fi],
                                                   Py_grid_a[ly + 1], 1, 0);
               nalu_hypre_StructMatrixInitialize(Py_a[ly]);
               nalu_hypre_PFMGSetupInterpOp(A_a[fi], 1, findex, stride,
                                       Py_a[ly], 0);
               RTy_a[ly] = Py_a[ly];
            }

            /* compute coarse-operator with Py */
            A_a[ci] = nalu_hypre_SparseMSGCreateRAPOp(RTy_a[ly], A_a[fi], Py_a[ly],
                                                 grid_a[ci], 1);
            nalu_hypre_StructMatrixInitialize(A_a[ci]);
            nalu_hypre_SetIndex3(stridePR, 1, 1, pow(2, lz));
            nalu_hypre_SparseMSGSetupRAPOp(RTy_a[ly], A_a[fi], Py_a[ly],
                                      1, cindex, stride, stridePR, A_a[ci]);
         }
      }

      /* RDF */
#if 0
      /* free up some coarse-operators to conserve memory */
      for (ly = 1; ly <= nalu_hypre_min((jump - lz), (num_grids[1] - 1)); ly++)
      {
         nalu_hypre_SparseMSGMapIndex(0, ly, lz, num_grids, fi);
         nalu_hypre_StructMatrixDestroy(A_a[fi]);
         A_a[fi] = NULL;
      }
#endif
      /* RDF */

      /*-------------------------------
       * compute Pz, RTz, and A
       *-------------------------------*/

      if (lz < (num_grids[2] - 1))
      {
         nalu_hypre_SparseMSGMapIndex(0, 0, lz, num_grids, fi);
         nalu_hypre_SparseMSGMapIndex(0, 0, (lz + 1), num_grids, ci);

         nalu_hypre_SparseMSGSetCIndex(2, cindex);
         nalu_hypre_SparseMSGSetFIndex(2, findex);
         nalu_hypre_SparseMSGSetStride(2, stride);

         /* compute z-transfer operators */
         Pz_a[lz] = nalu_hypre_PFMGCreateInterpOp(A_a[fi], Pz_grid_a[lz + 1], 2, 0);
         nalu_hypre_StructMatrixInitialize(Pz_a[lz]);
         nalu_hypre_PFMGSetupInterpOp(A_a[fi], 2, findex, stride, Pz_a[lz], 0);
         RTz_a[lz] = Pz_a[lz];

         /* compute coarse-operator with Pz */
         A_a[ci] = nalu_hypre_SparseMSGCreateRAPOp(RTz_a[lz], A_a[fi], Pz_a[lz],
                                              grid_a[ci], 2);
         nalu_hypre_StructMatrixInitialize(A_a[ci]);
         nalu_hypre_SetIndex3(stridePR, 1, 1, 1);
         nalu_hypre_SparseMSGSetupRAPOp(RTz_a[lz], A_a[fi], Pz_a[lz],
                                   2, cindex, stride, stridePR, A_a[ci]);
      }
   }

   /* RDF */
#if 0
   /* free up some coarse-operators to conserve memory */
   for (lz = 1; lz <= nalu_hypre_min((jump), (num_grids[2] - 1)); lz++)
   {
      nalu_hypre_SparseMSGMapIndex(0, 0, lz, num_grids, fi);
      nalu_hypre_StructMatrixDestroy(A_a[fi]);
      A_a[fi] = NULL;
   }
#endif
   /* RDF */

   (smsg_data -> A_array)   = A_a;
   (smsg_data -> Px_array)  = Px_a;
   (smsg_data -> Py_array)  = Py_a;
   (smsg_data -> Pz_array)  = Pz_a;
   (smsg_data -> RTx_array) = RTx_a;
   (smsg_data -> RTy_array) = RTy_a;
   (smsg_data -> RTz_array) = RTz_a;

   (smsg_data -> visitx_array) = visitx_a;
   (smsg_data -> visity_array) = visity_a;
   (smsg_data -> visitz_array) = visitz_a;
   (smsg_data -> grid_on)      = grid_on;

   /*------------------------------------------------------
    *  Set up vector structures
    *-----------------------------------------------------*/

   b_a = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   x_a = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   t_a = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   r_a = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   e_a = t_a;

   data_size = 0;

   b_a[0] = nalu_hypre_StructVectorRef(b);
   x_a[0] = nalu_hypre_StructVectorRef(x);

   t_a[0] = nalu_hypre_StructVectorCreate(comm, grid_a[0]);
   nalu_hypre_StructVectorSetNumGhost(t_a[0], x_num_ghost);
   nalu_hypre_StructVectorInitializeShell(t_a[0]);
   data_size += nalu_hypre_StructVectorDataSize(t_a[0]);

   r_a[0] = nalu_hypre_StructVectorCreate(comm, grid_a[0]);
   nalu_hypre_StructVectorSetNumGhost(r_a[0], x_num_ghost);
   nalu_hypre_StructVectorInitializeShell(r_a[0]);
   data_size += nalu_hypre_StructVectorDataSize(r_a[0]);

   for (lz = 0; lz < num_grids[2]; lz++)
   {
      for (ly = 0; ly < num_grids[1]; ly++)
      {
         for (lx = 0; lx < num_grids[0]; lx++)
         {
            l = lx + ly + lz;

            if (l >= 1)
            {
               nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);

               x_a[fi] = nalu_hypre_StructVectorCreate(comm, grid_a[fi]);
               nalu_hypre_StructVectorSetNumGhost(x_a[fi], x_num_ghost);
               nalu_hypre_StructVectorInitializeShell(x_a[fi]);
               data_size += nalu_hypre_StructVectorDataSize(x_a[fi]);

               t_a[fi] = nalu_hypre_StructVectorCreate(comm, grid_a[fi]);
               nalu_hypre_StructVectorSetNumGhost(t_a[fi], x_num_ghost);
               nalu_hypre_StructVectorInitializeShell(t_a[fi]);

               /* set vector structures in jump region */
               if (l <= jump)
               {
                  b_a[fi] = x_a[fi];
                  r_a[fi] = x_a[fi];
               }

               /* set vector structures outside of jump region */
               else
               {
                  b_a[fi] = nalu_hypre_StructVectorCreate(comm, grid_a[fi]);
                  nalu_hypre_StructVectorSetNumGhost(b_a[fi], b_num_ghost);
                  nalu_hypre_StructVectorInitializeShell(b_a[fi]);
                  data_size += nalu_hypre_StructVectorDataSize(b_a[fi]);

                  r_a[fi] = nalu_hypre_StructVectorCreate(comm, grid_a[fi]);
                  nalu_hypre_StructVectorSetNumGhost(r_a[fi], x_num_ghost);
                  nalu_hypre_StructVectorInitializeShell(r_a[fi]);
               }
            }
         }
      }
   }

   data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, data_size, memory_location);

   (smsg_data -> data) = data;
   (smsg_data -> memory_location) = memory_location;

   nalu_hypre_StructVectorInitializeData(t_a[0], data);
   nalu_hypre_StructVectorAssemble(t_a[0]);
   data += nalu_hypre_StructVectorDataSize(t_a[0]);

   nalu_hypre_StructVectorInitializeData(r_a[0], data);
   nalu_hypre_StructVectorAssemble(r_a[0]);
   data += nalu_hypre_StructVectorDataSize(r_a[0]);

   for (lz = 0; lz < num_grids[2]; lz++)
   {
      for (ly = 0; ly < num_grids[1]; ly++)
      {
         for (lx = 0; lx < num_grids[0]; lx++)
         {
            l = lx + ly + lz;

            if (l >= 1)
            {
               nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);

               nalu_hypre_StructVectorInitializeData(x_a[fi], data);
               nalu_hypre_StructVectorAssemble(x_a[fi]);
               data += nalu_hypre_StructVectorDataSize(x_a[fi]);

               tdata = nalu_hypre_StructVectorData(t_a[0]);
               nalu_hypre_StructVectorInitializeData(t_a[fi], tdata);

               /* set vector structures outside of jump region */
               if (l > jump)
               {
                  nalu_hypre_StructVectorInitializeData(b_a[fi], data);
                  nalu_hypre_StructVectorAssemble(b_a[fi]);
                  data += nalu_hypre_StructVectorDataSize(b_a[fi]);

                  tdata = nalu_hypre_StructVectorData(r_a[0]);
                  nalu_hypre_StructVectorInitializeData(r_a[fi], tdata);
               }
            }
         }
      }
   }

   (smsg_data -> b_array) = b_a;
   (smsg_data -> x_array) = x_a;
   (smsg_data -> t_array) = t_a;
   (smsg_data -> r_array) = r_a;
   (smsg_data -> e_array) = e_a;

   /*------------------------------------------------------
    *  Call setup routines
    *-----------------------------------------------------*/

   relax_a     = nalu_hypre_CTAlloc(void *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   matvec_a    = nalu_hypre_CTAlloc(void *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   restrictx_a = nalu_hypre_CTAlloc(void *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   restricty_a = nalu_hypre_CTAlloc(void *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   restrictz_a = nalu_hypre_CTAlloc(void *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   interpx_a   = nalu_hypre_CTAlloc(void *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   interpy_a   = nalu_hypre_CTAlloc(void *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);
   interpz_a   = nalu_hypre_CTAlloc(void *,  num_all_grids, NALU_HYPRE_MEMORY_HOST);

   /* set up x-transfer routines */
   for (lx = 0; lx < (num_grids[0] - 1); lx++)
   {
      nalu_hypre_SparseMSGSetCIndex(0, cindex);
      nalu_hypre_SparseMSGSetFIndex(0, findex);
      nalu_hypre_SparseMSGSetStride(0, stride);

      for (lz = 0; lz < num_grids[2]; lz++)
      {
         for (ly = 0; ly < num_grids[1]; ly++)
         {
            nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);
            nalu_hypre_SparseMSGMapIndex(lx + 1, ly, lz, num_grids, ci);

            nalu_hypre_SetIndex3(stridePR, 1, pow(2, ly), pow(2, lz));

            interpx_a[fi] = nalu_hypre_SparseMSGInterpCreate();
            nalu_hypre_SparseMSGInterpSetup(interpx_a[fi], Px_a[lx],
                                       x_a[ci], e_a[fi],
                                       cindex, findex, stride, stridePR);

            restrictx_a[fi] = nalu_hypre_SparseMSGRestrictCreate();
            nalu_hypre_SparseMSGRestrictSetup(restrictx_a[fi], RTx_a[lx],
                                         r_a[fi], b_a[ci],
                                         cindex, findex, stride, stridePR);
         }
      }
   }

   /* set up y-transfer routines */
   for (ly = 0; ly < (num_grids[1] - 1); ly++)
   {
      nalu_hypre_SparseMSGSetCIndex(1, cindex);
      nalu_hypre_SparseMSGSetFIndex(1, findex);
      nalu_hypre_SparseMSGSetStride(1, stride);

      for (lz = 0; lz < num_grids[2]; lz++)
      {
         for (lx = 0; lx < num_grids[0]; lx++)
         {
            nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);
            nalu_hypre_SparseMSGMapIndex(lx, ly + 1, lz, num_grids, ci);

            nalu_hypre_SetIndex3(stridePR, pow(2, lx), 1, pow(2, lz));

            interpy_a[fi] = nalu_hypre_SparseMSGInterpCreate();
            nalu_hypre_SparseMSGInterpSetup(interpy_a[fi], Py_a[ly],
                                       x_a[ci], e_a[fi],
                                       cindex, findex, stride, stridePR);

            restricty_a[fi] = nalu_hypre_SparseMSGRestrictCreate();
            nalu_hypre_SparseMSGRestrictSetup(restricty_a[fi], RTy_a[ly],
                                         r_a[fi], b_a[ci],
                                         cindex, findex, stride, stridePR);
         }
      }
   }

   /* set up z-transfer routines */
   for (lz = 0; lz < (num_grids[2] - 1); lz++)
   {
      nalu_hypre_SparseMSGSetCIndex(2, cindex);
      nalu_hypre_SparseMSGSetFIndex(2, findex);
      nalu_hypre_SparseMSGSetStride(2, stride);

      for (ly = 0; ly < num_grids[1]; ly++)
      {
         for (lx = 0; lx < num_grids[0]; lx++)
         {
            nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);
            nalu_hypre_SparseMSGMapIndex(lx, ly, lz + 1, num_grids, ci);

            nalu_hypre_SetIndex3(stridePR, pow(2, lx), pow(2, ly), 1);

            interpz_a[fi] = nalu_hypre_SparseMSGInterpCreate();
            nalu_hypre_SparseMSGInterpSetup(interpz_a[fi], Pz_a[lz],
                                       x_a[ci], e_a[fi],
                                       cindex, findex, stride, stridePR);

            restrictz_a[fi] = nalu_hypre_SparseMSGRestrictCreate();
            nalu_hypre_SparseMSGRestrictSetup(restrictz_a[fi], RTz_a[lz],
                                         r_a[fi], b_a[ci],
                                         cindex, findex, stride, stridePR);
         }
      }
   }

   /* set up fine grid relaxation */
   relax_a[0] = nalu_hypre_PFMGRelaxCreate(comm);
   nalu_hypre_PFMGRelaxSetTol(relax_a[0], 0.0);
   nalu_hypre_PFMGRelaxSetType(relax_a[0], relax_type);
   if (usr_jacobi_weight)
   {
      nalu_hypre_PFMGRelaxSetJacobiWeight(relax_a[0], jacobi_weight);
   }
   nalu_hypre_PFMGRelaxSetTempVec(relax_a[0], t_a[0]);
   nalu_hypre_PFMGRelaxSetup(relax_a[0], A_a[0], b_a[0], x_a[0]);
   /* set up the fine grid residual routine */
   matvec_a[0] = nalu_hypre_StructMatvecCreate();
   nalu_hypre_StructMatvecSetup(matvec_a[0], A_a[0], x_a[0]);
   if (num_levels > 1)
   {
      for (lz = 0; lz < num_grids[2]; lz++)
      {
         for (ly = 0; ly < num_grids[1]; ly++)
         {
            for (lx = 0; lx < num_grids[0]; lx++)
            {
               l = lx + ly + lz;

               if ((l > jump) && (l < (num_levels - 1)))
               {
                  nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);

                  /* set up relaxation */
                  relax_a[fi] = nalu_hypre_PFMGRelaxCreate(comm);
                  nalu_hypre_PFMGRelaxSetTol(relax_a[fi], 0.0);
                  nalu_hypre_PFMGRelaxSetType(relax_a[fi], relax_type);
                  if (usr_jacobi_weight)
                  {
                     nalu_hypre_PFMGRelaxSetJacobiWeight(relax_a[fi], jacobi_weight);
                  }
                  nalu_hypre_PFMGRelaxSetTempVec(relax_a[fi], t_a[fi]);
                  nalu_hypre_PFMGRelaxSetup(relax_a[fi], A_a[fi], b_a[fi], x_a[fi]);

                  /* set up the residual routine */
                  matvec_a[fi] = nalu_hypre_StructMatvecCreate();
                  nalu_hypre_StructMatvecSetup(matvec_a[fi], A_a[fi], x_a[fi]);
               }
            }
         }
      }
      /* set up coarsest grid relaxation */
      fi = num_all_grids - 1;
      relax_a[fi] = nalu_hypre_PFMGRelaxCreate(comm);
      nalu_hypre_PFMGRelaxSetTol(relax_a[fi], 0.0);
      nalu_hypre_PFMGRelaxSetMaxIter(relax_a[fi], 1);
      nalu_hypre_PFMGRelaxSetType(relax_a[fi], 0);
      if (usr_jacobi_weight)
      {
         nalu_hypre_PFMGRelaxSetJacobiWeight(relax_a[fi], jacobi_weight);
      }
      nalu_hypre_PFMGRelaxSetTempVec(relax_a[fi], t_a[fi]);
      nalu_hypre_PFMGRelaxSetup(relax_a[fi], A_a[fi], b_a[fi], x_a[fi]);
   }

   (smsg_data -> relax_array)     = relax_a;
   (smsg_data -> matvec_array)    = matvec_a;
   (smsg_data -> restrictx_array) = restrictx_a;
   (smsg_data -> restricty_array) = restricty_a;
   (smsg_data -> restrictz_array) = restrictz_a;
   (smsg_data -> interpx_array)   = interpx_a;
   (smsg_data -> interpy_array)   = interpy_a;
   (smsg_data -> interpz_array)   = interpz_a;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((smsg_data -> logging) > 0)
   {
      max_iter = (smsg_data -> max_iter);
      (smsg_data -> norms)     = nalu_hypre_TAlloc(NALU_HYPRE_Real,  max_iter, NALU_HYPRE_MEMORY_HOST);
      (smsg_data -> rel_norms) = nalu_hypre_TAlloc(NALU_HYPRE_Real,  max_iter, NALU_HYPRE_MEMORY_HOST);
   }

#if DEBUG
   for (lz = 0; lz < num_grids[2]; lz++)
   {
      for (ly = 0; ly < num_grids[1]; ly++)
      {
         for (lx = 0; lx < num_grids[0]; lx++)
         {
            l = lx + ly + lz;

            if ((l == 0) || (l > jump))
            {
               nalu_hypre_SparseMSGMapIndex(lx, ly, lz, num_grids, fi);

               nalu_hypre_sprintf(filename, "zoutSMSG_A.%d.%d.%d", lx, ly, lz);
               nalu_hypre_StructMatrixPrint(filename, A_a[fi], 0);

               nalu_hypre_sprintf(filename, "zoutSMSG_visitx.%d.%d.%d", lx, ly, lz);
               nalu_hypre_StructVectorPrint(filename, visitx_a[fi], 0);
               nalu_hypre_sprintf(filename, "zoutSMSG_visity.%d.%d.%d", lx, ly, lz);
               nalu_hypre_StructVectorPrint(filename, visity_a[fi], 0);
               nalu_hypre_sprintf(filename, "zoutSMSG_visitz.%d.%d.%d", lx, ly, lz);
               nalu_hypre_StructVectorPrint(filename, visitz_a[fi], 0);
            }
         }
      }
   }
   for (lx = 0; lx < num_grids[0] - 1; lx++)
   {
      nalu_hypre_sprintf(filename, "zoutSMSG_Px.%d", lx);
      nalu_hypre_StructMatrixPrint(filename, Px_a[lx], 0);
   }
   for (ly = 0; ly < num_grids[1] - 1; ly++)
   {
      nalu_hypre_sprintf(filename, "zoutSMSG_Py.%d", ly);
      nalu_hypre_StructMatrixPrint(filename, Py_a[ly], 0);
   }
   for (lz = 0; lz < num_grids[2] - 1; lz++)
   {
      nalu_hypre_sprintf(filename, "zoutSMSG_Pz.%d", lz);
      nalu_hypre_StructMatrixPrint(filename, Pz_a[lz], 0);
   }
#endif

   return ierr;
}

