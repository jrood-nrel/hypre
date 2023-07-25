/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "sys_pfmg.h"

#define DEBUG 0

#define nalu_hypre_PFMGSetCIndex(cdir, cindex)       \
   {                                            \
      nalu_hypre_SetIndex3(cindex, 0, 0, 0);          \
      nalu_hypre_IndexD(cindex, cdir) = 0;           \
   }

#define nalu_hypre_PFMGSetFIndex(cdir, findex)       \
   {                                            \
      nalu_hypre_SetIndex3(findex, 0, 0, 0);          \
      nalu_hypre_IndexD(findex, cdir) = 1;           \
   }

#define nalu_hypre_PFMGSetStride(cdir, stride)       \
   {                                            \
      nalu_hypre_SetIndex3(stride, 1, 1, 1);          \
      nalu_hypre_IndexD(stride, cdir) = 2;           \
   }

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetup( void                 *sys_pfmg_vdata,
                    nalu_hypre_SStructMatrix  *A_in,
                    nalu_hypre_SStructVector  *b_in,
                    nalu_hypre_SStructVector  *x_in        )
{
   nalu_hypre_SysPFMGData    *sys_pfmg_data = (nalu_hypre_SysPFMGData    *)sys_pfmg_vdata;

   MPI_Comm              comm = (sys_pfmg_data -> comm);

   nalu_hypre_SStructPMatrix *A;
   nalu_hypre_SStructPVector *b;
   nalu_hypre_SStructPVector *x;

   NALU_HYPRE_Int             relax_type = (sys_pfmg_data -> relax_type);
   NALU_HYPRE_Int             usr_jacobi_weight = (sys_pfmg_data -> usr_jacobi_weight);
   NALU_HYPRE_Real            jacobi_weight    = (sys_pfmg_data -> jacobi_weight);
   NALU_HYPRE_Int             skip_relax = (sys_pfmg_data -> skip_relax);
   NALU_HYPRE_Real           *dxyz       = (sys_pfmg_data -> dxyz);

   NALU_HYPRE_Int             max_iter;
   NALU_HYPRE_Int             max_levels;

   NALU_HYPRE_Int             num_levels;

   nalu_hypre_Index           cindex;
   nalu_hypre_Index           findex;
   nalu_hypre_Index           stride;

   nalu_hypre_Index           coarsen;

   NALU_HYPRE_Int              *cdir_l;
   NALU_HYPRE_Int              *active_l;
   nalu_hypre_SStructPGrid    **grid_l;
   nalu_hypre_SStructPGrid    **P_grid_l;

   nalu_hypre_SStructPMatrix  **A_l;
   nalu_hypre_SStructPMatrix  **P_l;
   nalu_hypre_SStructPMatrix  **RT_l;
   nalu_hypre_SStructPVector  **b_l;
   nalu_hypre_SStructPVector  **x_l;

   /* temp vectors */
   nalu_hypre_SStructPVector  **tx_l;
   nalu_hypre_SStructPVector  **r_l;
   nalu_hypre_SStructPVector  **e_l;

   void                **relax_data_l;
   void                **matvec_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   nalu_hypre_SStructPGrid     *grid;
   nalu_hypre_StructGrid       *sgrid;
   NALU_HYPRE_Int               dim;
   NALU_HYPRE_Int               full_periodic;

   nalu_hypre_Box            *cbox;

   NALU_HYPRE_Real           *relax_weights;
   NALU_HYPRE_Real           *mean, *deviation;
   NALU_HYPRE_Real            alpha, beta;
   NALU_HYPRE_Int             dxyz_flag;

   NALU_HYPRE_Real            min_dxyz;
   NALU_HYPRE_Int             cdir, periodic, cmaxsize;
   NALU_HYPRE_Int             d, l;
   NALU_HYPRE_Int             i;

   NALU_HYPRE_Real**              sys_dxyz;

   NALU_HYPRE_Int             nvars;

#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Refs to A,x,b (the PMatrix & PVectors within
    * the input SStructMatrix & SStructVectors)
    *-----------------------------------------------------*/
   nalu_hypre_SStructPMatrixRef(nalu_hypre_SStructMatrixPMatrix(A_in, 0), &A);
   nalu_hypre_SStructPVectorRef(nalu_hypre_SStructVectorPVector(b_in, 0), &b);
   nalu_hypre_SStructPVectorRef(nalu_hypre_SStructVectorPVector(x_in, 0), &x);

   /*--------------------------------------------------------
    * Allocate arrays for mesh sizes for each diagonal block
    *--------------------------------------------------------*/
   nvars    = nalu_hypre_SStructPMatrixNVars(A);
   sys_dxyz = nalu_hypre_TAlloc(NALU_HYPRE_Real *, nvars, NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < nvars; i++)
   {
      sys_dxyz[i] = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid  = nalu_hypre_SStructPMatrixPGrid(A);
   sgrid = nalu_hypre_SStructPGridSGrid(grid, 0);
   dim   = nalu_hypre_StructGridNDim(sgrid);

   /* Compute a new max_levels value based on the grid */
   cbox = nalu_hypre_BoxDuplicate(nalu_hypre_StructGridBoundingBox(sgrid));
   max_levels =
      nalu_hypre_Log2(nalu_hypre_BoxSizeD(cbox, 0)) + 2 +
      nalu_hypre_Log2(nalu_hypre_BoxSizeD(cbox, 1)) + 2 +
      nalu_hypre_Log2(nalu_hypre_BoxSizeD(cbox, 2)) + 2;
   if ((sys_pfmg_data -> max_levels) > 0)
   {
      max_levels = nalu_hypre_min(max_levels, (sys_pfmg_data -> max_levels));
   }
   (sys_pfmg_data -> max_levels) = max_levels;

   /* compute dxyz */
   dxyz_flag = 0;
   if ((dxyz[0] == 0) || (dxyz[1] == 0) || (dxyz[2] == 0))
   {
      mean      = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
      deviation = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);

      dxyz_flag = 0;
      for (i = 0; i < nvars; i++)
      {
         nalu_hypre_PFMGComputeDxyz(nalu_hypre_SStructPMatrixSMatrix(A, i, i), sys_dxyz[i],
                               mean, deviation);

         /* signal flag if any of the flag has a large (square) coeff. of
          * variation */
         if (!dxyz_flag)
         {
            for (d = 0; d < dim; d++)
            {
               deviation[d] -= mean[d] * mean[d];
               /* square of coeff. of variation */
               if (deviation[d] / (mean[d]*mean[d]) > .1)
               {
                  dxyz_flag = 1;
                  break;
               }
            }
         }

         for (d = 0; d < 3; d++)
         {
            dxyz[d] += sys_dxyz[i][d];
         }
      }
      nalu_hypre_TFree(mean, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(deviation, NALU_HYPRE_MEMORY_HOST);
   }

   grid_l = nalu_hypre_TAlloc(nalu_hypre_SStructPGrid *, max_levels, NALU_HYPRE_MEMORY_HOST);
   grid_l[0] = grid;
   P_grid_l = nalu_hypre_TAlloc(nalu_hypre_SStructPGrid *, max_levels, NALU_HYPRE_MEMORY_HOST);
   P_grid_l[0] = NULL;
   cdir_l = nalu_hypre_TAlloc(NALU_HYPRE_Int, max_levels, NALU_HYPRE_MEMORY_HOST);
   active_l = nalu_hypre_TAlloc(NALU_HYPRE_Int, max_levels, NALU_HYPRE_MEMORY_HOST);
   relax_weights = nalu_hypre_CTAlloc(NALU_HYPRE_Real, max_levels, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SetIndex3(coarsen, 1, 1, 1); /* forces relaxation on finest grid */
   for (l = 0; ; l++)
   {
      /* determine cdir */
      min_dxyz = dxyz[0] + dxyz[1] + dxyz[2] + 1;
      cdir = -1;
      alpha = 0.0;
      for (d = 0; d < dim; d++)
      {
         if ((nalu_hypre_BoxIMaxD(cbox, d) > nalu_hypre_BoxIMinD(cbox, d)) &&
             (dxyz[d] < min_dxyz))
         {
            min_dxyz = dxyz[d];
            cdir = d;
         }
         alpha += 1.0 / (dxyz[d] * dxyz[d]);
      }
      relax_weights[l] = 2.0 / 3.0;

      /* If it's possible to coarsen, change relax_weights */
      beta = 0.0;
      if (cdir != -1)
      {
         if (dxyz_flag)
         {
            relax_weights[l] = 2.0 / 3.0;
         }

         else
         {
            for (d = 0; d < dim; d++)
            {
               if (d != cdir)
               {
                  beta += 1.0 / (dxyz[d] * dxyz[d]);
               }
            }
            if (beta == alpha)
            {
               alpha = 0.0;
            }
            else
            {
               alpha = beta / alpha;
            }

            /* determine level Jacobi weights */
            if (dim > 1)
            {
               relax_weights[l] = 2.0 / (3.0 - alpha);
            }
            else
            {
               relax_weights[l] = 2.0 / 3.0; /* always 2/3 for 1-d */
            }
         }
      }

      if (cdir != -1)
      {
         /* don't coarsen if a periodic direction and not divisible by 2 */
         periodic = nalu_hypre_IndexD(nalu_hypre_StructGridPeriodic(grid_l[l]), cdir);
         if ((periodic) && (periodic % 2))
         {
            cdir = -1;
         }

         /* don't coarsen if we've reached max_levels */
         if (l == (max_levels - 1))
         {
            cdir = -1;
         }
      }

      /* stop coarsening */
      if (cdir == -1)
      {
         active_l[l] = 1; /* forces relaxation on coarsest grid */
         cmaxsize = 0;
         for (d = 0; d < dim; d++)
         {
            cmaxsize = nalu_hypre_max(cmaxsize, nalu_hypre_BoxSizeD(cbox, d));
         }

         break;
      }

      cdir_l[l] = cdir;

      if (nalu_hypre_IndexD(coarsen, cdir) != 0)
      {
         /* coarsened previously in this direction, relax level l */
         active_l[l] = 1;
         nalu_hypre_SetIndex3(coarsen, 0, 0, 0);
         nalu_hypre_IndexD(coarsen, cdir) = 1;
      }
      else
      {
         active_l[l] = 0;
         nalu_hypre_IndexD(coarsen, cdir) = 1;
      }

      /* set cindex, findex, and stride */
      nalu_hypre_PFMGSetCIndex(cdir, cindex);
      nalu_hypre_PFMGSetFIndex(cdir, findex);
      nalu_hypre_PFMGSetStride(cdir, stride);

      /* update dxyz and coarsen cbox*/
      dxyz[cdir] *= 2;
      nalu_hypre_ProjectBox(cbox, cindex, stride);
      nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(cbox), cindex, stride,
                                  nalu_hypre_BoxIMin(cbox));
      nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(cbox), cindex, stride,
                                  nalu_hypre_BoxIMax(cbox));

      /* build the interpolation grid */
      nalu_hypre_SysStructCoarsen(grid_l[l], findex, stride, 0, &P_grid_l[l + 1]);

      /* build the coarse grid */
      nalu_hypre_SysStructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l + 1]);
   }
   num_levels = l + 1;

   /*-----------------------------------------------------
    * For fully periodic problems, the coarsest grid
    * problem (a single node) can have zero diagonal
    * blocks. This causes problems with the gselim
    * routine (which doesn't do pivoting). We avoid
    * this by skipping relaxation.
    *-----------------------------------------------------*/

   full_periodic = 1;
   for (d = 0; d < dim; d++)
   {
      full_periodic *= nalu_hypre_IndexD(nalu_hypre_SStructPGridPeriodic(grid), d);
   }
   if ( full_periodic != 0)
   {
      nalu_hypre_SStructPGridDestroy(grid_l[num_levels - 1]);
      nalu_hypre_SStructPGridDestroy(P_grid_l[num_levels - 1]);
      num_levels -= 1;
   }

   /* free up some things */
   nalu_hypre_BoxDestroy(cbox);
   for ( i = 0; i < nvars; i++)
   {
      nalu_hypre_TFree(sys_dxyz[i], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(sys_dxyz, NALU_HYPRE_MEMORY_HOST);


   /* set all levels active if skip_relax = 0 */
   if (!skip_relax)
   {
      for (l = 0; l < num_levels; l++)
      {
         active_l[l] = 1;
      }
   }

   (sys_pfmg_data -> num_levels) = num_levels;
   (sys_pfmg_data -> cdir_l)     = cdir_l;
   (sys_pfmg_data -> active_l)   = active_l;
   (sys_pfmg_data -> grid_l)     = grid_l;
   (sys_pfmg_data -> P_grid_l)   = P_grid_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = nalu_hypre_TAlloc(nalu_hypre_SStructPMatrix *, num_levels, NALU_HYPRE_MEMORY_HOST);
   P_l  = nalu_hypre_TAlloc(nalu_hypre_SStructPMatrix *, num_levels - 1, NALU_HYPRE_MEMORY_HOST);
   RT_l = nalu_hypre_TAlloc(nalu_hypre_SStructPMatrix *, num_levels - 1, NALU_HYPRE_MEMORY_HOST);
   b_l  = nalu_hypre_TAlloc(nalu_hypre_SStructPVector *, num_levels, NALU_HYPRE_MEMORY_HOST);
   x_l  = nalu_hypre_TAlloc(nalu_hypre_SStructPVector *, num_levels, NALU_HYPRE_MEMORY_HOST);
   tx_l = nalu_hypre_TAlloc(nalu_hypre_SStructPVector *, num_levels, NALU_HYPRE_MEMORY_HOST);
   r_l  = tx_l;
   e_l  = tx_l;

   nalu_hypre_SStructPMatrixRef(A, &A_l[0]);
   nalu_hypre_SStructPVectorRef(b, &b_l[0]);
   nalu_hypre_SStructPVectorRef(x, &x_l[0]);

   nalu_hypre_SStructPVectorCreate(comm, grid_l[0], &tx_l[0]);
   nalu_hypre_SStructPVectorInitialize(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      cdir = cdir_l[l];

      P_l[l]  = nalu_hypre_SysPFMGCreateInterpOp(A_l[l], P_grid_l[l + 1], cdir);
      nalu_hypre_SStructPMatrixInitialize(P_l[l]);

      RT_l[l] = P_l[l];

      A_l[l + 1] = nalu_hypre_SysPFMGCreateRAPOp(RT_l[l], A_l[l], P_l[l],
                                            grid_l[l + 1], cdir);
      nalu_hypre_SStructPMatrixInitialize(A_l[l + 1]);

      nalu_hypre_SStructPVectorCreate(comm, grid_l[l + 1], &b_l[l + 1]);
      nalu_hypre_SStructPVectorInitialize(b_l[l + 1]);

      nalu_hypre_SStructPVectorCreate(comm, grid_l[l + 1], &x_l[l + 1]);
      nalu_hypre_SStructPVectorInitialize(x_l[l + 1]);

      nalu_hypre_SStructPVectorCreate(comm, grid_l[l + 1], &tx_l[l + 1]);
      nalu_hypre_SStructPVectorInitialize(tx_l[l + 1]);
   }

   nalu_hypre_SStructPVectorAssemble(tx_l[0]);
   for (l = 0; l < (num_levels - 1); l++)
   {
      nalu_hypre_SStructPVectorAssemble(b_l[l + 1]);
      nalu_hypre_SStructPVectorAssemble(x_l[l + 1]);
      nalu_hypre_SStructPVectorAssemble(tx_l[l + 1]);
   }

   (sys_pfmg_data -> A_l)  = A_l;
   (sys_pfmg_data -> P_l)  = P_l;
   (sys_pfmg_data -> RT_l) = RT_l;
   (sys_pfmg_data -> b_l)  = b_l;
   (sys_pfmg_data -> x_l)  = x_l;
   (sys_pfmg_data -> tx_l) = tx_l;
   (sys_pfmg_data -> r_l)  = r_l;
   (sys_pfmg_data -> e_l)  = e_l;

   /*-----------------------------------------------------
    * Set up multigrid operators and call setup routines
    *-----------------------------------------------------*/

   relax_data_l    = nalu_hypre_TAlloc(void *, num_levels, NALU_HYPRE_MEMORY_HOST);
   matvec_data_l   = nalu_hypre_TAlloc(void *, num_levels, NALU_HYPRE_MEMORY_HOST);
   restrict_data_l = nalu_hypre_TAlloc(void *, num_levels, NALU_HYPRE_MEMORY_HOST);
   interp_data_l   = nalu_hypre_TAlloc(void *, num_levels, NALU_HYPRE_MEMORY_HOST);

   for (l = 0; l < (num_levels - 1); l++)
   {
      cdir = cdir_l[l];

      nalu_hypre_PFMGSetCIndex(cdir, cindex);
      nalu_hypre_PFMGSetFIndex(cdir, findex);
      nalu_hypre_PFMGSetStride(cdir, stride);

      /* set up interpolation operator */
      nalu_hypre_SysPFMGSetupInterpOp(A_l[l], cdir, findex, stride, P_l[l]);

      /* set up the coarse grid operator */
      nalu_hypre_SysPFMGSetupRAPOp(RT_l[l], A_l[l], P_l[l],
                              cdir, cindex, stride, A_l[l + 1]);

      /* set up the interpolation routine */
      nalu_hypre_SysSemiInterpCreate(&interp_data_l[l]);
      nalu_hypre_SysSemiInterpSetup(interp_data_l[l], P_l[l], 0, x_l[l + 1], e_l[l],
                               cindex, findex, stride);

      /* set up the restriction routine */
      nalu_hypre_SysSemiRestrictCreate(&restrict_data_l[l]);
      nalu_hypre_SysSemiRestrictSetup(restrict_data_l[l], RT_l[l], 1, r_l[l], b_l[l + 1],
                                 cindex, findex, stride);
   }

   /* set up fine grid relaxation */
   relax_data_l[0] = nalu_hypre_SysPFMGRelaxCreate(comm);
   nalu_hypre_SysPFMGRelaxSetTol(relax_data_l[0], 0.0);
   if (usr_jacobi_weight)
   {
      nalu_hypre_SysPFMGRelaxSetJacobiWeight(relax_data_l[0], jacobi_weight);
   }
   else
   {
      nalu_hypre_SysPFMGRelaxSetJacobiWeight(relax_data_l[0], relax_weights[0]);
   }
   nalu_hypre_SysPFMGRelaxSetType(relax_data_l[0], relax_type);
   nalu_hypre_SysPFMGRelaxSetTempVec(relax_data_l[0], tx_l[0]);
   nalu_hypre_SysPFMGRelaxSetup(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
   if (num_levels > 1)
   {
      for (l = 1; l < num_levels; l++)
      {
         /* set relaxation parameters */
         relax_data_l[l] = nalu_hypre_SysPFMGRelaxCreate(comm);
         nalu_hypre_SysPFMGRelaxSetTol(relax_data_l[l], 0.0);
         if (usr_jacobi_weight)
         {
            nalu_hypre_SysPFMGRelaxSetJacobiWeight(relax_data_l[l], jacobi_weight);
         }
         else
         {
            nalu_hypre_SysPFMGRelaxSetJacobiWeight(relax_data_l[l], relax_weights[l]);
         }
         nalu_hypre_SysPFMGRelaxSetType(relax_data_l[l], relax_type);
         nalu_hypre_SysPFMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
      }

      /* change coarsest grid relaxation parameters */
      l = num_levels - 1;
      {
         NALU_HYPRE_Int maxwork, maxiter;
         nalu_hypre_SysPFMGRelaxSetType(relax_data_l[l], 0);
         /* do no more work on the coarsest grid than the cost of a V-cycle
          * (estimating roughly 4 communications per V-cycle level) */
         maxwork = 4 * num_levels;
         /* do sweeps proportional to the coarsest grid size */
         maxiter = nalu_hypre_min(maxwork, cmaxsize);
#if 0
         nalu_hypre_printf("maxwork = %d, cmaxsize = %d, maxiter = %d\n",
                      maxwork, cmaxsize, maxiter);
#endif
         nalu_hypre_SysPFMGRelaxSetMaxIter(relax_data_l[l], maxiter);
      }

      /* call relax setup */
      for (l = 1; l < num_levels; l++)
      {
         nalu_hypre_SysPFMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
      }
   }
   nalu_hypre_TFree(relax_weights, NALU_HYPRE_MEMORY_HOST);

   for (l = 0; l < num_levels; l++)
   {
      /* set up the residual routine */
      nalu_hypre_SStructPMatvecCreate(&matvec_data_l[l]);
      nalu_hypre_SStructPMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
   }

   (sys_pfmg_data -> relax_data_l)    = relax_data_l;
   (sys_pfmg_data -> matvec_data_l)   = matvec_data_l;
   (sys_pfmg_data -> restrict_data_l) = restrict_data_l;
   (sys_pfmg_data -> interp_data_l)   = interp_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((sys_pfmg_data -> logging) > 0)
   {
      max_iter = (sys_pfmg_data -> max_iter);
      (sys_pfmg_data -> norms)     = nalu_hypre_TAlloc(NALU_HYPRE_Real, max_iter, NALU_HYPRE_MEMORY_HOST);
      (sys_pfmg_data -> rel_norms) = nalu_hypre_TAlloc(NALU_HYPRE_Real, max_iter, NALU_HYPRE_MEMORY_HOST);
   }

#if DEBUG
   for (l = 0; l < (num_levels - 1); l++)
   {
      nalu_hypre_sprintf(filename, "zout_A.%02d", l);
      nalu_hypre_SStructPMatrixPrint(filename, A_l[l], 0);
      nalu_hypre_sprintf(filename, "zout_P.%02d", l);
      nalu_hypre_SStructPMatrixPrint(filename, P_l[l], 0);
   }
   nalu_hypre_sprintf(filename, "zout_A.%02d", l);
   nalu_hypre_SStructPMatrixPrint(filename, A_l[l], 0);
#endif

   /*-----------------------------------------------------
    * Destroy Refs to A,x,b (the PMatrix & PVectors within
    * the input SStructMatrix & SStructVectors).
    *-----------------------------------------------------*/
   nalu_hypre_SStructPMatrixDestroy(A);
   nalu_hypre_SStructPVectorDestroy(x);
   nalu_hypre_SStructPVectorDestroy(b);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysStructCoarsen( nalu_hypre_SStructPGrid  *fgrid,
                        nalu_hypre_Index          index,
                        nalu_hypre_Index          stride,
                        NALU_HYPRE_Int            prune,
                        nalu_hypre_SStructPGrid **cgrid_ptr )
{
   nalu_hypre_SStructPGrid   *cgrid;

   nalu_hypre_StructGrid     *sfgrid;
   nalu_hypre_StructGrid     *scgrid;

   MPI_Comm               comm;
   NALU_HYPRE_Int              ndim;
   NALU_HYPRE_Int              nvars;
   nalu_hypre_SStructVariable *vartypes;
   nalu_hypre_SStructVariable *new_vartypes;
   NALU_HYPRE_Int              i;
   NALU_HYPRE_Int              t;

   /*-----------------------------------------
    * Copy information from fine grid
    *-----------------------------------------*/

   comm      = nalu_hypre_SStructPGridComm(fgrid);
   ndim      = nalu_hypre_SStructPGridNDim(fgrid);
   nvars     = nalu_hypre_SStructPGridNVars(fgrid);
   vartypes  = nalu_hypre_SStructPGridVarTypes(fgrid);

   cgrid = nalu_hypre_TAlloc(nalu_hypre_SStructPGrid, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructPGridComm(cgrid)     = comm;
   nalu_hypre_SStructPGridNDim(cgrid)     = ndim;
   nalu_hypre_SStructPGridNVars(cgrid)    = nvars;
   new_vartypes = nalu_hypre_TAlloc(nalu_hypre_SStructVariable, nvars, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nvars; i++)
   {
      new_vartypes[i] = vartypes[i];
   }
   nalu_hypre_SStructPGridVarTypes(cgrid) = new_vartypes;

   for (t = 0; t < 8; t++)
   {
      nalu_hypre_SStructPGridVTSGrid(cgrid, t)     = NULL;
      nalu_hypre_SStructPGridVTIBoxArray(cgrid, t) = NULL;
   }

   /*-----------------------------------------
    * Set the coarse sgrid
    *-----------------------------------------*/

   sfgrid = nalu_hypre_SStructPGridCellSGrid(fgrid);
   nalu_hypre_StructCoarsen(sfgrid, index, stride, prune, &scgrid);

   nalu_hypre_CopyIndex(nalu_hypre_StructGridPeriodic(scgrid),
                   nalu_hypre_SStructPGridPeriodic(cgrid));

   nalu_hypre_SStructPGridSetCellSGrid(cgrid, scgrid);

   nalu_hypre_SStructPGridPNeighbors(cgrid) = nalu_hypre_BoxArrayCreate(0, ndim);
   nalu_hypre_SStructPGridPNborOffsets(cgrid) = NULL;

   nalu_hypre_SStructPGridLocalSize(cgrid)  = 0;
   nalu_hypre_SStructPGridGlobalSize(cgrid) = 0;
   nalu_hypre_SStructPGridGhlocalSize(cgrid) = 0;

   nalu_hypre_SStructPGridAssemble(cgrid);

   *cgrid_ptr = cgrid;

   return nalu_hypre_error_flag;
}

