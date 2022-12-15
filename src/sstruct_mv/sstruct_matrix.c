/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_SStructPMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

/*==========================================================================
 * SStructPMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatrixRef( nalu_hypre_SStructPMatrix  *matrix,
                         nalu_hypre_SStructPMatrix **matrix_ref )
{
   nalu_hypre_SStructPMatrixRefCount(matrix) ++;
   *matrix_ref = matrix;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatrixCreate( MPI_Comm               comm,
                            nalu_hypre_SStructPGrid    *pgrid,
                            nalu_hypre_SStructStencil **stencils,
                            nalu_hypre_SStructPMatrix **pmatrix_ptr )
{
   nalu_hypre_SStructPMatrix  *pmatrix;
   NALU_HYPRE_Int              nvars;
   NALU_HYPRE_Int            **smaps;
   nalu_hypre_StructStencil ***sstencils;
   nalu_hypre_StructMatrix  ***smatrices;
   NALU_HYPRE_Int            **symmetric;

   nalu_hypre_StructStencil   *sstencil;
   NALU_HYPRE_Int             *vars;
   nalu_hypre_Index           *sstencil_shape;
   NALU_HYPRE_Int              sstencil_size;
   NALU_HYPRE_Int              new_dim;
   NALU_HYPRE_Int             *new_sizes;
   nalu_hypre_Index          **new_shapes;
   NALU_HYPRE_Int              size;
   nalu_hypre_StructGrid      *sgrid;

   NALU_HYPRE_Int              vi, vj;
   NALU_HYPRE_Int              i, j, k;

   pmatrix = nalu_hypre_TAlloc(nalu_hypre_SStructPMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructPMatrixComm(pmatrix)     = comm;
   nalu_hypre_SStructPMatrixPGrid(pmatrix)    = pgrid;
   nalu_hypre_SStructPMatrixStencils(pmatrix) = stencils;
   nvars = nalu_hypre_SStructPGridNVars(pgrid);
   nalu_hypre_SStructPMatrixNVars(pmatrix) = nvars;

   /* create sstencils */
   smaps     = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nvars, NALU_HYPRE_MEMORY_HOST);
   sstencils = nalu_hypre_TAlloc(nalu_hypre_StructStencil **,  nvars, NALU_HYPRE_MEMORY_HOST);
   new_sizes  = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nvars, NALU_HYPRE_MEMORY_HOST);
   new_shapes = nalu_hypre_TAlloc(nalu_hypre_Index *,  nvars, NALU_HYPRE_MEMORY_HOST);
   size = 0;
   for (vi = 0; vi < nvars; vi++)
   {
      sstencils[vi] = nalu_hypre_TAlloc(nalu_hypre_StructStencil *,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         sstencils[vi][vj] = NULL;
         new_sizes[vj] = 0;
      }

      sstencil       = nalu_hypre_SStructStencilSStencil(stencils[vi]);
      vars           = nalu_hypre_SStructStencilVars(stencils[vi]);
      sstencil_shape = nalu_hypre_StructStencilShape(sstencil);
      sstencil_size  = nalu_hypre_StructStencilSize(sstencil);

      smaps[vi] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  sstencil_size, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < sstencil_size; i++)
      {
         j = vars[i];
         new_sizes[j]++;
      }
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            new_shapes[vj] = nalu_hypre_TAlloc(nalu_hypre_Index,  new_sizes[vj], NALU_HYPRE_MEMORY_HOST);
            new_sizes[vj] = 0;
         }
      }
      for (i = 0; i < sstencil_size; i++)
      {
         j = vars[i];
         k = new_sizes[j];
         nalu_hypre_CopyIndex(sstencil_shape[i], new_shapes[j][k]);
         smaps[vi][i] = k;
         new_sizes[j]++;
      }
      new_dim = nalu_hypre_StructStencilNDim(sstencil);
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            sstencils[vi][vj] =
               nalu_hypre_StructStencilCreate(new_dim, new_sizes[vj], new_shapes[vj]);
         }
         size = nalu_hypre_max(size, new_sizes[vj]);
      }
   }
   nalu_hypre_SStructPMatrixSMaps(pmatrix)     = smaps;
   nalu_hypre_SStructPMatrixSStencils(pmatrix) = sstencils;
   nalu_hypre_TFree(new_sizes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(new_shapes, NALU_HYPRE_MEMORY_HOST);

   /* create smatrices */
   smatrices = nalu_hypre_TAlloc(nalu_hypre_StructMatrix **,  nvars, NALU_HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      smatrices[vi] = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         smatrices[vi][vj] = NULL;
         if (sstencils[vi][vj] != NULL)
         {
            sgrid = nalu_hypre_SStructPGridSGrid(pgrid, vi);
            smatrices[vi][vj] =
               nalu_hypre_StructMatrixCreate(comm, sgrid, sstencils[vi][vj]);
         }
      }
   }
   nalu_hypre_SStructPMatrixSMatrices(pmatrix) = smatrices;

   /* create symmetric */
   symmetric = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nvars, NALU_HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      symmetric[vi] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         symmetric[vi][vj] = 0;
      }
   }
   nalu_hypre_SStructPMatrixSymmetric(pmatrix) = symmetric;

   nalu_hypre_SStructPMatrixSEntriesSize(pmatrix) = size;
   nalu_hypre_SStructPMatrixSEntries(pmatrix) = nalu_hypre_TAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructPMatrixRefCount(pmatrix) = 1;

   *pmatrix_ptr = pmatrix;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatrixDestroy( nalu_hypre_SStructPMatrix *pmatrix )
{
   nalu_hypre_SStructStencil  **stencils;
   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int             **smaps;
   nalu_hypre_StructStencil  ***sstencils;
   nalu_hypre_StructMatrix   ***smatrices;
   NALU_HYPRE_Int             **symmetric;
   NALU_HYPRE_Int               vi, vj;

   if (pmatrix)
   {
      nalu_hypre_SStructPMatrixRefCount(pmatrix) --;
      if (nalu_hypre_SStructPMatrixRefCount(pmatrix) == 0)
      {
         stencils  = nalu_hypre_SStructPMatrixStencils(pmatrix);
         nvars     = nalu_hypre_SStructPMatrixNVars(pmatrix);
         smaps     = nalu_hypre_SStructPMatrixSMaps(pmatrix);
         sstencils = nalu_hypre_SStructPMatrixSStencils(pmatrix);
         smatrices = nalu_hypre_SStructPMatrixSMatrices(pmatrix);
         symmetric = nalu_hypre_SStructPMatrixSymmetric(pmatrix);
         for (vi = 0; vi < nvars; vi++)
         {
            NALU_HYPRE_SStructStencilDestroy(stencils[vi]);
            nalu_hypre_TFree(smaps[vi], NALU_HYPRE_MEMORY_HOST);
            for (vj = 0; vj < nvars; vj++)
            {
               nalu_hypre_StructStencilDestroy(sstencils[vi][vj]);
               nalu_hypre_StructMatrixDestroy(smatrices[vi][vj]);
            }
            nalu_hypre_TFree(sstencils[vi], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(smatrices[vi], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(symmetric[vi], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(stencils, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smaps, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sstencils, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smatrices, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(symmetric, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_SStructPMatrixSEntries(pmatrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pmatrix, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_SStructPMatrixInitialize( nalu_hypre_SStructPMatrix *pmatrix )
{
   NALU_HYPRE_Int             nvars        = nalu_hypre_SStructPMatrixNVars(pmatrix);
   NALU_HYPRE_Int           **symmetric    = nalu_hypre_SStructPMatrixSymmetric(pmatrix);
   nalu_hypre_StructMatrix   *smatrix;
   NALU_HYPRE_Int             vi, vj;
   /* NALU_HYPRE_Int             num_ghost[2*NALU_HYPRE_MAXDIM]; */
   /* NALU_HYPRE_Int             vi, vj, d, ndim; */

#if 0
   ndim = nalu_hypre_SStructPMatrixNDim(pmatrix);
   /* RDF: Why are the ghosts being reset to one? Maybe it needs to be at least
    * one to set shared coefficients correctly, but not exactly one? */
   for (d = 0; d < ndim; d++)
   {
      num_ghost[2 * d] = num_ghost[2 * d + 1] = 1;
   }
#endif
   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            NALU_HYPRE_StructMatrixSetSymmetric(smatrix, symmetric[vi][vj]);
            /* nalu_hypre_StructMatrixSetNumGhost(smatrix, num_ghost); */
            nalu_hypre_StructMatrixInitialize(smatrix);
            /* needed to get AddTo accumulation correct between processors */
            nalu_hypre_StructMatrixClearGhostValues(smatrix);
         }
      }
   }

   nalu_hypre_SStructPMatrixAccumulated(pmatrix) = 0;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatrixSetValues( nalu_hypre_SStructPMatrix *pmatrix,
                               nalu_hypre_Index           index,
                               NALU_HYPRE_Int             var,
                               NALU_HYPRE_Int             nentries,
                               NALU_HYPRE_Int            *entries,
                               NALU_HYPRE_Complex        *values,
                               NALU_HYPRE_Int             action )
{
   nalu_hypre_SStructStencil *stencil = nalu_hypre_SStructPMatrixStencil(pmatrix, var);
   NALU_HYPRE_Int            *smap    = nalu_hypre_SStructPMatrixSMap(pmatrix, var);
   NALU_HYPRE_Int            *vars    = nalu_hypre_SStructStencilVars(stencil);
   nalu_hypre_StructMatrix   *smatrix;
   nalu_hypre_BoxArray       *grid_boxes;
   nalu_hypre_Box            *box, *grow_box;
   NALU_HYPRE_Int            *sentries;
   NALU_HYPRE_Int             i;

   smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entries[0]]);

   sentries = nalu_hypre_SStructPMatrixSEntries(pmatrix);
   for (i = 0; i < nentries; i++)
   {
      sentries[i] = smap[entries[i]];
   }

   /* set values inside the grid */
   nalu_hypre_StructMatrixSetValues(smatrix, index, nentries, sentries, values,
                               action, -1, 0);

   /* set (AddTo/Get) or clear (Set) values outside the grid in ghost zones */
   if (action != 0)
   {
      /* AddTo/Get */
      nalu_hypre_SStructPGrid  *pgrid = nalu_hypre_SStructPMatrixPGrid(pmatrix);
      nalu_hypre_Index          varoffset;
      NALU_HYPRE_Int            done = 0;

      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(smatrix));

      nalu_hypre_ForBoxI(i, grid_boxes)
      {
         box = nalu_hypre_BoxArrayBox(grid_boxes, i);
         if (nalu_hypre_IndexInBox(index, box))
         {
            done = 1;
            break;
         }
      }

      if (!done)
      {
         grow_box = nalu_hypre_BoxCreate(nalu_hypre_BoxArrayNDim(grid_boxes));
         nalu_hypre_SStructVariableGetOffset(nalu_hypre_SStructPGridVarType(pgrid, var),
                                        nalu_hypre_SStructPGridNDim(pgrid), varoffset);
         nalu_hypre_ForBoxI(i, grid_boxes)
         {
            box = nalu_hypre_BoxArrayBox(grid_boxes, i);
            nalu_hypre_CopyBox(box, grow_box);
            nalu_hypre_BoxGrowByIndex(grow_box, varoffset);
            if (nalu_hypre_IndexInBox(index, grow_box))
            {
               nalu_hypre_StructMatrixSetValues(smatrix, index, nentries, sentries,
                                           values, action, i, 1);
               break;
            }
         }
         nalu_hypre_BoxDestroy(grow_box);
      }
   }
   else
   {
      /* Set */
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(smatrix));

      nalu_hypre_ForBoxI(i, grid_boxes)
      {
         box = nalu_hypre_BoxArrayBox(grid_boxes, i);
         if (!nalu_hypre_IndexInBox(index, box))
         {
            nalu_hypre_StructMatrixClearValues(smatrix, index, nentries, sentries, i, 1);
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatrixSetBoxValues( nalu_hypre_SStructPMatrix *pmatrix,
                                  nalu_hypre_Box            *set_box,
                                  NALU_HYPRE_Int             var,
                                  NALU_HYPRE_Int             nentries,
                                  NALU_HYPRE_Int            *entries,
                                  nalu_hypre_Box            *value_box,
                                  NALU_HYPRE_Complex        *values,
                                  NALU_HYPRE_Int             action )
{
   NALU_HYPRE_Int             ndim    = nalu_hypre_SStructPMatrixNDim(pmatrix);
   nalu_hypre_SStructStencil *stencil = nalu_hypre_SStructPMatrixStencil(pmatrix, var);
   NALU_HYPRE_Int            *smap    = nalu_hypre_SStructPMatrixSMap(pmatrix, var);
   NALU_HYPRE_Int            *vars    = nalu_hypre_SStructStencilVars(stencil);
   nalu_hypre_StructMatrix   *smatrix;
   nalu_hypre_BoxArray       *grid_boxes;
   NALU_HYPRE_Int            *sentries;
   NALU_HYPRE_Int             i, j;

   smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entries[0]]);

   sentries = nalu_hypre_SStructPMatrixSEntries(pmatrix);
   for (i = 0; i < nentries; i++)
   {
      sentries[i] = smap[entries[i]];
   }

   /* set values inside the grid */
   nalu_hypre_StructMatrixSetBoxValues(smatrix, set_box, value_box, nentries, sentries,
                                  values, action, -1, 0);
   /* TODO: Why need DeviceSync? */
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#endif
   /* set (AddTo/Get) or clear (Set) values outside the grid in ghost zones */
   if (action != 0)
   {
      /* AddTo/Get */
      nalu_hypre_SStructPGrid  *pgrid = nalu_hypre_SStructPMatrixPGrid(pmatrix);
      nalu_hypre_Index          varoffset;
      nalu_hypre_BoxArray      *left_boxes, *done_boxes, *temp_boxes;
      nalu_hypre_Box           *left_box, *done_box, *int_box;

      nalu_hypre_SStructVariableGetOffset(nalu_hypre_SStructPGridVarType(pgrid, var),
                                     nalu_hypre_SStructPGridNDim(pgrid), varoffset);
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(smatrix));

      left_boxes = nalu_hypre_BoxArrayCreate(1, ndim);
      done_boxes = nalu_hypre_BoxArrayCreate(2, ndim);
      temp_boxes = nalu_hypre_BoxArrayCreate(0, ndim);

      /* done_box always points to the first box in done_boxes */
      done_box = nalu_hypre_BoxArrayBox(done_boxes, 0);
      /* int_box always points to the second box in done_boxes */
      int_box = nalu_hypre_BoxArrayBox(done_boxes, 1);

      nalu_hypre_CopyBox(set_box, nalu_hypre_BoxArrayBox(left_boxes, 0));
      nalu_hypre_BoxArraySetSize(left_boxes, 1);
      nalu_hypre_SubtractBoxArrays(left_boxes, grid_boxes, temp_boxes);

      nalu_hypre_BoxArraySetSize(done_boxes, 0);
      nalu_hypre_ForBoxI(i, grid_boxes)
      {
         nalu_hypre_SubtractBoxArrays(left_boxes, done_boxes, temp_boxes);
         nalu_hypre_BoxArraySetSize(done_boxes, 1);
         nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(grid_boxes, i), done_box);
         nalu_hypre_BoxGrowByIndex(done_box, varoffset);
         nalu_hypre_ForBoxI(j, left_boxes)
         {
            left_box = nalu_hypre_BoxArrayBox(left_boxes, j);
            nalu_hypre_IntersectBoxes(left_box, done_box, int_box);
            nalu_hypre_StructMatrixSetBoxValues(smatrix, int_box, value_box,
                                           nentries, sentries,
                                           values, action, i, 1);
         }
      }

      nalu_hypre_BoxArrayDestroy(left_boxes);
      nalu_hypre_BoxArrayDestroy(done_boxes);
      nalu_hypre_BoxArrayDestroy(temp_boxes);
   }
   else
   {
      /* Set */
      nalu_hypre_BoxArray  *diff_boxes;
      nalu_hypre_Box       *grid_box, *diff_box;

      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(smatrix));
      diff_boxes = nalu_hypre_BoxArrayCreate(0, ndim);

      nalu_hypre_ForBoxI(i, grid_boxes)
      {
         grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);
         nalu_hypre_BoxArraySetSize(diff_boxes, 0);
         nalu_hypre_SubtractBoxes(set_box, grid_box, diff_boxes);

         nalu_hypre_ForBoxI(j, diff_boxes)
         {
            diff_box = nalu_hypre_BoxArrayBox(diff_boxes, j);
            nalu_hypre_StructMatrixClearBoxValues(smatrix, diff_box, nentries, sentries,
                                             i, 1);
         }
      }
      nalu_hypre_BoxArrayDestroy(diff_boxes);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatrixAccumulate( nalu_hypre_SStructPMatrix *pmatrix )
{
   nalu_hypre_SStructPGrid    *pgrid    = nalu_hypre_SStructPMatrixPGrid(pmatrix);
   NALU_HYPRE_Int              nvars    = nalu_hypre_SStructPMatrixNVars(pmatrix);
   NALU_HYPRE_Int              ndim     = nalu_hypre_SStructPGridNDim(pgrid);
   NALU_HYPRE_SStructVariable *vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);

   nalu_hypre_StructMatrix    *smatrix;
   nalu_hypre_Index            varoffset;
   NALU_HYPRE_Int              num_ghost[2 * NALU_HYPRE_MAXDIM];
   nalu_hypre_StructGrid      *sgrid;
   NALU_HYPRE_Int              vi, vj, d;

   nalu_hypre_CommInfo        *comm_info;
   nalu_hypre_CommPkg         *comm_pkg;
   nalu_hypre_CommHandle      *comm_handle;

   /* if values already accumulated, just return */
   if (nalu_hypre_SStructPMatrixAccumulated(pmatrix))
   {
      return nalu_hypre_error_flag;
   }

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            sgrid = nalu_hypre_StructMatrixGrid(smatrix);
            /* assumes vi and vj vartypes are the same */
            nalu_hypre_SStructVariableGetOffset(vartypes[vi], ndim, varoffset);
            for (d = 0; d < ndim; d++)
            {
               num_ghost[2 * d]   = num_ghost[2 * d + 1] = nalu_hypre_IndexD(varoffset, d);
            }

            /* accumulate values from AddTo */
            nalu_hypre_CreateCommInfoFromNumGhost(sgrid, num_ghost, &comm_info);
            nalu_hypre_CommPkgCreate(comm_info,
                                nalu_hypre_StructMatrixDataSpace(smatrix),
                                nalu_hypre_StructMatrixDataSpace(smatrix),
                                nalu_hypre_StructMatrixNumValues(smatrix), NULL, 1,
                                nalu_hypre_StructMatrixComm(smatrix),
                                &comm_pkg);
            nalu_hypre_InitializeCommunication(comm_pkg,
                                          nalu_hypre_StructMatrixData(smatrix),
                                          nalu_hypre_StructMatrixData(smatrix),
                                          1, 0, &comm_handle);
            nalu_hypre_FinalizeCommunication(comm_handle);

            nalu_hypre_CommInfoDestroy(comm_info);
            nalu_hypre_CommPkgDestroy(comm_pkg);
         }
      }
   }

   nalu_hypre_SStructPMatrixAccumulated(pmatrix) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatrixAssemble( nalu_hypre_SStructPMatrix *pmatrix )
{
   NALU_HYPRE_Int              nvars    = nalu_hypre_SStructPMatrixNVars(pmatrix);
   nalu_hypre_StructMatrix    *smatrix;
   NALU_HYPRE_Int              vi, vj;

   nalu_hypre_SStructPMatrixAccumulate(pmatrix);

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            nalu_hypre_StructMatrixClearGhostValues(smatrix);
            nalu_hypre_StructMatrixAssemble(smatrix);
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatrixSetSymmetric( nalu_hypre_SStructPMatrix *pmatrix,
                                  NALU_HYPRE_Int             var,
                                  NALU_HYPRE_Int             to_var,
                                  NALU_HYPRE_Int             symmetric )
{
   NALU_HYPRE_Int **pmsymmetric = nalu_hypre_SStructPMatrixSymmetric(pmatrix);

   NALU_HYPRE_Int vstart = var;
   NALU_HYPRE_Int vsize  = 1;
   NALU_HYPRE_Int tstart = to_var;
   NALU_HYPRE_Int tsize  = 1;
   NALU_HYPRE_Int v, t;

   if (var == -1)
   {
      vstart = 0;
      vsize  = nalu_hypre_SStructPMatrixNVars(pmatrix);
   }
   if (to_var == -1)
   {
      tstart = 0;
      tsize  = nalu_hypre_SStructPMatrixNVars(pmatrix);
   }

   for (v = vstart; v < vsize; v++)
   {
      for (t = tstart; t < tsize; t++)
      {
         pmsymmetric[v][t] = symmetric;
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatrixPrint( const char           *filename,
                           nalu_hypre_SStructPMatrix *pmatrix,
                           NALU_HYPRE_Int             all )
{
   NALU_HYPRE_Int           nvars = nalu_hypre_SStructPMatrixNVars(pmatrix);
   nalu_hypre_StructMatrix *smatrix;
   NALU_HYPRE_Int           vi, vj;
   char                new_filename[255];

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            nalu_hypre_sprintf(new_filename, "%s.%02d.%02d", filename, vi, vj);
            nalu_hypre_StructMatrixPrint(new_filename, smatrix, all);
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*==========================================================================
 * SStructUMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructUMatrixInitialize( nalu_hypre_SStructMatrix *matrix )
{
   NALU_HYPRE_Int               ndim        = nalu_hypre_SStructMatrixNDim(matrix);
   NALU_HYPRE_IJMatrix          ijmatrix    = nalu_hypre_SStructMatrixIJMatrix(matrix);
   NALU_HYPRE_Int               matrix_type = nalu_hypre_SStructMatrixObjectType(matrix);
   nalu_hypre_SStructGraph     *graph       = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid      *grid        = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int               nparts      = nalu_hypre_SStructGraphNParts(graph);
   nalu_hypre_SStructPGrid    **pgrids      = nalu_hypre_SStructGraphPGrids(graph);
   nalu_hypre_SStructStencil ***stencils    = nalu_hypre_SStructGraphStencils(graph);
   NALU_HYPRE_Int               nUventries  = nalu_hypre_SStructGraphNUVEntries(graph);
   NALU_HYPRE_Int              *iUventries  = nalu_hypre_SStructGraphIUVEntries(graph);
   nalu_hypre_SStructUVEntry  **Uventries   = nalu_hypre_SStructGraphUVEntries(graph);
   NALU_HYPRE_Int             **nvneighbors = nalu_hypre_SStructGridNVNeighbors(grid);
   nalu_hypre_StructGrid       *sgrid;
   nalu_hypre_SStructStencil   *stencil;
   NALU_HYPRE_Int              *split;
   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int               nrows, rowstart, nnzs ;
   NALU_HYPRE_Int               part, var, entry, b, m, mi;
   NALU_HYPRE_Int              *row_sizes;
   NALU_HYPRE_Int               max_row_size;

   nalu_hypre_BoxArray         *boxes;
   nalu_hypre_Box              *box;
   nalu_hypre_Box              *ghost_box;
   nalu_hypre_IndexRef          start;
   nalu_hypre_Index             loop_size, stride;

   NALU_HYPRE_IJMatrixSetObjectType(ijmatrix, NALU_HYPRE_PARCSR);

#ifdef NALU_HYPRE_USING_OPENMP
   NALU_HYPRE_IJMatrixSetOMPFlag(ijmatrix, 1); /* Use OpenMP */
#endif

   if (matrix_type == NALU_HYPRE_SSTRUCT || matrix_type == NALU_HYPRE_STRUCT)
   {
      rowstart = nalu_hypre_SStructGridGhstartRank(grid);
      nrows = nalu_hypre_SStructGridGhlocalSize(grid) ;
   }
   else /* matrix_type == NALU_HYPRE_PARCSR */
   {
      rowstart = nalu_hypre_SStructGridStartRank(grid);
      nrows = nalu_hypre_SStructGridLocalSize(grid);
   }

   /* set row sizes */
   m = 0;
   max_row_size = 0;
   ghost_box = nalu_hypre_BoxCreate(ndim);
   row_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SetIndex(stride, 1);
   for (part = 0; part < nparts; part++)
   {
      nvars = nalu_hypre_SStructPGridNVars(pgrids[part]);
      for (var = 0; var < nvars; var++)
      {
         sgrid = nalu_hypre_SStructPGridSGrid(pgrids[part], var);

         stencil = stencils[part][var];
         split = nalu_hypre_SStructMatrixSplit(matrix, part, var);
         nnzs = 0;
         for (entry = 0; entry < nalu_hypre_SStructStencilSize(stencil); entry++)
         {
            if (split[entry] == -1)
            {
               nnzs++;
            }
         }
#if 0
         /* TODO: For now, assume stencil is full/complete */
         if (nalu_hypre_SStructMatrixSymmetric(matrix))
         {
            nnzs = 2 * nnzs - 1;
         }
#endif
         boxes = nalu_hypre_StructGridBoxes(sgrid);
         nalu_hypre_ForBoxI(b, boxes)
         {
            box = nalu_hypre_BoxArrayBox(boxes, b);
            nalu_hypre_CopyBox(box, ghost_box);
            if (matrix_type == NALU_HYPRE_SSTRUCT || matrix_type == NALU_HYPRE_STRUCT)
            {
               nalu_hypre_BoxGrowByArray(ghost_box, nalu_hypre_StructGridNumGhost(sgrid));
            }
            start = nalu_hypre_BoxIMin(box);
            nalu_hypre_BoxGetSize(box, loop_size);
            zypre_BoxLoop1Begin(nalu_hypre_SStructMatrixNDim(matrix), loop_size,
                                ghost_box, start, stride, mi);
            {
               row_sizes[m + mi] = nnzs;
            }
            zypre_BoxLoop1End(mi);

            m += nalu_hypre_BoxVolume(ghost_box);
         }

         max_row_size = nalu_hypre_max(max_row_size, nnzs);
         if (nvneighbors[part][var])
         {
            max_row_size =
               nalu_hypre_max(max_row_size, nalu_hypre_SStructStencilSize(stencil));
         }
      }
   }
   nalu_hypre_BoxDestroy(ghost_box);

   /* GEC0902 essentially for each UVentry we figure out how many extra columns
    * we need to add to the rowsizes                                   */

   /* RDF: THREAD? */
   for (entry = 0; entry < nUventries; entry++)
   {
      mi = iUventries[entry];
      m = nalu_hypre_SStructUVEntryRank(Uventries[mi]) - rowstart;
      if ((m > -1) && (m < nrows))
      {
         row_sizes[m] += nalu_hypre_SStructUVEntryNUEntries(Uventries[mi]);
         max_row_size = nalu_hypre_max(max_row_size, row_sizes[m]);
      }
   }

   /* ZTODO: Update row_sizes based on neighbor off-part couplings */
   NALU_HYPRE_IJMatrixSetRowSizes (ijmatrix, (const NALU_HYPRE_Int *) row_sizes);

   nalu_hypre_TFree(row_sizes, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructMatrixTmpSize(matrix) = max_row_size;
   nalu_hypre_SStructMatrixTmpRowCoords(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, max_row_size,
                                                           NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructMatrixTmpColCoords(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, max_row_size,
                                                           NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructMatrixTmpCoeffs(matrix)    = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, max_row_size,
                                                           NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_IJMatrixInitialize(ijmatrix);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *
 * 9/09 - AB: modified to use the box manager - here we need to check the
 *            neighbor box manager also
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructUMatrixSetValues( nalu_hypre_SStructMatrix *matrix,
                               NALU_HYPRE_Int            part,
                               nalu_hypre_Index          index,
                               NALU_HYPRE_Int            var,
                               NALU_HYPRE_Int            nentries,
                               NALU_HYPRE_Int           *entries,
                               NALU_HYPRE_Complex       *values,
                               NALU_HYPRE_Int            action )
{
   NALU_HYPRE_Int                ndim     = nalu_hypre_SStructMatrixNDim(matrix);
   NALU_HYPRE_IJMatrix           ijmatrix = nalu_hypre_SStructMatrixIJMatrix(matrix);
   nalu_hypre_SStructGraph      *graph    = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid       *grid     = nalu_hypre_SStructGraphGrid(graph);
   nalu_hypre_SStructGrid       *dom_grid = nalu_hypre_SStructGraphDomainGrid(graph);
   nalu_hypre_SStructStencil    *stencil  = nalu_hypre_SStructGraphStencil(graph, part, var);
   NALU_HYPRE_Int               *vars     = nalu_hypre_SStructStencilVars(stencil);
   nalu_hypre_Index             *shape    = nalu_hypre_SStructStencilShape(stencil);
   NALU_HYPRE_Int                size     = nalu_hypre_SStructStencilSize(stencil);
   nalu_hypre_IndexRef           offset;
   nalu_hypre_Index              to_index;
   nalu_hypre_SStructUVEntry    *Uventry;
   nalu_hypre_BoxManEntry       *boxman_entry;
   nalu_hypre_SStructBoxManInfo *entry_info;
   NALU_HYPRE_BigInt             row_coord;
   NALU_HYPRE_BigInt            *col_coords;
   NALU_HYPRE_Int                ncoeffs;
   NALU_HYPRE_Complex           *coeffs;
   NALU_HYPRE_Int                i, entry;
   NALU_HYPRE_BigInt             Uverank;
   NALU_HYPRE_Int                matrix_type = nalu_hypre_SStructMatrixObjectType(matrix);
   NALU_HYPRE_Complex           *h_values;
   NALU_HYPRE_MemoryLocation     memory_location = nalu_hypre_IJMatrixMemoryLocation(ijmatrix);

   nalu_hypre_SStructGridFindBoxManEntry(grid, part, index, var, &boxman_entry);

   /* if not local, check neighbors */
   if (boxman_entry == NULL)
   {
      nalu_hypre_SStructGridFindNborBoxManEntry(grid, part, index, var, &boxman_entry);
   }

   if (boxman_entry == NULL)
   {
      nalu_hypre_error_in_arg(1);
      nalu_hypre_error_in_arg(2);
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   else
   {
      nalu_hypre_BoxManEntryGetInfo(boxman_entry, (void **) &entry_info);
   }

   nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index,
                                         &row_coord, matrix_type);

   col_coords = nalu_hypre_SStructMatrixTmpColCoords(matrix);
   coeffs = nalu_hypre_SStructMatrixTmpCoeffs(matrix);

   if ( nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE )
   {
      h_values = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nentries, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(h_values, values, NALU_HYPRE_Complex, nentries, NALU_HYPRE_MEMORY_HOST, memory_location);
   }
   else
   {
      h_values = values;
   }

   /* RL: TODO Port it to GPU? */
   ncoeffs = 0;
   for (i = 0; i < nentries; i++)
   {
      entry = entries[i];

      if (entry < size)
      {
         /* stencil entries */
         offset = shape[entry];
         nalu_hypre_AddIndexes(index, offset, ndim, to_index);

         nalu_hypre_SStructGridFindBoxManEntry(dom_grid, part, to_index, vars[entry],
                                          &boxman_entry);

         /* if not local, check neighbors */
         if (boxman_entry == NULL)
         {
            nalu_hypre_SStructGridFindNborBoxManEntry(dom_grid, part, to_index,
                                                 vars[entry], &boxman_entry);
         }

         if (boxman_entry != NULL)
         {
            nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, to_index,
                                                  &col_coords[ncoeffs], matrix_type);

            coeffs[ncoeffs] = h_values[i];
            ncoeffs++;
         }
      }
      else
      {
         /* non-stencil entries */
         entry -= size;
         nalu_hypre_SStructGraphGetUVEntryRank(graph, part, var, index, &Uverank);

         if (Uverank > -1)
         {
            Uventry = nalu_hypre_SStructGraphUVEntry(graph, Uverank);
            col_coords[ncoeffs] = nalu_hypre_SStructUVEntryToRank(Uventry, entry);
            coeffs[ncoeffs] = h_values[i];
            ncoeffs++;
         }
      }
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   if ( nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE )
   {
      if (!nalu_hypre_SStructMatrixTmpRowCoordsDevice(matrix))
      {
         nalu_hypre_SStructMatrixTmpRowCoordsDevice(matrix) =
            nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nalu_hypre_SStructMatrixTmpSize(matrix), memory_location);
      }

      if (!nalu_hypre_SStructMatrixTmpColCoordsDevice(matrix))
      {
         nalu_hypre_SStructMatrixTmpColCoordsDevice(matrix) =
            nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nalu_hypre_SStructMatrixTmpSize(matrix), memory_location);
      }

      if (!nalu_hypre_SStructMatrixTmpCoeffsDevice(matrix))
      {
         nalu_hypre_SStructMatrixTmpCoeffsDevice(matrix) =
            nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nalu_hypre_SStructMatrixTmpSize(matrix), memory_location);
      }

      hypreDevice_BigIntFilln(nalu_hypre_SStructMatrixTmpRowCoordsDevice(matrix), ncoeffs, row_coord);

      nalu_hypre_TMemcpy(nalu_hypre_SStructMatrixTmpColCoordsDevice(matrix), col_coords, NALU_HYPRE_BigInt, ncoeffs,
                    memory_location, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TMemcpy(nalu_hypre_SStructMatrixTmpCoeffsDevice(matrix), coeffs, NALU_HYPRE_Complex, ncoeffs,
                    memory_location, NALU_HYPRE_MEMORY_HOST);

      if (action > 0)
      {
         NALU_HYPRE_IJMatrixAddToValues(ijmatrix, ncoeffs, NULL, nalu_hypre_SStructMatrixTmpRowCoordsDevice(matrix),
                                   (const NALU_HYPRE_BigInt *) nalu_hypre_SStructMatrixTmpColCoordsDevice(matrix),
                                   (const NALU_HYPRE_Complex *) nalu_hypre_SStructMatrixTmpCoeffsDevice(matrix));
      }
      else if (action > -1)
      {
         NALU_HYPRE_IJMatrixSetValues(ijmatrix, ncoeffs, NULL, nalu_hypre_SStructMatrixTmpRowCoordsDevice(matrix),
                                 (const NALU_HYPRE_BigInt *) nalu_hypre_SStructMatrixTmpColCoordsDevice(matrix),
                                 (const NALU_HYPRE_Complex *) nalu_hypre_SStructMatrixTmpCoeffsDevice(matrix));
      }
      else
      {
         // RL:TODO
         NALU_HYPRE_IJMatrixGetValues(ijmatrix, 1, &ncoeffs, &row_coord, col_coords, values);
      }
   }
   else
#endif
   {
      if (action > 0)
      {
         NALU_HYPRE_IJMatrixAddToValues(ijmatrix, 1, &ncoeffs, &row_coord,
                                   (const NALU_HYPRE_BigInt *) col_coords,
                                   (const NALU_HYPRE_Complex *) coeffs);
      }
      else if (action > -1)
      {
         NALU_HYPRE_IJMatrixSetValues(ijmatrix, 1, &ncoeffs, &row_coord,
                                 (const NALU_HYPRE_BigInt *) col_coords,
                                 (const NALU_HYPRE_Complex *) coeffs);
      }
      else
      {
         NALU_HYPRE_IJMatrixGetValues(ijmatrix, 1, &ncoeffs, &row_coord,
                                 col_coords, values);
      }
   }

   if (h_values != values)
   {
      nalu_hypre_TFree(h_values, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note: Entries must all be of type stencil or non-stencil, but not both.
 *
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *
 * 9/09 - AB: modified to use the box manager- here we need to check the
 *            neighbor box manager also
 *
 * To illustrate what is computed below before calling IJSetValues2(), consider
 * the following example of a 5-pt stencil (c,w,e,s,n) on a 3x2 grid (the 'x' in
 * arrays 'cols' and 'ijvalues' indicates "no data"):
 *
 *   nrows       = 6
 *   ncols       = 3         4         3         3         4         3
 *   rows        = 0         1         2         3         4         5
 *   row_indexes = 0         5         10        15        20        25
 *   cols        = . . . x x . . . . x . . . x x . . . x x . . . . x . . . x x
 *   ijvalues    = . . . x x . . . . x . . . x x . . . x x . . . . x . . . x x
 *   entry       = c e n     c w e n   c w n     c e s     c w e s   c w s
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructUMatrixSetBoxValues( nalu_hypre_SStructMatrix *matrix,
                                  NALU_HYPRE_Int            part,
                                  nalu_hypre_Box           *set_box,
                                  NALU_HYPRE_Int            var,
                                  NALU_HYPRE_Int            nentries,
                                  NALU_HYPRE_Int           *entries,
                                  nalu_hypre_Box           *value_box,
                                  NALU_HYPRE_Complex       *values,
                                  NALU_HYPRE_Int            action )
{
   NALU_HYPRE_Int             ndim     = nalu_hypre_SStructMatrixNDim(matrix);
   NALU_HYPRE_IJMatrix        ijmatrix = nalu_hypre_SStructMatrixIJMatrix(matrix);
   nalu_hypre_SStructGraph   *graph    = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid    *grid     = nalu_hypre_SStructGraphGrid(graph);
   nalu_hypre_SStructGrid    *dom_grid = nalu_hypre_SStructGraphDomainGrid(graph);
   nalu_hypre_SStructStencil *stencil  = nalu_hypre_SStructGraphStencil(graph, part, var);
   NALU_HYPRE_Int            *vars     = nalu_hypre_SStructStencilVars(stencil);
   nalu_hypre_Index          *shape    = nalu_hypre_SStructStencilShape(stencil);
   NALU_HYPRE_Int             size     = nalu_hypre_SStructStencilSize(stencil);
   nalu_hypre_IndexRef        offset;
   nalu_hypre_BoxManEntry   **boxman_entries;
   NALU_HYPRE_Int             nboxman_entries;
   nalu_hypre_BoxManEntry   **boxman_to_entries;
   NALU_HYPRE_Int             nboxman_to_entries;
   NALU_HYPRE_Int             nrows;
   NALU_HYPRE_Int            *ncols, *row_indexes;;
   NALU_HYPRE_BigInt         *rows, *cols;
   NALU_HYPRE_Complex        *ijvalues;
   nalu_hypre_Box            *box = nalu_hypre_BoxCreate(ndim);
   nalu_hypre_Box            *to_box;
   nalu_hypre_Box            *map_box;
   nalu_hypre_Box            *int_box;
   nalu_hypre_Index           index, stride, loop_size;
   nalu_hypre_IndexRef        start;
   nalu_hypre_Index           rs, cs;
   NALU_HYPRE_BigInt          row_base, col_base;
   NALU_HYPRE_Int             ei, entry, ii, jj;
   NALU_HYPRE_Int             matrix_type = nalu_hypre_SStructMatrixObjectType(matrix);
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_IJMatrixMemoryLocation(ijmatrix);

   /*------------------------------------------
    * all stencil entries
    *------------------------------------------*/

   if (entries[0] < size)
   {
      to_box  = nalu_hypre_BoxCreate(ndim);
      map_box = nalu_hypre_BoxCreate(ndim);
      int_box = nalu_hypre_BoxCreate(ndim);

      nrows       = nalu_hypre_BoxVolume(set_box);
      ncols       = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     nrows,            memory_location);
      rows        = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nrows,            memory_location);
      row_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     nrows,            memory_location);
      cols        = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nrows * nentries, memory_location);
      ijvalues    = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nrows * nentries, memory_location);

      nalu_hypre_SetIndex(stride, 1);

      nalu_hypre_SStructGridIntersect(grid, part, var, set_box, -1,
                                 &boxman_entries, &nboxman_entries);

      for (ii = 0; ii < nboxman_entries; ii++)
      {
         nalu_hypre_SStructBoxManEntryGetStrides(boxman_entries[ii], rs, matrix_type);

         nalu_hypre_CopyBox(set_box, box);
         nalu_hypre_BoxManEntryGetExtents(boxman_entries[ii],
                                     nalu_hypre_BoxIMin(map_box), nalu_hypre_BoxIMax(map_box));
         nalu_hypre_IntersectBoxes(box, map_box, int_box);
         nalu_hypre_CopyBox(int_box, box);

         /* For each index in 'box', compute a row of length <= nentries and
          * insert it into an nentries-length segment of 'cols' and 'ijvalues'.
          * This may result in gaps, but IJSetValues2() is designed for that. */

         nrows = nalu_hypre_BoxVolume(box);

#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(ncols,row_indexes)
         nalu_hypre_LoopBegin(nrows, i)
         {
            ncols[i] = 0;
            row_indexes[i] = i * nentries;
         }
         nalu_hypre_LoopEnd()
#undef DEVICE_VAR
#define DEVICE_VAR

         for (ei = 0; ei < nentries; ei++)
         {
            entry = entries[ei];

            nalu_hypre_CopyBox(box, to_box);

            offset = shape[entry];
            nalu_hypre_BoxShiftPos(to_box, offset);

            nalu_hypre_SStructGridIntersect(dom_grid, part, vars[entry], to_box, -1,
                                       &boxman_to_entries, &nboxman_to_entries);

            for (jj = 0; jj < nboxman_to_entries; jj++)
            {
               nalu_hypre_SStructBoxManEntryGetStrides(boxman_to_entries[jj], cs, matrix_type);

               nalu_hypre_BoxManEntryGetExtents(boxman_to_entries[jj],
                                           nalu_hypre_BoxIMin(map_box), nalu_hypre_BoxIMax(map_box));
               nalu_hypre_IntersectBoxes(to_box, map_box, int_box);

               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(int_box), index);
               nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_to_entries[jj],
                                                     index, &col_base, matrix_type);

               nalu_hypre_BoxShiftNeg(int_box, offset);

               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(int_box), index);
               nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entries[ii],
                                                     index, &row_base, matrix_type);

               start = nalu_hypre_BoxIMin(int_box);
               nalu_hypre_BoxGetSize(int_box, loop_size);

#if defined(NALU_HYPRE_USING_GPU)
               if ( nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE )
               {
                  nalu_hypre_assert(ndim <= 3);

                  NALU_HYPRE_Int rs_0, rs_1, rs_2;
                  NALU_HYPRE_Int cs_0, cs_1, cs_2;

                  if (ndim > 0)
                  {
                     rs_0 = rs[0];
                     cs_0 = cs[0];
                  }

                  if (ndim > 1)
                  {
                     rs_1 = rs[1];
                     cs_1 = cs[1];
                  }

                  if (ndim > 2)
                  {
                     rs_2 = rs[2];
                     cs_2 = cs[2];
                  }

#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(ncols,rows,cols,ijvalues,values)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      box,       start, stride, mi,
                                      value_box, start, stride, vi);
                  {
                     nalu_hypre_Index index;
                     NALU_HYPRE_Int   ci;

                     nalu_hypre_BoxLoopGetIndex(index);

                     ci = mi * nentries + ncols[mi];
                     rows[mi] = row_base;
                     cols[ci] = col_base;

                     if (ndim > 0)
                     {
                        rows[mi] += index[0] * rs_0;
                        cols[ci] += index[0] * cs_0;
                     }

                     if (ndim > 1)
                     {
                        rows[mi] += index[1] * rs_1;
                        cols[ci] += index[1] * cs_1;
                     }

                     if (ndim > 2)
                     {
                        rows[mi] += index[2] * rs_2;
                        cols[ci] += index[2] * cs_2;
                     }

                     ijvalues[ci] = values[ei + vi * nentries];
                     ncols[mi]++;
                  }
                  nalu_hypre_BoxLoop2End(mi, vi);
#undef DEVICE_VAR
#define DEVICE_VAR
               }
               else
#endif
               {
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      box,       start, stride, mi,
                                      value_box, start, stride, vi);
                  {
                     nalu_hypre_Index index;
                     NALU_HYPRE_Int   ci;

                     nalu_hypre_BoxLoopGetIndex(index);

                     ci = mi * nentries + ncols[mi];
                     rows[mi] = row_base;
                     cols[ci] = col_base;

                     NALU_HYPRE_Int d;
                     for (d = 0; d < ndim; d++)
                     {
                        rows[mi] += index[d] * rs[d];
                        cols[ci] += index[d] * cs[d];
                     }

                     ijvalues[ci] = values[ei + vi * nentries];
                     ncols[mi]++;
                  }
                  nalu_hypre_BoxLoop2End(mi, vi);
               }
            } /* end loop through boxman to entries */

            nalu_hypre_TFree(boxman_to_entries, NALU_HYPRE_MEMORY_HOST);

         } /* end of ei nentries loop */

         if (action > 0)
         {
            NALU_HYPRE_IJMatrixAddToValues2(ijmatrix, nrows, ncols,
                                       (const NALU_HYPRE_BigInt *) rows,
                                       (const NALU_HYPRE_Int *) row_indexes,
                                       (const NALU_HYPRE_BigInt *) cols,
                                       (const NALU_HYPRE_Complex *) ijvalues);
         }
         else if (action > -1)
         {
            NALU_HYPRE_IJMatrixSetValues2(ijmatrix, nrows, ncols,
                                     (const NALU_HYPRE_BigInt *) rows,
                                     (const NALU_HYPRE_Int *) row_indexes,
                                     (const NALU_HYPRE_BigInt *) cols,
                                     (const NALU_HYPRE_Complex *) ijvalues);
         }
         else
         {
            NALU_HYPRE_IJMatrixGetValues(ijmatrix, nrows, ncols, rows, cols, values);
         }

      } /* end loop through boxman entries */

      nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(ncols, memory_location);
      nalu_hypre_TFree(rows, memory_location);
      nalu_hypre_TFree(row_indexes, memory_location);
      nalu_hypre_TFree(cols, memory_location);
      nalu_hypre_TFree(ijvalues, memory_location);

      nalu_hypre_BoxDestroy(to_box);
      nalu_hypre_BoxDestroy(map_box);
      nalu_hypre_BoxDestroy(int_box);
   }

   /*------------------------------------------
    * non-stencil entries
    *------------------------------------------*/

   else
   {
      /* RDF: THREAD (Check safety on UMatrixSetValues call) */
      nalu_hypre_BoxGetSize(set_box, loop_size);
      nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
      {
         zypre_BoxLoopGetIndex(index);
         nalu_hypre_AddIndexes(index, nalu_hypre_BoxIMin(set_box), ndim, index);
         nalu_hypre_SStructUMatrixSetValues(matrix, part, index, var,
                                       nentries, entries, values, action);
         values += nentries;
      }
      nalu_hypre_SerialBoxLoop0End();
   }

   nalu_hypre_BoxDestroy(box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructUMatrixAssemble( nalu_hypre_SStructMatrix *matrix )
{
   NALU_HYPRE_IJMatrix ijmatrix = nalu_hypre_SStructMatrixIJMatrix(matrix);

   NALU_HYPRE_IJMatrixAssemble(ijmatrix);
   NALU_HYPRE_IJMatrixGetObject(
      ijmatrix, (void **) &nalu_hypre_SStructMatrixParCSRMatrix(matrix));

   return nalu_hypre_error_flag;
}

/*==========================================================================
 * SStructMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatrixRef( nalu_hypre_SStructMatrix  *matrix,
                        nalu_hypre_SStructMatrix **matrix_ref )
{
   nalu_hypre_SStructMatrixRefCount(matrix) ++;
   *matrix_ref = matrix;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatrixSplitEntries( nalu_hypre_SStructMatrix *matrix,
                                 NALU_HYPRE_Int            part,
                                 NALU_HYPRE_Int            var,
                                 NALU_HYPRE_Int            nentries,
                                 NALU_HYPRE_Int           *entries,
                                 NALU_HYPRE_Int           *nSentries_ptr,
                                 NALU_HYPRE_Int          **Sentries_ptr,
                                 NALU_HYPRE_Int           *nUentries_ptr,
                                 NALU_HYPRE_Int          **Uentries_ptr )
{
   nalu_hypre_SStructGraph   *graph   = nalu_hypre_SStructMatrixGraph(matrix);
   NALU_HYPRE_Int            *split   = nalu_hypre_SStructMatrixSplit(matrix, part, var);
   nalu_hypre_SStructStencil *stencil = nalu_hypre_SStructGraphStencil(graph, part, var);
   NALU_HYPRE_Int             entry;
   NALU_HYPRE_Int             i;

   NALU_HYPRE_Int             nSentries = 0;
   NALU_HYPRE_Int            *Sentries  = nalu_hypre_SStructMatrixSEntries(matrix);
   NALU_HYPRE_Int             nUentries = 0;
   NALU_HYPRE_Int            *Uentries  = nalu_hypre_SStructMatrixUEntries(matrix);

   for (i = 0; i < nentries; i++)
   {
      entry = entries[i];
      if (entry < nalu_hypre_SStructStencilSize(stencil))
      {
         /* stencil entries */
         if (split[entry] > -1)
         {
            Sentries[nSentries] = split[entry];
            nSentries++;
         }
         else
         {
            Uentries[nUentries] = entry;
            nUentries++;
         }
      }
      else
      {
         /* non-stencil entries */
         Uentries[nUentries] = entry;
         nUentries++;
      }
   }

   *nSentries_ptr = nSentries;
   *Sentries_ptr  = Sentries;
   *nUentries_ptr = nUentries;
   *Uentries_ptr  = Uentries;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatrixSetValues( NALU_HYPRE_SStructMatrix  matrix,
                              NALU_HYPRE_Int            part,
                              NALU_HYPRE_Int           *index,
                              NALU_HYPRE_Int            var,
                              NALU_HYPRE_Int            nentries,
                              NALU_HYPRE_Int           *entries,
                              NALU_HYPRE_Complex       *values,
                              NALU_HYPRE_Int            action )
{
   NALU_HYPRE_Int             ndim  = nalu_hypre_SStructMatrixNDim(matrix);
   nalu_hypre_SStructGraph   *graph = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid    *grid  = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int           **nvneighbors = nalu_hypre_SStructGridNVNeighbors(grid);
   NALU_HYPRE_Int            *Sentries;
   NALU_HYPRE_Int            *Uentries;
   NALU_HYPRE_Int             nSentries;
   NALU_HYPRE_Int             nUentries;
   nalu_hypre_SStructPMatrix *pmatrix;
   nalu_hypre_Index           cindex;

   nalu_hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   nalu_hypre_CopyToCleanIndex(index, ndim, cindex);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, part);
      nalu_hypre_SStructPMatrixSetValues(pmatrix, cindex, var,
                                    nSentries, Sentries, values, action);
      /* put inter-part couplings in UMatrix and zero them out in PMatrix
       * (possibly in ghost zones) */
      if (nvneighbors[part][var] > 0)
      {
         nalu_hypre_Box  *set_box;
         NALU_HYPRE_Int   d;
         /* This creates boxes with zeroed-out extents */
         set_box = nalu_hypre_BoxCreate(ndim);
         for (d = 0; d < ndim; d++)
         {
            nalu_hypre_BoxIMinD(set_box, d) = cindex[d];
            nalu_hypre_BoxIMaxD(set_box, d) = cindex[d];
         }
         nalu_hypre_SStructMatrixSetInterPartValues(matrix, part, set_box, var, nSentries, entries,
                                               set_box, values, action);
         nalu_hypre_BoxDestroy(set_box);
      }
   }

   /* U-matrix */
   if (nUentries > 0)
   {
      nalu_hypre_SStructUMatrixSetValues(matrix, part, cindex, var,
                                    nUentries, Uentries, values, action);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatrixSetBoxValues( NALU_HYPRE_SStructMatrix  matrix,
                                 NALU_HYPRE_Int            part,
                                 nalu_hypre_Box           *set_box,
                                 NALU_HYPRE_Int            var,
                                 NALU_HYPRE_Int            nentries,
                                 NALU_HYPRE_Int           *entries,
                                 nalu_hypre_Box           *value_box,
                                 NALU_HYPRE_Complex       *values,
                                 NALU_HYPRE_Int            action )
{
   nalu_hypre_SStructGraph      *graph = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid       *grid  = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int              **nvneighbors = nalu_hypre_SStructGridNVNeighbors(grid);
   NALU_HYPRE_Int               *Sentries;
   NALU_HYPRE_Int               *Uentries;
   NALU_HYPRE_Int                nSentries;
   NALU_HYPRE_Int                nUentries;
   nalu_hypre_SStructPMatrix    *pmatrix;


   nalu_hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, part);
      nalu_hypre_SStructPMatrixSetBoxValues(pmatrix, set_box, var, nSentries, Sentries,
                                       value_box, values, action);

      /* put inter-part couplings in UMatrix and zero them out in PMatrix
       * (possibly in ghost zones) */
      if (nvneighbors[part][var] > 0)
      {
         nalu_hypre_SStructMatrixSetInterPartValues(matrix, part, set_box, var, nSentries, entries,
                                               value_box, values, action);
      }
   }

   /* U-matrix */
   if (nUentries > 0)
   {
      nalu_hypre_SStructUMatrixSetBoxValues(matrix, part, set_box, var, nUentries, Uentries,
                                       value_box, values, action);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Put inter-part couplings in UMatrix and zero them out in PMatrix (possibly in
 * ghost zones).  Assumes that all entries are stencil entries.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatrixSetInterPartValues( NALU_HYPRE_SStructMatrix  matrix,
                                       NALU_HYPRE_Int            part,
                                       nalu_hypre_Box           *set_box,
                                       NALU_HYPRE_Int            var,
                                       NALU_HYPRE_Int            nentries,
                                       NALU_HYPRE_Int           *entries,
                                       nalu_hypre_Box           *value_box,
                                       NALU_HYPRE_Complex       *values,
                                       NALU_HYPRE_Int            action )
{
   NALU_HYPRE_Int                ndim  = nalu_hypre_SStructMatrixNDim(matrix);
   nalu_hypre_SStructGraph      *graph = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid       *grid  = nalu_hypre_SStructGraphGrid(graph);
   nalu_hypre_SStructPMatrix    *pmatrix;
   nalu_hypre_SStructPGrid      *pgrid;

   nalu_hypre_SStructStencil    *stencil;
   nalu_hypre_Index             *shape;
   NALU_HYPRE_Int               *smap;
   NALU_HYPRE_Int               *vars, frvartype, tovartype;
   nalu_hypre_StructMatrix      *smatrix;
   nalu_hypre_Box               *box, *ibox0, *ibox1, *tobox, *frbox;
   nalu_hypre_Index              stride, loop_size;
   nalu_hypre_IndexRef           offset, start;
   nalu_hypre_BoxManEntry      **frentries, **toentries;
   nalu_hypre_SStructBoxManInfo *frinfo, *toinfo;
   NALU_HYPRE_Complex           *tvalues = NULL;
   NALU_HYPRE_Int                tvalues_size = 0;
   NALU_HYPRE_Int                nfrentries, ntoentries, frpart, topart;
   NALU_HYPRE_Int                entry, sentry, ei, fri, toi;
   NALU_HYPRE_MemoryLocation     memory_location = nalu_hypre_IJMatrixMemoryLocation(nalu_hypre_SStructMatrixIJMatrix(
                                                                              matrix));

   pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, part);

   pgrid = nalu_hypre_SStructPMatrixPGrid(pmatrix);
   frvartype = nalu_hypre_SStructPGridVarType(pgrid, var);

   box   = nalu_hypre_BoxCreate(ndim);
   ibox0 = nalu_hypre_BoxCreate(ndim);
   ibox1 = nalu_hypre_BoxCreate(ndim);
   tobox = nalu_hypre_BoxCreate(ndim);
   frbox = nalu_hypre_BoxCreate(ndim);

   stencil = nalu_hypre_SStructPMatrixStencil(pmatrix, var);
   smap    = nalu_hypre_SStructPMatrixSMap(pmatrix, var);
   shape   = nalu_hypre_SStructStencilShape(stencil);
   vars    = nalu_hypre_SStructStencilVars(stencil);

   nalu_hypre_SetIndex(stride, 1);

   for (ei = 0; ei < nentries; ei++)
   {
      entry  = entries[ei];
      sentry = smap[entry];
      offset = shape[entry];
      smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entry]);
      tovartype = nalu_hypre_SStructPGridVarType(pgrid, vars[entry]);

      /* shift box in the stencil offset direction */
      nalu_hypre_CopyBox(set_box, box);

      nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(box), offset, ndim, nalu_hypre_BoxIMin(box));
      nalu_hypre_AddIndexes(nalu_hypre_BoxIMax(box), offset, ndim, nalu_hypre_BoxIMax(box));

      /* get "to" entries */
      nalu_hypre_SStructGridIntersect(grid, part, vars[entry], box, -1,
                                 &toentries, &ntoentries);

      for (toi = 0; toi < ntoentries; toi++)
      {
         nalu_hypre_BoxManEntryGetExtents(
            toentries[toi], nalu_hypre_BoxIMin(tobox), nalu_hypre_BoxIMax(tobox));
         nalu_hypre_IntersectBoxes(box, tobox, ibox0);
         if (nalu_hypre_BoxVolume(ibox0))
         {
            nalu_hypre_SStructBoxManEntryGetPart(toentries[toi], part, &topart);

            /* shift ibox0 back */
            nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(ibox0), offset, ndim,
                                  nalu_hypre_BoxIMin(ibox0));
            nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMax(ibox0), offset, ndim,
                                  nalu_hypre_BoxIMax(ibox0));

            /* get "from" entries */
            nalu_hypre_SStructGridIntersect(grid, part, var, ibox0, -1,
                                       &frentries, &nfrentries);
            for (fri = 0; fri < nfrentries; fri++)
            {
               /* don't set couplings within the same part unless possibly for
                * cell data (to simplify periodic conditions for users) */
               nalu_hypre_SStructBoxManEntryGetPart(frentries[fri], part, &frpart);
               if (topart == frpart)
               {
                  if ( (frvartype != NALU_HYPRE_SSTRUCT_VARIABLE_CELL) ||
                       (tovartype != NALU_HYPRE_SSTRUCT_VARIABLE_CELL) )
                  {
                     continue;
                  }
                  nalu_hypre_BoxManEntryGetInfo(frentries[fri], (void **) &frinfo);
                  nalu_hypre_BoxManEntryGetInfo(toentries[toi], (void **) &toinfo);
                  if ( nalu_hypre_SStructBoxManInfoType(frinfo) ==
                       nalu_hypre_SStructBoxManInfoType(toinfo) )
                  {
                     continue;
                  }
               }

               nalu_hypre_BoxManEntryGetExtents(
                  frentries[fri], nalu_hypre_BoxIMin(frbox), nalu_hypre_BoxIMax(frbox));
               nalu_hypre_IntersectBoxes(ibox0, frbox, ibox1);
               if (nalu_hypre_BoxVolume(ibox1))
               {
                  NALU_HYPRE_Int tvalues_new_size = nalu_hypre_BoxVolume(ibox1);
                  tvalues = nalu_hypre_TReAlloc_v2(tvalues, NALU_HYPRE_Complex, tvalues_size, NALU_HYPRE_Complex, tvalues_new_size,
                                              memory_location);
                  tvalues_size = tvalues_new_size;

                  if (action >= 0)
                  {
                     /* set or add */

                     /* copy values into tvalues */
                     start = nalu_hypre_BoxIMin(ibox1);
                     nalu_hypre_BoxGetSize(ibox1, loop_size);
#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(tvalues,values)
                     nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                         ibox1, start, stride, mi,
                                         value_box, start, stride, vi);
                     {
                        tvalues[mi] = values[ei + vi * nentries];
                     }
                     nalu_hypre_BoxLoop2End(mi, vi);
#undef DEVICE_VAR
#define DEVICE_VAR
                     /* put values into UMatrix */
                     nalu_hypre_SStructUMatrixSetBoxValues(
                        matrix, part, ibox1, var, 1, &entry, ibox1, tvalues, action);
                     /* zero out values in PMatrix (possibly in ghost) */
                     nalu_hypre_StructMatrixClearBoxValues(
                        smatrix, ibox1, 1, &sentry, -1, 1);
                  }
                  else
                  {
                     /* get */

                     /* get values from UMatrix */
                     nalu_hypre_SStructUMatrixSetBoxValues(
                        matrix, part, ibox1, var, 1, &entry, ibox1, tvalues, action);

                     /* copy tvalues into values */
                     start = nalu_hypre_BoxIMin(ibox1);
                     nalu_hypre_BoxGetSize(ibox1, loop_size);
#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(tvalues,values)
                     nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                         ibox1, start, stride, mi,
                                         value_box, start, stride, vi);
                     {
                        values[ei + vi * nentries] = tvalues[mi];
                     }
                     nalu_hypre_BoxLoop2End(mi, vi);
#undef DEVICE_VAR
#define DEVICE_VAR
                  } /* end if action */
               } /* end if nonzero ibox1 */
            } /* end of "from" boxman entries loop */
            nalu_hypre_TFree(frentries, NALU_HYPRE_MEMORY_HOST);
         } /* end if nonzero ibox0 */
      } /* end of "to" boxman entries loop */
      nalu_hypre_TFree(toentries, NALU_HYPRE_MEMORY_HOST);
   } /* end of entries loop */

   nalu_hypre_BoxDestroy(box);
   nalu_hypre_BoxDestroy(ibox0);
   nalu_hypre_BoxDestroy(ibox1);
   nalu_hypre_BoxDestroy(tobox);
   nalu_hypre_BoxDestroy(frbox);
   nalu_hypre_TFree(tvalues, memory_location);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_MemoryLocation
nalu_hypre_SStructMatrixMemoryLocation(nalu_hypre_SStructMatrix *matrix)
{
   NALU_HYPRE_Int type = nalu_hypre_SStructMatrixObjectType(matrix);

   if (type == NALU_HYPRE_SSTRUCT)
   {
      return nalu_hypre_ParCSRMatrixMemoryLocation(nalu_hypre_SStructMatrixParCSRMatrix(matrix));
   }

   void *object;
   NALU_HYPRE_SStructMatrixGetObject(matrix, &object);

   if (type == NALU_HYPRE_PARCSR)
   {
      return nalu_hypre_ParCSRMatrixMemoryLocation((nalu_hypre_ParCSRMatrix *) object);
   }

   if (type == NALU_HYPRE_STRUCT)
   {
      return nalu_hypre_StructMatrixMemoryLocation((nalu_hypre_StructMatrix *) object);
   }

   return NALU_HYPRE_MEMORY_UNDEFINED;
}

