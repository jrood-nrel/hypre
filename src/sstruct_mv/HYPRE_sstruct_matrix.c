/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructMatrix interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixCreate( MPI_Comm              comm,
                           NALU_HYPRE_SStructGraph    graph,
                           NALU_HYPRE_SStructMatrix  *matrix_ptr )
{
   nalu_hypre_SStructStencil ***stencils = nalu_hypre_SStructGraphStencils(graph);

   nalu_hypre_SStructMatrix    *matrix;
   NALU_HYPRE_Int            ***splits;
   NALU_HYPRE_Int               nparts;
   nalu_hypre_SStructPMatrix  **pmatrices;
   NALU_HYPRE_Int            ***symmetric;

   nalu_hypre_SStructPGrid     *pgrid;
   NALU_HYPRE_Int               nvars;

   NALU_HYPRE_Int               stencil_size;
   NALU_HYPRE_Int              *stencil_vars;
   NALU_HYPRE_Int               pstencil_size;

   NALU_HYPRE_SStructVariable   vitype, vjtype;
   NALU_HYPRE_Int               part, vi, vj, i;
   NALU_HYPRE_Int               size, rectangular;

   matrix = nalu_hypre_TAlloc(nalu_hypre_SStructMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructMatrixComm(matrix)  = comm;
   nalu_hypre_SStructMatrixNDim(matrix)  = nalu_hypre_SStructGraphNDim(graph);
   nalu_hypre_SStructGraphRef(graph, &nalu_hypre_SStructMatrixGraph(matrix));

   /* compute S/U-matrix split */
   nparts = nalu_hypre_SStructGraphNParts(graph);
   nalu_hypre_SStructMatrixNParts(matrix) = nparts;
   splits = nalu_hypre_TAlloc(NALU_HYPRE_Int **,  nparts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructMatrixSplits(matrix) = splits;
   pmatrices = nalu_hypre_TAlloc(nalu_hypre_SStructPMatrix *,  nparts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructMatrixPMatrices(matrix) = pmatrices;
   symmetric = nalu_hypre_TAlloc(NALU_HYPRE_Int **,  nparts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructMatrixSymmetric(matrix) = symmetric;
   /* is this a rectangular matrix? */
   rectangular = 0;
   if (nalu_hypre_SStructGraphGrid(graph) != nalu_hypre_SStructGraphDomainGrid(graph))
   {
      rectangular = 1;
   }
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGraphPGrid(graph, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      splits[part] = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nvars, NALU_HYPRE_MEMORY_HOST);
      symmetric[part] = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (vi = 0; vi < nvars; vi++)
      {
         stencil_size  = nalu_hypre_SStructStencilSize(stencils[part][vi]);
         stencil_vars  = nalu_hypre_SStructStencilVars(stencils[part][vi]);
         pstencil_size = 0;
         splits[part][vi] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
         symmetric[part][vi] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nvars, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < stencil_size; i++)
         {
            /* for rectangular matrices, put all coefficients in U-matrix */
            if (rectangular)
            {
               splits[part][vi][i] = -1;
            }
            else
            {
               vj = stencil_vars[i];
               vitype = nalu_hypre_SStructPGridVarType(pgrid, vi);
               vjtype = nalu_hypre_SStructPGridVarType(pgrid, vj);
               if (vjtype == vitype)
               {
                  splits[part][vi][i] = pstencil_size;
                  pstencil_size++;
               }
               else
               {
                  splits[part][vi][i] = -1;
               }
            }
         }
         for (vj = 0; vj < nvars; vj++)
         {
            symmetric[part][vi][vj] = 0;
         }
      }
   }

   /* GEC0902 move the IJ creation to the initialization phase
    * ilower = nalu_hypre_SStructGridGhstartRank(grid);
    * iupper = ilower + nalu_hypre_SStructGridGhlocalSize(grid) - 1;
    * NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper,
    *                    &nalu_hypre_SStructMatrixIJMatrix(matrix)); */

   nalu_hypre_SStructMatrixIJMatrix(matrix)     = NULL;
   nalu_hypre_SStructMatrixParCSRMatrix(matrix) = NULL;

   size = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGraphPGrid(graph, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      for (vi = 0; vi < nvars; vi++)
      {
         size = nalu_hypre_max(size, nalu_hypre_SStructStencilSize(stencils[part][vi]));
      }
   }
   nalu_hypre_SStructMatrixSEntries(matrix) = nalu_hypre_TAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
   size += nalu_hypre_SStructGraphUEMaxSize(graph);
   nalu_hypre_SStructMatrixUEntries(matrix) = nalu_hypre_TAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructMatrixEntriesSize(matrix) = size;
   nalu_hypre_SStructMatrixTmpRowCoords(matrix) = NULL;
   nalu_hypre_SStructMatrixTmpColCoords(matrix) = NULL;
   nalu_hypre_SStructMatrixTmpCoeffs(matrix)    = NULL;
   nalu_hypre_SStructMatrixTmpRowCoordsDevice(matrix) = NULL;
   nalu_hypre_SStructMatrixTmpColCoordsDevice(matrix) = NULL;
   nalu_hypre_SStructMatrixTmpCoeffsDevice(matrix)    = NULL;

   nalu_hypre_SStructMatrixNSSymmetric(matrix) = 0;
   nalu_hypre_SStructMatrixGlobalSize(matrix)  = 0;
   nalu_hypre_SStructMatrixRefCount(matrix)    = 1;

   /* GEC0902 setting the default of the object_type to NALU_HYPRE_SSTRUCT */

   nalu_hypre_SStructMatrixObjectType(matrix) = NALU_HYPRE_SSTRUCT;

   *matrix_ptr = matrix;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixDestroy( NALU_HYPRE_SStructMatrix matrix )
{
   nalu_hypre_SStructGraph     *graph;
   NALU_HYPRE_Int            ***splits;
   NALU_HYPRE_Int               nparts;
   nalu_hypre_SStructPMatrix  **pmatrices;
   NALU_HYPRE_Int            ***symmetric;
   nalu_hypre_SStructPGrid     *pgrid;
   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int               part, var;
   NALU_HYPRE_MemoryLocation    memory_location;

   if (matrix)
   {
      memory_location = nalu_hypre_SStructMatrixMemoryLocation(matrix);

      nalu_hypre_SStructMatrixRefCount(matrix) --;
      if (nalu_hypre_SStructMatrixRefCount(matrix) == 0)
      {
         graph        = nalu_hypre_SStructMatrixGraph(matrix);
         splits       = nalu_hypre_SStructMatrixSplits(matrix);
         nparts       = nalu_hypre_SStructMatrixNParts(matrix);
         pmatrices    = nalu_hypre_SStructMatrixPMatrices(matrix);
         symmetric    = nalu_hypre_SStructMatrixSymmetric(matrix);
         for (part = 0; part < nparts; part++)
         {
            pgrid = nalu_hypre_SStructGraphPGrid(graph, part);
            nvars = nalu_hypre_SStructPGridNVars(pgrid);
            for (var = 0; var < nvars; var++)
            {
               nalu_hypre_TFree(splits[part][var], NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(symmetric[part][var], NALU_HYPRE_MEMORY_HOST);
            }
            nalu_hypre_TFree(splits[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(symmetric[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_SStructPMatrixDestroy(pmatrices[part]);
         }
         NALU_HYPRE_SStructGraphDestroy(graph);
         nalu_hypre_TFree(splits, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pmatrices, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(symmetric, NALU_HYPRE_MEMORY_HOST);
         NALU_HYPRE_IJMatrixDestroy(nalu_hypre_SStructMatrixIJMatrix(matrix));
         nalu_hypre_TFree(nalu_hypre_SStructMatrixSEntries(matrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_SStructMatrixUEntries(matrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_SStructMatrixTmpRowCoords(matrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_SStructMatrixTmpColCoords(matrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_SStructMatrixTmpCoeffs(matrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_SStructMatrixTmpRowCoordsDevice(matrix), memory_location);
         nalu_hypre_TFree(nalu_hypre_SStructMatrixTmpColCoordsDevice(matrix), memory_location);
         nalu_hypre_TFree(nalu_hypre_SStructMatrixTmpCoeffsDevice(matrix), memory_location);
         nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixInitialize( NALU_HYPRE_SStructMatrix matrix )
{
   NALU_HYPRE_Int               nparts    = nalu_hypre_SStructMatrixNParts(matrix);
   nalu_hypre_SStructGraph     *graph     = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructPMatrix  **pmatrices = nalu_hypre_SStructMatrixPMatrices(matrix);
   NALU_HYPRE_Int            ***symmetric = nalu_hypre_SStructMatrixSymmetric(matrix);
   nalu_hypre_SStructStencil ***stencils  = nalu_hypre_SStructGraphStencils(graph);
   NALU_HYPRE_Int              *split;

   MPI_Comm                pcomm;
   nalu_hypre_SStructPGrid     *pgrid;
   nalu_hypre_SStructStencil  **pstencils;
   NALU_HYPRE_Int               nvars;

   NALU_HYPRE_Int               stencil_size;
   nalu_hypre_Index            *stencil_shape;
   NALU_HYPRE_Int              *stencil_vars;
   NALU_HYPRE_Int               pstencil_ndim;
   NALU_HYPRE_Int               pstencil_size;

   NALU_HYPRE_Int               part, var, i;

   /* GEC0902 addition of variables for ilower and iupper   */
   MPI_Comm                comm;
   nalu_hypre_SStructGrid      *grid, *domain_grid;
   NALU_HYPRE_Int               ilower, iupper, jlower, jupper;
   NALU_HYPRE_Int               matrix_type = nalu_hypre_SStructMatrixObjectType(matrix);

   /* S-matrix */
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGraphPGrid(graph, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      pstencils = nalu_hypre_TAlloc(nalu_hypre_SStructStencil *,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (var = 0; var < nvars; var++)
      {
         split = nalu_hypre_SStructMatrixSplit(matrix, part, var);
         stencil_size  = nalu_hypre_SStructStencilSize(stencils[part][var]);
         stencil_shape = nalu_hypre_SStructStencilShape(stencils[part][var]);
         stencil_vars  = nalu_hypre_SStructStencilVars(stencils[part][var]);
         pstencil_ndim = nalu_hypre_SStructStencilNDim(stencils[part][var]);
         pstencil_size = 0;
         for (i = 0; i < stencil_size; i++)
         {
            if (split[i] > -1)
            {
               pstencil_size++;
            }
         }
         NALU_HYPRE_SStructStencilCreate(pstencil_ndim, pstencil_size,
                                    &pstencils[var]);
         for (i = 0; i < stencil_size; i++)
         {
            if (split[i] > -1)
            {
               NALU_HYPRE_SStructStencilSetEntry(pstencils[var], split[i],
                                            stencil_shape[i],
                                            stencil_vars[i]);
            }
         }
      }
      pcomm = nalu_hypre_SStructPGridComm(pgrid);
      nalu_hypre_SStructPMatrixCreate(pcomm, pgrid, pstencils, &pmatrices[part]);
      for (var = 0; var < nvars; var++)
      {
         for (i = 0; i < nvars; i++)
         {
            nalu_hypre_SStructPMatrixSetSymmetric(pmatrices[part], var, i,
                                             symmetric[part][var][i]);
         }
      }
      nalu_hypre_SStructPMatrixInitialize(pmatrices[part]);
   }

   /* U-matrix */

   /* GEC0902  knowing the kind of matrix we can create the IJMATRIX with the
    *  the right dimension (NALU_HYPRE_PARCSR without ghosts) */

   grid = nalu_hypre_SStructGraphGrid(graph);
   domain_grid = nalu_hypre_SStructGraphDomainGrid(graph);
   comm =  nalu_hypre_SStructMatrixComm(matrix);

   if (matrix_type == NALU_HYPRE_PARCSR)
   {
      ilower = nalu_hypre_SStructGridStartRank(grid);
      iupper = ilower + nalu_hypre_SStructGridLocalSize(grid) - 1;
      jlower = nalu_hypre_SStructGridStartRank(domain_grid);
      jupper = jlower + nalu_hypre_SStructGridLocalSize(domain_grid) - 1;
   }

   if (matrix_type == NALU_HYPRE_SSTRUCT || matrix_type == NALU_HYPRE_STRUCT)
   {
      ilower = nalu_hypre_SStructGridGhstartRank(grid);
      iupper = ilower + nalu_hypre_SStructGridGhlocalSize(grid) - 1;
      jlower = nalu_hypre_SStructGridGhstartRank(domain_grid);
      jupper = jlower + nalu_hypre_SStructGridGhlocalSize(domain_grid) - 1;
   }

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper,
                        &nalu_hypre_SStructMatrixIJMatrix(matrix));

   nalu_hypre_SStructUMatrixInitialize(matrix);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixSetValues( NALU_HYPRE_SStructMatrix  matrix,
                              NALU_HYPRE_Int            part,
                              NALU_HYPRE_Int           *index,
                              NALU_HYPRE_Int            var,
                              NALU_HYPRE_Int            nentries,
                              NALU_HYPRE_Int           *entries,
                              NALU_HYPRE_Complex       *values )
{
   nalu_hypre_SStructMatrixSetValues(matrix, part, index, var,
                                nentries, entries, values, 0);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixAddToValues( NALU_HYPRE_SStructMatrix  matrix,
                                NALU_HYPRE_Int            part,
                                NALU_HYPRE_Int           *index,
                                NALU_HYPRE_Int            var,
                                NALU_HYPRE_Int            nentries,
                                NALU_HYPRE_Int           *entries,
                                NALU_HYPRE_Complex       *values )
{
   nalu_hypre_SStructMatrixSetValues(matrix, part, index, var,
                                nentries, entries, values, 1);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* ONLY3D - RDF: Why? */

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixAddFEMValues( NALU_HYPRE_SStructMatrix  matrix,
                                 NALU_HYPRE_Int            part,
                                 NALU_HYPRE_Int           *index,
                                 NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_Int           ndim         = nalu_hypre_SStructMatrixNDim(matrix);
   nalu_hypre_SStructGraph *graph        = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid  *grid         = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int           fem_nsparse  = nalu_hypre_SStructGraphFEMPNSparse(graph, part);
   NALU_HYPRE_Int          *fem_sparse_i = nalu_hypre_SStructGraphFEMPSparseI(graph, part);
   NALU_HYPRE_Int          *fem_entries  = nalu_hypre_SStructGraphFEMPEntries(graph, part);
   NALU_HYPRE_Int          *fem_vars     = nalu_hypre_SStructGridFEMPVars(grid, part);
   nalu_hypre_Index        *fem_offsets  = nalu_hypre_SStructGridFEMPOffsets(grid, part);
   NALU_HYPRE_Int           s, i, d, vindex[NALU_HYPRE_MAXDIM];

   /* Set one coefficient at a time */
   for (s = 0; s < fem_nsparse; s++)
   {
      i = fem_sparse_i[s];
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vindex[d] = index[d] + nalu_hypre_IndexD(fem_offsets[i], d);
      }
      NALU_HYPRE_SStructMatrixAddToValues(
         matrix, part, vindex, fem_vars[i], 1, &fem_entries[s], &values[s]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixGetValues( NALU_HYPRE_SStructMatrix  matrix,
                              NALU_HYPRE_Int            part,
                              NALU_HYPRE_Int           *index,
                              NALU_HYPRE_Int            var,
                              NALU_HYPRE_Int            nentries,
                              NALU_HYPRE_Int           *entries,
                              NALU_HYPRE_Complex       *values )
{
   nalu_hypre_SStructMatrixSetValues(matrix, part, index, var,
                                nentries, entries, values, -1);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* ONLY3D - RDF: Why? */

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixGetFEMValues( NALU_HYPRE_SStructMatrix  matrix,
                                 NALU_HYPRE_Int            part,
                                 NALU_HYPRE_Int           *index,
                                 NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_Int           ndim         = nalu_hypre_SStructMatrixNDim(matrix);
   nalu_hypre_SStructGraph *graph        = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid  *grid         = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int           fem_nsparse  = nalu_hypre_SStructGraphFEMPNSparse(graph, part);
   NALU_HYPRE_Int          *fem_sparse_i = nalu_hypre_SStructGraphFEMPSparseI(graph, part);
   NALU_HYPRE_Int          *fem_entries  = nalu_hypre_SStructGraphFEMPEntries(graph, part);
   NALU_HYPRE_Int          *fem_vars     = nalu_hypre_SStructGridFEMPVars(grid, part);
   nalu_hypre_Index        *fem_offsets  = nalu_hypre_SStructGridFEMPOffsets(grid, part);
   NALU_HYPRE_Int           s, i, d, vindex[NALU_HYPRE_MAXDIM];

   for (s = 0; s < fem_nsparse; s++)
   {
      i = fem_sparse_i[s];
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vindex[d] = index[d] + nalu_hypre_IndexD(fem_offsets[i], d);
      }
      nalu_hypre_SStructMatrixSetValues(
         matrix, part, vindex, fem_vars[i], 1, &fem_entries[s], &values[s], -1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixSetBoxValues( NALU_HYPRE_SStructMatrix  matrix,
                                 NALU_HYPRE_Int            part,
                                 NALU_HYPRE_Int           *ilower,
                                 NALU_HYPRE_Int           *iupper,
                                 NALU_HYPRE_Int            var,
                                 NALU_HYPRE_Int            nentries,
                                 NALU_HYPRE_Int           *entries,
                                 NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_SStructMatrixSetBoxValues2(matrix, part, ilower, iupper, var, nentries, entries,
                                    ilower, iupper, values);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixAddToBoxValues( NALU_HYPRE_SStructMatrix  matrix,
                                   NALU_HYPRE_Int            part,
                                   NALU_HYPRE_Int           *ilower,
                                   NALU_HYPRE_Int           *iupper,
                                   NALU_HYPRE_Int            var,
                                   NALU_HYPRE_Int            nentries,
                                   NALU_HYPRE_Int           *entries,
                                   NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_SStructMatrixAddToBoxValues2(matrix, part, ilower, iupper, var, nentries, entries,
                                      ilower, iupper, values);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixGetBoxValues( NALU_HYPRE_SStructMatrix  matrix,
                                 NALU_HYPRE_Int            part,
                                 NALU_HYPRE_Int           *ilower,
                                 NALU_HYPRE_Int           *iupper,
                                 NALU_HYPRE_Int            var,
                                 NALU_HYPRE_Int            nentries,
                                 NALU_HYPRE_Int           *entries,
                                 NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_SStructMatrixGetBoxValues2(matrix, part, ilower, iupper, var, nentries, entries,
                                    ilower, iupper, values);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixSetBoxValues2( NALU_HYPRE_SStructMatrix  matrix,
                                  NALU_HYPRE_Int            part,
                                  NALU_HYPRE_Int           *ilower,
                                  NALU_HYPRE_Int           *iupper,
                                  NALU_HYPRE_Int            var,
                                  NALU_HYPRE_Int            nentries,
                                  NALU_HYPRE_Int           *entries,
                                  NALU_HYPRE_Int           *vilower,
                                  NALU_HYPRE_Int           *viupper,
                                  NALU_HYPRE_Complex       *values )
{
   nalu_hypre_Box  *set_box, *value_box;
   NALU_HYPRE_Int   d, ndim = nalu_hypre_SStructMatrixNDim(matrix);

   /* This creates boxes with zeroed-out extents */
   set_box = nalu_hypre_BoxCreate(ndim);
   value_box = nalu_hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(set_box, d) = ilower[d];
      nalu_hypre_BoxIMaxD(set_box, d) = iupper[d];
      nalu_hypre_BoxIMinD(value_box, d) = vilower[d];
      nalu_hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   nalu_hypre_SStructMatrixSetBoxValues(matrix, part, set_box, var, nentries, entries,
                                   value_box, values, 0);

   nalu_hypre_BoxDestroy(set_box);
   nalu_hypre_BoxDestroy(value_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixAddToBoxValues2( NALU_HYPRE_SStructMatrix  matrix,
                                    NALU_HYPRE_Int            part,
                                    NALU_HYPRE_Int           *ilower,
                                    NALU_HYPRE_Int           *iupper,
                                    NALU_HYPRE_Int            var,
                                    NALU_HYPRE_Int            nentries,
                                    NALU_HYPRE_Int           *entries,
                                    NALU_HYPRE_Int           *vilower,
                                    NALU_HYPRE_Int           *viupper,
                                    NALU_HYPRE_Complex       *values )
{
   nalu_hypre_Box  *set_box, *value_box;
   NALU_HYPRE_Int   d, ndim = nalu_hypre_SStructMatrixNDim(matrix);

   /* This creates boxes with zeroed-out extents */
   set_box = nalu_hypre_BoxCreate(ndim);
   value_box = nalu_hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(set_box, d) = ilower[d];
      nalu_hypre_BoxIMaxD(set_box, d) = iupper[d];
      nalu_hypre_BoxIMinD(value_box, d) = vilower[d];
      nalu_hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   nalu_hypre_SStructMatrixSetBoxValues(matrix, part, set_box, var, nentries, entries,
                                   value_box, values, 1);

   nalu_hypre_BoxDestroy(set_box);
   nalu_hypre_BoxDestroy(value_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixGetBoxValues2( NALU_HYPRE_SStructMatrix  matrix,
                                  NALU_HYPRE_Int            part,
                                  NALU_HYPRE_Int           *ilower,
                                  NALU_HYPRE_Int           *iupper,
                                  NALU_HYPRE_Int            var,
                                  NALU_HYPRE_Int            nentries,
                                  NALU_HYPRE_Int           *entries,
                                  NALU_HYPRE_Int           *vilower,
                                  NALU_HYPRE_Int           *viupper,
                                  NALU_HYPRE_Complex       *values )
{
   nalu_hypre_Box  *set_box, *value_box;
   NALU_HYPRE_Int   d, ndim = nalu_hypre_SStructMatrixNDim(matrix);

   /* This creates boxes with zeroed-out extents */
   set_box = nalu_hypre_BoxCreate(ndim);
   value_box = nalu_hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(set_box, d) = ilower[d];
      nalu_hypre_BoxIMaxD(set_box, d) = iupper[d];
      nalu_hypre_BoxIMinD(value_box, d) = vilower[d];
      nalu_hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   nalu_hypre_SStructMatrixSetBoxValues(matrix, part, set_box, var, nentries, entries,
                                   value_box, values, -1);

   nalu_hypre_BoxDestroy(set_box);
   nalu_hypre_BoxDestroy(value_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixAddFEMBoxValues(NALU_HYPRE_SStructMatrix  matrix,
                                   NALU_HYPRE_Int            part,
                                   NALU_HYPRE_Int           *ilower,
                                   NALU_HYPRE_Int           *iupper,
                                   NALU_HYPRE_Complex       *values)
{
   NALU_HYPRE_Int             ndim            = nalu_hypre_SStructMatrixNDim(matrix);
   nalu_hypre_SStructGraph   *graph           = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid    *grid            = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_SStructMatrixMemoryLocation(matrix);

   NALU_HYPRE_Int             fem_nsparse     = nalu_hypre_SStructGraphFEMPNSparse(graph, part);
   NALU_HYPRE_Int            *fem_sparse_i    = nalu_hypre_SStructGraphFEMPSparseI(graph, part);
   NALU_HYPRE_Int            *fem_entries     = nalu_hypre_SStructGraphFEMPEntries(graph, part);
   NALU_HYPRE_Int            *fem_vars        = nalu_hypre_SStructGridFEMPVars(grid, part);
   nalu_hypre_Index          *fem_offsets     = nalu_hypre_SStructGridFEMPOffsets(grid, part);

   NALU_HYPRE_Complex        *tvalues;
   nalu_hypre_Box            *box;

   NALU_HYPRE_Int             s, i, d, vilower[NALU_HYPRE_MAXDIM], viupper[NALU_HYPRE_MAXDIM];
   NALU_HYPRE_Int             ei, vi, nelts;

   /* Set one coefficient at a time */
   box = nalu_hypre_BoxCreate(ndim);
   nalu_hypre_BoxSetExtents(box, ilower, iupper);
   nelts = nalu_hypre_BoxVolume(box);
   tvalues = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nelts, memory_location);

   for (s = 0; s < fem_nsparse; s++)
   {
      i = fem_sparse_i[s];
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vilower[d] = ilower[d] + nalu_hypre_IndexD(fem_offsets[i], d);
         viupper[d] = iupper[d] + nalu_hypre_IndexD(fem_offsets[i], d);
      }

#if defined(NALU_HYPRE_USING_GPU)
      if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
      {
         hypreDevice_ComplexStridedCopy(nelts, fem_nsparse, values + s, tvalues);
      }
      else
#endif
      {
         for (ei = 0, vi = s; ei < nelts; ei ++, vi += fem_nsparse)
         {
            tvalues[ei] = values[vi];
         }
      }

      NALU_HYPRE_SStructMatrixAddToBoxValues(matrix, part, vilower, viupper,
                                        fem_vars[i], 1, &fem_entries[s],
                                        tvalues);
   }

   /* Free memory */
   nalu_hypre_TFree(tvalues, memory_location);
   nalu_hypre_BoxDestroy(box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixAssemble( NALU_HYPRE_SStructMatrix matrix )
{
   NALU_HYPRE_Int               ndim           = nalu_hypre_SStructMatrixNDim(matrix);
   nalu_hypre_SStructGraph     *graph          = nalu_hypre_SStructMatrixGraph(matrix);
   NALU_HYPRE_Int               nparts         = nalu_hypre_SStructMatrixNParts(matrix);
   nalu_hypre_SStructPMatrix  **pmatrices      = nalu_hypre_SStructMatrixPMatrices(matrix);
   nalu_hypre_SStructGrid      *grid           = nalu_hypre_SStructGraphGrid(graph);
   nalu_hypre_SStructCommInfo **vnbor_comm_info = nalu_hypre_SStructGridVNborCommInfo(grid);
   NALU_HYPRE_Int               vnbor_ncomms    = nalu_hypre_SStructGridVNborNComms(grid);

   NALU_HYPRE_Int               part;

   nalu_hypre_CommInfo         *comm_info;
   NALU_HYPRE_Int               send_part,    recv_part;
   NALU_HYPRE_Int               send_var,     recv_var;
   nalu_hypre_StructMatrix     *send_matrix, *recv_matrix;
   nalu_hypre_CommPkg          *comm_pkg;
   nalu_hypre_CommHandle       *comm_handle;
   NALU_HYPRE_Int               ci;


   /*------------------------------------------------------
    * NOTE: Inter-part couplings were taken care of earlier.
    *------------------------------------------------------*/

   /*------------------------------------------------------
    * Communicate and accumulate within parts
    *------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_SStructPMatrixAccumulate(pmatrices[part]);
   }

   /*------------------------------------------------------
    * Communicate and accumulate between parts
    *------------------------------------------------------*/

   for (ci = 0; ci < vnbor_ncomms; ci++)
   {
      comm_info = nalu_hypre_SStructCommInfoCommInfo(vnbor_comm_info[ci]);
      send_part = nalu_hypre_SStructCommInfoSendPart(vnbor_comm_info[ci]);
      recv_part = nalu_hypre_SStructCommInfoRecvPart(vnbor_comm_info[ci]);
      send_var  = nalu_hypre_SStructCommInfoSendVar(vnbor_comm_info[ci]);
      recv_var  = nalu_hypre_SStructCommInfoRecvVar(vnbor_comm_info[ci]);

      send_matrix = nalu_hypre_SStructPMatrixSMatrix(
                       nalu_hypre_SStructMatrixPMatrix(matrix, send_part), send_var, send_var);
      recv_matrix = nalu_hypre_SStructPMatrixSMatrix(
                       nalu_hypre_SStructMatrixPMatrix(matrix, recv_part), recv_var, recv_var);

      if ((send_matrix != NULL) && (recv_matrix != NULL))
      {
         nalu_hypre_StructStencil *send_stencil = nalu_hypre_StructMatrixStencil(send_matrix);
         nalu_hypre_StructStencil *recv_stencil = nalu_hypre_StructMatrixStencil(recv_matrix);
         NALU_HYPRE_Int            num_values, stencil_size, num_transforms;
         NALU_HYPRE_Int           *symm;
         NALU_HYPRE_Int           *v_to_s, *s_to_v;
         nalu_hypre_Index         *coords, *dirs;
         NALU_HYPRE_Int          **orders, *order;
         nalu_hypre_IndexRef       sentry0;
         nalu_hypre_Index          sentry1;
         NALU_HYPRE_Int            ti, si, i, j;

         /* to compute 'orders', remember that we are doing reverse communication */
         num_values = nalu_hypre_StructMatrixNumValues(recv_matrix);
         symm = nalu_hypre_StructMatrixSymmElements(recv_matrix);
         stencil_size = nalu_hypre_StructStencilSize(recv_stencil);
         v_to_s = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_values, NALU_HYPRE_MEMORY_HOST);
         s_to_v = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
         for (si = 0, i = 0; si < stencil_size; si++)
         {
            s_to_v[si] = -1;
            if (symm[si] < 0)  /* this is a stored coefficient */
            {
               v_to_s[i] = si;
               s_to_v[si] = i;
               i++;
            }
         }
         nalu_hypre_CommInfoGetTransforms(comm_info, &num_transforms, &coords, &dirs);
         orders = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  num_transforms, NALU_HYPRE_MEMORY_HOST);
         order = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_values, NALU_HYPRE_MEMORY_HOST);
         for (ti = 0; ti < num_transforms; ti++)
         {
            for (i = 0; i < num_values; i++)
            {
               si = v_to_s[i];
               sentry0 = nalu_hypre_StructStencilElement(recv_stencil, si);
               for (j = 0; j < ndim; j++)
               {
                  nalu_hypre_IndexD(sentry1, nalu_hypre_IndexD(coords[ti], j)) =
                     nalu_hypre_IndexD(sentry0, j) * nalu_hypre_IndexD(dirs[ti], j);
               }
               order[i] = nalu_hypre_StructStencilElementRank(send_stencil, sentry1);
               /* currently, both send and recv transforms are parsed */
               if (order[i] > -1)
               {
                  order[i] = s_to_v[order[i]];
               }
            }
            /* want order to indicate the natural order on the remote process */
            orders[ti] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_values, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_values; i++)
            {
               orders[ti][i] = -1;
            }
            for (i = 0; i < num_values; i++)
            {
               if (order[i] > -1)
               {
                  orders[ti][order[i]] = i;
               }
            }
         }
         nalu_hypre_TFree(v_to_s, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(s_to_v, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(order, NALU_HYPRE_MEMORY_HOST);

         /* want to communicate and add ghost data to real data */
         nalu_hypre_CommPkgCreate(comm_info,
                             nalu_hypre_StructMatrixDataSpace(send_matrix),
                             nalu_hypre_StructMatrixDataSpace(recv_matrix),
                             num_values, orders, 1,
                             nalu_hypre_StructMatrixComm(send_matrix), &comm_pkg);
         /* note reversal of send/recv data here */
         nalu_hypre_InitializeCommunication(comm_pkg,
                                       nalu_hypre_StructMatrixData(recv_matrix),
                                       nalu_hypre_StructMatrixData(send_matrix),
                                       1, 0, &comm_handle);
         nalu_hypre_FinalizeCommunication(comm_handle);
         nalu_hypre_CommPkgDestroy(comm_pkg);

         for (ti = 0; ti < num_transforms; ti++)
         {
            nalu_hypre_TFree(orders[ti], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(orders, NALU_HYPRE_MEMORY_HOST);
      }
   }

   /*------------------------------------------------------
    * Assemble P and U matrices
    *------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_SStructPMatrixAssemble(pmatrices[part]);
   }

   /* U-matrix */
   nalu_hypre_SStructUMatrixAssemble(matrix);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: Should set things up so that this information can be passed
 * immediately to the PMatrix.  Unfortunately, the PMatrix is
 * currently not created until the SStructMatrix is initialized.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixSetSymmetric( NALU_HYPRE_SStructMatrix matrix,
                                 NALU_HYPRE_Int           part,
                                 NALU_HYPRE_Int           var,
                                 NALU_HYPRE_Int           to_var,
                                 NALU_HYPRE_Int           symmetric )
{
   NALU_HYPRE_Int          ***msymmetric = nalu_hypre_SStructMatrixSymmetric(matrix);
   nalu_hypre_SStructGraph   *graph      = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructPGrid   *pgrid;

   NALU_HYPRE_Int pstart = part;
   NALU_HYPRE_Int psize  = 1;
   NALU_HYPRE_Int vstart = var;
   NALU_HYPRE_Int vsize  = 1;
   NALU_HYPRE_Int tstart = to_var;
   NALU_HYPRE_Int tsize  = 1;
   NALU_HYPRE_Int p, v, t;

   if (part == -1)
   {
      pstart = 0;
      psize  = nalu_hypre_SStructMatrixNParts(matrix);
   }

   for (p = pstart; p < psize; p++)
   {
      pgrid = nalu_hypre_SStructGraphPGrid(graph, p);
      if (var == -1)
      {
         vstart = 0;
         vsize  = nalu_hypre_SStructPGridNVars(pgrid);
      }
      if (to_var == -1)
      {
         tstart = 0;
         tsize  = nalu_hypre_SStructPGridNVars(pgrid);
      }

      for (v = vstart; v < vsize; v++)
      {
         for (t = tstart; t < tsize; t++)
         {
            msymmetric[p][v][t] = symmetric;
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixSetNSSymmetric( NALU_HYPRE_SStructMatrix matrix,
                                   NALU_HYPRE_Int           symmetric )
{
   nalu_hypre_SStructMatrixNSSymmetric(matrix) = symmetric;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixSetObjectType( NALU_HYPRE_SStructMatrix  matrix,
                                  NALU_HYPRE_Int            type )
{
   nalu_hypre_SStructGraph     *graph    = nalu_hypre_SStructMatrixGraph(matrix);
   NALU_HYPRE_Int            ***splits   = nalu_hypre_SStructMatrixSplits(matrix);
   NALU_HYPRE_Int               nparts   = nalu_hypre_SStructMatrixNParts(matrix);
   nalu_hypre_SStructStencil ***stencils = nalu_hypre_SStructGraphStencils(graph);

   nalu_hypre_SStructPGrid     *pgrid;
   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int               stencil_size;
   NALU_HYPRE_Int               part, var, i;

   nalu_hypre_SStructMatrixObjectType(matrix) = type ;

   /* RDF: This and all other modifications to 'split' really belong
    * in the Initialize routine */
   if (type != NALU_HYPRE_SSTRUCT && type != NALU_HYPRE_STRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         pgrid = nalu_hypre_SStructGraphPGrid(graph, part);
         nvars = nalu_hypre_SStructPGridNVars(pgrid);
         for (var = 0; var < nvars; var++)
         {
            stencil_size = nalu_hypre_SStructStencilSize(stencils[part][var]);
            for (i = 0; i < stencil_size; i++)
            {
               splits[part][var][i] = -1;
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixGetObject( NALU_HYPRE_SStructMatrix   matrix,
                              void                **object )
{
   NALU_HYPRE_Int             type     = nalu_hypre_SStructMatrixObjectType(matrix);
   nalu_hypre_SStructPMatrix *pmatrix;
   nalu_hypre_StructMatrix   *smatrix;
   NALU_HYPRE_Int             part, var;

   if (type == NALU_HYPRE_SSTRUCT)
   {
      *object = matrix;
   }
   else if (type == NALU_HYPRE_PARCSR)
   {
      *object = nalu_hypre_SStructMatrixParCSRMatrix(matrix);
   }
   else if (type == NALU_HYPRE_STRUCT)
   {
      /* only one part & one variable */
      part = 0;
      var = 0;
      pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, part);
      smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, var, var);
      *object = smatrix;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMatrixPrint
 *
 * This function prints a SStructMatrix to file. Assumptions:
 *
 *   1) All StructMatrices have the same number of ghost layers.
 *   2) Range and domain num_ghosts are equal.
 *
 * TODO: Add GPU support
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixPrint( const char          *filename,
                          NALU_HYPRE_SStructMatrix  matrix,
                          NALU_HYPRE_Int            all )
{
   /* Matrix variables */
   MPI_Comm                comm = nalu_hypre_SStructMatrixComm(matrix);
   NALU_HYPRE_Int               nparts = nalu_hypre_SStructMatrixNParts(matrix);
   nalu_hypre_SStructGraph     *graph = nalu_hypre_SStructMatrixGraph(matrix);
   nalu_hypre_SStructGrid      *grid = nalu_hypre_SStructGraphGrid(graph);
   nalu_hypre_SStructStencil ***stencils = nalu_hypre_SStructGraphStencils(graph);
   nalu_hypre_SStructPMatrix   *pmatrix;
   nalu_hypre_StructMatrix     *smatrix;
   NALU_HYPRE_Int               data_size;

   /* Local variables */
   FILE                   *file;
   NALU_HYPRE_Int               myid;
   NALU_HYPRE_Int               part;
   NALU_HYPRE_Int               var, vi, vj, nvars;
   NALU_HYPRE_Int               num_symm_calls;
   char                    new_filename[255];

   /* Sanity check */
   nalu_hypre_assert(nparts > 0);

   /* Print auxiliary info */
   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_sprintf(new_filename, "%s.SMatrix.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_printf("Error: can't open output file %s\n", new_filename);
      nalu_hypre_error_in_arg(1);

      return nalu_hypre_error_flag;
   }

   /* Print grid info */
   nalu_hypre_fprintf(file, "SStructMatrix\n");
   nalu_hypre_SStructGridPrint(file, grid);

   /* Print stencil info */
   for (part = 0; part < nparts; part++)
   {
      pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, part);
      nvars = nalu_hypre_SStructPMatrixNVars(pmatrix);

      for (var = 0; var < nvars; var++)
      {
         nalu_hypre_fprintf(file, "\nStencil - (Part %d, Var %d):\n", part, var);
         NALU_HYPRE_SStructStencilPrint(file, stencils[part][var]);
      }
   }
   nalu_hypre_fprintf(file, "\n");

   /* Print graph info */
   NALU_HYPRE_SStructGraphPrint(file, graph);

   /* Print symmetric info */
   num_symm_calls = 0;
   for (part = 0; part < nparts; part++)
   {
      pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, part);
      nvars = nalu_hypre_SStructPMatrixNVars(pmatrix);

      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
            if (smatrix)
            {
               num_symm_calls++;
            }
         }
      }
   }
   nalu_hypre_fprintf(file, "\nMatrixNumSetSymmetric: %d", num_symm_calls);
   for (part = 0; part < nparts; part++)
   {
      pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, part);
      nvars = nalu_hypre_SStructPMatrixNVars(pmatrix);

      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
            if (smatrix)
            {
               nalu_hypre_fprintf(file, "\nMatrixSetSymmetric: %d %d %d %d",
                             part, vi, vj, nalu_hypre_StructMatrixSymmetric(smatrix));
            }
         }
      }
   }
   nalu_hypre_fprintf(file, "\n");

   /* Print data */
   for (part = 0; part < nparts; part++)
   {
      pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, part);
      nvars = nalu_hypre_SStructPMatrixNVars(pmatrix);

      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
            data_size = (smatrix) ? nalu_hypre_StructMatrixDataSize(smatrix) : 0;

            nalu_hypre_fprintf(file, "\nData - (Part %d, Vi %d, Vj %d): %d\n",
                          part, vi, vj, data_size);
            if (smatrix)
            {
               nalu_hypre_StructMatrixPrintData(file, smatrix, all);
            }
         }
      }
   }
   fclose(file);

   /* Print unstructured matrix (U-Matrix) */
   nalu_hypre_sprintf(new_filename, "%s.UMatrix", filename);
   NALU_HYPRE_IJMatrixPrint(nalu_hypre_SStructMatrixIJMatrix(matrix), new_filename);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructMatrixRead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixRead( MPI_Comm              comm,
                         const char           *filename,
                         NALU_HYPRE_SStructMatrix  *matrix_ptr )
{
   /* Matrix variables */
   NALU_HYPRE_SStructMatrix     matrix;
   nalu_hypre_SStructPMatrix   *pmatrix;
   nalu_hypre_StructMatrix     *smatrix;
   NALU_HYPRE_SStructGrid       grid;
   nalu_hypre_SStructPGrid     *pgrid;
   NALU_HYPRE_SStructGraph      graph;
   NALU_HYPRE_SStructStencil  **stencils;
   NALU_HYPRE_Int               nparts;
   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int               data_size;
   NALU_HYPRE_IJMatrix          umatrix;
   NALU_HYPRE_IJMatrix          h_umatrix;
   nalu_hypre_ParCSRMatrix     *h_parmatrix;
   nalu_hypre_ParCSRMatrix     *parmatrix = NULL;

   /* Local variables */
   FILE                   *file;
   NALU_HYPRE_Int               myid;
   NALU_HYPRE_Int               part, var;
   NALU_HYPRE_Int               p, v, i, j, vi, vj;
   NALU_HYPRE_Int               symmetric;
   NALU_HYPRE_Int               num_symm_calls;
   char                    new_filename[255];

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   /*-----------------------------------------------------------
    * Read S-Matrix
    *-----------------------------------------------------------*/

   nalu_hypre_sprintf(new_filename, "%s.SMatrix.%05d", filename, myid);
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      nalu_hypre_printf("Error: can't open input file %s\n", new_filename);
      nalu_hypre_error_in_arg(2);

      return nalu_hypre_error_flag;
   }

   /* Read grid info */
   nalu_hypre_fscanf(file, "SStructMatrix\n");
   nalu_hypre_SStructGridRead(comm, file, &grid);
   nparts = nalu_hypre_SStructGridNParts(grid);

   /* Read stencil info */
   stencils = nalu_hypre_TAlloc(NALU_HYPRE_SStructStencil *, nparts, NALU_HYPRE_MEMORY_HOST);
   for (p = 0; p < nparts; p++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, p);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);

      stencils[p] = nalu_hypre_TAlloc(NALU_HYPRE_SStructStencil, nvars, NALU_HYPRE_MEMORY_HOST);
      for (v = 0; v < nvars; v++)
      {
         nalu_hypre_fscanf(file, "\nStencil - (Part %d, Var %d):\n", &part, &var);
         NALU_HYPRE_SStructStencilRead(file, &stencils[part][var]);
      }
   }
   nalu_hypre_fscanf(file, "\n");

   /* Read graph info */
   NALU_HYPRE_SStructGraphRead(file, grid, stencils, &graph);

   /* Free memory */
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);

      for (var = 0; var < nvars; var++)
      {
         NALU_HYPRE_SStructStencilDestroy(stencils[part][var]);
      }
      nalu_hypre_TFree(stencils[part], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(stencils, NALU_HYPRE_MEMORY_HOST);

   /* Assemble graph */
   NALU_HYPRE_SStructGraphAssemble(graph);

   /* Create matrix */
   NALU_HYPRE_SStructMatrixCreate(comm, graph, &matrix);

   /* Read symmetric info */
   nalu_hypre_fscanf(file, "\nMatrixNumSetSymmetric: %d", &num_symm_calls);
   for (i = 0; i < num_symm_calls; i++)
   {
      nalu_hypre_fscanf(file, "\nMatrixSetSymmetric: %d %d %d %d",
                   &part, &vi, &vj, &symmetric);
      NALU_HYPRE_SStructMatrixSetSymmetric(matrix, part, vi, vj, symmetric);
   }
   nalu_hypre_fscanf(file, "\n");

   /* Initialize matrix */
   NALU_HYPRE_SStructMatrixInitialize(matrix);

   /* Read data */
   for (p = 0; p < nparts; p++)
   {
      pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, p);
      nvars = nalu_hypre_SStructPMatrixNVars(pmatrix);

      for (i = 0; i < nvars; i++)
      {
         for (j = 0; j < nvars; j++)
         {
            nalu_hypre_fscanf(file, "\nData - (Part %d, Vi %d, Vj %d): %d\n",
                         &part, &vi, &vj, &data_size);

            pmatrix = nalu_hypre_SStructMatrixPMatrix(matrix, part);
            smatrix = nalu_hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
            if (data_size > 0)
            {
               nalu_hypre_StructMatrixReadData(file, smatrix);
            }
         }
      }
   }
   fclose(file);

   /*-----------------------------------------------------------
    * Read U-Matrix
    *-----------------------------------------------------------*/

   /* Read unstructured matrix from file using host memory */
   nalu_hypre_sprintf(new_filename, "%s.UMatrix", filename);
   NALU_HYPRE_IJMatrixRead(new_filename, comm, NALU_HYPRE_PARCSR, &h_umatrix);
   h_parmatrix = (nalu_hypre_ParCSRMatrix*) nalu_hypre_IJMatrixObject(h_umatrix);

   /* Move ParCSRMatrix to device memory if necessary */
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
      parmatrix = nalu_hypre_ParCSRMatrixClone_v2(h_parmatrix, 1, memory_location);
   }
   else
   {
      parmatrix = h_parmatrix;
      nalu_hypre_IJMatrixObject(h_umatrix) = NULL;
   }

   /* Free memory */
   NALU_HYPRE_IJMatrixDestroy(h_umatrix);

   /* Update the umatrix with contents read from file,
      which now live on the correct memory location */
   umatrix = nalu_hypre_SStructMatrixIJMatrix(matrix);
   nalu_hypre_IJMatrixDestroyParCSR(umatrix);
   nalu_hypre_IJMatrixObject(umatrix) = (void*) parmatrix;
   nalu_hypre_SStructMatrixParCSRMatrix(matrix) = (nalu_hypre_ParCSRMatrix*) parmatrix;
   nalu_hypre_IJMatrixAssembleFlag(umatrix) = 1;

   /* Assemble SStructMatrix */
   NALU_HYPRE_SStructMatrixAssemble(matrix);

   /* Decrease ref counters */
   NALU_HYPRE_SStructGraphDestroy(graph);
   NALU_HYPRE_SStructGridDestroy(grid);

   *matrix_ptr = matrix;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructMatrixMatvec( NALU_HYPRE_Complex       alpha,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector x,
                           NALU_HYPRE_Complex       beta,
                           NALU_HYPRE_SStructVector y     )
{
   nalu_hypre_SStructMatvec(alpha, A, x, beta, y);

   return nalu_hypre_error_flag;
}
