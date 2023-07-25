/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* 9/09 AB - modified all functions to use the box manager */

/******************************************************************************
 *
 * NALU_HYPRE_SStructGrid interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridCreate( MPI_Comm           comm,
                         NALU_HYPRE_Int          ndim,
                         NALU_HYPRE_Int          nparts,
                         NALU_HYPRE_SStructGrid *grid_ptr )
{
   nalu_hypre_SStructGrid       *grid;
   nalu_hypre_SStructPGrid     **pgrids;
   nalu_hypre_SStructPGrid      *pgrid;
   NALU_HYPRE_Int               *nneighbors;
   nalu_hypre_SStructNeighbor  **neighbors;
   nalu_hypre_Index            **nbor_offsets;
   NALU_HYPRE_Int               *fem_nvars;
   NALU_HYPRE_Int              **fem_vars;
   nalu_hypre_Index            **fem_offsets;
   NALU_HYPRE_Int                num_ghost[2 * NALU_HYPRE_MAXDIM];
   NALU_HYPRE_Int                i;

   grid = nalu_hypre_TAlloc(nalu_hypre_SStructGrid,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructGridComm(grid)   = comm;
   nalu_hypre_SStructGridNDim(grid)   = ndim;
   nalu_hypre_SStructGridNParts(grid) = nparts;
   pgrids = nalu_hypre_TAlloc(nalu_hypre_SStructPGrid *,  nparts, NALU_HYPRE_MEMORY_HOST);
   nneighbors    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nparts, NALU_HYPRE_MEMORY_HOST);
   neighbors     = nalu_hypre_TAlloc(nalu_hypre_SStructNeighbor *,  nparts, NALU_HYPRE_MEMORY_HOST);
   nbor_offsets  = nalu_hypre_TAlloc(nalu_hypre_Index *,  nparts, NALU_HYPRE_MEMORY_HOST);
   fem_nvars     = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nparts, NALU_HYPRE_MEMORY_HOST);
   fem_vars      = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nparts, NALU_HYPRE_MEMORY_HOST);
   fem_offsets   = nalu_hypre_TAlloc(nalu_hypre_Index *,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nparts; i++)
   {
      nalu_hypre_SStructPGridCreate(comm, ndim, &pgrid);
      pgrids[i] = pgrid;
      nneighbors[i]    = 0;
      neighbors[i]     = NULL;
      nbor_offsets[i]  = NULL;
      fem_nvars[i]     = 0;
      fem_vars[i]      = NULL;
      fem_offsets[i]   = NULL;
   }
   nalu_hypre_SStructGridPGrids(grid)  = pgrids;
   nalu_hypre_SStructGridNNeighbors(grid)  = nneighbors;
   nalu_hypre_SStructGridNeighbors(grid)   = neighbors;
   nalu_hypre_SStructGridNborOffsets(grid) = nbor_offsets;
   nalu_hypre_SStructGridNUCVars(grid) = 0;
   nalu_hypre_SStructGridUCVars(grid)  = NULL;
   nalu_hypre_SStructGridFEMNVars(grid)   = fem_nvars;
   nalu_hypre_SStructGridFEMVars(grid)    = fem_vars;
   nalu_hypre_SStructGridFEMOffsets(grid) = fem_offsets;

   nalu_hypre_SStructGridBoxManagers(grid) = NULL;
   nalu_hypre_SStructGridNborBoxManagers(grid) = NULL;

   /* miscellaneous */
   nalu_hypre_SStructGridLocalSize(grid)     = 0;
   nalu_hypre_SStructGridGlobalSize(grid)    = 0;
   nalu_hypre_SStructGridRefCount(grid)      = 1;

   /* GEC0902 ghost addition to the grid    */
   nalu_hypre_SStructGridGhlocalSize(grid)   = 0;

   /* Initialize num ghost */
   for (i = 0; i < 2 * ndim; i++)
   {
      num_ghost[i] = 1;
   }
   nalu_hypre_SStructGridSetNumGhost(grid, num_ghost);

   *grid_ptr = grid;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridDestroy( NALU_HYPRE_SStructGrid grid )
{
   NALU_HYPRE_Int                      nparts;
   nalu_hypre_SStructPGrid           **pgrids;
   NALU_HYPRE_Int                     *nneighbors;
   nalu_hypre_SStructNeighbor        **neighbors;
   nalu_hypre_Index                  **nbor_offsets;
   NALU_HYPRE_Int                    **nvneighbors;
   nalu_hypre_SStructNeighbor       ***vneighbors;
   nalu_hypre_SStructCommInfo        **vnbor_comm_info;
   NALU_HYPRE_Int                      vnbor_ncomms;
   NALU_HYPRE_Int                     *fem_nvars;
   NALU_HYPRE_Int                    **fem_vars;
   nalu_hypre_Index                  **fem_offsets;
   nalu_hypre_BoxManager            ***managers;
   nalu_hypre_BoxManager            ***nbor_managers;
   NALU_HYPRE_Int                      nvars;
   NALU_HYPRE_Int                      part, var, i;

   if (grid)
   {
      nalu_hypre_SStructGridRefCount(grid) --;
      if (nalu_hypre_SStructGridRefCount(grid) == 0)
      {
         nparts  = nalu_hypre_SStructGridNParts(grid);
         pgrids  = nalu_hypre_SStructGridPGrids(grid);
         nneighbors   = nalu_hypre_SStructGridNNeighbors(grid);
         neighbors    = nalu_hypre_SStructGridNeighbors(grid);
         nbor_offsets = nalu_hypre_SStructGridNborOffsets(grid);
         nvneighbors = nalu_hypre_SStructGridNVNeighbors(grid);
         vneighbors  = nalu_hypre_SStructGridVNeighbors(grid);
         vnbor_comm_info = nalu_hypre_SStructGridVNborCommInfo(grid);
         vnbor_ncomms = nalu_hypre_SStructGridVNborNComms(grid);
         fem_nvars   = nalu_hypre_SStructGridFEMNVars(grid);
         fem_vars    = nalu_hypre_SStructGridFEMVars(grid);
         fem_offsets = nalu_hypre_SStructGridFEMOffsets(grid);
         managers  = nalu_hypre_SStructGridBoxManagers(grid);
         nbor_managers  = nalu_hypre_SStructGridNborBoxManagers(grid);

         for (part = 0; part < nparts; part++)
         {
            nvars = nalu_hypre_SStructPGridNVars(pgrids[part]);
            for (var = 0; var < nvars; var++)
            {
               nalu_hypre_TFree(vneighbors[part][var], NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_BoxManDestroy(managers[part][var]);
               nalu_hypre_BoxManDestroy(nbor_managers[part][var]);
            }
            nalu_hypre_TFree(neighbors[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(nbor_offsets[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(nvneighbors[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(vneighbors[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_SStructPGridDestroy(pgrids[part]);
            nalu_hypre_TFree(fem_vars[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(fem_offsets[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(managers[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(nbor_managers[part], NALU_HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < vnbor_ncomms; i++)
         {
            nalu_hypre_CommInfoDestroy(
               nalu_hypre_SStructCommInfoCommInfo(vnbor_comm_info[i]));
            nalu_hypre_TFree(vnbor_comm_info[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(vnbor_comm_info, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pgrids, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nneighbors, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(neighbors, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nbor_offsets, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(fem_nvars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(fem_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(fem_offsets, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nvneighbors, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(vneighbors, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(vnbor_comm_info, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(managers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nbor_managers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(grid, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridSetExtents( NALU_HYPRE_SStructGrid  grid,
                             NALU_HYPRE_Int          part,
                             NALU_HYPRE_Int         *ilower,
                             NALU_HYPRE_Int         *iupper )
{
   NALU_HYPRE_Int            ndim  = nalu_hypre_SStructGridNDim(grid);
   nalu_hypre_SStructPGrid  *pgrid = nalu_hypre_SStructGridPGrid(grid, part);
   nalu_hypre_Index          cilower;
   nalu_hypre_Index          ciupper;

   nalu_hypre_CopyToCleanIndex(ilower, ndim, cilower);
   nalu_hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   nalu_hypre_SStructPGridSetExtents(pgrid, cilower, ciupper);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridSetVariables( NALU_HYPRE_SStructGrid      grid,
                               NALU_HYPRE_Int              part,
                               NALU_HYPRE_Int              nvars,
                               NALU_HYPRE_SStructVariable *vartypes )
{
   nalu_hypre_SStructPGrid  *pgrid = nalu_hypre_SStructGridPGrid(grid, part);

   nalu_hypre_SStructPGridSetVariables(pgrid, nvars, vartypes);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridAddVariables( NALU_HYPRE_SStructGrid      grid,
                               NALU_HYPRE_Int              part,
                               NALU_HYPRE_Int             *index,
                               NALU_HYPRE_Int              nvars,
                               NALU_HYPRE_SStructVariable *vartypes )
{
   NALU_HYPRE_Int            ndim    = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int            nucvars = nalu_hypre_SStructGridNUCVars(grid);
   nalu_hypre_SStructUCVar **ucvars  = nalu_hypre_SStructGridUCVars(grid);
   nalu_hypre_SStructUCVar  *ucvar;

   NALU_HYPRE_Int            memchunk = 1000;
   NALU_HYPRE_Int            i;

   /* allocate more space if necessary */
   if ((nucvars % memchunk) == 0)
   {
      ucvars = nalu_hypre_TReAlloc(ucvars,  nalu_hypre_SStructUCVar *,
                              (nucvars + memchunk), NALU_HYPRE_MEMORY_HOST);
   }

   ucvar = nalu_hypre_TAlloc(nalu_hypre_SStructUCVar,  1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructUCVarUVars(ucvar) = nalu_hypre_TAlloc(nalu_hypre_SStructUVar,  nvars, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructUCVarPart(ucvar) = part;
   nalu_hypre_CopyToCleanIndex(index, ndim, nalu_hypre_SStructUCVarCell(ucvar));
   nalu_hypre_SStructUCVarNUVars(ucvar) = nvars;
   for (i = 0; i < nvars; i++)
   {
      nalu_hypre_SStructUCVarType(ucvar, i) = vartypes[i];
      nalu_hypre_SStructUCVarRank(ucvar, i) = -1;           /* don't know, yet */
      nalu_hypre_SStructUCVarProc(ucvar, i) = -1;           /* don't know, yet */
   }
   ucvars[nucvars] = ucvar;
   nucvars++;

   nalu_hypre_SStructGridNUCVars(grid) = nucvars;
   nalu_hypre_SStructGridUCVars(grid)  = ucvars;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * If ordering == NULL, use a default ordering.  This feature is mainly for
 * internal implementation reasons.
 *--------------------------------------------------------------------------*/

/* ONLY3D */

NALU_HYPRE_Int
NALU_HYPRE_SStructGridSetFEMOrdering( NALU_HYPRE_SStructGrid  grid,
                                 NALU_HYPRE_Int          part,
                                 NALU_HYPRE_Int         *ordering )
{
   NALU_HYPRE_Int               ndim     = nalu_hypre_SStructGridNDim(grid);
   nalu_hypre_SStructPGrid     *pgrid    = nalu_hypre_SStructGridPGrid(grid, part);
   NALU_HYPRE_Int               nvars    = nalu_hypre_SStructPGridNVars(pgrid);
   NALU_HYPRE_SStructVariable  *vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);
   NALU_HYPRE_Int               fem_nvars;
   NALU_HYPRE_Int              *fem_vars;
   nalu_hypre_Index            *fem_offsets;
   nalu_hypre_Index             varoffset;
   NALU_HYPRE_Int               i, j, d, nv, *block, off[3], loop[3];
   NALU_HYPRE_Int               clean = 0;

   /* compute fem_nvars */
   fem_nvars = 0;
   for (i = 0; i < nvars; i++)
   {
      nv = 1;
      nalu_hypre_SStructVariableGetOffset(vartypes[i], ndim, varoffset);
      for (d = 0; d < ndim; d++)
      {
         if (varoffset[d])
         {
            nv *= 2;
         }
      }
      fem_nvars += nv;
   }

   /* set default ordering */
   if (ordering == NULL)
   {
      clean = 1;
      ordering = nalu_hypre_TAlloc(NALU_HYPRE_Int,  (1 + ndim) * fem_nvars, NALU_HYPRE_MEMORY_HOST);
      j = 0;
      for (i = 0; i < nvars; i++)
      {
         nalu_hypre_SStructVariableGetOffset(vartypes[i], ndim, varoffset);
         for (d = 0; d < 3; d++)
         {
            loop[d] = 0;
            if ((d < ndim) && (varoffset[d] != 0))
            {
               loop[d] = 1;
            }
         }
         for (off[2] = -loop[2]; off[2] <= loop[2]; off[2] += 2)
         {
            for (off[1] = -loop[1]; off[1] <= loop[1]; off[1] += 2)
            {
               for (off[0] = -loop[0]; off[0] <= loop[0]; off[0] += 2)
               {
                  block = &ordering[(1 + ndim) * j];
                  block[0] = i;
                  for (d = 0; d < ndim; d++)
                  {
                     block[1 + d] = off[d];
                  }
                  j++;
               }
            }
         }
      }
   }

   fem_vars    = nalu_hypre_TReAlloc(nalu_hypre_SStructGridFEMPVars(grid, part), NALU_HYPRE_Int, fem_nvars,
                                NALU_HYPRE_MEMORY_HOST);
   fem_offsets = nalu_hypre_TReAlloc(nalu_hypre_SStructGridFEMPOffsets(grid, part), nalu_hypre_Index, fem_nvars,
                                NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < fem_nvars; i++)
   {
      block = &ordering[(1 + ndim) * i];
      fem_vars[i] = block[0];
      nalu_hypre_SetIndex(fem_offsets[i], 0);
      for (d = 0; d < ndim; d++)
      {
         /* modify the user offsets to contain only 0's and -1's */
         if (block[1 + d] < 0)
         {
            nalu_hypre_IndexD(fem_offsets[i], d) = -1;
         }
      }
   }

   nalu_hypre_SStructGridFEMPNVars(grid, part)   = fem_nvars;
   nalu_hypre_SStructGridFEMPVars(grid, part)    = fem_vars;
   nalu_hypre_SStructGridFEMPOffsets(grid, part) = fem_offsets;

   if (clean)
   {
      nalu_hypre_TFree(ordering, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridSetNeighborPart( NALU_HYPRE_SStructGrid  grid,
                                  NALU_HYPRE_Int          part,
                                  NALU_HYPRE_Int         *ilower,
                                  NALU_HYPRE_Int         *iupper,
                                  NALU_HYPRE_Int          nbor_part,
                                  NALU_HYPRE_Int         *nbor_ilower,
                                  NALU_HYPRE_Int         *nbor_iupper,
                                  NALU_HYPRE_Int         *index_map,
                                  NALU_HYPRE_Int         *index_dir )
{
   NALU_HYPRE_Int                ndim         = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int               *nneighbors   = nalu_hypre_SStructGridNNeighbors(grid);
   nalu_hypre_SStructNeighbor  **neighbors    = nalu_hypre_SStructGridNeighbors(grid);
   nalu_hypre_Index            **nbor_offsets = nalu_hypre_SStructGridNborOffsets(grid);
   nalu_hypre_SStructNeighbor   *neighbor;
   nalu_hypre_IndexRef           nbor_offset;

   nalu_hypre_Box               *box;
   nalu_hypre_Index              cilower;
   nalu_hypre_Index              ciupper;
   nalu_hypre_IndexRef           coord, dir, ilower_mapped;
   NALU_HYPRE_Int                memchunk = 10;
   NALU_HYPRE_Int                d, dd, tdir;

   /* allocate more memory if needed */
   if ((nneighbors[part] % memchunk) == 0)
   {
      neighbors[part] = nalu_hypre_TReAlloc(neighbors[part],  nalu_hypre_SStructNeighbor,
                                       (nneighbors[part] + memchunk), NALU_HYPRE_MEMORY_HOST);
      nbor_offsets[part] = nalu_hypre_TReAlloc(nbor_offsets[part],  nalu_hypre_Index,
                                          (nneighbors[part] + memchunk), NALU_HYPRE_MEMORY_HOST);
   }

   neighbor = &neighbors[part][nneighbors[part]];
   nbor_offset = nbor_offsets[part][nneighbors[part]];

   box = nalu_hypre_SStructNeighborBox(neighbor);
   nalu_hypre_CopyToCleanIndex(ilower, ndim, cilower);
   nalu_hypre_CopyToCleanIndex(iupper, ndim, ciupper);
   nalu_hypre_BoxInit(box, ndim);
   nalu_hypre_BoxSetExtents(box, cilower, ciupper);
   nalu_hypre_SetIndex(nbor_offset, 0);

   /* If the neighbor box is empty, return */
   if ( !(nalu_hypre_BoxVolume(box) > 0) )
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_SStructNeighborPart(neighbor) = nbor_part;

   coord = nalu_hypre_SStructNeighborCoord(neighbor);
   dir = nalu_hypre_SStructNeighborDir(neighbor);
   ilower_mapped = nalu_hypre_SStructNeighborILower(neighbor);
   nalu_hypre_CopyIndex(index_map, coord);
   nalu_hypre_CopyIndex(index_dir, dir);
   for (d = 0; d < ndim; d++)
   {
      dd = coord[d];
      tdir = dir[d];
      /* this effectively sorts nbor_ilower and nbor_iupper */
      if (nalu_hypre_IndexD(nbor_ilower, dd) > nalu_hypre_IndexD(nbor_iupper, dd))
      {
         tdir = -tdir;
      }
      if (tdir > 0)
      {
         nalu_hypre_IndexD(ilower_mapped, dd) = nalu_hypre_IndexD(nbor_ilower, dd);
      }
      else
      {
         nalu_hypre_IndexD(ilower_mapped, dd) = nalu_hypre_IndexD(nbor_iupper, dd);
      }
   }
   for (d = ndim; d < ndim; d++)
   {
      nalu_hypre_IndexD(coord, d) = d;
      nalu_hypre_IndexD(dir, d) = 1;
      nalu_hypre_IndexD(ilower_mapped, d) = 0;
   }

   nneighbors[part]++;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridSetSharedPart( NALU_HYPRE_SStructGrid  grid,
                                NALU_HYPRE_Int          part,
                                NALU_HYPRE_Int         *ilower,
                                NALU_HYPRE_Int         *iupper,
                                NALU_HYPRE_Int         *offset,
                                NALU_HYPRE_Int          shared_part,
                                NALU_HYPRE_Int         *shared_ilower,
                                NALU_HYPRE_Int         *shared_iupper,
                                NALU_HYPRE_Int         *shared_offset,
                                NALU_HYPRE_Int         *index_map,
                                NALU_HYPRE_Int         *index_dir )
{
   NALU_HYPRE_Int                ndim       = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int               *nneighbors = nalu_hypre_SStructGridNNeighbors(grid);
   nalu_hypre_SStructNeighbor  **neighbors  = nalu_hypre_SStructGridNeighbors(grid);
   nalu_hypre_Index            **nbor_offsets = nalu_hypre_SStructGridNborOffsets(grid);
   nalu_hypre_SStructNeighbor   *neighbor;
   nalu_hypre_IndexRef           nbor_offset;

   nalu_hypre_Box               *box;
   nalu_hypre_Index              cilower;
   nalu_hypre_Index              ciupper;
   nalu_hypre_IndexRef           coord, dir, ilower_mapped;
   NALU_HYPRE_Int                offset_mapped[NALU_HYPRE_MAXDIM];
   NALU_HYPRE_Int                memchunk = 10;
   NALU_HYPRE_Int                d, dd, tdir;

   /* allocate more memory if needed */
   if ((nneighbors[part] % memchunk) == 0)
   {
      neighbors[part] = nalu_hypre_TReAlloc(neighbors[part],  nalu_hypre_SStructNeighbor,
                                       (nneighbors[part] + memchunk), NALU_HYPRE_MEMORY_HOST);
      nbor_offsets[part] = nalu_hypre_TReAlloc(nbor_offsets[part],  nalu_hypre_Index,
                                          (nneighbors[part] + memchunk), NALU_HYPRE_MEMORY_HOST);
   }

   neighbor = &neighbors[part][nneighbors[part]];
   nbor_offset = nbor_offsets[part][nneighbors[part]];

   box = nalu_hypre_SStructNeighborBox(neighbor);
   nalu_hypre_CopyToCleanIndex(ilower, ndim, cilower);
   nalu_hypre_CopyToCleanIndex(iupper, ndim, ciupper);
   nalu_hypre_BoxInit(box, ndim);
   nalu_hypre_BoxSetExtents(box, cilower, ciupper);
   nalu_hypre_CopyToCleanIndex(offset, ndim, nbor_offset);

   /* If the neighbor box is empty, return */
   if ( !(nalu_hypre_BoxVolume(box) > 0) )
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_SStructNeighborPart(neighbor) = shared_part;

   coord = nalu_hypre_SStructNeighborCoord(neighbor);
   dir = nalu_hypre_SStructNeighborDir(neighbor);
   ilower_mapped = nalu_hypre_SStructNeighborILower(neighbor);
   nalu_hypre_CopyIndex(index_map, coord);
   nalu_hypre_CopyIndex(index_dir, dir);
   for (d = 0; d < ndim; d++)
   {
      dd = coord[d];
      tdir = dir[d];
      /* this effectively sorts shared_ilower and shared_iupper */
      if (nalu_hypre_IndexD(shared_ilower, dd) > nalu_hypre_IndexD(shared_iupper, dd))
      {
         tdir = -tdir;
      }
      if (tdir > 0)
      {
         nalu_hypre_IndexD(ilower_mapped, dd) = nalu_hypre_IndexD(shared_ilower, dd);
      }
      else
      {
         nalu_hypre_IndexD(ilower_mapped, dd) = nalu_hypre_IndexD(shared_iupper, dd);
      }
      /* Map the offset to the neighbor part and adjust ilower_mapped so that
       * NeighborILower is a direct mapping of NeighborBoxIMin.  This allows us
       * to eliminate shared_offset. */
      offset_mapped[dd] = offset[d] * dir[d];
      if (offset_mapped[dd] != shared_offset[dd])
      {
         nalu_hypre_IndexD(ilower_mapped, dd) -= offset_mapped[dd];
      }
   }
   for (d = ndim; d < NALU_HYPRE_MAXDIM; d++)
   {
      nalu_hypre_IndexD(coord, d) = d;
      nalu_hypre_IndexD(dir, d) = 1;
      nalu_hypre_IndexD(ilower_mapped, d) = 0;
   }

   nneighbors[part]++;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * *** placeholder ***
 *--------------------------------------------------------------------------*/

#if 0
NALU_HYPRE_Int
NALU_HYPRE_SStructGridAddUnstructuredPart( NALU_HYPRE_SStructGrid grid,
                                      NALU_HYPRE_Int        ilower,
                                      NALU_HYPRE_Int        iupper )
{
   nalu_hypre_SStructGridAssemble(grid);

   return nalu_hypre_error_flag;
}
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridAssemble( NALU_HYPRE_SStructGrid grid )
{
   NALU_HYPRE_Int                ndim         = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int                nparts       = nalu_hypre_SStructGridNParts(grid);
   nalu_hypre_SStructPGrid     **pgrids       = nalu_hypre_SStructGridPGrids(grid);
   NALU_HYPRE_Int               *nneighbors   = nalu_hypre_SStructGridNNeighbors(grid);
   nalu_hypre_SStructNeighbor  **neighbors    = nalu_hypre_SStructGridNeighbors(grid);
   nalu_hypre_Index            **nbor_offsets = nalu_hypre_SStructGridNborOffsets(grid);
   NALU_HYPRE_Int              **nvneighbors  = nalu_hypre_SStructGridNVNeighbors(grid);
   nalu_hypre_SStructNeighbor ***vneighbors   = nalu_hypre_SStructGridVNeighbors(grid);
   nalu_hypre_SStructNeighbor   *neighbor;
   nalu_hypre_IndexRef           nbor_offset;
   nalu_hypre_SStructNeighbor   *vneighbor;
   NALU_HYPRE_Int               *coord, *dir;
   nalu_hypre_Index             *fr_roots, *to_roots;
   nalu_hypre_BoxArrayArray     *nbor_boxes;
   nalu_hypre_BoxArray          *nbor_boxa;
   nalu_hypre_BoxArray          *sub_boxa;
   nalu_hypre_BoxArray          *tmp_boxa;
   nalu_hypre_Box               *nbor_box, *box;
   nalu_hypre_SStructPGrid      *pgrid;
   NALU_HYPRE_SStructVariable   *vartypes;
   nalu_hypre_Index              varoffset;
   NALU_HYPRE_Int                nvars;
   NALU_HYPRE_Int                part, var, b, vb, d, i, valid;
   NALU_HYPRE_Int                nbor_part, sub_part;

   /*-------------------------------------------------------------
    * if I own no data on some part, prune that part's neighbor info
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      if (nalu_hypre_StructGridNumBoxes(nalu_hypre_SStructPGridCellSGrid(pgrid)) == 0)
      {
         nneighbors[part] = 0;
         nalu_hypre_TFree(neighbors[part], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nbor_offsets[part], NALU_HYPRE_MEMORY_HOST);
      }
   }

   /*-------------------------------------------------------------
    * set pneighbors for each pgrid info to crop pgrids
    *-------------------------------------------------------------*/

   /*
    * ZTODO: Note that if neighbor boxes are not first intersected with
    * the global grid, then local pgrid info may be incorrectly cropped.
    * This would occur if users pass in neighbor extents that do not
    * actually live anywhere on the global grid.
    *
    * This is not an issue for cell-centered variables.
    */

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      for (b = 0; b < nneighbors[part]; b++)
      {
         neighbor = &neighbors[part][b];
         nbor_offset = nbor_offsets[part][b];

         /* if this part is not the owner of the shared data */
         if ( part > nalu_hypre_SStructNeighborPart(neighbor) )
         {
            nalu_hypre_SStructPGridSetPNeighbor(
               pgrid, nalu_hypre_SStructNeighborBox(neighbor), nbor_offset);
         }
      }
   }

   /*-------------------------------------------------------------
    * assemble the pgrids
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_SStructPGridAssemble(pgrids[part]);
   }

   /*-------------------------------------------------------------
    * re-organize u-variables to reference via local cell rank
    *-------------------------------------------------------------*/

   /* TODO */

   /*-------------------------------------------------------------
    * determine a unique u-variable data distribution
    *-------------------------------------------------------------*/

   /* TODO */

   /*-------------------------------------------------------------
    * set up the size info
    * GEC0902 calculation of the local ghost size for grid
    *-------------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nalu_hypre_SStructGridLocalSize(grid)   += nalu_hypre_SStructPGridLocalSize(pgrid);
      nalu_hypre_SStructGridGlobalSize(grid)  += nalu_hypre_SStructPGridGlobalSize(pgrid);
      nalu_hypre_SStructGridGhlocalSize(grid) += nalu_hypre_SStructPGridGhlocalSize(pgrid);
   }

   /*-------------------------------------------------
    * Set up the FEM ordering information
    *-------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      if (nalu_hypre_SStructGridFEMPNVars(grid, part) == 0)
      {
         /* use the default ordering */
         NALU_HYPRE_SStructGridSetFEMOrdering(grid, part, NULL);
      }
   }

   /*-------------------------------------------------
    * Set up vneighbor info
    *-------------------------------------------------*/

   box = nalu_hypre_BoxCreate(ndim);
   tmp_boxa = nalu_hypre_BoxArrayCreate(0, ndim);

   nvneighbors = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nparts, NALU_HYPRE_MEMORY_HOST);
   vneighbors  = nalu_hypre_TAlloc(nalu_hypre_SStructNeighbor **,  nparts, NALU_HYPRE_MEMORY_HOST);

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);
      nvneighbors[part] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nvars, NALU_HYPRE_MEMORY_HOST);
      vneighbors[part]  = nalu_hypre_TAlloc(nalu_hypre_SStructNeighbor *,  nvars, NALU_HYPRE_MEMORY_HOST);

      for (var = 0; var < nvars; var++)
      {
         /* Put each new vneighbor box into a BoxArrayArray so we can remove overlap */
         nbor_boxes = nalu_hypre_BoxArrayArrayCreate(nneighbors[part], ndim);
         fr_roots = nalu_hypre_TAlloc(nalu_hypre_Index,  nneighbors[part], NALU_HYPRE_MEMORY_HOST);
         to_roots = nalu_hypre_TAlloc(nalu_hypre_Index,  nneighbors[part], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_SStructVariableGetOffset((nalu_hypre_SStructVariable) vartypes[var], ndim, varoffset);
         nvneighbors[part][var] = 0;
         for (b = 0; b < nneighbors[part]; b++)
         {
            neighbor    = &neighbors[part][b];
            nbor_offset = nbor_offsets[part][b];

            /* Create var-centered vneighbor box from cell-centered neighbor box */
            nalu_hypre_CopyBox(nalu_hypre_SStructNeighborBox(neighbor), box);
            nalu_hypre_SStructCellBoxToVarBox(box, nbor_offset, varoffset, &valid);
            /* Sometimes we can't construct vneighbor boxes (valid = false).
             * For example, if only faces are shared (see SetSharedPart), then
             * there should be no vneighbor boxes for cell variables.  Note that
             * we ensure nonempty neighbor boxes when they are set up. */
            if (!valid)
            {
               continue;
            }

            /* Save root mapping information for later */
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(box), fr_roots[b]);
            nalu_hypre_CopyIndex(nalu_hypre_SStructNeighborILower(neighbor), to_roots[b]);

            /* It's important to adjust to_root (ilower) */
            coord = nalu_hypre_SStructNeighborCoord(neighbor);
            dir   = nalu_hypre_SStructNeighborDir(neighbor);
            for (d = 0; d < ndim; d++)
            {
               /* Compare the imin of the neighbor cell box ('i') to its imin
                * value after being converted to a variable box ('IMin(box,d)').
                * If the coordinates in the two parts move in the same direction
                * (i.e., dir[d] > 0) and the local imin changed, then also
                * change the corresponding neighbor ilower.  If the coordinates
                * in the two parts move in opposite directions and the local
                * imin did not change, then change the corresponding neighbor
                * ilower based on the value of 'varoffset'. */
               i = nalu_hypre_BoxIMinD(nalu_hypre_SStructNeighborBox(neighbor), d);
               if (((dir[d] > 0) && (nalu_hypre_BoxIMinD(box, d) != i)) ||
                   ((dir[d] < 0) && (nalu_hypre_BoxIMinD(box, d) == i)))
               {
                  nalu_hypre_IndexD(to_roots[b], coord[d]) -= nalu_hypre_IndexD(varoffset, d);
               }
            }

            /* Add box to the nbor_boxes */
            nbor_boxa = nalu_hypre_BoxArrayArrayBoxArray(nbor_boxes, b);
            nalu_hypre_AppendBox(box, nbor_boxa);

            /* Make sure that the nbor_boxes don't overlap */
            nbor_part = nalu_hypre_SStructNeighborPart(neighbor);
            for (i = 0; i < b; i++)
            {
               neighbor = &neighbors[part][i];
               sub_part = nalu_hypre_SStructNeighborPart(neighbor);
               /* Only subtract boxes on the same neighbor part */
               if (nbor_part == sub_part)
               {
                  sub_boxa = nalu_hypre_BoxArrayArrayBoxArray(nbor_boxes, i);
                  /* nbor_boxa -= sub_boxa */
                  nalu_hypre_SubtractBoxArrays(nbor_boxa, sub_boxa, tmp_boxa);
               }
            }

            nvneighbors[part][var] += nalu_hypre_BoxArraySize(nbor_boxa);
         }

         /* Set up vneighbors for this (part, var) */
         vneighbors[part][var] = nalu_hypre_TAlloc(nalu_hypre_SStructNeighbor,  nvneighbors[part][var],
                                              NALU_HYPRE_MEMORY_HOST);
         vb = 0;
         for (b = 0; b < nneighbors[part]; b++)
         {
            neighbor  = &neighbors[part][b];
            nbor_boxa = nalu_hypre_BoxArrayArrayBoxArray(nbor_boxes, b);
            nbor_part = nalu_hypre_SStructNeighborPart(neighbor);
            coord     = nalu_hypre_SStructNeighborCoord(neighbor);
            dir       = nalu_hypre_SStructNeighborDir(neighbor);
            nalu_hypre_ForBoxI(i, nbor_boxa)
            {
               vneighbor = &vneighbors[part][var][vb];
               nbor_box = nalu_hypre_BoxArrayBox(nbor_boxa, i);

               nalu_hypre_CopyBox(nbor_box, nalu_hypre_SStructNeighborBox(vneighbor));
               nalu_hypre_SStructNeighborPart(vneighbor) = nbor_part;
               nalu_hypre_SStructIndexToNborIndex(nalu_hypre_BoxIMin(nbor_box),
                                             fr_roots[b], to_roots[b], coord, dir, ndim,
                                             nalu_hypre_SStructNeighborILower(vneighbor));
               nalu_hypre_CopyIndex(coord, nalu_hypre_SStructNeighborCoord(vneighbor));
               nalu_hypre_CopyIndex(dir, nalu_hypre_SStructNeighborDir(vneighbor));

               vb++;
            }

         } /* end of vneighbor box loop */

         nalu_hypre_BoxArrayArrayDestroy(nbor_boxes);
         nalu_hypre_TFree(fr_roots, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(to_roots, NALU_HYPRE_MEMORY_HOST);

      } /* end of variables loop */
   } /* end of part loop */

   nalu_hypre_SStructGridNVNeighbors(grid) = nvneighbors;
   nalu_hypre_SStructGridVNeighbors(grid)  = vneighbors;

   nalu_hypre_BoxArrayDestroy(tmp_boxa);
   nalu_hypre_BoxDestroy(box);

   /*-------------------------------------------------
    * Assemble the box manager info
    *-------------------------------------------------*/

   nalu_hypre_SStructGridAssembleBoxManagers(grid);

   /*-------------------------------------------------
    * Assemble the neighbor box manager info
    *-------------------------------------------------*/

   nalu_hypre_SStructGridAssembleNborBoxManagers(grid);

   /*-------------------------------------------------
    * Compute the CommInfo component of the grid
    *-------------------------------------------------*/

   nalu_hypre_SStructGridCreateCommInfo(grid);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridSetPeriodic( NALU_HYPRE_SStructGrid  grid,
                              NALU_HYPRE_Int          part,
                              NALU_HYPRE_Int         *periodic )
{
   nalu_hypre_SStructPGrid *pgrid          = nalu_hypre_SStructGridPGrid(grid, part);
   nalu_hypre_IndexRef      pgrid_periodic = nalu_hypre_SStructPGridPeriodic(pgrid);
   NALU_HYPRE_Int           d;

   for (d = 0; d < nalu_hypre_SStructGridNDim(grid); d++)
   {
      nalu_hypre_IndexD(pgrid_periodic, d) = periodic[d];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC0902 a placeholder for a internal function that will set ghosts in each
 * of the sgrids of the grid
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGridSetNumGhost( NALU_HYPRE_SStructGrid grid,
                              NALU_HYPRE_Int      *num_ghost)
{
   nalu_hypre_SStructGridSetNumGhost(grid, num_ghost);

   return nalu_hypre_error_flag;
}
