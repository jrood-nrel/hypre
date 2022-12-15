/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_SStructGrid structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_SSTRUCT_GRID_HEADER
#define nalu_hypre_SSTRUCT_GRID_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructGrid:
 *
 * NOTE: Since variables may be replicated across different processes,
 * a separate set of "interface grids" is retained so that data can be
 * migrated onto and off of the internal (non-replicated) grids.
 *--------------------------------------------------------------------------*/

typedef NALU_HYPRE_Int nalu_hypre_SStructVariable;

typedef struct
{
   NALU_HYPRE_SStructVariable  type;
   NALU_HYPRE_Int              rank;     /* local rank */
   NALU_HYPRE_Int              proc;

} nalu_hypre_SStructUVar;

typedef struct
{
   NALU_HYPRE_Int              part;
   nalu_hypre_Index            cell;
   NALU_HYPRE_Int              nuvars;
   nalu_hypre_SStructUVar     *uvars;

} nalu_hypre_SStructUCVar;

typedef struct
{
   MPI_Comm                comm;             /* TODO: use different comms */
   NALU_HYPRE_Int               ndim;
   NALU_HYPRE_Int               nvars;            /* number of variables */
   NALU_HYPRE_SStructVariable  *vartypes;         /* types of variables */
   nalu_hypre_StructGrid       *sgrids[8];        /* struct grids for each vartype */
   nalu_hypre_BoxArray         *iboxarrays[8];    /* interface boxes */

   nalu_hypre_BoxArray         *pneighbors;
   nalu_hypre_Index            *pnbor_offsets;

   NALU_HYPRE_Int               local_size;       /* Number of variables locally */
   NALU_HYPRE_BigInt            global_size;      /* Total number of variables */

   nalu_hypre_Index             periodic;         /* Indicates if pgrid is periodic */

   /* GEC0902 additions for ghost expansion of boxes */

   NALU_HYPRE_Int               ghlocal_size;     /* Number of vars including ghosts */

   NALU_HYPRE_Int               cell_sgrid_done;  /* =1 implies cell grid already assembled */
} nalu_hypre_SStructPGrid;

typedef struct
{
   nalu_hypre_Box    box;
   NALU_HYPRE_Int    part;
   nalu_hypre_Index  ilower; /* box ilower, but on the neighbor index-space */
   nalu_hypre_Index  coord;  /* lives on local index-space */
   nalu_hypre_Index  dir;    /* lives on local index-space */

} nalu_hypre_SStructNeighbor;

enum nalu_hypre_SStructBoxManInfoType
{
   nalu_hypre_SSTRUCT_BOXMAN_INFO_DEFAULT  = 0,
   nalu_hypre_SSTRUCT_BOXMAN_INFO_NEIGHBOR = 1
};

typedef struct
{
   NALU_HYPRE_Int  type;
   NALU_HYPRE_BigInt offset;
   NALU_HYPRE_BigInt ghoffset;

} nalu_hypre_SStructBoxManInfo;

typedef struct
{
   NALU_HYPRE_Int    type;
   NALU_HYPRE_BigInt offset;   /* minimum offset for this box */
   NALU_HYPRE_BigInt ghoffset; /* minimum offset ghost for this box */
   NALU_HYPRE_Int    proc;     /* redundant with the proc in the entry, but
                             makes some coding easier */
   NALU_HYPRE_Int    boxnum;   /* this is different from the entry id */
   NALU_HYPRE_Int    part;     /* part the box lives on */
   nalu_hypre_Index  ilower;   /* box ilower, but on the neighbor index-space */
   nalu_hypre_Index  coord;    /* lives on local index-space */
   nalu_hypre_Index  dir;      /* lives on local index-space */
   nalu_hypre_Index  stride;   /* lives on local index-space */
   nalu_hypre_Index  ghstride; /* the ghost equivalent of strides */

} nalu_hypre_SStructBoxManNborInfo;

typedef struct
{
   nalu_hypre_CommInfo  *comm_info;
   NALU_HYPRE_Int        send_part;
   NALU_HYPRE_Int        recv_part;
   NALU_HYPRE_Int        send_var;
   NALU_HYPRE_Int        recv_var;

} nalu_hypre_SStructCommInfo;

typedef struct nalu_hypre_SStructGrid_struct
{
   MPI_Comm                   comm;
   NALU_HYPRE_Int                  ndim;
   NALU_HYPRE_Int                  nparts;

   /* s-variable info */
   nalu_hypre_SStructPGrid       **pgrids;

   /* neighbor info */
   NALU_HYPRE_Int                 *nneighbors;
   nalu_hypre_SStructNeighbor    **neighbors;
   nalu_hypre_Index              **nbor_offsets;
   NALU_HYPRE_Int                **nvneighbors;
   nalu_hypre_SStructNeighbor   ***vneighbors;
   nalu_hypre_SStructCommInfo    **vnbor_comm_info; /* for updating shared data */
   NALU_HYPRE_Int                  vnbor_ncomms;

   /* u-variables info: During construction, array entries are consecutive.
    * After 'Assemble', entries are referenced via local cell rank. */
   NALU_HYPRE_Int                  nucvars;
   nalu_hypre_SStructUCVar       **ucvars;

   /* info for fem-based user input (for each part) */
   NALU_HYPRE_Int                 *fem_nvars;
   NALU_HYPRE_Int                **fem_vars;
   nalu_hypre_Index              **fem_offsets;

   /* info for mapping (part, index, var) --> rank */
   nalu_hypre_BoxManager        ***boxmans;      /* manager for each part, var */
   nalu_hypre_BoxManager        ***nbor_boxmans; /* manager for each part, var */

   NALU_HYPRE_BigInt               start_rank;

   NALU_HYPRE_Int                  local_size;  /* Number of variables locally */
   NALU_HYPRE_BigInt               global_size; /* Total number of variables */

   NALU_HYPRE_Int                  ref_count;

   /* GEC0902 additions for ghost expansion of boxes */

   NALU_HYPRE_Int               ghlocal_size;  /* GEC0902 Number of vars including ghosts */
   NALU_HYPRE_BigInt            ghstart_rank;  /* GEC0902 start rank including ghosts  */
   NALU_HYPRE_Int               num_ghost[2 * NALU_HYPRE_MAXDIM]; /* ghost layer size */

} nalu_hypre_SStructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructGrid
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructGridComm(grid)           ((grid) -> comm)
#define nalu_hypre_SStructGridNDim(grid)           ((grid) -> ndim)
#define nalu_hypre_SStructGridNParts(grid)         ((grid) -> nparts)
#define nalu_hypre_SStructGridPGrids(grid)         ((grid) -> pgrids)
#define nalu_hypre_SStructGridPGrid(grid, part)    ((grid) -> pgrids[part])
#define nalu_hypre_SStructGridNNeighbors(grid)     ((grid) -> nneighbors)
#define nalu_hypre_SStructGridNeighbors(grid)      ((grid) -> neighbors)
#define nalu_hypre_SStructGridNborOffsets(grid)    ((grid) -> nbor_offsets)
#define nalu_hypre_SStructGridNVNeighbors(grid)    ((grid) -> nvneighbors)
#define nalu_hypre_SStructGridVNeighbors(grid)     ((grid) -> vneighbors)
#define nalu_hypre_SStructGridVNborCommInfo(grid)  ((grid) -> vnbor_comm_info)
#define nalu_hypre_SStructGridVNborNComms(grid)    ((grid) -> vnbor_ncomms)
#define nalu_hypre_SStructGridNUCVars(grid)        ((grid) -> nucvars)
#define nalu_hypre_SStructGridUCVars(grid)         ((grid) -> ucvars)
#define nalu_hypre_SStructGridUCVar(grid, i)       ((grid) -> ucvars[i])

#define nalu_hypre_SStructGridFEMNVars(grid)       ((grid) -> fem_nvars)
#define nalu_hypre_SStructGridFEMVars(grid)        ((grid) -> fem_vars)
#define nalu_hypre_SStructGridFEMOffsets(grid)     ((grid) -> fem_offsets)
#define nalu_hypre_SStructGridFEMPNVars(grid, part)   ((grid) -> fem_nvars[part])
#define nalu_hypre_SStructGridFEMPVars(grid, part)    ((grid) -> fem_vars[part])
#define nalu_hypre_SStructGridFEMPOffsets(grid, part) ((grid) -> fem_offsets[part])

#define nalu_hypre_SStructGridBoxManagers(grid)           ((grid) -> boxmans)
#define nalu_hypre_SStructGridBoxManager(grid, part, var) ((grid) -> boxmans[part][var])

#define nalu_hypre_SStructGridNborBoxManagers(grid)           ((grid) -> nbor_boxmans)
#define nalu_hypre_SStructGridNborBoxManager(grid, part, var) ((grid) -> nbor_boxmans[part][var])

#define nalu_hypre_SStructGridStartRank(grid)      ((grid) -> start_rank)
#define nalu_hypre_SStructGridLocalSize(grid)      ((grid) -> local_size)
#define nalu_hypre_SStructGridGlobalSize(grid)     ((grid) -> global_size)
#define nalu_hypre_SStructGridRefCount(grid)       ((grid) -> ref_count)
#define nalu_hypre_SStructGridGhlocalSize(grid)    ((grid) -> ghlocal_size)
#define nalu_hypre_SStructGridGhstartRank(grid)    ((grid) -> ghstart_rank)
#define nalu_hypre_SStructGridNumGhost(grid)       ((grid) -> num_ghost)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructPGrid
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructPGridComm(pgrid)             ((pgrid) -> comm)
#define nalu_hypre_SStructPGridNDim(pgrid)             ((pgrid) -> ndim)
#define nalu_hypre_SStructPGridNVars(pgrid)            ((pgrid) -> nvars)
#define nalu_hypre_SStructPGridVarTypes(pgrid)         ((pgrid) -> vartypes)
#define nalu_hypre_SStructPGridVarType(pgrid, var)     ((pgrid) -> vartypes[var])
#define nalu_hypre_SStructPGridCellSGridDone(pgrid)    ((pgrid) -> cell_sgrid_done)

#define nalu_hypre_SStructPGridSGrids(pgrid)           ((pgrid) -> sgrids)
#define nalu_hypre_SStructPGridSGrid(pgrid, var) \
((pgrid) -> sgrids[nalu_hypre_SStructPGridVarType(pgrid, var)])
#define nalu_hypre_SStructPGridCellSGrid(pgrid) \
((pgrid) -> sgrids[NALU_HYPRE_SSTRUCT_VARIABLE_CELL])
#define nalu_hypre_SStructPGridVTSGrid(pgrid, vartype) ((pgrid) -> sgrids[vartype])

#define nalu_hypre_SStructPGridIBoxArrays(pgrid)       ((pgrid) -> iboxarrays)
#define nalu_hypre_SStructPGridIBoxArray(pgrid, var) \
((pgrid) -> iboxarrays[nalu_hypre_SStructPGridVarType(pgrid, var)])
#define nalu_hypre_SStructPGridCellIBoxArray(pgrid) \
((pgrid) -> iboxarrays[NALU_HYPRE_SSTRUCT_VARIABLE_CELL])
#define nalu_hypre_SStructPGridVTIBoxArray(pgrid, vartype) \
((pgrid) -> iboxarrays[vartype])

#define nalu_hypre_SStructPGridPNeighbors(pgrid)       ((pgrid) -> pneighbors)
#define nalu_hypre_SStructPGridPNborOffsets(pgrid)     ((pgrid) -> pnbor_offsets)
#define nalu_hypre_SStructPGridLocalSize(pgrid)        ((pgrid) -> local_size)
#define nalu_hypre_SStructPGridGlobalSize(pgrid)       ((pgrid) -> global_size)
#define nalu_hypre_SStructPGridPeriodic(pgrid)         ((pgrid) -> periodic)
#define nalu_hypre_SStructPGridGhlocalSize(pgrid)      ((pgrid) -> ghlocal_size)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructBoxManInfo
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructBoxManInfoType(info)            ((info) -> type)
#define nalu_hypre_SStructBoxManInfoOffset(info)          ((info) -> offset)
#define nalu_hypre_SStructBoxManInfoGhoffset(info)        ((info) -> ghoffset)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructBoxManInfo
 *--------------------------------------------------------------------------*/

/* Use the MapInfo macros to access the first three structure components */
#define nalu_hypre_SStructBoxManNborInfoProc(info)    ((info) -> proc)
#define nalu_hypre_SStructBoxManNborInfoBoxnum(info)  ((info) -> boxnum)
#define nalu_hypre_SStructBoxManNborInfoPart(info)    ((info) -> part)
#define nalu_hypre_SStructBoxManNborInfoILower(info)  ((info) -> ilower)
#define nalu_hypre_SStructBoxManNborInfoCoord(info)   ((info) -> coord)
#define nalu_hypre_SStructBoxManNborInfoDir(info)     ((info) -> dir)
#define nalu_hypre_SStructBoxManNborInfoStride(info)  ((info) -> stride)
#define nalu_hypre_SStructBoxManNborInfoGhstride(info)  ((info) -> ghstride)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructNeighbor
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructNeighborBox(neighbor)    &((neighbor) -> box)
#define nalu_hypre_SStructNeighborPart(neighbor)    ((neighbor) -> part)
#define nalu_hypre_SStructNeighborILower(neighbor)  ((neighbor) -> ilower)
#define nalu_hypre_SStructNeighborCoord(neighbor)   ((neighbor) -> coord)
#define nalu_hypre_SStructNeighborDir(neighbor)     ((neighbor) -> dir)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructCommInfo
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructCommInfoCommInfo(cinfo)  ((cinfo) -> comm_info)
#define nalu_hypre_SStructCommInfoSendPart(cinfo)  ((cinfo) -> send_part)
#define nalu_hypre_SStructCommInfoRecvPart(cinfo)  ((cinfo) -> recv_part)
#define nalu_hypre_SStructCommInfoSendVar(cinfo)   ((cinfo) -> send_var)
#define nalu_hypre_SStructCommInfoRecvVar(cinfo)   ((cinfo) -> recv_var)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructUCVar
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructUCVarPart(uc)     ((uc) -> part)
#define nalu_hypre_SStructUCVarCell(uc)     ((uc) -> cell)
#define nalu_hypre_SStructUCVarNUVars(uc)   ((uc) -> nuvars)
#define nalu_hypre_SStructUCVarUVars(uc)    ((uc) -> uvars)
#define nalu_hypre_SStructUCVarType(uc, i)  ((uc) -> uvars[i].type)
#define nalu_hypre_SStructUCVarRank(uc, i)  ((uc) -> uvars[i].rank)
#define nalu_hypre_SStructUCVarProc(uc, i)  ((uc) -> uvars[i].proc)

#endif

