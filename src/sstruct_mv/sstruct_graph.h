/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_SStructGraph structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_SSTRUCT_GRAPH_HEADER
#define nalu_hypre_SSTRUCT_GRAPH_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructGraph:
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int     part;
   nalu_hypre_Index   index;
   NALU_HYPRE_Int     var;
   NALU_HYPRE_Int     to_part;
   nalu_hypre_Index   to_index;
   NALU_HYPRE_Int     to_var;

} nalu_hypre_SStructGraphEntry;

typedef struct
{
   NALU_HYPRE_Int     to_part;
   nalu_hypre_Index   to_index;
   NALU_HYPRE_Int     to_var;
   NALU_HYPRE_Int     to_boxnum;      /* local box number */
   NALU_HYPRE_Int     to_proc;
   NALU_HYPRE_Int     to_rank;

} nalu_hypre_SStructUEntry;

typedef struct
{
   NALU_HYPRE_Int            part;
   nalu_hypre_Index          index;
   NALU_HYPRE_Int            var;
   NALU_HYPRE_Int            rank;
   NALU_HYPRE_Int            nUentries;
   nalu_hypre_SStructUEntry *Uentries;

} nalu_hypre_SStructUVEntry;

typedef struct nalu_hypre_SStructGraph_struct
{
   MPI_Comm                comm;
   NALU_HYPRE_Int               ndim;
   nalu_hypre_SStructGrid      *grid;
   nalu_hypre_SStructGrid      *domain_grid; /* same as grid by default */
   NALU_HYPRE_Int               nparts;
   nalu_hypre_SStructPGrid    **pgrids;
   nalu_hypre_SStructStencil ***stencils; /* each (part, var) has a stencil */

   /* info for fem-based user input */
   NALU_HYPRE_Int              *fem_nsparse;
   NALU_HYPRE_Int             **fem_sparse_i;
   NALU_HYPRE_Int             **fem_sparse_j;
   NALU_HYPRE_Int             **fem_entries;

   /* U-graph info: Entries are referenced via a local rank that comes from an
    * ordering of the local grid boxes with ghost zones added. */
   NALU_HYPRE_Int               nUventries; /* number of Uventries */
   NALU_HYPRE_Int              *iUventries; /* rank indexes into Uventries */
   nalu_hypre_SStructUVEntry  **Uventries;
   NALU_HYPRE_Int               Uvesize;    /* size of Uventries array */
   NALU_HYPRE_Int               Uemaxsize;  /* max size of Uentries */
   NALU_HYPRE_BigInt          **Uveoffsets; /* offsets for computing rank indexes */

   NALU_HYPRE_Int               ref_count;

   NALU_HYPRE_Int               type;    /* GEC0203 */

   /* These are created in GraphAddEntries() then deleted in GraphAssemble() */
   nalu_hypre_SStructGraphEntry **graph_entries;
   NALU_HYPRE_Int               n_graph_entries; /* number graph entries */
   NALU_HYPRE_Int               a_graph_entries; /* alloced graph entries */

} nalu_hypre_SStructGraph;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructGraph
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructGraphComm(graph)           ((graph) -> comm)
#define nalu_hypre_SStructGraphNDim(graph)           ((graph) -> ndim)
#define nalu_hypre_SStructGraphGrid(graph)           ((graph) -> grid)
#define nalu_hypre_SStructGraphDomainGrid(graph)     ((graph) -> domain_grid)
#define nalu_hypre_SStructGraphNParts(graph)         ((graph) -> nparts)
#define nalu_hypre_SStructGraphPGrids(graph) \
   nalu_hypre_SStructGridPGrids(nalu_hypre_SStructGraphGrid(graph))
#define nalu_hypre_SStructGraphPGrid(graph, p) \
   nalu_hypre_SStructGridPGrid(nalu_hypre_SStructGraphGrid(graph), p)
#define nalu_hypre_SStructGraphStencils(graph)       ((graph) -> stencils)
#define nalu_hypre_SStructGraphStencil(graph, p, v)  ((graph) -> stencils[p][v])

#define nalu_hypre_SStructGraphFEMNSparse(graph)     ((graph) -> fem_nsparse)
#define nalu_hypre_SStructGraphFEMSparseI(graph)     ((graph) -> fem_sparse_i)
#define nalu_hypre_SStructGraphFEMSparseJ(graph)     ((graph) -> fem_sparse_j)
#define nalu_hypre_SStructGraphFEMEntries(graph)     ((graph) -> fem_entries)
#define nalu_hypre_SStructGraphFEMPNSparse(graph, p) ((graph) -> fem_nsparse[p])
#define nalu_hypre_SStructGraphFEMPSparseI(graph, p) ((graph) -> fem_sparse_i[p])
#define nalu_hypre_SStructGraphFEMPSparseJ(graph, p) ((graph) -> fem_sparse_j[p])
#define nalu_hypre_SStructGraphFEMPEntries(graph, p) ((graph) -> fem_entries[p])

#define nalu_hypre_SStructGraphNUVEntries(graph)     ((graph) -> nUventries)
#define nalu_hypre_SStructGraphIUVEntries(graph)     ((graph) -> iUventries)
#define nalu_hypre_SStructGraphIUVEntry(graph, i)    ((graph) -> iUventries[i])
#define nalu_hypre_SStructGraphUVEntries(graph)      ((graph) -> Uventries)
#define nalu_hypre_SStructGraphUVEntry(graph, i)     ((graph) -> Uventries[i])
#define nalu_hypre_SStructGraphUVESize(graph)        ((graph) -> Uvesize)
#define nalu_hypre_SStructGraphUEMaxSize(graph)      ((graph) -> Uemaxsize)
#define nalu_hypre_SStructGraphUVEOffsets(graph)     ((graph) -> Uveoffsets)
#define nalu_hypre_SStructGraphUVEOffset(graph, p, v)((graph) -> Uveoffsets[p][v])

#define nalu_hypre_SStructGraphRefCount(graph)       ((graph) -> ref_count)
#define nalu_hypre_SStructGraphObjectType(graph)     ((graph) -> type)
#define nalu_hypre_SStructGraphEntries(graph)        ((graph) -> graph_entries)
#define nalu_hypre_SStructNGraphEntries(graph)       ((graph) -> n_graph_entries)
#define nalu_hypre_SStructAGraphEntries(graph)       ((graph) -> a_graph_entries)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructUVEntry
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructUVEntryPart(Uv)        ((Uv) -> part)
#define nalu_hypre_SStructUVEntryIndex(Uv)       ((Uv) -> index)
#define nalu_hypre_SStructUVEntryVar(Uv)         ((Uv) -> var)
#define nalu_hypre_SStructUVEntryRank(Uv)        ((Uv) -> rank)
#define nalu_hypre_SStructUVEntryNUEntries(Uv)   ((Uv) -> nUentries)
#define nalu_hypre_SStructUVEntryUEntries(Uv)    ((Uv) -> Uentries)
#define nalu_hypre_SStructUVEntryUEntry(Uv, i)  &((Uv) -> Uentries[i])
#define nalu_hypre_SStructUVEntryToPart(Uv, i)   ((Uv) -> Uentries[i].to_part)
#define nalu_hypre_SStructUVEntryToIndex(Uv, i)  ((Uv) -> Uentries[i].to_index)
#define nalu_hypre_SStructUVEntryToVar(Uv, i)    ((Uv) -> Uentries[i].to_var)
#define nalu_hypre_SStructUVEntryToBoxnum(Uv, i) ((Uv) -> Uentries[i].to_boxnum)
#define nalu_hypre_SStructUVEntryToProc(Uv, i)   ((Uv) -> Uentries[i].to_proc)
#define nalu_hypre_SStructUVEntryToRank(Uv, i)   ((Uv) -> Uentries[i].to_rank)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructUEntry
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructUEntryToPart(U)   ((U) -> to_part)
#define nalu_hypre_SStructUEntryToIndex(U)  ((U) -> to_index)
#define nalu_hypre_SStructUEntryToVar(U)    ((U) -> to_var)
#define nalu_hypre_SStructUEntryToBoxnum(U) ((U) -> to_boxnum)
#define nalu_hypre_SStructUEntryToProc(U)   ((U) -> to_proc)
#define nalu_hypre_SStructUEntryToRank(U)   ((U) -> to_rank)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructGraphEntry
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructGraphEntryPart(g)     ((g) -> part)
#define nalu_hypre_SStructGraphEntryIndex(g)    ((g) -> index)
#define nalu_hypre_SStructGraphEntryVar(g)      ((g) -> var)
#define nalu_hypre_SStructGraphEntryToPart(g)   ((g) -> to_part)
#define nalu_hypre_SStructGraphEntryToIndex(g)  ((g) -> to_index)
#define nalu_hypre_SStructGraphEntryToVar(g)    ((g) -> to_var)

#endif
