/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_SStructGraph class.
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGraphRef( nalu_hypre_SStructGraph  *graph,
                       nalu_hypre_SStructGraph **graph_ref )
{
   nalu_hypre_SStructGraphRefCount(graph) ++;
   *graph_ref = graph;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Uventries are stored in an array indexed via a local rank that comes from an
 * ordering of the local grid boxes with ghost zones added.  Since a grid index
 * may intersect multiple grid boxes, the box with the smallest boxnum is used.
 *
 * RDF: Consider using another "local" BoxManager to optimize.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGraphGetUVEntryRank( nalu_hypre_SStructGraph    *graph,
                                  NALU_HYPRE_Int              part,
                                  NALU_HYPRE_Int              var,
                                  nalu_hypre_Index            index,
                                  NALU_HYPRE_BigInt          *rank )
{
   NALU_HYPRE_Int              ndim  = nalu_hypre_SStructGraphNDim(graph);
   nalu_hypre_SStructGrid     *grid  = nalu_hypre_SStructGraphGrid(graph);
   nalu_hypre_SStructPGrid    *pgrid = nalu_hypre_SStructGridPGrid(grid, part);
   nalu_hypre_StructGrid      *sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var);
   nalu_hypre_BoxArray        *boxes = nalu_hypre_StructGridBoxes(sgrid);
   nalu_hypre_Box             *box;
   NALU_HYPRE_Int              i, d, vol, found;


   *rank = nalu_hypre_SStructGraphUVEOffset(graph, part, var);
   nalu_hypre_ForBoxI(i, boxes)
   {
      box = nalu_hypre_BoxArrayBox(boxes, i);
      found = 1;
      for (d = 0; d < ndim; d++)
      {
         if ( (nalu_hypre_IndexD(index, d) < (nalu_hypre_BoxIMinD(box, d) - 1)) ||
              (nalu_hypre_IndexD(index, d) > (nalu_hypre_BoxIMaxD(box, d) + 1)) )
         {
            /* not in this box */
            found = 0;
            break;
         }
      }
      if (found)
      {
         vol = 0;
         for (d = (ndim - 1); d > -1; d--)
         {
            vol = vol * (nalu_hypre_BoxSizeD(box, d) + 2) +
                  (nalu_hypre_IndexD(index, d) - nalu_hypre_BoxIMinD(box, d) + 1);
         }
         *rank += (NALU_HYPRE_BigInt)vol;
         return nalu_hypre_error_flag;
      }
      else
      {
         vol = 1;
         for (d = 0; d < ndim; d++)
         {
            vol *= (nalu_hypre_BoxSizeD(box, d) + 2);
         }
         *rank += (NALU_HYPRE_BigInt)vol;
      }
   }

   /* a value of -1 indicates that the index was not found */
   *rank = -1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the local Uventries index for the endpt of a box. This index
 * can be used to localize a search for Uventries of a box.
 *      endpt= 0   start of boxes
 *      endpt= 1   end of boxes

 * 9/09 AB - modified to use the box manager
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGraphFindBoxEndpt(nalu_hypre_SStructGraph    *graph,
                               NALU_HYPRE_Int              part,
                               NALU_HYPRE_Int              var,
                               NALU_HYPRE_Int              proc,
                               NALU_HYPRE_Int              endpt,
                               NALU_HYPRE_Int              boxi)
{
   nalu_hypre_SStructGrid     *grid      = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int              type      = nalu_hypre_SStructGraphObjectType(graph);
   nalu_hypre_BoxManager      *boxman;
   nalu_hypre_BoxManEntry     *boxman_entry;
   nalu_hypre_StructGrid      *sgrid;
   nalu_hypre_Box             *box;
   NALU_HYPRE_BigInt           rank;

   /* Should we be checking the neighbor box manager also ?*/

   boxman = nalu_hypre_SStructGridBoxManager(grid, part, var);
   nalu_hypre_BoxManGetEntry(boxman, proc, boxi, &boxman_entry);

   sgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructGridPGrid(grid, part), var);
   box  = nalu_hypre_StructGridBox(sgrid, boxi);

   /* get the global rank of the endpt corner of box boxi */
   if (endpt < 1)
   {
      nalu_hypre_SStructBoxManEntryGetGlobalRank(
         boxman_entry, nalu_hypre_BoxIMin(box), &rank, type);
   }

   else
   {
      nalu_hypre_SStructBoxManEntryGetGlobalRank(
         boxman_entry, nalu_hypre_BoxIMax(box), &rank, type);
   }

   if (type == NALU_HYPRE_SSTRUCT || type ==  NALU_HYPRE_STRUCT)
   {
      rank -= nalu_hypre_SStructGridGhstartRank(grid);
   }
   if (type == NALU_HYPRE_PARCSR)
   {
      rank -= nalu_hypre_SStructGridStartRank(grid);
   }

   return rank;
}

/*--------------------------------------------------------------------------
 * Computes the local Uventries index for the start or end of each box of
 * a given sgrid.
 *      endpt= 0   start of boxes
 *      endpt= 1   end of boxes
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGraphFindSGridEndpts(nalu_hypre_SStructGraph    *graph,
                                  NALU_HYPRE_Int              part,
                                  NALU_HYPRE_Int              var,
                                  NALU_HYPRE_Int              proc,
                                  NALU_HYPRE_Int              endpt,
                                  NALU_HYPRE_Int             *endpts)
{
   nalu_hypre_SStructGrid     *grid      = nalu_hypre_SStructGraphGrid(graph);
   nalu_hypre_StructGrid      *sgrid;
   nalu_hypre_BoxArray        *boxes;
   NALU_HYPRE_Int              i;

   sgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructGridPGrid(grid, part), var);
   boxes = nalu_hypre_StructGridBoxes(sgrid);

   /* get the endpts using nalu_hypre_SStructGraphFindBoxEndpt */
   for (i = 0; i < nalu_hypre_BoxArraySize(boxes); i++)
   {
      endpts[i] = nalu_hypre_SStructGraphFindBoxEndpt(graph, part, var, proc, endpt, i);
   }

   return nalu_hypre_error_flag;
}

