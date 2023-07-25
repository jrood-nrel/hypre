/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*******************************************************************************

BoxManager:

AHB 10/06, updated 10/09 (changes to info object)

purpose::  organize arbitrary information in a spatial way

misc. notes/considerations/open questions:

  (1) In the struct code, we want to use Box Manager instead of
  current box neighbor stuff (see Struct function
  nalu_hypre_CreateCommInfoFromStencil.  For example, to get neighbors of
  box b, we can call Intersect with a larger box than b).

  (2) will associate a Box Manager with the struct grid (implement
  under the struct grid)

  (3) will interface with the Box Manager in the struct coarsen routine

    the coarsen routine:

    (a) get all the box manager entries from the current level,
    coarsen them, and create a new box manager for the coarse grid,
    adding the boxes via AddEntry

    (b) check the max_distance value and see if we have
        all the neighbor info we need in the current box manager.

    (c) if (b) is no, then call GatherEntries as needed on the coarse
    box manager


    (d) call assemble for the new coarse box manager (note: if gather
    entries has not been called, then no communication is required

  (4) We will associate an assumed partition with the box manager
      (this will be created in the box manager assemble routine)

  (5) We use the box manager with sstruct "on the side" as
  the boxmap is now, (at issue is modifying
  the "info" associated with an entry after the box manager has
  already been assembled through the underlying struct grid)

  (6) In SStruct we will have a separate box manager for the
      neighbor box information

********************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/******************************************************************************
 * Some specialized sorting routines used only in this file
 *****************************************************************************/

/* sort on NALU_HYPRE_Int i, move entry pointers ent */

void
nalu_hypre_entryswap2( NALU_HYPRE_Int  *v,
                  nalu_hypre_BoxManEntry ** ent,
                  NALU_HYPRE_Int  i,
                  NALU_HYPRE_Int  j )
{
   NALU_HYPRE_Int temp;

   nalu_hypre_BoxManEntry *temp_e;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;

   temp_e = ent[i];
   ent[i] = ent[j];
   ent[j] = temp_e;
}

void
nalu_hypre_entryqsort2( NALU_HYPRE_Int *v,
                   nalu_hypre_BoxManEntry ** ent,
                   NALU_HYPRE_Int  left,
                   NALU_HYPRE_Int  right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_entryswap2( v, ent, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_entryswap2(v, ent, ++last, i);
      }
   }
   nalu_hypre_entryswap2(v, ent, left, last);
   nalu_hypre_entryqsort2(v, ent, left, last - 1);
   nalu_hypre_entryqsort2(v, ent, last + 1, right);
}

/*--------------------------------------------------------------------------
 * This is not used
 *--------------------------------------------------------------------------*/

#if 0
NALU_HYPRE_Int
nalu_hypre_BoxManEntrySetInfo ( nalu_hypre_BoxManEntry *entry,
                           void *info )
{
   /* TO DO*/

   return nalu_hypre_error_flag;
}
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManEntryGetInfo (nalu_hypre_BoxManEntry *entry,
                          void **info_ptr )
{
   NALU_HYPRE_Int position = nalu_hypre_BoxManEntryPosition(entry);
   nalu_hypre_BoxManager *boxman;

   boxman = (nalu_hypre_BoxManager *) nalu_hypre_BoxManEntryBoxMan(entry);

   *info_ptr =  nalu_hypre_BoxManInfoObject(boxman, position);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManEntryGetExtents ( nalu_hypre_BoxManEntry *entry,
                              nalu_hypre_Index imin,
                              nalu_hypre_Index imax )
{
   nalu_hypre_IndexRef  entry_imin = nalu_hypre_BoxManEntryIMin(entry);
   nalu_hypre_IndexRef  entry_imax = nalu_hypre_BoxManEntryIMax(entry);
   NALU_HYPRE_Int       ndim       = nalu_hypre_BoxManEntryNDim(entry);

   NALU_HYPRE_Int  d;

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_IndexD(imin, d) = nalu_hypre_IndexD(entry_imin, d);
      nalu_hypre_IndexD(imax, d) = nalu_hypre_IndexD(entry_imax, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Warning: This does not copy the position or info!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManEntryCopy( nalu_hypre_BoxManEntry *fromentry,
                       nalu_hypre_BoxManEntry *toentry )
{
   NALU_HYPRE_Int ndim = nalu_hypre_BoxManEntryNDim(fromentry);
   NALU_HYPRE_Int d;

   nalu_hypre_Index imin;
   nalu_hypre_Index imax;

   nalu_hypre_IndexRef toentry_imin;
   nalu_hypre_IndexRef toentry_imax;

   /* copy extents */
   nalu_hypre_BoxManEntryGetExtents(fromentry, imin, imax);

   toentry_imin = nalu_hypre_BoxManEntryIMin(toentry);
   toentry_imax = nalu_hypre_BoxManEntryIMax(toentry);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_IndexD(toentry_imin, d) = nalu_hypre_IndexD(imin, d);
      nalu_hypre_IndexD(toentry_imax, d) = nalu_hypre_IndexD(imax, d);
   }
   nalu_hypre_BoxManEntryNDim(toentry) = ndim;

   /* copy proc and id */
   nalu_hypre_BoxManEntryProc(toentry) =  nalu_hypre_BoxManEntryProc(fromentry);
   nalu_hypre_BoxManEntryId(toentry) = nalu_hypre_BoxManEntryId(fromentry);

   /*copy ghost */
   for (d = 0; d < 2 * ndim; d++)
   {
      nalu_hypre_BoxManEntryNumGhost(toentry)[d] =
         nalu_hypre_BoxManEntryNumGhost(fromentry)[d];
   }

   /* copy box manager pointer */
   nalu_hypre_BoxManEntryBoxMan(toentry) = nalu_hypre_BoxManEntryBoxMan(fromentry) ;

   /* position - we don't copy this! */

   /* copy list pointer */
   nalu_hypre_BoxManEntryNext(toentry) =  nalu_hypre_BoxManEntryNext(fromentry);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManSetAllGlobalKnown ( nalu_hypre_BoxManager *manager,
                                NALU_HYPRE_Int known )
{
   nalu_hypre_BoxManAllGlobalKnown(manager) = known;

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGetAllGlobalKnown ( nalu_hypre_BoxManager *manager,
                                NALU_HYPRE_Int *known )
{
   *known = nalu_hypre_BoxManAllGlobalKnown(manager);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManSetIsEntriesSort ( nalu_hypre_BoxManager *manager,
                               NALU_HYPRE_Int is_sort )
{
   nalu_hypre_BoxManIsEntriesSort(manager) = is_sort;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGetIsEntriesSort ( nalu_hypre_BoxManager *manager,
                               NALU_HYPRE_Int *is_sort )
{
   *is_sort  =  nalu_hypre_BoxManIsEntriesSort(manager);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGetGlobalIsGatherCalled( nalu_hypre_BoxManager *manager,
                                     MPI_Comm  comm,
                                     NALU_HYPRE_Int *is_gather )
{
   NALU_HYPRE_Int loc_is_gather;
   NALU_HYPRE_Int nprocs;

   nalu_hypre_MPI_Comm_size(comm, &nprocs);

   loc_is_gather = nalu_hypre_BoxManIsGatherCalled(manager);

   if (nprocs > 1)
   {
      nalu_hypre_MPI_Allreduce(&loc_is_gather, is_gather, 1, NALU_HYPRE_MPI_INT,
                          nalu_hypre_MPI_LOR, comm);
   }
   else /* just one proc */
   {
      *is_gather = loc_is_gather;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGetAssumedPartition( nalu_hypre_BoxManager *manager,
                                 nalu_hypre_StructAssumedPart **assumed_partition )
{
   *assumed_partition = nalu_hypre_BoxManAssumedPartition(manager);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManSetAssumedPartition( nalu_hypre_BoxManager *manager,
                                 nalu_hypre_StructAssumedPart *assumed_partition )
{
   nalu_hypre_BoxManAssumedPartition(manager) = assumed_partition;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManSetBoundingBox ( nalu_hypre_BoxManager *manager,
                             nalu_hypre_Box *bounding_box )
{
   nalu_hypre_Box* bbox = nalu_hypre_BoxManBoundingBox(manager);

   nalu_hypre_BoxSetExtents(bbox,  nalu_hypre_BoxIMin(bounding_box),
                       nalu_hypre_BoxIMax(bounding_box));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManSetNumGhost( nalu_hypre_BoxManager *manager,
                         NALU_HYPRE_Int  *num_ghost )
{
   NALU_HYPRE_Int  i, ndim = nalu_hypre_BoxManNDim(manager);

   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_BoxManNumGhost(manager)[i] = num_ghost[i];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Delete multiple entries (and their corresponding info object) from the
 * manager.  The indices correspond to the ordering of the entries.  Assumes
 * indices given in ascending order - this is meant for internal use inside the
 * Assemble routime.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManDeleteMultipleEntriesAndInfo( nalu_hypre_BoxManager *manager,
                                          NALU_HYPRE_Int*  indices,
                                          NALU_HYPRE_Int num )
{
   NALU_HYPRE_Int  i, j, start;
   NALU_HYPRE_Int  array_size = nalu_hypre_BoxManNEntries(manager);

   NALU_HYPRE_Int  info_size = nalu_hypre_BoxManEntryInfoSize(manager);

   void *to_ptr;
   void *from_ptr;

   nalu_hypre_BoxManEntry  *entries  = nalu_hypre_BoxManEntries(manager);

   if (num > 0)
   {
      start = indices[0];

      j = 0;

      for (i = start; (i + j) < array_size; i++)
      {
         if (j < num)
         {
            while ((i + j) == indices[j]) /* see if deleting consecutive items */
            {
               j++; /*increase the shift*/
               if (j == num) { break; }
            }
         }

         if ( (i + j) < array_size) /* if deleting the last item then no moving */
         {
            /*copy the entry */
            nalu_hypre_BoxManEntryCopy(&entries[i + j], &entries[i]);

            /* change the position */
            nalu_hypre_BoxManEntryPosition(&entries[i]) = i;

            /* copy the info object */
            to_ptr = nalu_hypre_BoxManInfoObject(manager, i);
            from_ptr = nalu_hypre_BoxManInfoObject(manager, i + j);

            nalu_hypre_TMemcpy(to_ptr,  from_ptr, char, info_size, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         }
      }

      nalu_hypre_BoxManNEntries(manager) = array_size - num;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *  Allocate and initialize the box manager structure.
 *
 *  Notes:
 *
 * (1) max_nentries indicates how much storage you think you will need for
 * adding entries with BoxManAddEntry
 *
 * (2) info_size indicates the size (in bytes) of the info object that
 * will be attached to each entry in this box manager.
 *
 * (3) we will collect the bounding box - this is used by the AP
 *
 * (4) comm is needed for later calls to addentry - also used in the assemble
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManCreate( NALU_HYPRE_Int max_nentries,
                    NALU_HYPRE_Int info_size,
                    NALU_HYPRE_Int ndim,
                    nalu_hypre_Box *bounding_box,
                    MPI_Comm comm,
                    nalu_hypre_BoxManager **manager_ptr )
{
   nalu_hypre_BoxManager   *manager;
   nalu_hypre_Box          *bbox;

   NALU_HYPRE_Int  i, d;
   /* allocate object */
   manager = nalu_hypre_CTAlloc(nalu_hypre_BoxManager,  1, NALU_HYPRE_MEMORY_HOST);

   /* initialize */
   nalu_hypre_BoxManComm(manager) = comm;
   nalu_hypre_BoxManMaxNEntries(manager) = max_nentries;
   nalu_hypre_BoxManEntryInfoSize(manager) = info_size;
   nalu_hypre_BoxManNDim(manager) = ndim;
   nalu_hypre_BoxManIsAssembled(manager) = 0;

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxManIndexesD(manager, d) = NULL;
   }

   nalu_hypre_BoxManNEntries(manager) = 0;
   nalu_hypre_BoxManEntries(manager)  = nalu_hypre_CTAlloc(nalu_hypre_BoxManEntry,  max_nentries, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_BoxManInfoObjects(manager) = NULL;
   nalu_hypre_BoxManInfoObjects(manager) = nalu_hypre_TAlloc(char, max_nentries * info_size, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_BoxManIndexTable(manager) = NULL;

   nalu_hypre_BoxManNumProcsSort(manager)     = 0;
   nalu_hypre_BoxManIdsSort(manager)          = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_nentries, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxManProcsSort(manager)        = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_nentries, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxManProcsSortOffsets(manager) = NULL;

   nalu_hypre_BoxManFirstLocal(manager)      = 0;
   nalu_hypre_BoxManLocalProcOffset(manager) = 0;

   nalu_hypre_BoxManIsGatherCalled(manager)  = 0;
   nalu_hypre_BoxManGatherRegions(manager)   = nalu_hypre_BoxArrayCreate(0, ndim);
   nalu_hypre_BoxManAllGlobalKnown(manager)  = 0;

   nalu_hypre_BoxManIsEntriesSort(manager)   = 0;

   nalu_hypre_BoxManNumMyEntries(manager) = 0;
   nalu_hypre_BoxManMyIds(manager)        = NULL;
   nalu_hypre_BoxManMyEntries(manager)    = NULL;

   nalu_hypre_BoxManAssumedPartition(manager) = NULL;

   nalu_hypre_BoxManMyIds(manager) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_nentries, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxManMyEntries(manager) =
      nalu_hypre_CTAlloc(nalu_hypre_BoxManEntry *,  max_nentries, NALU_HYPRE_MEMORY_HOST);

   bbox =  nalu_hypre_BoxCreate(ndim);
   nalu_hypre_BoxManBoundingBox(manager) = bbox;
   nalu_hypre_BoxSetExtents(bbox, nalu_hypre_BoxIMin(bounding_box),
                       nalu_hypre_BoxIMax(bounding_box));

   nalu_hypre_BoxManNextId(manager) = 0;

   /* ghost points: we choose a default that will give zero everywhere..*/
   for (i = 0; i < 2 * NALU_HYPRE_MAXDIM; i++)
   {
      nalu_hypre_BoxManNumGhost(manager)[i] = 0;
   }

   /* return */
   *manager_ptr = manager;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Increase storage for entries (for future calls to BoxManAddEntry).
 *
 * Notes:
 *
 * In addition, we will dynamically allocate more memory if needed when a call
 * to BoxManAddEntry is made and there is not enough storage available.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManIncSize ( nalu_hypre_BoxManager *manager,
                      NALU_HYPRE_Int inc_size )
{
   NALU_HYPRE_Int   max_nentries = nalu_hypre_BoxManMaxNEntries(manager);
   NALU_HYPRE_Int  *ids          = nalu_hypre_BoxManIdsSort(manager);
   NALU_HYPRE_Int  *procs        = nalu_hypre_BoxManProcsSort(manager);
   NALU_HYPRE_Int   info_size    = nalu_hypre_BoxManEntryInfoSize(manager);

   void *info         = nalu_hypre_BoxManInfoObjects(manager);

   nalu_hypre_BoxManEntry  *entries = nalu_hypre_BoxManEntries(manager);

   /* increase size */
   max_nentries += inc_size;

   entries = nalu_hypre_TReAlloc(entries,  nalu_hypre_BoxManEntry,  max_nentries, NALU_HYPRE_MEMORY_HOST);
   ids = nalu_hypre_TReAlloc(ids,  NALU_HYPRE_Int,  max_nentries, NALU_HYPRE_MEMORY_HOST);
   procs =  nalu_hypre_TReAlloc(procs,  NALU_HYPRE_Int,  max_nentries, NALU_HYPRE_MEMORY_HOST);
   info = (void *) nalu_hypre_TReAlloc((char *)info, char, max_nentries * info_size, NALU_HYPRE_MEMORY_HOST);

   /* update manager */
   nalu_hypre_BoxManMaxNEntries(manager) = max_nentries;
   nalu_hypre_BoxManEntries(manager)     = entries;
   nalu_hypre_BoxManIdsSort(manager)     = ids;
   nalu_hypre_BoxManProcsSort(manager)   = procs;
   nalu_hypre_BoxManInfoObjects(manager) = info;

   /* my ids temporary structure (destroyed in assemble) */
   {
      NALU_HYPRE_Int *my_ids = nalu_hypre_BoxManMyIds(manager);
      nalu_hypre_BoxManEntry  **my_entries = nalu_hypre_BoxManMyEntries(manager);

      my_ids = nalu_hypre_TReAlloc(my_ids,  NALU_HYPRE_Int,  max_nentries, NALU_HYPRE_MEMORY_HOST);

      my_entries = nalu_hypre_TReAlloc(my_entries,  nalu_hypre_BoxManEntry *,  max_nentries, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_BoxManMyIds(manager) = my_ids;
      nalu_hypre_BoxManMyEntries(manager) = my_entries;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *  De-allocate the box manager structure.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManDestroy( nalu_hypre_BoxManager *manager )
{
   NALU_HYPRE_Int ndim = nalu_hypre_BoxManNDim(manager);
   NALU_HYPRE_Int d;

   if (manager)
   {
      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_TFree(nalu_hypre_BoxManIndexesD(manager,  d), NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(nalu_hypre_BoxManEntries(manager), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(nalu_hypre_BoxManInfoObjects(manager), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(nalu_hypre_BoxManIndexTable(manager), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(nalu_hypre_BoxManIdsSort(manager), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_BoxManProcsSort(manager), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_BoxManProcsSortOffsets(manager), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_BoxArrayDestroy(nalu_hypre_BoxManGatherRegions(manager));

      nalu_hypre_TFree(nalu_hypre_BoxManMyIds(manager), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_BoxManMyEntries(manager), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_StructAssumedPartitionDestroy(nalu_hypre_BoxManAssumedPartition(manager));

      nalu_hypre_BoxDestroy(nalu_hypre_BoxManBoundingBox(manager));

      nalu_hypre_TFree(manager, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Add a box (entry) to the box manager. Each entry is given a
 * unique id (proc_id, box_id).  Need to assemble after adding entries.
 *
 * Notes:
 *
 * (1) The id assigned may be any integer - though since (proc_id,
 * box_id) is unique, duplicates will be eliminated in the assemble.
 *
 * (2) If there is not enough storage available for this entry, then
 * increase the amount automatically
 *
 * (3) Only add entries whose boxes have non-zero volume.
 *
 * (4) The info object will be copied (according to the info size given in
 * the create) to storage within the box manager.
 *
 * (5) If the id passed in is negative (user doesn't care what it is) ,
 * then use the next_id stored in the box manager to assign the id
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManAddEntry( nalu_hypre_BoxManager *manager,
                      nalu_hypre_Index imin,
                      nalu_hypre_Index imax,
                      NALU_HYPRE_Int proc_id,
                      NALU_HYPRE_Int box_id,
                      void *info )
{
   NALU_HYPRE_Int           myid;
   NALU_HYPRE_Int           nentries = nalu_hypre_BoxManNEntries(manager);
   NALU_HYPRE_Int           info_size = nalu_hypre_BoxManEntryInfoSize(manager);
   NALU_HYPRE_Int           ndim = nalu_hypre_BoxManNDim(manager);

   nalu_hypre_BoxManEntry  *entries  = nalu_hypre_BoxManEntries(manager);
   nalu_hypre_BoxManEntry  *entry;

   nalu_hypre_IndexRef      entry_imin;
   nalu_hypre_IndexRef      entry_imax;

   NALU_HYPRE_Int           d;
   NALU_HYPRE_Int           *num_ghost = nalu_hypre_BoxManNumGhost(manager);
   NALU_HYPRE_Int           volume;

   NALU_HYPRE_Int           id;

   nalu_hypre_Box           *box;

   /* can only use before assembling */
   if (nalu_hypre_BoxManIsAssembled(manager))
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* check to see if we have a non-zero box volume (only add if non-zero) */
   box = nalu_hypre_BoxCreate(nalu_hypre_BoxManNDim(manager));
   nalu_hypre_BoxSetExtents( box, imin, imax );
   volume = nalu_hypre_BoxVolume(box);
   nalu_hypre_BoxDestroy(box);

   if (volume)
   {
      nalu_hypre_MPI_Comm_rank(nalu_hypre_BoxManComm(manager), &myid );

      /* check to make sure that there is enough storage available
         for this new entry - if not add space for 10 more */

      if (nentries + 1 > nalu_hypre_BoxManMaxNEntries(manager))
      {
         nalu_hypre_BoxManIncSize(manager, 10);

         entries = nalu_hypre_BoxManEntries(manager);
      }

      /* we add this to the end entry list - get pointer to location*/
      entry = &entries[nentries];
      entry_imin = nalu_hypre_BoxManEntryIMin(entry);
      entry_imax = nalu_hypre_BoxManEntryIMax(entry);

      /* copy information into entry */
      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_IndexD(entry_imin, d) = nalu_hypre_IndexD(imin, d);
         nalu_hypre_IndexD(entry_imax, d) = nalu_hypre_IndexD(imax, d);
      }
      nalu_hypre_BoxManEntryNDim(entry) = ndim;

      /* set the processor */
      nalu_hypre_BoxManEntryProc(entry) = proc_id;

      /* set the id */
      if (box_id >= 0)
      {
         id = box_id;
      }
      else /* negative means use id from box manager */
      {
         id = nalu_hypre_BoxManNextId(manager);
         /* increment fir next time */
         nalu_hypre_BoxManNextId(manager) = id + 1;
      }

      nalu_hypre_BoxManEntryId(entry) = id;

      /* this is the current position in the entries array */
      nalu_hypre_BoxManEntryPosition(entry) = nentries;

      /*this associates it with the box manager */
      nalu_hypre_BoxManEntryBoxMan(entry) = (void *) manager;

      /* copy the info object */
      if (info_size > 0)
      {
         void *index_ptr;

         /*point in the info array */
         index_ptr =  nalu_hypre_BoxManInfoObject(manager, nentries);
         nalu_hypre_TMemcpy(index_ptr,  info, char, info_size, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      }

      /* inherit and inject the numghost from manager into the entry (as
       * in boxmap) */
      for (d = 0; d < 2 * ndim; d++)
      {
         nalu_hypre_BoxManEntryNumGhost(entry)[d] = num_ghost[d];
      }
      nalu_hypre_BoxManEntryNext(entry) = NULL;

      /* add proc and id to procs_sort and ids_sort array */
      nalu_hypre_BoxManProcsSort(manager)[nentries] = proc_id;
      nalu_hypre_BoxManIdsSort(manager)[nentries] = id;

      /* here we need to keep track of my entries separately just to improve
         speed at the beginning of the assemble - then this gets deleted when
         the entries are sorted. */

      if (proc_id == myid)
      {
         NALU_HYPRE_Int *my_ids =   nalu_hypre_BoxManMyIds(manager);
         nalu_hypre_BoxManEntry **my_entries = nalu_hypre_BoxManMyEntries(manager);
         NALU_HYPRE_Int num_my_entries = nalu_hypre_BoxManNumMyEntries(manager);

         my_ids[num_my_entries] = id;
         my_entries[num_my_entries] = &entries[nentries];
         num_my_entries++;

         nalu_hypre_BoxManNumMyEntries(manager) = num_my_entries;
      }

      /* increment number of entries */
      nalu_hypre_BoxManNEntries(manager) = nentries + 1;

   } /* end of  vol > 0 */

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Given an id: (proc_id, box_id), return a pointer to the box entry.
 *
 * Notes:
 *
 * (1) Use of this is generally to get back something that has been
 * added by the above function.  If no entry is found, an error is returned.
 *
 * (2) This functionality will replace that previously provided by
 * nalu_hypre_BoxManFindBoxProcEntry.
 *
 * (3) Need to store entry information such that this information is
 * easily found. (During the assemble, we will sort on proc_id, then
 * box_id, and provide a pointer to the entries.  Then we can do a
 * search into the proc_id, and then into the box_id.)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGetEntry( nalu_hypre_BoxManager *manager,
                      NALU_HYPRE_Int proc,
                      NALU_HYPRE_Int id,
                      nalu_hypre_BoxManEntry **entry_ptr )
{
   /* find proc_id in procs array.  then find id in ids array, then grab the
      corresponding entry */

   nalu_hypre_BoxManEntry *entry;

   NALU_HYPRE_Int  myid;
   NALU_HYPRE_Int  i, offset;
   NALU_HYPRE_Int  start, finish;
   NALU_HYPRE_Int  location;
   NALU_HYPRE_Int  first_local  = nalu_hypre_BoxManFirstLocal(manager);
   NALU_HYPRE_Int *procs_sort   = nalu_hypre_BoxManProcsSort(manager);
   NALU_HYPRE_Int *ids_sort     = nalu_hypre_BoxManIdsSort(manager);
   NALU_HYPRE_Int  nentries     = nalu_hypre_BoxManNEntries(manager);
   NALU_HYPRE_Int  num_proc     = nalu_hypre_BoxManNumProcsSort(manager);
   NALU_HYPRE_Int *proc_offsets =  nalu_hypre_BoxManProcsSortOffsets(manager);

   /* can only use after assembling */
   if (!nalu_hypre_BoxManIsAssembled(manager))
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_rank(nalu_hypre_BoxManComm(manager), &myid );

   if (nentries)
   {
      /* check to see if it is the local id first - this will be the case most
       * of the time (currently it is only used in this manner)*/
      if (proc == myid)
      {
         start = first_local;
         if (start >= 0 )
         {
            finish =  proc_offsets[nalu_hypre_BoxManLocalProcOffset(manager) + 1];
         }
      }

      else /* otherwise find proc (TO DO: just have procs_sort not contain
              duplicates - then we could do a regular binary search (though this
              list is probably short)- this has to be changed in assemble, then
              also memory management in addentry - but currently this is not
              necessary because proc = myid for all current hypre calls) */
      {
         start = -1;
         for (i = 0; i < num_proc; i++)
         {
            offset = proc_offsets[i];
            if (proc == procs_sort[offset])
            {
               start = offset;
               finish = proc_offsets[i + 1];
               break;
            }
         }
      }
      if (start >= 0 )
      {
         /* now look for the id - returns -1 if not found*/
         location = nalu_hypre_BinarySearch(&ids_sort[start], id, finish - start);
      }
      else
      {
         location = -1;
      }
   }
   else
   {
      location = -1;
   }

   if (location >= 0 )
   {
      /* this location is relative to where we started searching - so fix if
       * non-negative */
      location += start;
      /* now grab entry */
      entry =  &nalu_hypre_BoxManEntries(manager)[location];
   }
   else
   {
      entry = NULL;
   }

   *entry_ptr = entry;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a list of all of the entries in the box manager (and the number of
 * entries). These are sorted by (proc, id) pairs.
 *
 * 11/06 - changed to return the pointer to the boxman entries rather than a
 * copy of the array (so calling code should not free this array!)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGetAllEntries( nalu_hypre_BoxManager *manager,
                           NALU_HYPRE_Int *num_entries,
                           nalu_hypre_BoxManEntry **entries)
{
   /* can only use after assembling */
   if (!nalu_hypre_BoxManIsAssembled(manager))
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* return */
   *num_entries = nalu_hypre_BoxManNEntries(manager);
   *entries =  nalu_hypre_BoxManEntries(manager);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a list of all of the boxes ONLY in the entries in the box manager.
 *
 * Notes: Should have already created the box array;
 *
 * TO DO: (?) Might want to just store the array of boxes seperate from the
 * entries array so we don't have to create the array everytime this function is
 * called.  (may be called quite a bit in some sstruct apps)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGetAllEntriesBoxes( nalu_hypre_BoxManager *manager,
                                nalu_hypre_BoxArray *boxes )
{
   nalu_hypre_BoxManEntry entry;

   NALU_HYPRE_Int          i, nentries;
   nalu_hypre_Index       ilower, iupper;

   nalu_hypre_BoxManEntry  *boxman_entries  = nalu_hypre_BoxManEntries(manager);

   /* can only use after assembling */
   if (!nalu_hypre_BoxManIsAssembled(manager))
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* set array size  */
   nentries = nalu_hypre_BoxManNEntries(manager);

   nalu_hypre_BoxArraySetSize(boxes, nentries);

   for (i = 0; i < nentries; i++)
   {
      entry = boxman_entries[i];
      nalu_hypre_BoxManEntryGetExtents(&entry, ilower, iupper);
      nalu_hypre_BoxSetExtents(nalu_hypre_BoxArrayBox(boxes, i), ilower, iupper);
   }

   /* return */

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a list of all of the boxes ONLY in the entries in the box manager that
 * belong to the calling processor.
 *
 * Notes: Should have already created the box array;
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGetLocalEntriesBoxes( nalu_hypre_BoxManager *manager,
                                  nalu_hypre_BoxArray *boxes )
{
   nalu_hypre_BoxManEntry entry;

   NALU_HYPRE_Int          i;

   nalu_hypre_Index        ilower, iupper;

   NALU_HYPRE_Int  start = nalu_hypre_BoxManFirstLocal(manager);
   NALU_HYPRE_Int  finish;
   NALU_HYPRE_Int  num_my_entries = nalu_hypre_BoxManNumMyEntries(manager);

   nalu_hypre_BoxManEntry  *boxman_entries  = nalu_hypre_BoxManEntries(manager);

   NALU_HYPRE_Int *offsets = nalu_hypre_BoxManProcsSortOffsets(manager);

   /* can only use after assembling */
   if (!nalu_hypre_BoxManIsAssembled(manager))
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* set array size  */
   nalu_hypre_BoxArraySetSize(boxes, num_my_entries);

   finish =  offsets[nalu_hypre_BoxManLocalProcOffset(manager) + 1];

   if (num_my_entries && ((finish - start) != num_my_entries))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Something's wrong with box manager!");
   }

   for (i = 0; i < num_my_entries; i++)
   {
      entry = boxman_entries[start + i];
      nalu_hypre_BoxManEntryGetExtents(&entry, ilower, iupper);
      nalu_hypre_BoxSetExtents(nalu_hypre_BoxArrayBox(boxes, i), ilower, iupper);
   }

   /* return */

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *  Get the boxes and the proc ids. The input procs array should be NULL.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGetAllEntriesBoxesProc( nalu_hypre_BoxManager *manager,
                                    nalu_hypre_BoxArray   *boxes,
                                    NALU_HYPRE_Int       **procs_ptr)
{
   nalu_hypre_BoxManEntry  entry;
   NALU_HYPRE_Int          i, nentries;
   nalu_hypre_Index        ilower, iupper;
   nalu_hypre_BoxManEntry *boxman_entries  = nalu_hypre_BoxManEntries(manager);
   NALU_HYPRE_Int         *procs;

   /* can only use after assembling */
   if (!nalu_hypre_BoxManIsAssembled(manager))
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* set array size  */
   nentries = nalu_hypre_BoxManNEntries(manager);
   nalu_hypre_BoxArraySetSize(boxes, nentries);
   procs = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nentries, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < nentries; i++)
   {
      entry = boxman_entries[i];
      nalu_hypre_BoxManEntryGetExtents(&entry, ilower, iupper);
      nalu_hypre_BoxSetExtents(nalu_hypre_BoxArrayBox(boxes, i), ilower, iupper);
      procs[i] = nalu_hypre_BoxManEntryProc(&entry);
   }

   /* return */
   *procs_ptr = procs;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * All global entries that lie within the boxes supplied to this function are
 * gathered from other processors during the assemble and stored in a
 * processor's local box manager.  Multiple calls may be made to this
 * function. The box extents supplied here are not retained after the assemble.
 *
 * Note:
 *
 * (1) This affects whether or not calls to BoxManIntersect() can be answered
 * correctly.  In other words, the user needs to anticipate the areas of the
 * grid where BoxManIntersect() calls will be made, and make sure that
 * information has been collected.
 *
 * (2) when this is called, the boolean "is_gather_entries" is set and the box
 * is added to gather_regions array.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManGatherEntries(nalu_hypre_BoxManager *manager,
                          nalu_hypre_Index imin,
                          nalu_hypre_Index imax )
{
   nalu_hypre_Box *box;

   nalu_hypre_BoxArray  *gather_regions;

   /* can only use before assembling */
   if (nalu_hypre_BoxManIsAssembled(manager))
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* initialize */
   nalu_hypre_BoxManIsGatherCalled(manager) = 1;
   gather_regions = nalu_hypre_BoxManGatherRegions(manager);

   /* add the box to the gather region array */
   box = nalu_hypre_BoxCreate(nalu_hypre_BoxManNDim(manager));
   nalu_hypre_BoxSetExtents( box, imin, imax );
   nalu_hypre_AppendBox( box, gather_regions); /* this is a copy */

   /* clean up */
   nalu_hypre_BoxDestroy(box);
   nalu_hypre_BoxManGatherRegions(manager) = gather_regions; /* may be a realloc */

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * In the assemble, we populate the local box manager with global box
 * information to be used by calls to BoxManIntersect().  Global box information
 * is gathered that corresponds to the regions input by calls to
 * nalu_hypre_BoxManGatherEntries().
 *
 * Notes:
 *
 * (1) In the assumed partition (AP) case, the boxes gathered are those that
 * correspond to boxes living in the assumed partition regions that intersect
 * the regions input to nalu_hypre_BoxManGatherEntries().  (We will have to check for
 * duplicates here as a box can be in more than one AP.)
 *
 * (2) If a box is gathered from a neighbor processor, then all the boxes from
 * that neighbor processor are retrieved.  So we can always assume that have all
 * the local information from neighbor processors.
 *
 * (3) If nalu_hypre_BoxManGatherEntries() has *not* been called, then only the box
 * information provided via calls to nalu_hypre_BoxManAddEntry will be in the box
 * manager.  (There is a global communication to check if GatherEntires has been
 * called on any processor).  In the non-AP case, if GatherEntries is called on
 * *any* processor, then all processors get *all* boxes (via allgatherv).
 *
 * (Don't call gather entries if all is known already)
 *
 * (4) Need to check for duplicate boxes (and eliminate) - based on pair
 * (proc_id, box_id).  Also sort this identifier pair so that GetEntry calls can
 * be made more easily.
 *
 * (5) ****TO DO****Particularly in the AP case, might want to think about a
 * "smart" algorithm to decide whether point-to-point communications or an
 * AllGather is the best way to collect the needed entries resulting from calls
 * to GatherEntries().  If this was done well, then the AP and non-AP would not
 * have to be treated separately at all!
 *
 * **Assumptions:
 *
 * 1. A processor has used "add entry" to put all of the boxes that it owns into
 * its box manager
 *
 * 2. The assemble routine is only called once for a box manager (i.e., you
 * don't assemble, then add more entries and then assemble again)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManAssemble( nalu_hypre_BoxManager *manager )
{
   NALU_HYPRE_Int  ndim = nalu_hypre_BoxManNDim(manager);
   NALU_HYPRE_Int  myid, nprocs;
   NALU_HYPRE_Int  is_gather, global_is_gather;
   NALU_HYPRE_Int  nentries;
   NALU_HYPRE_Int *procs_sort, *ids_sort;
   NALU_HYPRE_Int  i, j, k;

   NALU_HYPRE_Int need_to_sort = 1; /* default it to sort */
   //NALU_HYPRE_Int short_sort = 0; /*do abreviated sort */

   NALU_HYPRE_Int  non_ap_gather = 1; /* default to gather w/out ap*/

   NALU_HYPRE_Int  global_num_boxes = 0;

   nalu_hypre_BoxManEntry *entries;

   nalu_hypre_BoxArray  *gather_regions;

   MPI_Comm comm = nalu_hypre_BoxManComm(manager);

   /* cannot re-assemble */
   if (nalu_hypre_BoxManIsAssembled(manager))
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* initilize */
   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_MPI_Comm_size(comm, &nprocs);

   gather_regions = nalu_hypre_BoxManGatherRegions(manager);
   nentries = nalu_hypre_BoxManNEntries(manager);
   entries =  nalu_hypre_BoxManEntries(manager);
   procs_sort = nalu_hypre_BoxManProcsSort(manager);

   ids_sort = nalu_hypre_BoxManIdsSort(manager);

   /* do we need to gather entries - check to see if ANY processor called a
    * gather? */

   if (!nalu_hypre_BoxManAllGlobalKnown(manager))
   {
      if (nprocs > 1)
      {
         is_gather = nalu_hypre_BoxManIsGatherCalled(manager);
         nalu_hypre_MPI_Allreduce(&is_gather, &global_is_gather, 1, NALU_HYPRE_MPI_INT,
                             nalu_hypre_MPI_LOR, comm);
      }
      else /* just one proc */
      {
         global_is_gather = 0;
         nalu_hypre_BoxManAllGlobalKnown(manager) = 1;
      }
   }
   else /* global info is known - don't call a gather even if the use has
           called gather entries */
   {
      global_is_gather = 0;
   }

   /* ----------------------------GATHER? ------------------------------------*/

   if (global_is_gather)
   {

      NALU_HYPRE_Int *my_ids         = nalu_hypre_BoxManMyIds(manager);
      NALU_HYPRE_Int  num_my_entries = nalu_hypre_BoxManNumMyEntries(manager);

      nalu_hypre_BoxManEntry **my_entries = nalu_hypre_BoxManMyEntries(manager);

      /* Need to be able to find our own entry, given the box number - for the
         second data exchange - so do some sorting now.  Then we can use my_ids
         to quickly find an entry.  This will be freed when the sort table is
         created (it's redundant at that point).  (Note: we may be creating the
         AP here, so this sorting needs to be done at the beginning for that
         too).  If non-ap, then we want the allgatherv to already be sorted - so
         this takes care of that */

      /* my entries may already be sorted (if all entries are then my entries
         are - so check first */

      if (nalu_hypre_BoxManIsEntriesSort(manager) == 0)
      {
         nalu_hypre_entryqsort2(my_ids, my_entries, 0, num_my_entries - 1);
      }

      /* if AP, use AP to find out who owns the data we need.  In the non-AP,
         then just gather everything for now. */

      non_ap_gather = 0;

      /* Goal: Gather entries from the relevant processor and add to the entries
       * array.  Also add proc and id to the procs_sort and ids_sort arrays. */

      if (!non_ap_gather)   /*********** AP CASE! ***********/
      {
         NALU_HYPRE_Int  size;
         NALU_HYPRE_Int *tmp_proc_ids;
         NALU_HYPRE_Int  proc_count, proc_alloc;
         //NALU_HYPRE_Int  max_proc_count;
         NALU_HYPRE_Int *proc_array;
         NALU_HYPRE_Int *ap_proc_ids;
         NALU_HYPRE_Int  count;

         NALU_HYPRE_Int  max_response_size;
         NALU_HYPRE_Int  non_info_size, entry_size_bytes;
         NALU_HYPRE_Int *neighbor_proc_ids = NULL;
         NALU_HYPRE_Int *response_buf_starts;
         NALU_HYPRE_Int *response_buf;
         NALU_HYPRE_Int  response_size, tmp_int;

         NALU_HYPRE_Int *send_buf = NULL;
         NALU_HYPRE_Int *send_buf_starts = NULL;
         NALU_HYPRE_Int  d, proc, id, last_id;
         NALU_HYPRE_Int *tmp_int_ptr;
         NALU_HYPRE_Int *contact_proc_ids = NULL;

         NALU_HYPRE_Int max_regions, max_refinements, ologp;

         NALU_HYPRE_Int  *local_boxnums;

         NALU_HYPRE_Int statbuf[3];
         NALU_HYPRE_Int send_statbuf[3];

         NALU_HYPRE_Int ndim = nalu_hypre_BoxManNDim(manager);

         void *entry_response_buf;
         void *index_ptr;

         NALU_HYPRE_Real gamma;
         NALU_HYPRE_Real local_volume, global_volume;
         NALU_HYPRE_Real sendbuf2[2], recvbuf2[2];

         nalu_hypre_BoxArray *gather_regions;
         nalu_hypre_BoxArray *local_boxes;

         nalu_hypre_Box *box;

         nalu_hypre_StructAssumedPart *ap;

         nalu_hypre_DataExchangeResponse  response_obj, response_obj2;

         nalu_hypre_BoxManEntry *entry_ptr;

         nalu_hypre_Index imin, imax;

         nalu_hypre_IndexRef  min_ref, max_ref;

         /* 1.  Create an assumed partition? (may have been added in the coarsen
            routine) */

         if (nalu_hypre_BoxManAssumedPartition(manager) == NULL)
         {

            /* create an array of local boxes.  get the global box size/volume
               (as a NALU_HYPRE_Real). */

            local_boxes = nalu_hypre_BoxArrayCreate(num_my_entries, ndim);
            local_boxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_my_entries, NALU_HYPRE_MEMORY_HOST);

            local_volume = 0.0;

            for (i = 0; i < num_my_entries; i++)
            {
               /* get entry */
               entry_ptr = my_entries[i];

               /* copy box info to local_boxes */
               min_ref = nalu_hypre_BoxManEntryIMin(entry_ptr);
               max_ref =  nalu_hypre_BoxManEntryIMax(entry_ptr);
               box = nalu_hypre_BoxArrayBox(local_boxes, i);
               nalu_hypre_BoxSetExtents( box, min_ref, max_ref );

               /* keep box num also */
               local_boxnums[i] =   nalu_hypre_BoxManEntryId(entry_ptr);

               /* calculate volume */
               local_volume += (NALU_HYPRE_Real) nalu_hypre_BoxVolume(box);

            }/* end of local boxes */

            /* get the number of global entries and the global volume */

            sendbuf2[0] = local_volume;
            sendbuf2[1] = (NALU_HYPRE_Real) num_my_entries;

            nalu_hypre_MPI_Allreduce(&sendbuf2, &recvbuf2, 2, NALU_HYPRE_MPI_REAL,
                                nalu_hypre_MPI_SUM, comm);

            global_volume = recvbuf2[0];
            global_num_boxes = (NALU_HYPRE_Int) recvbuf2[1];

            /* estimates for the assumed partition */
            d = nprocs / 2;
            ologp = 0;
            while ( d > 0)
            {
               d = d / 2; /* note - d is an NALU_HYPRE_Int - so this is floored */
               ologp++;
            }

            max_regions =  nalu_hypre_min(nalu_hypre_pow2(ologp + 1), 10 * ologp);
            max_refinements = ologp;
            gamma = .6; /* percentage a region must be full to
                           avoid refinement */

            nalu_hypre_StructAssumedPartitionCreate(
               ndim, nalu_hypre_BoxManBoundingBox(manager), global_volume,
               global_num_boxes, local_boxes, local_boxnums,
               max_regions, max_refinements, gamma, comm, &ap);

            nalu_hypre_BoxManAssumedPartition(manager) = ap;

            nalu_hypre_BoxArrayDestroy(local_boxes);
            nalu_hypre_TFree(local_boxnums, NALU_HYPRE_MEMORY_HOST);
         }
         else
         {
            ap = nalu_hypre_BoxManAssumedPartition(manager);
         }

         /* 2.  Now go thru gather regions and find out which processor's AP
            region they intersect - only do the rest if we have global boxes!*/

         if (global_num_boxes)
         {
            gather_regions = nalu_hypre_BoxManGatherRegions(manager);

            /*allocate space to store info from one box */
            proc_count = 0;
            proc_alloc = nalu_hypre_pow2(ndim); /* Just an initial estimate */
            proc_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  proc_alloc, NALU_HYPRE_MEMORY_HOST);

            /* probably there will mostly be one proc per box - allocate space
             * for 2 */
            size = 2 * nalu_hypre_BoxArraySize(gather_regions);
            tmp_proc_ids =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
            count = 0;

            /* loop through all boxes */
            nalu_hypre_ForBoxI(i, gather_regions)
            {
               nalu_hypre_StructAssumedPartitionGetProcsFromBox(
                  ap, nalu_hypre_BoxArrayBox(gather_regions, i),
                  &proc_count, &proc_alloc, &proc_array);

               if ((count + proc_count) > size)
               {
                  size = size + proc_count
                         + 2 * (nalu_hypre_BoxArraySize(gather_regions) - i);
                  tmp_proc_ids = nalu_hypre_TReAlloc(tmp_proc_ids,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               }
               for (j = 0; j < proc_count; j++)
               {
                  tmp_proc_ids[count] = proc_array[j];
                  count++;
               }
            }

            nalu_hypre_TFree(proc_array, NALU_HYPRE_MEMORY_HOST);

            /* now get rid of redundencies in tmp_proc_ids (since a box can lie
               in more than one AP - put in ap_proc_ids*/
            nalu_hypre_qsort0(tmp_proc_ids, 0, count - 1);
            proc_count = 0;
            ap_proc_ids = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  count, NALU_HYPRE_MEMORY_HOST);

            if (count)
            {
               ap_proc_ids[0] = tmp_proc_ids[0];
               proc_count++;
            }
            for (i = 1; i < count; i++)
            {
               if (tmp_proc_ids[i]  != ap_proc_ids[proc_count - 1])
               {
                  ap_proc_ids[proc_count] = tmp_proc_ids[i];
                  proc_count++;
               }
            }
            nalu_hypre_TFree(tmp_proc_ids, NALU_HYPRE_MEMORY_HOST);

            /* 3.  now we have a sorted list with no duplicates in ap_proc_ids */
            /* for each of these processor ids, we need to get infomation about
               the boxes in their assumed partition region */

            /* get some stats: check how many point to point communications?
               (what is the max?) */
            /* also get the max distinct AP procs and the max # of entries) */
            send_statbuf[0] = proc_count;
            send_statbuf[1] =
               nalu_hypre_StructAssumedPartMyPartitionNumDistinctProcs(ap);
            send_statbuf[2] = num_my_entries;

            nalu_hypre_MPI_Allreduce(send_statbuf, statbuf, 3, NALU_HYPRE_MPI_INT,
                                nalu_hypre_MPI_MAX, comm);

            //max_proc_count = statbuf[0];

            /* we do not want a single processor to do a ton of point to point
               communications (relative to the number of total processors - how
               much is too much?*/

            /* is there a better way to figure the threshold? */

            /* 3/07 - take out threshold calculation - shouldn't be a problem on
             * large number of processors if box sizes are relativesly
             * similar */

#if 0
            threshold = nalu_hypre_min(12 * ologp, nprocs);

            if ( max_proc_count >=  threshold)
            {
               /* too many! */
               /*if (myid == 0)
                 nalu_hypre_printf("TOO BIG: check 1: max_proc_count = %d\n", max_proc_count);*/

               /* change coarse midstream!- now we will just gather everything! */
               non_ap_gather = 1;

               /*clean up from above */
               nalu_hypre_TFree(ap_proc_ids, NALU_HYPRE_MEMORY_HOST);
            }
#endif

            if (!non_ap_gather)
            {
               /* EXCHANGE DATA information (2 required) :

               if we simply return the boxes in the AP region, we will not have
               the entry information- in particular, we will not have the "info"
               obj.  So we have to get this info by doing a second communication
               where we contact the actual owners of the boxes and request the
               entry info...So:

               (1) exchange #1: contact the AP processor, get the ids of the
               procs with boxes in that AP region (for now we ignore the box
               numbers - since we will get all of the entries from each
               processor)

               (2) exchange #2: use this info to contact the owner processors
               and from them get the rest of the entry infomation: box extents,
               info object, etc. ***note: we will get all of the entries from
               that processor, not just the ones in a particular AP region
               (whose box numbers we ignored above) */

               /* exchange #1 - we send nothing, and the contacted proc returns
                * all of the procs with boxes in its AP region*/

               /* build response object*/
               response_obj.fill_response = nalu_hypre_FillResponseBoxManAssemble1;
               response_obj.data1 = ap; /* needed to fill responses*/
               response_obj.data2 = NULL;

               send_buf = NULL;
               send_buf_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  proc_count + 1, NALU_HYPRE_MEMORY_HOST);
               for (i = 0; i < proc_count + 1; i++)
               {
                  send_buf_starts[i] = 0;
               }

               response_buf = NULL; /*this and the next are allocated in
                                     * exchange data */
               response_buf_starts = NULL;

               /*we expect back the proc id for each box owned */
               size =  sizeof(NALU_HYPRE_Int);

               /* this parameter needs to be the same on all processors */
               /* max_response_size = (global_num_boxes/nprocs)*2;*/
               /* modification - should reduce data passed */
               max_response_size = statbuf[1]; /*max num of distinct procs */

               nalu_hypre_DataExchangeList(proc_count, ap_proc_ids,
                                      send_buf, send_buf_starts,
                                      0, size, &response_obj, max_response_size, 3,
                                      comm, (void**) &response_buf,
                                      &response_buf_starts);

               /*how many items were returned? */
               size = response_buf_starts[proc_count];

               /* alias the response buffer */
               neighbor_proc_ids = response_buf;

               /*clean up*/
               nalu_hypre_TFree(send_buf_starts, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(ap_proc_ids, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(response_buf_starts, NALU_HYPRE_MEMORY_HOST);

               /* create a contact list of these processors (eliminate duplicate
                * procs and also my id ) */

               /*first sort on proc_id  */
               nalu_hypre_qsort0(neighbor_proc_ids, 0, size - 1);

               /* new contact list: */
               contact_proc_ids = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               proc_count = 0; /* to determine the number of unique ids) */

               last_id = -1;

               for (i = 0; i < size; i++)
               {
                  if (neighbor_proc_ids[i] != last_id)
                  {
                     if (neighbor_proc_ids[i] != myid)
                     {
                        contact_proc_ids[proc_count] = neighbor_proc_ids[i];
                        last_id =  neighbor_proc_ids[i];
                        proc_count++;
                     }
                  }
               }

               /* check to see if we have any entries from a processor before
                  contacting(if we have one entry from a processor, then we have
                  all of the entries)

                  we will do we only do this if we have sorted - otherwise we
                  can't easily seach the proc list - this will be most common
                  usage anyways */

               if (nalu_hypre_BoxManIsEntriesSort(manager) && nentries)
               {
                  /* so we can eliminate duplicate contacts */

                  NALU_HYPRE_Int new_count = 0;
                  NALU_HYPRE_Int proc_spot = 0;
                  NALU_HYPRE_Int known_id, contact_id;

                  /* in this case, we can do the "short sort" because we will
                     not have any duplicate proc ids */
                  //short_sort = 1;

                  for (i = 0; i < proc_count; i++)
                  {
                     contact_id = contact_proc_ids[i];

                     while (proc_spot < nentries)
                     {
                        known_id = procs_sort[proc_spot];
                        if (contact_id > known_id)
                        {
                           proc_spot++;
                        }
                        else if (contact_id == known_id)
                        {
                           /* known already - remove from contact list - so go
                              to next i and spot*/
                           proc_spot++;
                           break;
                        }
                        else /* contact_id < known_id */
                        {
                           /* this contact_id is not known already - keep in
                              list*/
                           contact_proc_ids[new_count] = contact_id;
                           new_count++;
                           break;
                        }
                     }
                     if (proc_spot == nentries) /* keep the rest */
                     {
                        contact_proc_ids[new_count] = contact_id;
                        new_count++;
                     }
                  }

                  proc_count = new_count;
               }
#if 0
               /* also can do the short sort if we just have boxes that are
                  ours....here we also don't need to check for duplicates */
               if (nentries == num_my_entries)
               {
                  short_sort = 1;
               }
#endif

               send_buf_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  proc_count + 1, NALU_HYPRE_MEMORY_HOST);
               for (i = 0; i < proc_count + 1; i++)
               {
                  send_buf_starts[i] = 0;
               }
               send_buf = NULL;

               /* exchange #2 - now we contact processors (send nothing) and
                  that processor needs to send us all of their local entry
                  information*/

               entry_response_buf = NULL; /*this and the next are allocated
                                           * in exchange data */
               response_buf_starts = NULL;

               response_obj2.fill_response = nalu_hypre_FillResponseBoxManAssemble2;
               response_obj2.data1 = manager; /* needed to fill responses*/
               response_obj2.data2 = NULL;

               /* How big is an entry?
                    extents - 2*ndim NALU_HYPRE_Ints
                    proc    - 1 NALU_HYPRE_Int
                    id      - 1 NALU_HYPRE_Int
                    info    - info_size in bytes

                  Note: For now, we do not need to send num_ghost, position, or
                  boxman, since this is just generated in addentry. */

               non_info_size = 2 * ndim + 2;
               entry_size_bytes = non_info_size * sizeof(NALU_HYPRE_Int)
                                  + nalu_hypre_BoxManEntryInfoSize(manager);

               /* modification -  use an true max_response_size
                  (should be faster and less communication */
               max_response_size = statbuf[2]; /* max of num_my_entries */

               nalu_hypre_DataExchangeList(proc_count, contact_proc_ids,
                                      send_buf, send_buf_starts, sizeof(NALU_HYPRE_Int),
                                      entry_size_bytes, &response_obj2,
                                      max_response_size, 4,
                                      comm,  &entry_response_buf,
                                      &response_buf_starts);

               /* now we can add entries that are in response_buf - we check for
                  duplicates later  */

               /*how many entries do we have?*/
               response_size = response_buf_starts[proc_count];

               /* do we need more storage ?*/
               if (nentries + response_size >  nalu_hypre_BoxManMaxNEntries(manager))
               {
                  NALU_HYPRE_Int inc_size;

                  inc_size = (response_size + nentries
                              - nalu_hypre_BoxManMaxNEntries(manager));
                  nalu_hypre_BoxManIncSize ( manager, inc_size);

                  entries =  nalu_hypre_BoxManEntries(manager);
                  procs_sort = nalu_hypre_BoxManProcsSort(manager);
                  ids_sort = nalu_hypre_BoxManIdsSort(manager);
               }

               index_ptr = entry_response_buf; /* point into response buf */
               for (i = 0; i < response_size; i++)
               {
                  size = sizeof(NALU_HYPRE_Int);
                  /* imin */
                  for (d = 0; d < ndim; d++)
                  {
                     nalu_hypre_TMemcpy( &tmp_int,  index_ptr, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
                     index_ptr =  (void *) ((char *) index_ptr + size);
                     nalu_hypre_IndexD(imin, d) = tmp_int;
                  }

                  /*imax */
                  for (d = 0; d < ndim; d++)
                  {
                     nalu_hypre_TMemcpy( &tmp_int,  index_ptr, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
                     index_ptr =  (void *) ((char *) index_ptr + size);
                     nalu_hypre_IndexD(imax, d) = tmp_int;
                  }

                  /* proc */
                  tmp_int_ptr = (NALU_HYPRE_Int *) index_ptr;
                  proc = *tmp_int_ptr;
                  index_ptr =  (void *) ((char *) index_ptr + size);

                  /* id */
                  tmp_int_ptr = (NALU_HYPRE_Int *) index_ptr;
                  id = *tmp_int_ptr;
                  index_ptr =  (void *) ((char *) index_ptr + size);

                  /* the info object (now pointer to by index_ptr)
                     is copied by AddEntry*/
                  nalu_hypre_BoxManAddEntry(manager, imin, imax, proc, id, index_ptr);

                  /* start of next entry */
                  index_ptr = (void *)
                              ((char *) index_ptr + nalu_hypre_BoxManEntryInfoSize(manager));
               }

               /* clean up from this section of code*/
               nalu_hypre_TFree(entry_response_buf, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(response_buf_starts, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(send_buf_starts, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(contact_proc_ids, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(neighbor_proc_ids, NALU_HYPRE_MEMORY_HOST); /* response_buf - aliased */

            } /* end of nested non_ap_gather -exchange 1*/

         } /* end of if global boxes */

      } /********** end of gathering for the AP case *****************/

      if (non_ap_gather) /* beginning of gathering for the non-AP case */
      {
         /* collect global data - here we will just send each processor's local
            entries id = myid (not all of the entries in the table). Then we
            will just re-create the entries array instead of looking for
            duplicates and sorting */
         NALU_HYPRE_Int  entry_size_bytes;
         NALU_HYPRE_Int  send_count, send_count_bytes;
         NALU_HYPRE_Int *displs, *recv_counts;
         NALU_HYPRE_Int  recv_buf_size, recv_buf_size_bytes;
         NALU_HYPRE_Int  d;
         NALU_HYPRE_Int  size, non_info_size, position;
         NALU_HYPRE_Int  proc, id;
         NALU_HYPRE_Int  tmp_int;
         NALU_HYPRE_Int *tmp_int_ptr;

         void *send_buf = NULL;
         void *recv_buf = NULL;

         nalu_hypre_BoxManEntry  *entry;

         nalu_hypre_IndexRef index;

         nalu_hypre_Index imin, imax;

         void *index_ptr;
         void *info;

         /* How big is an entry?
            extents - 2*ndim NALU_HYPRE_Ints
            proc    - 1 NALU_HYPRE_Int
            id      - 1 NALU_HYPRE_Int
            info    - info_size in bytes

            Note: For now, we do not need to send num_ghost, position, or
            boxman, since this is just generated in addentry. */

         non_info_size = 2 * ndim + 2;
         entry_size_bytes = non_info_size * sizeof(NALU_HYPRE_Int)
                            + nalu_hypre_BoxManEntryInfoSize(manager);

         /* figure out how many entries each proc has - let the group know */
         send_count =  num_my_entries;
         send_count_bytes = send_count * entry_size_bytes;
         recv_counts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_MPI_Allgather(&send_count_bytes, 1, NALU_HYPRE_MPI_INT,
                             recv_counts, 1, NALU_HYPRE_MPI_INT, comm);

         displs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs, NALU_HYPRE_MEMORY_HOST);
         displs[0] = 0;
         recv_buf_size_bytes = recv_counts[0];
         for (i = 1; i < nprocs; i++)
         {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
            recv_buf_size_bytes += recv_counts[i];
         }
         recv_buf_size = recv_buf_size_bytes / entry_size_bytes;
         /* mydispls = displs[myid]/entry_size_bytes; */

         global_num_boxes = recv_buf_size;

         /* populate the send buffer with my entries (note: these are
            sorted above by increasing id */
         send_buf = nalu_hypre_TAlloc(char, send_count_bytes, NALU_HYPRE_MEMORY_HOST);
         recv_buf = nalu_hypre_TAlloc(char, recv_buf_size_bytes, NALU_HYPRE_MEMORY_HOST);

         index_ptr = send_buf; /* step through send_buf with this pointer */
         /* loop over my entries */
         for (i = 0; i < send_count; i++)
         {
            entry = my_entries[i];

            size = sizeof(NALU_HYPRE_Int);

            /* imin */
            index = nalu_hypre_BoxManEntryIMin(entry);
            for (d = 0; d < ndim; d++)
            {
               tmp_int = nalu_hypre_IndexD(index, d);
               nalu_hypre_TMemcpy( index_ptr,  &tmp_int, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
               index_ptr =  (void *) ((char *) index_ptr + size);
            }

            /* imax */
            index = nalu_hypre_BoxManEntryIMax(entry);
            for (d = 0; d < ndim; d++)
            {
               tmp_int = nalu_hypre_IndexD(index, d);
               nalu_hypre_TMemcpy( index_ptr,  &tmp_int, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
               index_ptr =  (void *) ((char *) index_ptr + size);
            }

            /* proc */
            tmp_int = nalu_hypre_BoxManEntryProc(entry);
            nalu_hypre_TMemcpy( index_ptr,  &tmp_int, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
            index_ptr =  (void *) ((char *) index_ptr + size);

            /* id */
            tmp_int = nalu_hypre_BoxManEntryId(entry);
            nalu_hypre_TMemcpy( index_ptr,  &tmp_int, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
            index_ptr =  (void *) ((char *) index_ptr + size);

            /*info object*/
            size = nalu_hypre_BoxManEntryInfoSize(manager);
            position = nalu_hypre_BoxManEntryPosition(entry);
            info = nalu_hypre_BoxManInfoObject(manager, position);

            nalu_hypre_TMemcpy(index_ptr,  info, char, size, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
            index_ptr =  (void *) ((char *) index_ptr + size);

         } /* end of loop over my entries */

         /* now send_buf is ready to go! */

         nalu_hypre_MPI_Allgatherv(send_buf, send_count_bytes, nalu_hypre_MPI_BYTE,
                              recv_buf, recv_counts, displs, nalu_hypre_MPI_BYTE, comm);

         /* unpack recv_buf into entries - let's just unpack them all into the
            entries table - this way they will already be sorted - so we set
            nentries to zero so that add entries starts at the beginning (i.e.,
            we are deleting the current entries and re-creating)*/

         if (recv_buf_size > nalu_hypre_BoxManMaxNEntries(manager))
         {
            NALU_HYPRE_Int inc_size;

            inc_size = (recv_buf_size - nalu_hypre_BoxManMaxNEntries(manager));
            nalu_hypre_BoxManIncSize ( manager, inc_size);

            nentries = nalu_hypre_BoxManNEntries(manager);
            entries =  nalu_hypre_BoxManEntries(manager);
            procs_sort = nalu_hypre_BoxManProcsSort(manager);
            ids_sort = nalu_hypre_BoxManIdsSort(manager);
         }

         /* now "empty" the entries array */
         nalu_hypre_BoxManNEntries(manager) = 0;
         nalu_hypre_BoxManNumMyEntries(manager) = 0;

         /* point into recv buf and then unpack */
         index_ptr = recv_buf;
         for (i = 0; i < recv_buf_size; i++)
         {

            size = sizeof(NALU_HYPRE_Int);
            /* imin */
            for (d = 0; d < ndim; d++)
            {
               nalu_hypre_TMemcpy( &tmp_int,  index_ptr, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
               index_ptr =  (void *) ((char *) index_ptr + size);
               nalu_hypre_IndexD(imin, d) = tmp_int;
            }

            /*imax */
            for (d = 0; d < ndim; d++)
            {
               nalu_hypre_TMemcpy( &tmp_int,  index_ptr, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
               index_ptr =  (void *) ((char *) index_ptr + size);
               nalu_hypre_IndexD(imax, d) = tmp_int;
            }

            /* proc */
            tmp_int_ptr = (NALU_HYPRE_Int *) index_ptr;
            proc = *tmp_int_ptr;
            index_ptr =  (void *) ((char *) index_ptr + size);

            /* id */
            tmp_int_ptr = (NALU_HYPRE_Int *) index_ptr;
            id = *tmp_int_ptr;
            index_ptr =  (void *) ((char *) index_ptr + size);

            /* info is copied by AddEntry and index_ptr is at info */
            nalu_hypre_BoxManAddEntry( manager, imin,
                                  imax, proc, id,
                                  index_ptr );

            /* start of next entry */
            index_ptr = (void *) ((char *) index_ptr +
                                  nalu_hypre_BoxManEntryInfoSize(manager));
         }

         nalu_hypre_BoxManAllGlobalKnown(manager) = 1;

         nalu_hypre_TFree(send_buf, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(recv_buf, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(recv_counts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(displs, NALU_HYPRE_MEMORY_HOST);

         /* now the entries and procs_sort and ids_sort are already
            sorted */
         need_to_sort = 0;
         nalu_hypre_BoxManIsEntriesSort(manager) = 1;

      } /********* end of non-AP gather *****************/

   }/* end of if (gather entries) for both AP and non-AP */
   else
   {
      /* no gather - so check to see if the entries have been sorted by the user
         - if so we don't need to sort! */
      if  (nalu_hypre_BoxManIsEntriesSort(manager))
      {
         need_to_sort = 0;
      }
   }

   /* we don't need special access to my entries anymore - because we will
      create the sort table */

   nalu_hypre_TFree(nalu_hypre_BoxManMyIds(manager), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_BoxManMyEntries(manager), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxManMyIds(manager) = NULL;
   nalu_hypre_BoxManMyEntries(manager) = NULL;

   /* -----------------------SORT--------------------------------------*/

   /* now everything we need is in entries, also ids and procs have * been added
      to procs_sort and ids_sort, but possibly not sorted. (check need_to_sort
      flag).  If sorted already, then duplicates have been removed. Also there
      may not be any duplicates in the AP case if a duplicate proc check was
      done (depends on if current entry info was sorted)*/

   /* check for and remove duplicate boxes - based on (proc, id) */
   /* at the same time sort the procs_sort and ids_sort and then sort the
    * entries*/
   {
      NALU_HYPRE_Int *order_index = NULL;
      NALU_HYPRE_Int *delete_array = NULL;
      NALU_HYPRE_Int  tmp_id, start, index;
      NALU_HYPRE_Int  first_local;
      NALU_HYPRE_Int  num_procs_sort;
      NALU_HYPRE_Int *proc_offsets;
      NALU_HYPRE_Int  myoffset;
      NALU_HYPRE_Int size;

      nalu_hypre_BoxManEntry  *new_entries;

      /* (TO DO): if we are sorting after the ap gather, then the box ids may
         already be sorted within processor number (depends on if the check for
         contacting duplicate processors was performed....if so, then there may
         be a faster way to sort the proc ids and not mess up the already sorted
         box ids - also there will not be any duplicates )*/

      /* initial... */
      nentries = nalu_hypre_BoxManNEntries(manager);
      entries =  nalu_hypre_BoxManEntries(manager);

      /* these are negative if a proc does not have any local entries in the
         manager */
      first_local = -1;
      myoffset = -1;

      if (need_to_sort)
      {

#if 0
         /* TO DO: add code for the "short sort" - which is don't check for
            duplicates and the boxids are already sorted within each processor
            id - but the proc ids are not sorted */

         if (short_sort)
         {
            /* TO DO: write this */
         }
         else
         {
            /*stuff below */
         }
#endif
         order_index = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nentries, NALU_HYPRE_MEMORY_HOST);
         delete_array =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nentries, NALU_HYPRE_MEMORY_HOST);
         index = 0;

         for (i = 0; i < nentries; i++)
         {
            order_index[i] = i;
         }
         /* sort by proc_id */
         nalu_hypre_qsort3i(procs_sort, ids_sort, order_index, 0, nentries - 1);
         num_procs_sort = 0;
         /* get first id */
         if (nentries)
         {
            tmp_id = procs_sort[0];
            num_procs_sort++;
         }

         /* now sort on ids within each processor number*/
         start = 0;
         for (i = 1; i < nentries; i++)
         {
            if (procs_sort[i] != tmp_id)
            {
               nalu_hypre_qsort2i(ids_sort, order_index, start, i - 1);
               /*now find duplicate ids */
               for (j = start + 1; j < i; j++)
               {
                  if (ids_sort[j] == ids_sort[j - 1])
                  {
                     delete_array[index++] = j;
                  }
               }
               /* update start and tmp_id */
               start = i;
               tmp_id = procs_sort[i];
               num_procs_sort++;
            }
         }
         /* final sort and purge (the last group doesn't get caught in the above
            loop) */
         if (nentries)
         {
            nalu_hypre_qsort2i(ids_sort, order_index, start, nentries - 1);
            /*now find duplicate boxnums */
            for (j = start + 1; j < nentries; j++)
            {
               if (ids_sort[j] == ids_sort[j - 1])
               {
                  delete_array[index++] = j;
               }
            }
         }
         /* now index = the number to delete (in delete_array) */

         if (index)
         {
            /* now delete from sort procs and sort ids -use delete_array because
               these have already been sorted.  also delete from order_index */
            start = delete_array[0];
            j = 0;
            for (i = start; (i + j) < nentries; i++)
            {
               if (j < index)
               {
                  while ((i + j) == delete_array[j]) /* see if deleting
                                                    * consec. items */
                  {
                     j++; /*increase the shift*/
                     if (j == index) { break; }
                  }
               }
               if ((i + j) < nentries) /* if deleting the last item then no moving */
               {
                  ids_sort[i] = ids_sort[i + j];
                  procs_sort[i] =  procs_sort[i + j];
                  order_index[i] = order_index[i + j];
               }
            }
         }

         /*** create new sorted entries and info arrays - delete old one ****/
         {
            NALU_HYPRE_Int position;
            NALU_HYPRE_Int info_size = nalu_hypre_BoxManEntryInfoSize(manager);

            void *index_ptr;
            void *new_info;
            void *info;

            size = nentries - index;
            new_entries =  nalu_hypre_CTAlloc(nalu_hypre_BoxManEntry,  size, NALU_HYPRE_MEMORY_HOST);

            new_info = nalu_hypre_TAlloc(char, size * info_size, NALU_HYPRE_MEMORY_HOST);
            index_ptr = new_info;

            for (i = 0; i < size; i++)
            {
               /* copy the entry */
               nalu_hypre_BoxManEntryCopy(&entries[order_index[i]], &new_entries[i]);

               /* set the new position */
               nalu_hypre_BoxManEntryPosition(&new_entries[i]) = i;

               /* copy the info object */
               position = nalu_hypre_BoxManEntryPosition(&entries[order_index[i]]);
               info = nalu_hypre_BoxManInfoObject(manager, position);

               nalu_hypre_TMemcpy(index_ptr,  info, char,  info_size, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
               index_ptr =  (void *) ((char *) index_ptr + info_size);

            }
            nalu_hypre_TFree(entries, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(nalu_hypre_BoxManInfoObjects(manager), NALU_HYPRE_MEMORY_HOST);

            nalu_hypre_BoxManEntries(manager) = new_entries;
            nalu_hypre_BoxManMaxNEntries(manager) = size;
            nalu_hypre_BoxManNEntries(manager) = size;

            nalu_hypre_BoxManInfoObjects(manager) = new_info;

            nentries = nalu_hypre_BoxManNEntries(manager);
            entries = nalu_hypre_BoxManEntries(manager);
         }

      } /* end of if (need_to_sort) */

      else
      {
         /* no sorting - just get num_procs_sort by looping through procs_sort
            array*/

         num_procs_sort = 0;
         if (nentries > 0)
         {
            tmp_id = procs_sort[0];
            num_procs_sort++;
         }
         for (i = 1; i < nentries; i++)
         {
            if (procs_sort[i] != tmp_id)
            {
               num_procs_sort++;
               tmp_id = procs_sort[i];
            }
         }
      }

      nalu_hypre_BoxManNumProcsSort(manager) = num_procs_sort;

      /* finally, create proc_offsets (myoffset corresponds to local id
         position) first_local is the position in entries; */
      proc_offsets = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs_sort + 1, NALU_HYPRE_MEMORY_HOST);
      proc_offsets[0] = 0;
      if (nentries > 0)
      {
         j = 1;
         tmp_id = procs_sort[0];
         if (myid == tmp_id)
         {
            myoffset = 0;
            first_local = 0;
         }

         for (i = 0; i < nentries; i++)
         {
            if (procs_sort[i] != tmp_id)
            {
               if (myid == procs_sort[i])
               {
                  myoffset = j;
                  first_local = i;
               }
               proc_offsets[j++] = i;
               tmp_id = procs_sort[i];
            }
         }
         proc_offsets[j] = nentries; /* last one */
      }

      nalu_hypre_BoxManProcsSortOffsets(manager) = proc_offsets;
      nalu_hypre_BoxManFirstLocal(manager) = first_local;
      nalu_hypre_BoxManLocalProcOffset(manager) = myoffset;

      /* clean up from this section of code */
      nalu_hypre_TFree(delete_array, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(order_index, NALU_HYPRE_MEMORY_HOST);

   }/* end bracket for all or the sorting stuff */

   {
      /* for the assumed partition case, we can check to see if all the global
         information is known (is a gather has been done) - this could prevent
         future comm costs */

      NALU_HYPRE_Int all_known = 0;
      NALU_HYPRE_Int global_all_known;

      nentries = nalu_hypre_BoxManNEntries(manager);

      if (!nalu_hypre_BoxManAllGlobalKnown(manager) && global_is_gather)
      {
         /*if every processor has its nentries = global_num_boxes, then all is
          * known */
         if (global_num_boxes == nentries) { all_known = 1; }

         nalu_hypre_MPI_Allreduce(&all_known, &global_all_known, 1, NALU_HYPRE_MPI_INT,
                             nalu_hypre_MPI_LAND, comm);

         nalu_hypre_BoxManAllGlobalKnown(manager) = global_all_known;
      }
   }

   /*------------------------------INDEX TABLE ---------------------------*/

   /* now build the index_table and indexes array */
   /* Note: for now we are using the same scheme as in BoxMap  */
   {
      NALU_HYPRE_Int *indexes[NALU_HYPRE_MAXDIM];
      NALU_HYPRE_Int  size[NALU_HYPRE_MAXDIM];
      NALU_HYPRE_Int  iminmax[2];
      NALU_HYPRE_Int  index_not_there;
      NALU_HYPRE_Int  d, e, itsize;
      NALU_HYPRE_Int  mystart, myfinish;
      NALU_HYPRE_Int  imin[NALU_HYPRE_MAXDIM];
      NALU_HYPRE_Int  imax[NALU_HYPRE_MAXDIM];
      NALU_HYPRE_Int  start_loop[NALU_HYPRE_MAXDIM];
      NALU_HYPRE_Int  end_loop[NALU_HYPRE_MAXDIM];
      NALU_HYPRE_Int  loop, range, loop_num;
      NALU_HYPRE_Int *proc_offsets;

      NALU_HYPRE_Int location, spot;

      nalu_hypre_BoxManEntry  **index_table;
      nalu_hypre_BoxManEntry   *entry;
      nalu_hypre_Box           *index_box, *table_box;
      nalu_hypre_Index          stride, loop_size;

      nalu_hypre_IndexRef entry_imin;
      nalu_hypre_IndexRef entry_imax;

      /* initial */
      nentries     = nalu_hypre_BoxManNEntries(manager);
      entries      = nalu_hypre_BoxManEntries(manager);
      proc_offsets = nalu_hypre_BoxManProcsSortOffsets(manager);

      /*------------------------------------------------------
       * Set up the indexes array and record the processor's
       * entries. This will be used in ordering the link list
       * of BoxManEntry- ones on this processor listed first.
       *------------------------------------------------------*/
      itsize = 0;
      for (d = 0; d < ndim; d++)
      {
         /* room for min and max of each entry in each dim */
         indexes[d] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2 * nentries, NALU_HYPRE_MEMORY_HOST);
         size[d] = 0;
      }
      /* loop through each entry and get index */
      for (e = 0; e < nentries; e++)
      {
         entry  = &entries[e]; /* grab the entry - get min and max extents */
         entry_imin = nalu_hypre_BoxManEntryIMin(entry);
         entry_imax = nalu_hypre_BoxManEntryIMax(entry);

         /* in each dim, check if min/max positions are already in the table */
         for (d = 0; d < ndim; d++)
         {
            iminmax[0] = nalu_hypre_IndexD(entry_imin, d);
            iminmax[1] = nalu_hypre_IndexD(entry_imax, d) + 1;

            /* do the min then the max */
            for (i = 0; i < 2; i++)
            {
               /* find the new index position in the indexes array */
               index_not_there = 1;

               if (!i)
               {
                  location = nalu_hypre_BinarySearch2(indexes[d], iminmax[i], 0,
                                                 size[d] - 1, &j);
                  if (location != -1) { index_not_there = 0; }
               }
               else /* for max, we can start seach at min position */
               {
                  location = nalu_hypre_BinarySearch2(indexes[d], iminmax[i], j,
                                                 size[d] - 1, &j);
                  if (location != -1) { index_not_there = 0; }
               }

               /* if the index is already there, don't add it again */
               if (index_not_there)
               {
                  for (k = size[d]; k > j; k--) /* make room for new index */
                  {
                     indexes[d][k] = indexes[d][k - 1];
                  }
                  indexes[d][j] = iminmax[i];
                  size[d]++; /* increase the size in that dimension */
               }
            } /* end of for min and max */
         } /* end of for each dimension of the entry */
      } /* end of for each entry loop */

      if (nentries)
      {
         itsize = 1;
         for (d = 0; d < ndim; d++)
         {
            size[d]--;
            itsize *= size[d];
         }
      }

      /*------------------------------------------------------
       * Set up the table - do offprocessor then on-processor
       *------------------------------------------------------*/

      /* allocate space for table */
      index_table = nalu_hypre_CTAlloc(nalu_hypre_BoxManEntry *,  itsize, NALU_HYPRE_MEMORY_HOST);

      index_box = nalu_hypre_BoxCreate(ndim);
      table_box = nalu_hypre_BoxCreate(ndim);

      /* create a table_box for use below */
      nalu_hypre_SetIndex(stride, 1);
      nalu_hypre_BoxSetExtents(table_box, stride, size);
      nalu_hypre_BoxShiftNeg(table_box, stride); /* Want box to start at 0*/

      /* which are my entries? (on-processor) */
      mystart = nalu_hypre_BoxManFirstLocal(manager);
      if (mystart >= 0 ) /*  we have local entries) because
                             firstlocal = -1 if no local entries */
      {
         loop_num = 3;
         /* basically we have need to do the same code fragment repeated three
            times so that we can do off-proc then on proc entries - this
            ordering is because creating the linked list for overlapping
            boxes */

         myfinish =  proc_offsets[nalu_hypre_BoxManLocalProcOffset(manager) + 1];
         /* #1 do off proc. entries - lower range */
         start_loop[0] = 0;
         end_loop[0] = mystart;
         /* #2 do off proc. entries - upper range */
         start_loop[1] = myfinish;
         end_loop[1] = nentries;
         /* #3 do ON proc. entries */
         start_loop[2] = mystart;
         end_loop[2] = myfinish;
      }
      else /* no on-proc entries */
      {
         loop_num = 1;
         start_loop[0] = 0;
         end_loop[0] = nentries;
      }

      for (loop = 0; loop < loop_num; loop++)
      {
         for (range = start_loop[loop]; range < end_loop[loop]; range++)
         {
            entry = &entries[range];
            entry_imin = nalu_hypre_BoxManEntryIMin(entry);
            entry_imax = nalu_hypre_BoxManEntryIMax(entry);

            /* find the indexes corresponding to the current box - put in imin
               and imax */
            for (d = 0; d < ndim; d++)
            {
               /* need to go to size[d] because that contains the last element */
               location = nalu_hypre_BinarySearch2(
                             indexes[d], nalu_hypre_IndexD(entry_imin, d), 0, size[d], &spot);
               nalu_hypre_IndexD(imin, d) = location;

               location = nalu_hypre_BinarySearch2(
                             indexes[d], nalu_hypre_IndexD(entry_imax, d) + 1, 0, size[d], &spot);
               nalu_hypre_IndexD(imax, d) = location - 1;

            } /* now have imin and imax location in index array*/

            /* set up index table */
            nalu_hypre_BoxSetExtents(index_box, imin, imax);
            nalu_hypre_BoxGetSize(index_box, loop_size);
            nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size, table_box, imin, stride, ii);
            {
               if (!index_table[ii]) /* no entry- add one */
               {
                  index_table[ii] = entry;
               }
               else /* already an entry there - so add to link list for
                       BoxMapEntry - overlapping */
               {
                  nalu_hypre_BoxManEntryNext(entry) = index_table[ii];
                  index_table[ii] = entry;
               }
            }
            nalu_hypre_SerialBoxLoop1End(ii);

         } /* end of subset of entries */
      }/* end of three loops over subsets */

      /* done with the index_table! */
      nalu_hypre_TFree( nalu_hypre_BoxManIndexTable(manager), NALU_HYPRE_MEMORY_HOST); /* in case this is a
                                                        re-assemble - shouldn't
                                                        be though */
      nalu_hypre_BoxManIndexTable(manager) = index_table;

      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_TFree(nalu_hypre_BoxManIndexesD(manager,  d), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_BoxManIndexesD(manager, d) = indexes[d];
         nalu_hypre_BoxManSizeD(manager, d) = size[d];
         nalu_hypre_BoxManLastIndexD(manager, d) = 0;
      }

      nalu_hypre_BoxDestroy(index_box);
      nalu_hypre_BoxDestroy(table_box);

   } /* end of building index table group */

   /* clean up and update*/

   nalu_hypre_BoxManNEntries(manager) = nentries;
   nalu_hypre_BoxManEntries(manager) = entries;

   nalu_hypre_BoxManIsGatherCalled(manager) = 0;
   nalu_hypre_BoxArrayDestroy(gather_regions);
   nalu_hypre_BoxManGatherRegions(manager) =  nalu_hypre_BoxArrayCreate(0, ndim);

   nalu_hypre_BoxManIsAssembled(manager) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Given a box (lower and upper indices), return a list of boxes in the global
 * grid that are intersected by this box. The user must insure that a processor
 * owns the correct global information to do the intersection. For now this is
 * virtually the same as the box map intersect.
 *
 * Notes:
 *
 * (1) This function can also be used in the way that nalu_hypre_BoxMapFindEntry was
 * previously used - just pass in iupper=ilower.
 *
 * (2) return NULL for entries if none are found
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxManIntersect ( nalu_hypre_BoxManager *manager,
                        nalu_hypre_Index ilower,
                        nalu_hypre_Index iupper,
                        nalu_hypre_BoxManEntry ***entries_ptr,
                        NALU_HYPRE_Int *nentries_ptr )
{
   NALU_HYPRE_Int           ndim = nalu_hypre_BoxManNDim(manager);
   NALU_HYPRE_Int           d;
   NALU_HYPRE_Int           find_index_d, current_index_d;
   NALU_HYPRE_Int          *man_indexes_d;
   NALU_HYPRE_Int           man_index_size_d;
   NALU_HYPRE_Int           nentries;
   NALU_HYPRE_Int          *marker, position;
   nalu_hypre_Box          *index_box, *table_box;
   nalu_hypre_Index         stride, loop_size;
   nalu_hypre_Index         man_ilower, man_iupper;
   nalu_hypre_BoxManEntry **index_table;
   nalu_hypre_BoxManEntry **entries;
   nalu_hypre_BoxManEntry  *entry;

#if 0
   NALU_HYPRE_Int   i, cnt;
   NALU_HYPRE_Int  *proc_ids, *ids, *unsort;
   NALU_HYPRE_Int   tmp_id, start;
#endif

   /* can only use after assembling */
   if (!nalu_hypre_BoxManIsAssembled(manager))
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* Check whether the box manager contains any entries */
   if (nalu_hypre_BoxManNEntries(manager) == 0)
   {
      *entries_ptr  = NULL;
      *nentries_ptr = 0;
      return nalu_hypre_error_flag;
   }

   /* Loop through each dimension */
   for (d = 0; d < ndim; d++)
   {
      /* Initialize */
      man_ilower[d] = 0;
      man_iupper[d] = 0;

      man_indexes_d = nalu_hypre_BoxManIndexesD(manager, d);
      man_index_size_d = nalu_hypre_BoxManSizeD(manager, d);

      /* -----find location of ilower[d] in  indexes-----*/
      find_index_d = nalu_hypre_IndexD(ilower, d);

      /* Start looking in place indicated by last_index stored in map */
      current_index_d = nalu_hypre_BoxManLastIndexD(manager, d);

      /* Loop downward if target index is less than current location */
      while ( (current_index_d >= 0 ) &&
              (find_index_d < man_indexes_d[current_index_d]) )
      {
         current_index_d --;
      }

      /* Loop upward if target index is greater than current location */
      while ( (current_index_d <= (man_index_size_d - 1)) &&
              (find_index_d >= man_indexes_d[current_index_d + 1]) )
      {
         current_index_d ++;
      }

      if ( current_index_d > (man_index_size_d - 1) )
      {
         *entries_ptr  = NULL;
         *nentries_ptr = 0;
         return nalu_hypre_error_flag;
      }
      else
      {
         man_ilower[d] = nalu_hypre_max(current_index_d, 0);
      }

      /* -----find location of iupper[d] in  indexes-----*/

      find_index_d = nalu_hypre_IndexD(iupper, d);

      /* Loop upward if target index is greater than current location */
      while ( (current_index_d <= (man_index_size_d - 1)) &&
              (find_index_d >= man_indexes_d[current_index_d + 1]) )
      {
         current_index_d ++;
      }
      if ( current_index_d < 0 )
      {
         *entries_ptr  = NULL;
         *nentries_ptr = 0;
         return nalu_hypre_error_flag;
      }
      else
      {
         man_iupper[d] = nalu_hypre_min(current_index_d, (man_index_size_d - 1));
      }
   }

   /*-----------------------------------------------------------------
    * If we reach this point, then set up the entries array.
    * Use a marker array to ensure unique entries.
    *-----------------------------------------------------------------*/

   nentries = nalu_hypre_BoxManMaxNEntries(manager);
   entries  = nalu_hypre_CTAlloc(nalu_hypre_BoxManEntry *,  nentries, NALU_HYPRE_MEMORY_HOST); /* realloc below */
   marker   = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nentries, NALU_HYPRE_MEMORY_HOST);
   index_table = nalu_hypre_BoxManIndexTable(manager);

   nentries = 0;

   table_box = nalu_hypre_BoxCreate(ndim);
   index_box = nalu_hypre_BoxCreate(ndim);

   nalu_hypre_SetIndex(stride, 1);
   nalu_hypre_BoxSetExtents(table_box, stride, nalu_hypre_BoxManSize(manager));
   nalu_hypre_BoxShiftNeg(table_box, stride); /* Want box to start at 0*/
   nalu_hypre_BoxSetExtents(index_box, man_ilower, man_iupper);
   nalu_hypre_BoxGetSize(index_box, loop_size);
   nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size, table_box, man_ilower, stride, ii);
   {
      entry = index_table[ii];

      while (entry != NULL)
      {
         position = nalu_hypre_BoxManEntryPosition(entry);

         if (marker[position] == 0) /* Add entry and mark as added */
         {
            entries[nentries] = entry;
            marker[position]  = 1;
            nentries++;
         }

         entry = nalu_hypre_BoxManEntryNext(entry);
      }
   }
   nalu_hypre_SerialBoxLoop1End(ii);

   entries  = nalu_hypre_TReAlloc(entries,  nalu_hypre_BoxManEntry *,  nentries, NALU_HYPRE_MEMORY_HOST);

   /* Reset the last index in the manager */
   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxManLastIndexD(manager, d) = man_ilower[d];
   }

   nalu_hypre_BoxDestroy(table_box);
   nalu_hypre_BoxDestroy(index_box);
   nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);

   *entries_ptr  = entries;
   *nentries_ptr = nentries;

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * contact message is null.  need to return the (proc) id of each box in our
 * assumed partition.
 *
 * 1/07 - just returning distinct proc ids.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_FillResponseBoxManAssemble1( void *p_recv_contact_buf,
                                   NALU_HYPRE_Int contact_size,
                                   NALU_HYPRE_Int contact_proc,
                                   void *ro, MPI_Comm comm,
                                   void **p_send_response_buf,
                                   NALU_HYPRE_Int *response_message_size )
{
   NALU_HYPRE_Int    myid, i, index;
   NALU_HYPRE_Int    size, num_boxes, num_objects;
   NALU_HYPRE_Int   *proc_ids;
   NALU_HYPRE_Int   *send_response_buf = (NALU_HYPRE_Int *) *p_send_response_buf;

   nalu_hypre_DataExchangeResponse  *response_obj = (nalu_hypre_DataExchangeResponse  *)ro;
   nalu_hypre_StructAssumedPart     *ap = (nalu_hypre_StructAssumedPart     *)response_obj->data1;

   NALU_HYPRE_Int overhead = response_obj->send_response_overhead;

   /* initialize stuff */
   nalu_hypre_MPI_Comm_rank(comm, &myid );

   proc_ids =  nalu_hypre_StructAssumedPartMyPartitionProcIds(ap);

   /* we need to send back the list of all the processor ids for the boxes */

   /* NOTE: in the AP, boxes with the same proc id are adjacent (but proc ids
      not in any sorted order) */

   /* how many boxes do we have in the AP?*/
   num_boxes = nalu_hypre_StructAssumedPartMyPartitionIdsSize(ap);
   /* how many procs do we have in the AP?*/
   num_objects = nalu_hypre_StructAssumedPartMyPartitionNumDistinctProcs(ap);

   /* num_objects is then how much we need to send*/

   /* check storage in send_buf for adding the information */
   /* note: we are returning objects that are 1 ints in size */

   if ( response_obj->send_response_storage  < num_objects  )
   {
      response_obj->send_response_storage =  nalu_hypre_max(num_objects, 10);
      size =  1 * (response_obj->send_response_storage + overhead);
      send_response_buf = nalu_hypre_TReAlloc( send_response_buf,  NALU_HYPRE_Int,
                                          size, NALU_HYPRE_MEMORY_HOST);
      *p_send_response_buf = send_response_buf;
   }

   /* populate send_response_buf with distinct proc ids*/
   index = 0;

   if (num_objects > 0)
   {
      send_response_buf[index++] = proc_ids[0];
   }

   for (i = 1; i < num_boxes && index < num_objects; i++)
   {
      /* processor id */
      if (proc_ids[i] != proc_ids[i - 1])
      {
         send_response_buf[index++] = proc_ids[i];
      }
   }

   /* return variables */
   *response_message_size = num_objects;
   *p_send_response_buf = send_response_buf;

   return nalu_hypre_error_flag;
}
/******************************************************************************
 * contact message is null.  the response needs to be the all our entries (with
 * id = myid).
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_FillResponseBoxManAssemble2( void *p_recv_contact_buf,
                                   NALU_HYPRE_Int contact_size,
                                   NALU_HYPRE_Int contact_proc,
                                   void *ro, MPI_Comm comm,
                                   void **p_send_response_buf,
                                   NALU_HYPRE_Int *response_message_size )
{
   NALU_HYPRE_Int          myid, i, d, size, position;
   NALU_HYPRE_Int          proc_id, box_id, tmp_int;
   NALU_HYPRE_Int          entry_size_bytes;
   nalu_hypre_BoxManEntry *entry;
   nalu_hypre_IndexRef     index;
   void              *info, *index_ptr;

   void                       *send_response_buf = (void *) *p_send_response_buf;
   nalu_hypre_DataExchangeResponse *response_obj = (nalu_hypre_DataExchangeResponse *)ro;
   nalu_hypre_BoxManager           *manager = (nalu_hypre_BoxManager           *)response_obj->data1;
   NALU_HYPRE_Int                   overhead = response_obj->send_response_overhead;

   NALU_HYPRE_Int           ndim = nalu_hypre_BoxManNDim(manager);
   nalu_hypre_BoxManEntry **my_entries = nalu_hypre_BoxManMyEntries(manager) ;
   NALU_HYPRE_Int           num_my_entries = nalu_hypre_BoxManNumMyEntries(manager);

   /*initialize stuff */
   nalu_hypre_MPI_Comm_rank(comm, &myid );

   entry_size_bytes = 8 * sizeof(NALU_HYPRE_Int) + nalu_hypre_BoxManEntryInfoSize(manager);

   /* num_my_entries is the amount of information to send */

   /*check storage in send_buf for adding the information */
   if ( response_obj->send_response_storage  < num_my_entries  )
   {
      response_obj->send_response_storage =  num_my_entries;
      size =  entry_size_bytes * (response_obj->send_response_storage + overhead);
      send_response_buf = nalu_hypre_TReAlloc( (char*)send_response_buf, char, size, NALU_HYPRE_MEMORY_HOST);
      *p_send_response_buf = send_response_buf;
   }

   index_ptr = send_response_buf; /* step through send_buf with this pointer */

   for (i = 0; i < num_my_entries; i++)
   {
      entry = my_entries[i];

      /*pack response buffer with information */

      size = sizeof(NALU_HYPRE_Int);
      /* imin */
      index = nalu_hypre_BoxManEntryIMin(entry);
      for (d = 0; d < ndim; d++)
      {
         tmp_int = nalu_hypre_IndexD(index, d);
         nalu_hypre_TMemcpy( index_ptr,  &tmp_int, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         index_ptr =  (void *) ((char *) index_ptr + size);
      }
      /* imax */
      index = nalu_hypre_BoxManEntryIMax(entry);
      for (d = 0; d < ndim; d++)
      {
         tmp_int = nalu_hypre_IndexD(index, d);
         nalu_hypre_TMemcpy( index_ptr,  &tmp_int, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         index_ptr =  (void *) ((char *) index_ptr + size);
      }
      /* proc */
      proc_id =  nalu_hypre_BoxManEntryProc(entry);
      nalu_hypre_TMemcpy( index_ptr,  &proc_id, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      index_ptr =  (void *) ((char *) index_ptr + size);

      /* id */
      box_id = nalu_hypre_BoxManEntryId(entry);
      nalu_hypre_TMemcpy( index_ptr,  &box_id, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      index_ptr =  (void *) ((char *) index_ptr + size);

      /*info*/
      size = nalu_hypre_BoxManEntryInfoSize(manager);
      position = nalu_hypre_BoxManEntryPosition(entry);
      info = nalu_hypre_BoxManInfoObject(manager, position);

      nalu_hypre_TMemcpy(index_ptr,  info, char, size, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

      index_ptr =  (void *) ((char *) index_ptr + size);

   }

   /* now send_response_buf is full */

   /* return variable */
   *response_message_size = num_my_entries;
   *p_send_response_buf = send_response_buf;

   return nalu_hypre_error_flag;
}
