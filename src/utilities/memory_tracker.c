/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Memory tracker
 * Do NOT use nalu_hypre_T* in this file since we don't want to track them,
 * Do NOT use nalu_hypre_printf, nalu_hypre_fprintf, which have nalu_hypre_TAlloc/Free
 * endless for-loop otherwise
 *--------------------------------------------------------------------------*/

#include "_nalu_hypre_utilities.h"

#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)

nalu_hypre_MemoryTracker *_nalu_hypre_memory_tracker = NULL;

/* accessor to the global ``_nalu_hypre_memory_tracker'' */
nalu_hypre_MemoryTracker*
nalu_hypre_memory_tracker(void)
{
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp critical
#endif
   {
      if (!_nalu_hypre_memory_tracker)
      {
         _nalu_hypre_memory_tracker = nalu_hypre_MemoryTrackerCreate();
      }
   }

   return _nalu_hypre_memory_tracker;
}

size_t nalu_hypre_total_bytes[nalu_hypre_NUM_MEMORY_LOCATION];
size_t nalu_hypre_peak_bytes[nalu_hypre_NUM_MEMORY_LOCATION];
size_t nalu_hypre_current_bytes[nalu_hypre_NUM_MEMORY_LOCATION];
NALU_HYPRE_Int nalu_hypre_memory_tracker_print = 0;
char nalu_hypre_memory_tracker_filename[NALU_HYPRE_MAX_FILE_NAME_LEN] = "HypreMemoryTrack.log";

char *nalu_hypre_basename(const char *name)
{
   const char *base = name;
   while (*name)
   {
      if (*name++ == '/')
      {
         base = name;
      }
   }
   return (char *) base;
}

nalu_hypre_MemcpyType
nalu_hypre_GetMemcpyType(nalu_hypre_MemoryLocation dst,
                    nalu_hypre_MemoryLocation src)
{
   NALU_HYPRE_Int d = 0, s = 0;

   if      (dst == nalu_hypre_MEMORY_HOST   || dst == nalu_hypre_MEMORY_HOST_PINNED) { d = 0; }
   else if (dst == nalu_hypre_MEMORY_DEVICE || dst == nalu_hypre_MEMORY_UNIFIED)     { d = 1; }

   if      (src == nalu_hypre_MEMORY_HOST   || src == nalu_hypre_MEMORY_HOST_PINNED) { s = 0; }
   else if (src == nalu_hypre_MEMORY_DEVICE || src == nalu_hypre_MEMORY_UNIFIED)     { s = 1; }

   if (d == 0 && s == 0) { return nalu_hypre_MEMCPY_H2H; }
   if (d == 0 && s == 1) { return nalu_hypre_MEMCPY_D2H; }
   if (d == 1 && s == 0) { return nalu_hypre_MEMCPY_H2D; }
   if (d == 1 && s == 1) { return nalu_hypre_MEMCPY_D2D; }

   return nalu_hypre_MEMCPY_NUM_TYPES;
}

nalu_hypre_int
nalu_hypre_MemoryTrackerQueueCompSort(const void *e1,
                                 const void *e2)
{
   void *p1 = ((nalu_hypre_MemoryTrackerEntry *) e1) -> ptr;
   void *p2 = ((nalu_hypre_MemoryTrackerEntry *) e2) -> ptr;

   if (p1 < p2) { return -1; }
   if (p1 > p2) { return  1; }

   size_t t1 = ((nalu_hypre_MemoryTrackerEntry *) e1) -> time_step;
   size_t t2 = ((nalu_hypre_MemoryTrackerEntry *) e2) -> time_step;

   if (t1 < t2) { return -1; }
   if (t1 > t2) { return  1; }

   return 0;
}


nalu_hypre_int
nalu_hypre_MemoryTrackerQueueCompSearch(const void *e1,
                                   const void *e2)
{
   void *p1 = ((nalu_hypre_MemoryTrackerEntry **) e1)[0] -> ptr;
   void *p2 = ((nalu_hypre_MemoryTrackerEntry **) e2)[0] -> ptr;

   if (p1 < p2) { return -1; }
   if (p1 > p2) { return  1; }

   return 0;
}

nalu_hypre_MemoryTrackerEvent
nalu_hypre_MemoryTrackerGetNext(nalu_hypre_MemoryTracker *tracker)
{
   nalu_hypre_MemoryTrackerEvent i, k = NALU_HYPRE_MEMORY_NUM_EVENTS;
   nalu_hypre_MemoryTrackerQueue *q = tracker->queue;

   for (i = NALU_HYPRE_MEMORY_EVENT_ALLOC; i < NALU_HYPRE_MEMORY_NUM_EVENTS; i++)
   {
      if (q[i].head >= q[i].actual_size)
      {
         continue;
      }

      if (k == NALU_HYPRE_MEMORY_NUM_EVENTS || q[i].data[q[i].head].time_step < q[k].data[q[k].head].time_step)
      {
         k = i;
      }
   }

   return k;
}

NALU_HYPRE_Int
nalu_hypre_MemoryTrackerSortQueue(nalu_hypre_MemoryTrackerQueue *q)
{
   size_t i = 0;

   if (!q) { return nalu_hypre_error_flag; }

   free(q->sorted_data);
   free(q->sorted_data_compressed_offset);
   free(q->sorted_data_compressed);

   q->sorted_data = (nalu_hypre_MemoryTrackerEntry *) malloc(q->actual_size * sizeof(
                                                           nalu_hypre_MemoryTrackerEntry));
   memcpy(q->sorted_data, q->data, q->actual_size * sizeof(nalu_hypre_MemoryTrackerEntry));
   qsort(q->sorted_data, q->actual_size, sizeof(nalu_hypre_MemoryTrackerEntry),
         nalu_hypre_MemoryTrackerQueueCompSort);

   q->sorted_data_compressed_len = 0;
   q->sorted_data_compressed_offset = (size_t *) malloc(q->actual_size * sizeof(size_t));
   q->sorted_data_compressed = (nalu_hypre_MemoryTrackerEntry **) malloc((q->actual_size + 1) * sizeof(
                                                                       nalu_hypre_MemoryTrackerEntry *));

   for (i = 0; i < q->actual_size; i++)
   {
      if (i == 0 || q->sorted_data[i].ptr != q->sorted_data[i - 1].ptr)
      {
         q->sorted_data_compressed_offset[q->sorted_data_compressed_len] = i;
         q->sorted_data_compressed[q->sorted_data_compressed_len] = &q->sorted_data[i];
         q->sorted_data_compressed_len ++;
      }
   }
   q->sorted_data_compressed[q->sorted_data_compressed_len] = q->sorted_data + q->actual_size;

   q->sorted_data_compressed_offset = (size_t *)
                                      realloc(q->sorted_data_compressed_offset, q->sorted_data_compressed_len * sizeof(size_t));

   q->sorted_data_compressed = (nalu_hypre_MemoryTrackerEntry **)
                               realloc(q->sorted_data_compressed,
                                       (q->sorted_data_compressed_len + 1) * sizeof(nalu_hypre_MemoryTrackerEntry *));

   return nalu_hypre_error_flag;
}

nalu_hypre_MemoryTracker *
nalu_hypre_MemoryTrackerCreate()
{
   nalu_hypre_MemoryTracker *ptr = (nalu_hypre_MemoryTracker *) calloc(1, sizeof(nalu_hypre_MemoryTracker));
   return ptr;
}

void
nalu_hypre_MemoryTrackerDestroy(nalu_hypre_MemoryTracker *tracker)
{
   if (tracker)
   {
      NALU_HYPRE_Int i;

      for (i = 0; i < NALU_HYPRE_MEMORY_NUM_EVENTS; i++)
      {
         free(tracker->queue[i].data);
         free(tracker->queue[i].sorted_data);
         free(tracker->queue[i].sorted_data_compressed_offset);
         free(tracker->queue[i].sorted_data_compressed);
      }

      free(tracker);
   }
}

NALU_HYPRE_Int
nalu_hypre_MemoryTrackerSetPrint(NALU_HYPRE_Int do_print)
{
   nalu_hypre_memory_tracker_print = do_print;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_MemoryTrackerSetFileName(const char *file_name)
{
   snprintf(nalu_hypre_memory_tracker_filename, NALU_HYPRE_MAX_FILE_NAME_LEN, "%s", file_name);

   return nalu_hypre_error_flag;
}

void
nalu_hypre_MemoryTrackerInsert1(const char           *action,
                           void                 *ptr,
                           size_t                nbytes,
                           nalu_hypre_MemoryLocation  memory_location,
                           const char           *filename,
                           const char           *function,
                           NALU_HYPRE_Int             line)
{
   nalu_hypre_MemoryTrackerInsert2(action, ptr, NULL, nbytes, memory_location, nalu_hypre_MEMORY_UNDEFINED,
                              filename, function, line);
}

void
nalu_hypre_MemoryTrackerInsert2(const char           *action,
                           void                 *ptr,
                           void                 *ptr2,
                           size_t                nbytes,
                           nalu_hypre_MemoryLocation  memory_location,
                           nalu_hypre_MemoryLocation  memory_location2,
                           const char           *filename,
                           const char           *function,
                           NALU_HYPRE_Int             line)
{
   if (ptr == NULL)
   {
      return;
   }

   nalu_hypre_MemoryTracker *tracker = nalu_hypre_memory_tracker();

   nalu_hypre_MemoryTrackerEvent q;

   /* Get the proper queue based on the action */

   if (strstr(action, "alloc") != NULL)
   {
      /* including malloc, alloc and the malloc in realloc */
      q = NALU_HYPRE_MEMORY_EVENT_ALLOC;
   }
   else if (strstr(action, "free") != NULL)
   {
      /* including free and the free in realloc */
      q = NALU_HYPRE_MEMORY_EVENT_FREE;
   }
   else if (strstr(action, "memcpy") != NULL)
   {
      /* including memcpy */
      q = NALU_HYPRE_MEMORY_EVENT_COPY;
   }
   else
   {
      return;
   }

   nalu_hypre_MemoryTrackerQueue *queue = &tracker->queue[q];

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp critical
#endif
   {
      /* resize if not enough space */

      if (queue->alloced_size <= queue->actual_size)
      {
         queue->alloced_size = 2 * queue->alloced_size + 1;
         queue->data = (nalu_hypre_MemoryTrackerEntry *) realloc(queue->data,
                                                            queue->alloced_size * sizeof(nalu_hypre_MemoryTrackerEntry));
      }

      nalu_hypre_assert(queue->actual_size < queue->alloced_size);

      /* insert an entry */
      nalu_hypre_MemoryTrackerEntry *entry = queue->data + queue->actual_size;

      entry->index = queue->actual_size;
      entry->time_step = tracker->curr_time_step;
      sprintf(entry->action, "%s", action);
      entry->ptr = ptr;
      entry->ptr2 = ptr2;
      entry->nbytes = nbytes;
      entry->memory_location = memory_location;
      entry->memory_location2 = memory_location2;
      sprintf(entry->filename, "%s", filename);
      sprintf(entry->function, "%s", function);
      entry->line = line;
      entry->pair = (size_t) -1;

#if 0
      NALU_HYPRE_Int myid;
      nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid);
      if (myid == 0 && entry->time_step == 28111) {assert(0);}
#endif

      /* increase the time step */
      tracker->curr_time_step ++;

      /* increase the queue length by 1 */
      queue->actual_size ++;
   }
}

NALU_HYPRE_Int
nalu_hypre_PrintMemoryTracker( size_t     *totl_bytes_o,
                          size_t     *peak_bytes_o,
                          size_t     *curr_bytes_o,
                          NALU_HYPRE_Int   do_print,
                          const char *fname )
{
   char   filename[NALU_HYPRE_MAX_FILE_NAME_LEN + 16];
   FILE  *file = NULL;
   size_t totl_bytes[nalu_hypre_NUM_MEMORY_LOCATION] = {0};
   size_t peak_bytes[nalu_hypre_NUM_MEMORY_LOCATION] = {0};
   size_t curr_bytes[nalu_hypre_NUM_MEMORY_LOCATION] = {0};
   size_t copy_bytes[nalu_hypre_MEMCPY_NUM_TYPES] = {0};
   size_t j;
   nalu_hypre_MemoryTrackerEvent i;
   //NALU_HYPRE_Real t0 = nalu_hypre_MPI_Wtime();

   NALU_HYPRE_Int leakcheck = 1;

   nalu_hypre_MemoryTracker *tracker = nalu_hypre_memory_tracker();
   nalu_hypre_MemoryTrackerQueue *qq = tracker->queue;
   nalu_hypre_MemoryTrackerQueue *qa = &qq[NALU_HYPRE_MEMORY_EVENT_ALLOC];
   nalu_hypre_MemoryTrackerQueue *qf = &qq[NALU_HYPRE_MEMORY_EVENT_FREE];

   if (do_print)
   {
      NALU_HYPRE_Int myid;
      nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid);

      if (fname)
      {
         nalu_hypre_sprintf(filename, "%s.%05d.csv", fname, myid);
      }
      else
      {
         nalu_hypre_sprintf(filename, "HypreMemoryTrack.log.%05d.csv", myid);
      }

      if ((file = fopen(filename, "w")) == NULL)
      {
         fprintf(stderr, "Error: can't open output file %s\n", filename);
         return nalu_hypre_error_flag;
      }

      fprintf(file, "\"==== Operations:\"\n");
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
              "ID", "EVENT", "ADDRESS1", "ADDRESS2", "BYTE", "LOCATION1", "LOCATION2",
              "FILE", "LINE", "FUNCTION", "HOST", "PINNED", "DEVICE", "UNIFIED");
   }

   if (leakcheck)
   {
      //NALU_HYPRE_Real t0 = nalu_hypre_MPI_Wtime();
      nalu_hypre_MemoryTrackerSortQueue(qf);
      //NALU_HYPRE_Real t1 = nalu_hypre_MPI_Wtime() - t0;
      //printf("Sort Time %.2f\n", t1);
   }

   size_t total_num_events = 0;
   size_t total_num_events_2 = 0;
   for (i = NALU_HYPRE_MEMORY_EVENT_ALLOC; i < NALU_HYPRE_MEMORY_NUM_EVENTS; i++)
   {
      total_num_events_2 += qq[i].actual_size;
   }

   for (i = nalu_hypre_MemoryTrackerGetNext(tracker); i < NALU_HYPRE_MEMORY_NUM_EVENTS;
        i = nalu_hypre_MemoryTrackerGetNext(tracker))
   {
      total_num_events ++;

      nalu_hypre_MemoryTrackerEntry *entry = &qq[i].data[qq[i].head++];

      if (strstr(entry->action, "alloc") != NULL)
      {
         totl_bytes[entry->memory_location] += entry->nbytes;

         if (leakcheck)
         {
            curr_bytes[entry->memory_location] += entry->nbytes;
            peak_bytes[entry->memory_location] = nalu_hypre_max( curr_bytes[entry->memory_location],
                                                            peak_bytes[entry->memory_location] );
         }

         if (leakcheck && entry->pair == (size_t) -1)
         {
            nalu_hypre_MemoryTrackerEntry key = { .ptr = entry->ptr };
            nalu_hypre_MemoryTrackerEntry *key_ptr = &key;

            nalu_hypre_MemoryTrackerEntry **result = bsearch(&key_ptr,
                                                        qf->sorted_data_compressed,
                                                        qf->sorted_data_compressed_len,
                                                        sizeof(nalu_hypre_MemoryTrackerEntry *),
                                                        nalu_hypre_MemoryTrackerQueueCompSearch);
            if (result)
            {
               j = result - qf->sorted_data_compressed;
               nalu_hypre_MemoryTrackerEntry *p = qf->sorted_data + qf->sorted_data_compressed_offset[j];

               if (p < qf->sorted_data_compressed[j + 1])
               {
                  nalu_hypre_assert(p->ptr == entry->ptr);
                  entry->pair = p->index;
                  nalu_hypre_assert(qf->data[p->index].pair == -1);
                  nalu_hypre_assert(qq[i].head - 1 == entry->index);
                  qf->data[p->index].pair = entry->index;
                  qf->data[p->index].nbytes = entry->nbytes;

                  qf->sorted_data_compressed_offset[j] ++;
               }
            }
         }
      }
      else if (leakcheck && strstr(entry->action, "free") != NULL)
      {
         if (entry->pair < qa->actual_size)
         {
            curr_bytes[entry->memory_location] -= qa->data[entry->pair].nbytes;
         }
      }
      else if (strstr(entry->action, "memcpy") != NULL)
      {
         copy_bytes[nalu_hypre_GetMemcpyType(entry->memory_location, entry->memory_location2)] += entry->nbytes;
      }

      if (do_print)
      {
         char memory_location[256];
         char memory_location2[256];
         char nbytes[32];

         nalu_hypre_GetMemoryLocationName(entry->memory_location, memory_location);
         nalu_hypre_GetMemoryLocationName(entry->memory_location2, memory_location2);

         if (entry->nbytes != (size_t) -1)
         {
            sprintf(nbytes, "%zu", entry->nbytes);
         }
         else
         {
            sprintf(nbytes, "%s", "--");
         }

         fprintf(file,
                 " %6zu, %9s, %16p, %16p, %10s, %10s, %10s, %28s, %8d, %54s, %11zu, %11zu, %11zu, %11zu\n",
                 entry->time_step,
                 entry->action,
                 entry->ptr,
                 entry->ptr2,
                 nbytes,
                 memory_location,
                 memory_location2,
                 nalu_hypre_basename(entry->filename),
                 entry->line,
                 entry->function,
                 curr_bytes[nalu_hypre_MEMORY_HOST],
                 curr_bytes[nalu_hypre_MEMORY_HOST_PINNED],
                 curr_bytes[nalu_hypre_MEMORY_DEVICE],
                 curr_bytes[nalu_hypre_MEMORY_UNIFIED]
                );
      }
   }

   nalu_hypre_assert(total_num_events == total_num_events_2);

   if (do_print)
   {
      fprintf(file, "\n\"==== Total Allocation (byte):\"\n");
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
              "", "", "", "", "", "", "", "", "", "", "HOST", "PINNED", "DEVICE", "UNIFIED");
      fprintf(file,
              " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
              "", "", "", "", "", "", "", "", "", "",
              totl_bytes[nalu_hypre_MEMORY_HOST],
              totl_bytes[nalu_hypre_MEMORY_HOST_PINNED],
              totl_bytes[nalu_hypre_MEMORY_DEVICE],
              totl_bytes[nalu_hypre_MEMORY_UNIFIED]);

      fprintf(file, "\n\"==== Peak Allocation (byte):\"\n");
      /*fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
            "", "", "", "", "", "", "", "", "", "", "HOST", "PINNED", "DEVICE", "UNIFIED"); */
      fprintf(file,
              " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
              "", "", "", "", "", "", "", "", "", "",
              peak_bytes[nalu_hypre_MEMORY_HOST],
              peak_bytes[nalu_hypre_MEMORY_HOST_PINNED],
              peak_bytes[nalu_hypre_MEMORY_DEVICE],
              peak_bytes[nalu_hypre_MEMORY_UNIFIED]);

      fprintf(file, "\n\"==== Reachable Allocation (byte):\"\n");
      /* fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
            "", "", "", "", "", "", "", "", "", "", "HOST", "PINNED", "DEVICE", "UNIFIED"); */
      fprintf(file,
              " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
              "", "", "", "", "", "", "", "", "", "",
              curr_bytes[nalu_hypre_MEMORY_HOST],
              curr_bytes[nalu_hypre_MEMORY_HOST_PINNED],
              curr_bytes[nalu_hypre_MEMORY_DEVICE],
              curr_bytes[nalu_hypre_MEMORY_UNIFIED]);

      fprintf(file, "\n\"==== Memory Copy (byte):\"\n");
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
              "", "", "", "", "", "", "", "", "", "", "H2H", "D2H", "H2D", "D2D");
      fprintf(file,
              " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
              "", "", "", "", "", "", "", "", "", "",
              copy_bytes[nalu_hypre_MEMCPY_H2H],
              copy_bytes[nalu_hypre_MEMCPY_D2H],
              copy_bytes[nalu_hypre_MEMCPY_H2D],
              copy_bytes[nalu_hypre_MEMCPY_D2D]);

   }

   if (totl_bytes_o)
   {
      totl_bytes_o[nalu_hypre_MEMORY_HOST] = totl_bytes[nalu_hypre_MEMORY_HOST];
      totl_bytes_o[nalu_hypre_MEMORY_HOST_PINNED] = totl_bytes[nalu_hypre_MEMORY_HOST_PINNED];
      totl_bytes_o[nalu_hypre_MEMORY_DEVICE] = totl_bytes[nalu_hypre_MEMORY_DEVICE];
      totl_bytes_o[nalu_hypre_MEMORY_UNIFIED] = totl_bytes[nalu_hypre_MEMORY_UNIFIED];
   }

   if (peak_bytes_o)
   {
      peak_bytes_o[nalu_hypre_MEMORY_HOST] = peak_bytes[nalu_hypre_MEMORY_HOST];
      peak_bytes_o[nalu_hypre_MEMORY_HOST_PINNED] = peak_bytes[nalu_hypre_MEMORY_HOST_PINNED];
      peak_bytes_o[nalu_hypre_MEMORY_DEVICE] = peak_bytes[nalu_hypre_MEMORY_DEVICE];
      peak_bytes_o[nalu_hypre_MEMORY_UNIFIED] = peak_bytes[nalu_hypre_MEMORY_UNIFIED];
   }

   if (curr_bytes_o)
   {
      curr_bytes_o[nalu_hypre_MEMORY_HOST] = curr_bytes[nalu_hypre_MEMORY_HOST];
      curr_bytes_o[nalu_hypre_MEMORY_HOST_PINNED] = curr_bytes[nalu_hypre_MEMORY_HOST_PINNED];
      curr_bytes_o[nalu_hypre_MEMORY_DEVICE] = curr_bytes[nalu_hypre_MEMORY_DEVICE];
      curr_bytes_o[nalu_hypre_MEMORY_UNIFIED] = curr_bytes[nalu_hypre_MEMORY_UNIFIED];
   }

#if defined(NALU_HYPRE_DEBUG)
   for (i = NALU_HYPRE_MEMORY_EVENT_ALLOC; i < NALU_HYPRE_MEMORY_NUM_EVENTS; i++)
   {
      nalu_hypre_assert(qq[i].head == qq[i].actual_size);
   }
#endif

   if (leakcheck && do_print)
   {
      fprintf(file, "\n\"==== Warnings:\"\n");

      for (j = 0; j < qa->actual_size; j++)
      {
         nalu_hypre_MemoryTrackerEntry *entry = &qa->data[j];
         if (entry->pair == (size_t) -1)
         {
            fprintf(file, " %6zu, %9s, %16p, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
                    entry->time_step, entry->action, entry->ptr, "", "", "", "", "", "", "Not freed", "", "", "", "");
         }
         else
         {
            nalu_hypre_assert(entry->pair < qf->actual_size);
            nalu_hypre_assert(qf->data[entry->pair].ptr == entry->ptr);
            nalu_hypre_assert(qf->data[entry->pair].nbytes == entry->nbytes);
            nalu_hypre_assert(qf->data[entry->pair].memory_location == entry->memory_location);
            nalu_hypre_assert(qf->data[entry->pair].pair == j);
         }
      }

      for (j = 0; j < qf->actual_size; j++)
      {
         nalu_hypre_MemoryTrackerEntry *entry = &qf->data[j];
         if (entry->pair == (size_t) -1)
         {
            fprintf(file, " %6zu, %9s, %16p, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
                    entry->time_step, entry->action, entry->ptr, "", "", "", "", "", "", "Unpaired free", "", "", "",
                    "");
         }
         else
         {
            nalu_hypre_assert(entry->pair < qa->actual_size);
            nalu_hypre_assert(qa->data[entry->pair].ptr == entry->ptr);
            nalu_hypre_assert(qa->data[entry->pair].nbytes == entry->nbytes);
            nalu_hypre_assert(qa->data[entry->pair].memory_location == entry->memory_location);
            nalu_hypre_assert(qa->data[entry->pair].pair == j);
         }
      }
   }

   if (file)
   {
      fclose(file);
   }

   if (leakcheck)
   {
      nalu_hypre_MemoryLocation t;

      for (t = nalu_hypre_MEMORY_HOST; t <= nalu_hypre_MEMORY_UNIFIED; t++)
      {
         if (curr_bytes[t])
         {
            char memory_location[256];
            nalu_hypre_GetMemoryLocationName(t, memory_location);
            fprintf(stderr, "%zu bytes of %s memory may not be freed\n", curr_bytes[t], memory_location);
         }

      }

      for (t = nalu_hypre_MEMORY_HOST; t <= nalu_hypre_MEMORY_UNIFIED; t++)
      {
         nalu_hypre_assert(curr_bytes[t] == 0);
      }
   }

   //NALU_HYPRE_Real t1 = nalu_hypre_MPI_Wtime() - t0;
   //printf("Tracker Print Time %.2f\n", t1);

   return nalu_hypre_error_flag;
}

#endif

