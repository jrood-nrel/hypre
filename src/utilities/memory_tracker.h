/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_MEMORY_TRACKER_HEADER
#define nalu_hypre_MEMORY_TRACKER_HEADER

#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)

extern size_t nalu_hypre_total_bytes[nalu_hypre_MEMORY_UNIFIED + 1];
extern size_t nalu_hypre_peak_bytes[nalu_hypre_MEMORY_UNIFIED + 1];
extern size_t nalu_hypre_current_bytes[nalu_hypre_MEMORY_UNIFIED + 1];
extern NALU_HYPRE_Int nalu_hypre_memory_tracker_print;
extern char nalu_hypre_memory_tracker_filename[NALU_HYPRE_MAX_FILE_NAME_LEN];

typedef enum _nalu_hypre_MemoryTrackerEvent
{
   NALU_HYPRE_MEMORY_EVENT_ALLOC = 0,
   NALU_HYPRE_MEMORY_EVENT_FREE,
   NALU_HYPRE_MEMORY_EVENT_COPY,
   NALU_HYPRE_MEMORY_NUM_EVENTS,
} nalu_hypre_MemoryTrackerEvent;

typedef enum _nalu_hypre_MemcpyType
{
   nalu_hypre_MEMCPY_H2H = 0,
   nalu_hypre_MEMCPY_D2H,
   nalu_hypre_MEMCPY_H2D,
   nalu_hypre_MEMCPY_D2D,
   nalu_hypre_MEMCPY_NUM_TYPES,
} nalu_hypre_MemcpyType;

typedef struct
{
   size_t                index;
   size_t                time_step;
   char                  action[16];
   void                 *ptr;
   void                 *ptr2;
   size_t                nbytes;
   nalu_hypre_MemoryLocation  memory_location;
   nalu_hypre_MemoryLocation  memory_location2;
   char                  filename[NALU_HYPRE_MAX_FILE_NAME_LEN];
   char                  function[256];
   NALU_HYPRE_Int             line;
   size_t                pair;
} nalu_hypre_MemoryTrackerEntry;

typedef struct
{
   size_t                     head;
   size_t                     actual_size;
   size_t                     alloced_size;
   nalu_hypre_MemoryTrackerEntry  *data;
   /* Free Queue is sorted based on (ptr, time_step) ascendingly */
   nalu_hypre_MemoryTrackerEntry  *sorted_data;
   /* compressed sorted_data with the same ptr */
   size_t                     sorted_data_compressed_len;
   size_t                    *sorted_data_compressed_offset;
   nalu_hypre_MemoryTrackerEntry **sorted_data_compressed;
} nalu_hypre_MemoryTrackerQueue;

typedef struct
{
   size_t                   curr_time_step;
   nalu_hypre_MemoryTrackerQueue queue[NALU_HYPRE_MEMORY_NUM_EVENTS];
} nalu_hypre_MemoryTracker;

#define nalu_hypre_TAlloc(type, count, location)                                                         \
(                                                                                                   \
{                                                                                                   \
   void *ptr = nalu_hypre_MAlloc((size_t)(sizeof(type) * (count)), location);                            \
                                                                                                    \
   nalu_hypre_MemoryLocation alocation = nalu_hypre_GetActualMemLocation(location);                           \
   nalu_hypre_MemoryTrackerInsert1("malloc", ptr, sizeof(type)*(count), alocation,                       \
                              __FILE__, __func__, __LINE__);                                        \
   (type *) ptr;                                                                                    \
}                                                                                                   \
)

#define nalu_hypre_CTAlloc(type, count, location)                                                        \
(                                                                                                   \
{                                                                                                   \
   void *ptr = nalu_hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location);                       \
                                                                                                    \
   nalu_hypre_MemoryLocation alocation = nalu_hypre_GetActualMemLocation(location);                           \
   nalu_hypre_MemoryTrackerInsert1("calloc", ptr, sizeof(type)*(count), alocation,                       \
                              __FILE__, __func__, __LINE__);                                        \
   (type *) ptr;                                                                                    \
}                                                                                                   \
)

#define nalu_hypre_TReAlloc(ptr, type, count, location)                                                  \
(                                                                                                   \
{                                                                                                   \
   void *new_ptr = nalu_hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location);          \
                                                                                                    \
   nalu_hypre_MemoryLocation alocation = nalu_hypre_GetActualMemLocation(location);                           \
   nalu_hypre_MemoryTrackerInsert1("rfree", ptr, (size_t) -1, alocation,                                 \
                              __FILE__, __func__, __LINE__);                                        \
   nalu_hypre_MemoryTrackerInsert1("rmalloc", new_ptr, sizeof(type)*(count), alocation,                  \
                              __FILE__, __func__, __LINE__);                                        \
   (type *) new_ptr;                                                                                \
}                                                                                                   \
)

#define nalu_hypre_TReAlloc_v2(ptr, old_type, old_count, new_type, new_count, location)                  \
(                                                                                                   \
{                                                                                                   \
   void *new_ptr = nalu_hypre_ReAlloc_v2((char *)ptr, (size_t)(sizeof(old_type)*(old_count)),            \
                                    (size_t)(sizeof(new_type)*(new_count)), location);              \
                                                                                                    \
   nalu_hypre_MemoryLocation alocation = nalu_hypre_GetActualMemLocation(location);                           \
   nalu_hypre_MemoryTrackerInsert1("rfree", ptr, sizeof(old_type)*(old_count), alocation,                \
                              __FILE__, __func__, __LINE__);                                        \
   nalu_hypre_MemoryTrackerInsert1("rmalloc", new_ptr, sizeof(new_type)*(new_count), alocation,          \
                              __FILE__, __func__, __LINE__);                                        \
   (new_type *) new_ptr;                                                                            \
}                                                                                                   \
)

#define nalu_hypre_TMemcpy(dst, src, type, count, locdst, locsrc)                                        \
(                                                                                                   \
{                                                                                                   \
   nalu_hypre_Memcpy((void *)(dst), (void *)(src), (size_t)(sizeof(type) * (count)), locdst, locsrc);    \
                                                                                                    \
   nalu_hypre_MemoryLocation alocation_dst = nalu_hypre_GetActualMemLocation(locdst);                         \
   nalu_hypre_MemoryLocation alocation_src = nalu_hypre_GetActualMemLocation(locsrc);                         \
   nalu_hypre_MemoryTrackerInsert2("memcpy", (void *) (dst), (void *) (src), sizeof(type)*(count),       \
                              alocation_dst, alocation_src,                                         \
                              __FILE__, __func__, __LINE__);                                        \
}                                                                                                   \
)

#define nalu_hypre_TFree(ptr, location)                                                                  \
(                                                                                                   \
{                                                                                                   \
   nalu_hypre_Free((void *)ptr, location);                                                               \
                                                                                                    \
   nalu_hypre_MemoryLocation alocation = nalu_hypre_GetActualMemLocation(location);                           \
   nalu_hypre_MemoryTrackerInsert1("free", ptr, (size_t) -1, alocation,                                  \
                              __FILE__, __func__, __LINE__);                                        \
   ptr = NULL;                                                                                      \
}                                                                                                   \
)

#define _nalu_hypre_TAlloc(type, count, location)                                                        \
(                                                                                                   \
{                                                                                                   \
   void *ptr = _nalu_hypre_MAlloc((size_t)(sizeof(type) * (count)), location);                           \
                                                                                                    \
   nalu_hypre_MemoryTrackerInsert1("malloc", ptr, sizeof(type)*(count), location,                        \
                              __FILE__, __func__, __LINE__);                                        \
   (type *) ptr;                                                                                    \
}                                                                                                   \
)

#define _nalu_hypre_TFree(ptr, location)                                                                 \
(                                                                                                   \
{                                                                                                   \
   _nalu_hypre_Free((void *)ptr, location);                                                              \
                                                                                                    \
   nalu_hypre_MemoryTrackerInsert1("free", ptr, (size_t) -1, location,                                   \
                             __FILE__, __func__, __LINE__);                                         \
   ptr = NULL;                                                                                      \
}                                                                                                   \
)

#endif /* #if defined(NALU_HYPRE_USING_MEMORY_TRACKER) */
#endif /* #ifndef nalu_hypre_MEMORY_TRACKER_HEADER */

