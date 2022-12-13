/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

static NALU_HYPRE_Int NearestPowerOfTwo( NALU_HYPRE_Int value )
{
   NALU_HYPRE_Int rc = 1;
   while (rc < value)
   {
      rc <<= 1;
   }
   return rc;
}

static void InitBucket(hypre_HopscotchBucket *b)
{
   b->hopInfo = 0;
   b->hash = NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
}

static void InitBigBucket(hypre_BigHopscotchBucket *b)
{
   b->hopInfo = 0;
   b->hash = NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
}

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
static void InitSegment(hypre_HopscotchSegment *s)
{
   s->timestamp = 0;
   omp_init_lock(&s->lock);
}

static void DestroySegment(hypre_HopscotchSegment *s)
{
   omp_destroy_lock(&s->lock);
}
#endif

void hypre_UnorderedIntSetCreate( hypre_UnorderedIntSet *s,
                                  NALU_HYPRE_Int inCapacity,
                                  NALU_HYPRE_Int concurrencyLevel)
{
   s->segmentMask = NearestPowerOfTwo(concurrencyLevel) - 1;
   if (inCapacity < s->segmentMask + 1)
   {
      inCapacity = s->segmentMask + 1;
   }

   //ADJUST INPUT ............................
   NALU_HYPRE_Int adjInitCap = NearestPowerOfTwo(inCapacity + 4096);
   NALU_HYPRE_Int num_buckets = adjInitCap + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE + 1;
   s->bucketMask = adjInitCap - 1;

   NALU_HYPRE_Int i;

   //ALLOCATE THE SEGMENTS ...................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   s->segments = hypre_TAlloc(hypre_HopscotchSegment,  s->segmentMask + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= s->segmentMask; ++i)
   {
      InitSegment(&s->segments[i]);
   }
#endif

   s->hopInfo = hypre_TAlloc(hypre_uint,  num_buckets, NALU_HYPRE_MEMORY_HOST);
   s->key = hypre_TAlloc(NALU_HYPRE_Int,  num_buckets, NALU_HYPRE_MEMORY_HOST);
   s->hash = hypre_TAlloc(NALU_HYPRE_Int,  num_buckets, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for
#endif
   for (i = 0; i < num_buckets; ++i)
   {
      s->hopInfo[i] = 0;
      s->hash[i] = NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
   }
}

void hypre_UnorderedBigIntSetCreate( hypre_UnorderedBigIntSet *s,
                                     NALU_HYPRE_Int inCapacity,
                                     NALU_HYPRE_Int concurrencyLevel)
{
   s->segmentMask = NearestPowerOfTwo(concurrencyLevel) - 1;
   if (inCapacity < s->segmentMask + 1)
   {
      inCapacity = s->segmentMask + 1;
   }

   //ADJUST INPUT ............................
   NALU_HYPRE_Int adjInitCap = NearestPowerOfTwo(inCapacity + 4096);
   NALU_HYPRE_Int num_buckets = adjInitCap + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE + 1;
   s->bucketMask = adjInitCap - 1;

   NALU_HYPRE_Int i;

   //ALLOCATE THE SEGMENTS ...................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   s->segments = hypre_TAlloc(hypre_HopscotchSegment,  s->segmentMask + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= s->segmentMask; ++i)
   {
      InitSegment(&s->segments[i]);
   }
#endif

   s->hopInfo = hypre_TAlloc(hypre_uint,  num_buckets, NALU_HYPRE_MEMORY_HOST);
   s->key = hypre_TAlloc(NALU_HYPRE_BigInt,  num_buckets, NALU_HYPRE_MEMORY_HOST);
   s->hash = hypre_TAlloc(NALU_HYPRE_BigInt,  num_buckets, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for
#endif
   for (i = 0; i < num_buckets; ++i)
   {
      s->hopInfo[i] = 0;
      s->hash[i] = NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
   }
}

void hypre_UnorderedIntMapCreate( hypre_UnorderedIntMap *m,
                                  NALU_HYPRE_Int inCapacity,
                                  NALU_HYPRE_Int concurrencyLevel)
{
   m->segmentMask = NearestPowerOfTwo(concurrencyLevel) - 1;
   if (inCapacity < m->segmentMask + 1)
   {
      inCapacity = m->segmentMask + 1;
   }

   //ADJUST INPUT ............................
   NALU_HYPRE_Int adjInitCap = NearestPowerOfTwo(inCapacity + 4096);
   NALU_HYPRE_Int num_buckets = adjInitCap + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE + 1;
   m->bucketMask = adjInitCap - 1;

   NALU_HYPRE_Int i;

   //ALLOCATE THE SEGMENTS ...................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   m->segments = hypre_TAlloc(hypre_HopscotchSegment,  m->segmentMask + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= m->segmentMask; i++)
   {
      InitSegment(&m->segments[i]);
   }
#endif

   m->table = hypre_TAlloc(hypre_HopscotchBucket,  num_buckets, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for
#endif
   for (i = 0; i < num_buckets; i++)
   {
      InitBucket(&m->table[i]);
   }
}

void hypre_UnorderedBigIntMapCreate( hypre_UnorderedBigIntMap *m,
                                     NALU_HYPRE_Int inCapacity,
                                     NALU_HYPRE_Int concurrencyLevel)
{
   m->segmentMask = NearestPowerOfTwo(concurrencyLevel) - 1;
   if (inCapacity < m->segmentMask + 1)
   {
      inCapacity = m->segmentMask + 1;
   }

   //ADJUST INPUT ............................
   NALU_HYPRE_Int adjInitCap = NearestPowerOfTwo(inCapacity + 4096);
   NALU_HYPRE_Int num_buckets = adjInitCap + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE + 1;
   m->bucketMask = adjInitCap - 1;

   NALU_HYPRE_Int i;

   //ALLOCATE THE SEGMENTS ...................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   m->segments = hypre_TAlloc(hypre_HopscotchSegment,  m->segmentMask + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= m->segmentMask; i++)
   {
      InitSegment(&m->segments[i]);
   }
#endif

   m->table = hypre_TAlloc(hypre_BigHopscotchBucket,  num_buckets, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for
#endif
   for (i = 0; i < num_buckets; i++)
   {
      InitBigBucket(&m->table[i]);
   }
}

void hypre_UnorderedIntSetDestroy( hypre_UnorderedIntSet *s )
{
   hypre_TFree(s->hopInfo, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(s->key, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(s->hash, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int i;
   for (i = 0; i <= s->segmentMask; i++)
   {
      DestroySegment(&s->segments[i]);
   }
   hypre_TFree(s->segments, NALU_HYPRE_MEMORY_HOST);
#endif
}

void hypre_UnorderedBigIntSetDestroy( hypre_UnorderedBigIntSet *s )
{
   hypre_TFree(s->hopInfo, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(s->key, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(s->hash, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int i;
   for (i = 0; i <= s->segmentMask; i++)
   {
      DestroySegment(&s->segments[i]);
   }
   hypre_TFree(s->segments, NALU_HYPRE_MEMORY_HOST);
#endif
}

void hypre_UnorderedIntMapDestroy( hypre_UnorderedIntMap *m)
{
   hypre_TFree(m->table, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int i;
   for (i = 0; i <= m->segmentMask; i++)
   {
      DestroySegment(&m->segments[i]);
   }
   hypre_TFree(m->segments, NALU_HYPRE_MEMORY_HOST);
#endif
}

void hypre_UnorderedBigIntMapDestroy( hypre_UnorderedBigIntMap *m)
{
   hypre_TFree(m->table, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int i;
   for (i = 0; i <= m->segmentMask; i++)
   {
      DestroySegment(&m->segments[i]);
   }
   hypre_TFree(m->segments, NALU_HYPRE_MEMORY_HOST);
#endif
}

NALU_HYPRE_Int *hypre_UnorderedIntSetCopyToArray( hypre_UnorderedIntSet *s, NALU_HYPRE_Int *len )
{
   /*NALU_HYPRE_Int prefix_sum_workspace[hypre_NumThreads() + 1];*/
   NALU_HYPRE_Int *prefix_sum_workspace;
   NALU_HYPRE_Int *ret_array = NULL;

   prefix_sum_workspace = hypre_TAlloc(NALU_HYPRE_Int,  hypre_NumThreads() + 1, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int n = s->bucketMask + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
      NALU_HYPRE_Int i_begin, i_end;
      hypre_GetSimpleThreadPartition(&i_begin, &i_end, n);

      NALU_HYPRE_Int cnt = 0;
      NALU_HYPRE_Int i;
      for (i = i_begin; i < i_end; i++)
      {
         if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) { cnt++; }
      }

      hypre_prefix_sum(&cnt, len, prefix_sum_workspace);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
      #pragma omp barrier
      #pragma omp master
#endif
      {
         ret_array = hypre_TAlloc(NALU_HYPRE_Int,  *len, NALU_HYPRE_MEMORY_HOST);
      }
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
      #pragma omp barrier
#endif

      for (i = i_begin; i < i_end; i++)
      {
         if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) { ret_array[cnt++] = s->key[i]; }
      }
   }

   hypre_TFree(prefix_sum_workspace, NALU_HYPRE_MEMORY_HOST);

   return ret_array;
}

NALU_HYPRE_BigInt *hypre_UnorderedBigIntSetCopyToArray( hypre_UnorderedBigIntSet *s, NALU_HYPRE_Int *len )
{
   /*NALU_HYPRE_Int prefix_sum_workspace[hypre_NumThreads() + 1];*/
   NALU_HYPRE_Int *prefix_sum_workspace;
   NALU_HYPRE_BigInt *ret_array = NULL;

   prefix_sum_workspace = hypre_TAlloc(NALU_HYPRE_Int,  hypre_NumThreads() + 1, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int n = s->bucketMask + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
      NALU_HYPRE_Int i_begin, i_end;
      hypre_GetSimpleThreadPartition(&i_begin, &i_end, n);

      NALU_HYPRE_Int cnt = 0;
      NALU_HYPRE_Int i;
      for (i = i_begin; i < i_end; i++)
      {
         if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) { cnt++; }
      }

      hypre_prefix_sum(&cnt, len, prefix_sum_workspace);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
      #pragma omp barrier
      #pragma omp master
#endif
      {
         ret_array = hypre_TAlloc(NALU_HYPRE_BigInt,  *len, NALU_HYPRE_MEMORY_HOST);
      }
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
      #pragma omp barrier
#endif

      for (i = i_begin; i < i_end; i++)
      {
         if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) { ret_array[cnt++] = s->key[i]; }
      }
   }

   hypre_TFree(prefix_sum_workspace, NALU_HYPRE_MEMORY_HOST);

   return ret_array;
}
