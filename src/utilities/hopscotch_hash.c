/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

static NALU_HYPRE_Int NearestPowerOfTwo( NALU_HYPRE_Int value )
{
   NALU_HYPRE_Int rc = 1;
   while (rc < value)
   {
      rc <<= 1;
   }
   return rc;
}

static void InitBucket(nalu_hypre_HopscotchBucket *b)
{
   b->hopInfo = 0;
   b->hash = NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
}

static void InitBigBucket(nalu_hypre_BigHopscotchBucket *b)
{
   b->hopInfo = 0;
   b->hash = NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
}

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
static void InitSegment(nalu_hypre_HopscotchSegment *s)
{
   s->timestamp = 0;
   omp_init_lock(&s->lock);
}

static void DestroySegment(nalu_hypre_HopscotchSegment *s)
{
   omp_destroy_lock(&s->lock);
}
#endif

void nalu_hypre_UnorderedIntSetCreate( nalu_hypre_UnorderedIntSet *s,
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
   s->segments = nalu_hypre_TAlloc(nalu_hypre_HopscotchSegment,  s->segmentMask + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= s->segmentMask; ++i)
   {
      InitSegment(&s->segments[i]);
   }
#endif

   s->hopInfo = nalu_hypre_TAlloc(nalu_hypre_uint,  num_buckets, NALU_HYPRE_MEMORY_HOST);
   s->key = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_buckets, NALU_HYPRE_MEMORY_HOST);
   s->hash = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_buckets, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for
#endif
   for (i = 0; i < num_buckets; ++i)
   {
      s->hopInfo[i] = 0;
      s->hash[i] = NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
   }
}

void nalu_hypre_UnorderedBigIntSetCreate( nalu_hypre_UnorderedBigIntSet *s,
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
   s->segments = nalu_hypre_TAlloc(nalu_hypre_HopscotchSegment,  s->segmentMask + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= s->segmentMask; ++i)
   {
      InitSegment(&s->segments[i]);
   }
#endif

   s->hopInfo = nalu_hypre_TAlloc(nalu_hypre_uint,  num_buckets, NALU_HYPRE_MEMORY_HOST);
   s->key = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  num_buckets, NALU_HYPRE_MEMORY_HOST);
   s->hash = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  num_buckets, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for
#endif
   for (i = 0; i < num_buckets; ++i)
   {
      s->hopInfo[i] = 0;
      s->hash[i] = NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
   }
}

void nalu_hypre_UnorderedIntMapCreate( nalu_hypre_UnorderedIntMap *m,
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
   m->segments = nalu_hypre_TAlloc(nalu_hypre_HopscotchSegment,  m->segmentMask + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= m->segmentMask; i++)
   {
      InitSegment(&m->segments[i]);
   }
#endif

   m->table = nalu_hypre_TAlloc(nalu_hypre_HopscotchBucket,  num_buckets, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for
#endif
   for (i = 0; i < num_buckets; i++)
   {
      InitBucket(&m->table[i]);
   }
}

void nalu_hypre_UnorderedBigIntMapCreate( nalu_hypre_UnorderedBigIntMap *m,
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
   m->segments = nalu_hypre_TAlloc(nalu_hypre_HopscotchSegment,  m->segmentMask + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= m->segmentMask; i++)
   {
      InitSegment(&m->segments[i]);
   }
#endif

   m->table = nalu_hypre_TAlloc(nalu_hypre_BigHopscotchBucket,  num_buckets, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for
#endif
   for (i = 0; i < num_buckets; i++)
   {
      InitBigBucket(&m->table[i]);
   }
}

void nalu_hypre_UnorderedIntSetDestroy( nalu_hypre_UnorderedIntSet *s )
{
   nalu_hypre_TFree(s->hopInfo, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(s->key, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(s->hash, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int i;
   for (i = 0; i <= s->segmentMask; i++)
   {
      DestroySegment(&s->segments[i]);
   }
   nalu_hypre_TFree(s->segments, NALU_HYPRE_MEMORY_HOST);
#endif
}

void nalu_hypre_UnorderedBigIntSetDestroy( nalu_hypre_UnorderedBigIntSet *s )
{
   nalu_hypre_TFree(s->hopInfo, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(s->key, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(s->hash, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int i;
   for (i = 0; i <= s->segmentMask; i++)
   {
      DestroySegment(&s->segments[i]);
   }
   nalu_hypre_TFree(s->segments, NALU_HYPRE_MEMORY_HOST);
#endif
}

void nalu_hypre_UnorderedIntMapDestroy( nalu_hypre_UnorderedIntMap *m)
{
   nalu_hypre_TFree(m->table, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int i;
   for (i = 0; i <= m->segmentMask; i++)
   {
      DestroySegment(&m->segments[i]);
   }
   nalu_hypre_TFree(m->segments, NALU_HYPRE_MEMORY_HOST);
#endif
}

void nalu_hypre_UnorderedBigIntMapDestroy( nalu_hypre_UnorderedBigIntMap *m)
{
   nalu_hypre_TFree(m->table, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int i;
   for (i = 0; i <= m->segmentMask; i++)
   {
      DestroySegment(&m->segments[i]);
   }
   nalu_hypre_TFree(m->segments, NALU_HYPRE_MEMORY_HOST);
#endif
}

NALU_HYPRE_Int *nalu_hypre_UnorderedIntSetCopyToArray( nalu_hypre_UnorderedIntSet *s, NALU_HYPRE_Int *len )
{
   /*NALU_HYPRE_Int prefix_sum_workspace[nalu_hypre_NumThreads() + 1];*/
   NALU_HYPRE_Int *prefix_sum_workspace;
   NALU_HYPRE_Int *ret_array = NULL;

   prefix_sum_workspace = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nalu_hypre_NumThreads() + 1, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int n = s->bucketMask + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
      NALU_HYPRE_Int i_begin, i_end;
      nalu_hypre_GetSimpleThreadPartition(&i_begin, &i_end, n);

      NALU_HYPRE_Int cnt = 0;
      NALU_HYPRE_Int i;
      for (i = i_begin; i < i_end; i++)
      {
         if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) { cnt++; }
      }

      nalu_hypre_prefix_sum(&cnt, len, prefix_sum_workspace);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
      #pragma omp barrier
      #pragma omp master
#endif
      {
         ret_array = nalu_hypre_TAlloc(NALU_HYPRE_Int,  *len, NALU_HYPRE_MEMORY_HOST);
      }
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
      #pragma omp barrier
#endif

      for (i = i_begin; i < i_end; i++)
      {
         if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) { ret_array[cnt++] = s->key[i]; }
      }
   }

   nalu_hypre_TFree(prefix_sum_workspace, NALU_HYPRE_MEMORY_HOST);

   return ret_array;
}

NALU_HYPRE_BigInt *nalu_hypre_UnorderedBigIntSetCopyToArray( nalu_hypre_UnorderedBigIntSet *s, NALU_HYPRE_Int *len )
{
   /*NALU_HYPRE_Int prefix_sum_workspace[nalu_hypre_NumThreads() + 1];*/
   NALU_HYPRE_Int *prefix_sum_workspace;
   NALU_HYPRE_BigInt *ret_array = NULL;

   prefix_sum_workspace = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nalu_hypre_NumThreads() + 1, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int n = s->bucketMask + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
      NALU_HYPRE_Int i_begin, i_end;
      nalu_hypre_GetSimpleThreadPartition(&i_begin, &i_end, n);

      NALU_HYPRE_Int cnt = 0;
      NALU_HYPRE_Int i;
      for (i = i_begin; i < i_end; i++)
      {
         if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) { cnt++; }
      }

      nalu_hypre_prefix_sum(&cnt, len, prefix_sum_workspace);

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
      #pragma omp barrier
      #pragma omp master
#endif
      {
         ret_array = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  *len, NALU_HYPRE_MEMORY_HOST);
      }
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
      #pragma omp barrier
#endif

      for (i = i_begin; i < i_end; i++)
      {
         if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i]) { ret_array[cnt++] = s->key[i]; }
      }
   }

   nalu_hypre_TFree(prefix_sum_workspace, NALU_HYPRE_MEMORY_HOST);

   return ret_array;
}
