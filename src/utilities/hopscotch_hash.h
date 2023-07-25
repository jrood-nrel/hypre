/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/**
 * Hopscotch hash is modified from the code downloaded from
 * https://sites.google.com/site/cconcurrencypackage/hopscotch-hashing
 * with the following terms of usage
 */

////////////////////////////////////////////////////////////////////////////////
//TERMS OF USAGE
//------------------------------------------------------------------------------
//
//  Permission to use, copy, modify and distribute this software and
//  its documentation for any purpose is hereby granted without fee,
//  provided that due acknowledgments to the authors are provided and
//  this permission notice appears in all copies of the software.
//  The software is provided "as is". There is no warranty of any kind.
//
//Authors:
//  Maurice Herlihy
//  Brown University
//  and
//  Nir Shavit
//  Tel-Aviv University
//  and
//  Moran Tzafrir
//  Tel-Aviv University
//
//  Date: July 15, 2008.
//
////////////////////////////////////////////////////////////////////////////////
// Programmer : Moran Tzafrir (MoranTza@gmail.com)
// Modified   : Jongsoo Park  (jongsoo.park@intel.com)
//              Oct 1, 2015.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef nalu_hypre_HOPSCOTCH_HASH_HEADER
#define nalu_hypre_HOPSCOTCH_HASH_HEADER

//#include <strings.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
//#include <math.h>

#ifdef NALU_HYPRE_USING_OPENMP
#include <omp.h>
#endif

//#include "_nalu_hypre_utilities.h"

// Potentially architecture specific features used here:
// __sync_val_compare_and_swap

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * This next section of code is here instead of in _nalu_hypre_utilities.h to get
 * around some portability issues with Visual Studio.  By putting it here, we
 * can explicitly include this '.h' file in a few files in hypre and compile
 * them with C++ instead of C (VS does not support C99 'inline').
 ******************************************************************************/

#ifdef NALU_HYPRE_USING_ATOMIC
static inline NALU_HYPRE_Int
nalu_hypre_compare_and_swap( NALU_HYPRE_Int *ptr, NALU_HYPRE_Int oldval, NALU_HYPRE_Int newval )
{
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
   return __sync_val_compare_and_swap(ptr, oldval, newval);
   //#elif defind _MSC_VER
   //return _InterlockedCompareExchange((long *)ptr, newval, oldval);
   //#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
   // JSP: not many compilers have implemented this, so comment out for now
   //_Atomic NALU_HYPRE_Int *atomic_ptr = ptr;
   //atomic_compare_exchange_strong(atomic_ptr, &oldval, newval);
   //return oldval;
#endif
}

static inline NALU_HYPRE_Int
nalu_hypre_fetch_and_add( NALU_HYPRE_Int *ptr, NALU_HYPRE_Int value )
{
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
   return __sync_fetch_and_add(ptr, value);
   //#elif defined _MSC_VER
   //return _InterlockedExchangeAdd((long *)ptr, value);
   //#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
   // JSP: not many compilers have implemented this, so comment out for now
   //_Atomic NALU_HYPRE_Int *atomic_ptr = ptr;
   //return atomic_fetch_add(atomic_ptr, value);
#endif
}
#else // !NALU_HYPRE_USING_ATOMIC
static inline NALU_HYPRE_Int
nalu_hypre_compare_and_swap( NALU_HYPRE_Int *ptr, NALU_HYPRE_Int oldval, NALU_HYPRE_Int newval )
{
   if (*ptr == oldval)
   {
      *ptr = newval;
      return oldval;
   }
   else { return *ptr; }
}

static inline NALU_HYPRE_Int
nalu_hypre_fetch_and_add( NALU_HYPRE_Int *ptr, NALU_HYPRE_Int value )
{
   NALU_HYPRE_Int oldval = *ptr;
   *ptr += value;
   return oldval;
}
#endif // !NALU_HYPRE_USING_ATOMIC

/******************************************************************************/

// Constants ................................................................
#define NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE    (32)
#define NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE (4*1024)

#define NALU_HYPRE_HOPSCOTCH_HASH_EMPTY (0)
#define NALU_HYPRE_HOPSCOTCH_HASH_BUSY  (1)

// Small Utilities ..........................................................
static inline NALU_HYPRE_Int
first_lsb_bit_indx( nalu_hypre_uint x )
{
   NALU_HYPRE_Int pos;
#if defined(_MSC_VER) || defined(__MINGW64__)
   if (x == 0)
   {
      pos = 0;
   }
   else
   {
      for (pos = 1; !(x & 1); ++pos)
      {
         x >>= 1;
      }
   }
#else
   pos = ffs(x);
#endif
   return (pos - 1);
}
/**
 * nalu_hypre_Hash is adapted from xxHash with the following license.
 */
/*
   xxHash - Extremely Fast Hash algorithm
   Header File
   Copyright (C) 2012-2015, Yann Collet.

   BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   You can contact the author at :
   - xxHash source repository : https://github.com/Cyan4973/xxHash
*/

/***************************************
*  Constants
***************************************/
#define NALU_HYPRE_XXH_PRIME32_1   2654435761U
#define NALU_HYPRE_XXH_PRIME32_2   2246822519U
#define NALU_HYPRE_XXH_PRIME32_3   3266489917U
#define NALU_HYPRE_XXH_PRIME32_4    668265263U
#define NALU_HYPRE_XXH_PRIME32_5    374761393U

#define NALU_HYPRE_XXH_PRIME64_1 11400714785074694791ULL
#define NALU_HYPRE_XXH_PRIME64_2 14029467366897019727ULL
#define NALU_HYPRE_XXH_PRIME64_3  1609587929392839161ULL
#define NALU_HYPRE_XXH_PRIME64_4  9650029242287828579ULL
#define NALU_HYPRE_XXH_PRIME64_5  2870177450012600261ULL

#define NALU_HYPRE_XXH_rotl32(x,r) ((x << r) | (x >> (32 - r)))
#define NALU_HYPRE_XXH_rotl64(x,r) ((x << r) | (x >> (64 - r)))

#if defined(NALU_HYPRE_MIXEDINT) || defined(NALU_HYPRE_BIGINT)
static inline NALU_HYPRE_BigInt
nalu_hypre_BigHash( NALU_HYPRE_BigInt input )
{
   nalu_hypre_ulonglongint h64 = NALU_HYPRE_XXH_PRIME64_5 + sizeof(input);

   nalu_hypre_ulonglongint k1 = input;
   k1 *= NALU_HYPRE_XXH_PRIME64_2;
   k1 = NALU_HYPRE_XXH_rotl64(k1, 31);
   k1 *= NALU_HYPRE_XXH_PRIME64_1;
   h64 ^= k1;
   h64 = NALU_HYPRE_XXH_rotl64(h64, 27) * NALU_HYPRE_XXH_PRIME64_1 + NALU_HYPRE_XXH_PRIME64_4;

   h64 ^= h64 >> 33;
   h64 *= NALU_HYPRE_XXH_PRIME64_2;
   h64 ^= h64 >> 29;
   h64 *= NALU_HYPRE_XXH_PRIME64_3;
   h64 ^= h64 >> 32;

#ifndef NDEBUG
   if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY == h64)
   {
      nalu_hypre_printf("hash(%lld) = %d\n", h64, NALU_HYPRE_HOPSCOTCH_HASH_EMPTY);
      nalu_hypre_assert(NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != h64);
   }
#endif

   return h64;
}

#else
static inline NALU_HYPRE_Int
nalu_hypre_BigHash(NALU_HYPRE_Int input)
{
   nalu_hypre_uint h32 = NALU_HYPRE_XXH_PRIME32_5 + sizeof(input);

   // 1665863975 is added to input so that
   // only -1073741824 gives NALU_HYPRE_HOPSCOTCH_HASH_EMPTY.
   // Hence, we're fine as long as key is non-negative.
   h32 += (input + 1665863975) * NALU_HYPRE_XXH_PRIME32_3;
   h32 = NALU_HYPRE_XXH_rotl32(h32, 17) * NALU_HYPRE_XXH_PRIME32_4;

   h32 ^= h32 >> 15;
   h32 *= NALU_HYPRE_XXH_PRIME32_2;
   h32 ^= h32 >> 13;
   h32 *= NALU_HYPRE_XXH_PRIME32_3;
   h32 ^= h32 >> 16;

   //nalu_hypre_assert(NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != h32);

   return h32;
}
#endif

#ifdef NALU_HYPRE_BIGINT
static inline NALU_HYPRE_Int
nalu_hypre_Hash(NALU_HYPRE_Int input)
{
   nalu_hypre_ulonglongint h64 = NALU_HYPRE_XXH_PRIME64_5 + sizeof(input);

   nalu_hypre_ulonglongint k1 = input;
   k1 *= NALU_HYPRE_XXH_PRIME64_2;
   k1 = NALU_HYPRE_XXH_rotl64(k1, 31);
   k1 *= NALU_HYPRE_XXH_PRIME64_1;
   h64 ^= k1;
   h64 = NALU_HYPRE_XXH_rotl64(h64, 27) * NALU_HYPRE_XXH_PRIME64_1 + NALU_HYPRE_XXH_PRIME64_4;

   h64 ^= h64 >> 33;
   h64 *= NALU_HYPRE_XXH_PRIME64_2;
   h64 ^= h64 >> 29;
   h64 *= NALU_HYPRE_XXH_PRIME64_3;
   h64 ^= h64 >> 32;

#ifndef NDEBUG
   if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY == h64)
   {
      nalu_hypre_printf("hash(%lld) = %d\n", h64, NALU_HYPRE_HOPSCOTCH_HASH_EMPTY);
      nalu_hypre_assert(NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != h64);
   }
#endif

   return h64;
}

#else
static inline NALU_HYPRE_Int
nalu_hypre_Hash(NALU_HYPRE_Int input)
{
   nalu_hypre_uint h32 = NALU_HYPRE_XXH_PRIME32_5 + sizeof(input);

   // 1665863975 is added to input so that
   // only -1073741824 gives NALU_HYPRE_HOPSCOTCH_HASH_EMPTY.
   // Hence, we're fine as long as key is non-negative.
   h32 += (input + 1665863975) * NALU_HYPRE_XXH_PRIME32_3;
   h32 = NALU_HYPRE_XXH_rotl32(h32, 17) * NALU_HYPRE_XXH_PRIME32_4;

   h32 ^= h32 >> 15;
   h32 *= NALU_HYPRE_XXH_PRIME32_2;
   h32 ^= h32 >> 13;
   h32 *= NALU_HYPRE_XXH_PRIME32_3;
   h32 ^= h32 >> 16;

   //nalu_hypre_assert(NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != h32);

   return h32;
}
#endif

static inline void
nalu_hypre_UnorderedIntSetFindCloserFreeBucket( nalu_hypre_UnorderedIntSet *s,
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
                                           nalu_hypre_HopscotchSegment *start_seg,
#endif
                                           NALU_HYPRE_Int *free_bucket,
                                           NALU_HYPRE_Int *free_dist )
{
   NALU_HYPRE_Int move_bucket = *free_bucket - (NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
   NALU_HYPRE_Int move_free_dist;
   for (move_free_dist = NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
   {
      nalu_hypre_uint start_hop_info = s->hopInfo[move_bucket];
      NALU_HYPRE_Int move_new_free_dist = -1;
      nalu_hypre_uint mask = 1;
      NALU_HYPRE_Int i;
      for (i = 0; i < move_free_dist; ++i, mask <<= 1)
      {
         if (mask & start_hop_info)
         {
            move_new_free_dist = i;
            break;
         }
      }
      if (-1 != move_new_free_dist)
      {
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         nalu_hypre_HopscotchSegment*  move_segment = &(s->segments[move_bucket & s->segmentMask]);

         if (start_seg != move_segment)
         {
            omp_set_lock(&move_segment->lock);
         }
#endif

         if (start_hop_info == s->hopInfo[move_bucket])
         {
            // new_free_bucket -> free_bucket and empty new_free_bucket
            NALU_HYPRE_Int new_free_bucket = move_bucket + move_new_free_dist;
            s->key[*free_bucket]  = s->key[new_free_bucket];
            s->hash[*free_bucket] = s->hash[new_free_bucket];

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            ++move_segment->timestamp;
            #pragma omp flush
#endif

            s->hopInfo[move_bucket] |= (1U << move_free_dist);
            s->hopInfo[move_bucket] &= ~(1U << move_new_free_dist);

            *free_bucket = new_free_bucket;
            *free_dist -= move_free_dist - move_new_free_dist;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            if (start_seg != move_segment)
            {
               omp_unset_lock(&move_segment->lock);
            }
#endif

            return;
         }
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         if (start_seg != move_segment)
         {
            omp_unset_lock(&move_segment->lock);
         }
#endif
      }
      ++move_bucket;
   }
   *free_bucket = -1;
   *free_dist = 0;
}

static inline void
nalu_hypre_UnorderedBigIntSetFindCloserFreeBucket( nalu_hypre_UnorderedBigIntSet *s,
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
                                              nalu_hypre_HopscotchSegment   *start_seg,
#endif
                                              NALU_HYPRE_Int *free_bucket,
                                              NALU_HYPRE_Int *free_dist )
{
   NALU_HYPRE_Int move_bucket = *free_bucket - (NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
   NALU_HYPRE_Int move_free_dist;
   for (move_free_dist = NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
   {
      nalu_hypre_uint start_hop_info = s->hopInfo[move_bucket];
      NALU_HYPRE_Int move_new_free_dist = -1;
      nalu_hypre_uint mask = 1;
      NALU_HYPRE_Int i;
      for (i = 0; i < move_free_dist; ++i, mask <<= 1)
      {
         if (mask & start_hop_info)
         {
            move_new_free_dist = i;
            break;
         }
      }
      if (-1 != move_new_free_dist)
      {
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         nalu_hypre_HopscotchSegment*  move_segment = &(s->segments[move_bucket & s->segmentMask]);

         if (start_seg != move_segment)
         {
            omp_set_lock(&move_segment->lock);
         }
#endif

         if (start_hop_info == s->hopInfo[move_bucket])
         {
            // new_free_bucket -> free_bucket and empty new_free_bucket
            NALU_HYPRE_Int new_free_bucket = move_bucket + move_new_free_dist;
            s->key[*free_bucket]  = s->key[new_free_bucket];
            s->hash[*free_bucket] = s->hash[new_free_bucket];

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            ++move_segment->timestamp;
            #pragma omp flush
#endif

            s->hopInfo[move_bucket] |= (1U << move_free_dist);
            s->hopInfo[move_bucket] &= ~(1U << move_new_free_dist);

            *free_bucket = new_free_bucket;
            *free_dist -= move_free_dist - move_new_free_dist;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            if (start_seg != move_segment)
            {
               omp_unset_lock(&move_segment->lock);
            }
#endif

            return;
         }
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         if (start_seg != move_segment)
         {
            omp_unset_lock(&move_segment->lock);
         }
#endif
      }
      ++move_bucket;
   }
   *free_bucket = -1;
   *free_dist = 0;
}

static inline void
nalu_hypre_UnorderedIntMapFindCloserFreeBucket( nalu_hypre_UnorderedIntMap  *m,
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
                                           nalu_hypre_HopscotchSegment *start_seg,
#endif
                                           nalu_hypre_HopscotchBucket **free_bucket,
                                           NALU_HYPRE_Int *free_dist)
{
   nalu_hypre_HopscotchBucket* move_bucket = *free_bucket - (NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
   NALU_HYPRE_Int move_free_dist;
   for (move_free_dist = NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
   {
      nalu_hypre_uint start_hop_info = move_bucket->hopInfo;
      NALU_HYPRE_Int move_new_free_dist = -1;
      nalu_hypre_uint mask = 1;
      NALU_HYPRE_Int i;
      for (i = 0; i < move_free_dist; ++i, mask <<= 1)
      {
         if (mask & start_hop_info)
         {
            move_new_free_dist = i;
            break;
         }
      }
      if (-1 != move_new_free_dist)
      {
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         nalu_hypre_HopscotchSegment* move_segment = &(m->segments[(move_bucket - m->table) & m->segmentMask]);

         if (start_seg != move_segment)
         {
            omp_set_lock(&move_segment->lock);
         }
#endif

         if (start_hop_info == move_bucket->hopInfo)
         {
            // new_free_bucket -> free_bucket and empty new_free_bucket
            nalu_hypre_HopscotchBucket* new_free_bucket = move_bucket + move_new_free_dist;
            (*free_bucket)->data = new_free_bucket->data;
            (*free_bucket)->key  = new_free_bucket->key;
            (*free_bucket)->hash = new_free_bucket->hash;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            ++move_segment->timestamp;

            #pragma omp flush
#endif

            move_bucket->hopInfo |= (1U << move_free_dist);
            move_bucket->hopInfo &= ~(1U << move_new_free_dist);

            *free_bucket = new_free_bucket;
            *free_dist -= move_free_dist - move_new_free_dist;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            if (start_seg != move_segment)
            {
               omp_unset_lock(&move_segment->lock);
            }
#endif
            return;
         }
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         if (start_seg != move_segment)
         {
            omp_unset_lock(&move_segment->lock);
         }
#endif
      }
      ++move_bucket;
   }
   *free_bucket = NULL;
   *free_dist = 0;
}

static inline void
nalu_hypre_UnorderedBigIntMapFindCloserFreeBucket( nalu_hypre_UnorderedBigIntMap   *m,
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
                                              nalu_hypre_HopscotchSegment     *start_seg,
#endif
                                              nalu_hypre_BigHopscotchBucket **free_bucket,
                                              NALU_HYPRE_Int *free_dist)
{
   nalu_hypre_BigHopscotchBucket* move_bucket = *free_bucket - (NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1);
   NALU_HYPRE_Int move_free_dist;
   for (move_free_dist = NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
   {
      nalu_hypre_uint start_hop_info = move_bucket->hopInfo;
      NALU_HYPRE_Int move_new_free_dist = -1;
      nalu_hypre_uint mask = 1;
      NALU_HYPRE_Int i;
      for (i = 0; i < move_free_dist; ++i, mask <<= 1)
      {
         if (mask & start_hop_info)
         {
            move_new_free_dist = i;
            break;
         }
      }
      if (-1 != move_new_free_dist)
      {
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         nalu_hypre_HopscotchSegment* move_segment = &(m->segments[(move_bucket - m->table) & m->segmentMask]);

         if (start_seg != move_segment)
         {
            omp_set_lock(&move_segment->lock);
         }
#endif

         if (start_hop_info == move_bucket->hopInfo)
         {
            // new_free_bucket -> free_bucket and empty new_free_bucket
            nalu_hypre_BigHopscotchBucket* new_free_bucket = move_bucket + move_new_free_dist;
            (*free_bucket)->data = new_free_bucket->data;
            (*free_bucket)->key  = new_free_bucket->key;
            (*free_bucket)->hash = new_free_bucket->hash;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            ++move_segment->timestamp;

            #pragma omp flush
#endif

            move_bucket->hopInfo |= (1U << move_free_dist);
            move_bucket->hopInfo &= ~(1U << move_new_free_dist);

            *free_bucket = new_free_bucket;
            *free_dist -= move_free_dist - move_new_free_dist;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            if (start_seg != move_segment)
            {
               omp_unset_lock(&move_segment->lock);
            }
#endif
            return;
         }
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         if (start_seg != move_segment)
         {
            omp_unset_lock(&move_segment->lock);
         }
#endif
      }
      ++move_bucket;
   }
   *free_bucket = NULL;
   *free_dist = 0;
}

void nalu_hypre_UnorderedIntSetCreate( nalu_hypre_UnorderedIntSet *s,
                                  NALU_HYPRE_Int inCapacity,
                                  NALU_HYPRE_Int concurrencyLevel);
void nalu_hypre_UnorderedBigIntSetCreate( nalu_hypre_UnorderedBigIntSet *s,
                                     NALU_HYPRE_Int inCapacity,
                                     NALU_HYPRE_Int concurrencyLevel);
void nalu_hypre_UnorderedIntMapCreate( nalu_hypre_UnorderedIntMap *m,
                                  NALU_HYPRE_Int inCapacity,
                                  NALU_HYPRE_Int concurrencyLevel);
void nalu_hypre_UnorderedBigIntMapCreate( nalu_hypre_UnorderedBigIntMap *m,
                                     NALU_HYPRE_Int inCapacity,
                                     NALU_HYPRE_Int concurrencyLevel);

void nalu_hypre_UnorderedIntSetDestroy( nalu_hypre_UnorderedIntSet *s );
void nalu_hypre_UnorderedBigIntSetDestroy( nalu_hypre_UnorderedBigIntSet *s );
void nalu_hypre_UnorderedIntMapDestroy( nalu_hypre_UnorderedIntMap *m );
void nalu_hypre_UnorderedBigIntMapDestroy( nalu_hypre_UnorderedBigIntMap *m );

// Query Operations .........................................................
static inline NALU_HYPRE_Int
nalu_hypre_UnorderedIntSetContains( nalu_hypre_UnorderedIntSet *s,
                               NALU_HYPRE_Int              key )
{
   //CALCULATE HASH ..........................
#ifdef NALU_HYPRE_BIGINT
   NALU_HYPRE_Int hash = nalu_hypre_BigHash(key);
#else
   NALU_HYPRE_Int hash = nalu_hypre_Hash(key);
#endif

   //CHECK IF ALREADY CONTAIN ................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment *segment = &s->segments[hash & s->segmentMask];
#endif
   NALU_HYPRE_Int bucket = hash & s->bucketMask;
   nalu_hypre_uint hopInfo = s->hopInfo[bucket];

   if (0 == hopInfo)
   {
      return 0;
   }
   else if (1 == hopInfo )
   {
      if (hash == s->hash[bucket] && key == s->key[bucket])
      {
         return 1;
      }
      else { return 0; }
   }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int startTimestamp = segment->timestamp;
#endif
   while (0 != hopInfo)
   {
      NALU_HYPRE_Int i = first_lsb_bit_indx(hopInfo);
      NALU_HYPRE_Int currElm = bucket + i;

      if (hash == s->hash[currElm] && key == s->key[currElm])
      {
         return 1;
      }
      hopInfo &= ~(1U << i);
   }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   if (segment->timestamp == startTimestamp)
   {
      return 0;
   }
#endif

   NALU_HYPRE_Int i;
   for (i = 0; i < NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i)
   {
      if (hash == s->hash[bucket + i] && key == s->key[bucket + i])
      {
         return 1;
      }
   }
   return 0;
}

static inline NALU_HYPRE_Int
nalu_hypre_UnorderedBigIntSetContains( nalu_hypre_UnorderedBigIntSet *s,
                                  NALU_HYPRE_BigInt key )
{
   //CALCULATE HASH ..........................
#if defined(NALU_HYPRE_BIGINT) || defined(NALU_HYPRE_MIXEDINT)
   NALU_HYPRE_BigInt hash = nalu_hypre_BigHash(key);
#else
   NALU_HYPRE_BigInt hash = nalu_hypre_Hash(key);
#endif

   //CHECK IF ALREADY CONTAIN ................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment *segment = &s->segments[(NALU_HYPRE_Int)(hash & s->segmentMask)];
#endif
   NALU_HYPRE_Int bucket = (NALU_HYPRE_Int)(hash & s->bucketMask);
   nalu_hypre_uint hopInfo = s->hopInfo[bucket];

   if (0 == hopInfo)
   {
      return 0;
   }
   else if (1 == hopInfo )
   {
      if (hash == s->hash[bucket] && key == s->key[bucket])
      {
         return 1;
      }
      else { return 0; }
   }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int startTimestamp = segment->timestamp;
#endif
   while (0 != hopInfo)
   {
      NALU_HYPRE_Int i = first_lsb_bit_indx(hopInfo);
      NALU_HYPRE_Int currElm = bucket + i;

      if (hash == s->hash[currElm] && key == s->key[currElm])
      {
         return 1;
      }
      hopInfo &= ~(1U << i);
   }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   if (segment->timestamp == startTimestamp)
   {
      return 0;
   }
#endif

   NALU_HYPRE_Int i;
   for (i = 0; i < NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i)
   {
      if (hash == s->hash[bucket + i] && key == s->key[bucket + i])
      {
         return 1;
      }
   }
   return 0;
}

/**
 * @ret -1 if key doesn't exist
 */
static inline NALU_HYPRE_Int
nalu_hypre_UnorderedIntMapGet( nalu_hypre_UnorderedIntMap *m,
                          NALU_HYPRE_Int key )
{
   //CALCULATE HASH ..........................
#ifdef NALU_HYPRE_BIGINT
   NALU_HYPRE_Int hash = nalu_hypre_BigHash(key);
#else
   NALU_HYPRE_Int hash = nalu_hypre_Hash(key);
#endif

   //CHECK IF ALREADY CONTAIN ................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
#endif
   nalu_hypre_HopscotchBucket *elmAry = &(m->table[hash & m->bucketMask]);
   nalu_hypre_uint hopInfo = elmAry->hopInfo;
   if (0 == hopInfo)
   {
      return -1;
   }
   else if (1 == hopInfo )
   {
      if (hash == elmAry->hash && key == elmAry->key)
      {
         return elmAry->data;
      }
      else { return -1; }
   }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int startTimestamp = segment->timestamp;
#endif
   while (0 != hopInfo)
   {
      NALU_HYPRE_Int i = first_lsb_bit_indx(hopInfo);
      nalu_hypre_HopscotchBucket* currElm = elmAry + i;
      if (hash == currElm->hash && key == currElm->key)
      {
         return currElm->data;
      }
      hopInfo &= ~(1U << i);
   }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   if (segment->timestamp == startTimestamp)
   {
      return -1;
   }
#endif

   nalu_hypre_HopscotchBucket *currBucket = &(m->table[hash & m->bucketMask]);
   NALU_HYPRE_Int i;
   for (i = 0; i < NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i, ++currBucket)
   {
      if (hash == currBucket->hash && key == currBucket->key)
      {
         return currBucket->data;
      }
   }
   return -1;
}

static inline
NALU_HYPRE_Int nalu_hypre_UnorderedBigIntMapGet( nalu_hypre_UnorderedBigIntMap *m,
                                       NALU_HYPRE_BigInt key )
{
   //CALCULATE HASH ..........................
#if defined(NALU_HYPRE_BIGINT) || defined(NALU_HYPRE_MIXEDINT)
   NALU_HYPRE_BigInt hash = nalu_hypre_BigHash(key);
#else
   NALU_HYPRE_BigInt hash = nalu_hypre_Hash(key);
#endif

   //CHECK IF ALREADY CONTAIN ................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment *segment = &m->segments[(NALU_HYPRE_Int)(hash & m->segmentMask)];
#endif
   nalu_hypre_BigHopscotchBucket *elmAry = &(m->table[(NALU_HYPRE_Int)(hash & m->bucketMask)]);
   nalu_hypre_uint hopInfo = elmAry->hopInfo;
   if (0 == hopInfo)
   {
      return -1;
   }
   else if (1 == hopInfo )
   {
      if (hash == elmAry->hash && key == elmAry->key)
      {
         return elmAry->data;
      }
      else { return -1; }
   }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   NALU_HYPRE_Int startTimestamp = segment->timestamp;
#endif
   while (0 != hopInfo)
   {
      NALU_HYPRE_Int i = first_lsb_bit_indx(hopInfo);
      nalu_hypre_BigHopscotchBucket* currElm = elmAry + i;
      if (hash == currElm->hash && key == currElm->key)
      {
         return currElm->data;
      }
      hopInfo &= ~(1U << i);
   }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   if (segment->timestamp == startTimestamp)
   {
      return -1;
   }
#endif

   nalu_hypre_BigHopscotchBucket *currBucket = &(m->table[hash & m->bucketMask]);
   NALU_HYPRE_Int i;
   for (i = 0; i < NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE; ++i, ++currBucket)
   {
      if (hash == currBucket->hash && key == currBucket->key)
      {
         return currBucket->data;
      }
   }
   return -1;
}

//status Operations .........................................................
static inline
NALU_HYPRE_Int nalu_hypre_UnorderedIntSetSize( nalu_hypre_UnorderedIntSet *s )
{
   NALU_HYPRE_Int counter = 0;
   NALU_HYPRE_Int n = s->bucketMask + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
   NALU_HYPRE_Int i;
   for (i = 0; i < n; ++i)
   {
      if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i])
      {
         ++counter;
      }
   }
   return counter;
}

static inline
NALU_HYPRE_Int nalu_hypre_UnorderedBigIntSetSize( nalu_hypre_UnorderedBigIntSet *s )
{
   NALU_HYPRE_Int counter = 0;
   NALU_HYPRE_BigInt n = s->bucketMask + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
   NALU_HYPRE_Int i;
   for (i = 0; i < n; ++i)
   {
      if (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != s->hash[i])
      {
         ++counter;
      }
   }
   return counter;
}

static inline NALU_HYPRE_Int
nalu_hypre_UnorderedIntMapSize( nalu_hypre_UnorderedIntMap *m )
{
   NALU_HYPRE_Int counter = 0;
   NALU_HYPRE_Int n = m->bucketMask + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
   NALU_HYPRE_Int i;
   for (i = 0; i < n; ++i)
   {
      if ( NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != m->table[i].hash )
      {
         ++counter;
      }
   }
   return counter;
}

static inline NALU_HYPRE_Int
nalu_hypre_UnorderedBigIntMapSize( nalu_hypre_UnorderedBigIntMap *m )
{
   NALU_HYPRE_Int counter = 0;
   NALU_HYPRE_Int n = m->bucketMask + NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE;
   NALU_HYPRE_Int i;
   for (i = 0; i < n; ++i)
   {
      if ( NALU_HYPRE_HOPSCOTCH_HASH_EMPTY != m->table[i].hash )
      {
         ++counter;
      }
   }
   return counter;
}

NALU_HYPRE_Int *nalu_hypre_UnorderedIntSetCopyToArray( nalu_hypre_UnorderedIntSet *s, NALU_HYPRE_Int *len );
NALU_HYPRE_BigInt *nalu_hypre_UnorderedBigIntSetCopyToArray( nalu_hypre_UnorderedBigIntSet *s, NALU_HYPRE_Int *len );

//modification Operations ...................................................
static inline void
nalu_hypre_UnorderedIntSetPut( nalu_hypre_UnorderedIntSet *s,
                          NALU_HYPRE_Int key )
{
   //CALCULATE HASH ..........................
#ifdef NALU_HYPRE_BIGINT
   NALU_HYPRE_Int hash = nalu_hypre_BigHash(key);
#else
   NALU_HYPRE_Int hash = nalu_hypre_Hash(key);
#endif

   //LOCK KEY HASH ENTERY ....................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment  *segment = &s->segments[hash & s->segmentMask];
   omp_set_lock(&segment->lock);
#endif
   NALU_HYPRE_Int bucket = hash & s->bucketMask;

   //CHECK IF ALREADY CONTAIN ................
   nalu_hypre_uint hopInfo = s->hopInfo[bucket];
   while (0 != hopInfo)
   {
      NALU_HYPRE_Int i = first_lsb_bit_indx(hopInfo);
      NALU_HYPRE_Int currElm = bucket + i;

      if (hash == s->hash[currElm] && key == s->key[currElm])
      {
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         omp_unset_lock(&segment->lock);
#endif
         return;
      }
      hopInfo &= ~(1U << i);
   }

   //LOOK FOR FREE BUCKET ....................
   NALU_HYPRE_Int free_bucket = bucket;
   NALU_HYPRE_Int free_dist = 0;
   for ( ; free_dist < NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
   {
      if ( (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY == s->hash[free_bucket]) &&
           (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY ==
            nalu_hypre_compare_and_swap((NALU_HYPRE_Int *)&s->hash[free_bucket],
                                   (NALU_HYPRE_Int)NALU_HYPRE_HOPSCOTCH_HASH_EMPTY,
                                   (NALU_HYPRE_Int)NALU_HYPRE_HOPSCOTCH_HASH_BUSY)) )
      {
         break;
      }
   }

   //PLACE THE NEW KEY .......................
   if (free_dist < NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE)
   {
      do
      {
         if (free_dist < NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE)
         {
            s->key[free_bucket]  = key;
            s->hash[free_bucket] = hash;
            s->hopInfo[bucket]  |= 1U << free_dist;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            omp_unset_lock(&segment->lock);
#endif
            return;
         }
         nalu_hypre_UnorderedIntSetFindCloserFreeBucket(s,
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
                                                   segment,
#endif
                                                   &free_bucket, &free_dist);
      }
      while (-1 != free_bucket);
   }

   //NEED TO RESIZE ..........................
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ERROR - RESIZE is not implemented\n");
   /*fprintf(stderr, "ERROR - RESIZE is not implemented\n");*/
   exit(1);
   return;
}

static inline void
nalu_hypre_UnorderedBigIntSetPut( nalu_hypre_UnorderedBigIntSet *s,
                             NALU_HYPRE_BigInt key )
{
   //CALCULATE HASH ..........................
#if defined(NALU_HYPRE_BIGINT) || defined(NALU_HYPRE_MIXEDINT)
   NALU_HYPRE_BigInt hash = nalu_hypre_BigHash(key);
#else
   NALU_HYPRE_BigInt hash = nalu_hypre_Hash(key);
#endif

   //LOCK KEY HASH ENTERY ....................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment  *segment = &s->segments[hash & s->segmentMask];
   omp_set_lock(&segment->lock);
#endif
   NALU_HYPRE_Int bucket = (NALU_HYPRE_Int)(hash & s->bucketMask);

   //CHECK IF ALREADY CONTAIN ................
   nalu_hypre_uint hopInfo = s->hopInfo[bucket];
   while (0 != hopInfo)
   {
      NALU_HYPRE_Int i = first_lsb_bit_indx(hopInfo);
      NALU_HYPRE_Int currElm = bucket + i;

      if (hash == s->hash[currElm] && key == s->key[currElm])
      {
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         omp_unset_lock(&segment->lock);
#endif
         return;
      }
      hopInfo &= ~(1U << i);
   }

   //LOOK FOR FREE BUCKET ....................
   NALU_HYPRE_Int free_bucket = bucket;
   NALU_HYPRE_Int free_dist = 0;
   for ( ; free_dist < NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
   {
      if ( (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY == s->hash[free_bucket]) &&
           (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY ==
            nalu_hypre_compare_and_swap((NALU_HYPRE_Int *)&s->hash[free_bucket],
                                   (NALU_HYPRE_Int)NALU_HYPRE_HOPSCOTCH_HASH_EMPTY,
                                   (NALU_HYPRE_Int)NALU_HYPRE_HOPSCOTCH_HASH_BUSY)) )
      {
         break;
      }
   }

   //PLACE THE NEW KEY .......................
   if (free_dist < NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE)
   {
      do
      {
         if (free_dist < NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE)
         {
            s->key[free_bucket]  = key;
            s->hash[free_bucket] = hash;
            s->hopInfo[bucket]  |= 1U << free_dist;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            omp_unset_lock(&segment->lock);
#endif
            return;
         }
         nalu_hypre_UnorderedBigIntSetFindCloserFreeBucket(s,
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
                                                      segment,
#endif
                                                      &free_bucket, &free_dist);
      }
      while (-1 != free_bucket);
   }

   //NEED TO RESIZE ..........................
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ERROR - RESIZE is not implemented\n");
   /*fprintf(stderr, "ERROR - RESIZE is not implemented\n");*/
   exit(1);
   return;
}

static inline NALU_HYPRE_Int
nalu_hypre_UnorderedIntMapPutIfAbsent( nalu_hypre_UnorderedIntMap *m,
                                  NALU_HYPRE_Int key, NALU_HYPRE_Int data )
{
   //CALCULATE HASH ..........................
#ifdef NALU_HYPRE_BIGINT
   NALU_HYPRE_Int hash = nalu_hypre_BigHash(key);
#else
   NALU_HYPRE_Int hash = nalu_hypre_Hash(key);
#endif

   //LOCK KEY HASH ENTERY ....................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
   omp_set_lock(&segment->lock);
#endif
   nalu_hypre_HopscotchBucket* startBucket = &(m->table[hash & m->bucketMask]);

   //CHECK IF ALREADY CONTAIN ................
   nalu_hypre_uint hopInfo = startBucket->hopInfo;
   while (0 != hopInfo)
   {
      NALU_HYPRE_Int i = first_lsb_bit_indx(hopInfo);
      nalu_hypre_HopscotchBucket* currElm = startBucket + i;
      if (hash == currElm->hash && key == currElm->key)
      {
         NALU_HYPRE_Int rc = currElm->data;
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         omp_unset_lock(&segment->lock);
#endif
         return rc;
      }
      hopInfo &= ~(1U << i);
   }

   //LOOK FOR FREE BUCKET ....................
   nalu_hypre_HopscotchBucket* free_bucket = startBucket;
   NALU_HYPRE_Int free_dist = 0;
   for ( ; free_dist < NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
   {
      if ( (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY == free_bucket->hash) &&
           (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY ==
            nalu_hypre_compare_and_swap((NALU_HYPRE_Int *)&free_bucket->hash,
                                   (NALU_HYPRE_Int)NALU_HYPRE_HOPSCOTCH_HASH_EMPTY,
                                   (NALU_HYPRE_Int)NALU_HYPRE_HOPSCOTCH_HASH_BUSY)) )
      {
         break;
      }
   }

   //PLACE THE NEW KEY .......................
   if (free_dist < NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE)
   {
      do
      {
         if (free_dist < NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE)
         {
            free_bucket->data     = data;
            free_bucket->key      = key;
            free_bucket->hash     = hash;
            startBucket->hopInfo |= 1U << free_dist;
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            omp_unset_lock(&segment->lock);
#endif
            return NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
         }
         nalu_hypre_UnorderedIntMapFindCloserFreeBucket(m,
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
                                                   segment,
#endif
                                                   &free_bucket, &free_dist);
      }
      while (NULL != free_bucket);
   }

   //NEED TO RESIZE ..........................
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ERROR - RESIZE is not implemented\n");
   /*fprintf(stderr, "ERROR - RESIZE is not implemented\n");*/
   exit(1);
   return NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
}

static inline NALU_HYPRE_Int
nalu_hypre_UnorderedBigIntMapPutIfAbsent( nalu_hypre_UnorderedBigIntMap *m,
                                     NALU_HYPRE_BigInt key, NALU_HYPRE_Int data)
{
   //CALCULATE HASH ..........................
#if defined(NALU_HYPRE_BIGINT) || defined(NALU_HYPRE_MIXEDINT)
   NALU_HYPRE_BigInt hash = nalu_hypre_BigHash(key);
#else
   NALU_HYPRE_BigInt hash = nalu_hypre_Hash(key);
#endif

   //LOCK KEY HASH ENTERY ....................
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
   omp_set_lock(&segment->lock);
#endif
   nalu_hypre_BigHopscotchBucket* startBucket = &(m->table[hash & m->bucketMask]);

   //CHECK IF ALREADY CONTAIN ................
   nalu_hypre_uint hopInfo = startBucket->hopInfo;
   while (0 != hopInfo)
   {
      NALU_HYPRE_Int i = first_lsb_bit_indx(hopInfo);
      nalu_hypre_BigHopscotchBucket* currElm = startBucket + i;
      if (hash == currElm->hash && key == currElm->key)
      {
         NALU_HYPRE_Int rc = currElm->data;
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         omp_unset_lock(&segment->lock);
#endif
         return rc;
      }
      hopInfo &= ~(1U << i);
   }

   //LOOK FOR FREE BUCKET ....................
   nalu_hypre_BigHopscotchBucket* free_bucket = startBucket;
   NALU_HYPRE_Int free_dist = 0;
   for ( ; free_dist < NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
   {
      if ( (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY == free_bucket->hash) &&
           (NALU_HYPRE_HOPSCOTCH_HASH_EMPTY ==
            nalu_hypre_compare_and_swap((NALU_HYPRE_Int *)&free_bucket->hash,
                                   (NALU_HYPRE_Int)NALU_HYPRE_HOPSCOTCH_HASH_EMPTY,
                                   (NALU_HYPRE_Int)NALU_HYPRE_HOPSCOTCH_HASH_BUSY)) )
      {
         break;
      }
   }

   //PLACE THE NEW KEY .......................
   if (free_dist < NALU_HYPRE_HOPSCOTCH_HASH_INSERT_RANGE)
   {
      do
      {
         if (free_dist < NALU_HYPRE_HOPSCOTCH_HASH_HOP_RANGE)
         {
            free_bucket->data     = data;
            free_bucket->key      = key;
            free_bucket->hash     = hash;
            startBucket->hopInfo |= 1U << free_dist;
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
            omp_unset_lock(&segment->lock);
#endif
            return NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
         }
         nalu_hypre_UnorderedBigIntMapFindCloserFreeBucket(m,
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
                                                      segment,
#endif
                                                      &free_bucket, &free_dist);
      }
      while (NULL != free_bucket);
   }

   //NEED TO RESIZE ..........................
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ERROR - RESIZE is not implemented\n");
   /*fprintf(stderr, "ERROR - RESIZE is not implemented\n");*/
   exit(1);
   return NALU_HYPRE_HOPSCOTCH_HASH_EMPTY;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // nalu_hypre_HOPSCOTCH_HASH_HEADER
