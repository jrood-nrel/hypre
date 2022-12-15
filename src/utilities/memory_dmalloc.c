/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Memory management utilities
 *
 * Routines to use "Debug Malloc Library", dmalloc
 *
 *****************************************************************************/

#ifdef NALU_HYPRE_MEMORY_DMALLOC

#include "memory.h"
#include <dmalloc.h>

char dmalloc_logpath_memory[256];

/*--------------------------------------------------------------------------
 * nalu_hypre_InitMemoryDebugDML
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_InitMemoryDebugDML( NALU_HYPRE_Int id  )
{
   NALU_HYPRE_Int  *iptr;

   /* do this to get the Debug Malloc Library started/initialized */
   iptr = nalu_hypre_TAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(iptr, NALU_HYPRE_MEMORY_HOST);

   dmalloc_logpath = dmalloc_logpath_memory;
   nalu_hypre_sprintf(dmalloc_logpath, "dmalloc.log.%04d", id);

   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FinalizeMemoryDebugDML
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FinalizeMemoryDebugDML( )
{
   dmalloc_verify(NULL);

   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MAllocDML
 *--------------------------------------------------------------------------*/

char *
nalu_hypre_MAllocDML( NALU_HYPRE_Int   size,
                 char *file,
                 NALU_HYPRE_Int   line )
{
   char *ptr;

   if (size > 0)
   {
      ptr = _malloc_leap(file, line, size);
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CAllocDML
 *--------------------------------------------------------------------------*/

char *
nalu_hypre_CAllocDML( NALU_HYPRE_Int   count,
                 NALU_HYPRE_Int   elt_size,
                 char *file,
                 NALU_HYPRE_Int   line    )
{
   char *ptr;
   NALU_HYPRE_Int   size = count * elt_size;

   if (size > 0)
   {
      ptr = _calloc_leap(file, line, count, elt_size);
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ReAllocDML
 *--------------------------------------------------------------------------*/

char *
nalu_hypre_ReAllocDML( char *ptr,
                  NALU_HYPRE_Int   size,
                  char *file,
                  NALU_HYPRE_Int   line )
{
   ptr = _realloc_leap(file, line, ptr, size);

   return ptr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FreeDML
 *--------------------------------------------------------------------------*/

void
nalu_hypre_FreeDML( char *ptr,
               char *file,
               NALU_HYPRE_Int   line )
{
   if (ptr)
   {
      _free_leap(file, line, ptr);
   }
}

#else

/* this is used only to eliminate compiler warnings */
char nalu_hypre_memory_dmalloc_empty;

#endif
