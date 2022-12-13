/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include <stdarg.h>
#include <stdio.h>

#define hypre_printf_buffer_len 4096
char hypre_printf_buffer[hypre_printf_buffer_len];

// #ifdef NALU_HYPRE_BIGINT

/* these prototypes are missing by default for some compilers */
/*
int vscanf( const char *format , va_list arg );
int vfscanf( FILE *stream , const char *format, va_list arg );
int vsscanf( const char *s , const char *format, va_list arg );
*/

NALU_HYPRE_Int
new_format( const char *format,
            char **newformat_ptr )
{
   const char *fp;
   char       *newformat, *nfp;
   NALU_HYPRE_Int   newformatlen;
   NALU_HYPRE_Int   copychar;
   NALU_HYPRE_Int   foundpercent = 0;

   newformatlen = 2 * strlen(format) + 1; /* worst case is all %d's to %lld's */

   if (newformatlen > hypre_printf_buffer_len)
   {
      newformat = hypre_TAlloc(char, newformatlen, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      newformat = hypre_printf_buffer;
   }

   nfp = newformat;
   for (fp = format; *fp != '\0'; fp++)
   {
      copychar = 1;
      if (*fp == '%')
      {
         foundpercent = 1;
      }
      else if (foundpercent)
      {
         if (*fp == 'l')
         {
            fp++; /* remove 'l' and maybe add it back in switch statement */
            if (*fp == 'l')
            {
               fp++; /* remove second 'l' if present */
            }
         }
         switch (*fp)
         {
            case 'b': /* used for BigInt type in hypre */
#if defined(NALU_HYPRE_BIGINT) || defined(NALU_HYPRE_MIXEDINT)
               *nfp = 'l'; nfp++;
               *nfp = 'l'; nfp++;
#endif
               *nfp = 'd'; nfp++; copychar = 0;
               foundpercent = 0; break;
            case 'd':
            case 'i':
#if defined(NALU_HYPRE_BIGINT)
               *nfp = 'l'; nfp++;
               *nfp = 'l'; nfp++;
#endif
               foundpercent = 0; break;
            case 'f':
            case 'e':
            case 'E':
            case 'g':
            case 'G':
#if defined(NALU_HYPRE_SINGLE)          /* no modifier */
#elif defined(NALU_HYPRE_LONG_DOUBLE)   /* modify with 'L' */
               *nfp = 'L'; nfp++;
#else                              /* modify with 'l' (default is _double_) */
               *nfp = 'l'; nfp++;
#endif
               foundpercent = 0; break;
            case 'c':
            case 'n':
            case 'o':
            case 'p':
            case 's':
            case 'u':
            case 'x':
            case 'X':
            case '%':
               foundpercent = 0; break;
         }
      }
      if (copychar)
      {
         *nfp = *fp; nfp++;
      }
   }
   *nfp = *fp;

   *newformat_ptr = newformat;

   /*   printf("\nNEWFORMAT: %s\n", *newformat_ptr);*/

   return 0;
}

NALU_HYPRE_Int
free_format( char *newformat )
{
   if (newformat != hypre_printf_buffer)
   {
      hypre_TFree(newformat, NALU_HYPRE_MEMORY_HOST);
   }

   return 0;
}

NALU_HYPRE_Int
hypre_ndigits( NALU_HYPRE_BigInt number )
{
   NALU_HYPRE_Int     ndigits = 0;

   while (number)
   {
      number /= 10;
      ndigits++;
   }

   return ndigits;
}

/* printf functions */

NALU_HYPRE_Int
hypre_printf( const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   NALU_HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vprintf(newformat, ap);
   free_format(newformat);
   va_end(ap);

   fflush(stdout);

   return ierr;
}

NALU_HYPRE_Int
hypre_fprintf( FILE *stream, const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   NALU_HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vfprintf(stream, newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

NALU_HYPRE_Int
hypre_sprintf( char *s, const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   NALU_HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vsprintf(s, newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

/* scanf functions */

NALU_HYPRE_Int
hypre_scanf( const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   NALU_HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vscanf(newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

NALU_HYPRE_Int
hypre_fscanf( FILE *stream, const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   NALU_HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vfscanf(stream, newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

NALU_HYPRE_Int
hypre_sscanf( char *s, const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   NALU_HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vsscanf(s, newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

NALU_HYPRE_Int
hypre_ParPrintf(MPI_Comm comm, const char *format, ...)
{
   NALU_HYPRE_Int my_id;
   NALU_HYPRE_Int ierr = hypre_MPI_Comm_rank(comm, &my_id);

   if (ierr)
   {
      return ierr;
   }

   if (!my_id)
   {
      va_list ap;
      char   *newformat;

      va_start(ap, format);
      new_format(format, &newformat);
      ierr = vprintf(newformat, ap);
      free_format(newformat);
      va_end(ap);

      fflush(stdout);
   }

   return ierr;
}
// #else
//
// /* this is used only to eliminate compiler warnings */
// NALU_HYPRE_Int hypre_printf_empty;
//
// #endif
