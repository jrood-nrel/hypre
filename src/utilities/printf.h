/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_PRINTF_HEADER
#define nalu_hypre_PRINTF_HEADER

#include <stdio.h>

/* printf.c */
// #ifdef NALU_HYPRE_BIGINT
NALU_HYPRE_Int nalu_hypre_ndigits( NALU_HYPRE_BigInt number );
NALU_HYPRE_Int nalu_hypre_printf( const char *format, ... );
NALU_HYPRE_Int nalu_hypre_fprintf( FILE *stream, const char *format, ... );
NALU_HYPRE_Int nalu_hypre_sprintf( char *s, const char *format, ... );
NALU_HYPRE_Int nalu_hypre_scanf( const char *format, ... );
NALU_HYPRE_Int nalu_hypre_fscanf( FILE *stream, const char *format, ... );
NALU_HYPRE_Int nalu_hypre_sscanf( char *s, const char *format, ... );
NALU_HYPRE_Int nalu_hypre_ParPrintf(MPI_Comm comm, const char *format, ...);
// #else
// #define nalu_hypre_printf  printf
// #define nalu_hypre_fprintf fprintf
// #define nalu_hypre_sprintf sprintf
// #define nalu_hypre_scanf   scanf
// #define nalu_hypre_fscanf  fscanf
// #define nalu_hypre_sscanf  sscanf
// #endif

#endif
