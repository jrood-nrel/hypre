/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_THREADING_HEADER
#define hypre_THREADING_HEADER

#ifdef NALU_HYPRE_USING_OPENMP

NALU_HYPRE_Int hypre_NumThreads( void );
NALU_HYPRE_Int hypre_NumActiveThreads( void );
NALU_HYPRE_Int hypre_GetThreadNum( void );
void      hypre_SetNumThreads(NALU_HYPRE_Int nt);

#else

#define hypre_NumThreads() 1
#define hypre_NumActiveThreads() 1
#define hypre_GetThreadNum() 0
#define hypre_SetNumThreads(x)

#endif

void hypre_GetSimpleThreadPartition( NALU_HYPRE_Int *begin, NALU_HYPRE_Int *end, NALU_HYPRE_Int n );

#endif

