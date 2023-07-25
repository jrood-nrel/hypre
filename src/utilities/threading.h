/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_THREADING_HEADER
#define nalu_hypre_THREADING_HEADER

#ifdef NALU_HYPRE_USING_OPENMP

NALU_HYPRE_Int nalu_hypre_NumThreads( void );
NALU_HYPRE_Int nalu_hypre_NumActiveThreads( void );
NALU_HYPRE_Int nalu_hypre_GetThreadNum( void );
void      nalu_hypre_SetNumThreads(NALU_HYPRE_Int nt);

#else

#define nalu_hypre_NumThreads() 1
#define nalu_hypre_NumActiveThreads() 1
#define nalu_hypre_GetThreadNum() 0
#define nalu_hypre_SetNumThreads(x)

#endif

void nalu_hypre_GetSimpleThreadPartition( NALU_HYPRE_Int *begin, NALU_HYPRE_Int *end, NALU_HYPRE_Int n );

#endif

