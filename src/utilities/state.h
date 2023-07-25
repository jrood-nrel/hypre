/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_STATE_HEADER
#define nalu_hypre_STATE_HEADER

/*--------------------------------------------------------------------------
 * hypre library state
 *--------------------------------------------------------------------------*/

typedef enum nalu_hypre_State_enum
{
   NALU_HYPRE_STATE_NONE        = 0,
   NALU_HYPRE_STATE_INITIALIZED = 1,
   NALU_HYPRE_STATE_FINALIZED   = 2
} nalu_hypre_State;

extern nalu_hypre_State nalu_hypre__global_state;

#endif /* nalu_hypre_STATE_HEADER */
