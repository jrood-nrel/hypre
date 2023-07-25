/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

/* Global variable: library state (initialized, finalized, or none) */
nalu_hypre_State nalu_hypre__global_state = NALU_HYPRE_STATE_NONE;

/*--------------------------------------------------------------------------
 * NALU_HYPRE_Initialized
 *
 * Public function for nalu_hypre_Initialized
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_Initialized( void )
{
   return nalu_hypre_Initialized();
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_Finalized
 *
 * Public function for nalu_hypre_Finalized
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_Finalized( void )
{
   return nalu_hypre_Finalized();
}

/*--------------------------------------------------------------------------
 * nalu_hypre_Initialized
 *
 * This function returns True when the library has been initialized, but not
 * finalized yet.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_Initialized( void )
{
   return (nalu_hypre__global_state == NALU_HYPRE_STATE_INITIALIZED);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_Finalized
 *
 * This function returns True when the library is in finalized state;
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_Finalized( void )
{
   return (nalu_hypre__global_state == NALU_HYPRE_STATE_FINALIZED);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SetInitialized
 *
 * This function sets the library state to initialized
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SetInitialized( void )
{
   nalu_hypre__global_state = NALU_HYPRE_STATE_INITIALIZED;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SetFinalized
 *
 * This function sets the library state to finalized
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SetFinalized( void )
{
   nalu_hypre__global_state = NALU_HYPRE_STATE_FINALIZED;

   return nalu_hypre_error_flag;
}
