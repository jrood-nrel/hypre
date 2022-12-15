/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for doing timing
 *
 *****************************************************************************/

#ifndef NALU_HYPRE_TIMING_HEADER
#define NALU_HYPRE_TIMING_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Prototypes for low-level timing routines
 *--------------------------------------------------------------------------*/

/* timer.c */
NALU_HYPRE_Real time_getWallclockSeconds( void );
NALU_HYPRE_Real time_getCPUSeconds( void );
NALU_HYPRE_Real time_get_wallclock_seconds_( void );
NALU_HYPRE_Real time_get_cpu_seconds_( void );

/*--------------------------------------------------------------------------
 * With timing off
 *--------------------------------------------------------------------------*/

#ifndef NALU_HYPRE_TIMING

#define nalu_hypre_InitializeTiming(name) 0
#define nalu_hypre_FinalizeTiming(index)
#define nalu_hypre_IncFLOPCount(inc)
#define nalu_hypre_BeginTiming(i)
#define nalu_hypre_EndTiming(i)
#define nalu_hypre_PrintTiming(heading, comm)
#define nalu_hypre_ClearTiming()
#define nalu_hypre_GetTiming()

/*--------------------------------------------------------------------------
 * With timing on
 *--------------------------------------------------------------------------*/

#else

/*-------------------------------------------------------
 * Global timing structure
 *-------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Real  *wall_time;
   NALU_HYPRE_Real  *cpu_time;
   NALU_HYPRE_Real  *flops;
   char   **name;
   NALU_HYPRE_Int     *state;     /* boolean flag to allow for recursive timing */
   NALU_HYPRE_Int     *num_regs;  /* count of how many times a name is registered */

   NALU_HYPRE_Int      num_names;
   NALU_HYPRE_Int      size;

   NALU_HYPRE_Real   wall_count;
   NALU_HYPRE_Real   CPU_count;
   NALU_HYPRE_Real   FLOP_count;

} nalu_hypre_TimingType;

#ifdef NALU_HYPRE_TIMING_GLOBALS
nalu_hypre_TimingType *nalu_hypre_global_timing = NULL;
#else
extern nalu_hypre_TimingType *nalu_hypre_global_timing;
#endif

/*-------------------------------------------------------
 * Accessor functions
 *-------------------------------------------------------*/

#define nalu_hypre_TimingWallTime(i) (nalu_hypre_global_timing -> wall_time[(i)])
#define nalu_hypre_TimingCPUTime(i)  (nalu_hypre_global_timing -> cpu_time[(i)])
#define nalu_hypre_TimingFLOPS(i)    (nalu_hypre_global_timing -> flops[(i)])
#define nalu_hypre_TimingName(i)     (nalu_hypre_global_timing -> name[(i)])
#define nalu_hypre_TimingState(i)    (nalu_hypre_global_timing -> state[(i)])
#define nalu_hypre_TimingNumRegs(i)  (nalu_hypre_global_timing -> num_regs[(i)])
#define nalu_hypre_TimingWallCount   (nalu_hypre_global_timing -> wall_count)
#define nalu_hypre_TimingCPUCount    (nalu_hypre_global_timing -> CPU_count)
#define nalu_hypre_TimingFLOPCount   (nalu_hypre_global_timing -> FLOP_count)

/*-------------------------------------------------------
 * Prototypes
 *-------------------------------------------------------*/

/* timing.c */
NALU_HYPRE_Int nalu_hypre_InitializeTiming( const char *name );
NALU_HYPRE_Int nalu_hypre_FinalizeTiming( NALU_HYPRE_Int time_index );
NALU_HYPRE_Int nalu_hypre_FinalizeAllTimings();
NALU_HYPRE_Int nalu_hypre_IncFLOPCount( NALU_HYPRE_BigInt inc );
NALU_HYPRE_Int nalu_hypre_BeginTiming( NALU_HYPRE_Int time_index );
NALU_HYPRE_Int nalu_hypre_EndTiming( NALU_HYPRE_Int time_index );
NALU_HYPRE_Int nalu_hypre_ClearTiming( void );
NALU_HYPRE_Int nalu_hypre_PrintTiming( const char *heading, MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_GetTiming( const char *heading, NALU_HYPRE_Real *wall_time_ptr, MPI_Comm comm );

#endif

#ifdef __cplusplus
}
#endif

#endif

