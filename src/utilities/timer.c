/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * File: timer.c
 * Author:  Scott Kohn (skohn@llnl.gov)
 * Description:   somewhat portable timing routines for C++, C, and Fortran
 *
 * This has been modified many times since the original author's version.
 */

#include "_nalu_hypre_utilities.h"

#include <time.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/times.h>
#endif

NALU_HYPRE_Real time_getWallclockSeconds(void)
{
#ifndef NALU_HYPRE_SEQUENTIAL
   return (nalu_hypre_MPI_Wtime());
#else
#ifdef WIN32
   clock_t cl = clock();
   return (((NALU_HYPRE_Real) cl) / ((NALU_HYPRE_Real) CLOCKS_PER_SEC));
#else
   struct tms usage;
   nalu_hypre_longint wallclock = times(&usage);
   return (((NALU_HYPRE_Real) wallclock) / ((NALU_HYPRE_Real) sysconf(_SC_CLK_TCK)));
#endif
#endif
}

NALU_HYPRE_Real time_getCPUSeconds(void)
{
#ifndef TIMER_NO_SYS
   clock_t cpuclock = clock();
   return (((NALU_HYPRE_Real) (cpuclock)) / ((NALU_HYPRE_Real) CLOCKS_PER_SEC));
#else
   return (0.0);
#endif
}

NALU_HYPRE_Real time_get_wallclock_seconds_(void)
{
   return (time_getWallclockSeconds());
}

NALU_HYPRE_Real time_get_cpu_seconds_(void)
{
   return (time_getCPUSeconds());
}
