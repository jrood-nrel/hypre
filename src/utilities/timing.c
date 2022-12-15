/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Routines for doing timing.
 *
 *****************************************************************************/

#define NALU_HYPRE_TIMING
#define NALU_HYPRE_TIMING_GLOBALS
#include "_nalu_hypre_utilities.h"
#include "timing.h"

/*-------------------------------------------------------
 * Timing macros
 *-------------------------------------------------------*/

#define nalu_hypre_StartTiming() \
nalu_hypre_TimingWallCount -= time_getWallclockSeconds();\
nalu_hypre_TimingCPUCount -= time_getCPUSeconds()

#define nalu_hypre_StopTiming() \
nalu_hypre_TimingWallCount += time_getWallclockSeconds();\
nalu_hypre_TimingCPUCount += time_getCPUSeconds()

#define nalu_hypre_global_timing_ref(index,field) nalu_hypre_global_timing->field

/*--------------------------------------------------------------------------
 * nalu_hypre_InitializeTiming
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_InitializeTiming( const char *name )
{
   NALU_HYPRE_Int      time_index;

   NALU_HYPRE_Real  *old_wall_time;
   NALU_HYPRE_Real  *old_cpu_time;
   NALU_HYPRE_Real  *old_flops;
   char   **old_name;
   NALU_HYPRE_Int     *old_state;
   NALU_HYPRE_Int     *old_num_regs;

   NALU_HYPRE_Int      new_name;
   NALU_HYPRE_Int      i;

   /*-------------------------------------------------------
    * Allocate global TimingType structure if needed
    *-------------------------------------------------------*/

   if (nalu_hypre_global_timing == NULL)
   {
      nalu_hypre_global_timing = nalu_hypre_CTAlloc(nalu_hypre_TimingType,  1, NALU_HYPRE_MEMORY_HOST);
   }

   /*-------------------------------------------------------
    * Check to see if name has already been registered
    *-------------------------------------------------------*/

   new_name = 1;
   for (i = 0; i < (nalu_hypre_global_timing_ref(threadid, size)); i++)
   {
      if (nalu_hypre_TimingNumRegs(i) > 0)
      {
         if (strcmp(name, nalu_hypre_TimingName(i)) == 0)
         {
            new_name = 0;
            time_index = i;
            nalu_hypre_TimingNumRegs(time_index) ++;
            break;
         }
      }
   }

   if (new_name)
   {
      for (i = 0; i < nalu_hypre_global_timing_ref(threadid, size); i++)
      {
         if (nalu_hypre_TimingNumRegs(i) == 0)
         {
            break;
         }
      }
      time_index = i;
   }

   /*-------------------------------------------------------
    * Register the new timing name
    *-------------------------------------------------------*/

   if (new_name)
   {
      if (time_index == (nalu_hypre_global_timing_ref(threadid, size)))
      {
         old_wall_time = (nalu_hypre_global_timing_ref(threadid, wall_time));
         old_cpu_time  = (nalu_hypre_global_timing_ref(threadid, cpu_time));
         old_flops     = (nalu_hypre_global_timing_ref(threadid, flops));
         old_name      = (nalu_hypre_global_timing_ref(threadid, name));
         old_state     = (nalu_hypre_global_timing_ref(threadid, state));
         old_num_regs  = (nalu_hypre_global_timing_ref(threadid, num_regs));

         (nalu_hypre_global_timing_ref(threadid, wall_time)) =
            nalu_hypre_CTAlloc(NALU_HYPRE_Real,  (time_index + 1), NALU_HYPRE_MEMORY_HOST);
         (nalu_hypre_global_timing_ref(threadid, cpu_time))  =
            nalu_hypre_CTAlloc(NALU_HYPRE_Real,  (time_index + 1), NALU_HYPRE_MEMORY_HOST);
         (nalu_hypre_global_timing_ref(threadid, flops))     =
            nalu_hypre_CTAlloc(NALU_HYPRE_Real,  (time_index + 1), NALU_HYPRE_MEMORY_HOST);
         (nalu_hypre_global_timing_ref(threadid, name))      =
            nalu_hypre_CTAlloc(char *,  (time_index + 1), NALU_HYPRE_MEMORY_HOST);
         (nalu_hypre_global_timing_ref(threadid, state))     =
            nalu_hypre_CTAlloc(NALU_HYPRE_Int,     (time_index + 1), NALU_HYPRE_MEMORY_HOST);
         (nalu_hypre_global_timing_ref(threadid, num_regs))  =
            nalu_hypre_CTAlloc(NALU_HYPRE_Int,     (time_index + 1), NALU_HYPRE_MEMORY_HOST);
         (nalu_hypre_global_timing_ref(threadid, size)) ++;

         for (i = 0; i < time_index; i++)
         {
            nalu_hypre_TimingWallTime(i) = old_wall_time[i];
            nalu_hypre_TimingCPUTime(i)  = old_cpu_time[i];
            nalu_hypre_TimingFLOPS(i)    = old_flops[i];
            nalu_hypre_TimingName(i)     = old_name[i];
            nalu_hypre_TimingState(i)    = old_state[i];
            nalu_hypre_TimingNumRegs(i)  = old_num_regs[i];
         }

         nalu_hypre_TFree(old_wall_time, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(old_cpu_time, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(old_flops, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(old_name, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(old_state, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(old_num_regs, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TimingName(time_index) = nalu_hypre_CTAlloc(char,  80, NALU_HYPRE_MEMORY_HOST);
      strncpy(nalu_hypre_TimingName(time_index), name, 79);
      nalu_hypre_TimingState(time_index)   = 0;
      nalu_hypre_TimingNumRegs(time_index) = 1;
      (nalu_hypre_global_timing_ref(threadid, num_names)) ++;
   }

   return time_index;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FinalizeTiming
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FinalizeTiming( NALU_HYPRE_Int time_index )
{
   NALU_HYPRE_Int  ierr = 0;
   NALU_HYPRE_Int  i;

   if (nalu_hypre_global_timing == NULL)
   {
      return ierr;
   }

   if (time_index < (nalu_hypre_global_timing_ref(threadid, size)))
   {
      if (nalu_hypre_TimingNumRegs(time_index) > 0)
      {
         nalu_hypre_TimingNumRegs(time_index) --;
      }

      if (nalu_hypre_TimingNumRegs(time_index) == 0)
      {
         nalu_hypre_TFree(nalu_hypre_TimingName(time_index), NALU_HYPRE_MEMORY_HOST);
         (nalu_hypre_global_timing_ref(threadid, num_names)) --;
      }
   }

   if ((nalu_hypre_global_timing -> num_names) == 0)
   {
      for (i = 0; i < (nalu_hypre_global_timing -> size); i++)
      {
         nalu_hypre_TFree(nalu_hypre_global_timing_ref(i,  wall_time), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_global_timing_ref(i,  cpu_time), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_global_timing_ref(i,  flops), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_global_timing_ref(i,  name), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_global_timing_ref(i,  state), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_global_timing_ref(i,  num_regs), NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(nalu_hypre_global_timing, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_global_timing = NULL;
   }

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_FinalizeAllTimings()
{
   NALU_HYPRE_Int time_index, ierr = 0;

   if (nalu_hypre_global_timing == NULL)
   {
      return ierr;
   }

   NALU_HYPRE_Int size = nalu_hypre_global_timing_ref(threadid, size);

   for (time_index = 0; time_index < size; time_index++)
   {
      ierr += nalu_hypre_FinalizeTiming(time_index);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IncFLOPCount
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IncFLOPCount( NALU_HYPRE_BigInt inc )
{
   NALU_HYPRE_Int  ierr = 0;

   if (nalu_hypre_global_timing == NULL)
   {
      return ierr;
   }

   nalu_hypre_TimingFLOPCount += (NALU_HYPRE_Real) (inc);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BeginTiming
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BeginTiming( NALU_HYPRE_Int time_index )
{
   NALU_HYPRE_Int  ierr = 0;

   if (nalu_hypre_global_timing == NULL)
   {
      return ierr;
   }

   if (nalu_hypre_TimingState(time_index) == 0)
   {
      nalu_hypre_StopTiming();
      nalu_hypre_TimingWallTime(time_index) -= nalu_hypre_TimingWallCount;
      nalu_hypre_TimingCPUTime(time_index)  -= nalu_hypre_TimingCPUCount;
      nalu_hypre_TimingFLOPS(time_index)    -= nalu_hypre_TimingFLOPCount;

      nalu_hypre_StartTiming();
   }
   nalu_hypre_TimingState(time_index) ++;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_EndTiming
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_EndTiming( NALU_HYPRE_Int time_index )
{
   NALU_HYPRE_Int  ierr = 0;

   if (nalu_hypre_global_timing == NULL)
   {
      return ierr;
   }

   nalu_hypre_TimingState(time_index) --;
   if (nalu_hypre_TimingState(time_index) == 0)
   {
#if defined(NALU_HYPRE_USING_GPU)
      nalu_hypre_Handle *nalu_hypre_handle_ = nalu_hypre_handle();
      if (nalu_hypre_HandleDefaultExecPolicy(nalu_hypre_handle_) == NALU_HYPRE_EXEC_DEVICE)
      {
         nalu_hypre_SyncCudaDevice(nalu_hypre_handle_);
      }
#endif
      nalu_hypre_StopTiming();
      nalu_hypre_TimingWallTime(time_index) += nalu_hypre_TimingWallCount;
      nalu_hypre_TimingCPUTime(time_index)  += nalu_hypre_TimingCPUCount;
      nalu_hypre_TimingFLOPS(time_index)    += nalu_hypre_TimingFLOPCount;
      nalu_hypre_StartTiming();
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ClearTiming
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ClearTiming( )
{
   NALU_HYPRE_Int  ierr = 0;
   NALU_HYPRE_Int  i;

   if (nalu_hypre_global_timing == NULL)
   {
      return ierr;
   }

   for (i = 0; i < (nalu_hypre_global_timing_ref(threadid, size)); i++)
   {
      nalu_hypre_TimingWallTime(i) = 0.0;
      nalu_hypre_TimingCPUTime(i)  = 0.0;
      nalu_hypre_TimingFLOPS(i)    = 0.0;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PrintTiming
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PrintTiming( const char     *heading,
                   MPI_Comm        comm  )
{
   NALU_HYPRE_Int  ierr = 0;

   NALU_HYPRE_Real  local_wall_time;
   NALU_HYPRE_Real  local_cpu_time;
   NALU_HYPRE_Real  wall_time;
   NALU_HYPRE_Real  cpu_time;
   NALU_HYPRE_Real  wall_mflops;
   NALU_HYPRE_Real  cpu_mflops;

   NALU_HYPRE_Int     i;
   NALU_HYPRE_Int     myrank;

   if (nalu_hypre_global_timing == NULL)
   {
      return ierr;
   }

   nalu_hypre_MPI_Comm_rank(comm, &myrank );

   /* print heading */
   if (myrank == 0)
   {
      nalu_hypre_printf("=============================================\n");
      nalu_hypre_printf("%s:\n", heading);
      nalu_hypre_printf("=============================================\n");
   }

   for (i = 0; i < (nalu_hypre_global_timing -> size); i++)
   {
      if (nalu_hypre_TimingNumRegs(i) > 0)
      {
         local_wall_time = nalu_hypre_TimingWallTime(i);
         local_cpu_time  = nalu_hypre_TimingCPUTime(i);
         nalu_hypre_MPI_Allreduce(&local_wall_time, &wall_time, 1,
                             nalu_hypre_MPI_REAL, nalu_hypre_MPI_MAX, comm);
         nalu_hypre_MPI_Allreduce(&local_cpu_time, &cpu_time, 1,
                             nalu_hypre_MPI_REAL, nalu_hypre_MPI_MAX, comm);

         if (myrank == 0)
         {
            nalu_hypre_printf("%s:\n", nalu_hypre_TimingName(i));

            /* print wall clock info */
            nalu_hypre_printf("  wall clock time = %f seconds\n", wall_time);
            if (wall_time)
            {
               wall_mflops = nalu_hypre_TimingFLOPS(i) / wall_time / 1.0E6;
            }
            else
            {
               wall_mflops = 0.0;
            }
            nalu_hypre_printf("  wall MFLOPS     = %f\n", wall_mflops);

            /* print CPU clock info */
            nalu_hypre_printf("  cpu clock time  = %f seconds\n", cpu_time);
            if (cpu_time)
            {
               cpu_mflops = nalu_hypre_TimingFLOPS(i) / cpu_time / 1.0E6;
            }
            else
            {
               cpu_mflops = 0.0;
            }
            nalu_hypre_printf("  cpu MFLOPS      = %f\n\n", cpu_mflops);
         }
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GetTiming
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_GetTiming( const char     *heading,
                 NALU_HYPRE_Real     *wall_time_ptr,
                 MPI_Comm        comm  )
{
   NALU_HYPRE_Int  ierr = 0;

   NALU_HYPRE_Real  local_wall_time;
   NALU_HYPRE_Real  wall_time;

   NALU_HYPRE_Int     i;
   NALU_HYPRE_Int     myrank;

   if (nalu_hypre_global_timing == NULL)
   {
      return ierr;
   }

   nalu_hypre_MPI_Comm_rank(comm, &myrank );

   /* print heading */
   if (myrank == 0)
   {
      nalu_hypre_printf("=============================================\n");
      nalu_hypre_printf("%s:\n", heading);
      nalu_hypre_printf("=============================================\n");
   }

   for (i = 0; i < (nalu_hypre_global_timing -> size); i++)
   {
      if (nalu_hypre_TimingNumRegs(i) > 0)
      {
         local_wall_time = nalu_hypre_TimingWallTime(i);
         nalu_hypre_MPI_Allreduce(&local_wall_time, &wall_time, 1,
                             nalu_hypre_MPI_REAL, nalu_hypre_MPI_MAX, comm);

         if (myrank == 0)
         {
            nalu_hypre_printf("%s:\n", nalu_hypre_TimingName(i));

            /* print wall clock info */
            nalu_hypre_printf("  wall clock time = %f seconds\n", wall_time);
         }
      }
   }

   *wall_time_ptr = wall_time;
   return ierr;
}
