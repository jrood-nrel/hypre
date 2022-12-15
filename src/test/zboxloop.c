/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_nalu_hypre_utilities.h"
#include "NALU_HYPRE_struct_ls.h"
#include "NALU_HYPRE_krylov.h"

#include "_nalu_hypre_struct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * Test driver to time new boxloops and compare to the old ones
 *--------------------------------------------------------------------------*/

nalu_hypre_int
main( nalu_hypre_int argc,
      char *argv[] )
{
   NALU_HYPRE_Int         arg_index;
   NALU_HYPRE_Int         print_usage;
   NALU_HYPRE_Int         nx, ny, nz;
   NALU_HYPRE_Int         P, Q, R;
   NALU_HYPRE_Int         time_index;
   NALU_HYPRE_Int         num_procs, myid;
   NALU_HYPRE_Int         dim;
   NALU_HYPRE_Int         rep, reps, fail, sum;
   NALU_HYPRE_Int         size;
   nalu_hypre_Box        *x1_data_box, *x2_data_box, *x3_data_box, *x4_data_box;
   //NALU_HYPRE_Int         xi1, xi2, xi3, xi4;
   NALU_HYPRE_Int         xi1;
   NALU_HYPRE_Real       *xp1, *xp2, *xp3, *xp4;
   NALU_HYPRE_Real       *d_xp1, *d_xp2, *d_xp3, *d_xp4;
   nalu_hypre_Index       loop_size, start, unit_stride, index;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   NALU_HYPRE_Init();

#if defined(NALU_HYPRE_USING_KOKKOS)
   Kokkos::initialize (argc, argv);
#endif

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   dim = 3;

   nx = 10;
   ny = 10;
   nz = 10;

   P  = num_procs;
   Q  = 1;
   R  = 1;

   reps = -1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-d") == 0 )
      {
         arg_index++;
         dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-reps") == 0 )
      {
         arg_index++;
         reps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
         break;
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/

   if ( (print_usage) && (myid == 0) )
   {
      nalu_hypre_printf("\n");
      nalu_hypre_printf("Usage: %s [<options>]\n", argv[0]);
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -n <nx> <ny> <nz>   : problem size per block\n");
      nalu_hypre_printf("  -P <Px> <Py> <Pz>   : processor topology\n");
      nalu_hypre_printf("  -d <dim>            : problem dimension (2 or 3)\n");
      nalu_hypre_printf("\n");
   }

   if ( print_usage )
   {
      exit(1);
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) > num_procs)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("Error: PxQxR is more than the number of processors\n");
      }
      exit(1);
   }
   else if ((P * Q * R) < num_procs)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("Warning: PxQxR is less than the number of processors\n");
      }
   }

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_SetIndex3(start, 1, 1, 1);
   nalu_hypre_SetIndex3(loop_size, nx, ny, nz);
   nalu_hypre_SetIndex3(unit_stride, 1, 1, 1);

   x1_data_box = nalu_hypre_BoxCreate(dim);
   x2_data_box = nalu_hypre_BoxCreate(dim);
   x3_data_box = nalu_hypre_BoxCreate(dim);
   x4_data_box = nalu_hypre_BoxCreate(dim);
   nalu_hypre_SetIndex3(nalu_hypre_BoxIMin(x1_data_box), 0, 0, 0);
   nalu_hypre_SetIndex3(nalu_hypre_BoxIMax(x1_data_box), nx + 1, ny + 1, nz + 1);
   nalu_hypre_CopyBox(x1_data_box, x2_data_box);
   nalu_hypre_CopyBox(x1_data_box, x3_data_box);
   nalu_hypre_CopyBox(x1_data_box, x4_data_box);

   size = (nx + 2) * (ny + 2) * (nz + 2);
   xp1 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_HOST);
   xp2 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_HOST);
   xp3 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_HOST);
   xp4 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_HOST);

   d_xp1 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_DEVICE);
   d_xp2 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_DEVICE);
   d_xp3 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_DEVICE);
   d_xp4 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_DEVICE);

   if (reps < 0)
   {
      reps = 1000000000 / (nx * ny * nz + 1000);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("Running with these driver parameters:\n");
      nalu_hypre_printf("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      nalu_hypre_printf("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      nalu_hypre_printf("  dim             = %d\n", dim);
      nalu_hypre_printf("  reps            = %d\n", reps);
   }

   /*-----------------------------------------------------------
    * Check new boxloops
    *-----------------------------------------------------------*/

   /* xp1 is already initialized to 0 */

   zypre_BoxLoop1Begin(dim, loop_size,
                       x1_data_box, start, unit_stride, xi1);
   {
      xp1[xi1] ++;
   }
   zypre_BoxLoop1End(xi1);

   /* Use old boxloop to check that values are set to 1 */
   fail = 0;
   sum = 0;
   nalu_hypre_SerialBoxLoop1Begin(3, loop_size,
                             x1_data_box, start, unit_stride, xi1);
   {
      sum += xp1[xi1];
      if (xp1[xi1] != 1)
      {
         zypre_BoxLoopGetIndex(index);
         nalu_hypre_printf("*(%d,%d,%d) = %d\n",
                      index[0], index[1], index[2], (NALU_HYPRE_Int) xp1[xi1]);
         fail = 1;
      }
   }
   nalu_hypre_SerialBoxLoop1End(xi1);

   if (sum != (nx * ny * nz))
   {
      nalu_hypre_printf("*sum = %d\n", sum);
      fail = 1;
   }
   if (fail)
   {
      exit(1);
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Barrier(nalu_hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Time (device) boxloops
    *-----------------------------------------------------------*/

   /* Time BoxLoop0 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop0");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      xi1 = 0;
#define DEVICE_VAR is_device_ptr(d_xp1)
      nalu_hypre_BoxLoop0Begin(3, loop_size);
      {
         d_xp1[xi1] += d_xp1[xi1];
         //xi1++;
      }
      nalu_hypre_BoxLoop0End();
#undef DEVICE_VAR
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop1 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop1");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
#define DEVICE_VAR is_device_ptr(d_xp1)
      nalu_hypre_BoxLoop1Begin(3, loop_size,
                          x1_data_box, start, unit_stride, xi1);
      {
         d_xp1[xi1] += d_xp1[xi1];
      }
      nalu_hypre_BoxLoop1End(xi1);
#undef DEVICE_VAR
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop2 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop2");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
#define DEVICE_VAR is_device_ptr(d_xp1,d_xp2)
      nalu_hypre_BoxLoop2Begin(3, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2);
      {
         d_xp1[xi1] += d_xp1[xi1] + d_xp2[xi2];
      }
      nalu_hypre_BoxLoop2End(xi1, xi2);
#undef DEVICE_VAR
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop3 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop3");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
#define DEVICE_VAR is_device_ptr(d_xp1,d_xp2,d_xp3)
      nalu_hypre_BoxLoop3Begin(3, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2,
                          x3_data_box, start, unit_stride, xi3);
      {
         d_xp1[xi1] += d_xp1[xi1] + d_xp2[xi2] + d_xp3[xi3];
      }
      nalu_hypre_BoxLoop3End(xi1, xi2, xi3);
#undef DEVICE_VAR
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop4 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop4");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
#define DEVICE_VAR is_device_ptr(d_xp1,d_xp2,d_xp3,d_xp4)
      nalu_hypre_BoxLoop4Begin(3, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2,
                          x3_data_box, start, unit_stride, xi3,
                          x4_data_box, start, unit_stride, xi4);
      {
         d_xp1[xi1] += d_xp1[xi1] + d_xp2[xi2] + d_xp3[xi3] + d_xp4[xi4];
      }
      nalu_hypre_BoxLoop4End(xi1, xi2, xi3, xi4);
#undef DEVICE_VAR
   }
   nalu_hypre_EndTiming(time_index);

   nalu_hypre_PrintTiming("BoxLoop times [DEVICE]", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeAllTimings();
   nalu_hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Time host boxloops
    *-----------------------------------------------------------*/

   /* Time BoxLoop0 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop0");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      xi1 = 0;
#undef NALU_HYPRE_OMP_CLAUSE
#define NALU_HYPRE_OMP_CLAUSE firstprivate(xi1)
      zypre_BoxLoop0Begin(dim, loop_size);
      {
         xp1[xi1] += xp1[xi1];
         xi1++;
      }
      zypre_BoxLoop0End();
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop1 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop1");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      zypre_BoxLoop1Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1);
      {
         xp1[xi1] += xp1[xi1];
      }
      zypre_BoxLoop1End(xi1);
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop2 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop2");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      zypre_BoxLoop2Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2);
      {
         xp1[xi1] += xp1[xi1] + xp2[xi2];
      }
      zypre_BoxLoop2End(xi1, xi2);
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop3 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop3");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      zypre_BoxLoop3Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2,
                          x3_data_box, start, unit_stride, xi3);
      {
         xp1[xi1] += xp1[xi1] + xp2[xi2] + xp3[xi3];
      }
      zypre_BoxLoop3End(xi1, xi2, xi3);
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop4 */
   time_index = nalu_hypre_InitializeTiming("BoxLoop4");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      zypre_BoxLoop4Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2,
                          x3_data_box, start, unit_stride, xi3,
                          x4_data_box, start, unit_stride, xi4);
      {
         xp1[xi1] += xp1[xi1] + xp2[xi2] + xp3[xi3] + xp4[xi4];
      }
      zypre_BoxLoop4End(xi1, xi2, xi3, xi4);
   }
   nalu_hypre_EndTiming(time_index);

   nalu_hypre_PrintTiming("BoxLoop times [HOST]", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeAllTimings();
   nalu_hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Reduction Loops
    *-----------------------------------------------------------*/
   {
      NALU_HYPRE_Int i;
      for (i = 0; i < size; i++)
      {
         xp1[i] = cos(i + 1.0);
         xp2[i] = sin(i + 2.0);
      }
      nalu_hypre_TMemcpy(d_xp1, xp1, NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(d_xp2, xp2, NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
   }

#if defined(NALU_HYPRE_USING_KOKKOS)
   NALU_HYPRE_Real reducer = 0.0;
#elif defined(NALU_HYPRE_USING_RAJA)
   ReduceSum<nalu_hypre_raja_reduce_policy, NALU_HYPRE_Real> reducer(0.0);
#elif defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   ReduceSum<NALU_HYPRE_Real> reducer(0.0);
#else
   NALU_HYPRE_Real reducer = 0.0;
#endif
   NALU_HYPRE_Real box_sum1 = 0.0, box_sum2 = 0.0;

#undef NALU_HYPRE_BOX_REDUCTION
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#define NALU_HYPRE_BOX_REDUCTION map(tofrom:reducer) reduction(+:reducer)
#else
#define NALU_HYPRE_BOX_REDUCTION reduction(+:reducer)
#endif

   /*-----------------------------------------------------------
    * Time (device) boxloops
    *-----------------------------------------------------------*/

   /* Time BoxLoop1Reduction */
   time_index = nalu_hypre_InitializeTiming("BoxLoopReduction1");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      reducer = 0.0;
#define DEVICE_VAR is_device_ptr(d_xp1)
      nalu_hypre_BoxLoop1ReductionBegin(3, loop_size,
                                   x1_data_box, start, unit_stride, xi1,
                                   reducer);
      {
         reducer += 1.0 / d_xp1[xi1];
      }
      nalu_hypre_BoxLoop1ReductionEnd(xi1, reducer);
#undef DEVICE_VAR
      box_sum1 += (NALU_HYPRE_Real) reducer;
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop2Reduction */
   time_index = nalu_hypre_InitializeTiming("BoxLoopReduction2");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      reducer = 0.0;
#define DEVICE_VAR is_device_ptr(d_xp1,d_xp2)
      nalu_hypre_BoxLoop2ReductionBegin(3, loop_size,
                                   x1_data_box, start, unit_stride, xi1,
                                   x2_data_box, start, unit_stride, xi2,
                                   reducer);
      {
         reducer += 1.0 / d_xp1[xi1] + d_xp2[xi2] * 3.1415926;
      }
      nalu_hypre_BoxLoop2ReductionEnd(xi1, xi2, reducer);
#undef DEVICE_VAR
      box_sum2 += (NALU_HYPRE_Real) reducer;
   }
   nalu_hypre_EndTiming(time_index);

   nalu_hypre_PrintTiming("BoxLoopReduction times [DEVICE]", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeAllTimings();
   nalu_hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Time host boxloops
    *-----------------------------------------------------------*/
   NALU_HYPRE_Real zbox_sum1 = 0.0, zbox_sum2 = 0.0;

   /* Time BoxLoop1 */
   time_index = nalu_hypre_InitializeTiming("BoxLoopReduction1");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
#undef NALU_HYPRE_BOX_REDUCTION
#define NALU_HYPRE_BOX_REDUCTION reduction(+:zbox_sum1)
      zypre_BoxLoop1Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1);
      {
         zbox_sum1 += 1.0 / xp1[xi1];
      }
      zypre_BoxLoop1End(xi1);
   }
   nalu_hypre_EndTiming(time_index);

   /* Time BoxLoop2 */
   time_index = nalu_hypre_InitializeTiming("BoxLoopReduction2");
   nalu_hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
#undef NALU_HYPRE_BOX_REDUCTION
#define NALU_HYPRE_BOX_REDUCTION reduction(+:zbox_sum2)
      zypre_BoxLoop2Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2);
      {
         zbox_sum2 += 1.0 / xp1[xi1] + xp2[xi2] * 3.1415926;
      }
      zypre_BoxLoop2End(xi1, xi2);
   }
   nalu_hypre_EndTiming(time_index);

   nalu_hypre_PrintTiming("BoxLoopReduction times [HOST]", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeAllTimings();
   nalu_hypre_ClearTiming();

   nalu_hypre_printf("BoxLoopReduction1, error %e\n", nalu_hypre_abs((zbox_sum1 - box_sum1) / zbox_sum1));
   nalu_hypre_printf("BoxLoopReduction2, error %e\n", nalu_hypre_abs((zbox_sum2 - box_sum2) / zbox_sum2));

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   nalu_hypre_BoxDestroy(x1_data_box);
   nalu_hypre_BoxDestroy(x2_data_box);
   nalu_hypre_BoxDestroy(x3_data_box);
   nalu_hypre_BoxDestroy(x4_data_box);

   nalu_hypre_TFree(xp1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(xp2, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(xp3, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(xp4, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(d_xp1, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_xp2, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_xp3, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_xp4, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_KOKKOS)
   Kokkos::finalize ();
#endif

   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   return (0);
}

