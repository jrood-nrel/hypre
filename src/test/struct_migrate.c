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
#include "NALU_HYPRE_struct_mv.h"
/* RDF: This include is only needed for AddValuesVector() */
#include "_nalu_hypre_struct_mv.h"

NALU_HYPRE_Int AddValuesVector( nalu_hypre_StructGrid   *grid,
                           nalu_hypre_StructVector *vector,
                           NALU_HYPRE_Real          value );

/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 *--------------------------------------------------------------------------*/

/*----------------------------------------------------------------------
 * Standard 7-point laplacian in 3D with grid and anisotropy determined
 * as command line arguments.  Do `driver -help' for usage info.
 *----------------------------------------------------------------------*/

nalu_hypre_int
main( nalu_hypre_int argc,
      char *argv[] )
{
   NALU_HYPRE_Int           arg_index;
   NALU_HYPRE_Int           print_usage;
   NALU_HYPRE_Int           nx, ny, nz;
   NALU_HYPRE_Int           P, Q, R;
   NALU_HYPRE_Int           bx, by, bz;

   NALU_HYPRE_StructGrid    from_grid, to_grid;
   NALU_HYPRE_StructVector  from_vector, to_vector, check_vector;
   NALU_HYPRE_CommPkg       comm_pkg;

   NALU_HYPRE_Int           time_index;
   NALU_HYPRE_Int           num_procs, myid;

   NALU_HYPRE_Int           p, q, r;
   NALU_HYPRE_Int           dim;
   NALU_HYPRE_Int           nblocks ;
   NALU_HYPRE_Int         **ilower, **iupper, **iupper2;
   NALU_HYPRE_Int           istart[3];
   NALU_HYPRE_Int           i, ix, iy, iz, ib;
   NALU_HYPRE_Int           print_system = 0;

   NALU_HYPRE_Real          check;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before NALU_HYPRE_Init() and should not be changed after
    *-----------------------------------------------------------------*/
   nalu_hypre_bind_device(myid, num_procs, nalu_hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   NALU_HYPRE_Init();

#if defined(NALU_HYPRE_USING_KOKKOS)
   Kokkos::initialize (argc, argv);
#endif

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   dim = 3;

   nx = 2;
   ny = 2;
   nz = 2;

   P  = num_procs;
   Q  = 1;
   R  = 1;

   bx = 1;
   by = 1;
   bz = 1;

   istart[0] = 1;
   istart[1] = 1;
   istart[2] = 1;

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
      else if ( strcmp(argv[arg_index], "-istart") == 0 )
      {
         arg_index++;
         istart[0] = atoi(argv[arg_index++]);
         istart[1] = atoi(argv[arg_index++]);
         istart[2] = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         bx = atoi(argv[arg_index++]);
         by = atoi(argv[arg_index++]);
         bz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-d") == 0 )
      {
         arg_index++;
         dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
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
      nalu_hypre_printf("  -istart <ix> <iy> <iz> : start of box\n");
      nalu_hypre_printf("  -P <Px> <Py> <Pz>   : processor topology\n");
      nalu_hypre_printf("  -b <bx> <by> <bz>   : blocking per processor\n");
      nalu_hypre_printf("  -d <dim>            : problem dimension (2 or 3)\n");
      nalu_hypre_printf("  -print              : print vectors\n");
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
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("Running with these driver parameters:\n");
      nalu_hypre_printf("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      nalu_hypre_printf("  (ix, iy, iz)    = (%d, %d, %d)\n",
                   istart[0], istart[1], istart[2]);
      nalu_hypre_printf("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      nalu_hypre_printf("  (bx, by, bz)    = (%d, %d, %d)\n", bx, by, bz);
      nalu_hypre_printf("  dim             = %d\n", dim);
   }

   /*-----------------------------------------------------------
    * Set up the stencil structure (7 points) when matrix is NOT read from file
    * Set up the grid structure used when NO files are read
    *-----------------------------------------------------------*/

   switch (dim)
   {
      case 1:
         nblocks = bx;
         p = myid % P;
         break;

      case 2:
         nblocks = bx * by;
         p = myid % P;
         q = (( myid - p) / P) % Q;
         break;

      case 3:
         nblocks = bx * by * bz;
         p = myid % P;
         q = (( myid - p) / P) % Q;
         r = ( myid - p - P * q) / ( P * Q );
         break;
   }

   if (myid >= (P * Q * R))
   {
      /* My processor has no data on it */
      nblocks = bx = by = bz = 0;
   }

   /*-----------------------------------------------------------
    * prepare space for the extents
    *-----------------------------------------------------------*/

   ilower = nalu_hypre_CTAlloc(NALU_HYPRE_Int*,  nblocks, NALU_HYPRE_MEMORY_HOST);
   iupper = nalu_hypre_CTAlloc(NALU_HYPRE_Int*,  nblocks, NALU_HYPRE_MEMORY_HOST);
   iupper2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int*,  nblocks, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nblocks; i++)
   {
      ilower[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  dim, NALU_HYPRE_MEMORY_HOST);
      iupper[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  dim, NALU_HYPRE_MEMORY_HOST);
      iupper2[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  dim, NALU_HYPRE_MEMORY_HOST);
   }

   ib = 0;
   switch (dim)
   {
      case 1:
         for (ix = 0; ix < bx; ix++)
         {
            ilower[ib][0] = istart[0] + nx * (bx * p + ix);
            iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
            iupper2[ib][0] = iupper[ib][0];
            if ( (ix == (bx - 1)) && (p < (P - 1)) )
            {
               iupper2[ib][0] = iupper[ib][0] + 1;
            }
            ib++;
         }
         break;
      case 2:
         for (iy = 0; iy < by; iy++)
            for (ix = 0; ix < bx; ix++)
            {
               ilower[ib][0] = istart[0] + nx * (bx * p + ix);
               iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
               ilower[ib][1] = istart[1] + ny * (by * q + iy);
               iupper[ib][1] = istart[1] + ny * (by * q + iy + 1) - 1;
               iupper2[ib][0] = iupper[ib][0];
               iupper2[ib][1] = iupper[ib][1];
               if ( (ix == (bx - 1)) && (p < (P - 1)) )
               {
                  iupper2[ib][0] = iupper[ib][0] + 1;
               }
               if ( (iy == (by - 1)) && (q < (Q - 1)) )
               {
                  iupper2[ib][1] = iupper[ib][1] + 1;
               }
               ib++;
            }
         break;
      case 3:
         for (iz = 0; iz < bz; iz++)
            for (iy = 0; iy < by; iy++)
               for (ix = 0; ix < bx; ix++)
               {
                  ilower[ib][0] = istart[0] + nx * (bx * p + ix);
                  iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
                  ilower[ib][1] = istart[1] + ny * (by * q + iy);
                  iupper[ib][1] = istart[1] + ny * (by * q + iy + 1) - 1;
                  ilower[ib][2] = istart[2] + nz * (bz * r + iz);
                  iupper[ib][2] = istart[2] + nz * (bz * r + iz + 1) - 1;
                  iupper2[ib][0] = iupper[ib][0];
                  iupper2[ib][1] = iupper[ib][1];
                  iupper2[ib][2] = iupper[ib][2];
                  if ( (ix == (bx - 1)) && (p < (P - 1)) )
                  {
                     iupper2[ib][0] = iupper[ib][0] + 1;
                  }
                  if ( (iy == (by - 1)) && (q < (Q - 1)) )
                  {
                     iupper2[ib][1] = iupper[ib][1] + 1;
                  }
                  if ( (iz == (bz - 1)) && (r < (R - 1)) )
                  {
                     iupper2[ib][2] = iupper[ib][2] + 1;
                  }
                  ib++;
               }
         break;
   }

   NALU_HYPRE_StructGridCreate(nalu_hypre_MPI_COMM_WORLD, dim, &from_grid);
   NALU_HYPRE_StructGridCreate(nalu_hypre_MPI_COMM_WORLD, dim, &to_grid);
   for (ib = 0; ib < nblocks; ib++)
   {
      NALU_HYPRE_StructGridSetExtents(from_grid, ilower[ib], iupper[ib]);
      NALU_HYPRE_StructGridSetExtents(to_grid, ilower[ib], iupper2[ib]);
   }
   NALU_HYPRE_StructGridAssemble(from_grid);
   NALU_HYPRE_StructGridAssemble(to_grid);

   /*-----------------------------------------------------------
    * Set up the vectors
    *-----------------------------------------------------------*/

   NALU_HYPRE_StructVectorCreate(nalu_hypre_MPI_COMM_WORLD, from_grid, &from_vector);
   NALU_HYPRE_StructVectorInitialize(from_vector);
   AddValuesVector(from_grid, from_vector, 1.0);
   NALU_HYPRE_StructVectorAssemble(from_vector);

   NALU_HYPRE_StructVectorCreate(nalu_hypre_MPI_COMM_WORLD, to_grid, &to_vector);
   NALU_HYPRE_StructVectorInitialize(to_vector);
   AddValuesVector(to_grid, to_vector, 0.0);
   NALU_HYPRE_StructVectorAssemble(to_vector);

   /* Vector used to check the migration */
   NALU_HYPRE_StructVectorCreate(nalu_hypre_MPI_COMM_WORLD, to_grid, &check_vector);
   NALU_HYPRE_StructVectorInitialize(check_vector);
   AddValuesVector(to_grid, check_vector, 1.0);
   NALU_HYPRE_StructVectorAssemble(check_vector);

   /*-----------------------------------------------------------
    * Migrate
    *-----------------------------------------------------------*/

   time_index = nalu_hypre_InitializeTiming("Struct Migrate");
   nalu_hypre_BeginTiming(time_index);

   NALU_HYPRE_StructVectorGetMigrateCommPkg(from_vector, to_vector, &comm_pkg);
   NALU_HYPRE_StructVectorMigrate(comm_pkg, from_vector, to_vector);
   NALU_HYPRE_CommPkgDestroy(comm_pkg);

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Struct Migrate", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);

   /*-----------------------------------------------------------
    * Check the migration and print the result
    *-----------------------------------------------------------*/

   nalu_hypre_StructAxpy(-1.0, to_vector, check_vector);
   check = nalu_hypre_StructInnerProd (check_vector, check_vector);

   if (myid == 0)
   {
      nalu_hypre_printf("\nCheck = %1.0f (success = 0)\n\n", check);
   }

   /*-----------------------------------------------------------
    * Print out the vectors
    *-----------------------------------------------------------*/

   if (print_system)
   {
      NALU_HYPRE_StructVectorPrint("struct_migrate.out.xfr", from_vector, 0);
      NALU_HYPRE_StructVectorPrint("struct_migrate.out.xto", to_vector, 0);
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   NALU_HYPRE_StructGridDestroy(from_grid);
   NALU_HYPRE_StructGridDestroy(to_grid);

   for (i = 0; i < nblocks; i++)
   {
      nalu_hypre_TFree(ilower[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(iupper[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(iupper2[i], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(ilower, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(iupper, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(iupper2, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_StructVectorDestroy(from_vector);
   NALU_HYPRE_StructVectorDestroy(to_vector);
   NALU_HYPRE_StructVectorDestroy(check_vector);

#if defined(NALU_HYPRE_USING_KOKKOS)
   Kokkos::finalize ();
#endif

   /* Finalize Hypre */
   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   return (0);
}

/*-------------------------------------------------------------------------
 * Add constant values to a vector.
 *-------------------------------------------------------------------------*/

NALU_HYPRE_Int
AddValuesVector( nalu_hypre_StructGrid   *grid,
                 nalu_hypre_StructVector *vector,
                 NALU_HYPRE_Real          value )
{
   NALU_HYPRE_Int          i, ierr = 0;
   nalu_hypre_BoxArray    *gridboxes;
   NALU_HYPRE_Int          ib;
   nalu_hypre_IndexRef     ilower;
   nalu_hypre_IndexRef     iupper;
   nalu_hypre_Box         *box;
   NALU_HYPRE_Real        *values;
   NALU_HYPRE_Real        *values_h;
   NALU_HYPRE_Int          volume;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_StructVectorMemoryLocation(vector);

   gridboxes = nalu_hypre_StructGridBoxes(grid);

   ib = 0;
   nalu_hypre_ForBoxI(ib, gridboxes)
   {
      box    = nalu_hypre_BoxArrayBox(gridboxes, ib);
      volume = nalu_hypre_BoxVolume(box);
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, volume, memory_location);
      values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, volume, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < volume; i++)
      {
         values_h[i] = value;
      }

      nalu_hypre_TMemcpy(values, values_h, NALU_HYPRE_Real, volume, memory_location, NALU_HYPRE_MEMORY_HOST);

      ilower = nalu_hypre_BoxIMin(box);
      iupper = nalu_hypre_BoxIMax(box);
      NALU_HYPRE_StructVectorSetBoxValues(vector, ilower, iupper, values);
      nalu_hypre_TFree(values, memory_location);
      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}

