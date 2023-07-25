/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 1

   Interface:    Structured interface (Struct)

   Compile with: make ex1 (may need to edit NALU_HYPRE_DIR in Makefile)

   Sample run:   mpirun -np 2 ex1

   Description:  This is a two processor example.  Each processor owns one
                 box in the grid.  For reference, the two grid boxes are those
                 in the example diagram in the struct interface chapter
                 of the User's Manual. Note that in this example code, we have
                 used the two boxes shown in the diagram as belonging
                 to processor 0 (and given one box to each processor). The
                 solver is PCG with no preconditioner.

                 We recommend viewing examples 1-4 sequentially for
                 a nice overview/tutorial of the struct interface.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Struct linear solvers header */
#include "NALU_HYPRE_struct_ls.h"
#include "ex.h"

#ifdef NALU_HYPRE_EXVIS
#include "vis.c"
#endif

int main (int argc, char *argv[])
{
   int i, j, myid, num_procs;

   int vis = 0;

   NALU_HYPRE_StructGrid     grid;
   NALU_HYPRE_StructStencil  stencil;
   NALU_HYPRE_StructMatrix   A;
   NALU_HYPRE_StructVector   b;
   NALU_HYPRE_StructVector   x;
   NALU_HYPRE_StructSolver   solver;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   if (num_procs != 2)
   {
      if (myid == 0) { printf("Must run with 2 processors!\n"); }
      MPI_Finalize();

      return (0);
   }

   /* Initialize HYPRE */
   NALU_HYPRE_Initialize();

   /* Print GPU info */
   /* NALU_HYPRE_PrintDeviceInfo(); */

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-vis") == 0 )
         {
            arg_index++;
            vis = 1;
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

      if ((print_usage) && (myid == 0))
      {
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -vis : save the solution for GLVis visualization\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* 1. Set up a grid. Each processor describes the piece
      of the grid that it owns. */
   {
      /* Create an empty 2D grid object */
      NALU_HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);

      /* Add boxes to the grid */
      if (myid == 0)
      {
         int ilower[2] = {-3, 1}, iupper[2] = {-1, 2};
         NALU_HYPRE_StructGridSetExtents(grid, ilower, iupper);
      }
      else if (myid == 1)
      {
         int ilower[2] = {0, 1}, iupper[2] = {2, 4};
         NALU_HYPRE_StructGridSetExtents(grid, ilower, iupper);
      }

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      NALU_HYPRE_StructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
   {
      /* Create an empty 2D, 5-pt stencil object */
      NALU_HYPRE_StructStencilCreate(2, 5, &stencil);

      /* Define the geometry of the stencil. Each represents a
         relative offset (in the index space). */
      {
         int entry;
         int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

         /* Assign each of the 5 stencil entries */
         for (entry = 0; entry < 5; entry++)
         {
            NALU_HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
         }
      }
   }

   /* 3. Set up a Struct Matrix */
   {
      /* Create an empty matrix object */
      NALU_HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);

      /* Indicate that the matrix coefficients are ready to be set */
      NALU_HYPRE_StructMatrixInitialize(A);

      /* Set the matrix coefficients.  Each processor assigns coefficients
         for the boxes in the grid that it owns. Note that the coefficients
         associated with each stencil entry may vary from grid point to grid
         point if desired.  Here, we first set the same stencil entries for
         each grid point.  Then we make modifications to grid points near
         the boundary. */
      if (myid == 0)
      {
         int ilower[2] = {-3, 1}, iupper[2] = {-1, 2};
         int stencil_indices[5] = {0, 1, 2, 3, 4}; /* labels for the stencil entries -
                                                  these correspond to the offsets
                                                  defined above */
         int nentries = 5;
         int nvalues  = 30; /* 6 grid points, each with 5 stencil entries */
         /* double values[30]; OK to use constant-length arrays for CPUs */
         double *values = (double *) malloc(30 * sizeof(double));

         /* We have 6 grid points, each with 5 stencil entries */
         for (i = 0; i < nvalues; i += nentries)
         {
            values[i] = 4.0;
            for (j = 1; j < nentries; j++)
            {
               values[i + j] = -1.0;
            }
         }

         NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                        stencil_indices, values);

         free(values);
      }
      else if (myid == 1)
      {
         int ilower[2] = {0, 1}, iupper[2] = {2, 4};
         int stencil_indices[5] = {0, 1, 2, 3, 4};
         int nentries = 5;
         int nvalues  = 60; /* 12 grid points, each with 5 stencil entries */
         /* double values[60]; OK to use constant-length arrays for CPUs */
         double *values = (double *) malloc(60 * sizeof(double));

         for (i = 0; i < nvalues; i += nentries)
         {
            values[i] = 4.0;
            for (j = 1; j < nentries; j++)
            {
               values[i + j] = -1.0;
            }
         }

         NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                        stencil_indices, values);

         free(values);
      }

      /* Set the coefficients reaching outside of the boundary to 0 */
      if (myid == 0)
      {
         /* double values[3]; OK to use constant-length arrays for CPUs */
         double *values = (double *) malloc(3 * sizeof(double));
         for (i = 0; i < 3; i++)
         {
            values[i] = 0.0;
         }
         {
            /* values below our box */
            int ilower[2] = {-3, 1}, iupper[2] = {-1, 1};
            int stencil_indices[1] = {3};
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }
         {
            /* values to the left of our box */
            int ilower[2] = {-3, 1}, iupper[2] = {-3, 2};
            int stencil_indices[1] = {1};
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }
         {
            /* values above our box */
            int ilower[2] = {-3, 2}, iupper[2] = {-1, 2};
            int stencil_indices[1] = {4};
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }
         free(values);
      }
      else if (myid == 1)
      {
         /* double values[4]; OK to use constant-length arrays for CPUs */
         double *values = (double *) malloc(4 * sizeof(double));
         for (i = 0; i < 4; i++)
         {
            values[i] = 0.0;
         }
         {
            /* values below our box */
            int ilower[2] = {0, 1}, iupper[2] = {2, 1};
            int stencil_indices[1] = {3};
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }
         {
            /* values to the right of our box */
            int ilower[2] = {2, 1}, iupper[2] = {2, 4};
            int stencil_indices[1] = {2};
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }
         {
            /* values above our box */
            int ilower[2] = {0, 4}, iupper[2] = {2, 4};
            int stencil_indices[1] = {4};
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }
         {
            /* values to the left of our box
               (that do not border the other box on proc. 0) */
            int ilower[2] = {0, 3}, iupper[2] = {0, 4};
            int stencil_indices[1] = {1};
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
         }
         free(values);
      }

      /* This is a collective call finalizing the matrix assembly.
         The matrix is now ``ready to be used'' */
      NALU_HYPRE_StructMatrixAssemble(A);
   }

   /* 4. Set up Struct Vectors for b and x.  Each processor sets the vectors
      corresponding to its boxes. */
   {
      /* Create an empty vector object */
      NALU_HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
      NALU_HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* Indicate that the vector coefficients are ready to be set */
      NALU_HYPRE_StructVectorInitialize(b);
      NALU_HYPRE_StructVectorInitialize(x);

      /* Set the vector coefficients */
      if (myid == 0)
      {
         int ilower[2] = {-3, 1}, iupper[2] = {-1, 2};
         /* double values[6]; OK to use constant-length arrays for CPUs */
         double *values = (double *) malloc(6 * sizeof(double)); /* 6 grid points */

         for (i = 0; i < 6; i ++)
         {
            values[i] = 1.0;
         }
         NALU_HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

         for (i = 0; i < 6; i ++)
         {
            values[i] = 0.0;
         }
         NALU_HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
         free(values);
      }
      else if (myid == 1)
      {
         int ilower[2] = {0, 1}, iupper[2] = {2, 4};
         /* double values[12]; OK to use constant-length arrays for CPUs */
         double *values = (double *) malloc(12 * sizeof(double)); /* 12 grid points */

         for (i = 0; i < 12; i ++)
         {
            values[i] = 1.0;
         }
         NALU_HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

         for (i = 0; i < 12; i ++)
         {
            values[i] = 0.0;
         }
         NALU_HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
         free(values);
      }

      /* This is a collective call finalizing the vector assembly.
         The vectors are now ``ready to be used'' */
      NALU_HYPRE_StructVectorAssemble(b);
      NALU_HYPRE_StructVectorAssemble(x);
   }

   /* 5. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   {
      /* Create an empty PCG Struct solver */
      NALU_HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters */
      NALU_HYPRE_StructPCGSetTol(solver, 1.0e-06); /* convergence tolerance */
      NALU_HYPRE_StructPCGSetPrintLevel(solver, 2); /* amount of info. printed */

      /* Setup and solve */
      NALU_HYPRE_StructPCGSetup(solver, A, b, x);
      NALU_HYPRE_StructPCGSolve(solver, A, b, x);
   }

   /* Save the solution for GLVis visualization, see vis/glvis-ex1.sh */
   if (vis)
   {
#ifdef NALU_HYPRE_EXVIS
      GLVis_PrintStructGrid(grid, "vis/ex1.mesh", myid, NULL, NULL);
      GLVis_PrintStructVector(x, "vis/ex1.sol", myid);
      GLVis_PrintData("vis/ex1.data", myid, num_procs);
#endif
   }

   /* Free memory */
   NALU_HYPRE_StructGridDestroy(grid);
   NALU_HYPRE_StructStencilDestroy(stencil);
   NALU_HYPRE_StructMatrixDestroy(A);
   NALU_HYPRE_StructVectorDestroy(b);
   NALU_HYPRE_StructVectorDestroy(x);
   NALU_HYPRE_StructPCGDestroy(solver);

   /* Finalize Hypre */
   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}
