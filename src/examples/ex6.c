/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 6

   Interface:    Semi-Structured interface (SStruct)

   Compile with: make ex6

   Sample run:   mpirun -np 2 ex6

   Description:  This is a two processor example and is the same problem
                 as is solved with the structured interface in Example 2.
                 (The grid boxes are exactly those in the example
                 diagram in the struct interface chapter of the User's Manual.
                 Processor 0 owns two boxes and processor 1 owns one box.)

                 This is the simplest sstruct example, and it demonstrates how
                 the semi-structured interface can be used for structured problems.
                 There is one part and one variable.  The solver is PCG with SMG
                 preconditioner. We use a structured solver for this example.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* SStruct linear solvers headers */
#include "NALU_HYPRE_sstruct_ls.h"
#include "ex.h"

#ifdef NALU_HYPRE_EXVIS
#include "vis.c"
#endif

int main (int argc, char *argv[])
{
   int myid, num_procs;

   int vis = 0;

   NALU_HYPRE_SStructGrid     grid;
   NALU_HYPRE_SStructGraph    graph;
   NALU_HYPRE_SStructStencil  stencil;
   NALU_HYPRE_SStructMatrix   A;
   NALU_HYPRE_SStructVector   b;
   NALU_HYPRE_SStructVector   x;

   /* We are using struct solvers for this example */
   NALU_HYPRE_StructSolver solver;
   NALU_HYPRE_StructSolver precond;

   int object_type;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   NALU_HYPRE_Init();

   /* Print GPU info */
   /* NALU_HYPRE_PrintDeviceInfo(); */

   if (num_procs != 2)
   {
      if (myid == 0) { printf("Must run with 2 processors!\n"); }
      MPI_Finalize();

      return (0);
   }

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

   /* 1. Set up the 2D grid.  This gives the index space in each part.
      Here we only use one part and one variable. (So the part id is 0
      and the variable id is 0) */
   {
      int ndim = 2;
      int nparts = 1;
      int part = 0;

      /* Create an empty 2D grid object */
      NALU_HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &grid);

      /* Set the extents of the grid - each processor sets its grid
         boxes.  Each part has its own relative index space numbering,
         but in this example all boxes belong to the same part. */

      /* Processor 0 owns two boxes in the grid. */
      if (myid == 0)
      {
         /* Add a new box to the grid */
         {
            int ilower[2] = {-3, 1};
            int iupper[2] = {-1, 2};

            NALU_HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
         }

         /* Add a new box to the grid */
         {
            int ilower[2] = {0, 1};
            int iupper[2] = {2, 4};

            NALU_HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
         }
      }

      /* Processor 1 owns one box in the grid. */
      else if (myid == 1)
      {
         /* Add a new box to the grid */
         {
            int ilower[2] = {3, 1};
            int iupper[2] = {6, 4};

            NALU_HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
         }
      }

      /* Set the variable type and number of variables on each part. */
      {
         int i;
         int nvars = 1;
         NALU_HYPRE_SStructVariable vartypes[1] = {NALU_HYPRE_SSTRUCT_VARIABLE_CELL};

         for (i = 0; i < nparts; i++)
         {
            NALU_HYPRE_SStructGridSetVariables(grid, i, nvars, vartypes);
         }
      }

      /* Now the grid is ready to use */
      NALU_HYPRE_SStructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil(s) */
   {
      /* Create an empty 2D, 5-pt stencil object */
      NALU_HYPRE_SStructStencilCreate(2, 5, &stencil);

      /* Define the geometry of the stencil. Each represents a
         relative offset (in the index space). */
      {
         int entry;
         int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
         int var = 0;

         /* Assign numerical values to the offsets so that we can
            easily refer to them  - the last argument indicates the
            variable for which we are assigning this stencil - we are
            just using one variable in this example so it is the first one (0) */
         for (entry = 0; entry < 5; entry++)
         {
            NALU_HYPRE_SStructStencilSetEntry(stencil, entry, offsets[entry], var);
         }
      }
   }

   /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix and allows non-stencil relationships between the parts */
   {
      int var = 0;
      int part = 0;

      /* Create the graph object */
      NALU_HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

      /* See MatrixSetObjectType below */
      object_type = NALU_HYPRE_STRUCT;
      NALU_HYPRE_SStructGraphSetObjectType(graph, object_type);

      /* Now we need to tell the graph which stencil to use for each
         variable on each part (we only have one variable and one part) */
      NALU_HYPRE_SStructGraphSetStencil(graph, part, var, stencil);

      /* Here we could establish connections between parts if we
         had more than one part using the graph. For example, we could
         use NALU_HYPRE_GraphAddEntries() routine or NALU_HYPRE_GridSetNeighborBox() */

      /* Assemble the graph */
      NALU_HYPRE_SStructGraphAssemble(graph);
   }

   /* 4. Set up a SStruct Matrix */
   {
      int i, j;
      int part = 0;
      int var = 0;

      /* Create the empty matrix object */
      NALU_HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);

      /* Set the object type (by default NALU_HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  If you want to use unstructured
         solvers, e.g. BoomerAMG, the object type should be NALU_HYPRE_PARCSR.
         If the problem is purely structured (with one part), you may want to use
         NALU_HYPRE_STRUCT to access the structured solvers. Here we have a purely
         structured example. */
      object_type = NALU_HYPRE_STRUCT;
      NALU_HYPRE_SStructMatrixSetObjectType(A, object_type);

      /* Get ready to set values */
      NALU_HYPRE_SStructMatrixInitialize(A);

      /* Each processor must set the stencil values for their boxes on each part.
         In this example, we only set stencil entries and therefore use
         NALU_HYPRE_SStructMatrixSetBoxValues.  If we need to set non-stencil entries,
         we have to use NALU_HYPRE_SStructMatrixSetValues (shown in a later example). */

      if (myid == 0)
      {
         /* Set the matrix coefficients for some set of stencil entries
            over all the gridpoints in my first box (account for boundary
            grid points later) */
         {
            int ilower[2] = {-3, 1};
            int iupper[2] = {-1, 2};

            int nentries = 5;
            int nvalues  = 30; /* 6 grid points, each with 5 stencil entries */
            /* double values[30]; OK to use constant-length array for CPUs */
            double *values = (double *) malloc(30 * sizeof(double));

            int stencil_indices[5];
            for (j = 0; j < nentries; j++) /* label the stencil indices -
                                              these correspond to the offsets
                                              defined above */
            {
               stencil_indices[j] = j;
            }

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
               {
                  values[i + j] = -1.0;
               }
            }

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, nentries,
                                            stencil_indices, values);

            free(values);
         }

         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my second box */
         {
            int ilower[2] = {0, 1};
            int iupper[2] = {2, 4};

            int nentries = 5;
            int nvalues  = 60; /* 12 grid points, each with 5 stencil entries */
            /* double values[60]; OK to use constant-length array for CPUs */
            double *values = (double *) malloc(60 * sizeof(double));

            int stencil_indices[5];
            for (j = 0; j < nentries; j++)
            {
               stencil_indices[j] = j;
            }

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
               {
                  values[i + j] = -1.0;
               }
            }

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, nentries,
                                            stencil_indices, values);

            free(values);
         }
      }
      else if (myid == 1)
      {
         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my box */
         {
            int ilower[2] = {3, 1};
            int iupper[2] = {6, 4};

            int nentries = 5;
            int nvalues  = 80; /* 16 grid points, each with 5 stencil entries */
            /* double values[80]; OK to use constant-length array for CPUs */
            double *values = (double *) malloc(80 * sizeof(double));

            int stencil_indices[5];
            for (j = 0; j < nentries; j++)
            {
               stencil_indices[j] = j;
            }

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
               {
                  values[i + j] = -1.0;
               }
            }

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, nentries,
                                            stencil_indices, values);

            free(values);
         }
      }

      /* For each box, set any coefficients that reach ouside of the
         boundary to 0 */
      if (myid == 0)
      {
         int maxnvalues = 6;
         /* double values[6]; OK to use constant-length array for CPUs */
         double *values = (double *) malloc(6 * sizeof(double));

         for (i = 0; i < maxnvalues; i++)
         {
            values[i] = 0.0;
         }

         {
            /* Values below our first AND second box */
            int ilower[2] = {-3, 1};
            int iupper[2] = { 2, 1};

            int stencil_indices[1] = {3};

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
         }

         {
            /* Values to the left of our first box */
            int ilower[2] = {-3, 1};
            int iupper[2] = {-3, 2};

            int stencil_indices[1] = {1};

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
         }

         {
            /* Values above our first box */
            int ilower[2] = {-3, 2};
            int iupper[2] = {-1, 2};

            int stencil_indices[1] = {4};

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
         }

         {
            /* Values to the left of our second box (that do not border the
               first box). */
            int ilower[2] = { 0, 3};
            int iupper[2] = { 0, 4};

            int stencil_indices[1] = {1};

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
         }

         {
            /* Values above our second box */
            int ilower[2] = { 0, 4};
            int iupper[2] = { 2, 4};

            int stencil_indices[1] = {4};

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
         }

         free(values);
      }
      else if (myid == 1)
      {
         int maxnvalues = 4;
         /* double values[4]; OK to use constant-length array for CPUs */
         double *values = (double *) malloc(4 * sizeof(double));

         for (i = 0; i < maxnvalues; i++)
         {
            values[i] = 0.0;
         }

         {
            /* Values below our box */
            int ilower[2] = { 3, 1};
            int iupper[2] = { 6, 1};

            int stencil_indices[1] = {3};

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
         }

         {
            /* Values to the right of our box */
            int ilower[2] = { 6, 1};
            int iupper[2] = { 6, 4};

            int stencil_indices[1] = {2};

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
         }

         {
            /* Values above our box */
            int ilower[2] = { 3, 4};
            int iupper[2] = { 6, 4};

            int stencil_indices[1] = {4};

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
         }

         free(values);
      }

      /* This is a collective call finalizing the matrix assembly.
         The matrix is now ``ready to be used'' */
      NALU_HYPRE_SStructMatrixAssemble(A);
   }


   /* 5. Set up SStruct Vectors for b and x */
   {
      int i;

      /* We have one part and one variable. */
      int part = 0;
      int var = 0;

      /* Create an empty vector object */
      NALU_HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
      NALU_HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* As with the matrix,  set the object type for the vectors
         to be the struct type */
      object_type = NALU_HYPRE_STRUCT;
      NALU_HYPRE_SStructVectorSetObjectType(b, object_type);
      NALU_HYPRE_SStructVectorSetObjectType(x, object_type);

      /* Indicate that the vector coefficients are ready to be set */
      NALU_HYPRE_SStructVectorInitialize(b);
      NALU_HYPRE_SStructVectorInitialize(x);

      if (myid == 0)
      {
         /* Set the vector coefficients over the gridpoints in my first box */
         {
            int ilower[2] = {-3, 1};
            int iupper[2] = {-1, 2};

            int nvalues = 6;  /* 6 grid points */
            /* double values[6]; OK to use constant-length array for CPUs */
            double *values = (double *) malloc(6 * sizeof(double));

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 1.0;
            }
            NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 0.0;
            }
            NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

            free(values);
         }

         /* Set the vector coefficients over the gridpoints in my second box */
         {
            int ilower[2] = { 0, 1};
            int iupper[2] = { 2, 4};

            int nvalues = 12; /* 12 grid points */
            /* double values[12]; OK to use constant-length array for CPUs */
            double *values = (double *) malloc(12 * sizeof(double));

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 1.0;
            }
            NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 0.0;
            }
            NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

            free(values);
         }
      }
      else if (myid == 1)
      {
         /* Set the vector coefficients over the gridpoints in my box */
         {
            int ilower[2] = { 3, 1};
            int iupper[2] = { 6, 4};

            int nvalues = 16; /* 16 grid points */
            /* double values[16]; OK to use constant-length array for CPUs */
            double *values = (double *) malloc(16 * sizeof(double));

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 1.0;
            }
            NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 0.0;
            }
            NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

            free(values);
         }
      }

      /* This is a collective call finalizing the vector assembly.
         The vectors are now ``ready to be used'' */
      NALU_HYPRE_SStructVectorAssemble(b);
      NALU_HYPRE_SStructVectorAssemble(x);
   }

   /* 6. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   {
      NALU_HYPRE_StructMatrix sA;
      NALU_HYPRE_StructVector sb;
      NALU_HYPRE_StructVector sx;

      /* Because we are using a struct solver, we need to get the
         object of the matrix and vectors to pass in to the struct solvers */
      NALU_HYPRE_SStructMatrixGetObject(A, (void **) &sA);
      NALU_HYPRE_SStructVectorGetObject(b, (void **) &sb);
      NALU_HYPRE_SStructVectorGetObject(x, (void **) &sx);

      /* Create an empty PCG Struct solver */
      NALU_HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set PCG parameters */
      NALU_HYPRE_StructPCGSetTol(solver, 1.0e-06);
      NALU_HYPRE_StructPCGSetPrintLevel(solver, 2);
      NALU_HYPRE_StructPCGSetMaxIter(solver, 50);

      /* Create the Struct SMG solver for use as a preconditioner */
      NALU_HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);

      /* Set SMG parameters */
      NALU_HYPRE_StructSMGSetMaxIter(precond, 1);
      NALU_HYPRE_StructSMGSetTol(precond, 0.0);
      NALU_HYPRE_StructSMGSetZeroGuess(precond);
      NALU_HYPRE_StructSMGSetNumPreRelax(precond, 1);
      NALU_HYPRE_StructSMGSetNumPostRelax(precond, 1);

      /* Set preconditioner and solve */
      NALU_HYPRE_StructPCGSetPrecond(solver, NALU_HYPRE_StructSMGSolve,
                                NALU_HYPRE_StructSMGSetup, precond);
      NALU_HYPRE_StructPCGSetup(solver, sA, sb, sx);
      NALU_HYPRE_StructPCGSolve(solver, sA, sb, sx);
   }

   /* Save the solution for GLVis visualization, see vis/glvis-ex6.sh */
   if (vis)
   {
#ifdef NALU_HYPRE_EXVIS
      GLVis_PrintSStructGrid(grid, "vis/ex6.mesh", myid, NULL, NULL);
      GLVis_PrintSStructVector(x, 0, "vis/ex6.sol", myid);
      GLVis_PrintData("vis/ex6.data", myid, num_procs);
#endif
   }

   /* Free memory */
   NALU_HYPRE_SStructGridDestroy(grid);
   NALU_HYPRE_SStructStencilDestroy(stencil);
   NALU_HYPRE_SStructGraphDestroy(graph);
   NALU_HYPRE_SStructMatrixDestroy(A);
   NALU_HYPRE_SStructVectorDestroy(b);
   NALU_HYPRE_SStructVectorDestroy(x);

   NALU_HYPRE_StructPCGDestroy(solver);
   NALU_HYPRE_StructSMGDestroy(precond);

   /* Finalize HYPRE */
   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}
