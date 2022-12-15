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

/* SStruct linear solvers headers */
#include "NALU_HYPRE_sstruct_ls.h"

/*     include fortran headers       */
#ifdef NALU_HYPRE_FORTRAN
#include "fortran.h"
#include "nalu_hypre_struct_fortran_test.h"
#include "nalu_hypre_sstruct_fortran_test.h"
#endif

NALU_HYPRE_Int main (NALU_HYPRE_Int argc, char *argv[])
{
   NALU_HYPRE_Int myid, num_procs;

   /* We are using struct solvers for this example */
#ifdef NALU_HYPRE_FORTRAN
   nalu_hypre_F90_Obj     grid;
   nalu_hypre_F90_Obj     graph;
   nalu_hypre_F90_Obj     stencil;
   nalu_hypre_F90_Obj     A;
   nalu_hypre_F90_Obj     b;
   nalu_hypre_F90_Obj     x;

   nalu_hypre_F90_Obj     solver;
   nalu_hypre_F90_Obj     precond;

   nalu_hypre_F90_Obj     long_temp_COMM;
   NALU_HYPRE_Int          temp_COMM;

   NALU_HYPRE_Int          precond_id;

   NALU_HYPRE_Int          nalu_hypre_var_cell = NALU_HYPRE_SSTRUCT_VARIABLE_CELL;

   NALU_HYPRE_Int          one = 1;
   NALU_HYPRE_Int          two = 2;
   NALU_HYPRE_Int          five = 5;
   NALU_HYPRE_Int          fifty = 50;

   NALU_HYPRE_Real   tol = 1.e-6;
   NALU_HYPRE_Real   zerodot = 0.;
#else
   NALU_HYPRE_SStructGrid     grid;
   NALU_HYPRE_SStructGraph    graph;
   NALU_HYPRE_SStructStencil  stencil;
   NALU_HYPRE_SStructMatrix   A;
   NALU_HYPRE_SStructVector   b;
   NALU_HYPRE_SStructVector   x;

   NALU_HYPRE_StructSolver solver;
   NALU_HYPRE_StructSolver precond;
#endif

   NALU_HYPRE_Int object_type;

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid);
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs);

   if (num_procs != 2)
   {
      if (myid == 0) { nalu_hypre_printf("Must run with 2 processors!\n"); }
      nalu_hypre_MPI_Finalize();

      return (0);
   }
#ifdef NALU_HYPRE_FORTRAN
   temp_COMM = (NALU_HYPRE_Int) nalu_hypre_MPI_COMM_WORLD;
   long_temp_COMM = (nalu_hypre_F90_Obj) nalu_hypre_MPI_COMM_WORLD;
#endif

   /* 1. Set up the 2D grid.  This gives the index space in each part.
      Here we only use one part and one variable. (So the part id is 0
      and the variable id is 0) */
   {
      NALU_HYPRE_Int ndim = 2;
      NALU_HYPRE_Int nparts = 1;
      NALU_HYPRE_Int part = 0;

      /* Create an empty 2D grid object */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGridCreate(&temp_COMM, &ndim, &nparts, &grid);
#else
      NALU_HYPRE_SStructGridCreate(nalu_hypre_MPI_COMM_WORLD, ndim, nparts, &grid);
#endif

      /* Set the extents of the grid - each processor sets its grid
         boxes.  Each part has its own relative index space numbering,
         but in this example all boxes belong to the same part. */

      /* Processor 0 owns two boxes in the grid. */
      if (myid == 0)
      {
         /* Add a new box to the grid */
         {
            NALU_HYPRE_Int ilower[2] = {-3, 1};
            NALU_HYPRE_Int iupper[2] = {-1, 2};

#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructGridSetExtents(&grid, &part, &ilower[0], &iupper[0]);
#else
            NALU_HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
#endif
         }

         /* Add a new box to the grid */
         {
            NALU_HYPRE_Int ilower[2] = {0, 1};
            NALU_HYPRE_Int iupper[2] = {2, 4};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructGridSetExtents(&grid, &part, &ilower[0], &iupper[0]);
#else
            NALU_HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
#endif
         }
      }

      /* Processor 1 owns one box in the grid. */
      else if (myid == 1)
      {
         /* Add a new box to the grid */
         {
            NALU_HYPRE_Int ilower[2] = {3, 1};
            NALU_HYPRE_Int iupper[2] = {6, 4};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructGridSetExtents(&grid, &part, &ilower[0], &iupper[0]);
#else
            NALU_HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
#endif
         }
      }

      /* Set the variable type and number of variables on each part. */
      {
         NALU_HYPRE_Int i;
         NALU_HYPRE_Int nvars = 1;

#ifdef NALU_HYPRE_FORTRAN
         nalu_hypre_F90_Obj vartypes[1] = {NALU_HYPRE_SSTRUCT_VARIABLE_CELL};
#else
         NALU_HYPRE_SStructVariable vartypes[1] = {NALU_HYPRE_SSTRUCT_VARIABLE_CELL};
#endif

         for (i = 0; i < nparts; i++)

#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructGridSetVariables(&grid, &i, &nvars, &vartypes[0]);
#else
            NALU_HYPRE_SStructGridSetVariables(grid, i, nvars, vartypes);
#endif
      }

      /* Now the grid is ready to use */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGridAssemble(&grid);
#else
      NALU_HYPRE_SStructGridAssemble(grid);
#endif
   }

   /* 2. Define the discretization stencil(s) */
   {
      /* Create an empty 2D, 5-pt stencil object */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructStencilCreate(&two, &five, &stencil);
#else
      NALU_HYPRE_SStructStencilCreate(2, 5, &stencil);
#endif

      /* Define the geometry of the stencil. Each represents a
         relative offset (in the index space). */
      {
         NALU_HYPRE_Int entry;
         NALU_HYPRE_Int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
         NALU_HYPRE_Int var = 0;

         /* Assign numerical values to the offsets so that we can
            easily refer to them  - the last argument indicates the
            variable for which we are assigning this stencil - we are
            just using one variable in this example so it is the first one (0) */
         for (entry = 0; entry < 5; entry++)

#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructStencilSetEntry(&stencil, &entry, &offsets[entry][0], &var);
#else
            NALU_HYPRE_SStructStencilSetEntry(stencil, entry, offsets[entry], var);
#endif
      }
   }

   /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix and allows non-stencil relationships between the parts */
   {
      NALU_HYPRE_Int var = 0;
      NALU_HYPRE_Int part = 0;

      /* Create the graph object */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGraphCreate(&temp_COMM, &grid, &graph);
#else
      NALU_HYPRE_SStructGraphCreate(nalu_hypre_MPI_COMM_WORLD, grid, &graph);
#endif

      /* Now we need to tell the graph which stencil to use for each
         variable on each part (we only have one variable and one part) */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGraphSetStencil(&graph, &part, &var, &stencil);
#else
      NALU_HYPRE_SStructGraphSetStencil(graph, part, var, stencil);
#endif

      /* Here we could establish connections between parts if we
         had more than one part using the graph. For example, we could
         use NALU_HYPRE_GraphAddEntries() routine or NALU_HYPRE_GridSetNeighborBox() */

      /* Assemble the graph */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGraphAssemble(&graph);
#else
      NALU_HYPRE_SStructGraphAssemble(graph);
#endif
   }

   /* 4. Set up a SStruct Matrix */
   {
      NALU_HYPRE_Int i, j;
      NALU_HYPRE_Int part = 0;
      NALU_HYPRE_Int var = 0;

      /* Create the empty matrix object */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixCreate(&temp_COMM, &graph, &A);
#else
      NALU_HYPRE_SStructMatrixCreate(nalu_hypre_MPI_COMM_WORLD, graph, &A);
#endif

      /* Set the object type (by default NALU_HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  If you want to use unstructured
         solvers, e.g. BoomerAMG, the object type should be NALU_HYPRE_PARCSR.
         If the problem is purely structured (with one part), you may want to use
         NALU_HYPRE_STRUCT to access the structured solvers. Here we have a purely
         structured example. */
      object_type = NALU_HYPRE_STRUCT;

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixSetObjectType(&A, &object_type);
#else
      NALU_HYPRE_SStructMatrixSetObjectType(A, object_type);
#endif

      /* Get ready to set values */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixInitialize(&A);
#else
      NALU_HYPRE_SStructMatrixInitialize(A);
#endif

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
            NALU_HYPRE_Int ilower[2] = {-3, 1};
            NALU_HYPRE_Int iupper[2] = {-1, 2};

            NALU_HYPRE_Int nentries = 5;
            NALU_HYPRE_Int nvalues  = 30; /* 6 grid points, each with 5 stencil entries */
            NALU_HYPRE_Real values[30];

            NALU_HYPRE_Int stencil_indices[5];
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


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &nentries,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, nentries,
                                            stencil_indices, values);
#endif
         }

         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my second box */
         {
            NALU_HYPRE_Int ilower[2] = {0, 1};
            NALU_HYPRE_Int iupper[2] = {2, 4};

            NALU_HYPRE_Int nentries = 5;
            NALU_HYPRE_Int nvalues  = 60; /* 12 grid points, each with 5 stencil entries */
            NALU_HYPRE_Real values[60];

            NALU_HYPRE_Int stencil_indices[5];
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


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &nentries,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, nentries,
                                            stencil_indices, values);
#endif
         }
      }
      else if (myid == 1)
      {
         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my box */
         {
            NALU_HYPRE_Int ilower[2] = {3, 1};
            NALU_HYPRE_Int iupper[2] = {6, 4};

            NALU_HYPRE_Int nentries = 5;
            NALU_HYPRE_Int nvalues  = 80; /* 16 grid points, each with 5 stencil entries */
            NALU_HYPRE_Real values[80];

            NALU_HYPRE_Int stencil_indices[5];
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


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &nentries,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, nentries,
                                            stencil_indices, values);
#endif
         }
      }

      /* For each box, set any coefficients that reach ouside of the
         boundary to 0 */
      if (myid == 0)
      {
         NALU_HYPRE_Int maxnvalues = 6;
         NALU_HYPRE_Real values[6];

         for (i = 0; i < maxnvalues; i++)
         {
            values[i] = 0.0;
         }

         {
            /* Values below our first AND second box */
            NALU_HYPRE_Int ilower[2] = {-3, 1};
            NALU_HYPRE_Int iupper[2] = { 2, 1};

            NALU_HYPRE_Int stencil_indices[1] = {3};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values to the left of our first box */
            NALU_HYPRE_Int ilower[2] = {-3, 1};
            NALU_HYPRE_Int iupper[2] = {-3, 2};

            NALU_HYPRE_Int stencil_indices[1] = {1};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values above our first box */
            NALU_HYPRE_Int ilower[2] = {-3, 2};
            NALU_HYPRE_Int iupper[2] = {-1, 2};

            NALU_HYPRE_Int stencil_indices[1] = {4};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values to the left of our second box (that do not border the
               first box). */
            NALU_HYPRE_Int ilower[2] = { 0, 3};
            NALU_HYPRE_Int iupper[2] = { 0, 4};

            NALU_HYPRE_Int stencil_indices[1] = {1};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values above our second box */
            NALU_HYPRE_Int ilower[2] = { 0, 4};
            NALU_HYPRE_Int iupper[2] = { 2, 4};

            NALU_HYPRE_Int stencil_indices[1] = {4};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }
      }
      else if (myid == 1)
      {
         NALU_HYPRE_Int maxnvalues = 4;
         NALU_HYPRE_Real values[4];
         for (i = 0; i < maxnvalues; i++)
         {
            values[i] = 0.0;
         }

         {
            /* Values below our box */
            NALU_HYPRE_Int ilower[2] = { 3, 1};
            NALU_HYPRE_Int iupper[2] = { 6, 1};

            NALU_HYPRE_Int stencil_indices[1] = {3};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values to the right of our box */
            NALU_HYPRE_Int ilower[2] = { 6, 1};
            NALU_HYPRE_Int iupper[2] = { 6, 4};

            NALU_HYPRE_Int stencil_indices[1] = {2};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values above our box */
            NALU_HYPRE_Int ilower[2] = { 3, 4};
            NALU_HYPRE_Int iupper[2] = { 6, 4};

            NALU_HYPRE_Int stencil_indices[1] = {4};


#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }
      }

      /* This is a collective call finalizing the matrix assembly.
         The matrix is now ``ready to be used'' */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixAssemble(&A);
#else
      NALU_HYPRE_SStructMatrixAssemble(A);
#endif
   }


   /* 5. Set up SStruct Vectors for b and x */
   {
      NALU_HYPRE_Int i;

      /* We have one part and one variable. */
      NALU_HYPRE_Int part = 0;
      NALU_HYPRE_Int var = 0;

      /* Create an empty vector object */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructVectorCreate(&temp_COMM, &grid, &b);
      NALU_HYPRE_SStructVectorCreate(&temp_COMM, &grid, &x);
#else
      NALU_HYPRE_SStructVectorCreate(nalu_hypre_MPI_COMM_WORLD, grid, &b);
      NALU_HYPRE_SStructVectorCreate(nalu_hypre_MPI_COMM_WORLD, grid, &x);
#endif

      /* As with the matrix,  set the object type for the vectors
         to be the struct type */
      object_type = NALU_HYPRE_STRUCT;

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructVectorSetObjectType(&b, &object_type);
      NALU_HYPRE_SStructVectorSetObjectType(&x, &object_type);
#else
      NALU_HYPRE_SStructVectorSetObjectType(b, object_type);
      NALU_HYPRE_SStructVectorSetObjectType(x, object_type);
#endif

      /* Indicate that the vector coefficients are ready to be set */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructVectorInitialize(&b);
      NALU_HYPRE_SStructVectorInitialize(&x);
#else
      NALU_HYPRE_SStructVectorInitialize(b);
      NALU_HYPRE_SStructVectorInitialize(x);
#endif

      if (myid == 0)
      {
         /* Set the vector coefficients over the gridpoints in my first box */
         {
            NALU_HYPRE_Int ilower[2] = {-3, 1};
            NALU_HYPRE_Int iupper[2] = {-1, 2};

            NALU_HYPRE_Int nvalues = 6;  /* 6 grid points */
            NALU_HYPRE_Real values[6];

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 1.0;
            }

#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructVectorSetBoxValues(&b, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
#endif

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 0.0;
            }

#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructVectorSetBoxValues(&x, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
#endif
         }

         /* Set the vector coefficients over the gridpoints in my second box */
         {
            NALU_HYPRE_Int ilower[2] = { 0, 1};
            NALU_HYPRE_Int iupper[2] = { 2, 4};

            NALU_HYPRE_Int nvalues = 12; /* 12 grid points */
            NALU_HYPRE_Real values[12];

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 1.0;
            }

#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructVectorSetBoxValues(&b, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
#endif

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 0.0;
            }

#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructVectorSetBoxValues(&x, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
#endif
         }
      }
      else if (myid == 1)
      {
         /* Set the vector coefficients over the gridpoints in my box */
         {
            NALU_HYPRE_Int ilower[2] = { 3, 1};
            NALU_HYPRE_Int iupper[2] = { 6, 4};

            NALU_HYPRE_Int nvalues = 16; /* 16 grid points */
            NALU_HYPRE_Real values[16];

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 1.0;
            }

#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructVectorSetBoxValues(&b, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
#endif

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 0.0;
            }

#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructVectorSetBoxValues(&x, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
#endif
         }
      }

      /* This is a collective call finalizing the vector assembly.
         The vectors are now ``ready to be used'' */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructVectorAssemble(&b);
      NALU_HYPRE_SStructVectorAssemble(&x);
#else
      NALU_HYPRE_SStructVectorAssemble(b);
      NALU_HYPRE_SStructVectorAssemble(x);
#endif
   }


   /* 6. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   {

#ifdef NALU_HYPRE_FORTRAN
      nalu_hypre_F90_Obj sA;
      nalu_hypre_F90_Obj sb;
      nalu_hypre_F90_Obj sx;
#else
      NALU_HYPRE_StructMatrix sA;
      NALU_HYPRE_StructVector sb;
      NALU_HYPRE_StructVector sx;
#endif

      /* Because we are using a struct solver, we need to get the
         object of the matrix and vectors to pass in to the struct solvers */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixGetObject(&A, &sA);
      NALU_HYPRE_SStructVectorGetObject(&b, &sb);
      NALU_HYPRE_SStructVectorGetObject(&x, &sx);
#else
      NALU_HYPRE_SStructMatrixGetObject(A, (void **) &sA);
      NALU_HYPRE_SStructVectorGetObject(b, (void **) &sb);
      NALU_HYPRE_SStructVectorGetObject(x, (void **) &sx);
#endif

      /* Create an empty PCG Struct solver */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructPCGCreate(&temp_COMM, &solver);
#else
      NALU_HYPRE_StructPCGCreate(nalu_hypre_MPI_COMM_WORLD, &solver);
#endif

      /* Set PCG parameters */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructPCGSetTol(&solver, &tol);
      NALU_HYPRE_StructPCGSetPrintLevel(&solver, &two);
      NALU_HYPRE_StructPCGSetMaxIter(&solver, &fifty);
#else
      NALU_HYPRE_StructPCGSetTol(solver, 1.0e-06);
      NALU_HYPRE_StructPCGSetPrintLevel(solver, 2);
      NALU_HYPRE_StructPCGSetMaxIter(solver, 50);
#endif

      /* Create the Struct SMG solver for use as a preconditioner */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructSMGCreate(&temp_COMM, &precond);
#else
      NALU_HYPRE_StructSMGCreate(nalu_hypre_MPI_COMM_WORLD, &precond);
#endif

      /* Set SMG parameters */

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructSMGSetMaxIter(&precond, &one);
      NALU_HYPRE_StructSMGSetTol(&precond, &zerodot);
      NALU_HYPRE_StructSMGSetZeroGuess(&precond);
      NALU_HYPRE_StructSMGSetNumPreRelax(&precond, &one);
      NALU_HYPRE_StructSMGSetNumPostRelax(&precond, &one);
#else
      NALU_HYPRE_StructSMGSetMaxIter(precond, 1);
      NALU_HYPRE_StructSMGSetTol(precond, 0.0);
      NALU_HYPRE_StructSMGSetZeroGuess(precond);
      NALU_HYPRE_StructSMGSetNumPreRelax(precond, 1);
      NALU_HYPRE_StructSMGSetNumPostRelax(precond, 1);
#endif

      /* Set preconditioner and solve */

#ifdef NALU_HYPRE_FORTRAN
      precond_id = 0;
      NALU_HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
      NALU_HYPRE_StructPCGSetup(&solver, &sA, &sb, &sx);
      NALU_HYPRE_StructPCGSolve(&solver, &sA, &sb, &sx);
#else
      NALU_HYPRE_StructPCGSetPrecond(solver, NALU_HYPRE_StructSMGSolve,
                                NALU_HYPRE_StructSMGSetup, precond);
      NALU_HYPRE_StructPCGSetup(solver, sA, sb, sx);
      NALU_HYPRE_StructPCGSolve(solver, sA, sb, sx);
#endif
   }

   /* Free memory */

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructGridDestroy(&grid);
   NALU_HYPRE_SStructStencilDestroy(&stencil);
   NALU_HYPRE_SStructGraphDestroy(&graph);
   NALU_HYPRE_SStructMatrixDestroy(&A);
   NALU_HYPRE_SStructVectorDestroy(&b);
   NALU_HYPRE_SStructVectorDestroy(&x);

   NALU_HYPRE_StructPCGDestroy(&solver);
   NALU_HYPRE_StructSMGDestroy(&precond);
#else
   NALU_HYPRE_SStructGridDestroy(grid);
   NALU_HYPRE_SStructStencilDestroy(stencil);
   NALU_HYPRE_SStructGraphDestroy(graph);
   NALU_HYPRE_SStructMatrixDestroy(A);
   NALU_HYPRE_SStructVectorDestroy(b);
   NALU_HYPRE_SStructVectorDestroy(x);

   NALU_HYPRE_StructPCGDestroy(solver);
   NALU_HYPRE_StructSMGDestroy(precond);
#endif

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   return (0);
}
