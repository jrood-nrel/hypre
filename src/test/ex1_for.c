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

/* Struct linear solvers header */
#include "NALU_HYPRE_struct_ls.h"

#ifdef NALU_HYPRE_FORTRAN
#include "fortran.h"
#include "hypre_struct_fortran_test.h"
#endif

NALU_HYPRE_Int main (NALU_HYPRE_Int argc, char *argv[])
{
   NALU_HYPRE_Int i, j, myid;

#ifdef NALU_HYPRE_FORTRAN
   hypre_F90_Obj grid;
   hypre_F90_Obj stencil;
   hypre_F90_Obj A;
   hypre_F90_Obj b;
   hypre_F90_Obj x;
   hypre_F90_Obj solver;
   NALU_HYPRE_Int temp_COMM;
   NALU_HYPRE_Int one = 1;
   NALU_HYPRE_Int two = 2;
   NALU_HYPRE_Int five = 5;
   NALU_HYPRE_Real tol = 1.e-6;
#else
   NALU_HYPRE_StructGrid     grid;
   NALU_HYPRE_StructStencil  stencil;
   NALU_HYPRE_StructMatrix   A;
   NALU_HYPRE_StructVector   b;
   NALU_HYPRE_StructVector   x;
   NALU_HYPRE_StructSolver   solver;
#endif

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   /* 1. Set up a grid. Each processor describes the piece
      of the grid that it owns. */
   {
      /* Create an empty 2D grid object */
#ifdef NALU_HYPRE_FORTRAN
      temp_COMM = (NALU_HYPRE_Int) hypre_MPI_COMM_WORLD;
      NALU_HYPRE_StructGridCreate(&temp_COMM, &two, &grid);
#else
      NALU_HYPRE_StructGridCreate(hypre_MPI_COMM_WORLD, 2, &grid);
#endif

      /* Add boxes to the grid */
      if (myid == 0)
      {
         NALU_HYPRE_Int ilower[2] = {-3, 1}, iupper[2] = {-1, 2};
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructGridSetExtents(&grid, &ilower[0], &iupper[0]);
#else
         NALU_HYPRE_StructGridSetExtents(grid, ilower, iupper);
#endif
      }
      else if (myid == 1)
      {
         NALU_HYPRE_Int ilower[2] = {0, 1}, iupper[2] = {2, 4};
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructGridSetExtents(&grid, &ilower[0], &iupper[0]);
#else
         NALU_HYPRE_StructGridSetExtents(grid, ilower, iupper);
#endif
      }

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructGridAssemble(&grid);
#else
      NALU_HYPRE_StructGridAssemble(grid);
#endif
   }

   /* 2. Define the discretization stencil */
   {
      /* Create an empty 2D, 5-pt stencil object */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructStencilCreate(&two, &five, &stencil);
#else
      NALU_HYPRE_StructStencilCreate(2, 5, &stencil);
#endif

      /* Define the geometry of the stencil. Each represents a
         relative offset (in the index space). */
      {
         NALU_HYPRE_Int entry;
         NALU_HYPRE_Int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

         /* Assign each of the 5 stencil entries */
#ifdef NALU_HYPRE_FORTRAN
         for (entry = 0; entry < 5; entry++)
         {
            NALU_HYPRE_StructStencilSetElement(&stencil, &entry, offsets[entry]);
         }
#else
         for (entry = 0; entry < 5; entry++)
         {
            NALU_HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
         }
#endif
      }
   }

   /* 3. Set up a Struct Matrix */
   {
      /* Create an empty matrix object */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructMatrixCreate(&temp_COMM, &grid, &stencil, &A);
#else
      NALU_HYPRE_StructMatrixCreate(hypre_MPI_COMM_WORLD, grid, stencil, &A);
#endif

      /* Indicate that the matrix coefficients are ready to be set */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructMatrixInitialize(&A);
#else
      NALU_HYPRE_StructMatrixInitialize(A);
#endif

      /* Set the matrix coefficients.  Each processor assigns coefficients
         for the boxes in the grid that it owns. Note that the coefficients
         associated with each stencil entry may vary from grid point to grid
         point if desired.  Here, we first set the same stencil entries for
         each grid point.  Then we make modifications to grid points near
         the boundary. */
      if (myid == 0)
      {
         NALU_HYPRE_Int ilower[2] = {-3, 1}, iupper[2] = {-1, 2};
         NALU_HYPRE_Int stencil_indices[5] = {0, 1, 2, 3, 4}; /* labels for the stencil entries -
                                                  these correspond to the offsets
                                                  defined above */
         NALU_HYPRE_Int nentries = 5;
         NALU_HYPRE_Int nvalues  = 30; /* 6 grid points, each with 5 stencil entries */
         NALU_HYPRE_Real values[30];

         /* We have 6 grid points, each with 5 stencil entries */
         for (i = 0; i < nvalues; i += nentries)
         {
            values[i] = 4.0;
            for (j = 1; j < nentries; j++)
            {
               values[i + j] = -1.0;
            }
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &nentries,
                                        &stencil_indices[0], &values[0]);
#else
         NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                        stencil_indices, values);
#endif
      }
      else if (myid == 1)
      {
         NALU_HYPRE_Int ilower[2] = {0, 1}, iupper[2] = {2, 4};
         NALU_HYPRE_Int stencil_indices[5] = {0, 1, 2, 3, 4};
         NALU_HYPRE_Int nentries = 5;
         NALU_HYPRE_Int nvalues  = 60; /* 12 grid points, each with 5 stencil entries */
         NALU_HYPRE_Real values[60];

         for (i = 0; i < nvalues; i += nentries)
         {
            values[i] = 4.0;
            for (j = 1; j < nentries; j++)
            {
               values[i + j] = -1.0;
            }
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &nentries,
                                        &stencil_indices[0], &values[0]);
#else
         NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                        stencil_indices, values);
#endif
      }

      /* Set the coefficients reaching outside of the boundary to 0 */
      if (myid == 0)
      {
         NALU_HYPRE_Real values[3];
         for (i = 0; i < 3; i++)
         {
            values[i] = 0.0;
         }
         {
            /* values below our box */
            NALU_HYPRE_Int ilower[2] = {-3, 1}, iupper[2] = {-1, 1};
            NALU_HYPRE_Int stencil_indices[1] = {3};
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                           &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values to the left of our box */
            NALU_HYPRE_Int ilower[2] = {-3, 1}, iupper[2] = {-3, 2};
            NALU_HYPRE_Int stencil_indices[1] = {1};
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                           &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values above our box */
            NALU_HYPRE_Int ilower[2] = {-3, 2}, iupper[2] = {-1, 2};
            NALU_HYPRE_Int stencil_indices[1] = {4};
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                           &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
      }
      else if (myid == 1)
      {
         NALU_HYPRE_Real values[4];
         for (i = 0; i < 4; i++)
         {
            values[i] = 0.0;
         }
         {
            /* values below our box */
            NALU_HYPRE_Int ilower[2] = {0, 1}, iupper[2] = {2, 1};
            NALU_HYPRE_Int stencil_indices[1] = {3};
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                           &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values to the right of our box
               (that do not border the other box on proc. 0) */
            NALU_HYPRE_Int ilower[2] = {2, 1}, iupper[2] = {2, 4};
            NALU_HYPRE_Int stencil_indices[1] = {2};
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                           &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values above our box */
            NALU_HYPRE_Int ilower[2] = {0, 4}, iupper[2] = {2, 4};
            NALU_HYPRE_Int stencil_indices[1] = {4};
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                           &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values to the left of our box */
            NALU_HYPRE_Int ilower[2] = {0, 3}, iupper[2] = {0, 4};
            NALU_HYPRE_Int stencil_indices[1] = {1};
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                           &stencil_indices[0], &values[0]);
#else
            NALU_HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
      }

      /* This is a collective call finalizing the matrix assembly.
         The matrix is now ``ready to be used'' */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructMatrixAssemble(&A);
#else
      NALU_HYPRE_StructMatrixAssemble(A);
#endif
   }

   /* 4. Set up Struct Vectors for b and x.  Each processor sets the vectors
      corresponding to its boxes. */
   {
      /* Create an empty vector object */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructVectorCreate(&temp_COMM, &grid, &b);
      NALU_HYPRE_StructVectorCreate(&temp_COMM, &grid, &x);
#else
      NALU_HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, grid, &b);
      NALU_HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, grid, &x);
#endif

      /* Indicate that the vector coefficients are ready to be set */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructVectorInitialize(&b);
      NALU_HYPRE_StructVectorInitialize(&x);
#else
      NALU_HYPRE_StructVectorInitialize(b);
      NALU_HYPRE_StructVectorInitialize(x);
#endif

      /* Set the vector coefficients */
      if (myid == 0)
      {
         NALU_HYPRE_Int ilower[2] = {-3, 1}, iupper[2] = {-1, 2};
         NALU_HYPRE_Real values[6]; /* 6 grid points */

         for (i = 0; i < 6; i ++)
         {
            values[i] = 1.0;
         }
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructVectorSetBoxValues(&b, &ilower[0], &iupper[0], &values[0]);
#else
         NALU_HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);
#endif

         for (i = 0; i < 6; i ++)
         {
            values[i] = 0.0;
         }
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructVectorSetBoxValues(&x, &ilower[0], &iupper[0], &values[0]);
#else
         NALU_HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
#endif
      }
      else if (myid == 1)
      {
         NALU_HYPRE_Int ilower[2] = {0, 1}, iupper[2] = {2, 4};
         NALU_HYPRE_Real values[12]; /* 12 grid points */

         for (i = 0; i < 12; i ++)
         {
            values[i] = 1.0;
         }
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructVectorSetBoxValues(&b, &ilower[0], &iupper[0], &values[0]);
#else
         NALU_HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);
#endif

         for (i = 0; i < 12; i ++)
         {
            values[i] = 0.0;
         }
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructVectorSetBoxValues(&x, &ilower[0], &iupper[0], &values[0]);
#else
         NALU_HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
#endif
      }

      /* This is a collective call finalizing the vector assembly.
         The vectors are now ``ready to be used'' */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructVectorAssemble(&b);
      NALU_HYPRE_StructVectorAssemble(&x);
#else
      NALU_HYPRE_StructVectorAssemble(b);
      NALU_HYPRE_StructVectorAssemble(x);
#endif
   }

   /* 5. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   {
      /* Create an empty PCG Struct solver */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructPCGCreate(&temp_COMM, &solver);
#else
      NALU_HYPRE_StructPCGCreate(hypre_MPI_COMM_WORLD, &solver);
#endif

      /* Set some parameters */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructPCGSetTol(&solver, &tol); /* convergence tolerance */
      NALU_HYPRE_StructPCGSetPrintLevel(&solver, &two); /* amount of info. printed */
#else
      NALU_HYPRE_StructPCGSetTol(solver, 1.0e-06); /* convergence tolerance */
      NALU_HYPRE_StructPCGSetPrintLevel(solver, 2); /* amount of info. printed */
#endif

      /* Setup and solve */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructPCGSetup(&solver, &A, &b, &x);
      NALU_HYPRE_StructPCGSolve(&solver, &A, &b, &x);
#else
      NALU_HYPRE_StructPCGSetup(solver, A, b, x);
      NALU_HYPRE_StructPCGSolve(solver, A, b, x);
#endif
   }

   /* Free memory */
#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_StructGridDestroy(&grid);
   NALU_HYPRE_StructStencilDestroy(&stencil);
   NALU_HYPRE_StructMatrixDestroy(&A);
   NALU_HYPRE_StructVectorDestroy(&b);
   NALU_HYPRE_StructVectorDestroy(&x);
   NALU_HYPRE_StructPCGDestroy(&solver);
#else
   NALU_HYPRE_StructGridDestroy(grid);
   NALU_HYPRE_StructStencilDestroy(stencil);
   NALU_HYPRE_StructMatrixDestroy(A);
   NALU_HYPRE_StructVectorDestroy(b);
   NALU_HYPRE_StructVectorDestroy(x);
   NALU_HYPRE_StructPCGDestroy(solver);
#endif

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}
