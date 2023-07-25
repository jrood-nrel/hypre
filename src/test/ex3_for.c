/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 3

   Interface:      Structured interface (Struct)

   Compile with:   make ex3

   Sample run:     mpirun -np 16 ex3 -n 33 -solver 0 -v 1 1

   To see options: ex3 -help

   Description:    This code solves a system corresponding to a discretization
                   of the Laplace equation with zero boundary conditions on the
                   unit square. The domain is split into an N x N processor grid.
                   Thus, the given number of processors should be a perfect square.
                   Each processor's piece of the grid has n x n cells with n x n
                   nodes connected by the standard 5-point stencil. Note that the
                   struct interface assumes a cell-centered grid, and, therefore,
                   the nodes are not shared.  This example demonstrates more
                   features than the previous two struct examples (Example 1 and
                   Example 2).  Two solvers are available.

                   To incorporate the boundary conditions, we do the following:
                   Let x_i and x_b be the interior and boundary parts of the
                   solution vector x. We can split the matrix A as
                                   A = [A_ii A_ib; A_bi A_bb].
                   Let u_0 be the Dirichlet B.C.  We can simply say that x_b = u_0.
                   If b_i is the right-hand side, then we just need to solve in
                   the interior:
                                    A_ii x_i = b_i - A_ib u_0.
                   For this partitcular example, u_0 = 0, so we are just solving
                   A_ii x_i = b_i.

                   We recommend viewing examples 1 and 2 before viewing this
                   example.
*/

#include <math.h>
#include "_nalu_hypre_utilities.h"
#include "NALU_HYPRE_struct_ls.h"

#ifdef NALU_HYPRE_FORTRAN
#include "NALU_HYPRE_config.h"
#include "fortran.h"
#include "nalu_hypre_struct_fortran_test.h"
#endif


NALU_HYPRE_Int main (NALU_HYPRE_Int argc, char *argv[])
{
   NALU_HYPRE_Int i, j;

   NALU_HYPRE_Int myid, num_procs;

   NALU_HYPRE_Int n, N, pi, pj;
   NALU_HYPRE_Real h, h2;
   NALU_HYPRE_Int ilower[2], iupper[2];

   NALU_HYPRE_Int solver_id;
   NALU_HYPRE_Int n_pre, n_post;

#ifdef NALU_HYPRE_FORTRAN
   nalu_hypre_F90_Obj grid;
   nalu_hypre_F90_Obj stencil;
   nalu_hypre_F90_Obj A;
   nalu_hypre_F90_Obj b;
   nalu_hypre_F90_Obj x;
   nalu_hypre_F90_Obj solver;
   nalu_hypre_F90_Obj precond;
   NALU_HYPRE_Int temp_COMM;
   NALU_HYPRE_Int precond_id;
   NALU_HYPRE_Int zero = 0;
   NALU_HYPRE_Int one = 1;
   NALU_HYPRE_Int two = 2;
   NALU_HYPRE_Int five = 5;
   NALU_HYPRE_Int fifty = 50;
   NALU_HYPRE_Real zero_dot = 0.0;
   NALU_HYPRE_Real tol = 1.e-6;
#else
   NALU_HYPRE_StructGrid     grid;
   NALU_HYPRE_StructStencil  stencil;
   NALU_HYPRE_StructMatrix   A;
   NALU_HYPRE_StructVector   b;
   NALU_HYPRE_StructVector   x;
   NALU_HYPRE_StructSolver   solver;
   NALU_HYPRE_StructSolver   precond;
#endif

   NALU_HYPRE_Int num_iterations;
   NALU_HYPRE_Real final_res_norm;

   NALU_HYPRE_Int print_solution;

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid);
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs);

   /* Set defaults */
   n = 33;
   solver_id = 0;
   n_pre  = 1;
   n_post = 1;
   print_solution  = 0;

   /* Parse command line */
   {
      NALU_HYPRE_Int arg_index = 0;
      NALU_HYPRE_Int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-n") == 0 )
         {
            arg_index++;
            n = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-v") == 0 )
         {
            arg_index++;
            n_pre = atoi(argv[arg_index++]);
            n_post = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-print_solution") == 0 )
         {
            arg_index++;
            print_solution = 1;
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
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Usage: %s [<options>]\n", argv[0]);
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -n <n>              : problem size per procesor (default: 8)\n");
         nalu_hypre_printf("  -solver <ID>        : solver ID\n");
         nalu_hypre_printf("                        0  - PCG with SMG precond (default)\n");
         nalu_hypre_printf("                        1  - SMG\n");
         nalu_hypre_printf("  -v <n_pre> <n_post> : number of pre and post relaxations (default: 1 1)\n");
         nalu_hypre_printf("  -print_solution     : print the solution vector\n");
         nalu_hypre_printf("\n");
      }

      if (print_usage)
      {
         nalu_hypre_MPI_Finalize();
         return (0);
      }
   }

   /* Figure out the processor grid (N x N).  The local problem
      size for the interior nodes is indicated by n (n x n).
      pi and pj indicate position in the processor grid. */
   N  = nalu_hypre_sqrt(num_procs);
   h  = 1.0 / (N * n + 1); /* note that when calculating h we must
                          remember to count the bounday nodes */
   h2 = h * h;
   pj = myid / N;
   pi = myid - pj * N;

   /* Figure out the extents of each processor's piece of the grid. */
   ilower[0] = pi * n;
   ilower[1] = pj * n;

   iupper[0] = ilower[0] + n - 1;
   iupper[1] = ilower[1] + n - 1;

   /* 1. Set up a grid */
   {
#ifdef NALU_HYPRE_FORTRAN
      temp_COMM = (NALU_HYPRE_Int) nalu_hypre_MPI_COMM_WORLD;
      /* Create an empty 2D grid object */
      NALU_HYPRE_StructGridCreate(&temp_COMM, &two, &grid);

      /* Add a new box to the grid */
      NALU_HYPRE_StructGridSetExtents(&grid, &ilower[0], &iupper[0]);

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      NALU_HYPRE_StructGridAssemble(&grid);
#else
      /* Create an empty 2D grid object */
      NALU_HYPRE_StructGridCreate(nalu_hypre_MPI_COMM_WORLD, 2, &grid);

      /* Add a new box to the grid */
      NALU_HYPRE_StructGridSetExtents(grid, ilower, iupper);

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      NALU_HYPRE_StructGridAssemble(grid);
#endif
   }

   /* 2. Define the discretization stencil */
   {
#ifdef NALU_HYPRE_FORTRAN
      /* Create an empty 2D, 5-pt stencil object */
      NALU_HYPRE_StructStencilCreate(&two, &five, &stencil);

      /* Define the geometry of the stencil */
      {
         NALU_HYPRE_Int entry;
         NALU_HYPRE_Int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

         for (entry = 0; entry < 5; entry++)
         {
            NALU_HYPRE_StructStencilSetElement(&stencil, &entry, offsets[entry]);
         }
      }
#else
      /* Create an empty 2D, 5-pt stencil object */
      NALU_HYPRE_StructStencilCreate(2, 5, &stencil);

      /* Define the geometry of the stencil */
      {
         NALU_HYPRE_Int entry;
         NALU_HYPRE_Int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

         for (entry = 0; entry < 5; entry++)
         {
            NALU_HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
         }
      }
#endif
   }

   /* 3. Set up a Struct Matrix */
   {
      NALU_HYPRE_Int nentries = 5;
      NALU_HYPRE_Int nvalues = nentries * n * n;
      NALU_HYPRE_Real *values;
      NALU_HYPRE_Int stencil_indices[5];

#ifdef NALU_HYPRE_FORTRAN
      temp_COMM = (NALU_HYPRE_Int) nalu_hypre_MPI_COMM_WORLD;
      /* Create an empty matrix object */
      NALU_HYPRE_StructMatrixCreate(&temp_COMM, &grid, &stencil, &A);

      /* Indicate that the matrix coefficients are ready to be set */
      NALU_HYPRE_StructMatrixInitialize(&A);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nvalues, NALU_HYPRE_MEMORY_HOST);

      for (j = 0; j < nentries; j++)
      {
         stencil_indices[j] = j;
      }

      /* Set the standard stencil at each grid point,
         we will fix the boundaries later */
      for (i = 0; i < nvalues; i += nentries)
      {
         values[i] = 4.0;
         for (j = 1; j < nentries; j++)
         {
            values[i + j] = -1.0;
         }
      }

      NALU_HYPRE_StructMatrixSetBoxValues(&A, ilower, iupper, &nentries,
                                     stencil_indices, values);

      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
#else
      /* Create an empty matrix object */
      NALU_HYPRE_StructMatrixCreate(nalu_hypre_MPI_COMM_WORLD, grid, stencil, &A);

      /* Indicate that the matrix coefficients are ready to be set */
      NALU_HYPRE_StructMatrixInitialize(A);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nvalues, NALU_HYPRE_MEMORY_HOST);

      for (j = 0; j < nentries; j++)
      {
         stencil_indices[j] = j;
      }

      /* Set the standard stencil at each grid point,
         we will fix the boundaries later */
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

      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
#endif
   }

   /* 4. Incorporate the zero boundary conditions: go along each edge of
         the domain and set the stencil entry that reaches to the boundary to
         zero.*/
   {
      NALU_HYPRE_Int bc_ilower[2];
      NALU_HYPRE_Int bc_iupper[2];
      NALU_HYPRE_Int nentries = 1;
      NALU_HYPRE_Int nvalues  = nentries * n; /*  number of stencil entries times the length
                                     of one side of my grid box */
      NALU_HYPRE_Real *values;
      NALU_HYPRE_Int stencil_indices[1];

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nvalues, NALU_HYPRE_MEMORY_HOST);
      for (j = 0; j < nvalues; j++)
      {
         values[j] = 0.0;
      }

      /* Recall: pi and pj describe position in the processor grid */
      if (pj == 0)
      {
         /* bottom row of grid points */
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 3;

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructMatrixSetBoxValues(&A, bc_ilower, bc_iupper, &nentries,
                                        stencil_indices, values);
#else
         NALU_HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);
#endif
      }

      if (pj == N - 1)
      {
         /* upper row of grid points */
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n + n - 1;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 4;

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructMatrixSetBoxValues(&A, bc_ilower, bc_iupper, &nentries,
                                        stencil_indices, values);
#else
         NALU_HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);
#endif
      }

      if (pi == 0)
      {
         /* left row of grid points */
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         stencil_indices[0] = 1;

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructMatrixSetBoxValues(&A, bc_ilower, bc_iupper, &nentries,
                                        stencil_indices, values);
#else
         NALU_HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);
#endif
      }

      if (pi == N - 1)
      {
         /* right row of grid points */
         bc_ilower[0] = pi * n + n - 1;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         stencil_indices[0] = 2;

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructMatrixSetBoxValues(&A, bc_ilower, bc_iupper, &nentries,
                                        stencil_indices, values);
#else
         NALU_HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);
#endif
      }

      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
   }

   /* This is a collective call finalizing the matrix assembly.
      The matrix is now ``ready to be used'' */
#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_StructMatrixAssemble(&A);
#else
   NALU_HYPRE_StructMatrixAssemble(A);
#endif

   /* 5. Set up Struct Vectors for b and x */
   {
      NALU_HYPRE_Int    nvalues = n * n;
      NALU_HYPRE_Real *values;

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nvalues, NALU_HYPRE_MEMORY_HOST);

      /* Create an empty vector object */
#ifdef NALU_HYPRE_FORTRAN
      temp_COMM = (NALU_HYPRE_Int) nalu_hypre_MPI_COMM_WORLD;
      NALU_HYPRE_StructVectorCreate(&temp_COMM, &grid, &b);
      NALU_HYPRE_StructVectorCreate(&temp_COMM, &grid, &x);
#else
      NALU_HYPRE_StructVectorCreate(nalu_hypre_MPI_COMM_WORLD, grid, &b);
      NALU_HYPRE_StructVectorCreate(nalu_hypre_MPI_COMM_WORLD, grid, &x);
#endif

      /* Indicate that the vector coefficients are ready to be set */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructVectorInitialize(&b);
      NALU_HYPRE_StructVectorInitialize(&x);
#else
      NALU_HYPRE_StructVectorInitialize(b);
      NALU_HYPRE_StructVectorInitialize(x);
#endif

      /* Set the values */
      for (i = 0; i < nvalues; i ++)
      {
         values[i] = h2;
      }
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructVectorSetBoxValues(&b, ilower, iupper, values);
#else
      NALU_HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);
#endif

      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 0.0;
      }
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructVectorSetBoxValues(&x, ilower, iupper, values);
#else
      NALU_HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
#endif

      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      /* This is a collective call finalizing the vector assembly.
         The vector is now ``ready to be used'' */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_StructVectorAssemble(&b);
      NALU_HYPRE_StructVectorAssemble(&x);
#else
      NALU_HYPRE_StructVectorAssemble(b);
      NALU_HYPRE_StructVectorAssemble(x);
#endif
   }

   /* 6. Set up and use a struct solver
      (Solver options can be found in the Reference Manual.) */
   if (solver_id == 0)
   {
#ifdef NALU_HYPRE_FORTRAN
      temp_COMM = (NALU_HYPRE_Int) nalu_hypre_MPI_COMM_WORLD;
      NALU_HYPRE_StructPCGCreate(&temp_COMM, &solver);
      NALU_HYPRE_StructPCGSetMaxIter(&solver, &fifty );
      NALU_HYPRE_StructPCGSetTol(&solver, &tol );
      NALU_HYPRE_StructPCGSetTwoNorm(&solver, &one );
      NALU_HYPRE_StructPCGSetRelChange(&solver, &zero );
      NALU_HYPRE_StructPCGSetPrintLevel(&solver, &two ); /* print each CG iteration */
      NALU_HYPRE_StructPCGSetLogging(&solver, &one);

      /* Use symmetric SMG as preconditioner */
      NALU_HYPRE_StructSMGCreate(&temp_COMM, &precond);
      NALU_HYPRE_StructSMGSetMemoryUse(&precond, &zero);
      NALU_HYPRE_StructSMGSetMaxIter(&precond, &one);
      NALU_HYPRE_StructSMGSetTol(&precond, &zero_dot);
      NALU_HYPRE_StructSMGSetZeroGuess(&precond);
      NALU_HYPRE_StructSMGSetNumPreRelax(&precond, &one);
      NALU_HYPRE_StructSMGSetNumPostRelax(&precond, &one);

      /* Set the preconditioner and solve */
      precond_id = 0;
      NALU_HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
      NALU_HYPRE_StructPCGSetup(&solver, &A, &b, &x);
      NALU_HYPRE_StructPCGSolve(&solver, &A, &b, &x);

      /* Get some info on the run */
      NALU_HYPRE_StructPCGGetNumIterations(&solver, &num_iterations);
      NALU_HYPRE_StructPCGGetFinalRelativeResidualNorm(&solver, &final_res_norm);

      /* Clean up */
      NALU_HYPRE_StructPCGDestroy(&solver);
#else
      NALU_HYPRE_StructPCGCreate(nalu_hypre_MPI_COMM_WORLD, &solver);
      NALU_HYPRE_StructPCGSetMaxIter(solver, 50 );
      NALU_HYPRE_StructPCGSetTol(solver, 1.0e-06 );
      NALU_HYPRE_StructPCGSetTwoNorm(solver, 1 );
      NALU_HYPRE_StructPCGSetRelChange(solver, 0 );
      NALU_HYPRE_StructPCGSetPrintLevel(solver, 2 ); /* print each CG iteration */
      NALU_HYPRE_StructPCGSetLogging(solver, 1);

      /* Use symmetric SMG as preconditioner */
      NALU_HYPRE_StructSMGCreate(nalu_hypre_MPI_COMM_WORLD, &precond);
      NALU_HYPRE_StructSMGSetMemoryUse(precond, 0);
      NALU_HYPRE_StructSMGSetMaxIter(precond, 1);
      NALU_HYPRE_StructSMGSetTol(precond, 0.0);
      NALU_HYPRE_StructSMGSetZeroGuess(precond);
      NALU_HYPRE_StructSMGSetNumPreRelax(precond, 1);
      NALU_HYPRE_StructSMGSetNumPostRelax(precond, 1);

      /* Set the preconditioner and solve */
      NALU_HYPRE_StructPCGSetPrecond(solver, NALU_HYPRE_StructSMGSolve,
                                NALU_HYPRE_StructSMGSetup, precond);
      NALU_HYPRE_StructPCGSetup(solver, A, b, x);
      NALU_HYPRE_StructPCGSolve(solver, A, b, x);

      /* Get some info on the run */
      NALU_HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      /* Clean up */
      NALU_HYPRE_StructPCGDestroy(solver);
#endif
   }

   if (solver_id == 1)
   {
#ifdef NALU_HYPRE_FORTRAN
      temp_COMM = (NALU_HYPRE_Int) nalu_hypre_MPI_COMM_WORLD;
      NALU_HYPRE_StructSMGCreate(&temp_COMM, &solver);
      NALU_HYPRE_StructSMGSetMemoryUse(&solver, &zero);
      NALU_HYPRE_StructSMGSetMaxIter(&solver, &fifty);
      NALU_HYPRE_StructSMGSetTol(&solver, &tol);
      NALU_HYPRE_StructSMGSetRelChange(&solver, &zero);
      NALU_HYPRE_StructSMGSetNumPreRelax(&solver, &n_pre);
      NALU_HYPRE_StructSMGSetNumPostRelax(&solver, &n_post);
      /* Logging must be on to get iterations and residual norm info below */
      NALU_HYPRE_StructSMGSetLogging(&solver, &one);

      /* Setup and solve */
      NALU_HYPRE_StructSMGSetup(&solver, &A, &b, &x);
      NALU_HYPRE_StructSMGSolve(&solver, &A, &b, &x);

      /* Get some info on the run */
      NALU_HYPRE_StructSMGGetNumIterations(&solver, &num_iterations);
      NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm(&solver, &final_res_norm);

      /* Clean up */
      NALU_HYPRE_StructSMGDestroy(&solver);
#else
      NALU_HYPRE_StructSMGCreate(nalu_hypre_MPI_COMM_WORLD, &solver);
      NALU_HYPRE_StructSMGSetMemoryUse(solver, 0);
      NALU_HYPRE_StructSMGSetMaxIter(solver, 50);
      NALU_HYPRE_StructSMGSetTol(solver, 1.0e-06);
      NALU_HYPRE_StructSMGSetRelChange(solver, 0);
      NALU_HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
      NALU_HYPRE_StructSMGSetNumPostRelax(solver, n_post);
      /* Logging must be on to get iterations and residual norm info below */
      NALU_HYPRE_StructSMGSetLogging(solver, 1);

      /* Setup and solve */
      NALU_HYPRE_StructSMGSetup(solver, A, b, x);
      NALU_HYPRE_StructSMGSolve(solver, A, b, x);

      /* Get some info on the run */
      NALU_HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      /* Clean up */
      NALU_HYPRE_StructSMGDestroy(solver);
#endif
   }

   /* Print the solution and other info */
#ifdef NALU_HYPRE_FORTRAN
   if (print_solution)
   {
      NALU_HYPRE_StructVectorPrint(&x, &zero);
   }
#else
   if (print_solution)
   {
      NALU_HYPRE_StructVectorPrint("struct.out.x", x, 0);
   }
#endif

   if (myid == 0)
   {
      nalu_hypre_printf("\n");
      nalu_hypre_printf("Iterations = %d\n", num_iterations);
      nalu_hypre_printf("Final Relative Residual Norm = %g\n", final_res_norm);
      nalu_hypre_printf("\n");
   }

   /* Free memory */
#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_StructGridDestroy(&grid);
   NALU_HYPRE_StructStencilDestroy(&stencil);
   NALU_HYPRE_StructMatrixDestroy(&A);
   NALU_HYPRE_StructVectorDestroy(&b);
   NALU_HYPRE_StructVectorDestroy(&x);
#else
   NALU_HYPRE_StructGridDestroy(grid);
   NALU_HYPRE_StructStencilDestroy(stencil);
   NALU_HYPRE_StructMatrixDestroy(A);
   NALU_HYPRE_StructVectorDestroy(b);
   NALU_HYPRE_StructVectorDestroy(x);
#endif

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   return (0);
}
