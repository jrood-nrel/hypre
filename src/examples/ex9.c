/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 9

   Interface:      Semi-Structured interface (SStruct)

   Compile with:   make ex9

   Sample run:     mpirun -np 16 ex9 -n 33 -solver 0 -v 1 1

   To see options: ex9 -help

   Description:    This code solves a system corresponding to a discretization
                   of the biharmonic problem treated as a system of equations
                   on the unit square.  Specifically, instead of solving
                   Delta^2(u) = f with zero boundary conditions for u and
                   Delta(u), we solve the system A x = b, where

                   A = [ Delta -I ; 0 Delta], x = [ u ; v] and b = [ 0 ; f]

                   The corresponding boundary conditions are u = 0 and v = 0.

                   The domain is split into an N x N processor grid.  Thus, the
                   given number of processors should be a perfect square.
                   Each processor's piece of the grid has n x n cells with n x n
                   nodes. We use cell-centered variables, and, therefore, the
                   nodes are not shared. Note that we have two variables, u and
                   v, and need only one part to describe the domain. We use the
                   standard 5-point stencil to discretize the Laplace operators.
                   The boundary conditions are incorporated as in Example 3.

                   We recommend viewing Examples 3, 6 and 7 before this example.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "NALU_HYPRE_sstruct_ls.h"
#include "NALU_HYPRE_krylov.h"
#include "ex.h"

#ifdef NALU_HYPRE_EXVIS
#include "vis.c"
#endif

int main (int argc, char *argv[])
{
   int i, j;

   int myid, num_procs;

   int n, N, pi, pj;
   double h, h2;
   int ilower[2], iupper[2];

   int solver_id;
   int n_pre, n_post;

   int vis;
   int object_type;

   NALU_HYPRE_SStructGrid     grid;
   NALU_HYPRE_SStructGraph    graph;
   NALU_HYPRE_SStructStencil  stencil_v;
   NALU_HYPRE_SStructStencil  stencil_u;
   NALU_HYPRE_SStructMatrix   A;
   NALU_HYPRE_SStructVector   b;
   NALU_HYPRE_SStructVector   x;

   /* sstruct solvers */
   NALU_HYPRE_SStructSolver   solver;
   NALU_HYPRE_SStructSolver   precond;

   /* parcsr solvers */
   NALU_HYPRE_Solver          par_solver;
   NALU_HYPRE_Solver          par_precond;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   NALU_HYPRE_Initialize();

   /* Print GPU info */
   /* NALU_HYPRE_PrintDeviceInfo(); */

   /* Set defaults */
   n = 33;
   solver_id = 0;
   n_pre  = 1;
   n_post = 1;
   vis = 0;

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

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
         else if ( strcmp(argv[arg_index], "-vis") == 0 )
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
         printf("  -n <n>              : problem size per processor (default: 33)\n");
         printf("  -solver <ID>        : solver ID\n");
         printf("                        0  - GMRES with sysPFMG precond (default)\n");
         printf("                        1  - sysPFMG\n");
         printf("                        2  - GMRES with AMG precond\n");
         printf("                        3  - AMG\n");
         printf("  -v <n_pre> <n_post> : number of pre and post relaxations for SysPFMG (default: 1 1)\n");
         printf("  -vis                : save the solution for GLVis visualization\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Figure out the processor grid (N x N).  The local problem
      size for the interior nodes is indicated by n (n x n).
      pi and pj indicate position in the processor grid. */
   N  = sqrt(num_procs);
   h  = 1.0 / (N * n + 1); /* note that when calculating h we must
                          remember to count the boundary nodes */
   h2 = h * h;
   pj = myid / N;
   pi = myid - pj * N;

   /* Figure out the extents of each processor's piece of the grid. */
   ilower[0] = pi * n;
   ilower[1] = pj * n;

   iupper[0] = ilower[0] + n - 1;
   iupper[1] = ilower[1] + n - 1;

   /* 1. Set up a grid - we have one part and two variables */
   {
      int nparts = 1;
      int part = 0;
      int ndim = 2;

      /* Create an empty 2D grid object */
      NALU_HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &grid);

      /* Add a new box to the grid */
      NALU_HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);

      /* Set the variable type and number of variables on each part.*/
      {
         int i;
         int nvars = 2;
         NALU_HYPRE_SStructVariable vartypes[2] =
         {
            NALU_HYPRE_SSTRUCT_VARIABLE_CELL,
            NALU_HYPRE_SSTRUCT_VARIABLE_CELL
         };

         for (i = 0; i < nparts; i++)
         {
            NALU_HYPRE_SStructGridSetVariables(grid, i, nvars, vartypes);
         }
      }

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      NALU_HYPRE_SStructGridAssemble(grid);
   }

   /* 2. Define the discretization stencils */
   {
      int entry;
      int stencil_size;
      int var;
      int ndim = 2;

      /* Stencil object for variable u (labeled as variable 0) */
      {
         int offsets[6][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {0, 0}};
         stencil_size = 6;

         NALU_HYPRE_SStructStencilCreate(ndim, stencil_size, &stencil_u);

         /* The first 5 entries are for the u-u connections */
         var = 0; /* connect to variable 0 */
         for (entry = 0; entry < stencil_size - 1 ; entry++)
         {
            NALU_HYPRE_SStructStencilSetEntry(stencil_u, entry, offsets[entry], var);
         }

         /* The last entry is for the u-v connection */
         var = 1;  /* connect to variable 1 */
         entry = 5;
         NALU_HYPRE_SStructStencilSetEntry(stencil_u, entry, offsets[entry], var);
      }

      /* Stencil object for variable v  (variable 1) */
      {
         int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
         stencil_size = 5;

         NALU_HYPRE_SStructStencilCreate(ndim, stencil_size, &stencil_v);

         /* These are all v-v connections */
         var = 1; /* Connect to variable 1 */
         for (entry = 0; entry < stencil_size; entry++)
         {
            NALU_HYPRE_SStructStencilSetEntry(stencil_v, entry, offsets[entry], var);
         }
      }
   }

   /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix and allows non-stencil relationships between the parts. */
   {
      int var;
      int part = 0;

      /* Create the graph object */
      NALU_HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

      /* See MatrixSetObjectType below */
      if (solver_id > 1 && solver_id < 4)
      {
         object_type = NALU_HYPRE_PARCSR;
      }
      else
      {
         object_type = NALU_HYPRE_SSTRUCT;
      }
      NALU_HYPRE_SStructGraphSetObjectType(graph, object_type);

      /* Assign the u-stencil we created to variable u (variable 0) */
      var = 0;
      NALU_HYPRE_SStructGraphSetStencil(graph, part, var, stencil_u);

      /* Assign the v-stencil we created to variable v (variable 1) */
      var = 1;
      NALU_HYPRE_SStructGraphSetStencil(graph, part, var, stencil_v);

      /* Assemble the graph */
      NALU_HYPRE_SStructGraphAssemble(graph);
   }

   /* 4. Set up the SStruct Matrix */
   {
      int nentries;
      int nvalues;
      int var;
      int part = 0;

      /* Create an empty matrix object */
      NALU_HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);

      /* Set the object type (by default NALU_HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  If you want to use
         unstructured solvers, e.g. BoomerAMG, the object type should be
         NALU_HYPRE_PARCSR. If the problem is purely structured (with one part), you
         may want to use NALU_HYPRE_STRUCT to access the structured solvers.  */
      NALU_HYPRE_SStructMatrixSetObjectType(A, object_type);

      /* Indicate that the matrix coefficients are ready to be set */
      NALU_HYPRE_SStructMatrixInitialize(A);

      /* Each processor must set the stencil values for their boxes on each part.
         In this example, we only set stencil entries and therefore use
         NALU_HYPRE_SStructMatrixSetBoxValues.  If we need to set non-stencil entries,
         we have to use NALU_HYPRE_SStructMatrixSetValues. */

      /* First set the u-stencil entries.  Note that
         NALU_HYPRE_SStructMatrixSetBoxValues can only set values corresponding
         to stencil entries for the same variable. Therefore, we must set the
         entries for each variable within a stencil with separate function calls.
         For example, below the u-u connections and u-v connections are handled
         in separate calls.  */
      {
         int     i, j;
         double *u_values;
         int     u_v_indices[1] = {5};
         int     u_u_indices[5] = {0, 1, 2, 3, 4};

         var = 0; /* Set values for the u connections */

         /*  First the u-u connections */
         nentries = 5;
         nvalues = nentries * n * n;
         u_values = (double*) calloc(nvalues, sizeof(double));

         for (i = 0; i < nvalues; i += nentries)
         {
            u_values[i] = 4.0;
            for (j = 1; j < nentries; j++)
            {
               u_values[i + j] = -1.0;
            }
         }

         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, nentries,
                                         u_u_indices, u_values);
         free(u_values);

         /* Next the u-v connections */
         nentries = 1;
         nvalues = nentries * n * n;
         u_values = (double*) calloc(nvalues, sizeof(double));

         for (i = 0; i < nvalues; i++)
         {
            u_values[i] = -h2;
         }

         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, nentries,
                                         u_v_indices, u_values);

         free(u_values);
      }

      /*  Now set the v-stencil entries */
      {
         int     i, j;
         double *v_values;
         int     v_v_indices[5] = {0, 1, 2, 3, 4};

         var = 1; /* the v connections */

         /* the v-v connections */
         nentries = 5;
         nvalues = nentries * n * n;
         v_values = (double*) calloc(nvalues, sizeof(double));

         for (i = 0; i < nvalues; i += nentries)
         {
            v_values[i] = 4.0;
            for (j = 1; j < nentries; j++)
            {
               v_values[i + j] = -1.0;
            }
         }

         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, nentries,
                                         v_v_indices, v_values);

         free(v_values);

         /* There are no v-u connections to set */
      }
   }

   /* 5. Incorporate the zero boundary conditions: go along each edge of
         the domain and set the stencil entry that reaches to the boundary
         to zero.*/
   {
      int bc_ilower[2];
      int bc_iupper[2];
      int nentries = 1;
      int nvalues  = nentries * n; /*  number of stencil entries times the length
                                     of one side of my grid box */
      int var;
      double *values;
      int stencil_indices[1];

      int part = 0;

      values = (double*) calloc(nvalues, sizeof(double));
      for (j = 0; j < nvalues; j++)
      {
         values[j] = 0.0;
      }

      /* Recall: pi and pj describe position in the processor grid */
      if (pj == 0)
      {
         /* Bottom row of grid points */
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 3;

         /* Need to do this for u and for v */
         var = 0;
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
      }

      if (pj == N - 1)
      {
         /* upper row of grid points */
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n + n - 1;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 4;

         /* Need to do this for u and for v */
         var = 0;
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

      }

      if (pi == 0)
      {
         /* Left row of grid points */
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         stencil_indices[0] = 1;

         /* Need to do this for u and for v */
         var = 0;
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
      }

      if (pi == N - 1)
      {
         /* Right row of grid points */
         bc_ilower[0] = pi * n + n - 1;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         stencil_indices[0] = 2;

         /* Need to do this for u and for v */
         var = 0;
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
      }

      free(values);
   }

   /* This is a collective call finalizing the matrix assembly.
      The matrix is now ``ready to be used'' */
   NALU_HYPRE_SStructMatrixAssemble(A);

   /* 5. Set up SStruct Vectors for b and x */
   {
      int    nvalues = n * n;
      double *values;
      int part = 0;
      int var;

      values = (double*) calloc(nvalues, sizeof(double));

      /* Create an empty vector object */
      NALU_HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
      NALU_HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* Set the object type for the vectors
         to be the same as was already set for the matrix */
      NALU_HYPRE_SStructVectorSetObjectType(b, object_type);
      NALU_HYPRE_SStructVectorSetObjectType(x, object_type);

      /* Indicate that the vector coefficients are ready to be set */
      NALU_HYPRE_SStructVectorInitialize(b);
      NALU_HYPRE_SStructVectorInitialize(x);

      /* Set the values for b */
      for (i = 0; i < nvalues; i ++)
      {
         values[i] = h2;
      }
      var = 1;
      NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 0.0;
      }
      var = 0;
      NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

      /* Set the values for the initial guess */
      var = 0;
      NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

      var = 1;
      NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

      free(values);

      /* This is a collective call finalizing the vector assembly.
         The vector is now ``ready to be used'' */
      NALU_HYPRE_SStructVectorAssemble(b);
      NALU_HYPRE_SStructVectorAssemble(x);
   }

   /* 6. Set up and use a solver
      (Solver options can be found in the Reference Manual.) */
   {
      double final_res_norm;
      int its;

      NALU_HYPRE_ParCSRMatrix    par_A;
      NALU_HYPRE_ParVector       par_b;
      NALU_HYPRE_ParVector       par_x;

      /* If we are using a parcsr solver, we need to get the object for the
         matrix and vectors. */
      if (object_type == NALU_HYPRE_PARCSR)
      {
         NALU_HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
         NALU_HYPRE_SStructVectorGetObject(b, (void **) &par_b);
         NALU_HYPRE_SStructVectorGetObject(x, (void **) &par_x);
      }

      if (solver_id == 0 ) /* GMRES with SysPFMG - the default*/
      {
         NALU_HYPRE_SStructGMRESCreate(MPI_COMM_WORLD, &solver);

         /* GMRES parameters */
         NALU_HYPRE_SStructGMRESSetMaxIter(solver, 50 );
         NALU_HYPRE_SStructGMRESSetTol(solver, 1.0e-06 );
         NALU_HYPRE_SStructGMRESSetPrintLevel(solver, 2 ); /* print each GMRES
                                                         iteration */
         NALU_HYPRE_SStructGMRESSetLogging(solver, 1);

         /* use SysPFMG as precondititioner */
         NALU_HYPRE_SStructSysPFMGCreate(MPI_COMM_WORLD, &precond);

         /* Set sysPFMG parameters */
         NALU_HYPRE_SStructSysPFMGSetTol(precond, 0.0);
         NALU_HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
         NALU_HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
         NALU_HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
         NALU_HYPRE_SStructSysPFMGSetPrintLevel(precond, 0);
         NALU_HYPRE_SStructSysPFMGSetZeroGuess(precond);

         /* Set the preconditioner*/
         NALU_HYPRE_SStructGMRESSetPrecond(solver, NALU_HYPRE_SStructSysPFMGSolve,
                                      NALU_HYPRE_SStructSysPFMGSetup, precond);
         /* do the setup */
         NALU_HYPRE_SStructGMRESSetup(solver, A, b, x);

         /* do the solve */
         NALU_HYPRE_SStructGMRESSolve(solver, A, b, x);

         /* get some info */
         NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm(solver,
                                                        &final_res_norm);
         NALU_HYPRE_SStructGMRESGetNumIterations(solver, &its);

         /* clean up */
         NALU_HYPRE_SStructSysPFMGDestroy(precond);
         NALU_HYPRE_SStructGMRESDestroy(solver);
      }
      else if (solver_id == 1) /* SysPFMG */
      {
         NALU_HYPRE_SStructSysPFMGCreate(MPI_COMM_WORLD, &solver);

         /* Set sysPFMG parameters */
         NALU_HYPRE_SStructSysPFMGSetTol(solver, 1.0e-6);
         NALU_HYPRE_SStructSysPFMGSetMaxIter(solver, 50);
         NALU_HYPRE_SStructSysPFMGSetNumPreRelax(solver, n_pre);
         NALU_HYPRE_SStructSysPFMGSetNumPostRelax(solver, n_post);
         NALU_HYPRE_SStructSysPFMGSetPrintLevel(solver, 0);
         NALU_HYPRE_SStructSysPFMGSetLogging(solver, 1);

         /* do the setup */
         NALU_HYPRE_SStructSysPFMGSetup(solver, A, b, x);

         /* do the solve */
         NALU_HYPRE_SStructSysPFMGSolve(solver, A, b, x);

         /* get some info */
         NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(solver,
                                                          &final_res_norm);
         NALU_HYPRE_SStructSysPFMGGetNumIterations(solver, &its);

         /* clean up */
         NALU_HYPRE_SStructSysPFMGDestroy(solver);
      }
      else if (solver_id == 2) /* GMRES with AMG */
      {
         NALU_HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &par_solver);

         /* set the GMRES paramaters */
         NALU_HYPRE_GMRESSetKDim(par_solver, 5);
         NALU_HYPRE_GMRESSetMaxIter(par_solver, 100);
         NALU_HYPRE_GMRESSetTol(par_solver, 1.0e-06);
         NALU_HYPRE_GMRESSetPrintLevel(par_solver, 2);
         NALU_HYPRE_GMRESSetLogging(par_solver, 1);

         /* use BoomerAMG as preconditioner */
         NALU_HYPRE_BoomerAMGCreate(&par_precond);
         NALU_HYPRE_BoomerAMGSetCoarsenType(par_precond, 6);
         NALU_HYPRE_BoomerAMGSetOldDefault(par_precond);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         NALU_HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         NALU_HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         NALU_HYPRE_BoomerAMGSetPrintFileName(par_precond, "ex9.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(par_precond, 1);

         /* set the preconditioner */
         NALU_HYPRE_ParCSRGMRESSetPrecond(par_solver,
                                     NALU_HYPRE_BoomerAMGSolve,
                                     NALU_HYPRE_BoomerAMGSetup,
                                     par_precond);

         /* do the setup */
         NALU_HYPRE_ParCSRGMRESSetup(par_solver, par_A, par_b, par_x);

         /* do the solve */
         NALU_HYPRE_ParCSRGMRESSolve(par_solver, par_A, par_b, par_x);

         /* get some info */
         NALU_HYPRE_GMRESGetNumIterations(par_solver, &its);
         NALU_HYPRE_GMRESGetFinalRelativeResidualNorm(par_solver,
                                                 &final_res_norm);
         /* clean up */
         NALU_HYPRE_ParCSRGMRESDestroy(par_solver);
         NALU_HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 3) /* AMG */
      {
         NALU_HYPRE_BoomerAMGCreate(&par_solver);
         NALU_HYPRE_BoomerAMGSetCoarsenType(par_solver, 6);
         NALU_HYPRE_BoomerAMGSetOldDefault(par_solver);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(par_solver, 0.25);
         NALU_HYPRE_BoomerAMGSetTol(par_solver, 1.9e-6);
         NALU_HYPRE_BoomerAMGSetPrintLevel(par_solver, 1);
         NALU_HYPRE_BoomerAMGSetPrintFileName(par_solver, "ex9.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(par_solver, 50);

         /* do the setup */
         NALU_HYPRE_BoomerAMGSetup(par_solver, par_A, par_b, par_x);

         /* do the solve */
         NALU_HYPRE_BoomerAMGSolve(par_solver, par_A, par_b, par_x);

         /* get some info */
         NALU_HYPRE_BoomerAMGGetNumIterations(par_solver, &its);
         NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm(par_solver,
                                                     &final_res_norm);
         /* clean up */
         NALU_HYPRE_BoomerAMGDestroy(par_solver);
      }
      else
      {
         if (myid == 0) { printf("\n ERROR: Invalid solver id specified.\n"); }
      }

      /* Gather the solution vector.  This needs to be done if:
         (1) the  object  type is parcsr OR
         (2) any one of the variables is NOT cell-centered */
      if (object_type == NALU_HYPRE_PARCSR)
      {
         NALU_HYPRE_SStructVectorGather(x);
      }

      /* Save the solution for GLVis visualization, see vis/glvis-ex7.sh */
      if (vis)
      {
#ifdef NALU_HYPRE_EXVIS
         FILE *file;
         char filename[255];

         int k, part = 0, var;
         int nvalues = n * n;
         double *values = (double*) calloc(nvalues, sizeof(double));

         /* save local solution for variable u */
         var = 0;
         NALU_HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                         var, values);

         sprintf(filename, "%s.%06d", "vis/ex9-u.sol", myid);
         if ((file = fopen(filename, "w")) == NULL)
         {
            printf("Error: can't open output file %s\n", filename);
            MPI_Finalize();
            exit(1);
         }

         /* save solution with global unknown numbers */
         k = 0;
         for (j = 0; j < n; j++)
            for (i = 0; i < n; i++)
            {
               fprintf(file, "%06d %.14e\n", pj * N * n * n + pi * n + j * N * n + i, values[k++]);
            }

         fflush(file);
         fclose(file);

         /* save local solution for variable v */
         var = 1;
         NALU_HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                         var, values);

         sprintf(filename, "%s.%06d", "vis/ex9-v.sol", myid);
         if ((file = fopen(filename, "w")) == NULL)
         {
            printf("Error: can't open output file %s\n", filename);
            MPI_Finalize();
            exit(1);
         }

         /* save solution with global unknown numbers */
         k = 0;
         for (j = 0; j < n; j++)
            for (i = 0; i < n; i++)
            {
               fprintf(file, "%06d %.14e\n", pj * N * n * n + pi * n + j * N * n + i, values[k++]);
            }

         fflush(file);
         fclose(file);

         free(values);

         /* save global finite element mesh */
         if (myid == 0)
         {
            GLVis_PrintGlobalSquareMesh("vis/ex9.mesh", N * n - 1);
         }
#endif
      }

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", its);
         printf("Final Relative Residual Norm = %g\n", final_res_norm);
         printf("\n");
      }
   }

   /* Free memory */
   NALU_HYPRE_SStructGridDestroy(grid);
   NALU_HYPRE_SStructStencilDestroy(stencil_v);
   NALU_HYPRE_SStructStencilDestroy(stencil_u);
   NALU_HYPRE_SStructGraphDestroy(graph);
   NALU_HYPRE_SStructMatrixDestroy(A);
   NALU_HYPRE_SStructVectorDestroy(b);
   NALU_HYPRE_SStructVectorDestroy(x);

   /* Finalize HYPRE */
   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}
