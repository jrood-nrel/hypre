/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 7 -- FORTRAN Test Version

   Interface:      SStructured interface (SStruct)

   Compile with:   make ex7

   Sample run:     mpirun -np 16 ex7_for -n 33 -solver 10 -K 3 -B 0 -C 1 -U0 2 -F 4

   To see options: ex7 -help

   Description:    This example uses the sstruct interface to solve the same
                   problem as was solved in Example 4 with the struct interface.
                   Therefore, there is only one part and one variable.

                   This code solves the convection-reaction-diffusion problem
                   div (-K grad u + B u) + C u = F in the unit square with
                   boundary condition u = U0.  The domain is split into N x N
                   processor grid.  Thus, the given number of processors should
                   be a perfect square. Each processor has a n x n grid, with
                   nodes connected by a 5-point stencil.  We use cell-centered
                   variables, and, therefore, the nodes are not shared.

                   To incorporate the boundary conditions, we do the following:
                   Let x_i and x_b be the interior and boundary parts of the
                   solution vector x. If we split the matrix A as
                             A = [A_ii A_ib; A_bi A_bb],
                   then we solve
                             [A_ii 0; 0 I] [x_i ; x_b] = [b_i - A_ib u_0; u_0].
                   Note that this differs from Example 3 in that we
                   are actually solving for the boundary conditions (so they
                   may not be exact as in ex3, where we only solved for the
                   interior).  This approach is useful for more general types
                   of b.c.

                   As in the previous example (Example 6), we use a structured
                   solver.  A number of structured solvers are available.
                   More information can be found in the Solvers and Preconditioners
                   chapter of the User's Manual.
*/

#include <math.h>
#include "_nalu_hypre_utilities.h"
#include "NALU_HYPRE_krylov.h"
#include "NALU_HYPRE_sstruct_ls.h"

#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif

/* Macro to evaluate a function F in the grid point (i,j) */
#define Eval(F,i,j) (F( (ilower[0]+(i))*h, (ilower[1]+(j))*h ))
#define bcEval(F,i,j) (F( (bc_ilower[0]+(i))*h, (bc_ilower[1]+(j))*h ))

#ifdef NALU_HYPRE_FORTRAN
#include "fortran.h"
#include "nalu_hypre_struct_fortran_test.h"
#include "nalu_hypre_sstruct_fortran_test.h"
#endif

NALU_HYPRE_Int optionK, optionB, optionC, optionU0, optionF;

/* Diffusion coefficient */
NALU_HYPRE_Real K(NALU_HYPRE_Real x, NALU_HYPRE_Real y)
{
   switch (optionK)
   {
      case 0:
         return 1.0;
      case 1:
         return x * x + exp(y);
      case 2:
         if ((fabs(x - 0.5) < 0.25) && (fabs(y - 0.5) < 0.25))
         {
            return 100.0;
         }
         else
         {
            return 1.0;
         }
      case 3:
         if (((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) < 0.0625)
         {
            return 10.0;
         }
         else
         {
            return 1.0;
         }
      default:
         return 1.0;
   }
}

/* Convection vector, first component */
NALU_HYPRE_Real B1(NALU_HYPRE_Real x, NALU_HYPRE_Real y)
{
   switch (optionB)
   {
      case 0:
         return 0.0;
      case 1:
         return -0.1;
      case 2:
         return 0.25;
      case 3:
         return 1.0;
      default:
         return 0.0;
   }
}

/* Convection vector, second component */
NALU_HYPRE_Real B2(NALU_HYPRE_Real x, NALU_HYPRE_Real y)
{
   switch (optionB)
   {
      case 0:
         return 0.0;
      case 1:
         return 0.1;
      case 2:
         return -0.25;
      case 3:
         return 1.0;
      default:
         return 0.0;
   }
}

/* Reaction coefficient */
NALU_HYPRE_Real C(NALU_HYPRE_Real x, NALU_HYPRE_Real y)
{
   switch (optionC)
   {
      case 0:
         return 0.0;
      case 1:
         return 10.0;
      case 2:
         return 100.0;
      default:
         return 0.0;
   }
}

/* Boundary condition */
NALU_HYPRE_Real U0(NALU_HYPRE_Real x, NALU_HYPRE_Real y)
{
   switch (optionU0)
   {
      case 0:
         return 0.0;
      case 1:
         return (x + y) / 100;
      case 2:
         return (nalu_hypre_sin(5 * PI * x) + nalu_hypre_sin(5 * PI * y)) / 1000;
      default:
         return 0.0;
   }
}

/* Right-hand side */
NALU_HYPRE_Real F(NALU_HYPRE_Real x, NALU_HYPRE_Real y)
{
   switch (optionF)
   {
      case 0:
         return 1.0;
      case 1:
         return 0.0;
      case 2:
         return 2 * PI * PI * nalu_hypre_sin(PI * x) * nalu_hypre_sin(PI * y);
      case 3:
         if ((fabs(x - 0.5) < 0.25) && (fabs(y - 0.5) < 0.25))
         {
            return -1.0;
         }
         else
         {
            return 1.0;
         }
      case 4:
         if (((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) < 0.0625)
         {
            return -1.0;
         }
         else
         {
            return 1.0;
         }
      default:
         return 1.0;
   }
}

NALU_HYPRE_Int main (NALU_HYPRE_Int argc, char *argv[])
{
   NALU_HYPRE_Int i, j, k;

   NALU_HYPRE_Int myid, num_procs;

   NALU_HYPRE_Int n, N, pi, pj;
   NALU_HYPRE_Real h, h2;
   NALU_HYPRE_Int ilower[2], iupper[2];

   NALU_HYPRE_Int solver_id;
   NALU_HYPRE_Int n_pre, n_post;
   NALU_HYPRE_Int rap, relax, skip, sym;
   NALU_HYPRE_Int time_index;

   NALU_HYPRE_Int object_type;

   NALU_HYPRE_Int num_iterations;
   NALU_HYPRE_Real final_res_norm;

   NALU_HYPRE_Int print_solution;

   /* We are using struct solvers for this example */
#ifdef NALU_HYPRE_FORTRAN
   nalu_hypre_F90_Obj grid;
   nalu_hypre_F90_Obj stencil;
   nalu_hypre_F90_Obj graph;
   nalu_hypre_F90_Obj A;
   nalu_hypre_F90_Obj b;
   nalu_hypre_F90_Obj x;

   nalu_hypre_F90_Obj solver;
   nalu_hypre_F90_Obj precond;
   NALU_HYPRE_Int      precond_id;

   nalu_hypre_F90_Obj long_temp_COMM;
   NALU_HYPRE_Int      temp_COMM;

   NALU_HYPRE_Int      zero = 0;
   NALU_HYPRE_Int      one = 1;
   NALU_HYPRE_Int      two = 2;
   NALU_HYPRE_Int      three = 3;
   NALU_HYPRE_Int      five = 5;
   NALU_HYPRE_Int      fifty = 50;
   NALU_HYPRE_Int      twohundred = 200;
   NALU_HYPRE_Int      fivehundred = 500;

   NALU_HYPRE_Real   zerodot = 0.;
   NALU_HYPRE_Real   tol = 1.e-6;
#else
   NALU_HYPRE_SStructGrid     grid;
   NALU_HYPRE_SStructStencil  stencil;
   NALU_HYPRE_SStructGraph    graph;
   NALU_HYPRE_SStructMatrix   A;
   NALU_HYPRE_SStructVector   b;
   NALU_HYPRE_SStructVector   x;

   NALU_HYPRE_StructSolver   solver;
   NALU_HYPRE_StructSolver   precond;
#endif

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid);
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs);

   /* Set default parameters */
   n         = 33;
   optionK   = 0;
   optionB   = 0;
   optionC   = 0;
   optionU0  = 0;
   optionF   = 0;
   solver_id = 10;
   n_pre     = 1;
   n_post    = 1;
   rap       = 0;
   relax     = 1;
   skip      = 0;
   sym       = 0;

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
         else if ( strcmp(argv[arg_index], "-K") == 0 )
         {
            arg_index++;
            optionK = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-B") == 0 )
         {
            arg_index++;
            optionB = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-C") == 0 )
         {
            arg_index++;
            optionC = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-U0") == 0 )
         {
            arg_index++;
            optionU0 = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-F") == 0 )
         {
            arg_index++;
            optionF = atoi(argv[arg_index++]);
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
         else if ( strcmp(argv[arg_index], "-rap") == 0 )
         {
            arg_index++;
            rap = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-relax") == 0 )
         {
            arg_index++;
            relax = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-skip") == 0 )
         {
            arg_index++;
            skip = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-sym") == 0 )
         {
            arg_index++;
            sym = atoi(argv[arg_index++]);
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
         nalu_hypre_printf("  -n  <n>             : problem size per processor (default: 8)\n");
         nalu_hypre_printf("  -K  <K>             : choice for the diffusion coefficient (default: 1.0)\n");
         nalu_hypre_printf("  -B  <B>             : choice for the convection vector (default: 0.0)\n");
         nalu_hypre_printf("  -C  <C>             : choice for the reaction coefficient (default: 0.0)\n");
         nalu_hypre_printf("  -U0 <U0>            : choice for the boundary condition (default: 0.0)\n");
         nalu_hypre_printf("  -F  <F>             : choice for the right-hand side (default: 1.0) \n");
         nalu_hypre_printf("  -solver <ID>        : solver ID\n");
         nalu_hypre_printf("                        0  - SMG \n");
         nalu_hypre_printf("                        1  - PFMG\n");
         nalu_hypre_printf("                        10 - CG with SMG precond (default)\n");
         nalu_hypre_printf("                        11 - CG with PFMG precond\n");
         nalu_hypre_printf("                        17 - CG with 2-step Jacobi\n");
         nalu_hypre_printf("                        18 - CG with diagonal scaling\n");
         nalu_hypre_printf("                        19 - CG\n");
         nalu_hypre_printf("                        30 - GMRES with SMG precond\n");
         nalu_hypre_printf("                        31 - GMRES with PFMG precond\n");
         nalu_hypre_printf("                        37 - GMRES with 2-step Jacobi\n");
         nalu_hypre_printf("                        38 - GMRES with diagonal scaling\n");
         nalu_hypre_printf("                        39 - GMRES\n");
         nalu_hypre_printf("  -v <n_pre> <n_post> : number of pre and post relaxations\n");
         nalu_hypre_printf("  -rap <r>            : coarse grid operator type\n");
         nalu_hypre_printf("                        0 - Galerkin (default)\n");
         nalu_hypre_printf("                        1 - non-Galerkin ParFlow operators\n");
         nalu_hypre_printf("                        2 - Galerkin, general operators\n");
         nalu_hypre_printf("  -relax <r>          : relaxation type\n");
         nalu_hypre_printf("                        0 - Jacobi\n");
         nalu_hypre_printf("                        1 - Weighted Jacobi (default)\n");
         nalu_hypre_printf("                        2 - R/B Gauss-Seidel\n");
         nalu_hypre_printf("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
         nalu_hypre_printf("  -skip <s>           : skip levels in PFMG (0 or 1)\n");
         nalu_hypre_printf("  -sym <s>            : symmetric storage (1) or not (0)\n");
         nalu_hypre_printf("  -print_solution     : print the solution vector\n");
         nalu_hypre_printf("\n");
      }

      if (print_usage)
      {
         nalu_hypre_MPI_Finalize();
         return (0);
      }
   }

   /* Convection produces non-symmetric matrices */
   if (optionB && sym)
   {
      optionB = 0;
   }

   /* Figure out the processor grid (N x N).  The local
      problem size is indicated by n (n x n). pi and pj
      indicate position in the processor grid. */
   N  = nalu_hypre_sqrt(num_procs);
   h  = 1.0 / (N * n - 1);
   h2 = h * h;
   pj = myid / N;
   pi = myid - pj * N;

   /* Define the nodes owned by the current processor (each processor's
      piece of the global grid) */
   ilower[0] = pi * n;
   ilower[1] = pj * n;
   iupper[0] = ilower[0] + n - 1;
   iupper[1] = ilower[1] + n - 1;

   /* 1. Set up a 2D grid */
   {
      NALU_HYPRE_Int ndim = 2;
      NALU_HYPRE_Int nparts = 1;
      NALU_HYPRE_Int nvars = 1;
      NALU_HYPRE_Int part = 0;
      NALU_HYPRE_Int i;

      /* Create an empty 2D grid object */
#ifdef NALU_HYPRE_FORTRAN
      temp_COMM = (NALU_HYPRE_Int) nalu_hypre_MPI_COMM_WORLD;
      long_temp_COMM = (nalu_hypre_F90_Obj) nalu_hypre_MPI_COMM_WORLD;
      NALU_HYPRE_SStructGridCreate(&temp_COMM, &ndim, &nparts, &grid);
#else
      NALU_HYPRE_SStructGridCreate(nalu_hypre_MPI_COMM_WORLD, ndim, nparts, &grid);
#endif

      /* Add a new box to the grid */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGridSetExtents(&grid, &part, &ilower[0], &iupper[0]);
#else
      NALU_HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
#endif

      /* Set the variable type for each part */
      {
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

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGridAssemble(&grid);
#else
      NALU_HYPRE_SStructGridAssemble(grid);
#endif
   }

   /* 2. Define the discretization stencil */
   {
      NALU_HYPRE_Int ndim = 2;
      NALU_HYPRE_Int var = 0;

      if (sym == 0)
      {
         /* Define the geometry of the stencil */
         NALU_HYPRE_Int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

         /* Create an empty 2D, 5-pt stencil object */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructStencilCreate(&ndim, &five, &stencil);
#else
         NALU_HYPRE_SStructStencilCreate(ndim, 5, &stencil);
#endif

         /* Assign stencil entries */
         for (i = 0; i < 5; i++)
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructStencilSetEntry(&stencil, &i, offsets[i], &var);
#else
            NALU_HYPRE_SStructStencilSetEntry(stencil, i, offsets[i], var);
#endif
      }
      else /* Symmetric storage */
      {
         /* Define the geometry of the stencil */
         NALU_HYPRE_Int offsets[3][2] = {{0, 0}, {1, 0}, {0, 1}};

         /* Create an empty 2D, 3-pt stencil object */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructStencilCreate(&ndim, &three, &stencil);
#else
         NALU_HYPRE_SStructStencilCreate(ndim, 3, &stencil);
#endif

         /* Assign stencil entries */
         for (i = 0; i < 3; i++)
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructStencilSetEntry(&stencil, &i, offsets[i], &var);
#else
            NALU_HYPRE_SStructStencilSetEntry(stencil, i, offsets[i], var);
#endif
      }
   }

   /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix */
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
         variable on each part (we only have one variable and one part)*/
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGraphSetStencil(&graph, &part, &var, &stencil);
#else
      NALU_HYPRE_SStructGraphSetStencil(graph, part, var, stencil);
#endif

      /* Here we could establish connections between parts if we
         had more than one part. */

      /* Assemble the graph */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGraphAssemble(&graph);
#else
      NALU_HYPRE_SStructGraphAssemble(graph);
#endif
   }

   /* 4. Set up SStruct Vectors for b and x */
   {
      NALU_HYPRE_Real *values;

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

      /* Set the object type (by default NALU_HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  If you want to use unstructured
         solvers, e.g. BoomerAMG, the object type should be NALU_HYPRE_PARCSR.
         If the problem is purely structured (with one part), you may want to use
         NALU_HYPRE_STRUCT to access the structured solvers. Here we have a purely
         structured example. */
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

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, (n * n), NALU_HYPRE_MEMORY_HOST);

      /* Set the values of b in left-to-right, bottom-to-top order */
      for (k = 0, j = 0; j < n; j++)
         for (i = 0; i < n; i++, k++)
         {
            values[k] = h2 * Eval(F, i, j);
         }
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructVectorSetBoxValues(&b, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
      NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
#endif

      /* Set x = 0 */
      for (i = 0; i < (n * n); i ++)
      {
         values[i] = 0.0;
      }
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructVectorSetBoxValues(&x, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
      NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
#endif

      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      /* Assembling is postponed since the vectors will be further modified */
   }

   /* 4. Set up a SStruct Matrix */
   {
      /* We have one part and one variable. */
      NALU_HYPRE_Int part = 0;
      NALU_HYPRE_Int var = 0;

      /* Create an empty matrix object */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixCreate(&temp_COMM, &graph, &A);
#else
      NALU_HYPRE_SStructMatrixCreate(nalu_hypre_MPI_COMM_WORLD, graph, &A);
#endif

      /* Use symmetric storage? The function below is for symmetric stencil entries
         (use NALU_HYPRE_SStructMatrixSetNSSymmetric for non-stencil entries) */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixSetSymmetric(&A, &part, &var, &var, &sym);
#else
      NALU_HYPRE_SStructMatrixSetSymmetric(A, part, var, var, sym);
#endif

      /* As with the vectors,  set the object type for the vectors
         to be the struct type */
      object_type = NALU_HYPRE_STRUCT;
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixSetObjectType(&A, &object_type);
#else
      NALU_HYPRE_SStructMatrixSetObjectType(A, object_type);
#endif

      /* Indicate that the matrix coefficients are ready to be set */
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixInitialize(&A);
#else
      NALU_HYPRE_SStructMatrixInitialize(A);
#endif

      /* Set the stencil values in the interior. Here we set the values
         at every node. We will modify the boundary nodes later. */
      if (sym == 0)
      {
         NALU_HYPRE_Int stencil_indices[5] = {0, 1, 2, 3, 4}; /* labels correspond
                                                      to the offsets */
         NALU_HYPRE_Real *values;

         values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 5 * (n * n), NALU_HYPRE_MEMORY_HOST);

         /* The order is left-to-right, bottom-to-top */
         for (k = 0, j = 0; j < n; j++)
            for (i = 0; i < n; i++, k += 5)
            {
               values[k + 1] = - Eval(K, i - 0.5, j) - Eval(B1, i - 0.5, j);

               values[k + 2] = - Eval(K, i + 0.5, j) + Eval(B1, i + 0.5, j);

               values[k + 3] = - Eval(K, i, j - 0.5) - Eval(B2, i, j - 0.5);

               values[k + 4] = - Eval(K, i, j + 0.5) + Eval(B2, i, j + 0.5);

               values[k] = h2 * Eval(C, i, j)
                           + Eval(K, i - 0.5, j) + Eval(K, i + 0.5, j)
                           + Eval(K, i, j - 0.5) + Eval(K, i, j + 0.5)
                           - Eval(B1, i - 0.5, j) + Eval(B1, i + 0.5, j)
                           - Eval(B2, i, j - 0.5) + Eval(B2, i, j + 0.5);
            }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                         &var, &five,
                                         &stencil_indices[0], &values[0]);
#else
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, 5,
                                         stencil_indices, values);
#endif

         nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
      }
      else /* Symmetric storage */
      {
         NALU_HYPRE_Int stencil_indices[3] = {0, 1, 2};
         NALU_HYPRE_Real *values;

         values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 3 * (n * n), NALU_HYPRE_MEMORY_HOST);

         /* The order is left-to-right, bottom-to-top */
         for (k = 0, j = 0; j < n; j++)
            for (i = 0; i < n; i++, k += 3)
            {
               values[k + 1] = - Eval(K, i + 0.5, j);
               values[k + 2] = - Eval(K, i, j + 0.5);
               values[k] = h2 * Eval(C, i, j)
                           + Eval(K, i + 0.5, j) + Eval(K, i, j + 0.5)
                           + Eval(K, i - 0.5, j) + Eval(K, i, j - 0.5);
            }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                         &var, &three,
                                         &stencil_indices[0], &values[0]);
#else
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, 3,
                                         stencil_indices, values);
#endif

         nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
      }
   }

   /* 5. Set the boundary conditions, while eliminating the coefficients
         reaching ouside of the domain boundary. We must modify the matrix
         stencil and the corresponding rhs entries. */
   {
      NALU_HYPRE_Int bc_ilower[2];
      NALU_HYPRE_Int bc_iupper[2];

      NALU_HYPRE_Int stencil_indices[5] = {0, 1, 2, 3, 4};
      NALU_HYPRE_Real *values, *bvalues;

      NALU_HYPRE_Int nentries;

      /* We have one part and one variable. */
      NALU_HYPRE_Int part = 0;
      NALU_HYPRE_Int var = 0;

      if (sym == 0)
      {
         nentries = 5;
      }
      else
      {
         nentries = 3;
      }

      values  = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nentries * n, NALU_HYPRE_MEMORY_HOST);
      bvalues = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST);

      /* The stencil at the boundary nodes is 1-0-0-0-0. Because
         we have I x_b = u_0; */
      for (i = 0; i < nentries * n; i += nentries)
      {
         values[i] = 1.0;
         for (j = 1; j < nentries; j++)
         {
            values[i + j] = 0.0;
         }
      }

      /* Processors at y = 0 */
      if (pj == 0)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         /* Modify the matrix */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &nentries,
                                         &stencil_indices[0], &values[0]);
#else
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
#endif

         /* Put the boundary conditions in b */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = bcEval(U0, i, 0);
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructVectorSetBoxValues(&b, &part, &bc_ilower[0],
                                         &bc_iupper[0], &var, &bvalues[0]);
#else
         NALU_HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower,
                                         bc_iupper, var, bvalues);
#endif
      }

      /* Processors at y = 1 */
      if (pj == N - 1)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n + n - 1;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         /* Modify the matrix */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &nentries,
                                         &stencil_indices[0], &values[0]);
#else
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
#endif

         /* Put the boundary conditions in b */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = bcEval(U0, i, 0);
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructVectorSetBoxValues(&b, &part, &bc_ilower[0],
                                         &bc_iupper[0], &var, &bvalues[0]);
#else
         NALU_HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower, bc_iupper, var, bvalues);
#endif
      }

      /* Processors at x = 0 */
      if (pi == 0)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         /* Modify the matrix */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &nentries,
                                         &stencil_indices[0], &values[0]);
#else
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
#endif

         /* Put the boundary conditions in b */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = bcEval(U0, 0, j);
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructVectorSetBoxValues(&b, &part, &bc_ilower[0],
                                         &bc_iupper[0], &var, &bvalues[0]);
#else
         NALU_HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower, bc_iupper,
                                         var, bvalues);
#endif
      }

      /* Processors at x = 1 */
      if (pi == N - 1)
      {
         bc_ilower[0] = pi * n + n - 1;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         /* Modify the matrix */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &nentries,
                                         &stencil_indices[0], &values[0]);
#else
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
#endif

         /* Put the boundary conditions in b */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = bcEval(U0, 0, j);
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructVectorSetBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &bvalues[0]);
#else
         NALU_HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower, bc_iupper,
                                         var, bvalues);
#endif
      }

      /* Recall that the system we are solving is:
         [A_ii 0; 0 I] [x_i ; x_b] = [b_i - A_ib u_0; u_0].
         This requires removing the connections between the interior
         and boundary nodes that we have set up when we set the
         5pt stencil at each node. We adjust for removing
         these connections by appropriately modifying the rhs.
         For the symm ordering scheme, just do the top and right
         boundary */

      /* Processors at y = 0, neighbors of boundary nodes */
      if (pj == 0)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n + 1;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 3;

         /* Modify the matrix */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = 0.0;
         }

         if (sym == 0)
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &bvalues[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                            var, 1,
                                            stencil_indices, bvalues);
#endif

         /* Eliminate the boundary conditions in b */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = bcEval(U0, i, -1) * (bcEval(K, i, -0.5) + bcEval(B2, i, -0.5));
         }

         if (pi == 0)
         {
            bvalues[0] = 0.0;
         }

         if (pi == N - 1)
         {
            bvalues[n - 1] = 0.0;
         }

         /* Note the use of AddToBoxValues (because we have already set values
            at these nodes) */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructVectorAddToBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                           &var, &bvalues[0]);
#else
         NALU_HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper,
                                           var, bvalues);
#endif
      }

      /* Processors at x = 0, neighbors of boundary nodes */
      if (pi == 0)
      {
         bc_ilower[0] = pi * n + 1;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         stencil_indices[0] = 1;

         /* Modify the matrix */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = 0.0;
         }

         if (sym == 0)
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &bvalues[0]);
#else
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                            var, 1,
                                            stencil_indices, bvalues);
#endif

         /* Eliminate the boundary conditions in b */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = bcEval(U0, -1, j) * (bcEval(K, -0.5, j) + bcEval(B1, -0.5, j));
         }

         if (pj == 0)
         {
            bvalues[0] = 0.0;
         }

         if (pj == N - 1)
         {
            bvalues[n - 1] = 0.0;
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructVectorAddToBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                           &var, &bvalues[0]);
#else
         NALU_HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper, var, bvalues);
#endif
      }

      /* Processors at y = 1, neighbors of boundary nodes */
      if (pj == N - 1)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n + (n - 1) - 1;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         if (sym == 0)
         {
            stencil_indices[0] = 4;
         }
         else
         {
            stencil_indices[0] = 2;
         }

         /* Modify the matrix */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = 0.0;
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &one,
                                         &stencil_indices[0], &bvalues[0]);
#else
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, var, 1,
                                         stencil_indices, bvalues);
#endif

         /* Eliminate the boundary conditions in b */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = bcEval(U0, i, 1) * (bcEval(K, i, 0.5) + bcEval(B2, i, 0.5));
         }

         if (pi == 0)
         {
            bvalues[0] = 0.0;
         }

         if (pi == N - 1)
         {
            bvalues[n - 1] = 0.0;
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructVectorAddToBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                           &var, &bvalues[0]);
#else
         NALU_HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper,
                                           var, bvalues);
#endif
      }

      /* Processors at x = 1, neighbors of boundary nodes */
      if (pi == N - 1)
      {
         bc_ilower[0] = pi * n + (n - 1) - 1;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         if (sym == 0)
         {
            stencil_indices[0] = 2;
         }
         else
         {
            stencil_indices[0] = 1;
         }

         /* Modify the matrix */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = 0.0;
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &one,
                                         &stencil_indices[0], &bvalues[0]);
#else
         NALU_HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, 1,
                                         stencil_indices, bvalues);
#endif

         /* Eliminate the boundary conditions in b */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = bcEval(U0, 1, j) * (bcEval(K, 0.5, j) + bcEval(B1, 0.5, j));
         }

         if (pj == 0)
         {
            bvalues[0] = 0.0;
         }

         if (pj == N - 1)
         {
            bvalues[n - 1] = 0.0;
         }

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructVectorAddToBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                           &var, &bvalues[0]);
#else
         NALU_HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper, var, bvalues);
#endif
      }

      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(bvalues, NALU_HYPRE_MEMORY_HOST);
   }

   /* Finalize the vector and matrix assembly */
#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructMatrixAssemble(&A);
   NALU_HYPRE_SStructVectorAssemble(&b);
   NALU_HYPRE_SStructVectorAssemble(&x);
#else
   NALU_HYPRE_SStructMatrixAssemble(A);
   NALU_HYPRE_SStructVectorAssemble(b);
   NALU_HYPRE_SStructVectorAssemble(x);
#endif

   /* 6. Set up and use a solver */
   {
#ifdef NALU_HYPRE_FORTRAN
      nalu_hypre_F90_Obj sA;
      nalu_hypre_F90_Obj sb;
      nalu_hypre_F90_Obj sx;
#else
      NALU_HYPRE_StructMatrix    sA;
      NALU_HYPRE_StructVector    sb;
      NALU_HYPRE_StructVector    sx;
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

      if (solver_id == 0) /* SMG */
      {
         /* Start timing */
         time_index = nalu_hypre_InitializeTiming("SMG Setup");
         nalu_hypre_BeginTiming(time_index);

         /* Options and setup */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructSMGCreate(&temp_COMM, &solver);
         NALU_HYPRE_StructSMGSetMemoryUse(&solver, &zero);
         NALU_HYPRE_StructSMGSetMaxIter(&solver, &fifty);
         NALU_HYPRE_StructSMGSetTol(&solver, &tol);
         NALU_HYPRE_StructSMGSetRelChange(&solver, &zero);
         NALU_HYPRE_StructSMGSetNumPreRelax(&solver, &n_pre);
         NALU_HYPRE_StructSMGSetNumPostRelax(&solver, &n_post);
         NALU_HYPRE_StructSMGSetPrintLevel(&solver, &one);
         NALU_HYPRE_StructSMGSetLogging(&solver, &one);
         NALU_HYPRE_StructSMGSetup(&solver, &sA, &sb, &sx);
#else
         NALU_HYPRE_StructSMGCreate(nalu_hypre_MPI_COMM_WORLD, &solver);
         NALU_HYPRE_StructSMGSetMemoryUse(solver, 0);
         NALU_HYPRE_StructSMGSetMaxIter(solver, 50);
         NALU_HYPRE_StructSMGSetTol(solver, 1.0e-06);
         NALU_HYPRE_StructSMGSetRelChange(solver, 0);
         NALU_HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
         NALU_HYPRE_StructSMGSetNumPostRelax(solver, n_post);
         NALU_HYPRE_StructSMGSetPrintLevel(solver, 1);
         NALU_HYPRE_StructSMGSetLogging(solver, 1);
         NALU_HYPRE_StructSMGSetup(solver, sA, sb, sx);
#endif

         /* Finalize current timing */
         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         /* Start timing again */
         time_index = nalu_hypre_InitializeTiming("SMG Solve");
         nalu_hypre_BeginTiming(time_index);

         /* Solve */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructSMGSolve(&solver, &sA, &sb, &sx);
#else
         NALU_HYPRE_StructSMGSolve(solver, sA, sb, sx);
#endif
         nalu_hypre_EndTiming(time_index);
         /* Finalize current timing */

         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         /* Get info and release memory */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructSMGGetNumIterations(&solver, &num_iterations);
         NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm(&solver, &final_res_norm);
         NALU_HYPRE_StructSMGDestroy(&solver);
#else
         NALU_HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
         NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         NALU_HYPRE_StructSMGDestroy(solver);
#endif
      }

      if (solver_id == 1) /* PFMG */
      {
         /* Start timing */
         time_index = nalu_hypre_InitializeTiming("PFMG Setup");
         nalu_hypre_BeginTiming(time_index);

         /* Options and setup */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructPFMGCreate(&temp_COMM, &solver);
         NALU_HYPRE_StructPFMGSetMaxIter(&solver, &fifty);
         NALU_HYPRE_StructPFMGSetTol(&solver, &tol);
         NALU_HYPRE_StructPFMGSetRelChange(&solver, &zero);
         NALU_HYPRE_StructPFMGSetRAPType(&solver, &rap);
         NALU_HYPRE_StructPFMGSetRelaxType(&solver, &relax);
         NALU_HYPRE_StructPFMGSetNumPreRelax(&solver, &n_pre);
         NALU_HYPRE_StructPFMGSetNumPostRelax(&solver, &n_post);
         NALU_HYPRE_StructPFMGSetSkipRelax(&solver, &skip);
         NALU_HYPRE_StructPFMGSetPrintLevel(&solver, &one);
         NALU_HYPRE_StructPFMGSetLogging(&solver, &one);
         NALU_HYPRE_StructPFMGSetup(&solver, &sA, &sb, &sx);
#else
         NALU_HYPRE_StructPFMGCreate(nalu_hypre_MPI_COMM_WORLD, &solver);
         NALU_HYPRE_StructPFMGSetMaxIter(solver, 50);
         NALU_HYPRE_StructPFMGSetTol(solver, 1.0e-06);
         NALU_HYPRE_StructPFMGSetRelChange(solver, 0);
         NALU_HYPRE_StructPFMGSetRAPType(solver, rap);
         NALU_HYPRE_StructPFMGSetRelaxType(solver, relax);
         NALU_HYPRE_StructPFMGSetNumPreRelax(solver, n_pre);
         NALU_HYPRE_StructPFMGSetNumPostRelax(solver, n_post);
         NALU_HYPRE_StructPFMGSetSkipRelax(solver, skip);
         NALU_HYPRE_StructPFMGSetPrintLevel(solver, 1);
         NALU_HYPRE_StructPFMGSetLogging(solver, 1);
         NALU_HYPRE_StructPFMGSetup(solver, sA, sb, sx);
#endif

         /* Finalize current timing */
         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         /* Start timing again */
         time_index = nalu_hypre_InitializeTiming("PFMG Solve");
         nalu_hypre_BeginTiming(time_index);

         /* Solve */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructPFMGSolve(&solver, &sA, &sb, &sx);
#else
         NALU_HYPRE_StructPFMGSolve(solver, sA, sb, sx);
#endif

         /* Finalize current timing */
         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         /* Get info and release memory */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructPFMGGetNumIterations(&solver, &num_iterations);
         NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm(&solver, &final_res_norm);
         NALU_HYPRE_StructPFMGDestroy(&solver);
#else
         NALU_HYPRE_StructPFMGGetNumIterations(solver, &num_iterations);
         NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         NALU_HYPRE_StructPFMGDestroy(solver);
#endif
      }

      /* Preconditioned CG */
      if ((solver_id > 9) && (solver_id < 20))
      {
         time_index = nalu_hypre_InitializeTiming("PCG Setup");
         nalu_hypre_BeginTiming(time_index);

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructPCGCreate(&temp_COMM, &solver);
         NALU_HYPRE_StructPCGSetMaxIter(&solver, &twohundred );
         NALU_HYPRE_StructPCGSetTol(&solver, &tol );
         NALU_HYPRE_StructPCGSetTwoNorm(&solver, &one );
         NALU_HYPRE_StructPCGSetRelChange(&solver, &zero );
         NALU_HYPRE_StructPCGSetPrintLevel(&solver, &two );
#else
         NALU_HYPRE_StructPCGCreate(nalu_hypre_MPI_COMM_WORLD, &solver);
         NALU_HYPRE_StructPCGSetMaxIter(solver, 200 );
         NALU_HYPRE_StructPCGSetTol(solver, 1.0e-06 );
         NALU_HYPRE_StructPCGSetTwoNorm(solver, 1 );
         NALU_HYPRE_StructPCGSetRelChange(solver, 0 );
         NALU_HYPRE_StructPCGSetPrintLevel(solver, 2 );
#endif

         if (solver_id == 10)
         {
            /* use symmetric SMG as preconditioner */
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructSMGCreate(&temp_COMM, &precond);
            NALU_HYPRE_StructSMGSetMemoryUse(&precond, &zero);
            NALU_HYPRE_StructSMGSetMaxIter(&precond, &one);
            NALU_HYPRE_StructSMGSetTol(&precond, &zerodot);
            NALU_HYPRE_StructSMGSetZeroGuess(&precond);
            NALU_HYPRE_StructSMGSetNumPreRelax(&precond, &n_pre);
            NALU_HYPRE_StructSMGSetNumPostRelax(&precond, &n_post);
            NALU_HYPRE_StructSMGSetPrintLevel(&precond, &zero);
            NALU_HYPRE_StructSMGSetLogging(&precond, &zero);
            precond_id = 0;
            NALU_HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
#else
            NALU_HYPRE_StructSMGCreate(nalu_hypre_MPI_COMM_WORLD, &precond);
            NALU_HYPRE_StructSMGSetMemoryUse(precond, 0);
            NALU_HYPRE_StructSMGSetMaxIter(precond, 1);
            NALU_HYPRE_StructSMGSetTol(precond, 0.0);
            NALU_HYPRE_StructSMGSetZeroGuess(precond);
            NALU_HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            NALU_HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            NALU_HYPRE_StructSMGSetPrintLevel(precond, 0);
            NALU_HYPRE_StructSMGSetLogging(precond, 0);
            NALU_HYPRE_StructPCGSetPrecond(solver,
                                      NALU_HYPRE_StructSMGSolve,
                                      NALU_HYPRE_StructSMGSetup,
                                      precond);
#endif
         }

         else if (solver_id == 11)
         {
            /* use symmetric PFMG as preconditioner */
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructPFMGCreate(&temp_COMM, &precond);
            NALU_HYPRE_StructPFMGSetMaxIter(&precond, &one);
            NALU_HYPRE_StructPFMGSetTol(&precond, &zerodot);
            NALU_HYPRE_StructPFMGSetZeroGuess(&precond);
            NALU_HYPRE_StructPFMGSetRAPType(&precond, &rap);
            NALU_HYPRE_StructPFMGSetRelaxType(&precond, &relax);
            NALU_HYPRE_StructPFMGSetNumPreRelax(&precond, &n_pre);
            NALU_HYPRE_StructPFMGSetNumPostRelax(&precond, &n_post);
            NALU_HYPRE_StructPFMGSetSkipRelax(&precond, &skip);
            NALU_HYPRE_StructPFMGSetPrintLevel(&precond, &zero);
            NALU_HYPRE_StructPFMGSetLogging(&precond, &zero);
            precond_id = 1;
            NALU_HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
#else
            NALU_HYPRE_StructPFMGCreate(nalu_hypre_MPI_COMM_WORLD, &precond);
            NALU_HYPRE_StructPFMGSetMaxIter(precond, 1);
            NALU_HYPRE_StructPFMGSetTol(precond, 0.0);
            NALU_HYPRE_StructPFMGSetZeroGuess(precond);
            NALU_HYPRE_StructPFMGSetRAPType(precond, rap);
            NALU_HYPRE_StructPFMGSetRelaxType(precond, relax);
            NALU_HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            NALU_HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            NALU_HYPRE_StructPFMGSetSkipRelax(precond, skip);
            NALU_HYPRE_StructPFMGSetPrintLevel(precond, 0);
            NALU_HYPRE_StructPFMGSetLogging(precond, 0);
            NALU_HYPRE_StructPCGSetPrecond(solver,
                                      NALU_HYPRE_StructPFMGSolve,
                                      NALU_HYPRE_StructPFMGSetup,
                                      precond);
#endif
         }

         else if (solver_id == 17)
         {
            /* use two-step Jacobi as preconditioner */
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructJacobiCreate(&temp_COMM, &precond);
            NALU_HYPRE_StructJacobiSetMaxIter(&precond, &two);
            NALU_HYPRE_StructJacobiSetTol(&precond, &zerodot);
            NALU_HYPRE_StructJacobiSetZeroGuess(&precond);
            precond_id = 7;
            NALU_HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
#else
            NALU_HYPRE_StructJacobiCreate(nalu_hypre_MPI_COMM_WORLD, &precond);
            NALU_HYPRE_StructJacobiSetMaxIter(precond, 2);
            NALU_HYPRE_StructJacobiSetTol(precond, 0.0);
            NALU_HYPRE_StructJacobiSetZeroGuess(precond);
            NALU_HYPRE_StructPCGSetPrecond( solver,
                                       NALU_HYPRE_StructJacobiSolve,
                                       NALU_HYPRE_StructJacobiSetup,
                                       precond);
#endif
         }

         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
#ifdef NALU_HYPRE_FORTRAN
            precond_id = 8;
            NALU_HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
#else
            precond = NULL;
            NALU_HYPRE_StructPCGSetPrecond(solver,
                                      NALU_HYPRE_StructDiagScale,
                                      NALU_HYPRE_StructDiagScaleSetup,
                                      precond);
#endif
         }

         /* PCG Setup */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructPCGSetup(&solver, &sA, &sb, &sx );
#else
         NALU_HYPRE_StructPCGSetup(solver, sA, sb, sx );
#endif

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         time_index = nalu_hypre_InitializeTiming("PCG Solve");
         nalu_hypre_BeginTiming(time_index);

         /* PCG Solve */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructPCGSolve(&solver, &sA, &sb, &sx );
#else
         NALU_HYPRE_StructPCGSolve(solver, sA, sb, sx);
#endif

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         /* Get info and release memory */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructPCGGetNumIterations(&solver, &num_iterations );
         NALU_HYPRE_StructPCGGetFinalRelativeResidualNorm(&solver, &final_res_norm );
         NALU_HYPRE_StructPCGDestroy(&solver);
#else
         NALU_HYPRE_StructPCGGetNumIterations( solver, &num_iterations );
         NALU_HYPRE_StructPCGGetFinalRelativeResidualNorm( solver, &final_res_norm );
         NALU_HYPRE_StructPCGDestroy(solver);
#endif

         if (solver_id == 10)
         {
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructSMGDestroy(&precond);
#else
            NALU_HYPRE_StructSMGDestroy(precond);
#endif
         }
         else if (solver_id == 11 )
         {
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructPFMGDestroy(&precond);
#else
            NALU_HYPRE_StructPFMGDestroy(precond);
#endif
         }
         else if (solver_id == 17)
         {
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructJacobiDestroy(&precond);
#else
            NALU_HYPRE_StructJacobiDestroy(precond);
#endif
         }
      }

      /* Preconditioned GMRES */
      if ((solver_id > 29) && (solver_id < 40))
      {
         time_index = nalu_hypre_InitializeTiming("GMRES Setup");
         nalu_hypre_BeginTiming(time_index);

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructGMRESCreate(&temp_COMM, &solver);
#else
         NALU_HYPRE_StructGMRESCreate(nalu_hypre_MPI_COMM_WORLD, &solver);
#endif

         /* Note that GMRES can be used with all the interfaces - not
            just the struct.  So here we demonstrate the
            more generic GMRES interface functions. Since we have chosen
            a struct solver then we must type cast to the more generic
            NALU_HYPRE_Solver when setting options with these generic functions.
            Note that one could declare the solver to be
            type NALU_HYPRE_Solver, and then the casting would not be necessary.*/

         /*  Using struct GMRES routines to test FORTRAN Interface --3/3/2006  */

#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructGMRESSetMaxIter(&solver, &fivehundred );
         NALU_HYPRE_StructGMRESSetTol(&solver, &tol );
         NALU_HYPRE_StructGMRESSetPrintLevel(&solver, &two );
         NALU_HYPRE_StructGMRESSetLogging(&solver, &one );
#else
         NALU_HYPRE_StructGMRESSetMaxIter(solver, 500 );
         NALU_HYPRE_StructGMRESSetTol(solver, 1.0e-6 );
         NALU_HYPRE_StructGMRESSetPrintLevel(solver, 2 );
         NALU_HYPRE_StructGMRESSetLogging(solver, 1 );
#endif

         if (solver_id == 30)
         {
            /* use symmetric SMG as preconditioner */
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructSMGCreate(&temp_COMM, &precond);
            NALU_HYPRE_StructSMGSetMemoryUse(&precond, &zero);
            NALU_HYPRE_StructSMGSetMaxIter(&precond, &one);
            NALU_HYPRE_StructSMGSetTol(&precond, &zerodot);
            NALU_HYPRE_StructSMGSetZeroGuess(&precond);
            NALU_HYPRE_StructSMGSetNumPreRelax(&precond, &n_pre);
            NALU_HYPRE_StructSMGSetNumPostRelax(&precond, &n_post);
            NALU_HYPRE_StructSMGSetPrintLevel(&precond, &zero);
            NALU_HYPRE_StructSMGSetLogging(&precond, &zero);
            precond_id = 0;
            NALU_HYPRE_StructGMRESSetPrecond(&solver, &precond_id, &precond);
#else
            NALU_HYPRE_StructSMGCreate(nalu_hypre_MPI_COMM_WORLD, &precond);
            NALU_HYPRE_StructSMGSetMemoryUse(precond, 0);
            NALU_HYPRE_StructSMGSetMaxIter(precond, 1);
            NALU_HYPRE_StructSMGSetTol(precond, 0.0);
            NALU_HYPRE_StructSMGSetZeroGuess(precond);
            NALU_HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            NALU_HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            NALU_HYPRE_StructSMGSetPrintLevel(precond, 0);
            NALU_HYPRE_StructSMGSetLogging(precond, 0);
            NALU_HYPRE_StructGMRESSetPrecond(solver,
                                        NALU_HYPRE_StructSMGSolve,
                                        NALU_HYPRE_StructSMGSetup,
                                        precond);
#endif
         }

         else if (solver_id == 31)
         {
            /* use symmetric PFMG as preconditioner */
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructPFMGCreate(&temp_COMM, &precond);
            NALU_HYPRE_StructPFMGSetMaxIter(&precond, &one);
            NALU_HYPRE_StructPFMGSetTol(&precond, &zerodot);
            NALU_HYPRE_StructPFMGSetZeroGuess(&precond);
            NALU_HYPRE_StructPFMGSetRAPType(&precond, &rap);
            NALU_HYPRE_StructPFMGSetRelaxType(&precond, &relax);
            NALU_HYPRE_StructPFMGSetNumPreRelax(&precond, &n_pre);
            NALU_HYPRE_StructPFMGSetNumPostRelax(&precond, &n_post);
            NALU_HYPRE_StructPFMGSetSkipRelax(&precond, &skip);
            NALU_HYPRE_StructPFMGSetPrintLevel(&precond, &zero);
            NALU_HYPRE_StructPFMGSetLogging(&precond, &zero);
            precond_id = 1;
            NALU_HYPRE_StructGMRESSetPrecond(&solver, &precond_id, &precond);
#else
            NALU_HYPRE_StructPFMGCreate(nalu_hypre_MPI_COMM_WORLD, &precond);
            NALU_HYPRE_StructPFMGSetMaxIter(precond, 1);
            NALU_HYPRE_StructPFMGSetTol(precond, 0.0);
            NALU_HYPRE_StructPFMGSetZeroGuess(precond);
            NALU_HYPRE_StructPFMGSetRAPType(precond, rap);
            NALU_HYPRE_StructPFMGSetRelaxType(precond, relax);
            NALU_HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            NALU_HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            NALU_HYPRE_StructPFMGSetSkipRelax(precond, skip);
            NALU_HYPRE_StructPFMGSetPrintLevel(precond, 0);
            NALU_HYPRE_StructPFMGSetLogging(precond, 0);
            NALU_HYPRE_StructGMRESSetPrecond( solver,
                                         NALU_HYPRE_StructPFMGSolve,
                                         NALU_HYPRE_StructPFMGSetup,
                                         precond);
#endif
         }

         else if (solver_id == 37)
         {
            /* use two-step Jacobi as preconditioner */
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructJacobiCreate(&temp_COMM, &precond);
            NALU_HYPRE_StructJacobiSetMaxIter(&precond, &two);
            NALU_HYPRE_StructJacobiSetTol(&precond, &zerodot);
            NALU_HYPRE_StructJacobiSetZeroGuess(&precond);
            precond_id = 7;
            NALU_HYPRE_StructGMRESSetPrecond(&solver, &precond_id, &precond);
#else
            NALU_HYPRE_StructJacobiCreate(nalu_hypre_MPI_COMM_WORLD, &precond);
            NALU_HYPRE_StructJacobiSetMaxIter(precond, 2);
            NALU_HYPRE_StructJacobiSetTol(precond, 0.0);
            NALU_HYPRE_StructJacobiSetZeroGuess(precond);
            NALU_HYPRE_StructGMRESSetPrecond( solver,
                                         NALU_HYPRE_StructJacobiSolve,
                                         NALU_HYPRE_StructJacobiSetup,
                                         precond);
#endif
         }

         else if (solver_id == 38)
         {
            /* use diagonal scaling as preconditioner */
#ifdef NALU_HYPRE_FORTRAN
            precond_id = 8;
            NALU_HYPRE_StructGMRESSetPrecond(&solver, &precond_id, &precond);
#else
            precond = NULL;
            NALU_HYPRE_StructGMRESSetPrecond( solver,
                                         NALU_HYPRE_StructDiagScale,
                                         NALU_HYPRE_StructDiagScaleSetup,
                                         precond);
#endif
         }

         /* GMRES Setup */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructGMRESSetup(&solver, &sA, &sb, &sx );
#else
         NALU_HYPRE_StructGMRESSetup(solver, sA, sb, sx );
#endif

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         time_index = nalu_hypre_InitializeTiming("GMRES Solve");
         nalu_hypre_BeginTiming(time_index);

         /* GMRES Solve */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructGMRESSolve(&solver, &sA, &sb, &sx );
#else
         NALU_HYPRE_StructGMRESSolve(solver, sA, sb, sx);
#endif

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         /* Get info and release memory */
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_StructGMRESGetNumIterations(&solver, &num_iterations);
         NALU_HYPRE_StructGMRESGetFinalRelativeResidualNorm(&solver, &final_res_norm);
         NALU_HYPRE_StructGMRESDestroy(&solver);
#else
         NALU_HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
         NALU_HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         NALU_HYPRE_StructGMRESDestroy(solver);
#endif

         if (solver_id == 30)
         {
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructSMGDestroy(&precond);
#else
            NALU_HYPRE_StructSMGDestroy(precond);
#endif
         }
         else if (solver_id == 31)
         {
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructPFMGDestroy(&precond);
#else
            NALU_HYPRE_StructPFMGDestroy(precond);
#endif
         }
         else if (solver_id == 37)
         {
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_StructJacobiDestroy(&precond);
#else
            NALU_HYPRE_StructJacobiDestroy(precond);
#endif
         }
      }

   }

   /* Print the solution and other info */
   if (print_solution)
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructVectorPrint("sstruct.out.x", &x, &zero);
#else
      NALU_HYPRE_SStructVectorPrint("sstruct.out.x", x, 0);
#endif

   if (myid == 0)
   {
      nalu_hypre_printf("\n");
      nalu_hypre_printf("Iterations = %d\n", num_iterations);
      nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      nalu_hypre_printf("\n");
   }

   /* Free memory */
#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructGridDestroy(&grid);
   NALU_HYPRE_SStructStencilDestroy(&stencil);
   NALU_HYPRE_SStructGraphDestroy(&graph);
   NALU_HYPRE_SStructMatrixDestroy(&A);
   NALU_HYPRE_SStructVectorDestroy(&b);
   NALU_HYPRE_SStructVectorDestroy(&x);
#else
   NALU_HYPRE_SStructGridDestroy(grid);
   NALU_HYPRE_SStructStencilDestroy(stencil);
   NALU_HYPRE_SStructGraphDestroy(graph);
   NALU_HYPRE_SStructMatrixDestroy(A);
   NALU_HYPRE_SStructVectorDestroy(b);
   NALU_HYPRE_SStructVectorDestroy(x);
#endif

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   return (0);
}
