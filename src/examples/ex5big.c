/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 5big

   Interface:    Linear-Algebraic (IJ)

   Compile with: make ex5big

   Sample run:   mpirun -np 4 ex5big

   Description:  This example is a slight modification of Example 5 that
                 illustrates the 64-bit integer support in hypre needed to run
                 problems with more than 2B unknowns.

                 Specifically, the changes compared to Example 5 are as follows:

                 1) All integer arguments to HYPRE functions should be declared
                    of type NALU_HYPRE_Int.

                 2) Variables of type NALU_HYPRE_Int are 64-bit integers, so they
                    should be printed in the %lld format (not %d).

                 To enable the 64-bit integer support, you need to build hypre
                 with the --enable-bigint option of the configure script.  We
                 recommend comparing this example with Example 5.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "NALU_HYPRE_krylov.h"
#include "NALU_HYPRE.h"
#include "NALU_HYPRE_parcsr_ls.h"

int hypre_FlexGMRESModifyPCAMGExample(void *precond_data, int iterations,
                                      double rel_residual_norm);

#define my_min(a,b)  (((a)<(b)) ? (a) : (b))

int main (int argc, char *argv[])
{
   NALU_HYPRE_Int i;
   int myid, num_procs;
   int N, n;

   NALU_HYPRE_Int ilower, iupper;
   NALU_HYPRE_Int local_size, extra;

   int solver_id;
   int print_system;

   double h, h2;

   NALU_HYPRE_IJMatrix A;
   NALU_HYPRE_ParCSRMatrix parcsr_A;
   NALU_HYPRE_IJVector b;
   NALU_HYPRE_ParVector par_b;
   NALU_HYPRE_IJVector x;
   NALU_HYPRE_ParVector par_x;

   NALU_HYPRE_Solver solver, precond;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   NALU_HYPRE_Init();

   /* Default problem parameters */
   n = 33;
   solver_id = 0;
   print_system = 0;


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
         else if ( strcmp(argv[arg_index], "-print_system") == 0 )
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

      if ((print_usage) && (myid == 0))
      {
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -n <n>              : problem size in each direction (default: 33)\n");
         printf("  -solver <ID>        : solver ID\n");
         printf("                        0  - AMG (default) \n");
         printf("                        1  - AMG-PCG\n");
         printf("                        8  - ParaSails-PCG\n");
         printf("                        50 - PCG\n");
         printf("                        61 - AMG-FlexGMRES\n");
         printf("  -print_system       : print the matrix and rhs\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Preliminaries: want at least one processor per row */
   if (n * n < num_procs) { n = sqrt(num_procs) + 1; }
   N = n * n; /* global number of rows */
   h = 1.0 / (n + 1); /* mesh size*/
   h2 = h * h;

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N / num_procs;
   extra = N - local_size * num_procs;

   ilower = local_size * myid;
   ilower += my_min(myid, extra);

   iupper = local_size * (myid + 1);
   iupper += my_min(myid + 1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   NALU_HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

   /* Choose a parallel csr format storage (see the User's Manual) */
   NALU_HYPRE_IJMatrixSetObjectType(A, NALU_HYPRE_PARCSR);

   /* Initialize before setting coefficients */
   NALU_HYPRE_IJMatrixInitialize(A);

   /* Now go through my local rows and set the matrix entries.
      Each row has at most 5 entries. For example, if n=3:

      A = [M -I 0; -I M -I; 0 -I M]
      M = [4 -1 0; -1 4 -1; 0 -1 4]

      Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
   */
   {
      NALU_HYPRE_Int nnz;
      double values[5];
      NALU_HYPRE_Int cols[5];

      for (i = ilower; i <= iupper; i++)
      {
         nnz = 0;

         /* The left identity block:position i-n */
         if ((i - n) >= 0)
         {
            cols[nnz] = i - n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The left -1: position i-1 */
         if (i % n)
         {
            cols[nnz] = i - 1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the diagonal: position i */
         cols[nnz] = i;
         values[nnz] = 4.0;
         nnz++;

         /* The right -1: position i+1 */
         if ((i + 1) % n)
         {
            cols[nnz] = i + 1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The right identity block:position i+n */
         if ((i + n) < N)
         {
            cols[nnz] = i + n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the values for row i */
         NALU_HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);
      }
   }

   /* Assemble after setting the coefficients */
   NALU_HYPRE_IJMatrixAssemble(A);

   /* Note: for the testing of small problems, one may wish to read
      in a matrix in IJ format (for the format, see the output files
      from the -print_system option).
      In this case, one would use the following routine:
      NALU_HYPRE_IJMatrixRead( <filename>, MPI_COMM_WORLD,
                          NALU_HYPRE_PARCSR, &A );
      <filename>  = IJ.A.out to read in what has been printed out
      by -print_system (processor numbers are omitted).
      A call to NALU_HYPRE_IJMatrixRead is an *alternative* to the
      following sequence of NALU_HYPRE_IJMatrix calls:
      Create, SetObjectType, Initialize, SetValues, and Assemble
   */


   /* Get the parcsr matrix object to use */
   NALU_HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);


   /* Create the rhs and solution */
   NALU_HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
   NALU_HYPRE_IJVectorSetObjectType(b, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(b);

   NALU_HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
   NALU_HYPRE_IJVectorSetObjectType(x, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(x);

   /* Set the rhs values to h^2 and the solution to zero */
   {
      double *rhs_values, *x_values;
      NALU_HYPRE_Int *rows;

      rhs_values = (double*) calloc(local_size, sizeof(double));
      x_values = (double*) calloc(local_size, sizeof(double));
      rows = (NALU_HYPRE_Int*) calloc(local_size, sizeof(NALU_HYPRE_Int));

      for (i = 0; i < local_size; i++)
      {
         rhs_values[i] = h2;
         x_values[i] = 0.0;
         rows[i] = ilower + i;
      }

      NALU_HYPRE_IJVectorSetValues(b, local_size, rows, rhs_values);
      NALU_HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

      free(x_values);
      free(rhs_values);
      free(rows);
   }


   NALU_HYPRE_IJVectorAssemble(b);
   /*  As with the matrix, for testing purposes, one may wish to read in a rhs:
       NALU_HYPRE_IJVectorRead( <filename>, MPI_COMM_WORLD,
                                 NALU_HYPRE_PARCSR, &b );
       as an alternative to the
       following sequence of NALU_HYPRE_IJVectors calls:
       Create, SetObjectType, Initialize, SetValues, and Assemble
   */
   NALU_HYPRE_IJVectorGetObject(b, (void **) &par_b);

   NALU_HYPRE_IJVectorAssemble(x);
   NALU_HYPRE_IJVectorGetObject(x, (void **) &par_x);


   /*  Print out the system  - files names will be IJ.out.A.XXXXX
        and IJ.out.b.XXXXX, where XXXXX = processor id */
   if (print_system)
   {
      NALU_HYPRE_IJMatrixPrint(A, "IJ.out.A");
      NALU_HYPRE_IJVectorPrint(b, "IJ.out.b");
   }


   /* Choose a solver and solve the system */

   /* AMG */
   if (solver_id == 0)
   {
      NALU_HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
      NALU_HYPRE_BoomerAMGCreate(&solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      NALU_HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
      NALU_HYPRE_BoomerAMGSetOldDefault(solver); /* Falgout coarsening with modified classical interpolation */
      NALU_HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
      NALU_HYPRE_BoomerAMGSetRelaxOrder(solver, 1);   /* Uses C/F relaxation */
      NALU_HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
      NALU_HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
      NALU_HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */

      /* Now setup and solve! */
      NALU_HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
      NALU_HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);

      /* Run info - needed logging turned on */
      NALU_HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      NALU_HYPRE_BoomerAMGDestroy(solver);
   }
   /* PCG */
   else if (solver_id == 50)
   {
      NALU_HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
      NALU_HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      NALU_HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      NALU_HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      NALU_HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      NALU_HYPRE_PCGSetPrintLevel(solver, 2); /* prints out the iteration info */
      NALU_HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      /* Now setup and solve! */
      NALU_HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      NALU_HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

      /* Run info - needed logging turned on */
      NALU_HYPRE_PCGGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      NALU_HYPRE_ParCSRPCGDestroy(solver);
   }
   /* PCG with AMG preconditioner */
   else if (solver_id == 1)
   {
      NALU_HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
      NALU_HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      NALU_HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      NALU_HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      NALU_HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      NALU_HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      NALU_HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      /* Now set up the AMG preconditioner and specify any parameters */
      NALU_HYPRE_BoomerAMGCreate(&precond);
      NALU_HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
      NALU_HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      NALU_HYPRE_BoomerAMGSetOldDefault(precond);
      NALU_HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      NALU_HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      NALU_HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      NALU_HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

      /* Set the PCG preconditioner */
      NALU_HYPRE_PCGSetPrecond(solver, (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                          (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup, precond);

      /* Now setup and solve! */
      NALU_HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      NALU_HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

      /* Run info - needed logging turned on */
      NALU_HYPRE_PCGGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver and preconditioner */
      NALU_HYPRE_ParCSRPCGDestroy(solver);
      NALU_HYPRE_BoomerAMGDestroy(precond);
   }
   /* PCG with Parasails Preconditioner */
   else if (solver_id == 8)
   {
      NALU_HYPRE_Int num_iterations;
      double final_res_norm;

      int      sai_max_levels = 1;
      double   sai_threshold = 0.1;
      double   sai_filter = 0.05;
      int      sai_sym = 1;

      /* Create solver */
      NALU_HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      NALU_HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      NALU_HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      NALU_HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      NALU_HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      NALU_HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      /* Now set up the ParaSails preconditioner and specify any parameters */
      NALU_HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);

      /* Set some parameters (See Reference Manual for more parameters) */
      NALU_HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
      NALU_HYPRE_ParaSailsSetFilter(precond, sai_filter);
      NALU_HYPRE_ParaSailsSetSym(precond, sai_sym);
      NALU_HYPRE_ParaSailsSetLogging(precond, 3);

      /* Set the PCG preconditioner */
      NALU_HYPRE_PCGSetPrecond(solver, (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSolve,
                          (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSetup, precond);

      /* Now setup and solve! */
      NALU_HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      NALU_HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);


      /* Run info - needed logging turned on */
      NALU_HYPRE_PCGGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      NALU_HYPRE_ParCSRPCGDestroy(solver);
      NALU_HYPRE_ParaSailsDestroy(precond);
   }
   /* Flexible GMRES with  AMG Preconditioner */
   else if (solver_id == 61)
   {
      NALU_HYPRE_Int num_iterations;
      double final_res_norm;
      int    restart = 30;
      int    modify = 1;


      /* Create solver */
      NALU_HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      NALU_HYPRE_FlexGMRESSetKDim(solver, restart);
      NALU_HYPRE_FlexGMRESSetMaxIter(solver, 1000); /* max iterations */
      NALU_HYPRE_FlexGMRESSetTol(solver, 1e-7); /* conv. tolerance */
      NALU_HYPRE_FlexGMRESSetPrintLevel(solver, 2); /* print solve info */
      NALU_HYPRE_FlexGMRESSetLogging(solver, 1); /* needed to get run info later */


      /* Now set up the AMG preconditioner and specify any parameters */
      NALU_HYPRE_BoomerAMGCreate(&precond);
      NALU_HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
      NALU_HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      NALU_HYPRE_BoomerAMGSetOldDefault(precond);
      NALU_HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      NALU_HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      NALU_HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      NALU_HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

      /* Set the FlexGMRES preconditioner */
      NALU_HYPRE_FlexGMRESSetPrecond(solver, (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup, precond);


      if (modify)
      {
         /* this is an optional call  - if you don't call it, hypre_FlexGMRESModifyPCDefault
            is used - which does nothing.  Otherwise, you can define your own, similar to
            the one used here */
         NALU_HYPRE_FlexGMRESSetModifyPC(
            solver, (NALU_HYPRE_PtrToModifyPCFcn) hypre_FlexGMRESModifyPCAMGExample);
      }


      /* Now setup and solve! */
      NALU_HYPRE_ParCSRFlexGMRESSetup(solver, parcsr_A, par_b, par_x);
      NALU_HYPRE_ParCSRFlexGMRESSolve(solver, parcsr_A, par_b, par_x);

      /* Run info - needed logging turned on */
      NALU_HYPRE_FlexGMRESGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      NALU_HYPRE_ParCSRFlexGMRESDestroy(solver);
      NALU_HYPRE_BoomerAMGDestroy(precond);

   }
   else
   {
      if (myid == 0) { printf("Invalid solver id specified.\n"); }
   }

   /* Clean up */
   NALU_HYPRE_IJMatrixDestroy(A);
   NALU_HYPRE_IJVectorDestroy(b);
   NALU_HYPRE_IJVectorDestroy(x);

   /* Finalize HYPRE */
   NALU_HYPRE_Finalize();

   /* Finalize MPI*/
   MPI_Finalize();

   return (0);
}

/*--------------------------------------------------------------------------
   hypre_FlexGMRESModifyPCAMGExample -

    This is an example (not recommended)
   of how we can modify things about AMG that
   affect the solve phase based on how FlexGMRES is doing...For
   another preconditioner it may make sense to modify the tolerance..

 *--------------------------------------------------------------------------*/

int hypre_FlexGMRESModifyPCAMGExample(void *precond_data, int iterations,
                                      double rel_residual_norm)
{


   if (rel_residual_norm > .1)
   {
      NALU_HYPRE_BoomerAMGSetNumSweeps((NALU_HYPRE_Solver)precond_data, 10);
   }
   else
   {
      NALU_HYPRE_BoomerAMGSetNumSweeps((NALU_HYPRE_Solver)precond_data, 1);
   }


   return 0;
}
