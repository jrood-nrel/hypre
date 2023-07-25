/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "NALU_HYPRE_parcsr_mv.h"

#include "NALU_HYPRE_IJ_mv.h"
#include "NALU_HYPRE_parcsr_ls.h"
#include "NALU_HYPRE_LinSysCore.h"

NALU_HYPRE_Int BuildParLaplacian27pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix *A_ptr );

#define SECOND_TIME 0

NALU_HYPRE_Int main( NALU_HYPRE_Int   argc, char *argv[] )
{
   NALU_HYPRE_Int                 arg_index;
   NALU_HYPRE_Int                 print_usage;
   NALU_HYPRE_Int                 build_matrix_arg_index;
   NALU_HYPRE_Int                 solver_id;
   NALU_HYPRE_Int                 ierr, i, j;
   NALU_HYPRE_Int                 num_iterations;

   NALU_HYPRE_ParCSRMatrix  parcsr_A;
   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 local_row;
   NALU_HYPRE_Int             time_index;
   MPI_Comm            comm;
   NALU_HYPRE_Int                 M, N;
   NALU_HYPRE_Int                 first_local_row, last_local_row;
   NALU_HYPRE_Int                 first_local_col, last_local_col;
   NALU_HYPRE_Int                 size, *col_ind;
   NALU_HYPRE_Real          *values;

   /* parameters for BoomerAMG */
   NALU_HYPRE_Real          strong_threshold;
   NALU_HYPRE_Int                 num_grid_sweeps;
   NALU_HYPRE_Real          relax_weight;

   /* parameters for GMRES */
   NALU_HYPRE_Int                  k_dim;

   char *paramString = new char[100];

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Init(&argc, &argv);
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   build_matrix_arg_index = argc;
   solver_id              = 0;
   strong_threshold       = 0.25;
   num_grid_sweeps        = 2;
   relax_weight           = 0.5;
   k_dim                  = 20;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
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
      nalu_hypre_printf("  -solver <ID>           : solver ID\n");
      nalu_hypre_printf("       0=DS-PCG      1=ParaSails-PCG \n");
      nalu_hypre_printf("       2=AMG-PCG     3=DS-GMRES     \n");
      nalu_hypre_printf("       4=PILUT-GMRES 5=AMG-GMRES    \n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -rlx <val>             : relaxation type\n");
      nalu_hypre_printf("       0=Weighted Jacobi  \n");
      nalu_hypre_printf("       1=Gauss-Seidel (very slow!)  \n");
      nalu_hypre_printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
      nalu_hypre_printf("\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("Running with these driver parameters:\n");
      nalu_hypre_printf("  solver ID    = %d\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   strcpy(paramString, "LS Interface");
   time_index = nalu_hypre_InitializeTiming(paramString);
   nalu_hypre_BeginTiming(time_index);

   BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);

   /*-----------------------------------------------------------
    * Copy the parcsr matrix into the LSI through interface calls
    *-----------------------------------------------------------*/

   ierr = NALU_HYPRE_ParCSRMatrixGetComm( parcsr_A, &comm );
   ierr += NALU_HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );
   ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                           &first_local_row, &last_local_row,
                                           &first_local_col, &last_local_col );

   NALU_HYPRE_LinSysCore H(nalu_hypre_MPI_COMM_WORLD);
   NALU_HYPRE_Int numLocalEqns = last_local_row - first_local_row + 1;
   H.createMatricesAndVectors(M, first_local_row + 1, numLocalEqns);

   NALU_HYPRE_Int index;
   NALU_HYPRE_Int *rowLengths = new NALU_HYPRE_Int[numLocalEqns];
   NALU_HYPRE_Int **colIndices = new NALU_HYPRE_Int*[numLocalEqns];

   local_row = 0;
   for (i = first_local_row; i <= last_local_row; i++)
   {
      ierr += NALU_HYPRE_ParCSRMatrixGetRow(parcsr_A, i, &size, &col_ind, &values );
      rowLengths[local_row] = size;
      colIndices[local_row] = new NALU_HYPRE_Int[size];
      for (j = 0; j < size; j++) { colIndices[local_row][j] = col_ind[j] + 1; }
      local_row++;
      NALU_HYPRE_ParCSRMatrixRestoreRow(parcsr_A, i, &size, &col_ind, &values);
   }
   H.allocateMatrix(colIndices, rowLengths);
   delete [] rowLengths;
   for (i = 0; i < numLocalEqns; i++) { delete [] colIndices[i]; }
   delete [] colIndices;

   NALU_HYPRE_Int *newColInd;

   for (i = first_local_row; i <= last_local_row; i++)
   {
      ierr += NALU_HYPRE_ParCSRMatrixGetRow(parcsr_A, i, &size, &col_ind, &values );
      newColInd = new NALU_HYPRE_Int[size];
      for (j = 0; j < size; j++) { newColInd[j] = col_ind[j] + 1; }
      H.sumIntoSystemMatrix(i + 1, size, (const NALU_HYPRE_Real*)values,
                            (const NALU_HYPRE_Int*)newColInd);
      delete [] newColInd;
      ierr += NALU_HYPRE_ParCSRMatrixRestoreRow(parcsr_A, i, &size, &col_ind, &values);
   }
   H.matrixLoadComplete();
   NALU_HYPRE_ParCSRMatrixDestroy(parcsr_A);

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   NALU_HYPRE_Real ddata = 1.0;
   NALU_HYPRE_Int  status;

   for (i = first_local_row; i <= last_local_row; i++)
   {
      index = i + 1;
      H.sumIntoRHSVector(1, (const NALU_HYPRE_Real*) &ddata, (const NALU_HYPRE_Int*) &index);
   }

   nalu_hypre_EndTiming(time_index);
   strcpy(paramString, "LS Interface");
   nalu_hypre_PrintTiming(paramString, nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if ( solver_id == 0 )
   {
      strcpy(paramString, "solver cg");
      H.parameters(1, &paramString);
      if (myid == 0) { nalu_hypre_printf("Solver: DS-PCG\n"); }

      strcpy(paramString, "preconditioner diagonal");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 1 )
   {
      strcpy(paramString, "solver cg");
      H.parameters(1, &paramString);
      if (myid == 0) { nalu_hypre_printf("Solver: ParaSails-PCG\n"); }

      strcpy(paramString, "preconditioner parasails");
      H.parameters(1, &paramString);
      strcpy(paramString, "parasailsNlevels 1");
      H.parameters(1, &paramString);
      strcpy(paramString, "parasailsThreshold 0.1");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 2 )
   {
      strcpy(paramString, "solver cg");
      H.parameters(1, &paramString);
      if (myid == 0) { nalu_hypre_printf("Solver: AMG-PCG\n"); }

      strcpy(paramString, "preconditioner boomeramg");
      H.parameters(1, &paramString);
      strcpy(paramString, "amgCoarsenType falgout");
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "amgStrongThreshold %e", strong_threshold);
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "amgNumSweeps %d", num_grid_sweeps);
      H.parameters(1, &paramString);
      strcpy(paramString, "amgRelaxType jacobi");
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "amgRelaxWeight %e", relax_weight);
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 3 )
   {
      strcpy(paramString, "solver cg");
      H.parameters(1, &paramString);
      if (myid == 0) { nalu_hypre_printf("Solver: Poly-PCG\n"); }

      strcpy(paramString, "preconditioner poly");
      H.parameters(1, &paramString);
      strcpy(paramString, "polyOrder 9");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 4 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { nalu_hypre_printf("Solver: DS-GMRES\n"); }

      strcpy(paramString, "preconditioner diagonal");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 5 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { nalu_hypre_printf("Solver: PILUT-GMRES\n"); }

      strcpy(paramString, "preconditioner pilut");
      H.parameters(1, &paramString);
      strcpy(paramString, "pilutRowSize 0");
      H.parameters(1, &paramString);
      strcpy(paramString, "pilutDropTol 0.0");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 6 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { nalu_hypre_printf("Solver: AMG-GMRES\n"); }

      strcpy(paramString, "preconditioner boomeramg");
      H.parameters(1, &paramString);
      strcpy(paramString, "amgCoarsenType falgout");
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "amgStrongThreshold %e", strong_threshold);
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "amgNumSweeps %d", num_grid_sweeps);
      H.parameters(1, &paramString);
      strcpy(paramString, "amgRelaxType jacobi");
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "amgRelaxWeight %e", relax_weight);
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 7 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { nalu_hypre_printf("Solver: DDILUT-GMRES\n"); }

      strcpy(paramString, "preconditioner ddilut");
      H.parameters(1, &paramString);
      strcpy(paramString, "ddilutFillin 5.0");
      H.parameters(1, &paramString);
      strcpy(paramString, "ddilutDropTol 0.0");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 8 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      nalu_hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { nalu_hypre_printf("Solver: POLY-GMRES\n"); }

      strcpy(paramString, "preconditioner poly");
      H.parameters(1, &paramString);
      strcpy(paramString, "polyOrder 5");
      H.parameters(1, &paramString);
   }

   strcpy(paramString, "Krylov Solve");
   time_index = nalu_hypre_InitializeTiming(paramString);
   nalu_hypre_BeginTiming(time_index);

   H.launchSolver(status, num_iterations);

   nalu_hypre_EndTiming(time_index);
   strcpy(paramString, "Solve phase times");
   nalu_hypre_PrintTiming(paramString, nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   if (myid == 0)
   {
      nalu_hypre_printf("\n Iterations = %d\n", num_iterations);
      nalu_hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   delete [] paramString;
   nalu_hypre_MPI_Finalize();

   return (0);
}

/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParLaplacian27pt( NALU_HYPRE_Int                  argc,
                       char                *argv[],
                       NALU_HYPRE_Int                  arg_index,
                       NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q, r;
   NALU_HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 20;
   ny = 20;
   nz = 20;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      nalu_hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Laplacian_27pt:\n");
      nalu_hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  2, NALU_HYPRE_MEMORY_HOST);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
   {
      values[0] = 8.0;
   }
   if (nx * ny == 1 || nx * nz == 1 || ny * nz == 1)
   {
      values[0] = 2.0;
   }
   values[1] = -1.0;

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian27pt(nalu_hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}
