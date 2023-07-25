/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interfoace (parcsr storage).
 * Do `driver -help' for usage info.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
main( NALU_HYPRE_Int   argc,
      char *argv[] )
{
   NALU_HYPRE_Int                 arg_index;
   NALU_HYPRE_Int                 print_usage;
   NALU_HYPRE_Int                 build_matrix_type;
   NALU_HYPRE_Int                 build_matrix_arg_index;
   NALU_HYPRE_Int                 build_rhs_type;
   NALU_HYPRE_Int                 build_rhs_arg_index;
   NALU_HYPRE_Int                 solver_id;
   NALU_HYPRE_Int                 ioutdat;
   NALU_HYPRE_Int                 debug_flag;
   NALU_HYPRE_Int                 ierr, i;
   NALU_HYPRE_Int                 max_levels = 25;
   NALU_HYPRE_Int                 num_iterations;
   NALU_HYPRE_Real          norm;
   NALU_HYPRE_Real          final_res_norm;


   NALU_HYPRE_ParCSRMatrix  A;
   NALU_HYPRE_ParVector     b;
   NALU_HYPRE_ParVector     x;

   NALU_HYPRE_Solver        amg_solver;
   NALU_HYPRE_Solver        pcg_solver;
   NALU_HYPRE_Solver        pcg_precond;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 global_m, global_n;
   NALU_HYPRE_Int                *partitioning;

   NALU_HYPRE_Int             time_index;

   /* parameters for BoomerAMG */
   NALU_HYPRE_Real   strong_threshold;
   NALU_HYPRE_Real   trunc_factor;
   NALU_HYPRE_Int      cycle_type;
   NALU_HYPRE_Int      coarsen_type = 0;
   NALU_HYPRE_Int      hybrid = 1;
   NALU_HYPRE_Int      measure_type = 0;
   NALU_HYPRE_Int     *num_grid_sweeps;
   NALU_HYPRE_Int     *grid_relax_type;
   NALU_HYPRE_Int    **grid_relax_points;
   NALU_HYPRE_Int      relax_default;
   NALU_HYPRE_Real  *relax_weight;
   NALU_HYPRE_Real   tol = 1.0e-6;

   /* parameters for PILUT */
   NALU_HYPRE_Real   drop_tol = -1;
   NALU_HYPRE_Int      nonzeros_to_keep = -1;

   /* parameters for GMRES */
   NALU_HYPRE_Int       k_dim;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   build_matrix_type      = 1;
   build_matrix_arg_index = argc;
   build_rhs_type = 0;
   build_rhs_arg_index = argc;
   relax_default = 3;
   debug_flag = 0;

   solver_id = 0;

   ioutdat = 3;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonefile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 3;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 4;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 1;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromonefile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 2;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsrand") == 0 )
      {
         arg_index++;
         build_rhs_type      = 3;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-ruge") == 0 )
      {
         arg_index++;
         coarsen_type      = 1;
      }
      else if ( strcmp(argv[arg_index], "-ruge2b") == 0 )
      {
         arg_index++;
         coarsen_type      = 2;
      }
      else if ( strcmp(argv[arg_index], "-ruge3") == 0 )
      {
         arg_index++;
         coarsen_type      = 3;
      }
      else if ( strcmp(argv[arg_index], "-ruge3c") == 0 )
      {
         arg_index++;
         coarsen_type      = 4;
      }
      else if ( strcmp(argv[arg_index], "-rugerlx") == 0 )
      {
         arg_index++;
         coarsen_type      = 5;
      }
      else if ( strcmp(argv[arg_index], "-falgout") == 0 )
      {
         arg_index++;
         coarsen_type      = 6;
      }
      else if ( strcmp(argv[arg_index], "-nohybrid") == 0 )
      {
         arg_index++;
         hybrid      = -1;
      }
      else if ( strcmp(argv[arg_index], "-gm") == 0 )
      {
         arg_index++;
         measure_type      = 1;
      }
      else if ( strcmp(argv[arg_index], "-xisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 4;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index++;
         relax_default = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         debug_flag = atoi(argv[arg_index++]);
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

   /* for CGNR preconditioned with Boomeramg, only relaxation scheme 2 is
      implemented, i.e. Jacobi relaxation with Matvec */
   if (solver_id == 5) { relax_default = 2; }

   /* defaults for BoomerAMG */
   strong_threshold = 0.25;
   trunc_factor = 0.0;
   cycle_type = 1;

   num_grid_sweeps = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
   grid_relax_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
   grid_relax_points = nalu_hypre_CTAlloc(NALU_HYPRE_Int *, 4, NALU_HYPRE_MEMORY_HOST);
   relax_weight = nalu_hypre_CTAlloc(NALU_HYPRE_Real, max_levels, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < max_levels; i++)
   {
      relax_weight[i] = 0.0;
   }
   if (coarsen_type == 5)
   {
      /* fine grid */
      num_grid_sweeps[0] = 3;
      grid_relax_type[0] = relax_default;
      grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[0][0] = -2;
      grid_relax_points[0][1] = -1;
      grid_relax_points[0][2] = 1;

      /* down cycle */
      num_grid_sweeps[1] = 4;
      grid_relax_type[1] = relax_default;
      grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[1][0] = -1;
      grid_relax_points[1][1] = 1;
      grid_relax_points[1][2] = -2;
      grid_relax_points[1][3] = -2;

      /* up cycle */
      num_grid_sweeps[2] = 4;
      grid_relax_type[2] = relax_default;
      grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[2][0] = -2;
      grid_relax_points[2][1] = -2;
      grid_relax_points[2][2] = 1;
      grid_relax_points[2][3] = -1;
   }
   else
   {
      /* fine grid */
      num_grid_sweeps[0] = 2;
      grid_relax_type[0] = relax_default;
      grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[0][0] = 1;
      grid_relax_points[0][1] = -1;

      /* down cycle */
      num_grid_sweeps[1] = 2;
      grid_relax_type[1] = relax_default;
      grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[1][0] = 1;
      grid_relax_points[1][1] = -1;

      /* up cycle */
      num_grid_sweeps[2] = 2;
      grid_relax_type[2] = relax_default;
      grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[2][0] = -1;
      grid_relax_points[2][1] = 1;
   }
   /* coarsest grid */
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 9;
   grid_relax_points[3] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
   grid_relax_points[3][0] = 0;

   /* defaults for GMRES */

   k_dim = 5;

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         relax_weight[0] = atof(argv[arg_index++]);
         for (i = 1; i < max_levels; i++)
         {
            relax_weight[i] = relax_weight[0];
         }
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-drop_tol") == 0 )
      {
         arg_index++;
         drop_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nonzeros_to_keep") == 0 )
      {
         arg_index++;
         nonzeros_to_keep  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tr") == 0 )
      {
         arg_index++;
         trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-iout") == 0 )
      {
         arg_index++;
         ioutdat  = atoi(argv[arg_index++]);
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
      nalu_hypre_printf("  -fromfile <filename>   : matrix from distributed file\n");
      nalu_hypre_printf("  -fromonefile <filename>: matrix from standard CSR file\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -laplacian [<options>] : build laplacian matrix\n");
      nalu_hypre_printf("  -9pt [<opts>] : build 9pt 2D laplacian matrix\n");
      nalu_hypre_printf("  -27pt [<opts>] : build 27pt 3D laplacian matrix\n");
      nalu_hypre_printf("  -difconv [<opts>]      : build convection-diffusion matrix\n");
      nalu_hypre_printf("    -n <nx> <ny> <nz>    : problem size per processor\n");
      nalu_hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      nalu_hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      nalu_hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("   -rhsfromfile          : from distributed file (NOT YET)\n");
      nalu_hypre_printf("   -rhsfromonefile       : from vector file \n");
      nalu_hypre_printf("   -rhsrand              : rhs is random vector, ||x||=1\n");
      nalu_hypre_printf("   -xisone               : rhs of all ones\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -solver <ID>           : solver ID\n");
      nalu_hypre_printf("       1=AMG-PCG    2=DS-PCG   \n");
      nalu_hypre_printf("       3=AMG-GMRES  4=DS-GMRES  \n");
      nalu_hypre_printf("       5=AMG-CGNR   6=DS-CGNR  \n");
      nalu_hypre_printf("       7=PILUT-GMRES  \n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("   -ruge                 : Ruge coarsening (local)\n");
      nalu_hypre_printf("   -ruge3                : third pass on boundary\n");
      nalu_hypre_printf("   -ruge3c               : third pass on boundary, keep c-points\n");
      nalu_hypre_printf("   -ruge2b               : 2nd pass is global\n");
      nalu_hypre_printf("   -rugerlx              : relaxes special points\n");
      nalu_hypre_printf("   -falgout              : local ruge followed by LJP\n");
      nalu_hypre_printf("   -nohybrid             : no switch in coarsening\n");
      nalu_hypre_printf("   -gm                   : use global measures\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -rlx <val>             : relaxation type\n");
      nalu_hypre_printf("       0=Weighted Jacobi  \n");
      nalu_hypre_printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -th <val>              : set AMG threshold Theta = val \n");
      nalu_hypre_printf("  -tr <val>              : set AMG interpolation truncation factor = val \n");
      nalu_hypre_printf("  -tol <val>             : set AMG convergence tolerance to val\n");
      nalu_hypre_printf("  -w  <val>              : set Jacobi relax weight = val\n");
      nalu_hypre_printf("  -k  <val>              : dimension Krylov space for GMRES\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -drop_tol  <val>       : set threshold for dropping in PILUT\n");
      nalu_hypre_printf("  -nonzeros_to_keep <val>: number of nonzeros in each row to keep\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -iout <val>            : set output flag\n");
      nalu_hypre_printf("       0=no output    1=matrix stats\n");
      nalu_hypre_printf("       2=cycle stats  3=matrix & cycle stats\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -dbg <val>             : set debug flag\n");
      nalu_hypre_printf("       0=no debugging\n       1=internal timing\n       2=interpolation truncation\n       3=more detailed timing in coarsening routine\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

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

   if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &A);
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &A);
   }
   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

#if 0
   NALU_HYPRE_ParCSRMatrixPrint(A, "driver.out.A");
#endif

   if (build_rhs_type == 1)
   {
      /* BuildRHSParFromFile(argc, argv, build_rhs_arg_index, &b); */
      nalu_hypre_printf("Rhs from file not yet implemented.  Defaults to b=0\n");
      NALU_HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      NALU_HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      NALU_HYPRE_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_m, partitioning, &b);
      NALU_HYPRE_ParVectorInitialize(b);
      NALU_HYPRE_ParVectorSetConstantValues(b, 0.0);

      NALU_HYPRE_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_n, partitioning, &x);
      NALU_HYPRE_ParVectorInitialize(x);
      NALU_HYPRE_ParVectorSetConstantValues(x, 1.0);
   }
   else if ( build_rhs_type == 2 )
   {
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, A, &b);

      NALU_HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      NALU_HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      NALU_HYPRE_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_n, partitioning, &x);
      NALU_HYPRE_ParVectorInitialize(x);
      NALU_HYPRE_ParVectorSetConstantValues(x, 0.0);
   }
   else if ( build_rhs_type == 3 )
   {

      NALU_HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      NALU_HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      NALU_HYPRE_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_m, partitioning, &b);
      NALU_HYPRE_ParVectorInitialize(b);
      NALU_HYPRE_ParVectorSetRandomValues(b, 22775);
      NALU_HYPRE_ParVectorInnerProd(b, b, &norm);
      norm = 1.0 / nalu_hypre_sqrt(norm);
      ierr = NALU_HYPRE_ParVectorScale(norm, b);

      NALU_HYPRE_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_n, partitioning, &x);
      NALU_HYPRE_ParVectorInitialize(x);
      NALU_HYPRE_ParVectorSetConstantValues(x, 0.0);
   }
   else if ( build_rhs_type == 4 )
   {

      NALU_HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      NALU_HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      NALU_HYPRE_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_n, partitioning, &x);
      NALU_HYPRE_ParVectorInitialize(x);
      NALU_HYPRE_ParVectorSetConstantValues(x, 1.0);

      NALU_HYPRE_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_m, partitioning, &b);
      NALU_HYPRE_ParVectorInitialize(b);
      NALU_HYPRE_ParCSRMatrixMatvec(1.0, A, x, 0.0, b);

      NALU_HYPRE_ParVectorSetConstantValues(x, 0.0);
   }
   else /* if ( build_rhs_type == 0 ) */
   {
      NALU_HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
      NALU_HYPRE_ParCSRMatrixGetDims(A, &global_m, &global_n);
      NALU_HYPRE_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_m, partitioning, &b);
      NALU_HYPRE_ParVectorInitialize(b);
      NALU_HYPRE_ParVectorSetConstantValues(b, 0.0);

      NALU_HYPRE_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_n, partitioning, &x);
      NALU_HYPRE_ParVectorInitialize(x);
      NALU_HYPRE_ParVectorSetConstantValues(x, 1.0);
   }
   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      time_index = nalu_hypre_InitializeTiming("BoomerAMG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BoomerAMGCreate(&amg_solver);
      NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, (hybrid * coarsen_type));
      NALU_HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      NALU_HYPRE_BoomerAMGSetTol(amg_solver, tol);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      NALU_HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
      NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, ioutdat);
      NALU_HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log");
      NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      NALU_HYPRE_BoomerAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      NALU_HYPRE_BoomerAMGSetGridRelaxType(amg_solver, grid_relax_type);
      NALU_HYPRE_BoomerAMGSetRelaxWeight(amg_solver, relax_weight);
      NALU_HYPRE_BoomerAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      NALU_HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      NALU_HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);

      NALU_HYPRE_BoomerAMGSetup(amg_solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("BoomerAMG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BoomerAMGSolve(amg_solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_BoomerAMGDestroy(amg_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2)
   {
      time_index = nalu_hypre_InitializeTiming("PCG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRPCGCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_ParCSRPCGSetMaxIter(pcg_solver, 500);
      NALU_HYPRE_ParCSRPCGSetTol(pcg_solver, tol);
      NALU_HYPRE_ParCSRPCGSetTwoNorm(pcg_solver, 1);
      NALU_HYPRE_ParCSRPCGSetRelChange(pcg_solver, 0);
      NALU_HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, 1);

      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, ioutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   NALU_HYPRE_BoomerAMGSolve,
                                   NALU_HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if (solver_id == 2)
      {
         /* use diagonal scaling as preconditioner */

         pcg_precond = NULL;

         NALU_HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                                   NALU_HYPRE_ParCSRDiagScale,
                                   NALU_HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
      }

      NALU_HYPRE_ParCSRPCGSetup(pcg_solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("PCG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRPCGSolve(pcg_solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
      NALU_HYPRE_ParCSRPCGDestroy(pcg_solver);

      if (solver_id == 1)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7)
   {
      time_index = nalu_hypre_InitializeTiming("GMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRGMRESCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_ParCSRGMRESSetKDim(pcg_solver, k_dim);
      NALU_HYPRE_ParCSRGMRESSetMaxIter(pcg_solver, 100);
      NALU_HYPRE_ParCSRGMRESSetTol(pcg_solver, tol);
      NALU_HYPRE_ParCSRGMRESSetLogging(pcg_solver, 1);

      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */

         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, ioutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                     NALU_HYPRE_BoomerAMGSolve,
                                     NALU_HYPRE_BoomerAMGSetup,
                                     pcg_precond);
      }
      else if (solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */

         pcg_precond = NULL;

         NALU_HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                     NALU_HYPRE_ParCSRDiagScale,
                                     NALU_HYPRE_ParCSRDiagScaleSetup,
                                     pcg_precond);
      }
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         ierr = NALU_HYPRE_ParCSRPilutCreate( nalu_hypre_MPI_COMM_WORLD, &pcg_precond );
         if (ierr)
         {
            nalu_hypre_printf("Error in ParPilutCreate\n");
         }

         NALU_HYPRE_ParCSRGMRESSetPrecond(pcg_solver,
                                     NALU_HYPRE_ParCSRPilutSolve,
                                     NALU_HYPRE_ParCSRPilutSetup,
                                     pcg_precond);

         if (drop_tol >= 0 )
            NALU_HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            NALU_HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
      }

      NALU_HYPRE_ParCSRGMRESSetup(pcg_solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("GMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRGMRESSolve(pcg_solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_ParCSRGMRESGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
      NALU_HYPRE_ParCSRGMRESDestroy(pcg_solver);

      if (solver_id == 3)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (solver_id == 7)
      {
         NALU_HYPRE_ParCSRPilutDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("GMRES Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using CGNR
    *-----------------------------------------------------------*/

   if (solver_id == 5 || solver_id == 6)
   {
      time_index = nalu_hypre_InitializeTiming("CGNR Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRCGNRCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_ParCSRCGNRSetMaxIter(pcg_solver, 1000);
      NALU_HYPRE_ParCSRCGNRSetTol(pcg_solver, tol);
      NALU_HYPRE_ParCSRCGNRSetLogging(pcg_solver, 1);

      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, ioutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_ParCSRCGNRSetPrecond(pcg_solver,
                                    NALU_HYPRE_BoomerAMGSolve,
                                    NALU_HYPRE_BoomerAMGSolveT,
                                    NALU_HYPRE_BoomerAMGSetup,
                                    pcg_precond);
      }
      else if (solver_id == 6)
      {
         /* use diagonal scaling as preconditioner */

         pcg_precond = NULL;

         NALU_HYPRE_ParCSRCGNRSetPrecond(pcg_solver,
                                    NALU_HYPRE_ParCSRDiagScale,
                                    NALU_HYPRE_ParCSRDiagScale,
                                    NALU_HYPRE_ParCSRDiagScaleSetup,
                                    pcg_precond);
      }

      NALU_HYPRE_ParCSRCGNRSetup(pcg_solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("CGNR Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRCGNRSolve(pcg_solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_ParCSRCGNRGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
      NALU_HYPRE_ParCSRCGNRDestroy(pcg_solver);

      if (solver_id == 5)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

#if 0
   NALU_HYPRE_PrintCSRVector(x, "driver.out.x");
#endif


   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   NALU_HYPRE_ParCSRMatrixDestroy(A);
   NALU_HYPRE_ParVectorDestroy(b);
   NALU_HYPRE_ParVectorDestroy(x);

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from file. Expects three files on each processor.
 * filename.D.n contains the diagonal part, filename.O.n contains
 * the offdiagonal part and filename.INFO.n contains global row
 * and column numbers, number of columns of offdiagonal matrix
 * and the mapping of offdiagonal column numbers to global column numbers.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParFromFile( NALU_HYPRE_Int                  argc,
                  char                *argv[],
                  NALU_HYPRE_Int                  arg_index,
                  NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      nalu_hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   A = NALU_HYPRE_ParCSRMatrixRead(nalu_hypre_MPI_COMM_WORLD, filename);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParLaplacian( NALU_HYPRE_Int                  argc,
                   char                *argv[],
                   NALU_HYPRE_Int                  arg_index,
                   NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;
   NALU_HYPRE_Real          cx, cy, cz;

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

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

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
      nalu_hypre_printf("  Laplacian:\n");
      nalu_hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      nalu_hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
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

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  4, NALU_HYPRE_MEMORY_HOST);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.0;
   if (nx > 1)
   {
      values[0] += 2.0 * cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0 * cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0 * cz;
   }

   A = (NALU_HYPRE_ParCSRMatrix)
       GenerateLaplacian(nalu_hypre_MPI_COMM_WORLD, nx, ny, nz, P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point convection-diffusion operator
 * Parameters given in command line.
 * Operator:
 *
 *  -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f
 *
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParDifConv( NALU_HYPRE_Int                  argc,
                 char                *argv[],
                 NALU_HYPRE_Int                  arg_index,
                 NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;
   NALU_HYPRE_Real          cx, cy, cz;
   NALU_HYPRE_Real          ax, ay, az;
   NALU_HYPRE_Real          hinx, hiny, hinz;

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

   nx = 10;
   ny = 10;
   nz = 10;

   hinx = 1.0 / (nx + 1);
   hiny = 1.0 / (ny + 1);
   hinz = 1.0 / (nz + 1);

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

   ax = 1.0;
   ay = 1.0;
   az = 1.0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = atof(argv[arg_index++]);
         ay = atof(argv[arg_index++]);
         az = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

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
      nalu_hypre_printf("  Convection-Diffusion: \n");
      nalu_hypre_printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");
      nalu_hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      nalu_hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      nalu_hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n", ax, ay, az);
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

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  7, NALU_HYPRE_MEMORY_HOST);

   values[1] = -cx / (hinx * hinx);
   values[2] = -cy / (hiny * hiny);
   values[3] = -cz / (hinz * hinz);
   values[4] = -cx / (hinx * hinx) + ax / hinx;
   values[5] = -cy / (hiny * hiny) + ay / hiny;
   values[6] = -cz / (hinz * hinz) + az / hinz;

   values[0] = 0.0;
   if (nx > 1)
   {
      values[0] += 2.0 * cx / (hinx * hinx) - 1.0 * ax / hinx;
   }
   if (ny > 1)
   {
      values[0] += 2.0 * cy / (hiny * hiny) - 1.0 * ay / hiny;
   }
   if (nz > 1)
   {
      values[0] += 2.0 * cz / (hinz * hinz) - 1.0 * az / hinz;
   }

   A = (NALU_HYPRE_ParCSRMatrix) GenerateDifConv(nalu_hypre_MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from one file on Proc. 0. Expects matrix to be in
 * CSR format. Distributes matrix across processors giving each about
 * the same number of rows.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParFromOneFile( NALU_HYPRE_Int                  argc,
                     char                *argv[],
                     NALU_HYPRE_Int                  arg_index,
                     NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   NALU_HYPRE_ParCSRMatrix  A;
   NALU_HYPRE_CSRMatrix  A_CSR;

   NALU_HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      nalu_hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix
       *-----------------------------------------------------------*/

      A_CSR = NALU_HYPRE_CSRMatrixRead(filename);
   }
   A = NALU_HYPRE_CSRMatrixToParCSRMatrix(nalu_hypre_MPI_COMM_WORLD, A_CSR, NULL, NULL);

   *A_ptr = A;

   NALU_HYPRE_CSRMatrixDestroy(A_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildRhsParFromOneFile( NALU_HYPRE_Int                  argc,
                        char                *argv[],
                        NALU_HYPRE_Int                  arg_index,
                        NALU_HYPRE_ParCSRMatrix   A,
                        NALU_HYPRE_ParVector     *b_ptr     )
{
   char               *filename;

   NALU_HYPRE_ParVector  b;
   NALU_HYPRE_Vector     b_CSR;

   NALU_HYPRE_Int                 myid;
   NALU_HYPRE_Int            *partitioning;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      nalu_hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Rhs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix
       *-----------------------------------------------------------*/

      b_CSR = NALU_HYPRE_VectorRead(filename);
   }
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning(A, &partitioning);
   b = NALU_HYPRE_VectorToParVector(nalu_hypre_MPI_COMM_WORLD, b_CSR, partitioning);

   *b_ptr = b;

   NALU_HYPRE_VectorDestroy(b_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParLaplacian9pt( NALU_HYPRE_Int                  argc,
                      char                *argv[],
                      NALU_HYPRE_Int                  arg_index,
                      NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny;
   NALU_HYPRE_Int                 P, Q;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q;
   NALU_HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q) != num_procs)
   {
      nalu_hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Laplacian 9pt:\n");
      nalu_hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      nalu_hypre_printf("    (Px, Py) = (%d, %d)\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p) / P;

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  2, NALU_HYPRE_MEMORY_HOST);

   values[1] = -1.0;

   values[0] = 0.0;
   if (nx > 1)
   {
      values[0] += 2.0;
   }
   if (ny > 1)
   {
      values[0] += 2.0;
   }
   if (nx > 1 && ny > 1)
   {
      values[0] += 4.0;
   }

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian9pt(nalu_hypre_MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

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

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

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
