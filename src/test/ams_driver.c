/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   This test driver performs the following operations:

1. Read a linear system corresponding to a parallel finite element
   discretization of Maxwell's equations.

2. Call the AMS solver in HYPRE to solve that linear system.
*/

/* hypre/AMS prototypes */
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_IJ_mv.h"
#include "NALU_HYPRE.h"

void CheckIfFileExists(char *file)
{
   FILE *test;
   if (!(test = fopen(file, "r")))
   {
      nalu_hypre_MPI_Finalize();
      nalu_hypre_printf("Can't find the input file \"%s\"\n", file);
      exit(1);
   }
   fclose(test);
}

void AMSDriverMatrixRead(const char *file, NALU_HYPRE_ParCSRMatrix *A)
{
   FILE *test;
   char file0[100];
   sprintf(file0, "%s.D.0", file);
   if (!(test = fopen(file0, "r")))
   {
      sprintf(file0, "%s.00000", file);
      if (!(test = fopen(file0, "r")))
      {
         nalu_hypre_MPI_Finalize();
         nalu_hypre_printf("Can't find the input file \"%s\"\n", file);
         exit(1);
      }
      else /* Read in IJ format*/
      {
         NALU_HYPRE_IJMatrix ij_A;
         void *object;
         NALU_HYPRE_IJMatrixRead(file, nalu_hypre_MPI_COMM_WORLD, NALU_HYPRE_PARCSR, &ij_A);
         NALU_HYPRE_IJMatrixGetObject(ij_A, &object);
         *A = (NALU_HYPRE_ParCSRMatrix) object;
         nalu_hypre_IJMatrixObject((nalu_hypre_IJMatrix *)ij_A) = NULL;
         NALU_HYPRE_IJMatrixDestroy(ij_A);
      }
   }
   else /* Read in ParCSR format*/
   {
      NALU_HYPRE_ParCSRMatrixRead(nalu_hypre_MPI_COMM_WORLD, file, A);
   }
   fclose(test);
}

void AMSDriverVectorRead(const char *file, NALU_HYPRE_ParVector *x)
{
   FILE *test;
   char file0[100];
   sprintf(file0, "%s.0", file);
   if (!(test = fopen(file0, "r")))
   {
      sprintf(file0, "%s.00000", file);
      if (!(test = fopen(file0, "r")))
      {
         nalu_hypre_MPI_Finalize();
         nalu_hypre_printf("Can't find the input file \"%s\"\n", file);
         exit(1);
      }
      else /* Read in IJ format*/
      {
         NALU_HYPRE_IJVector ij_x;
         void *object;
         NALU_HYPRE_IJVectorRead(file, nalu_hypre_MPI_COMM_WORLD, NALU_HYPRE_PARCSR, &ij_x);
         NALU_HYPRE_IJVectorGetObject(ij_x, &object);
         *x = (NALU_HYPRE_ParVector) object;
         nalu_hypre_IJVectorObject((nalu_hypre_IJVector *)ij_x) = NULL;
         NALU_HYPRE_IJVectorDestroy(ij_x);
      }
   }
   else /* Read in ParCSR format*/
   {
      NALU_HYPRE_ParVectorRead(nalu_hypre_MPI_COMM_WORLD, file, x);
   }
   fclose(test);
}

nalu_hypre_int main (nalu_hypre_int argc, char *argv[])
{
   NALU_HYPRE_Int num_procs, myid;
   NALU_HYPRE_Int time_index;

   NALU_HYPRE_Int solver_id;
   NALU_HYPRE_Int maxit, cycle_type, rlx_type, coarse_rlx_type, rlx_sweeps, dim;
   NALU_HYPRE_Real rlx_weight, rlx_omega;
   NALU_HYPRE_Int amg_coarsen_type, amg_rlx_type, amg_agg_levels, amg_interp_type, amg_Pmax;
   NALU_HYPRE_Int h1_method, singular_problem, coordinates;
   NALU_HYPRE_Real tol, theta;
   NALU_HYPRE_Real rtol;
   NALU_HYPRE_Int rr;
   NALU_HYPRE_Int zero_cond;
   NALU_HYPRE_Int blockSize;
   NALU_HYPRE_Solver solver, precond;

   NALU_HYPRE_ParCSRMatrix A = 0, G = 0, Aalpha = 0, Abeta = 0, M = 0;
   NALU_HYPRE_ParVector x0 = 0, b = 0;
   NALU_HYPRE_ParVector Gx = 0, Gy = 0, Gz = 0;
   NALU_HYPRE_ParVector x = 0, y = 0, z = 0;

   NALU_HYPRE_ParVector interior_nodes = 0;

   /* default execution policy and memory space */
#if defined(NALU_HYPRE_TEST_USING_HOST)
   NALU_HYPRE_MemoryLocation memory_location = NALU_HYPRE_MEMORY_HOST;
   NALU_HYPRE_ExecutionPolicy default_exec_policy = NALU_HYPRE_EXEC_HOST;
#else
   NALU_HYPRE_MemoryLocation memory_location = NALU_HYPRE_MEMORY_DEVICE;
   NALU_HYPRE_ExecutionPolicy default_exec_policy = NALU_HYPRE_EXEC_DEVICE;
#endif

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid);

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before NALU_HYPRE_Init() and should not be changed after
    *-----------------------------------------------------------------*/
   nalu_hypre_bind_device(myid, num_procs, nalu_hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   NALU_HYPRE_Init();

   /* default memory location */
   NALU_HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   NALU_HYPRE_SetExecutionPolicy(default_exec_policy);

#if defined(NALU_HYPRE_USING_GPU)
   /* use vendor implementation for SpGEMM */
   NALU_HYPRE_SetSpGemmUseVendor(0);
   /* use cuRand for PMIS */
   NALU_HYPRE_SetUseGpuRand(1);
#endif

   /* Set defaults */
   solver_id = 3;
   maxit = 100;
   tol = 1e-6;
   dim = 3;
   coordinates = 0;
   h1_method = 0;
   singular_problem = 0;
   rlx_sweeps = 1;
   rlx_weight = 1.0; rlx_omega = 1.0;
   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      cycle_type = 1; amg_coarsen_type =  8; amg_agg_levels = 1; amg_rlx_type = 8;
      coarse_rlx_type = 8, rlx_type = 2; /* PMIS */
   }
   else
   {
      cycle_type = 1; amg_coarsen_type = 10; amg_agg_levels = 1; amg_rlx_type = 8;
      coarse_rlx_type = 8, rlx_type = 2; /* HMIS-1 */
   }

   /* cycle_type = 1; amg_coarsen_type = 10; amg_agg_levels = 0; amg_rlx_type = 3; */ /* HMIS-0 */
   /* cycle_type = 1; amg_coarsen_type = 8; amg_agg_levels = 1; amg_rlx_type = 3;  */ /* PMIS-1 */
   /* cycle_type = 1; amg_coarsen_type = 8; amg_agg_levels = 0; amg_rlx_type = 3;  */ /* PMIS-0 */
   /* cycle_type = 7; amg_coarsen_type = 6; amg_agg_levels = 0; amg_rlx_type = 6;  */ /* Falgout-0 */
   amg_interp_type = 6; amg_Pmax = 4;     /* long-range interpolation */
   /* amg_interp_type = 0; amg_Pmax = 0; */  /* standard interpolation */
   theta = 0.25;
   blockSize = 5;
   rtol = 0;
   rr = 0;
   zero_cond = 0;

   /* Parse command line */
   {
      NALU_HYPRE_Int arg_index = 0;
      NALU_HYPRE_Int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-maxit") == 0 )
         {
            arg_index++;
            maxit = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-tol") == 0 )
         {
            arg_index++;
            tol = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-type") == 0 )
         {
            arg_index++;
            cycle_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlx") == 0 )
         {
            arg_index++;
            rlx_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlxn") == 0 )
         {
            arg_index++;
            rlx_sweeps = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlxw") == 0 )
         {
            arg_index++;
            rlx_weight = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlxo") == 0 )
         {
            arg_index++;
            rlx_omega = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ctype") == 0 )
         {
            arg_index++;
            amg_coarsen_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-amgrlx") == 0 )
         {
            arg_index++;
            amg_rlx_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-crlx") == 0 )
         {
            arg_index++;
            coarse_rlx_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-agg") == 0 )
         {
            arg_index++;
            amg_agg_levels = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-itype") == 0 )
         {
            arg_index++;
            amg_interp_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-pmax") == 0 )
         {
            arg_index++;
            amg_Pmax = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-dim") == 0 )
         {
            arg_index++;
            dim = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-coord") == 0 )
         {
            arg_index++;
            coordinates = 1;
         }
         else if ( strcmp(argv[arg_index], "-h1") == 0 )
         {
            arg_index++;
            h1_method = 1;
         }
         else if ( strcmp(argv[arg_index], "-sing") == 0 )
         {
            arg_index++;
            singular_problem = 1;
         }
         else if ( strcmp(argv[arg_index], "-theta") == 0 )
         {
            arg_index++;
            theta = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-bsize") == 0 )
         {
            arg_index++;
            blockSize = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rtol") == 0 )
         {
            arg_index++;
            rtol = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rr") == 0 )
         {
            arg_index++;
            rr = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-zc") == 0 )
         {
            arg_index++;
            zero_cond = 1;
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

      if (argc == 1)
      {
         print_usage = 1;
      }

      if ((print_usage) && (myid == 0))
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Usage: mpirun -np <np> %s [<options>]\n", argv[0]);
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  Hypre solvers options:                                       \n");
         nalu_hypre_printf("    -solver <ID>         : solver ID                           \n");
         nalu_hypre_printf("                           0  - AMG                            \n");
         nalu_hypre_printf("                           1  - AMG-PCG                        \n");
         nalu_hypre_printf("                           2  - AMS                            \n");
         nalu_hypre_printf("                           3  - AMS-PCG (default)              \n");
         nalu_hypre_printf("                           4  - DS-PCG                         \n");
         nalu_hypre_printf("                           5  - AME eigensolver                \n");
         nalu_hypre_printf("    -maxit <num>         : maximum number of iterations (100)  \n");
         nalu_hypre_printf("    -tol <num>           : convergence tolerance (1e-6)        \n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  AMS solver options:                                          \n");
         nalu_hypre_printf("    -dim <num>           : space dimension                     \n");
         nalu_hypre_printf("    -type <num>          : 3-level cycle type (0-8, 11-14)     \n");
         nalu_hypre_printf("    -theta <num>         : BoomerAMG threshold (0.25)          \n");
         nalu_hypre_printf("    -ctype <num>         : BoomerAMG coarsening type           \n");
         nalu_hypre_printf("    -agg <num>           : Levels of BoomerAMG agg. coarsening \n");
         nalu_hypre_printf("    -amgrlx <num>        : BoomerAMG relaxation type           \n");
         nalu_hypre_printf("    -itype <num>         : BoomerAMG interpolation type        \n");
         nalu_hypre_printf("    -pmax <num>          : BoomerAMG interpolation truncation  \n");
         nalu_hypre_printf("    -rlx <num>           : relaxation type                     \n");
         nalu_hypre_printf("    -rlxn <num>          : number of relaxation sweeps         \n");
         nalu_hypre_printf("    -rlxw <num>          : damping parameter (usually <=1)     \n");
         nalu_hypre_printf("    -rlxo <num>          : SOR parameter (usuallyin (0,2))     \n");
         nalu_hypre_printf("    -coord               : use coordinate vectors              \n");
         nalu_hypre_printf("    -h1                  : use block-diag Poisson solves       \n");
         nalu_hypre_printf("    -sing                : curl-curl only (singular) problem   \n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  AME eigensolver options:                                     \n");
         nalu_hypre_printf("    -bsize<num>          : number of eigenvalues to compute    \n");
         nalu_hypre_printf("\n");
      }

      if (print_usage)
      {
         nalu_hypre_MPI_Finalize();
         return (0);
      }
   }

   AMSDriverMatrixRead("mfem.A", &A);
   AMSDriverVectorRead("mfem.x0", &x0);
   AMSDriverVectorRead("mfem.b", &b);
   AMSDriverMatrixRead("mfem.G", &G);

   /* Vectors Gx, Gy and Gz */
   if (!coordinates)
   {
      AMSDriverVectorRead("mfem.Gx", &Gx);
      AMSDriverVectorRead("mfem.Gy", &Gy);
      if (dim == 3)
      {
         AMSDriverVectorRead("mfem.Gz", &Gz);
      }
   }

   /* Vectors x, y and z */
   if (coordinates)
   {
      AMSDriverVectorRead("mfem.x", &x);
      AMSDriverVectorRead("mfem.y", &y);
      if (dim == 3)
      {
         AMSDriverVectorRead("mfem.z", &z);
      }
   }

   /* Poisson matrices */
   if (h1_method)
   {
      AMSDriverMatrixRead("mfem.Aalpha", &Aalpha);
      AMSDriverMatrixRead("mfem.Abeta", &Abeta);
   }

   if (zero_cond)
   {
      AMSDriverVectorRead("mfem.inodes", &interior_nodes);
   }

   if (!myid)
   {
      nalu_hypre_printf("Problem size: %d\n\n",
                   nalu_hypre_ParCSRMatrixGlobalNumRows((nalu_hypre_ParCSRMatrix*)A));
   }

   nalu_hypre_ParCSRMatrixMigrate(A,      nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParCSRMatrixMigrate(G,      nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParCSRMatrixMigrate(Aalpha, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParCSRMatrixMigrate(Abeta,  nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));

   nalu_hypre_ParVectorMigrate(x0, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(b,  nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(Gx, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(Gy, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(Gz, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(x,  nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(y,  nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(z,  nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(interior_nodes, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));

   nalu_hypre_MPI_Barrier(nalu_hypre_MPI_COMM_WORLD);

   /* AMG */
   if (solver_id == 0)
   {
      NALU_HYPRE_Int num_iterations;
      NALU_HYPRE_Real final_res_norm;

      /* Start timing */
      time_index = nalu_hypre_InitializeTiming("BoomerAMG Setup");
      nalu_hypre_BeginTiming(time_index);

      /* Create solver */
      NALU_HYPRE_BoomerAMGCreate(&solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      NALU_HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
      NALU_HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
      NALU_HYPRE_BoomerAMGSetRelaxType(solver, rlx_type); /* G-S/Jacobi hybrid relaxation */
      NALU_HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
      NALU_HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
      NALU_HYPRE_BoomerAMGSetTol(solver, tol);       /* conv. tolerance */
      NALU_HYPRE_BoomerAMGSetMaxIter(solver, maxit); /* maximum number of iterations */
      NALU_HYPRE_BoomerAMGSetStrongThreshold(solver, theta);

      NALU_HYPRE_BoomerAMGSetup(solver, A, b, x0);

      /* Finalize setup timing */
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      /* Start timing again */
      time_index = nalu_hypre_InitializeTiming("BoomerAMG Solve");
      nalu_hypre_BeginTiming(time_index);

      /* Solve */
      NALU_HYPRE_BoomerAMGSolve(solver, A, b, x0);

      /* Finalize solve timing */
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      /* Run info - needed logging turned on */
      NALU_HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

      /* Destroy solver */
      NALU_HYPRE_BoomerAMGDestroy(solver);
   }

   /* AMS */
   if (solver_id == 2)
   {
      /* Start timing */
      time_index = nalu_hypre_InitializeTiming("AMS Setup");
      nalu_hypre_BeginTiming(time_index);

      /* Create solver */
      NALU_HYPRE_AMSCreate(&solver);

      /* Set AMS parameters */
      NALU_HYPRE_AMSSetDimension(solver, dim);
      NALU_HYPRE_AMSSetMaxIter(solver, maxit);
      NALU_HYPRE_AMSSetTol(solver, tol);
      NALU_HYPRE_AMSSetCycleType(solver, cycle_type);
      NALU_HYPRE_AMSSetPrintLevel(solver, 1);
      NALU_HYPRE_AMSSetDiscreteGradient(solver, G);

      /* Vectors Gx, Gy and Gz */
      if (!coordinates)
      {
         NALU_HYPRE_AMSSetEdgeConstantVectors(solver, Gx, Gy, Gz);
      }

      /* Vectors x, y and z */
      if (coordinates)
      {
         NALU_HYPRE_AMSSetCoordinateVectors(solver, x, y, z);
      }

      /* Poisson matrices */
      if (h1_method)
      {
         NALU_HYPRE_AMSSetAlphaPoissonMatrix(solver, Aalpha);
         NALU_HYPRE_AMSSetBetaPoissonMatrix(solver, Abeta);
      }

      if (singular_problem)
      {
         NALU_HYPRE_AMSSetBetaPoissonMatrix(solver, NULL);
      }

      /* Smoothing and AMG options */
      NALU_HYPRE_AMSSetSmoothingOptions(solver, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
      NALU_HYPRE_AMSSetAlphaAMGOptions(solver, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                  amg_interp_type, amg_Pmax);
      NALU_HYPRE_AMSSetBetaAMGOptions(solver, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                 amg_interp_type, amg_Pmax);
      NALU_HYPRE_AMSSetAlphaAMGCoarseRelaxType(solver, coarse_rlx_type);
      NALU_HYPRE_AMSSetBetaAMGCoarseRelaxType(solver, coarse_rlx_type);

      NALU_HYPRE_AMSSetup(solver, A, b, x0);

      /* Finalize setup timing */
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      /* Start timing again */
      time_index = nalu_hypre_InitializeTiming("AMS Solve");
      nalu_hypre_BeginTiming(time_index);

      /* Solve */
      NALU_HYPRE_AMSSolve(solver, A, b, x0);

      /* Finalize solve timing */
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      /* Destroy solver */
      NALU_HYPRE_AMSDestroy(solver);
   }

   /* PCG solvers */
   else if (solver_id == 1 || solver_id == 3 || solver_id == 4)
   {
      NALU_HYPRE_Int num_iterations;
      NALU_HYPRE_Real final_res_norm;

      /* Start timing */
      if (solver_id == 1)
      {
         time_index = nalu_hypre_InitializeTiming("BoomerAMG-PCG Setup");
      }
      else if (solver_id == 3)
      {
         time_index = nalu_hypre_InitializeTiming("AMS-PCG Setup");
      }
      else if (solver_id == 4)
      {
         time_index = nalu_hypre_InitializeTiming("DS-PCG Setup");
      }
      nalu_hypre_BeginTiming(time_index);

      /* Create solver */
      NALU_HYPRE_ParCSRPCGCreate(nalu_hypre_MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      NALU_HYPRE_PCGSetMaxIter(solver, maxit); /* max iterations */
      NALU_HYPRE_PCGSetTol(solver, tol); /* conv. tolerance */
      NALU_HYPRE_PCGSetTwoNorm(solver, 0); /* use the two norm as the stopping criteria */
      NALU_HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      NALU_HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      /* PCG with AMG preconditioner */
      if (solver_id == 1)
      {
         /* Now set up the AMG preconditioner and specify any parameters */
         NALU_HYPRE_BoomerAMGCreate(&precond);
         NALU_HYPRE_BoomerAMGSetPrintLevel(precond, 1);  /* print amg solution info */
         NALU_HYPRE_BoomerAMGSetCoarsenType(precond, 6); /* Falgout coarsening */
         NALU_HYPRE_BoomerAMGSetRelaxType(precond, rlx_type);   /* Sym G.S./Jacobi hybrid */
         NALU_HYPRE_BoomerAMGSetNumSweeps(precond, 1);   /* Sweeeps on each level */
         NALU_HYPRE_BoomerAMGSetMaxLevels(precond, 20);  /* maximum number of levels */
         NALU_HYPRE_BoomerAMGSetTol(precond, 0.0);      /* conv. tolerance (if needed) */
         NALU_HYPRE_BoomerAMGSetMaxIter(precond, 1);     /* do only one iteration! */
         NALU_HYPRE_BoomerAMGSetStrongThreshold(precond, theta);

         /* Set the PCG preconditioner */
         NALU_HYPRE_PCGSetPrecond(solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                             precond);
      }
      /* PCG with AMS preconditioner */
      if (solver_id == 3)
      {
         /* Now set up the AMS preconditioner and specify any parameters */
         NALU_HYPRE_AMSCreate(&precond);
         NALU_HYPRE_AMSSetDimension(precond, dim);
         NALU_HYPRE_AMSSetMaxIter(precond, 1);
         NALU_HYPRE_AMSSetTol(precond, 0.0);
         NALU_HYPRE_AMSSetCycleType(precond, cycle_type);
         NALU_HYPRE_AMSSetPrintLevel(precond, 0);
         NALU_HYPRE_AMSSetDiscreteGradient(precond, G);

         if (zero_cond)
         {
            NALU_HYPRE_AMSSetInteriorNodes(precond, interior_nodes);
            NALU_HYPRE_AMSSetProjectionFrequency(precond, 5);
         }
         NALU_HYPRE_PCGSetResidualTol(solver, rtol);
         NALU_HYPRE_PCGSetRecomputeResidualP(solver, rr);

         /* Vectors Gx, Gy and Gz */
         if (!coordinates)
         {
            NALU_HYPRE_AMSSetEdgeConstantVectors(precond, Gx, Gy, Gz);
         }

         /* Vectors x, y and z */
         if (coordinates)
         {
            NALU_HYPRE_AMSSetCoordinateVectors(precond, x, y, z);
         }

         /* Poisson matrices */
         if (h1_method)
         {
            NALU_HYPRE_AMSSetAlphaPoissonMatrix(precond, Aalpha);
            NALU_HYPRE_AMSSetBetaPoissonMatrix(precond, Abeta);
         }

         if (singular_problem)
         {
            NALU_HYPRE_AMSSetBetaPoissonMatrix(precond, NULL);
         }

         /* Smoothing and AMG options */
         NALU_HYPRE_AMSSetSmoothingOptions(precond, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
         NALU_HYPRE_AMSSetAlphaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                     amg_interp_type, amg_Pmax);
         NALU_HYPRE_AMSSetBetaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                    amg_interp_type, amg_Pmax);
         NALU_HYPRE_AMSSetAlphaAMGCoarseRelaxType(precond, coarse_rlx_type);
         NALU_HYPRE_AMSSetBetaAMGCoarseRelaxType(precond, coarse_rlx_type);

         /* Set the PCG preconditioner */
         NALU_HYPRE_PCGSetPrecond(solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_AMSSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_AMSSetup,
                             precond);
      }
      /* PCG with diagonal scaling preconditioner */
      else if (solver_id == 4)
      {
         /* Set the PCG preconditioner */
         NALU_HYPRE_PCGSetPrecond(solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                             NULL);
      }

      /* Setup */
      NALU_HYPRE_ParCSRPCGSetup(solver, A, b, x0);

      /* Finalize setup timing */
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      /* Start timing again */
      if (solver_id == 1)
      {
         time_index = nalu_hypre_InitializeTiming("BoomerAMG-PCG Solve");
      }
      else if (solver_id == 3)
      {
         time_index = nalu_hypre_InitializeTiming("AMS-PCG Solve");
      }
      else if (solver_id == 4)
      {
         time_index = nalu_hypre_InitializeTiming("DS-PCG Solve");
      }
      nalu_hypre_BeginTiming(time_index);

      /* Solve */
      NALU_HYPRE_ParCSRPCGSolve(solver, A, b, x0);

      /* Finalize solve timing */
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      /* Run info - needed logging turned on */
      NALU_HYPRE_PCGGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

      /* Destroy solver and preconditioner */
      NALU_HYPRE_ParCSRPCGDestroy(solver);
      if (solver_id == 1)
      {
         NALU_HYPRE_BoomerAMGDestroy(precond);
      }
      else if (solver_id == 3)
      {
         NALU_HYPRE_AMSDestroy(precond);
      }
   }

   if (solver_id == 5)
   {
      AMSDriverMatrixRead("mfem.M", &M);

      nalu_hypre_ParCSRMatrixMigrate(M, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));

      time_index = nalu_hypre_InitializeTiming("AME Setup");
      nalu_hypre_BeginTiming(time_index);

      /* Create AMS preconditioner and specify any parameters */
      NALU_HYPRE_AMSCreate(&precond);
      NALU_HYPRE_AMSSetDimension(precond, dim);
      NALU_HYPRE_AMSSetMaxIter(precond, 1);
      NALU_HYPRE_AMSSetTol(precond, 0.0);
      NALU_HYPRE_AMSSetCycleType(precond, cycle_type);
      NALU_HYPRE_AMSSetPrintLevel(precond, 0);
      NALU_HYPRE_AMSSetDiscreteGradient(precond, G);

      /* Vectors Gx, Gy and Gz */
      if (!coordinates)
      {
         NALU_HYPRE_AMSSetEdgeConstantVectors(precond, Gx, Gy, Gz);
      }

      /* Vectors x, y and z */
      if (coordinates)
      {
         NALU_HYPRE_AMSSetCoordinateVectors(precond, x, y, z);
      }

      /* Poisson matrices */
      if (h1_method)
      {
         NALU_HYPRE_AMSSetAlphaPoissonMatrix(precond, Aalpha);
         NALU_HYPRE_AMSSetBetaPoissonMatrix(precond, Abeta);
      }

      if (singular_problem)
      {
         NALU_HYPRE_AMSSetBetaPoissonMatrix(precond, NULL);
      }

      /* Smoothing and AMG options */
      NALU_HYPRE_AMSSetSmoothingOptions(precond, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
      NALU_HYPRE_AMSSetAlphaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                  amg_interp_type, amg_Pmax);
      NALU_HYPRE_AMSSetBetaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                 amg_interp_type, amg_Pmax);
      NALU_HYPRE_AMSSetAlphaAMGCoarseRelaxType(precond, coarse_rlx_type);
      NALU_HYPRE_AMSSetBetaAMGCoarseRelaxType(precond, coarse_rlx_type);

      /* Set up the AMS preconditioner */
      NALU_HYPRE_AMSSetup(precond, A, b, x0);

      /* Create AME object */
      NALU_HYPRE_AMECreate(&solver);

      /* Set main parameters */
      NALU_HYPRE_AMESetAMSSolver(solver, precond);
      NALU_HYPRE_AMESetMassMatrix(solver, M);
      NALU_HYPRE_AMESetBlockSize(solver, blockSize);

      /* Set additional parameters */
      NALU_HYPRE_AMESetMaxIter(solver, maxit); /* max iterations */
      NALU_HYPRE_AMESetTol(solver, tol); /* conv. tolerance */
      if (myid == 0)
      {
         NALU_HYPRE_AMESetPrintLevel(solver, 1);   /* print solve info */
      }
      else
      {
         NALU_HYPRE_AMESetPrintLevel(solver, 0);
      }

      /* Setup */
      NALU_HYPRE_AMESetup(solver);

      /* Finalize setup timing */
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("AME Solve");
      nalu_hypre_BeginTiming(time_index);

      /* Solve */
      NALU_HYPRE_AMESolve(solver);

      /* Finalize solve timing */
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      /* Destroy solver and preconditioner */
      NALU_HYPRE_AMEDestroy(solver);
      NALU_HYPRE_AMSDestroy(precond);
   }

   /* Save the solution */
   /* NALU_HYPRE_ParVectorPrint(x0,"x.ams"); */

   /* Clean-up */
   NALU_HYPRE_ParCSRMatrixDestroy(A);
   NALU_HYPRE_ParVectorDestroy(x0);
   NALU_HYPRE_ParVectorDestroy(b);
   NALU_HYPRE_ParCSRMatrixDestroy(G);

   if (M) { NALU_HYPRE_ParCSRMatrixDestroy(M); }

   if (Gx) { NALU_HYPRE_ParVectorDestroy(Gx); }
   if (Gy) { NALU_HYPRE_ParVectorDestroy(Gy); }
   if (Gz) { NALU_HYPRE_ParVectorDestroy(Gz); }

   if (x) { NALU_HYPRE_ParVectorDestroy(x); }
   if (y) { NALU_HYPRE_ParVectorDestroy(y); }
   if (z) { NALU_HYPRE_ParVectorDestroy(z); }

   if (Aalpha) { NALU_HYPRE_ParCSRMatrixDestroy(Aalpha); }
   if (Abeta) { NALU_HYPRE_ParCSRMatrixDestroy(Abeta); }

   if (zero_cond)
   {
      NALU_HYPRE_ParVectorDestroy(interior_nodes);
   }

   /* Finalize Hypre */
   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   if (NALU_HYPRE_GetError() && !myid)
   {
      nalu_hypre_fprintf(stderr, "nalu_hypre_error_flag = %d\n", NALU_HYPRE_GetError());
   }

   return 0;
}
