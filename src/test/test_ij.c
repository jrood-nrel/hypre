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
#include "_nalu_hypre_parcsr_mv.h"
#include "NALU_HYPRE_krylov.h"

#include "nalu_hypre_test.h"

/* begin lobpcg */

#define NO_SOLVER -9198

#include <time.h>

#include "NALU_HYPRE_lobpcg.h"

NALU_HYPRE_Int
BuildParIsoLaplacian( NALU_HYPRE_Int argc, char** argv, NALU_HYPRE_ParCSRMatrix *A_ptr );

/* end lobpcg */

NALU_HYPRE_Int BuildParFromFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                            NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParLaplacian (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                             NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParDifConv (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                           NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParFromOneFile2(NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_Int num_functions, NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildFuncsFromFiles (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_Int **dof_func_ptr );
NALU_HYPRE_Int BuildFuncsFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_Int **dof_func_ptr );
NALU_HYPRE_Int BuildRhsParFromOneFile2(NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                  NALU_HYPRE_Int *partitioning, NALU_HYPRE_ParVector *b_ptr );
NALU_HYPRE_Int BuildParLaplacian9pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParLaplacian27pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix *A_ptr );

#define SECOND_TIME 0

NALU_HYPRE_Int
main( NALU_HYPRE_Int   argc,
      char *argv[] )
{
   NALU_HYPRE_Int                 arg_index;
   NALU_HYPRE_Int                 print_usage;
   NALU_HYPRE_Int                 sparsity_known = 0;
   NALU_HYPRE_Int                 build_matrix_type;
   NALU_HYPRE_Int                 build_matrix_arg_index;
   NALU_HYPRE_Int                 build_rhs_type;
   NALU_HYPRE_Int                 build_rhs_arg_index;
   NALU_HYPRE_Int                 build_src_type;
   NALU_HYPRE_Int                 build_src_arg_index;
   NALU_HYPRE_Int                 build_funcs_type;
   NALU_HYPRE_Int                 build_funcs_arg_index;
   NALU_HYPRE_Int                 matrix_id;
   NALU_HYPRE_Int                 solver_id;
   NALU_HYPRE_Int                 precond_id;
   NALU_HYPRE_Int                 solver_type = 1;
   NALU_HYPRE_Int                 ioutdat;
   NALU_HYPRE_Int                 poutdat;
   NALU_HYPRE_Int                 debug_flag;
   NALU_HYPRE_Int                 ierr = 0;
   NALU_HYPRE_Int                 i, j, k;
   NALU_HYPRE_Int                 indx, rest, tms;
   NALU_HYPRE_Int                 max_levels = 25;
   NALU_HYPRE_Int                 num_iterations;
   NALU_HYPRE_Int                 pcg_num_its;
   NALU_HYPRE_Int                 dscg_num_its;
   NALU_HYPRE_Int                 pcg_max_its;
   NALU_HYPRE_Int                 dscg_max_its;
   NALU_HYPRE_Real          cf_tol = 0.9;
   NALU_HYPRE_Real          norm;
   NALU_HYPRE_Real          final_res_norm;
   void               *object;

   NALU_HYPRE_IJMatrix      ij_A;
   NALU_HYPRE_IJVector      ij_b;
   NALU_HYPRE_IJVector      ij_x;

   NALU_HYPRE_ParCSRMatrix  parcsr_A;
   NALU_HYPRE_ParVector     b;
   NALU_HYPRE_ParVector     x;

   NALU_HYPRE_Solver        amg_solver;
   NALU_HYPRE_Solver        pcg_solver;
   NALU_HYPRE_Solver        pcg_precond, pcg_precond_gotten;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 local_row;
   NALU_HYPRE_Int                *row_sizes;
   NALU_HYPRE_Int                *diag_sizes;
   NALU_HYPRE_Int                *offdiag_sizes;
   NALU_HYPRE_Int                 size;
   NALU_HYPRE_Int                *col_inds;
   NALU_HYPRE_Int                *dof_func;
   NALU_HYPRE_Int                 num_functions = 1;

   NALU_HYPRE_Int                 time_index;
   MPI_Comm            comm = nalu_hypre_MPI_COMM_WORLD;
   NALU_HYPRE_Int M, N;
   NALU_HYPRE_Int first_local_row, last_local_row, local_num_rows;
   NALU_HYPRE_Int first_local_col, last_local_col, local_num_cols;
   NALU_HYPRE_Int local_num_vars;
   NALU_HYPRE_Int variant, overlap, domain_type;
   NALU_HYPRE_Real schwarz_rlx_weight;
   NALU_HYPRE_Real *values;

   const NALU_HYPRE_Real dt_inf = 1.e40;
   NALU_HYPRE_Real dt = dt_inf;

   NALU_HYPRE_Int      print_system = 0;

   /* begin lobpcg */

   NALU_HYPRE_Int lobpcgFlag = 0;
   NALU_HYPRE_Int lobpcgGen = 0;
   NALU_HYPRE_Int constrained = 0;
   NALU_HYPRE_Int vFromFileFlag = 0;
   NALU_HYPRE_Int lobpcgSeed = 0;
   NALU_HYPRE_Int blockSize = 1;
   NALU_HYPRE_Int verbosity = 1;
   NALU_HYPRE_Int iterations;
   NALU_HYPRE_Int maxIterations = 100;
   NALU_HYPRE_Int checkOrtho = 0;
   NALU_HYPRE_Int printLevel = 0;
   NALU_HYPRE_Int pcgIterations = 0;
   NALU_HYPRE_Int pcgMode = 1;
   NALU_HYPRE_Real pcgTol = 1e-2;
   NALU_HYPRE_Real nonOrthF;

   FILE* filePtr;

   mv_MultiVectorPtr eigenvectors = NULL;
   mv_MultiVectorPtr constraints = NULL;
   mv_MultiVectorPtr workspace = NULL;

   NALU_HYPRE_Real* eigenvalues = NULL;

   NALU_HYPRE_Real* residuals;
   utilities_FortranMatrix* residualNorms;
   utilities_FortranMatrix* residualNormsHistory;
   utilities_FortranMatrix* eigenvaluesHistory;
   utilities_FortranMatrix* printBuffer;
   utilities_FortranMatrix* gramXX;
   utilities_FortranMatrix* identity;

   NALU_HYPRE_Solver        lobpcg_solver;

   mv_InterfaceInterpreter* interpreter;
   NALU_HYPRE_MatvecFunctions matvec_fn;

   NALU_HYPRE_IJMatrix      ij_B;
   NALU_HYPRE_ParCSRMatrix  parcsr_B;

   /* end lobpcg */

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

   build_matrix_type = 2;
   build_matrix_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;
   build_src_type = -1;
   build_src_arg_index = argc;
   build_funcs_type = 0;
   build_funcs_arg_index = argc;
   relax_default = 3;
   debug_flag = 0;

   matrix_id = NALU_HYPRE_PARCSR;
   solver_id = NALU_HYPRE_PCG;
   precond_id = 0;

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
         build_matrix_type      = -1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromparcsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonecsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
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
      else if ( strcmp(argv[arg_index], "-funcsfromonefile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 1;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromfile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 2;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-exact_size") == 0 )
      {
         arg_index++;
         sparsity_known = 1;
      }
      else if ( strcmp(argv[arg_index], "-storage_low") == 0 )
      {
         arg_index++;
         sparsity_known = 2;
      }
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-matrix") == 0 )
      {
         arg_index++;
         if ( strcmp(argv[arg_index], "parcsr") == 0 )
         {
            matrix_id = NALU_HYPRE_PARCSR;
         }
         else if ( strcmp(argv[arg_index], "sstruct") == 0 )
         {
            matrix_id = NALU_HYPRE_SSTRUCT;
         }
         else if ( strcmp(argv[arg_index], "struct") == 0 )
         {
            matrix_id = NALU_HYPRE_STRUCT;
         };
         arg_index++;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;

         /* begin lobpcg */
         if ( strcmp(argv[arg_index], "none") == 0 )
         {
            solver_id = NO_SOLVER;
         }                          /* end lobpcg */
         else if ( strcmp(argv[arg_index], "boomeramg") == 0  ||
                   strcmp(argv[arg_index], "amg") == 0 )
         {
            solver_id = NALU_HYPRE_BOOMERAMG;
         }
         else if ( strcmp(argv[arg_index], "bicgstab") == 0 )
         {
            solver_id = NALU_HYPRE_BICGSTAB;
         }
         else if ( strcmp(argv[arg_index], "cgnr") == 0 )
         {
            solver_id = NALU_HYPRE_CGNR;
         }
         else if ( strcmp(argv[arg_index], "diagscale") == 0 )
         {
            solver_id = NALU_HYPRE_DIAGSCALE;
         }
         else if ( strcmp(argv[arg_index], "euclid") == 0 )
         {
            solver_id = NALU_HYPRE_EUCLID;
         }
         else if ( strcmp(argv[arg_index], "gmres") == 0 )
         {
            solver_id = NALU_HYPRE_GMRES;
         }
         else if ( strcmp(argv[arg_index], "gsmg") == 0 )
         {
            solver_id = NALU_HYPRE_GSMG;
         }
         else if ( strcmp(argv[arg_index], "hybrid") == 0 )
         {
            solver_id = NALU_HYPRE_HYBRID;
         }
         else if ( strcmp(argv[arg_index], "jacobi") == 0 )
         {
            solver_id = NALU_HYPRE_JACOBI;
         }
         else if ( strcmp(argv[arg_index], "parasails") == 0 )
         {
            solver_id = NALU_HYPRE_PARASAILS;
         }
         else if ( strcmp(argv[arg_index], "pcg") == 0 )
         {
            solver_id = NALU_HYPRE_PCG;
         }
         else if ( strcmp(argv[arg_index], "pfmg") == 0 )
         {
            solver_id = NALU_HYPRE_PFMG;
         }
         else if ( strcmp(argv[arg_index], "pilut") == 0 )
         {
            solver_id = NALU_HYPRE_PILUT;
         }
         else if ( strcmp(argv[arg_index], "schwarz") == 0 )
         {
            solver_id = NALU_HYPRE_SCHWARZ;
         }
         else if ( strcmp(argv[arg_index], "smg") == 0 )
         {
            solver_id = NALU_HYPRE_SMG;
         }
         else if ( strcmp(argv[arg_index], "sparsemsg") == 0 )
         {
            solver_id = NALU_HYPRE_SPARSEMSG;
         }
         else if ( strcmp(argv[arg_index], "split") == 0 )
         {
            solver_id = NALU_HYPRE_SPLIT;
         }
         else if ( strcmp(argv[arg_index], "splitpfmg") == 0 )
         {
            solver_id = NALU_HYPRE_SPLITPFMG;
         }
         else if ( strcmp(argv[arg_index], "splitsmg") == 0 )
         {
            solver_id = NALU_HYPRE_SPLITSMG;
         }
         else if ( strcmp(argv[arg_index], "syspfmg") == 0 )
         {
            solver_id = NALU_HYPRE_SYSPFMG;
         };
         arg_index++;
      }
      else if ( strcmp(argv[arg_index], "-precond") == 0 )
      {
         arg_index++;

         /* begin lobpcg */
         if ( strcmp(argv[arg_index], "none") == 0 )
         {
            precond_id = NO_SOLVER;
         }                          /* end lobpcg */
         else if ( strcmp(argv[arg_index], "boomeramg") == 0  ||
                   strcmp(argv[arg_index], "amg") == 0 )
         {
            precond_id = NALU_HYPRE_BOOMERAMG;
         }
         else if ( strcmp(argv[arg_index], "bicgstab") == 0 )
         {
            precond_id = NALU_HYPRE_BICGSTAB;
         }
         else if ( strcmp(argv[arg_index], "cgnr") == 0 )
         {
            precond_id = NALU_HYPRE_CGNR;
         }
         else if ( strcmp(argv[arg_index], "diagscale") == 0 )
         {
            precond_id = NALU_HYPRE_DIAGSCALE;
         }
         else if ( strcmp(argv[arg_index], "euclid") == 0 )
         {
            precond_id = NALU_HYPRE_EUCLID;
         }
         else if ( strcmp(argv[arg_index], "gmres") == 0 )
         {
            precond_id = NALU_HYPRE_GMRES;
         }
         else if ( strcmp(argv[arg_index], "gsmg") == 0 )
         {
            precond_id = NALU_HYPRE_GSMG;
         }
         else if ( strcmp(argv[arg_index], "hybrid") == 0 )
         {
            precond_id = NALU_HYPRE_HYBRID;
         }
         else if ( strcmp(argv[arg_index], "jacobi") == 0 )
         {
            precond_id = NALU_HYPRE_JACOBI;
         }
         else if ( strcmp(argv[arg_index], "parasails") == 0 )
         {
            precond_id = NALU_HYPRE_PARASAILS;
         }
         else if ( strcmp(argv[arg_index], "pcg") == 0 )
         {
            precond_id = NALU_HYPRE_PCG;
         }
         else if ( strcmp(argv[arg_index], "pfmg") == 0 )
         {
            precond_id = NALU_HYPRE_PFMG;
         }
         else if ( strcmp(argv[arg_index], "pilut") == 0 )
         {
            precond_id = NALU_HYPRE_PILUT;
         }
         else if ( strcmp(argv[arg_index], "schwarz") == 0 )
         {
            precond_id = NALU_HYPRE_SCHWARZ;
         }
         else if ( strcmp(argv[arg_index], "smg") == 0 )
         {
            precond_id = NALU_HYPRE_SMG;
         }
         else if ( strcmp(argv[arg_index], "sparsemsg") == 0 )
         {
            precond_id = NALU_HYPRE_SPARSEMSG;
         }
         else if ( strcmp(argv[arg_index], "split") == 0 )
         {
            precond_id = NALU_HYPRE_SPLIT;
         }
         else if ( strcmp(argv[arg_index], "splitpfmg") == 0 )
         {
            precond_id = NALU_HYPRE_SPLITPFMG;
         }
         else if ( strcmp(argv[arg_index], "splitsmg") == 0 )
         {
            precond_id = NALU_HYPRE_SPLITSMG;
         }
         else if ( strcmp(argv[arg_index], "syspfmg") == 0 )
         {
            precond_id = NALU_HYPRE_SYSPFMG;
         };
         arg_index++;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 0;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromonefile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 1;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsisone") == 0 )
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
      else if ( strcmp(argv[arg_index], "-xisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 4;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         build_rhs_type      = 5;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcfromfile") == 0 )
      {
         arg_index++;
         build_src_type      = 0;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcfromonefile") == 0 )
      {
         arg_index++;
         build_src_type      = 1;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcisone") == 0 )
      {
         arg_index++;
         build_src_type      = 2;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcrand") == 0 )
      {
         arg_index++;
         build_src_type      = 3;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srczero") == 0 )
      {
         arg_index++;
         build_src_type      = 4;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-cljp") == 0 )
      {
         arg_index++;
         coarsen_type      = 0;
      }
      else if ( strcmp(argv[arg_index], "-cljp1") == 0 )
      {
         arg_index++;
         coarsen_type      = 7;
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
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index++;
         relax_default = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smtype") == 0 )
      {
         arg_index++;
         smooth_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smlv") == 0 )
      {
         arg_index++;
         smooth_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxl") == 0 )
      {
         arg_index++;
         max_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         debug_flag = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nf") == 0 )
      {
         arg_index++;
         num_functions = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns") == 0 )
      {
         arg_index++;
         num_sweep = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sns") == 0 )
      {
         arg_index++;
         smooth_num_sweeps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dt") == 0 )
      {
         arg_index++;
         dt = atof(argv[arg_index++]);
         build_rhs_type = -1;
         if ( build_src_type == -1 ) { build_src_type = 2; }
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      /* begin lobpcg */
      else if ( strcmp(argv[arg_index], "-lobpcg") == 0 )
      {
         /* use lobpcg */
         arg_index++;
         lobpcgFlag = 1;
      }
      else if ( strcmp(argv[arg_index], "-gen") == 0 )
      {
         /* generalized evp */
         arg_index++;
         lobpcgGen = 1;
      }
      else if ( strcmp(argv[arg_index], "-con") == 0 )
      {
         /* constrained evp */
         arg_index++;
         constrained = 1;
      }
      else if ( strcmp(argv[arg_index], "-orthchk") == 0 )
      {
         /* lobpcg: check orthonormality */
         arg_index++;
         checkOrtho = 1;
      }
      else if ( strcmp(argv[arg_index], "-vfromfile") == 0 )
      {
         /* lobpcg: get initial vectors from file */
         arg_index++;
         vFromFileFlag = 1;
      }
      else if ( strcmp(argv[arg_index], "-vrand") == 0 )
      {
         /* lobpcg: block size */
         arg_index++;
         blockSize = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seed") == 0 )
      {
         /* lobpcg: seed for srand */
         arg_index++;
         lobpcgSeed = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-itr") == 0 )
      {
         /* lobpcg: max # of iterations */
         arg_index++;
         maxIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-verb") == 0 )
      {
         /* lobpcg: verbosity level */
         arg_index++;
         verbosity = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vout") == 0 )
      {
         /* lobpcg: print level */
         arg_index++;
         printLevel = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgitr") == 0 )
      {
         /* lobpcg: inner pcg iterations */
         arg_index++;
         pcgIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgtol") == 0 )
      {
         /* lobpcg: inner pcg iterations */
         arg_index++;
         pcgTol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgmode") == 0 )
      {
         /* lobpcg: initial guess for inner pcg */
         arg_index++;      /* 0: zero, otherwise rhs */
         pcgMode = atoi(argv[arg_index++]);
      }
      /* end lobpcg */
      else
      {
         arg_index++;
      }
   }

   /* begin lobpcg */

   if ( solver_id == NALU_HYPRE_BOOMERAMG && lobpcgFlag )
   {
      solver_id = NALU_HYPRE_BOOMERAMG;
      precond_id = NALU_HYPRE_PCG;
   };

   /* end lobpcg */

   if (solver_id == NALU_HYPRE_PARASAILS)
   {
      max_levels = 1;
   }

   /* defaults for BoomerAMG */
   if (solver_id == NALU_HYPRE_BOOMERAMG)
   {
      strong_threshold = 0.25;
      trunc_factor = 0.;
      cycle_type = 1;

      num_grid_sweeps   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_type   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points = nalu_hypre_CTAlloc(NALU_HYPRE_Int *, 4, NALU_HYPRE_MEMORY_HOST);
      relax_weight      = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  max_levels, NALU_HYPRE_MEMORY_HOST);
      omega      = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  max_levels, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < max_levels; i++)
      {
         relax_weight[i] = 1.;
         omega[i] = 1.;
      }

      /* for CGNR preconditioned with Boomeramg, only relaxation scheme 0 is
         implemented, i.e. Jacobi relaxation */
      if (precond_id == NALU_HYPRE_CGNR)
      {
         /* fine grid */
         relax_default = 7;
         grid_relax_type[0] = relax_default;
         num_grid_sweeps[0] = num_sweep;
         grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sweep; i++)
         {
            grid_relax_points[0][i] = 0;
         }
         /* down cycle */
         grid_relax_type[1] = relax_default;
         num_grid_sweeps[1] = num_sweep;
         grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sweep; i++)
         {
            grid_relax_points[1][i] = 0;
         }
         /* up cycle */
         grid_relax_type[2] = relax_default;
         num_grid_sweeps[2] = num_sweep;
         grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sweep; i++)
         {
            grid_relax_points[2][i] = 0;
         }
      }
      else if (coarsen_type == 5)
      {
         /* fine grid */
         num_grid_sweeps[0] = 3;
         grid_relax_type[0] = relax_default;
         grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  3, NALU_HYPRE_MEMORY_HOST);
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
         grid_relax_type[0] = relax_default;
         /*num_grid_sweeps[0] = num_sweep;
         grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i=0; i<num_sweep; i++)
         {
            grid_relax_points[0][i] = 0;
         } */
         num_grid_sweeps[0] = 2 * num_sweep;
         grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2 * num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < 2 * num_sweep; i += 2)
         {
            grid_relax_points[0][i] = 1;
            grid_relax_points[0][i + 1] = -1;
         }

         /* down cycle */
         grid_relax_type[1] = relax_default;
         /* num_grid_sweeps[1] = num_sweep;
         grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i=0; i<num_sweep; i++)
         {
            grid_relax_points[1][i] = 0;
         } */
         num_grid_sweeps[1] = 2 * num_sweep;
         grid_relax_type[1] = relax_default;
         grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2 * num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < 2 * num_sweep; i += 2)
         {
            grid_relax_points[1][i] = 1;
            grid_relax_points[1][i + 1] = -1;
         }

         /* up cycle */
         grid_relax_type[2] = relax_default;
         /* num_grid_sweeps[2] = num_sweep;
         grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i=0; i<num_sweep; i++)
         {
            grid_relax_points[2][i] = 0;
         } */
         num_grid_sweeps[2] = 2 * num_sweep;
         grid_relax_type[2] = relax_default;
         grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2 * num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < 2 * num_sweep; i += 2)
         {
            grid_relax_points[2][i] = -1;
            grid_relax_points[2][i + 1] = 1;
         }
      }

      /* coarsest grid */
      num_grid_sweeps[3] = 1;
      grid_relax_type[3] = 9;
      grid_relax_points[3] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[3][0] = 0;
   }

   /* defaults for Schwarz */

   variant = 0;  /* multiplicative */
   overlap = 1;  /* 1 layer overlap */
   domain_type = 2; /* through agglomeration */
   schwarz_rlx_weight = 1.;

   /* defaults for GMRES */

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
         if (solver_id == NALU_HYPRE_BOOMERAMG || solver_id == NALU_HYPRE_GSMG ||
             (solver_id == NALU_HYPRE_HYBRID && precond_id == NALU_HYPRE_BOOMERAMG))
         {
            relax_weight[0] = atof(argv[arg_index++]);
            for (i = 1; i < max_levels; i++)
            {
               relax_weight[i] = relax_weight[0];
            }
         }
      }
      else if ( strcmp(argv[arg_index], "-om") == 0 )
      {
         arg_index++;
         if (solver_id == NALU_HYPRE_BOOMERAMG || solver_id == NALU_HYPRE_GSMG ||
             (solver_id == NALU_HYPRE_HYBRID && precond_id == NALU_HYPRE_BOOMERAMG))
         {
            omega[0] = atof(argv[arg_index++]);
            for (i = 1; i < max_levels; i++)
            {
               omega[i] = omega[0];
            }
         }
      }
      else if ( strcmp(argv[arg_index], "-sw") == 0 )
      {
         arg_index++;
         schwarz_rlx_weight = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index++;
         max_row_sum  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_th") == 0 )
      {
         arg_index++;
         sai_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_filt") == 0 )
      {
         arg_index++;
         sai_filter  = atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-solver_type") == 0 )
      {
         arg_index++;
         solver_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-iout") == 0 )
      {
         arg_index++;
         ioutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pout") == 0 )
      {
         arg_index++;
         poutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-var") == 0 )
      {
         arg_index++;
         variant  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ov") == 0 )
      {
         arg_index++;
         overlap  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dom") == 0 )
      {
         arg_index++;
         domain_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mu") == 0 )
      {
         arg_index++;
         cycle_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-numsamp") == 0 )
      {
         arg_index++;
         gsmg_samples  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-interptype") == 0 )
      {
         arg_index++;
         interp_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
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
      nalu_hypre_printf("  -fromfile <filename>       : ");
      nalu_hypre_printf("matrix read from multiple files (IJ format)\n");
      nalu_hypre_printf("  -fromparcsrfile <filename> : ");
      nalu_hypre_printf("matrix read from multiple files (ParCSR format)\n");
      nalu_hypre_printf("  -fromonecsrfile <filename> : ");
      nalu_hypre_printf("matrix read from a single file (CSR format)\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -laplacian [<options>] : build 5pt 2D laplacian problem (default) \n");
      nalu_hypre_printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
      nalu_hypre_printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
      nalu_hypre_printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      nalu_hypre_printf("    -n <nx> <ny> <nz>    : total problem size \n");
      nalu_hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      nalu_hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      nalu_hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -exact_size            : inserts immediately into ParCSR structure\n");
      nalu_hypre_printf("  -storage_low           : allocates not enough storage for aux struct\n");
      nalu_hypre_printf("  -concrete_parcsr       : use parcsr matrix type as concrete type\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -rhsfromfile           : ");
      nalu_hypre_printf("rhs read from multiple files (IJ format)\n");
      nalu_hypre_printf("  -rhsfromonefile        : ");
      nalu_hypre_printf("rhs read from a single file (CSR format)\n");
      nalu_hypre_printf("  -rhsrand               : rhs is random vector\n");
      nalu_hypre_printf("  -rhsisone              : rhs is vector with unit components (default)\n");
      nalu_hypre_printf("  -xisone                : solution of all ones\n");
      nalu_hypre_printf("  -rhszero               : rhs is zero vector\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -dt <val>              : specify finite backward Euler time step\n");
      nalu_hypre_printf("                         :    -rhsfromfile, -rhsfromonefile, -rhsrand,\n");
      nalu_hypre_printf("                         :    -rhsrand, or -xisone will be ignored\n");
      nalu_hypre_printf("  -srcfromfile           : ");
      nalu_hypre_printf("backward Euler source read from multiple files (IJ format)\n");
      nalu_hypre_printf("  -srcfromonefile        : ");
      nalu_hypre_printf("backward Euler source read from a single file (IJ format)\n");
      nalu_hypre_printf("  -srcrand               : ");
      nalu_hypre_printf("backward Euler source is random vector with components in range 0 - 1\n");
      nalu_hypre_printf("  -srcisone              : ");
      nalu_hypre_printf("backward Euler source is vector with unit components (default)\n");
      nalu_hypre_printf("  -srczero               : ");
      nalu_hypre_printf("backward Euler source is zero-vector\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -matrix <ID>           : matrix ID\n");
      nalu_hypre_printf("    allowed options:  \n");
      nalu_hypre_printf("                parcsr   \n");
      nalu_hypre_printf("                sstruct  \n");
      nalu_hypre_printf("                struct   \n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -solver <ID>           : solver ID\n");
      nalu_hypre_printf("    allowed options:  \n");
      nalu_hypre_printf("                amg || boomeramg   \n");
      nalu_hypre_printf("                bicgstab           \n");
      nalu_hypre_printf("                cgnr               \n");
      nalu_hypre_printf("                diagscale || ds    \n");
      nalu_hypre_printf("                euclid             \n");
      nalu_hypre_printf("                gmres              \n");
      nalu_hypre_printf("                gsmg               \n");
      nalu_hypre_printf("                hybrid             \n");
      nalu_hypre_printf("                jacobi             \n");
      nalu_hypre_printf("                parasails          \n");
      nalu_hypre_printf("                pcg                \n");
      nalu_hypre_printf("                pfmg               \n");
      nalu_hypre_printf("                pilut              \n");
      nalu_hypre_printf("                schwarz            \n");
      nalu_hypre_printf("                smg                \n");
      nalu_hypre_printf("                sparsemsg          \n");
      nalu_hypre_printf("                split              \n");
      nalu_hypre_printf("                splitpfmg          \n");
      nalu_hypre_printf("                splitsmg           \n");
      nalu_hypre_printf("                syspfmg            \n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -preconditioner <ID>  : precond ID\n");
      nalu_hypre_printf("    allowed options:  \n");
      nalu_hypre_printf("                amg || boomeramg   \n");
      nalu_hypre_printf("                bicgstab           \n");
      nalu_hypre_printf("                cgnr               \n");
      nalu_hypre_printf("                diagscale || ds    \n");
      nalu_hypre_printf("                euclid             \n");
      nalu_hypre_printf("                gmres              \n");
      nalu_hypre_printf("                gsmg               \n");
      nalu_hypre_printf("                hybrid             \n");
      nalu_hypre_printf("                jacobi             \n");
      nalu_hypre_printf("                parasails          \n");
      nalu_hypre_printf("                pcg                \n");
      nalu_hypre_printf("                pfmg               \n");
      nalu_hypre_printf("                pilut              \n");
      nalu_hypre_printf("                schwarz            \n");
      nalu_hypre_printf("                smg                \n");
      nalu_hypre_printf("                sparsemsg          \n");
      nalu_hypre_printf("                split              \n");
      nalu_hypre_printf("                splitpfmg          \n");
      nalu_hypre_printf("                splitsmg           \n");
      nalu_hypre_printf("                syspfmg            \n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -cljp                 : CLJP coarsening \n");
      nalu_hypre_printf("  -ruge                 : Ruge coarsening (local)\n");
      nalu_hypre_printf("  -ruge3                : third pass on boundary\n");
      nalu_hypre_printf("  -ruge3c               : third pass on boundary, keep c-points\n");
      nalu_hypre_printf("  -ruge2b               : 2nd pass is global\n");
      nalu_hypre_printf("  -rugerlx              : relaxes special points\n");
      nalu_hypre_printf("  -falgout              : local ruge followed by LJP\n");
      nalu_hypre_printf("  -nohybrid             : no switch in coarsening\n");
      nalu_hypre_printf("  -gm                   : use global measures\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -rlx  <val>            : relaxation type\n");
      nalu_hypre_printf("       0=Weighted Jacobi  \n");
      nalu_hypre_printf("       1=Gauss-Seidel (very slow!)  \n");
      nalu_hypre_printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
      nalu_hypre_printf("  -ns <val>              : Use <val> sweeps on each level\n");
      nalu_hypre_printf("                           (default C/F down, F/C up, F/C fine\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n");
      nalu_hypre_printf("  -th   <val>            : set AMG threshold Theta = val \n");
      nalu_hypre_printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
      nalu_hypre_printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");
      nalu_hypre_printf("  -nf <val>              : set number of functions for systems AMG\n");
      nalu_hypre_printf("  -numsamp <val>         : set number of sample vectors for GSMG\n");
      nalu_hypre_printf("  -interptype <val>      : set to 1 to get LS interpolation\n");
      nalu_hypre_printf("                         : set to 2 to get interpolation for hyperbolic equations\n");

      nalu_hypre_printf("  -solver_type <val>     : sets solver within Hybrid solver\n");
      nalu_hypre_printf("                         : 1  PCG  (default)\n");
      nalu_hypre_printf("                         : 2  GMRES\n");
      nalu_hypre_printf("                         : 3  BiCGSTAB\n");

      nalu_hypre_printf("  -w   <val>             : set Jacobi relax weight = val\n");
      nalu_hypre_printf("  -k   <val>             : dimension Krylov space for GMRES\n");
      nalu_hypre_printf("  -mxl  <val>            : maximum number of levels (AMG, ParaSAILS)\n");
      nalu_hypre_printf("  -tol  <val>            : set solver convergence tolerance = val\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -sai_th   <val>        : set ParaSAILS threshold = val \n");
      nalu_hypre_printf("  -sai_filt <val>        : set ParaSAILS filter = val \n");
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
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -print                 : print out the system\n");
      nalu_hypre_printf("\n");

      /* begin lobpcg */

      nalu_hypre_printf("LOBPCG options:\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -lobpcg                 : run LOBPCG instead of PCG\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -gen                    : solve generalized EVP with B = Laplacian\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -con                    : solve constrained EVP using 'vectors.*.*'\n");
      nalu_hypre_printf("                            as constraints (see -vout 1 below)\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -solver none            : no HYPRE preconditioner is used\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -itr <val>              : maximal number of LOBPCG iterations\n");
      nalu_hypre_printf("                            (default 100);\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -vrand <val>            : compute <val> eigenpairs using random\n");
      nalu_hypre_printf("                            initial vectors (default 1)\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -seed <val>             : use <val> as the seed for the random\n");
      nalu_hypre_printf("                            number generator(default seed is based\n");
      nalu_hypre_printf("                            on the time of the run)\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -vfromfile              : read initial vectors from files\n");
      nalu_hypre_printf("                            vectors.i.j where i is vector number\n");
      nalu_hypre_printf("                            and j is processor number\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -orthchk                : check eigenvectors for orthonormality\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -verb <val>             : verbosity level\n");
      nalu_hypre_printf("  -verb 0                 : no print\n");
      nalu_hypre_printf("  -verb 1                 : print initial eigenvalues and residuals,\n");
      nalu_hypre_printf("                            the iteration number, the number of\n");
      nalu_hypre_printf("                            non-convergent eigenpairs and final\n");
      nalu_hypre_printf("                            eigenvalues and residuals (default)\n");
      nalu_hypre_printf("  -verb 2                 : print eigenvalues and residuals on each\n");
      nalu_hypre_printf("                            iteration\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -pcgitr <val>           : maximal number of inner PCG iterations\n");
      nalu_hypre_printf("                            for preconditioning (default 1);\n");
      nalu_hypre_printf("                            if <val> = 0 then the preconditioner\n");
      nalu_hypre_printf("                            is applied directly\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -pcgtol <val>           : residual tolerance for inner iterations\n");
      nalu_hypre_printf("                            (default 0.01)\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -vout <val>             : file output level\n");
      nalu_hypre_printf("  -vout 0                 : no files created (default)\n");
      nalu_hypre_printf("  -vout 1                 : write eigenvalues to values.txt, residuals\n");
      nalu_hypre_printf("                            to residuals.txt and eigenvectors to \n");
      nalu_hypre_printf("                            vectors.i.j where i is vector number\n");
      nalu_hypre_printf("                            and j is processor number\n");
      nalu_hypre_printf("  -vout 2                 : in addition to the above, write the\n");
      nalu_hypre_printf("                            eigenvalues history (the matrix whose\n");
      nalu_hypre_printf("                            i-th column contains eigenvalues at\n");
      nalu_hypre_printf("                            (i+1)-th iteration) to val_hist.txt and\n");
      nalu_hypre_printf("                            residuals history to res_hist.txt\n");
      nalu_hypre_printf("\nNOTE: in this test driver LOBPCG only works with solvers 1, 2, 8, 12, 14 and 43\n");
      nalu_hypre_printf("\ndefault solver is 1\n");
      nalu_hypre_printf("\n");

      /* end lobpcg */

      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("Running with these driver parameters:\n");
      nalu_hypre_printf("  matrix ID    = %d\n", matrix_id);
      nalu_hypre_printf("  solver ID    = %d\n", solver_id);
      nalu_hypre_printf("  precond ID   = %d\n\n", precond_id);
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   if ( myid == 0 && dt != dt_inf)
   {
      nalu_hypre_printf("  Backward Euler time step with dt = %e\n", dt);
      nalu_hypre_printf("  Dirichlet 0 BCs are implicit in the spatial operator\n");
   }

   if ( build_matrix_type == -1 )
   {
      NALU_HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                          NALU_HYPRE_PARCSR, &ij_A );
   }
   else if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParFromOneFile2(argc, argv, build_matrix_arg_index, num_functions,
                           &parcsr_A);
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else
   {
      nalu_hypre_printf("You have asked for an unsupported problem with\n");
      nalu_hypre_printf("build_matrix_type = %d.\n", build_matrix_type);
      return (-1);
   }

   time_index = nalu_hypre_InitializeTiming("Spatial operator");
   nalu_hypre_BeginTiming(time_index);

   if (build_matrix_type < 0)
   {
      ierr += NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_A = (NALU_HYPRE_ParCSRMatrix) object;

      ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                              &first_local_row, &last_local_row,
                                              &first_local_col, &last_local_col );

      local_num_rows = last_local_row - first_local_row + 1;
      local_num_cols = last_local_col - first_local_col + 1;
   }
   else
   {

      /*-----------------------------------------------------------
       * Copy the parcsr matrix into the IJMatrix through interface calls
       *-----------------------------------------------------------*/

      ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                              &first_local_row, &last_local_row,
                                              &first_local_col, &last_local_col );

      local_num_rows = last_local_row - first_local_row + 1;
      local_num_cols = last_local_col - first_local_col + 1;
      ierr += NALU_HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );

      ierr += NALU_HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                    first_local_col, last_local_col,
                                    &ij_A );

      ierr += NALU_HYPRE_IJMatrixSetObjectType( ij_A, NALU_HYPRE_PARCSR );


      /* the following shows how to build an IJMatrix if one has only an
         estimate for the row sizes */
      if (sparsity_known == 1)
      {
         /*  build IJMatrix using exact row_sizes for diag and offdiag */

         diag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
         offdiag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
         local_row = 0;
         for (i = first_local_row; i <= last_local_row; i++)
         {
            ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                              &col_inds, &values );

            for (j = 0; j < size; j++)
            {
               if (col_inds[j] < first_local_row || col_inds[j] > last_local_row)
               {
                  offdiag_sizes[local_row]++;
               }
               else
               {
                  diag_sizes[local_row]++;
               }
            }
            local_row++;
            ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                                  &col_inds, &values );
         }
         ierr += NALU_HYPRE_IJMatrixSetDiagOffdSizes( ij_A,
                                                 (const NALU_HYPRE_Int *) diag_sizes,
                                                 (const NALU_HYPRE_Int *) offdiag_sizes );
         nalu_hypre_TFree(diag_sizes, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(offdiag_sizes, NALU_HYPRE_MEMORY_HOST);

         ierr = NALU_HYPRE_IJMatrixInitialize( ij_A );

         for (i = first_local_row; i <= last_local_row; i++)
         {
            ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                              &col_inds, &values );

            ierr += NALU_HYPRE_IJMatrixSetValues( ij_A, 1, &size, &i,
                                             (const NALU_HYPRE_Int *) col_inds,
                                             (const NALU_HYPRE_Real *) values );

            ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                                  &col_inds, &values );
         }
      }
      else
      {
         row_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);

         size = 5; /* this is in general too low, and supposed to test
                    the capability of the reallocation of the interface */

         if (sparsity_known == 0) /* tries a more accurate estimate of the
                                    storage */
         {
            if (build_matrix_type == 2) { size = 7; }
            if (build_matrix_type == 3) { size = 9; }
            if (build_matrix_type == 4) { size = 27; }
         }

         for (i = 0; i < local_num_rows; i++)
         {
            row_sizes[i] = size;
         }

         ierr = NALU_HYPRE_IJMatrixSetRowSizes ( ij_A, (const NALU_HYPRE_Int *) row_sizes );

         nalu_hypre_TFree(row_sizes, NALU_HYPRE_MEMORY_HOST);

         ierr = NALU_HYPRE_IJMatrixInitialize( ij_A );

         /* Loop through all locally stored rows and insert them into ij_matrix */
         for (i = first_local_row; i <= last_local_row; i++)
         {
            ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                              &col_inds, &values );

            ierr += NALU_HYPRE_IJMatrixSetValues( ij_A, 1, &size, &i,
                                             (const NALU_HYPRE_Int *) col_inds,
                                             (const NALU_HYPRE_Real *) values );

            ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                                  &col_inds, &values );
         }
      }

      ierr += NALU_HYPRE_IJMatrixAssemble( ij_A );

   }

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("IJ Matrix Setup", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   if (ierr)
   {
      nalu_hypre_printf("Error in driver building IJMatrix from parcsr matrix. \n");
      return (-1);
   }

   /* This is to emphasize that one can IJMatrixAddToValues after an
      IJMatrixRead or an IJMatrixAssemble.  After an IJMatrixRead,
      assembly is unnecessary if the sparsity pattern of the matrix is
      not changed somehow.  If one has not used IJMatrixRead, one has
      the opportunity to IJMatrixAddTo before a IJMatrixAssemble. */


   ierr = NALU_HYPRE_IJMatrixInitialize( ij_A );

   /* Loop through all locally stored rows and insert them into ij_matrix */
   for (i = first_local_row; i <= last_local_row; i++)
   {
      ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                        &col_inds, &values );

      ierr += NALU_HYPRE_IJMatrixSetValues( ij_A, 1, &size, &i,
                                       (const NALU_HYPRE_Int *) col_inds,
                                       (const NALU_HYPRE_Real *) values );

      ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                            &col_inds, &values );
   }

   /* If sparsity pattern is not changed since last IJMatrixAssemble call,
      this should be a no-op */

   ierr += NALU_HYPRE_IJMatrixAssemble( ij_A );

   /*-----------------------------------------------------------
    * Fetch the resulting underlying matrix out
    *-----------------------------------------------------------*/

   if (build_matrix_type > -1)
   {
      ierr += NALU_HYPRE_ParCSRMatrixDestroy(parcsr_A);
   }

   ierr += NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
   parcsr_A = (NALU_HYPRE_ParCSRMatrix) object;

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = nalu_hypre_InitializeTiming("RHS and Initial Guess");
   nalu_hypre_BeginTiming(time_index);

   if ( build_rhs_type == 0 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      ierr = NALU_HYPRE_IJVectorRead( argv[build_rhs_arg_index], nalu_hypre_MPI_COMM_WORLD,
                                 NALU_HYPRE_PARCSR, &ij_b );
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 1 )
   {
      nalu_hypre_printf("build_rhs_type == 1 not currently implemented\n");
      return (-1);

#if 0
      /* RHS */
      BuildRhsParFromOneFile2(argc, argv, build_rhs_arg_index, part_b, &b);
#endif
   }
   else if ( build_rhs_type == 2 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector has unit components\n");
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 1.0;
      }
      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 3 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector has random components and unit 2-norm\n");
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* For purposes of this test, NALU_HYPRE_ParVector functions are used, but these are
         not necessary.  For a clean use of the interface, the user "should"
         modify components of ij_x by using functions NALU_HYPRE_IJVectorSetValues or
         NALU_HYPRE_IJVectorAddToValues */

      NALU_HYPRE_ParVectorSetRandomValues(b, 22775);
      NALU_HYPRE_ParVectorInnerProd(b, b, &norm);
      norm = 1. / sqrt(norm);
      ierr = NALU_HYPRE_ParVectorScale(norm, b);

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 4 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector set for solution with unit components\n");
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      /* Temporary use of solution vector */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 1.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      NALU_HYPRE_ParCSRMatrixMatvec(1., parcsr_A, x, 0., b);

      /* Initial guess */
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
   }
   else if ( build_rhs_type == 5 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector is 0\n");
         nalu_hypre_printf("  Initial guess has unit components\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 1.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }

   if ( build_src_type == 0 )
   {
#if 0
      /* RHS */
      BuildRhsParFromFile(argc, argv, build_src_arg_index, &b);
#endif

      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector read from file %s\n", argv[build_src_arg_index]);
         nalu_hypre_printf("  Initial unknown vector in evolution is 0\n");
      }

      ierr = NALU_HYPRE_IJVectorRead( argv[build_src_arg_index], nalu_hypre_MPI_COMM_WORLD,
                                 NALU_HYPRE_PARCSR, &ij_b );

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial unknown vector */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 1 )
   {
      nalu_hypre_printf("build_src_type == 1 not currently implemented\n");
      return (-1);

#if 0
      BuildRhsParFromOneFile2(argc, argv, build_src_arg_index, part_b, &b);
#endif
   }
   else if ( build_src_type == 2 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector has unit components\n");
         nalu_hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 1.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 3 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector has random components in range 0 - 1\n");
         nalu_hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = nalu_hypre_Rand();
      }

      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 4 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector is 0 \n");
         nalu_hypre_printf("  Initial unknown vector has random components in range 0 - 1\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = nalu_hypre_Rand() / dt;
      }
      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = nalu_hypre_Rand();
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("IJ Vector Setup", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   /* NALU_HYPRE_IJMatrixPrint(ij_A, "driver.out.A");
   NALU_HYPRE_IJVectorPrint(ij_x, "driver.out.x0"); */

   if (num_functions > 1)
   {
      dof_func = NULL;
      if (build_funcs_type == 1)
      {
         BuildFuncsFromOneFile(argc, argv, build_funcs_arg_index, parcsr_A, &dof_func);
      }
      else if (build_funcs_type == 2)
      {
         BuildFuncsFromFiles(argc, argv, build_funcs_arg_index, parcsr_A, &dof_func);
      }
      else
      {
         local_num_vars = local_num_rows;
         dof_func = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_vars, NALU_HYPRE_MEMORY_HOST);
         if (myid == 0)
         {
            nalu_hypre_printf (" Number of unknown functions = %d \n", num_functions);
         }
         rest = first_local_row - ((first_local_row / num_functions) * num_functions);
         indx = num_functions - rest;
         if (rest == 0) { indx = 0; }
         k = num_functions - 1;
         for (j = indx - 1; j > -1; j--)
         {
            dof_func[j] = k--;
         }
         tms = local_num_vars / num_functions;
         if (tms * num_functions + indx > local_num_vars) { tms--; }
         for (j = 0; j < tms; j++)
         {
            for (k = 0; k < num_functions; k++)
            {
               dof_func[indx++] = k;
            }
         }
         k = 0;
         while (indx < local_num_vars)
         {
            dof_func[indx++] = k++;
         }
      }
   }

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
      NALU_HYPRE_IJMatrixPrint(ij_A, "IJ.out.A");
      NALU_HYPRE_IJVectorPrint(ij_b, "IJ.out.b");
      NALU_HYPRE_IJVectorPrint(ij_x, "IJ.out.x0");
   }

   /*-----------------------------------------------------------
    * Solve the system using the hybrid solver
    *-----------------------------------------------------------*/

   if (matrix_id == NALU_HYPRE_PARCSR  &&  solver_id == NALU_HYPRE_HYBRID)
   {
      dscg_max_its = 1000;
      pcg_max_its = 200;
      if (myid == 0) { nalu_hypre_printf("Solver:  AMG_Hybrid\n"); }
      time_index = nalu_hypre_InitializeTiming("AMG_hybrid Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRHybridCreate(&amg_solver);
      NALU_HYPRE_ParCSRHybridSetTol(amg_solver, tol);
      NALU_HYPRE_ParCSRHybridSetConvergenceTol(amg_solver, cf_tol);
      NALU_HYPRE_ParCSRHybridSetSolverType(amg_solver, solver_type);
      NALU_HYPRE_ParCSRHybridSetLogging(amg_solver, ioutdat);
      NALU_HYPRE_ParCSRHybridSetPrintLevel(amg_solver, poutdat);
      NALU_HYPRE_ParCSRHybridSetDSCGMaxIter(amg_solver, dscg_max_its );
      NALU_HYPRE_ParCSRHybridSetPCGMaxIter(amg_solver, pcg_max_its );
      NALU_HYPRE_ParCSRHybridSetCoarsenType(amg_solver, (hybrid * coarsen_type));
      NALU_HYPRE_ParCSRHybridSetStrongThreshold(amg_solver, strong_threshold);
      NALU_HYPRE_ParCSRHybridSetTruncFactor(amg_solver, trunc_factor);
      NALU_HYPRE_ParCSRHybridSetNumGridSweeps(amg_solver, num_grid_sweeps);
      NALU_HYPRE_ParCSRHybridSetGridRelaxType(amg_solver, grid_relax_type);
      NALU_HYPRE_ParCSRHybridSetRelaxWeight(amg_solver, relax_weight);
      NALU_HYPRE_ParCSRHybridSetOmega(amg_solver, omega);
      NALU_HYPRE_ParCSRHybridSetGridRelaxPoints(amg_solver, grid_relax_points);
      NALU_HYPRE_ParCSRHybridSetMaxLevels(amg_solver, max_levels);
      NALU_HYPRE_ParCSRHybridSetMaxRowSum(amg_solver, max_row_sum);

      NALU_HYPRE_ParCSRHybridSetup(amg_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("ParCSR Hybrid Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRHybridSolve(amg_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_ParCSRHybridGetNumIterations(amg_solver, &num_iterations);
      NALU_HYPRE_ParCSRHybridGetPCGNumIterations(amg_solver, &pcg_num_its);
      NALU_HYPRE_ParCSRHybridGetDSCGNumIterations(amg_solver, &dscg_num_its);
      NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(amg_solver,
                                                     &final_res_norm);

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("PCG_Iterations = %d\n", pcg_num_its);
         nalu_hypre_printf("DSCG_Iterations = %d\n", dscg_num_its);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
      NALU_HYPRE_ParCSRHybridDestroy(amg_solver);
   }
   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == NALU_HYPRE_BOOMERAMG)
   {
      if (myid == 0) { nalu_hypre_printf("Solver:  AMG\n"); }
      time_index = nalu_hypre_InitializeTiming("BoomerAMG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BoomerAMGCreate(&amg_solver);
      NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      NALU_HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, (hybrid * coarsen_type));
      NALU_HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      NALU_HYPRE_BoomerAMGSetTol(amg_solver, tol);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      NALU_HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
      /* note: log is written to standard output, not to file */
      NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
      NALU_HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log");
      NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      NALU_HYPRE_BoomerAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      NALU_HYPRE_BoomerAMGSetGridRelaxType(amg_solver, grid_relax_type);
      NALU_HYPRE_BoomerAMGSetRelaxWeight(amg_solver, relax_weight);
      NALU_HYPRE_BoomerAMGSetOmega(amg_solver, omega);
      NALU_HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
      NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
      NALU_HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, smooth_num_levels);
      NALU_HYPRE_BoomerAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      NALU_HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      NALU_HYPRE_BoomerAMGSetMaxRowSum(amg_solver, max_row_sum);
      NALU_HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      NALU_HYPRE_BoomerAMGSetVariant(amg_solver, variant);
      NALU_HYPRE_BoomerAMGSetOverlap(amg_solver, overlap);
      NALU_HYPRE_BoomerAMGSetDomainType(amg_solver, domain_type);
      NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_solver, schwarz_rlx_weight);
      NALU_HYPRE_BoomerAMGSetNumFunctions(amg_solver, num_functions);
      if (num_functions > 1)
      {
         NALU_HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);
      }

      NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("BoomerAMG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);
      NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
#endif

      NALU_HYPRE_BoomerAMGDestroy(amg_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using GSMG
    *-----------------------------------------------------------*/

   if (solver_id == NALU_HYPRE_GSMG)
   {
      /* reset some smoother parameters */

      /* fine grid */
      num_grid_sweeps[0] = num_sweep;
      grid_relax_type[0] = relax_default;
      nalu_hypre_TFree(grid_relax_points[0], NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sweep; i++)
      {
         grid_relax_points[0][i] = 0;
      }

      /* down cycle */
      num_grid_sweeps[1] = num_sweep;
      grid_relax_type[1] = relax_default;
      nalu_hypre_TFree(grid_relax_points[1], NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sweep; i++)
      {
         grid_relax_points[1][i] = 0;
      }

      /* up cycle */
      num_grid_sweeps[2] = num_sweep;
      grid_relax_type[2] = relax_default;
      nalu_hypre_TFree(grid_relax_points[2], NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sweep; i++)
      {
         grid_relax_points[2][i] = 0;
      }

      /* coarsest grid */
      num_grid_sweeps[3] = 1;
      grid_relax_type[3] = 9;
      nalu_hypre_TFree(grid_relax_points[3], NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[3] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[3][0] = 0;

      if (myid == 0) { nalu_hypre_printf("Solver:  GSMG\n"); }
      time_index = nalu_hypre_InitializeTiming("BoomerAMG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BoomerAMGCreate(&amg_solver);
      NALU_HYPRE_BoomerAMGSetGSMG(amg_solver, 4); /* specify GSMG */
      NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      NALU_HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, (hybrid * coarsen_type));
      NALU_HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      NALU_HYPRE_BoomerAMGSetTol(amg_solver, tol);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      NALU_HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
      /* note: log is written to standard output, not to file */
      NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
      NALU_HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log");
      NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      NALU_HYPRE_BoomerAMGSetNumGridSweeps(amg_solver, num_grid_sweeps);
      NALU_HYPRE_BoomerAMGSetGridRelaxType(amg_solver, grid_relax_type);
      NALU_HYPRE_BoomerAMGSetRelaxWeight(amg_solver, relax_weight);
      NALU_HYPRE_BoomerAMGSetOmega(amg_solver, omega);
      NALU_HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
      NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
      NALU_HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, smooth_num_levels);
      NALU_HYPRE_BoomerAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
      NALU_HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      NALU_HYPRE_BoomerAMGSetMaxRowSum(amg_solver, max_row_sum);
      NALU_HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      NALU_HYPRE_BoomerAMGSetVariant(amg_solver, variant);
      NALU_HYPRE_BoomerAMGSetOverlap(amg_solver, overlap);
      NALU_HYPRE_BoomerAMGSetDomainType(amg_solver, domain_type);
      NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_solver, schwarz_rlx_weight);
      NALU_HYPRE_BoomerAMGSetNumFunctions(amg_solver, num_functions);
      if (num_functions > 1)
      {
         NALU_HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);
      }

      NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("BoomerAMG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);
      NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
#endif

      NALU_HYPRE_BoomerAMGDestroy(amg_solver);
   }

   if (solver_id == NALU_HYPRE_PARASAILS)
   {
      NALU_HYPRE_IJMatrix ij_M;
      NALU_HYPRE_ParCSRMatrix  parcsr_mat;

      /* use ParaSails preconditioner */
      if (myid == 0) { nalu_hypre_printf("Test ParaSails Build IJMatrix\n"); }

      NALU_HYPRE_IJMatrixPrint(ij_A, "parasails.in");

      NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
      NALU_HYPRE_ParaSailsSetParams(pcg_precond, 0., 0);
      NALU_HYPRE_ParaSailsSetFilter(pcg_precond, 0.);
      NALU_HYPRE_ParaSailsSetLogging(pcg_precond, ioutdat);

      NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_mat = (NALU_HYPRE_ParCSRMatrix) object;

      NALU_HYPRE_ParaSailsSetup(pcg_precond, parcsr_mat, NULL, NULL);
      NALU_HYPRE_ParaSailsBuildIJMatrix(pcg_precond, &ij_M);
      NALU_HYPRE_IJMatrixPrint(ij_M, "parasails.out");

      if (myid == 0) { nalu_hypre_printf("Printed to parasails.out.\n"); }
      exit(0);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   /* begin lobpcg */
   if ( !lobpcgFlag && ( solver_id == NALU_HYPRE_PCG) )
      /*end lobpcg */
   {
      time_index = nalu_hypre_InitializeTiming("PCG Setup");
      nalu_hypre_BeginTiming(time_index);
      ioutdat = 2;

      NALU_HYPRE_ParCSRPCGCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_PCGSetMaxIter(pcg_solver, 1000);
      NALU_HYPRE_PCGSetTol(pcg_solver, tol);
      NALU_HYPRE_PCGSetTwoNorm(pcg_solver, 1);
      NALU_HYPRE_PCGSetRelChange(pcg_solver, 0);
      NALU_HYPRE_PCGSetPrintLevel(pcg_solver, ioutdat);

      if (precond_id == NALU_HYPRE_EUCLID)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver: Euclid-PCG\n"); }

         NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time
            parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally
            parse the command line.
         */
         NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         NALU_HYPRE_PCGSetPrecond(pcg_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                             pcg_precond);
      }
      else
      {
         nalu_hypre_set_precond(matrix_id, solver_id, precond_id, pcg_solver, pcg_precond);
      }

      NALU_HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
         nalu_hypre_printf("NALU_HYPRE_ParCSRPCGGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         nalu_hypre_printf("NALU_HYPRE_ParCSRPCGGetPrecond got good precond\n");
      }

      NALU_HYPRE_PCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("PCG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_PCGSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_PCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
      NALU_HYPRE_PCGSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
#endif

      NALU_HYPRE_ParCSRPCGDestroy(pcg_solver);

      nalu_hypre_destroy_precond(precond_id, pcg_precond);


      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

   }

   /* begin lobpcg */

   /*-----------------------------------------------------------
    * Solve the eigenvalue problem using LOBPCG
    *-----------------------------------------------------------*/

   if ( lobpcgFlag )
   {

      interpreter = nalu_hypre_CTAlloc(mv_InterfaceInterpreter, 1, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_ParCSRSetupInterpreter( interpreter );
      NALU_HYPRE_ParCSRSetupMatvec(&matvec_fn);

      if (myid != 0)
      {
         verbosity = 0;
      }

      if ( lobpcgGen )
      {
         BuildParIsoLaplacian(argc, argv, &parcsr_B);

         ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_B,
                                                 &first_local_row, &last_local_row,
                                                 &first_local_col, &last_local_col );

         local_num_rows = last_local_row - first_local_row + 1;
         local_num_cols = last_local_col - first_local_col + 1;
         ierr += NALU_HYPRE_ParCSRMatrixGetDims( parcsr_B, &M, &N );

         ierr += NALU_HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                       first_local_col, last_local_col,
                                       &ij_B );

         ierr += NALU_HYPRE_IJMatrixSetObjectType( ij_B, NALU_HYPRE_PARCSR );

         if (sparsity_known == 1)
         {
            diag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
            offdiag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
            local_row = 0;
            for (i = first_local_row; i <= last_local_row; i++)
            {
               ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_B, i, &size,
                                                 &col_inds, &values );
               for (j = 0; j < size; j++)
               {
                  if (col_inds[j] < first_local_row || col_inds[j] > last_local_row)
                  {
                     offdiag_sizes[local_row]++;
                  }
                  else
                  {
                     diag_sizes[local_row]++;
                  }
               }
               local_row++;
               ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_B, i, &size,
                                                     &col_inds, &values );
            }
            ierr += NALU_HYPRE_IJMatrixSetDiagOffdSizes( ij_B,
                                                    (const NALU_HYPRE_Int *) diag_sizes,
                                                    (const NALU_HYPRE_Int *) offdiag_sizes );
            nalu_hypre_TFree(diag_sizes, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(offdiag_sizes, NALU_HYPRE_MEMORY_HOST);

            ierr = NALU_HYPRE_IJMatrixInitialize( ij_B );

            for (i = first_local_row; i <= last_local_row; i++)
            {
               ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_B, i, &size,
                                                 &col_inds, &values );

               ierr += NALU_HYPRE_IJMatrixSetValues( ij_B, 1, &size, &i,
                                                (const NALU_HYPRE_Int *) col_inds,
                                                (const NALU_HYPRE_Real *) values );

               ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_B, i, &size,
                                                     &col_inds, &values );
            }
         }
         else
         {
            row_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);

            size = 5; /* this is in general too low, and supposed to test
                       the capability of the reallocation of the interface */

            if (sparsity_known == 0) /* tries a more accurate estimate of the
                                      storage */
            {
               if (build_matrix_type == 2) { size = 7; }
               if (build_matrix_type == 3) { size = 9; }
               if (build_matrix_type == 4) { size = 27; }
            }

            for (i = 0; i < local_num_rows; i++)
            {
               row_sizes[i] = size;
            }

            ierr = NALU_HYPRE_IJMatrixSetRowSizes ( ij_B, (const NALU_HYPRE_Int *) row_sizes );

            nalu_hypre_TFree(row_sizes, NALU_HYPRE_MEMORY_HOST);

            ierr = NALU_HYPRE_IJMatrixInitialize( ij_B );

            /* Loop through all locally stored rows and insert them into ij_matrix */
            for (i = first_local_row; i <= last_local_row; i++)
            {
               ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_B, i, &size,
                                                 &col_inds, &values );

               ierr += NALU_HYPRE_IJMatrixSetValues( ij_B, 1, &size, &i,
                                                (const NALU_HYPRE_Int *) col_inds,
                                                (const NALU_HYPRE_Real *) values );

               ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_B, i, &size,
                                                     &col_inds, &values );
            }
         }

         ierr += NALU_HYPRE_IJMatrixAssemble( ij_B );

         ierr += NALU_HYPRE_ParCSRMatrixDestroy(parcsr_B);

         ierr += NALU_HYPRE_IJMatrixGetObject( ij_B, &object);
         parcsr_B = (NALU_HYPRE_ParCSRMatrix) object;

      } /* if ( lobpcgGen ) */

      if ( pcgIterations > 0 )   /* do inner pcg iterations */
      {

         time_index = nalu_hypre_InitializeTiming("PCG Setup");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_ParCSRPCGCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
         NALU_HYPRE_PCGSetMaxIter(pcg_solver, pcgIterations);
         NALU_HYPRE_PCGSetTol(pcg_solver, pcgTol);
         NALU_HYPRE_PCGSetTwoNorm(pcg_solver, 1);
         NALU_HYPRE_PCGSetRelChange(pcg_solver, 0);
         NALU_HYPRE_PCGSetPrintLevel(pcg_solver, 0);

         NALU_HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond);

         if (solver_id == 1)
         {
            /* use BoomerAMG as preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: AMG-PCG\n"); }
            NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
            NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
            NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
            NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
            NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
            NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
            NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
            NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
            NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
            NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
            NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
            NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
            NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
            NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
            NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
            NALU_HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
            NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
            NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
            NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
            NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
            NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
            NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
            NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
            if (num_functions > 1)
            {
               NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
            }
            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                pcg_precond);
         }
         else if (solver_id == 2)
         {

            /* use diagonal scaling as preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: DS-PCG\n"); }
            pcg_precond = NULL;

            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                pcg_precond);
         }
         else if (solver_id == 8)
         {
            /* use ParaSails preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: ParaSails-PCG\n"); }

            NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
            NALU_HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
            NALU_HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
            NALU_HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);

            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSetup,
                                pcg_precond);
         }
         else if (solver_id == 12)
         {
            /* use Schwarz preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: Schwarz-PCG\n"); }

            NALU_HYPRE_SchwarzCreate(&pcg_precond);
            NALU_HYPRE_SchwarzSetVariant(pcg_precond, variant);
            NALU_HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);

            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSetup,
                                pcg_precond);
         }
         else if (solver_id == 14)
         {
            /* use GSMG as preconditioner */

            /* reset some smoother parameters */

            /* fine grid */
            num_grid_sweeps[0] = num_sweep;
            grid_relax_type[0] = relax_default;
            nalu_hypre_TFree(grid_relax_points[0], NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sweep; i++)
            {
               grid_relax_points[0][i] = 0;
            }

            /* down cycle */
            num_grid_sweeps[1] = num_sweep;
            grid_relax_type[1] = relax_default;
            nalu_hypre_TFree(grid_relax_points[1], NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sweep; i++)
            {
               grid_relax_points[1][i] = 0;
            }

            /* up cycle */
            num_grid_sweeps[2] = num_sweep;
            grid_relax_type[2] = relax_default;
            nalu_hypre_TFree(grid_relax_points[2], NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sweep; i++)
            {
               grid_relax_points[2][i] = 0;
            }

            /* coarsest grid */
            num_grid_sweeps[3] = 1;
            grid_relax_type[3] = 9;
            nalu_hypre_TFree(grid_relax_points[3], NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[3] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[3][0] = 0;

            if (myid == 0) { nalu_hypre_printf("Solver: GSMG-PCG\n"); }
            NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
            NALU_HYPRE_BoomerAMGSetGSMG(pcg_precond, 4);
            NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
            NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
            NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
            NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
            NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
            NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
            NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
            NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
            NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
            NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
            NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
            NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
            NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
            NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
            NALU_HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
            NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
            NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
            NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
            NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
            NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
            NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
            NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
            if (num_functions > 1)
            {
               NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
            }
            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                pcg_precond);
         }
         else if (solver_id == 43)
         {
            /* use Euclid preconditioning */
            if (myid == 0) { nalu_hypre_printf("Solver: Euclid-PCG\n"); }

            NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

            /* note: There are three three methods of setting run-time
               parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here
               we'll use what I think is simplest: let Euclid internally
               parse the command line.
               */
            NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);

            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                pcg_precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if ( verbosity )
            {
               nalu_hypre_printf("Solver ID not recognized - running inner PCG iterations without preconditioner\n\n");
            }
         }

         NALU_HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  pcg_precond)
         {
            nalu_hypre_printf("NALU_HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            nalu_hypre_printf("NALU_HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }

         /*      NALU_HYPRE_PCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                 (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x); */

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         NALU_HYPRE_LOBPCGCreate(interpreter, &matvec_fn, &lobpcg_solver);

         NALU_HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
         NALU_HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
         NALU_HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
         NALU_HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

         NALU_HYPRE_LOBPCGSetPrecond(lobpcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_PCGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_PCGSetup,
                                pcg_solver);

         NALU_HYPRE_LOBPCGSetupT(lobpcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                            (NALU_HYPRE_Vector)x);

         NALU_HYPRE_LOBPCGSetup(lobpcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                           (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

         if ( lobpcgGen )
            NALU_HYPRE_LOBPCGSetupB(lobpcg_solver, (NALU_HYPRE_Matrix)parcsr_B,
                               (NALU_HYPRE_Vector)x);

         if ( vFromFileFlag )
         {
            eigenvectors = mv_MultiVectorWrap( interpreter,
                                               nalu_hypre_ParCSRMultiVectorRead(nalu_hypre_MPI_COMM_WORLD,
                                                                           interpreter,
                                                                           "vectors" ), 1);
            nalu_hypre_assert( eigenvectors != NULL );
            blockSize = mv_MultiVectorWidth( eigenvectors );
         }
         else
         {
            eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                                 blockSize,
                                                                 x );
            if ( lobpcgSeed )
            {
               mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
            }
            else
            {
               mv_MultiVectorSetRandom( eigenvectors, (NALU_HYPRE_Int)time(0) );
            }
         }

         if ( constrained )
         {
            constraints = mv_MultiVectorWrap( interpreter,
                                              nalu_hypre_ParCSRMultiVectorRead(nalu_hypre_MPI_COMM_WORLD,
                                                                          interpreter,
                                                                          "vectors" ), 1);
            nalu_hypre_assert( constraints != NULL );
         }

         eigenvalues = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  blockSize, NALU_HYPRE_MEMORY_HOST);

         time_index = nalu_hypre_InitializeTiming("LOBPCG Solve");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_LOBPCGSolve(lobpcg_solver, constraints, eigenvectors, eigenvalues );

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();


         if ( checkOrtho )
         {

            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
            utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

            if ( lobpcgGen )
            {
               workspace = mv_MultiVectorCreateCopy( eigenvectors, 0 );
               nalu_hypre_LOBPCGMultiOperatorB( lobpcg_solver,
                                           mv_MultiVectorGetData(eigenvectors),
                                           mv_MultiVectorGetData(workspace) );
               lobpcg_MultiVectorByMultiVector( eigenvectors, workspace, gramXX );
            }
            else
            {
               lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
            }

            utilities_FortranMatrixSetToIdentity( identity );
            utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
            nonOrthF = utilities_FortranMatrixFNorm( gramXX );
            if ( myid == 0 )
            {
               nalu_hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );

         }

         if ( printLevel )
         {

            nalu_hypre_ParCSRMultiVectorPrint( mv_MultiVectorGetData(eigenvectors), "vectors" );

            if ( myid == 0 )
            {
               if ( (filePtr = fopen("values.txt", "w")) )
               {
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ( (filePtr = fopen("residuals.txt", "w")) )
               {
                  residualNorms = NALU_HYPRE_LOBPCGResidualNorms( lobpcg_solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if ( printLevel > 1 )
               {

                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = NALU_HYPRE_LOBPCGIterations( lobpcg_solver );

                  eigenvaluesHistory = NALU_HYPRE_LOBPCGEigenvaluesHistory( lobpcg_solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

                  residualNormsHistory = NALU_HYPRE_LOBPCGResidualNormsHistory( lobpcg_solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                  utilities_FortranMatrixDestroy( printBuffer );
               }
            }
         }

         NALU_HYPRE_LOBPCGDestroy(lobpcg_solver);
         mv_MultiVectorDestroy( eigenvectors );
         if ( constrained )
         {
            mv_MultiVectorDestroy( constraints );
         }
         if ( lobpcgGen )
         {
            mv_MultiVectorDestroy( workspace );
         }
         nalu_hypre_TFree(eigenvalues, NALU_HYPRE_MEMORY_HOST);

         NALU_HYPRE_ParCSRPCGDestroy(pcg_solver);

         if (solver_id == 1)
         {
            NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
         }
         else if (solver_id == 8)
         {
            NALU_HYPRE_ParaSailsDestroy(pcg_precond);
         }
         else if (solver_id == 12)
         {
            NALU_HYPRE_SchwarzDestroy(pcg_precond);
         }
         else if (solver_id == 14)
         {
            NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
         }
         else if (solver_id == 43)
         {
            NALU_HYPRE_EuclidDestroy(pcg_precond);
         }

      }
      else   /* pcgIterations <= 0 --> use the preconditioner directly */
      {

         time_index = nalu_hypre_InitializeTiming("LOBPCG Setup");
         nalu_hypre_BeginTiming(time_index);
         if (myid != 0)
         {
            verbosity = 0;
         }

         NALU_HYPRE_LOBPCGCreate(interpreter, &matvec_fn, &pcg_solver);
         NALU_HYPRE_LOBPCGSetMaxIter(pcg_solver, maxIterations);
         NALU_HYPRE_LOBPCGSetTol(pcg_solver, tol);
         NALU_HYPRE_LOBPCGSetPrintLevel(pcg_solver, verbosity);

         NALU_HYPRE_LOBPCGGetPrecond(pcg_solver, &pcg_precond);

         if (solver_id == 1)
         {
            /* use BoomerAMG as preconditioner */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: AMG-PCG\n");
            }

            NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
            NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
            NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
            NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
            NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
            NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
            NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
            NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
            NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
            NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
            NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
            NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
            NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
            NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
            NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
            NALU_HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
            NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
            NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
            NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
            NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
            NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
            NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
            NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
            if (num_functions > 1)
            {
               NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
            }

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                   pcg_precond);
         }
         else if (solver_id == 2)
         {

            /* use diagonal scaling as preconditioner */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: DS-PCG\n");
            }

            pcg_precond = NULL;

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
         }
         else if (solver_id == 8)
         {
            /* use ParaSails preconditioner */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: ParaSails-PCG\n");
            }

            NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
            NALU_HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
            NALU_HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
            NALU_HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSetup,
                                   pcg_precond);
         }
         else if (solver_id == 12)
         {
            /* use Schwarz preconditioner */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: Schwarz-PCG\n");
            }

            NALU_HYPRE_SchwarzCreate(&pcg_precond);
            NALU_HYPRE_SchwarzSetVariant(pcg_precond, variant);
            NALU_HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSetup,
                                   pcg_precond);
         }
         else if (solver_id == 14)
         {
            /* use GSMG as preconditioner */

            /* reset some smoother parameters */

            /* fine grid */
            num_grid_sweeps[0] = num_sweep;
            grid_relax_type[0] = relax_default;
            nalu_hypre_TFree(grid_relax_points[0], NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sweep; i++)
            {
               grid_relax_points[0][i] = 0;
            }

            /* down cycle */
            num_grid_sweeps[1] = num_sweep;
            grid_relax_type[1] = relax_default;
            nalu_hypre_TFree(grid_relax_points[1], NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sweep; i++)
            {
               grid_relax_points[1][i] = 0;
            }

            /* up cycle */
            num_grid_sweeps[2] = num_sweep;
            grid_relax_type[2] = relax_default;
            nalu_hypre_TFree(grid_relax_points[2], NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sweep; i++)
            {
               grid_relax_points[2][i] = 0;
            }

            /* coarsest grid */
            num_grid_sweeps[3] = 1;
            grid_relax_type[3] = 9;
            nalu_hypre_TFree(grid_relax_points[3], NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[3] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
            grid_relax_points[3][0] = 0;

            if (myid == 0) { nalu_hypre_printf("Solver: GSMG-PCG\n"); }
            NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
            NALU_HYPRE_BoomerAMGSetGSMG(pcg_precond, 4);
            NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
            NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
            NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
            NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
            NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
            NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
            NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
            NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
            NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
            NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
            NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
            NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
            NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
            NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
            NALU_HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
            NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
            NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
            NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
            NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
            NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
            NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
            NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
            if (num_functions > 1)
            {
               NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
            }

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                   pcg_precond);
         }
         else if (solver_id == 43)
         {
            /* use Euclid preconditioning */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: Euclid-PCG\n");
            }

            NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

            /* note: There are three three methods of setting run-time
               parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here
               we'll use what I think is simplest: let Euclid internally
               parse the command line.
               */
            NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                   pcg_precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if ( verbosity )
            {
               nalu_hypre_printf("Solver ID not recognized - running LOBPCG without preconditioner\n\n");
            }
         }

         NALU_HYPRE_LOBPCGGetPrecond(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  pcg_precond && pcgIterations)
         {
            nalu_hypre_printf("NALU_HYPRE_ParCSRLOBPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            nalu_hypre_printf("NALU_HYPRE_ParCSRLOBPCGGetPrecond got good precond\n");
         }

         NALU_HYPRE_LOBPCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                           (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

         if ( lobpcgGen )
            NALU_HYPRE_LOBPCGSetupB(pcg_solver, (NALU_HYPRE_Matrix)parcsr_B,
                               (NALU_HYPRE_Vector)x);

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         if ( vFromFileFlag )
         {
            eigenvectors = mv_MultiVectorWrap( interpreter,
                                               nalu_hypre_ParCSRMultiVectorRead(nalu_hypre_MPI_COMM_WORLD,
                                                                           interpreter,
                                                                           "vectors" ), 1);
            nalu_hypre_assert( eigenvectors != NULL );
            blockSize = mv_MultiVectorWidth( eigenvectors );
         }
         else
         {
            eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                                 blockSize,
                                                                 x );
            if ( lobpcgSeed )
            {
               mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
            }
            else
            {
               mv_MultiVectorSetRandom( eigenvectors, (NALU_HYPRE_Int)time(0) );
            }
         }

         if ( constrained )
         {
            constraints = mv_MultiVectorWrap( interpreter,
                                              nalu_hypre_ParCSRMultiVectorRead(nalu_hypre_MPI_COMM_WORLD,
                                                                          interpreter,
                                                                          "vectors" ), 1);
            nalu_hypre_assert( constraints != NULL );
         }

         eigenvalues = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  blockSize, NALU_HYPRE_MEMORY_HOST);

         time_index = nalu_hypre_InitializeTiming("LOBPCG Solve");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_LOBPCGSolve(pcg_solver, constraints, eigenvectors, eigenvalues);

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         if ( checkOrtho )
         {

            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
            utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

            if ( lobpcgGen )
            {
               workspace = mv_MultiVectorCreateCopy( eigenvectors, 0 );
               nalu_hypre_LOBPCGMultiOperatorB( pcg_solver,
                                           mv_MultiVectorGetData(eigenvectors),
                                           mv_MultiVectorGetData(workspace) );
               lobpcg_MultiVectorByMultiVector( eigenvectors, workspace, gramXX );
            }
            else
            {
               lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
            }

            utilities_FortranMatrixSetToIdentity( identity );
            utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
            nonOrthF = utilities_FortranMatrixFNorm( gramXX );
            if ( myid == 0 )
            {
               nalu_hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );

         }

         if ( printLevel )
         {

            nalu_hypre_ParCSRMultiVectorPrint( mv_MultiVectorGetData(eigenvectors), "vectors" );

            if ( myid == 0 )
            {
               if ( (filePtr = fopen("values.txt", "w")) )
               {
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ( (filePtr = fopen("residuals.txt", "w")) )
               {
                  residualNorms = NALU_HYPRE_LOBPCGResidualNorms( pcg_solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if ( printLevel > 1 )
               {

                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = NALU_HYPRE_LOBPCGIterations( pcg_solver );

                  eigenvaluesHistory = NALU_HYPRE_LOBPCGEigenvaluesHistory( pcg_solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

                  residualNormsHistory = NALU_HYPRE_LOBPCGResidualNormsHistory( pcg_solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                  utilities_FortranMatrixDestroy( printBuffer );
               }
            }
         }

#if SECOND_TIME
         /* run a second time to check for memory leaks */
         mv_MultiVectorSetRandom( eigenvectors, 775 );
         NALU_HYPRE_LOBPCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                           (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
         NALU_HYPRE_LOBPCGSolve(pcg_solver, constraints, eigenvectors, eigenvalues );
#endif

         NALU_HYPRE_LOBPCGDestroy(pcg_solver);

         if (solver_id == 1)
         {
            NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
         }
         else if (solver_id == 8)
         {
            NALU_HYPRE_ParaSailsDestroy(pcg_precond);
         }
         else if (solver_id == 12)
         {
            NALU_HYPRE_SchwarzDestroy(pcg_precond);
         }
         else if (solver_id == 14)
         {
            NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
         }
         else if (solver_id == 43)
         {
            NALU_HYPRE_EuclidDestroy(pcg_precond);
         }

         mv_MultiVectorDestroy( eigenvectors );
         if ( constrained )
         {
            mv_MultiVectorDestroy( constraints );
         }
         if ( lobpcgGen )
         {
            mv_MultiVectorDestroy( workspace );
         }
         nalu_hypre_TFree(eigenvalues, NALU_HYPRE_MEMORY_HOST);
      } /* if ( pcgIterations > 0 ) */

      nalu_hypre_TFree( interpreter, NALU_HYPRE_MEMORY_HOST);

      if ( lobpcgGen )
      {
         NALU_HYPRE_IJMatrixDestroy(ij_B);
      }

   } /* if ( lobpcgFlag ) */

   /* end lobpcg */

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7 ||
       solver_id == 15 || solver_id == 18 || solver_id == 44)
   {
      time_index = nalu_hypre_InitializeTiming("GMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      ioutdat = 2;
      NALU_HYPRE_ParCSRGMRESCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_GMRESSetKDim(pcg_solver, k_dim);
      NALU_HYPRE_GMRESSetMaxIter(pcg_solver, 1000);
      NALU_HYPRE_GMRESSetTol(pcg_solver, tol);
      NALU_HYPRE_GMRESSetLogging(pcg_solver, 1);
      NALU_HYPRE_GMRESSetPrintLevel(pcg_solver, ioutdat);

      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-GMRES\n"); }

         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         NALU_HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                               pcg_precond);
      }
      else if (solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-GMRES\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                               pcg_precond);
      }
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: PILUT-GMRES\n"); }

         ierr = NALU_HYPRE_ParCSRPilutCreate( nalu_hypre_MPI_COMM_WORLD, &pcg_precond );
         if (ierr)
         {
            nalu_hypre_printf("Error in ParPilutCreate\n");
         }

         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSetup,
                               pcg_precond);

         if (drop_tol >= 0 )
            NALU_HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            NALU_HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
      }
      else if (solver_id == 15)
      {
         /* use GSMG as preconditioner */

         /* reset some smoother parameters */

         /* fine grid */
         num_grid_sweeps[0] = num_sweep;
         grid_relax_type[0] = relax_default;
         nalu_hypre_TFree(grid_relax_points[0], NALU_HYPRE_MEMORY_HOST);
         grid_relax_points[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sweep; i++)
         {
            grid_relax_points[0][i] = 0;
         }

         /* down cycle */
         num_grid_sweeps[1] = num_sweep;
         grid_relax_type[1] = relax_default;
         nalu_hypre_TFree(grid_relax_points[1], NALU_HYPRE_MEMORY_HOST);
         grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sweep; i++)
         {
            grid_relax_points[1][i] = 0;
         }

         /* up cycle */
         num_grid_sweeps[2] = num_sweep;
         grid_relax_type[2] = relax_default;
         nalu_hypre_TFree(grid_relax_points[2], NALU_HYPRE_MEMORY_HOST);
         grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sweep, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sweep; i++)
         {
            grid_relax_points[2][i] = 0;
         }

         /* coarsest grid */
         num_grid_sweeps[3] = 1;
         grid_relax_type[3] = 9;
         nalu_hypre_TFree(grid_relax_points[3], NALU_HYPRE_MEMORY_HOST);
         grid_relax_points[3] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
         grid_relax_points[3][0] = 0;

         if (myid == 0) { nalu_hypre_printf("Solver: GSMG-GMRES\n"); }
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetGSMG(pcg_precond, 4);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         NALU_HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                               pcg_precond);
      }
      else if (solver_id == 18)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: ParaSails-GMRES\n"); }

         NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
         NALU_HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
         NALU_HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
         NALU_HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);
         NALU_HYPRE_ParaSailsSetSym(pcg_precond, 0);

         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSetup,
                               pcg_precond);
      }
      else if (solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver: Euclid-GMRES\n"); }

         NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time
            parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally
            parse the command line.
         */
         NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         NALU_HYPRE_GMRESSetPrecond (pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                pcg_precond);
      }

      NALU_HYPRE_GMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         nalu_hypre_printf("NALU_HYPRE_GMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         nalu_hypre_printf("NALU_HYPRE_GMRESGetPrecond got good precond\n");
      }
      NALU_HYPRE_GMRESSetup
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("GMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_GMRESSolve
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_GMRESGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_GMRESSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                       (NALU_HYPRE_Vector)x);
      NALU_HYPRE_GMRESSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                       (NALU_HYPRE_Vector)x);
#endif

      NALU_HYPRE_ParCSRGMRESDestroy(pcg_solver);

      if (solver_id == 3 || solver_id == 15)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (solver_id == 7)
      {
         NALU_HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
      else if (solver_id == 18)
      {
         NALU_HYPRE_ParaSailsDestroy(pcg_precond);
      }
      else if (solver_id == 44)
      {
         NALU_HYPRE_EuclidDestroy(pcg_precond);
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
    * Solve the system using BiCGSTAB
    *-----------------------------------------------------------*/

   if (solver_id == 9 || solver_id == 10 || solver_id == 11 || solver_id == 45)
   {
      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Setup");
      nalu_hypre_BeginTiming(time_index);

      ioutdat = 2;
      NALU_HYPRE_ParCSRBiCGSTABCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_BiCGSTABSetMaxIter(pcg_solver, 1000);
      NALU_HYPRE_BiCGSTABSetTol(pcg_solver, tol);
      NALU_HYPRE_BiCGSTABSetLogging(pcg_solver, ioutdat);
      NALU_HYPRE_BiCGSTABSetPrintLevel(pcg_solver, ioutdat);

      if (solver_id == 9)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-BiCGSTAB\n"); }
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         NALU_HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                  pcg_precond);
      }
      else if (solver_id == 10)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-BiCGSTAB\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                  pcg_precond);
      }
      else if (solver_id == 11)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: PILUT-BiCGSTAB\n"); }

         ierr = NALU_HYPRE_ParCSRPilutCreate( nalu_hypre_MPI_COMM_WORLD, &pcg_precond );
         if (ierr)
         {
            nalu_hypre_printf("Error in ParPilutCreate\n");
         }

         NALU_HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSetup,
                                  pcg_precond);

         if (drop_tol >= 0 )
            NALU_HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            NALU_HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
      }
      else if (solver_id == 45)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver: Euclid-BICGSTAB\n"); }

         NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time
            parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally
            parse the command line.
         */
         NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         NALU_HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                  pcg_precond);
      }

      NALU_HYPRE_BiCGSTABSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                          (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BiCGSTABSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                          (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_BiCGSTABGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_BiCGSTABSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                          (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
      NALU_HYPRE_BiCGSTABSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                          (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
#endif

      NALU_HYPRE_ParCSRBiCGSTABDestroy(pcg_solver);

      if (solver_id == 9)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (solver_id == 11)
      {
         NALU_HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
      else if (solver_id == 45)
      {
         NALU_HYPRE_EuclidDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("BiCGSTAB Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final BiCGSTAB Relative Residual Norm = %e\n", final_res_norm);
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

      ioutdat = 2;
      NALU_HYPRE_ParCSRCGNRCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_CGNRSetMaxIter(pcg_solver, 1000);
      NALU_HYPRE_CGNRSetTol(pcg_solver, tol);
      NALU_HYPRE_CGNRSetLogging(pcg_solver, ioutdat);

      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-CGNR\n"); }
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         NALU_HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         NALU_HYPRE_BoomerAMGSetOmega(pcg_precond, omega);
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_CGNRSetPrecond(pcg_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolveT,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                              pcg_precond);
      }
      else if (solver_id == 6)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-CGNR\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_CGNRSetPrecond(pcg_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                              pcg_precond);
      }

      NALU_HYPRE_CGNRGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         nalu_hypre_printf("NALU_HYPRE_ParCSRCGNRGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         nalu_hypre_printf("NALU_HYPRE_ParCSRCGNRGetPrecond got good precond\n");
      }
      NALU_HYPRE_CGNRSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                      (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("CGNR Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_CGNRSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                      (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_CGNRGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_CGNRGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_CGNRSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                      (NALU_HYPRE_Vector)x);
      NALU_HYPRE_CGNRSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                      (NALU_HYPRE_Vector)x);
#endif

      NALU_HYPRE_ParCSRCGNRDestroy(pcg_solver);

      if (solver_id == 5)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      if (myid == 0 /* begin lobpcg */ && !lobpcgFlag /* end lobpcg */)
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

   NALU_HYPRE_IJVectorGetObjectType(ij_b, &j);
   /* NALU_HYPRE_IJVectorPrint(ij_b, "driver.out.b");
   NALU_HYPRE_IJVectorPrint(ij_x, "driver.out.x"); */

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   NALU_HYPRE_IJMatrixDestroy(ij_A);
   NALU_HYPRE_IJVectorDestroy(ij_b);
   NALU_HYPRE_IJVectorDestroy(ij_x);

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

   NALU_HYPRE_ParCSRMatrix A;

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

   NALU_HYPRE_ParCSRMatrixRead(nalu_hypre_MPI_COMM_WORLD, filename, &A);

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

   cx = 1.;
   cy = 1.;
   cz = 1.;

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
      nalu_hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
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

   values[0] = 0.;
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

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian(nalu_hypre_MPI_COMM_WORLD,
                                              nx, ny, nz, P, Q, R, p, q, r, values);

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

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   ax = 1.;
   ay = 1.;
   az = 1.;

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
      nalu_hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   hinx = 1. / (nx + 1);
   hiny = 1. / (ny + 1);
   hinz = 1. / (nz + 1);

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

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
   }
   if (ny > 1)
   {
      values[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
   }
   if (nz > 1)
   {
      values[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
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
BuildParFromOneFile2(NALU_HYPRE_Int                  argc,
                     char                *argv[],
                     NALU_HYPRE_Int                  arg_index,
                     NALU_HYPRE_Int                  num_functions,
                     NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   NALU_HYPRE_ParCSRMatrix  A;
   NALU_HYPRE_CSRMatrix  A_CSR = NULL;

   NALU_HYPRE_Int                 myid, numprocs;
   NALU_HYPRE_Int                 i, rest, size, num_nodes, num_dofs;
   NALU_HYPRE_Int                *row_part;
   NALU_HYPRE_Int                *col_part;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &numprocs );

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

   row_part = NULL;
   col_part = NULL;
   if (myid == 0 && num_functions > 1)
   {
      NALU_HYPRE_CSRMatrixGetNumRows(A_CSR, &num_dofs);
      num_nodes = num_dofs / num_functions;
      if (num_dofs != num_functions * num_nodes)
      {
         row_part = NULL;
         col_part = NULL;
      }
      else
      {
         row_part = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  numprocs + 1, NALU_HYPRE_MEMORY_HOST);
         row_part[0] = 0;
         size = num_nodes / numprocs;
         rest = num_nodes - size * numprocs;
         for (i = 0; i < numprocs; i++)
         {
            row_part[i + 1] = row_part[i] + size * num_functions;
            if (i < rest) { row_part[i + 1] += num_functions; }
         }
         col_part = row_part;
      }
   }

   NALU_HYPRE_CSRMatrixToParCSRMatrix(nalu_hypre_MPI_COMM_WORLD, A_CSR, row_part, col_part, &A);

   *A_ptr = A;

   if (myid == 0) { NALU_HYPRE_CSRMatrixDestroy(A_CSR); }

   return (0);
}

/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildFuncsFromFiles(    NALU_HYPRE_Int                  argc,
                        char                *argv[],
                        NALU_HYPRE_Int                  arg_index,
                        NALU_HYPRE_ParCSRMatrix   parcsr_A,
                        NALU_HYPRE_Int                **dof_func_ptr     )
{
   /*----------------------------------------------------------------------
    * Build Function array from files on different processors
    *----------------------------------------------------------------------*/

   nalu_hypre_printf (" Feature is not implemented yet!\n");
   return (0);

}


NALU_HYPRE_Int
BuildFuncsFromOneFile(  NALU_HYPRE_Int                  argc,
                        char                *argv[],
                        NALU_HYPRE_Int                  arg_index,
                        NALU_HYPRE_ParCSRMatrix   parcsr_A,
                        NALU_HYPRE_Int                **dof_func_ptr     )
{
   char           *filename;

   NALU_HYPRE_Int             myid, num_procs;
   NALU_HYPRE_Int            *partitioning;
   NALU_HYPRE_Int            *dof_func;
   NALU_HYPRE_Int            *dof_func_local;
   NALU_HYPRE_Int             i, j;
   NALU_HYPRE_Int             local_size, global_size;
   nalu_hypre_MPI_Request    *requests;
   nalu_hypre_MPI_Status     *status, status0;
   MPI_Comm              comm;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   comm = nalu_hypre_MPI_COMM_WORLD;
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );

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
      FILE *fp;
      nalu_hypre_printf("  Funcs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * read in the data
       *-----------------------------------------------------------*/
      fp = fopen(filename, "r");

      nalu_hypre_fscanf(fp, "%d", &global_size);
      dof_func = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  global_size, NALU_HYPRE_MEMORY_HOST);

      for (j = 0; j < global_size; j++)
      {
         nalu_hypre_fscanf(fp, "%d", &dof_func[j]);
      }

      fclose(fp);

   }
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A, &partitioning);
   local_size = partitioning[myid + 1] - partitioning[myid];
   dof_func_local = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_size, NALU_HYPRE_MEMORY_HOST);

   if (myid == 0)
   {
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      j = 0;
      for (i = 1; i < num_procs; i++)
         nalu_hypre_MPI_Isend(&dof_func[partitioning[i]],
                         partitioning[i + 1] - partitioning[i],
                         NALU_HYPRE_MPI_INT, i, 0, comm, &requests[j++]);
      for (i = 0; i < local_size; i++)
      {
         dof_func_local[i] = dof_func[i];
      }
      nalu_hypre_MPI_Waitall(num_procs - 1, requests, status);
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      nalu_hypre_MPI_Recv(dof_func_local, local_size, NALU_HYPRE_MPI_INT, 0, 0, comm, &status0);
   }

   *dof_func_ptr = dof_func_local;

   if (myid == 0) { nalu_hypre_TFree(dof_func, NALU_HYPRE_MEMORY_HOST); }

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildRhsParFromOneFile2(NALU_HYPRE_Int                  argc,
                        char                *argv[],
                        NALU_HYPRE_Int                  arg_index,
                        NALU_HYPRE_Int                 *partitioning,
                        NALU_HYPRE_ParVector     *b_ptr     )
{
   char           *filename;

   NALU_HYPRE_ParVector b;
   NALU_HYPRE_Vector    b_CSR;

   NALU_HYPRE_Int             myid;

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
   NALU_HYPRE_VectorToParVector(nalu_hypre_MPI_COMM_WORLD, b_CSR, partitioning, &b);

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
      nalu_hypre_printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
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

   values[1] = -1.;

   values[0] = 0.;
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
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
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
   values[1] = -1.;

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian27pt(nalu_hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/* begin lobpcg */

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParIsoLaplacian( NALU_HYPRE_Int argc, char** argv, NALU_HYPRE_ParCSRMatrix *A_ptr )
{

   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Real          cx, cy, cz;

   NALU_HYPRE_Int                 P, Q, R;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q, r;
   NALU_HYPRE_Real         *values;

   NALU_HYPRE_Int arg_index;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   P  = 1;
   Q  = num_procs;
   R  = 1;

   nx = 10;
   ny = 10;
   nz = 10;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;


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
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Laplacian:\n");
      nalu_hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      nalu_hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
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

   values[0] = 0.;
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

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian(nalu_hypre_MPI_COMM_WORLD,
                                              nx, ny, nz, P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/* end lobpcg */
