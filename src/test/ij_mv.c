/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix-vector interface.
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "NALU_HYPRE_parcsr_mv.h"

#include "NALU_HYPRE_IJ_mv.h"
#include "NALU_HYPRE_parcsr_ls.h"
#include "NALU_HYPRE_krylov.h"

NALU_HYPRE_Int BuildParFromFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                            NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParLaplacian (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                             NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParDifConv (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                           NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildRhsParFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                  NALU_HYPRE_BigInt *partitioning, NALU_HYPRE_ParVector *b_ptr );
NALU_HYPRE_Int BuildParLaplacian9pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParLaplacian27pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix *A_ptr );

#define SECOND_TIME 0

nalu_hypre_int
main( nalu_hypre_int argc,
      char *argv[] )
{
   NALU_HYPRE_Int           arg_index;
   NALU_HYPRE_Int           print_usage;
   NALU_HYPRE_Int           sparsity_known = 0;
   NALU_HYPRE_Int           build_matrix_type;
   NALU_HYPRE_Int           build_matrix_arg_index;
   NALU_HYPRE_Int           build_rhs_type;
   NALU_HYPRE_Int           build_rhs_arg_index;
   NALU_HYPRE_Real          norm;
   void               *object;

   NALU_HYPRE_IJMatrix      ij_A;
   NALU_HYPRE_IJVector      ij_b = NULL;
   NALU_HYPRE_IJVector      ij_x = NULL;
   NALU_HYPRE_IJVector      ij_v;

   NALU_HYPRE_ParCSRMatrix  parcsr_A;
   NALU_HYPRE_ParVector     b;
   NALU_HYPRE_ParVector     x;

   NALU_HYPRE_Int           num_procs, myid;
   NALU_HYPRE_Int           local_row;
   NALU_HYPRE_BigInt       *indices;
   NALU_HYPRE_Int          *row_sizes;
   NALU_HYPRE_Int          *diag_sizes;
   NALU_HYPRE_Int          *offdiag_sizes;
   NALU_HYPRE_BigInt       *rows;
   NALU_HYPRE_Int           size;
   NALU_HYPRE_Int          *ncols;
   NALU_HYPRE_BigInt       *col_inds;

   MPI_Comm            comm = nalu_hypre_MPI_COMM_WORLD;
   NALU_HYPRE_Int           time_index;
   NALU_HYPRE_Int           ierr = 0;
   NALU_HYPRE_BigInt        M, N, big_i;
   NALU_HYPRE_Int           i, j;
   NALU_HYPRE_Int           local_num_rows, local_num_cols;
   NALU_HYPRE_BigInt        first_local_row, last_local_row;
   NALU_HYPRE_BigInt        first_local_col, last_local_col;
   NALU_HYPRE_Real         *values;

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

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromijfile") == 0 )
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
      nalu_hypre_printf("  -fromijfile <filename>     : ");
      nalu_hypre_printf("matrix read in IJ format from distributed files\n");
      nalu_hypre_printf("  -fromparcsrfile <filename> : ");
      nalu_hypre_printf("matrix read in ParCSR format from distributed files\n");
      nalu_hypre_printf("  -fromonecsrfile <filename> : ");
      nalu_hypre_printf("matrix read in CSR format from a file on one processor\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -laplacian [<options>] : build laplacian problem\n");
      nalu_hypre_printf("  -9pt [<opts>] : build 9pt 2D laplacian problem\n");
      nalu_hypre_printf("  -27pt [<opts>] : build 27pt 3D laplacian problem\n");
      nalu_hypre_printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      nalu_hypre_printf("    -n <nx> <ny> <nz>    : total problem size \n");
      nalu_hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      nalu_hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      nalu_hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -exact_size           : inserts immediately into ParCSR structure\n");
      nalu_hypre_printf("  -storage_low          : allocates not enough storage for aux struct\n");
      nalu_hypre_printf("  -concrete_parcsr      : use parcsr matrix type as concrete type\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -rhsfromfile           : rhs read in IJ form from distributed files\n");
      nalu_hypre_printf("  -rhsfromonefile        : rhs read from a file one one processor\n");
      nalu_hypre_printf("  -rhsrand               : rhs is random vector\n");
      nalu_hypre_printf("  -rhsisone              : rhs is vector with unit components (default)\n");
      nalu_hypre_printf("  -xisone                : solution of all ones\n");
      nalu_hypre_printf("  -rhszero               : rhs is zero vector\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("Running with these driver parameters:\n");
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

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
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, &parcsr_A);
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
      nalu_hypre_printf("You have asked for an unsupported test with\n");
      nalu_hypre_printf("build_matrix_type = %d.\n", build_matrix_type);
      return (-1);
   }

   time_index = nalu_hypre_InitializeTiming("Spatial Operator");
   nalu_hypre_BeginTiming(time_index);

   if (build_matrix_type < 2)
   {
      ierr += NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_A = (NALU_HYPRE_ParCSRMatrix) object;

      ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                              &first_local_row, &last_local_row,
                                              &first_local_col, &last_local_col );

      local_num_rows = last_local_row - first_local_row + 1;
      local_num_cols = last_local_col - first_local_col + 1;

      ierr = NALU_HYPRE_IJMatrixInitialize( ij_A );
   }
   else
   {
      /*--------------------------------------------------------------------
       * Copy the parcsr matrix into the IJMatrix through interface calls
       *--------------------------------------------------------------------*/

      ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                              &first_local_row, &last_local_row,
                                              &first_local_col, &last_local_col );

      local_num_rows = (NALU_HYPRE_Int)(last_local_row - first_local_row + 1);
      local_num_cols = (NALU_HYPRE_Int)(last_local_col - first_local_col + 1);

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
         for (big_i = first_local_row; big_i <= last_local_row; big_i++)
         {
            ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_A, big_i, &size,
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
            ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_A, big_i, &size,
                                                  &col_inds, &values );
         }
         ierr += NALU_HYPRE_IJMatrixSetDiagOffdSizes( ij_A,
                                                 (const NALU_HYPRE_Int *) diag_sizes,
                                                 (const NALU_HYPRE_Int *) offdiag_sizes );
         nalu_hypre_TFree(diag_sizes, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(offdiag_sizes, NALU_HYPRE_MEMORY_HOST);

         ierr = NALU_HYPRE_IJMatrixInitialize( ij_A );

         for (big_i = first_local_row; big_i <= last_local_row; big_i++)
         {
            ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_A, big_i, &size,
                                              &col_inds, &values );

            ierr += NALU_HYPRE_IJMatrixSetValues( ij_A, 1, &size, &big_i,
                                             (const NALU_HYPRE_BigInt *) col_inds,
                                             (const NALU_HYPRE_Real *) values );

            ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_A, big_i, &size,
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
         for (big_i = first_local_row; big_i <= last_local_row; big_i++)
         {
            ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_A, big_i, &size,
                                              &col_inds, &values );

            ierr += NALU_HYPRE_IJMatrixSetValues( ij_A, 1, &size, &big_i,
                                             (const NALU_HYPRE_BigInt *) col_inds,
                                             (const NALU_HYPRE_Real *) values );

            ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_A, big_i, &size,
                                                  &col_inds, &values );
         }
      }

      ierr += NALU_HYPRE_IJMatrixAssemble( ij_A );

   }

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Initial IJ Matrix Setup", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   if (ierr)
   {
      nalu_hypre_printf("Error in driver building IJMatrix from parcsr matrix. \n");
      return (-1);
   }

   time_index = nalu_hypre_InitializeTiming("Backward Euler Time Step");
   nalu_hypre_BeginTiming(time_index);

   /* This is to emphasize that one can IJMatrixAddToValues after an
      IJMatrixRead or an IJMatrixAssemble.  After an IJMatrixRead,
      assembly is unnecessary if the sparsity pattern of the matrix is
      not changed somehow.  If one has not used IJMatrixRead, one has
      the opportunity to IJMatrixAddTo before a IJMatrixAssemble. */

   ncols    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);
   rows     = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);
   col_inds = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);
   values   = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);

   for (big_i = first_local_row; big_i <= last_local_row; big_i++)
   {
      j = (NALU_HYPRE_Int)(big_i - first_local_row);
      rows[j] = big_i;
      ncols[j] = 1;
      col_inds[j] = big_i;
      values[j] = -27.8;
   }

   ierr += NALU_HYPRE_IJMatrixAddToValues( ij_A,
                                      local_num_rows,
                                      ncols, rows,
                                      (const NALU_HYPRE_BigInt *) col_inds,
                                      (const NALU_HYPRE_Real *) values );

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(col_inds, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(rows, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ncols, NALU_HYPRE_MEMORY_HOST);

   /* If sparsity pattern is not changed since last IJMatrixAssemble call,
      this should be a no-op */

   ierr += NALU_HYPRE_IJMatrixAssemble( ij_A );

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("IJ Matrix Diagonal Augmentation", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Fetch the resulting underlying matrix out
    *-----------------------------------------------------------*/

   if (build_matrix_type > 1)
   {
      ierr += NALU_HYPRE_ParCSRMatrixDestroy(parcsr_A);
   }

   ierr += NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
   parcsr_A = (NALU_HYPRE_ParCSRMatrix) object;

   NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_v);
   NALU_HYPRE_IJVectorSetObjectType(ij_v, NALU_HYPRE_PARCSR );
   NALU_HYPRE_IJVectorInitialize(ij_v);

   values  = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);

   /*-------------------------------------------------------------------
    * Check NALU_HYPRE_IJVectorSet(Get)Values calls
    *
    * All local components changed -- NULL indices
    *-------------------------------------------------------------------*/

   for (i = 0; i < local_num_cols; i++)
   {
      values[i] = 1.;
   }

   NALU_HYPRE_IJVectorSetValues(ij_v, local_num_cols, NULL, values);

   for (i = 0; i < local_num_cols; i++)
   {
      values[i] = (NALU_HYPRE_Real)i;
   }

   NALU_HYPRE_IJVectorAddToValues(ij_v, local_num_cols / 2, NULL, values);

   NALU_HYPRE_IJVectorGetValues(ij_v, local_num_cols, NULL, values);

   ierr = 0;
   for (i = 0; i < local_num_cols / 2; i++)
      if (values[i] != (NALU_HYPRE_Real)i + 1.) { ++ierr; }
   for (i = local_num_cols / 2; i < local_num_cols; i++)
      if (values[i] != 1.) { ++ierr; }
   if (ierr)
   {
      nalu_hypre_printf("One of NALU_HYPRE_IJVectorSet(AddTo,Get)Values\n");
      nalu_hypre_printf("calls with NULL indices bad\n");
      nalu_hypre_printf("IJVector Error 1 with ierr = %d\n", ierr);
      exit(1);
   }

   /*-------------------------------------------------------------------
    * All local components changed, assigned reverse-ordered values
    *   as specified by indices
    *-------------------------------------------------------------------*/

   indices = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  local_num_cols, NALU_HYPRE_MEMORY_HOST);

   for (big_i = first_local_col; big_i <= last_local_col; big_i++)
   {
      j = (NALU_HYPRE_Int)(big_i - first_local_col);
      values[j] = (NALU_HYPRE_Real)big_i;
      indices[j] = last_local_col - big_i;
   }

   NALU_HYPRE_IJVectorSetValues(ij_v, local_num_cols, indices, values);

   for (big_i = first_local_col; big_i <= last_local_col; big_i++)
   {
      j = (NALU_HYPRE_Int)(big_i - first_local_col);
      values[j] = (NALU_HYPRE_Real)big_i * big_i;
   }

   NALU_HYPRE_IJVectorAddToValues(ij_v, local_num_cols, indices, values);

   NALU_HYPRE_IJVectorGetValues(ij_v, local_num_cols, indices, values);

   nalu_hypre_TFree(indices, NALU_HYPRE_MEMORY_HOST);

   ierr = 0;
   for (big_i = first_local_col; big_i <= last_local_col; big_i++)
   {
      j = (NALU_HYPRE_Int)(big_i - first_local_col);
      if (values[j] != (NALU_HYPRE_Real)(big_i * big_i + big_i)) { ++ierr; }
   }

   if (ierr)
   {
      nalu_hypre_printf("One of NALU_HYPRE_IJVectorSet(Get)Values\n");
      nalu_hypre_printf("calls bad\n");
      nalu_hypre_printf("IJVector Error 2 with ierr = %d\n", ierr);
      exit(1);
   }

   NALU_HYPRE_IJVectorDestroy(ij_v);

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
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, part_b, &b);
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
      norm = 1. / nalu_hypre_sqrt(norm);
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

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("IJ Vector Setup", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   NALU_HYPRE_IJVectorGetObjectType(ij_b, &j);
   NALU_HYPRE_IJVectorPrint(ij_b, "driver.out.b");
   NALU_HYPRE_IJVectorPrint(ij_x, "driver.out.x");

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
   NALU_HYPRE_BigInt              nx, ny, nz;
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
         cx = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cy = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cz = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
      nalu_hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
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
   NALU_HYPRE_BigInt              nx, ny, nz;
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
         cx = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cy = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cz = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         ay = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         az = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
      nalu_hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
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
   NALU_HYPRE_CSRMatrix  A_CSR = NULL;

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
   NALU_HYPRE_CSRMatrixToParCSRMatrix(nalu_hypre_MPI_COMM_WORLD, A_CSR, NULL, NULL, &A);

   *A_ptr = A;

   if (myid == 0) { NALU_HYPRE_CSRMatrixDestroy(A_CSR); }

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildRhsParFromOneFile( NALU_HYPRE_Int            argc,
                        char                *argv[],
                        NALU_HYPRE_Int            arg_index,
                        NALU_HYPRE_BigInt        *partitioning,
                        NALU_HYPRE_ParVector     *b_ptr     )
{
   char           *filename;

   NALU_HYPRE_ParVector b;
   NALU_HYPRE_Vector    b_CSR = NULL;

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
BuildParLaplacian9pt( NALU_HYPRE_Int            argc,
                      char                *argv[],
                      NALU_HYPRE_Int            arg_index,
                      NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_BigInt              nx, ny;
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
      nalu_hypre_printf("    (nx, ny) = (%b, %b)\n", nx, ny);
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
      nalu_hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
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
   values[1] = -1.0;

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian27pt(nalu_hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}
