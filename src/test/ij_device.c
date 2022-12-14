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

#include "_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "NALU_HYPRE_parcsr_mv.h"

#include "NALU_HYPRE_IJ_mv.h"
#include "NALU_HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "NALU_HYPRE_krylov.h"

#include "cuda_profiler_api.h"

#ifdef __cplusplus
extern "C" {
#endif



NALU_HYPRE_Int BuildParFromFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                            NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParRhsFromFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_ParVector *b_ptr );

NALU_HYPRE_Int BuildParLaplacian (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                             NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParSysLaplacian (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParDifConv (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                           NALU_HYPRE_ParCSRMatrix *A_ptr);
NALU_HYPRE_Int BuildFuncsFromFiles (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_Int **dof_func_ptr );
NALU_HYPRE_Int BuildFuncsFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_Int **dof_func_ptr );
NALU_HYPRE_Int BuildParLaplacian9pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParLaplacian27pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParRotate7pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                             NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParVarDifConv (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                              NALU_HYPRE_ParCSRMatrix *A_ptr, NALU_HYPRE_ParVector *rhs_ptr );
NALU_HYPRE_ParCSRMatrix GenerateSysLaplacian (MPI_Comm comm, NALU_HYPRE_Int nx, NALU_HYPRE_Int ny, NALU_HYPRE_Int nz,
                                         NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                         NALU_HYPRE_Int num_fun, NALU_HYPRE_Real *mtrx, NALU_HYPRE_Real *value);
NALU_HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef (MPI_Comm comm, NALU_HYPRE_Int nx, NALU_HYPRE_Int ny,
                                              NALU_HYPRE_Int nz,
                                              NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                              NALU_HYPRE_Int num_fun, NALU_HYPRE_Real *mtrx, NALU_HYPRE_Real *value);
NALU_HYPRE_Int SetSysVcoefValues(NALU_HYPRE_Int num_fun, NALU_HYPRE_Int nx, NALU_HYPRE_Int ny, NALU_HYPRE_Int nz,
                            NALU_HYPRE_Real vcx, NALU_HYPRE_Real vcy, NALU_HYPRE_Real vcz, NALU_HYPRE_Int mtx_entry, NALU_HYPRE_Real *values);

NALU_HYPRE_Int BuildParCoordinates (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_Int *coorddim_ptr, float **coord_ptr );

void testPMIS(NALU_HYPRE_ParCSRMatrix parcsr_A);
void testPMIS2(NALU_HYPRE_ParCSRMatrix parcsr_A);
void testPMIS3(NALU_HYPRE_ParCSRMatrix parcsr_A);
void testTranspose(NALU_HYPRE_ParCSRMatrix parcsr_A);
void testAdd(NALU_HYPRE_ParCSRMatrix parcsr_A);
void testFFFC(NALU_HYPRE_ParCSRMatrix parcsr_A);

NALU_HYPRE_Int CompareParCSRDH(NALU_HYPRE_ParCSRMatrix hmat, NALU_HYPRE_ParCSRMatrix dmat, NALU_HYPRE_Real tol);

#ifdef __cplusplus
}
#endif

hypre_int
main( hypre_int argc,
      char *argv[] )
{
   NALU_HYPRE_Int           arg_index;
   NALU_HYPRE_Int           print_usage;
   NALU_HYPRE_Int           build_matrix_type;
   NALU_HYPRE_Int           build_matrix_arg_index;
   NALU_HYPRE_Int           ierr = 0;
   void               *object;

   NALU_HYPRE_IJMatrix     ij_A = NULL;
   NALU_HYPRE_ParCSRMatrix parcsr_A   = NULL;

   NALU_HYPRE_Int       time_index;
   MPI_Comm        comm = hypre_MPI_COMM_WORLD;
   NALU_HYPRE_Int       test = 1;
   //NALU_HYPRE_Int       i;
   //NALU_HYPRE_Real      *data;

   NALU_HYPRE_Int myid, num_procs;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before NALU_HYPRE_Init() and should not be changed after
    *-----------------------------------------------------------------*/
   hypre_bind_device(myid, num_procs, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   NALU_HYPRE_Init();

   hypre_SetNumThreads(5);
   hypre_printf("CPU #OMP THREADS %d\n", hypre_NumThreads());

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
   build_matrix_type = 2;
   build_matrix_arg_index = argc;

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
      else if ( strcmp(argv[arg_index], "-vardifconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 6;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rotate") == 0 )
      {
         arg_index++;
         build_matrix_type      = 7;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-test") == 0 )
      {
         arg_index++;
         test = atoi(argv[arg_index++]);
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
   if ( print_usage )
   {
      goto final;
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   NALU_HYPRE_SetSpGemmUseVendor(0);
   /* use cuRand for PMIS */
   NALU_HYPRE_SetUseGpuRand(1);

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/
   time_index = hypre_InitializeTiming("Generate Matrix");
   hypre_BeginTiming(time_index);
   if ( build_matrix_type == -1 )
   {
      ierr = NALU_HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                                 NALU_HYPRE_PARCSR, &ij_A );
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the system matrix!\n");
         exit(1);
      }
   }
   else if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
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
   else if ( build_matrix_type == 7 )
   {
      BuildParRotate7pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else
   {
      hypre_printf("You have asked for an unsupported problem with\n");
      hypre_printf("build_matrix_type = %d.\n", build_matrix_type);
      return (-1);
   }

   if (build_matrix_type < 0)
   {
      ierr += NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_A = (NALU_HYPRE_ParCSRMatrix) object;
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Generate Matrix", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   hypre_ParCSRMatrixMigrate(parcsr_A, hypre_HandleMemoryLocation(hypre_handle()));

   /*
    * TESTS
    */
   if (test == 1)
   {
      testPMIS(parcsr_A);
   }
   else if (test == 2)
   {
      testPMIS2(parcsr_A);
   }
   else if (test == 3)
   {
      testPMIS3(parcsr_A);
   }
   else if (test == 4)
   {
      testTranspose(parcsr_A);
   }
   else if (test == 5)
   {
      testAdd(parcsr_A);
   }

   //testFFFC(parcsr_A);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   if (build_matrix_type == -1)
   {
      NALU_HYPRE_IJMatrixDestroy(ij_A);
   }
   else
   {
      NALU_HYPRE_ParCSRMatrixDestroy(parcsr_A);
   }

final:

   /* Finalize Hypre */
   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

   /* when using cuda-memcheck --leak-check full, uncomment this */
#if defined(NALU_HYPRE_USING_GPU)
   hypre_ResetCudaDevice(hypre_handle());
#endif

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
BuildParFromFile( NALU_HYPRE_Int            argc,
                  char                *argv[],
                  NALU_HYPRE_Int            arg_index,
                  NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   NALU_HYPRE_ParCSRMatrix A;

   NALU_HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   NALU_HYPRE_ParCSRMatrixRead(hypre_MPI_COMM_WORLD, filename, &A);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build rhs from file. Expects two files on each processor.
 * filename.n contains the data and
 * and filename.INFO.n contains global row
 * numbers
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParRhsFromFile( NALU_HYPRE_Int            argc,
                     char                *argv[],
                     NALU_HYPRE_Int            arg_index,
                     NALU_HYPRE_ParVector      *b_ptr     )
{
   char               *filename;

   NALU_HYPRE_ParVector b;

   NALU_HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  RhsFromParFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   NALU_HYPRE_ParVectorRead(hypre_MPI_COMM_WORLD, filename, &b);

   *b_ptr = b;

   return (0);
}




/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParLaplacian( NALU_HYPRE_Int            argc,
                   char                *argv[],
                   NALU_HYPRE_Int            arg_index,
                   NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;
   NALU_HYPRE_Real          cx, cy, cz;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q, r;
   NALU_HYPRE_Int                 num_fun = 1;
   NALU_HYPRE_Real         *values;
   NALU_HYPRE_Real         *mtrx;

   NALU_HYPRE_Real          ep = .1;

   NALU_HYPRE_Int                 system_vcoef = 0;
   NALU_HYPRE_Int                 sys_opt = 0;
   NALU_HYPRE_Int                 vcoef_opt = 0;


   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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
      else if ( strcmp(argv[arg_index], "-sysL") == 0 )
      {
         arg_index++;
         num_fun = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sysL_opt") == 0 )
      {
         arg_index++;
         sys_opt = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sys_vcoef") == 0 )
      {
         /* have to use -sysL for this to */
         arg_index++;
         system_vcoef = 1;
      }
      else if ( strcmp(argv[arg_index], "-sys_vcoef_opt") == 0 )
      {
         arg_index++;
         vcoef_opt = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ep") == 0 )
      {
         arg_index++;
         ep = atof(argv[arg_index++]);
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
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian:   num_fun = %d\n", num_fun);
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
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

   values = hypre_CTAlloc(NALU_HYPRE_Real,  4, NALU_HYPRE_MEMORY_HOST);

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

   if (num_fun == 1)
      A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD,
                                                 nx, ny, nz, P, Q, R, p, q, r, values);
   else
   {
      mtrx = hypre_CTAlloc(NALU_HYPRE_Real,  num_fun * num_fun, NALU_HYPRE_MEMORY_HOST);

      if (num_fun == 2)
      {
         if (sys_opt == 1) /* identity  */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 20.0;
         }
         else if (sys_opt == 3) /* similar to barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 2.0;
            mtrx[2] = 2.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 4) /* can use with vcoef to get barry's ex*/
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.0;
            mtrx[2] = 1.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 5) /* barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.1;
            mtrx[2] = 1.1;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 6) /*  */
         {
            mtrx[0] = 1.1;
            mtrx[1] = 1.0;
            mtrx[2] = 1.0;
            mtrx[3] = 1.1;
         }

         else /* == 0 */
         {
            mtrx[0] = 2;
            mtrx[1] = 1;
            mtrx[2] = 1;
            mtrx[3] = 2;
         }
      }
      else if (num_fun == 3)
      {
         if (sys_opt == 1)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 0.0;
            mtrx[4] = 1.0;
            mtrx[5] = 0.0;
            mtrx[6] = 0.0;
            mtrx[7] = 0.0;
            mtrx[8] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 0.0;
            mtrx[4] = 20.0;
            mtrx[5] = 0.0;
            mtrx[6] = 0.0;
            mtrx[7] = 0.0;
            mtrx[8] = .01;
         }
         else if (sys_opt == 3)
         {
            mtrx[0] = 1.01;
            mtrx[1] = 1;
            mtrx[2] = 0.0;
            mtrx[3] = 1;
            mtrx[4] = 2;
            mtrx[5] = 1;
            mtrx[6] = 0.0;
            mtrx[7] = 1;
            mtrx[8] = 1.01;
         }
         else if (sys_opt == 4) /* barry ex4 */
         {
            mtrx[0] = 3;
            mtrx[1] = 1;
            mtrx[2] = 0.0;
            mtrx[3] = 1;
            mtrx[4] = 4;
            mtrx[5] = 2;
            mtrx[6] = 0.0;
            mtrx[7] = 2;
            mtrx[8] = .25;
         }
         else /* == 0 */
         {
            mtrx[0] = 2.0;
            mtrx[1] = 1.0;
            mtrx[2] = 0.0;
            mtrx[3] = 1.0;
            mtrx[4] = 2.0;
            mtrx[5] = 1.0;
            mtrx[6] = 0.0;
            mtrx[7] = 1.0;
            mtrx[8] = 2.0;
         }

      }
      else if (num_fun == 4)
      {
         mtrx[0] = 1.01;
         mtrx[1] = 1;
         mtrx[2] = 0.0;
         mtrx[3] = 0.0;
         mtrx[4] = 1;
         mtrx[5] = 2;
         mtrx[6] = 1;
         mtrx[7] = 0.0;
         mtrx[8] = 0.0;
         mtrx[9] = 1;
         mtrx[10] = 1.01;
         mtrx[11] = 0.0;
         mtrx[12] = 2;
         mtrx[13] = 1;
         mtrx[14] = 0.0;
         mtrx[15] = 1;
      }




      if (!system_vcoef)
      {
         A = (NALU_HYPRE_ParCSRMatrix) GenerateSysLaplacian(hypre_MPI_COMM_WORLD,
                                                       nx, ny, nz, P, Q,
                                                       R, p, q, r, num_fun, mtrx, values);
      }
      else
      {


         NALU_HYPRE_Real *mtrx_values;

         mtrx_values = hypre_CTAlloc(NALU_HYPRE_Real,  num_fun * num_fun * 4, NALU_HYPRE_MEMORY_HOST);

         if (num_fun == 2)
         {
            if (vcoef_opt == 1)
            {
               /* Barry's talk * - must also have sys_opt = 4, all fail */
               mtrx[0] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, .10, 1.0, 0, mtrx_values);

               mtrx[1]  = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, .1, 1.0, 1.0, 1, mtrx_values);

               mtrx[2] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, .01, 1.0, 1.0, 2, mtrx_values);

               mtrx[3] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, .02, 1.0, 3, mtrx_values);

            }
            else if (vcoef_opt == 2)
            {
               /* Barry's talk * - ex2 - if have sys-opt = 4*/
               mtrx[0] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, .010, 1.0, 0, mtrx_values);

               mtrx[1]  = 200.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 1, mtrx_values);

               mtrx[2] = 200.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 2, mtrx_values);

               mtrx[3] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, .02, 1.0, 3, mtrx_values);

            }
            else if (vcoef_opt == 3) /* use with default sys_opt  - ulrike ex 3*/
            {

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 4) /* use with default sys_opt  - ulrike ex 4*/
            {
               NALU_HYPRE_Real ep2 = ep;

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep * 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep2 * 1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 5) /* use with default sys_opt  - */
            {
               NALU_HYPRE_Real  alp, beta;
               alp = .001;
               beta = 10;

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta * 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta * 1.0, 1.0, 3, mtrx_values);
            }
            else  /* = 0 */
            {
               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 2.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, 1.0, 0.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 3.0, 1.0, 3, mtrx_values);
            }

         }
         else if (num_fun == 3)
         {
            mtrx[0] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, .01, 1, 0, mtrx_values);

            mtrx[1] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 1, mtrx_values);

            mtrx[2] = 0.0;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 2, mtrx_values);

            mtrx[3] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 3, mtrx_values);

            mtrx[4] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 2, .02, 1, 4, mtrx_values);

            mtrx[5] = 2;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 5, mtrx_values);

            mtrx[6] = 0.0;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 6, mtrx_values);

            mtrx[7] = 2;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 7, mtrx_values);

            mtrx[8] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1.5, .04, 1, 8, mtrx_values);

         }

         A = (NALU_HYPRE_ParCSRMatrix) GenerateSysLaplacianVCoef(hypre_MPI_COMM_WORLD,
                                                            nx, ny, nz, P, Q,
                                                            R, p, q, r, num_fun, mtrx, mtrx_values);





         hypre_TFree(mtrx_values, NALU_HYPRE_MEMORY_HOST);
      }

      hypre_TFree(mtrx, NALU_HYPRE_MEMORY_HOST);
   }

   hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * returns the sign of a real number
 *  1 : positive
 *  0 : zero
 * -1 : negative
 *----------------------------------------------------------------------*/
static inline NALU_HYPRE_Int sign_double(NALU_HYPRE_Real a)
{
   return ( (0.0 < a) - (0.0 > a) );
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
BuildParDifConv( NALU_HYPRE_Int            argc,
                 char                *argv[],
                 NALU_HYPRE_Int            arg_index,
                 NALU_HYPRE_ParCSRMatrix  *A_ptr)
{
   NALU_HYPRE_Int           nx, ny, nz;
   NALU_HYPRE_Int           P, Q, R;
   NALU_HYPRE_Real          cx, cy, cz;
   NALU_HYPRE_Real          ax, ay, az, atype;
   NALU_HYPRE_Real          hinx, hiny, hinz;
   NALU_HYPRE_Int           sign_prod;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int           num_procs, myid;
   NALU_HYPRE_Int           p, q, r;
   NALU_HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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

   atype = 0;

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
      else if ( strcmp(argv[arg_index], "-atype") == 0 )
      {
         arg_index++;
         atype = atoi(argv[arg_index++]);
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
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Convection-Diffusion: \n");
      hypre_printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
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
   /* values[7]:
    *    [0]: center
    *    [1]: X-
    *    [2]: Y-
    *    [3]: Z-
    *    [4]: X+
    *    [5]: Y+
    *    [6]: Z+
    */
   values = hypre_CTAlloc(NALU_HYPRE_Real,  7, NALU_HYPRE_MEMORY_HOST);

   values[0] = 0.;

   if (0 == atype) /* forward scheme for conv */
   {
      values[1] = -cx / (hinx * hinx);
      values[2] = -cy / (hiny * hiny);
      values[3] = -cz / (hinz * hinz);
      values[4] = -cx / (hinx * hinx) + ax / hinx;
      values[5] = -cy / (hiny * hiny) + ay / hiny;
      values[6] = -cz / (hinz * hinz) + az / hinz;

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
   }
   else if (1 == atype) /* backward scheme for conv */
   {
      values[1] = -cx / (hinx * hinx) - ax / hinx;
      values[2] = -cy / (hiny * hiny) - ay / hiny;
      values[3] = -cz / (hinz * hinz) - az / hinz;
      values[4] = -cx / (hinx * hinx);
      values[5] = -cy / (hiny * hiny);
      values[6] = -cz / (hinz * hinz);

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
      }
   }
   else if (3 == atype) /* upwind scheme */
   {
      sign_prod = sign_double(cx) * sign_double(ax);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[1] = -cx / (hinx * hinx) - ax / hinx;
         values[4] = -cx / (hinx * hinx);
         if (nx > 1)
         {
            values[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[1] = -cx / (hinx * hinx);
         values[4] = -cx / (hinx * hinx) + ax / hinx;
         if (nx > 1)
         {
            values[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
         }
      }

      sign_prod = sign_double(cy) * sign_double(ay);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[2] = -cy / (hiny * hiny) - ay / hiny;
         values[5] = -cy / (hiny * hiny);
         if (ny > 1)
         {
            values[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[2] = -cy / (hiny * hiny);
         values[5] = -cy / (hiny * hiny) + ay / hiny;
         if (ny > 1)
         {
            values[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
         }
      }

      sign_prod = sign_double(cz) * sign_double(az);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[3] = -cz / (hinz * hinz) - az / hinz;
         values[6] = -cz / (hinz * hinz);
         if (nz > 1)
         {
            values[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[3] = -cz / (hinz * hinz);
         values[6] = -cz / (hinz * hinz) + az / hinz;
         if (nz > 1)
         {
            values[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
         }
      }
   }
   else /* centered difference scheme */
   {
      values[1] = -cx / (hinx * hinx) - ax / (2.*hinx);
      values[2] = -cy / (hiny * hiny) - ay / (2.*hiny);
      values[3] = -cz / (hinz * hinz) - az / (2.*hinz);
      values[4] = -cx / (hinx * hinx) + ax / (2.*hinx);
      values[5] = -cy / (hiny * hiny) + ay / (2.*hiny);
      values[6] = -cz / (hinz * hinz) + az / (2.*hinz);

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx);
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny);
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz);
      }
   }

   A = (NALU_HYPRE_ParCSRMatrix) GenerateDifConv(hypre_MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

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

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian 9pt:\n");
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
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

   values = hypre_CTAlloc(NALU_HYPRE_Real,  2, NALU_HYPRE_MEMORY_HOST);

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

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian9pt(hypre_MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values);

   hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

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

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
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

   values = hypre_CTAlloc(NALU_HYPRE_Real,  2, NALU_HYPRE_MEMORY_HOST);

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

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build 7-point in 2D
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParRotate7pt( NALU_HYPRE_Int                  argc,
                   char                *argv[],
                   NALU_HYPRE_Int                  arg_index,
                   NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny;
   NALU_HYPRE_Int                 P, Q;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q;
   NALU_HYPRE_Real          eps, alpha;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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
      else if ( strcmp(argv[arg_index], "-alpha") == 0 )
      {
         arg_index++;
         alpha  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = atof(argv[arg_index++]);
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
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Rotate 7pt:\n");
      hypre_printf("    alpha = %f, eps = %f\n", alpha, eps);
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n", P,  Q);
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

   A = (NALU_HYPRE_ParCSRMatrix) GenerateRotate7pt(hypre_MPI_COMM_WORLD,
                                              nx, ny, P, Q, p, q, alpha, eps);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point difference operator using centered differences
 *
 *  eps*(a(x,y,z) ux)x + (b(x,y,z) uy)y + (c(x,y,z) uz)z
 *  d(x,y,z) ux + e(x,y,z) uy + f(x,y,z) uz + g(x,y,z) u
 *
 *  functions a,b,c,d,e,f,g need to be defined inside par_vardifconv.c
 *
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParVarDifConv( NALU_HYPRE_Int                  argc,
                    char                *argv[],
                    NALU_HYPRE_Int                  arg_index,
                    NALU_HYPRE_ParCSRMatrix  *A_ptr,
                    NALU_HYPRE_ParVector  *rhs_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;

   NALU_HYPRE_ParCSRMatrix  A;
   NALU_HYPRE_ParVector  rhs;

   NALU_HYPRE_Int           num_procs, myid;
   NALU_HYPRE_Int           p, q, r;
   NALU_HYPRE_Int           type;
   NALU_HYPRE_Real          eps;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;
   P  = 1;
   Q  = num_procs;
   R  = 1;
   eps = 1.0;

   /* type: 0   : default FD;
    *       1-3 : FD and examples 1-3 in Ruge-Stuben paper */
   type = 0;

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
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vardifconvRS") == 0 )
      {
         arg_index++;
         type = atoi(argv[arg_index++]);
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
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  ell PDE: eps = %f\n", eps);
      hypre_printf("    Dx(aDxu) + Dy(bDyu) + Dz(cDzu) + d Dxu + e Dyu + f Dzu  + g u= f\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
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

   if (0 == type)
   {
      A = (NALU_HYPRE_ParCSRMatrix) GenerateVarDifConv(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, eps, &rhs);
   }
   else
   {
      A = (NALU_HYPRE_ParCSRMatrix) GenerateRSVarDifConv(hypre_MPI_COMM_WORLD,
                                                    nx, ny, nz, P, Q, R, p, q, r, eps, &rhs,
                                                    type);
   }

   *A_ptr = A;
   *rhs_ptr = rhs;

   return (0);
}

/**************************************************************************/


NALU_HYPRE_Int SetSysVcoefValues(NALU_HYPRE_Int num_fun, NALU_HYPRE_Int nx, NALU_HYPRE_Int ny, NALU_HYPRE_Int nz,
                            NALU_HYPRE_Real vcx,
                            NALU_HYPRE_Real vcy, NALU_HYPRE_Real vcz, NALU_HYPRE_Int mtx_entry, NALU_HYPRE_Real *values)
{


   NALU_HYPRE_Int sz = num_fun * num_fun;

   values[1 * sz + mtx_entry] = -vcx;
   values[2 * sz + mtx_entry] = -vcy;
   values[3 * sz + mtx_entry] = -vcz;
   values[0 * sz + mtx_entry] = 0.0;

   if (nx > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcx;
   }
   if (ny > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcy;
   }
   if (nz > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcz;
   }

   return 0;

}

/*----------------------------------------------------------------------
 * Build coordinates for 1D/2D/3D
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParCoordinates( NALU_HYPRE_Int                  argc,
                     char                *argv[],
                     NALU_HYPRE_Int                  arg_index,
                     NALU_HYPRE_Int                 *coorddim_ptr,
                     float               **coord_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q, r;

   NALU_HYPRE_Int                 coorddim;
   float               *coordinates;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

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

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the coordinates
    *-----------------------------------------------------------*/

   coorddim = 3;
   if (nx < 2) { coorddim--; }
   if (ny < 2) { coorddim--; }
   if (nz < 2) { coorddim--; }

   if (coorddim > 0)
      coordinates = GenerateCoordinates (hypre_MPI_COMM_WORLD,
                                         nx, ny, nz, P, Q, R, p, q, r, coorddim);
   else
   {
      coordinates = NULL;
   }

   *coorddim_ptr = coorddim;
   *coord_ptr = coordinates;
   return (0);
}


void
testPMIS(NALU_HYPRE_ParCSRMatrix parcsr_A)
{
   NALU_HYPRE_Int         nC1 = 0, nC2 = 0, i;
   NALU_HYPRE_Int         time_index;
   hypre_IntArray   *h_CF_marker  = NULL;
   hypre_IntArray   *h_CF_marker2 = NULL;
   NALU_HYPRE_Int        *d_CF_marker  = NULL;
   NALU_HYPRE_Real        max_row_sum = 1.0;
   NALU_HYPRE_Int         num_functions = 1;
   NALU_HYPRE_Real        strong_threshold = 0.25;
   NALU_HYPRE_Int         debug_flag = 0;
   NALU_HYPRE_Int         local_num_rows;
   NALU_HYPRE_Int         first_local_row, last_local_row;
   NALU_HYPRE_Int         first_local_col, last_local_col;
   MPI_Comm          comm = hypre_MPI_COMM_WORLD;
   NALU_HYPRE_Int         num_procs, myid;

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                    &first_local_row, &last_local_row,
                                    &first_local_col, &last_local_col );

   local_num_rows = last_local_row - first_local_row + 1;
   //local_num_cols = last_local_col - first_local_col + 1;

   /* Soc on HOST */
   NALU_HYPRE_ParCSRMatrix parcsr_S   = NULL;

   hypre_BoomerAMGCreateSHost(parcsr_A, strong_threshold, max_row_sum,
                              num_functions, NULL, &parcsr_S);

   /* PMIS on HOST */
   time_index = hypre_InitializeTiming("Host PMIS");
   hypre_BeginTiming(time_index);

   hypre_BoomerAMGCoarsenPMISHost(parcsr_S, parcsr_A, 2, debug_flag, &h_CF_marker);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Host PMIS", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /* Soc on DEVICE */
   NALU_HYPRE_ParCSRMatrix parcsr_S_device  = NULL;
   hypre_BoomerAMGCreateSDevice(parcsr_A, strong_threshold, max_row_sum,
                                num_functions, NULL, &parcsr_S_device);
   /* PMIS on DEVICE */
   time_index = hypre_InitializeTiming("Device PMIS");
   hypre_BeginTiming(time_index);

   h_CF_marker2 = hypre_IntArrayCreate(local_num_rows);
   hypre_IntArrayInitialize(h_CF_marker2);
   hypre_BoomerAMGCoarsenPMISDevice(parcsr_S_device, parcsr_A, 2, debug_flag, &h_CF_marker2);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Device PMIS", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   //h_CF_marker2 = hypre_TAlloc(NALU_HYPRE_Int, local_num_rows, NALU_HYPRE_MEMORY_HOST);
   //hypre_TMemcpy(h_CF_marker2, d_CF_marker, NALU_HYPRE_Int, local_num_rows, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

   for (i = 0; i < local_num_rows; i++)
   {
      if (hypre_IntArrayData(h_CF_marker)[i] > 0)
      {
         nC1++;
      }

      hypre_assert(hypre_IntArrayData(h_CF_marker2)[i] == 1 ||
                   hypre_IntArrayData(h_CF_marker2)[i] == -1 || hypre_IntArrayData(h_CF_marker2)[i] == -3);

      //hypre_assert(h_CF_marker[i] == h_CF_marker2[i]);

      if (hypre_IntArrayData(h_CF_marker2)[i] > 0)
      {
         nC2++;
      }
   }

   NALU_HYPRE_Int allnC1, allnC2;
   hypre_MPI_Allreduce(&nC1, &allnC1, 1, NALU_HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   hypre_MPI_Allreduce(&nC2, &allnC2, 1, NALU_HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   if (myid == 0)
   {
      printf("nC1 %d nC2 %d\n", allnC1, allnC2);
   }

   hypre_ParCSRMatrixDestroy(parcsr_S);
   hypre_ParCSRMatrixDestroy(parcsr_S_device);
   hypre_IntArrayDestroy(h_CF_marker);
   hypre_IntArrayDestroy(h_CF_marker2);
   hypre_TFree(d_CF_marker,  NALU_HYPRE_MEMORY_DEVICE);
}

void
testPMIS3(NALU_HYPRE_ParCSRMatrix parcsr_A)
{
   NALU_HYPRE_Int         nC2 = 0, i;
   hypre_IntArray   *h_CF_marker2 = NULL;
   NALU_HYPRE_Real        max_row_sum = 1.0;
   NALU_HYPRE_Int         num_functions = 1;
   NALU_HYPRE_Real        strong_threshold = 0.25;
   NALU_HYPRE_Int         debug_flag = 0;
   NALU_HYPRE_Int         local_num_rows;
   NALU_HYPRE_Int         first_local_row, last_local_row;
   NALU_HYPRE_Int         first_local_col, last_local_col;
   MPI_Comm          comm = hypre_MPI_COMM_WORLD;
   NALU_HYPRE_Int         num_procs, myid;

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                    &first_local_row, &last_local_row,
                                    &first_local_col, &last_local_col );

   local_num_rows = last_local_row - first_local_row + 1;
   //local_num_cols = last_local_col - first_local_col + 1;

   /* Soc on DEVICE */
   NALU_HYPRE_ParCSRMatrix parcsr_S_device  = NULL;
   hypre_BoomerAMGCreateSDevice(parcsr_A, strong_threshold, max_row_sum,
                                num_functions, NULL, &parcsr_S_device);
   /* PMIS on DEVICE */
   h_CF_marker2 = hypre_IntArrayCreate(local_num_rows);
   hypre_IntArrayInitialize(h_CF_marker2);
   hypre_BoomerAMGCoarsenPMISDevice(parcsr_S_device, parcsr_A, 2, debug_flag, &h_CF_marker2);

   for (i = 0; i < local_num_rows; i++)
   {
      hypre_assert(hypre_IntArrayData(h_CF_marker2)[i] == 1 ||
                   hypre_IntArrayData(h_CF_marker2)[i] == -1 || hypre_IntArrayData(h_CF_marker2)[i] == -3);

      if (hypre_IntArrayData(h_CF_marker2)[i] > 0)
      {
         nC2++;
      }
   }

   NALU_HYPRE_Int allnC2;
   hypre_MPI_Allreduce(&nC2, &allnC2, 1, NALU_HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   if (myid == 0)
   {
      printf("nC2 %d\n", allnC2);
   }

   hypre_ParCSRMatrixDestroy(parcsr_S_device);
   hypre_IntArrayDestroy(h_CF_marker2);
}

void
testPMIS2(NALU_HYPRE_ParCSRMatrix parcsr_A)
{
   NALU_HYPRE_Int         nC2 = 0, i;
   hypre_IntArray   *h_CF_marker  = NULL;
   hypre_IntArray   *h_CF_marker2 = NULL;
   NALU_HYPRE_Real        max_row_sum = 1.0;
   NALU_HYPRE_Int         num_functions = 1;
   NALU_HYPRE_Real        strong_threshold = 0.25;
   NALU_HYPRE_Int         debug_flag = 0;
   NALU_HYPRE_Int         local_num_rows;
   NALU_HYPRE_Int         first_local_row, last_local_row;
   NALU_HYPRE_Int         first_local_col, last_local_col;
   MPI_Comm          comm = hypre_ParCSRMatrixComm(parcsr_A);
   NALU_HYPRE_Int         num_procs, myid;

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                    &first_local_row, &last_local_row,
                                    &first_local_col, &last_local_col );

   local_num_rows = last_local_row - first_local_row + 1;
   //local_num_cols = last_local_col - first_local_col + 1;

   /* Soc on DEVICE */
   NALU_HYPRE_ParCSRMatrix parcsr_S_device  = NULL;
   hypre_BoomerAMGCreateSDevice(parcsr_A, strong_threshold, max_row_sum,
                                num_functions, NULL, &parcsr_S_device);
   /* PMIS on DEVICE */
   h_CF_marker = hypre_IntArrayCreate(local_num_rows);
   hypre_IntArrayInitialize(h_CF_marker);
   hypre_BoomerAMGCoarsenPMISDevice(parcsr_S_device, parcsr_A, 2, debug_flag, &h_CF_marker);

   NALU_HYPRE_Int coarse_pnts_global[2];
   hypre_BoomerAMGCoarseParms(comm, local_num_rows, 1, NULL, h_CF_marker, NULL,
                              coarse_pnts_global);

   /* interp */
   hypre_ParCSRMatrix *P;
   hypre_BoomerAMGBuildDirInterpDevice(parcsr_A, hypre_IntArrayData(h_CF_marker), parcsr_S_device,
                                       coarse_pnts_global, 1, NULL,
                                       debug_flag, 0.0, 0, 3, &P);

   hypre_ParCSRMatrix *AH = hypre_ParCSRMatrixRAPKTDevice(P, parcsr_A, P, 1);
   hypre_ParCSRMatrixSetNumNonzeros(AH);

   //printf("AH %d, %d\n", hypre_ParCSRMatrixGlobalNumRows(AH), hypre_ParCSRMatrixNumNonzeros(AH));

   hypre_ParCSRMatrixPrintIJ(AH, 0, 0, "AH");

   NALU_HYPRE_Int local_num_rows2 = hypre_ParCSRMatrixNumRows(AH);

   hypre_ParCSRMatrix *S2;
   hypre_BoomerAMGCreateSDevice(AH, strong_threshold, max_row_sum,
                                num_functions, NULL, &S2);

   h_CF_marker2 = hypre_IntArrayCreate(local_num_rows2);
   hypre_IntArrayInitialize(h_CF_marker2);

   hypre_BoomerAMGCoarsenPMISDevice(S2, AH, 2, debug_flag, &h_CF_marker2);

   for (i = 0; i < local_num_rows2; i++)
   {
      hypre_assert(hypre_IntArrayData(h_CF_marker2)[i] == 1 ||
                   hypre_IntArrayData(h_CF_marker2)[i] == -1 || hypre_IntArrayData(h_CF_marker2)[i] == -3);

      if (hypre_IntArrayData(h_CF_marker2)[i] > 0)
      {
         nC2++;
      }
   }

   NALU_HYPRE_Int allnC2;
   hypre_MPI_Allreduce(&nC2, &allnC2, 1, NALU_HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   if (myid == 0)
   {
      printf("nC2 %d\n", allnC2);
   }

   hypre_ParCSRMatrixDestroy(parcsr_S_device);
   hypre_ParCSRMatrixDestroy(S2);
   hypre_IntArrayDestroy(h_CF_marker);
   hypre_IntArrayDestroy(h_CF_marker2);
}

void
testTranspose(NALU_HYPRE_ParCSRMatrix parcsr_A)
{
   NALU_HYPRE_Int    myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   NALU_HYPRE_Real tol = 0.0;
   NALU_HYPRE_Int  ierr = 0;

   NALU_HYPRE_ParCSRMatrix parcsr_AT;
   hypre_ParCSRMatrixTransposeDevice(parcsr_A, &parcsr_AT, 1);

   NALU_HYPRE_ParCSRMatrix parcsr_AT_h;
   NALU_HYPRE_ParCSRMatrix parcsr_A_h = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, NALU_HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixTransposeHost(parcsr_A_h, &parcsr_AT_h, 1);

   ierr += CompareParCSRDH(parcsr_AT_h, parcsr_AT, tol); hypre_assert(!ierr);

   hypre_ParCSRMatrixDestroy(parcsr_AT);
   hypre_ParCSRMatrixDestroy(parcsr_AT_h);

   //
   hypre_ParCSRMatrixTransposeDevice(parcsr_A, &parcsr_AT, 0);
   hypre_ParCSRMatrixTransposeHost(parcsr_A_h, &parcsr_AT_h, 0);

   hypre_ParCSRMatrixSetConstantValues(parcsr_AT, 1.0);
   hypre_ParCSRMatrixSetConstantValues(parcsr_AT_h, 1.0);

   ierr += CompareParCSRDH(parcsr_AT_h, parcsr_AT, tol); hypre_assert(!ierr);

   hypre_ParCSRMatrixDestroy(parcsr_AT);
   hypre_ParCSRMatrixDestroy(parcsr_AT_h);
   hypre_ParCSRMatrixDestroy(parcsr_A_h);

   //
   NALU_HYPRE_Int    first_local_row, last_local_row;
   NALU_HYPRE_Int    first_local_col, last_local_col;
   NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                    &first_local_row, &last_local_row,
                                    &first_local_col, &last_local_col );
   NALU_HYPRE_Int local_num_rows = last_local_row - first_local_row + 1;
   NALU_HYPRE_ParCSRMatrix parcsr_S_device  = NULL;
   hypre_BoomerAMGCreateSDevice(parcsr_A, 0.25, 1.0, 1, NULL, &parcsr_S_device);
   hypre_IntArray *h_CF_marker = hypre_IntArrayCreate(local_num_rows);
   hypre_IntArrayInitialize(h_CF_marker);
   hypre_BoomerAMGCoarsenPMISDevice(parcsr_S_device, parcsr_A, 2, 0, &h_CF_marker);
   hypre_ParCSRMatrix *P, *PT, *P_h, *PT_h, *P2;
   NALU_HYPRE_Int coarse_pnts_global[2];
   MPI_Comm comm = hypre_ParCSRMatrixComm(parcsr_A);
   hypre_BoomerAMGCoarseParms(comm, local_num_rows, 1, NULL, h_CF_marker, NULL, coarse_pnts_global);
   hypre_BoomerAMGBuildDirInterpDevice(parcsr_A, hypre_IntArrayData(h_CF_marker), parcsr_S_device,
                                       coarse_pnts_global, 1, NULL,
                                       0, 0.0, 0, 3, &P);
   P_h = hypre_ParCSRMatrixClone_v2(P, 1, NALU_HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrixTransposeDevice(P, &PT, 1);
   hypre_ParCSRMatrixTransposeHost(P_h, &PT_h, 1);
   hypre_ParCSRMatrixTransposeDevice(PT, &P2, 1);

   ierr += CompareParCSRDH(PT_h, PT, tol); hypre_assert(!ierr);
   ierr += CompareParCSRDH(P_h, P2, tol); hypre_assert(!ierr);

   if (myid == 0 && !ierr)
   {
      printf("[hypre_ParCSRMatrixTranspose] All Tests were OK ...\n");
   }

   hypre_ParCSRMatrixDestroy(P);
   hypre_ParCSRMatrixDestroy(PT);
   hypre_ParCSRMatrixDestroy(P_h);
   hypre_ParCSRMatrixDestroy(PT_h);
   hypre_ParCSRMatrixDestroy(P2);
   hypre_IntArrayDestroy(h_CF_marker);
}

void
testAdd(NALU_HYPRE_ParCSRMatrix parcsr_A)
{
   NALU_HYPRE_Int    myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   NALU_HYPRE_Real tol = 1e-14;
   NALU_HYPRE_Int  ierr = 0;
   NALU_HYPRE_Real alpha = 3.141592654, beta = 2.718281828 * 9.9;

   NALU_HYPRE_ParCSRMatrix parcsr_A2 = hypre_ParCSRMatMatDevice(parcsr_A, parcsr_A);
   NALU_HYPRE_ParCSRMatrix parcsr_C;
   hypre_ParCSRMatrixAddDevice(alpha, parcsr_A, beta, parcsr_A2, &parcsr_C);

   NALU_HYPRE_ParCSRMatrix parcsr_A_h = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_ParCSRMatrix parcsr_A2_h = hypre_ParCSRMatrixClone_v2(parcsr_A2, 1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_ParCSRMatrix parcsr_C_h;
   hypre_ParCSRMatrixAddHost(alpha, parcsr_A_h, beta, parcsr_A2_h, &parcsr_C_h);
   ierr += CompareParCSRDH(parcsr_C_h, parcsr_C, tol); hypre_assert(!ierr);

   hypre_ParCSRMatrixDestroy(parcsr_A2);
   hypre_ParCSRMatrixDestroy(parcsr_C);
   hypre_ParCSRMatrixDestroy(parcsr_A_h);
   hypre_ParCSRMatrixDestroy(parcsr_A2_h);
   hypre_ParCSRMatrixDestroy(parcsr_C_h);

   //
   NALU_HYPRE_Int    first_local_row, last_local_row;
   NALU_HYPRE_Int    first_local_col, last_local_col;
   NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                    &first_local_row, &last_local_row,
                                    &first_local_col, &last_local_col );
   NALU_HYPRE_Int local_num_rows = last_local_row - first_local_row + 1;
   NALU_HYPRE_ParCSRMatrix parcsr_S_device  = NULL;
   hypre_BoomerAMGCreateSDevice(parcsr_A, 0.25, 1.0, 1, NULL, &parcsr_S_device);
   hypre_IntArray *h_CF_marker = hypre_IntArrayCreate(local_num_rows);
   hypre_IntArrayInitialize(h_CF_marker);
   hypre_BoomerAMGCoarsenPMISDevice(parcsr_S_device, parcsr_A, 2, 0, &h_CF_marker);
   hypre_ParCSRMatrix *P, *AP, *P_h, *AP_h;
   NALU_HYPRE_Int coarse_pnts_global[2];
   MPI_Comm comm = hypre_ParCSRMatrixComm(parcsr_A);
   hypre_BoomerAMGCoarseParms(comm, local_num_rows, 1, NULL, h_CF_marker, NULL, coarse_pnts_global);
   hypre_BoomerAMGBuildDirInterpDevice(parcsr_A, hypre_IntArrayData(h_CF_marker), parcsr_S_device,
                                       coarse_pnts_global, 1, NULL,
                                       0, 0.0, 0, 3, &P);
   AP = hypre_ParCSRMatMatDevice(parcsr_A, P);
   P_h = hypre_ParCSRMatrixClone_v2(P, 1, NALU_HYPRE_MEMORY_HOST);
   AP_h = hypre_ParCSRMatrixClone_v2(AP, 1, NALU_HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixAddDevice(alpha, P, beta, AP, &parcsr_C);
   hypre_ParCSRMatrixAddHost(alpha, P_h, beta, AP_h, &parcsr_C_h);

   ierr += CompareParCSRDH(parcsr_C_h, parcsr_C, tol); hypre_assert(!ierr);

   hypre_ParCSRMatrixDestroy(P);
   hypre_ParCSRMatrixDestroy(AP);
   hypre_ParCSRMatrixDestroy(P_h);
   hypre_ParCSRMatrixDestroy(AP_h);
   hypre_ParCSRMatrixDestroy(parcsr_C);
   hypre_ParCSRMatrixDestroy(parcsr_C_h);
   hypre_ParCSRMatrixDestroy(parcsr_S_device);
   hypre_IntArrayDestroy(h_CF_marker);

   if (myid == 0 && !ierr)
   {
      printf("[hypre_ParCSRMatrixAdd] All Tests were OK ...\n");
   }
}

void
testFFFC(NALU_HYPRE_ParCSRMatrix parcsr_A)
{
   NALU_HYPRE_Real        max_row_sum = 1.0;
   NALU_HYPRE_Real        strong_threshold = 0.25;
   NALU_HYPRE_Int         debug_flag = 0;
   NALU_HYPRE_Int         num_functions = 1;
   hypre_IntArray   *h_CF_marker  = NULL;
   NALU_HYPRE_Int         local_num_rows;
   NALU_HYPRE_Int         first_local_row, last_local_row;
   NALU_HYPRE_Int         first_local_col, last_local_col;
   NALU_HYPRE_BigInt      coarse_pnts_global[2];
   NALU_HYPRE_Int         ierr = 0;
   NALU_HYPRE_Int         myid;

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   NALU_HYPRE_ParCSRMatrix parcsr_A_h;
   NALU_HYPRE_ParCSRMatrix AFF, AFC, W;
   NALU_HYPRE_ParCSRMatrix AFF_h, AFC_h, W_h;
   NALU_HYPRE_ParCSRMatrix parcsr_S_h, parcsr_S_device;

   NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                    &first_local_row, &last_local_row,
                                    &first_local_col, &last_local_col );

   local_num_rows = last_local_row - first_local_row + 1;

   /* Soc on DEVICE */
   hypre_BoomerAMGCreateSDevice(parcsr_A, strong_threshold, max_row_sum, num_functions, NULL,
                                &parcsr_S_device);
   h_CF_marker = hypre_IntArrayCreate(local_num_rows);
   hypre_IntArrayInitialize(h_CF_marker);
   hypre_BoomerAMGCoarsenPMISDevice(parcsr_S_device, parcsr_A, 0, debug_flag, &h_CF_marker);
   hypre_BoomerAMGCoarseParms(hypre_ParCSRMatrixComm(parcsr_A), local_num_rows, num_functions, NULL,
                              h_CF_marker, NULL, coarse_pnts_global);

   /* FFFC on Device */
   hypre_ParCSRMatrixGenerateFFFCDevice(parcsr_A, hypre_IntArrayData(h_CF_marker), coarse_pnts_global,
                                        parcsr_S_device, &AFC, &AFF);

   /* FFFC on Host */
   parcsr_A_h = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, NALU_HYPRE_MEMORY_HOST);
   parcsr_S_h = hypre_ParCSRMatrixClone_v2(parcsr_S_device, 0, NALU_HYPRE_MEMORY_HOST);
   hypre_MatvecCommPkgCreate(parcsr_A_h);
   hypre_ParCSRMatrixGenerateFFFCHost(parcsr_A_h, hypre_IntArrayData(h_CF_marker),
                                      coarse_pnts_global, parcsr_S_h, &AFC_h, &AFF_h);

   /* AFF * AFC */
   W_h = hypre_ParCSRMatMatHost(AFF_h, AFC_h);
   W   = hypre_ParCSRMatMatDevice(AFF, AFC);

   /* check */
   NALU_HYPRE_Real tol = 1e-15;
   ierr += CompareParCSRDH(AFF_h, AFF, tol); hypre_assert(!ierr);
   ierr += CompareParCSRDH(AFC_h, AFC, tol); hypre_assert(!ierr);
   ierr += CompareParCSRDH(W_h,     W, tol); hypre_assert(!ierr);

   if (myid == 0 && !ierr)
   {
      printf("All Tests were OK ...\n");
   }

   /* done */
   hypre_TFree(h_CF_marker, NALU_HYPRE_MEMORY_HOST);
   hypre_IntArrayDestroy(h_CF_marker);
   hypre_ParCSRMatrixDestroy(parcsr_A_h);
   hypre_ParCSRMatrixDestroy(AFF);
   hypre_ParCSRMatrixDestroy(AFC);
   hypre_ParCSRMatrixDestroy(W);
   hypre_ParCSRMatrixDestroy(AFF_h);
   hypre_ParCSRMatrixDestroy(AFC_h);
   hypre_ParCSRMatrixDestroy(W_h);
}

NALU_HYPRE_Int
CompareParCSRDH(NALU_HYPRE_ParCSRMatrix hmat, NALU_HYPRE_ParCSRMatrix dmat, NALU_HYPRE_Real tol)
{
   NALU_HYPRE_ParCSRMatrix hmat2, emat;
   NALU_HYPRE_Real enorm, fnorm, rnorm;
   NALU_HYPRE_Int i, ecode = 0, ecode_total = 0;

   hmat2 = hypre_ParCSRMatrixClone_v2(dmat, 1, NALU_HYPRE_MEMORY_HOST);

   if (hypre_ParCSRMatrixNumRows(hmat) != hypre_ParCSRMatrixNumRows(hmat2))
   {
      ecode ++;
   }
   if (hypre_ParCSRMatrixNumCols(hmat) != hypre_ParCSRMatrixNumCols(hmat2))
   {
      ecode ++;
   }
   if (hypre_CSRMatrixNumRows(hypre_ParCSRMatrixOffd(hmat)) != hypre_CSRMatrixNumRows(
          hypre_ParCSRMatrixOffd(hmat2)))
   {
      ecode ++;
   }
   if (hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(hmat)) != hypre_CSRMatrixNumCols(
          hypre_ParCSRMatrixOffd(hmat2)))
   {
      ecode ++;
   }
   for (i = 0; i < hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(hmat)); i++)
   {
      if (hypre_ParCSRMatrixColMapOffd(hmat)[i] != hypre_ParCSRMatrixColMapOffd(hmat2)[i])
      {
         ecode++;
         break;
      }
   }
   if (hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(hmat)) != hypre_CSRMatrixNumNonzeros(
          hypre_ParCSRMatrixDiag(hmat2)))
   {
      ecode ++;
   }
   if (hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(hmat)) != hypre_CSRMatrixNumNonzeros(
          hypre_ParCSRMatrixOffd(hmat2)))
   {
      ecode ++;
   }

   hypre_MPI_Allreduce(&ecode, &ecode_total, 1, NALU_HYPRE_MPI_INT, hypre_MPI_SUM, hypre_MPI_COMM_WORLD);

   hypre_ParCSRMatrixAdd(1.0, hmat, -1.0, hmat2, &emat);
   enorm = hypre_ParCSRMatrixFnorm(emat);

   fnorm = hypre_ParCSRMatrixFnorm(hmat);
   rnorm = fnorm > 0 ? enorm / fnorm : enorm;
   if ( rnorm > tol )
   {
      ecode_total ++;
   }

   printf("relative error %e = %e / %e\n", rnorm, enorm, fnorm);

   hypre_ParCSRMatrixDestroy(hmat2);
   hypre_ParCSRMatrixDestroy(emat);

   return ecode_total;
}
