/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ).
 *
 * It tests the assembly phase of an IJ matrix in both CPU and GPU.
 *--------------------------------------------------------------------------*/

#include "NALU_HYPRE.h"
#include "NALU_HYPRE_utilities.h"
#include "_nalu_hypre_IJ_mv.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "NALU_HYPRE_parcsr_ls.h"
//#include "_nalu_hypre_utilities.hpp"

NALU_HYPRE_Int buildMatrixEntries(MPI_Comm comm,
                             NALU_HYPRE_Int nx, NALU_HYPRE_Int ny, NALU_HYPRE_Int nz,
                             NALU_HYPRE_Int Px, NALU_HYPRE_Int Py, NALU_HYPRE_Int Pz,
                             NALU_HYPRE_Real cx, NALU_HYPRE_Real cy, NALU_HYPRE_Real cz,
                             NALU_HYPRE_BigInt *ilower, NALU_HYPRE_BigInt *iupper,
                             NALU_HYPRE_BigInt *jlower, NALU_HYPRE_BigInt *jupper,
                             NALU_HYPRE_Int *nrows, NALU_HYPRE_BigInt *num_nonzeros,
                             NALU_HYPRE_Int **nnzrow_ptr, NALU_HYPRE_BigInt **rows_ptr,
                             NALU_HYPRE_BigInt **rows2_ptr, NALU_HYPRE_BigInt **cols_ptr,
                             NALU_HYPRE_Real **coefs_ptr, NALU_HYPRE_Int stencil, NALU_HYPRE_ParCSRMatrix *parcsr_ptr);

NALU_HYPRE_Int getParCSRMatrixData(NALU_HYPRE_ParCSRMatrix  A, NALU_HYPRE_Int *nrows_ptr,
                              NALU_HYPRE_BigInt *num_nonzeros_ptr,
                              NALU_HYPRE_Int **nnzrow_ptr, NALU_HYPRE_BigInt **rows_ptr, NALU_HYPRE_BigInt **rows2_ptr,
                              NALU_HYPRE_BigInt **cols_ptr, NALU_HYPRE_Real **coefs_ptr);

NALU_HYPRE_Int checkMatrix(NALU_HYPRE_ParCSRMatrix parcsr_ref, NALU_HYPRE_IJMatrix ij_A);

NALU_HYPRE_Int test_Set(MPI_Comm comm, NALU_HYPRE_MemoryLocation memory_location, NALU_HYPRE_Int option,
                   NALU_HYPRE_BigInt ilower, NALU_HYPRE_BigInt iupper,
                   NALU_HYPRE_Int nrows, NALU_HYPRE_BigInt num_nonzeros,
                   NALU_HYPRE_Int nchunks, NALU_HYPRE_Int *h_nnzrow, NALU_HYPRE_Int *nnzrow,
                   NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols,
                   NALU_HYPRE_Real *coefs, NALU_HYPRE_IJMatrix *ij_A_ptr);

NALU_HYPRE_Int test_SetOffProc(NALU_HYPRE_ParCSRMatrix parcsr_A, NALU_HYPRE_MemoryLocation memory_location,
                          NALU_HYPRE_Int nchunks, NALU_HYPRE_Int option, NALU_HYPRE_IJMatrix *ij_AT_ptr);

NALU_HYPRE_Int test_SetSet(MPI_Comm comm, NALU_HYPRE_MemoryLocation memory_location, NALU_HYPRE_Int option,
                      NALU_HYPRE_BigInt ilower, NALU_HYPRE_BigInt iupper,
                      NALU_HYPRE_Int nrows, NALU_HYPRE_BigInt num_nonzeros,
                      NALU_HYPRE_Int nchunks, NALU_HYPRE_Int *h_nnzrow, NALU_HYPRE_Int *nnzrow,
                      NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols,
                      NALU_HYPRE_Real *coefs, NALU_HYPRE_IJMatrix *ij_A_ptr);

NALU_HYPRE_Int test_AddSet(MPI_Comm comm, NALU_HYPRE_MemoryLocation memory_location, NALU_HYPRE_Int option,
                      NALU_HYPRE_BigInt ilower, NALU_HYPRE_BigInt iupper,
                      NALU_HYPRE_Int nrows, NALU_HYPRE_BigInt num_nonzeros,
                      NALU_HYPRE_Int nchunks, NALU_HYPRE_Int *h_nnzrow, NALU_HYPRE_Int *nnzrow,
                      NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols,
                      NALU_HYPRE_Real *coefs, NALU_HYPRE_IJMatrix *ij_A_ptr);

NALU_HYPRE_Int test_SetAddSet(MPI_Comm comm, NALU_HYPRE_MemoryLocation memory_location, NALU_HYPRE_Int option,
                         NALU_HYPRE_BigInt ilower, NALU_HYPRE_BigInt iupper,
                         NALU_HYPRE_Int nrows, NALU_HYPRE_BigInt num_nonzeros,
                         NALU_HYPRE_Int nchunks, NALU_HYPRE_Int *h_nnzrow, NALU_HYPRE_Int *nnzrow,
                         NALU_HYPRE_BigInt *rows, NALU_HYPRE_BigInt *cols,
                         NALU_HYPRE_Real *coefs, NALU_HYPRE_IJMatrix *ij_A_ptr);

//#define CUDA_PROFILER

nalu_hypre_int
main( nalu_hypre_int  argc,
      char      *argv[] )
{
   MPI_Comm                  comm = nalu_hypre_MPI_COMM_WORLD;
   NALU_HYPRE_Int                 num_procs;
   NALU_HYPRE_Int                 myid;
   NALU_HYPRE_Int                 arg_index;
   NALU_HYPRE_Int                 time_index;
   NALU_HYPRE_Int                 print_usage;
   NALU_HYPRE_MemoryLocation      memory_location;
#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_ExecutionPolicy    default_exec_policy;
#endif
   char                      memory_location_name[8];

   NALU_HYPRE_Int                 nrows;
   NALU_HYPRE_BigInt              num_nonzeros;
   NALU_HYPRE_BigInt              ilower, iupper;
   NALU_HYPRE_BigInt              jlower, jupper;
   NALU_HYPRE_Int                *nnzrow, *h_nnzrow, *d_nnzrow;
   NALU_HYPRE_BigInt             *rows,   *h_rows,   *d_rows;
   NALU_HYPRE_BigInt             *rows2,  *h_rows2,  *d_rows2;
   NALU_HYPRE_BigInt             *cols,   *h_cols,   *d_cols;
   NALU_HYPRE_Real               *coefs,  *h_coefs,  *d_coefs;
   NALU_HYPRE_IJMatrix            ij_A;
   NALU_HYPRE_IJMatrix            ij_AT;
   NALU_HYPRE_ParCSRMatrix        parcsr_ref;

   // Driver input parameters
   NALU_HYPRE_Int                 Px, Py, Pz;
   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Real                cx, cy, cz;
   NALU_HYPRE_Int                 nchunks;
   NALU_HYPRE_Int                 mode;
   NALU_HYPRE_Int                 option;
   NALU_HYPRE_Int                 stencil;
   NALU_HYPRE_Int                 print_matrix;

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);
   nalu_hypre_MPI_Comm_size(comm, &num_procs );
   nalu_hypre_MPI_Comm_rank(comm, &myid );

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before NALU_HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   nalu_hypre_bind_device(myid, num_procs, nalu_hypre_MPI_COMM_WORLD);

   /* Initialize Hypre */
   /* Initialize Hypre: must be the first Hypre function to call */
   time_index = nalu_hypre_InitializeTiming("Hypre init");
   nalu_hypre_BeginTiming(time_index);
   NALU_HYPRE_Initialize();
   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Hypre init times", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Set default parameters
    *-----------------------------------------------------------*/
   Px = num_procs;
   Py = 1;
   Pz = 1;

   nx = 100;
   ny = 101;
   nz = 102;

   cx = 1.0;
   cy = 2.0;
   cz = 3.0;

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   default_exec_policy = NALU_HYPRE_EXEC_DEVICE;
#endif
   memory_location     = NALU_HYPRE_MEMORY_DEVICE;
   mode                = 1;
   option              = 1;
   nchunks             = 1;
   print_matrix        = 0;
   stencil             = 7;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   print_usage = 0;
   arg_index = 1;
   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-memory_location") == 0 )
      {
         arg_index++;
         memory_location = (NALU_HYPRE_MemoryLocation) atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         Px = atoi(argv[arg_index++]);
         Py = atoi(argv[arg_index++]);
         Pz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx  = atoi(argv[arg_index++]);
         ny  = atoi(argv[arg_index++]);
         nz  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cy = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cz = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mode") == 0 )
      {
         arg_index++;
         mode = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-option") == 0 )
      {
         arg_index++;
         option = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         stencil = 9;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         stencil = 27;
      }
      else if ( strcmp(argv[arg_index], "-nchunks") == 0 )
      {
         arg_index++;
         nchunks = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_matrix = 1;
      }
      else
      {
         print_usage = 1; break;
      }
   }

   /*-----------------------------------------------------------
    * Safety checks
    *-----------------------------------------------------------*/
   if (Px * Py * Pz != num_procs)
   {
      nalu_hypre_printf("Px x Py x Pz is different than the number of MPI processes");
      return (-1);
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
   if ( print_usage )
   {
      if ( myid == 0 )
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Usage: %s [<options>]\n", argv[0]);
         nalu_hypre_printf("\n");
         nalu_hypre_printf("      -n <nx> <ny> <nz>      : total problem size \n");
         nalu_hypre_printf("      -P <Px> <Py> <Pz>      : processor topology\n");
         nalu_hypre_printf("      -c <cx> <cy> <cz>      : diffusion coefficients\n");
         nalu_hypre_printf("      -memory_location <val> : memory location of the assembled matrix\n");
         nalu_hypre_printf("             0 = HOST\n");
         nalu_hypre_printf("             1 = DEVICE (default)\n");
         nalu_hypre_printf("      -nchunks <val>         : number of chunks passed to Set/AddValues\n");
         nalu_hypre_printf("      -mode <val>            : tests to be performed\n");
         nalu_hypre_printf("             1 = Set (default)\n");
         nalu_hypre_printf("             2 = SetOffProc\n");
         nalu_hypre_printf("             4 = SetSet\n");
         nalu_hypre_printf("             8 = AddSet\n");
         nalu_hypre_printf("            16 = SetAddSet\n");
         nalu_hypre_printf("      -option <val>          : interface option of Set/AddToValues\n");
         nalu_hypre_printf("             1 = CSR-like (default)\n");
         nalu_hypre_printf("             2 = COO-like\n");
         nalu_hypre_printf("      -print                 : print matrices\n");
         nalu_hypre_printf("\n");
      }

      return (0);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
   switch (memory_location)
   {
      case NALU_HYPRE_MEMORY_UNDEFINED:
         return -1;

      case NALU_HYPRE_MEMORY_DEVICE:
         nalu_hypre_sprintf(memory_location_name, "Device"); break;

      case NALU_HYPRE_MEMORY_HOST:
         nalu_hypre_sprintf(memory_location_name, "Host"); break;
   }

   if (myid == 0)
   {
      nalu_hypre_printf("  Memory location: %s\n", memory_location_name);
      nalu_hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", Px, Py, Pz);
      nalu_hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      nalu_hypre_printf("\n");
   }

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   nalu_hypre_HandleDefaultExecPolicy(nalu_hypre_handle()) = default_exec_policy;
#endif

   /*-----------------------------------------------------------
    * Build matrix entries
    *-----------------------------------------------------------*/
   buildMatrixEntries(comm, nx, ny, nz, Px, Py, Pz, cx, cy, cz,
                      &ilower, &iupper, &jlower, &jupper, &nrows, &num_nonzeros,
                      &h_nnzrow, &h_rows, &h_rows2, &h_cols, &h_coefs, stencil, &parcsr_ref);

   switch (memory_location)
   {
      case NALU_HYPRE_MEMORY_DEVICE:
         d_nnzrow = nalu_hypre_TAlloc(NALU_HYPRE_Int,    nrows,        NALU_HYPRE_MEMORY_DEVICE);
         d_rows   = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nrows,        NALU_HYPRE_MEMORY_DEVICE);
         d_rows2  = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE);
         d_cols   = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE);
         d_coefs  = nalu_hypre_TAlloc(NALU_HYPRE_Real,   num_nonzeros, NALU_HYPRE_MEMORY_DEVICE);

         nalu_hypre_TMemcpy(d_nnzrow, h_nnzrow, NALU_HYPRE_Int,    nrows,        NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(d_rows,   h_rows,   NALU_HYPRE_BigInt, nrows,        NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(d_rows2,  h_rows2,  NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(d_cols,   h_cols,   NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(d_coefs,  h_coefs,  NALU_HYPRE_Real,   num_nonzeros, NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_HOST);

         nnzrow = d_nnzrow;
         rows   = d_rows;
         rows2  = d_rows2;
         cols   = d_cols;
         coefs  = d_coefs;
         break;

      case NALU_HYPRE_MEMORY_HOST:
         nnzrow = h_nnzrow;
         rows   = h_rows;
         rows2  = h_rows2;
         cols   = h_cols;
         coefs  = h_coefs;
         break;

      case NALU_HYPRE_MEMORY_UNDEFINED:
         return -1;
   }

   /*-----------------------------------------------------------
    * Test different Set/Add combinations
    *-----------------------------------------------------------*/
   /* Test Set */
   if (mode & 1)
   {
      test_Set(comm, memory_location, option, ilower, iupper, nrows, num_nonzeros,
               nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows2, cols, coefs, &ij_A);

      checkMatrix(parcsr_ref, ij_A);
      if (print_matrix)
      {
         NALU_HYPRE_IJMatrixPrint(ij_A, "ij_Set");
      }
      NALU_HYPRE_IJMatrixDestroy(ij_A);
   }

   /* Test SetOffProc */
   if (mode & 2)
   {
      test_SetOffProc(parcsr_ref, memory_location, nchunks, option, &ij_AT);
      checkMatrix(parcsr_ref, ij_AT);
      if (print_matrix)
      {
         NALU_HYPRE_IJMatrixPrint(ij_A, "ij_SetOffProc");
      }
      NALU_HYPRE_IJMatrixDestroy(ij_AT);
   }

   /* Test Set/Set */
   if (mode & 4)
   {
      test_SetSet(comm, memory_location, option, ilower, iupper, nrows, num_nonzeros,
                  nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows2, cols, coefs, &ij_A);

      checkMatrix(parcsr_ref, ij_A);
      if (print_matrix)
      {
         NALU_HYPRE_IJMatrixPrint(ij_A, "ij_SetSet");
      }
      NALU_HYPRE_IJMatrixDestroy(ij_A);
   }

   /* Test Add/Set */
   if (mode & 8)
   {
      test_AddSet(comm, memory_location, option, ilower, iupper, nrows, num_nonzeros,
                  nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows2, cols, coefs, &ij_A);

      checkMatrix(parcsr_ref, ij_A);
      if (print_matrix)
      {
         NALU_HYPRE_IJMatrixPrint(ij_A, "ij_AddSet");
      }
      NALU_HYPRE_IJMatrixDestroy(ij_A);
   }

   /* Test Set/Add/Set */
   if (mode & 16)
   {
      test_SetAddSet(comm, memory_location, option, ilower, iupper, nrows, num_nonzeros,
                     nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows2, cols, coefs, &ij_A);

      checkMatrix(parcsr_ref, ij_A);
      if (print_matrix)
      {
         NALU_HYPRE_IJMatrixPrint(ij_A, "ij_SetAddSet");
      }
      NALU_HYPRE_IJMatrixDestroy(ij_A);
   }

   /*-----------------------------------------------------------
    * Free memory
    *-----------------------------------------------------------*/
   if (memory_location == NALU_HYPRE_MEMORY_DEVICE)
   {
      nalu_hypre_TFree(d_nnzrow, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(d_rows,   NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(d_rows2,  NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(d_cols,   NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(d_coefs,  NALU_HYPRE_MEMORY_DEVICE);
   }
   nalu_hypre_TFree(h_nnzrow, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(h_rows,   NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(h_rows2,  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(h_cols,   NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(h_coefs,  NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_ParCSRMatrixDestroy(parcsr_ref);

   /* Finalize Hypre */
   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   /* when using cuda-memcheck --leak-check full, uncomment this */
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_ResetCudaDevice(nalu_hypre_handle());
#endif

   return (0);
}

NALU_HYPRE_Int
buildMatrixEntries(MPI_Comm            comm,
                   NALU_HYPRE_Int           nx,
                   NALU_HYPRE_Int           ny,
                   NALU_HYPRE_Int           nz,
                   NALU_HYPRE_Int           Px,
                   NALU_HYPRE_Int           Py,
                   NALU_HYPRE_Int           Pz,
                   NALU_HYPRE_Real           cx,
                   NALU_HYPRE_Real           cy,
                   NALU_HYPRE_Real           cz,
                   NALU_HYPRE_BigInt       *ilower_ptr,
                   NALU_HYPRE_BigInt       *iupper_ptr,
                   NALU_HYPRE_BigInt       *jlower_ptr,
                   NALU_HYPRE_BigInt       *jupper_ptr,
                   NALU_HYPRE_Int          *nrows_ptr,
                   NALU_HYPRE_BigInt       *num_nonzeros_ptr,
                   NALU_HYPRE_Int         **nnzrow_ptr,
                   NALU_HYPRE_BigInt      **rows_ptr,   /* row indices of length nrows */
                   NALU_HYPRE_BigInt      **rows2_ptr,  /* row indices of length nnz */
                   NALU_HYPRE_BigInt      **cols_ptr,   /* col indices of length nnz */
                   NALU_HYPRE_Real        **coefs_ptr,  /* values of length nnz */
                   NALU_HYPRE_Int           stencil,
                   NALU_HYPRE_ParCSRMatrix *parcsr_ptr)
{
   NALU_HYPRE_Int        num_procs;
   NALU_HYPRE_Int        myid;
   NALU_HYPRE_Real       values[4];
   NALU_HYPRE_ParCSRMatrix A;

   nalu_hypre_MPI_Comm_size(comm, &num_procs );
   nalu_hypre_MPI_Comm_rank(comm, &myid );

   NALU_HYPRE_Int ip = myid % Px;
   NALU_HYPRE_Int iq = (( myid - ip) / Px) % Py;
   NALU_HYPRE_Int ir = ( myid - ip - Px * iq) / ( Px * Py );

   values[0] = 0;
   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   if (stencil == 7)
   {
      A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian(comm, nx, ny, nz, Px, Py, Pz, ip, iq, ir, values);
   }
   else if (stencil == 9)
   {
      A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian9pt(comm, nx, ny, Px, Py, ip, iq, values);
   }
   else if (stencil == 27)
   {
      A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian27pt(comm, nx, ny, nz, Px, Py, Pz, ip, iq, ir, values);
   }
   else
   {
      nalu_hypre_assert(0);
   }

   nalu_hypre_ParCSRMatrixMigrate(A, NALU_HYPRE_MEMORY_HOST);
   getParCSRMatrixData(A, nrows_ptr, num_nonzeros_ptr, nnzrow_ptr, rows_ptr, rows2_ptr, cols_ptr,
                       coefs_ptr);

   // Set pointers
   *ilower_ptr = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   *iupper_ptr = nalu_hypre_ParCSRMatrixLastRowIndex(A);
   *jlower_ptr = nalu_hypre_ParCSRMatrixFirstColDiag(A);
   *jupper_ptr = nalu_hypre_ParCSRMatrixLastColDiag(A);
   *parcsr_ptr = A;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
getParCSRMatrixData(NALU_HYPRE_ParCSRMatrix  A,
                    NALU_HYPRE_Int          *nrows_ptr,
                    NALU_HYPRE_BigInt       *num_nonzeros_ptr,
                    NALU_HYPRE_Int         **nnzrow_ptr,
                    NALU_HYPRE_BigInt      **rows_ptr,
                    NALU_HYPRE_BigInt      **rows2_ptr,
                    NALU_HYPRE_BigInt      **cols_ptr,
                    NALU_HYPRE_Real        **coefs_ptr)
{
   nalu_hypre_CSRMatrix    *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix    *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int          *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int          *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int          *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int          *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_BigInt       *col_map_offd_A = nalu_hypre_ParCSRMatrixColMapOffd(A);

   NALU_HYPRE_BigInt       ilower = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_BigInt       jlower = nalu_hypre_ParCSRMatrixFirstColDiag(A);

   NALU_HYPRE_Int          nrows;
   NALU_HYPRE_BigInt       num_nonzeros;
   NALU_HYPRE_Int         *nnzrow;
   NALU_HYPRE_BigInt      *rows;
   NALU_HYPRE_BigInt      *rows2;
   NALU_HYPRE_BigInt      *cols;
   NALU_HYPRE_Real        *coefs;
   NALU_HYPRE_Int          i, j, k;

   nrows  = nalu_hypre_ParCSRMatrixNumRows(A);
   num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A_diag) + nalu_hypre_CSRMatrixNumNonzeros(A_offd);
   nnzrow = nalu_hypre_CTAlloc(NALU_HYPRE_Int,    nrows,        NALU_HYPRE_MEMORY_HOST);
   rows   = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nrows,        NALU_HYPRE_MEMORY_HOST);
   rows2  = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_HOST);
   cols   = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_HOST);
   coefs  = nalu_hypre_CTAlloc(NALU_HYPRE_Real,   num_nonzeros, NALU_HYPRE_MEMORY_HOST);

   k = 0;
#if 0
   for (i = 0; i < nrows; i++)
   {
      nnzrow[i] = A_diag_i[i + 1] - A_diag_i[i] +
                  A_offd_i[i + 1] - A_offd_i[i];
      rows[i]   = ilower + i;

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         rows2[k]   = ilower + (NALU_HYPRE_BigInt) i;
         cols[k]    = jlower + (NALU_HYPRE_BigInt) A_diag_j[j];
         coefs[k++] = nalu_hypre_CSRMatrixData(A_diag)[j];
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         rows2[k]   = ilower + (NALU_HYPRE_BigInt) i;
         cols[k]    = nalu_hypre_ParCSRMatrixColMapOffd(A)[A_offd_j[j]];
         coefs[k++] = nalu_hypre_CSRMatrixData(A_offd)[j];
      }
   }
#else
   for (i = nrows - 1; i >= 0; i--)
   {
      nnzrow[nrows - 1 - i] = A_diag_i[i + 1] - A_diag_i[i] +
                              A_offd_i[i + 1] - A_offd_i[i];
      rows[nrows - 1 - i]   = ilower + i;

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         rows2[k]   = ilower + (NALU_HYPRE_BigInt) i;
         cols[k]    = jlower + (NALU_HYPRE_BigInt) A_diag_j[j];
         coefs[k++] = nalu_hypre_CSRMatrixData(A_diag)[j];
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         rows2[k]   = ilower + (NALU_HYPRE_BigInt) i;
         cols[k]    = col_map_offd_A[A_offd_j[j]];
         coefs[k++] = nalu_hypre_CSRMatrixData(A_offd)[j];
      }
   }
#endif

   nalu_hypre_assert(k == num_nonzeros);

   // Set pointers
   *nrows_ptr        = nrows;
   *num_nonzeros_ptr = num_nonzeros;
   *nnzrow_ptr       = nnzrow;
   *rows_ptr         = rows;
   *rows2_ptr        = rows2;
   *cols_ptr         = cols;
   *coefs_ptr        = coefs;

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
checkMatrix(NALU_HYPRE_ParCSRMatrix h_parcsr_ref, NALU_HYPRE_IJMatrix ij_A)
{
   MPI_Comm            comm         = nalu_hypre_IJMatrixComm(ij_A);
   NALU_HYPRE_ParCSRMatrix  parcsr_A     = (NALU_HYPRE_ParCSRMatrix) nalu_hypre_IJMatrixObject(ij_A);
   NALU_HYPRE_ParCSRMatrix  h_parcsr_A;
   NALU_HYPRE_ParCSRMatrix  parcsr_error;
   NALU_HYPRE_Int           myid;
   NALU_HYPRE_Real          fnorm;

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   h_parcsr_A = nalu_hypre_ParCSRMatrixClone_v2(parcsr_A, 1, NALU_HYPRE_MEMORY_HOST);

   // Check norm of (parcsr_ref - parcsr_A)
   nalu_hypre_ParCSRMatrixAdd(1.0, h_parcsr_ref, -1.0, h_parcsr_A, &parcsr_error);
   fnorm = nalu_hypre_ParCSRMatrixFnorm(parcsr_error);

   if (myid == 0)
   {
      nalu_hypre_printf("Frobenius norm of (A_ref - A): %e\n", fnorm);
   }

   NALU_HYPRE_ParCSRMatrixDestroy(h_parcsr_A);
   NALU_HYPRE_ParCSRMatrixDestroy(parcsr_error);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
test_Set(MPI_Comm             comm,
         NALU_HYPRE_MemoryLocation memory_location,
         NALU_HYPRE_Int            option,           /* 1 or 2 */
         NALU_HYPRE_BigInt         ilower,
         NALU_HYPRE_BigInt         iupper,
         NALU_HYPRE_Int            nrows,
         NALU_HYPRE_BigInt         num_nonzeros,
         NALU_HYPRE_Int            nchunks,
         NALU_HYPRE_Int           *h_nnzrow,
         NALU_HYPRE_Int           *nnzrow,
         NALU_HYPRE_BigInt
         *rows,             /* option = 1: length of nrows, = 2: length of num_nonzeros */
         NALU_HYPRE_BigInt        *cols,
         NALU_HYPRE_Real          *coefs,
         NALU_HYPRE_IJMatrix      *ij_A_ptr)
{
   NALU_HYPRE_IJMatrix  ij_A;
   NALU_HYPRE_Int       i, chunk, chunk_size;
   NALU_HYPRE_Int       time_index;
   NALU_HYPRE_Int      *h_rowptr;

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_A);
   NALU_HYPRE_IJMatrixSetObjectType(ij_A, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);
   NALU_HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   h_rowptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   nalu_hypre_assert(h_rowptr[nrows] == num_nonzeros);

   chunk_size = nrows / nchunks;

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   time_index = nalu_hypre_InitializeTiming("Test SetValues");
   nalu_hypre_BeginTiming(time_index);
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = nalu_hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   NALU_HYPRE_IJMatrixAssemble(ij_A);

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Test SetValues", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   // Free memory
   nalu_hypre_TFree(h_rowptr, NALU_HYPRE_MEMORY_HOST);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
test_SetOffProc(NALU_HYPRE_ParCSRMatrix    parcsr_A,
                NALU_HYPRE_MemoryLocation  memory_location,
                NALU_HYPRE_Int             nchunks,
                NALU_HYPRE_Int             option,           /* 1 or 2 */
                NALU_HYPRE_IJMatrix       *ij_AT_ptr)
{
   MPI_Comm            comm = nalu_hypre_ParCSRMatrixComm(parcsr_A);
   NALU_HYPRE_ParCSRMatrix  parcsr_AT;
   NALU_HYPRE_IJMatrix      ij_AT;

   NALU_HYPRE_Int           nrows;
   NALU_HYPRE_BigInt        num_nonzeros;
   NALU_HYPRE_BigInt        ilower, iupper;

   NALU_HYPRE_Int          *h_nnzrow;
   NALU_HYPRE_BigInt       *h_rows1;
   NALU_HYPRE_BigInt       *h_rows2;
   NALU_HYPRE_BigInt       *h_cols;
   NALU_HYPRE_Real         *h_coefs;

   NALU_HYPRE_Int          *d_nnzrow;
   NALU_HYPRE_BigInt       *d_rows;
   NALU_HYPRE_BigInt       *d_cols;
   NALU_HYPRE_Real         *d_coefs;

   NALU_HYPRE_Int          *nnzrow;
   NALU_HYPRE_BigInt       *rows;
   NALU_HYPRE_BigInt       *cols;
   NALU_HYPRE_Real         *coefs;

   NALU_HYPRE_Int          *h_rowptr;

   NALU_HYPRE_Int           time_index;
   NALU_HYPRE_Int           chunk_size;
   NALU_HYPRE_Int           chunk;
   NALU_HYPRE_Int           i;

   nalu_hypre_ParCSRMatrixTranspose(parcsr_A, &parcsr_AT, 1);
   ilower = nalu_hypre_ParCSRMatrixFirstRowIndex(parcsr_AT);
   iupper = nalu_hypre_ParCSRMatrixLastRowIndex(parcsr_AT);
   getParCSRMatrixData(parcsr_AT, &nrows, &num_nonzeros, &h_nnzrow, &h_rows1, &h_rows2, &h_cols,
                       &h_coefs);
   NALU_HYPRE_ParCSRMatrixDestroy(parcsr_AT);

   switch (memory_location)
   {
      case NALU_HYPRE_MEMORY_DEVICE:
         d_nnzrow = nalu_hypre_TAlloc(NALU_HYPRE_Int,    nrows,        NALU_HYPRE_MEMORY_DEVICE);
         d_cols   = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE);
         d_coefs  = nalu_hypre_TAlloc(NALU_HYPRE_Real,   num_nonzeros, NALU_HYPRE_MEMORY_DEVICE);
         if (option == 1)
         {
            d_rows  = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nrows,        NALU_HYPRE_MEMORY_DEVICE);
            nalu_hypre_TMemcpy(d_rows,  h_rows1,  NALU_HYPRE_BigInt, nrows,        NALU_HYPRE_MEMORY_DEVICE,
                          NALU_HYPRE_MEMORY_HOST);
         }
         else
         {
            d_rows  = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE);
            nalu_hypre_TMemcpy(d_rows,  h_rows2,  NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE,
                          NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TMemcpy(d_nnzrow, h_nnzrow, NALU_HYPRE_Int,    nrows,        NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(d_cols,   h_cols,   NALU_HYPRE_BigInt, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(d_coefs,  h_coefs,  NALU_HYPRE_Real,   num_nonzeros, NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_HOST);

         nnzrow = d_nnzrow;
         rows   = d_rows;
         cols   = d_cols;
         coefs  = d_coefs;
         break;

      case NALU_HYPRE_MEMORY_HOST:
         nnzrow = h_nnzrow;
         rows   = (option == 1) ? h_rows1 : h_rows2;
         cols   = h_cols;
         coefs  = h_coefs;
         break;

      case NALU_HYPRE_MEMORY_UNDEFINED:
         return -1;
   }

   // Create transpose with SetValues
   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_AT);
   NALU_HYPRE_IJMatrixSetObjectType(ij_AT, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize_v2(ij_AT, memory_location);
   NALU_HYPRE_IJMatrixSetOMPFlag(ij_AT, 1);

   h_rowptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   nalu_hypre_assert(h_rowptr[nrows] == num_nonzeros);

   chunk_size = nrows / nchunks;

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#endif

   time_index = nalu_hypre_InitializeTiming("Test SetValues OffProc");
   nalu_hypre_BeginTiming(time_index);

   //cudaProfilerStart();

   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = nalu_hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         NALU_HYPRE_IJMatrixSetValues(ij_AT, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         NALU_HYPRE_IJMatrixSetValues(ij_AT, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   NALU_HYPRE_IJMatrixAssemble(ij_AT);

   //cudaProfilerStop();

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#endif

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Test SetValues OffProc", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   // Set pointer to output
   *ij_AT_ptr = ij_AT;

   // Free memory
   nalu_hypre_TFree(h_rowptr, NALU_HYPRE_MEMORY_HOST);
   if (memory_location == NALU_HYPRE_MEMORY_DEVICE)
   {
      nalu_hypre_TFree(d_nnzrow, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(d_rows,   NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(d_cols,   NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(d_coefs,  NALU_HYPRE_MEMORY_DEVICE);
   }
   nalu_hypre_TFree(h_nnzrow, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(h_rows1,  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(h_rows2,  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(h_cols,   NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(h_coefs,  NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
test_SetSet(MPI_Comm             comm,
            NALU_HYPRE_MemoryLocation memory_location,
            NALU_HYPRE_Int            option,           /* 1 or 2 */
            NALU_HYPRE_BigInt         ilower,
            NALU_HYPRE_BigInt         iupper,
            NALU_HYPRE_Int            nrows,
            NALU_HYPRE_BigInt         num_nonzeros,
            NALU_HYPRE_Int            nchunks,
            NALU_HYPRE_Int           *h_nnzrow,
            NALU_HYPRE_Int           *nnzrow,
            NALU_HYPRE_BigInt
            *rows,             /* option = 1: length of nrows, = 2: length of num_nonzeros */
            NALU_HYPRE_BigInt        *cols,
            NALU_HYPRE_Real          *coefs,
            NALU_HYPRE_IJMatrix      *ij_A_ptr)
{
   NALU_HYPRE_IJMatrix  ij_A;
   NALU_HYPRE_Int       i, chunk, chunk_size;
   NALU_HYPRE_Int       time_index;
   NALU_HYPRE_Int      *h_rowptr;
   NALU_HYPRE_Real     *new_coefs;

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_A);
   NALU_HYPRE_IJMatrixSetObjectType(ij_A, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);
   NALU_HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   h_rowptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   nalu_hypre_assert(h_rowptr[nrows] == num_nonzeros);

   chunk_size = nrows / nchunks;
   new_coefs = nalu_hypre_TAlloc(NALU_HYPRE_Real, num_nonzeros, memory_location);

   if (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_HOST)
   {
      for (i = 0; i < num_nonzeros; i++)
      {
         new_coefs[i] = 2.0 * coefs[i];
      }
   }
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   else
   {
      nalu_hypre_TMemcpy(new_coefs, coefs, NALU_HYPRE_Real, num_nonzeros, memory_location, memory_location);
      hypreDevice_ComplexScalen(new_coefs, num_nonzeros, new_coefs, 2.0);
   }
#endif

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   // First Set
   time_index = nalu_hypre_InitializeTiming("Test Set/Set");
   nalu_hypre_BeginTiming(time_index);
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = nalu_hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &new_coefs[h_rowptr[chunk]]);
      }
      else
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &new_coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   NALU_HYPRE_IJMatrixAssemble(ij_A);

   // Second set
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = nalu_hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   NALU_HYPRE_IJMatrixAssemble(ij_A);

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Test Set/Set", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   // Free memory
   nalu_hypre_TFree(h_rowptr, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(new_coefs, memory_location);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
test_AddSet(MPI_Comm             comm,
            NALU_HYPRE_MemoryLocation memory_location,
            NALU_HYPRE_Int            option,           /* 1 or 2 */
            NALU_HYPRE_BigInt         ilower,
            NALU_HYPRE_BigInt         iupper,
            NALU_HYPRE_Int            nrows,
            NALU_HYPRE_BigInt         num_nonzeros,
            NALU_HYPRE_Int            nchunks,
            NALU_HYPRE_Int           *h_nnzrow,
            NALU_HYPRE_Int           *nnzrow,
            NALU_HYPRE_BigInt
            *rows,             /* option = 1: length of nrows, = 2: length of num_nonzeros */
            NALU_HYPRE_BigInt        *cols,
            NALU_HYPRE_Real          *coefs,
            NALU_HYPRE_IJMatrix      *ij_A_ptr)
{
   NALU_HYPRE_IJMatrix  ij_A;
   NALU_HYPRE_Int       i, chunk, chunk_size;
   NALU_HYPRE_Int       time_index;
   NALU_HYPRE_Int      *h_rowptr;
   NALU_HYPRE_Real     *new_coefs;

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_A);
   NALU_HYPRE_IJMatrixSetObjectType(ij_A, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);
   NALU_HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   h_rowptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   nalu_hypre_assert(h_rowptr[nrows] == num_nonzeros);

   chunk_size = nrows / nchunks;
   new_coefs = nalu_hypre_TAlloc(NALU_HYPRE_Real, num_nonzeros, memory_location);

   if (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_HOST)
   {
      for (i = 0; i < num_nonzeros; i++)
      {
         new_coefs[i] = 2.0 * coefs[i];
      }
   }
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   else
   {
      nalu_hypre_TMemcpy(new_coefs, coefs, NALU_HYPRE_Real, num_nonzeros, memory_location, memory_location);
      hypreDevice_ComplexScalen(new_coefs, num_nonzeros, new_coefs, 2.0);
   }
#endif

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   // First Add
   time_index = nalu_hypre_InitializeTiming("Test Add/Set");
   nalu_hypre_BeginTiming(time_index);
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = nalu_hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         NALU_HYPRE_IJMatrixAddToValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                   &cols[h_rowptr[chunk]], &new_coefs[h_rowptr[chunk]]);
      }
      else
      {
         NALU_HYPRE_IJMatrixAddToValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                   NULL, &rows[h_rowptr[chunk]],
                                   &cols[h_rowptr[chunk]], &new_coefs[h_rowptr[chunk]]);
      }
   }

   // Then Set
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = nalu_hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   NALU_HYPRE_IJMatrixAssemble(ij_A);

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Test Add/Set", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   // Free memory
   nalu_hypre_TFree(h_rowptr, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(new_coefs, memory_location);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
test_SetAddSet(MPI_Comm             comm,
               NALU_HYPRE_MemoryLocation memory_location,
               NALU_HYPRE_Int            option,           /* 1 or 2 */
               NALU_HYPRE_BigInt         ilower,
               NALU_HYPRE_BigInt         iupper,
               NALU_HYPRE_Int            nrows,
               NALU_HYPRE_BigInt         num_nonzeros,
               NALU_HYPRE_Int            nchunks,
               NALU_HYPRE_Int           *h_nnzrow,
               NALU_HYPRE_Int           *nnzrow,
               NALU_HYPRE_BigInt
               *rows,             /* option = 1: length of nrows, = 2: length of num_nonzeros */
               NALU_HYPRE_BigInt        *cols,
               NALU_HYPRE_Real          *coefs,
               NALU_HYPRE_IJMatrix      *ij_A_ptr)
{
   NALU_HYPRE_IJMatrix  ij_A;
   NALU_HYPRE_Int       i, chunk, chunk_size;
   NALU_HYPRE_Int       time_index;
   NALU_HYPRE_Int      *h_rowptr;

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_A);
   NALU_HYPRE_IJMatrixSetObjectType(ij_A, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);
   NALU_HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   h_rowptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   nalu_hypre_assert(h_rowptr[nrows] == num_nonzeros);
   chunk_size = nrows / nchunks;

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   // First Set
   time_index = nalu_hypre_InitializeTiming("Test Set/Add/Set");
   nalu_hypre_BeginTiming(time_index);
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = nalu_hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Then Add
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = nalu_hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         NALU_HYPRE_IJMatrixAddToValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                   &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         NALU_HYPRE_IJMatrixAddToValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                   NULL, &rows[h_rowptr[chunk]],
                                   &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Then Set
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = nalu_hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         NALU_HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   NALU_HYPRE_IJMatrixAssemble(ij_A);

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Test Set/Add/Set", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   // Free memory
   nalu_hypre_TFree(h_rowptr, NALU_HYPRE_MEMORY_HOST);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return nalu_hypre_error_flag;
}
