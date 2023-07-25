/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* driver_commpkg.c*/
/* AHB 06/04 */
/* purpose:  to test a new communication package for the ij interface */

/* 11/06 - if you want to use this, the the nalu_hypre_NewCommPkgCreate has to be
   reinstated in parcsr_mv/new_commpkg.c - currently it won't compile*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
/*   #include <mpi.h>   */

#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "NALU_HYPRE_parcsr_ls.h"

/* #include "_nalu_hypre_parcsr_ls.h"
 #include "NALU_HYPRE.h"
 #include "NALU_HYPRE_parcsr_mv.h"
 #include "NALU_HYPRE_krylov.h"  */



/*some debugging tools*/
#define   mydebug 0
#define   mpip_on 0

/*time an allgather in addition to the current commpkg -
  since the allgather happens outside of the communication package.*/
#define   time_gather 1

/* for timing multiple commpkg setup (if you want the time to be larger in the
   hopes of getting smaller stds - often not effective) */
#define   LOOP2  1


NALU_HYPRE_Int myBuildParLaplacian (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_ParCSRMatrix *A_ptr, NALU_HYPRE_Int parmprint );
NALU_HYPRE_Int myBuildParLaplacian27pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                   NALU_HYPRE_ParCSRMatrix *A_ptr, NALU_HYPRE_Int parmprint );


void stats_mo(NALU_HYPRE_Real*, NALU_HYPRE_Int, NALU_HYPRE_Real *, NALU_HYPRE_Real *);

/*==========================================================================*/


/*------------------------------------------------------------------
 *
 * This tests an alternate comm package for ij
 *
 * options:
 *         -laplacian              3D 7pt stencil
 *         -27pt                   3D 27pt laplacian
 *         -fromonecsrfile         read matrix from a csr file
 *         -commpkg <NALU_HYPRE_Int>          1 = new comm. package
 *                                 2  =old
 *                                 3 = both (default)
 *         -loop <NALU_HYPRE_Int>             number of times to loop (default is 0)
 *         -verbose                print more error checking
 *         -noparmprint            don't print the parameters
 *-------------------------------------------------------------------*/


NALU_HYPRE_Int
main( NALU_HYPRE_Int   argc,
      char *argv[] )
{


   NALU_HYPRE_Int        num_procs, myid;
   NALU_HYPRE_Int        verbose = 0, build_matrix_type = 1;
   NALU_HYPRE_Int        index, matrix_arg_index, commpkg_flag = 3;
   NALU_HYPRE_Int        i, k, ierr = 0;
   NALU_HYPRE_Int        row_start, row_end;
   NALU_HYPRE_Int        col_start, col_end, global_num_rows;
   NALU_HYPRE_Int       *row_part, *col_part;
   char      *csrfilename;
   NALU_HYPRE_Int        preload = 0, loop = 0, loop2 = LOOP2;
   NALU_HYPRE_Int        bcast_rows[2], *info;



   nalu_hypre_ParCSRMatrix    *parcsr_A, *small_A;
   NALU_HYPRE_ParCSRMatrix    A_temp, A_temp_small;
   nalu_hypre_CSRMatrix       *A_CSR;
   nalu_hypre_ParCSRCommPkg   *comm_pkg;


   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;
   NALU_HYPRE_Int                 p, q, r;
   NALU_HYPRE_Real          values[4];

   nalu_hypre_ParVector     *x_new;
   nalu_hypre_ParVector     *y_new, *y;
   NALU_HYPRE_Int                 *row_starts;
   NALU_HYPRE_Real          ans;
   NALU_HYPRE_Real          start_time, end_time, total_time, *loop_times;
   NALU_HYPRE_Real          T_avg, T_std;

   NALU_HYPRE_Int                   noparmprint = 0;

#if mydebug
   NALU_HYPRE_Int  j, tmp_int;
#endif

   /*-----------------------------------------------------------
    * Initialize MPI
    *-----------------------------------------------------------*/


   nalu_hypre_MPI_Init(&argc, &argv);

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );



   /*-----------------------------------------------------------
    * default - is 27pt laplace
    *-----------------------------------------------------------*/


   build_matrix_type = 2;
   matrix_arg_index = argc;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   index = 1;
   while ( index < argc)
   {
      if  ( strcmp(argv[index], "-verbose") == 0 )
      {
         index++;
         verbose = 1;
      }
      else if ( strcmp(argv[index], "-fromonecsrfile") == 0 )
      {
         index++;
         build_matrix_type      = 1;
         matrix_arg_index = index; /*this tells where the name is*/
      }
      else if  ( strcmp(argv[index], "-commpkg") == 0 )
      {
         index++;
         commpkg_flag = atoi(argv[index++]);
      }
      else if ( strcmp(argv[index], "-laplacian") == 0 )
      {
         index++;
         build_matrix_type      = 2;
         matrix_arg_index = index;
      }
      else if ( strcmp(argv[index], "-27pt") == 0 )
      {
         index++;
         build_matrix_type      = 4;
         matrix_arg_index = index;
      }
#if 0
      else if  ( strcmp(argv[index], "-nopreload") == 0 )
      {
         index++;
         preload = 0;
      }
#endif
      else if  ( strcmp(argv[index], "-loop") == 0 )
      {
         index++;
         loop = atoi(argv[index++]);
      }
      else if  ( strcmp(argv[index], "-noparmprint") == 0 )
      {
         index++;
         noparmprint = 1;

      }
      else
      {
         index++;
         /*nalu_hypre_printf("Warning: Unrecogized option '%s'\n",argv[index++] );*/
      }
   }



   /*-----------------------------------------------------------
    * Setup the Matrix problem
    *-----------------------------------------------------------*/

   /*-----------------------------------------------------------
     *  Get actual partitioning-
     *  read in an actual csr matrix.
     *-----------------------------------------------------------*/


   if (build_matrix_type == 1) /*read in a csr matrix from one file */
   {
      if (matrix_arg_index < argc)
      {
         csrfilename = argv[matrix_arg_index];
      }
      else
      {
         nalu_hypre_printf("Error: No filename specified \n");
         exit(1);
      }
      if (myid == 0)
      {
         /*nalu_hypre_printf("  FromFile: %s\n", csrfilename);*/
         A_CSR = nalu_hypre_CSRMatrixRead(csrfilename);
      }
      row_part = NULL;
      col_part = NULL;

      parcsr_A = nalu_hypre_CSRMatrixToParCSRMatrix(nalu_hypre_MPI_COMM_WORLD, A_CSR,
                                               row_part, col_part);

      if (myid == 0) { nalu_hypre_CSRMatrixDestroy(A_CSR); }
   }
   else if (build_matrix_type == 2)
   {

      myBuildParLaplacian(argc, argv, matrix_arg_index,  &A_temp, !noparmprint);
      parcsr_A = (nalu_hypre_ParCSRMatrix *) A_temp;

   }
   else if (build_matrix_type == 4)
   {
      myBuildParLaplacian27pt(argc, argv, matrix_arg_index, &A_temp, !noparmprint);
      parcsr_A = (nalu_hypre_ParCSRMatrix *) A_temp;
   }


   /*-----------------------------------------------------------
    * create a small problem so that timings are more accurate -
    * code gets run twice (small laplace)
    *-----------------------------------------------------------*/

   /*this is no longer being used - preload = 0 is set at the beginning */

   if (preload == 1)
   {

      /*nalu_hypre_printf("preload!\n");*/


      values[1] = -1;
      values[2] = -1;
      values[3] = -1;
      values[0] = - 6.0    ;

      nx = 2;
      ny = num_procs;
      nz = 2;

      P  = 1;
      Q  = num_procs;
      R  = 1;

      p = myid % P;
      q = (( myid - p) / P) % Q;
      r = ( myid - p - P * q) / ( P * Q );

      A_temp_small = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian(nalu_hypre_MPI_COMM_WORLD, nx, ny, nz,
                                                            P, Q, R, p, q, r, values);
      small_A = (nalu_hypre_ParCSRMatrix *) A_temp_small;

      /*do comm packages*/
      nalu_hypre_NewCommPkgCreate(small_A);
      nalu_hypre_NewCommPkgDestroy(small_A);

      nalu_hypre_MatvecCommPkgCreate(small_A);
      nalu_hypre_ParCSRMatrixDestroy(small_A);

   }





   /*-----------------------------------------------------------
    *  Prepare for timing
    *-----------------------------------------------------------*/

   /* instead of preloading, let's not time the first one if more than one*/


   if (!loop)
   {
      loop = 1;
      /* and don't do any timings */

   }
   else
   {

      loop += 1;
      if (loop < 2) { loop = 2; }
   }


   loop_times = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  loop, NALU_HYPRE_MEMORY_HOST);



   /******************************************************************************************/

   nalu_hypre_MPI_Barrier(nalu_hypre_MPI_COMM_WORLD);

   if (commpkg_flag == 1 || commpkg_flag == 3 )
   {

      /*-----------------------------------------------------------
       *  Create new comm package
       *-----------------------------------------------------------*/



      if (!myid) { nalu_hypre_printf("********************************************************\n" ); }

      /*do loop times*/
      for (i = 0; i < loop; i++)
      {
         loop_times[i] = 0.0;
         for (k = 0; k < loop2; k++)
         {

            nalu_hypre_MPI_Barrier(nalu_hypre_MPI_COMM_WORLD);

            start_time = nalu_hypre_MPI_Wtime();

#if mpip_on
            if (i == (loop - 1)) { nalu_hypre_MPI_Pcontrol(1); }
#endif

            nalu_hypre_NewCommPkgCreate(parcsr_A);

#if mpip_on
            if (i == (loop - 1)) { nalu_hypre_MPI_Pcontrol(0); }
#endif

            end_time = nalu_hypre_MPI_Wtime();

            end_time = end_time - start_time;

            nalu_hypre_MPI_Allreduce(&end_time, &total_time, 1,
                                NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_MAX, nalu_hypre_MPI_COMM_WORLD);

            loop_times[i] += total_time;

            if (  !((i + 1) == loop  &&  (k + 1) == loop2)) { nalu_hypre_NewCommPkgDestroy(parcsr_A); }

         }/*end of loop2 */


      } /*end of loop*/



      /* calculate the avg and std. */
      if (loop > 1)
      {

         /* calculate the avg and std. */
         stats_mo(loop_times, loop, &T_avg, &T_std);

         if (!myid) { nalu_hypre_printf(" NewCommPkgCreate:  AVG. wall clock time =  %f seconds\n", T_avg); }
         if (!myid) { nalu_hypre_printf("                    STD. for %d  runs     =  %f\n", loop - 1, T_std); }
         if (!myid) { nalu_hypre_printf("                    (Note: avg./std. timings exclude run 0.)\n"); }
         if (!myid) { nalu_hypre_printf("********************************************************\n" ); }
         for (i = 0; i < loop; i++)
         {
            if (!myid) { nalu_hypre_printf("      run %d  =  %f sec.\n", i, loop_times[i]); }
         }
         if (!myid) { nalu_hypre_printf("********************************************************\n" ); }

      }
      else
      {
         if (!myid) { nalu_hypre_printf("********************************************************\n" ); }
         if (!myid) { nalu_hypre_printf(" NewCommPkgCreate:\n"); }
         if (!myid) { nalu_hypre_printf("      run time =  %f sec.\n", loop_times[0]); }
         if (!myid) { nalu_hypre_printf("********************************************************\n" ); }
      }


      /*-----------------------------------------------------------
        *  Verbose printing
        *-----------------------------------------------------------*/

      /*some verification*/

      global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(parcsr_A);

      if (verbose)
      {

         ierr = nalu_hypre_ParCSRMatrixGetLocalRange( parcsr_A,
                                                 &row_start, &row_end,
                                                 &col_start, &col_end );


         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(parcsr_A);

         nalu_hypre_printf("myid = %i, my ACTUAL local range: [%i, %i]\n", myid,
                      row_start, row_end);


         ierr = nalu_hypre_GetAssumedPartitionRowRange( myid, global_num_rows, &row_start,
                                                   &row_end);


         nalu_hypre_printf("myid = %i, my assumed local range: [%i, %i]\n", myid,
                      row_start, row_end);

         nalu_hypre_printf("myid = %d, num_recvs = %d\n", myid,
                      nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg)  );

#if mydebug
         for (i = 0; i < nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg); i++)
         {
            nalu_hypre_printf("myid = %d, recv proc = %d, vec_starts = [%d : %d]\n",
                         myid,  nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg)[i],
                         nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i],
                         nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i + 1] - 1);
         }
#endif
         nalu_hypre_printf("myid = %d, num_sends = %d\n", myid,
                      nalu_hypre_ParCSRCommPkgNumSends(comm_pkg)  );

#if mydebug
         for (i = 0; i < nalu_hypre_ParCSRCommPkgNumSends(comm_pkg) ; i++)
         {
            tmp_int =  nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i + 1] -
                       nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
            index = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
            for (j = 0; j < tmp_int; j++)
            {
               nalu_hypre_printf("myid = %d, send proc = %d, send element = %d\n", myid,
                            nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg)[i],
                            nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg)[index + j]);
            }
         }
#endif
      }
      /*-----------------------------------------------------------
       *  To verify correctness (if commpkg_flag = 3)
       *-----------------------------------------------------------*/

      if (commpkg_flag == 3 )
      {
         /*do a matvec - we are assuming a square matrix */
         row_starts = nalu_hypre_ParCSRMatrixRowStarts(parcsr_A);

         x_new = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_num_rows, row_starts);
         nalu_hypre_ParVectorInitialize(x_new);
         nalu_hypre_ParVectorSetRandomValues(x_new, 1);

         y_new = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_num_rows, row_starts);
         nalu_hypre_ParVectorInitialize(y_new);
         nalu_hypre_ParVectorSetConstantValues(y_new, 0.0);

         /*y = 1.0*A*x+1.0*y */
         nalu_hypre_ParCSRMatrixMatvec (1.0, parcsr_A, x_new, 1.0, y_new);
      }

      /*-----------------------------------------------------------
       *  Clean up after MyComm
       *-----------------------------------------------------------*/


      nalu_hypre_NewCommPkgDestroy(parcsr_A);

   }






   /******************************************************************************************/
   /******************************************************************************************/

   nalu_hypre_MPI_Barrier(nalu_hypre_MPI_COMM_WORLD);


   if (commpkg_flag > 1 )
   {

      /*-----------------------------------------------------------
       *  Set up standard comm package
       *-----------------------------------------------------------*/

      bcast_rows[0] = 23;
      bcast_rows[1] = 1789;

      if (!myid) { nalu_hypre_printf("********************************************************\n" ); }
      /*do loop times*/
      for (i = 0; i < loop; i++)
      {

         loop_times[i] = 0.0;
         for (k = 0; k < loop2; k++)
         {


            nalu_hypre_MPI_Barrier(nalu_hypre_MPI_COMM_WORLD);


            start_time = nalu_hypre_MPI_Wtime();

#if time_gather

            info = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);

            nalu_hypre_MPI_Allgather(bcast_rows, 1, NALU_HYPRE_MPI_INT, info, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_COMM_WORLD);

#endif

            nalu_hypre_MatvecCommPkgCreate(parcsr_A);

            end_time = nalu_hypre_MPI_Wtime();


            end_time = end_time - start_time;

            nalu_hypre_MPI_Allreduce(&end_time, &total_time, 1,
                                NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_MAX, nalu_hypre_MPI_COMM_WORLD);

            loop_times[i] += total_time;


            if (  !((i + 1) == loop  &&  (k + 1) == loop2)) { nalu_hypre_MatvecCommPkgDestroy(nalu_hypre_ParCSRMatrixCommPkg(parcsr_A)); }

         }/* end of loop 2*/


      } /*end of loop*/

      /* calculate the avg and std. */
      if (loop > 1)
      {

         stats_mo(loop_times, loop, &T_avg, &T_std);
         if (!myid) { nalu_hypre_printf("Current CommPkgCreate:  AVG. wall clock time =  %f seconds\n", T_avg); }
         if (!myid) { nalu_hypre_printf("                        STD. for %d  runs     =  %f\n", loop - 1, T_std); }
         if (!myid) { nalu_hypre_printf("                        (Note: avg./std. timings exclude run 0.)\n"); }
         if (!myid) { nalu_hypre_printf("********************************************************\n" ); }
         for (i = 0; i < loop; i++)
         {
            if (!myid) { nalu_hypre_printf("      run %d  =  %f sec.\n", i, loop_times[i]); }
         }
         if (!myid) { nalu_hypre_printf("********************************************************\n" ); }

      }
      else
      {
         if (!myid) { nalu_hypre_printf("********************************************************\n" ); }
         if (!myid) { nalu_hypre_printf(" Current CommPkgCreate:\n"); }
         if (!myid) { nalu_hypre_printf("      run time =  %f sec.\n", loop_times[0]); }
         if (!myid) { nalu_hypre_printf("********************************************************\n" ); }
      }





      /*-----------------------------------------------------------
       * Verbose printing
       *-----------------------------------------------------------*/

      /*some verification*/


      if (verbose)
      {

         ierr = nalu_hypre_ParCSRMatrixGetLocalRange( parcsr_A,
                                                 &row_start, &row_end,
                                                 &col_start, &col_end );


         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(parcsr_A);

         nalu_hypre_printf("myid = %i, std - my local range: [%i, %i]\n", myid,
                      row_start, row_end);

         ierr = nalu_hypre_ParCSRMatrixGetLocalRange( parcsr_A,
                                                 &row_start, &row_end,
                                                 &col_start, &col_end );

         nalu_hypre_printf("myid = %d, std - num_recvs = %d\n", myid,
                      nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg)  );

#if mydebug
         for (i = 0; i < nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg); i++)
         {
            nalu_hypre_printf("myid = %d, std - recv proc = %d, vec_starts = [%d : %d]\n",
                         myid,  nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg)[i],
                         nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i],
                         nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)[i + 1] - 1);
         }
#endif
         nalu_hypre_printf("myid = %d, std - num_sends = %d\n", myid,
                      nalu_hypre_ParCSRCommPkgNumSends(comm_pkg));


#if mydebug
         for (i = 0; i < nalu_hypre_ParCSRCommPkgNumSends(comm_pkg) ; i++)
         {
            tmp_int =  nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i + 1] -
                       nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
            index = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg)[i];
            for (j = 0; j < tmp_int; j++)
            {
               nalu_hypre_printf("myid = %d, std - send proc = %d, send element = %d\n", myid,
                            nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg)[i],
                            nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg)[index + j]);
            }
         }
#endif
      }

      /*-----------------------------------------------------------
       * Verify correctness
       *-----------------------------------------------------------*/



      if (commpkg_flag == 3 )
      {
         global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(parcsr_A);
         row_starts = nalu_hypre_ParCSRMatrixRowStarts(parcsr_A);


         y = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_num_rows, row_starts);
         nalu_hypre_ParVectorInitialize(y);
         nalu_hypre_ParVectorSetConstantValues(y, 0.0);

         nalu_hypre_ParCSRMatrixMatvec (1.0, parcsr_A, x_new, 1.0, y);
      }
   }






   /*-----------------------------------------------------------
    *  Compare matvecs for both comm packages (3)
    *-----------------------------------------------------------*/

   if (commpkg_flag == 3 )
   {
      /*make sure that y and y_new are the same  - now y_new should=0*/
      nalu_hypre_ParVectorAxpy( -1.0, y, y_new );


      nalu_hypre_ParVectorSetRandomValues(y, 1);

      ans = nalu_hypre_ParVectorInnerProd( y, y_new );
      if (!myid)
      {

         if ( fabs(ans) > 1e-8 )
         {
            nalu_hypre_printf("!!!!! WARNING !!!!! should be zero if correct = %6.10f\n",
                         ans);
         }
         else
         {
            nalu_hypre_printf("Matvecs match ( should be zero = %6.10f )\n",
                         ans);
         }
      }


   }


   /*-----------------------------------------------------------
    *  Clean up
    *-----------------------------------------------------------*/


   nalu_hypre_ParCSRMatrixDestroy(parcsr_A); /*this calls the standard comm
                                          package destroy - but we'll destroy
                                          ours separately until it is
                                          incorporated */

   if (commpkg_flag == 3 )
   {

      nalu_hypre_ParVectorDestroy(x_new);
      nalu_hypre_ParVectorDestroy(y);
      nalu_hypre_ParVectorDestroy(y_new);
   }




   nalu_hypre_MPI_Finalize();

   return (ierr);


}





/*------------------------------------
 *    Calculate the average and STD
 *     throw away 1st timing
 *------------------------------------*/

void stats_mo(NALU_HYPRE_Real array[], NALU_HYPRE_Int n, NALU_HYPRE_Real *Tavg, NALU_HYPRE_Real *Tstd)
{

   NALU_HYPRE_Int i;
   NALU_HYPRE_Real atmp, tmp = 0.0;
   NALU_HYPRE_Real avg = 0.0, std;


   for (i = 1; i < n; i++)
   {
      atmp = array[i];
      avg += atmp;
      tmp += atmp * atmp;
   }

   n = n - 1;
   avg = avg / (NALU_HYPRE_Real) n;
   tmp = tmp / (NALU_HYPRE_Real) n;

   tmp = fabs(tmp - avg * avg);
   std = nalu_hypre_sqrt(tmp);

   *Tavg = avg;
   *Tstd = std;
}



/*These next two functions are from ij.c in linear_solvers/tests */


/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
myBuildParLaplacian27pt( NALU_HYPRE_Int                  argc,
                         char                *argv[],
                         NALU_HYPRE_Int                  arg_index,
                         NALU_HYPRE_ParCSRMatrix  *A_ptr, NALU_HYPRE_Int parmprint  )
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

   if (myid == 0 && parmprint)
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


/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/


NALU_HYPRE_Int
myBuildParLaplacian( NALU_HYPRE_Int                  argc,
                     char                *argv[],
                     NALU_HYPRE_Int                  arg_index,
                     NALU_HYPRE_ParCSRMatrix  *A_ptr, NALU_HYPRE_Int parmprint    )
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

   if (myid == 0 && parmprint)
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

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian(nalu_hypre_MPI_COMM_WORLD, nx, ny, nz,
                                              P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);


   *A_ptr = A;

   return (0);
}
