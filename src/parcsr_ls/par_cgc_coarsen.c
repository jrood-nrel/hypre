/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */


#include "_nalu_hypre_parcsr_ls.h"
#include "../NALU_HYPRE.h" /* BM Aug 15, 2006 */
#include "_nalu_hypre_IJ_mv.h"

#define C_PT 1
#define F_PT -1
#define Z_PT -2
#define SF_PT -3  /* special fine points */
#define UNDECIDED 0


/**************************************************************
 *
 *      CGC Coarsening routine
 *
 **************************************************************/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGCoarsenCGCb( nalu_hypre_ParCSRMatrix    *S,
                            nalu_hypre_ParCSRMatrix    *A,
                            NALU_HYPRE_Int                    measure_type,
                            NALU_HYPRE_Int                    coarsen_type,
                            NALU_HYPRE_Int                    cgc_its,
                            NALU_HYPRE_Int                    debug_flag,
                            nalu_hypre_IntArray             **CF_marker_ptr)
{
#ifdef NALU_HYPRE_MIXEDINT
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "CGC coarsening is not enabled in mixedint mode!");
   return nalu_hypre_error_flag;
#endif

   MPI_Comm         comm          = nalu_hypre_ParCSRMatrixComm(S);
   nalu_hypre_ParCSRCommPkg   *comm_pkg      = nalu_hypre_ParCSRMatrixCommPkg(S);
   nalu_hypre_ParCSRCommHandle *comm_handle;
   nalu_hypre_CSRMatrix *S_diag        = nalu_hypre_ParCSRMatrixDiag(S);
   nalu_hypre_CSRMatrix *S_offd        = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int             *S_i           = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int             *S_j           = nalu_hypre_CSRMatrixJ(S_diag);
   NALU_HYPRE_Int             *S_offd_i      = nalu_hypre_CSRMatrixI(S_offd);
   /*NALU_HYPRE_Int             *S_offd_j      = nalu_hypre_CSRMatrixJ(S_offd);*/
   NALU_HYPRE_Int              num_variables = nalu_hypre_CSRMatrixNumRows(S_diag);
   NALU_HYPRE_Int              num_cols_offd = nalu_hypre_CSRMatrixNumCols(S_offd);

   nalu_hypre_CSRMatrix *S_ext;
   NALU_HYPRE_Int             *S_ext_i;
   NALU_HYPRE_BigInt          *S_ext_j;

   nalu_hypre_CSRMatrix *ST;
   NALU_HYPRE_Int             *ST_i;
   NALU_HYPRE_Int             *ST_j;

   NALU_HYPRE_Int             *CF_marker;
   NALU_HYPRE_Int             *CF_marker_offd = NULL;
   NALU_HYPRE_Int              ci_tilde = -1;
   NALU_HYPRE_Int              ci_tilde_mark = -1;

   NALU_HYPRE_Int             *measure_array;
   NALU_HYPRE_Int             *measure_array_master;
   NALU_HYPRE_Int             *graph_array;
   NALU_HYPRE_Int             *int_buf_data = NULL;
   /*NALU_HYPRE_Int           *ci_array=NULL;*/

   NALU_HYPRE_Int              i, j, k, l, jS;
   NALU_HYPRE_Int              ji, jj, index;
   NALU_HYPRE_Int              set_empty = 1;
   NALU_HYPRE_Int              C_i_nonempty = 0;
   NALU_HYPRE_Int              num_nonzeros;
   NALU_HYPRE_Int              num_procs, my_id;
   NALU_HYPRE_Int              num_sends = 0;
   NALU_HYPRE_BigInt           first_col;
   NALU_HYPRE_Int              start;
   /*NALU_HYPRE_Int            col_0, col_n;*/

   nalu_hypre_LinkList   LoL_head;
   nalu_hypre_LinkList   LoL_tail;

   NALU_HYPRE_Int             *lists, *where;
   NALU_HYPRE_Int              measure, new_meas;
   NALU_HYPRE_Int              num_left;
   NALU_HYPRE_Int              nabor, nabor_two;

   NALU_HYPRE_Int              use_commpkg_A = 0;
   NALU_HYPRE_Real             wall_time;

   NALU_HYPRE_Int              measure_max; /* BM Aug 30, 2006: maximal measure, needed for CGC */

   if (coarsen_type < 0) { coarsen_type = -coarsen_type; }

   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);
   where = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   NALU_HYPRE_Int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a CSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > nalu_hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < nalu_hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      use_commpkg_A = 1;
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

   /*if (num_cols_offd) S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);*/

   jS = S_i[num_variables];

   ST = nalu_hypre_CSRMatrixCreate(num_variables, num_variables, jS);
   ST_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_variables + 1, NALU_HYPRE_MEMORY_HOST);
   ST_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, jS, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CSRMatrixI(ST) = ST_i;
   nalu_hypre_CSRMatrixJ(ST) = ST_j;
   nalu_hypre_CSRMatrixMemoryLocation(ST) = NALU_HYPRE_MEMORY_HOST;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i = 0; i <= num_variables; i++)
   {
      ST_i[i] = 0;
   }

   for (i = 0; i < jS; i++)
   {
      ST_i[S_j[i] + 1]++;
   }
   for (i = 0; i < num_variables; i++)
   {
      ST_i[i + 1] += ST_i[i];
   }
   for (i = 0; i < num_variables; i++)
   {
      for (j = S_i[i]; j < S_i[i + 1]; j++)
      {
         index = S_j[j];
         ST_j[ST_i[index]] = i;
         ST_i[index]++;
      }
   }
   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i - 1];
   }
   ST_i[0] = 0;

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are given by the row sums of ST.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    * correct actual measures through adding influences from
    * neighbor processors
    *----------------------------------------------------------*/

   measure_array_master = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);
   measure_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_variables; i++)
   {
      measure_array_master[i] = ST_i[i + 1] - ST_i[i];
   }

   if ((measure_type || (coarsen_type != 1 && coarsen_type != 11))
       && num_procs > 1)
   {
      if (use_commpkg_A)
      {
         S_ext      = nalu_hypre_ParCSRMatrixExtractBExt(S, A, 0);
      }
      else
      {
         S_ext      = nalu_hypre_ParCSRMatrixExtractBExt(S, S, 0);
      }
      S_ext_i    = nalu_hypre_CSRMatrixI(S_ext);
      S_ext_j    = nalu_hypre_CSRMatrixBigJ(S_ext);
      num_nonzeros = S_ext_i[num_cols_offd];
      first_col = nalu_hypre_ParCSRMatrixFirstColDiag(S);
      /*col_0 = first_col-1;
        col_n = col_0+num_variables;*/
      if (measure_type)
      {
         for (i = 0; i < num_nonzeros; i++)
         {
            index = (NALU_HYPRE_Int)(S_ext_j[i] - first_col);
            if (index > -1 && index < num_variables)
            {
               measure_array_master[index]++;
            }
         }
      }
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   /* first coarsening phase */

   /*************************************************************
    *
    *   Initialize the lists
    *
    *************************************************************/

   /* Allocate CF_marker if not done before */
   if (*CF_marker_ptr == NULL)
   {
      *CF_marker_ptr = nalu_hypre_IntArrayCreate(num_variables);
      nalu_hypre_IntArrayInitialize(*CF_marker_ptr);
   }
   CF_marker = nalu_hypre_IntArrayData(*CF_marker_ptr);

   num_left = 0;
   for (j = 0; j < num_variables; j++)
   {
      if ((S_i[j + 1] - S_i[j]) == 0 &&
          (S_offd_i[j + 1] - S_offd_i[j]) == 0)
      {
         CF_marker[j] = SF_PT;
         measure_array_master[j] = 0;
      }
      else
      {
         CF_marker[j] = UNDECIDED;
         /*        num_left++; */ /* BM May 19, 2006: see below*/
      }
   }

   if (coarsen_type == 22)
   {
      /* BM Sep 8, 2006: allow_emptygrids only if the following holds for all points j:
         (a) the point has no strong connections at all, OR
         (b) the point has a strong connection across a boundary */
      for (j = 0; j < num_variables; j++)
         if (S_i[j + 1] > S_i[j] && S_offd_i[j + 1] == S_offd_i[j]) {coarsen_type = 21; break;}
   }

   for (l = 1; l <= cgc_its; l++)
   {
      LoL_head = NULL;
      LoL_tail = NULL;
      num_left = 0;  /* compute num_left before each RS coarsening loop */
      nalu_hypre_TMemcpy(measure_array, measure_array_master, NALU_HYPRE_Int, num_variables, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_HOST);
      memset (lists, 0, sizeof(NALU_HYPRE_Int)*num_variables);
      memset (where, 0, sizeof(NALU_HYPRE_Int)*num_variables);

      for (j = 0; j < num_variables; j++)
      {
         measure = measure_array[j];
         if (CF_marker[j] != SF_PT)
         {
            if (measure > 0)
            {
               nalu_hypre_enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
               num_left++; /* compute num_left before each RS coarsening loop */
            }
            else if (CF_marker[j] == 0) /* increase weight of strongly coupled neighbors only
                                           if j is not conained in a previously constructed coarse grid.
                                           Reason: these neighbors should start with the same initial weight
                                           in each CGC iteration.                    BM Aug 30, 2006 */

            {
               if (measure < 0) { nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "negative measure!\n"); }
               /* CF_marker[j] = f_pnt; */
               for (k = S_i[j]; k < S_i[j + 1]; k++)
               {
                  nabor = S_j[k];
                  /* if (CF_marker[nabor] != SF_PT)  */
                  if (CF_marker[nabor] == 0)  /* BM Aug 30, 2006: don't alter weights of points
                                                 contained in other candidate coarse grids */
                  {
                     if (nabor < j)
                     {
                        new_meas = measure_array[nabor];
                        if (new_meas > 0)
                           nalu_hypre_remove_point(&LoL_head, &LoL_tail, new_meas,
                                              nabor, lists, where);
                        else { num_left++; } /* BM Aug 29, 2006 */

                        new_meas = ++(measure_array[nabor]);
                        nalu_hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                             nabor, lists, where);
                     }
                     else
                     {
                        new_meas = ++(measure_array[nabor]);
                     }
                  }
               }
               /* --num_left; */ /* BM May 19, 2006 */
            }
         }
      }

      /* BM Aug 30, 2006: first iteration: determine maximal weight */
      if (num_left && l == 1) { measure_max = measure_array[LoL_head->head]; }
      /* BM Aug 30, 2006: break CGC iteration if no suitable
         starting point is available any more */
      if (!num_left || measure_array[LoL_head->head] < measure_max)
      {
         while (LoL_head)
         {
            nalu_hypre_LinkList list_ptr = LoL_head;
            LoL_head = LoL_head->next_elt;
            nalu_hypre_dispose_elt (list_ptr);
         }
         break;
      }

      /****************************************************************
       *
       *  Main loop of Ruge-Stueben first coloring pass.
       *
       *  WHILE there are still points to classify DO:
       *        1) find first point, i,  on list with max_measure
       *           make i a C-point, remove it from the lists
       *        2) For each point, j,  in S_i^T,
       *           a) Set j to be an F-point
       *           b) For each point, k, in S_j
       *                  move k to the list in LoL with measure one
       *                  greater than it occupies (creating new LoL
       *                  entry if necessary)
       *        3) For each point, j,  in S_i,
       *                  move j to the list in LoL with measure one
       *                  smaller than it occupies (creating new LoL
       *                  entry if necessary)
       *
       ****************************************************************/

      while (num_left > 0)
      {
         index = LoL_head -> head;
         /*         index = LoL_head -> tail;  */

         /*        CF_marker[index] = C_PT; */
         CF_marker[index] = l;  /* BM Aug 18, 2006 */
         measure = measure_array[index];
         measure_array[index] = 0;
         measure_array_master[index] = 0; /* BM May 19: for CGC */
         --num_left;

         nalu_hypre_remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);

         for (j = ST_i[index]; j < ST_i[index + 1]; j++)
         {
            nabor = ST_j[j];
            /*          if (CF_marker[nabor] == UNDECIDED) */
            if (measure_array[nabor] > 0) /* undecided point */
            {
               /* CF_marker[nabor] = F_PT; */ /* BM Aug 18, 2006 */
               measure = measure_array[nabor];
               measure_array[nabor] = 0;

               nalu_hypre_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
               --num_left;

               for (k = S_i[nabor]; k < S_i[nabor + 1]; k++)
               {
                  nabor_two = S_j[k];
                  /* if (CF_marker[nabor_two] == UNDECIDED) */
                  if (measure_array[nabor_two] > 0) /* undecided point */
                  {
                     measure = measure_array[nabor_two];
                     nalu_hypre_remove_point(&LoL_head, &LoL_tail, measure,
                                        nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);

                     nalu_hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                          nabor_two, lists, where);
                  }
               }
            }
         }
         for (j = S_i[index]; j < S_i[index + 1]; j++)
         {
            nabor = S_j[j];
            /*          if (CF_marker[nabor] == UNDECIDED) */
            if (measure_array[nabor] > 0) /* undecided point */
            {
               measure = measure_array[nabor];

               nalu_hypre_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

               measure_array[nabor] = --measure;

               if (measure > 0)
                  nalu_hypre_enter_on_lists(&LoL_head, &LoL_tail, measure, nabor,
                                       lists, where);
               else
               {
                  /* CF_marker[nabor] = F_PT; */ /* BM Aug 18, 2006 */
                  --num_left;

                  for (k = S_i[nabor]; k < S_i[nabor + 1]; k++)
                  {
                     nabor_two = S_j[k];
                     /* if (CF_marker[nabor_two] == UNDECIDED) */
                     if (measure_array[nabor_two] > 0)
                     {
                        new_meas = measure_array[nabor_two];
                        nalu_hypre_remove_point(&LoL_head, &LoL_tail, new_meas,
                                           nabor_two, lists, where);

                        new_meas = ++(measure_array[nabor_two]);

                        nalu_hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                             nabor_two, lists, where);
                     }
                  }
               }
            }
         }
      }
      if (LoL_head) { nalu_hypre_error_w_msg (NALU_HYPRE_ERROR_GENERIC, "Linked list not empty!\n"); } /*head: %d\n",LoL_head->head);*/
   }
   l--; /* BM Aug 15, 2006 */

   nalu_hypre_TFree(measure_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(measure_array_master, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CSRMatrixDestroy(ST);

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d    Coarsen 1st pass = %f\n",
                   my_id, wall_time);
   }

   nalu_hypre_TFree(lists, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(where, NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }
      nalu_hypre_BoomerAMGCoarsenCGC (S, l, coarsen_type, CF_marker);

      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         nalu_hypre_printf("Proc = %d    Coarsen CGC = %f\n",
                      my_id, wall_time);
      }
   }
   else
   {
      /* the first candiate coarse grid is the coarse grid */
      for (j = 0; j < num_variables; j++)
      {
         if (CF_marker[j] == 1) { CF_marker[j] = C_PT; }
         else { CF_marker[j] = F_PT; }
      }
   }

   /* BM May 19, 2006:
      Set all undecided points to be fine grid points. */
   for (j = 0; j < num_variables; j++)
      if (!CF_marker[j]) { CF_marker[j] = F_PT; }

   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   for (i = 0; i < num_variables; i++)
   {
      if (ci_tilde_mark != i) { ci_tilde = -1; }
      if (CF_marker[i] == -1)
      {
         for (ji = S_i[i]; ji < S_i[i + 1]; ji++)
         {
            j = S_j[ji];
            if (CF_marker[j] > 0)
            {
               graph_array[j] = i;
            }
         }
         for (ji = S_i[i]; ji < S_i[i + 1]; ji++)
         {
            j = S_j[ji];
            if (CF_marker[j] == -1)
            {
               set_empty = 1;
               for (jj = S_i[j]; jj < S_i[j + 1]; jj++)
               {
                  index = S_j[jj];
                  if (graph_array[index] == i)
                  {
                     set_empty = 0;
                     break;
                  }
               }
               if (set_empty)
               {
                  if (C_i_nonempty)
                  {
                     CF_marker[i] = 1;
                     if (ci_tilde > -1)
                     {
                        CF_marker[ci_tilde] = -1;
                        ci_tilde = -1;
                     }
                     C_i_nonempty = 0;
                     break;
                  }
                  else
                  {
                     ci_tilde = j;
                     ci_tilde_mark = i;
                     CF_marker[j] = 1;
                     C_i_nonempty = 1;
                     i--;
                     break;
                  }
               }
            }
         }
      }
   }

   if (debug_flag == 3 && coarsen_type != 2)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d    Coarsen 2nd pass = %f\n",
                   my_id, wall_time);
   }

   /* third pass, check boundary fine points for coarse neighbors */

   /*------------------------------------------------
    * Exchange boundary data for CF_marker
    *------------------------------------------------*/

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
   int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                            num_sends), NALU_HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         int_buf_data[index++]
            = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   if (num_procs > 1)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                 CF_marker_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }
   nalu_hypre_AmgCGCBoundaryFix (S, CF_marker, CF_marker_offd);
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d    CGC boundary fix = %f\n",
                   my_id, wall_time);
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   /*if (coarsen_type != 1)
     { */
   if (CF_marker_offd) { nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST); }  /* BM Aug 21, 2006 */
   if (int_buf_data) { nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST); } /* BM Aug 21, 2006 */
   /*if (ci_array) nalu_hypre_TFree(ci_array);*/ /* BM Aug 21, 2006 */
   /*} */
   nalu_hypre_TFree(graph_array, NALU_HYPRE_MEMORY_HOST);
   if ((measure_type || (coarsen_type != 1 && coarsen_type != 11))
       && num_procs > 1)
   {
      nalu_hypre_CSRMatrixDestroy(S_ext);
   }

   return nalu_hypre_error_flag;
}

/* begin Bram added */

NALU_HYPRE_Int nalu_hypre_BoomerAMGCoarsenCGC (nalu_hypre_ParCSRMatrix    *S, NALU_HYPRE_Int numberofgrids,
                                     NALU_HYPRE_Int coarsen_type, NALU_HYPRE_Int *CF_marker)
/* CGC algorithm
 * ====================================================================================================
 * coupling : the strong couplings
 * numberofgrids : the number of grids
 * coarsen_type : the coarsening type
 * gridpartition : the grid partition
 * =====================================================================================================*/
{
   NALU_HYPRE_Int j,/*p,*/mpisize, mpirank,/*rstart,rend,*/choice, *coarse;
   NALU_HYPRE_Int *vertexrange = NULL;
   NALU_HYPRE_Int *vertexrange_all = NULL;
   NALU_HYPRE_Int *CF_marker_offd = NULL;
   NALU_HYPRE_Int num_variables = nalu_hypre_CSRMatrixNumRows (nalu_hypre_ParCSRMatrixDiag(S));
   /*   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols (nalu_hypre_ParCSRMatrixOffd (S)); */
   /*   NALU_HYPRE_Int *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd (S); */

   /*   NALU_HYPRE_Real wall_time; */

   NALU_HYPRE_IJMatrix ijG;
   nalu_hypre_ParCSRMatrix *G;
   nalu_hypre_CSRMatrix *Gseq;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(S);

   nalu_hypre_MPI_Comm_size (comm, &mpisize);
   nalu_hypre_MPI_Comm_rank (comm, &mpirank);

#if 0
   if (!mpirank)
   {
      wall_time = time_getWallclockSeconds();
      nalu_hypre_printf ("Starting CGC preparation\n");
   }
#endif
   nalu_hypre_AmgCGCPrepare (S, numberofgrids, CF_marker, &CF_marker_offd, coarsen_type, &vertexrange);
#if 0 /* debugging */
   if (!mpirank)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf ("Finished CGC preparation, wall_time = %f s\n", wall_time);
      wall_time = time_getWallclockSeconds();
      nalu_hypre_printf ("Starting CGC matrix assembly\n");
   }
#endif
   nalu_hypre_AmgCGCGraphAssemble (S, vertexrange, CF_marker, CF_marker_offd, coarsen_type, &ijG);
#if 0
   NALU_HYPRE_IJMatrixPrint (ijG, "graph.txt");
#endif
   NALU_HYPRE_IJMatrixGetObject (ijG, (void**)&G);
#if 0 /* debugging */
   if (!mpirank)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf ("Finished CGC matrix assembly, wall_time = %f s\n", wall_time);
      wall_time = time_getWallclockSeconds();
      nalu_hypre_printf ("Starting CGC matrix communication\n");
   }
#endif
   {
      /* classical CGC does not really make sense with an assumed partition, but
         anyway, here it is: */
      NALU_HYPRE_Int nlocal = vertexrange[1] - vertexrange[0];
      vertexrange_all = nalu_hypre_CTAlloc(NALU_HYPRE_Int, mpisize + 1, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_MPI_Allgather (&nlocal, 1, NALU_HYPRE_MPI_INT, vertexrange_all + 1, 1, NALU_HYPRE_MPI_INT, comm);
      vertexrange_all[0] = 0;
      for (j = 2; j <= mpisize; j++) { vertexrange_all[j] += vertexrange_all[j - 1]; }
   }
   Gseq = nalu_hypre_ParCSRMatrixToCSRMatrixAll (G);
#if 0 /* debugging */
   if (!mpirank)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf ("Finished CGC matrix communication, wall_time = %f s\n", wall_time);
   }
#endif

   if (Gseq)   /* BM Aug 31, 2006: Gseq==NULL if G has no local rows */
   {
#if 0 /* debugging */
      if (!mpirank)
      {
         wall_time = time_getWallclockSeconds();
         nalu_hypre_printf ("Starting CGC election\n");
      }
#endif
      nalu_hypre_AmgCGCChoose (Gseq, vertexrange_all, mpisize, &coarse);
#if 0 /* debugging */
      if (!mpirank)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         nalu_hypre_printf ("Finished CGC election, wall_time = %f s\n", wall_time);
      }
#endif

#if 0 /* debugging */
      if (!mpirank)
      {
         for (j = 0; j < mpisize; j++)
         {
            nalu_hypre_printf ("Processor %d, choice = %d of range %d - %d\n", j, coarse[j], vertexrange_all[j] + 1,
                          vertexrange_all[j + 1]);
         }
      }
      fflush(stdout);
#endif
#if 0 /* debugging */
      if (!mpirank)
      {
         wall_time = time_getWallclockSeconds();
         nalu_hypre_printf ("Starting CGC CF assignment\n");
      }
#endif
      choice = coarse[mpirank];
      for (j = 0; j < num_variables; j++)
      {
         if (CF_marker[j] == choice)
         {
            CF_marker[j] = C_PT;
         }
         else
         {
            CF_marker[j] = F_PT;
         }
      }

      nalu_hypre_CSRMatrixDestroy (Gseq);
      nalu_hypre_TFree(coarse, NALU_HYPRE_MEMORY_HOST);
   }
   else
      for (j = 0; j < num_variables; j++) { CF_marker[j] = F_PT; }
#if 0
   if (!mpirank)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf ("Finished CGC CF assignment, wall_time = %f s\n", wall_time);
   }
#endif

#if 0 /* debugging */
   if (!mpirank)
   {
      wall_time = time_getWallclockSeconds();
      nalu_hypre_printf ("Starting CGC cleanup\n");
   }
#endif
   NALU_HYPRE_IJMatrixDestroy (ijG);
   nalu_hypre_TFree(vertexrange, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vertexrange_all, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
#if 0
   if (!mpirank)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf ("Finished CGC cleanup, wall_time = %f s\n", wall_time);
   }
#endif
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_AmgCGCPrepare (nalu_hypre_ParCSRMatrix *S, NALU_HYPRE_Int nlocal, NALU_HYPRE_Int *CF_marker,
                               NALU_HYPRE_Int **CF_marker_offd, NALU_HYPRE_Int coarsen_type, NALU_HYPRE_Int **vrange)
/* assemble a graph representing the connections between the grids
 * ================================================================================================
 * S : the strength matrix
 * nlocal : the number of locally created coarse grids
 * CF_marker, CF_marker_offd : the coare/fine markers
 * coarsen_type : the coarsening type
 * vrange : the ranges of the vertices representing coarse grids
 * ================================================================================================*/
{
   NALU_HYPRE_Int mpisize, mpirank;
   NALU_HYPRE_Int num_sends;
   NALU_HYPRE_Int *vertexrange = NULL;
   NALU_HYPRE_Int vstart/*,vend*/;
   NALU_HYPRE_Int *int_buf_data;
   NALU_HYPRE_Int start;
   NALU_HYPRE_Int i, ii, j;
   NALU_HYPRE_Int num_variables = nalu_hypre_CSRMatrixNumRows (nalu_hypre_ParCSRMatrixDiag(S));
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols (nalu_hypre_ParCSRMatrixOffd (S));

   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(S);
   /*   nalu_hypre_MPI_Status status; */

   nalu_hypre_ParCSRCommPkg    *comm_pkg    = nalu_hypre_ParCSRMatrixCommPkg (S);
   nalu_hypre_ParCSRCommHandle *comm_handle;


   nalu_hypre_MPI_Comm_size (comm, &mpisize);
   nalu_hypre_MPI_Comm_rank (comm, &mpirank);

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate (S);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg (S);
   }
   num_sends = nalu_hypre_ParCSRCommPkgNumSends (comm_pkg);

   if (coarsen_type % 2 == 0) { nlocal++; } /* even coarsen_type means allow_emptygrids */
   {
      NALU_HYPRE_Int scan_recv;

      vertexrange = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 2, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_MPI_Scan(&nlocal, &scan_recv, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm);
      /* first point in my range */
      vertexrange[0] = scan_recv - nlocal;
      /* first point in next proc's range */
      vertexrange[1] = scan_recv;
      vstart = vertexrange[0];
      /*vend   = vertexrange[1];*/
   }

   /* Note: vstart uses 0-based indexing, while CF_marker uses 1-based indexing */
   if (coarsen_type % 2 == 1)   /* see above */
   {
      for (i = 0; i < num_variables; i++)
         if (CF_marker[i] > 0)
         {
            CF_marker[i] += vstart;
         }
   }
   else
   {
      /*      nalu_hypre_printf ("processor %d: empty grid allowed\n",mpirank);  */
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] > 0)
         {
            CF_marker[i] += vstart + 1;
         } /* add one because vertexrange[mpirank]+1 denotes the empty grid.
                                       Hence, vertexrange[mpirank]+2 is the first coarse grid denoted in
                                       global indices, ... */
      }
   }

   /* exchange data */
   *CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd, NALU_HYPRE_MEMORY_HOST);
   int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart (comm_pkg, num_sends),
                                NALU_HYPRE_MEMORY_HOST);

   for (i = 0, ii = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart (comm_pkg, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart (comm_pkg, i + 1); j++)
      {
         int_buf_data [ii++] = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   if (mpisize > 1)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate (11, comm_pkg, int_buf_data, *CF_marker_offd);
      nalu_hypre_ParCSRCommHandleDestroy (comm_handle);
   }
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   *vrange = vertexrange;
   return nalu_hypre_error_flag;
}

#define tag_pointrange 301
#define tag_vertexrange 302

NALU_HYPRE_Int nalu_hypre_AmgCGCGraphAssemble (nalu_hypre_ParCSRMatrix *S, NALU_HYPRE_Int *vertexrange,
                                     NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *CF_marker_offd, NALU_HYPRE_Int coarsen_type,
                                     NALU_HYPRE_IJMatrix *ijG)
/* assemble a graph representing the connections between the grids
 * ================================================================================================
 * S : the strength matrix
 * vertexrange : the parallel layout of the candidate coarse grid vertices
 * CF_marker, CF_marker_offd : the coarse/fine markers
 * coarsen_type : the coarsening type
 * ijG : the created graph
 * ================================================================================================*/
{
   NALU_HYPRE_Int i,/* ii,ip,*/ j, jj, m, n, p;
   NALU_HYPRE_Int mpisize, mpirank;

   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(S);
   /*   nalu_hypre_MPI_Status status; */

   NALU_HYPRE_IJMatrix ijmatrix;
   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag (S);
   nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd (S);
   /*   NALU_HYPRE_Int *S_i = nalu_hypre_CSRMatrixI(S_diag); */
   /*   NALU_HYPRE_Int *S_j = nalu_hypre_CSRMatrixJ(S_diag); */
   NALU_HYPRE_Int *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int *S_offd_j = NULL;
   NALU_HYPRE_Int num_variables = nalu_hypre_CSRMatrixNumRows (S_diag);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols (S_offd);
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd (S);
   NALU_HYPRE_BigInt *pointrange;
   NALU_HYPRE_Int *pointrange_nonlocal, *pointrange_strong = NULL;
   NALU_HYPRE_Int vertexrange_start, vertexrange_end;
   NALU_HYPRE_Int *vertexrange_strong = NULL;
   NALU_HYPRE_Int *vertexrange_nonlocal;
   NALU_HYPRE_Int num_recvs, num_recvs_strong;
   NALU_HYPRE_Int *recv_procs, *recv_procs_strong = NULL;
   NALU_HYPRE_Int /* *zeros,*rownz,*/*rownz_diag, *rownz_offd;
   NALU_HYPRE_Int nz;
   NALU_HYPRE_Int nlocal;
   //NALU_HYPRE_Int one=1;

   nalu_hypre_ParCSRCommPkg    *comm_pkg    = nalu_hypre_ParCSRMatrixCommPkg (S);

   nalu_hypre_MPI_Comm_size (comm, &mpisize);
   nalu_hypre_MPI_Comm_rank (comm, &mpirank);

   /* determine neighbor processors */
   num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs (comm_pkg);
   recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs (comm_pkg);
   pointrange = nalu_hypre_ParCSRMatrixRowStarts (S);
   pointrange_nonlocal = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2 * num_recvs, NALU_HYPRE_MEMORY_HOST);
   vertexrange_nonlocal = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2 * num_recvs, NALU_HYPRE_MEMORY_HOST);
   {
      NALU_HYPRE_Int num_sends  =  nalu_hypre_ParCSRCommPkgNumSends (comm_pkg);
      NALU_HYPRE_Int *send_procs =  nalu_hypre_ParCSRCommPkgSendProcs (comm_pkg);
      NALU_HYPRE_Int *int_buf_data   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4 * num_sends, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Int *int_buf_data2  = int_buf_data + 2 * num_sends;
      nalu_hypre_MPI_Request *sendrequest, *recvrequest;
      NALU_HYPRE_Int pointrange_start, pointrange_end;

      nlocal = vertexrange[1] - vertexrange[0];
      pointrange_start = pointrange[0];
      pointrange_end   = pointrange[1];
      vertexrange_start = vertexrange[0];
      vertexrange_end   = vertexrange[1];
      sendrequest = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, 2 * (num_sends + num_recvs), NALU_HYPRE_MEMORY_HOST);
      recvrequest = sendrequest + 2 * num_sends;

      for (i = 0; i < num_recvs; i++)
      {
         nalu_hypre_MPI_Irecv (pointrange_nonlocal + 2 * i, 2, NALU_HYPRE_MPI_INT, recv_procs[i], tag_pointrange, comm,
                          &recvrequest[2 * i]);
         nalu_hypre_MPI_Irecv (vertexrange_nonlocal + 2 * i, 2, NALU_HYPRE_MPI_INT, recv_procs[i], tag_vertexrange,
                          comm,
                          &recvrequest[2 * i + 1]);
      }
      for (i = 0; i < num_sends; i++)
      {
         int_buf_data[2 * i] = pointrange_start;
         int_buf_data[2 * i + 1] = pointrange_end;
         int_buf_data2[2 * i] = vertexrange_start;
         int_buf_data2[2 * i + 1] = vertexrange_end;
         nalu_hypre_MPI_Isend (int_buf_data + 2 * i, 2, NALU_HYPRE_MPI_INT, send_procs[i], tag_pointrange, comm,
                          &sendrequest[2 * i]);
         nalu_hypre_MPI_Isend (int_buf_data2 + 2 * i, 2, NALU_HYPRE_MPI_INT, send_procs[i], tag_vertexrange, comm,
                          &sendrequest[2 * i + 1]);
      }
      nalu_hypre_MPI_Waitall (2 * (num_sends + num_recvs), sendrequest, nalu_hypre_MPI_STATUSES_IGNORE);
      nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sendrequest, NALU_HYPRE_MEMORY_HOST);
   }
   /* now we have the array recv_procs. However, it may contain too many entries as it is
      inherited from A. We now have to determine the subset which contains only the
      strongly connected neighbors */
   if (num_cols_offd)
   {
      S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

      recv_procs_strong = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs, NALU_HYPRE_MEMORY_HOST);
      memset (recv_procs_strong, 0, num_recvs * sizeof(NALU_HYPRE_Int));
      /* don't forget to shorten the pointrange and vertexrange arrays accordingly */
      pointrange_strong = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 2 * num_recvs, NALU_HYPRE_MEMORY_HOST);
      memset (pointrange_strong, 0, 2 * num_recvs * sizeof(NALU_HYPRE_Int));
      vertexrange_strong = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 2 * num_recvs, NALU_HYPRE_MEMORY_HOST);
      memset (vertexrange_strong, 0, 2 * num_recvs * sizeof(NALU_HYPRE_Int));

      for (i = 0; i < num_variables; i++)
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            jj = col_map_offd[S_offd_j[j]];
            for (p = 0; p < num_recvs; p++) /* S_offd_j is NOT sorted! */
               if (jj >= pointrange_nonlocal[2 * p] && jj < pointrange_nonlocal[2 * p + 1]) { break; }
#if 0
            nalu_hypre_printf ("Processor %d, remote point %d on processor %d\n", mpirank, jj, recv_procs[p]);
#endif
            recv_procs_strong [p] = 1;
         }

      for (p = 0, num_recvs_strong = 0; p < num_recvs; p++)
      {
         if (recv_procs_strong[p])
         {
            recv_procs_strong[num_recvs_strong] = recv_procs[p];
            pointrange_strong[2 * num_recvs_strong] = pointrange_nonlocal[2 * p];
            pointrange_strong[2 * num_recvs_strong + 1] = pointrange_nonlocal[2 * p + 1];
            vertexrange_strong[2 * num_recvs_strong] = vertexrange_nonlocal[2 * p];
            vertexrange_strong[2 * num_recvs_strong + 1] = vertexrange_nonlocal[2 * p + 1];
            num_recvs_strong++;
         }
      }
   }
   else { num_recvs_strong = 0; }

   nalu_hypre_TFree(pointrange_nonlocal, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vertexrange_nonlocal, NALU_HYPRE_MEMORY_HOST);

   rownz_diag = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 2 * nlocal, NALU_HYPRE_MEMORY_HOST);
   rownz_offd = rownz_diag + nlocal;
   for (p = 0, nz = 0; p < num_recvs_strong; p++)
   {
      nz += vertexrange_strong[2 * p + 1] - vertexrange_strong[2 * p];
   }
   for (m = 0; m < nlocal; m++)
   {
      rownz_diag[m] = nlocal - 1;
      rownz_offd[m] = nz;
   }

   NALU_HYPRE_IJMatrixCreate(comm, vertexrange_start, vertexrange_end - 1, vertexrange_start,
                        vertexrange_end - 1, &ijmatrix);
   NALU_HYPRE_IJMatrixSetObjectType(ijmatrix, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixSetDiagOffdSizes (ijmatrix, rownz_diag, rownz_offd);
   NALU_HYPRE_IJMatrixInitialize(ijmatrix);
   nalu_hypre_TFree(rownz_diag, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_IJMatrixMemoryLocation(ijmatrix);
   NALU_HYPRE_BigInt *big_m_n = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, 2, memory_location);
   NALU_HYPRE_Real *weight = nalu_hypre_TAlloc(NALU_HYPRE_Real, 1, memory_location);

   /* initialize graph */
   weight[0] = -1;
   for (m = vertexrange_start; m < vertexrange_end; m++)
   {
      big_m_n[0] = (NALU_HYPRE_BigInt) m;
      for (p = 0; p < num_recvs_strong; p++)
      {
         for (n = vertexrange_strong[2 * p]; n < vertexrange_strong[2 * p + 1]; n++)
         {
            big_m_n[1] = (NALU_HYPRE_BigInt) n;
            NALU_HYPRE_IJMatrixAddToValues (ijmatrix, 1, NULL, &big_m_n[0], &big_m_n[1], &weight[0]);
            /*#if 0
              if (ierr) nalu_hypre_printf ("Processor %d: error %d while initializing graphs at (%d, %d)\n",mpirank,ierr,m,n);
            #endif*/
         }
      }
   }

   /* weight graph */
   for (i = 0; i < num_variables; i++)
   {

      for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
      {
         jj = S_offd_j[j]; /* jj is not a global index!!! */
         /* determine processor */
         for (p = 0; p < num_recvs_strong; p++)
            if (col_map_offd[jj] >= pointrange_strong[2 * p] &&
                col_map_offd[jj] < pointrange_strong[2 * p + 1]) { break; }
         /*ip=recv_procs_strong[p];*/
         /* loop over all coarse grids constructed on this processor domain */
         for (m = vertexrange_start; m < vertexrange_end; m++)
         {
            big_m_n[0] = (NALU_HYPRE_BigInt) m;
            /* loop over all coarse grids constructed on neighbor processor domain */
            for (n = vertexrange_strong[2 * p]; n < vertexrange_strong[2 * p + 1]; n++)
            {
               big_m_n[1] = (NALU_HYPRE_BigInt) n;
               /* coarse grid counting inside gridpartition->local/gridpartition->nonlocal starts with one
                  while counting inside range starts with zero */
               if (CF_marker[i] - 1 == m && CF_marker_offd[jj] - 1 == n)
                  /* C-C-coupling */
               {
                  weight[0] = -1;
               }
               else if ( (CF_marker[i] - 1 == m && (CF_marker_offd[jj] == 0 || CF_marker_offd[jj] - 1 != n) )
                         || ( (CF_marker[i] == 0 || CF_marker[i] - 1 != m) && CF_marker_offd[jj] - 1 == n ) )
                  /* C-F-coupling */
               {
                  weight[0] = 0;
               }
               else { weight[0] = -8; } /* F-F-coupling */
               NALU_HYPRE_IJMatrixAddToValues (ijmatrix, 1, NULL, &big_m_n[0], &big_m_n[1], &weight[0]);
               /*#if 0
                 if (ierr) nalu_hypre_printf ("Processor %d: error %d while adding %lf to entry (%d, %d)\n",mpirank,ierr,weight,m,n);
               #endif*/
            }
         }
      }
   }

   /* assemble */
   NALU_HYPRE_IJMatrixAssemble (ijmatrix);
   /*if (num_recvs_strong) {*/
   nalu_hypre_TFree(recv_procs_strong, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(pointrange_strong, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vertexrange_strong, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(big_m_n, memory_location);
   nalu_hypre_TFree(weight, memory_location);

   /*} */

   *ijG = ijmatrix;
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_AmgCGCChoose (nalu_hypre_CSRMatrix *G, NALU_HYPRE_Int *vertexrange, NALU_HYPRE_Int mpisize,
                              NALU_HYPRE_Int **coarse)
/* chooses one grid for every processor
 * ============================================================
 * G : the connectivity graph
 * map : the parallel layout
 * mpisize : number of procs
 * coarse : the chosen coarse grids
 * ===========================================================*/
{
   NALU_HYPRE_Int i, j, jj, p, choice, *processor;
   NALU_HYPRE_Int measure, new_measure;

   /*   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(G); */

   /*   nalu_hypre_ParCSRCommPkg    *comm_pkg    = nalu_hypre_ParCSRMatrixCommPkg (G); */
   /*   nalu_hypre_ParCSRCommHandle *comm_handle; */

   NALU_HYPRE_Real *G_data = nalu_hypre_CSRMatrixData (G);
   NALU_HYPRE_Real max;
   NALU_HYPRE_Int *G_i = nalu_hypre_CSRMatrixI(G);
   NALU_HYPRE_Int *G_j = nalu_hypre_CSRMatrixJ(G);
   nalu_hypre_CSRMatrix *H, *HT;
   NALU_HYPRE_Int *H_i, *H_j, *HT_i, *HT_j;
   NALU_HYPRE_Int jG, jH;
   NALU_HYPRE_Int num_vertices = nalu_hypre_CSRMatrixNumRows (G);
   NALU_HYPRE_Int *measure_array;
   NALU_HYPRE_Int *lists, *where;

   nalu_hypre_LinkList LoL_head = NULL;
   nalu_hypre_LinkList LoL_tail = NULL;

   processor = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_vertices, NALU_HYPRE_MEMORY_HOST);
   *coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int, mpisize, NALU_HYPRE_MEMORY_HOST);
   memset (*coarse, 0, sizeof(NALU_HYPRE_Int)*mpisize);

   measure_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_vertices, NALU_HYPRE_MEMORY_HOST);
   lists = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_vertices, NALU_HYPRE_MEMORY_HOST);
   where = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_vertices, NALU_HYPRE_MEMORY_HOST);

   /*   for (p=0;p<mpisize;p++) nalu_hypre_printf ("%d: %d-%d\n",p,range[p]+1,range[p+1]); */

   /******************************************************************
    * determine heavy edges
    ******************************************************************/

   jG  = G_i[num_vertices];
   H   = nalu_hypre_CSRMatrixCreate (num_vertices, num_vertices, jG);
   H_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_vertices + 1, NALU_HYPRE_MEMORY_HOST);
   H_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, jG, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CSRMatrixI(H) = H_i;
   nalu_hypre_CSRMatrixJ(H) = H_j;
   nalu_hypre_CSRMatrixMemoryLocation(H) = NALU_HYPRE_MEMORY_HOST;

   for (i = 0, p = 0; i < num_vertices; i++)
   {
      while (vertexrange[p + 1] <= i) { p++; }
      processor[i] = p;
   }

   H_i[0] = 0;
   for (i = 0, jj = 0; i < num_vertices; i++)
   {
#if 0
      nalu_hypre_printf ("neighbors of grid %d:", i);
#endif
      H_i[i + 1] = H_i[i];
      for (j = G_i[i], choice = -1, max = 0; j < G_i[i + 1]; j++)
      {
#if 0
         if (G_data[j] >= 0.0)
         {
            nalu_hypre_printf ("G[%d,%d]=0. G_j(j)=%d, G_data(j)=%f.\n", i, G_j[j], j, G_data[j]);
         }
#endif
         /* G_data is always negative, so this test is sufficient */
         if (choice == -1 || G_data[j] > max)
         {
            choice = G_j[j];
            max = G_data[j];
         }
         if (j == G_i[i + 1] - 1 || processor[G_j[j + 1]] > processor[choice])
         {
            /* we are done for this processor boundary */
            H_j[jj++] = choice;
            H_i[i + 1]++;
#if 0
            nalu_hypre_printf (" %d", choice);
#endif
            choice = -1; max = 0;
         }
      }
#if 0
      nalu_hypre_printf("\n");
#endif
   }

   /******************************************************************
    * compute H^T, the transpose of H
    ******************************************************************/

   jH = H_i[num_vertices];
   HT = nalu_hypre_CSRMatrixCreate (num_vertices, num_vertices, jH);
   HT_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_vertices + 1, NALU_HYPRE_MEMORY_HOST);
   HT_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, jH, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CSRMatrixI(HT) = HT_i;
   nalu_hypre_CSRMatrixJ(HT) = HT_j;
   nalu_hypre_CSRMatrixMemoryLocation(HT) = NALU_HYPRE_MEMORY_HOST;

   for (i = 0; i <= num_vertices; i++)
   {
      HT_i[i] = 0;
   }
   for (i = 0; i < jH; i++)
   {
      HT_i[H_j[i] + 1]++;
   }
   for (i = 0; i < num_vertices; i++)
   {
      HT_i[i + 1] += HT_i[i];
   }
   for (i = 0; i < num_vertices; i++)
   {
      for (j = H_i[i]; j < H_i[i + 1]; j++)
      {
         NALU_HYPRE_Int myindex = H_j[j];
         HT_j[HT_i[myindex]] = i;
         HT_i[myindex]++;
      }
   }
   for (i = num_vertices; i > 0; i--)
   {
      HT_i[i] = HT_i[i - 1];
   }
   HT_i[0] = 0;

   /*****************************************************************
    * set initial vertex weights
    *****************************************************************/

   for (i = 0; i < num_vertices; i++)
   {
      measure_array[i] = H_i[i + 1] - H_i[i] + HT_i[i + 1] - HT_i[i];
      nalu_hypre_enter_on_lists (&LoL_head, &LoL_tail, measure_array[i], i, lists, where);
   }

   /******************************************************************
    * apply CGC iteration
    ******************************************************************/

   while (LoL_head && measure_array[LoL_head->head])
   {


      choice = LoL_head->head;
      measure = measure_array[choice];
#if 0
      nalu_hypre_printf ("Choice: %d, measure %d, processor %d\n", choice, measure, processor[choice]);
      fflush(stdout);
#endif

      (*coarse)[processor[choice]] = choice
                                     + 1; /* add one because coarsegrid indexing starts with 1, not 0 */
      /* new maximal weight */
      new_measure = measure + 1;
      for (i = vertexrange[processor[choice]]; i < vertexrange[processor[choice] + 1]; i++)
      {
         /* set weights for all remaining vertices on this processor to zero */
         measure = measure_array[i];
         nalu_hypre_remove_point (&LoL_head, &LoL_tail, measure, i, lists, where);
         measure_array[i] = 0;
      }
      for (j = H_i[choice]; j < H_i[choice + 1]; j++)
      {
         jj = H_j[j];
         /* if no vertex is chosen on this proc, set weights of all heavily coupled vertices to max1 */
         if (!(*coarse)[processor[jj]])
         {
            measure = measure_array[jj];
            nalu_hypre_remove_point (&LoL_head, &LoL_tail, measure, jj, lists, where);
            nalu_hypre_enter_on_lists (&LoL_head, &LoL_tail, new_measure, jj, lists, where);
            measure_array[jj] = new_measure;
         }
      }
      for (j = HT_i[choice]; j < HT_i[choice + 1]; j++)
      {
         jj = HT_j[j];
         /* if no vertex is chosen on this proc, set weights of all heavily coupled vertices to max1 */
         if (!(*coarse)[processor[jj]])
         {
            measure = measure_array[jj];
            nalu_hypre_remove_point (&LoL_head, &LoL_tail, measure, jj, lists, where);
            nalu_hypre_enter_on_lists (&LoL_head, &LoL_tail, new_measure, jj, lists, where);
            measure_array[jj] = new_measure;
         }
      }
   }

   /* remove remaining list elements, if they exist. They all should have measure 0 */
   while (LoL_head)
   {
      i = LoL_head->head;
      measure = measure_array[i];
#if 0
      nalu_hypre_assert (measure == 0);
#endif
      nalu_hypre_remove_point (&LoL_head, &LoL_tail, measure, i, lists, where);
   }


   for (p = 0; p < mpisize; p++)
      /* if the algorithm has not determined a coarse vertex for this proc, simply take the last one
         Do not take the first one, it might by empty! */
      if (!(*coarse)[p])
      {
         (*coarse)[p] = vertexrange[p + 1];
         /*       nalu_hypre_printf ("choice for processor %d: %d\n",p,range[p]+1); */
      }

   /********************************************
    * clean up
    ********************************************/

   nalu_hypre_CSRMatrixDestroy (H);
   nalu_hypre_CSRMatrixDestroy (HT);


   nalu_hypre_TFree(processor, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(measure_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(lists, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(where, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_AmgCGCBoundaryFix (nalu_hypre_ParCSRMatrix *S, NALU_HYPRE_Int *CF_marker,
                                   NALU_HYPRE_Int *CF_marker_offd)
/* Checks whether an interpolation is possible for a fine grid point with strong couplings.
 * Required after CGC coarsening
 * ========================================================================================
 * S : the strength matrix
 * CF_marker, CF_marker_offd : the coarse/fine markers
 * ========================================================================================*/
{
   NALU_HYPRE_Int mpirank, i, j, has_c_pt;
   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag (S);
   nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd (S);
   NALU_HYPRE_Int *S_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int *S_j = nalu_hypre_CSRMatrixJ(S_diag);
   NALU_HYPRE_Int *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int *S_offd_j = NULL;
   NALU_HYPRE_Int num_variables = nalu_hypre_CSRMatrixNumRows (S_diag);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols (S_offd);
   NALU_HYPRE_Int added_cpts = 0;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(S);

   nalu_hypre_MPI_Comm_rank (comm, &mpirank);
   if (num_cols_offd)
   {
      S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);
   }

   for (i = 0; i < num_variables; i++)
   {
      if (S_offd_i[i] == S_offd_i[i + 1] || CF_marker[i] == C_PT) { continue; }
      has_c_pt = 0;

      /* fine grid point with strong connections across the boundary */
      for (j = S_i[i]; j < S_i[i + 1]; j++)
         if (CF_marker[S_j[j]] == C_PT) {has_c_pt = 1; break;}
      if (has_c_pt) { continue; }

      for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         if (CF_marker_offd[S_offd_j[j]] == C_PT) {has_c_pt = 1; break;}
      if (has_c_pt) { continue; }

      /* all points i is strongly coupled to are fine: make i C_PT */
      CF_marker[i] = C_PT;
#if 0
      nalu_hypre_printf ("Processor %d: added point %d in nalu_hypre_AmgCGCBoundaryFix\n", mpirank, i);
#endif
      added_cpts++;
   }
#if 0
   if (added_cpts) { nalu_hypre_printf ("Processor %d: added %d points in nalu_hypre_AmgCGCBoundaryFix\n", mpirank, added_cpts); }
   fflush(stdout);
#endif
   return nalu_hypre_error_flag;
}
