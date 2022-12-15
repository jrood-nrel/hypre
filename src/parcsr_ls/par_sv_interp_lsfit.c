/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "Common.h"
#include "_nalu_hypre_blas.h"
#include "_nalu_hypre_lapack.h"

#define ADJUST(a,b)  (adjust_list[(a)*(num_functions-1)+(b)])

/******************************************************************************
 * nalu_hypre_BoomerAMGFitInterpVectors
 *
  This routine for updating the interp operator to interpolate the
  supplied smooth vectors with a L.S. fitting.  This code (varient 0)
  was used for the Baker, Kolev and Yang elasticity paper in section 3
  to evaluate the least squares fitting methed proposed by Stuben in
  his talk (see paper for details).  So this code is basically a
  post-processing step that performs the LS fit (the size and sparsity
  of P do not change).

  Note: truncation only works correctly for 1 processor - needs to
        just use the other truncation rouitne


  Variant = 0: do L.S. fit to existing interp weights (default)


  Variant = 1: extends the neighborhood to incl. other unknowns on the
  same node - ASSUMES A NODAL COARSENING, ASSUMES VARIABLES ORDERED
  GRID POINT, THEN UNKNOWN (e.g., u0, v0, u1, v1, etc. ), AND AT MOST
  3 FCNS (NOTE: **only** works with 1 processor)

  This code is not compiled or accessible through hypre at this time
  (it was not particularly effective - compared to the LN and GM
  approaches), but is checked-in in case there is interest in the
  future.

 ******************************************************************************/
NALU_HYPRE_Int nalu_hypre_BoomerAMGFitInterpVectors( nalu_hypre_ParCSRMatrix *A,
                                           nalu_hypre_ParCSRMatrix **P,
                                           NALU_HYPRE_Int num_smooth_vecs,
                                           nalu_hypre_ParVector **smooth_vecs,
                                           nalu_hypre_ParVector **coarse_smooth_vecs,
                                           NALU_HYPRE_Real delta,
                                           NALU_HYPRE_Int num_functions,
                                           NALU_HYPRE_Int *dof_func,
                                           NALU_HYPRE_Int *CF_marker,
                                           NALU_HYPRE_Int max_elmts,
                                           NALU_HYPRE_Real trunc_factor,
                                           NALU_HYPRE_Int variant, NALU_HYPRE_Int level)
{

   NALU_HYPRE_Int  i, j, k;

   NALU_HYPRE_Int  one_i = 1;
   NALU_HYPRE_Int  info;
   NALU_HYPRE_Int  coarse_index;;
   NALU_HYPRE_Int  num_coarse_diag;
   NALU_HYPRE_Int  num_coarse_offd;
   NALU_HYPRE_Int  num_nonzeros = 0;
   NALU_HYPRE_Int  coarse_point = 0;
   NALU_HYPRE_Int  k_size;
   NALU_HYPRE_Int  k_alloc;
   NALU_HYPRE_Int  counter;
   NALU_HYPRE_Int  *piv;
   NALU_HYPRE_Int  tmp_int;
   NALU_HYPRE_Int  num_sends;

   NALU_HYPRE_Real *alpha;
   NALU_HYPRE_Real *Beta;
   NALU_HYPRE_Real *w;
   NALU_HYPRE_Real *w_old;
   NALU_HYPRE_Real *B_s;

   NALU_HYPRE_Real tmp_double;
   NALU_HYPRE_Real one = 1.0;
   NALU_HYPRE_Real mone = -1.0;;
   NALU_HYPRE_Real *vec_data;

   nalu_hypre_CSRMatrix *P_diag = nalu_hypre_ParCSRMatrixDiag(*P);
   nalu_hypre_CSRMatrix *P_offd = nalu_hypre_ParCSRMatrixOffd(*P);
   NALU_HYPRE_Real      *P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
   NALU_HYPRE_Int       *P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
   NALU_HYPRE_Int       *P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
   NALU_HYPRE_Real      *P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
   NALU_HYPRE_Int       *P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
   NALU_HYPRE_Int       *P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
   NALU_HYPRE_Int       num_rows_P = nalu_hypre_CSRMatrixNumRows(P_diag);
   NALU_HYPRE_Int        P_diag_size = P_diag_i[num_rows_P];
   NALU_HYPRE_Int        P_offd_size = P_offd_i[num_rows_P];
   NALU_HYPRE_Int        num_cols_P_offd = nalu_hypre_CSRMatrixNumCols(P_offd);
   NALU_HYPRE_BigInt    *col_map_offd_P = NULL;

   nalu_hypre_CSRMatrix  *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int         num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(*P);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   MPI_Comm                 comm;


   NALU_HYPRE_Real  *dbl_buf_data;
   NALU_HYPRE_Real  *smooth_vec_offd = NULL;
   NALU_HYPRE_Real  *offd_vec_data;

   NALU_HYPRE_Int   index, start;
   NALU_HYPRE_Int  *P_marker;
   NALU_HYPRE_Int   num_procs;

   nalu_hypre_ParVector *vector;

   NALU_HYPRE_Int   new_nnz, orig_start, j_pos, fcn_num, num_elements;
   NALU_HYPRE_Int  *P_diag_j_new;
   NALU_HYPRE_Real *P_diag_data_new;
   NALU_HYPRE_Int   adjust_3D[] = {1, 2, -1, 1, -2, -1};
   NALU_HYPRE_Int   adjust_2D[] = {1, -1};
   NALU_HYPRE_Int  *adjust_list;

   if (variant == 1 && num_functions > 1)
   {
      /* First add new entries to P with value 0.0 corresponding to weights from
         other unknowns on the same grid point */
      /* Loop through each row */

      new_nnz = P_diag_size * num_functions; /* this is an over-estimate */
      P_diag_j_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  new_nnz, NALU_HYPRE_MEMORY_HOST);
      P_diag_data_new = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  new_nnz, NALU_HYPRE_MEMORY_HOST);


      if (num_functions == 2)
      {
         adjust_list = adjust_2D;
      }
      else if (num_functions == 3)
      {
         adjust_list = adjust_3D;
      }

      j_pos = 0;
      orig_start = 0;
      /* loop through rows */
      for (i = 0; i < num_rows_P; i++)
      {
         fcn_num = (NALU_HYPRE_Int) fmod(i, num_functions);
         if (fcn_num != dof_func[i])
         {
            printf("WARNING - ROWS incorrectly ordered!\n");
         }

         /* loop through elements */
         num_elements = P_diag_i[i + 1] - orig_start;

         /* add zeros corrresponding to other unknowns */
         if (num_elements > 1)
         {
            for (j = 0; j < num_elements; j++)
            {
               P_diag_j_new[j_pos] = P_diag_j[orig_start + j];
               P_diag_data_new[j_pos++] = P_diag_data[orig_start + j];

               for (k = 0; k < num_functions - 1; k++)
               {
                  P_diag_j_new[j_pos] = P_diag_j[orig_start + j] + ADJUST(fcn_num, k);
                  P_diag_data_new[j_pos++] = 0.0;
               }
            }
         }
         else if (num_elements == 1)/* only one element - just copy to new */
         {
            P_diag_j_new[j_pos] = P_diag_j[orig_start];
            P_diag_data_new[j_pos++] = P_diag_data[orig_start];
         }
         orig_start = P_diag_i[i + 1];
         if (num_elements > 1)
         {
            P_diag_i[i + 1] =  P_diag_i[i] + num_elements * num_functions;
         }
         else
         {
            P_diag_i[i + 1] = P_diag_i[i] + num_elements;
         }

         if (j_pos != P_diag_i[i + 1]) { printf("Problem!\n"); }


      }/* end loop through rows */

      /* modify P */
      nalu_hypre_TFree(P_diag_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_diag_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixJ(P_diag) = P_diag_j_new;
      nalu_hypre_CSRMatrixData(P_diag) = P_diag_data_new;
      nalu_hypre_CSRMatrixNumNonzeros(P_diag) = P_diag_i[num_rows_P];
      P_diag_j = P_diag_j_new;
      P_diag_data = P_diag_data_new;

      /* check if there is already a comm pkg - if so, destroy*/
      if (comm_pkg)
      {
         nalu_hypre_MatvecCommPkgDestroy(comm_pkg );
         comm_pkg = NULL;

      }


   } /* end variant == 1 and num functions > 0 */



   /* For each row, we are updating the weights by
      solving w = w_old + (delta)(Beta^T)Bs^(-1)(alpha - (Beta)w_old).
      let s = num_smooth_vectors
      let k = # of interp points for fine point i
      Then:
      w = new weights (k x 1)
      w_old = old weights (k x 1)
      delta is a scalar weight in [0,1]
      alpha = s x 1 vector of s smooth vector values at fine point i
      Beta = s x k matrix of s smooth vector values at k interp points of i
      Bs = delta*Beta*Beta^T+(1-delta)*I_s (I_s is sxs identity matrix)
   */



#if 0
   /* print smoothvecs */
   {
      char new_file[80];

      for (i = 0; i < num_smooth_vecs; i++)
      {
         sprintf(new_file, "%s.%d.level.%d", "smoothvec", i, level );
         nalu_hypre_ParVectorPrint(smooth_vecs[i], new_file);
      }
   }

#endif

   /*initial*/
   if (num_smooth_vecs == 0)
   {
      return nalu_hypre_error_flag;
   }

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate ( *P );
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(*P);
   }


   comm      = nalu_hypre_ParCSRCommPkgComm(comm_pkg);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(P_diag)
                  + nalu_hypre_CSRMatrixNumNonzeros(P_offd);

   /* number of coarse points = number of cols */
   coarse_points = nalu_hypre_CSRMatrixNumCols(P_diag) + nalu_hypre_CSRMatrixNumCols(P_offd);

   /* allocate */
   alpha = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_smooth_vecs, NALU_HYPRE_MEMORY_HOST);
   piv = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_smooth_vecs, NALU_HYPRE_MEMORY_HOST);
   B_s = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_smooth_vecs * num_smooth_vecs, NALU_HYPRE_MEMORY_HOST);

   /*estimate the max number of weights per row (coarse points only have one weight)*/
   k_alloc = (num_nonzeros - coarse_points) / (num_rows_P - coarse_points);
   k_alloc += 5;

   Beta = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  k_alloc * num_smooth_vecs, NALU_HYPRE_MEMORY_HOST);
   w = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  k_alloc, NALU_HYPRE_MEMORY_HOST);
   w_old = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  k_alloc, NALU_HYPRE_MEMORY_HOST);

   /* Get smooth vec components for the off-processor columns */

   if (num_procs > 1)
   {

      smooth_vec_offd =  nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_cols_P_offd * num_smooth_vecs, NALU_HYPRE_MEMORY_HOST);

      /* for now, do a seperate comm for each smooth vector */
      for (k = 0; k < num_smooth_vecs; k++)
      {

         vector = smooth_vecs[k];
         vec_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(vector));

         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         dbl_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                                   num_sends), NALU_HYPRE_MEMORY_HOST);
         /* point into smooth_vec_offd */
         offd_vec_data =  smooth_vec_offd + k * num_cols_P_offd;

         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
               dbl_buf_data[index++]
                  = vec_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }

         comm_handle = nalu_hypre_ParCSRCommHandleCreate( 1, comm_pkg, dbl_buf_data,
                                                     offd_vec_data);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         nalu_hypre_TFree(dbl_buf_data, NALU_HYPRE_MEMORY_HOST);
      }
   }/*end num procs > 1 */
   /* now off-proc smooth vec data is in smoothvec_offd */

   /* Loop through each row */
   for (i = 0; i < num_rows_P; i++)
   {

      /* only need to modify rows belonging to fine points */
      if (CF_marker[i] >= 0) /* coarse */
      {
         continue;
      }

      num_coarse_diag = P_diag_i[i + 1] - P_diag_i[i];
      num_coarse_offd =  P_offd_i[i + 1] - P_offd_i[i];

      k_size = num_coarse_diag + num_coarse_offd;


      /* only need to modify rows that interpolate from coarse points */
      if (k_size == 0)
      {
         continue;
      }

#if 0
      /* only change the weights if we have at least as many coarse points
         as smooth vectors - do we want to do this? NO */

      too_few = 0;
      if (k_size < num_smooth_vecs)
      {
         too_few++;
         continue;
      }
#endif

      /*verify that we have enough space allocated */
      if (k_size > k_alloc)
      {
         k_alloc = k_size + 2;

         Beta = nalu_hypre_TReAlloc(Beta,  NALU_HYPRE_Real,  k_alloc * num_smooth_vecs, NALU_HYPRE_MEMORY_HOST);
         w = nalu_hypre_TReAlloc(w,  NALU_HYPRE_Real,  k_alloc, NALU_HYPRE_MEMORY_HOST);
         w_old = nalu_hypre_TReAlloc(w_old,  NALU_HYPRE_Real,  k_alloc, NALU_HYPRE_MEMORY_HOST);
      }

      /* put current weights into w*/
      counter = 0;
      for (j = P_diag_i[i]; j <  P_diag_i[i + 1]; j++)
      {
         w[counter++] = P_diag_data[j];
      }
      for (j = P_offd_i[i]; j <  P_offd_i[i + 1]; j++)
      {
         w[counter++] = P_offd_data[j];
      }

      /* copy w to w_old */
      for (j = 0; j < k_size; j++)
      {
         w_old[j] = w[j];
      }

      /* get alpha and Beta */
      /* alpha is the smooth vector values at fine point i */
      /* Beta is the smooth vector values at the points that
         i interpolates from */

      /* Note - for using BLAS/LAPACK - need to store Beta in
       * column-major order */

      for (j = 0; j < num_smooth_vecs; j++)
      {
         vector = smooth_vecs[j];
         vec_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(vector));
         /* point into smooth_vec_offd */
         offd_vec_data = smooth_vec_offd + j * num_cols_P_offd;

         alpha[j] = vec_data[i];

         vector = coarse_smooth_vecs[j];
         vec_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(vector));
         /* on processor */
         counter = 0;

         for (k = P_diag_i[i]; k <  P_diag_i[i + 1]; k++)
         {
            coarse_index = P_diag_j[k];
            /*Beta(j, counter) */
            Beta[counter * num_smooth_vecs + j] = vec_data[coarse_index];
            counter++;
         }
         /* off-processor */
         for (k = P_offd_i[i]; k <  P_offd_i[i + 1]; k++)
         {
            coarse_index = P_offd_j[k];
            Beta[counter * num_smooth_vecs + j] = offd_vec_data[coarse_index];
            counter++;

         }

      }

      /* form B_s: delta*Beta*Beta^T + (1-delta)*I_s */

      /* first B_s <- (1-delta)*I_s */
      tmp_double = 1.0 - delta;
      for (j = 0; j < num_smooth_vecs * num_smooth_vecs; j++)
      {
         B_s[j] = 0.0;
      }
      for (j = 0; j < num_smooth_vecs; j++)
      {
         B_s[j * num_smooth_vecs + j] = tmp_double;
      }

      /* now  B_s <-delta*Beta*Beta^T + B_s */
      /* usage: DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
                C := alpha*op( A )*op( B ) + beta*C */
      nalu_hypre_dgemm("N", "T", &num_smooth_vecs,
                  &num_smooth_vecs, &k_size,
                  &delta, Beta, &num_smooth_vecs, Beta,
                  &num_smooth_vecs, &one, B_s, &num_smooth_vecs);

      /* now do alpha <- (alpha - beta*w)*/
      /* usage: DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
                y := alpha*A*x + beta*y */
      nalu_hypre_dgemv("N", &num_smooth_vecs, &k_size, &mone,
                  Beta, &num_smooth_vecs, w_old, &one_i,
                  &one, alpha, &one_i);

      /* now get alpha <- inv(B_s)*alpha */
      /*write over B_s with LU */
      nalu_hypre_dgetrf(&num_smooth_vecs, &num_smooth_vecs,
                   B_s, &num_smooth_vecs, piv, &info);

      /*now get alpha  */
      nalu_hypre_dgetrs("N", &num_smooth_vecs, &one_i, B_s,
                   &num_smooth_vecs, piv, alpha,
                   &num_smooth_vecs, &info);

      /* now w <- w + (delta)*(Beta)^T*(alpha) */
      nalu_hypre_dgemv("T", &num_smooth_vecs, &k_size, &delta,
                  Beta, &num_smooth_vecs, alpha, &one_i,
                  &one, w, &one_i);

      /* note:we have w_old still, but we don't need it unless we
       * want to use it in the future for something */

      /* now update the weights in P*/
      counter = 0;
      for (j = P_diag_i[i]; j <  P_diag_i[i + 1]; j++)
      {
         P_diag_data[j] = w[counter++];
      }
      for (j = P_offd_i[i]; j <  P_offd_i[i + 1]; j++)
      {
         P_offd_data[j] = w[counter++];
      }
   }/* end of loop through each row */


   /* clean up from L.S. fitting*/
   nalu_hypre_TFree(alpha, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Beta, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(w, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(w_old, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(piv, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(B_s, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(smooth_vec_offd, NALU_HYPRE_MEMORY_HOST);

   /* Now we truncate here (instead of after forming the interp matrix) */

   /* SAME code as in othr interp routines:
      Compress P, removing coefficients smaller than trunc_factor * Max ,
      or when there are more than max_elements*/

   if (trunc_factor != 0.0 || max_elmts > 0)
   {

      /* To DO: THIS HAS A BUG IN PARALLEL! */

      tmp_int =  P_offd_size;

      nalu_hypre_BoomerAMGInterpTruncation(*P, trunc_factor, max_elmts);
      P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
      P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
      P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
      P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
      P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
      P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[num_rows_P];

      P_offd_size = P_offd_i[num_rows_P];


      /* if truncation occurred, we need to re-do the col_map_offd... */
      if (tmp_int != P_offd_size)
      {
         NALU_HYPRE_Int *tmp_map_offd;
         num_cols_P_offd = 0;
         P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

         for (i = 0; i < num_cols_A_offd; i++)
         {
            P_marker[i] = 0;
         }

         num_cols_P_offd = 0;
         for (i = 0; i < P_offd_size; i++)
         {
            index = P_offd_j[i];
            if (!P_marker[index])
            {
               num_cols_P_offd++;
               P_marker[index] = 1;
            }
         }

         col_map_offd_P = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_P_offd, NALU_HYPRE_MEMORY_HOST);
         tmp_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_P_offd, NALU_HYPRE_MEMORY_HOST);

         index = 0;
         for (i = 0; i < num_cols_P_offd; i++)
         {
            while (P_marker[index] == 0) { index++; }
            tmp_map_offd[i] = index++;
         }
         for (i = 0; i < P_offd_size; i++)
            P_offd_j[i] = nalu_hypre_BinarySearch(tmp_map_offd,
                                             P_offd_j[i],
                                             num_cols_P_offd);
         nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree( nalu_hypre_ParCSRMatrixColMapOffd(*P), NALU_HYPRE_MEMORY_HOST);

         /* assign new col map */
         nalu_hypre_ParCSRMatrixColMapOffd(*P) = col_map_offd_P;
         nalu_hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;


         /* destroy the old and get a new commpkg....*/
         nalu_hypre_MatvecCommPkgDestroy(comm_pkg);
         nalu_hypre_MatvecCommPkgCreate ( *P );
         nalu_hypre_TFree(tmp_map_offd);

      }/*end re-do col_map_offd */

   }/*end trucation */

   return nalu_hypre_error_flag;


}
