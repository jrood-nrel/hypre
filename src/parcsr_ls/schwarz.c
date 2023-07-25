/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "Common.h"
#include "_nalu_hypre_lapack.h"

/* -------------------------------------------------------------------------
   dof_domain: for each dof defines neighborhood to build interpolation,
   using
   domain_diagmat (for cut--off scaling) and
   i_domain_dof, j_dof_domain (for extracting the block of A);

   domain_matrixinverse: contains the inverse of subdomain matrix;

   B can be used to define strength matrix;
   ----------------------------------------------------------------------- */

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGNodalSchwarzSmoother: (Not used currently)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGNodalSchwarzSmoother( nalu_hypre_CSRMatrix    *A,
                               NALU_HYPRE_Int                 num_functions,
                               NALU_HYPRE_Int                   option,
                               nalu_hypre_CSRMatrix   **domain_structure_pointer)
{
   /*  option =      0: nodal symGS;
       1: next to nodal symGS (overlapping Schwarz) */

   NALU_HYPRE_Int *i_domain_dof, *j_domain_dof;
   NALU_HYPRE_Real *domain_matrixinverse;
   NALU_HYPRE_Int num_domains;
   nalu_hypre_CSRMatrix *domain_structure;

   NALU_HYPRE_Int *i_dof_node, *j_dof_node;
   NALU_HYPRE_Int *i_node_dof, *j_node_dof;

   NALU_HYPRE_Int *i_node_dof_dof, *j_node_dof_dof;

   NALU_HYPRE_Int *i_node_node, *j_node_node;

   NALU_HYPRE_Int num_nodes;

   NALU_HYPRE_Int *i_dof_dof = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int *j_dof_dof = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Real *a_dof_dof = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int num_dofs = nalu_hypre_CSRMatrixNumRows(A);


   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int i, j, k, l_loc, i_loc, j_loc;
   NALU_HYPRE_Int i_dof, j_dof;
   NALU_HYPRE_Int *i_local_to_global;
   NALU_HYPRE_Int *i_global_to_local;

   NALU_HYPRE_Int *i_int;
   NALU_HYPRE_Int *i_int_to_local;

   NALU_HYPRE_Int int_dof_counter, local_dof_counter, max_local_dof_counter = 0;

   NALU_HYPRE_Int domain_dof_counter = 0, domain_matrixinverse_counter = 0;

   NALU_HYPRE_Real *AE;

   char uplo = 'L';

   NALU_HYPRE_Int cnt;

   /* PCG arrays: ---------------------------------------------------
      NALU_HYPRE_Real *x, *rhs, *v, *w, *d, *aux;

      NALU_HYPRE_Int max_iter;

      ------------------------------------------------------------------ */

   /* build dof_node graph: ----------------------------------------- */

   num_nodes = num_dofs / num_functions;

   /*nalu_hypre_printf("\nnum_nodes: %d, num_dofs: %d = %d x %d\n", num_nodes, num_dofs,
     num_nodes, num_functions);*/

   i_dof_node = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_dofs + 1, NALU_HYPRE_MEMORY_HOST);
   j_dof_node = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_dofs, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_dofs + 1; i++)
   {
      i_dof_node[i] = i;
   }

   for (j = 0; j < num_nodes; j++)
      for (k = 0; k < num_functions; k++)
      {
         j_dof_node[j * num_functions + k] = j;
      }

   /* build node_dof graph: ----------------------------------------- */

   ierr = transpose_matrix_create(&i_node_dof, &j_node_dof,
                                  i_dof_node, j_dof_node,
                                  num_dofs, num_nodes);


   /* build node_node graph: ----------------------------------------- */

   ierr = matrix_matrix_product(&i_node_dof_dof,
                                &j_node_dof_dof,
                                i_node_dof, j_node_dof,
                                i_dof_dof, j_dof_dof,
                                num_nodes, num_dofs, num_dofs);

   ierr = matrix_matrix_product(&i_node_node,
                                &j_node_node,
                                i_node_dof_dof,
                                j_node_dof_dof,
                                i_dof_node, j_dof_node,
                                num_nodes, num_dofs, num_nodes);

   nalu_hypre_TFree(i_node_dof_dof, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_node_dof_dof, NALU_HYPRE_MEMORY_HOST);

   /* compute for each node the local information: -------------------- */

   i_global_to_local = i_dof_node;

   for (i_dof = 0; i_dof < num_dofs; i_dof++)
   {
      i_global_to_local[i_dof] = -1;
   }

   domain_matrixinverse_counter = 0;
   domain_dof_counter = 0;
   for (i = 0; i < num_nodes; i++)
   {
      local_dof_counter = 0;

      for (j = i_node_node[i]; j < i_node_node[i + 1]; j++)
         for (k = i_node_dof[j_node_node[j]];
              k < i_node_dof[j_node_node[j] + 1]; k++)
         {
            j_dof = j_node_dof[k];

            if (i_global_to_local[j_dof] < 0)
            {
               i_global_to_local[j_dof] = local_dof_counter;
               local_dof_counter++;
            }

         }
      domain_matrixinverse_counter += local_dof_counter * local_dof_counter;
      domain_dof_counter += local_dof_counter;

      if (local_dof_counter > max_local_dof_counter)
      {
         max_local_dof_counter = local_dof_counter;
      }

      for (j = i_node_node[i]; j < i_node_node[i + 1]; j++)
         for (k = i_node_dof[j_node_node[j]];
              k < i_node_dof[j_node_node[j] + 1]; k++)
         {
            j_dof = j_node_dof[k];
            i_global_to_local[j_dof] = -1;
         }
   }

   num_domains = num_nodes;

   i_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_domains + 1, NALU_HYPRE_MEMORY_HOST);

   if (option == 1)
   {
      j_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  domain_dof_counter, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      j_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_dofs, NALU_HYPRE_MEMORY_HOST);
   }

   if (option == 1)
   {
      domain_matrixinverse = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  domain_matrixinverse_counter, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      domain_matrixinverse = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_dofs * num_functions, NALU_HYPRE_MEMORY_HOST);
   }

   i_local_to_global = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_local_dof_counter, NALU_HYPRE_MEMORY_HOST);

   i_int_to_local = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_local_dof_counter, NALU_HYPRE_MEMORY_HOST);
   i_int          = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_local_dof_counter, NALU_HYPRE_MEMORY_HOST);

   for (l_loc = 0; l_loc < max_local_dof_counter; l_loc++)
   {
      i_int[l_loc] = -1;
   }

   domain_dof_counter = 0;
   domain_matrixinverse_counter = 0;
   for (i = 0; i < num_nodes; i++)
   {
      i_domain_dof[i] = domain_dof_counter;
      local_dof_counter = 0;

      for (j = i_node_node[i]; j < i_node_node[i + 1]; j++)
         for (k = i_node_dof[j_node_node[j]];
              k < i_node_dof[j_node_node[j] + 1]; k++)
         {
            j_dof = j_node_dof[k];

            if (i_global_to_local[j_dof] < 0)
            {
               i_global_to_local[j_dof] = local_dof_counter;
               i_local_to_global[local_dof_counter] = j_dof;
               local_dof_counter++;
            }

         }

      for (j = i_node_dof[i]; j < i_node_dof[i + 1]; j++)
         for (k = i_dof_dof[j_node_dof[j]]; k < i_dof_dof[j_node_dof[j] + 1]; k++)
            if (i_global_to_local[j_dof_dof[k]] < 0)
            {
               nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "WRONG local indexing: ====================== \n");
            }

      int_dof_counter = 0;
      for (k = i_node_dof[i]; k < i_node_dof[i + 1]; k++)
      {
         i_dof = j_node_dof[k];
         i_loc = i_global_to_local[i_dof];
         i_int[i_loc] = int_dof_counter;
         i_int_to_local[int_dof_counter] = i_loc;
         int_dof_counter++;
      }

      /* get local matrix AE: ======================================== */

      if (option == 1)
      {
         AE = &domain_matrixinverse[domain_matrixinverse_counter];
         cnt = 0;
         for (i_loc = 0; i_loc < local_dof_counter; i_loc++)
            for (j_loc = 0; j_loc < local_dof_counter; j_loc++)
            {
               AE[cnt++] = 0.e0;
            }

         for (i_loc = 0; i_loc < local_dof_counter; i_loc++)
         {
            i_dof = i_local_to_global[i_loc];
            for (j = i_dof_dof[i_dof]; j < i_dof_dof[i_dof + 1]; j++)
            {
               j_loc = i_global_to_local[j_dof_dof[j]];
               if (j_loc >= 0)
               {
                  AE[i_loc + j_loc * local_dof_counter] = a_dof_dof[j];
               }
            }
         }

         /* get block for Schwarz smoother: ============================= */
         /* ierr = nalu_hypre_matinv(XE, AE, local_dof_counter); */
         /* nalu_hypre_printf("ierr_AE_inv: %d\n", ierr); */
         nalu_hypre_dpotrf(&uplo, &local_dof_counter, AE,
                      &local_dof_counter, &ierr);
         if (ierr == 1) { nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error! Matrix not SPD\n"); }

         for (i_loc = 0; i_loc < local_dof_counter; i_loc++)
            j_domain_dof[domain_dof_counter + i_loc]
               = i_local_to_global[i_loc];
      }

      if (option == 0)
      {
         AE = &domain_matrixinverse[domain_matrixinverse_counter];
         for (i_loc = 0; i_loc < int_dof_counter; i_loc++)
            for (j_loc = 0; j_loc < int_dof_counter; j_loc++)
            {
               AE[i_loc + j_loc * int_dof_counter] = 0.e0;
            }

         for (l_loc = 0; l_loc < int_dof_counter; l_loc++)
         {
            i_loc = i_int_to_local[l_loc];
            i_dof = i_local_to_global[i_loc];
            for (j = i_dof_dof[i_dof]; j < i_dof_dof[i_dof + 1]; j++)
            {
               j_loc = i_global_to_local[j_dof_dof[j]];
               if (j_loc >= 0)
                  if (i_int[j_loc] >= 0)
                     AE[i_loc + i_int[j_loc] * int_dof_counter]
                        = a_dof_dof[j];
            }
         }

         /* ierr = nalu_hypre_matinv(XE, AE, int_dof_counter); */
         nalu_hypre_dpotrf(&uplo, &local_dof_counter, AE,
                      &local_dof_counter, &ierr);

         if (ierr) { nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " error in dpotrf !!!\n"); }

         for (i_loc = 0; i_loc < int_dof_counter; i_loc++)
         {
            j_domain_dof[domain_dof_counter + i_loc] =
               i_local_to_global[i_int_to_local[i_loc]];

            for (j_loc = 0; j_loc < int_dof_counter; j_loc++)
               domain_matrixinverse[domain_matrixinverse_counter
                                    + i_loc + j_loc * int_dof_counter]
                  = AE[i_loc + j_loc * int_dof_counter];
         }

         domain_dof_counter += int_dof_counter;
         domain_matrixinverse_counter += int_dof_counter * int_dof_counter;
      }
      else
      {
         domain_dof_counter += local_dof_counter;
         domain_matrixinverse_counter += local_dof_counter * local_dof_counter;
      }

      for (l_loc = 0; l_loc < local_dof_counter; l_loc++)
      {
         i_int[l_loc] = -1;
         i_global_to_local[i_local_to_global[l_loc]] = -1;
      }
   }

   i_domain_dof[num_nodes] = domain_dof_counter;

   nalu_hypre_TFree(i_dof_node, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_dof_node, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(i_node_dof, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_node_dof, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(i_node_node, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_node_node, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(i_int, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(i_int_to_local, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(i_local_to_global, NALU_HYPRE_MEMORY_HOST);

   domain_structure = nalu_hypre_CSRMatrixCreate(num_domains, max_local_dof_counter,
                                            i_domain_dof[num_domains]);
   nalu_hypre_CSRMatrixI(domain_structure) = i_domain_dof;
   nalu_hypre_CSRMatrixJ(domain_structure) = j_domain_dof;
   nalu_hypre_CSRMatrixData(domain_structure) = domain_matrixinverse;

   *domain_structure_pointer = domain_structure;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_ParMPSchwarzSolve(nalu_hypre_ParCSRMatrix *par_A,
                                  nalu_hypre_CSRMatrix *A_boundary,
                                  nalu_hypre_ParVector *rhs_vector,
                                  nalu_hypre_CSRMatrix *domain_structure,
                                  nalu_hypre_ParVector *par_x,
                                  NALU_HYPRE_Real relax_wt,
                                  NALU_HYPRE_Real *scale,
                                  nalu_hypre_ParVector *Vtemp, NALU_HYPRE_Int *pivots,
                                  NALU_HYPRE_Int use_nonsymm)
{
   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(par_A);
   NALU_HYPRE_Int num_sends = 0;
   NALU_HYPRE_Int *send_map_starts;
   NALU_HYPRE_Int *send_map_elmts;

   nalu_hypre_ParCSRCommHandle *comm_handle;

   NALU_HYPRE_Int ierr = 0;
   /* NALU_HYPRE_Int num_dofs; */
   nalu_hypre_CSRMatrix *A_diag;
   NALU_HYPRE_Int *A_diag_i;
   NALU_HYPRE_Int *A_diag_j;
   NALU_HYPRE_Real *A_diag_data;
   nalu_hypre_CSRMatrix *A_offd;
   NALU_HYPRE_Int *A_offd_i;
   NALU_HYPRE_Int *A_offd_j;
   NALU_HYPRE_Real *A_offd_data;
   NALU_HYPRE_Real *x;
   NALU_HYPRE_Real *x_ext;
   NALU_HYPRE_Real *x_ext_old;
   NALU_HYPRE_Real *rhs;
   NALU_HYPRE_Real *rhs_ext;
   NALU_HYPRE_Real *vtemp_data;
   NALU_HYPRE_Real *aux;
   NALU_HYPRE_Real *buf_data;
   /*nalu_hypre_Vector *x_vector;*/
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(par_A);
   NALU_HYPRE_Int num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   NALU_HYPRE_Int max_domain_size = nalu_hypre_CSRMatrixNumCols(domain_structure);
   NALU_HYPRE_Int *i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   NALU_HYPRE_Int *j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   NALU_HYPRE_Real *domain_matrixinverse = nalu_hypre_CSRMatrixData(domain_structure);
   NALU_HYPRE_Int *A_boundary_i;
   NALU_HYPRE_Int *A_boundary_j;
   NALU_HYPRE_Real *A_boundary_data;
   NALU_HYPRE_Int num_variables;
   NALU_HYPRE_Int num_cols_offd;

   NALU_HYPRE_Int piv_counter = 0;
   NALU_HYPRE_Int one = 1;
   char uplo = 'L';

   NALU_HYPRE_Int jj, i, j, k, j_loc, k_loc;
   NALU_HYPRE_Int index;

   NALU_HYPRE_Int matrix_size, matrix_size_counter = 0;

   NALU_HYPRE_Int num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /* initiate:      ----------------------------------------------- */
   /* num_dofs = nalu_hypre_CSRMatrixNumRows(A); */

   A_diag = nalu_hypre_ParCSRMatrixDiag(par_A);
   A_offd = nalu_hypre_ParCSRMatrixOffd(par_A);
   num_variables = nalu_hypre_CSRMatrixNumRows(A_diag);
   num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   x = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(par_x));
   vtemp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Vtemp));
   rhs = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(rhs_vector));

   if (use_nonsymm)
   {
      uplo = 'N';
   }

   /*x_vector = nalu_hypre_ParVectorLocalVector(par_x);*/
   A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   if (num_cols_offd)
   {
      A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
      A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
      A_boundary_i = nalu_hypre_CSRMatrixI(A_boundary);
      A_boundary_j = nalu_hypre_CSRMatrixJ(A_boundary);
      A_boundary_data = nalu_hypre_CSRMatrixData(A_boundary);
   }
   aux = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  max_domain_size, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParVectorCopy(rhs_vector, Vtemp);
   nalu_hypre_ParCSRMatrixMatvec(-1.0, par_A, par_x, 1.0, Vtemp);

   if (comm_pkg)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

      buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
      x_ext = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      x_ext_old = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      rhs_ext = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            buf_data[index++] = vtemp_data[send_map_elmts[j]];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data,
                                                 rhs_ext);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            buf_data[index++] = x[send_map_elmts[j]];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data, x_ext);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   /* correction of residual for exterior points to be updated locally */
   for (i = 0; i < num_cols_offd; i++)
   {
      x_ext_old[i] = x_ext[i];
      for (j = A_boundary_i[i]; j < A_boundary_i[i + 1]; j++)
      {
         k_loc = A_boundary_j[j];
         if (k_loc < num_variables)
         {
            rhs_ext[i] += A_boundary_data[j] * x[k_loc];
         }
         else
         {
            rhs_ext[i] += A_boundary_data[j] * x_ext[k_loc - num_variables];
         }
      }
   }
   /* forward solve: ----------------------------------------------- */

   matrix_size_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         j_loc = j_domain_dof[j];
         if (j_loc < num_variables)
         {
            aux[jj] = rhs[j_loc];
            for (k = A_diag_i[j_loc]; k < A_diag_i[j_loc + 1]; k++)
            {
               aux[jj] -= A_diag_data[k] * x[A_diag_j[k]];
            }
            for (k = A_offd_i[j_loc]; k < A_offd_i[j_loc + 1]; k++)
            {
               aux[jj] -= A_offd_data[k] * x_ext[A_offd_j[k]];
            }
         }
         else
         {
            j_loc -= num_variables;
            aux[jj] = rhs_ext[j_loc];
            for (k = A_boundary_i[j_loc]; k < A_boundary_i[j_loc + 1]; k++)
            {
               k_loc = A_boundary_j[k];
            }
            if (k_loc < num_variables)
            {
               aux[jj] -= A_boundary_data[k] * x[k_loc];
            }
            else
            {
               aux[jj] -= A_boundary_data[k] * x_ext[k_loc - num_variables];
            }
         }
         jj++;
      }
      /* solve for correction: ------------------------------------- */
      if (use_nonsymm)
      {
         nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, &pivots[piv_counter], aux,
                      &matrix_size, &ierr);
      }
      else
      {
         nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, aux,
                      &matrix_size, &ierr);
      }
      if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         j_loc = j_domain_dof[j];
         if (j_loc < num_variables)
         {
            x[j_loc] +=  relax_wt * aux[jj++];
         }
         else
         {
            x_ext[j_loc - num_variables] +=  relax_wt * aux[jj++];
         }
      }
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;


   }
   /*
     for (i=0; i < num_cols_offd; i++)
     x_ext[i] -= x_ext_old[i];

     if (comm_pkg)
     {
     comm_handle=nalu_hypre_ParCSRCommHandleCreate (2,comm_pkg,x_ext,buf_data);

     nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
     comm_handle = NULL;

     index = 0;
     for (i=0; i < num_sends; i++)
     {
     for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
     x[send_map_elmts[j]] += buf_data[index++];
     }
     }
     for (i=0; i < num_variables; i++)
     x[i] *= scale[i];

     nalu_hypre_ParVectorCopy(rhs_vector,Vtemp);
     nalu_hypre_ParCSRMatrixMatvec(-1.0,par_A,par_x,1.0,Vtemp);

     if (comm_pkg)
     {
     index = 0;
     for (i=0; i < num_sends; i++)
     {
     for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
     buf_data[index++] = vtemp_data[send_map_elmts[j]];
     }

     comm_handle = nalu_hypre_ParCSRCommHandleCreate(1,comm_pkg,buf_data,
     rhs_ext);
     nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
     comm_handle = NULL;

     index = 0;
     for (i=0; i < num_sends; i++)
     {
     for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
     buf_data[index++] = x[send_map_elmts[j]];
     }

     comm_handle = nalu_hypre_ParCSRCommHandleCreate(1,comm_pkg,buf_data,x_ext);
     nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
     comm_handle = NULL;
     }
   */
   /* correction of residual for exterior points to be updated locally */
   /*   for (i=0; i < num_cols_offd; i++)
        {
        x_ext_old[i] = x_ext[i];
        for (j = A_boundary_i[i]; j < A_boundary_i[i+1]; j++)
        {
        k_loc = A_boundary_j[j];
        if (k_loc < num_variables)
        rhs_ext[i] += A_boundary_i[k]*x[k_loc];
        else
        rhs_ext[i] += A_boundary_i[k]*x_ext[k_loc-num_variables];
        }
        }
   */
   /* backward solve: ------------------------------------------------ */
   for (i = num_domains - 1; i > -1; i--)
   {
      matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];
      matrix_size_counter -= matrix_size * matrix_size;
      piv_counter -= matrix_size;

      /* compute residual: ---------------------------------------- */
      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         j_loc = j_domain_dof[j];
         if (j_loc < num_variables)
         {
            aux[jj] = rhs[j_loc];
            for (k = A_diag_i[j_loc]; k < A_diag_i[j_loc + 1]; k++)
            {
               aux[jj] -= A_diag_data[k] * x[A_diag_j[k]];
            }
            for (k = A_offd_i[j_loc]; k < A_offd_i[j_loc + 1]; k++)
            {
               aux[jj] -= A_offd_data[k] * x_ext[A_offd_j[k]];
            }
         }
         else
         {
            j_loc -= num_variables;
            aux[jj] = rhs_ext[j_loc];
            for (k = A_boundary_i[j_loc]; k < A_boundary_i[j_loc + 1]; k++)
            {
               k_loc = A_boundary_j[k];
            }
            if (k_loc < num_variables)
            {
               aux[jj] -= A_boundary_data[k] * x[k_loc];
            }
            else
            {
               aux[jj] -= A_boundary_data[k] * x_ext[k_loc - num_variables];
            }
         }
         jj++;
      }

      /* solve for correction: ------------------------------------- */
      if (use_nonsymm)
      {
         nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, &pivots[piv_counter], aux,
                      &matrix_size, &ierr);
      }
      else
      {
         nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, aux,
                      &matrix_size, &ierr);
      }

      if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         j_loc = j_domain_dof[j];
         if (j_loc < num_variables)
         {
            x[j_loc] +=  relax_wt * aux[jj++];
         }
         else
         {
            x_ext[j_loc - num_variables] +=  relax_wt * aux[jj++];
         }
      }
   }

   for (i = 0; i < num_cols_offd; i++)
   {
      x_ext[i] -= x_ext_old[i];
   }

   if (comm_pkg)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate (2, comm_pkg, x_ext, buf_data);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            x[send_map_elmts[j]] += buf_data[index++];
         }
      }

      nalu_hypre_TFree(buf_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(x_ext, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(x_ext_old, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(rhs_ext, NALU_HYPRE_MEMORY_HOST);
   }
   for (i = 0; i < num_variables; i++)
   {
      x[i] *= scale[i];
   }

   nalu_hypre_TFree(aux, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;

}

NALU_HYPRE_Int nalu_hypre_MPSchwarzSolve(nalu_hypre_ParCSRMatrix *par_A,
                               nalu_hypre_Vector *rhs_vector,
                               nalu_hypre_CSRMatrix *domain_structure,
                               nalu_hypre_ParVector *par_x,
                               NALU_HYPRE_Real relax_wt,
                               nalu_hypre_Vector *aux_vector, NALU_HYPRE_Int *pivots,
                               NALU_HYPRE_Int use_nonsymm)
{
   NALU_HYPRE_Int ierr = 0;
   /* NALU_HYPRE_Int num_dofs; */
   NALU_HYPRE_Int *i_dof_dof;
   NALU_HYPRE_Int *j_dof_dof;
   NALU_HYPRE_Real *a_dof_dof;
   NALU_HYPRE_Real *x;
   nalu_hypre_Vector *rhs;
   NALU_HYPRE_Real *aux;
   nalu_hypre_CSRMatrix *A;
   nalu_hypre_Vector *x_vector;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(par_A);
   NALU_HYPRE_Int num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   NALU_HYPRE_Int *i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   NALU_HYPRE_Int *j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   NALU_HYPRE_Real *domain_matrixinverse = nalu_hypre_CSRMatrixData(domain_structure);

   NALU_HYPRE_Int piv_counter = 0;
   NALU_HYPRE_Int one = 1;
   char uplo = 'L';

   NALU_HYPRE_Int jj, i, j, k; /*, j_loc, k_loc;*/


   NALU_HYPRE_Int matrix_size, matrix_size_counter = 0;

   NALU_HYPRE_Int num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /* initiate:      ----------------------------------------------- */
   /* num_dofs = nalu_hypre_CSRMatrixNumRows(A); */
   x_vector = nalu_hypre_ParVectorLocalVector(par_x);
   A = nalu_hypre_ParCSRMatrixDiag(par_A);
   i_dof_dof = nalu_hypre_CSRMatrixI(A);
   j_dof_dof = nalu_hypre_CSRMatrixJ(A);
   a_dof_dof = nalu_hypre_CSRMatrixData(A);
   x = nalu_hypre_VectorData(x_vector);
   aux = nalu_hypre_VectorData(aux_vector);
   /* for (i=0; i < num_dofs; i++)
      x[i] = 0.e0; */

   if (use_nonsymm)
   {
      uplo = 'N';
   }


   if (num_procs > 1)
   {
      nalu_hypre_parCorrRes(par_A, par_x, rhs_vector, &rhs);
   }
   else
   {
      rhs = rhs_vector;
   }

   /* forward solve: ----------------------------------------------- */

   matrix_size_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         aux[jj] = nalu_hypre_VectorData(rhs)[j_domain_dof[j]];
         for (k = i_dof_dof[j_domain_dof[j]];
              k < i_dof_dof[j_domain_dof[j] + 1]; k++)
         {
            aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
         }
         jj++;
      }
      /* solve for correction: ------------------------------------- */
      if (use_nonsymm)
      {
         nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, &pivots[piv_counter], aux,
                      &matrix_size, &ierr);
      }
      else
      {
         nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, aux,
                      &matrix_size, &ierr);
      }

      if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         x[j_domain_dof[j]] +=  relax_wt * aux[jj++];
      }
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;
   }

   /* backward solve: ------------------------------------------------ */
   for (i = num_domains - 1; i > -1; i--)
   {
      matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];
      matrix_size_counter -= matrix_size * matrix_size;
      piv_counter -= matrix_size;

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         aux[jj] = nalu_hypre_VectorData(rhs)[j_domain_dof[j]];
         for (k = i_dof_dof[j_domain_dof[j]];
              k < i_dof_dof[j_domain_dof[j] + 1]; k++)
         {
            aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
         }
         jj++;
      }

      /* solve for correction: ------------------------------------- */

      if (use_nonsymm)
      {
         nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, &pivots[piv_counter], aux,
                      &matrix_size, &ierr);
      }
      else
      {
         nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, aux,
                      &matrix_size, &ierr);
      }

      if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         x[j_domain_dof[j]] += relax_wt * aux[jj++];
      }
   }

   if (num_procs > 1)
   {
      nalu_hypre_SeqVectorDestroy(rhs);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_MPSchwarzCFSolve(nalu_hypre_ParCSRMatrix *par_A,
                                 nalu_hypre_Vector *rhs_vector,
                                 nalu_hypre_CSRMatrix *domain_structure,
                                 nalu_hypre_ParVector *par_x,
                                 NALU_HYPRE_Real relax_wt,
                                 nalu_hypre_Vector *aux_vector,
                                 NALU_HYPRE_Int *CF_marker,
                                 NALU_HYPRE_Int rlx_pt, NALU_HYPRE_Int *pivots,
                                 NALU_HYPRE_Int use_nonsymm)
{
   NALU_HYPRE_Int ierr = 0;
   /* NALU_HYPRE_Int num_dofs; */
   NALU_HYPRE_Int *i_dof_dof;
   NALU_HYPRE_Int *j_dof_dof;
   NALU_HYPRE_Real *a_dof_dof;
   NALU_HYPRE_Real *x;
   nalu_hypre_Vector *rhs;
   NALU_HYPRE_Real *aux;
   nalu_hypre_CSRMatrix *A;
   nalu_hypre_Vector *x_vector;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(par_A);
   NALU_HYPRE_Int num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   NALU_HYPRE_Int *i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   NALU_HYPRE_Int *j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   NALU_HYPRE_Real *domain_matrixinverse = nalu_hypre_CSRMatrixData(domain_structure);

   NALU_HYPRE_Int piv_counter = 0;
   NALU_HYPRE_Int one = 1;
   char uplo = 'L';
   NALU_HYPRE_Int jj, i, j, k; /*, j_loc, k_loc;*/

   NALU_HYPRE_Int matrix_size, matrix_size_counter = 0;

   NALU_HYPRE_Int num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /* initiate:      ----------------------------------------------- */
   /* num_dofs = nalu_hypre_CSRMatrixNumRows(A); */
   x_vector = nalu_hypre_ParVectorLocalVector(par_x);
   A = nalu_hypre_ParCSRMatrixDiag(par_A);
   i_dof_dof = nalu_hypre_CSRMatrixI(A);
   j_dof_dof = nalu_hypre_CSRMatrixJ(A);
   a_dof_dof = nalu_hypre_CSRMatrixData(A);
   x = nalu_hypre_VectorData(x_vector);
   aux = nalu_hypre_VectorData(aux_vector);
   /* for (i=0; i < num_dofs; i++)
      x[i] = 0.e0; */

   if (use_nonsymm)
   {
      uplo = 'N';
   }

   if (num_procs > 1)
   {
      nalu_hypre_parCorrRes(par_A, par_x, rhs_vector, &rhs);
   }
   else
   {
      rhs = rhs_vector;
   }

   /* forward solve: ----------------------------------------------- */

   matrix_size_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      if (CF_marker[i] == rlx_pt)
      {
         matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];

         /* compute residual: ---------------------------------------- */

         jj = 0;
         for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
         {
            aux[jj] = nalu_hypre_VectorData(rhs)[j_domain_dof[j]];
            if (CF_marker[j_domain_dof[j]] == rlx_pt)
            {
               for (k = i_dof_dof[j_domain_dof[j]];
                    k < i_dof_dof[j_domain_dof[j] + 1]; k++)
               {
                  aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
               }
            }
            jj++;
         }
         /* solve for correction: ------------------------------------- */
         if (use_nonsymm)
         {
            nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                         &domain_matrixinverse[matrix_size_counter],
                         &matrix_size, &pivots[piv_counter], aux,
                         &matrix_size, &ierr);
         }
         else
         {
            nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                         &domain_matrixinverse[matrix_size_counter],
                         &matrix_size, aux,
                         &matrix_size, &ierr);
         }

         if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
         jj = 0;
         for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
         {
            x[j_domain_dof[j]] +=  relax_wt * aux[jj++];
         }
         matrix_size_counter += matrix_size * matrix_size;
         piv_counter += matrix_size;
      }
   }

   /* backward solve: ------------------------------------------------ */
   for (i = num_domains - 1; i > -1; i--)
   {
      if (CF_marker[i] == rlx_pt)
      {
         matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];
         matrix_size_counter -= matrix_size * matrix_size;
         piv_counter -= matrix_size;

         /* compute residual: ---------------------------------------- */
         jj = 0;
         for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
         {
            aux[jj] = nalu_hypre_VectorData(rhs)[j_domain_dof[j]];
            if (CF_marker[j_domain_dof[j]] == rlx_pt)
            {
               for (k = i_dof_dof[j_domain_dof[j]];
                    k < i_dof_dof[j_domain_dof[j] + 1]; k++)
               {
                  aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
               }
            }
            jj++;
         }

         /* solve for correction: ------------------------------------- */
         if (use_nonsymm)
         {
            nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                         &domain_matrixinverse[matrix_size_counter],
                         &matrix_size, &pivots[piv_counter], aux,
                         &matrix_size, &ierr);
         }
         else
         {
            nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                         &domain_matrixinverse[matrix_size_counter],
                         &matrix_size, aux,
                         &matrix_size, &ierr);
         }

         if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
         jj = 0;
         for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
         {
            x[j_domain_dof[j]] +=  relax_wt * aux[jj++];
         }
      }
   }

   if (num_procs > 1)
   {
      nalu_hypre_SeqVectorDestroy(rhs);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_MPSchwarzFWSolve(nalu_hypre_ParCSRMatrix *par_A,
                                 nalu_hypre_Vector *rhs_vector,
                                 nalu_hypre_CSRMatrix *domain_structure,
                                 nalu_hypre_ParVector *par_x,
                                 NALU_HYPRE_Real relax_wt,
                                 nalu_hypre_Vector *aux_vector, NALU_HYPRE_Int *pivots,
                                 NALU_HYPRE_Int use_nonsymm)
{
   NALU_HYPRE_Int ierr = 0;
   /* NALU_HYPRE_Int num_dofs; */
   NALU_HYPRE_Int *i_dof_dof;
   NALU_HYPRE_Int *j_dof_dof;
   NALU_HYPRE_Real *a_dof_dof;
   NALU_HYPRE_Real *x;
   nalu_hypre_Vector *rhs;
   NALU_HYPRE_Real *aux;
   nalu_hypre_CSRMatrix *A;
   nalu_hypre_Vector *x_vector;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(par_A);
   NALU_HYPRE_Int num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   NALU_HYPRE_Int *i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   NALU_HYPRE_Int *j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   NALU_HYPRE_Real *domain_matrixinverse = nalu_hypre_CSRMatrixData(domain_structure);


   NALU_HYPRE_Int piv_counter = 0;
   NALU_HYPRE_Int one = 1;
   char uplo = 'L';
   NALU_HYPRE_Int jj, i, j, k; /*, j_loc, k_loc;*/


   NALU_HYPRE_Int matrix_size, matrix_size_counter = 0;

   NALU_HYPRE_Int num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /* initiate:      ----------------------------------------------- */
   /* num_dofs = nalu_hypre_CSRMatrixNumRows(A); */
   x_vector = nalu_hypre_ParVectorLocalVector(par_x);
   A = nalu_hypre_ParCSRMatrixDiag(par_A);
   i_dof_dof = nalu_hypre_CSRMatrixI(A);
   j_dof_dof = nalu_hypre_CSRMatrixJ(A);
   a_dof_dof = nalu_hypre_CSRMatrixData(A);
   x = nalu_hypre_VectorData(x_vector);
   aux = nalu_hypre_VectorData(aux_vector);
   /* for (i=0; i < num_dofs; i++)
      x[i] = 0.e0; */

   if (num_procs > 1)
   {
      nalu_hypre_parCorrRes(par_A, par_x, rhs_vector, &rhs);
   }
   else
   {
      rhs = rhs_vector;
   }

   /* forward solve: ----------------------------------------------- */

   matrix_size_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         aux[jj] = nalu_hypre_VectorData(rhs)[j_domain_dof[j]];
         for (k = i_dof_dof[j_domain_dof[j]];
              k < i_dof_dof[j_domain_dof[j] + 1]; k++)
         {
            aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
         }
         jj++;
      }
      /* solve for correction: ------------------------------------- */
      if (use_nonsymm)
      {
         nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, &pivots[piv_counter], aux,
                      &matrix_size, &ierr);
      }
      else
      {
         nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, aux,
                      &matrix_size, &ierr);
      }


      if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         x[j_domain_dof[j]] +=  relax_wt * aux[jj++];
      }
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;
   }

   if (num_procs > 1)
   {
      nalu_hypre_SeqVectorDestroy(rhs);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_MPSchwarzCFFWSolve(nalu_hypre_ParCSRMatrix *par_A,
                                   nalu_hypre_Vector *rhs_vector,
                                   nalu_hypre_CSRMatrix *domain_structure,
                                   nalu_hypre_ParVector *par_x,
                                   NALU_HYPRE_Real relax_wt,
                                   nalu_hypre_Vector *aux_vector,
                                   NALU_HYPRE_Int *CF_marker,
                                   NALU_HYPRE_Int rlx_pt, NALU_HYPRE_Int *pivots,
                                   NALU_HYPRE_Int use_nonsymm)
{
   NALU_HYPRE_Int ierr = 0;
   /* NALU_HYPRE_Int num_dofs; */
   NALU_HYPRE_Int *i_dof_dof;
   NALU_HYPRE_Int *j_dof_dof;
   NALU_HYPRE_Real *a_dof_dof;
   NALU_HYPRE_Real *x;
   nalu_hypre_Vector *rhs;
   NALU_HYPRE_Real *aux;
   nalu_hypre_CSRMatrix *A;
   nalu_hypre_Vector *x_vector;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(par_A);
   NALU_HYPRE_Int num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   NALU_HYPRE_Int *i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   NALU_HYPRE_Int *j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   NALU_HYPRE_Real *domain_matrixinverse = nalu_hypre_CSRMatrixData(domain_structure);

   NALU_HYPRE_Int piv_counter = 0;
   NALU_HYPRE_Int one = 1;

   char uplo = 'L';
   NALU_HYPRE_Int jj, i, j, k; /*, j_loc, k_loc;*/


   NALU_HYPRE_Int matrix_size, matrix_size_counter = 0;

   NALU_HYPRE_Int num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /* initiate:      ----------------------------------------------- */
   /* num_dofs = nalu_hypre_CSRMatrixNumRows(A); */
   x_vector = nalu_hypre_ParVectorLocalVector(par_x);
   A = nalu_hypre_ParCSRMatrixDiag(par_A);
   i_dof_dof = nalu_hypre_CSRMatrixI(A);
   j_dof_dof = nalu_hypre_CSRMatrixJ(A);
   a_dof_dof = nalu_hypre_CSRMatrixData(A);
   x = nalu_hypre_VectorData(x_vector);
   aux = nalu_hypre_VectorData(aux_vector);
   /* for (i=0; i < num_dofs; i++)
      x[i] = 0.e0; */

   if (use_nonsymm)
   {
      uplo = 'N';
   }

   if (num_procs > 1)
   {
      nalu_hypre_parCorrRes(par_A, par_x, rhs_vector, &rhs);
   }
   else
   {
      rhs = rhs_vector;
   }

   /* forward solve: ----------------------------------------------- */

   matrix_size_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      if (CF_marker[i] == rlx_pt)
      {
         matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];

         /* compute residual: ---------------------------------------- */

         jj = 0;
         for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
         {
            aux[jj] = nalu_hypre_VectorData(rhs)[j_domain_dof[j]];
            if (CF_marker[j_domain_dof[j]] == rlx_pt)
            {
               for (k = i_dof_dof[j_domain_dof[j]];
                    k < i_dof_dof[j_domain_dof[j] + 1]; k++)
               {
                  aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
               }
            }
            jj++;
         }
         /* solve for correction: ------------------------------------- */
         if (use_nonsymm)
         {
            nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                         &domain_matrixinverse[matrix_size_counter],
                         &matrix_size, &pivots[piv_counter], aux,
                         &matrix_size, &ierr);
         }

         else
         {
            nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                         &domain_matrixinverse[matrix_size_counter],
                         &matrix_size, aux,
                         &matrix_size, &ierr);
         }

         if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
         jj = 0;
         for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
         {
            x[j_domain_dof[j]] +=  relax_wt * aux[jj++];
         }
         matrix_size_counter += matrix_size * matrix_size;
         piv_counter += matrix_size;
      }
   }

   if (num_procs > 1)
   {
      nalu_hypre_SeqVectorDestroy(rhs);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
transpose_matrix_create(  NALU_HYPRE_Int **i_face_element_pointer,
                          NALU_HYPRE_Int **j_face_element_pointer,
                          NALU_HYPRE_Int *i_element_face, NALU_HYPRE_Int *j_element_face,
                          NALU_HYPRE_Int num_elements, NALU_HYPRE_Int num_faces)
{
   /* FILE *f; */
   NALU_HYPRE_Int i, j;

   NALU_HYPRE_Int *i_face_element, *j_face_element;

   /* ======================================================================
      first create face_element graph: -------------------------------------
      ====================================================================== */

   i_face_element = nalu_hypre_TAlloc(NALU_HYPRE_Int, (num_faces + 1), NALU_HYPRE_MEMORY_HOST);
   j_face_element = nalu_hypre_TAlloc(NALU_HYPRE_Int, i_element_face[num_elements], NALU_HYPRE_MEMORY_HOST);


   for (i = 0; i < num_faces; i++)
   {
      i_face_element[i] = 0;
   }

   for (i = 0; i < num_elements; i++)
      for (j = i_element_face[i]; j < i_element_face[i + 1]; j++)
      {
         i_face_element[j_element_face[j]]++;
      }

   i_face_element[num_faces] = i_element_face[num_elements];

   for (i = num_faces - 1; i > -1; i--)
   {
      i_face_element[i] = i_face_element[i + 1] - i_face_element[i];
   }

   for (i = 0; i < num_elements; i++)
      for (j = i_element_face[i]; j < i_element_face[i + 1]; j++)
      {
         j_face_element[i_face_element[j_element_face[j]]] = i;
         i_face_element[j_element_face[j]]++;
      }

   for (i = num_faces - 1; i > -1; i--)
   {
      i_face_element[i + 1] = i_face_element[i];
   }

   i_face_element[0] = 0;

   /* nalu_hypre_printf("end building face--element graph: ++++++++++++++++++\n"); */

   /* END building face_element graph; ================================ */

   *i_face_element_pointer = i_face_element;
   *j_face_element_pointer = j_face_element;

   return 0;

}
NALU_HYPRE_Int
matrix_matrix_product(    NALU_HYPRE_Int **i_element_edge_pointer,
                          NALU_HYPRE_Int **j_element_edge_pointer,
                          NALU_HYPRE_Int *i_element_face, NALU_HYPRE_Int *j_element_face,
                          NALU_HYPRE_Int *i_face_edge, NALU_HYPRE_Int *j_face_edge,
                          NALU_HYPRE_Int num_elements, NALU_HYPRE_Int num_faces, NALU_HYPRE_Int num_edges)
{
   /* FILE *f; */
   NALU_HYPRE_Int i, j, k, l, m;

   NALU_HYPRE_Int i_edge_on_local_list, i_edge_on_list;
   NALU_HYPRE_Int local_element_edge_counter = 0, element_edge_counter = 0;
   NALU_HYPRE_Int *j_local_element_edge;

   NALU_HYPRE_Int *i_element_edge, *j_element_edge;

   j_local_element_edge = nalu_hypre_TAlloc(NALU_HYPRE_Int, (num_edges + 1), NALU_HYPRE_MEMORY_HOST);

   i_element_edge = nalu_hypre_TAlloc(NALU_HYPRE_Int, (num_elements + 1), NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_elements + 1; i++)
   {
      i_element_edge[i] = 0;
   }

   for (i = 0; i < num_elements; i++)
   {
      local_element_edge_counter = 0;
      for (j = i_element_face[i]; j < i_element_face[i + 1]; j++)
      {
         k = j_element_face[j];

         for (l = i_face_edge[k]; l < i_face_edge[k + 1]; l++)
         {
            /* element i  and edge j_face_edge[l] are connected */

            /* nalu_hypre_printf("element %d  contains edge %d;\n",
               i, j_face_edge[l]);  */

            i_edge_on_local_list = -1;
            for (m = 0; m < local_element_edge_counter; m++)
               if (j_local_element_edge[m] == j_face_edge[l])
               {
                  i_edge_on_local_list++;
                  break;
               }

            if (i_edge_on_local_list == -1)
            {
               i_element_edge[i]++;
               j_local_element_edge[local_element_edge_counter] =
                  j_face_edge[l];
               local_element_edge_counter++;
            }
         }
      }
   }

   nalu_hypre_TFree(j_local_element_edge, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_elements; i++)
   {
      i_element_edge[i + 1] += i_element_edge[i];
   }

   for (i = num_elements; i > 0; i--)
   {
      i_element_edge[i] = i_element_edge[i - 1];
   }

   i_element_edge[0] = 0;

   j_element_edge = nalu_hypre_TAlloc(NALU_HYPRE_Int, i_element_edge[num_elements]
                                 , NALU_HYPRE_MEMORY_HOST);

   /* fill--in the actual j_element_edge array: --------------------- */

   element_edge_counter = 0;
   for (i = 0; i < num_elements; i++)
   {
      i_element_edge[i] = element_edge_counter;
      for (j = i_element_face[i]; j < i_element_face[i + 1]; j++)
      {
         for (k = i_face_edge[j_element_face[j]];
              k < i_face_edge[j_element_face[j] + 1]; k++)
         {
            /* check if edge j_face_edge[k] is already on list ***/

            i_edge_on_list = -1;
            for (l = i_element_edge[i];
                 l < element_edge_counter; l++)
               if (j_element_edge[l] == j_face_edge[k])
               {
                  i_edge_on_list++;
                  break;
               }

            if (i_edge_on_list == -1)
            {
               if (element_edge_counter >=
                   i_element_edge[num_elements])
               {
                  nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "error in j_element_edge size: \n");
                  break;
               }

               j_element_edge[element_edge_counter] =
                  j_face_edge[k];
               element_edge_counter++;
            }
         }
      }

   }

   i_element_edge[num_elements] = element_edge_counter;

   /*------------------------------------------------------------------
     f = fopen("element_edge", "w");
     for (i=0; i < num_elements; i++)
     {
     nalu_hypre_printf("\nelement: %d has edges:\n", i);
     for (j=i_element_edge[i]; j < i_element_edge[i+1]; j++)
     {
     nalu_hypre_printf("%d ", j_element_edge[j]);
     nalu_hypre_fprintf(f, "%d %d\n", i, j_element_edge[j]);
     }

     nalu_hypre_printf("\n");
     }

     fclose(f);
   */

   /* nalu_hypre_printf("end element_edge computation: ++++++++++++++++++++++++ \n");*/

   *i_element_edge_pointer = i_element_edge;
   *j_element_edge_pointer = j_element_edge;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGCreateDomainDof:
 *--------------------------------------------------------------------------*/

/*****************************************************************************
 *
 * Routine for constructing graph domain_dof with minimal overlap
 *             and computing the respective matrix inverses to be
 *             used in an overlapping Schwarz procedure (like smoother
 *             in AMG);
 *
 *****************************************************************************/
NALU_HYPRE_Int
nalu_hypre_AMGCreateDomainDof(nalu_hypre_CSRMatrix     *A,
                         NALU_HYPRE_Int domain_type, NALU_HYPRE_Int overlap,
                         NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                         nalu_hypre_CSRMatrix    **domain_structure_pointer,
                         NALU_HYPRE_Int **piv_pointer, NALU_HYPRE_Int use_nonsymm)
{

   NALU_HYPRE_Int *i_domain_dof, *j_domain_dof;
   NALU_HYPRE_Real *domain_matrixinverse;
   NALU_HYPRE_Int num_domains;
   nalu_hypre_CSRMatrix *domain_structure = NULL;

   NALU_HYPRE_Int *i_dof_dof = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int *j_dof_dof = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Real *a_dof_dof = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int num_dofs = nalu_hypre_CSRMatrixNumRows(A);

   /* NALU_HYPRE_Int *i_dof_to_accept_weight; */
   NALU_HYPRE_Int *i_dof_to_prefer_weight,
             *w_dof_dof, *i_dof_weight;
   NALU_HYPRE_Int *i_dof_to_aggregate, *i_aggregate_dof, *j_aggregate_dof;

   NALU_HYPRE_Int *i_dof_index;

   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int i, j, k,  l_loc, i_loc, j_loc;
   NALU_HYPRE_Int i_dof;
   NALU_HYPRE_Int *i_local_to_global;
   NALU_HYPRE_Int *i_global_to_local;

   NALU_HYPRE_Int local_dof_counter, max_local_dof_counter = 0;

   NALU_HYPRE_Int domain_dof_counter = 0, domain_matrixinverse_counter = 0;
   NALU_HYPRE_Int nf;

   NALU_HYPRE_Real *AE;

   NALU_HYPRE_Int piv_counter = 0;
   NALU_HYPRE_Int *ipiv;
   NALU_HYPRE_Int *piv = NULL;
   char uplo = 'L';
   NALU_HYPRE_Int cnt;

   /* --------------------------------------------------------------------- */

   /*=======================================================================*/
   /*    create artificial domains by agglomeration;                        */
   /*=======================================================================*/

   /*nalu_hypre_printf("----------- create artificials domain by agglomeration;  ======\n");
    */

   if (num_dofs == 0)
   {
      *domain_structure_pointer = domain_structure;

      *piv_pointer = piv;

      return nalu_hypre_error_flag;
   }

   i_aggregate_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_dofs + 1, NALU_HYPRE_MEMORY_HOST);
   j_aggregate_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_dofs, NALU_HYPRE_MEMORY_HOST);

   if (domain_type == 2)
   {
      i_dof_to_prefer_weight = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_dofs, NALU_HYPRE_MEMORY_HOST);
      w_dof_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int, i_dof_dof[num_dofs], NALU_HYPRE_MEMORY_HOST);
      i_dof_weight = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_dofs, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_dofs; i++)
         for (j = i_dof_dof[i]; j < i_dof_dof[i + 1]; j++)
         {
            if (j_dof_dof[j] == i)
            {
               w_dof_dof[j] = 0;
            }
            else
            {
               w_dof_dof[j] = 1;
            }
         }

      /*nalu_hypre_printf("end computing weights for agglomeration procedure: --------\n");
       */
      nalu_hypre_AMGeAgglomerate(i_aggregate_dof, j_aggregate_dof,
                            i_dof_dof, j_dof_dof, w_dof_dof,
                            i_dof_dof, j_dof_dof,
                            i_dof_dof, j_dof_dof,
                            i_dof_to_prefer_weight,
                            i_dof_weight,
                            num_dofs, num_dofs,
                            &num_domains);

      nalu_hypre_TFree(i_dof_to_prefer_weight, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_weight, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(w_dof_dof, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      nf = 1;
      if (domain_type == 1) { nf = num_functions; }

      num_domains = num_dofs / nf;
      for (i = 0; i < num_domains + 1; i++)
      {
         i_aggregate_dof[i] = nf * i;
      }
      for (i = 0; i < num_dofs; i++)
      {
         j_aggregate_dof[i] = i;
      }
   }
   /*nalu_hypre_printf("num_dofs: %d, num_domains: %d\n", num_dofs, num_domains);*/


   /*
     nalu_hypre_printf("========================================================\n");
     nalu_hypre_printf("== artificial non--overlapping domains (aggregates): ===\n");
     nalu_hypre_printf("========================================================\n");


     for (i=0; i < num_domains; i++)
     {
     nalu_hypre_printf("\n aggregate %d:\n", i);
     for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
     nalu_hypre_printf("%d, ", j_aggregate_dof[j]);

     nalu_hypre_printf("\n");
     }
   */

   /* make domains from aggregates: *********************************/

   if (overlap == 1)
   {
      i_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_domains + 1, NALU_HYPRE_MEMORY_HOST);

      i_dof_index = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_dofs, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_dofs; i++)
      {
         i_dof_index[i] = -1;
      }

      i_dof_to_aggregate = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_dofs, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_domains; i++)
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            i_dof_to_aggregate[j_aggregate_dof[j]] = i;
         }

      domain_dof_counter = 0;
      for (i = 0; i < num_domains; i++)
      {
         i_domain_dof[i] =  domain_dof_counter;
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            i_dof_index[j_aggregate_dof[j]] = -1;
         }

         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
            for (k = i_dof_dof[j_aggregate_dof[j]];
                 k < i_dof_dof[j_aggregate_dof[j] + 1]; k++)
               if (i_dof_to_aggregate[j_dof_dof[k]] >= i
                   && i_dof_index[j_dof_dof[k]] == -1)
               {
                  i_dof_index[j_dof_dof[k]]++;
                  domain_dof_counter++;
               }
      }

      i_domain_dof[num_domains] =  domain_dof_counter;
      j_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int, domain_dof_counter, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_dofs; i++)
      {
         i_dof_index[i] = -1;
      }

      domain_dof_counter = 0;
      for (i = 0; i < num_domains; i++)
      {
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            i_dof_index[j_aggregate_dof[j]] = -1;
         }

         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
            for (k = i_dof_dof[j_aggregate_dof[j]];
                 k < i_dof_dof[j_aggregate_dof[j] + 1]; k++)
               if (i_dof_to_aggregate[j_dof_dof[k]] >= i
                   && i_dof_index[j_dof_dof[k]] == -1)
               {
                  i_dof_index[j_dof_dof[k]]++;
                  j_domain_dof[domain_dof_counter] = j_dof_dof[k];
                  domain_dof_counter++;
               }
      }

      nalu_hypre_TFree(i_aggregate_dof, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_aggregate_dof, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_to_aggregate, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_index, NALU_HYPRE_MEMORY_HOST);
   }
   else if (overlap == 2)
   {
      i_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_domains + 1, NALU_HYPRE_MEMORY_HOST);

      i_dof_index = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_dofs, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_dofs; i++)
      {
         i_dof_index[i] = -1;
      }

      domain_dof_counter = 0;
      for (i = 0; i < num_domains; i++)
      {
         i_domain_dof[i] =  domain_dof_counter;
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
            for (k = i_dof_dof[j_aggregate_dof[j]];
                 k < i_dof_dof[j_aggregate_dof[j] + 1]; k++)
               if (i_dof_index[j_dof_dof[k]] == -1)
               {
                  i_dof_index[j_dof_dof[k]]++;
                  domain_dof_counter++;
               }

         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
            for (k = i_dof_dof[j_aggregate_dof[j]];
                 k < i_dof_dof[j_aggregate_dof[j] + 1]; k++)
            {
               i_dof_index[j_dof_dof[k]] = -1;
            }

      }

      for (i = 0; i < num_dofs; i++)
      {
         i_dof_index[i] = -1;
      }

      i_domain_dof[num_domains] =  domain_dof_counter;
      j_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int, domain_dof_counter, NALU_HYPRE_MEMORY_HOST);

      domain_dof_counter = 0;
      for (i = 0; i < num_domains; i++)
      {
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
            for (k = i_dof_dof[j_aggregate_dof[j]];
                 k < i_dof_dof[j_aggregate_dof[j] + 1]; k++)
               if ( i_dof_index[j_dof_dof[k]] == -1)
               {
                  i_dof_index[j_dof_dof[k]]++;
                  j_domain_dof[domain_dof_counter] = j_dof_dof[k];
                  domain_dof_counter++;
               }

         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
            for (k = i_dof_dof[j_aggregate_dof[j]];
                 k < i_dof_dof[j_aggregate_dof[j] + 1]; k++)
            {
               i_dof_index[j_dof_dof[k]] = -1;
            }
      }

      nalu_hypre_TFree(i_aggregate_dof, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_aggregate_dof, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_index, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      i_domain_dof = i_aggregate_dof;
      j_domain_dof = j_aggregate_dof;
   }

   /*nalu_hypre_printf("END domain_dof computations: =================================\n");
    */
   domain_matrixinverse_counter = 0;
   local_dof_counter = 0;
   piv_counter = 0;

   for (i = 0; i < num_domains; i++)
   {
      local_dof_counter = i_domain_dof[i + 1] - i_domain_dof[i];
      domain_matrixinverse_counter += local_dof_counter * local_dof_counter;
      piv_counter += local_dof_counter;

      if (local_dof_counter > max_local_dof_counter)
      {
         max_local_dof_counter = local_dof_counter;
      }
   }

   domain_matrixinverse = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  domain_matrixinverse_counter, NALU_HYPRE_MEMORY_HOST);
   if (use_nonsymm)
   {
      piv = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  piv_counter, NALU_HYPRE_MEMORY_HOST);
   }

   i_local_to_global = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_local_dof_counter, NALU_HYPRE_MEMORY_HOST);

   i_global_to_local = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_dofs, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_dofs; i++)
   {
      i_global_to_local[i] = -1;
   }

   piv_counter = 0;
   domain_matrixinverse_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      local_dof_counter = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         i_global_to_local[j_domain_dof[j]] = local_dof_counter;
         i_local_to_global[local_dof_counter] = j_domain_dof[j];
         local_dof_counter++;
      }

      /* get local matrix in AE: ======================================== */
      cnt = 0;

      AE = &domain_matrixinverse[domain_matrixinverse_counter];
      ipiv = &piv[piv_counter];
      for (i_loc = 0; i_loc < local_dof_counter; i_loc++)
         for (j_loc = 0; j_loc < local_dof_counter; j_loc++)
         {
            AE[cnt++] = 0.e0;
         }

      for (i_loc = 0; i_loc < local_dof_counter; i_loc++)
      {
         i_dof = i_local_to_global[i_loc];
         for (j = i_dof_dof[i_dof]; j < i_dof_dof[i_dof + 1]; j++)
         {
            j_loc = i_global_to_local[j_dof_dof[j]];
            if (j_loc >= 0)
            {
               AE[i_loc + j_loc * local_dof_counter] = a_dof_dof[j];
            }
         }
      }

      if (use_nonsymm)
      {
         nalu_hypre_dgetrf(&local_dof_counter,
                      &local_dof_counter, AE,
                      &local_dof_counter, ipiv, &ierr);
         piv_counter += local_dof_counter;
      }

      else
      {
         nalu_hypre_dpotrf(&uplo, &local_dof_counter, AE,
                      &local_dof_counter, &ierr);
      }

      domain_matrixinverse_counter += local_dof_counter * local_dof_counter;

      for (l_loc = 0; l_loc < local_dof_counter; l_loc++)
      {
         i_global_to_local[i_local_to_global[l_loc]] = -1;
      }

   }

   nalu_hypre_TFree(i_local_to_global, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(i_global_to_local, NALU_HYPRE_MEMORY_HOST);

   domain_structure = nalu_hypre_CSRMatrixCreate(num_domains, max_local_dof_counter,
                                            i_domain_dof[num_domains]);

   nalu_hypre_CSRMatrixMemoryLocation(domain_structure) = NALU_HYPRE_MEMORY_HOST;

   nalu_hypre_CSRMatrixI(domain_structure) = i_domain_dof;
   nalu_hypre_CSRMatrixJ(domain_structure) = j_domain_dof;
   nalu_hypre_CSRMatrixData(domain_structure) = domain_matrixinverse;

   *domain_structure_pointer = domain_structure;

   *piv_pointer = piv;

   return nalu_hypre_error_flag;
}

/* unacceptable faces: i_face_to_prefer_weight[] = -1; ------------------*/

NALU_HYPRE_Int nalu_hypre_AMGeAgglomerate(NALU_HYPRE_Int *i_AE_element, NALU_HYPRE_Int *j_AE_element,
                                NALU_HYPRE_Int *i_face_face, NALU_HYPRE_Int *j_face_face, NALU_HYPRE_Int *w_face_face,
                                NALU_HYPRE_Int *i_face_element, NALU_HYPRE_Int *j_face_element,
                                NALU_HYPRE_Int *i_element_face, NALU_HYPRE_Int *j_element_face,
                                NALU_HYPRE_Int *i_face_to_prefer_weight,
                                NALU_HYPRE_Int *i_face_weight,
                                NALU_HYPRE_Int num_faces, NALU_HYPRE_Int num_elements,
                                NALU_HYPRE_Int *num_AEs_pointer)
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int i, j, k, l;

   NALU_HYPRE_Int face_to_eliminate;
   NALU_HYPRE_Int max_weight_old, max_weight;

   NALU_HYPRE_Int AE_counter = 0, AE_element_counter = 0;

   /* NALU_HYPRE_Int i_element_face_counter; */

   NALU_HYPRE_Int *i_element_to_AE;

   NALU_HYPRE_Int *previous, *next, *first;
   NALU_HYPRE_Int head, tail, last;

   NALU_HYPRE_Int face_max_weight, face_local_max_weight, preferred_weight;

   NALU_HYPRE_Int weight, weight_max;

   max_weight = 1;
   for (i = 0; i < num_faces; i++)
   {
      weight = 1;
      for (j = i_face_face[i]; j < i_face_face[i + 1]; j++)
      {
         weight += w_face_face[j];
      }
      if (max_weight < weight) { max_weight = weight; }
   }

   first = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_weight + 1, NALU_HYPRE_MEMORY_HOST);

   next = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_faces, NALU_HYPRE_MEMORY_HOST);

   previous = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_faces + 1, NALU_HYPRE_MEMORY_HOST);

   tail = num_faces;
   head = -1;

   for (i = 0; i < num_faces; i++)
   {
      next[i] = i + 1;
      previous[i] = i - 1;
   }

   last = num_faces - 1;
   previous[tail] = last;

   for (weight = 1; weight <= max_weight; weight++)
   {
      first[weight] = tail;
   }

   i_element_to_AE = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_elements, NALU_HYPRE_MEMORY_HOST);

   /*=======================================================================
     AGGLOMERATION PROCEDURE:
     ======================================================================= */

   for (k = 0; k < num_elements; k++)
   {
      i_element_to_AE[k] = -1;
   }

   for (k = 0; k < num_faces; k++)
   {
      i_face_weight[k] = 1;
   }

   first[0] = 0;
   first[1] = 0;

   last = previous[tail];
   weight_max = i_face_weight[last];

   k = last;
   face_max_weight = -1;
   while (k != head)
   {
      if (i_face_to_prefer_weight[k] > -1)
      {
         face_max_weight = k;
      }

      if (face_max_weight > -1) { break; }

      k = previous[k];
   }

   /* this will be used if the faces have been sorted: *****************
      k = last;
      face_max_weight = -1;
      while (k != head)
      {
      if (i_face_to_prefer_weight[k] > -1)
      face_max_weight = k;


      if (face_max_weight > -1)
      {
      max_weight = i_face_weight[face_max_weight];
      l = face_max_weight;

      while (previous[l] != head)
      {

      if (i_face_weight[previous[l]] < max_weight)
      break;
      else
      if (i_face_to_prefer_weight[previous[l]] >
      i_face_to_prefer_weight[face_max_weight])
      {
      l = previous[l];
      face_max_weight = l;
      }
      else
      l = previous[l];
      }

      break;
      }

      l =previous[k];

      weight = i_face_weight[k];
      last = previous[tail];
      if (last == head)
      weight_max = 0;
      else
      weight_max = i_face_weight[last];

      ierr = nalu_hypre_remove_entry(weight, &weight_max,
      previous, next, first, &last,
      head, tail,
      k);

      k=l;
      }
   */

   if (face_max_weight == -1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "all faces are unacceptable, i.e., no faces to eliminate !\n");

      *num_AEs_pointer = 1;

      i_AE_element[0] = 0;
      for (i = 0; i < num_elements; i++)
      {
         i_element_to_AE[i] = 0;
         j_AE_element[i] = i;
      }

      i_AE_element[1] = num_elements;

      return nalu_hypre_error_flag;
   }

   for (k = 0; k < num_faces; k++)
      if (i_face_to_prefer_weight[k] > i_face_to_prefer_weight[face_max_weight])
      {
         face_max_weight = k;
      }

   max_weight = i_face_weight[face_max_weight];

   AE_counter = 0;
   AE_element_counter = 0;

   i_AE_element[AE_counter] = AE_element_counter;

   max_weight_old = -1;

   face_local_max_weight = face_max_weight;

eliminate_face:

   face_to_eliminate = face_local_max_weight;

   max_weight = i_face_weight[face_to_eliminate];

   last = previous[tail];
   if (last == head)
   {
      weight_max = 0;
   }
   else
   {
      weight_max = i_face_weight[last];
   }

   ierr = nalu_hypre_remove_entry(max_weight, &weight_max,
                             previous, next, first, &last,
                             head, tail,
                             face_to_eliminate);

   i_face_weight[face_to_eliminate] = 0;

   /*----------------------------------------------------------
    *  agglomeration step:
    *
    *  put on AE_element -- list all elements
    *  that share face "face_to_eliminate";
    *----------------------------------------------------------*/

   for (k = i_face_element[face_to_eliminate];
        k < i_face_element[face_to_eliminate + 1]; k++)
   {
      /* check if element j_face_element[k] is already on the list: */

      if (j_face_element[k] < num_elements)
      {
         if (i_element_to_AE[j_face_element[k]] == -1)
         {
            j_AE_element[AE_element_counter] = j_face_element[k];
            i_element_to_AE[j_face_element[k]] = AE_counter;
            AE_element_counter++;
         }
      }
   }

   /* local update & search:==================================== */

   for (j = i_face_face[face_to_eliminate];
        j < i_face_face[face_to_eliminate + 1]; j++)
      if (i_face_weight[j_face_face[j]] > 0)
      {
         weight = i_face_weight[j_face_face[j]];


         last = previous[tail];
         if (last == head)
         {
            weight_max = 0;
         }
         else
         {
            weight_max = i_face_weight[last];
         }

         ierr = nalu_hypre_move_entry(weight, &weight_max,
                                 previous, next, first, &last,
                                 head, tail,
                                 j_face_face[j]);

         i_face_weight[j_face_face[j]] += w_face_face[j];

         weight = i_face_weight[j_face_face[j]];

         /* nalu_hypre_printf("update entry: %d\n", j_face_face[j]);  */

         last = previous[tail];
         if (last == head)
         {
            weight_max = 0;
         }
         else
         {
            weight_max = i_face_weight[last];
         }

         ierr = nalu_hypre_update_entry(weight, &weight_max,
                                   previous, next, first, &last,
                                   head, tail,
                                   j_face_face[j]);

         last = previous[tail];
         if (last == head)
         {
            weight_max = 0;
         }
         else
         {
            weight_max = i_face_weight[last];
         }
      }

   /* find a face of the elements that have already been agglomerated
      with a maximal weight: ====================================== */

   max_weight_old = max_weight;

   face_local_max_weight = -1;
   preferred_weight = -1;

   for (l = i_AE_element[AE_counter];
        l < AE_element_counter; l++)
   {
      for (j = i_element_face[j_AE_element[l]];
           j < i_element_face[j_AE_element[l] + 1]; j++)
      {
         i = j_element_face[j];

         if (max_weight_old > 1 && i_face_weight[i] > 0 &&
             i_face_to_prefer_weight[i] > -1)
         {
            if ( max_weight < i_face_weight[i])
            {
               face_local_max_weight = i;
               max_weight = i_face_weight[i];
               preferred_weight = i_face_to_prefer_weight[i];
            }

            if ( max_weight == i_face_weight[i]
                 && i_face_to_prefer_weight[i] > preferred_weight)
            {
               face_local_max_weight = i;
               preferred_weight = i_face_to_prefer_weight[i];
            }
         }
      }
   }

   if (face_local_max_weight > -1) { goto eliminate_face; }

   /* ----------------------------------------------------------------
    * eliminate and label with i_face_weight[ ] = -1
    * "boundary faces of agglomerated elements";
    * those faces will be preferred for the next coarse spaces
    * in case multiple coarse spaces are to be built;
    * ---------------------------------------------------------------*/

   for (k = i_AE_element[AE_counter]; k < AE_element_counter; k++)
   {
      for (j = i_element_face[j_AE_element[k]];
           j < i_element_face[j_AE_element[k] + 1]; j++)
      {
         if (i_face_weight[j_element_face[j]] > 0)
         {
            weight = i_face_weight[j_element_face[j]];
            last = previous[tail];
            if (last == head)
            {
               weight_max = 0;
            }
            else
            {
               weight_max = i_face_weight[last];
            }


            ierr = nalu_hypre_remove_entry(weight, &weight_max,
                                      previous, next, first, &last,
                                      head, tail,
                                      j_element_face[j]);

            i_face_weight[j_element_face[j]] = -1;

         }
      }
   }

   if (AE_element_counter > i_AE_element[AE_counter])
   {
      /* nalu_hypre_printf("completing agglomerated element: %d\n",
         AE_counter);   */
      AE_counter++;
   }

   i_AE_element[AE_counter] = AE_element_counter;

   /* find a face with maximal weight: ---------------------------*/

   last = previous[tail];
   if (last == head) { goto end_agglomerate; }

   weight_max = i_face_weight[last];

   /* nalu_hypre_printf("global search: ======================================\n"); */

   face_max_weight = -1;

   k = last;
   while (k != head)
   {
      if (i_face_to_prefer_weight[k] > -1)
      {
         face_max_weight = k;
      }

      if (face_max_weight > -1)
      {
         max_weight = i_face_weight[face_max_weight];
         l = face_max_weight;

         while (previous[l] != head)
         {

            if (i_face_weight[previous[l]] < max_weight)
            {
               break;
            }
            else if (i_face_to_prefer_weight[previous[l]] >
                     i_face_to_prefer_weight[face_max_weight])
            {
               l = previous[l];
               face_max_weight = l;
            }
            else
            {
               l = previous[l];
            }
         }

         break;
      }

      l = previous[k];
      /* remove face k: ---------------------------------------*/

      weight = i_face_weight[k];
      last = previous[tail];
      if (last == head)
      {
         weight_max = 0;
      }
      else
      {
         weight_max = i_face_weight[last];
      }


      ierr = nalu_hypre_remove_entry(weight, &weight_max,
                                previous, next, first, &last,
                                head, tail,
                                k);

      /* i_face_weight[k] = -1; */

      k = l;
   }

   if (face_max_weight == -1) { goto end_agglomerate; }

   max_weight = i_face_weight[face_max_weight];

   face_local_max_weight = face_max_weight;

   goto eliminate_face;

end_agglomerate:

   /* eliminate isolated elements: ----------------------------------*/

   for (i = 0; i < num_elements; i++)
   {

      if (i_element_to_AE[i] == -1)
      {
         for (j = i_element_face[i]; j < i_element_face[i + 1]
              && i_element_to_AE[i] == -1; j++)
            if (i_face_to_prefer_weight[j_element_face[j]] > -1)
               for (k = i_face_element[j_element_face[j]];
                    k < i_face_element[j_element_face[j] + 1]
                    && i_element_to_AE[i] == -1; k++)
                  if (i_element_to_AE[j_face_element[k]] != -1)
                  {
                     i_element_to_AE[i] = i_element_to_AE[j_face_element[k]];
                  }
      }

      /*
        if (i_element_to_AE[i] == -1)
        {
        i_element_face_counter = 0;
        for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
        if (i_face_to_prefer_weight[j_element_face[j]] > -1)
        i_element_face_counter++;

        if (i_element_face_counter == 1)
        {
        for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
        if (i_face_to_prefer_weight[j_element_face[j]] > -1)
        for (k=i_face_element[j_element_face[j]];
        k<i_face_element[j_element_face[j]+1]; k++)
        if (i_element_to_AE[j_face_element[k]] != -1)
        i_element_to_AE[i] = i_element_to_AE[j_face_element[k]];
        }
        }
      */

      if (i_element_to_AE[i] == -1)
      {
         i_element_to_AE[i] = AE_counter;
         AE_counter++;
      }
   }

   num_AEs_pointer[0] = AE_counter;

   /* compute adjoint graph: -------------------------------------------*/

   for (i = 0; i < AE_counter; i++)
   {
      i_AE_element[i] = 0;
   }

   for (i = 0; i < num_elements; i++)
   {
      i_AE_element[i_element_to_AE[i]]++;
   }

   i_AE_element[AE_counter] = num_elements;

   for (i = AE_counter - 1; i > -1; i--)
   {
      i_AE_element[i] = i_AE_element[i + 1] - i_AE_element[i];
   }

   for (i = 0; i < num_elements; i++)
   {
      j_AE_element[i_AE_element[i_element_to_AE[i]]] = i;
      i_AE_element[i_element_to_AE[i]]++;
   }

   for (i = AE_counter - 1; i > -1; i--)
   {
      i_AE_element[i + 1] = i_AE_element[i];
   }

   i_AE_element[0] = 0;

   /*--------------------------------------------------------------------*/
   for (i = 0; i < num_faces; i++)
      if (i_face_to_prefer_weight[i] == -1) { i_face_weight[i] = -1; }


   nalu_hypre_TFree(i_element_to_AE, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(previous, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(next, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(first, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int nalu_hypre_update_entry(NALU_HYPRE_Int weight, NALU_HYPRE_Int *weight_max,
                             NALU_HYPRE_Int *previous, NALU_HYPRE_Int *next, NALU_HYPRE_Int *first, NALU_HYPRE_Int *last,
                             NALU_HYPRE_Int head, NALU_HYPRE_Int tail,
                             NALU_HYPRE_Int i)

{
   NALU_HYPRE_Int weight0;

   if (previous[i] != head) { next[previous[i]] = next[i]; }
   previous[next[i]] = previous[i];

   if (first[weight] == tail)
   {
      if (weight <= weight_max[0])
      {
         nalu_hypre_printf("ERROR IN UPDATE_ENTRY: ===================\n");
         nalu_hypre_printf("weight: %d, weight_max: %d\n",
                      weight, weight_max[0]);
         return -1;
      }
      for (weight0 = weight_max[0] + 1; weight0 <= weight; weight0++)
      {
         first[weight0] = i;
         /* nalu_hypre_printf("create first[%d] = %d\n", weight0, i); */
      }

      previous[i] = previous[tail];
      next[i] = tail;
      if (previous[tail] > head)
      {
         next[previous[tail]] = i;
      }
      previous[tail] = i;

   }
   else
      /* first[weight] already exists: =====================*/
   {
      previous[i] = previous[first[weight]];
      next[i] = first[weight];

      if (previous[first[weight]] != head)
      {
         next[previous[first[weight]]] = i;
      }

      previous[first[weight]] = i;

      for (weight0 = 1; weight0 <= weight; weight0++)
         if (first[weight0] == first[weight])
         {
            first[weight0] = i;
         }

   }

   return 0;
}

NALU_HYPRE_Int nalu_hypre_remove_entry(NALU_HYPRE_Int weight, NALU_HYPRE_Int *weight_max,
                             NALU_HYPRE_Int *previous, NALU_HYPRE_Int *next, NALU_HYPRE_Int *first, NALU_HYPRE_Int *last,
                             NALU_HYPRE_Int head, NALU_HYPRE_Int tail,
                             NALU_HYPRE_Int i)
{
   NALU_HYPRE_Int weight0;

   if (previous[i] != head) { next[previous[i]] = next[i]; }
   previous[next[i]] = previous[i];

   for (weight0 = 1; weight0 <= weight_max[0]; weight0++)
   {
      /* nalu_hypre_printf("first[%d}: %d\n", weight0,  first[weight0]); */
      if (first[weight0] == i)
      {
         first[weight0] = next[i];
         /* nalu_hypre_printf("shift: first[%d]= %d to %d\n",
            weight0, i, next[i]);
            if (i == last[0])
            nalu_hypre_printf("i= last[0]: %d\n", i); */
      }
   }

   next[i] = i;
   previous[i] = i;

   return 0;
}

NALU_HYPRE_Int nalu_hypre_move_entry(NALU_HYPRE_Int weight, NALU_HYPRE_Int *weight_max,
                           NALU_HYPRE_Int *previous, NALU_HYPRE_Int *next, NALU_HYPRE_Int *first, NALU_HYPRE_Int *last,
                           NALU_HYPRE_Int head, NALU_HYPRE_Int tail,
                           NALU_HYPRE_Int i)
{
   NALU_HYPRE_Int  weight0;

   if (previous[i] != head) { next[previous[i]] = next[i]; }
   previous[next[i]] = previous[i];

   for (weight0 = 1; weight0 <= weight_max[0]; weight0++)
   {
      if (first[weight0] == i)
      {
         first[weight0] = next[i];
      }
   }

   return 0;
}

/*---------------------------------------------------------------------
  nalu_hypre_matinv:  X <--  A**(-1) ;  A IS POSITIVE DEFINITE (non--symmetric);
  ---------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_matinv(NALU_HYPRE_Real *x, NALU_HYPRE_Real *a, NALU_HYPRE_Int k)
{
   NALU_HYPRE_Int i, j, l, ierr = 0;

   for (i = 0; i < k; i++)
   {
      if (a[i + i * k] <= 0.e0)
      {
         if (i < k - 1)
         {
            /*********
                      nalu_hypre_printf("indefinite singular matrix in *** matinv ***:\n");
                      nalu_hypre_printf("i:%d;  diagonal entry: %e\n", i, a[i+k*i]);
            */
            ierr = -1;
         }

         a[i + i * k] = 0.e0;
      }
      else
      {
         a[i + k * i] = 1.0 / a[i + i * k];
      }

      for (j = 1; j < k - i; j++)
      {
         for (l = 1; l < k - i; l++)
         {
            a[i + l + k * (i + j)] -= a[i + l + k * i] * a[i + k * i] * a[i + k * (i + j)];
         }
      }

      for (j = 1; j < k - i; j++)
      {
         a[i + j + k * i] = a[i + j + k * i] * a[i + k * i];
         a[i + k * (i + j)] = a[i + k * (i + j)] * a[i + k * i];
      }
   }

   /* FULL INVERSION: --------------------------------------------*/

   x[k * k - 1] = a[k * k - 1];
   for (i = k - 1; i > -1; i--)
   {
      for (j = 1; j < k - i; j++)
      {
         x[i + j + k * i] = 0;
         x[i + k * (i + j)] = 0;

         for (l = 1; l < k - i; l++)
         {
            x[i + j + k * i] -= x[i + j + k * (i + l)] * a[i + l + k * i];
            x[i + k * (i + j)] -= a[i + k * (i + l)] * x[i + l + k * (i + j)];
         }
      }

      x[i + k * i] = a[i + k * i];
      for (j = 1; j < k - i; j++)
      {
         x[i + k * i] -= x[i + k * (i + j)] * a[i + j + k * i];
      }
   }

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_parCorrRes( nalu_hypre_ParCSRMatrix *A,
                  nalu_hypre_ParVector    *x,
                  nalu_hypre_Vector       *rhs,
                  nalu_hypre_Vector      **tmp_ptr )
{
   NALU_HYPRE_Int i, j, index, start;
   NALU_HYPRE_Int num_sends, num_cols_offd;
   NALU_HYPRE_Int local_size;
   NALU_HYPRE_Real *x_buf_data, *x_tmp_data, *x_local_data;
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_ParCSRCommPkg *comm_pkg;
   nalu_hypre_CSRMatrix *offd;
   nalu_hypre_Vector *x_local, *x_tmp, *tmp_vector;
   nalu_hypre_ParCSRCommHandle *comm_handle;

   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   offd = nalu_hypre_ParCSRMatrixOffd(A);
   num_cols_offd = nalu_hypre_CSRMatrixNumCols(offd);

   x_local = nalu_hypre_ParVectorLocalVector(x);
   x_local_data = nalu_hypre_VectorData(x_local);
   local_size = nalu_hypre_VectorSize(x_local);

   if (num_cols_offd)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      x_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,
                                 nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends), NALU_HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            x_buf_data[index++]
               = x_local_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      x_tmp = nalu_hypre_SeqVectorCreate(num_cols_offd);
      nalu_hypre_SeqVectorInitialize_v2(x_tmp, memory_location);
      x_tmp_data = nalu_hypre_VectorData(x_tmp);

      comm_handle = nalu_hypre_ParCSRCommHandleCreate( 1, comm_pkg, x_buf_data,
                                                  x_tmp_data);

      tmp_vector = nalu_hypre_SeqVectorCreate(local_size);
      nalu_hypre_SeqVectorInitialize_v2(tmp_vector, memory_location);
      nalu_hypre_SeqVectorCopy(rhs, tmp_vector);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      nalu_hypre_CSRMatrixMatvec(-1.0, offd, x_tmp, 1.0, tmp_vector);

      nalu_hypre_SeqVectorDestroy(x_tmp);
      nalu_hypre_TFree(x_buf_data, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      tmp_vector = nalu_hypre_SeqVectorCreate(local_size);
      nalu_hypre_SeqVectorInitialize_v2(tmp_vector, memory_location);
      nalu_hypre_SeqVectorCopy(rhs, tmp_vector);
   }

   *tmp_ptr = tmp_vector;

   return 0;
}


NALU_HYPRE_Int nalu_hypre_AdSchwarzSolve(nalu_hypre_ParCSRMatrix *par_A,
                               nalu_hypre_ParVector *par_rhs,
                               nalu_hypre_CSRMatrix *domain_structure,
                               NALU_HYPRE_Real *scale,
                               nalu_hypre_ParVector *par_x,
                               nalu_hypre_ParVector *par_aux, NALU_HYPRE_Int *pivots,
                               NALU_HYPRE_Int use_nonsymm)
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Real *x;
   NALU_HYPRE_Real *aux;
   NALU_HYPRE_Real *tmp;
   nalu_hypre_Vector *x_vector;
   nalu_hypre_Vector *aux_vector;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(par_A);
   NALU_HYPRE_Int num_domains;
   NALU_HYPRE_Int max_domain_size;
   NALU_HYPRE_Int *i_domain_dof;
   NALU_HYPRE_Int *j_domain_dof;
   NALU_HYPRE_Real *domain_matrixinverse;

   NALU_HYPRE_Int piv_counter = 0;
   NALU_HYPRE_Int one = 1;
   char uplo = 'L';

   NALU_HYPRE_Int jj, i, j; /*, j_loc, k_loc;*/


   NALU_HYPRE_Int matrix_size, matrix_size_counter = 0;

   NALU_HYPRE_Int num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /* initiate:      ----------------------------------------------- */
   x_vector = nalu_hypre_ParVectorLocalVector(par_x);
   aux_vector = nalu_hypre_ParVectorLocalVector(par_aux);
   x = nalu_hypre_VectorData(x_vector);
   aux = nalu_hypre_VectorData(aux_vector);
   num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   max_domain_size = nalu_hypre_CSRMatrixNumCols(domain_structure);
   i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   domain_matrixinverse = nalu_hypre_CSRMatrixData(domain_structure);

   if (use_nonsymm)
   {
      uplo = 'N';
   }

   nalu_hypre_ParVectorCopy(par_rhs, par_aux);
   nalu_hypre_ParCSRMatrixMatvec(-1.0, par_A, par_x, 1.0, par_aux);
   tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Real, max_domain_size, NALU_HYPRE_MEMORY_HOST);

   /* forward solve: ----------------------------------------------- */

   matrix_size_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         tmp[jj] = aux[j_domain_dof[j]];
         jj++;
      }
      /* solve for correction: ------------------------------------- */
      if (use_nonsymm)
      {
         nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, &pivots[piv_counter], tmp,
                      &matrix_size, &ierr);
      }
      else
      {
         nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, tmp,
                      &matrix_size, &ierr);
      }

      if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         x[j_domain_dof[j]] +=  scale[j_domain_dof[j]] * tmp[jj++];
      }
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;

   }

   nalu_hypre_TFree(tmp, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_AdSchwarzCFSolve(nalu_hypre_ParCSRMatrix *par_A,
                                 nalu_hypre_ParVector *par_rhs,
                                 nalu_hypre_CSRMatrix *domain_structure,
                                 NALU_HYPRE_Real *scale,
                                 nalu_hypre_ParVector *par_x,
                                 nalu_hypre_ParVector *par_aux,
                                 NALU_HYPRE_Int *CF_marker,
                                 NALU_HYPRE_Int rlx_pt, NALU_HYPRE_Int *pivots,
                                 NALU_HYPRE_Int use_nonsymm)
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Real *x;
   NALU_HYPRE_Real *aux;
   NALU_HYPRE_Real *tmp;
   nalu_hypre_Vector *x_vector;
   nalu_hypre_Vector *aux_vector;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(par_A);
   NALU_HYPRE_Int num_domains;
   NALU_HYPRE_Int max_domain_size;
   NALU_HYPRE_Int *i_domain_dof;
   NALU_HYPRE_Int *j_domain_dof;
   NALU_HYPRE_Real *domain_matrixinverse;

   NALU_HYPRE_Int piv_counter = 0;
   NALU_HYPRE_Int one = 1;

   char uplo = 'L';
   NALU_HYPRE_Int jj, i, j; /*, j_loc, k_loc;*/

   NALU_HYPRE_Int matrix_size, matrix_size_counter = 0;

   NALU_HYPRE_Int num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /* initiate:      ----------------------------------------------- */
   x_vector = nalu_hypre_ParVectorLocalVector(par_x);
   aux_vector = nalu_hypre_ParVectorLocalVector(par_aux);
   x = nalu_hypre_VectorData(x_vector);
   aux = nalu_hypre_VectorData(aux_vector);
   num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   max_domain_size = nalu_hypre_CSRMatrixNumCols(domain_structure);
   i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   domain_matrixinverse = nalu_hypre_CSRMatrixData(domain_structure);

   if (use_nonsymm)
   {
      uplo = 'N';
   }

   nalu_hypre_ParVectorCopy(par_rhs, par_aux);
   nalu_hypre_ParCSRMatrixMatvec(-1.0, par_A, par_x, 1.0, par_aux);
   tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Real, max_domain_size, NALU_HYPRE_MEMORY_HOST);

   /* forward solve: ----------------------------------------------- */

   matrix_size_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      if (CF_marker[i] == rlx_pt)
      {
         matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];

         /* compute residual: ---------------------------------------- */

         jj = 0;
         for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
         {
            tmp[jj] = aux[j_domain_dof[j]];
            jj++;
         }
         /* solve for correction: ------------------------------------- */
         if (use_nonsymm)
         {
            nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                         &domain_matrixinverse[matrix_size_counter],
                         &matrix_size, &pivots[piv_counter], tmp,
                         &matrix_size, &ierr);
         }

         else
         {
            nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                         &domain_matrixinverse[matrix_size_counter],
                         &matrix_size, tmp,
                         &matrix_size, &ierr);
         }

         if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
         jj = 0;
         for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
         {
            x[j_domain_dof[j]] +=  scale[j_domain_dof[j]] * tmp[jj++];
         }
         matrix_size_counter += matrix_size * matrix_size;
         piv_counter += matrix_size;
      }
   }

   nalu_hypre_TFree(tmp, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_GenerateScale(nalu_hypre_CSRMatrix *domain_structure,
                    NALU_HYPRE_Int              num_variables,
                    NALU_HYPRE_Real       relaxation_weight,
                    NALU_HYPRE_Real     **scale_pointer)
{
   NALU_HYPRE_Int num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   NALU_HYPRE_Int *i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   NALU_HYPRE_Int *j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Real *scale;

   scale = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_variables, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_domains; i++)
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         scale[j_domain_dof[j]] += 1.0;
      }

   for (i = 0; i < num_variables; i++)
   {
      scale[i] = relaxation_weight / scale[i];
   }

   *scale_pointer = scale;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_ParAdSchwarzSolve(nalu_hypre_ParCSRMatrix *A,
                                  nalu_hypre_ParVector *F,
                                  nalu_hypre_CSRMatrix *domain_structure,
                                  NALU_HYPRE_Real *scale,
                                  nalu_hypre_ParVector *X,
                                  nalu_hypre_ParVector *Vtemp,
                                  NALU_HYPRE_Int *pivots,
                                  NALU_HYPRE_Int use_nonsymm)
{
   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int num_sends = 0;
   NALU_HYPRE_Int *send_map_starts;
   NALU_HYPRE_Int *send_map_elmts;

   nalu_hypre_ParCSRCommHandle *comm_handle;

   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Real *x_data;
   NALU_HYPRE_Real *x_ext_data;
   NALU_HYPRE_Real *aux;
   NALU_HYPRE_Real *vtemp_data;
   NALU_HYPRE_Real *vtemp_ext_data;
   NALU_HYPRE_Int num_domains, max_domain_size;
   NALU_HYPRE_Int *i_domain_dof;
   NALU_HYPRE_Int *j_domain_dof;
   NALU_HYPRE_Real *domain_matrixinverse;
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int num_variables;
   NALU_HYPRE_Int num_cols_offd;
   NALU_HYPRE_Real *scale_ext;
   NALU_HYPRE_Real *buf_data;
   NALU_HYPRE_Int index;

   NALU_HYPRE_Int piv_counter = 0;
   NALU_HYPRE_Int one = 1;

   char uplo = 'L';
   NALU_HYPRE_Int jj, i, j, j_loc; /*, j_loc, k_loc;*/

   NALU_HYPRE_Int matrix_size, matrix_size_counter = 0;

   /* initiate:      ----------------------------------------------- */
   num_variables = nalu_hypre_CSRMatrixNumRows(A_diag);
   num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A));
   x_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(X));
   vtemp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Vtemp));

   if (use_nonsymm)
   {
      uplo = 'N';
   }

   nalu_hypre_ParVectorCopy(F, Vtemp);
   nalu_hypre_ParCSRMatrixMatvec(-1.0, A, X, 1.0, Vtemp);

   /* forward solve: ----------------------------------------------- */

   num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   max_domain_size = nalu_hypre_CSRMatrixNumCols(domain_structure);
   i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   domain_matrixinverse = nalu_hypre_CSRMatrixData(domain_structure);
   aux = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  max_domain_size, NALU_HYPRE_MEMORY_HOST);

   if (comm_pkg)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

      buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
      x_ext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      vtemp_ext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      scale_ext = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            buf_data[index++] = vtemp_data[send_map_elmts[j]];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data,
                                                 vtemp_ext_data);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            buf_data[index++] = scale[send_map_elmts[j]];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data, scale_ext);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   for (i = 0; i < num_cols_offd; i++)
   {
      x_ext_data[i] = 0.0;
   }

   matrix_size_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      matrix_size = i_domain_dof[i + 1] - i_domain_dof[i];

      /* copy data contiguously into aux  --------------------------- */

      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         j_loc = j_domain_dof[j];
         if (j_loc < num_variables)
         {
            aux[jj] = vtemp_data[j_loc];
         }
         else
         {
            aux[jj] = vtemp_ext_data[j_loc - num_variables];
         }
         jj++;
      }
      /* solve for correction: ------------------------------------- */
      if (use_nonsymm)
      {
         nalu_hypre_dgetrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, &pivots[piv_counter], aux,
                      &matrix_size, &ierr);
      }
      else
      {
         nalu_hypre_dpotrs(&uplo, &matrix_size, &one,
                      &domain_matrixinverse[matrix_size_counter],
                      &matrix_size, aux,
                      &matrix_size, &ierr);
      }

      if (ierr) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }
      jj = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         j_loc = j_domain_dof[j];
         if (j_loc < num_variables)
         {
            x_data[j_loc] += scale[j_loc] * aux[jj++];
         }
         else
         {
            j_loc -= num_variables;
            x_ext_data[j_loc] += scale_ext[j_loc] * aux[jj++];
         }
      }
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;
   }

   if (comm_pkg)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate (2, comm_pkg, x_ext_data, buf_data);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            x_data[send_map_elmts[j]] += buf_data[index++];
         }
      }

      nalu_hypre_TFree(buf_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(x_ext_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(vtemp_ext_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(scale_ext, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(aux, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParAMGCreateDomainDof:
 *--------------------------------------------------------------------------*/

/*****************************************************************************
 *
 * Routine for constructing graph domain_dof with minimal overlap
 *             and computing the respective matrix inverses to be
 *             used in an overlapping additive Schwarz procedure (smoother
 *             in AMG);
 *
 *****************************************************************************/
NALU_HYPRE_Int
nalu_hypre_ParAMGCreateDomainDof(nalu_hypre_ParCSRMatrix   *A,
                            NALU_HYPRE_Int domain_type, NALU_HYPRE_Int overlap,
                            NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                            nalu_hypre_CSRMatrix     **domain_structure_pointer,
                            NALU_HYPRE_Int **piv_pointer, NALU_HYPRE_Int use_nonsymm)

{
   nalu_hypre_CSRMatrix *domain_structure = NULL;
   NALU_HYPRE_Int *i_domain_dof, *j_domain_dof;
   NALU_HYPRE_Real *domain_matrixinverse;
   NALU_HYPRE_Int num_domains;

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int *a_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int *a_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real *a_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int num_variables = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt first_col_diag = nalu_hypre_ParCSRMatrixFirstColDiag(A);
   NALU_HYPRE_BigInt col_0 = first_col_diag - 1 ;
   NALU_HYPRE_BigInt col_n = first_col_diag + (NALU_HYPRE_BigInt)num_variables;

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int *a_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int *a_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real *a_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(A);

   nalu_hypre_CSRMatrix *A_ext;
   NALU_HYPRE_Int *a_ext_i;
   NALU_HYPRE_BigInt *a_ext_j;
   NALU_HYPRE_Real *a_ext_data;

   /* NALU_HYPRE_Int *i_dof_to_accept_weight; */
   NALU_HYPRE_Int *i_dof_to_prefer_weight,
             *w_dof_dof, *i_dof_weight;
   NALU_HYPRE_Int *i_dof_to_aggregate, *i_aggregate_dof, *j_aggregate_dof;

   NALU_HYPRE_Int *i_dof_index;
   NALU_HYPRE_Int *i_dof_index_offd;
   NALU_HYPRE_Int *i_proc;
   /* NALU_HYPRE_Int *row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);*/
   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int num_recvs = 0;
   NALU_HYPRE_Int *recv_vec_starts = NULL;

   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int i, j, k, l_loc, i_loc, j_loc;
   NALU_HYPRE_Int i_dof;
   NALU_HYPRE_Int nf;
   NALU_HYPRE_Int *i_local_to_global;
   NALU_HYPRE_Int *i_global_to_local;

   NALU_HYPRE_Int local_dof_counter, max_local_dof_counter = 0;

   NALU_HYPRE_Int domain_dof_counter = 0, domain_matrixinverse_counter = 0;

   NALU_HYPRE_Real *AE;


   NALU_HYPRE_Int *ipiv;
   char uplo = 'L';
   NALU_HYPRE_Int piv_counter;
   NALU_HYPRE_Int *piv = NULL;

   NALU_HYPRE_Int cnt, indx;
   NALU_HYPRE_Int num_procs, my_id;

   if (num_variables == 0)
   {
      *domain_structure_pointer = domain_structure;

      *piv_pointer = piv;

      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_size(nalu_hypre_ParCSRMatrixComm(A), &num_procs);
   nalu_hypre_MPI_Comm_size(nalu_hypre_ParCSRMatrixComm(A), &my_id);

   /* --------------------------------------------------------------------- */

   /*=======================================================================*/
   /*    create artificial domains by agglomeration;                        */
   /*=======================================================================*/

   /*nalu_hypre_printf("----------- create artificials domain by agglomeration;  ======\n");
    */
   i_aggregate_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_variables + 1, NALU_HYPRE_MEMORY_HOST);
   j_aggregate_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_variables, NALU_HYPRE_MEMORY_HOST);

   if (domain_type == 2)
   {
      i_dof_to_prefer_weight = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);
      w_dof_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  a_diag_i[num_variables], NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_variables; i++)
         for (j = a_diag_i[i]; j < a_diag_i[i + 1]; j++)
         {
            if (a_diag_j[j] == i)
            {
               w_dof_dof[j] = 0;
            }
            else
            {
               w_dof_dof[j] = 1;
            }
         }

      /*nalu_hypre_printf("end computing weights for agglomeration procedure: --------\n");
       */

      i_dof_weight = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_AMGeAgglomerate(i_aggregate_dof, j_aggregate_dof,
                            a_diag_i, a_diag_j, w_dof_dof,
                            a_diag_i, a_diag_j,
                            a_diag_i, a_diag_j,
                            i_dof_to_prefer_weight,
                            i_dof_weight,
                            num_variables, num_variables,
                            &num_domains);

      nalu_hypre_TFree(i_dof_to_prefer_weight, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_weight, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(w_dof_dof, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      nf = 1;
      if (domain_type == 1) { nf = num_functions; }

      num_domains = num_variables / nf;
      for (i = 0; i < num_domains + 1; i++)
      {
         i_aggregate_dof[i] = nf * i;
      }
      for (i = 0; i < num_variables; i++)
      {
         j_aggregate_dof[i] = i;
      }
   }
   /*nalu_hypre_printf("num_variables: %d, num_domains: %d\n", num_variables, num_domains);
    */
   if (overlap == 1)
   {
      i_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_domains + 1, NALU_HYPRE_MEMORY_HOST);

      i_dof_to_aggregate = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_domains; i++)
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            i_dof_to_aggregate[j_aggregate_dof[j]] = i;
         }

      i_proc = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_cols_offd; i++)
      {
         i_proc[i] = 0;
      }

      if (comm_pkg)
      {
         num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
         recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      }
      else if (num_procs > 1)
      {

         nalu_hypre_MatvecCommPkgCreate(A);

         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
         num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
         recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      }

      for (i = 0; i < num_recvs; i++)
         for (indx = recv_vec_starts[i]; indx < recv_vec_starts[i + 1]; indx++)
         {
            i_proc[indx] = i;
         }

      /* make domains from aggregates: *********************************/

      i_dof_index = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);

      i_dof_index_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_variables; i++)
      {
         i_dof_index[i] = -1;
      }

      for (i = 0; i < num_cols_offd; i++)
      {
         i_dof_index_offd[i] = -1;
      }

      domain_dof_counter = 0;
      for (i = 0; i < num_domains; i++)
      {
         i_domain_dof[i] =  domain_dof_counter;
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            i_dof_index[j_aggregate_dof[j]] = -1;
         }
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            for (k = a_diag_i[j_aggregate_dof[j]];
                 k < a_diag_i[j_aggregate_dof[j] + 1]; k++)
               if (i_dof_to_aggregate[a_diag_j[k]] >= i
                   && i_dof_index[a_diag_j[k]] == -1)
               {
                  i_dof_index[a_diag_j[k]]++;
                  domain_dof_counter++;
               }
            for (k = a_offd_i[j_aggregate_dof[j]];
                 k < a_offd_i[j_aggregate_dof[j] + 1]; k++)
               if (i_proc[a_offd_j[k]] > my_id
                   && i_dof_index_offd[a_offd_j[k]] == -1)
               {
                  i_dof_index_offd[a_offd_j[k]]++;
                  domain_dof_counter++;
               }
         }
      }

      for (i = 0; i < num_variables; i++)
      {
         i_dof_index[i] = -1;
      }

      for (i = 0; i < num_cols_offd; i++)
      {
         i_dof_index_offd[i] = -1;
      }

      i_domain_dof[num_domains] =  domain_dof_counter;
      j_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  domain_dof_counter, NALU_HYPRE_MEMORY_HOST);

      domain_dof_counter = 0;
      for (i = 0; i < num_domains; i++)
      {
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            i_dof_index[j_aggregate_dof[j]] = -1;
         }
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            for (k = a_diag_i[j_aggregate_dof[j]];
                 k < a_diag_i[j_aggregate_dof[j] + 1]; k++)
               if (i_dof_to_aggregate[a_diag_j[k]] >= i
                   && i_dof_index[a_diag_j[k]] == -1)
               {
                  i_dof_index[a_diag_j[k]]++;
                  j_domain_dof[domain_dof_counter] = a_diag_j[k];
                  domain_dof_counter++;
               }
            for (k = a_offd_i[j_aggregate_dof[j]];
                 k < a_offd_i[j_aggregate_dof[j] + 1]; k++)
               if (i_proc[a_offd_j[k]] > my_id
                   && i_dof_index_offd[a_offd_j[k]] == -1)
               {
                  i_dof_index_offd[a_offd_j[k]]++;
                  j_domain_dof[domain_dof_counter] = a_offd_j[k] + num_variables;
                  domain_dof_counter++;
               }
         }
      }

      nalu_hypre_TFree(i_aggregate_dof, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_aggregate_dof, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_to_aggregate, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_index, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_index_offd, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_proc, NALU_HYPRE_MEMORY_HOST);
   }
   else if (overlap == 2)
   {
      i_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_domains + 1, NALU_HYPRE_MEMORY_HOST);

      i_dof_to_aggregate = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_domains; i++)
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            i_dof_to_aggregate[j_aggregate_dof[j]] = i;
         }

      /* make domains from aggregates: *********************************/

      i_dof_index = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables, NALU_HYPRE_MEMORY_HOST);

      i_dof_index_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_variables; i++)
      {
         i_dof_index[i] = -1;
      }

      for (i = 0; i < num_cols_offd; i++)
      {
         i_dof_index_offd[i] = -1;
      }

      domain_dof_counter = 0;
      for (i = 0; i < num_domains; i++)
      {
         i_domain_dof[i] =  domain_dof_counter;
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            for (k = a_diag_i[j_aggregate_dof[j]];
                 k < a_diag_i[j_aggregate_dof[j] + 1]; k++)
               if ( i_dof_index[a_diag_j[k]] == -1)
               {
                  i_dof_index[a_diag_j[k]]++;
                  domain_dof_counter++;
               }
            for (k = a_offd_i[j_aggregate_dof[j]];
                 k < a_offd_i[j_aggregate_dof[j] + 1]; k++)
               if ( i_dof_index_offd[a_offd_j[k]] == -1)
               {
                  i_dof_index_offd[a_offd_j[k]]++;
                  domain_dof_counter++;
               }
         }
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            for (k = a_diag_i[j_aggregate_dof[j]];
                 k < a_diag_i[j_aggregate_dof[j] + 1]; k++)
            {
               i_dof_index[a_diag_j[k]] = -1;
            }
            for (k = a_offd_i[j_aggregate_dof[j]];
                 k < a_offd_i[j_aggregate_dof[j] + 1]; k++)
            {
               i_dof_index_offd[a_offd_j[k]] = -1;
            }
         }
      }

      for (i = 0; i < num_variables; i++)
      {
         i_dof_index[i] = -1;
      }

      for (i = 0; i < num_cols_offd; i++)
      {
         i_dof_index_offd[i] = -1;
      }

      i_domain_dof[num_domains] =  domain_dof_counter;
      j_domain_dof = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  domain_dof_counter, NALU_HYPRE_MEMORY_HOST);

      domain_dof_counter = 0;
      for (i = 0; i < num_domains; i++)
      {
         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            for (k = a_diag_i[j_aggregate_dof[j]];
                 k < a_diag_i[j_aggregate_dof[j] + 1]; k++)
               if ( i_dof_index[a_diag_j[k]] == -1)
               {
                  i_dof_index[a_diag_j[k]]++;
                  j_domain_dof[domain_dof_counter] = a_diag_j[k];
                  domain_dof_counter++;
               }
            for (k = a_offd_i[j_aggregate_dof[j]];
                 k < a_offd_i[j_aggregate_dof[j] + 1]; k++)
               if ( i_dof_index_offd[a_offd_j[k]] == -1)
               {
                  i_dof_index_offd[a_offd_j[k]]++;
                  j_domain_dof[domain_dof_counter] = a_offd_j[k] + num_variables;
                  domain_dof_counter++;
               }
         }

         for (j = i_aggregate_dof[i]; j < i_aggregate_dof[i + 1]; j++)
         {
            for (k = a_diag_i[j_aggregate_dof[j]];
                 k < a_diag_i[j_aggregate_dof[j] + 1]; k++)
            {
               i_dof_index[a_diag_j[k]] = -1;
            }
            for (k = a_offd_i[j_aggregate_dof[j]];
                 k < a_offd_i[j_aggregate_dof[j] + 1]; k++)
            {
               i_dof_index_offd[a_offd_j[k]] = -1;
            }
         }
      }

      nalu_hypre_TFree(i_aggregate_dof, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_aggregate_dof, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_to_aggregate, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_index, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_dof_index_offd, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      i_domain_dof = i_aggregate_dof;
      j_domain_dof = j_aggregate_dof;
   }

   /*nalu_hypre_printf("END domain_dof computations: =================================\n");
    */
   domain_matrixinverse_counter = 0;
   local_dof_counter = 0;
   piv_counter = 0;

   for (i = 0; i < num_domains; i++)
   {
      local_dof_counter = i_domain_dof[i + 1] - i_domain_dof[i];
      domain_matrixinverse_counter += local_dof_counter * local_dof_counter;
      piv_counter += local_dof_counter;

      if (local_dof_counter > max_local_dof_counter)
      {
         max_local_dof_counter = local_dof_counter;
      }
   }

   domain_matrixinverse = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  domain_matrixinverse_counter, NALU_HYPRE_MEMORY_HOST);
   if (use_nonsymm)
   {
      piv = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  piv_counter, NALU_HYPRE_MEMORY_HOST);
   }

   if (num_procs > 1)
   {
      A_ext = nalu_hypre_ParCSRMatrixExtractBExt(A, A, 1);
      a_ext_i = nalu_hypre_CSRMatrixI(A_ext);
      a_ext_j = nalu_hypre_CSRMatrixBigJ(A_ext);
      a_ext_data = nalu_hypre_CSRMatrixData(A_ext);
   }
   else
   {
      A_ext = NULL;
   }

   i_local_to_global = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_local_dof_counter, NALU_HYPRE_MEMORY_HOST);

   i_global_to_local = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_variables + num_cols_offd, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_variables + num_cols_offd; i++)
   {
      i_global_to_local[i] = -1;
   }

   piv_counter = 0;
   domain_matrixinverse_counter = 0;
   for (i = 0; i < num_domains; i++)
   {
      local_dof_counter = 0;
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         i_global_to_local[j_domain_dof[j]] = local_dof_counter;
         i_local_to_global[local_dof_counter] = j_domain_dof[j];
         local_dof_counter++;
      }

      /* get local matrix in AE: ======================================== */

      AE = &domain_matrixinverse[domain_matrixinverse_counter];
      ipiv = &piv[piv_counter];

      cnt = 0;
      for (i_loc = 0; i_loc < local_dof_counter; i_loc++)
         for (j_loc = 0; j_loc < local_dof_counter; j_loc++)
         {
            AE[cnt++] = 0.e0;
         }

      for (i_loc = 0; i_loc < local_dof_counter; i_loc++)
      {
         i_dof = i_local_to_global[i_loc];
         if (i_dof < num_variables)
         {
            for (j = a_diag_i[i_dof]; j < a_diag_i[i_dof + 1]; j++)
            {
               j_loc = i_global_to_local[a_diag_j[j]];
               if (j_loc >= 0)
               {
                  AE[i_loc + j_loc * local_dof_counter] = a_diag_data[j];
               }
            }
            for (j = a_offd_i[i_dof]; j < a_offd_i[i_dof + 1]; j++)
            {
               j_loc = i_global_to_local[a_offd_j[j] + num_variables];
               if (j_loc >= 0)
               {
                  AE[i_loc + j_loc * local_dof_counter] = a_offd_data[j];
               }
            }
         }
         else
         {
            NALU_HYPRE_BigInt jj;
            NALU_HYPRE_Int j2;
            i_dof -= num_variables;
            for (j = a_ext_i[i_dof]; j < a_ext_i[i_dof + 1]; j++)
            {
               jj = a_ext_j[j];
               if (jj > col_0 && jj < col_n)
               {
                  j2 = (NALU_HYPRE_Int)(jj - first_col_diag);
               }
               else
               {
                  j2 = nalu_hypre_BigBinarySearch(col_map_offd, jj, num_cols_offd);
                  if (j2 > -1) { j2 += num_variables; }
               }
               if (j2 > -1)
               {
                  j_loc = i_global_to_local[j2];
                  if (j_loc >= 0)
                  {
                     AE[i_loc + j_loc * local_dof_counter] = a_ext_data[j];
                  }
               }
            }
         }
      }

      if (use_nonsymm)
      {
         nalu_hypre_dgetrf(&local_dof_counter,
                      &local_dof_counter, AE,
                      &local_dof_counter, ipiv, &ierr);
         piv_counter += local_dof_counter;
      }

      else
      {
         nalu_hypre_dpotrf(&uplo, &local_dof_counter, AE,
                      &local_dof_counter, &ierr);
      }

      domain_matrixinverse_counter += local_dof_counter * local_dof_counter;



      for (l_loc = 0; l_loc < local_dof_counter; l_loc++)
      {
         i_global_to_local[i_local_to_global[l_loc]] = -1;
      }

   }

   nalu_hypre_TFree(i_local_to_global, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(i_global_to_local, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CSRMatrixDestroy(A_ext);

   domain_structure = nalu_hypre_CSRMatrixCreate(num_domains, max_local_dof_counter,
                                            i_domain_dof[num_domains]);

   nalu_hypre_CSRMatrixI(domain_structure) = i_domain_dof;
   nalu_hypre_CSRMatrixJ(domain_structure) = j_domain_dof;
   nalu_hypre_CSRMatrixData(domain_structure) = domain_matrixinverse;

   *domain_structure_pointer = domain_structure;
   *piv_pointer = piv;
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_ParGenerateScale(nalu_hypre_ParCSRMatrix *A,
                       nalu_hypre_CSRMatrix *domain_structure,
                       NALU_HYPRE_Real       relaxation_weight,
                       NALU_HYPRE_Real     **scale_pointer)
{
   NALU_HYPRE_Int num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   NALU_HYPRE_Int *i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   NALU_HYPRE_Int *j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Real *scale;
   NALU_HYPRE_Real *scale_ext;
   NALU_HYPRE_Real *scale_int;

   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int num_sends = 0;
   NALU_HYPRE_Int *send_map_starts;
   NALU_HYPRE_Int *send_map_elmts;

   NALU_HYPRE_Int num_variables = nalu_hypre_ParCSRMatrixNumRows(A);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A));
   NALU_HYPRE_Int j_loc, index, start;

   nalu_hypre_ParCSRCommHandle *comm_handle;

   if (comm_pkg)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   }

   scale = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_variables, NALU_HYPRE_MEMORY_HOST);
   if (num_cols_offd) { scale_ext = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_cols_offd, NALU_HYPRE_MEMORY_HOST); }

   for (i = 0; i < num_domains; i++)
   {
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         j_loc = j_domain_dof[j];
         if (j_loc < num_variables)
         {
            scale[j_loc] += 1.0;
         }
         else
         {
            scale_ext[j_loc - num_variables] += 1.0;
         }
      }
   }
   if (comm_pkg)
   {
      scale_int = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
      comm_handle = nalu_hypre_ParCSRCommHandleCreate (2, comm_pkg, scale_ext, scale_int);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = send_map_starts[i];
      for (j = start; j < send_map_starts[i + 1]; j++)
      {
         scale[send_map_elmts[j]] += scale_int[index++];
      }
   }

   if (comm_pkg) { nalu_hypre_TFree(scale_int, NALU_HYPRE_MEMORY_HOST); }
   if (num_cols_offd) { nalu_hypre_TFree(scale_ext, NALU_HYPRE_MEMORY_HOST); }

   for (i = 0; i < num_variables; i++)
   {
      scale[i] = relaxation_weight / scale[i];
   }

   *scale_pointer = scale;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_ParGenerateHybridScale(nalu_hypre_ParCSRMatrix *A,
                             nalu_hypre_CSRMatrix  *domain_structure,
                             nalu_hypre_CSRMatrix **A_boundary_pointer,
                             NALU_HYPRE_Real      **scale_pointer)
{
   nalu_hypre_CSRMatrix *A_ext;
   NALU_HYPRE_Int *A_ext_i;
   NALU_HYPRE_BigInt *A_ext_j;
   NALU_HYPRE_Real *A_ext_data;

   nalu_hypre_CSRMatrix *A_boundary;
   NALU_HYPRE_Int *A_boundary_i;
   NALU_HYPRE_Int *A_boundary_j;
   NALU_HYPRE_Real *A_boundary_data;

   NALU_HYPRE_Int num_domains = nalu_hypre_CSRMatrixNumRows(domain_structure);
   NALU_HYPRE_Int *i_domain_dof = nalu_hypre_CSRMatrixI(domain_structure);
   NALU_HYPRE_Int *j_domain_dof = nalu_hypre_CSRMatrixJ(domain_structure);
   NALU_HYPRE_Int i, j, jj;
   NALU_HYPRE_Real *scale;
   NALU_HYPRE_Real *scale_ext;
   NALU_HYPRE_Real *scale_int;

   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int num_sends = 0;
   NALU_HYPRE_Int *send_map_starts;
   NALU_HYPRE_Int *send_map_elmts;
   NALU_HYPRE_Int *index_ext;

   NALU_HYPRE_Int num_variables = nalu_hypre_ParCSRMatrixNumRows(A);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A));
   NALU_HYPRE_Int j_loc, index, start;
   NALU_HYPRE_BigInt col_0, col_n;
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(A);

   nalu_hypre_ParCSRCommHandle *comm_handle;

   col_0 = nalu_hypre_ParCSRMatrixFirstColDiag(A) - 1;
   col_n = col_0 + (NALU_HYPRE_Int)num_variables;

   A_boundary = NULL;

   if (comm_pkg)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   }

   scale = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_variables, NALU_HYPRE_MEMORY_HOST);
   if (num_cols_offd)
   {
      scale_ext = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      index_ext = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_variables; i++)
   {
      scale[i] = 1;
   }

   for (i = 0; i < num_cols_offd; i++)
   {
      index_ext[i] = -1;
   }

   for (i = 0; i < num_domains; i++)
   {
      for (j = i_domain_dof[i]; j < i_domain_dof[i + 1]; j++)
      {
         j_loc = j_domain_dof[j];
         if (j_loc >= num_variables)
         {
            j_loc -= num_variables;
            if (index_ext[j_loc] == -1)
            {
               scale_ext[j_loc] += 1.0;
               index_ext[j_loc] ++;
            }
         }
      }
   }
   if (comm_pkg)
   {
      scale_int = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(2, comm_pkg, scale_ext, scale_int);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      A_ext = nalu_hypre_ParCSRMatrixExtractBExt(A, A, 1);
      A_ext_i = nalu_hypre_CSRMatrixI(A_ext);
      A_boundary_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd + 1, NALU_HYPRE_MEMORY_HOST);
      A_ext_j = nalu_hypre_CSRMatrixBigJ(A_ext);
      A_ext_data = nalu_hypre_CSRMatrixData(A_ext);
      /* compress A_ext to contain only local data and
         necessary boundary points*/
      index = 0;
      for (i = 0; i < num_cols_offd; i++)
      {
         A_boundary_i[i] = index;
         for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
         {
            NALU_HYPRE_BigInt j_col;
            j_col = A_ext_j[j];
            if (j_col > col_0 && j_col < col_n)
            {
               A_ext_j[j] = j_col - col_0;
               index++;
            }
            else
            {
               jj = nalu_hypre_BigBinarySearch(col_map_offd, j_col, num_cols_offd);
               if (jj > -1 && (scale_ext[jj] > 0))
               {
                  A_ext_j[j] = num_variables + jj;
                  index++;
               }
               else
               {
                  A_ext_j[j] = -1;
               }
            }
         }
      }
      A_boundary_i[num_cols_offd] = index;

      A_boundary_j = NULL;
      A_boundary_data = NULL;

      if (index)
      {
         A_boundary_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, index, NALU_HYPRE_MEMORY_HOST);
         A_boundary_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, index, NALU_HYPRE_MEMORY_HOST);
      }

      index = 0;
      for (i = 0; i < A_ext_i[num_cols_offd]; i++)
      {
         if (A_ext_j[i] > -1)
         {
            A_boundary_j[index] = (NALU_HYPRE_Int)A_ext_j[i];
            A_boundary_data[index] = A_ext_data[i];
            index++;
         }
      }
      A_boundary = nalu_hypre_CSRMatrixCreate(num_cols_offd, num_variables, index);
      nalu_hypre_CSRMatrixI(A_boundary) = A_boundary_i;
      nalu_hypre_CSRMatrixJ(A_boundary) = A_boundary_j;
      nalu_hypre_CSRMatrixData(A_boundary) = A_boundary_data;
      nalu_hypre_CSRMatrixDestroy(A_ext);
   }

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = send_map_starts[i];
      for (j = start; j < send_map_starts[i + 1]; j++)
      {
         scale[send_map_elmts[j]] += scale_int[index++];
      }
   }

   if (comm_pkg) { nalu_hypre_TFree(scale_int, NALU_HYPRE_MEMORY_HOST); }
   if (num_cols_offd)
   {
      nalu_hypre_TFree(scale_ext, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(index_ext, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_variables; i++)
   {
      scale[i] = 1.0 / scale[i];
   }

   *scale_pointer = scale;
   *A_boundary_pointer = A_boundary;

   return nalu_hypre_error_flag;
}

