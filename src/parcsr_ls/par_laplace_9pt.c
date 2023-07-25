/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_GenerateLaplacian9pt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_ParCSRMatrix
GenerateLaplacian9pt( MPI_Comm comm,
                      NALU_HYPRE_BigInt   nx,
                      NALU_HYPRE_BigInt   ny,
                      NALU_HYPRE_Int      P,
                      NALU_HYPRE_Int      Q,
                      NALU_HYPRE_Int      p,
                      NALU_HYPRE_Int      q,
                      NALU_HYPRE_Real  *value )
{
   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_CSRMatrix *diag;
   nalu_hypre_CSRMatrix *offd;

   NALU_HYPRE_Int    *diag_i;
   NALU_HYPRE_Int    *diag_j;
   NALU_HYPRE_Real *diag_data;

   NALU_HYPRE_Int    *offd_i;
   NALU_HYPRE_Int    *offd_j;
   NALU_HYPRE_Real *offd_data;

   NALU_HYPRE_BigInt global_part[2];
   NALU_HYPRE_BigInt ix, iy;
   NALU_HYPRE_Int cnt, o_cnt;
   NALU_HYPRE_Int local_num_rows;
   NALU_HYPRE_BigInt *col_map_offd;
   NALU_HYPRE_BigInt *big_offd_j;
   NALU_HYPRE_Int row_index;
   NALU_HYPRE_Int i;

   NALU_HYPRE_Int nx_local, ny_local;
   NALU_HYPRE_Int num_cols_offd;
   NALU_HYPRE_BigInt grid_size;

   NALU_HYPRE_BigInt *nx_part;
   NALU_HYPRE_BigInt *ny_part;

   NALU_HYPRE_Int num_procs;
   NALU_HYPRE_Int P_busy, Q_busy;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   grid_size = nx * ny;

   nalu_hypre_GeneratePartitioning(nx, P, &nx_part);
   nalu_hypre_GeneratePartitioning(ny, Q, &ny_part);

   nx_local = (NALU_HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (NALU_HYPRE_Int)(ny_part[q + 1] - ny_part[q]);

   local_num_rows = nx_local * ny_local;

   global_part[0] = ny_part[q] * nx + nx_part[p] * ny_local;
   global_part[1] = global_part[0] + (NALU_HYPRE_BigInt)local_num_rows;

   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);

   P_busy = nalu_hypre_min(nx, P);
   Q_busy = nalu_hypre_min(ny, Q);

   num_cols_offd = 0;
   if (p) { num_cols_offd += ny_local; }
   if (p < P_busy - 1) { num_cols_offd += ny_local; }
   if (q) { num_cols_offd += nx_local; }
   if (q < Q_busy - 1) { num_cols_offd += nx_local; }
   if (p && q) { num_cols_offd++; }
   if (p && q < Q_busy - 1 ) { num_cols_offd++; }
   if (p < P_busy - 1 && q ) { num_cols_offd++; }
   if (p < P_busy - 1 && q < Q_busy - 1 ) { num_cols_offd++; }

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

   cnt = 0;
   o_cnt = 0;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
   {
      for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
      {
         cnt++;
         o_cnt++;
         diag_i[cnt] = diag_i[cnt - 1];
         offd_i[o_cnt] = offd_i[o_cnt - 1];
         diag_i[cnt]++;
         if (iy > ny_part[q])
         {
            diag_i[cnt]++;
            if (ix > nx_part[p])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (ix)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (ix < nx_part[p + 1] - 1)
            {
               diag_i[cnt]++;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  offd_i[o_cnt]++;
               }
            }
         }
         else
         {
            if (iy)
            {
               offd_i[o_cnt]++;
               if (ix > nx_part[p])
               {
                  offd_i[o_cnt]++;
               }
               else if (ix)
               {
                  offd_i[o_cnt]++;
               }
               if (ix < nx_part[p + 1] - 1)
               {
                  offd_i[o_cnt]++;
               }
               else if (ix < nx - 1)
               {
                  offd_i[o_cnt]++;
               }
            }
         }
         if (ix > nx_part[p])
         {
            diag_i[cnt]++;
         }
         else
         {
            if (ix)
            {
               offd_i[o_cnt]++;
            }
         }
         if (ix + 1 < nx_part[p + 1])
         {
            diag_i[cnt]++;
         }
         else
         {
            if (ix + 1 < nx)
            {
               offd_i[o_cnt]++;
            }
         }
         if (iy + 1 < ny_part[q + 1])
         {
            diag_i[cnt]++;
            if (ix > nx_part[p])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (ix)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (ix < nx_part[p + 1] - 1)
            {
               diag_i[cnt]++;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  offd_i[o_cnt]++;
               }
            }
         }
         else
         {
            if (iy + 1 < ny)
            {
               offd_i[o_cnt]++;
               if (ix > nx_part[p])
               {
                  offd_i[o_cnt]++;
               }
               else if (ix)
               {
                  offd_i[o_cnt]++;
               }
               if (ix < nx_part[p + 1] - 1)
               {
                  offd_i[o_cnt]++;
               }
               else if (ix < nx - 1)
               {
                  offd_i[o_cnt]++;
               }
            }
         }
      }
   }

   diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      big_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
   {
      for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
      {
         diag_j[cnt] = row_index;
         diag_data[cnt++] = value[0];
         if (iy > ny_part[q])
         {
            if (ix > nx_part[p])
            {
               diag_j[cnt] = row_index - nx_local - 1 ;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix - 1, iy - 1, p - 1, q, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            diag_j[cnt] = row_index - nx_local;
            diag_data[cnt++] = value[1];
            if (ix < nx_part[p + 1] - 1)
            {
               diag_j[cnt] = row_index - nx_local + 1 ;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix + 1, iy - 1, p + 1, q, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
         }
         else
         {
            if (iy)
            {
               if (ix > nx_part[p])
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix - 1, iy - 1, p, q - 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               else if (ix)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix - 1, iy - 1, p - 1, q - 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               big_offd_j[o_cnt] = nalu_hypre_map2(ix, iy - 1, p, q - 1, nx,
                                              nx_part, ny_part);
               offd_data[o_cnt++] = value[1];
               if (ix < nx_part[p + 1] - 1)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix + 1, iy - 1, p, q - 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               else if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix + 1, iy - 1, p + 1, q - 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
         }
         if (ix > nx_part[p])
         {
            diag_j[cnt] = row_index - 1;
            diag_data[cnt++] = value[1];
         }
         else
         {
            if (ix)
            {
               big_offd_j[o_cnt] = nalu_hypre_map2(ix - 1, iy, p - 1, q, nx,
                                              nx_part, ny_part);
               offd_data[o_cnt++] = value[1];
            }
         }
         if (ix + 1 < nx_part[p + 1])
         {
            diag_j[cnt] = row_index + 1;
            diag_data[cnt++] = value[1];
         }
         else
         {
            if (ix + 1 < nx)
            {
               big_offd_j[o_cnt] = nalu_hypre_map2(ix + 1, iy, p + 1, q, nx,
                                              nx_part, ny_part);
               offd_data[o_cnt++] = value[1];
            }
         }
         if (iy + 1 < ny_part[q + 1])
         {
            if (ix > nx_part[p])
            {
               diag_j[cnt] = row_index + nx_local - 1 ;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix - 1, iy + 1, p - 1, q, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            diag_j[cnt] = row_index + nx_local;
            diag_data[cnt++] = value[1];
            if (ix < nx_part[p + 1] - 1)
            {
               diag_j[cnt] = row_index + nx_local + 1 ;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix + 1, iy + 1, p + 1, q, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
         }
         else
         {
            if (iy + 1 < ny)
            {
               if (ix > nx_part[p])
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix - 1, iy + 1, p, q + 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               else if (ix)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix - 1, iy + 1, p - 1, q + 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               big_offd_j[o_cnt] = nalu_hypre_map2(ix, iy + 1, p, q + 1, nx,
                                              nx_part, ny_part);
               offd_data[o_cnt++] = value[1];
               if (ix < nx_part[p + 1] - 1)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix + 1, iy + 1, p, q + 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
               else if (ix < nx - 1)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map2(ix + 1, iy + 1, p + 1, q + 1, nx,
                                                 nx_part, ny_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
         }
         row_index++;
      }
   }

   if (num_procs > 1)
   {
      NALU_HYPRE_BigInt *tmp = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, o_cnt, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < o_cnt; i++)
      {
         tmp[i] = big_offd_j[i];
      }

      nalu_hypre_BigQsort0(tmp, 0, o_cnt - 1);

      col_map_offd[0] = tmp[0];
      cnt = 0;
      for (i = 0; i < o_cnt; i++)
      {
         if (tmp[i] > col_map_offd[cnt])
         {
            cnt++;
            col_map_offd[cnt] = tmp[i];
         }
      }

      for (i = 0; i < o_cnt; i++)
      {
         offd_j[i] = nalu_hypre_BigBinarySearch(col_map_offd, big_offd_j[i], num_cols_offd);
      }

      nalu_hypre_TFree(big_offd_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(tmp, NALU_HYPRE_MEMORY_HOST);
   }

   A = nalu_hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, num_cols_offd,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);

   nalu_hypre_ParCSRMatrixColMapOffd(A) = col_map_offd;

   diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrixI(diag) = diag_i;
   nalu_hypre_CSRMatrixJ(diag) = diag_j;
   nalu_hypre_CSRMatrixData(diag) = diag_data;

   offd = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_CSRMatrixI(offd) = offd_i;
   if (num_cols_offd)
   {
      nalu_hypre_CSRMatrixJ(offd) = offd_j;
      nalu_hypre_CSRMatrixData(offd) = offd_data;
   }

   nalu_hypre_CSRMatrixMemoryLocation(diag) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixMemoryLocation(offd) = NALU_HYPRE_MEMORY_HOST;

   nalu_hypre_ParCSRMatrixMigrate(A, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));

   nalu_hypre_TFree(nx_part, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ny_part, NALU_HYPRE_MEMORY_HOST);

   return (NALU_HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_BigInt
nalu_hypre_map2( NALU_HYPRE_BigInt  ix,
            NALU_HYPRE_BigInt  iy,
            NALU_HYPRE_Int  p,
            NALU_HYPRE_Int  q,
            NALU_HYPRE_BigInt  nx,
            NALU_HYPRE_BigInt *nx_part,
            NALU_HYPRE_BigInt *ny_part)
{
   NALU_HYPRE_Int nx_local;
   NALU_HYPRE_Int ny_local;
   NALU_HYPRE_Int ix_local;
   NALU_HYPRE_Int iy_local;
   NALU_HYPRE_BigInt global_index;

   nx_local = (NALU_HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (NALU_HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   ix_local = (NALU_HYPRE_Int)(ix - nx_part[p]);
   iy_local = (NALU_HYPRE_Int)(iy - ny_part[q]);
   global_index = ny_part[q] * nx + nx_part[p] * (NALU_HYPRE_BigInt)ny_local;
   global_index += (NALU_HYPRE_BigInt)(iy_local * nx_local + ix_local);

   return global_index;
}
