/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_GenerateLaplacian
 *--------------------------------------------------------------------------*/

NALU_HYPRE_ParCSRMatrix
GenerateLaplacian( MPI_Comm       comm,
                   NALU_HYPRE_BigInt   nx,
                   NALU_HYPRE_BigInt   ny,
                   NALU_HYPRE_BigInt   nz,
                   NALU_HYPRE_Int      P,
                   NALU_HYPRE_Int      Q,
                   NALU_HYPRE_Int      R,
                   NALU_HYPRE_Int      ip,
                   NALU_HYPRE_Int      iq,
                   NALU_HYPRE_Int      ir,
                   NALU_HYPRE_Real    *value )
{
   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_CSRMatrix *diag;
   nalu_hypre_CSRMatrix *offd;

   NALU_HYPRE_Int    *diag_i;
   NALU_HYPRE_Int    *diag_j;
   NALU_HYPRE_Real *diag_data;

   NALU_HYPRE_Int    *offd_i;
   NALU_HYPRE_Int    *offd_j = NULL;
   NALU_HYPRE_BigInt *big_offd_j = NULL;
   NALU_HYPRE_Real *offd_data = NULL;

   NALU_HYPRE_BigInt global_part[2];
   NALU_HYPRE_BigInt ix, iy, iz;
   NALU_HYPRE_Int cnt, o_cnt;
   NALU_HYPRE_Int local_num_rows;
   NALU_HYPRE_BigInt *col_map_offd;
   NALU_HYPRE_Int row_index;
   NALU_HYPRE_Int i, j;

   NALU_HYPRE_Int nx_local, ny_local, nz_local;
   NALU_HYPRE_Int num_cols_offd;
   NALU_HYPRE_BigInt grid_size;

   NALU_HYPRE_BigInt *nx_part;
   NALU_HYPRE_BigInt *ny_part;
   NALU_HYPRE_BigInt *nz_part;

   NALU_HYPRE_Int num_procs;
   NALU_HYPRE_Int P_busy, Q_busy, R_busy;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   grid_size = nx * ny * nz;

   nalu_hypre_GeneratePartitioning(nx, P, &nx_part);
   nalu_hypre_GeneratePartitioning(ny, Q, &ny_part);
   nalu_hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (NALU_HYPRE_Int)(nx_part[ip + 1] - nx_part[ip]);
   ny_local = (NALU_HYPRE_Int)(ny_part[iq + 1] - ny_part[iq]);
   nz_local = (NALU_HYPRE_Int)(nz_part[ir + 1] - nz_part[ir]);

   local_num_rows = nx_local * ny_local * nz_local;

   global_part[0] = nz_part[ir] * nx * ny + (ny_part[iq] * nx + nx_part[ip] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (NALU_HYPRE_BigInt)local_num_rows;

   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);

   P_busy = nalu_hypre_min(nx, P);
   Q_busy = nalu_hypre_min(ny, Q);
   R_busy = nalu_hypre_min(nz, R);

   num_cols_offd = 0;
   if (ip) { num_cols_offd += ny_local * nz_local; }
   if (ip < P_busy - 1) { num_cols_offd += ny_local * nz_local; }
   if (iq) { num_cols_offd += nx_local * nz_local; }
   if (iq < Q_busy - 1) { num_cols_offd += nx_local * nz_local; }
   if (ir) { num_cols_offd += nx_local * ny_local; }
   if (ir < R_busy - 1) { num_cols_offd += nx_local * ny_local; }

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[ir]; iz < nz_part[ir + 1]; iz++)
   {
      for (iy = ny_part[iq];  iy < ny_part[iq + 1]; iy++)
      {
         for (ix = nx_part[ip]; ix < nx_part[ip + 1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[o_cnt] = offd_i[o_cnt - 1];
            diag_i[cnt]++;
            if (iz > nz_part[ir])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (iz)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iy > ny_part[iq] )
            {
               diag_i[cnt]++;
            }
            else
            {
               if (iy)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (ix > nx_part[ip] )
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
            if (ix + 1 < nx_part[ip + 1])
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
            if (iy + 1 < ny_part[iq + 1])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz + 1 < nz_part[ir + 1])
            {
               diag_i[cnt]++;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  offd_i[o_cnt]++;
               }
            }
            cnt++;
            o_cnt++;
         }
      }
   }

   diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      big_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      offd_j     = nalu_hypre_CTAlloc(NALU_HYPRE_Int,    offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      offd_data  = nalu_hypre_CTAlloc(NALU_HYPRE_Real,   offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_part[ir]; iz < nz_part[ir + 1]; iz++)
   {
      for (iy = ny_part[iq];  iy < ny_part[iq + 1]; iy++)
      {
         for (ix = nx_part[ip]; ix < nx_part[ip + 1]; ix++)
         {
            diag_j[cnt] = row_index;
            diag_data[cnt++] = value[0];
            if (iz > nz_part[ir])
            {
               diag_j[cnt] = row_index - nx_local * ny_local;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (iz)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix, iy, iz - 1, ip, iq, ir - 1, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
            if (iy > ny_part[iq])
            {
               diag_j[cnt] = row_index - nx_local;
               diag_data[cnt++] = value[2];
            }
            else
            {
               if (iy)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix, iy - 1, iz, ip, iq - 1, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[2];
               }
            }
            if (ix > nx_part[ip])
            {
               diag_j[cnt] = row_index - 1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix - 1, iy, iz, ip - 1, iq, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (ix + 1 < nx_part[ip + 1])
            {
               diag_j[cnt] = row_index + 1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix + 1, iy, iz, ip + 1, iq, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (iy + 1 < ny_part[iq + 1])
            {
               diag_j[cnt] = row_index + nx_local;
               diag_data[cnt++] = value[2];
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix, iy + 1, iz, ip, iq + 1, ir, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[2];
               }
            }
            if (iz + 1 < nz_part[ir + 1])
            {
               diag_j[cnt] = row_index + nx_local * ny_local;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix, iy, iz + 1, ip, iq, ir + 1, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
      for (i = 0; i < num_cols_offd; i++)
      {
         col_map_offd[i] = big_offd_j[i];
      }

      nalu_hypre_BigQsort0(col_map_offd, 0, num_cols_offd - 1);

      for (i = 0; i < num_cols_offd; i++)
         for (j = 0; j < num_cols_offd; j++)
            if (big_offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }

      /*for (i=0; i < offd_i[local_num_rows]; i++)
      {
         offd_j[i] = nalu_hypre_BigBinarySearch(col_map_offd,big_offd_j[i],num_cols_offd);
      }*/
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

   nalu_hypre_TFree(nx_part,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ny_part,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nz_part,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_offd_j,  NALU_HYPRE_MEMORY_HOST);

   return (NALU_HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_BigInt
nalu_hypre_map(NALU_HYPRE_BigInt  ix,
          NALU_HYPRE_BigInt  iy,
          NALU_HYPRE_BigInt  iz,
          NALU_HYPRE_Int     p,
          NALU_HYPRE_Int     q,
          NALU_HYPRE_Int     r,
          NALU_HYPRE_BigInt  nx,
          NALU_HYPRE_BigInt  ny,
          NALU_HYPRE_BigInt *nx_part,
          NALU_HYPRE_BigInt *ny_part,
          NALU_HYPRE_BigInt *nz_part)
{
   NALU_HYPRE_Int nx_local = (NALU_HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   NALU_HYPRE_Int ny_local = (NALU_HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   NALU_HYPRE_Int nz_local = (NALU_HYPRE_Int)(nz_part[r + 1] - nz_part[r]);
   NALU_HYPRE_Int ix_local = (NALU_HYPRE_Int)(ix - nx_part[p]);
   NALU_HYPRE_Int iy_local = (NALU_HYPRE_Int)(iy - ny_part[q]);
   NALU_HYPRE_Int iz_local = (NALU_HYPRE_Int)(iz - nz_part[r]);
   NALU_HYPRE_BigInt global_index;

   global_index = nz_part[r] * nx * ny + ny_part[q] * nx * (NALU_HYPRE_BigInt)nz_local + nx_part[p] *
                  (NALU_HYPRE_BigInt)(ny_local * nz_local);
   global_index += (NALU_HYPRE_BigInt)((iz_local * ny_local + iy_local) * nx_local + ix_local);
   return global_index;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GenerateVectorLaplacian - this is the systems version
 *--------------------------------------------------------------------------*/

NALU_HYPRE_ParCSRMatrix
GenerateSysLaplacian( MPI_Comm comm,
                      NALU_HYPRE_BigInt   nx,
                      NALU_HYPRE_BigInt   ny,
                      NALU_HYPRE_BigInt   nz,
                      NALU_HYPRE_Int      P,
                      NALU_HYPRE_Int      Q,
                      NALU_HYPRE_Int      R,
                      NALU_HYPRE_Int      p,
                      NALU_HYPRE_Int      q,
                      NALU_HYPRE_Int      r,
                      NALU_HYPRE_Int      num_fun,
                      NALU_HYPRE_Real  *mtrx,
                      NALU_HYPRE_Real  *value )
{
   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_CSRMatrix *diag;
   nalu_hypre_CSRMatrix *offd;

   NALU_HYPRE_Int    *diag_i;
   NALU_HYPRE_Int    *diag_j;
   NALU_HYPRE_Real *diag_data;

   NALU_HYPRE_Int    *offd_i;
   NALU_HYPRE_Int    *offd_j = NULL;
   NALU_HYPRE_BigInt *big_offd_j = NULL;
   NALU_HYPRE_Real *offd_data;

   NALU_HYPRE_BigInt global_part[2];
   NALU_HYPRE_Int ix, iy, iz;
   NALU_HYPRE_Int cnt, o_cnt;
   NALU_HYPRE_Int local_num_rows;
   NALU_HYPRE_BigInt *col_map_offd;
   NALU_HYPRE_Int row_index, row, col;
   NALU_HYPRE_Int index, diag_index;
   NALU_HYPRE_Int i, j;

   NALU_HYPRE_Int nx_local, ny_local, nz_local;
   NALU_HYPRE_Int num_cols_offd;
   NALU_HYPRE_BigInt grid_size;
   NALU_HYPRE_Int local_grid_size;
   NALU_HYPRE_Int first_j, j_ind;
   NALU_HYPRE_BigInt big_first_j, big_num_fun = (NALU_HYPRE_BigInt)num_fun;
   NALU_HYPRE_Int num_coeffs, num_offd_coeffs;

   NALU_HYPRE_BigInt *nx_part;
   NALU_HYPRE_BigInt *ny_part;
   NALU_HYPRE_BigInt *nz_part;

   NALU_HYPRE_Int num_procs;
   NALU_HYPRE_Int P_busy, Q_busy, R_busy;
   NALU_HYPRE_Real val;
   NALU_HYPRE_Int gp_size;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   grid_size = nx * ny * nz;

   nalu_hypre_GeneratePartitioning(nx, P, &nx_part);
   nalu_hypre_GeneratePartitioning(ny, Q, &ny_part);
   nalu_hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (NALU_HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (NALU_HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (NALU_HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_grid_size = nx_local * ny_local * nz_local;
   local_num_rows = num_fun * local_grid_size;

   global_part[0] = nz_part[r] * nx * ny + (ny_part[q] * nx + nx_part[p] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (NALU_HYPRE_BigInt)local_grid_size;
   gp_size = 2;

   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);

   P_busy = nalu_hypre_min(nx, P);
   Q_busy = nalu_hypre_min(ny, Q);
   R_busy = nalu_hypre_min(nz, R);

   num_cols_offd = 0;
   if (p) { num_cols_offd += ny_local * nz_local; }
   if (p < P_busy - 1) { num_cols_offd += ny_local * nz_local; }
   if (q) { num_cols_offd += nx_local * nz_local; }
   if (q < Q_busy - 1) { num_cols_offd += nx_local * nz_local; }
   if (r) { num_cols_offd += nx_local * ny_local; }
   if (r < R_busy - 1) { num_cols_offd += nx_local * ny_local; }
   num_cols_offd *= num_fun;

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

   cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[cnt] = offd_i[cnt - 1];
            diag_i[cnt] += num_fun;
            if (iz > nz_part[r])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iz)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iy > ny_part[q])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iy)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (ix > nx_part[p])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (ix)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (ix + 1 < nx_part[p + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            num_coeffs = diag_i[cnt] - diag_i[cnt - 1];
            num_offd_coeffs = offd_i[cnt] - offd_i[cnt - 1];
            cnt++;
            for (i = 1; i < num_fun; i++)
            {
               diag_i[cnt] = diag_i[cnt - 1] + num_coeffs;
               offd_i[cnt] = offd_i[cnt - 1] + num_offd_coeffs;
               cnt++;
            }
         }
      }
   }

   diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      offd_j     = nalu_hypre_CTAlloc(NALU_HYPRE_Int,    offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      big_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      offd_data  = nalu_hypre_CTAlloc(NALU_HYPRE_Real,   offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            cnt = diag_i[row_index];;
            o_cnt = offd_i[row_index];;
            num_coeffs = diag_i[row_index + 1] - diag_i[row_index];
            num_offd_coeffs = offd_i[row_index + 1] - offd_i[row_index];
            first_j = row_index;
            for (i = 0; i < num_fun; i++)
            {
               for (j = 0; j < num_fun; j++)
               {
                  j_ind = cnt + i * num_coeffs + j;
                  diag_j[j_ind] = first_j + j;
                  diag_data[j_ind] = value[0] * mtrx[i * num_fun + j];
               }
            }
            cnt += num_fun;
            if (iz > nz_part[r])
            {
               first_j = row_index - nx_local * ny_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[3] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iz)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix, iy, iz - 1, p, q, r - 1, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[3] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iy > ny_part[q])
            {
               first_j = row_index - nx_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[2] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iy)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix, iy - 1, iz, p, q - 1, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[2] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (ix > nx_part[p])
            {
               first_j = row_index - num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[1] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (ix)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix - 1, iy, iz, p - 1, q, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[1] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (ix + 1 < nx_part[p + 1])
            {
               first_j = row_index + num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[1] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix + 1, iy, iz, p + 1, q, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[1] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               first_j = row_index + nx_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[2] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix, iy + 1, iz, p, q + 1, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[2] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               first_j = row_index + nx_local * ny_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[3] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix, iy, iz + 1, p, q, r + 1, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[3] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            row_index += num_fun;
         }
      }
   }

   if (num_procs > 1)
   {
      cnt = 0;
      for (i = 0; i < local_num_rows; i += num_fun)
      {
         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            col_map_offd[cnt++] = big_offd_j[j];
         }
      }

      nalu_hypre_BigQsort0(col_map_offd, 0, num_cols_offd - 1);

      for (i = 0; i < num_fun * num_cols_offd; i++)
         for (j = nalu_hypre_min(0, nalu_hypre_abs(i - num_fun)); j < num_cols_offd; j++)
            if (big_offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
   }

   for (i = 0; i < gp_size; i++)
   {
      global_part[i] *= big_num_fun;
   }

   for (j = 1; j < num_fun; j++)
   {
      for (i = 0; i < local_grid_size; i++)
      {
         row = i * num_fun + j;
         diag_index = diag_i[row];
         index = diag_index + j;
         val = diag_data[diag_index];
         col = diag_j[diag_index];
         diag_data[diag_index] = diag_data[index];
         diag_j[diag_index] = diag_j[index];
         diag_data[index] = val;
         diag_j[index] = col;
      }
   }

   A = nalu_hypre_ParCSRMatrixCreate(comm, big_num_fun * grid_size, big_num_fun * grid_size,
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

   nalu_hypre_TFree(nx_part,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ny_part,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nz_part,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_offd_j,  NALU_HYPRE_MEMORY_HOST);

   return (NALU_HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GenerateVectorLaplacian - this is also a systems version and we can
 * vary the diffusion coefficients in each block
 *--------------------------------------------------------------------------*/

NALU_HYPRE_ParCSRMatrix
GenerateSysLaplacianVCoef( MPI_Comm       comm,
                           NALU_HYPRE_BigInt   nx,
                           NALU_HYPRE_BigInt   ny,
                           NALU_HYPRE_BigInt   nz,
                           NALU_HYPRE_Int      P,
                           NALU_HYPRE_Int      Q,
                           NALU_HYPRE_Int      R,
                           NALU_HYPRE_Int      p,
                           NALU_HYPRE_Int      q,
                           NALU_HYPRE_Int      r,
                           NALU_HYPRE_Int      num_fun,
                           NALU_HYPRE_Real    *mtrx,
                           NALU_HYPRE_Real    *value )
{
   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_CSRMatrix *diag;
   nalu_hypre_CSRMatrix *offd;

   NALU_HYPRE_Int    *diag_i;
   NALU_HYPRE_Int    *diag_j;
   NALU_HYPRE_Real *diag_data;

   NALU_HYPRE_Int    *offd_i;
   NALU_HYPRE_Int    *offd_j;
   NALU_HYPRE_BigInt *big_offd_j;
   NALU_HYPRE_Real *offd_data;

   NALU_HYPRE_BigInt global_part[2];
   NALU_HYPRE_BigInt ix, iy, iz;
   NALU_HYPRE_Int cnt, o_cnt;
   NALU_HYPRE_Int local_num_rows;
   NALU_HYPRE_BigInt *col_map_offd;
   NALU_HYPRE_Int row_index, row, col;
   NALU_HYPRE_Int index, diag_index;
   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Int gp_size;

   NALU_HYPRE_Int nx_local, ny_local, nz_local;
   NALU_HYPRE_Int num_cols_offd;
   NALU_HYPRE_BigInt grid_size;
   NALU_HYPRE_Int local_grid_size;
   NALU_HYPRE_Int first_j, j_ind;
   NALU_HYPRE_BigInt big_first_j, big_num_fun = (NALU_HYPRE_BigInt) num_fun;
   NALU_HYPRE_Int num_coeffs, num_offd_coeffs;

   NALU_HYPRE_BigInt *nx_part;
   NALU_HYPRE_BigInt *ny_part;
   NALU_HYPRE_BigInt *nz_part;

   NALU_HYPRE_Int num_procs, P_busy, Q_busy, R_busy;
   NALU_HYPRE_Real val;

   /* for indexing in values */
   NALU_HYPRE_Int sz = num_fun * num_fun;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   grid_size = nx * ny * nz;

   nalu_hypre_GeneratePartitioning(nx, P, &nx_part);
   nalu_hypre_GeneratePartitioning(ny, Q, &ny_part);
   nalu_hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (NALU_HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (NALU_HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (NALU_HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_grid_size = nx_local * ny_local * nz_local;
   global_part[0] = nz_part[r] * nx * ny + (ny_part[q] * nx + nx_part[p] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (NALU_HYPRE_BigInt)local_grid_size;
   gp_size = 2;

   local_num_rows = num_fun * local_grid_size;
   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);

   P_busy = nalu_hypre_min(nx, P);
   Q_busy = nalu_hypre_min(ny, Q);
   R_busy = nalu_hypre_min(nz, R);

   num_cols_offd = 0;
   if (p) { num_cols_offd += ny_local * nz_local; }
   if (p < P_busy - 1) { num_cols_offd += ny_local * nz_local; }
   if (q) { num_cols_offd += nx_local * nz_local; }
   if (q < Q_busy - 1) { num_cols_offd += nx_local * nz_local; }
   if (r) { num_cols_offd += nx_local * ny_local; }
   if (r < R_busy - 1) { num_cols_offd += nx_local * ny_local; }
   num_cols_offd *= num_fun;

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

   cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[cnt] = offd_i[cnt - 1];
            diag_i[cnt] += num_fun;
            if (iz > nz_part[r])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iz)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iy > ny_part[q])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iy)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (ix > nx_part[p])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (ix)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (ix + 1 < nx_part[p + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               diag_i[cnt] += num_fun;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  offd_i[cnt] += num_fun;
               }
            }
            num_coeffs = diag_i[cnt] - diag_i[cnt - 1];
            num_offd_coeffs = offd_i[cnt] - offd_i[cnt - 1];
            cnt++;
            for (i = 1; i < num_fun; i++)
            {
               diag_i[cnt] = diag_i[cnt - 1] + num_coeffs;
               offd_i[cnt] = offd_i[cnt - 1] + num_offd_coeffs;
               cnt++;
            }
         }
      }
   }

   diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      offd_j     = nalu_hypre_CTAlloc(NALU_HYPRE_Int,    offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      big_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      offd_data  = nalu_hypre_CTAlloc(NALU_HYPRE_Real,   offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            cnt = diag_i[row_index];;
            o_cnt = offd_i[row_index];;
            num_coeffs = diag_i[row_index + 1] - diag_i[row_index];
            num_offd_coeffs = offd_i[row_index + 1] - offd_i[row_index];
            first_j = row_index;
            for (i = 0; i < num_fun; i++)
            {
               for (j = 0; j < num_fun; j++)
               {
                  j_ind = cnt + i * num_coeffs + j;
                  diag_j[j_ind] = first_j + j;
                  diag_data[j_ind] = value[0 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
               }
            }
            cnt += num_fun;
            if (iz > nz_part[r])
            {
               first_j = row_index - nx_local * ny_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[3 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iz)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix, iy, iz - 1, p, q, r - 1, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[3 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iy > ny_part[q])
            {
               first_j = row_index - nx_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[2 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iy)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix, iy - 1, iz, p, q - 1, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[2 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (ix > nx_part[p])
            {
               first_j = row_index - num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[1 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (ix)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix - 1, iy, iz, p - 1, q, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[1 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (ix + 1 < nx_part[p + 1])
            {
               first_j = row_index + num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[1 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix + 1, iy, iz, p + 1, q, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[1 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iy + 1 < ny_part[q + 1])
            {
               first_j = row_index + nx_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[2 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix, iy + 1, iz, p, q + 1, r, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[2 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            if (iz + 1 < nz_part[r + 1])
            {
               first_j = row_index + nx_local * ny_local * num_fun;
               for (i = 0; i < num_fun; i++)
               {
                  for (j = 0; j < num_fun; j++)
                  {
                     j_ind = cnt + i * num_coeffs + j;
                     diag_j[j_ind] = first_j + j;
                     diag_data[j_ind] = value[3 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                  }
               }
               cnt += num_fun;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_first_j = big_num_fun * nalu_hypre_map(ix, iy, iz + 1, p, q, r + 1, nx, ny,
                                                        nx_part, ny_part, nz_part);
                  for (i = 0; i < num_fun; i++)
                  {
                     for (j = 0; j < num_fun; j++)
                     {
                        j_ind = o_cnt + i * num_offd_coeffs + j;
                        big_offd_j[j_ind] = big_first_j + (NALU_HYPRE_BigInt)j;
                        offd_data[j_ind] = value[3 * sz + i * num_fun + j] * mtrx[i * num_fun + j];
                     }
                  }
                  o_cnt += num_fun;
               }
            }
            row_index += num_fun;
         }
      }
   }

   if (num_procs > 1)
   {
      cnt = 0;
      for (i = 0; i < local_num_rows; i += num_fun)
      {
         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            col_map_offd[cnt++] = big_offd_j[j];
         }
      }

      nalu_hypre_BigQsort0(col_map_offd, 0, num_cols_offd - 1);

      for (i = 0; i < num_fun * num_cols_offd; i++)
         for (j = nalu_hypre_min(0, nalu_hypre_abs(i - num_fun)); j < num_cols_offd; j++)
            if (big_offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
   }

   for (i = 0; i < gp_size; i++)
   {
      global_part[i] *= num_fun;
   }

   for (j = 1; j < num_fun; j++)
   {
      for (i = 0; i < local_grid_size; i++)
      {
         row = i * num_fun + j;
         diag_index = diag_i[row];
         index = diag_index + j;
         val = diag_data[diag_index];
         col = diag_j[diag_index];
         diag_data[diag_index] = diag_data[index];
         diag_j[diag_index] = diag_j[index];
         diag_data[index] = val;
         diag_j[index] = col;
      }
   }

   A = nalu_hypre_ParCSRMatrixCreate(comm, num_fun * grid_size, num_fun * grid_size,
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

   nalu_hypre_TFree(nx_part,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ny_part,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nz_part,     NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_offd_j,  NALU_HYPRE_MEMORY_HOST);

   return (NALU_HYPRE_ParCSRMatrix) A;
}
