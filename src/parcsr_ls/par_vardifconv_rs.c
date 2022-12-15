/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

/* examples in Ruge & Stuben paper */
static NALU_HYPRE_Int rs_example = 1;
static NALU_HYPRE_Real rs_l = 3.0;

/*--------------------------------------------------------------------------
 * nalu_hypre_GenerateVarDifConv: with the FD discretization and examples
 *                           in Ruge-Stuben's paper ``Algebraic Multigrid''
 *--------------------------------------------------------------------------*/

NALU_HYPRE_ParCSRMatrix
GenerateRSVarDifConv( MPI_Comm         comm,
                      NALU_HYPRE_BigInt     nx,
                      NALU_HYPRE_BigInt     ny,
                      NALU_HYPRE_BigInt     nz,
                      NALU_HYPRE_Int        P,
                      NALU_HYPRE_Int        Q,
                      NALU_HYPRE_Int        R,
                      NALU_HYPRE_Int        p,
                      NALU_HYPRE_Int        q,
                      NALU_HYPRE_Int        r,
                      NALU_HYPRE_Real       eps,
                      NALU_HYPRE_ParVector *rhs_ptr,
                      NALU_HYPRE_Int        type)
{
   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_CSRMatrix *diag;
   nalu_hypre_CSRMatrix *offd;
   nalu_hypre_ParVector *par_rhs;
   nalu_hypre_Vector *rhs;
   NALU_HYPRE_Real *rhs_data;

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
   NALU_HYPRE_Int row_index;
   NALU_HYPRE_Int i, j;

   NALU_HYPRE_Int nx_local, ny_local, nz_local;
   NALU_HYPRE_Int num_cols_offd;
   NALU_HYPRE_BigInt grid_size;


   NALU_HYPRE_BigInt *nx_part;
   NALU_HYPRE_BigInt *ny_part;
   NALU_HYPRE_BigInt *nz_part;

   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int P_busy, Q_busy, R_busy;

   NALU_HYPRE_Real hhx, hhy, hhz;
   NALU_HYPRE_Real xx, yy, zz;
   NALU_HYPRE_Real afp, afm, bfp, bfm, cfp, cfm, di, ai, mux, ei, bi,
              muy, fi, ci, muz, dfm, dfp, efm, efp, ffm, ffp, gi;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (type >= 1 && type <= 3)
   {
      rs_example = type;
   }

   grid_size = nx * ny * nz;

   nalu_hypre_GeneratePartitioning(nx, P, &nx_part);
   nalu_hypre_GeneratePartitioning(ny, Q, &ny_part);
   nalu_hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (NALU_HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (NALU_HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (NALU_HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_num_rows = nx_local * ny_local * nz_local;

   global_part[0] = nz_part[r] * nx * ny + (ny_part[q] * nx + nx_part[p] * ny_local) * nz_local;
   global_part[1] = global_part[0] + (NALU_HYPRE_BigInt)local_num_rows;

   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   rhs_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);

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

   if (!local_num_rows) { num_cols_offd = 0; }

   col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);

   hhx = 1.0 / (NALU_HYPRE_Real)(nx + 1);
   hhy = 1.0 / (NALU_HYPRE_Real)(ny + 1);
   hhz = 1.0 / (NALU_HYPRE_Real)(nz + 1);

   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt - 1];
            offd_i[o_cnt] = offd_i[o_cnt - 1];
            diag_i[cnt]++;
            if (iz > nz_part[r])
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
            if (iy > ny_part[q])
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
            }
            else
            {
               if (iy + 1 < ny)
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz + 1 < nz_part[r + 1])
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

   diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  diag_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      big_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
      offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  offd_i[local_num_rows], NALU_HYPRE_MEMORY_HOST);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      zz = (NALU_HYPRE_Real)(iz + 1) * hhz;
      for (iy = ny_part[q]; iy < ny_part[q + 1]; iy++)
      {
         yy = (NALU_HYPRE_Real)(iy + 1) * hhy;
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            xx = (NALU_HYPRE_Real)(ix + 1) * hhx;
            afp = -eps * afun_rs(xx + 0.5 * hhx, yy, zz) / hhx / hhx;
            afm = -eps * afun_rs(xx - 0.5 * hhx, yy, zz) / hhx / hhx;
            bfp = -eps * bfun_rs(xx, yy + 0.5 * hhy, zz) / hhy / hhy;
            bfm = -eps * bfun_rs(xx, yy - 0.5 * hhy, zz) / hhy / hhy;
            cfp = -eps * cfun_rs(xx, yy, zz + 0.5 * hhz) / hhz / hhz;
            cfm = -eps * cfun_rs(xx, yy, zz - 0.5 * hhz) / hhz / hhz;
            /* first order terms */
            /* x-direction */
            di = dfun_rs(xx, yy, zz);
            ai = afun_rs(xx, yy, zz);
            if (di * hhx > eps * ai)
            {
               mux = eps * ai / (2.0 * di * hhx);
            }
            else if (di * hhx < -eps * ai)
            {
               mux = 1.0 + eps * ai / (2.0 * di * hhx);
            }
            else
            {
               mux = 0.5;
            }
            /* y-direction */
            ei = efun_rs(xx, yy, zz);
            bi = bfun_rs(xx, yy, zz);
            if (ei * hhy > eps * bi)
            {
               muy = eps * bi / (2.0 * ei * hhy);
            }
            else if (ei * hhy < -eps * bi)
            {
               muy = 1.0 + eps * bi / (2.0 * ei * hhy);
            }
            else
            {
               muy = 0.5;
            }
            /* z-direction */
            fi = ffun_rs(xx, yy, zz);
            ci = cfun_rs(xx, yy, zz);
            if (fi * hhz > eps * ci)
            {
               muz = eps * ci / (2.0 * fi * hhz);
            }
            else if (fi * hhz < -eps * ci)
            {
               muz = 1.0 + eps * ci / (2.0 * fi * hhz);
            }
            else
            {
               muz = 0.5;
            }

            dfm = di * (mux - 1.0) / hhx;
            dfp = di * mux / hhx;
            efm = ei * (muy - 1.0) / hhy;
            efp = ei * muy / hhy;
            ffm = fi * (muz - 1.0) / hhz;
            ffp = fi * muz / hhz;
            gi = gfun_rs(xx, yy, zz);
            /* stencil: center */
            diag_j[cnt] = row_index;
            diag_data[cnt++] = -(afp + afm + bfp + bfm + cfp + cfm  +
                                 dfp + dfm + efp + efm + ffp + ffm) + gi;
            /* rhs vector */
            rhs_data[row_index] = rfun_rs(xx, yy, zz);
            /* apply boundary conditions */
            if (ix == 0) { rhs_data[row_index] -= (afm + dfm) * bndfun_rs(0, yy, zz); }
            if (iy == 0) { rhs_data[row_index] -= (bfm + efm) * bndfun_rs(xx, 0, zz); }
            if (iz == 0) { rhs_data[row_index] -= (cfm + ffm) * bndfun_rs(xx, yy, 0); }
            if (ix + 1 == nx) { rhs_data[row_index] -= (afp + dfp) * bndfun_rs(1.0, yy, zz); }
            if (iy + 1 == ny) { rhs_data[row_index] -= (bfp + efp) * bndfun_rs(xx, 1.0, zz); }
            if (iz + 1 == nz) { rhs_data[row_index] -= (cfp + ffp) * bndfun_rs(xx, yy, 1.0); }
            /* stencil: z- */
            if (iz > nz_part[r])
            {
               diag_j[cnt] = row_index - nx_local * ny_local;
               diag_data[cnt++] = cfm + ffm;
            }
            else
            {
               if (iz)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix, iy, iz - 1, p, q, r - 1, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = cfm + ffm;
               }
            }
            /* stencil: y- */
            if (iy > ny_part[q])
            {
               diag_j[cnt] = row_index - nx_local;
               diag_data[cnt++] = bfm + efm;
            }
            else
            {
               if (iy)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix, iy - 1, iz, p, q - 1, r, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = bfm + efm;
               }
            }
            /* stencil: x- */
            if (ix > nx_part[p])
            {
               diag_j[cnt] = row_index - 1;
               diag_data[cnt++] = afm + dfm;
            }
            else
            {
               if (ix)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix - 1, iy, iz, p - 1, q, r, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = afm + dfm;
               }
            }
            /* stencil: x+ */
            if (ix + 1 < nx_part[p + 1])
            {
               diag_j[cnt] = row_index + 1;
               diag_data[cnt++] = afp + dfp;
            }
            else
            {
               if (ix + 1 < nx)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix + 1, iy, iz, p + 1, q, r, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = afp + dfp;
               }
            }
            /* stencil: y+ */
            if (iy + 1 < ny_part[q + 1])
            {
               diag_j[cnt] = row_index + nx_local;
               diag_data[cnt++] = bfp + efp;
            }
            else
            {
               if (iy + 1 < ny)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix, iy + 1, iz, p, q + 1, r, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = bfp + efp;
               }
            }
            /* stencil: z+ */
            if (iz + 1 < nz_part[r + 1])
            {
               diag_j[cnt] = row_index + nx_local * ny_local;
               diag_data[cnt++] = cfp + ffp;
            }
            else
            {
               if (iz + 1 < nz)
               {
                  big_offd_j[o_cnt] = nalu_hypre_map(ix, iy, iz + 1, p, q, r + 1, nx, ny,
                                                nx_part, ny_part, nz_part);
                  offd_data[o_cnt++] = cfp + ffp;
               }
            }
            /* done with this row */
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
      nalu_hypre_TFree(big_offd_j, NALU_HYPRE_MEMORY_HOST);
   }

   par_rhs = nalu_hypre_ParVectorCreate(comm, grid_size, global_part);
   rhs = nalu_hypre_ParVectorLocalVector(par_rhs);
   nalu_hypre_VectorData(rhs) = rhs_data;

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

   nalu_hypre_TFree(nx_part, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ny_part, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nz_part, NALU_HYPRE_MEMORY_HOST);

   *rhs_ptr = (NALU_HYPRE_ParVector) par_rhs;

   return (NALU_HYPRE_ParCSRMatrix) A;
}

NALU_HYPRE_Real afun_rs(NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz)
{
   NALU_HYPRE_Real value;
   value = 1.0;
   return value;
}

NALU_HYPRE_Real bfun_rs(NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz)
{
   NALU_HYPRE_Real value;
   value = 1.0;
   return value;
}

NALU_HYPRE_Real cfun_rs(NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz)
{
   NALU_HYPRE_Real value;
   value = 1.0;
   return value;
}

NALU_HYPRE_Real dfun_rs(NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz)
{
   NALU_HYPRE_Real value;
   if (rs_example == 1)
   {
      value = sin(rs_l * M_PI / 8.0);
   }
   else if (rs_example == 2)
   {
      value = (2.0 * yy - 1.0) * (1.0 - xx * xx);
   }
   else
   {
      value = 4.0 * xx * (xx - 1.0) * (1.0 - 2.0 * yy);
   }
   return value;
}

NALU_HYPRE_Real efun_rs(NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz)
{
   NALU_HYPRE_Real value;
   if (rs_example == 1)
   {
      value = cos(rs_l * M_PI / 8.0);
   }
   else if (rs_example == 2)
   {
      value = 2.0 * xx * yy * (yy - 1.0);
   }
   else
   {
      value = -4.0 * yy * (yy - 1.0) * (1.0 - 2.0 * xx);
   }
   return value;
}

NALU_HYPRE_Real ffun_rs(NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz)
{
   NALU_HYPRE_Real value;
   value = efun_rs(xx, yy, zz);
   return value;
}

NALU_HYPRE_Real gfun_rs(NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz)
{
   NALU_HYPRE_Real value;
   value = 0.0;
   return value;
}

NALU_HYPRE_Real rfun_rs(NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz)
{
   NALU_HYPRE_Real value;
   value = 1.0;
   return value;
}

NALU_HYPRE_Real bndfun_rs(NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz)
{
   NALU_HYPRE_Real value;
   value = 0.0;
   return value;
}
