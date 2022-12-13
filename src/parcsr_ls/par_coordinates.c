/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * GenerateCoordinates
 *--------------------------------------------------------------------------*/

float *
GenerateCoordinates( MPI_Comm comm,
                     NALU_HYPRE_BigInt      nx,
                     NALU_HYPRE_BigInt      ny,
                     NALU_HYPRE_BigInt      nz,
                     NALU_HYPRE_Int      P,
                     NALU_HYPRE_Int      Q,
                     NALU_HYPRE_Int      R,
                     NALU_HYPRE_Int      p,
                     NALU_HYPRE_Int      q,
                     NALU_HYPRE_Int      r,
                     NALU_HYPRE_Int      coorddim)
{
   NALU_HYPRE_BigInt ix, iy, iz;
   NALU_HYPRE_Int cnt;

   NALU_HYPRE_Int nx_local, ny_local, nz_local;
   NALU_HYPRE_Int local_num_rows;

   NALU_HYPRE_BigInt *nx_part;
   NALU_HYPRE_BigInt *ny_part;
   NALU_HYPRE_BigInt *nz_part;

   float *coord = NULL;

   if (coorddim < 1 || coorddim > 3)
   {
      return NULL;
   }

   hypre_GeneratePartitioning(nx, P, &nx_part);
   hypre_GeneratePartitioning(ny, Q, &ny_part);
   hypre_GeneratePartitioning(nz, R, &nz_part);

   nx_local = (NALU_HYPRE_Int)(nx_part[p + 1] - nx_part[p]);
   ny_local = (NALU_HYPRE_Int)(ny_part[q + 1] - ny_part[q]);
   nz_local = (NALU_HYPRE_Int)(nz_part[r + 1] - nz_part[r]);

   local_num_rows = nx_local * ny_local * nz_local;

   coord = hypre_CTAlloc(float,  coorddim * local_num_rows, NALU_HYPRE_MEMORY_HOST);

   cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r + 1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q + 1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p + 1]; ix++)
         {
            /* set coordinates BM Oct 17, 2006 */
            if (coord)
            {
               if (nx > 1) { coord[cnt++] = ix; }
               if (ny > 1) { coord[cnt++] = iy; }
               if (nz > 1) { coord[cnt++] = iz; }
            }
         }
      }
   }

   hypre_TFree(nx_part, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part, NALU_HYPRE_MEMORY_HOST);

   return coord;
}
