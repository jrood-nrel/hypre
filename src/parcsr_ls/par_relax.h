/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_PAR_RELAX_HEADER
#define NALU_HYPRE_PAR_RELAX_HEADER

/* Non-Scale version */
static inline void
nalu_hypre_HybridGaussSeidelNS( NALU_HYPRE_Int     *A_diag_i,
                           NALU_HYPRE_Int     *A_diag_j,
                           NALU_HYPRE_Complex *A_diag_data,
                           NALU_HYPRE_Int     *A_offd_i,
                           NALU_HYPRE_Int     *A_offd_j,
                           NALU_HYPRE_Complex *A_offd_data,
                           NALU_HYPRE_Complex *f_data,
                           NALU_HYPRE_Int     *cf_marker,
                           NALU_HYPRE_Int      relax_points,
                           NALU_HYPRE_Complex *l1_norms,
                           NALU_HYPRE_Complex *u_data,
                           NALU_HYPRE_Complex *v_tmp_data,
                           NALU_HYPRE_Complex *v_ext_data,
                           NALU_HYPRE_Int      ibegin,
                           NALU_HYPRE_Int      iend,
                           NALU_HYPRE_Int      iorder,
                           NALU_HYPRE_Int      Skip_diag )
{
   NALU_HYPRE_Int i;
   const NALU_HYPRE_Complex zero = 0.0;

   /*-----------------------------------------------------------
    * Relax only C or F points as determined by relax_points.
    * If i is of the right type ( C or F or All) and diagonal is
    * nonzero, relax point i; otherwise, skip it.
    *-----------------------------------------------------------*/
   if (l1_norms)
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && l1_norms[i] != zero )
         {
            NALU_HYPRE_Int jj;
            NALU_HYPRE_Complex res = f_data[i];

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_diag_j[jj];
               res -= A_diag_data[jj] * u_data[ii];
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] = res / l1_norms[i];
            }
            else
            {
               u_data[i] += res / l1_norms[i];
            }
         }
      } /* for ( i = ...) */
   }
   else
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
         {
            NALU_HYPRE_Int jj;
            NALU_HYPRE_Complex res = f_data[i];

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_diag_j[jj];
               res -= A_diag_data[jj] * u_data[ii];
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
            else
            {
               u_data[i] += res / A_diag_data[A_diag_i[i]];
            }
         }
      } /* for ( i = ...) */
   }
}

/* Non-Scale Threaded version */
static inline void
nalu_hypre_HybridGaussSeidelNSThreads( NALU_HYPRE_Int     *A_diag_i,
                                  NALU_HYPRE_Int     *A_diag_j,
                                  NALU_HYPRE_Complex *A_diag_data,
                                  NALU_HYPRE_Int     *A_offd_i,
                                  NALU_HYPRE_Int     *A_offd_j,
                                  NALU_HYPRE_Complex *A_offd_data,
                                  NALU_HYPRE_Complex *f_data,
                                  NALU_HYPRE_Int     *cf_marker,
                                  NALU_HYPRE_Int      relax_points,
                                  NALU_HYPRE_Complex *l1_norms,
                                  NALU_HYPRE_Complex *u_data,
                                  NALU_HYPRE_Complex *v_tmp_data,
                                  NALU_HYPRE_Complex *v_ext_data,
                                  NALU_HYPRE_Int      ns,
                                  NALU_HYPRE_Int      ne,
                                  NALU_HYPRE_Int      ibegin,
                                  NALU_HYPRE_Int      iend,
                                  NALU_HYPRE_Int      iorder,
                                  NALU_HYPRE_Int      Skip_diag )
{
   NALU_HYPRE_Int i;
   const NALU_HYPRE_Complex zero = 0.0;

   /*-----------------------------------------------------------
    * Relax only C or F points as determined by relax_points.
    * If i is of the right type ( C or F or All) and diagonal is
    * nonzero, relax point i; otherwise, skip it.
    *-----------------------------------------------------------*/

   if (l1_norms)
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         /*-----------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          * If i is of the right type ( C or F or All) and diagonal is
          * nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && l1_norms[i] != zero )
         {
            NALU_HYPRE_Int jj;
            NALU_HYPRE_Complex res = f_data[i];

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_diag_j[jj];
               if (ii >= ns && ii < ne)
               {
                  res -= A_diag_data[jj] * u_data[ii];
               }
               else
               {
                  res -= A_diag_data[jj] * v_tmp_data[ii];
               }
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] = res / l1_norms[i];
            }
            else
            {
               u_data[i] += res / l1_norms[i];
            }
         }
      } /* for ( i = ...) */
   }
   else
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
         {
            NALU_HYPRE_Int jj;
            NALU_HYPRE_Complex res = f_data[i];

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_diag_j[jj];
               if (ii >= ns && ii < ne)
               {
                  res -= A_diag_data[jj] * u_data[ii];
               }
               else
               {
                  res -= A_diag_data[jj] * v_tmp_data[ii];
               }
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
            else
            {
               u_data[i] += res / A_diag_data[A_diag_i[i]];
            }
         }
      } /* for ( i = ...) */
   }
}

/* Scaled version */
static inline void
nalu_hypre_HybridGaussSeidel( NALU_HYPRE_Int     *A_diag_i,
                         NALU_HYPRE_Int     *A_diag_j,
                         NALU_HYPRE_Complex *A_diag_data,
                         NALU_HYPRE_Int     *A_offd_i,
                         NALU_HYPRE_Int     *A_offd_j,
                         NALU_HYPRE_Complex *A_offd_data,
                         NALU_HYPRE_Complex *f_data,
                         NALU_HYPRE_Int     *cf_marker,
                         NALU_HYPRE_Int      relax_points,
                         NALU_HYPRE_Real     relax_weight,
                         NALU_HYPRE_Real     omega,
                         NALU_HYPRE_Real     one_minus_omega,
                         NALU_HYPRE_Real     prod,
                         NALU_HYPRE_Complex *l1_norms,
                         NALU_HYPRE_Complex *u_data,
                         NALU_HYPRE_Complex *v_tmp_data,
                         NALU_HYPRE_Complex *v_ext_data,
                         NALU_HYPRE_Int      ibegin,
                         NALU_HYPRE_Int      iend,
                         NALU_HYPRE_Int      iorder,
                         NALU_HYPRE_Int      Skip_diag )
{
   NALU_HYPRE_Int i;
   const NALU_HYPRE_Complex zero = 0.0;

   /*-----------------------------------------------------------
    * Relax only C or F points as determined by relax_points.
    * If i is of the right type ( C or F or All) and diagonal is
    * nonzero, relax point i; otherwise, skip it.
    *-----------------------------------------------------------*/

   if (l1_norms)
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && l1_norms[i] != zero )
         {
            NALU_HYPRE_Int jj;
            NALU_HYPRE_Complex res = f_data[i];
            NALU_HYPRE_Complex res0 = 0.0;
            NALU_HYPRE_Complex res2 = 0.0;

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_diag_j[jj];
               res0 -= A_diag_data[jj] * u_data[ii];
               res2 += A_diag_data[jj] * v_tmp_data[ii];
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] *= prod;
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) / l1_norms[i];
            }
            else
            {
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) / l1_norms[i];
            }
         }
      } /* for ( i = ...) */
   }
   else
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
         {
            NALU_HYPRE_Int jj;
            NALU_HYPRE_Complex res = f_data[i];
            NALU_HYPRE_Complex res0 = 0.0;
            NALU_HYPRE_Complex res2 = 0.0;

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_diag_j[jj];
               res0 -= A_diag_data[jj] * u_data[ii];
               res2 += A_diag_data[jj] * v_tmp_data[ii];
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] *= prod;
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) /
                            A_diag_data[A_diag_i[i]];
            }
            else
            {
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) /
                            A_diag_data[A_diag_i[i]];
            }
         }
      } /* for ( i = ...) */
   }
}

/* Scaled Threaded version */
static inline void
nalu_hypre_HybridGaussSeidelThreads( NALU_HYPRE_Int     *A_diag_i,
                                NALU_HYPRE_Int     *A_diag_j,
                                NALU_HYPRE_Complex *A_diag_data,
                                NALU_HYPRE_Int     *A_offd_i,
                                NALU_HYPRE_Int     *A_offd_j,
                                NALU_HYPRE_Complex *A_offd_data,
                                NALU_HYPRE_Complex *f_data,
                                NALU_HYPRE_Int     *cf_marker,
                                NALU_HYPRE_Int      relax_points,
                                NALU_HYPRE_Real     relax_weight,
                                NALU_HYPRE_Real     omega,
                                NALU_HYPRE_Real     one_minus_omega,
                                NALU_HYPRE_Real     prod,
                                NALU_HYPRE_Complex *l1_norms,
                                NALU_HYPRE_Complex *u_data,
                                NALU_HYPRE_Complex *v_tmp_data,
                                NALU_HYPRE_Complex *v_ext_data,
                                NALU_HYPRE_Int      ns,
                                NALU_HYPRE_Int      ne,
                                NALU_HYPRE_Int      ibegin,
                                NALU_HYPRE_Int      iend,
                                NALU_HYPRE_Int      iorder,
                                NALU_HYPRE_Int      Skip_diag )
{
   NALU_HYPRE_Int i;
   const NALU_HYPRE_Complex zero = 0.0;

   /*-----------------------------------------------------------
    * Relax only C or F points as determined by relax_points.
    * If i is of the right type ( C or F or All) and diagonal is
    * nonzero, relax point i; otherwise, skip it.
    *-----------------------------------------------------------*/
   if (l1_norms)
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && l1_norms[i] != zero )
         {
            NALU_HYPRE_Int jj;
            NALU_HYPRE_Complex res = f_data[i];
            NALU_HYPRE_Complex res0 = 0.0;
            NALU_HYPRE_Complex res2 = 0.0;

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_diag_j[jj];
               if (ii >= ns && ii < ne)
               {
                  res0 -= A_diag_data[jj] * u_data[ii];
                  res2 += A_diag_data[jj] * v_tmp_data[ii];
               }
               else
               {
                  res -= A_diag_data[jj] * v_tmp_data[ii];
               }
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] *= prod;
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) / l1_norms[i];
            }
            else
            {
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) / l1_norms[i];
            }
         }
      } /* for ( i = ...) */
   }
   else
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
         {
            NALU_HYPRE_Int jj;
            NALU_HYPRE_Complex res = f_data[i];
            NALU_HYPRE_Complex res0 = 0.0;
            NALU_HYPRE_Complex res2 = 0.0;

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_diag_j[jj];
               if (ii >= ns && ii < ne)
               {
                  res0 -= A_diag_data[jj] * u_data[ii];
                  res2 += A_diag_data[jj] * v_tmp_data[ii];
               }
               else
               {
                  res -= A_diag_data[jj] * v_tmp_data[ii];
               }
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const NALU_HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] *= prod;
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) /
                            A_diag_data[A_diag_i[i]];
            }
            else
            {
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) /
                            A_diag_data[A_diag_i[i]];
            }
         }
      } /* for ( i = ...) */
   }
}


/* Ordered Version */
static inline void
nalu_hypre_HybridGaussSeidelOrderedNS( NALU_HYPRE_Int     *A_diag_i,
                                  NALU_HYPRE_Int     *A_diag_j,
                                  NALU_HYPRE_Complex *A_diag_data,
                                  NALU_HYPRE_Int     *A_offd_i,
                                  NALU_HYPRE_Int     *A_offd_j,
                                  NALU_HYPRE_Complex *A_offd_data,
                                  NALU_HYPRE_Complex *f_data,
                                  NALU_HYPRE_Int     *cf_marker,
                                  NALU_HYPRE_Int      relax_points,
                                  NALU_HYPRE_Complex *u_data,
                                  NALU_HYPRE_Complex *v_tmp_data,
                                  NALU_HYPRE_Complex *v_ext_data,
                                  NALU_HYPRE_Int      ibegin,
                                  NALU_HYPRE_Int      iend,
                                  NALU_HYPRE_Int      iorder,
                                  NALU_HYPRE_Int     *proc_ordering )
{
   NALU_HYPRE_Int j;
   const NALU_HYPRE_Complex zero = 0.0;

   for (j = ibegin; j != iend; j += iorder)
   {
      const NALU_HYPRE_Int i = proc_ordering[j];
      /*-----------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       * If i is of the right type ( C or F or All) and diagonal is
       * nonzero, relax point i; otherwise, skip it.
       *-----------------------------------------------------------*/
      if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
      {
         NALU_HYPRE_Int jj;
         NALU_HYPRE_Complex res = f_data[i];

         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            const NALU_HYPRE_Int ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
         }

         for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
         {
            const NALU_HYPRE_Int ii = A_offd_j[jj];
            res -= A_offd_data[jj] * v_ext_data[ii];
         }

         u_data[i] = res / A_diag_data[A_diag_i[i]];
      }
   } /* for ( i = ...) */
}

#endif /* #ifndef NALU_HYPRE_PAR_RELAX_HEADER */

