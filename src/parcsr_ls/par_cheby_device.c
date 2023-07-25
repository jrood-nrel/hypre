/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Chebyshev setup and solve Device
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "float.h"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#include "_nalu_hypre_utilities.hpp"

/**
 * @brief waxpyz
 *
 * Performs
 * w = a*x+y.*z
 * For scalars w,x,y,z and constant a (indices 0, 1, 2, 3 respectively)
 */
template <typename T>
struct waxpyz
{
   typedef thrust::tuple<T &, T, T, T> Tuple;

   const T scale;
   waxpyz(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<0>(t) = scale * thrust::get<1>(t) + thrust::get<2>(t) * thrust::get<3>(t);
   }
};

/**
 * @brief wxypz
 *
 * Performs
 * o = x * (y .+ z)
 * For scalars o,x,y,z (indices 0, 1, 2, 3 respectively)
 */
template <typename T>
struct wxypz
{
   typedef thrust::tuple<T &, T, T, T> Tuple;
   __host__ __device__ void            operator()(Tuple t)
   {
      thrust::get<0>(t) = thrust::get<1>(t) * (thrust::get<2>(t) + thrust::get<3>(t));
   }
};
/**
 * @brief Saves u into o, then scales r placing the result in u
 *
 * Performs
 * o = u
 * u = r * a
 * For scalars o and u, with constant a
 */
template <typename T>
struct save_and_scale
{
   typedef thrust::tuple<T &, T &, T> Tuple;

   const T scale;

   save_and_scale(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<0>(t) = thrust::get<1>(t);
      thrust::get<1>(t) = thrust::get<2>(t) * scale;
   }
};

/**
 * @brief xpyz
 *
 * Performs
 * y = x + y .* z
 * For scalars x,y,z (indices 1,0,2 respectively)
 */
template <typename T>
struct xpyz
{
   typedef thrust::tuple<T &, T, T> Tuple;

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<0>(t) = thrust::get<1>(t) + thrust::get<2>(t) * thrust::get<0>(t);
   }
};

/**
 * @brief Solve using a chebyshev polynomial on the device
 *
 * @param[in] A Matrix to relax with
 * @param[in] f right-hand side
 * @param[in] ds_data Diagonal information
 * @param[in] coefs Polynomial coefficients
 * @param[in] order Order of the polynomial
 * @param[in] scale Whether or not to scale by diagonal
 * @param[in] scale Whether or not to use a variant
 * @param[in,out] u Initial/updated approximation
 * @param[out] v Temp vector
 * @param[out] v Temp Vector
 */
NALU_HYPRE_Int
nalu_hypre_ParCSRRelax_Cheby_SolveDevice(nalu_hypre_ParCSRMatrix *A, /* matrix to relax with */
                                    nalu_hypre_ParVector    *f, /* right-hand side */
                                    NALU_HYPRE_Real         *ds_data,
                                    NALU_HYPRE_Real         *coefs,
                                    NALU_HYPRE_Int           order, /* polynomial order */
                                    NALU_HYPRE_Int           scale, /* scale by diagonal?*/
                                    NALU_HYPRE_Int           variant,
                                    nalu_hypre_ParVector    *u,          /* initial/updated approximation */
                                    nalu_hypre_ParVector    *v,          /* temporary vector */
                                    nalu_hypre_ParVector    *r,          /*another temp vector */
                                    nalu_hypre_ParVector    *orig_u_vec, /*another temp vector */
                                    nalu_hypre_ParVector    *tmp_vec)       /*a potential temp vector */
{
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *u_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(u));
   NALU_HYPRE_Real      *f_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(f));
   NALU_HYPRE_Real      *v_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(v));

   NALU_HYPRE_Real *r_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(r));

   NALU_HYPRE_Int i;
   NALU_HYPRE_Int num_rows = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Real  mult;

   NALU_HYPRE_Int cheby_order;

   NALU_HYPRE_Real *tmp_data;

   /* u = u + p(A)r */

   if (order > 4) { order = 4; }
   if (order < 1) { order = 1; }

   /* we are using the order of p(A) */
   cheby_order = order - 1;

   nalu_hypre_assert(nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(orig_u_vec)) >= num_rows);
   NALU_HYPRE_Real *orig_u = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(orig_u_vec));

   if (!scale)
   {
      /* get residual: r = f - A*u */
      nalu_hypre_ParVectorCopy(f, r);
      nalu_hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      /* o = u; u = r .* coef */
      NALU_HYPRE_THRUST_CALL(
         for_each,
         thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)),
         thrust::make_zip_iterator(thrust::make_tuple(orig_u + num_rows, u_data + num_rows,
                                                      r_data + num_rows)),
         save_and_scale<NALU_HYPRE_Real>(coefs[cheby_order]));

      for (i = cheby_order - 1; i >= 0; i--)
      {
         nalu_hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, v);
         mult = coefs[i];

         /* u = mult * r + v */
         hypreDevice_ComplexAxpyn( r_data, num_rows, v_data, u_data, mult );
      }

      /* u = o + u */
      hypreDevice_ComplexAxpyn( orig_u, num_rows, u_data, u_data, 1.0);
   }
   else /* scaling! */
   {

      /*grab 1/nalu_hypre_sqrt(diagonal) */

      tmp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(tmp_vec));

      /* get ds_data and get scaled residual: r = D^(-1/2)f -
       * D^(-1/2)A*u */

      nalu_hypre_ParCSRMatrixMatvec(-1.0, A, u, 0.0, tmp_vec);
      /* r = ds .* (f + tmp) */

      /* TODO: It might be possible to merge this and the next call to:
       * r[j] = ds_data[j] * (f_data[j] + tmp_data[j]); o[j] = u[j]; u[j] = r[j] * coef */
      NALU_HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, f_data, tmp_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, f_data, tmp_data)) + num_rows,
                        wxypz<NALU_HYPRE_Real>());

      /* save original u, then start
         the iteration by multiplying r by the cheby coef.*/

      /* o = u;  u = r * coef */
      NALU_HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)) + num_rows,
                        save_and_scale<NALU_HYPRE_Real>(coefs[cheby_order]));

      /* now do the other coefficients */
      for (i = cheby_order - 1; i >= 0; i--)
      {
         /* v = D^(-1/2)AD^(-1/2)u */
         /* tmp = ds .* u */
         NALU_HYPRE_THRUST_CALL( transform, ds_data, ds_data + num_rows, u_data, tmp_data, _1 * _2 );

         nalu_hypre_ParCSRMatrixMatvec(1.0, A, tmp_vec, 0.0, v);

         /* u_new = coef*r + v*/
         mult = coefs[i];

         /* u = coef * r + ds .* v */
         NALU_HYPRE_THRUST_CALL(for_each,
                           thrust::make_zip_iterator(thrust::make_tuple(u_data, r_data, ds_data, v_data)),
                           thrust::make_zip_iterator(thrust::make_tuple(u_data, r_data, ds_data, v_data)) + num_rows,
                           waxpyz<NALU_HYPRE_Real>(mult));
      } /* end of cheby_order loop */

      /* now we have to scale u_data before adding it to u_orig*/

      /* u = orig_u + ds .* u */
      NALU_HYPRE_THRUST_CALL(
         for_each,
         thrust::make_zip_iterator(thrust::make_tuple(u_data, orig_u, ds_data)),
         thrust::make_zip_iterator(thrust::make_tuple(u_data + num_rows, orig_u + num_rows,
                                                      ds_data + num_rows)),
         xpyz<NALU_HYPRE_Real>());


   } /* end of scaling code */

   return nalu_hypre_error_flag;
}
#endif
