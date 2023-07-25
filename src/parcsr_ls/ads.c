/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_ls.h"
#include "float.h"
#include "ams.h"
#include "ads.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)
#if defined(NALU_HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_AMSComputePi_copy1(nalu_hypre_DeviceItem &item, NALU_HYPRE_Int nnz,
                                                  NALU_HYPRE_Int dim,
                                                  NALU_HYPRE_Int *j_in,
                                                  NALU_HYPRE_Int *j_out);
#if defined(NALU_HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_AMSComputePi_copy2(nalu_hypre_DeviceItem &item, NALU_HYPRE_Int nrows,
                                                  NALU_HYPRE_Int dim,
                                                  NALU_HYPRE_Int *i_in,
                                                  NALU_HYPRE_Real *data_in, NALU_HYPRE_Real *Gx_data, NALU_HYPRE_Real *Gy_data, NALU_HYPRE_Real *Gz_data,
                                                  NALU_HYPRE_Real *data_out);
#if defined(NALU_HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_AMSComputePixyz_copy(nalu_hypre_DeviceItem &item, NALU_HYPRE_Int nrows,
                                                    NALU_HYPRE_Int dim,
                                                    NALU_HYPRE_Int *i_in, NALU_HYPRE_Real *data_in, NALU_HYPRE_Real *Gx_data, NALU_HYPRE_Real *Gy_data, NALU_HYPRE_Real *Gz_data,
                                                    NALU_HYPRE_Real *data_x_out, NALU_HYPRE_Real *data_y_out, NALU_HYPRE_Real *data_z_out );
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSCreate
 *
 * Allocate the ADS solver structure.
 *--------------------------------------------------------------------------*/

void * nalu_hypre_ADSCreate(void)
{
   nalu_hypre_ADSData *ads_data;

   ads_data = nalu_hypre_CTAlloc(nalu_hypre_ADSData, 1, NALU_HYPRE_MEMORY_HOST);

   /* Default parameters */

   ads_data -> maxit = 20;             /* perform at most 20 iterations */
   ads_data -> tol = 1e-6;             /* convergence tolerance */
   ads_data -> print_level = 1;        /* print residual norm at each step */
   ads_data -> cycle_type = 1;         /* a 3-level multiplicative solver */
   ads_data -> A_relax_type = 2;       /* offd-l1-scaled GS */
   ads_data -> A_relax_times = 1;      /* one relaxation sweep */
   ads_data -> A_relax_weight = 1.0;   /* damping parameter */
   ads_data -> A_omega = 1.0;          /* SSOR coefficient */
   ads_data -> A_cheby_order = 2;      /* Cheby: order (1 -4 are vaild) */
   ads_data -> A_cheby_fraction = 0.3; /* Cheby: fraction of spectrum to smooth */

   ads_data -> B_C_cycle_type = 11;    /* a 5-level multiplicative solver */
   ads_data -> B_C_coarsen_type = 10;  /* HMIS coarsening */
   ads_data -> B_C_agg_levels = 1;     /* Levels of aggressive coarsening */
   ads_data -> B_C_relax_type = 3;     /* hybrid G-S/Jacobi */
   ads_data -> B_C_theta = 0.25;       /* strength threshold */
   ads_data -> B_C_interp_type = 0;    /* interpolation type */
   ads_data -> B_C_Pmax = 0;           /* max nonzero elements in interp. rows */
   ads_data -> B_Pi_coarsen_type = 10; /* HMIS coarsening */
   ads_data -> B_Pi_agg_levels = 1;    /* Levels of aggressive coarsening */
   ads_data -> B_Pi_relax_type = 3;    /* hybrid G-S/Jacobi */
   ads_data -> B_Pi_theta = 0.25;      /* strength threshold */
   ads_data -> B_Pi_interp_type = 0;   /* interpolation type */
   ads_data -> B_Pi_Pmax = 0;          /* max nonzero elements in interp. rows */

   /* The rest of the fields are initialized using the Set functions */

   ads_data -> A     = NULL;
   ads_data -> C     = NULL;
   ads_data -> A_C   = NULL;
   ads_data -> B_C   = 0;
   ads_data -> Pi    = NULL;
   ads_data -> A_Pi  = NULL;
   ads_data -> B_Pi  = 0;
   ads_data -> Pix    = NULL;
   ads_data -> Piy    = NULL;
   ads_data -> Piz    = NULL;
   ads_data -> A_Pix  = NULL;
   ads_data -> A_Piy  = NULL;
   ads_data -> A_Piz  = NULL;
   ads_data -> B_Pix  = 0;
   ads_data -> B_Piy  = 0;
   ads_data -> B_Piz  = 0;
   ads_data -> G     = NULL;
   ads_data -> x     = NULL;
   ads_data -> y     = NULL;
   ads_data -> z     = NULL;
   ads_data -> zz  = NULL;

   ads_data -> r0  = NULL;
   ads_data -> g0  = NULL;
   ads_data -> r1  = NULL;
   ads_data -> g1  = NULL;
   ads_data -> r2  = NULL;
   ads_data -> g2  = NULL;

   ads_data -> A_l1_norms = NULL;
   ads_data -> A_max_eig_est = 0;
   ads_data -> A_min_eig_est = 0;

   ads_data -> owns_Pi = 1;
   ads_data -> ND_Pi   = NULL;
   ads_data -> ND_Pix  = NULL;
   ads_data -> ND_Piy  = NULL;
   ads_data -> ND_Piz  = NULL;

   return (void *) ads_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSDestroy
 *
 * Deallocate the ADS solver structure. Note that the input data (given
 * through the Set functions) is not destroyed.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSDestroy(void *solver)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;

   if (!ads_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (ads_data -> A_C)
   {
      nalu_hypre_ParCSRMatrixDestroy(ads_data -> A_C);
   }
   if (ads_data -> B_C)
   {
      NALU_HYPRE_AMSDestroy(ads_data -> B_C);
   }

   if (ads_data -> owns_Pi && ads_data -> Pi)
   {
      nalu_hypre_ParCSRMatrixDestroy(ads_data -> Pi);
   }
   if (ads_data -> A_Pi)
   {
      nalu_hypre_ParCSRMatrixDestroy(ads_data -> A_Pi);
   }
   if (ads_data -> B_Pi)
   {
      NALU_HYPRE_BoomerAMGDestroy(ads_data -> B_Pi);
   }

   if (ads_data -> owns_Pi && ads_data -> Pix)
   {
      nalu_hypre_ParCSRMatrixDestroy(ads_data -> Pix);
   }
   if (ads_data -> A_Pix)
   {
      nalu_hypre_ParCSRMatrixDestroy(ads_data -> A_Pix);
   }
   if (ads_data -> B_Pix)
   {
      NALU_HYPRE_BoomerAMGDestroy(ads_data -> B_Pix);
   }
   if (ads_data -> owns_Pi && ads_data -> Piy)
   {
      nalu_hypre_ParCSRMatrixDestroy(ads_data -> Piy);
   }
   if (ads_data -> A_Piy)
   {
      nalu_hypre_ParCSRMatrixDestroy(ads_data -> A_Piy);
   }
   if (ads_data -> B_Piy)
   {
      NALU_HYPRE_BoomerAMGDestroy(ads_data -> B_Piy);
   }
   if (ads_data -> owns_Pi && ads_data -> Piz)
   {
      nalu_hypre_ParCSRMatrixDestroy(ads_data -> Piz);
   }
   if (ads_data -> A_Piz)
   {
      nalu_hypre_ParCSRMatrixDestroy(ads_data -> A_Piz);
   }
   if (ads_data -> B_Piz)
   {
      NALU_HYPRE_BoomerAMGDestroy(ads_data -> B_Piz);
   }

   if (ads_data -> r0)
   {
      nalu_hypre_ParVectorDestroy(ads_data -> r0);
   }
   if (ads_data -> g0)
   {
      nalu_hypre_ParVectorDestroy(ads_data -> g0);
   }
   if (ads_data -> r1)
   {
      nalu_hypre_ParVectorDestroy(ads_data -> r1);
   }
   if (ads_data -> g1)
   {
      nalu_hypre_ParVectorDestroy(ads_data -> g1);
   }
   if (ads_data -> r2)
   {
      nalu_hypre_ParVectorDestroy(ads_data -> r2);
   }
   if (ads_data -> g2)
   {
      nalu_hypre_ParVectorDestroy(ads_data -> g2);
   }
   if (ads_data -> zz)
   {
      nalu_hypre_ParVectorDestroy(ads_data -> zz);
   }

   nalu_hypre_SeqVectorDestroy(ads_data -> A_l1_norms);

   /* C, G, x, y and z are not destroyed */

   if (ads_data)
   {
      nalu_hypre_TFree(ads_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetDiscreteCurl
 *
 * Set the discrete curl matrix C.
 * This function should be called before nalu_hypre_ADSSetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetDiscreteCurl(void *solver,
                                   nalu_hypre_ParCSRMatrix *C)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> C = C;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetDiscreteGradient
 *
 * Set the discrete gradient matrix G.
 * This function should be called before nalu_hypre_ADSSetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetDiscreteGradient(void *solver,
                                       nalu_hypre_ParCSRMatrix *G)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> G = G;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetCoordinateVectors
 *
 * Set the x, y and z coordinates of the vertices in the mesh.
 * This function should be called before nalu_hypre_ADSSetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetCoordinateVectors(void *solver,
                                        nalu_hypre_ParVector *x,
                                        nalu_hypre_ParVector *y,
                                        nalu_hypre_ParVector *z)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> x = x;
   ads_data -> y = y;
   ads_data -> z = z;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetInterpolations
 *
 * Set the (components of) the Raviart-Thomas (RT_Pi) and the Nedelec (ND_Pi)
 * interpolation matrices.
 *
 * This function is generally intended to be used only for high-order H(div)
 * discretizations (in the lowest order case, these matrices are constructed
 * internally in ADS from the discreet gradient and curl matrices and the
 * coordinates of the vertices), though it can also be used in the lowest-order
 * case or for other types of discretizations.
 *
 * By definition, RT_Pi and ND_Pi are the matrix representations of the linear
 * operators that interpolate (high-order) vector nodal finite elements into the
 * (high-order) Raviart-Thomas and Nedelec spaces. The component matrices are
 * defined in both cases as Pix phi = Pi (phi,0,0) and similarly for Piy and
 * Piz. Note that all these operators depend on the choice of the basis and
 * degrees of freedom in the high-order spaces.
 *
 * The column numbering of RT_Pi and ND_Pi should be node-based, i.e. the x/y/z
 * components of the first node (vertex or high-order dof) should be listed
 * first, followed by the x/y/z components of the second node and so on (see the
 * documentation of NALU_HYPRE_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before nalu_hypre_ADSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * {RT_Pi} and {RT_Pix,RT_Piy,RT_Piz} needs to be specified (though it is OK to
 * provide both).  If RT_Pix is NULL, then scalar Pi-based ADS cycles, i.e.
 * those with cycle_type > 10, will be unavailable. Similarly, ADS cycles based
 * on monolithic Pi (cycle_type < 10) require that RT_Pi is not NULL. The same
 * restrictions hold for the sets {ND_Pi} and {ND_Pix,ND_Piy,ND_Piz} -- only one
 * of them needs to be specified, and the availability of each enables different
 * AMS cycle type options for the subspace solve.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetInterpolations(void *solver,
                                     nalu_hypre_ParCSRMatrix *RT_Pi,
                                     nalu_hypre_ParCSRMatrix *RT_Pix,
                                     nalu_hypre_ParCSRMatrix *RT_Piy,
                                     nalu_hypre_ParCSRMatrix *RT_Piz,
                                     nalu_hypre_ParCSRMatrix *ND_Pi,
                                     nalu_hypre_ParCSRMatrix *ND_Pix,
                                     nalu_hypre_ParCSRMatrix *ND_Piy,
                                     nalu_hypre_ParCSRMatrix *ND_Piz)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> Pi = RT_Pi;
   ads_data -> Pix = RT_Pix;
   ads_data -> Piy = RT_Piy;
   ads_data -> Piz = RT_Piz;
   ads_data -> ND_Pi = ND_Pi;
   ads_data -> ND_Pix = ND_Pix;
   ads_data -> ND_Piy = ND_Piy;
   ads_data -> ND_Piz = ND_Piz;
   ads_data -> owns_Pi = 0;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetMaxIter
 *
 * Set the maximum number of iterations in the auxiliary-space method.
 * The default value is 20. To use the ADS solver as a preconditioner,
 * set maxit to 1, tol to 0.0 and print_level to 0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetMaxIter(void *solver,
                              NALU_HYPRE_Int maxit)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> maxit = maxit;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetTol
 *
 * Set the convergence tolerance (if the method is used as a solver).
 * The default value is 1e-6.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetTol(void *solver,
                          NALU_HYPRE_Real tol)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> tol = tol;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetCycleType
 *
 * Choose which three-level solver to use. Possible values are:
 *
 *   1 = 3-level multipl. solver (01210)      <-- small solution time
 *   2 = 3-level additive solver (0+1+2)
 *   3 = 3-level multipl. solver (02120)
 *   4 = 3-level additive solver (010+2)
 *   5 = 3-level multipl. solver (0102010)    <-- small solution time
 *   6 = 3-level additive solver (1+020)
 *   7 = 3-level multipl. solver (0201020)    <-- small number of iterations
 *   8 = 3-level additive solver (0(1+2)0)    <-- small solution time
 *   9 = 3-level multipl. solver (01210) with discrete divergence
 *  11 = 5-level multipl. solver (013454310)  <-- small solution time, memory
 *  12 = 5-level additive solver (0+1+3+4+5)
 *  13 = 5-level multipl. solver (034515430)  <-- small solution time, memory
 *  14 = 5-level additive solver (01(3+4+5)10)
 *
 * The default value is 1.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetCycleType(void *solver,
                                NALU_HYPRE_Int cycle_type)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> cycle_type = cycle_type;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetPrintLevel
 *
 * Control how much information is printed during the solution iterations.
 * The defaut values is 1 (print residual norm at each step).
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetPrintLevel(void *solver,
                                 NALU_HYPRE_Int print_level)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> print_level = print_level;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetSmoothingOptions
 *
 * Set relaxation parameters for A. Default values: 2, 1, 1.0, 1.0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetSmoothingOptions(void *solver,
                                       NALU_HYPRE_Int A_relax_type,
                                       NALU_HYPRE_Int A_relax_times,
                                       NALU_HYPRE_Real A_relax_weight,
                                       NALU_HYPRE_Real A_omega)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> A_relax_type = A_relax_type;
   ads_data -> A_relax_times = A_relax_times;
   ads_data -> A_relax_weight = A_relax_weight;
   ads_data -> A_omega = A_omega;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetChebySmoothingOptions
 *
 * Set parameters for Chebyshev relaxation. Default values: 2, 0.3.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetChebySmoothingOptions(void *solver,
                                            NALU_HYPRE_Int A_cheby_order,
                                            NALU_HYPRE_Real A_cheby_fraction)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> A_cheby_order =  A_cheby_order;
   ads_data -> A_cheby_fraction =  A_cheby_fraction;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetAMSOptions
 *
 * Set AMS parameters for B_C. Default values: 11, 10, 1, 3, 0.25, 0, 0.
 *
 * Note that B_C_cycle_type should be greater than 10, unless the high-order
 * interface of nalu_hypre_ADSSetInterpolations is being used!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetAMSOptions(void *solver,
                                 NALU_HYPRE_Int B_C_cycle_type,
                                 NALU_HYPRE_Int B_C_coarsen_type,
                                 NALU_HYPRE_Int B_C_agg_levels,
                                 NALU_HYPRE_Int B_C_relax_type,
                                 NALU_HYPRE_Real B_C_theta,
                                 NALU_HYPRE_Int B_C_interp_type,
                                 NALU_HYPRE_Int B_C_Pmax)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> B_C_cycle_type = B_C_cycle_type;
   ads_data -> B_C_coarsen_type = B_C_coarsen_type;
   ads_data -> B_C_agg_levels = B_C_agg_levels;
   ads_data -> B_C_relax_type = B_C_relax_type;
   ads_data -> B_C_theta = B_C_theta;
   ads_data -> B_C_interp_type = B_C_interp_type;
   ads_data -> B_C_Pmax = B_C_Pmax;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetAMGOptions
 *
 * Set AMG parameters for B_Pi. Default values: 10, 1, 3, 0.25, 0, 0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetAMGOptions(void *solver,
                                 NALU_HYPRE_Int B_Pi_coarsen_type,
                                 NALU_HYPRE_Int B_Pi_agg_levels,
                                 NALU_HYPRE_Int B_Pi_relax_type,
                                 NALU_HYPRE_Real B_Pi_theta,
                                 NALU_HYPRE_Int B_Pi_interp_type,
                                 NALU_HYPRE_Int B_Pi_Pmax)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   ads_data -> B_Pi_coarsen_type = B_Pi_coarsen_type;
   ads_data -> B_Pi_agg_levels = B_Pi_agg_levels;
   ads_data -> B_Pi_relax_type = B_Pi_relax_type;
   ads_data -> B_Pi_theta = B_Pi_theta;
   ads_data -> B_Pi_interp_type = B_Pi_interp_type;
   ads_data -> B_Pi_Pmax = B_Pi_Pmax;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSComputePi
 *
 * Construct the Pi interpolation matrix, which maps the space of vector
 * linear finite elements to the space of face finite elements.
 *
 * The construction is based on the fact that Pi = [Pi_x, Pi_y, Pi_z], where
 * each block has the same sparsity structure as C*G, with entries that can be
 * computed from the vectors RT100, RT010 and RT001.
 *
 * We assume a constant number of vertices per face (no prisms or pyramids).
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSComputePi(nalu_hypre_ParCSRMatrix *A,
                             nalu_hypre_ParCSRMatrix *C,
                             nalu_hypre_ParCSRMatrix *G,
                             nalu_hypre_ParVector *x,
                             nalu_hypre_ParVector *y,
                             nalu_hypre_ParVector *z,
                             nalu_hypre_ParCSRMatrix *PiNDx,
                             nalu_hypre_ParCSRMatrix *PiNDy,
                             nalu_hypre_ParCSRMatrix *PiNDz,
                             nalu_hypre_ParCSRMatrix **Pi_ptr)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   nalu_hypre_ParCSRMatrix *Pi;

   /* Compute the representations of the coordinate vectors, RT100, RT010 and
      RT001, in the Raviart-Thomas space, by observing that the RT coordinates
      of (1,0,0) = -curl (0,z,0) are given by C*PiNDy*z, etc. (We ignore the
      minus sign since it is irrelevant for the coarse-grid correction.) */
   nalu_hypre_ParVector *RT100, *RT010, *RT001;
   {
      nalu_hypre_ParVector *PiNDlin = nalu_hypre_ParVectorInRangeOf(PiNDx);

      RT100 = nalu_hypre_ParVectorInRangeOf(C);
      nalu_hypre_ParCSRMatrixMatvec(1.0, PiNDy, z, 0.0, PiNDlin);
      nalu_hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT100);
      RT010 = nalu_hypre_ParVectorInRangeOf(C);
      nalu_hypre_ParCSRMatrixMatvec(1.0, PiNDz, x, 0.0, PiNDlin);
      nalu_hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT010);
      RT001 = nalu_hypre_ParVectorInRangeOf(C);
      nalu_hypre_ParCSRMatrixMatvec(1.0, PiNDx, y, 0.0, PiNDlin);
      nalu_hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT001);

      nalu_hypre_ParVectorDestroy(PiNDlin);
   }

   /* Compute Pi = [Pi_x, Pi_y, Pi_z] */
   {
      NALU_HYPRE_Int i, j, d;

      NALU_HYPRE_Real *RT100_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(RT100));
      NALU_HYPRE_Real *RT010_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(RT010));
      NALU_HYPRE_Real *RT001_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(RT001));

      /* Each component of Pi has the sparsity pattern of the topological
         face-to-vertex matrix. */
      nalu_hypre_ParCSRMatrix *F2V;
#if defined(NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         F2V = nalu_hypre_ParCSRMatMat(C, G);
      }
      else
#endif
      {
         F2V = nalu_hypre_ParMatmul(C, G);
      }

      /* Create the parallel interpolation matrix */
      {
         MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(F2V);
         NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(F2V);
         NALU_HYPRE_BigInt global_num_cols = 3 * nalu_hypre_ParCSRMatrixGlobalNumCols(F2V);
         NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(F2V);
         NALU_HYPRE_BigInt *col_starts;
         NALU_HYPRE_Int col_starts_size;
         NALU_HYPRE_Int num_cols_offd = 3 * nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(F2V));
         NALU_HYPRE_Int num_nonzeros_diag = 3 * nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(F2V));
         NALU_HYPRE_Int num_nonzeros_offd = 3 * nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(F2V));
         NALU_HYPRE_BigInt *col_starts_F2V = nalu_hypre_ParCSRMatrixColStarts(F2V);
         col_starts_size = 2;
         col_starts = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, col_starts_size, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < col_starts_size; i++)
         {
            col_starts[i] = 3 * col_starts_F2V[i];
         }

         Pi = nalu_hypre_ParCSRMatrixCreate(comm,
                                       global_num_rows,
                                       global_num_cols,
                                       row_starts,
                                       col_starts,
                                       num_cols_offd,
                                       num_nonzeros_diag,
                                       num_nonzeros_offd);

         nalu_hypre_ParCSRMatrixOwnsData(Pi) = 1;
         nalu_hypre_ParCSRMatrixInitialize(Pi);
      }

      /* Fill-in the diagonal part */
      {
         nalu_hypre_CSRMatrix *F2V_diag = nalu_hypre_ParCSRMatrixDiag(F2V);
         NALU_HYPRE_Int *F2V_diag_I = nalu_hypre_CSRMatrixI(F2V_diag);
         NALU_HYPRE_Int *F2V_diag_J = nalu_hypre_CSRMatrixJ(F2V_diag);

         NALU_HYPRE_Int F2V_diag_nrows = nalu_hypre_CSRMatrixNumRows(F2V_diag);
         NALU_HYPRE_Int F2V_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(F2V_diag);

         nalu_hypre_CSRMatrix *Pi_diag = nalu_hypre_ParCSRMatrixDiag(Pi);
         NALU_HYPRE_Int *Pi_diag_I = nalu_hypre_CSRMatrixI(Pi_diag);
         NALU_HYPRE_Int *Pi_diag_J = nalu_hypre_CSRMatrixJ(Pi_diag);
         NALU_HYPRE_Real *Pi_diag_data = nalu_hypre_CSRMatrixData(Pi_diag);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            hypreDevice_IntScalen( F2V_diag_I, F2V_diag_nrows + 1, Pi_diag_I, 3);

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(F2V_diag_nnz, "thread", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              F2V_diag_nnz, 3, F2V_diag_J, Pi_diag_J );

            gDim = nalu_hypre_GetDefaultDeviceGridDimension(F2V_diag_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy2, gDim, bDim,
                              F2V_diag_nrows, 3, F2V_diag_I, NULL, RT100_data, RT010_data, RT001_data,
                              Pi_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < F2V_diag_nrows + 1; i++)
            {
               Pi_diag_I[i] = 3 * F2V_diag_I[i];
            }

            for (i = 0; i < F2V_diag_nnz; i++)
               for (d = 0; d < 3; d++)
               {
                  Pi_diag_J[3 * i + d] = 3 * F2V_diag_J[i] + d;
               }

            for (i = 0; i < F2V_diag_nrows; i++)
               for (j = F2V_diag_I[i]; j < F2V_diag_I[i + 1]; j++)
               {
                  *Pi_diag_data++ = RT100_data[i];
                  *Pi_diag_data++ = RT010_data[i];
                  *Pi_diag_data++ = RT001_data[i];
               }
         }
      }

      /* Fill-in the off-diagonal part */
      {
         nalu_hypre_CSRMatrix *F2V_offd = nalu_hypre_ParCSRMatrixOffd(F2V);
         NALU_HYPRE_Int *F2V_offd_I = nalu_hypre_CSRMatrixI(F2V_offd);
         NALU_HYPRE_Int *F2V_offd_J = nalu_hypre_CSRMatrixJ(F2V_offd);

         NALU_HYPRE_Int F2V_offd_nrows = nalu_hypre_CSRMatrixNumRows(F2V_offd);
         NALU_HYPRE_Int F2V_offd_ncols = nalu_hypre_CSRMatrixNumCols(F2V_offd);
         NALU_HYPRE_Int F2V_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(F2V_offd);

         nalu_hypre_CSRMatrix *Pi_offd = nalu_hypre_ParCSRMatrixOffd(Pi);
         NALU_HYPRE_Int *Pi_offd_I = nalu_hypre_CSRMatrixI(Pi_offd);
         NALU_HYPRE_Int *Pi_offd_J = nalu_hypre_CSRMatrixJ(Pi_offd);
         NALU_HYPRE_Real *Pi_offd_data = nalu_hypre_CSRMatrixData(Pi_offd);

         NALU_HYPRE_BigInt *F2V_cmap = nalu_hypre_ParCSRMatrixColMapOffd(F2V);
         NALU_HYPRE_BigInt *Pi_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Pi);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            if (F2V_offd_ncols)
            {
               hypreDevice_IntScalen( F2V_offd_I, F2V_offd_nrows + 1, Pi_offd_I, 3 );
            }

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(F2V_offd_nnz, "thread", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              F2V_offd_nnz, 3, F2V_offd_J, Pi_offd_J );

            gDim = nalu_hypre_GetDefaultDeviceGridDimension(F2V_offd_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy2, gDim, bDim,
                              F2V_offd_nrows, 3, F2V_offd_I, NULL, RT100_data, RT010_data, RT001_data,
                              Pi_offd_data );
         }
         else
#endif
         {
            if (F2V_offd_ncols)
               for (i = 0; i < F2V_offd_nrows + 1; i++)
               {
                  Pi_offd_I[i] = 3 * F2V_offd_I[i];
               }

            for (i = 0; i < F2V_offd_nnz; i++)
               for (d = 0; d < 3; d++)
               {
                  Pi_offd_J[3 * i + d] = 3 * F2V_offd_J[i] + d;
               }

            for (i = 0; i < F2V_offd_nrows; i++)
               for (j = F2V_offd_I[i]; j < F2V_offd_I[i + 1]; j++)
               {
                  *Pi_offd_data++ = RT100_data[i];
                  *Pi_offd_data++ = RT010_data[i];
                  *Pi_offd_data++ = RT001_data[i];
               }
         }

         for (i = 0; i < F2V_offd_ncols; i++)
            for (d = 0; d < 3; d++)
            {
               Pi_cmap[3 * i + d] = 3 * F2V_cmap[i] + (NALU_HYPRE_BigInt)d;
            }
      }

      nalu_hypre_ParCSRMatrixDestroy(F2V);
   }

   nalu_hypre_ParVectorDestroy(RT100);
   nalu_hypre_ParVectorDestroy(RT010);
   nalu_hypre_ParVectorDestroy(RT001);

   *Pi_ptr = Pi;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSComputePixyz
 *
 * Construct the components Pix, Piy, Piz of the interpolation matrix Pi, which
 * maps the space of vector linear finite elements to the space of face finite
 * elements.
 *
 * The construction is based on the fact that each component has the same
 * sparsity structure as the matrix C*G, with entries that can be computed from
 * the vectors RT100, RT010 and RT001.
 *
 * We assume a constant number of vertices per face (no prisms or pyramids).
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSComputePixyz(nalu_hypre_ParCSRMatrix *A,
                                nalu_hypre_ParCSRMatrix *C,
                                nalu_hypre_ParCSRMatrix *G,
                                nalu_hypre_ParVector *x,
                                nalu_hypre_ParVector *y,
                                nalu_hypre_ParVector *z,
                                nalu_hypre_ParCSRMatrix *PiNDx,
                                nalu_hypre_ParCSRMatrix *PiNDy,
                                nalu_hypre_ParCSRMatrix *PiNDz,
                                nalu_hypre_ParCSRMatrix **Pix_ptr,
                                nalu_hypre_ParCSRMatrix **Piy_ptr,
                                nalu_hypre_ParCSRMatrix **Piz_ptr)
{
   nalu_hypre_ParCSRMatrix *Pix, *Piy, *Piz;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   /* Compute the representations of the coordinate vectors, RT100, RT010 and
      RT001, in the Raviart-Thomas space, by observing that the RT coordinates
      of (1,0,0) = -curl (0,z,0) are given by C*PiNDy*z, etc. (We ignore the
      minus sign since it is irrelevant for the coarse-grid correction.) */
   nalu_hypre_ParVector *RT100, *RT010, *RT001;
   {
      nalu_hypre_ParVector *PiNDlin = nalu_hypre_ParVectorInRangeOf(PiNDx);

      RT100 = nalu_hypre_ParVectorInRangeOf(C);
      nalu_hypre_ParCSRMatrixMatvec(1.0, PiNDy, z, 0.0, PiNDlin);
      nalu_hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT100);
      RT010 = nalu_hypre_ParVectorInRangeOf(C);
      nalu_hypre_ParCSRMatrixMatvec(1.0, PiNDz, x, 0.0, PiNDlin);
      nalu_hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT010);
      RT001 = nalu_hypre_ParVectorInRangeOf(C);
      nalu_hypre_ParCSRMatrixMatvec(1.0, PiNDx, y, 0.0, PiNDlin);
      nalu_hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT001);

      nalu_hypre_ParVectorDestroy(PiNDlin);
   }

   /* Compute Pix, Piy, Piz */
   {
      NALU_HYPRE_Int i, j;

      NALU_HYPRE_Real *RT100_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(RT100));
      NALU_HYPRE_Real *RT010_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(RT010));
      NALU_HYPRE_Real *RT001_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(RT001));

      /* Each component of Pi has the sparsity pattern of the topological
         face-to-vertex matrix. */
      nalu_hypre_ParCSRMatrix *F2V;

#if defined(NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         F2V = nalu_hypre_ParCSRMatMat(C, G);
      }
      else
#endif
      {
         F2V = nalu_hypre_ParMatmul(C, G);
      }

      /* Create the components of the parallel interpolation matrix */
      {
         MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(F2V);
         NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(F2V);
         NALU_HYPRE_BigInt global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(F2V);
         NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(F2V);
         NALU_HYPRE_BigInt *col_starts = nalu_hypre_ParCSRMatrixColStarts(F2V);
         NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(F2V));
         NALU_HYPRE_Int num_nonzeros_diag = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(F2V));
         NALU_HYPRE_Int num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(F2V));

         Pix = nalu_hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         nalu_hypre_ParCSRMatrixOwnsData(Pix) = 1;
         nalu_hypre_ParCSRMatrixInitialize(Pix);

         Piy = nalu_hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         nalu_hypre_ParCSRMatrixOwnsData(Piy) = 1;
         nalu_hypre_ParCSRMatrixInitialize(Piy);

         Piz = nalu_hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         nalu_hypre_ParCSRMatrixOwnsData(Piz) = 1;
         nalu_hypre_ParCSRMatrixInitialize(Piz);
      }

      /* Fill-in the diagonal part */
      {
         nalu_hypre_CSRMatrix *F2V_diag = nalu_hypre_ParCSRMatrixDiag(F2V);
         NALU_HYPRE_Int *F2V_diag_I = nalu_hypre_CSRMatrixI(F2V_diag);
         NALU_HYPRE_Int *F2V_diag_J = nalu_hypre_CSRMatrixJ(F2V_diag);

         NALU_HYPRE_Int F2V_diag_nrows = nalu_hypre_CSRMatrixNumRows(F2V_diag);
         NALU_HYPRE_Int F2V_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(F2V_diag);

         nalu_hypre_CSRMatrix *Pix_diag = nalu_hypre_ParCSRMatrixDiag(Pix);
         NALU_HYPRE_Int *Pix_diag_I = nalu_hypre_CSRMatrixI(Pix_diag);
         NALU_HYPRE_Int *Pix_diag_J = nalu_hypre_CSRMatrixJ(Pix_diag);
         NALU_HYPRE_Real *Pix_diag_data = nalu_hypre_CSRMatrixData(Pix_diag);

         nalu_hypre_CSRMatrix *Piy_diag = nalu_hypre_ParCSRMatrixDiag(Piy);
         NALU_HYPRE_Int *Piy_diag_I = nalu_hypre_CSRMatrixI(Piy_diag);
         NALU_HYPRE_Int *Piy_diag_J = nalu_hypre_CSRMatrixJ(Piy_diag);
         NALU_HYPRE_Real *Piy_diag_data = nalu_hypre_CSRMatrixData(Piy_diag);

         nalu_hypre_CSRMatrix *Piz_diag = nalu_hypre_ParCSRMatrixDiag(Piz);
         NALU_HYPRE_Int *Piz_diag_I = nalu_hypre_CSRMatrixI(Piz_diag);
         NALU_HYPRE_Int *Piz_diag_J = nalu_hypre_CSRMatrixJ(Piz_diag);
         NALU_HYPRE_Real *Piz_diag_data = nalu_hypre_CSRMatrixData(Piz_diag);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if defined(NALU_HYPRE_USING_SYCL)
            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(F2V_diag_I, F2V_diag_I, F2V_diag_I),
                               F2V_diag_nrows + 1,
                               oneapi::dpl::make_zip_iterator(Pix_diag_I, Piy_diag_I, Piz_diag_I) );

            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(F2V_diag_J, F2V_diag_J, F2V_diag_J),
                               F2V_diag_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_diag_J, Piy_diag_J, Piz_diag_J) );
#else
            NALU_HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(F2V_diag_I, F2V_diag_I, F2V_diag_I)),
                               F2V_diag_nrows + 1,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_I, Piy_diag_I, Piz_diag_I)) );

            NALU_HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(F2V_diag_J, F2V_diag_J, F2V_diag_J)),
                               F2V_diag_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_J, Piy_diag_J, Piz_diag_J)) );
#endif

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(F2V_diag_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              F2V_diag_nrows, 3, F2V_diag_I, NULL, RT100_data, RT010_data, RT001_data,
                              Pix_diag_data, Piy_diag_data, Piz_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < F2V_diag_nrows + 1; i++)
            {
               Pix_diag_I[i] = F2V_diag_I[i];
               Piy_diag_I[i] = F2V_diag_I[i];
               Piz_diag_I[i] = F2V_diag_I[i];
            }

            for (i = 0; i < F2V_diag_nnz; i++)
            {
               Pix_diag_J[i] = F2V_diag_J[i];
               Piy_diag_J[i] = F2V_diag_J[i];
               Piz_diag_J[i] = F2V_diag_J[i];
            }

            for (i = 0; i < F2V_diag_nrows; i++)
               for (j = F2V_diag_I[i]; j < F2V_diag_I[i + 1]; j++)
               {
                  *Pix_diag_data++ = RT100_data[i];
                  *Piy_diag_data++ = RT010_data[i];
                  *Piz_diag_data++ = RT001_data[i];
               }
         }
      }

      /* Fill-in the off-diagonal part */
      {
         nalu_hypre_CSRMatrix *F2V_offd = nalu_hypre_ParCSRMatrixOffd(F2V);
         NALU_HYPRE_Int *F2V_offd_I = nalu_hypre_CSRMatrixI(F2V_offd);
         NALU_HYPRE_Int *F2V_offd_J = nalu_hypre_CSRMatrixJ(F2V_offd);

         NALU_HYPRE_Int F2V_offd_nrows = nalu_hypre_CSRMatrixNumRows(F2V_offd);
         NALU_HYPRE_Int F2V_offd_ncols = nalu_hypre_CSRMatrixNumCols(F2V_offd);
         NALU_HYPRE_Int F2V_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(F2V_offd);

         nalu_hypre_CSRMatrix *Pix_offd = nalu_hypre_ParCSRMatrixOffd(Pix);
         NALU_HYPRE_Int *Pix_offd_I = nalu_hypre_CSRMatrixI(Pix_offd);
         NALU_HYPRE_Int *Pix_offd_J = nalu_hypre_CSRMatrixJ(Pix_offd);
         NALU_HYPRE_Real *Pix_offd_data = nalu_hypre_CSRMatrixData(Pix_offd);

         nalu_hypre_CSRMatrix *Piy_offd = nalu_hypre_ParCSRMatrixOffd(Piy);
         NALU_HYPRE_Int *Piy_offd_I = nalu_hypre_CSRMatrixI(Piy_offd);
         NALU_HYPRE_Int *Piy_offd_J = nalu_hypre_CSRMatrixJ(Piy_offd);
         NALU_HYPRE_Real *Piy_offd_data = nalu_hypre_CSRMatrixData(Piy_offd);

         nalu_hypre_CSRMatrix *Piz_offd = nalu_hypre_ParCSRMatrixOffd(Piz);
         NALU_HYPRE_Int *Piz_offd_I = nalu_hypre_CSRMatrixI(Piz_offd);
         NALU_HYPRE_Int *Piz_offd_J = nalu_hypre_CSRMatrixJ(Piz_offd);
         NALU_HYPRE_Real *Piz_offd_data = nalu_hypre_CSRMatrixData(Piz_offd);

         NALU_HYPRE_BigInt *F2V_cmap = nalu_hypre_ParCSRMatrixColMapOffd(F2V);
         NALU_HYPRE_BigInt *Pix_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Pix);
         NALU_HYPRE_BigInt *Piy_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Piy);
         NALU_HYPRE_BigInt *Piz_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Piz);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if defined(NALU_HYPRE_USING_SYCL)
            if (F2V_offd_ncols)
            {
               NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                                  oneapi::dpl::make_zip_iterator(F2V_offd_I, F2V_offd_I, F2V_offd_I),
                                  F2V_offd_nrows + 1,
                                  oneapi::dpl::make_zip_iterator(Pix_offd_I, Piy_offd_I, Piz_offd_I) );
            }

            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(F2V_offd_J, F2V_offd_J, F2V_offd_J),
                               F2V_offd_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_offd_J, Piy_offd_J, Piz_offd_J) );
#else
            if (F2V_offd_ncols)
            {
               NALU_HYPRE_THRUST_CALL( copy_n,
                                  thrust::make_zip_iterator(thrust::make_tuple(F2V_offd_I, F2V_offd_I, F2V_offd_I)),
                                  F2V_offd_nrows + 1,
                                  thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_I, Piy_offd_I, Piz_offd_I)) );
            }

            NALU_HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(F2V_offd_J, F2V_offd_J, F2V_offd_J)),
                               F2V_offd_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_J, Piy_offd_J, Piz_offd_J)) );
#endif

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(F2V_offd_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              F2V_offd_nrows, 3, F2V_offd_I, NULL, RT100_data, RT010_data, RT001_data,
                              Pix_offd_data, Piy_offd_data, Piz_offd_data );
         }
         else
#endif
         {
            if (F2V_offd_ncols)
               for (i = 0; i < F2V_offd_nrows + 1; i++)
               {
                  Pix_offd_I[i] = F2V_offd_I[i];
                  Piy_offd_I[i] = F2V_offd_I[i];
                  Piz_offd_I[i] = F2V_offd_I[i];
               }

            for (i = 0; i < F2V_offd_nnz; i++)
            {
               Pix_offd_J[i] = F2V_offd_J[i];
               Piy_offd_J[i] = F2V_offd_J[i];
               Piz_offd_J[i] = F2V_offd_J[i];
            }

            for (i = 0; i < F2V_offd_nrows; i++)
               for (j = F2V_offd_I[i]; j < F2V_offd_I[i + 1]; j++)
               {
                  *Pix_offd_data++ = RT100_data[i];
                  *Piy_offd_data++ = RT010_data[i];
                  *Piz_offd_data++ = RT001_data[i];
               }
         }

         for (i = 0; i < F2V_offd_ncols; i++)
         {
            Pix_cmap[i] = F2V_cmap[i];
            Piy_cmap[i] = F2V_cmap[i];
            Piz_cmap[i] = F2V_cmap[i];
         }
      }

      if (NALU_HYPRE_AssumedPartitionCheck())
      {
         nalu_hypre_ParCSRMatrixDestroy(F2V);
      }
      else
      {
         nalu_hypre_ParCSRBooleanMatrixDestroy((nalu_hypre_ParCSRBooleanMatrix*)F2V);
      }
   }

   nalu_hypre_ParVectorDestroy(RT100);
   nalu_hypre_ParVectorDestroy(RT010);
   nalu_hypre_ParVectorDestroy(RT001);

   *Pix_ptr = Pix;
   *Piy_ptr = Piy;
   *Piz_ptr = Piz;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSetup
 *
 * Construct the ADS solver components.
 *
 * The following functions need to be called before nalu_hypre_ADSSetup():
 * - nalu_hypre_ADSSetDiscreteCurl()
 * - nalu_hypre_ADSSetDiscreteGradient()
 * - nalu_hypre_ADSSetCoordinateVectors()
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSetup(void *solver,
                         nalu_hypre_ParCSRMatrix *A,
                         nalu_hypre_ParVector *b,
                         nalu_hypre_ParVector *x)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   nalu_hypre_AMSData *ams_data;

   ads_data -> A = A;

   /* Make sure that the first entry in each row is the diagonal one. */
   /* nalu_hypre_CSRMatrixReorder(nalu_hypre_ParCSRMatrixDiag(ads_data -> A)); */

   /* Compute the l1 norm of the rows of A */
   if (ads_data -> A_relax_type >= 1 && ads_data -> A_relax_type <= 4)
   {
      NALU_HYPRE_Real *l1_norm_data = NULL;

      nalu_hypre_ParCSRComputeL1Norms(ads_data -> A, ads_data -> A_relax_type, NULL, &l1_norm_data);

      ads_data -> A_l1_norms = nalu_hypre_SeqVectorCreate(nalu_hypre_ParCSRMatrixNumRows(ads_data -> A));
      nalu_hypre_VectorData(ads_data -> A_l1_norms) = l1_norm_data;
      nalu_hypre_SeqVectorInitialize_v2(ads_data -> A_l1_norms,
                                   nalu_hypre_ParCSRMatrixMemoryLocation(ads_data -> A));
   }

   /* Chebyshev? */
   if (ads_data -> A_relax_type == 16)
   {
      nalu_hypre_ParCSRMaxEigEstimateCG(ads_data->A, 1, 10,
                                   &ads_data->A_max_eig_est,
                                   &ads_data->A_min_eig_est);
   }

   /* Create the AMS solver on the range of C^T */
   {
      NALU_HYPRE_AMSCreate(&ads_data -> B_C);
      NALU_HYPRE_AMSSetDimension(ads_data -> B_C, 3);

      /* B_C is a preconditioner */
      NALU_HYPRE_AMSSetMaxIter(ads_data -> B_C, 1);
      NALU_HYPRE_AMSSetTol(ads_data -> B_C, 0.0);
      NALU_HYPRE_AMSSetPrintLevel(ads_data -> B_C, 0);

      NALU_HYPRE_AMSSetCycleType(ads_data -> B_C, ads_data -> B_C_cycle_type);
      NALU_HYPRE_AMSSetDiscreteGradient(ads_data -> B_C,
                                   (NALU_HYPRE_ParCSRMatrix) ads_data -> G);

      if (ads_data -> ND_Pi == NULL && ads_data -> ND_Pix == NULL)
      {
         if (ads_data -> B_C_cycle_type < 10)
         {
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "Unsupported AMS cycle type in ADS!");
         }
         NALU_HYPRE_AMSSetCoordinateVectors(ads_data -> B_C,
                                       (NALU_HYPRE_ParVector) ads_data -> x,
                                       (NALU_HYPRE_ParVector) ads_data -> y,
                                       (NALU_HYPRE_ParVector) ads_data -> z);
      }
      else
      {
         if ((ads_data -> B_C_cycle_type < 10 && ads_data -> ND_Pi == NULL) ||
             (ads_data -> B_C_cycle_type > 10 && ads_data -> ND_Pix == NULL))
         {
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "Unsupported AMS cycle type in ADS!");
         }
         NALU_HYPRE_AMSSetInterpolations(ads_data -> B_C,
                                    (NALU_HYPRE_ParCSRMatrix) ads_data -> ND_Pi,
                                    (NALU_HYPRE_ParCSRMatrix) ads_data -> ND_Pix,
                                    (NALU_HYPRE_ParCSRMatrix) ads_data -> ND_Piy,
                                    (NALU_HYPRE_ParCSRMatrix) ads_data -> ND_Piz);
      }

      /* beta=0 in the subspace */
      NALU_HYPRE_AMSSetBetaPoissonMatrix(ads_data -> B_C, NULL);

      /* Reuse A's relaxation parameters for A_C */
      NALU_HYPRE_AMSSetSmoothingOptions(ads_data -> B_C,
                                   ads_data -> A_relax_type,
                                   ads_data -> A_relax_times,
                                   ads_data -> A_relax_weight,
                                   ads_data -> A_omega);

      NALU_HYPRE_AMSSetAlphaAMGOptions(ads_data -> B_C, ads_data -> B_C_coarsen_type,
                                  ads_data -> B_C_agg_levels, ads_data -> B_C_relax_type,
                                  ads_data -> B_C_theta, ads_data -> B_C_interp_type,
                                  ads_data -> B_C_Pmax);
      /* No need to call NALU_HYPRE_AMSSetBetaAMGOptions */

      /* Construct the coarse space matrix by RAP */
      if (!ads_data -> A_C)
      {
         if (!nalu_hypre_ParCSRMatrixCommPkg(ads_data -> C))
         {
            nalu_hypre_MatvecCommPkgCreate(ads_data -> C);
         }

         if (!nalu_hypre_ParCSRMatrixCommPkg(ads_data -> A))
         {
            nalu_hypre_MatvecCommPkgCreate(ads_data -> A);
         }

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            ads_data -> A_C = nalu_hypre_ParCSRMatrixRAPKT(ads_data -> C,
                                                      ads_data -> A,
                                                      ads_data -> C, 1);
         }
         else
#endif
         {
            nalu_hypre_BoomerAMGBuildCoarseOperator(ads_data -> C,
                                               ads_data -> A,
                                               ads_data -> C,
                                               &ads_data -> A_C);
         }

         /* Make sure that A_C has no zero rows (this can happen if beta is zero
            in part of the domain). */
         nalu_hypre_ParCSRMatrixFixZeroRows(ads_data -> A_C);
      }

      NALU_HYPRE_AMSSetup(ads_data -> B_C, (NALU_HYPRE_ParCSRMatrix)ads_data -> A_C, 0, 0);
   }

   ams_data = (nalu_hypre_AMSData *) ads_data -> B_C;

   if (ads_data -> Pi == NULL && ads_data -> Pix == NULL)
   {
      if (ads_data -> cycle_type > 10)
         /* Construct Pi{x,y,z} instead of Pi = [Pix,Piy,Piz] */
         nalu_hypre_ADSComputePixyz(ads_data -> A,
                               ads_data -> C,
                               ads_data -> G,
                               ads_data -> x,
                               ads_data -> y,
                               ads_data -> z,
                               ams_data -> Pix,
                               ams_data -> Piy,
                               ams_data -> Piz,
                               &ads_data -> Pix,
                               &ads_data -> Piy,
                               &ads_data -> Piz);
      else
         /* Construct the Pi interpolation matrix */
         nalu_hypre_ADSComputePi(ads_data -> A,
                            ads_data -> C,
                            ads_data -> G,
                            ads_data -> x,
                            ads_data -> y,
                            ads_data -> z,
                            ams_data -> Pix,
                            ams_data -> Piy,
                            ams_data -> Piz,
                            &ads_data -> Pi);
   }

   if (ads_data -> cycle_type > 10)
      /* Create the AMG solvers on the range of Pi{x,y,z}^T */
   {
      NALU_HYPRE_BoomerAMGCreate(&ads_data -> B_Pix);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ads_data -> B_Pix, ads_data -> B_Pi_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ads_data -> B_Pix, ads_data -> B_Pi_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ads_data -> B_Pix, ads_data -> B_Pi_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ads_data -> B_Pix, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ads_data -> B_Pix, 25);
      NALU_HYPRE_BoomerAMGSetTol(ads_data -> B_Pix, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ads_data -> B_Pix, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ads_data -> B_Pix, ads_data -> B_Pi_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ads_data -> B_Pix, ads_data -> B_Pi_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ads_data -> B_Pix, ads_data -> B_Pi_Pmax);

      NALU_HYPRE_BoomerAMGCreate(&ads_data -> B_Piy);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ads_data -> B_Piy, ads_data -> B_Pi_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ads_data -> B_Piy, ads_data -> B_Pi_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ads_data -> B_Piy, ads_data -> B_Pi_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ads_data -> B_Piy, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ads_data -> B_Piy, 25);
      NALU_HYPRE_BoomerAMGSetTol(ads_data -> B_Piy, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ads_data -> B_Piy, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ads_data -> B_Piy, ads_data -> B_Pi_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ads_data -> B_Piy, ads_data -> B_Pi_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ads_data -> B_Piy, ads_data -> B_Pi_Pmax);

      NALU_HYPRE_BoomerAMGCreate(&ads_data -> B_Piz);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ads_data -> B_Piz, ads_data -> B_Pi_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ads_data -> B_Piz, ads_data -> B_Pi_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ads_data -> B_Piz, ads_data -> B_Pi_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ads_data -> B_Piz, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ads_data -> B_Piz, 25);
      NALU_HYPRE_BoomerAMGSetTol(ads_data -> B_Piz, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ads_data -> B_Piz, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ads_data -> B_Piz, ads_data -> B_Pi_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ads_data -> B_Piz, ads_data -> B_Pi_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ads_data -> B_Piz, ads_data -> B_Pi_Pmax);

      /* Don't use exact solve on the coarsest level (matrices may be singular) */
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ads_data -> B_Pix,
                                       ads_data -> B_Pi_relax_type, 3);
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ads_data -> B_Piy,
                                       ads_data -> B_Pi_relax_type, 3);
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ads_data -> B_Piz,
                                       ads_data -> B_Pi_relax_type, 3);

      /* Construct the coarse space matrices by RAP */
      if (!nalu_hypre_ParCSRMatrixCommPkg(ads_data -> Pix))
      {
         nalu_hypre_MatvecCommPkgCreate(ads_data -> Pix);
      }

#if defined(NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         ads_data -> A_Pix = nalu_hypre_ParCSRMatrixRAPKT(ads_data -> Pix,
                                                     ads_data -> A,
                                                     ads_data -> Pix, 1);
      }
      else
#endif
      {
         nalu_hypre_BoomerAMGBuildCoarseOperator(ads_data -> Pix,
                                            ads_data -> A,
                                            ads_data -> Pix,
                                            &ads_data -> A_Pix);
      }

      NALU_HYPRE_BoomerAMGSetup(ads_data -> B_Pix,
                           (NALU_HYPRE_ParCSRMatrix)ads_data -> A_Pix,
                           NULL, NULL);

      if (!nalu_hypre_ParCSRMatrixCommPkg(ads_data -> Piy))
      {
         nalu_hypre_MatvecCommPkgCreate(ads_data -> Piy);
      }

#if defined(NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         ads_data -> A_Piy = nalu_hypre_ParCSRMatrixRAPKT(ads_data -> Piy,
                                                     ads_data -> A,
                                                     ads_data -> Piy, 1);
      }
      else
#endif
      {
         nalu_hypre_BoomerAMGBuildCoarseOperator(ads_data -> Piy,
                                            ads_data -> A,
                                            ads_data -> Piy,
                                            &ads_data -> A_Piy);
      }

      NALU_HYPRE_BoomerAMGSetup(ads_data -> B_Piy,
                           (NALU_HYPRE_ParCSRMatrix)ads_data -> A_Piy,
                           NULL, NULL);

      if (!nalu_hypre_ParCSRMatrixCommPkg(ads_data -> Piz))
      {
         nalu_hypre_MatvecCommPkgCreate(ads_data -> Piz);
      }

#if defined(NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         ads_data -> A_Piz = nalu_hypre_ParCSRMatrixRAPKT(ads_data -> Piz,
                                                     ads_data -> A,
                                                     ads_data -> Piz, 1);
      }
      else
#endif
      {
         nalu_hypre_BoomerAMGBuildCoarseOperator(ads_data -> Piz,
                                            ads_data -> A,
                                            ads_data -> Piz,
                                            &ads_data -> A_Piz);
      }

      NALU_HYPRE_BoomerAMGSetup(ads_data -> B_Piz,
                           (NALU_HYPRE_ParCSRMatrix)ads_data -> A_Piz,
                           NULL, NULL);
   }
   else
      /* Create the AMG solver on the range of Pi^T */
   {
      NALU_HYPRE_BoomerAMGCreate(&ads_data -> B_Pi);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ads_data -> B_Pi, ads_data -> B_Pi_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ads_data -> B_Pi, ads_data -> B_Pi_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ads_data -> B_Pi, ads_data -> B_Pi_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ads_data -> B_Pi, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ads_data -> B_Pi, 25);
      NALU_HYPRE_BoomerAMGSetTol(ads_data -> B_Pi, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ads_data -> B_Pi, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ads_data -> B_Pi, ads_data -> B_Pi_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ads_data -> B_Pi, ads_data -> B_Pi_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ads_data -> B_Pi, ads_data -> B_Pi_Pmax);

      /* Don't use exact solve on the coarsest level (matrix may be singular) */
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ads_data -> B_Pi,
                                       ads_data -> B_Pi_relax_type,
                                       3);

      /* Construct the coarse space matrix by RAP and notify BoomerAMG that this
         is a 3 x 3 block system. */
      if (!ads_data -> A_Pi)
      {
         if (!nalu_hypre_ParCSRMatrixCommPkg(ads_data -> Pi))
         {
            nalu_hypre_MatvecCommPkgCreate(ads_data -> Pi);
         }

         if (!nalu_hypre_ParCSRMatrixCommPkg(ads_data -> A))
         {
            nalu_hypre_MatvecCommPkgCreate(ads_data -> A);
         }

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            ads_data -> A_Pi = nalu_hypre_ParCSRMatrixRAPKT(ads_data -> Pi,
                                                       ads_data -> A,
                                                       ads_data -> Pi, 1);
         }
         else
#endif
         {
            nalu_hypre_BoomerAMGBuildCoarseOperator(ads_data -> Pi,
                                               ads_data -> A,
                                               ads_data -> Pi,
                                               &ads_data -> A_Pi);
         }

         NALU_HYPRE_BoomerAMGSetNumFunctions(ads_data -> B_Pi, 3);
         /* NALU_HYPRE_BoomerAMGSetNodal(ads_data -> B_Pi, 1); */
      }

      NALU_HYPRE_BoomerAMGSetup(ads_data -> B_Pi,
                           (NALU_HYPRE_ParCSRMatrix)ads_data -> A_Pi,
                           NULL, NULL);
   }

   /* Allocate temporary vectors */
   ads_data -> r0 = nalu_hypre_ParVectorInRangeOf(ads_data -> A);
   ads_data -> g0 = nalu_hypre_ParVectorInRangeOf(ads_data -> A);
   if (ads_data -> A_C)
   {
      ads_data -> r1 = nalu_hypre_ParVectorInRangeOf(ads_data -> A_C);
      ads_data -> g1 = nalu_hypre_ParVectorInRangeOf(ads_data -> A_C);
   }
   if (ads_data -> cycle_type > 10)
   {
      ads_data -> r2 = nalu_hypre_ParVectorInDomainOf(ads_data -> Pix);
      ads_data -> g2 = nalu_hypre_ParVectorInDomainOf(ads_data -> Pix);
   }
   else
   {
      ads_data -> r2 = nalu_hypre_ParVectorInDomainOf(ads_data -> Pi);
      ads_data -> g2 = nalu_hypre_ParVectorInDomainOf(ads_data -> Pi);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSSolve
 *
 * Solve the system A x = b.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSSolve(void *solver,
                         nalu_hypre_ParCSRMatrix *A,
                         nalu_hypre_ParVector *b,
                         nalu_hypre_ParVector *x)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;

   NALU_HYPRE_Int i, my_id = -1;
   NALU_HYPRE_Real r0_norm, r_norm, b_norm, relative_resid = 0, old_resid;

   char cycle[30];
   nalu_hypre_ParCSRMatrix *Ai[5], *Pi[5];
   NALU_HYPRE_Solver Bi[5];
   NALU_HYPRE_PtrToSolverFcn HBi[5];
   nalu_hypre_ParVector *ri[5], *gi[5];
   NALU_HYPRE_Int needZ = 0;

   nalu_hypre_ParVector *z = ads_data -> zz;

   Ai[0] = ads_data -> A_C;    Pi[0] = ads_data -> C;
   Ai[1] = ads_data -> A_Pi;   Pi[1] = ads_data -> Pi;
   Ai[2] = ads_data -> A_Pix;  Pi[2] = ads_data -> Pix;
   Ai[3] = ads_data -> A_Piy;  Pi[3] = ads_data -> Piy;
   Ai[4] = ads_data -> A_Piz;  Pi[4] = ads_data -> Piz;

   Bi[0] = ads_data -> B_C;    HBi[0] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_AMSSolve;
   Bi[1] = ads_data -> B_Pi;   HBi[1] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_BoomerAMGBlockSolve;
   Bi[2] = ads_data -> B_Pix;  HBi[2] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_BoomerAMGSolve;
   Bi[3] = ads_data -> B_Piy;  HBi[3] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_BoomerAMGSolve;
   Bi[4] = ads_data -> B_Piz;  HBi[4] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_BoomerAMGSolve;

   ri[0] = ads_data -> r1;     gi[0] = ads_data -> g1;
   ri[1] = ads_data -> r2;     gi[1] = ads_data -> g2;
   ri[2] = ads_data -> r2;     gi[2] = ads_data -> g2;
   ri[3] = ads_data -> r2;     gi[3] = ads_data -> g2;
   ri[4] = ads_data -> r2;     gi[4] = ads_data -> g2;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   /* may need to create an additional temporary vector for relaxation */
#if defined(NALU_HYPRE_USING_GPU)
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      needZ = ads_data -> A_relax_type == 2 || ads_data -> A_relax_type == 4 ||
              ads_data -> A_relax_type == 16;
   }
   else
#endif
   {
      needZ = nalu_hypre_NumThreads() > 1 || ads_data -> A_relax_type == 16;
   }

   if (needZ && !z)
   {
      z = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixRowStarts(A));
      nalu_hypre_ParVectorInitialize(z);
      ads_data -> zz = z;
   }

   if (ads_data -> print_level > 0)
   {
      nalu_hypre_MPI_Comm_rank(nalu_hypre_ParCSRMatrixComm(A), &my_id);
   }

   switch (ads_data -> cycle_type)
   {
      case 1:
      default:
         nalu_hypre_sprintf(cycle, "%s", "01210");
         break;
      case 2:
         nalu_hypre_sprintf(cycle, "%s", "(0+1+2)");
         break;
      case 3:
         nalu_hypre_sprintf(cycle, "%s", "02120");
         break;
      case 4:
         nalu_hypre_sprintf(cycle, "%s", "(010+2)");
         break;
      case 5:
         nalu_hypre_sprintf(cycle, "%s", "0102010");
         break;
      case 6:
         nalu_hypre_sprintf(cycle, "%s", "(020+1)");
         break;
      case 7:
         nalu_hypre_sprintf(cycle, "%s", "0201020");
         break;
      case 8:
         nalu_hypre_sprintf(cycle, "%s", "0(+1+2)0");
         break;
      case 9:
         nalu_hypre_sprintf(cycle, "%s", "01210");
         break;
      case 11:
         nalu_hypre_sprintf(cycle, "%s", "013454310");
         break;
      case 12:
         nalu_hypre_sprintf(cycle, "%s", "(0+1+3+4+5)");
         break;
      case 13:
         nalu_hypre_sprintf(cycle, "%s", "034515430");
         break;
      case 14:
         nalu_hypre_sprintf(cycle, "%s", "01(+3+4+5)10");
         break;
   }

   for (i = 0; i < ads_data -> maxit; i++)
   {
      /* Compute initial residual norms */
      if (ads_data -> maxit > 1 && i == 0)
      {
         nalu_hypre_ParVectorCopy(b, ads_data -> r0);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, ads_data -> A, x, 1.0, ads_data -> r0);
         r_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(ads_data -> r0, ads_data -> r0));
         r0_norm = r_norm;
         b_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(b, b));
         if (b_norm)
         {
            relative_resid = r_norm / b_norm;
         }
         else
         {
            relative_resid = r_norm;
         }
         if (my_id == 0 && ads_data -> print_level > 0)
         {
            nalu_hypre_printf("                                            relative\n");
            nalu_hypre_printf("               residual        factor       residual\n");
            nalu_hypre_printf("               --------        ------       --------\n");
            nalu_hypre_printf("    Initial    %e                 %e\n",
                         r_norm, relative_resid);
         }
      }

      /* Apply the preconditioner */
      nalu_hypre_ParCSRSubspacePrec(ads_data -> A,
                               ads_data -> A_relax_type,
                               ads_data -> A_relax_times,
                               ads_data -> A_l1_norms ? nalu_hypre_VectorData(ads_data -> A_l1_norms) : NULL,
                               ads_data -> A_relax_weight,
                               ads_data -> A_omega,
                               ads_data -> A_max_eig_est,
                               ads_data -> A_min_eig_est,
                               ads_data -> A_cheby_order,
                               ads_data -> A_cheby_fraction,
                               Ai, Bi, HBi, Pi, ri, gi,
                               b, x,
                               ads_data -> r0,
                               ads_data -> g0,
                               cycle,
                               z);

      /* Compute new residual norms */
      if (ads_data -> maxit > 1)
      {
         old_resid = r_norm;
         nalu_hypre_ParVectorCopy(b, ads_data -> r0);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, ads_data -> A, x, 1.0, ads_data -> r0);
         r_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(ads_data -> r0, ads_data -> r0));
         if (b_norm)
         {
            relative_resid = r_norm / b_norm;
         }
         else
         {
            relative_resid = r_norm;
         }
         if (my_id == 0 && ads_data -> print_level > 0)
            nalu_hypre_printf("    Cycle %2d   %e    %f     %e \n",
                         i + 1, r_norm, r_norm / old_resid, relative_resid);
      }

      if (relative_resid < ads_data -> tol)
      {
         i++;
         break;
      }
   }

   if (my_id == 0 && ads_data -> print_level > 0 && ads_data -> maxit > 1)
      nalu_hypre_printf("\n\n Average Convergence Factor = %f\n\n",
                   nalu_hypre_pow((r_norm / r0_norm), (1.0 / (NALU_HYPRE_Real) i)));

   ads_data -> num_iterations = i;
   ads_data -> rel_resid_norm = relative_resid;

   if (ads_data -> num_iterations == ads_data -> maxit && ads_data -> tol > 0.0)
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_CONV);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSGetNumIterations
 *
 * Get the number of ADS iterations.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSGetNumIterations(void *solver,
                                    NALU_HYPRE_Int *num_iterations)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   *num_iterations = ads_data -> num_iterations;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ADSGetFinalRelativeResidualNorm
 *
 * Get the final relative residual norm in ADS.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ADSGetFinalRelativeResidualNorm(void *solver,
                                                NALU_HYPRE_Real *rel_resid_norm)
{
   nalu_hypre_ADSData *ads_data = (nalu_hypre_ADSData *) solver;
   *rel_resid_norm = ads_data -> rel_resid_norm;
   return nalu_hypre_error_flag;
}
