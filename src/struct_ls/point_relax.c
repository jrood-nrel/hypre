/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"

/* this currently cannot be greater than 7 */
#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;

   NALU_HYPRE_Real              tol;       /* tolerance, set =0 for no convergence testing */
   NALU_HYPRE_Real              rresnorm;  /* relative residual norm, computed only if tol>0.0 */
   NALU_HYPRE_Int               max_iter;
   NALU_HYPRE_Int               rel_change;         /* not yet used */
   NALU_HYPRE_Int               zero_guess;
   NALU_HYPRE_Real              weight;

   NALU_HYPRE_Int               num_pointsets;
   NALU_HYPRE_Int              *pointset_sizes;
   NALU_HYPRE_Int              *pointset_ranks;
   nalu_hypre_Index            *pointset_strides;
   nalu_hypre_Index           **pointset_indices;

   nalu_hypre_StructMatrix     *A;
   nalu_hypre_StructVector     *b;
   nalu_hypre_StructVector     *x;
   nalu_hypre_StructVector     *t;

   NALU_HYPRE_Int               diag_rank;

   nalu_hypre_ComputePkg      **compute_pkgs;

   /* log info (always logged) */
   NALU_HYPRE_Int               num_iterations;
   NALU_HYPRE_Int               time_index;
   NALU_HYPRE_BigInt            flops;

} nalu_hypre_PointRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_PointRelaxCreate( MPI_Comm  comm )
{
   nalu_hypre_PointRelaxData *relax_data;

   nalu_hypre_Index           stride;
   nalu_hypre_Index           indices[1];

   relax_data = nalu_hypre_CTAlloc(nalu_hypre_PointRelaxData,  1, NALU_HYPRE_MEMORY_HOST);

   (relax_data -> comm)       = comm;
   (relax_data -> time_index) = nalu_hypre_InitializeTiming("PointRelax");

   /* set defaults */
   (relax_data -> tol)              = 0.0;  /* tol=0 means no convergence testing */
   (relax_data -> rresnorm)         = 0.0;
   (relax_data -> max_iter)         = 1000;
   (relax_data -> rel_change)       = 0;
   (relax_data -> zero_guess)       = 0;
   (relax_data -> weight)           = 1.0;
   (relax_data -> num_pointsets)    = 0;
   (relax_data -> pointset_sizes)   = NULL;
   (relax_data -> pointset_ranks)   = NULL;
   (relax_data -> pointset_strides) = NULL;
   (relax_data -> pointset_indices) = NULL;
   (relax_data -> A)                = NULL;
   (relax_data -> b)                = NULL;
   (relax_data -> x)                = NULL;
   (relax_data -> t)                = NULL;
   (relax_data -> compute_pkgs)     = NULL;

   nalu_hypre_SetIndex3(stride, 1, 1, 1);
   nalu_hypre_SetIndex3(indices[0], 0, 0, 0);
   nalu_hypre_PointRelaxSetNumPointsets((void *) relax_data, 1);
   nalu_hypre_PointRelaxSetPointset((void *) relax_data, 0, 1, stride, indices);

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxDestroy( void *relax_vdata )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;
   NALU_HYPRE_Int             i;

   if (relax_data)
   {
      for (i = 0; i < (relax_data -> num_pointsets); i++)
      {
         nalu_hypre_TFree(relax_data -> pointset_indices[i], NALU_HYPRE_MEMORY_HOST);
      }
      if (relax_data -> compute_pkgs)
      {
         for (i = 0; i < (relax_data -> num_pointsets); i++)
         {
            nalu_hypre_ComputePkgDestroy(relax_data -> compute_pkgs[i]);
         }
      }
      nalu_hypre_TFree(relax_data -> pointset_sizes, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> pointset_ranks, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> pointset_strides, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> pointset_indices, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_StructMatrixDestroy(relax_data -> A);
      nalu_hypre_StructVectorDestroy(relax_data -> b);
      nalu_hypre_StructVectorDestroy(relax_data -> x);
      nalu_hypre_StructVectorDestroy(relax_data -> t);
      nalu_hypre_TFree(relax_data -> compute_pkgs, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_FinalizeTiming(relax_data -> time_index);
      nalu_hypre_TFree(relax_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxSetup( void               *relax_vdata,
                       nalu_hypre_StructMatrix *A,
                       nalu_hypre_StructVector *b,
                       nalu_hypre_StructVector *x           )
{
   nalu_hypre_PointRelaxData  *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   NALU_HYPRE_Int              num_pointsets    = (relax_data -> num_pointsets);
   NALU_HYPRE_Int             *pointset_sizes   = (relax_data -> pointset_sizes);
   nalu_hypre_Index           *pointset_strides = (relax_data -> pointset_strides);
   nalu_hypre_Index          **pointset_indices = (relax_data -> pointset_indices);
   NALU_HYPRE_Int              ndim = nalu_hypre_StructMatrixNDim(A);
   nalu_hypre_StructVector    *t;
   NALU_HYPRE_Int              diag_rank;
   nalu_hypre_ComputeInfo     *compute_info;
   nalu_hypre_ComputePkg     **compute_pkgs;

   nalu_hypre_Index            diag_index;
   nalu_hypre_IndexRef         stride;
   nalu_hypre_IndexRef         index;

   nalu_hypre_StructGrid      *grid;
   nalu_hypre_StructStencil   *stencil;

   nalu_hypre_BoxArrayArray   *orig_indt_boxes;
   nalu_hypre_BoxArrayArray   *orig_dept_boxes;
   nalu_hypre_BoxArrayArray   *box_aa;
   nalu_hypre_BoxArray        *box_a;
   nalu_hypre_Box             *box;
   NALU_HYPRE_Int              box_aa_size;
   NALU_HYPRE_Int              box_a_size;
   nalu_hypre_BoxArrayArray   *new_box_aa;
   nalu_hypre_BoxArray        *new_box_a;
   nalu_hypre_Box             *new_box;

   NALU_HYPRE_Real             scale;
   NALU_HYPRE_Int              frac;

   NALU_HYPRE_Int              i, j, k, p, m, compute_i;

   /*----------------------------------------------------------
    * Set up the temp vector
    *----------------------------------------------------------*/

   if ((relax_data -> t) == NULL)
   {
      t = nalu_hypre_StructVectorCreate(nalu_hypre_StructVectorComm(b),
                                   nalu_hypre_StructVectorGrid(b));
      nalu_hypre_StructVectorSetNumGhost(t, nalu_hypre_StructVectorNumGhost(b));
      nalu_hypre_StructVectorInitialize(t);
      nalu_hypre_StructVectorAssemble(t);
      (relax_data -> t) = t;
   }

   /*----------------------------------------------------------
    * Find the matrix diagonal
    *----------------------------------------------------------*/

   grid    = nalu_hypre_StructMatrixGrid(A);
   stencil = nalu_hypre_StructMatrixStencil(A);

   nalu_hypre_SetIndex3(diag_index, 0, 0, 0);
   diag_rank = nalu_hypre_StructStencilElementRank(stencil, diag_index);

   /*----------------------------------------------------------
    * Set up the compute packages
    *----------------------------------------------------------*/

   compute_pkgs = nalu_hypre_CTAlloc(nalu_hypre_ComputePkg *,  num_pointsets, NALU_HYPRE_MEMORY_HOST);

   for (p = 0; p < num_pointsets; p++)
   {
      nalu_hypre_CreateComputeInfo(grid, stencil, &compute_info);
      orig_indt_boxes = nalu_hypre_ComputeInfoIndtBoxes(compute_info);
      orig_dept_boxes = nalu_hypre_ComputeInfoDeptBoxes(compute_info);

      stride = pointset_strides[p];

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
               box_aa = orig_indt_boxes;
               break;

            case 1:
               box_aa = orig_dept_boxes;
               break;
         }
         box_aa_size = nalu_hypre_BoxArrayArraySize(box_aa);
         new_box_aa = nalu_hypre_BoxArrayArrayCreate(box_aa_size, ndim);

         for (i = 0; i < box_aa_size; i++)
         {
            box_a = nalu_hypre_BoxArrayArrayBoxArray(box_aa, i);
            box_a_size = nalu_hypre_BoxArraySize(box_a);
            new_box_a = nalu_hypre_BoxArrayArrayBoxArray(new_box_aa, i);
            nalu_hypre_BoxArraySetSize(new_box_a, box_a_size * pointset_sizes[p]);

            k = 0;
            for (m = 0; m < pointset_sizes[p]; m++)
            {
               index  = pointset_indices[p][m];

               for (j = 0; j < box_a_size; j++)
               {
                  box = nalu_hypre_BoxArrayBox(box_a, j);
                  new_box = nalu_hypre_BoxArrayBox(new_box_a, k);

                  nalu_hypre_CopyBox(box, new_box);
                  nalu_hypre_ProjectBox(new_box, index, stride);

                  k++;
               }
            }
         }

         switch (compute_i)
         {
            case 0:
               nalu_hypre_ComputeInfoIndtBoxes(compute_info) = new_box_aa;
               break;

            case 1:
               nalu_hypre_ComputeInfoDeptBoxes(compute_info) = new_box_aa;
               break;
         }
      }

      nalu_hypre_CopyIndex(stride, nalu_hypre_ComputeInfoStride(compute_info));

      nalu_hypre_ComputePkgCreate(compute_info, nalu_hypre_StructVectorDataSpace(x), 1,
                             grid, &compute_pkgs[p]);

      nalu_hypre_BoxArrayArrayDestroy(orig_indt_boxes);
      nalu_hypre_BoxArrayArrayDestroy(orig_dept_boxes);
   }

   /*----------------------------------------------------------
    * Set up the relax data structure
    *----------------------------------------------------------*/

   (relax_data -> A) = nalu_hypre_StructMatrixRef(A);
   (relax_data -> x) = nalu_hypre_StructVectorRef(x);
   (relax_data -> b) = nalu_hypre_StructVectorRef(b);
   (relax_data -> diag_rank)    = diag_rank;
   (relax_data -> compute_pkgs) = compute_pkgs;

   /*-----------------------------------------------------
    * Compute flops
    *-----------------------------------------------------*/

   scale = 0.0;
   for (p = 0; p < num_pointsets; p++)
   {
      stride = pointset_strides[p];
      frac   = nalu_hypre_IndexX(stride);
      frac  *= nalu_hypre_IndexY(stride);
      frac  *= nalu_hypre_IndexZ(stride);
      scale += (pointset_sizes[p] / frac);
   }
   (relax_data -> flops) = (NALU_HYPRE_BigInt)scale * (nalu_hypre_StructMatrixGlobalSize(A) +
                                                  nalu_hypre_StructVectorGlobalSize(x));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelax( void               *relax_vdata,
                  nalu_hypre_StructMatrix *A,
                  nalu_hypre_StructVector *b,
                  nalu_hypre_StructVector *x           )
{
   nalu_hypre_PointRelaxData  *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   NALU_HYPRE_Int              max_iter         = (relax_data -> max_iter);
   NALU_HYPRE_Int              zero_guess       = (relax_data -> zero_guess);
   NALU_HYPRE_Real             weight           = (relax_data -> weight);
   NALU_HYPRE_Int              num_pointsets    = (relax_data -> num_pointsets);
   NALU_HYPRE_Int             *pointset_ranks   = (relax_data -> pointset_ranks);
   nalu_hypre_Index           *pointset_strides = (relax_data -> pointset_strides);
   nalu_hypre_StructVector    *t                = (relax_data -> t);
   NALU_HYPRE_Int              diag_rank        = (relax_data -> diag_rank);
   nalu_hypre_ComputePkg     **compute_pkgs     = (relax_data -> compute_pkgs);
   NALU_HYPRE_Real             tol              = (relax_data -> tol);
   NALU_HYPRE_Real             tol2             = tol * tol;

   nalu_hypre_ComputePkg      *compute_pkg;
   nalu_hypre_CommHandle      *comm_handle;

   nalu_hypre_BoxArrayArray   *compute_box_aa;
   nalu_hypre_BoxArray        *compute_box_a;
   nalu_hypre_Box             *compute_box;

   nalu_hypre_Box             *A_data_box;
   nalu_hypre_Box             *b_data_box;
   nalu_hypre_Box             *x_data_box;
   nalu_hypre_Box             *t_data_box;

   NALU_HYPRE_Real            *Ap;
   NALU_HYPRE_Real            AAp0;
   NALU_HYPRE_Real            *bp;
   NALU_HYPRE_Real            *xp;
   NALU_HYPRE_Real            *tp;
   void                  *matvec_data = NULL;

   NALU_HYPRE_Int              Ai;

   nalu_hypre_IndexRef         stride;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            loop_size;

   NALU_HYPRE_Int              constant_coefficient;

   NALU_HYPRE_Int              iter, p, compute_i, i, j;
   NALU_HYPRE_Int              pointset;

   NALU_HYPRE_Real             bsumsq, rsumsq;

   /*----------------------------------------------------------
    * Initialize some things and deal with special cases
    *----------------------------------------------------------*/

   nalu_hypre_BeginTiming(relax_data -> time_index);

   nalu_hypre_StructMatrixDestroy(relax_data -> A);
   nalu_hypre_StructVectorDestroy(relax_data -> b);
   nalu_hypre_StructVectorDestroy(relax_data -> x);
   (relax_data -> A) = nalu_hypre_StructMatrixRef(A);
   (relax_data -> x) = nalu_hypre_StructVectorRef(x);
   (relax_data -> b) = nalu_hypre_StructVectorRef(b);

   (relax_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         nalu_hypre_StructVectorSetConstantValues(x, 0.0);
      }

      nalu_hypre_EndTiming(relax_data -> time_index);
      return nalu_hypre_error_flag;
   }

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(A);
   if (constant_coefficient) { nalu_hypre_StructVectorClearBoundGhostValues(x, 0); }

   rsumsq = 0.0;
   if ( tol > 0.0 )
   {
      bsumsq = nalu_hypre_StructInnerProd( b, b );
   }

   /*----------------------------------------------------------
    * Do zero_guess iteration
    *----------------------------------------------------------*/

   p    = 0;
   iter = 0;
   if ( tol > 0.0)
   {
      matvec_data = nalu_hypre_StructMatvecCreate();
      nalu_hypre_StructMatvecSetup( matvec_data, A, x );
   }

   if (zero_guess)
   {
      if ( p == 0 ) { rsumsq = 0.0; }
      if (num_pointsets > 1)
      {
         nalu_hypre_StructVectorSetConstantValues(x, 0.0);
      }
      pointset = pointset_ranks[p];
      compute_pkg = compute_pkgs[pointset];
      stride = pointset_strides[pointset];

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         nalu_hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_data_box =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
            b_data_box =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(b), i);
            x_data_box =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);

            Ap = nalu_hypre_StructMatrixBoxData(A, i, diag_rank);
            bp = nalu_hypre_StructVectorBoxData(b, i);
            xp = nalu_hypre_StructVectorBoxData(x, i);

            nalu_hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

               start  = nalu_hypre_BoxIMin(compute_box);
               nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

               /* all matrix coefficients are constant */
               if ( constant_coefficient == 1 )
               {
                  Ai = nalu_hypre_CCBoxIndexRank( A_data_box, start );
                  AAp0 = 1 / Ap[Ai];
#define DEVICE_VAR is_device_ptr(xp,bp)
                  nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                                      b_data_box, start, stride, bi,
                                      x_data_box, start, stride, xi);
                  {
                     xp[xi] = bp[bi] * AAp0;
                  }
                  nalu_hypre_BoxLoop2End(bi, xi);
#undef DEVICE_VAR
               }
               /* constant_coefficent 0 (variable) or 2 (variable diagonal
                  only) are the same for the diagonal */
               else
               {
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                  nalu_hypre_BoxLoop3Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                                      A_data_box, start, stride, Ai,
                                      b_data_box, start, stride, bi,
                                      x_data_box, start, stride, xi);
                  {
                     xp[xi] = bp[bi] / Ap[Ai];
                  }
                  nalu_hypre_BoxLoop3End(Ai, bi, xi);
#undef DEVICE_VAR
               }
            }
         }
      }

      if (weight != 1.0)
      {
         nalu_hypre_StructScale(weight, x);
      }

      p    = (p + 1) % num_pointsets;
      iter = iter + (p == 0);

      if ( tol > 0.0 && p == 0 )
         /* ... p==0 here means we've finished going through all the pointsets,
            i.e. this iteration is complete.
            tol>0.0 means to do a convergence test, using tol.
            The test is simply ||r||/||b||<tol, where r=residual, b=r.h.s., unweighted L2 norm */
      {
         nalu_hypre_StructCopy( b, t ); /* t = b */
         nalu_hypre_StructMatvecCompute( matvec_data,
                                    -1.0, A, x, 1.0, t );  /* t = - A x + t = - A x + b */
         rsumsq = nalu_hypre_StructInnerProd( t, t ); /* <t,t> */
         if ( rsumsq / bsumsq < tol2 ) { max_iter = iter; } /* converged; reset max_iter to prevent more iterations */
      }
   }

   /*----------------------------------------------------------
    * Do regular iterations
    *----------------------------------------------------------*/

   while (iter < max_iter)
   {
      if ( p == 0 ) { rsumsq = 0.0; }
      pointset = pointset_ranks[p];
      compute_pkg = compute_pkgs[pointset];
      stride = pointset_strides[pointset];

      /*nalu_hypre_StructCopy(x, t); ... not needed as long as the copy at the end of the loop
        is restricted to the current pointset (nalu_hypre_relax_copy, nalu_hypre_relax_wtx */

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               xp = nalu_hypre_StructVectorData(x);
               nalu_hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
               compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               nalu_hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         nalu_hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_data_box =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
            b_data_box =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(b), i);
            x_data_box =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
            t_data_box =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(t), i);

            bp = nalu_hypre_StructVectorBoxData(b, i);
            xp = nalu_hypre_StructVectorBoxData(x, i);
            tp = nalu_hypre_StructVectorBoxData(t, i);

            nalu_hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

               if ( constant_coefficient == 1 || constant_coefficient == 2 )
               {
                  nalu_hypre_PointRelax_core12(
                     relax_vdata, A, constant_coefficient,
                     compute_box, bp, xp, tp, i,
                     A_data_box, b_data_box, x_data_box, t_data_box,
                     stride
                  );
               }

               else
               {
                  nalu_hypre_PointRelax_core0(
                     relax_vdata, A, constant_coefficient,
                     compute_box, bp, xp, tp, i,
                     A_data_box, b_data_box, x_data_box, t_data_box,
                     stride
                  );
               }

               Ap = nalu_hypre_StructMatrixBoxData(A, i, diag_rank);

               if ( constant_coefficient == 0 || constant_coefficient == 2 )
                  /* divide by the variable diagonal */
               {
                  start  = nalu_hypre_BoxIMin(compute_box);
                  nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);
#define DEVICE_VAR is_device_ptr(tp,Ap)
                  nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                                      A_data_box, start, stride, Ai,
                                      t_data_box, start, stride, ti);
                  {
                     tp[ti] /= Ap[Ai];
                  }
                  nalu_hypre_BoxLoop2End(Ai, ti);
#undef DEVICE_VAR
               }
            }
         }
      }


      if (weight != 1.0)
      {
         /*        nalu_hypre_StructScale((1.0 - weight), x);
                   nalu_hypre_StructAxpy(weight, t, x);*/
         nalu_hypre_relax_wtx( relax_data, pointset, t, x ); /* x=w*t+(1-w)*x on pointset */
      }
      else
      {
         nalu_hypre_relax_copy( relax_data, pointset, t, x ); /* x=t on pointset */
         /* nalu_hypre_StructCopy(t, x);*/
      }

      p    = (p + 1) % num_pointsets;
      iter = iter + (p == 0);

      if ( tol > 0.0 && p == 0 )
         /* ... p==0 here means we've finished going through all the pointsets,
            i.e. this iteration is complete.
            tol>0.0 means to do a convergence test, using tol.
            The test is simply ||r||/||b||<tol, where r=residual, b=r.h.s., unweighted L2 norm */
      {
         nalu_hypre_StructCopy( b, t ); /* t = b */
         nalu_hypre_StructMatvecCompute( matvec_data,
                                    -1.0, A, x, 1.0, t );  /* t = - A x + t = - A x + b */
         rsumsq = nalu_hypre_StructInnerProd( t, t ); /* <t,t> */
         if ( rsumsq / bsumsq < tol2 ) { break; }
      }
   }

   if ( tol > 0.0 )
   {
      nalu_hypre_StructMatvecDestroy( matvec_data );
   }

   if ( tol > 0.0 ) { (relax_data -> rresnorm) = sqrt( rsumsq / bsumsq ); }
   (relax_data -> num_iterations) = iter;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   nalu_hypre_IncFLOPCount(relax_data -> flops);
   nalu_hypre_EndTiming(relax_data -> time_index);

   return nalu_hypre_error_flag;
}

/* for constant_coefficient==0, all coefficients may vary ...*/
NALU_HYPRE_Int
nalu_hypre_PointRelax_core0( void               *relax_vdata,
                        nalu_hypre_StructMatrix *A,
                        NALU_HYPRE_Int           constant_coefficient,
                        nalu_hypre_Box          *compute_box,
                        NALU_HYPRE_Real         *bp,
                        NALU_HYPRE_Real         *xp,
                        NALU_HYPRE_Real         *tp,
                        NALU_HYPRE_Int           boxarray_id,
                        nalu_hypre_Box          *A_data_box,
                        nalu_hypre_Box          *b_data_box,
                        nalu_hypre_Box          *x_data_box,
                        nalu_hypre_Box          *t_data_box,
                        nalu_hypre_IndexRef      stride
                      )
{
   nalu_hypre_PointRelaxData  *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   NALU_HYPRE_Real            *Ap0;
   NALU_HYPRE_Real            *Ap1;
   NALU_HYPRE_Real            *Ap2;
   NALU_HYPRE_Real            *Ap3;
   NALU_HYPRE_Real            *Ap4;
   NALU_HYPRE_Real            *Ap5;
   NALU_HYPRE_Real            *Ap6;

   NALU_HYPRE_Int              xoff0;
   NALU_HYPRE_Int              xoff1;
   NALU_HYPRE_Int              xoff2;
   NALU_HYPRE_Int              xoff3;
   NALU_HYPRE_Int              xoff4;
   NALU_HYPRE_Int              xoff5;
   NALU_HYPRE_Int              xoff6;

   nalu_hypre_StructStencil   *stencil;
   nalu_hypre_Index           *stencil_shape;
   NALU_HYPRE_Int              stencil_size;

   NALU_HYPRE_Int              diag_rank        = (relax_data -> diag_rank);
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            loop_size;
   NALU_HYPRE_Int              si, sk, ssi[MAX_DEPTH], depth, k;

   stencil       = nalu_hypre_StructMatrixStencil(A);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);

   start  = nalu_hypre_BoxIMin(compute_box);
   nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(tp,bp)
   nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                       b_data_box, start, stride, bi,
                       t_data_box, start, stride, ti);
   {
      tp[ti] = bp[bi];
   }
   nalu_hypre_BoxLoop2End(bi, ti);
#undef DEVICE_VAR

   /* unroll up to depth MAX_DEPTH */
   for (si = 0; si < stencil_size; si += MAX_DEPTH)
   {
      depth = nalu_hypre_min(MAX_DEPTH, (stencil_size - si));

      for (k = 0, sk = si; k < depth; sk++)
      {
         if (sk == diag_rank)
         {
            depth--;
         }
         else
         {
            ssi[k] = sk;
            k++;
         }
      }

      switch (depth)
      {
         case 7:
            Ap6 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[6]);
            xoff6 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[6]]);

         case 6:
            Ap5 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[5]);
            xoff5 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[5]]);

         case 5:
            Ap4 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[4]);
            xoff4 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[4]]);

         case 4:
            Ap3 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[3]);
            xoff3 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[3]]);

         case 3:
            Ap2 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[2]);
            xoff2 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[2]]);

         case 2:
            Ap1 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[1]);
            xoff1 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[1]]);

         case 1:
            Ap0 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[0]);
            xoff0 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[0]]);

         case 0:

            break;
      }

      switch (depth)
      {
         case 7:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,xp)
            nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4] +
                  Ap5[Ai] * xp[xi + xoff5] +
                  Ap6[Ai] * xp[xi + xoff6];
            }
            nalu_hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 6:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,xp)
            nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4] +
                  Ap5[Ai] * xp[xi + xoff5];
            }
            nalu_hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 5:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,Ap3,Ap4,xp)
            nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4];
            }
            nalu_hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 4:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,Ap3,xp)
            nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3];
            }
            nalu_hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 3:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,Ap2,xp)
            nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2];
            }
            nalu_hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 2:
#define DEVICE_VAR is_device_ptr(tp,Ap0,Ap1,xp)
            nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1];
            }
            nalu_hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 1:
#define DEVICE_VAR is_device_ptr(tp,Ap0,xp)
            nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  Ap0[Ai] * xp[xi + xoff0];
            }
            nalu_hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
            break;

         case 0:
            break;
      }
   }

   return nalu_hypre_error_flag;
}


/* for constant_coefficient==1 or 2, all offdiagonal coefficients constant over space ...*/
NALU_HYPRE_Int
nalu_hypre_PointRelax_core12( void               *relax_vdata,
                         nalu_hypre_StructMatrix *A,
                         NALU_HYPRE_Int           constant_coefficient,
                         nalu_hypre_Box          *compute_box,
                         NALU_HYPRE_Real         *bp,
                         NALU_HYPRE_Real         *xp,
                         NALU_HYPRE_Real         *tp,
                         NALU_HYPRE_Int           boxarray_id,
                         nalu_hypre_Box          *A_data_box,
                         nalu_hypre_Box          *b_data_box,
                         nalu_hypre_Box          *x_data_box,
                         nalu_hypre_Box          *t_data_box,
                         nalu_hypre_IndexRef      stride
                       )
{
   nalu_hypre_PointRelaxData  *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   NALU_HYPRE_Real            *Apd;
   NALU_HYPRE_Real            *Ap0;
   NALU_HYPRE_Real            *Ap1;
   NALU_HYPRE_Real            *Ap2;
   NALU_HYPRE_Real            *Ap3;
   NALU_HYPRE_Real            *Ap4;
   NALU_HYPRE_Real            *Ap5;
   NALU_HYPRE_Real            *Ap6;
   NALU_HYPRE_Real            AAp0;
   NALU_HYPRE_Real            AAp1;
   NALU_HYPRE_Real            AAp2;
   NALU_HYPRE_Real            AAp3;
   NALU_HYPRE_Real            AAp4;
   NALU_HYPRE_Real            AAp5;
   NALU_HYPRE_Real            AAp6;
   NALU_HYPRE_Real            AApd;

   NALU_HYPRE_Int              xoff0;
   NALU_HYPRE_Int              xoff1;
   NALU_HYPRE_Int              xoff2;
   NALU_HYPRE_Int              xoff3;
   NALU_HYPRE_Int              xoff4;
   NALU_HYPRE_Int              xoff5;
   NALU_HYPRE_Int              xoff6;

   nalu_hypre_StructStencil   *stencil;
   nalu_hypre_Index           *stencil_shape;
   NALU_HYPRE_Int              stencil_size;

   NALU_HYPRE_Int              diag_rank        = (relax_data -> diag_rank);
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            loop_size;
   NALU_HYPRE_Int              si, sk, ssi[MAX_DEPTH], depth, k;
   NALU_HYPRE_Int              Ai;

   stencil       = nalu_hypre_StructMatrixStencil(A);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);

   start  = nalu_hypre_BoxIMin(compute_box);
   nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

   /* The standard (variable coefficient) algorithm initializes
      tp=bp.  Do it here, but for constant diagonal, also
      divide by the diagonal (and set up AApd for other
      division-equivalents.
      For a variable diagonal, this diagonal division is done
      at the end of the computation. */
   Ai = nalu_hypre_CCBoxIndexRank( A_data_box, start );

#define DEVICE_VAR is_device_ptr(tp,bp)
   if ( constant_coefficient == 1 ) /* constant diagonal */
   {
      Apd = nalu_hypre_StructMatrixBoxData(A, boxarray_id, diag_rank);
      AApd = 1 / Apd[Ai];

      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          b_data_box, start, stride, bi,
                          t_data_box, start, stride, ti);
      {
         tp[ti] = AApd * bp[bi];
      }
      nalu_hypre_BoxLoop2End(bi, ti);
   }
   else /* constant_coefficient==2, variable diagonal */
   {
      AApd = 1;
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          b_data_box, start, stride, bi,
                          t_data_box, start, stride, ti);
      {
         tp[ti] = bp[bi];
      }
      nalu_hypre_BoxLoop2End(bi, ti);
   }
#undef DEVICE_VAR

   /* unroll up to depth MAX_DEPTH */
   for (si = 0; si < stencil_size; si += MAX_DEPTH)
   {
      depth = nalu_hypre_min(MAX_DEPTH, (stencil_size - si));

      for (k = 0, sk = si; k < depth; sk++)
      {
         if (sk == diag_rank)
         {
            depth--;
         }
         else
         {
            ssi[k] = sk;
            k++;
         }
      }

      switch (depth)
      {
         case 7:
            Ap6 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[6]);
            xoff6 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[6]]);

         case 6:
            Ap5 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[5]);
            xoff5 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[5]]);

         case 5:
            Ap4 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[4]);
            xoff4 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[4]]);

         case 4:
            Ap3 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[3]);
            xoff3 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[3]]);

         case 3:
            Ap2 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[2]);
            xoff2 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[2]]);

         case 2:
            Ap1 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[1]);
            xoff1 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[1]]);

         case 1:
            Ap0 = nalu_hypre_StructMatrixBoxData(A, boxarray_id, ssi[0]);
            xoff0 = nalu_hypre_BoxOffsetDistance(
                       x_data_box, stencil_shape[ssi[0]]);

         case 0:

            break;
      }

#define DEVICE_VAR is_device_ptr(tp,xp)
      switch (depth)
      {
         case 7:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            AAp3 = Ap3[Ai] * AApd;
            AAp4 = Ap4[Ai] * AApd;
            AAp5 = Ap5[Ai] * AApd;
            AAp6 = Ap6[Ai] * AApd;
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2] +
                  AAp3 * xp[xi + xoff3] +
                  AAp4 * xp[xi + xoff4] +
                  AAp5 * xp[xi + xoff5] +
                  AAp6 * xp[xi + xoff6];
            }
            nalu_hypre_BoxLoop2End(xi, ti);
            break;

         case 6:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            AAp3 = Ap3[Ai] * AApd;
            AAp4 = Ap4[Ai] * AApd;
            AAp5 = Ap5[Ai] * AApd;
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2] +
                  AAp3 * xp[xi + xoff3] +
                  AAp4 * xp[xi + xoff4] +
                  AAp5 * xp[xi + xoff5];
            }
            nalu_hypre_BoxLoop2End(xi, ti);
            break;

         case 5:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            AAp3 = Ap3[Ai] * AApd;
            AAp4 = Ap4[Ai] * AApd;
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2] +
                  AAp3 * xp[xi + xoff3] +
                  AAp4 * xp[xi + xoff4];
            }
            nalu_hypre_BoxLoop2End(xi, ti);
            break;

         case 4:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            AAp3 = Ap3[Ai] * AApd;
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2] +
                  AAp3 * xp[xi + xoff3];
            }
            nalu_hypre_BoxLoop2End(xi, ti);
            break;

         case 3:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            AAp2 = Ap2[Ai] * AApd;
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1] +
                  AAp2 * xp[xi + xoff2];
            }
            nalu_hypre_BoxLoop2End(xi, ti);
            break;

         case 2:
            AAp0 = Ap0[Ai] * AApd;
            AAp1 = Ap1[Ai] * AApd;
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0] +
                  AAp1 * xp[xi + xoff1];
            }
            nalu_hypre_BoxLoop2End(xi, ti);
            break;

         case 1:
            AAp0 = Ap0[Ai] * AApd;
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               tp[ti] -=
                  AAp0 * xp[xi + xoff0];
            }
            nalu_hypre_BoxLoop2End(xi, ti);
            break;

         case 0:
            break;
      }
#undef DEVICE_VAR
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxSetTol( void   *relax_vdata,
                        NALU_HYPRE_Real  tol         )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   (relax_data -> tol) = tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxGetTol( void   *relax_vdata,
                        NALU_HYPRE_Real *tol         )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   *tol = (relax_data -> tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxSetMaxIter( void *relax_vdata,
                            NALU_HYPRE_Int   max_iter    )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   (relax_data -> max_iter) = max_iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxGetMaxIter( void *relax_vdata,
                            NALU_HYPRE_Int * max_iter    )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   *max_iter = (relax_data -> max_iter);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxSetZeroGuess( void *relax_vdata,
                              NALU_HYPRE_Int   zero_guess  )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   (relax_data -> zero_guess) = zero_guess;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxGetZeroGuess( void *relax_vdata,
                              NALU_HYPRE_Int * zero_guess  )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   *zero_guess = (relax_data -> zero_guess);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxGetNumIterations( void *relax_vdata,
                                  NALU_HYPRE_Int * num_iterations  )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   *num_iterations = (relax_data -> num_iterations);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxSetWeight( void    *relax_vdata,
                           NALU_HYPRE_Real   weight      )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   (relax_data -> weight) = weight;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxSetNumPointsets( void *relax_vdata,
                                 NALU_HYPRE_Int   num_pointsets )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;
   NALU_HYPRE_Int             i;

   /* free up old pointset memory */
   for (i = 0; i < (relax_data -> num_pointsets); i++)
   {
      nalu_hypre_TFree(relax_data -> pointset_indices[i], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(relax_data -> pointset_sizes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(relax_data -> pointset_ranks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(relax_data -> pointset_strides, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(relax_data -> pointset_indices, NALU_HYPRE_MEMORY_HOST);

   /* alloc new pointset memory */
   (relax_data -> num_pointsets)    = num_pointsets;
   (relax_data -> pointset_sizes)   = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_pointsets, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> pointset_ranks)   = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_pointsets, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> pointset_strides) = nalu_hypre_TAlloc(nalu_hypre_Index,  num_pointsets, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> pointset_indices) = nalu_hypre_TAlloc(nalu_hypre_Index *,
                                                   num_pointsets, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_pointsets; i++)
   {
      (relax_data -> pointset_sizes[i]) = 0;
      (relax_data -> pointset_ranks[i]) = i;
      (relax_data -> pointset_indices[i]) = NULL;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxSetPointset( void        *relax_vdata,
                             NALU_HYPRE_Int    pointset,
                             NALU_HYPRE_Int    pointset_size,
                             nalu_hypre_Index  pointset_stride,
                             nalu_hypre_Index *pointset_indices )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;
   NALU_HYPRE_Int             i;

   /* free up old pointset memory */
   nalu_hypre_TFree(relax_data -> pointset_indices[pointset], NALU_HYPRE_MEMORY_HOST);

   /* alloc new pointset memory */
   (relax_data -> pointset_indices[pointset]) =
      nalu_hypre_TAlloc(nalu_hypre_Index,  pointset_size, NALU_HYPRE_MEMORY_HOST);

   (relax_data -> pointset_sizes[pointset]) = pointset_size;
   nalu_hypre_CopyIndex(pointset_stride,
                   (relax_data -> pointset_strides[pointset]));
   for (i = 0; i < pointset_size; i++)
   {
      nalu_hypre_CopyIndex(pointset_indices[i],
                      (relax_data -> pointset_indices[pointset][i]));
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxSetPointsetRank( void *relax_vdata,
                                 NALU_HYPRE_Int   pointset,
                                 NALU_HYPRE_Int   pointset_rank )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   (relax_data -> pointset_ranks[pointset]) = pointset_rank;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PointRelaxSetTempVec( void               *relax_vdata,
                            nalu_hypre_StructVector *t           )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   nalu_hypre_StructVectorDestroy(relax_data -> t);
   (relax_data -> t) = nalu_hypre_StructVectorRef(t);

   return nalu_hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_PointRelaxGetFinalRelativeResidualNorm( void * relax_vdata, NALU_HYPRE_Real * norm )
{
   nalu_hypre_PointRelaxData *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;

   *norm = relax_data -> rresnorm;
   return 0;
}

/*--------------------------------------------------------------------------
 * Special vector operation for use in nalu_hypre_PointRelax -
 * convex combination of vectors on specified pointsets.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_relax_wtx( void *relax_vdata, NALU_HYPRE_Int pointset,
                           nalu_hypre_StructVector *t, nalu_hypre_StructVector *x )
/* Sets x to a convex combination of x and t,  x = weight * t + (1-weight) * x,
   but only in the specified pointset */
{
   nalu_hypre_PointRelaxData  *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;
   NALU_HYPRE_Real             weight           = (relax_data -> weight);
   nalu_hypre_Index           *pointset_strides = (relax_data -> pointset_strides);
   nalu_hypre_ComputePkg     **compute_pkgs     = (relax_data -> compute_pkgs);
   nalu_hypre_ComputePkg      *compute_pkg;

   nalu_hypre_IndexRef         stride;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            loop_size;

   NALU_HYPRE_Real weightc = 1 - weight;
   NALU_HYPRE_Real *xp, *tp;
   NALU_HYPRE_Int compute_i, i, j;

   nalu_hypre_BoxArrayArray   *compute_box_aa;
   nalu_hypre_BoxArray        *compute_box_a;
   nalu_hypre_Box             *compute_box;
   nalu_hypre_Box             *x_data_box;
   nalu_hypre_Box             *t_data_box;

   compute_pkg = compute_pkgs[pointset];
   stride = pointset_strides[pointset];

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(compute_pkg);
         }
         break;

         case 1:
         {
            compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      nalu_hypre_ForBoxArrayI(i, compute_box_aa)
      {
         compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

         x_data_box =
            nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
         t_data_box =
            nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(t), i);

         xp = nalu_hypre_StructVectorBoxData(x, i);
         tp = nalu_hypre_StructVectorBoxData(t, i);

         nalu_hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

            start  = nalu_hypre_BoxIMin(compute_box);
            nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,tp)
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               xp[xi] = weight * tp[ti] + weightc * xp[xi];
            }
            nalu_hypre_BoxLoop2End(xi, ti);
#undef DEVICE_VAR
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Special vector operation for use in nalu_hypre_PointRelax -
 * vector copy on specified pointsets.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_relax_copy( void *relax_vdata, NALU_HYPRE_Int pointset,
                            nalu_hypre_StructVector *t, nalu_hypre_StructVector *x )
/* Sets x to t, x=t, but only in the specified pointset. */
{
   nalu_hypre_PointRelaxData  *relax_data = (nalu_hypre_PointRelaxData *)relax_vdata;
   nalu_hypre_Index           *pointset_strides = (relax_data -> pointset_strides);
   nalu_hypre_ComputePkg     **compute_pkgs     = (relax_data -> compute_pkgs);
   nalu_hypre_ComputePkg      *compute_pkg;

   nalu_hypre_IndexRef         stride;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            loop_size;

   NALU_HYPRE_Real *xp, *tp;
   NALU_HYPRE_Int compute_i, i, j;

   nalu_hypre_BoxArrayArray   *compute_box_aa;
   nalu_hypre_BoxArray        *compute_box_a;
   nalu_hypre_Box             *compute_box;
   nalu_hypre_Box             *x_data_box;
   nalu_hypre_Box             *t_data_box;

   compute_pkg = compute_pkgs[pointset];
   stride = pointset_strides[pointset];

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(compute_pkg);
         }
         break;

         case 1:
         {
            compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      nalu_hypre_ForBoxArrayI(i, compute_box_aa)
      {
         compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

         x_data_box =
            nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
         t_data_box =
            nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(t), i);

         xp = nalu_hypre_StructVectorBoxData(x, i);
         tp = nalu_hypre_StructVectorBoxData(t, i);

         nalu_hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

            start  = nalu_hypre_BoxIMin(compute_box);
            nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,tp)
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                                x_data_box, start, stride, xi,
                                t_data_box, start, stride, ti);
            {
               xp[xi] = tp[ti];
            }
            nalu_hypre_BoxLoop2End(xi, ti);
#undef DEVICE_VAR
         }
      }
   }

   return nalu_hypre_error_flag;
}
