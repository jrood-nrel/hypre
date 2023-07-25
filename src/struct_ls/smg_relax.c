/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int               setup_temp_vec;
   NALU_HYPRE_Int               setup_a_rem;
   NALU_HYPRE_Int               setup_a_sol;

   MPI_Comm                comm;

   NALU_HYPRE_Int               memory_use;
   NALU_HYPRE_Real              tol;
   NALU_HYPRE_Int               max_iter;
   NALU_HYPRE_Int               zero_guess;

   NALU_HYPRE_Int               num_spaces;
   NALU_HYPRE_Int              *space_indices;
   NALU_HYPRE_Int              *space_strides;

   NALU_HYPRE_Int               num_pre_spaces;
   NALU_HYPRE_Int               num_reg_spaces;
   NALU_HYPRE_Int              *pre_space_ranks;
   NALU_HYPRE_Int              *reg_space_ranks;

   nalu_hypre_Index             base_index;
   nalu_hypre_Index             base_stride;
   nalu_hypre_BoxArray         *base_box_array;

   NALU_HYPRE_Int               stencil_dim;

   nalu_hypre_StructMatrix     *A;
   nalu_hypre_StructVector     *b;
   nalu_hypre_StructVector     *x;

   nalu_hypre_StructVector     *temp_vec;
   nalu_hypre_StructMatrix     *A_sol;  /* Coefficients of A that make up
                                      the (sol)ve part of the relaxation */
   nalu_hypre_StructMatrix     *A_rem;  /* Coefficients of A (rem)aining:
                                      A_rem = A - A_sol                  */
   void                  **residual_data;  /* Array of size `num_spaces' */
   void                  **solve_data;     /* Array of size `num_spaces' */

   /* log info (always logged) */
   NALU_HYPRE_Int               num_iterations;
   NALU_HYPRE_Int               time_index;

   NALU_HYPRE_Int               num_pre_relax;
   NALU_HYPRE_Int               num_post_relax;

   NALU_HYPRE_Int               max_level;
} nalu_hypre_SMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SMGRelaxCreate( MPI_Comm  comm )
{
   nalu_hypre_SMGRelaxData *relax_data;

   relax_data = nalu_hypre_CTAlloc(nalu_hypre_SMGRelaxData,  1, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;
   (relax_data -> comm)           = comm;
   (relax_data -> base_box_array) = NULL;
   (relax_data -> time_index)     = nalu_hypre_InitializeTiming("SMGRelax");
   /* set defaults */
   (relax_data -> memory_use)         = 0;
   (relax_data -> tol)                = 1.0e-06;
   (relax_data -> max_iter)           = 1000;
   (relax_data -> zero_guess)         = 0;
   (relax_data -> num_spaces)         = 1;
   (relax_data -> space_indices)      = nalu_hypre_TAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> space_strides)      = nalu_hypre_TAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> space_indices[0])   = 0;
   (relax_data -> space_strides[0])   = 1;
   (relax_data -> num_pre_spaces)     = 0;
   (relax_data -> num_reg_spaces)     = 1;
   (relax_data -> pre_space_ranks)    = NULL;
   (relax_data -> reg_space_ranks)    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> reg_space_ranks[0]) = 0;
   nalu_hypre_SetIndex3((relax_data -> base_index), 0, 0, 0);
   nalu_hypre_SetIndex3((relax_data -> base_stride), 1, 1, 1);
   (relax_data -> A)                  = NULL;
   (relax_data -> b)                  = NULL;
   (relax_data -> x)                  = NULL;
   (relax_data -> temp_vec)           = NULL;

   (relax_data -> num_pre_relax)  = 1;
   (relax_data -> num_post_relax) = 1;
   (relax_data -> max_level)      = -1;
   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxDestroyTempVec( void *relax_vdata )
{
   nalu_hypre_SMGRelaxData  *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   nalu_hypre_StructVectorDestroy(relax_data -> temp_vec);
   (relax_data -> setup_temp_vec) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxDestroyARem( void *relax_vdata )
{
   nalu_hypre_SMGRelaxData  *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;
   NALU_HYPRE_Int            i;

   if (relax_data -> A_rem)
   {
      for (i = 0; i < (relax_data -> num_spaces); i++)
      {
         nalu_hypre_SMGResidualDestroy(relax_data -> residual_data[i]);
      }
      nalu_hypre_TFree(relax_data -> residual_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_StructMatrixDestroy(relax_data -> A_rem);
      (relax_data -> A_rem) = NULL;
   }
   (relax_data -> setup_a_rem) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxDestroyASol( void *relax_vdata )
{
   nalu_hypre_SMGRelaxData  *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;
   NALU_HYPRE_Int            stencil_dim;
   NALU_HYPRE_Int            i;

   if (relax_data -> A_sol)
   {
      stencil_dim = (relax_data -> stencil_dim);
      for (i = 0; i < (relax_data -> num_spaces); i++)
      {
         if (stencil_dim > 2)
         {
            nalu_hypre_SMGDestroy(relax_data -> solve_data[i]);
         }
         else
         {
            nalu_hypre_CyclicReductionDestroy(relax_data -> solve_data[i]);
         }
      }
      nalu_hypre_TFree(relax_data -> solve_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_StructMatrixDestroy(relax_data -> A_sol);
      (relax_data -> A_sol) = NULL;
   }
   (relax_data -> setup_a_sol) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxDestroy( void *relax_vdata )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   if (relax_data)
   {
      nalu_hypre_TFree(relax_data -> space_indices, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> space_strides, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> pre_space_ranks, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> reg_space_ranks, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_BoxArrayDestroy(relax_data -> base_box_array);

      nalu_hypre_StructMatrixDestroy(relax_data -> A);
      nalu_hypre_StructVectorDestroy(relax_data -> b);
      nalu_hypre_StructVectorDestroy(relax_data -> x);

      nalu_hypre_SMGRelaxDestroyTempVec(relax_vdata);
      nalu_hypre_SMGRelaxDestroyARem(relax_vdata);
      nalu_hypre_SMGRelaxDestroyASol(relax_vdata);

      nalu_hypre_FinalizeTiming(relax_data -> time_index);
      nalu_hypre_TFree(relax_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelax( void               *relax_vdata,
                nalu_hypre_StructMatrix *A,
                nalu_hypre_StructVector *b,
                nalu_hypre_StructVector *x           )
{
   nalu_hypre_SMGRelaxData   *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   NALU_HYPRE_Int             zero_guess;
   NALU_HYPRE_Int             stencil_dim;
   nalu_hypre_StructVector   *temp_vec;
   nalu_hypre_StructMatrix   *A_sol;
   nalu_hypre_StructMatrix   *A_rem;
   void                **residual_data;
   void                **solve_data;

   nalu_hypre_IndexRef        base_stride;
   nalu_hypre_BoxArray       *base_box_a;
   NALU_HYPRE_Real            zero = 0.0;

   NALU_HYPRE_Int             max_iter;
   NALU_HYPRE_Int             num_spaces;
   NALU_HYPRE_Int            *space_ranks;

   NALU_HYPRE_Int             i, j, k, is;

   /*----------------------------------------------------------
    * Note: The zero_guess stuff is not handled correctly
    * for general relaxation parameters.  It is correct when
    * the spaces are independent sets in the direction of
    * relaxation.
    *----------------------------------------------------------*/

   nalu_hypre_BeginTiming(relax_data -> time_index);

   /*----------------------------------------------------------
    * Set up the solver
    *----------------------------------------------------------*/

   /* insure that the solver memory gets fully set up */
   if ((relax_data -> setup_a_sol) > 0)
   {
      (relax_data -> setup_a_sol) = 2;
   }

   nalu_hypre_SMGRelaxSetup(relax_vdata, A, b, x);

   zero_guess      = (relax_data -> zero_guess);
   stencil_dim     = (relax_data -> stencil_dim);
   temp_vec        = (relax_data -> temp_vec);
   A_sol           = (relax_data -> A_sol);
   A_rem           = (relax_data -> A_rem);
   residual_data   = (relax_data -> residual_data);
   solve_data      = (relax_data -> solve_data);

   /*----------------------------------------------------------
    * Set zero values
    *----------------------------------------------------------*/

   if (zero_guess)
   {
      base_stride = (relax_data -> base_stride);
      base_box_a = (relax_data -> base_box_array);
      nalu_hypre_SMGSetStructVectorConstantValues(x, zero, base_box_a, base_stride);
   }

   /*----------------------------------------------------------
    * Iterate
    *----------------------------------------------------------*/

   for (k = 0; k < 2; k++)
   {
      switch (k)
      {
         /* Do pre-relaxation iterations */
         case 0:
            max_iter    = 1;
            num_spaces  = (relax_data -> num_pre_spaces);
            space_ranks = (relax_data -> pre_space_ranks);
            break;

         /* Do regular relaxation iterations */
         case 1:
            max_iter    = (relax_data -> max_iter);
            num_spaces  = (relax_data -> num_reg_spaces);
            space_ranks = (relax_data -> reg_space_ranks);
            break;
      }

      for (i = 0; i < max_iter; i++)
      {
         for (j = 0; j < num_spaces; j++)
         {
            is = space_ranks[j];

            nalu_hypre_SMGResidual(residual_data[is], A_rem, x, b, temp_vec);

            if (stencil_dim > 2)
            {
               nalu_hypre_SMGSolve(solve_data[is], A_sol, temp_vec, x);
            }
            else
            {
               nalu_hypre_CyclicReduction(solve_data[is], A_sol, temp_vec, x);
            }
         }

         (relax_data -> num_iterations) = (i + 1);
      }
   }

   /*----------------------------------------------------------
    * Free up memory according to memory_use parameter
    *----------------------------------------------------------*/

   if ((stencil_dim - 1) <= (relax_data -> memory_use))
   {
      nalu_hypre_SMGRelaxDestroyASol(relax_vdata);
   }

   nalu_hypre_EndTiming(relax_data -> time_index);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetup( void               *relax_vdata,
                     nalu_hypre_StructMatrix *A,
                     nalu_hypre_StructVector *b,
                     nalu_hypre_StructVector *x           )
{
   nalu_hypre_SMGRelaxData  *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;
   NALU_HYPRE_Int            stencil_dim;
   NALU_HYPRE_Int            a_sol_test;

   stencil_dim = nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A));
   (relax_data -> stencil_dim) = stencil_dim;
   nalu_hypre_StructMatrixDestroy(relax_data -> A);
   nalu_hypre_StructVectorDestroy(relax_data -> b);
   nalu_hypre_StructVectorDestroy(relax_data -> x);
   (relax_data -> A) = nalu_hypre_StructMatrixRef(A);
   (relax_data -> b) = nalu_hypre_StructVectorRef(b);
   (relax_data -> x) = nalu_hypre_StructVectorRef(x);

   /*----------------------------------------------------------
    * Set up memory according to memory_use parameter.
    *
    * If a subset of the solver memory is not to be set up
    * until the solve is actually done, it's "setup" tag
    * should have a value greater than 1.
    *----------------------------------------------------------*/

   if ((stencil_dim - 1) <= (relax_data -> memory_use))
   {
      a_sol_test = 1;
   }
   else
   {
      a_sol_test = 0;
   }

   /*----------------------------------------------------------
    * Set up the solver
    *----------------------------------------------------------*/

   if ((relax_data -> setup_temp_vec) > 0)
   {
      nalu_hypre_SMGRelaxSetupTempVec(relax_vdata, A, b, x);
   }

   if ((relax_data -> setup_a_rem) > 0)
   {
      nalu_hypre_SMGRelaxSetupARem(relax_vdata, A, b, x);
   }

   if ((relax_data -> setup_a_sol) > a_sol_test)
   {
      nalu_hypre_SMGRelaxSetupASol(relax_vdata, A, b, x);
   }

   if ((relax_data -> base_box_array) == NULL)
   {
      nalu_hypre_SMGRelaxSetupBaseBoxArray(relax_vdata, A, b, x);
   }


   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetupTempVec( void               *relax_vdata,
                            nalu_hypre_StructMatrix *A,
                            nalu_hypre_StructVector *b,
                            nalu_hypre_StructVector *x           )
{
   nalu_hypre_SMGRelaxData  *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;
   nalu_hypre_StructVector  *temp_vec   = (relax_data -> temp_vec);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   if ((relax_data -> temp_vec) == NULL)
   {
      temp_vec = nalu_hypre_StructVectorCreate(nalu_hypre_StructVectorComm(b),
                                          nalu_hypre_StructVectorGrid(b));
      nalu_hypre_StructVectorSetNumGhost(temp_vec, nalu_hypre_StructVectorNumGhost(b));
      nalu_hypre_StructVectorInitialize(temp_vec);
      nalu_hypre_StructVectorAssemble(temp_vec);
      (relax_data -> temp_vec) = temp_vec;
   }
   (relax_data -> setup_temp_vec) = 0;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SMGRelaxSetupARem
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetupARem( void               *relax_vdata,
                         nalu_hypre_StructMatrix *A,
                         nalu_hypre_StructVector *b,
                         nalu_hypre_StructVector *x           )
{
   nalu_hypre_SMGRelaxData   *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   NALU_HYPRE_Int             num_spaces    = (relax_data -> num_spaces);
   NALU_HYPRE_Int            *space_indices = (relax_data -> space_indices);
   NALU_HYPRE_Int            *space_strides = (relax_data -> space_strides);
   nalu_hypre_StructVector   *temp_vec      = (relax_data -> temp_vec);

   nalu_hypre_StructStencil  *stencil       = nalu_hypre_StructMatrixStencil(A);
   nalu_hypre_Index          *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   NALU_HYPRE_Int             stencil_size  = nalu_hypre_StructStencilSize(stencil);
   NALU_HYPRE_Int             stencil_dim   = nalu_hypre_StructStencilNDim(stencil);

   nalu_hypre_StructMatrix   *A_rem;
   void                **residual_data;

   nalu_hypre_Index           base_index;
   nalu_hypre_Index           base_stride;

   NALU_HYPRE_Int             num_stencil_indices;
   NALU_HYPRE_Int            *stencil_indices;

   NALU_HYPRE_Int             i;

   /*----------------------------------------------------------
    * Free up old data before putting new data into structure
    *----------------------------------------------------------*/

   nalu_hypre_SMGRelaxDestroyARem(relax_vdata);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   nalu_hypre_CopyIndex((relax_data -> base_index),  base_index);
   nalu_hypre_CopyIndex((relax_data -> base_stride), base_stride);

   stencil_indices = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (nalu_hypre_IndexD(stencil_shape[i], (stencil_dim - 1)) != 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }
   A_rem = nalu_hypre_StructMatrixCreateMask(A, num_stencil_indices, stencil_indices);
   nalu_hypre_TFree(stencil_indices, NALU_HYPRE_MEMORY_HOST);

   /* Set up residual_data */
   residual_data = nalu_hypre_TAlloc(void *,  num_spaces, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_spaces; i++)
   {
      nalu_hypre_IndexD(base_index,  (stencil_dim - 1)) = space_indices[i];
      nalu_hypre_IndexD(base_stride, (stencil_dim - 1)) = space_strides[i];

      residual_data[i] = nalu_hypre_SMGResidualCreate();
      nalu_hypre_SMGResidualSetBase(residual_data[i], base_index, base_stride);
      nalu_hypre_SMGResidualSetup(residual_data[i], A_rem, x, b, temp_vec);
   }

   (relax_data -> A_rem)         = A_rem;
   (relax_data -> residual_data) = residual_data;

   (relax_data -> setup_a_rem) = 0;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetupASol( void               *relax_vdata,
                         nalu_hypre_StructMatrix *A,
                         nalu_hypre_StructVector *b,
                         nalu_hypre_StructVector *x           )
{
   nalu_hypre_SMGRelaxData   *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   NALU_HYPRE_Int             num_spaces    = (relax_data -> num_spaces);
   NALU_HYPRE_Int            *space_indices = (relax_data -> space_indices);
   NALU_HYPRE_Int            *space_strides = (relax_data -> space_strides);
   nalu_hypre_StructVector   *temp_vec      = (relax_data -> temp_vec);

   NALU_HYPRE_Int             num_pre_relax   = (relax_data -> num_pre_relax);
   NALU_HYPRE_Int             num_post_relax  = (relax_data -> num_post_relax);

   nalu_hypre_StructStencil  *stencil       = nalu_hypre_StructMatrixStencil(A);
   nalu_hypre_Index          *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   NALU_HYPRE_Int             stencil_size  = nalu_hypre_StructStencilSize(stencil);
   NALU_HYPRE_Int             stencil_dim   = nalu_hypre_StructStencilNDim(stencil);

   nalu_hypre_StructMatrix   *A_sol;
   void                **solve_data;

   nalu_hypre_Index           base_index;
   nalu_hypre_Index           base_stride;

   NALU_HYPRE_Int             num_stencil_indices;
   NALU_HYPRE_Int            *stencil_indices;

   NALU_HYPRE_Int             i;

   /*----------------------------------------------------------
    * Free up old data before putting new data into structure
    *----------------------------------------------------------*/

   nalu_hypre_SMGRelaxDestroyASol(relax_vdata);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   nalu_hypre_CopyIndex((relax_data -> base_index),  base_index);
   nalu_hypre_CopyIndex((relax_data -> base_stride), base_stride);

   stencil_indices = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (nalu_hypre_IndexD(stencil_shape[i], (stencil_dim - 1)) == 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }

   A_sol = nalu_hypre_StructMatrixCreateMask(A, num_stencil_indices, stencil_indices);
   nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A_sol)) = stencil_dim - 1;
   nalu_hypre_TFree(stencil_indices, NALU_HYPRE_MEMORY_HOST);

   /* Set up solve_data */
   solve_data    = nalu_hypre_TAlloc(void *,  num_spaces, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_spaces; i++)
   {
      nalu_hypre_IndexD(base_index,  (stencil_dim - 1)) = space_indices[i];
      nalu_hypre_IndexD(base_stride, (stencil_dim - 1)) = space_strides[i];

      if (stencil_dim > 2)
      {
         solve_data[i] = nalu_hypre_SMGCreate(relax_data -> comm);
         nalu_hypre_SMGSetNumPreRelax( solve_data[i], num_pre_relax);
         nalu_hypre_SMGSetNumPostRelax( solve_data[i], num_post_relax);
         nalu_hypre_SMGSetBase(solve_data[i], base_index, base_stride);
         nalu_hypre_SMGSetMemoryUse(solve_data[i], (relax_data -> memory_use));
         nalu_hypre_SMGSetTol(solve_data[i], 0.0);
         nalu_hypre_SMGSetMaxIter(solve_data[i], 1);
         nalu_hypre_StructSMGSetMaxLevel(solve_data[i], (relax_data -> max_level));
         nalu_hypre_SMGSetup(solve_data[i], A_sol, temp_vec, x);
      }
      else
      {
         solve_data[i] = nalu_hypre_CyclicReductionCreate(relax_data -> comm);
         nalu_hypre_CyclicReductionSetBase(solve_data[i], base_index, base_stride);
         //nalu_hypre_CyclicReductionSetMaxLevel(solve_data[i], -1);//(relax_data -> max_level)+10);
         nalu_hypre_CyclicReductionSetup(solve_data[i], A_sol, temp_vec, x);
      }
   }

   (relax_data -> A_sol)      = A_sol;
   (relax_data -> solve_data) = solve_data;

   (relax_data -> setup_a_sol) = 0;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetTempVec( void               *relax_vdata,
                          nalu_hypre_StructVector *temp_vec    )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   nalu_hypre_SMGRelaxDestroyTempVec(relax_vdata);
   (relax_data -> temp_vec) = nalu_hypre_StructVectorRef(temp_vec);

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetMemoryUse( void *relax_vdata,
                            NALU_HYPRE_Int   memory_use  )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> memory_use) = memory_use;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetTol( void   *relax_vdata,
                      NALU_HYPRE_Real  tol         )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> tol) = tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetMaxIter( void *relax_vdata,
                          NALU_HYPRE_Int   max_iter    )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> max_iter) = max_iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetZeroGuess( void *relax_vdata,
                            NALU_HYPRE_Int   zero_guess  )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> zero_guess) = zero_guess;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetNumSpaces( void *relax_vdata,
                            NALU_HYPRE_Int   num_spaces      )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;
   NALU_HYPRE_Int           i;

   (relax_data -> num_spaces) = num_spaces;

   nalu_hypre_TFree(relax_data -> space_indices, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(relax_data -> space_strides, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(relax_data -> pre_space_ranks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(relax_data -> reg_space_ranks, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> space_indices)   = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_spaces, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> space_strides)   = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_spaces, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> num_pre_spaces)  = 0;
   (relax_data -> num_reg_spaces)  = num_spaces;
   (relax_data -> pre_space_ranks) = NULL;
   (relax_data -> reg_space_ranks) = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_spaces, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_spaces; i++)
   {
      (relax_data -> space_indices[i]) = 0;
      (relax_data -> space_strides[i]) = 1;
      (relax_data -> reg_space_ranks[i]) = i;
   }

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetNumPreSpaces( void *relax_vdata,
                               NALU_HYPRE_Int   num_pre_spaces )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;
   NALU_HYPRE_Int           i;

   (relax_data -> num_pre_spaces) = num_pre_spaces;

   nalu_hypre_TFree(relax_data -> pre_space_ranks, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> pre_space_ranks) = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_pre_spaces, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_pre_spaces; i++)
   {
      (relax_data -> pre_space_ranks[i]) = 0;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetNumRegSpaces( void *relax_vdata,
                               NALU_HYPRE_Int   num_reg_spaces )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;
   NALU_HYPRE_Int           i;

   (relax_data -> num_reg_spaces) = num_reg_spaces;

   nalu_hypre_TFree(relax_data -> reg_space_ranks, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> reg_space_ranks) = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_reg_spaces, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_reg_spaces; i++)
   {
      (relax_data -> reg_space_ranks[i]) = 0;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetSpace( void *relax_vdata,
                        NALU_HYPRE_Int   i,
                        NALU_HYPRE_Int   space_index,
                        NALU_HYPRE_Int   space_stride )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> space_indices[i]) = space_index;
   (relax_data -> space_strides[i]) = space_stride;

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetRegSpaceRank( void *relax_vdata,
                               NALU_HYPRE_Int   i,
                               NALU_HYPRE_Int   reg_space_rank )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> reg_space_ranks[i]) = reg_space_rank;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetPreSpaceRank( void *relax_vdata,
                               NALU_HYPRE_Int   i,
                               NALU_HYPRE_Int   pre_space_rank  )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> pre_space_ranks[i]) = pre_space_rank;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetBase( void        *relax_vdata,
                       nalu_hypre_Index  base_index,
                       nalu_hypre_Index  base_stride )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;
   NALU_HYPRE_Int           d;

   for (d = 0; d < 3; d++)
   {
      nalu_hypre_IndexD((relax_data -> base_index),  d) =
         nalu_hypre_IndexD(base_index,  d);
      nalu_hypre_IndexD((relax_data -> base_stride), d) =
         nalu_hypre_IndexD(base_stride, d);
   }

   if ((relax_data -> base_box_array) != NULL)
   {
      nalu_hypre_BoxArrayDestroy((relax_data -> base_box_array));
      (relax_data -> base_box_array) = NULL;
   }

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note that we require at least 1 pre-relax sweep.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetNumPreRelax( void *relax_vdata,
                              NALU_HYPRE_Int   num_pre_relax )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> num_pre_relax) = nalu_hypre_max(num_pre_relax, 1);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetNumPostRelax( void *relax_vdata,
                               NALU_HYPRE_Int   num_post_relax )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> num_post_relax) = num_post_relax;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetNewMatrixStencil( void                *relax_vdata,
                                   nalu_hypre_StructStencil *diff_stencil )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   nalu_hypre_Index        *stencil_shape = nalu_hypre_StructStencilShape(diff_stencil);
   NALU_HYPRE_Int           stencil_size  = nalu_hypre_StructStencilSize(diff_stencil);
   NALU_HYPRE_Int           stencil_dim   = nalu_hypre_StructStencilNDim(diff_stencil);

   NALU_HYPRE_Int           i;

   for (i = 0; i < stencil_size; i++)
   {
      if (nalu_hypre_IndexD(stencil_shape[i], (stencil_dim - 1)) != 0)
      {
         (relax_data -> setup_a_rem) = 1;
      }
      else
      {
         (relax_data -> setup_a_sol) = 1;
      }
   }

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_SMGRelaxSetupBaseBoxArray
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetupBaseBoxArray( void               *relax_vdata,
                                 nalu_hypre_StructMatrix *A,
                                 nalu_hypre_StructVector *b,
                                 nalu_hypre_StructVector *x           )
{
   nalu_hypre_SMGRelaxData  *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   nalu_hypre_StructGrid    *grid;
   nalu_hypre_BoxArray      *boxes;
   nalu_hypre_BoxArray      *base_box_array;

   grid  = nalu_hypre_StructVectorGrid(x);
   boxes = nalu_hypre_StructGridBoxes(grid);

   base_box_array = nalu_hypre_BoxArrayDuplicate(boxes);
   nalu_hypre_ProjectBoxArray(base_box_array,
                         (relax_data -> base_index),
                         (relax_data -> base_stride));

   (relax_data -> base_box_array) = base_box_array;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGRelaxSetMaxLevel( void *relax_vdata,
                           NALU_HYPRE_Int   num_max_level )
{
   nalu_hypre_SMGRelaxData *relax_data = (nalu_hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> max_level) = num_max_level;

   return nalu_hypre_error_flag;
}
