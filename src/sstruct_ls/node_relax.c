/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_struct_mv.hpp"

#include "gselim.h"

/* TODO consider adding it to semistruct header files */
#define NALU_HYPRE_MAXVARS 4

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;

   NALU_HYPRE_Real              tol;                /* not yet used */
   NALU_HYPRE_Int               max_iter;
   NALU_HYPRE_Int               rel_change;         /* not yet used */
   NALU_HYPRE_Int               zero_guess;
   NALU_HYPRE_Real              weight;

   NALU_HYPRE_Int               num_nodesets;
   NALU_HYPRE_Int              *nodeset_sizes;
   NALU_HYPRE_Int              *nodeset_ranks;
   nalu_hypre_Index            *nodeset_strides;
   nalu_hypre_Index           **nodeset_indices;

   nalu_hypre_SStructPMatrix   *A;
   nalu_hypre_SStructPVector   *b;
   nalu_hypre_SStructPVector   *x;

   nalu_hypre_SStructPVector   *t;

   NALU_HYPRE_Int             **diag_rank;

   /* defines sends and recieves for each struct_vector */
   nalu_hypre_ComputePkg     ***svec_compute_pkgs;
   nalu_hypre_CommHandle      **comm_handle;

   /* defines independent and dependent boxes for computations */
   nalu_hypre_ComputePkg      **compute_pkgs;

   /* pointers to local storage used to invert diagonal blocks */
   /*
   NALU_HYPRE_Real            *A_loc;
   NALU_HYPRE_Real            *x_loc;
   */

   /* pointers for vector and matrix data */
   NALU_HYPRE_MemoryLocation    memory_location;
   NALU_HYPRE_Real            **Ap;
   NALU_HYPRE_Real            **bp;
   NALU_HYPRE_Real            **xp;
   NALU_HYPRE_Real            **tp;

   /* log info (always logged) */
   NALU_HYPRE_Int               num_iterations;
   NALU_HYPRE_Int               time_index;
   NALU_HYPRE_Int               flops;

} nalu_hypre_NodeRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_NodeRelaxCreate( MPI_Comm  comm )
{
   nalu_hypre_NodeRelaxData *relax_data;

   nalu_hypre_Index          stride;
   nalu_hypre_Index          indices[1];

   relax_data = nalu_hypre_CTAlloc(nalu_hypre_NodeRelaxData,  1, NALU_HYPRE_MEMORY_HOST);

   (relax_data -> comm)       = comm;
   (relax_data -> time_index) = nalu_hypre_InitializeTiming("NodeRelax");

   /* set defaults */
   (relax_data -> tol)              = 1.0e-06;
   (relax_data -> max_iter)         = 1000;
   (relax_data -> rel_change)       = 0;
   (relax_data -> zero_guess)       = 0;
   (relax_data -> weight)           = 1.0;
   (relax_data -> num_nodesets)     = 0;
   (relax_data -> nodeset_sizes)    = NULL;
   (relax_data -> nodeset_ranks)    = NULL;
   (relax_data -> nodeset_strides)  = NULL;
   (relax_data -> nodeset_indices)  = NULL;
   (relax_data -> diag_rank)        = NULL;
   (relax_data -> t)                = NULL;
   /*
   (relax_data -> A_loc)            = NULL;
   (relax_data -> x_loc)            = NULL;
   */
   (relax_data -> Ap)               = NULL;
   (relax_data -> bp)               = NULL;
   (relax_data -> xp)               = NULL;
   (relax_data -> tp)               = NULL;
   (relax_data -> comm_handle)      = NULL;
   (relax_data -> svec_compute_pkgs) = NULL;
   (relax_data -> compute_pkgs)     = NULL;

   nalu_hypre_SetIndex3(stride, 1, 1, 1);
   nalu_hypre_SetIndex3(indices[0], 0, 0, 0);
   nalu_hypre_NodeRelaxSetNumNodesets((void *) relax_data, 1);
   nalu_hypre_NodeRelaxSetNodeset((void *) relax_data, 0, 1, stride, indices);

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxDestroy( void *relax_vdata )
{
   nalu_hypre_NodeRelaxData  *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;
   NALU_HYPRE_Int             i, vi;
   NALU_HYPRE_Int             nvars;

   if (relax_data)
   {
      NALU_HYPRE_MemoryLocation memory_location = relax_data -> memory_location;

      nvars = nalu_hypre_SStructPMatrixNVars(relax_data -> A);

      for (i = 0; i < (relax_data -> num_nodesets); i++)
      {
         nalu_hypre_TFree(relax_data -> nodeset_indices[i], NALU_HYPRE_MEMORY_HOST);
         for (vi = 0; vi < nvars; vi++)
         {
            nalu_hypre_ComputePkgDestroy(relax_data -> svec_compute_pkgs[i][vi]);
         }
         nalu_hypre_TFree(relax_data -> svec_compute_pkgs[i], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ComputePkgDestroy(relax_data -> compute_pkgs[i]);
      }
      nalu_hypre_TFree(relax_data -> nodeset_sizes, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> nodeset_ranks, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> nodeset_strides, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> nodeset_indices, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SStructPMatrixDestroy(relax_data -> A);
      nalu_hypre_SStructPVectorDestroy(relax_data -> b);
      nalu_hypre_SStructPVectorDestroy(relax_data -> x);
      nalu_hypre_TFree(relax_data -> svec_compute_pkgs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> comm_handle, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(relax_data -> compute_pkgs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SStructPVectorDestroy(relax_data -> t);
      /*
      nalu_hypre_TFree(relax_data -> x_loc, memory_location);
      nalu_hypre_TFree(relax_data -> A_loc, memory_location);
      */
      nalu_hypre_TFree(relax_data -> bp, memory_location);
      nalu_hypre_TFree(relax_data -> xp, memory_location);
      nalu_hypre_TFree(relax_data -> tp, memory_location);
      nalu_hypre_TFree(relax_data -> Ap, memory_location);
      for (vi = 0; vi < nvars; vi++)
      {
         nalu_hypre_TFree((relax_data -> diag_rank)[vi], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(relax_data -> diag_rank, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_FinalizeTiming(relax_data -> time_index);
      nalu_hypre_TFree(relax_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxSetup(  void                 *relax_vdata,
                       nalu_hypre_SStructPMatrix *A,
                       nalu_hypre_SStructPVector *b,
                       nalu_hypre_SStructPVector *x           )
{
   nalu_hypre_NodeRelaxData   *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;

   NALU_HYPRE_Int              num_nodesets    = (relax_data -> num_nodesets);
   NALU_HYPRE_Int             *nodeset_sizes   = (relax_data -> nodeset_sizes);
   nalu_hypre_Index           *nodeset_strides = (relax_data -> nodeset_strides);
   nalu_hypre_Index          **nodeset_indices = (relax_data -> nodeset_indices);
   NALU_HYPRE_Int              ndim = nalu_hypre_SStructPMatrixNDim(A);

   nalu_hypre_SStructPVector  *t;
   NALU_HYPRE_Int            **diag_rank;
   /*
   NALU_HYPRE_Real            *A_loc;
   NALU_HYPRE_Real            *x_loc;
   */
   NALU_HYPRE_Real           **Ap;
   NALU_HYPRE_Real           **bp;
   NALU_HYPRE_Real           **xp;
   NALU_HYPRE_Real           **tp;

   nalu_hypre_ComputeInfo     *compute_info;
   nalu_hypre_ComputePkg     **compute_pkgs;
   nalu_hypre_ComputePkg    ***svec_compute_pkgs;
   nalu_hypre_CommHandle     **comm_handle;

   nalu_hypre_Index            diag_index;
   nalu_hypre_IndexRef         stride;
   nalu_hypre_IndexRef         index;

   nalu_hypre_StructGrid      *sgrid;

   nalu_hypre_StructStencil   *sstencil;
   nalu_hypre_Index           *sstencil_shape;
   NALU_HYPRE_Int              sstencil_size;

   nalu_hypre_StructStencil   *sstencil_union;
   nalu_hypre_Index           *sstencil_union_shape;
   NALU_HYPRE_Int              sstencil_union_count;

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

   NALU_HYPRE_Int              i, j, k, p, m, s, compute_i;

   NALU_HYPRE_Int              vi, vj;
   NALU_HYPRE_Int              nvars;
   NALU_HYPRE_Int              dim;

   NALU_HYPRE_MemoryLocation   memory_location;

   /*----------------------------------------------------------
    * Set up the temp vector
    *----------------------------------------------------------*/

   if ((relax_data -> t) == NULL)
   {
      nalu_hypre_SStructPVectorCreate(nalu_hypre_SStructPVectorComm(b),
                                 nalu_hypre_SStructPVectorPGrid(b), &t);
      nalu_hypre_SStructPVectorInitialize(t);
      nalu_hypre_SStructPVectorAssemble(t);
      (relax_data -> t) = t;
   }

   /*----------------------------------------------------------
    * Find the matrix diagonals, use diag_rank[vi][vj] = -1 to
    * mark that the coresponding StructMatrix is NULL.
    *----------------------------------------------------------*/

   nvars = nalu_hypre_SStructPMatrixNVars(A);

   nalu_hypre_assert(nvars <= NALU_HYPRE_MAXVARS);

   diag_rank = nalu_hypre_CTAlloc(NALU_HYPRE_Int *, nvars, NALU_HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      diag_rank[vi] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         if (nalu_hypre_SStructPMatrixSMatrix(A, vi, vj) != NULL)
         {
            sstencil = nalu_hypre_SStructPMatrixSStencil(A, vi, vj);
            nalu_hypre_SetIndex3(diag_index, 0, 0, 0);
            diag_rank[vi][vj] = nalu_hypre_StructStencilElementRank(sstencil, diag_index);
         }
         else
         {
            diag_rank[vi][vj] = -1;
         }
      }
   }

   memory_location = nalu_hypre_StructMatrixMemoryLocation(nalu_hypre_SStructPMatrixSMatrix(A, 0, 0));

   /*----------------------------------------------------------
    * Allocate storage used to invert local diagonal blocks
    *----------------------------------------------------------*/
   /*
   i = nalu_hypre_NumThreads();
   x_loc = nalu_hypre_TAlloc(NALU_HYPRE_Real  , i*nvars,       memory_location);
   A_loc = nalu_hypre_TAlloc(NALU_HYPRE_Real  , i*nvars*nvars, memory_location);
   */

   /* Allocate pointers for vector and matrix */
   bp = nalu_hypre_TAlloc(NALU_HYPRE_Real *, nvars, memory_location);
   xp = nalu_hypre_TAlloc(NALU_HYPRE_Real *, nvars, memory_location);
   tp = nalu_hypre_TAlloc(NALU_HYPRE_Real *, nvars, memory_location);
   Ap = nalu_hypre_TAlloc(NALU_HYPRE_Real *, nvars * nvars, memory_location);

   /*----------------------------------------------------------
    * Set up the compute packages for each nodeset
    *----------------------------------------------------------*/

   sgrid = nalu_hypre_StructMatrixGrid(nalu_hypre_SStructPMatrixSMatrix(A, 0, 0));
   dim = nalu_hypre_StructStencilNDim(nalu_hypre_SStructPMatrixSStencil(A, 0, 0));

   compute_pkgs = nalu_hypre_CTAlloc(nalu_hypre_ComputePkg *, num_nodesets,
                                NALU_HYPRE_MEMORY_HOST);

   svec_compute_pkgs = nalu_hypre_CTAlloc(nalu_hypre_ComputePkg **, num_nodesets,
                                     NALU_HYPRE_MEMORY_HOST);

   comm_handle = nalu_hypre_CTAlloc(nalu_hypre_CommHandle *, nvars, NALU_HYPRE_MEMORY_HOST);

   for (p = 0; p < num_nodesets; p++)
   {
      /*----------------------------------------------------------
       * Set up the compute packages to define sends and recieves
       * for each struct_vector (svec_compute_pkgs) and the compute
       * package to define independent and dependent computations
       * (compute_pkgs).
       *----------------------------------------------------------*/
      svec_compute_pkgs[p] = nalu_hypre_CTAlloc(nalu_hypre_ComputePkg *, nvars, NALU_HYPRE_MEMORY_HOST);

      for (vi = -1; vi < nvars; vi++)
      {

         /*----------------------------------------------------------
          * The first execution (vi=-1) sets up the stencil to
          * define independent and dependent computations. The
          * stencil is the "union" over i,j of all stencils for
          * for struct_matrix A_ij.
          *
          * Other executions (vi > -1) set up the stencil to
          * define sends and recieves for the struct_vector vi.
          * The stencil for vector i is the "union" over j of all
          * stencils for struct_matrix A_ji.
          *----------------------------------------------------------*/
         sstencil_union_count = 0;
         if (vi == -1)
         {
            for (i = 0; i < nvars; i++)
            {
               for (vj = 0; vj < nvars; vj++)
               {
                  if (nalu_hypre_SStructPMatrixSMatrix(A, vj, i) != NULL)
                  {
                     sstencil = nalu_hypre_SStructPMatrixSStencil(A, vj, i);
                     sstencil_union_count += nalu_hypre_StructStencilSize(sstencil);
                  }
               }
            }
         }
         else
         {
            for (vj = 0; vj < nvars; vj++)
            {
               if (nalu_hypre_SStructPMatrixSMatrix(A, vj, vi) != NULL)
               {
                  sstencil = nalu_hypre_SStructPMatrixSStencil(A, vj, vi);
                  sstencil_union_count += nalu_hypre_StructStencilSize(sstencil);
               }
            }
         }
         sstencil_union_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,
                                              sstencil_union_count, NALU_HYPRE_MEMORY_HOST);
         sstencil_union_count = 0;
         if (vi == -1)
         {
            for (i = 0; i < nvars; i++)
            {
               for (vj = 0; vj < nvars; vj++)
               {
                  if (nalu_hypre_SStructPMatrixSMatrix(A, vj, i) != NULL)
                  {
                     sstencil = nalu_hypre_SStructPMatrixSStencil(A, vj, i);
                     sstencil_size = nalu_hypre_StructStencilSize(sstencil);
                     sstencil_shape = nalu_hypre_StructStencilShape(sstencil);
                     for (s = 0; s < sstencil_size; s++)
                     {
                        nalu_hypre_CopyIndex(sstencil_shape[s],
                                        sstencil_union_shape[sstencil_union_count]);
                        sstencil_union_count++;
                     }
                  }
               }
            }
         }
         else
         {
            for (vj = 0; vj < nvars; vj++)
            {
               if (nalu_hypre_SStructPMatrixSMatrix(A, vj, vi) != NULL)
               {
                  sstencil = nalu_hypre_SStructPMatrixSStencil(A, vj, vi);
                  sstencil_size = nalu_hypre_StructStencilSize(sstencil);
                  sstencil_shape = nalu_hypre_StructStencilShape(sstencil);
                  for (s = 0; s < sstencil_size; s++)
                  {
                     nalu_hypre_CopyIndex(sstencil_shape[s],
                                     sstencil_union_shape[sstencil_union_count]);
                     sstencil_union_count++;
                  }
               }
            }
         }

         sstencil_union = nalu_hypre_StructStencilCreate(dim, sstencil_union_count,
                                                    sstencil_union_shape);


         nalu_hypre_CreateComputeInfo(sgrid, sstencil_union, &compute_info);
         orig_indt_boxes = nalu_hypre_ComputeInfoIndtBoxes(compute_info);
         orig_dept_boxes = nalu_hypre_ComputeInfoDeptBoxes(compute_info);

         stride = nodeset_strides[p];

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
               nalu_hypre_BoxArraySetSize(new_box_a,
                                     box_a_size * nodeset_sizes[p]);

               k = 0;
               for (m = 0; m < nodeset_sizes[p]; m++)
               {
                  index  = nodeset_indices[p][m];

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

         if (vi == -1)
         {
            nalu_hypre_ComputePkgCreate(compute_info,
                                   nalu_hypre_StructVectorDataSpace(
                                      nalu_hypre_SStructPVectorSVector(x, 0)),
                                   1, sgrid, &compute_pkgs[p]);
         }
         else
         {
            nalu_hypre_ComputePkgCreate(compute_info,
                                   nalu_hypre_StructVectorDataSpace(
                                      nalu_hypre_SStructPVectorSVector(x, vi)),
                                   1, sgrid, &svec_compute_pkgs[p][vi]);
         }

         nalu_hypre_BoxArrayArrayDestroy(orig_indt_boxes);
         nalu_hypre_BoxArrayArrayDestroy(orig_dept_boxes);

         nalu_hypre_StructStencilDestroy(sstencil_union);
      }
   }

   /*----------------------------------------------------------
    * Set up the relax data structure
    *----------------------------------------------------------*/

   nalu_hypre_SStructPMatrixRef(A, &(relax_data -> A));
   nalu_hypre_SStructPVectorRef(x, &(relax_data -> x));
   nalu_hypre_SStructPVectorRef(b, &(relax_data -> b));

   (relax_data -> diag_rank) = diag_rank;
   /*
   (relax_data -> A_loc)     = A_loc;
   (relax_data -> x_loc)     = x_loc;
   */
   (relax_data -> Ap)    = Ap;
   (relax_data -> bp)    = bp;
   (relax_data -> tp)    = tp;
   (relax_data -> xp)    = xp;
   (relax_data -> memory_location) = memory_location;
   (relax_data -> compute_pkgs) = compute_pkgs;
   (relax_data -> svec_compute_pkgs) = svec_compute_pkgs;
   (relax_data -> comm_handle) = comm_handle;

   /*-----------------------------------------------------
    * Compute flops
    *-----------------------------------------------------*/

   scale = 0.0;
   for (p = 0; p < num_nodesets; p++)
   {
      stride = nodeset_strides[p];
      frac   = nalu_hypre_IndexX(stride);
      frac  *= nalu_hypre_IndexY(stride);
      frac  *= nalu_hypre_IndexZ(stride);
      scale += (nodeset_sizes[p] / frac);
   }
   /* REALLY Rough Estimate = num_nodes * nvar^3 */
   (relax_data -> flops) = scale * nvars * nvars * nvars *
                           nalu_hypre_StructVectorGlobalSize(
                              nalu_hypre_SStructPVectorSVector(x, 0) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelax(  void                 *relax_vdata,
                  nalu_hypre_SStructPMatrix *A,
                  nalu_hypre_SStructPVector *b,
                  nalu_hypre_SStructPVector *x           )
{
   nalu_hypre_NodeRelaxData  *relax_data        = (nalu_hypre_NodeRelaxData  *)relax_vdata;

   NALU_HYPRE_Int             max_iter          = (relax_data -> max_iter);
   NALU_HYPRE_Int             zero_guess        = (relax_data -> zero_guess);
   NALU_HYPRE_Real            weight            = (relax_data -> weight);
   NALU_HYPRE_Int             num_nodesets      = (relax_data -> num_nodesets);
   NALU_HYPRE_Int            *nodeset_ranks     = (relax_data -> nodeset_ranks);
   nalu_hypre_Index          *nodeset_strides   = (relax_data -> nodeset_strides);
   nalu_hypre_SStructPVector *t                 = (relax_data -> t);
   NALU_HYPRE_Int           **diag_rank         = (relax_data -> diag_rank);
   nalu_hypre_ComputePkg    **compute_pkgs      = (relax_data -> compute_pkgs);
   nalu_hypre_ComputePkg   ***svec_compute_pkgs = (relax_data ->svec_compute_pkgs);
   nalu_hypre_CommHandle    **comm_handle       = (relax_data -> comm_handle);

   nalu_hypre_ComputePkg     *compute_pkg;
   nalu_hypre_ComputePkg     *svec_compute_pkg;

   nalu_hypre_BoxArrayArray  *compute_box_aa;
   nalu_hypre_BoxArray       *compute_box_a;
   nalu_hypre_Box            *compute_box;
   nalu_hypre_Box            *A_data_box;
   nalu_hypre_Box            *b_data_box;
   nalu_hypre_Box            *x_data_box;
   nalu_hypre_Box            *t_data_box;

   /*
   NALU_HYPRE_Real           *tA_loc = (relax_data -> A_loc);
   NALU_HYPRE_Real           *tx_loc = (relax_data -> x_loc);
   */
   NALU_HYPRE_Real          **Ap = (relax_data -> Ap);
   NALU_HYPRE_Real          **bp = (relax_data -> bp);
   NALU_HYPRE_Real          **xp = (relax_data -> xp);
   NALU_HYPRE_Real          **tp = (relax_data -> tp);
   NALU_HYPRE_Real           *_h_Ap[NALU_HYPRE_MAXVARS * NALU_HYPRE_MAXVARS];
   NALU_HYPRE_Real           *_h_bp[NALU_HYPRE_MAXVARS];
   NALU_HYPRE_Real           *_h_xp[NALU_HYPRE_MAXVARS];
   NALU_HYPRE_Real           *_h_tp[NALU_HYPRE_MAXVARS];
   NALU_HYPRE_Real          **h_Ap;
   NALU_HYPRE_Real          **h_bp;
   NALU_HYPRE_Real          **h_xp;
   NALU_HYPRE_Real          **h_tp;

   NALU_HYPRE_MemoryLocation  memory_location = relax_data -> memory_location;

   /* Ap, bp, xp, tp are device pointers */
   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      h_Ap = _h_Ap;
      h_bp = _h_bp;
      h_xp = _h_xp;
      h_tp = _h_tp;
   }
   else
   {
      h_Ap = Ap;
      h_bp = bp;
      h_xp = xp;
      h_tp = tp;
   }

   nalu_hypre_StructMatrix    *A_block;
   nalu_hypre_StructVector    *x_block;

   nalu_hypre_IndexRef         stride;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            loop_size;

   nalu_hypre_StructStencil   *stencil;
   nalu_hypre_Index           *stencil_shape;
   NALU_HYPRE_Int              stencil_size;

   NALU_HYPRE_Int              iter, p, compute_i, i, j, si;
   NALU_HYPRE_Int              nodeset;

   NALU_HYPRE_Int              nvars, ndim;
   NALU_HYPRE_Int              vi, vj;

   /*----------------------------------------------------------
    * Initialize some things and deal with special cases
    *----------------------------------------------------------*/

   nalu_hypre_BeginTiming(relax_data -> time_index);

   nalu_hypre_SStructPMatrixDestroy(relax_data -> A);
   nalu_hypre_SStructPVectorDestroy(relax_data -> b);
   nalu_hypre_SStructPVectorDestroy(relax_data -> x);
   nalu_hypre_SStructPMatrixRef(A, &(relax_data -> A));
   nalu_hypre_SStructPVectorRef(x, &(relax_data -> x));
   nalu_hypre_SStructPVectorRef(b, &(relax_data -> b));

   (relax_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         nalu_hypre_SStructPVectorSetConstantValues(x, 0.0);
      }

      nalu_hypre_EndTiming(relax_data -> time_index);
      return nalu_hypre_error_flag;
   }

   /*----------------------------------------------------------
    * Do zero_guess iteration
    *----------------------------------------------------------*/

   p    = 0;
   iter = 0;

   nvars = nalu_hypre_SStructPMatrixNVars(relax_data -> A);
   ndim = nalu_hypre_SStructPMatrixNDim(relax_data -> A);

   if (zero_guess)
   {
      if (num_nodesets > 1)
      {
         nalu_hypre_SStructPVectorSetConstantValues(x, 0.0);
      }
      nodeset = nodeset_ranks[p];
      compute_pkg = compute_pkgs[nodeset];
      stride = nodeset_strides[nodeset];

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

            A_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(
                                              nalu_hypre_SStructPMatrixSMatrix(A, 0, 0)), i);
            b_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(
                                              nalu_hypre_SStructPVectorSVector(b, 0)), i);
            x_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(
                                              nalu_hypre_SStructPVectorSVector(x, 0)), i);

            for (vi = 0; vi < nvars; vi++)
            {
               for (vj = 0; vj < nvars; vj++)
               {
                  if (nalu_hypre_SStructPMatrixSMatrix(A, vi, vj) != NULL)
                  {
                     h_Ap[vi * nvars + vj] = nalu_hypre_StructMatrixBoxData( nalu_hypre_SStructPMatrixSMatrix(A, vi, vj),
                                                                        i, diag_rank[vi][vj] );
                  }
                  else
                  {
                     h_Ap[vi * nvars + vj] = NULL;
                  }
               }
               h_bp[vi] = nalu_hypre_StructVectorBoxData( nalu_hypre_SStructPVectorSVector(b, vi), i );
               h_xp[vi] = nalu_hypre_StructVectorBoxData( nalu_hypre_SStructPVectorSVector(x, vi), i );
            }

            if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
            {
               nalu_hypre_TMemcpy(Ap, h_Ap, NALU_HYPRE_Real *, nvars * nvars, memory_location, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TMemcpy(bp, h_bp, NALU_HYPRE_Real *, nvars, memory_location, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TMemcpy(xp, h_xp, NALU_HYPRE_Real *, nvars, memory_location, NALU_HYPRE_MEMORY_HOST);
            }

            nalu_hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

               start = nalu_hypre_BoxIMin(compute_box);
               nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(bp,Ap,xp)
               nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                   A_data_box, start, stride, Ai,
                                   b_data_box, start, stride, bi,
                                   x_data_box, start, stride, xi);
               {
                  NALU_HYPRE_Int vi, vj, err;
                  //NALU_HYPRE_Real *A_loc = tA_loc + nalu_hypre_BoxLoopBlock() * nvars * nvars;
                  //NALU_HYPRE_Real *x_loc = tx_loc + nalu_hypre_BoxLoopBlock() * nvars;
                  NALU_HYPRE_Real A_loc[NALU_HYPRE_MAXVARS * NALU_HYPRE_MAXVARS];
                  NALU_HYPRE_Real x_loc[NALU_HYPRE_MAXVARS];
                  /*------------------------------------------------
                   * Copy rhs and matrix for diagonal coupling
                   * (intra-nodal) into local storage.
                   *----------------------------------------------*/
                  for (vi = 0; vi < nvars; vi++)
                  {
                     NALU_HYPRE_Real *bpi = bp[vi];
                     x_loc[vi] = bpi[bi];
                     for (vj = 0; vj < nvars; vj++)
                     {
                        NALU_HYPRE_Real *Apij = Ap[vi * nvars + vj];
                        A_loc[vi * nvars + vj] = Apij ? Apij[Ai] : 0.0;
                     }
                  }

                  /*------------------------------------------------
                   * Invert intra-nodal coupling
                   *----------------------------------------------*/
                  nalu_hypre_gselim(A_loc, x_loc, nvars, err);
                  /*------------------------------------------------
                   * Copy solution from local storage.
                   *----------------------------------------------*/
                  for (vi = 0; vi < nvars; vi++)
                  {
                     NALU_HYPRE_Real *xpi = xp[vi];
                     xpi[xi] = x_loc[vi];
                  }
               }
               nalu_hypre_BoxLoop3End(Ai, bi, xi);
#undef DEVICE_VAR
            }
         }
      }

      if (weight != 1.0)
      {
         nalu_hypre_SStructPScale(weight, x);
      }

      p    = (p + 1) % num_nodesets;
      iter = iter + (p == 0);
   }

   /*----------------------------------------------------------
    * Do regular iterations
    *----------------------------------------------------------*/

   while (iter < max_iter)
   {
      nodeset = nodeset_ranks[p];
      compute_pkg = compute_pkgs[nodeset];
      stride = nodeset_strides[nodeset];

      nalu_hypre_SStructPCopy(x, t);

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               for (vi = 0; vi < nvars; vi++)
               {
                  x_block = nalu_hypre_SStructPVectorSVector(x, vi);
                  h_xp[vi] = nalu_hypre_StructVectorData(x_block);
                  svec_compute_pkg = svec_compute_pkgs[nodeset][vi];
                  nalu_hypre_InitializeIndtComputations(svec_compute_pkg,
                                                   h_xp[vi], &comm_handle[vi]);
               }
               compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               for (vi = 0; vi < nvars; vi++)
               {
                  nalu_hypre_FinalizeIndtComputations(comm_handle[vi]);
               }
               compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         nalu_hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_data_box = nalu_hypre_BoxArrayBox( nalu_hypre_StructMatrixDataSpace(
                                               nalu_hypre_SStructPMatrixSMatrix(A, 0, 0)), i );
            b_data_box = nalu_hypre_BoxArrayBox( nalu_hypre_StructVectorDataSpace(
                                               nalu_hypre_SStructPVectorSVector(b, 0)), i );
            x_data_box = nalu_hypre_BoxArrayBox( nalu_hypre_StructVectorDataSpace(
                                               nalu_hypre_SStructPVectorSVector(x, 0)), i );
            t_data_box = nalu_hypre_BoxArrayBox( nalu_hypre_StructVectorDataSpace(
                                               nalu_hypre_SStructPVectorSVector(t, 0)), i );

            for (vi = 0; vi < nvars; vi++)
            {
               h_bp[vi] = nalu_hypre_StructVectorBoxData( nalu_hypre_SStructPVectorSVector(b, vi), i );
               h_tp[vi] = nalu_hypre_StructVectorBoxData( nalu_hypre_SStructPVectorSVector(t, vi), i );
            }

            if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
            {
               nalu_hypre_TMemcpy(bp, h_bp, NALU_HYPRE_Real *, nvars, memory_location, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TMemcpy(tp, h_tp, NALU_HYPRE_Real *, nvars, memory_location, NALU_HYPRE_MEMORY_HOST);
            }

            nalu_hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

               start  = nalu_hypre_BoxIMin(compute_box);
               nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(tp,bp)
               nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                   b_data_box, start, stride, bi,
                                   t_data_box, start, stride, ti);
               {
                  NALU_HYPRE_Int vi;
                  /* Copy rhs into temp vector */
                  for (vi = 0; vi < nvars; vi++)
                  {
                     NALU_HYPRE_Real *tpi = tp[vi];
                     NALU_HYPRE_Real *bpi = bp[vi];
                     tpi[ti] = bpi[bi];
                  }
               }
               nalu_hypre_BoxLoop2End(bi, ti);
#undef DEVICE_VAR

               for (vi = 0; vi < nvars; vi++)
               {
                  for (vj = 0; vj < nvars; vj++)
                  {
                     if (nalu_hypre_SStructPMatrixSMatrix(A, vi, vj) != NULL)
                     {
                        A_block = nalu_hypre_SStructPMatrixSMatrix(A, vi, vj);
                        x_block = nalu_hypre_SStructPVectorSVector(x, vj);
                        stencil = nalu_hypre_StructMatrixStencil(A_block);
                        stencil_shape = nalu_hypre_StructStencilShape(stencil);
                        stencil_size  = nalu_hypre_StructStencilSize(stencil);
                        for (si = 0; si < stencil_size; si++)
                        {
                           if (si != diag_rank[vi][vj])
                           {
                              NALU_HYPRE_Real *Apij = nalu_hypre_StructMatrixBoxData(A_block, i, si);
                              NALU_HYPRE_Real *xpj  = nalu_hypre_StructVectorBoxData(x_block, i) +
                                                 nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[si]);
                              NALU_HYPRE_Real *tpi  = h_tp[vi];

#define DEVICE_VAR is_device_ptr(tpi,Apij,xpj)
                              nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                                  A_data_box, start, stride, Ai,
                                                  x_data_box, start, stride, xi,
                                                  t_data_box, start, stride, ti);
                              {
                                 tpi[ti] -= Apij[Ai] * xpj[xi];
                              }
                              nalu_hypre_BoxLoop3End(Ai, xi, ti);
#undef DEVICE_VAR
                           }
                        }
                     }
                  }
               }

               for (vi = 0; vi < nvars; vi++)
               {
                  for (vj = 0; vj < nvars; vj++)
                  {
                     if (nalu_hypre_SStructPMatrixSMatrix(A, vi, vj) != NULL)
                     {
                        h_Ap[vi * nvars + vj] = nalu_hypre_StructMatrixBoxData( nalu_hypre_SStructPMatrixSMatrix(A, vi, vj),
                                                                           i, diag_rank[vi][vj]);
                     }
                     else
                     {
                        h_Ap[vi * nvars + vj] = NULL;
                     }
                  }
               }

               if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
               {
                  nalu_hypre_TMemcpy(Ap, h_Ap, NALU_HYPRE_Real *, nvars * nvars, memory_location, NALU_HYPRE_MEMORY_HOST);
               }

#define DEVICE_VAR is_device_ptr(tp,Ap)
               nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                   A_data_box, start, stride, Ai,
                                   t_data_box, start, stride, ti);
               {
                  NALU_HYPRE_Int vi, vj, err;
                  /*
                  NALU_HYPRE_Real *A_loc = tA_loc + nalu_hypre_BoxLoopBlock() * nvars * nvars;
                  NALU_HYPRE_Real *x_loc = tx_loc + nalu_hypre_BoxLoopBlock() * nvars;
                  */
                  NALU_HYPRE_Real A_loc[NALU_HYPRE_MAXVARS * NALU_HYPRE_MAXVARS];
                  NALU_HYPRE_Real x_loc[NALU_HYPRE_MAXVARS];

                  /*------------------------------------------------
                   * Copy rhs and matrix for diagonal coupling
                   * (intra-nodal) into local storage.
                   *----------------------------------------------*/
                  for (vi = 0; vi < nvars; vi++)
                  {
                     NALU_HYPRE_Real *tpi = tp[vi];
                     x_loc[vi] = tpi[ti];
                     for (vj = 0; vj < nvars; vj++)
                     {
                        NALU_HYPRE_Real *Apij = Ap[vi * nvars + vj];
                        A_loc[vi * nvars + vj] = Apij ? Apij[Ai] : 0.0;
                     }
                  }

                  /*------------------------------------------------
                   * Invert intra-nodal coupling
                   *----------------------------------------------*/
                  nalu_hypre_gselim(A_loc, x_loc, nvars, err);
                  /*------------------------------------------------
                   * Copy solution from local storage.
                   *----------------------------------------------*/
                  for (vi = 0; vi < nvars; vi++)
                  {
                     NALU_HYPRE_Real *tpi = tp[vi];
                     tpi[ti] = x_loc[vi];
                  }

               }
               nalu_hypre_BoxLoop2End(Ai, ti);
#undef DEVICE_VAR
            }
         }
      }

      if (weight != 1.0)
      {
         nalu_hypre_SStructPScale((1.0 - weight), x);
         nalu_hypre_SStructPAxpy(weight, t, x);
      }
      else
      {
         nalu_hypre_SStructPCopy(t, x);
      }

      p    = (p + 1) % num_nodesets;
      iter = iter + (p == 0);
   }

   (relax_data -> num_iterations) = iter;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   nalu_hypre_IncFLOPCount(relax_data -> flops);
   nalu_hypre_EndTiming(relax_data -> time_index);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxSetTol( void   *relax_vdata,
                       NALU_HYPRE_Real  tol         )
{
   nalu_hypre_NodeRelaxData *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;

   (relax_data -> tol) = tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxSetMaxIter( void *relax_vdata,
                           NALU_HYPRE_Int   max_iter    )
{
   nalu_hypre_NodeRelaxData *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;

   (relax_data -> max_iter) = max_iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxSetZeroGuess( void *relax_vdata,
                             NALU_HYPRE_Int   zero_guess  )
{
   nalu_hypre_NodeRelaxData *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;

   (relax_data -> zero_guess) = zero_guess;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxSetWeight( void    *relax_vdata,
                          NALU_HYPRE_Real   weight      )
{
   nalu_hypre_NodeRelaxData *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;

   (relax_data -> weight) = weight;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxSetNumNodesets( void *relax_vdata,
                               NALU_HYPRE_Int   num_nodesets )
{
   nalu_hypre_NodeRelaxData *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;
   NALU_HYPRE_Int            i;

   /* free up old nodeset memory */
   for (i = 0; i < (relax_data -> num_nodesets); i++)
   {
      nalu_hypre_TFree(relax_data -> nodeset_indices[i], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(relax_data -> nodeset_sizes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(relax_data -> nodeset_ranks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(relax_data -> nodeset_strides, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(relax_data -> nodeset_indices, NALU_HYPRE_MEMORY_HOST);

   /* alloc new nodeset memory */
   (relax_data -> num_nodesets)    = num_nodesets;
   (relax_data -> nodeset_sizes)   = nalu_hypre_TAlloc(NALU_HYPRE_Int,     num_nodesets, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> nodeset_ranks)   = nalu_hypre_TAlloc(NALU_HYPRE_Int,     num_nodesets, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> nodeset_strides) = nalu_hypre_TAlloc(nalu_hypre_Index,   num_nodesets, NALU_HYPRE_MEMORY_HOST);
   (relax_data -> nodeset_indices) = nalu_hypre_TAlloc(nalu_hypre_Index *, num_nodesets, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_nodesets; i++)
   {
      (relax_data -> nodeset_sizes[i]) = 0;
      (relax_data -> nodeset_ranks[i]) = i;
      (relax_data -> nodeset_indices[i]) = NULL;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxSetNodeset( void        *relax_vdata,
                           NALU_HYPRE_Int    nodeset,
                           NALU_HYPRE_Int    nodeset_size,
                           nalu_hypre_Index  nodeset_stride,
                           nalu_hypre_Index *nodeset_indices )
{
   nalu_hypre_NodeRelaxData *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;
   NALU_HYPRE_Int            i;

   /* free up old nodeset memory */
   nalu_hypre_TFree(relax_data -> nodeset_indices[nodeset], NALU_HYPRE_MEMORY_HOST);

   /* alloc new nodeset memory */
   (relax_data -> nodeset_indices[nodeset]) =
      nalu_hypre_TAlloc(nalu_hypre_Index,  nodeset_size, NALU_HYPRE_MEMORY_HOST);

   (relax_data -> nodeset_sizes[nodeset]) = nodeset_size;
   nalu_hypre_CopyIndex(nodeset_stride,
                   (relax_data -> nodeset_strides[nodeset]));
   for (i = 0; i < nodeset_size; i++)
   {
      nalu_hypre_CopyIndex(nodeset_indices[i],
                      (relax_data -> nodeset_indices[nodeset][i]));
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxSetNodesetRank( void *relax_vdata,
                               NALU_HYPRE_Int   nodeset,
                               NALU_HYPRE_Int   nodeset_rank )
{
   nalu_hypre_NodeRelaxData *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;

   (relax_data -> nodeset_ranks[nodeset]) = nodeset_rank;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NodeRelaxSetTempVec( void                 *relax_vdata,
                           nalu_hypre_SStructPVector *t           )
{
   nalu_hypre_NodeRelaxData  *relax_data = (nalu_hypre_NodeRelaxData  *)relax_vdata;

   nalu_hypre_SStructPVectorDestroy(relax_data -> t);
   nalu_hypre_SStructPVectorRef(t, &(relax_data -> t));

   return nalu_hypre_error_flag;
}

