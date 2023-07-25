/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the SMG solver
 *
 *****************************************************************************/

#ifndef nalu_hypre_SMG_HEADER
#define nalu_hypre_SMG_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_SMGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   NALU_HYPRE_Int             memory_use;
   NALU_HYPRE_Real            tol;
   NALU_HYPRE_Int             max_iter;
   NALU_HYPRE_Int             rel_change;
   NALU_HYPRE_Int             zero_guess;
   NALU_HYPRE_Int             max_levels;  /* max_level <= 0 means no limit */

   NALU_HYPRE_Int             num_levels;

   NALU_HYPRE_Int             num_pre_relax;  /* number of pre relaxation sweeps */
   NALU_HYPRE_Int             num_post_relax; /* number of post relaxation sweeps */

   NALU_HYPRE_Int             cdir;  /* coarsening direction */

   /* base index space info */
   nalu_hypre_Index           base_index;
   nalu_hypre_Index           base_stride;

   nalu_hypre_StructGrid    **grid_l;
   nalu_hypre_StructGrid    **PT_grid_l;

   NALU_HYPRE_MemoryLocation  memory_location; /* memory location of data */
   NALU_HYPRE_Real           *data;
   NALU_HYPRE_Real           *data_const;
   nalu_hypre_StructMatrix  **A_l;
   nalu_hypre_StructMatrix  **PT_l;
   nalu_hypre_StructMatrix  **R_l;
   nalu_hypre_StructVector  **b_l;
   nalu_hypre_StructVector  **x_l;

   /* temp vectors */
   nalu_hypre_StructVector  **tb_l;
   nalu_hypre_StructVector  **tx_l;
   nalu_hypre_StructVector  **r_l;
   nalu_hypre_StructVector  **e_l;

   void                **relax_data_l;
   void                **residual_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   /* log info (always logged) */
   NALU_HYPRE_Int             num_iterations;
   NALU_HYPRE_Int             time_index;

   NALU_HYPRE_Int             print_level;

   /* additional log info (logged when `logging' > 0) */
   NALU_HYPRE_Int             logging;
   NALU_HYPRE_Real           *norms;
   NALU_HYPRE_Real           *rel_norms;
} nalu_hypre_SMGData;

/*--------------------------------------------------------------------------
 * Utility routines:
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SMGSetBIndex(base_index, base_stride, level, bindex) \
{\
   if (level > 0)\
      nalu_hypre_SetIndex3(bindex, 0, 0, 0);\
   else\
      nalu_hypre_CopyIndex(base_index, bindex);\
}

#define nalu_hypre_SMGSetBStride(base_index, base_stride, level, bstride) \
{\
   if (level > 0)\
      nalu_hypre_SetIndex3(bstride, 1, 1, 1);\
   else\
      nalu_hypre_CopyIndex(base_stride, bstride);\
}

#define nalu_hypre_SMGSetCIndex(base_index, base_stride, level, cdir, cindex) \
{\
   nalu_hypre_SMGSetBIndex(base_index, base_stride, level, cindex);\
   nalu_hypre_IndexD(cindex, cdir) += 0;\
}

#define nalu_hypre_SMGSetFIndex(base_index, base_stride, level, cdir, findex) \
{\
   nalu_hypre_SMGSetBIndex(base_index, base_stride, level, findex);\
   nalu_hypre_IndexD(findex, cdir) += 1;\
}

#define nalu_hypre_SMGSetStride(base_index, base_stride, level, cdir, stride) \
{\
   nalu_hypre_SMGSetBStride(base_index, base_stride, level, stride);\
   nalu_hypre_IndexD(stride, cdir) *= 2;\
}

#endif
