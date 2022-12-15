/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Auxiliary Parallel Vector data structures
 *
 * Note: this vector currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef nalu_hypre_AUX_PAR_VECTOR_HEADER
#define nalu_hypre_AUX_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int            max_off_proc_elmts;      /* length of off processor stash for
                                                    SetValues and AddToValues*/
   NALU_HYPRE_Int            current_off_proc_elmts;  /* current no. of elements stored in stash */
   NALU_HYPRE_BigInt        *off_proc_i;              /* contains column indices */
   NALU_HYPRE_Complex       *off_proc_data;           /* contains corresponding data */

   NALU_HYPRE_MemoryLocation memory_location;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int            max_stack_elmts;      /* length of stash for SetValues and AddToValues*/
   NALU_HYPRE_Int            current_stack_elmts;  /* current no. of elements stored in stash */
   NALU_HYPRE_BigInt        *stack_i;              /* contains row indices */
   NALU_HYPRE_BigInt        *stack_voff;           /* contains vector offsets for multivectors */
   NALU_HYPRE_Complex       *stack_data;           /* contains corresponding data */
   char                *stack_sora;
   NALU_HYPRE_Int            usr_off_proc_elmts;   /* the num of off-proc elements usr guided */
   NALU_HYPRE_Real           init_alloc_factor;
   NALU_HYPRE_Real           grow_factor;
#endif
} nalu_hypre_AuxParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel Vector structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_AuxParVectorMaxOffProcElmts(vector)      ((vector) -> max_off_proc_elmts)
#define nalu_hypre_AuxParVectorCurrentOffProcElmts(vector)  ((vector) -> current_off_proc_elmts)
#define nalu_hypre_AuxParVectorOffProcI(vector)             ((vector) -> off_proc_i)
#define nalu_hypre_AuxParVectorOffProcData(vector)          ((vector) -> off_proc_data)

#define nalu_hypre_AuxParVectorMemoryLocation(vector)       ((vector) -> memory_location)

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
#define nalu_hypre_AuxParVectorMaxStackElmts(vector)        ((vector) -> max_stack_elmts)
#define nalu_hypre_AuxParVectorCurrentStackElmts(vector)    ((vector) -> current_stack_elmts)
#define nalu_hypre_AuxParVectorStackI(vector)               ((vector) -> stack_i)
#define nalu_hypre_AuxParVectorStackVoff(vector)            ((vector) -> stack_voff)
#define nalu_hypre_AuxParVectorStackData(vector)            ((vector) -> stack_data)
#define nalu_hypre_AuxParVectorStackSorA(vector)            ((vector) -> stack_sora)
#define nalu_hypre_AuxParVectorUsrOffProcElmts(vector)      ((vector) -> usr_off_proc_elmts)
#define nalu_hypre_AuxParVectorInitAllocFactor(vector)      ((vector) -> init_alloc_factor)
#define nalu_hypre_AuxParVectorGrowFactor(vector)           ((vector) -> grow_factor)
#endif

#endif /* #ifndef nalu_hypre_AUX_PAR_VECTOR_HEADER */
