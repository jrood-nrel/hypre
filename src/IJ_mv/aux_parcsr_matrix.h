/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Auxiliary Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef nalu_hypre_AUX_PARCSR_MATRIX_HEADER
#define nalu_hypre_AUX_PARCSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int            local_num_rows;    /* defines number of rows on this processor */
   NALU_HYPRE_Int            local_num_rownnz;  /* defines number of nonzero rows on this processor */
   NALU_HYPRE_Int            local_num_cols;    /* defines number of cols of diag */

   NALU_HYPRE_Int            need_aux;                /* if need_aux = 1, aux_j, aux_data are used to
                                                    generate the parcsr matrix (default),
                                                    for need_aux = 0, data is put directly into
                                                    parcsr structure (requires the knowledge of
                                                    offd_i and diag_i ) */

   NALU_HYPRE_Int           *rownnz;                  /* row_nnz[i] contains the i-th nonzero row id */
   NALU_HYPRE_Int           *row_length;              /* row_length[i] contains number of stored
                                                    elements in i-th row */
   NALU_HYPRE_Int           *row_space;               /* row_space[i] contains space allocated to
                                                    i-th row */

   NALU_HYPRE_Int           *diag_sizes;              /* user input row lengths of diag */
   NALU_HYPRE_Int           *offd_sizes;              /* user input row lengths of diag */

   NALU_HYPRE_BigInt       **aux_j;                   /* contains collected column indices */
   NALU_HYPRE_Complex      **aux_data;                /* contains collected data */

   NALU_HYPRE_Int           *indx_diag;               /* indx_diag[i] points to first empty space of portion
                                                    in diag_j , diag_data assigned to row i */
   NALU_HYPRE_Int           *indx_offd;               /* indx_offd[i] points to first empty space of portion
                                                    in offd_j , offd_data assigned to row i */

   NALU_HYPRE_Int            max_off_proc_elmts;      /* length of off processor stash set for
                                                    SetValues and AddTOValues */
   NALU_HYPRE_Int            current_off_proc_elmts;  /* current no. of elements stored in stash */
   NALU_HYPRE_Int            off_proc_i_indx;         /* pointer to first empty space in
                                                    set_off_proc_i_set */
   NALU_HYPRE_BigInt        *off_proc_i;              /* length 2*num_off_procs_elmts, contains info pairs
                                                    (code, no. of elmts) where code contains global
                                                    row no. if  SetValues, and (-global row no. -1)
                                                    if  AddToValues */
   NALU_HYPRE_BigInt        *off_proc_j;              /* contains column indices
                                                  * ( global col id.)    if SetValues,
                                                  * (-global col id. -1) if AddToValues */
   NALU_HYPRE_Complex       *off_proc_data;           /* contains corresponding data */

   NALU_HYPRE_MemoryLocation memory_location;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_BigInt         max_stack_elmts;
   NALU_HYPRE_BigInt         current_stack_elmts;
   NALU_HYPRE_BigInt        *stack_i;
   NALU_HYPRE_BigInt        *stack_j;
   NALU_HYPRE_Complex       *stack_data;
   char                *stack_sora;              /* Set (1) or Add (0) */
   NALU_HYPRE_Int            usr_on_proc_elmts;       /* user given num elmt on-proc */
   NALU_HYPRE_Int            usr_off_proc_elmts;      /* user given num elmt off-proc */
   NALU_HYPRE_BigInt         init_alloc_factor;
   NALU_HYPRE_BigInt         grow_factor;
#endif
} nalu_hypre_AuxParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_AuxParCSRMatrixLocalNumRows(matrix)         ((matrix) -> local_num_rows)
#define nalu_hypre_AuxParCSRMatrixLocalNumRownnz(matrix)       ((matrix) -> local_num_rownnz)
#define nalu_hypre_AuxParCSRMatrixLocalNumCols(matrix)         ((matrix) -> local_num_cols)

#define nalu_hypre_AuxParCSRMatrixNeedAux(matrix)              ((matrix) -> need_aux)
#define nalu_hypre_AuxParCSRMatrixRownnz(matrix)               ((matrix) -> rownnz)
#define nalu_hypre_AuxParCSRMatrixRowLength(matrix)            ((matrix) -> row_length)
#define nalu_hypre_AuxParCSRMatrixRowSpace(matrix)             ((matrix) -> row_space)
#define nalu_hypre_AuxParCSRMatrixAuxJ(matrix)                 ((matrix) -> aux_j)
#define nalu_hypre_AuxParCSRMatrixAuxData(matrix)              ((matrix) -> aux_data)

#define nalu_hypre_AuxParCSRMatrixIndxDiag(matrix)             ((matrix) -> indx_diag)
#define nalu_hypre_AuxParCSRMatrixIndxOffd(matrix)             ((matrix) -> indx_offd)

#define nalu_hypre_AuxParCSRMatrixDiagSizes(matrix)            ((matrix) -> diag_sizes)
#define nalu_hypre_AuxParCSRMatrixOffdSizes(matrix)            ((matrix) -> offd_sizes)

#define nalu_hypre_AuxParCSRMatrixMaxOffProcElmts(matrix)      ((matrix) -> max_off_proc_elmts)
#define nalu_hypre_AuxParCSRMatrixCurrentOffProcElmts(matrix)  ((matrix) -> current_off_proc_elmts)
#define nalu_hypre_AuxParCSRMatrixOffProcIIndx(matrix)         ((matrix) -> off_proc_i_indx)
#define nalu_hypre_AuxParCSRMatrixOffProcI(matrix)             ((matrix) -> off_proc_i)
#define nalu_hypre_AuxParCSRMatrixOffProcJ(matrix)             ((matrix) -> off_proc_j)
#define nalu_hypre_AuxParCSRMatrixOffProcData(matrix)          ((matrix) -> off_proc_data)

#define nalu_hypre_AuxParCSRMatrixMemoryLocation(matrix)       ((matrix) -> memory_location)

#if defined(NALU_HYPRE_USING_GPU)
#define nalu_hypre_AuxParCSRMatrixMaxStackElmts(matrix)        ((matrix) -> max_stack_elmts)
#define nalu_hypre_AuxParCSRMatrixCurrentStackElmts(matrix)    ((matrix) -> current_stack_elmts)
#define nalu_hypre_AuxParCSRMatrixStackI(matrix)               ((matrix) -> stack_i)
#define nalu_hypre_AuxParCSRMatrixStackJ(matrix)               ((matrix) -> stack_j)
#define nalu_hypre_AuxParCSRMatrixStackData(matrix)            ((matrix) -> stack_data)
#define nalu_hypre_AuxParCSRMatrixStackSorA(matrix)            ((matrix) -> stack_sora)
#define nalu_hypre_AuxParCSRMatrixUsrOnProcElmts(matrix)       ((matrix) -> usr_on_proc_elmts)
#define nalu_hypre_AuxParCSRMatrixUsrOffProcElmts(matrix)      ((matrix) -> usr_off_proc_elmts)
#define nalu_hypre_AuxParCSRMatrixInitAllocFactor(matrix)      ((matrix) -> init_alloc_factor)
#define nalu_hypre_AuxParCSRMatrixGrowFactor(matrix)           ((matrix) -> grow_factor)
#endif

#endif /* #ifndef nalu_hypre_AUX_PARCSR_MATRIX_HEADER */
