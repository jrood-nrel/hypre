/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef nalu_hypre_PAR_CSR_MATRIX_HEADER
#define nalu_hypre_PAR_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

#ifndef NALU_HYPRE_PAR_CSR_MATRIX_STRUCT
#define NALU_HYPRE_PAR_CSR_MATRIX_STRUCT
#endif

typedef struct nalu_hypre_ParCSRMatrix_struct
{
   MPI_Comm              comm;

   NALU_HYPRE_BigInt          global_num_rows;
   NALU_HYPRE_BigInt          global_num_cols;
   NALU_HYPRE_BigInt          global_num_rownnz;
   NALU_HYPRE_BigInt          first_row_index;
   NALU_HYPRE_BigInt          first_col_diag;
   /* need to know entire local range in case row_starts and col_starts
      are null  (i.e., bgl) AHB 6/05*/
   NALU_HYPRE_BigInt          last_row_index;
   NALU_HYPRE_BigInt          last_col_diag;

   nalu_hypre_CSRMatrix      *diag;
   nalu_hypre_CSRMatrix      *offd;
   nalu_hypre_CSRMatrix      *diagT, *offdT;
   /* JSP: transposed matrices are created lazily and optional */
   NALU_HYPRE_BigInt         *col_map_offd;
   NALU_HYPRE_BigInt         *device_col_map_offd;
   /* maps columns of offd to global columns */
   NALU_HYPRE_BigInt          row_starts[2];
   /* row_starts[0] is start of local rows
      row_starts[1] is start of next processor's rows */
   NALU_HYPRE_BigInt          col_starts[2];
   /* col_starts[0] is start of local columns
      col_starts[1] is start of next processor's columns */

   nalu_hypre_ParCSRCommPkg  *comm_pkg;
   nalu_hypre_ParCSRCommPkg  *comm_pkgT;

   /* Does the ParCSRMatrix create/destroy `diag', `offd', `col_map_offd'? */
   NALU_HYPRE_Int             owns_data;

   NALU_HYPRE_BigInt          num_nonzeros;
   NALU_HYPRE_Real            d_num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   NALU_HYPRE_BigInt         *rowindices;
   NALU_HYPRE_Complex        *rowvalues;
   NALU_HYPRE_Int             getrowactive;

   nalu_hypre_IJAssumedPart  *assumed_partition;
   NALU_HYPRE_Int             owns_assumed_partition;
   /* Array to store ordering of local diagonal block to relax. In particular,
   used for triangulr matrices that are not ordered to be triangular. */
   NALU_HYPRE_Int            *proc_ordering;

   /* Save block diagonal inverse */
   NALU_HYPRE_Int             bdiag_size;
   NALU_HYPRE_Complex        *bdiaginv;
   nalu_hypre_ParCSRCommPkg  *bdiaginv_comm_pkg;

#if defined(NALU_HYPRE_USING_GPU)
   /* these two arrays are reserveed for SoC matrices on GPUs to help build interpolation */
   NALU_HYPRE_Int            *soc_diag_j;
   NALU_HYPRE_Int            *soc_offd_j;
#endif

} nalu_hypre_ParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ParCSRMatrixComm(matrix)                   ((matrix) -> comm)
#define nalu_hypre_ParCSRMatrixGlobalNumRows(matrix)          ((matrix) -> global_num_rows)
#define nalu_hypre_ParCSRMatrixGlobalNumCols(matrix)          ((matrix) -> global_num_cols)
#define nalu_hypre_ParCSRMatrixGlobalNumRownnz(matrix)        ((matrix) -> global_num_rownnz)
#define nalu_hypre_ParCSRMatrixFirstRowIndex(matrix)          ((matrix) -> first_row_index)
#define nalu_hypre_ParCSRMatrixFirstColDiag(matrix)           ((matrix) -> first_col_diag)
#define nalu_hypre_ParCSRMatrixLastRowIndex(matrix)           ((matrix) -> last_row_index)
#define nalu_hypre_ParCSRMatrixLastColDiag(matrix)            ((matrix) -> last_col_diag)
#define nalu_hypre_ParCSRMatrixDiag(matrix)                   ((matrix) -> diag)
#define nalu_hypre_ParCSRMatrixOffd(matrix)                   ((matrix) -> offd)
#define nalu_hypre_ParCSRMatrixDiagT(matrix)                  ((matrix) -> diagT)
#define nalu_hypre_ParCSRMatrixOffdT(matrix)                  ((matrix) -> offdT)
#define nalu_hypre_ParCSRMatrixColMapOffd(matrix)             ((matrix) -> col_map_offd)
#define nalu_hypre_ParCSRMatrixDeviceColMapOffd(matrix)       ((matrix) -> device_col_map_offd)
#define nalu_hypre_ParCSRMatrixRowStarts(matrix)              ((matrix) -> row_starts)
#define nalu_hypre_ParCSRMatrixColStarts(matrix)              ((matrix) -> col_starts)
#define nalu_hypre_ParCSRMatrixCommPkg(matrix)                ((matrix) -> comm_pkg)
#define nalu_hypre_ParCSRMatrixCommPkgT(matrix)               ((matrix) -> comm_pkgT)
#define nalu_hypre_ParCSRMatrixOwnsData(matrix)               ((matrix) -> owns_data)
#define nalu_hypre_ParCSRMatrixNumNonzeros(matrix)            ((matrix) -> num_nonzeros)
#define nalu_hypre_ParCSRMatrixDNumNonzeros(matrix)           ((matrix) -> d_num_nonzeros)
#define nalu_hypre_ParCSRMatrixRowindices(matrix)             ((matrix) -> rowindices)
#define nalu_hypre_ParCSRMatrixRowvalues(matrix)              ((matrix) -> rowvalues)
#define nalu_hypre_ParCSRMatrixGetrowactive(matrix)           ((matrix) -> getrowactive)
#define nalu_hypre_ParCSRMatrixAssumedPartition(matrix)       ((matrix) -> assumed_partition)
#define nalu_hypre_ParCSRMatrixOwnsAssumedPartition(matrix)   ((matrix) -> owns_assumed_partition)
#define nalu_hypre_ParCSRMatrixProcOrdering(matrix)           ((matrix) -> proc_ordering)
#if defined(NALU_HYPRE_USING_GPU)
#define nalu_hypre_ParCSRMatrixSocDiagJ(matrix)               ((matrix) -> soc_diag_j)
#define nalu_hypre_ParCSRMatrixSocOffdJ(matrix)               ((matrix) -> soc_offd_j)
#endif

#define nalu_hypre_ParCSRMatrixNumRows(matrix) nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(matrix))
#define nalu_hypre_ParCSRMatrixNumCols(matrix) nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixDiag(matrix))

static inline NALU_HYPRE_MemoryLocation
nalu_hypre_ParCSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrix *matrix)
{
   if (!matrix) { return NALU_HYPRE_MEMORY_UNDEFINED; }

   nalu_hypre_CSRMatrix *diag = nalu_hypre_ParCSRMatrixDiag(matrix);
   nalu_hypre_CSRMatrix *offd = nalu_hypre_ParCSRMatrixOffd(matrix);
   NALU_HYPRE_MemoryLocation memory_diag = diag ? nalu_hypre_CSRMatrixMemoryLocation(
                                         diag) : NALU_HYPRE_MEMORY_UNDEFINED;
   NALU_HYPRE_MemoryLocation memory_offd = offd ? nalu_hypre_CSRMatrixMemoryLocation(
                                         offd) : NALU_HYPRE_MEMORY_UNDEFINED;

   if (diag && offd)
   {
      if (memory_diag != memory_offd)
      {
         char err_msg[1024];
         nalu_hypre_sprintf(err_msg, "Error: ParCSRMatrix Memory Location Diag (%d) != Offd (%d)\n", memory_diag,
                       memory_offd);
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_MEMORY, err_msg);
         nalu_hypre_assert(0);

         return NALU_HYPRE_MEMORY_UNDEFINED;
      }

      return memory_diag;
   }

   if (diag) { return memory_diag; }
   if (offd) { return memory_offd; }

   return NALU_HYPRE_MEMORY_UNDEFINED;
}

/*--------------------------------------------------------------------------
 * Parallel CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
   NALU_HYPRE_BigInt            global_num_rows;
   NALU_HYPRE_BigInt            global_num_cols;
   NALU_HYPRE_BigInt            first_row_index;
   NALU_HYPRE_BigInt            first_col_diag;
   NALU_HYPRE_BigInt            last_row_index;
   NALU_HYPRE_BigInt            last_col_diag;
   nalu_hypre_CSRBooleanMatrix *diag;
   nalu_hypre_CSRBooleanMatrix *offd;
   NALU_HYPRE_BigInt           *col_map_offd;
   NALU_HYPRE_BigInt           *row_starts;
   NALU_HYPRE_BigInt           *col_starts;
   nalu_hypre_ParCSRCommPkg    *comm_pkg;
   nalu_hypre_ParCSRCommPkg    *comm_pkgT;
   NALU_HYPRE_Int               owns_data;
   NALU_HYPRE_Int               owns_row_starts;
   NALU_HYPRE_Int               owns_col_starts;
   NALU_HYPRE_BigInt            num_nonzeros;
   NALU_HYPRE_BigInt           *rowindices;
   NALU_HYPRE_Int               getrowactive;

} nalu_hypre_ParCSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ParCSRBooleanMatrix_Get_Comm(matrix)          ((matrix)->comm)
#define nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix)   ((matrix)->global_num_rows)
#define nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix)   ((matrix)->global_num_cols)
#define nalu_hypre_ParCSRBooleanMatrix_Get_StartRow(matrix)      ((matrix)->first_row_index)
#define nalu_hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(matrix) ((matrix)->first_row_index)
#define nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(matrix)  ((matrix)->first_col_diag)
#define nalu_hypre_ParCSRBooleanMatrix_Get_LastRowIndex(matrix)  ((matrix)->last_row_index)
#define nalu_hypre_ParCSRBooleanMatrix_Get_LastColDiag(matrix)   ((matrix)->last_col_diag)
#define nalu_hypre_ParCSRBooleanMatrix_Get_Diag(matrix)          ((matrix)->diag)
#define nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix)          ((matrix)->offd)
#define nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix)    ((matrix)->col_map_offd)
#define nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix)     ((matrix)->row_starts)
#define nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix)     ((matrix)->col_starts)
#define nalu_hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix)       ((matrix)->comm_pkg)
#define nalu_hypre_ParCSRBooleanMatrix_Get_CommPkgT(matrix)      ((matrix)->comm_pkgT)
#define nalu_hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix)      ((matrix)->owns_data)
#define nalu_hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) ((matrix)->owns_row_starts)
#define nalu_hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) ((matrix)->owns_col_starts)
#define nalu_hypre_ParCSRBooleanMatrix_Get_NRows(matrix)         ((matrix->diag->num_rows))
#define nalu_hypre_ParCSRBooleanMatrix_Get_NCols(matrix)         ((matrix->diag->num_cols))
#define nalu_hypre_ParCSRBooleanMatrix_Get_NNZ(matrix)           ((matrix)->num_nonzeros)
#define nalu_hypre_ParCSRBooleanMatrix_Get_Rowindices(matrix)    ((matrix)->rowindices)
#define nalu_hypre_ParCSRBooleanMatrix_Get_Getrowactive(matrix)  ((matrix)->getrowactive)

#endif
