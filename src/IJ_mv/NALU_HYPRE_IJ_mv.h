/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_IJ_MV_HEADER
#define NALU_HYPRE_IJ_MV_HEADER

#include "NALU_HYPRE_config.h"
#include "NALU_HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup IJSystemInterface IJ System Interface
 *
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 *
 * @memo A linear-algebraic conceptual interface
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Matrices
 *
 * @{
 **/

struct nalu_hypre_IJMatrix_struct;
/**
 * The matrix object.
 **/
typedef struct nalu_hypre_IJMatrix_struct *NALU_HYPRE_IJMatrix;

/**
 * Create a matrix object.  Each process owns some unique consecutive
 * range of rows, indicated by the global row indices \e ilower and
 * \e iupper.  The row data is required to be such that the value
 * of \e ilower on any process \f$p\f$ be exactly one more than the
 * value of \e iupper on process \f$p-1\f$.  Note that the first row of
 * the global matrix may start with any integer value.  In particular,
 * one may use zero- or one-based indexing.
 *
 * For square matrices, \e jlower and \e jupper typically should
 * match \e ilower and \e iupper, respectively.  For rectangular
 * matrices, \e jlower and \e jupper should define a
 * partitioning of the columns.  This partitioning must be used for
 * any vector \f$v\f$ that will be used in matrix-vector products with the
 * rectangular matrix.  The matrix data structure may use \e jlower
 * and \e jupper to store the diagonal blocks (rectangular in
 * general) of the matrix separately from the rest of the matrix.
 *
 * Collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixCreate(MPI_Comm        comm,
                               NALU_HYPRE_BigInt    ilower,
                               NALU_HYPRE_BigInt    iupper,
                               NALU_HYPRE_BigInt    jlower,
                               NALU_HYPRE_BigInt    jupper,
                               NALU_HYPRE_IJMatrix *matrix);

/**
 * Destroy a matrix object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixDestroy(NALU_HYPRE_IJMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.  This
 * routine will also re-initialize an already assembled matrix,
 * allowing users to modify coefficient values.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixInitialize(NALU_HYPRE_IJMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.  This
 * routine will also re-initialize an already assembled matrix,
 * allowing users to modify coefficient values. This routine
 * also specifies the memory location, i.e. host or device.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixInitialize_v2(NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_MemoryLocation memory_location);

/**
 * Sets values for \e nrows rows or partial rows of the matrix.
 * The arrays \e ncols
 * and \e rows are of dimension \e nrows and contain the number
 * of columns in each row and the row indices, respectively.  The
 * array \e cols contains the column indices for each of the \e
 * rows, and is ordered by rows.  The data in the \e values array
 * corresponds directly to the column entries in \e cols.  Erases
 * any previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before, inserts a
 * new one if set locally. Note that it is not possible to set values
 * on other processors. If one tries to set a value from proc i on proc j,
 * proc i will erase all previous occurrences of this value in its stack
 * (including values generated with AddToValues), and treat it like
 * a zero value. The actual value needs to be set on proc j.
 *
 * Note that a threaded version (threaded over the number of rows)
 * will be called if
 * NALU_HYPRE_IJMatrixSetOMPFlag is set to a value != 0.
 * This requires that rows[i] != rows[j] for i!= j
 * and is only efficient if a large number of rows is set in one call
 * to NALU_HYPRE_IJMatrixSetValues.
 *
 * Not collective.
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetValues(NALU_HYPRE_IJMatrix       matrix,
                                  NALU_HYPRE_Int            nrows,
                                  NALU_HYPRE_Int           *ncols,
                                  const NALU_HYPRE_BigInt  *rows,
                                  const NALU_HYPRE_BigInt  *cols,
                                  const NALU_HYPRE_Complex *values);

/**
 * Sets all  matrix coefficients of an already assembled matrix to
 * \e value
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetConstantValues(NALU_HYPRE_IJMatrix matrix,
                                          NALU_HYPRE_Complex value);

/**
 * Adds to values for \e nrows rows or partial rows of the matrix.
 * Usage details are analogous to \ref NALU_HYPRE_IJMatrixSetValues.
 * Adds to any previous values at the specified locations, or, if
 * there was no value there before, inserts a new one.
 * AddToValues can be used to add to values on other processors.
 *
 * Note that a threaded version (threaded over the number of rows)
 * will be called if
 * NALU_HYPRE_IJMatrixSetOMPFlag is set to a value != 0.
 * This requires that rows[i] != rows[j] for i!= j
 * and is only efficient if a large number of rows is added in one call
 * to NALU_HYPRE_IJMatrixAddToValues.
 *
 * Not collective.
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixAddToValues(NALU_HYPRE_IJMatrix       matrix,
                                    NALU_HYPRE_Int            nrows,
                                    NALU_HYPRE_Int           *ncols,
                                    const NALU_HYPRE_BigInt  *rows,
                                    const NALU_HYPRE_BigInt  *cols,
                                    const NALU_HYPRE_Complex *values);

/**
 * Sets values for \e nrows rows or partial rows of the matrix.
 *
 * Same as IJMatrixSetValues, but with an additional \e row_indexes array
 * that provides indexes into the \e cols and \e values arrays.  Because
 * of this, there can be gaps between the row data in these latter two arrays.
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetValues2(NALU_HYPRE_IJMatrix       matrix,
                                   NALU_HYPRE_Int            nrows,
                                   NALU_HYPRE_Int           *ncols,
                                   const NALU_HYPRE_BigInt  *rows,
                                   const NALU_HYPRE_Int     *row_indexes,
                                   const NALU_HYPRE_BigInt  *cols,
                                   const NALU_HYPRE_Complex *values);

/**
 * Adds to values for \e nrows rows or partial rows of the matrix.
 *
 * Same as IJMatrixAddToValues, but with an additional \e row_indexes array
 * that provides indexes into the \e cols and \e values arrays.  Because
 * of this, there can be gaps between the row data in these latter two arrays.
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixAddToValues2(NALU_HYPRE_IJMatrix       matrix,
                                     NALU_HYPRE_Int            nrows,
                                     NALU_HYPRE_Int           *ncols,
                                     const NALU_HYPRE_BigInt  *rows,
                                     const NALU_HYPRE_Int     *row_indexes,
                                     const NALU_HYPRE_BigInt  *cols,
                                     const NALU_HYPRE_Complex *values);

/**
 * Finalize the construction of the matrix before using.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixAssemble(NALU_HYPRE_IJMatrix matrix);

/**
 * Gets number of nonzeros elements for \e nrows rows specified in \e rows
 * and returns them in \e ncols, which needs to be allocated by the
 * user.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetRowCounts(NALU_HYPRE_IJMatrix  matrix,
                                     NALU_HYPRE_Int       nrows,
                                     NALU_HYPRE_BigInt   *rows,
                                     NALU_HYPRE_Int      *ncols);

/**
 * Gets values for \e nrows rows or partial rows of the matrix.
 * Usage details are mostly
 * analogous to \ref NALU_HYPRE_IJMatrixSetValues.
 * Note that if nrows is negative, the routine will return
 * the column_indices and matrix coefficients of the
 * (-nrows) rows contained in rows.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetValues(NALU_HYPRE_IJMatrix  matrix,
                                  NALU_HYPRE_Int       nrows,
                                  NALU_HYPRE_Int      *ncols,
                                  NALU_HYPRE_BigInt   *rows,
                                  NALU_HYPRE_BigInt   *cols,
                                  NALU_HYPRE_Complex  *values);

/**
 * Set the storage type of the matrix object to be constructed.
 * Currently, \e type can only be \c NALU_HYPRE_PARCSR.
 *
 * Not collective, but must be the same on all processes.
 *
 * @see NALU_HYPRE_IJMatrixGetObject
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetObjectType(NALU_HYPRE_IJMatrix matrix,
                                      NALU_HYPRE_Int      type);

/**
 * Get the storage type of the constructed matrix object.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetObjectType(NALU_HYPRE_IJMatrix  matrix,
                                      NALU_HYPRE_Int      *type);

/**
 * Gets range of rows owned by this processor and range
 * of column partitioning for this processor.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetLocalRange(NALU_HYPRE_IJMatrix  matrix,
                                      NALU_HYPRE_BigInt   *ilower,
                                      NALU_HYPRE_BigInt   *iupper,
                                      NALU_HYPRE_BigInt   *jlower,
                                      NALU_HYPRE_BigInt   *jupper);

/**
 * Get a reference to the constructed matrix object.
 *
 * @see NALU_HYPRE_IJMatrixSetObjectType
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixGetObject(NALU_HYPRE_IJMatrix   matrix,
                                  void           **object);

/**
 * (Optional) Set the max number of nonzeros to expect in each row.
 * The array \e sizes contains estimated sizes for each row on this
 * process.  This call can significantly improve the efficiency of
 * matrix construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetRowSizes(NALU_HYPRE_IJMatrix   matrix,
                                    const NALU_HYPRE_Int *sizes);

/**
 * (Optional) Sets the exact number of nonzeros in each row of
 * the diagonal and off-diagonal blocks.  The diagonal block is the
 * submatrix whose column numbers correspond to rows owned by this
 * process, and the off-diagonal block is everything else.  The arrays
 * \e diag_sizes and \e offdiag_sizes contain estimated sizes
 * for each row of the diagonal and off-diagonal blocks, respectively.
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetDiagOffdSizes(NALU_HYPRE_IJMatrix   matrix,
                                         const NALU_HYPRE_Int *diag_sizes,
                                         const NALU_HYPRE_Int *offdiag_sizes);

/**
 * (Optional) Sets the maximum number of elements that are expected to be set
 * (or added) on other processors from this processor
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetMaxOffProcElmts(NALU_HYPRE_IJMatrix matrix,
                                           NALU_HYPRE_Int      max_off_proc_elmts);

/**
 * (Optional) Sets the print level, if the user wants to print
 * error messages. The default is 0, i.e. no error messages are printed.
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetPrintLevel(NALU_HYPRE_IJMatrix matrix,
                                      NALU_HYPRE_Int      print_level);

/**
 * (Optional) if set, will use a threaded version of
 * NALU_HYPRE_IJMatrixSetValues and NALU_HYPRE_IJMatrixAddToValues.
 * This is only useful if a large number of rows is set or added to
 * at once.
 *
 * NOTE that the values in the rows array of NALU_HYPRE_IJMatrixSetValues
 * or NALU_HYPRE_IJMatrixAddToValues must be different from each other !!!
 *
 * This option is VERY inefficient if only a small number of rows
 * is set or added at once and/or
 * if reallocation of storage is required and/or
 * if values are added to off processor values.
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixSetOMPFlag(NALU_HYPRE_IJMatrix matrix,
                                   NALU_HYPRE_Int      omp_flag);

/**
 * Read the matrix from file.  This is mainly for debugging purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixRead(const char     *filename,
                             MPI_Comm        comm,
                             NALU_HYPRE_Int       type,
                             NALU_HYPRE_IJMatrix *matrix);

/**
 * Read the matrix from MM file.  This is mainly for debugging purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixReadMM(const char     *filename,
                               MPI_Comm        comm,
                               NALU_HYPRE_Int       type,
                               NALU_HYPRE_IJMatrix *matrix);

/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJMatrixPrint(NALU_HYPRE_IJMatrix  matrix,
                              const char     *filename);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Vectors
 *
 * @{
 **/

struct nalu_hypre_IJVector_struct;
/**
 * The vector object.
 **/
typedef struct nalu_hypre_IJVector_struct *NALU_HYPRE_IJVector;

/**
 * Create a vector object.  Each process owns some unique consecutive
 * range of vector unknowns, indicated by the global indices \e
 * jlower and \e jupper.  The data is required to be such that the
 * value of \e jlower on any process \f$p\f$ be exactly one more than
 * the value of \e jupper on process \f$p-1\f$.  Note that the first
 * index of the global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 *
 * Collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorCreate(MPI_Comm        comm,
                               NALU_HYPRE_BigInt    jlower,
                               NALU_HYPRE_BigInt    jupper,
                               NALU_HYPRE_IJVector *vector);

/**
 * Destroy a vector object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorDestroy(NALU_HYPRE_IJVector vector);

/**
 * Prepare a vector object for setting coefficient values.  This
 * routine will also re-initialize an already assembled vector,
 * allowing users to modify coefficient values.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorInitialize(NALU_HYPRE_IJVector vector);

/**
 * Prepare a vector object for setting coefficient values.  This
 * routine will also re-initialize an already assembled vector,
 * allowing users to modify coefficient values. This routine
 * also specifies the memory location, i.e. host or device.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorInitialize_v2( NALU_HYPRE_IJVector vector,
                                       NALU_HYPRE_MemoryLocation memory_location );

/**
 * (Optional) Sets the maximum number of elements that are expected to be set
 * (or added) on other processors from this processor
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetMaxOffProcElmts(NALU_HYPRE_IJVector vector,
                                           NALU_HYPRE_Int      max_off_proc_elmts);

/**
 * (Optional) Sets the number of components (vectors) of a multivector. A vector
 * is assumed to have a single component when this function is not called.
 * This function must be called prior to NALU_HYPRE_IJVectorInitialize.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetNumComponents(NALU_HYPRE_IJVector  vector,
                                         NALU_HYPRE_Int       num_components);

/**
 * (Optional) Sets the component identifier of a vector with multiple components (multivector).
 * This can be used for Set/AddTo/Get purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetComponent(NALU_HYPRE_IJVector  vector,
                                     NALU_HYPRE_Int       component);

/**
 * Sets values in vector.  The arrays \e values and \e indices
 * are of dimension \e nvalues and contain the vector values to be
 * set and the corresponding global vector indices, respectively.
 * Erases any previous values at the specified locations and replaces
 * them with new ones.  Note that it is not possible to set values
 * on other processors. If one tries to set a value from proc i on proc j,
 * proc i will erase all previous occurrences of this value in its stack
 * (including values generated with AddToValues), and treat it like
 * a zero value. The actual value needs to be set on proc j.
 *
 * Not collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetValues(NALU_HYPRE_IJVector       vector,
                                  NALU_HYPRE_Int            nvalues,
                                  const NALU_HYPRE_BigInt  *indices,
                                  const NALU_HYPRE_Complex *values);

/**
 * Adds to values in vector.  Usage details are analogous to
 * \ref NALU_HYPRE_IJVectorSetValues.
 * Adds to any previous values at the specified locations, or, if
 * there was no value there before, inserts a new one.
 * AddToValues can be used to add to values on other processors.
 *
 * Not collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorAddToValues(NALU_HYPRE_IJVector       vector,
                                    NALU_HYPRE_Int            nvalues,
                                    const NALU_HYPRE_BigInt  *indices,
                                    const NALU_HYPRE_Complex *values);

/**
 * Finalize the construction of the vector before using.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorAssemble(NALU_HYPRE_IJVector vector);

/**
 * Update vectors by setting (action 1) or
 * adding to (action 0) values in 'vector'.
 * Note that this function cannot update values owned by other processes
 * and does not allow repeated index values in 'indices'.
 *
 * Not collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorUpdateValues(NALU_HYPRE_IJVector       vector,
                                     NALU_HYPRE_Int            nvalues,
                                     const NALU_HYPRE_BigInt  *indices,
                                     const NALU_HYPRE_Complex *values,
                                     NALU_HYPRE_Int            action);

/**
 * Gets values in vector.  Usage details are analogous to
 * \ref NALU_HYPRE_IJVectorSetValues.
 *
 * Not collective.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorGetValues(NALU_HYPRE_IJVector   vector,
                                  NALU_HYPRE_Int        nvalues,
                                  const NALU_HYPRE_BigInt *indices,
                                  NALU_HYPRE_Complex   *values);

/**
 * Set the storage type of the vector object to be constructed.
 * Currently, \e type can only be \c NALU_HYPRE_PARCSR.
 *
 * Not collective, but must be the same on all processes.
 *
 * @see NALU_HYPRE_IJVectorGetObject
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetObjectType(NALU_HYPRE_IJVector vector,
                                      NALU_HYPRE_Int      type);

/**
 * Get the storage type of the constructed vector object.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorGetObjectType(NALU_HYPRE_IJVector  vector,
                                      NALU_HYPRE_Int      *type);

/**
 * Returns range of the part of the vector owned by this processor.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorGetLocalRange(NALU_HYPRE_IJVector  vector,
                                      NALU_HYPRE_BigInt   *jlower,
                                      NALU_HYPRE_BigInt   *jupper);

/**
 * Get a reference to the constructed vector object.
 *
 * @see NALU_HYPRE_IJVectorSetObjectType
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorGetObject(NALU_HYPRE_IJVector   vector,
                                  void           **object);

/**
 * (Optional) Sets the print level, if the user wants to print
 * error messages. The default is 0, i.e. no error messages are printed.
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorSetPrintLevel(NALU_HYPRE_IJVector vector,
                                      NALU_HYPRE_Int      print_level);

/**
 * Read the vector from file.  This is mainly for debugging purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorRead(const char     *filename,
                             MPI_Comm        comm,
                             NALU_HYPRE_Int       type,
                             NALU_HYPRE_IJVector *vector);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorPrint(NALU_HYPRE_IJVector  vector,
                              const char     *filename);

/**
 * Computes the inner product between two vectors
 **/
NALU_HYPRE_Int NALU_HYPRE_IJVectorInnerProd(NALU_HYPRE_IJVector  x,
                                  NALU_HYPRE_IJVector  y,
                                  NALU_HYPRE_Real     *prod);

/**@}*/
/**@}*/

#ifdef __cplusplus
}
#endif

#endif
