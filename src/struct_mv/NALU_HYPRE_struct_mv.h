/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_STRUCT_MV_HEADER
#define NALU_HYPRE_STRUCT_MV_HEADER

#include "NALU_HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/* forward declarations */
#ifndef NALU_HYPRE_StructVector_defined
#define NALU_HYPRE_StructVector_defined
struct nalu_hypre_StructVector_struct;
typedef struct nalu_hypre_StructVector_struct *NALU_HYPRE_StructVector;
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup StructSystemInterface Struct System Interface
 *
 * This interface represents a structured-grid conceptual view of a linear
 * system.
 *
 * @memo A structured-grid conceptual interface
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Grids
 *
 * @{
 **/

struct nalu_hypre_StructGrid_struct;
/**
 * A grid object is constructed out of several "boxes", defined on a global
 * abstract index space.
 **/
typedef struct nalu_hypre_StructGrid_struct *NALU_HYPRE_StructGrid;

/**
 * Create an <em>ndim</em>-dimensional grid object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructGridCreate(MPI_Comm          comm,
                                 NALU_HYPRE_Int         ndim,
                                 NALU_HYPRE_StructGrid *grid);

/**
 * Destroy a grid object.  An object should be explicitly destroyed using this
 * destructor when the user's code no longer needs direct access to it.  Once
 * destroyed, the object must not be referenced again.  Note that the object may
 * not be deallocated at the completion of this call, since there may be
 * internal package references to the object.  The object will then be destroyed
 * when all internal reference counts go to zero.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructGridDestroy(NALU_HYPRE_StructGrid grid);

/**
 * Set the extents for a box on the grid.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructGridSetExtents(NALU_HYPRE_StructGrid  grid,
                                     NALU_HYPRE_Int        *ilower,
                                     NALU_HYPRE_Int        *iupper);

/**
 * Finalize the construction of the grid before using.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructGridAssemble(NALU_HYPRE_StructGrid grid);

/**
 * Set the periodicity for the grid.
 *
 * The argument \e periodic is an <em>ndim</em>-dimensional integer array that
 * contains the periodicity for each dimension.  A zero value for a dimension
 * means non-periodic, while a nonzero value means periodic and contains the
 * actual period.  For example, periodicity in the first and third dimensions
 * for a 10x11x12 grid is indicated by the array [10,0,12].
 *
 * NOTE: Some of the solvers in hypre have power-of-two restrictions on the size
 * of the periodic dimensions.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructGridSetPeriodic(NALU_HYPRE_StructGrid  grid,
                                      NALU_HYPRE_Int        *periodic);

/**
 * Set the ghost layer in the grid object
 **/
NALU_HYPRE_Int NALU_HYPRE_StructGridSetNumGhost(NALU_HYPRE_StructGrid  grid,
                                      NALU_HYPRE_Int        *num_ghost);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Stencils
 *
 * @{
 **/

struct nalu_hypre_StructStencil_struct;
/**
 * The stencil object.
 **/
typedef struct nalu_hypre_StructStencil_struct *NALU_HYPRE_StructStencil;

/**
 * Create a stencil object for the specified number of spatial dimensions and
 * stencil entries.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructStencilCreate(NALU_HYPRE_Int            ndim,
                                    NALU_HYPRE_Int            size,
                                    NALU_HYPRE_StructStencil *stencil);

/**
 * Destroy a stencil object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructStencilDestroy(NALU_HYPRE_StructStencil stencil);

/**
 * Set a stencil entry.
 *
 * NOTE: The name of this routine will eventually be changed to \e
 * HYPRE\_StructStencilSetEntry.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructStencilSetElement(NALU_HYPRE_StructStencil  stencil,
                                        NALU_HYPRE_Int            entry,
                                        NALU_HYPRE_Int           *offset);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Matrices
 *
 * @{
 **/

struct nalu_hypre_StructMatrix_struct;
/**
 * The matrix object.
 **/
typedef struct nalu_hypre_StructMatrix_struct *NALU_HYPRE_StructMatrix;

/**
 * Create a matrix object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixCreate(MPI_Comm             comm,
                                   NALU_HYPRE_StructGrid     grid,
                                   NALU_HYPRE_StructStencil  stencil,
                                   NALU_HYPRE_StructMatrix  *matrix);

/**
 * Destroy a matrix object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixDestroy(NALU_HYPRE_StructMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixInitialize(NALU_HYPRE_StructMatrix matrix);

/**
 * Set matrix coefficients index by index.  The \e values array is of length
 * \e nentries.
 *
 * NOTE: For better efficiency, use \ref NALU_HYPRE_StructMatrixSetBoxValues to set
 * coefficients a box at a time.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetValues(NALU_HYPRE_StructMatrix  matrix,
                                      NALU_HYPRE_Int          *index,
                                      NALU_HYPRE_Int           nentries,
                                      NALU_HYPRE_Int          *entries,
                                      NALU_HYPRE_Complex      *values);

/**
 * Add to matrix coefficients index by index.  The \e values array is of
 * length \e nentries.
 *
 * NOTE: For better efficiency, use \ref NALU_HYPRE_StructMatrixAddToBoxValues to
 * set coefficients a box at a time.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixAddToValues(NALU_HYPRE_StructMatrix  matrix,
                                        NALU_HYPRE_Int          *index,
                                        NALU_HYPRE_Int           nentries,
                                        NALU_HYPRE_Int          *entries,
                                        NALU_HYPRE_Complex      *values);

/**
 * Set matrix coefficients which are constant over the grid.  The \e values
 * array is of length \e nentries.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetConstantValues(NALU_HYPRE_StructMatrix  matrix,
                                              NALU_HYPRE_Int           nentries,
                                              NALU_HYPRE_Int          *entries,
                                              NALU_HYPRE_Complex      *values);
/**
 * Add to matrix coefficients which are constant over the grid.  The \e
 * values array is of length \e nentries.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixAddToConstantValues(NALU_HYPRE_StructMatrix  matrix,
                                                NALU_HYPRE_Int           nentries,
                                                NALU_HYPRE_Int          *entries,
                                                NALU_HYPRE_Complex      *values);

/**
 * Set matrix coefficients a box at a time.  The data in \e values is ordered
 * as follows:
 *
   \verbatim
   m = 0;
   for (k = ilower[2]; k <= iupper[2]; k++)
      for (j = ilower[1]; j <= iupper[1]; j++)
         for (i = ilower[0]; i <= iupper[0]; i++)
            for (entry = 0; entry < nentries; entry++)
            {
               values[m] = ...;
               m++;
            }
   \endverbatim
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetBoxValues(NALU_HYPRE_StructMatrix  matrix,
                                         NALU_HYPRE_Int          *ilower,
                                         NALU_HYPRE_Int          *iupper,
                                         NALU_HYPRE_Int           nentries,
                                         NALU_HYPRE_Int          *entries,
                                         NALU_HYPRE_Complex      *values);
/**
 * Add to matrix coefficients a box at a time.  The data in \e values is
 * ordered as in \ref NALU_HYPRE_StructMatrixSetBoxValues.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixAddToBoxValues(NALU_HYPRE_StructMatrix  matrix,
                                           NALU_HYPRE_Int          *ilower,
                                           NALU_HYPRE_Int          *iupper,
                                           NALU_HYPRE_Int           nentries,
                                           NALU_HYPRE_Int          *entries,
                                           NALU_HYPRE_Complex      *values);

/**
 * Set matrix coefficients a box at a time.  The \e values array is logically
 * box shaped with value-box extents \e vilower and \e viupper that must
 * contain the set-box extents \e ilower and \e iupper .  The data in the
 * \e values array is ordered as in \ref NALU_HYPRE_StructMatrixSetBoxValues, but
 * based on the value-box extents.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetBoxValues2(NALU_HYPRE_StructMatrix  matrix,
                                          NALU_HYPRE_Int          *ilower,
                                          NALU_HYPRE_Int          *iupper,
                                          NALU_HYPRE_Int           nentries,
                                          NALU_HYPRE_Int          *entries,
                                          NALU_HYPRE_Int          *vilower,
                                          NALU_HYPRE_Int          *viupper,
                                          NALU_HYPRE_Complex      *values);
/**
 * Add to matrix coefficients a box at a time.  The data in \e values is
 * ordered as in \ref NALU_HYPRE_StructMatrixSetBoxValues2.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixAddToBoxValues2(NALU_HYPRE_StructMatrix  matrix,
                                            NALU_HYPRE_Int          *ilower,
                                            NALU_HYPRE_Int          *iupper,
                                            NALU_HYPRE_Int           nentries,
                                            NALU_HYPRE_Int          *entries,
                                            NALU_HYPRE_Int          *vilower,
                                            NALU_HYPRE_Int          *viupper,
                                            NALU_HYPRE_Complex      *values);

/**
 * Finalize the construction of the matrix before using.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixAssemble(NALU_HYPRE_StructMatrix matrix);

/**
 * Get matrix coefficients index by index.  The \e values array is of length
 * \e nentries.
 *
 * NOTE: For better efficiency, use \ref NALU_HYPRE_StructMatrixGetBoxValues to get
 * coefficients a box at a time.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixGetValues(NALU_HYPRE_StructMatrix  matrix,
                                      NALU_HYPRE_Int          *index,
                                      NALU_HYPRE_Int           nentries,
                                      NALU_HYPRE_Int          *entries,
                                      NALU_HYPRE_Complex      *values);

/**
 * Get matrix coefficients a box at a time.  The data in \e values is
 * ordered as in \ref NALU_HYPRE_StructMatrixSetBoxValues.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixGetBoxValues(NALU_HYPRE_StructMatrix  matrix,
                                         NALU_HYPRE_Int          *ilower,
                                         NALU_HYPRE_Int          *iupper,
                                         NALU_HYPRE_Int           nentries,
                                         NALU_HYPRE_Int          *entries,
                                         NALU_HYPRE_Complex      *values);

/**
 * Get matrix coefficients a box at a time.  The data in \e values is
 * ordered as in \ref NALU_HYPRE_StructMatrixSetBoxValues2.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixGetBoxValues2(NALU_HYPRE_StructMatrix  matrix,
                                          NALU_HYPRE_Int          *ilower,
                                          NALU_HYPRE_Int          *iupper,
                                          NALU_HYPRE_Int           nentries,
                                          NALU_HYPRE_Int          *entries,
                                          NALU_HYPRE_Int          *vilower,
                                          NALU_HYPRE_Int          *viupper,
                                          NALU_HYPRE_Complex      *values);

/**
 * Define symmetry properties of the matrix.  By default, matrices are assumed
 * to be nonsymmetric.  Significant storage savings can be made if the matrix is
 * symmetric.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetSymmetric(NALU_HYPRE_StructMatrix  matrix,
                                         NALU_HYPRE_Int           symmetric);

/**
 * Specify which stencil entries are constant over the grid.  Declaring entries
 * to be "constant over the grid" yields significant memory savings because
 * the value for each declared entry will only be stored once.  However, not all
 * solvers are able to utilize this feature.
 *
 * Presently supported:
 *    - no entries constant (this function need not be called)
 *    - all entries constant
 *    - all but the diagonal entry constant
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetConstantEntries( NALU_HYPRE_StructMatrix matrix,
                                                NALU_HYPRE_Int          nentries,
                                                NALU_HYPRE_Int         *entries );

/**
 * Set the ghost layer in the matrix
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixSetNumGhost(NALU_HYPRE_StructMatrix  matrix,
                                        NALU_HYPRE_Int          *num_ghost);


/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixPrint(const char         *filename,
                                  NALU_HYPRE_StructMatrix  matrix,
                                  NALU_HYPRE_Int           all);

/**
 * Read the matrix from file.  This is mainly for debugging purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixRead( MPI_Comm             comm,
                                  const char          *filename,
                                  NALU_HYPRE_Int           *num_ghost,
                                  NALU_HYPRE_StructMatrix  *matrix );

/**
 * Matvec operator.  This operation is \f$y = \alpha A x + \beta y\f$ .
 * Note that you can do a simple matrix-vector multiply by setting
 * \f$\alpha=1\f$ and \f$\beta=0\f$.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructMatrixMatvec ( NALU_HYPRE_Complex alpha,
                                     NALU_HYPRE_StructMatrix A,
                                     NALU_HYPRE_StructVector x,
                                     NALU_HYPRE_Complex beta,
                                     NALU_HYPRE_StructVector y );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Vectors
 *
 * @{
 **/

struct nalu_hypre_StructVector_struct;
/**
 * The vector object.
 **/
#ifndef NALU_HYPRE_StructVector_defined
typedef struct nalu_hypre_StructVector_struct *NALU_HYPRE_StructVector;
#endif

/**
 * Create a vector object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorCreate(MPI_Comm            comm,
                                   NALU_HYPRE_StructGrid    grid,
                                   NALU_HYPRE_StructVector *vector);

/**
 * Destroy a vector object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorDestroy(NALU_HYPRE_StructVector vector);

/**
 * Prepare a vector object for setting coefficient values.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorInitialize(NALU_HYPRE_StructVector vector);

/**
 * Set vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \ref NALU_HYPRE_StructVectorSetBoxValues to set
 * coefficients a box at a time.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorSetValues(NALU_HYPRE_StructVector  vector,
                                      NALU_HYPRE_Int          *index,
                                      NALU_HYPRE_Complex       value);

/**
 * Add to vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \ref NALU_HYPRE_StructVectorAddToBoxValues to
 * set coefficients a box at a time.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorAddToValues(NALU_HYPRE_StructVector  vector,
                                        NALU_HYPRE_Int          *index,
                                        NALU_HYPRE_Complex       value);

/**
 * Set vector coefficients a box at a time.  The data in \e values is ordered
 * as follows:
 *
   \verbatim
   m = 0;
   for (k = ilower[2]; k <= iupper[2]; k++)
      for (j = ilower[1]; j <= iupper[1]; j++)
         for (i = ilower[0]; i <= iupper[0]; i++)
         {
            values[m] = ...;
            m++;
         }
   \endverbatim
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorSetBoxValues(NALU_HYPRE_StructVector  vector,
                                         NALU_HYPRE_Int          *ilower,
                                         NALU_HYPRE_Int          *iupper,
                                         NALU_HYPRE_Complex      *values);
/**
 * Add to vector coefficients a box at a time.  The data in \e values is
 * ordered as in \ref NALU_HYPRE_StructVectorSetBoxValues.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorAddToBoxValues(NALU_HYPRE_StructVector  vector,
                                           NALU_HYPRE_Int          *ilower,
                                           NALU_HYPRE_Int          *iupper,
                                           NALU_HYPRE_Complex      *values);

/**
 * Set vector coefficients a box at a time.  The \e values array is logically
 * box shaped with value-box extents \e vilower and \e viupper that must
 * contain the set-box extents \e ilower and \e iupper .  The data in the
 * \e values array is ordered as in \ref NALU_HYPRE_StructVectorSetBoxValues, but
 * based on the value-box extents.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorSetBoxValues2(NALU_HYPRE_StructVector  vector,
                                          NALU_HYPRE_Int          *ilower,
                                          NALU_HYPRE_Int          *iupper,
                                          NALU_HYPRE_Int          *vilower,
                                          NALU_HYPRE_Int          *viupper,
                                          NALU_HYPRE_Complex      *values);
/**
 * Add to vector coefficients a box at a time.  The data in \e values is
 * ordered as in \ref NALU_HYPRE_StructVectorSetBoxValues2.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorAddToBoxValues2(NALU_HYPRE_StructVector  vector,
                                            NALU_HYPRE_Int          *ilower,
                                            NALU_HYPRE_Int          *iupper,
                                            NALU_HYPRE_Int          *vilower,
                                            NALU_HYPRE_Int          *viupper,
                                            NALU_HYPRE_Complex      *values);

/**
 * Finalize the construction of the vector before using.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorAssemble(NALU_HYPRE_StructVector vector);

/**
 * Get vector coefficients index by index.
 *
 * NOTE: For better efficiency, use \ref NALU_HYPRE_StructVectorGetBoxValues to get
 * coefficients a box at a time.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorGetValues(NALU_HYPRE_StructVector  vector,
                                      NALU_HYPRE_Int          *index,
                                      NALU_HYPRE_Complex      *value);

/**
 * Get vector coefficients a box at a time.  The data in \e values is ordered
 * as in \ref NALU_HYPRE_StructVectorSetBoxValues.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorGetBoxValues(NALU_HYPRE_StructVector  vector,
                                         NALU_HYPRE_Int          *ilower,
                                         NALU_HYPRE_Int          *iupper,
                                         NALU_HYPRE_Complex      *values);

/**
 * Get vector coefficients a box at a time.  The data in \e values is ordered
 * as in \ref NALU_HYPRE_StructVectorSetBoxValues2.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorGetBoxValues2(NALU_HYPRE_StructVector  vector,
                                          NALU_HYPRE_Int          *ilower,
                                          NALU_HYPRE_Int          *iupper,
                                          NALU_HYPRE_Int          *vilower,
                                          NALU_HYPRE_Int          *viupper,
                                          NALU_HYPRE_Complex      *values);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorPrint(const char         *filename,
                                  NALU_HYPRE_StructVector  vector,
                                  NALU_HYPRE_Int           all);

/**
 * Read the vector from file.  This is mainly for debugging purposes.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructVectorRead( MPI_Comm             comm,
                                  const char          *filename,
                                  NALU_HYPRE_Int           *num_ghost,
                                  NALU_HYPRE_StructVector  *vector );

/**@}*/
/**@}*/

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_StructMatrixGetGrid(NALU_HYPRE_StructMatrix  matrix,
                                    NALU_HYPRE_StructGrid   *grid);

struct nalu_hypre_CommPkg_struct;
typedef struct nalu_hypre_CommPkg_struct *NALU_HYPRE_CommPkg;

NALU_HYPRE_Int NALU_HYPRE_StructVectorSetNumGhost(NALU_HYPRE_StructVector  vector,
                                        NALU_HYPRE_Int          *num_ghost);

NALU_HYPRE_Int NALU_HYPRE_StructVectorSetConstantValues(NALU_HYPRE_StructVector vector,
                                              NALU_HYPRE_Complex      values);

NALU_HYPRE_Int NALU_HYPRE_StructVectorGetMigrateCommPkg(NALU_HYPRE_StructVector  from_vector,
                                              NALU_HYPRE_StructVector  to_vector,
                                              NALU_HYPRE_CommPkg      *comm_pkg);

NALU_HYPRE_Int NALU_HYPRE_StructVectorMigrate(NALU_HYPRE_CommPkg      comm_pkg,
                                    NALU_HYPRE_StructVector from_vector,
                                    NALU_HYPRE_StructVector to_vector);

NALU_HYPRE_Int NALU_HYPRE_CommPkgDestroy(NALU_HYPRE_CommPkg comm_pkg);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int
NALU_HYPRE_StructGridSetDataLocation( NALU_HYPRE_StructGrid grid, NALU_HYPRE_MemoryLocation data_location );
#endif

#ifdef __cplusplus
}
#endif

#endif
