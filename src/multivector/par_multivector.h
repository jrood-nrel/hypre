/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Parallel Vector data structure
 *
 *****************************************************************************/
#ifndef hypre_PAR_MULTIVECTOR_HEADER
#define hypre_PAR_MULTIVECTOR_HEADER

#include "_hypre_utilities.h"
#include "seq_Multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * hypre_ParMultiVector
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm       comm;
   NALU_HYPRE_Int                global_size;
   NALU_HYPRE_Int                first_index;
   NALU_HYPRE_Int               *partitioning;
   NALU_HYPRE_Int               owns_data;
   NALU_HYPRE_Int               num_vectors;
   hypre_Multivector  *local_vector;

   /* using mask on "parallel" level seems to be inconvenient, so i (IL) moved it to
          "sequential" level. Also i now store it as a number of active indices and an array of
          active indices. hypre_ParMultiVectorSetMask converts user-provided "(1,1,0,1,...)" mask
          to the format above.
      NALU_HYPRE_Int                *mask;
   */

} hypre_ParMultivector;


/*--------------------------------------------------------------------------
 * Accessor macros for the Vector structure;
 * kinda strange macros; right hand side looks much convenient than left.....
 *--------------------------------------------------------------------------*/

#define hypre_ParMultiVectorComm(vector)             ((vector) -> comm)
#define hypre_ParMultiVectorGlobalSize(vector)       ((vector) -> global_size)
#define hypre_ParMultiVectorFirstIndex(vector)       ((vector) -> first_index)
#define hypre_ParMultiVectorPartitioning(vector)     ((vector) -> partitioning)
#define hypre_ParMultiVectorLocalVector(vector)      ((vector) -> local_vector)
#define hypre_ParMultiVectorOwnsData(vector)         ((vector) -> owns_data)
#define hypre_ParMultiVectorNumVectors(vector)       ((vector) -> num_vectors)

/* field "mask" moved to "sequential" level, see structure above
#define hypre_ParMultiVectorMask(vector)             ((vector) -> mask)
*/

/* function prototypes for working with hypre_ParMultiVector */
hypre_ParMultiVector *hypre_ParMultiVectorCreate(MPI_Comm, NALU_HYPRE_Int, NALU_HYPRE_Int *, NALU_HYPRE_Int);
NALU_HYPRE_Int hypre_ParMultiVectorDestroy(hypre_ParMultiVector *);
NALU_HYPRE_Int hypre_ParMultiVectorInitialize(hypre_ParMultiVector *);
NALU_HYPRE_Int hypre_ParMultiVectorSetDataOwner(hypre_ParMultiVector *, NALU_HYPRE_Int);
NALU_HYPRE_Int hypre_ParMultiVectorSetMask(hypre_ParMultiVector *, NALU_HYPRE_Int *);
NALU_HYPRE_Int hypre_ParMultiVectorSetConstantValues(hypre_ParMultiVector *, NALU_HYPRE_Complex);
NALU_HYPRE_Int hypre_ParMultiVectorSetRandomValues(hypre_ParMultiVector *, NALU_HYPRE_Int);
NALU_HYPRE_Int hypre_ParMultiVectorCopy(hypre_ParMultiVector *, hypre_ParMultiVector *);
NALU_HYPRE_Int hypre_ParMultiVectorScale(NALU_HYPRE_Complex, hypre_ParMultiVector *);
NALU_HYPRE_Int hypre_ParMultiVectorMultiScale(NALU_HYPRE_Complex *, hypre_ParMultiVector *);
NALU_HYPRE_Int hypre_ParMultiVectorAxpy(NALU_HYPRE_Complex, hypre_ParMultiVector *,
                                   hypre_ParMultiVector *);

NALU_HYPRE_Int hypre_ParMultiVectorByDiag(  hypre_ParMultiVector *x,
                                       NALU_HYPRE_Int                *mask,
                                       NALU_HYPRE_Int                n,
                                       NALU_HYPRE_Complex      *alpha,
                                       hypre_ParMultiVector *y);

NALU_HYPRE_Int hypre_ParMultiVectorInnerProd(hypre_ParMultiVector *,
                                        hypre_ParMultiVector *, NALU_HYPRE_Real *, NALU_HYPRE_Real *);
NALU_HYPRE_Int hypre_ParMultiVectorInnerProdDiag(hypre_ParMultiVector *,
                                            hypre_ParMultiVector *, NALU_HYPRE_Real *, NALU_HYPRE_Real *);
NALU_HYPRE_Int
hypre_ParMultiVectorCopyWithoutMask(hypre_ParMultiVector *x, hypre_ParMultiVector *y);
NALU_HYPRE_Int
hypre_ParMultiVectorByMatrix(hypre_ParMultiVector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                             NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, hypre_ParMultiVector * y);
NALU_HYPRE_Int
hypre_ParMultiVectorXapy(hypre_ParMultiVector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                         NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, hypre_ParMultiVector * y);

NALU_HYPRE_Int
hypre_ParMultiVectorEval(void (*f)( void*, void*, void* ), void* par,
                         hypre_ParMultiVector * x, hypre_ParMultiVector * y);

/* to be replaced by better implementation when format for multivector files established */
hypre_ParMultiVector * hypre_ParMultiVectorTempRead(MPI_Comm comm, const char *file_name);
NALU_HYPRE_Int hypre_ParMultiVectorTempPrint(hypre_ParMultiVector *vector, const char *file_name);

#ifdef __cplusplus
}
#endif

#endif   /* hypre_PAR_MULTIVECTOR_HEADER */
