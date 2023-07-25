/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Parallel Vector data structure
 *
 *****************************************************************************/
#ifndef nalu_hypre_PAR_MULTIVECTOR_HEADER
#define nalu_hypre_PAR_MULTIVECTOR_HEADER

#include "_nalu_hypre_utilities.h"
#include "seq_Multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVector
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm       comm;
   NALU_HYPRE_Int                global_size;
   NALU_HYPRE_Int                first_index;
   NALU_HYPRE_Int               *partitioning;
   NALU_HYPRE_Int               owns_data;
   NALU_HYPRE_Int               num_vectors;
   nalu_hypre_Multivector  *local_vector;

   /* using mask on "parallel" level seems to be inconvenient, so i (IL) moved it to
          "sequential" level. Also i now store it as a number of active indices and an array of
          active indices. nalu_hypre_ParMultiVectorSetMask converts user-provided "(1,1,0,1,...)" mask
          to the format above.
      NALU_HYPRE_Int                *mask;
   */

} nalu_hypre_ParMultivector;


/*--------------------------------------------------------------------------
 * Accessor macros for the Vector structure;
 * kinda strange macros; right hand side looks much convenient than left.....
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ParMultiVectorComm(vector)             ((vector) -> comm)
#define nalu_hypre_ParMultiVectorGlobalSize(vector)       ((vector) -> global_size)
#define nalu_hypre_ParMultiVectorFirstIndex(vector)       ((vector) -> first_index)
#define nalu_hypre_ParMultiVectorPartitioning(vector)     ((vector) -> partitioning)
#define nalu_hypre_ParMultiVectorLocalVector(vector)      ((vector) -> local_vector)
#define nalu_hypre_ParMultiVectorOwnsData(vector)         ((vector) -> owns_data)
#define nalu_hypre_ParMultiVectorNumVectors(vector)       ((vector) -> num_vectors)

/* field "mask" moved to "sequential" level, see structure above
#define nalu_hypre_ParMultiVectorMask(vector)             ((vector) -> mask)
*/

/* function prototypes for working with nalu_hypre_ParMultiVector */
nalu_hypre_ParMultiVector *nalu_hypre_ParMultiVectorCreate(MPI_Comm, NALU_HYPRE_Int, NALU_HYPRE_Int *, NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorDestroy(nalu_hypre_ParMultiVector *);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorInitialize(nalu_hypre_ParMultiVector *);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorSetDataOwner(nalu_hypre_ParMultiVector *, NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorSetMask(nalu_hypre_ParMultiVector *, NALU_HYPRE_Int *);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorSetConstantValues(nalu_hypre_ParMultiVector *, NALU_HYPRE_Complex);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorSetRandomValues(nalu_hypre_ParMultiVector *, NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorCopy(nalu_hypre_ParMultiVector *, nalu_hypre_ParMultiVector *);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorScale(NALU_HYPRE_Complex, nalu_hypre_ParMultiVector *);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorMultiScale(NALU_HYPRE_Complex *, nalu_hypre_ParMultiVector *);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorAxpy(NALU_HYPRE_Complex, nalu_hypre_ParMultiVector *,
                                   nalu_hypre_ParMultiVector *);

NALU_HYPRE_Int nalu_hypre_ParMultiVectorByDiag(  nalu_hypre_ParMultiVector *x,
                                       NALU_HYPRE_Int                *mask,
                                       NALU_HYPRE_Int                n,
                                       NALU_HYPRE_Complex      *alpha,
                                       nalu_hypre_ParMultiVector *y);

NALU_HYPRE_Int nalu_hypre_ParMultiVectorInnerProd(nalu_hypre_ParMultiVector *,
                                        nalu_hypre_ParMultiVector *, NALU_HYPRE_Real *, NALU_HYPRE_Real *);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorInnerProdDiag(nalu_hypre_ParMultiVector *,
                                            nalu_hypre_ParMultiVector *, NALU_HYPRE_Real *, NALU_HYPRE_Real *);
NALU_HYPRE_Int
nalu_hypre_ParMultiVectorCopyWithoutMask(nalu_hypre_ParMultiVector *x, nalu_hypre_ParMultiVector *y);
NALU_HYPRE_Int
nalu_hypre_ParMultiVectorByMatrix(nalu_hypre_ParMultiVector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                             NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, nalu_hypre_ParMultiVector * y);
NALU_HYPRE_Int
nalu_hypre_ParMultiVectorXapy(nalu_hypre_ParMultiVector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                         NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, nalu_hypre_ParMultiVector * y);

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorEval(void (*f)( void*, void*, void* ), void* par,
                         nalu_hypre_ParMultiVector * x, nalu_hypre_ParMultiVector * y);

/* to be replaced by better implementation when format for multivector files established */
nalu_hypre_ParMultiVector * nalu_hypre_ParMultiVectorTempRead(MPI_Comm comm, const char *file_name);
NALU_HYPRE_Int nalu_hypre_ParMultiVectorTempPrint(nalu_hypre_ParMultiVector *vector, const char *file_name);

#ifdef __cplusplus
}
#endif

#endif   /* nalu_hypre_PAR_MULTIVECTOR_HEADER */
