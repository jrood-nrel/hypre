/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Multivector data structure
 *
 *****************************************************************************/

#ifndef nalu_hypre_MULTIVECTOR_HEADER
#define nalu_hypre_MULTIVECTOR_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_Multivector
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Complex  *data;
   NALU_HYPRE_Int      size;
   NALU_HYPRE_Int      owns_data;
   NALU_HYPRE_Int      num_vectors;  /* the above "size" is size of one vector */

   NALU_HYPRE_Int      num_active_vectors;
   NALU_HYPRE_Int     *active_indices;  /* indices of active vectors; 0-based notation */

} nalu_hypre_Multivector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Multivector structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_MultivectorData(vector)      ((vector) -> data)
#define nalu_hypre_MultivectorSize(vector)      ((vector) -> size)
#define nalu_hypre_MultivectorOwnsData(vector)  ((vector) -> owns_data)
#define nalu_hypre_MultivectorNumVectors(vector) ((vector) -> num_vectors)

nalu_hypre_Multivector * nalu_hypre_SeqMultivectorCreate(NALU_HYPRE_Int size, NALU_HYPRE_Int num_vectors);
nalu_hypre_Multivector *nalu_hypre_SeqMultivectorRead(char *file_name);

NALU_HYPRE_Int nalu_hypre_SeqMultivectorDestroy(nalu_hypre_Multivector *vector);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorInitialize(nalu_hypre_Multivector *vector);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorSetDataOwner(nalu_hypre_Multivector *vector, NALU_HYPRE_Int owns_data);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorPrint(nalu_hypre_Multivector *vector, char *file_name);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorSetConstantValues(nalu_hypre_Multivector *v, NALU_HYPRE_Complex value);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorSetRandomValues(nalu_hypre_Multivector *v, NALU_HYPRE_Int seed);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorCopy(nalu_hypre_Multivector *x, nalu_hypre_Multivector *y);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorScale(NALU_HYPRE_Complex alpha, nalu_hypre_Multivector *y, NALU_HYPRE_Int *mask);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorAxpy(NALU_HYPRE_Complex alpha, nalu_hypre_Multivector *x,
                                   nalu_hypre_Multivector *y);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorInnerProd(nalu_hypre_Multivector *x, nalu_hypre_Multivector *y,
                                        NALU_HYPRE_Real *results);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorMultiScale(NALU_HYPRE_Complex *alpha, nalu_hypre_Multivector *v,
                                         NALU_HYPRE_Int *mask);
NALU_HYPRE_Int nalu_hypre_SeqMultivectorByDiag(nalu_hypre_Multivector *x, NALU_HYPRE_Int *mask, NALU_HYPRE_Int n,
                                     NALU_HYPRE_Complex *alpha, nalu_hypre_Multivector *y);

NALU_HYPRE_Int nalu_hypre_SeqMultivectorInnerProdDiag(nalu_hypre_Multivector *x,
                                            nalu_hypre_Multivector *y,
                                            NALU_HYPRE_Real *diagResults );

NALU_HYPRE_Int nalu_hypre_SeqMultivectorSetMask(nalu_hypre_Multivector *mvector, NALU_HYPRE_Int * mask);

NALU_HYPRE_Int nalu_hypre_SeqMultivectorCopyWithoutMask(nalu_hypre_Multivector *x,
                                              nalu_hypre_Multivector *y);

NALU_HYPRE_Int nalu_hypre_SeqMultivectorByMatrix(nalu_hypre_Multivector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                                       NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, nalu_hypre_Multivector *y);

NALU_HYPRE_Int nalu_hypre_SeqMultivectorXapy (nalu_hypre_Multivector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                                    NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, nalu_hypre_Multivector *y);

#ifdef __cplusplus
}
#endif

#endif /* nalu_hypre_MULTIVECTOR_HEADER */
