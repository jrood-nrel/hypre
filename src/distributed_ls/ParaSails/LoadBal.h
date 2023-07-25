/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * LoadBal.h header file.
 *
 *****************************************************************************/

#ifndef _LOADBAL_H
#define _LOADBAL_H

#define LOADBAL_REQ_TAG  888
#define LOADBAL_REP_TAG  889

typedef struct
{
    NALU_HYPRE_Int  pe;
    NALU_HYPRE_Int  beg_row;
    NALU_HYPRE_Int  end_row;
    NALU_HYPRE_Int *buffer;
}
DonorData;

typedef struct
{
    NALU_HYPRE_Int     pe;
    Matrix *mat;
    NALU_HYPRE_Real *buffer;
}
RecipData;

typedef struct
{
    NALU_HYPRE_Int         num_given;
    NALU_HYPRE_Int         num_taken;
    DonorData  *donor_data;
    RecipData  *recip_data;
    NALU_HYPRE_Int         beg_row;    /* local beginning row, after all donated rows */
}
LoadBal;

LoadBal *LoadBalDonate(MPI_Comm comm, Matrix *mat, Numbering *numb,
  NALU_HYPRE_Real local_cost, NALU_HYPRE_Real beta);
void LoadBalReturn(LoadBal *p, MPI_Comm comm, Matrix *mat);

#endif /* _LOADBAL_H */
