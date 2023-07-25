/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Parallel Chord Matrix data structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_PAR_CHORD_MATRIX_HEADER
#define nalu_hypre_PAR_CHORD_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Parallel Chord Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm comm;

   /*  A structure: -------------------------------------------------------- */
   NALU_HYPRE_Int num_inprocessors;
   NALU_HYPRE_Int *inprocessor;

   /* receiving in idof from different (in)processors; ---------------------- */
   NALU_HYPRE_Int *num_idofs_inprocessor;
   NALU_HYPRE_Int **idof_inprocessor;

   /* symmetric information: ----------------------------------------------- */
   /* this can be replaces by CSR format: ---------------------------------- */
   NALU_HYPRE_Int     *num_inchords;
   NALU_HYPRE_Int     **inchord_idof;
   NALU_HYPRE_Int     **inchord_rdof;
   NALU_HYPRE_Complex **inchord_data;

   NALU_HYPRE_Int num_idofs;
   NALU_HYPRE_Int num_rdofs;

   NALU_HYPRE_Int *firstindex_idof; /* not owned by my_id; ---------------------- */
   NALU_HYPRE_Int *firstindex_rdof; /* not owned by my_id; ---------------------- */

   /* --------------------------- mirror information: ---------------------- */
   /* participation of rdof in different processors; ----------------------- */

   NALU_HYPRE_Int num_toprocessors;
   NALU_HYPRE_Int *toprocessor;

   /* rdofs to be sentto toprocessors; -------------------------------------
      ---------------------------------------------------------------------- */
   NALU_HYPRE_Int *num_rdofs_toprocessor;
   NALU_HYPRE_Int **rdof_toprocessor;

} nalu_hypre_ParChordMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ParChordMatrixComm(matrix)                  ((matrix) -> comm)

/*  matrix structure: ----------------------------------------------------- */

#define nalu_hypre_ParChordMatrixNumInprocessors(matrix)  ((matrix) -> num_inprocessors)
#define nalu_hypre_ParChordMatrixInprocessor(matrix) ((matrix) -> inprocessor)
#define nalu_hypre_ParChordMatrixNumIdofsInprocessor(matrix) ((matrix) -> num_idofs_inprocessor)
#define nalu_hypre_ParChordMatrixIdofInprocessor(matrix) ((matrix) -> idof_inprocessor)

#define nalu_hypre_ParChordMatrixNumInchords(matrix) ((matrix) -> num_inchords)

#define nalu_hypre_ParChordMatrixInchordIdof(matrix) ((matrix) -> inchord_idof)
#define nalu_hypre_ParChordMatrixInchordRdof(matrix) ((matrix) -> inchord_rdof)
#define nalu_hypre_ParChordMatrixInchordData(matrix) ((matrix) -> inchord_data)
#define nalu_hypre_ParChordMatrixNumIdofs(matrix)    ((matrix) -> num_idofs)
#define nalu_hypre_ParChordMatrixNumRdofs(matrix)    ((matrix) -> num_rdofs)

#define nalu_hypre_ParChordMatrixFirstindexIdof(matrix) ((matrix) -> firstindex_idof)
#define nalu_hypre_ParChordMatrixFirstindexRdof(matrix) ((matrix) -> firstindex_rdof)

/* participation of rdof in different processors; ---------- */

#define nalu_hypre_ParChordMatrixNumToprocessors(matrix) ((matrix) -> num_toprocessors)
#define nalu_hypre_ParChordMatrixToprocessor(matrix)  ((matrix) -> toprocessor)
#define nalu_hypre_ParChordMatrixNumRdofsToprocessor(matrix) ((matrix) -> num_rdofs_toprocessor)
#define nalu_hypre_ParChordMatrixRdofToprocessor(matrix) ((matrix) -> rdof_toprocessor)

#endif

