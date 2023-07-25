/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Note: this module contains functionality for reading/writing
         Euclid's binary io format, and opening and closing files.
         Additional io can be found in in mat_dh_private, which contains
         private functions for reading/writing various matrix and
         vector formats; functions in that module are called by
         public class methods of the Mat_dh and Vec_dh classes.
*/

#ifndef IO_DH
#define IO_DH

/* #include "euclid_common.h" */

/*--------------------------------------------------------------------------
 * open and close files, with error checking
 *--------------------------------------------------------------------------*/
extern FILE * openFile_dh(const char *filenameIN, const char *modeIN);
extern void closeFile_dh(FILE *fpIN);

/*---------------------------------------------------------------------------
 * binary io; these are called by functions in mat_dh_private
 *---------------------------------------------------------------------------*/

bool isSmallEndian(void);

/* seq only ?? */
extern void io_dh_print_ebin_mat_private(NALU_HYPRE_Int m, NALU_HYPRE_Int beg_row,
                                NALU_HYPRE_Int *rp, NALU_HYPRE_Int *cval, NALU_HYPRE_Real *aval,
                           NALU_HYPRE_Int *n2o, NALU_HYPRE_Int *o2n, Hash_i_dh hash, char *filename);

/* seq only ?? */
extern void io_dh_read_ebin_mat_private(NALU_HYPRE_Int *m, NALU_HYPRE_Int **rp, NALU_HYPRE_Int **cval,
                                     NALU_HYPRE_Real **aval, char *filename);

/* seq only */
extern void io_dh_print_ebin_vec_private(NALU_HYPRE_Int n, NALU_HYPRE_Int beg_row, NALU_HYPRE_Real *vals,
                           NALU_HYPRE_Int *n2o, NALU_HYPRE_Int *o2n, Hash_i_dh hash, char *filename);
/* seq only */
extern void io_dh_read_ebin_vec_private(NALU_HYPRE_Int *n, NALU_HYPRE_Real **vals, char *filename);


#endif

