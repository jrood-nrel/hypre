/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef MAT_DH_DH
#define MAT_DH_DH

/* #include "euclid_common.h" */

  /* this stuff for experimental internal timing */
#define MAT_DH_BINS      10
#define MATVEC_TIME       0  /* time to actually perform matvec */
#define MATVEC_MPI_TIME   1  /* time for comms + vector copying needed */
#define MATVEC_MPI_TIME2  5  /* time for comms, + vector copying needed */
#define MATVEC_TOTAL_TIME 2  /* MATVEC_TIME+MATVEC_MPI_TIME */
#define MATVEC_RATIO      3  /* computation/communication ratio */
#define MATVEC_WORDS      4  /* total words sent to other procs. */

struct _mat_dh {
  NALU_HYPRE_Int m, n;    /* dimensions of local rectangular submatrix;
                * the global matrix is n by n.
                */
  NALU_HYPRE_Int beg_row;   /* global number of 1st locally owned row */
  NALU_HYPRE_Int bs;        /* block size */

  /* sparse row-oriented storage for locally owned submatrix */
  NALU_HYPRE_Int *rp;       
  NALU_HYPRE_Int *len;   /* length of each row; only used for MPI triangular solves */
  NALU_HYPRE_Int *cval;
  NALU_HYPRE_Int *fill;
  NALU_HYPRE_Int *diag;
  NALU_HYPRE_Real *aval;
  bool owner;  /* for MPI triangular solves */

  /* working space for getRow */
  NALU_HYPRE_Int len_private;
  NALU_HYPRE_Int rowCheckedOut;
  NALU_HYPRE_Int *cval_private;
  NALU_HYPRE_Real *aval_private;

  /* row permutations to increase positive definiteness */
  NALU_HYPRE_Int *row_perm;

  /* for timing matvecs in experimental studies */
  NALU_HYPRE_Real time[MAT_DH_BINS];
  NALU_HYPRE_Real time_max[MAT_DH_BINS];
  NALU_HYPRE_Real time_min[MAT_DH_BINS];
  bool matvec_timing;

  /* used for MatVecs */
  NALU_HYPRE_Int          num_recv; 
  NALU_HYPRE_Int          num_send;   /* used in destructor */
  hypre_MPI_Request  *recv_req;
  hypre_MPI_Request  *send_req; 
  NALU_HYPRE_Real   *recvbuf, *sendbuf;  
  NALU_HYPRE_Int          *sendind;
  NALU_HYPRE_Int          sendlen;               
  NALU_HYPRE_Int          recvlen;               
  bool         matvecIsSetup;
  Numbering_dh numb;
  hypre_MPI_Status   *status;  

  bool debug;
};

extern void Mat_dhCreate(Mat_dh *mat);
extern void Mat_dhDestroy(Mat_dh mat);

extern void Mat_dhTranspose(Mat_dh matIN, Mat_dh *matOUT);
extern void Mat_dhMakeStructurallySymmetric(Mat_dh A);

  /* adopted from ParaSails, by Edmond Chow */
extern void Mat_dhMatVecSetup(Mat_dh mat);
extern void Mat_dhMatVecSetdown(Mat_dh mat);

/*========================================================================*/
/* notes: if not compiled with OpenMP, Mat_dhMatVec() and Mat_dhMatVec_omp()
          perform identically; similarly for Mat_dhMatVec_uni()
          and Mat_dhMatVec_uni_omp()
*/

extern void Mat_dhMatVec(Mat_dh mat, NALU_HYPRE_Real *lhs, NALU_HYPRE_Real *rhs);
  /* unthreaded MPI version */

extern void Mat_dhMatVec_omp(Mat_dh mat, NALU_HYPRE_Real *lhs, NALU_HYPRE_Real *rhs);
  /* OpenMP/MPI version */

extern void Mat_dhMatVec_uni(Mat_dh mat, NALU_HYPRE_Real *lhs, NALU_HYPRE_Real *rhs);
  /* unthreaded, single-task version */

extern void Mat_dhMatVec_uni_omp(Mat_dh mat, NALU_HYPRE_Real *lhs, NALU_HYPRE_Real *rhs);
  /* OpenMP/single primary task version */


extern NALU_HYPRE_Int Mat_dhReadNz(Mat_dh mat);

  /* for next five, SubdomainGraph_dh() may be NULL; if not null,
     caller must ensure it has been properly initialized;
     if not null, matrix is permuted before printing.

     note: use "-matlab" when calling Mat_dhPrintTriples, to
           insert small value in place of 0.

     Mat_dhPrintCSR only implemented for single cpu, no reordering.
   */
extern void Mat_dhPrintGraph(Mat_dh mat, SubdomainGraph_dh sg, FILE *fp);
extern void Mat_dhPrintRows(Mat_dh mat, SubdomainGraph_dh sg, FILE *fp);

extern void Mat_dhPrintCSR(Mat_dh mat, SubdomainGraph_dh sg, char *filename);
extern void Mat_dhPrintTriples(Mat_dh mat, SubdomainGraph_dh sg, char *filename);
extern void Mat_dhPrintBIN(Mat_dh mat, SubdomainGraph_dh sg, char *filename);

extern void Mat_dhReadCSR(Mat_dh *mat, char *filename);
extern void Mat_dhReadTriples(Mat_dh *mat, NALU_HYPRE_Int ignore, char *filename);
extern void Mat_dhReadBIN(Mat_dh *mat, char *filename);


extern void Mat_dhPermute(Mat_dh Ain, NALU_HYPRE_Int *pIN, Mat_dh *Bout);
  /* for single cpu only! */

extern void Mat_dhFixDiags(Mat_dh A);
  /* inserts diagonal if not explicitly present;
     sets diagonal value in row i to sum of absolute
     values of all elts in row i.
  */

extern void Mat_dhPrintDiags(Mat_dh A, FILE *fp);

extern void Mat_dhGetRow(Mat_dh B, NALU_HYPRE_Int globalRow, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val);
extern void Mat_dhRestoreRow(Mat_dh B, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val);

  /* partition matrix into "k" blocks.  User must free storage. */
extern void Mat_dhPartition(Mat_dh mat, NALU_HYPRE_Int k, NALU_HYPRE_Int **beg_rowOUT, 
                            NALU_HYPRE_Int **row_countOUT, NALU_HYPRE_Int **n2oOUT, NALU_HYPRE_Int **o2nOUT);




extern void Mat_dhZeroTiming(Mat_dh mat);
extern void Mat_dhReduceTiming(Mat_dh mat);


extern void Mat_dhRowPermute(Mat_dh);

extern void dldperm(NALU_HYPRE_Int job, NALU_HYPRE_Int n, NALU_HYPRE_Int nnz, NALU_HYPRE_Int colptr[], NALU_HYPRE_Int adjncy[],
                NALU_HYPRE_Real nzval[], NALU_HYPRE_Int *perm, NALU_HYPRE_Real u[], NALU_HYPRE_Real v[]);


#endif
