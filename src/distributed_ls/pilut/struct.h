/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef STRUCT_H
#define STRUCT_H

/*
 * struct.h
 *
 * This file contains data structures for ILU routines.
 *
 * Started 9/26/95
 * George
 *
 * 7/8
 *  - change to generic NALU_HYPRE_Int and NALU_HYPRE_Real (in all files) and verified
 *  - added rrowlen to rmat and verified
 * 7/9
 *  - add recv info to the LDU communication struct TriSolveCommType
 * 7/29
 *  - add maxntogo and remove unused out and address buffers from cinfo
 *  - rearranged all structures to have ptrs first, then ints, ints, structs.
 *    This is under the assumption that that is the most likely order
 *    for things to be natural word length, so reduces padding.
 *
 * $Id$
 */

#ifndef __cplusplus
#ifndef true
# define true  1
# define false 0
#endif

#ifndef bool
# ifdef Boolean
   typedef Boolean bool;
# else
   typedef unsigned char bool;
# endif
#endif
#endif
 
/*************************************************************************
* This data structure holds the data distribution
**************************************************************************/
struct distdef {
  NALU_HYPRE_Int ddist_nrows;		/* The order of the distributed matrix */
  NALU_HYPRE_Int ddist_lnrows;           /* The local number of rows */
  NALU_HYPRE_Int *ddist_rowdist;	/* How the rows are distributed among processors */
};

typedef struct distdef DataDistType;

#define DataDistTypeNrows(data_dist)      ((data_dist)->    ddist_nrows)
#define DataDistTypeLnrows(data_dist)     ((data_dist)->   ddist_lnrows)
#define DataDistTypeRowdist(data_dist)    ((data_dist)->  ddist_rowdist)

/*************************************************************************
* The following data structure stores info for a communication phase during
* the triangular solvers.
**************************************************************************/
struct cphasedef {
  NALU_HYPRE_Real **raddr;	/* A rnbrpes+1 list of addresses to recv data into */

  NALU_HYPRE_Int *spes;	/* A snbrpes    list of PEs to send data */
  NALU_HYPRE_Int *sptr;	/* An snbrpes+1 list indexing sindex for each spes[i] */
  NALU_HYPRE_Int *sindex;	/* The packets to send per PE */
  NALU_HYPRE_Int *auxsptr;	/* Auxiliary send ptr, used at intermediate points */

  NALU_HYPRE_Int *rpes;	/* A rnbrpes   list of PEs to recv data */
  NALU_HYPRE_Int *rdone;	/* A rnbrpes   list of # elements recv'd in this hypre_LDUSolve */
  NALU_HYPRE_Int *rnum;        /* A nlevels x npes array of the number of elements to recieve */

  NALU_HYPRE_Int snbrpes;		/* The total number of neighboring PEs (to send to)   */
  NALU_HYPRE_Int rnbrpes;		/* The total number of neighboring PEs (to recv from) */
};

typedef struct cphasedef TriSolveCommType;


/*************************************************************************
* This data structure holds the factored matrix
**************************************************************************/
struct factormatdef {
  NALU_HYPRE_Int *lsrowptr;	/* Pointers to the locally stored rows start */
  NALU_HYPRE_Int *lerowptr;	/* Pointers to the locally stored rows end */
  NALU_HYPRE_Int *lcolind;	/* Array of column indices of lnrows */
   NALU_HYPRE_Real *lvalues;	/* Array of locally stored values */
  NALU_HYPRE_Int *lrowptr;

  NALU_HYPRE_Int *usrowptr;	/* Pointers to the locally stored rows start */
  NALU_HYPRE_Int *uerowptr;	/* Pointers to the locally stored rows end */
  NALU_HYPRE_Int *ucolind;	/* Array of column indices of lnrows */
   NALU_HYPRE_Real *uvalues;	/* Array of locally stored values */
  NALU_HYPRE_Int *urowptr;

  NALU_HYPRE_Real *dvalues;	/* Diagonal values */

  NALU_HYPRE_Real *nrm2s;	/* Array of the 2-norms of the rows for tolerance testing */

  NALU_HYPRE_Int *perm;		/* perm and invperm arrays for factorization */
  NALU_HYPRE_Int *iperm;

  /* Communication info for triangular system solution */
  NALU_HYPRE_Real *gatherbuf;            /* maxsend*snbrpes buffer for sends */

  NALU_HYPRE_Real *lx;
  NALU_HYPRE_Real *ux;
  NALU_HYPRE_Int lxlen, uxlen;

  NALU_HYPRE_Int nlevels;			/* The number of reductions performed */
  NALU_HYPRE_Int nnodes[MAXNLEVEL];	/* The number of nodes at each reduction level */

  TriSolveCommType lcomm;	/* Communication info during the Lx=y solve */
  TriSolveCommType ucomm;	/* Communication info during the Ux=y solve */
};

typedef struct factormatdef FactorMatType;


/*************************************************************************
* This data structure holds the reduced matrix
**************************************************************************/
struct reducematdef {
  NALU_HYPRE_Int *rmat_rnz;		/* Pointers to the locally stored rows */
  NALU_HYPRE_Int *rmat_rrowlen;	/* Length allocated for each row */
  NALU_HYPRE_Int **rmat_rcolind;	/* Array of column indices of lnrows */
   NALU_HYPRE_Real **rmat_rvalues;	/* Array of locally stored values */

  NALU_HYPRE_Int rmat_ndone;	     /* The number of vertices factored so far */
  NALU_HYPRE_Int rmat_ntogo;  /* The number of vertices not factored. This is the size of rmat */
  NALU_HYPRE_Int rmat_nlevel;	     /* The number of reductions performed so far */
};

typedef struct reducematdef ReduceMatType;



/*************************************************************************
* This data structure stores information about the send in each phase 
* of parallel hypre_ILUT
**************************************************************************/
struct comminfodef {
  NALU_HYPRE_Real *gatherbuf;	/* Assembly buffer for sending colind & values */

  NALU_HYPRE_Int *incolind;	/* Receive buffer for colind */
   NALU_HYPRE_Real *invalues;	/* Receive buffer for values */

  NALU_HYPRE_Int *rnbrind;	/* The neighbor processors */
  NALU_HYPRE_Int *rrowind;	/* The indices that are received */
  NALU_HYPRE_Int *rnbrptr;	/* Array of size rnnbr+1 into rrowind */

  NALU_HYPRE_Int *snbrind;	/* The neighbor processors */
  NALU_HYPRE_Int *srowind;	/* The indices that are sent */
  NALU_HYPRE_Int *snbrptr;	/* Array of size snnbr+1 into srowind */

  NALU_HYPRE_Int maxnsend;		/* The maximum number of rows being sent */
  NALU_HYPRE_Int maxnrecv;		/* The maximum number of rows being received */
  NALU_HYPRE_Int maxntogo;         /* The maximum number of rows left on any PE */

  NALU_HYPRE_Int rnnbr;		/* Number of neighbor processors */
  NALU_HYPRE_Int snnbr;		/* Number of neighbor processors */
};

typedef struct comminfodef CommInfoType;


/*************************************************************************
* The following data structure stores communication info for mat-vec
**************************************************************************/
struct mvcommdef {
  NALU_HYPRE_Int *spes;	/* Array of PE numbers */
  NALU_HYPRE_Int *sptr;	/* Array of send indices */
  NALU_HYPRE_Int *sindex;	/* Array that stores the actual indices */

  NALU_HYPRE_Int *rpes;
  NALU_HYPRE_Real **raddr;

  NALU_HYPRE_Real *bsec;		/* Stores the actual b vector */
  NALU_HYPRE_Real *gatherbuf;	/* Used to gather the outgoing packets */
  NALU_HYPRE_Int *perm;	/* Used to map the LIND back to GIND */

  NALU_HYPRE_Int snpes;		/* Number of send PE's */
  NALU_HYPRE_Int rnpes;
};

typedef struct mvcommdef MatVecCommType;


/*************************************************************************
* The following data structure stores key-value pair
**************************************************************************/
struct KeyValueType {
  NALU_HYPRE_Int key;
  NALU_HYPRE_Int val;
};

typedef struct KeyValueType KeyValueType;


#endif
