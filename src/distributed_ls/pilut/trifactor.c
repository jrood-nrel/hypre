/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * trifactor.c
 *
 * This file contains a number of fuction that are used in solving
 * the triangular systems resulting from the nalu_hypre_ILUT
 *
 * Started 11/13/95
 * George
 *
 * 7/8
 *  - seperate nalu_hypre_SetUpFactor from nalu_hypre_SetUpLUFactor and verify
 * 7/9
 *  - MPI support, adding to the comm structure
 *  - timing of the nalu_hypre_LDUSolve. The computation is very scalable, but the
 *    communication and sync is not. Partially this comes from sending
 *    zero length messages. I'll fix that.
 * 7/10
 *  - MPI and validation. Doesn't seem to work with Edinburgh, but
 *    I haven't the slightest idea why not. (Artifact of running
 *    along with shmem?)
 * 7/11
 *  - cleaned up code a little. Added timer macros.
 *
 * $Id$
 */

#include "ilu.h"
#include "DistributedMatrixPilutSolver.h"


/*************************************************************************
* This function performs the forward and backward substitution.
* It solves the system LDUx = b.
**************************************************************************/
void nalu_hypre_LDUSolve(DataDistType *ddist, FactorMatType *ldu, NALU_HYPRE_Real *x, NALU_HYPRE_Real *b,
                   nalu_hypre_PilutSolverGlobals *globals)
{
  NALU_HYPRE_Int ii, i, j, l, TAG;
  NALU_HYPRE_Int nlevels, snbrpes, rnbrpes;
  NALU_HYPRE_Int *perm, *iperm, *nnodes, *rowptr, *colind,
    *spes, *sptr, *sindex, *auxsptr, *rpes, *rdone, *rnum;
  NALU_HYPRE_Real *lx, *ux, *values, *dvalues, *gatherbuf, **raddr, xx;
  nalu_hypre_MPI_Status Status;
  nalu_hypre_MPI_Request *receive_requests;

  /* nalu_hypre_PrintLine("nalu_hypre_LDUSolve start", globals); */

  lnrows    = ddist->ddist_lnrows;
  perm      = ldu->perm;
  iperm     = ldu->iperm;
  nnodes    = ldu->nnodes;
  nlevels   = ldu->nlevels;
  dvalues   = ldu->dvalues;
  gatherbuf = ldu->gatherbuf;

  lx = ldu->lx;
  ux = ldu->ux;

  /******************************************************************
  * Do the L(lx) = b, first
  *******************************************************************/
  snbrpes = ldu->lcomm.snbrpes;
  spes    = ldu->lcomm.spes;
  sptr    = ldu->lcomm.sptr;
  sindex  = ldu->lcomm.sindex;
  auxsptr = ldu->lcomm.auxsptr;
  if( sptr != NULL ) nalu_hypre_memcpy_idx(auxsptr, sptr, snbrpes+1);

  rnbrpes = ldu->lcomm.rnbrpes;
  raddr   = ldu->lcomm.raddr;
  rpes    = ldu->lcomm.rpes;
  rdone   = ldu->lcomm.rdone;
  for (i=0; i<rnbrpes; i++)
    rdone[i] = 0 ;

  rowptr = ldu->lrowptr;
  colind = ldu->lcolind;
  values = ldu->lvalues;

#ifdef NALU_HYPRE_TIMING
  nalu_hypre_BeginTiming( globals->Ll_timer );
#endif

  /* Do the local first.
   * For forward substitution we do local+1st MIS == nnodes[1] (NOT [0]!) */
  for (i=0; i<nnodes[nalu_hypre_max(0,nalu_hypre_min(1,nlevels))]; i++) {
    xx = 0.0;
    for (j=rowptr[i]; j<rowptr[i+1]; j++)
      xx += values[j]*lx[colind[j]];
    lx[i] = b[perm[i]] - xx;
  }
#ifdef NALU_HYPRE_TIMING
  nalu_hypre_EndTiming( globals->Ll_timer );
#endif


  /* Allocate requests */
  receive_requests = nalu_hypre_CTAlloc( nalu_hypre_MPI_Request,  npes , NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_TIMING
  nalu_hypre_BeginTiming( globals->Lp_timer );
#endif
  /* Do the distributed next */
  for (ii=1; ii<nlevels; ii++) {
    /* make MPI LX tags unique for this level (so we don't have to sync) */
    TAG = (TAG_LDU_lx | ii);

    /* get number of recieves for this level */
    rnum = &(ldu->lcomm.rnum[(ii-1)*rnbrpes]) ;

    /* Recv the required lx elements from the appropriate processors */
    for (i=0; i<rnbrpes; i++) {
      if ( rnum[i] > 0 ) { /* Something to recv */
	nalu_hypre_MPI_Irecv( raddr[i]+rdone[i], rnum[i], nalu_hypre_MPI_REAL,
		  rpes[i], TAG, pilut_comm, &receive_requests[i] );

	rdone[i] += rnum[i] ;
      }
    }

    /* Send the required lx elements to the appropriate processors */
    for (i=0; i<snbrpes; i++) {
      if (sptr[i+1] > auxsptr[i]  &&  sindex[auxsptr[i]]<nnodes[ii]) { /* Something to send */
        for (j=auxsptr[i], l=0;   j<sptr[i+1] && sindex[j]<nnodes[ii];   j++, l++)
          gatherbuf[l] = lx[sindex[j]];

	nalu_hypre_MPI_Send( gatherbuf, l, nalu_hypre_MPI_REAL,
		  spes[i], TAG, pilut_comm );

        auxsptr[i] = j;
      }
    }

    /* Wait for receives */
    for (i=0; i<rnbrpes; i++) {
      if ( rnum[i] > 0 ) { /* Something to recv */
        nalu_hypre_MPI_Wait( &receive_requests[i], &Status);
      }
    }

    /* solve for this MIS set
     * by construction all remote lx elements needed are filled in */
    for (i=nnodes[ii]; i<nnodes[ii+1]; i++) {
      xx = 0.0;
      for (j=rowptr[i]; j<rowptr[i+1]; j++) {
        xx += values[j]*lx[colind[j]];
      }
      lx[i] = b[perm[i]] - xx;
    }
  }
#ifdef NALU_HYPRE_TIMING
  nalu_hypre_EndTiming( globals->Lp_timer );
#endif


  /******************************************************************
  * Do the U(ly) = (lx), next
  *******************************************************************/
  snbrpes = ldu->ucomm.snbrpes;
  spes    = ldu->ucomm.spes;
  sptr    = ldu->ucomm.sptr;
  sindex  = ldu->ucomm.sindex;
  auxsptr = ldu->ucomm.auxsptr;
  nalu_hypre_memcpy_idx(auxsptr, sptr, snbrpes+1);

  rnbrpes = ldu->ucomm.rnbrpes;
  raddr   = ldu->ucomm.raddr;
  rpes    = ldu->ucomm.rpes;
  rdone   = ldu->ucomm.rdone;
  for (i=0; i<rnbrpes; i++)
    rdone[i] = 0 ;

  rowptr = ldu->urowptr;
  colind = ldu->ucolind;
  values = ldu->uvalues;

#ifdef NALU_HYPRE_TIMING
  nalu_hypre_BeginTiming( globals->Up_timer );
#endif
  /* Do the distributed */
  for (ii=nlevels; ii>0; ii--) {
    /* Solve for this MIS set
     * by construction all remote lx elements needed are filled in */
    for (i=nnodes[ii]-1; i>=nnodes[ii-1]; i--) {
      xx = 0.0;
      for (j=rowptr[i]; j<rowptr[i+1]; j++)
        xx += values[j]*ux[colind[j]];
      ux[i] = dvalues[i]*(lx[i] - xx);
    }

    /* make MPI UX tags unique for this level (so we don't have to sync) */
    TAG = (TAG_LDU_ux | ii);

    /* get number of recieves for this level */
    rnum = &(ldu->ucomm.rnum[(ii-1)*rnbrpes]);

    /* Recv the required ux elements from the appropriate processors */
    for (i=0; i<rnbrpes; i++) {
      if ( rnum[i] > 0 ) { /* Something to recv */
	nalu_hypre_MPI_Irecv( raddr[i]+rdone[i], rnum[i], nalu_hypre_MPI_REAL,
		  rpes[i], TAG, pilut_comm, &receive_requests[ i ] );

	rdone[i] += rnum[i] ;
      }
    }

    /* Send the required ux elements to the appropriate processors */
    for (i=0; i<snbrpes; i++) {
      if (sptr[i+1] > auxsptr[i]  &&  sindex[auxsptr[i]]>=nnodes[ii-1]) { /* Something to send */
        for (j=auxsptr[i], l=0;   j<sptr[i+1] && sindex[j]>=nnodes[ii-1];   j++, l++)
          gatherbuf[l] = ux[sindex[j]];

	nalu_hypre_MPI_Send( gatherbuf, l, nalu_hypre_MPI_REAL,
		  spes[i], TAG, pilut_comm );

        auxsptr[i] = j;
      }
    }

    /* Finish receives */
    for (i=0; i<rnbrpes; i++) {
      if ( rnum[i] > 0 ) { /* Something to recv */
	nalu_hypre_MPI_Wait( &receive_requests[ i ], &Status );
      }
    }

  }



#ifdef NALU_HYPRE_TIMING
  nalu_hypre_EndTiming( globals->Up_timer );
#endif
#ifdef NALU_HYPRE_TIMING
  nalu_hypre_BeginTiming( globals->Ul_timer );
#endif
  /* Do the local next */
  for (i=nnodes[0]-1; i>=0; i--) {
    xx = 0.0;
    for (j=rowptr[i]; j<rowptr[i+1]; j++)
      xx += values[j]*ux[colind[j]];
    ux[i] = dvalues[i]*(lx[i] - xx);
  }
#ifdef NALU_HYPRE_TIMING
  nalu_hypre_EndTiming( globals->Ul_timer );
#endif


  /* Permute the solution to back to x */
  for (i=0; i<lnrows; i++)
    x[i] = ux[iperm[i]];

  nalu_hypre_TFree( receive_requests , NALU_HYPRE_MEMORY_HOST);
}


/*************************************************************************
* This function sets-up the communication parameters for the forward
* and backward substitution, and relabels the L and U matrices
**************************************************************************/
NALU_HYPRE_Int nalu_hypre_SetUpLUFactor(DataDistType *ddist, FactorMatType *ldu, NALU_HYPRE_Int maxnz,
                   nalu_hypre_PilutSolverGlobals *globals )
{
  NALU_HYPRE_Int maxsend;
  NALU_HYPRE_Int *petotal, *rind, *imap;

  petotal = nalu_hypre_idx_malloc(npes+1,       "nalu_hypre_SetUpLUFactor: petotal");
  rind    = nalu_hypre_idx_malloc(ddist->ddist_nrows, "nalu_hypre_SetUpLUFactor: rind"   );
  imap    = nalu_hypre_idx_malloc_init(ddist->ddist_nrows, -1, "nalu_hypre_SetUpLUFactor: imap");

  /* This is the global maximum for both L and U */
  maxsend = 0;

#ifdef NALU_HYPRE_TIMING
{
   NALU_HYPRE_Int Ltimer;

   Ltimer = nalu_hypre_InitializeTiming( "nalu_hypre_SetUpFactor for L" );

   nalu_hypre_BeginTiming( Ltimer );
#endif
  /* Work on L first */
  nalu_hypre_SetUpFactor( ddist, ldu, maxnz,   petotal, rind, imap, &maxsend,   true,
               globals  );
#ifdef NALU_HYPRE_TIMING
   nalu_hypre_EndTiming( Ltimer );
   /* nalu_hypre_FinalizeTiming( Ltimer ); */
}
#endif

#ifdef NALU_HYPRE_TIMING
 {
   NALU_HYPRE_Int Utimer;

   Utimer = nalu_hypre_InitializeTiming( "nalu_hypre_SetUpFactor for U" );

   nalu_hypre_BeginTiming( Utimer );
#endif
  /* Now work on U   */
  nalu_hypre_SetUpFactor( ddist, ldu, maxnz,   petotal, rind, imap, &maxsend,   false,
               globals );
#ifdef NALU_HYPRE_TIMING
   nalu_hypre_EndTiming( Utimer );
   /* nalu_hypre_FinalizeTiming( Utimer ); */
 }
#endif

  /* Allocate memory for the gather buffer. This is an overestimate */
  ldu->gatherbuf = nalu_hypre_fp_malloc(maxsend, "nalu_hypre_SetUpLUFactor: ldu->gatherbuf");

  /*nalu_hypre_free_multi(petotal, rind, imap, -1);*/
  nalu_hypre_TFree(petotal, NALU_HYPRE_MEMORY_HOST);
  nalu_hypre_TFree(rind, NALU_HYPRE_MEMORY_HOST);
  nalu_hypre_TFree(imap, NALU_HYPRE_MEMORY_HOST);

  return(0);
}

/*************************************************************************
* This function sets-up the communication parameters for the forward
* and backward substitution, and relabels the L and U matrices.
* This function is called twice--once for L and once for U. DoingL
* differentiates the two calls for the minor differences between them.
* These differences are marked by **** in comments
**************************************************************************/
void nalu_hypre_SetUpFactor(DataDistType *ddist, FactorMatType *ldu, NALU_HYPRE_Int maxnz,
		 NALU_HYPRE_Int *petotal, NALU_HYPRE_Int *rind, NALU_HYPRE_Int *imap,
		 NALU_HYPRE_Int *maxsendP, NALU_HYPRE_Int DoingL,
                   nalu_hypre_PilutSolverGlobals *globals )
{
  NALU_HYPRE_Int i, ii, j, k, l,
    nlevels, nrecv, nsend, snbrpes, rnbrpes;
  NALU_HYPRE_Int *rowdist, *sptr, *sindex, *spes, *rpes,
    *perm, *iperm, *newrowptr, *newcolind,
    *srowptr, *erowptr, *colind, *rnum ;
  NALU_HYPRE_Real *newvalues, *values, *x, **raddr;
  TriSolveCommType *TriSolveComm;
  nalu_hypre_MPI_Status Status;
  nalu_hypre_MPI_Request *receive_requests;
  nalu_hypre_MPI_Datatype MyColType_rnbr;

  /* data common to L and U */
  lnrows   = ddist->ddist_lnrows;
  nlevels  = ldu->nlevels;
  rowdist  = ddist->ddist_rowdist;
  firstrow = rowdist[mype];
  lastrow  = rowdist[mype+1];
  perm     = ldu->perm;
  iperm    = ldu->iperm;

  /**** choose between L and U data ****/
  srowptr = (DoingL  ?  ldu->lsrowptr  :  ldu->usrowptr);
  erowptr = (DoingL  ?  ldu->lerowptr  :  ldu->uerowptr);
  colind  = (DoingL  ?  ldu->lcolind   :  ldu->ucolind );
  values  = (DoingL  ?  ldu->lvalues   :  ldu->uvalues );
  TriSolveComm    = (DoingL  ?  &(ldu->lcomm)  :  &(ldu->ucomm));

  /* Determine the needed indices for L (U) */
  nrecv   = 0;
  for (ii=ldu->nnodes[0]; ii<lnrows; ii++) {
    i = perm[ii];
    for (j=srowptr[i]; j<erowptr[i]; j++) {
      k = colind[j];
      if ((k < firstrow || k >= lastrow) && imap[k] == -1) {
        rind[nrecv++] = k;
        imap[k] = -2;
      }
    }
  }
  nalu_hypre_sincsort_fast(nrecv, rind);

  /**** select between L and U "x" vectors ****/
  if ( DoingL ) {
    ldu->lxlen = nalu_hypre_GlobalSEMax(lnrows+nrecv, pilut_comm );
    x = ldu->lx  = nalu_hypre_fp_malloc_init(ldu->lxlen, 0, "nalu_hypre_SetUpFactor: ldu->lx");
  }
  else {
    ldu->uxlen = nalu_hypre_GlobalSEMax(lnrows+nrecv, pilut_comm);
    x = ldu->ux  = nalu_hypre_fp_malloc_init(ldu->uxlen, 0, "nalu_hypre_SetUpFactor: ldu->ux");
  }

  /* Determine processor boundaries */
  j = 0;
  for (i=0; i<npes; i++) {
    k = j;
    for (; j<nrecv; j++) {
      if (rind[j] >= rowdist[i+1])
        break;
    }
    petotal[i] = j-k;
  }

  /* Tell the processors how many elements I'll be sending */
  rnbrpes = 0;
  for (i=0; i<npes; i++) {
    if (petotal[i] > 0) {
      rnbrpes++;
    }
  }
  TriSolveComm->rnbrpes = rnbrpes ;

  nalu_hypre_MPI_Alltoall( petotal, 1, NALU_HYPRE_MPI_INT,
		lu_recv, 1, NALU_HYPRE_MPI_INT, pilut_comm );

  /* Determine to how many processors you will be sending data */
  snbrpes = 0;
  nsend = 0;
  for (i=0; i<npes; i++) {
    if (lu_recv[i] > 0) {
      snbrpes++;
      nsend += lu_recv[i];
      if ((*maxsendP) < lu_recv[i])
        (*maxsendP) = lu_recv[i];
    }
  }
  TriSolveComm->snbrpes = snbrpes;

  /* Allocate sufficient memory for the various data structures for TriSolveComm */
          TriSolveComm->auxsptr = nalu_hypre_idx_malloc(snbrpes+1, "nalu_hypre_SetUpFactor: TriSolveComm->auxsptr");
  spes  = TriSolveComm->spes    = nalu_hypre_idx_malloc(snbrpes,   "nalu_hypre_SetUpFactor: TriSolveComm->spes"   );
  sptr  = TriSolveComm->sptr    = nalu_hypre_idx_malloc(snbrpes+1, "nalu_hypre_SetUpFactor: TriSolveComm->sptr"   );
  sindex  = TriSolveComm->sindex    = nalu_hypre_idx_malloc(nalu_hypre_GlobalSEMax(nsend, pilut_comm), "nalu_hypre_SetUpFactor: TriSolveComm->sindex");

          TriSolveComm->rdone   = nalu_hypre_idx_malloc(rnbrpes,  "nalu_hypre_SetUpFactor: TriSolveComm->rpes");
  rpes  = TriSolveComm->rpes    = nalu_hypre_idx_malloc(rnbrpes,  "nalu_hypre_SetUpFactor: TriSolveComm->rpes" );
  raddr = TriSolveComm->raddr   = (NALU_HYPRE_Real**) nalu_hypre_mymalloc( sizeof(NALU_HYPRE_Real*)*(rnbrpes+1),
					       "nalu_hypre_SetUpFactor: TriSolveComm->raddr");

  /* Save send addresses, lengths, and construct spes */
  snbrpes = 0;
  for (i=0; i<npes; i++) {
    if (lu_recv[i] > 0) {
      spes[snbrpes] = i;
      sptr[snbrpes] = lu_recv[i];
      snbrpes++;

      lu_recv[i] = 0;
    }
  }
  nalu_hypre_assert( TriSolveComm->snbrpes == snbrpes );

  /* Create a sptr array into sindex */
  for (i=1; i<snbrpes; i++)
    sptr[i] += sptr[i-1];
  for (i=snbrpes; i>0; i--)
    sptr[i] = sptr[i-1];
  sptr[0] = 0;

  /* Allocate requests */
  receive_requests = nalu_hypre_CTAlloc( nalu_hypre_MPI_Request,  npes , NALU_HYPRE_MEMORY_HOST);

  /* Start asynchronous receives */
  for (i=0; i<snbrpes; i++) {
    nalu_hypre_MPI_Irecv( sindex+sptr[i], sptr[i+1]-sptr[i], NALU_HYPRE_MPI_INT,
	      spes[i], TAG_SetUp_rind, pilut_comm, &receive_requests[i] );
  }

  /* Send the rind sets to the processors */
  rnbrpes = 0;
  k = 0;
  for (i=0; i<npes; i++) {
    if (petotal[i] > 0) {
      nalu_hypre_MPI_Send( rind+k, petotal[i], NALU_HYPRE_MPI_INT ,
		i, TAG_SetUp_rind, pilut_comm );

      /* recv info for nalu_hypre_LDUSolve */
      raddr[rnbrpes] = x + k + lnrows;
      rpes [rnbrpes] = i;
      rnbrpes++;
      k += petotal[i];

      nalu_hypre_assert( k < ddist->ddist_nrows );
    }
  }
  /* this last one is to compute (raddr[i+1] - raddr[i]) */
  raddr[rnbrpes] = x + k + lnrows;
  nalu_hypre_assert( TriSolveComm->rnbrpes == rnbrpes );

  /* complete asynchronous receives */
  for (i=0; i<snbrpes; i++) {
    nalu_hypre_MPI_Wait( &receive_requests[i], &Status );
  }

  /* At this point, the set of indexes that you need to send to processors are
     stored in (sptr, sindex) */
  /* Apply the iperm[] onto the sindex in order to sort them according to MIS */
  for (i=0; i<nsend; i++) {
    nalu_hypre_CheckBounds(firstrow, sindex[i], lastrow, globals);
    sindex[i] = iperm[sindex[i]-firstrow];
  }

  /**** Go and do a segmented sort of the elements of the sindex.
   **** L is sorted increasing, U is sorted decreasing. ****/
  if ( DoingL ) {
    for (i=0; i<snbrpes; i++)
      nalu_hypre_sincsort_fast(sptr[i+1]-sptr[i], sindex+sptr[i]);
  }
  else {
    for (i=0; i<snbrpes; i++)
      nalu_hypre_sdecsort_fast(sptr[i+1]-sptr[i], sindex+sptr[i]);
  }

  /* Apply the perm[] onto the sindex to take it back to the original index space */
  for (i=0; i<nsend; i++) {
    nalu_hypre_CheckBounds(0, sindex[i], lnrows, globals);
    sindex[i] = perm[sindex[i]]+firstrow;
  }

  /* Start Recvs from the processors that send them to me */
  k = 0;
  for (i=0; i<npes; i++) {
    if (petotal[i] > 0) {
      nalu_hypre_MPI_Irecv( rind+k, petotal[i], NALU_HYPRE_MPI_INT,
	        i, TAG_SetUp_reord, pilut_comm, &receive_requests[i] );
      k += petotal[i];
    }
  }

  /* Write them back to the processors that send them to me */
  for (i=0; i<snbrpes; i++) {
    nalu_hypre_MPI_Send( sindex+sptr[i], sptr[i+1]-sptr[i], NALU_HYPRE_MPI_INT,
	      spes[i], TAG_SetUp_reord, pilut_comm );
  }

  /* Finish Recv  */
  for (i=0; i<npes; i++) {
    if (petotal[i] > 0) {
      nalu_hypre_MPI_Wait( &receive_requests[i], &Status );
    }
  }

  /* Apply the iperm[] onto the sindex for easy indexing during solution */
  for (i=0; i<nsend; i++)
    sindex[i] = iperm[sindex[i]-firstrow];

  /* Create imap array for relabeling L */
  for (i=0; i<nrecv; i++) {
    nalu_hypre_assert(imap[rind[i]] == -2);
    imap[rind[i]] = lnrows+i;
  }

  /* Construct the IMAP array of the locally stored rows */
  for (i=0; i<lnrows; i++)
    imap[firstrow+perm[i]] = i;

  /* rnum is a 2D array of nlevels rows of rnbrpes columns each */
  TriSolveComm->rnum = nalu_hypre_idx_malloc(nlevels * rnbrpes, "nalu_hypre_SetUpFactor: TriSolveComm->rnum");
        rnum = nalu_hypre_idx_malloc(nlevels, "nalu_hypre_SetUpFactor: rnum"      );
  nalu_hypre_memcpy_idx(TriSolveComm->auxsptr, sptr, snbrpes+1);

  /**** send the number of elements we are going to send to each PE.
   **** Note the inner for loop has no body, and L and U differ slightly.
   **** For L, rnum[nlevels-1] is undefined and rnum only has (nlevels-1) entries ****/
  for (i=0; i<snbrpes; i++) {
    if ( DoingL ) {
      for (ii=1; ii<nlevels; ii++) {
	for (j=TriSolveComm->auxsptr[i], l=0;   j<sptr[i+1] && sindex[j]<ldu->nnodes[ii];     j++, l++)
	  ;

	rnum[ii-1] = l;
	TriSolveComm->auxsptr[i] = j;
      }
      rnum[nlevels-1] = 0; /* never used */
    }
    else {
      for (ii=nlevels; ii>0; ii--) {
	for (j=TriSolveComm->auxsptr[i], l=0;   j<sptr[i+1] && sindex[j]>=ldu->nnodes[ii-1];  j++, l++)
	  ;

	rnum[ii-1] = l;
	TriSolveComm->auxsptr[i] = j;
      }
    }

    nalu_hypre_MPI_Send( rnum, nlevels, NALU_HYPRE_MPI_INT,
	      spes[i], TAG_SetUp_rnum, pilut_comm );
  }

  if (rnum) nalu_hypre_TFree(rnum,NALU_HYPRE_MEMORY_HOST);

  /* recieve data as columns rather than rows */
  nalu_hypre_MPI_Type_vector( nlevels, 1, rnbrpes, NALU_HYPRE_MPI_INT, &MyColType_rnbr );
  nalu_hypre_MPI_Type_commit( &MyColType_rnbr );

  /* receive each column */
  for (i=0; i<rnbrpes; i++) {
    nalu_hypre_MPI_Recv( TriSolveComm->rnum+i, 1, MyColType_rnbr,
	      rpes[i], TAG_SetUp_rnum, pilut_comm, &Status );
  }

  nalu_hypre_MPI_Type_free( &MyColType_rnbr );

  /* Now, go and create the renumbered L (U) that is also in CSR format */
  newrowptr = nalu_hypre_idx_malloc(lnrows+1,     "nalu_hypre_SetUpFactor: rowptr");
  newcolind = nalu_hypre_idx_malloc(lnrows*maxnz, "nalu_hypre_SetUpFactor: colind");
  newvalues =  nalu_hypre_fp_malloc(lnrows*maxnz, "nalu_hypre_SetUpFactor: values");

  newrowptr[0] = 0;
  k = 0;
  for (ii=0; ii<lnrows; ii++) {
    i = perm[ii];
    for (j=srowptr[i]; j<erowptr[i]; j++) {
      nalu_hypre_assert(imap[colind[j]] != -1);
      newcolind[k] = imap[colind[j]];
      newvalues[k] = values[j];
      k++;
    }
    newrowptr[ii+1] = k;
  }

  /**** Store new L (DU) into LDU ****/
  if ( DoingL ) {
    /* Free memory that stored the L so far and relink the data structures */
    /*nalu_hypre_free_multi(ldu->lsrowptr, ldu->lerowptr, ldu->lcolind, ldu->lvalues, -1);*/
    nalu_hypre_TFree(ldu->lsrowptr, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(ldu->lerowptr, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(ldu->lcolind, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(ldu->lvalues, NALU_HYPRE_MEMORY_HOST);
    ldu->lrowptr = newrowptr;
    ldu->lcolind = newcolind;
    ldu->lvalues = newvalues;
  }
  else {
    /* Use uvalues as a buffer to permute the dvalues */
    for (i=0; i<lnrows; i++)
      values[i] = ldu->dvalues[perm[i]];
    nalu_hypre_memcpy_fp(ldu->dvalues, values, lnrows);

    /* Free memory that stored the U so far and relink the data structures */
    /*nalu_hypre_free_multi(ldu->usrowptr, ldu->uerowptr, ldu->ucolind, ldu->uvalues, -1);*/
    nalu_hypre_TFree(ldu->usrowptr, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(ldu->uerowptr, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(ldu->ucolind, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(ldu->uvalues, NALU_HYPRE_MEMORY_HOST);
    ldu->urowptr = newrowptr;
    ldu->ucolind = newcolind;
    ldu->uvalues = newvalues;
  }

  /* clean up memory */
  nalu_hypre_TFree(receive_requests, NALU_HYPRE_MEMORY_HOST);

  /* Reset the imap by only touching the appropriate elements */
  for (i=0; i<nrecv; i++)
    imap[rind[i]] = -1;
  for (i=0; i<lnrows; i++)
    imap[firstrow+i] = -1;
}

