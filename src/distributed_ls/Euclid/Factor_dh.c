/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_Euclid.h"
/* #include "Factor_dh.h" */
/* #include "Vec_dh.h" */
/* #include "Mat_dh.h" */
/* #include "SubdomainGraph_dh.h" */
/* #include "TimeLog_dh.h" */
/* #include "Mem_dh.h" */
/* #include "Numbering_dh.h" */
/* #include "Hash_i_dh.h" */
/* #include "Parser_dh.h" */
/* #include "mat_dh_private.h" */
/* #include "getRow_dh.h" */
/* #include "Euclid_dh.h" */
/* #include "io_dh.h" */

/* suppress compiler complaints */
void Factor_dh_junk(void)
{
}

static void adjust_bj_private(Factor_dh mat);
static void unadjust_bj_private(Factor_dh mat);


#undef __FUNC__
#define __FUNC__ "Factor_dhCreate"
void Factor_dhCreate(Factor_dh *mat)
{
  START_FUNC_DH
  NALU_HYPRE_Int i;
  struct _factor_dh* tmp; 

  if (np_dh > MAX_MPI_TASKS) {
    SET_V_ERROR("you must change MAX_MPI_TASKS and recompile!");
  }

  tmp = (struct _factor_dh*)MALLOC_DH(sizeof(struct _factor_dh)); CHECK_V_ERROR;
  *mat = tmp;

  tmp->m = 0;
  tmp->n = 0;
  tmp->id = myid_dh;
  tmp->beg_row = 0; 
  tmp->first_bdry = 0;
  tmp->bdry_count = 0;
  tmp->blockJacobi = false;

  tmp->rp = NULL;
  tmp->cval = NULL;
  tmp->aval = NULL;
  tmp->fill = NULL;
  tmp->diag = NULL;
  tmp->alloc = 0;

  tmp->work_y_lo = tmp->work_x_hi = NULL;
  tmp->sendbufLo = tmp->sendbufHi = NULL;
  tmp->sendindLo = tmp->sendindHi = NULL;
  tmp->num_recvLo = tmp->num_recvHi = 0;
  tmp->num_sendLo = tmp->num_sendHi = 0;
  tmp->sendlenLo = tmp->sendlenHi = 0;

  tmp->solveIsSetup = false;
  tmp->numbSolve = NULL;

  tmp->debug = Parser_dhHasSwitch(parser_dh, "-debug_Factor");
  
  /* initialize MPI request to null */
  for(i=0; i<MAX_MPI_TASKS; i++)
  {
     tmp->recv_reqLo[i] = nalu_hypre_MPI_REQUEST_NULL;
     tmp->recv_reqHi[i] = nalu_hypre_MPI_REQUEST_NULL;
     tmp->send_reqLo[i] = nalu_hypre_MPI_REQUEST_NULL;
     tmp->send_reqHi[i] = nalu_hypre_MPI_REQUEST_NULL;
     tmp->requests[i] = nalu_hypre_MPI_REQUEST_NULL;
  }
/*  Factor_dhZeroTiming(tmp); CHECK_V_ERROR; */
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Factor_dhDestroy"
void Factor_dhDestroy(Factor_dh mat)
{
  START_FUNC_DH
  NALU_HYPRE_Int i;

  if (mat->rp != NULL) { FREE_DH(mat->rp); CHECK_V_ERROR; }
  if (mat->cval != NULL) { FREE_DH(mat->cval); CHECK_V_ERROR; }
  if (mat->aval != NULL) { FREE_DH(mat->aval); CHECK_V_ERROR; }
  if (mat->diag != NULL) { FREE_DH(mat->diag); CHECK_V_ERROR; }
  if (mat->fill != NULL) { FREE_DH(mat->fill); CHECK_V_ERROR; }

  if (mat->work_y_lo != NULL) { FREE_DH(mat->work_y_lo); CHECK_V_ERROR; }
  if (mat->work_x_hi != NULL) { FREE_DH(mat->work_x_hi); CHECK_V_ERROR; }
  if (mat->sendbufLo != NULL) { FREE_DH(mat->sendbufLo); CHECK_V_ERROR; }
  if (mat->sendbufHi != NULL) { FREE_DH(mat->sendbufHi); CHECK_V_ERROR; }
  if (mat->sendindLo != NULL) { FREE_DH(mat->sendindLo); CHECK_V_ERROR; }
  if (mat->sendindHi != NULL) { FREE_DH(mat->sendindHi); CHECK_V_ERROR; }

  if (mat->numbSolve != NULL) { Numbering_dhDestroy(mat->numbSolve); CHECK_V_ERROR; }
  
  /* cleanup MPI requests */
  for(i=0; i<MAX_MPI_TASKS; i++)
  {
     if(mat->recv_reqLo[i] != nalu_hypre_MPI_REQUEST_NULL) nalu_hypre_MPI_Request_free(&(mat->recv_reqLo[i]));
     if(mat->recv_reqHi[i] != nalu_hypre_MPI_REQUEST_NULL) nalu_hypre_MPI_Request_free(&(mat->recv_reqHi[i]));     
     if(mat->send_reqLo[i] != nalu_hypre_MPI_REQUEST_NULL) nalu_hypre_MPI_Request_free(&(mat->send_reqLo[i]));
     if(mat->send_reqHi[i] != nalu_hypre_MPI_REQUEST_NULL) nalu_hypre_MPI_Request_free(&(mat->send_reqHi[i]));
     if(mat->requests[i] != nalu_hypre_MPI_REQUEST_NULL) nalu_hypre_MPI_Request_free(&(mat->requests[i]));     
  }
  FREE_DH(mat); CHECK_V_ERROR; 
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "create_fake_mat_private"
static void create_fake_mat_private(Factor_dh mat, Mat_dh *matFakeIN)
{
  START_FUNC_DH
  Mat_dh matFake;
  Mat_dhCreate(matFakeIN); CHECK_V_ERROR;
  matFake = *matFakeIN;
  matFake->m = mat->m;
  matFake->n = mat->n;
  matFake->rp = mat->rp;
  matFake->cval = mat->cval;
  matFake->aval = mat->aval;
  matFake->beg_row = mat->beg_row;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "destroy_fake_mat_private"
static void destroy_fake_mat_private(Mat_dh matFake)
{
  START_FUNC_DH
  matFake->rp = NULL;
  matFake->cval = NULL;
  matFake->aval = NULL;
  Mat_dhDestroy(matFake); CHECK_V_ERROR;
  END_FUNC_DH
}



#undef __FUNC__
#define __FUNC__ "Factor_dhReadNz"
NALU_HYPRE_Int Factor_dhReadNz(Factor_dh mat)
{
  START_FUNC_DH
  NALU_HYPRE_Int ierr, retval = mat->rp[mat->m];
  NALU_HYPRE_Int nz = retval;
  ierr = nalu_hypre_MPI_Allreduce(&nz, &retval, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm_dh); CHECK_MPI_ERROR(ierr);
  END_FUNC_VAL(retval)
}



#undef __FUNC__
#define __FUNC__ "Factor_dhPrintRows"
void Factor_dhPrintRows(Factor_dh mat, FILE *fp)
{
  START_FUNC_DH
  NALU_HYPRE_Int beg_row = mat->beg_row;
  NALU_HYPRE_Int m = mat->m, i, j;
  bool noValues;

  noValues = (Parser_dhHasSwitch(parser_dh, "-noValues"));
  if (mat->aval == NULL) noValues = true;

  if (mat->blockJacobi) { adjust_bj_private(mat); CHECK_V_ERROR; }

  nalu_hypre_fprintf(fp, "\n----------------------- Factor_dhPrintRows ------------------\n");
  if (mat->blockJacobi) {
    nalu_hypre_fprintf(fp, "@@@ Block Jacobi ILU; adjusted values from zero-based @@@\n");
  }

  for (i=0; i<m; ++i) {
    nalu_hypre_fprintf(fp, "%i :: ", 1+i+beg_row);
    for (j=mat->rp[i]; j<mat->rp[i+1]; ++j) {
      if (noValues) {
        nalu_hypre_fprintf(fp, "%i ", 1+mat->cval[j]);
      } else {
        nalu_hypre_fprintf(fp, "%i,%g ; ", 1+mat->cval[j], mat->aval[j]);
      }
    }
    nalu_hypre_fprintf(fp, "\n");
  }

  if (mat->blockJacobi) { unadjust_bj_private(mat); CHECK_V_ERROR; }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Factor_dhPrintDiags"
void Factor_dhPrintDiags(Factor_dh mat, FILE *fp)
{
  START_FUNC_DH
  NALU_HYPRE_Int beg_row = mat->beg_row;
  NALU_HYPRE_Int m = mat->m, i, pe, *diag = mat->diag;
  REAL_DH *aval = mat->aval; 

  
  fprintf_dh(fp, "\n----------------------- Factor_dhPrintDiags ------------------\n");
  fprintf_dh(fp, "(grep for 'ZERO')\n");

  for (pe=0; pe<np_dh; ++pe) {
    nalu_hypre_MPI_Barrier(comm_dh); 
    if (mat->id == pe) {
      nalu_hypre_fprintf(fp, "----- subdomain: %i  processor: %i\n", pe, myid_dh);
      for (i=0; i<m; ++i) {
        REAL_DH val = aval[diag[i]];
        if (val) {
          nalu_hypre_fprintf(fp, "%i %g\n", i+1+beg_row, aval[diag[i]]);
        } else {
          nalu_hypre_fprintf(fp, "%i %g ZERO\n", i+1+beg_row, aval[diag[i]]);
        }
      }
    }
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Factor_dhPrintGraph"
void Factor_dhPrintGraph(Factor_dh mat, char *filename)
{
  START_FUNC_DH
  FILE *fp;
  NALU_HYPRE_Int i, j, m = mat->m, *work, *rp = mat->rp, *cval = mat->cval;

  if (np_dh > 1) SET_V_ERROR("only implemented for single mpi task");

  work = (NALU_HYPRE_Int*)MALLOC_DH(m*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;

  fp=openFile_dh(filename, "w"); CHECK_V_ERROR;

  for (i=0; i<m; ++i) {
    for (j=0; j<m; ++j) work[j] = 0;
    for (j=rp[i]; j<rp[i]; ++j) work[cval[j]] = 1;

    for (j=0; j<m; ++j) {
      if (work[j]) {
        nalu_hypre_fprintf(fp, " x ");
      } else {
        nalu_hypre_fprintf(fp, "   ");
      }
    }
    nalu_hypre_fprintf(fp, "\n");
  }

  closeFile_dh(fp); CHECK_V_ERROR;

  FREE_DH(work);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Factor_dhPrintTriples"
void Factor_dhPrintTriples(Factor_dh mat, char *filename)
{
  START_FUNC_DH
  NALU_HYPRE_Int pe, i, j;
  NALU_HYPRE_Int m = mat->m, *rp = mat->rp;
  NALU_HYPRE_Int beg_row = mat->beg_row;
  REAL_DH *aval = mat->aval;
  bool noValues;
  FILE *fp;

  if (mat->blockJacobi) { adjust_bj_private(mat); CHECK_V_ERROR; }

  noValues = (Parser_dhHasSwitch(parser_dh, "-noValues"));
  if (noValues) aval = NULL;

  for (pe=0; pe<np_dh; ++pe) {
    nalu_hypre_MPI_Barrier(comm_dh); 
    if (mat->id == pe) {
      if (pe == 0) { 
        fp=openFile_dh(filename, "w"); CHECK_V_ERROR;
      } 
      else { 
        fp=openFile_dh(filename, "a"); CHECK_V_ERROR;
      }

      for (i=0; i<m; ++i) {
        for (j=rp[i]; j<rp[i+1]; ++j) {
          if (noValues) {
            nalu_hypre_fprintf(fp, "%i %i\n", 1+i+beg_row, 1+mat->cval[j]);
          } else {
            nalu_hypre_fprintf(fp, TRIPLES_FORMAT, 
                        1+i+beg_row, 1+mat->cval[j], aval[j]);
          }
        }
      }
      closeFile_dh(fp); CHECK_V_ERROR;
    }
  }

  if (mat->blockJacobi) { unadjust_bj_private(mat); CHECK_V_ERROR; }
  END_FUNC_DH
}

/*--------------------------------------------------------------------------------
 * Functions to setup the matrix for triangular solves.  These are similar to
 * MatVecSetup(), except that there are two cases: subdomains ordered lower than
 * ourselves, and subdomains ordered higher than ourselves.  This SolveSetup
 * is used for Parallel ILU (PILU).  The following are adopted/modified from
 * Edmond Chow's ParaSails
 *--------------------------------------------------------------------------------*/

/* adopted from Edmond Chow's ParaSails */

/* 1. start receives of node data to be received from other processors;
   2. send to other processors the list of nodes this processor needs
      to receive from them.
   Returns: the number of processors from whom nodes will be received.
*/
#undef __FUNC__
#define __FUNC__ "setup_receives_private"
static NALU_HYPRE_Int setup_receives_private(Factor_dh mat, NALU_HYPRE_Int *beg_rows, NALU_HYPRE_Int *end_rows, 
                                  NALU_HYPRE_Real *recvBuf, nalu_hypre_MPI_Request *req,
                                  NALU_HYPRE_Int *reqind, NALU_HYPRE_Int reqlen, 
                                  NALU_HYPRE_Int *outlist, bool debug)
{
  START_FUNC_DH
  NALU_HYPRE_Int i, j, this_pe, num_recv = 0;
  nalu_hypre_MPI_Request request;

  if (debug) {
    nalu_hypre_fprintf(logFile, "\nFACT ========================================================\n");
    nalu_hypre_fprintf(logFile, "FACT STARTING: setup_receives_private\n");
  }

  for (i=0; i<reqlen; i=j) { /* j is set below */ 
    /* determine the processor that owns the row with index reqind[i] */
    this_pe = mat_find_owner(beg_rows, end_rows, reqind[i]); CHECK_ERROR(-1);

    /* Figure out other rows we need from this_pe */
    for (j=i+1; j<reqlen; j++) {
      NALU_HYPRE_Int idx = reqind[j];
      if (idx < beg_rows[this_pe] || idx >= end_rows[this_pe]) {
        break;
      }
    }

    if (debug) {
      NALU_HYPRE_Int k;
      nalu_hypre_fprintf(logFile, "FACT need nodes from P_%i: ", this_pe);
      for (k=i; k<j; ++k) nalu_hypre_fprintf(logFile, "%i ", 1+reqind[k]);
      nalu_hypre_fprintf(logFile,"\n");
    }

    /* Record the number of number of indices needed from this_pe */
    outlist[this_pe] = j-i;

    /* Request rows in reqind[i..j-1] */
    /* Note: the receiving processor, this_pe, doesn't yet know
       about the incoming request, hence, can't set up a matching
       receive; this matching receive will be started later,
       in setup_sends_private.
    */
    nalu_hypre_MPI_Isend(reqind+i, j-i, NALU_HYPRE_MPI_INT, this_pe, 444, comm_dh, &request); 
    nalu_hypre_MPI_Request_free(&request); 

    /* set up persistent comms for receiving the values from this_pe */
    nalu_hypre_MPI_Recv_init(recvBuf+i, j-i, nalu_hypre_MPI_REAL, this_pe, 555,
                        comm_dh, req+num_recv); 
    ++num_recv;
  }

  END_FUNC_VAL(num_recv);
}

/*
   1. start receive to get list of nodes that this processor
      needs to send to other processors
   2. start persistent comms to send the data
*/
#undef __FUNC__
#define __FUNC__ "setup_sends_private"
static void setup_sends_private(Factor_dh mat, NALU_HYPRE_Int *inlist, 
                                  NALU_HYPRE_Int *o2n_subdomain, bool debug)
{
  START_FUNC_DH
  NALU_HYPRE_Int         i, jLo, jHi, sendlenLo, sendlenHi, first = mat->beg_row;
  nalu_hypre_MPI_Request *requests = mat->requests, *sendReq;
  nalu_hypre_MPI_Status  *statuses = mat->status;
  bool        isHigher;
  NALU_HYPRE_Int         *rcvBuf;
  NALU_HYPRE_Real  *sendBuf;
  NALU_HYPRE_Int         myidNEW = o2n_subdomain[myid_dh];
  NALU_HYPRE_Int         count;

  if (debug) {
    nalu_hypre_fprintf(logFile, "FACT \nSTARTING: setup_sends_private\n");
  }

  /* Determine size of and allocate sendbuf and sendind */
  sendlenLo = sendlenHi = 0;
  for (i=0; i<np_dh; i++) {
    if (inlist[i]) {
      if (o2n_subdomain[i] < myidNEW) { sendlenLo += inlist[i]; }
      else                            { sendlenHi += inlist[i]; }
    }
  }

  mat->sendlenLo = sendlenLo;
  mat->sendlenHi = sendlenHi;
  mat->sendbufLo = (NALU_HYPRE_Real *)MALLOC_DH(sendlenLo * sizeof(NALU_HYPRE_Real)); CHECK_V_ERROR;
  mat->sendbufHi = (NALU_HYPRE_Real *)MALLOC_DH(sendlenHi * sizeof(NALU_HYPRE_Real)); CHECK_V_ERROR;
  mat->sendindLo = (NALU_HYPRE_Int *)MALLOC_DH(sendlenLo * sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
  mat->sendindHi = (NALU_HYPRE_Int *)MALLOC_DH(sendlenHi * sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;

  count = 0;  /* number of calls to nalu_hypre_MPI_Irecv() */
  jLo = jHi = 0;
  mat->num_sendLo = 0;
  mat->num_sendHi = 0;
  for (i=0; i<np_dh; i++) {
    if (inlist[i]) {
      isHigher = (o2n_subdomain[i] < myidNEW) ? false : true;

      /* Post receive for the actual indices */
      if (isHigher) {
        rcvBuf = &mat->sendindHi[jHi];
        sendBuf = &mat->sendbufHi[jHi];
        sendReq = &mat->send_reqHi[mat->num_sendHi];
        mat->num_sendHi++;
        jHi += inlist[i];
      } else {
        rcvBuf = &mat->sendindLo[jLo];
        sendBuf = &mat->sendbufLo[jLo];
        sendReq = &mat->send_reqLo[mat->num_sendLo];
        mat->num_sendLo++;
        jLo += inlist[i];
      }

      /* matching receive, for list of unknowns that will be sent,
         during the triangular solves, from ourselves to P_i
       */
      nalu_hypre_MPI_Irecv(rcvBuf, inlist[i], NALU_HYPRE_MPI_INT, i, 444, comm_dh, requests+count); 
      ++count;

      /* Set up the send */
      nalu_hypre_MPI_Send_init(sendBuf, inlist[i], nalu_hypre_MPI_REAL, i, 555, comm_dh, sendReq); 
    }
  }

  /* note: count = mat->num_sendLo = mat->num_sendHi */
  nalu_hypre_MPI_Waitall(count, requests, statuses); 

  if (debug) {
    NALU_HYPRE_Int j;
    jLo = jHi = 0;

    nalu_hypre_fprintf(logFile, "\nFACT columns that I must send to other subdomains:\n");
    for (i=0; i<np_dh; i++) {
      if (inlist[i]) {
        isHigher = (o2n_subdomain[i] < myidNEW) ? false : true;
        if (isHigher) {
          rcvBuf = &mat->sendindHi[jHi];
          jHi += inlist[i];
        } else {
          rcvBuf = &mat->sendindLo[jLo];
          jLo += inlist[i];
        }

        nalu_hypre_fprintf(logFile, "FACT  send to P_%i: ", i);
        for (j=0; j<inlist[i]; ++j) nalu_hypre_fprintf(logFile, "%i ", rcvBuf[j]+1);
        nalu_hypre_fprintf(logFile, "\n");
      }
    }
  }

  /* convert global indices to local indices */
  /* these are all indices on this processor */
  for (i=0; i<mat->sendlenLo; i++) mat->sendindLo[i] -= first;
  for (i=0; i<mat->sendlenHi; i++) mat->sendindHi[i] -= first;
  END_FUNC_DH
}



#undef __FUNC__ 
#define __FUNC__ "Factor_dhSolveSetup"
void Factor_dhSolveSetup(Factor_dh mat, SubdomainGraph_dh sg)
{
  START_FUNC_DH
  NALU_HYPRE_Int *outlist, *inlist;
  NALU_HYPRE_Int i, row, *rp = mat->rp, *cval = mat->cval;
  Numbering_dh numb;
  NALU_HYPRE_Int m = mat->m;
  /* NALU_HYPRE_Int firstLocalRow = mat->beg_row; */
  NALU_HYPRE_Int *beg_rows = sg->beg_rowP, *row_count = sg->row_count, *end_rows;
  Mat_dh matFake;
  bool debug = false;
  NALU_HYPRE_Real *recvBuf;

  if (mat->debug && logFile != NULL) debug = true;

  end_rows = (NALU_HYPRE_Int *)MALLOC_DH(np_dh*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
  outlist = (NALU_HYPRE_Int *)MALLOC_DH(np_dh*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
  inlist  = (NALU_HYPRE_Int *)MALLOC_DH(np_dh*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<np_dh; ++i) {
    inlist[i] = 0;
    outlist[i] = 0;
    end_rows[i] = beg_rows[i]+row_count[i];
  }

  /* Create Numbering object */
  create_fake_mat_private(mat, &matFake); CHECK_V_ERROR;
  Numbering_dhCreate(&(mat->numbSolve)); CHECK_V_ERROR;
  numb = mat->numbSolve;
  Numbering_dhSetup(numb, matFake); CHECK_V_ERROR;
  destroy_fake_mat_private(matFake); CHECK_V_ERROR;

  if (debug) {
    nalu_hypre_fprintf(stderr, "Numbering_dhSetup completed\n");
  }

  /* Allocate recvbuf; recvbuf has numlocal entries saved for local part of x */
  i = m+numb->num_ext;
  mat->work_y_lo = (NALU_HYPRE_Real*)MALLOC_DH(i*sizeof(NALU_HYPRE_Real)); CHECK_V_ERROR;
  mat->work_x_hi = (NALU_HYPRE_Real*)MALLOC_DH(i*sizeof(NALU_HYPRE_Real)); CHECK_V_ERROR;
  if (debug) {
    nalu_hypre_fprintf(logFile, "FACT num_extLo= %i  num_extHi= %i\n", numb->num_extLo, numb->num_extHi);
  }

  mat->num_recvLo = 0;
  mat->num_recvHi = 0;
  if (numb->num_extLo) {
    recvBuf = mat->work_y_lo + m;
    mat->num_recvLo = setup_receives_private(mat, beg_rows, end_rows, 
                             recvBuf, mat->recv_reqLo,
                             numb->idx_extLo, numb->num_extLo,
                             outlist, debug); CHECK_V_ERROR;

  }

  if (numb->num_extHi) {
    recvBuf = mat->work_x_hi + m + numb->num_extLo;
    mat->num_recvHi = setup_receives_private(mat, beg_rows, end_rows,
                            recvBuf, mat->recv_reqHi,
                            numb->idx_extHi, numb->num_extHi, 
                            outlist, debug); CHECK_V_ERROR;
  }

  nalu_hypre_MPI_Alltoall(outlist, 1, NALU_HYPRE_MPI_INT, inlist, 1, NALU_HYPRE_MPI_INT, comm_dh); 
  /* At this point, inlist[j] contains the number of indices 
     that this processor must send to P_j.  Processors next need
     to exchange the actual lists of required indices; this is done
     in setup_sends_private()
  */

  setup_sends_private(mat, inlist, sg->o2n_sub, debug); CHECK_V_ERROR;

  /* Convert column indices in each row to local indices */
  for (row=0; row<m; row++) {
    NALU_HYPRE_Int len = rp[row+1]-rp[row];
    NALU_HYPRE_Int *ind = cval+rp[row];
    Numbering_dhGlobalToLocal(numb, len, ind, ind); CHECK_V_ERROR;
  }

  FREE_DH(outlist); CHECK_V_ERROR;
  FREE_DH(inlist); CHECK_V_ERROR;
  FREE_DH(end_rows); CHECK_V_ERROR;

  if (debug) {
    NALU_HYPRE_Int ii, jj;

    nalu_hypre_fprintf(logFile, "\n--------- row/col structure, after global to local renumbering\n");
    for (ii=0; ii<mat->m; ++ii) {
      nalu_hypre_fprintf(logFile, "local row %i :: ", ii+1);
      for (jj=mat->rp[ii]; jj<mat->rp[ii+1]; ++jj) {
        nalu_hypre_fprintf(logFile, "%i ", 1+mat->cval[jj]);
      }
      nalu_hypre_fprintf(logFile, "\n");
    }
    nalu_hypre_fprintf(logFile, "\n");
    fflush(logFile);
  }
  END_FUNC_DH
}

/* solve for MPI implementation of PILU.  This function is
   so similar to MatVec, that I put it here, instead of with
   the other solves located in Euclid_apply.c.
*/
static void forward_solve_private(NALU_HYPRE_Int m, NALU_HYPRE_Int from, NALU_HYPRE_Int to, 
                            NALU_HYPRE_Int *rp, NALU_HYPRE_Int *cval, NALU_HYPRE_Int *diag, NALU_HYPRE_Real *aval, 
                            NALU_HYPRE_Real *rhs, NALU_HYPRE_Real *work_y, bool debug);

static void backward_solve_private(NALU_HYPRE_Int m, NALU_HYPRE_Int from, NALU_HYPRE_Int to, 
                       NALU_HYPRE_Int *rp, NALU_HYPRE_Int *cval, NALU_HYPRE_Int *diag, NALU_HYPRE_Real *aval, 
                       NALU_HYPRE_Real *work_y, NALU_HYPRE_Real *work_x, bool debug);

static NALU_HYPRE_Int beg_rowG;


#undef __FUNC__
#define __FUNC__ "Factor_dhSolve"
void Factor_dhSolve(NALU_HYPRE_Real *rhs, NALU_HYPRE_Real *lhs, Euclid_dh ctx)
{
  START_FUNC_DH
  Factor_dh mat = ctx->F;
  NALU_HYPRE_Int    from, to;
  NALU_HYPRE_Int    ierr, i, m = mat->m, first_bdry = mat->first_bdry;
  NALU_HYPRE_Int    offsetLo = mat->numbSolve->num_extLo;
  NALU_HYPRE_Int    offsetHi = mat->numbSolve->num_extHi;
  NALU_HYPRE_Int    *rp = mat->rp, *cval = mat->cval, *diag = mat->diag;
  NALU_HYPRE_Real *aval = mat->aval;
  NALU_HYPRE_Int    *sendindLo = mat->sendindLo, *sendindHi = mat->sendindHi;
  NALU_HYPRE_Int    sendlenLo = mat->sendlenLo, sendlenHi = mat->sendlenHi;
  NALU_HYPRE_Real *sendbufLo = mat->sendbufLo, *sendbufHi = mat->sendbufHi; 
  NALU_HYPRE_Real *work_y = mat->work_y_lo;
  NALU_HYPRE_Real *work_x = mat->work_x_hi;
  bool debug = false;

  if (mat->debug && logFile != NULL) debug = true;
  if (debug) beg_rowG = ctx->F->beg_row;

/*
for (i=0; i<m+offsetLo+offsetHi; ++i) {
  work_y[i] = -99;
  work_x[i] = -99;
}
*/

  if (debug) {
    nalu_hypre_fprintf(logFile, "\n=====================================================\n");
    nalu_hypre_fprintf(logFile, "FACT Factor_dhSolve: num_recvLo= %i num_recvHi = %i\n",
                                         mat->num_recvLo, mat->num_recvHi);
  }

  /* start receives from higher and lower ordered subdomains */
  if (mat->num_recvLo) {
    nalu_hypre_MPI_Startall(mat->num_recvLo, mat->recv_reqLo); 
  }
  if (mat->num_recvHi) {
    nalu_hypre_MPI_Startall(mat->num_recvHi, mat->recv_reqHi); 
  }

  /*-------------------------------------------------------------
   * PART 1: Forward Solve Ly = rhs for y ('y' is called 'work')
   *-------------------------------------------------------------*/
  /* forward triangular solve on interior nodes */
  from = 0;
  to = first_bdry;
  if (from != to) {
    forward_solve_private(m, from, to, rp, cval, diag, aval, 
                          rhs, work_y, debug); CHECK_V_ERROR;
  }

  /* wait for receives from lower ordered subdomains, then
     complete forward solve on boundary nodes.
  */
  if (mat->num_recvLo) {
    nalu_hypre_MPI_Waitall(mat->num_recvLo, mat->recv_reqLo, mat->status);

    /* debug block */
    if (debug) {
      nalu_hypre_fprintf(logFile, "FACT got 'y' values from lower neighbors; work buffer:\n  ");
      for (i=0; i<offsetLo; ++i) {
        nalu_hypre_fprintf(logFile, "%g ", work_y[m+i]);
      }
    }
  }

  /* forward triangular solve on boundary nodes */
  from = first_bdry;
  to = m;
  if (from != to) {
    forward_solve_private(m, from, to, rp, cval, diag, aval, 
                          rhs, work_y, debug); CHECK_V_ERROR;
  }

  /*  send boundary elements from work vector 'y' to higher ordered subdomains */
  if (mat->num_sendHi) {

    /* copy elements to send buffer */
    for (i=0; i<sendlenHi; i++) {
      sendbufHi[i] = work_y[sendindHi[i]]; 
    }

    /* start the sends */
    nalu_hypre_MPI_Startall(mat->num_sendHi, mat->send_reqHi); 

    /* debug block */
    if (debug) {
      nalu_hypre_fprintf(logFile, "\nFACT sending 'y' values to higher neighbor:\nFACT   ");
      for (i=0; i<sendlenHi; i++) {
        nalu_hypre_fprintf(logFile, "%g ", sendbufHi[i]);
      }
      nalu_hypre_fprintf(logFile, "\n");
    }
  }

  /*----------------------------------------------------------
   * PART 2: Backward Solve
   *----------------------------------------------------------*/
  /* wait for bdry nodes 'x' from higher-ordered processsors */
  if (mat->num_recvHi) {
    ierr = nalu_hypre_MPI_Waitall(mat->num_recvHi, mat->recv_reqHi, mat->status); CHECK_MPI_V_ERROR(ierr);

    /* debug block */
    if (debug) {
      nalu_hypre_fprintf(logFile, "FACT got 'x' values from higher neighbors:\n  ");
      for (i=m+offsetLo; i<m+offsetLo+offsetHi; ++i) {
        nalu_hypre_fprintf(logFile, "%g ", work_x[i]);
      }
      nalu_hypre_fprintf(logFile, "\n");
    }
  }

  /* backward solve boundary nodes */
  from = m;
  to = first_bdry;
  if (from != to) {
    backward_solve_private(m, from, to, rp, cval, diag, aval, 
                           work_y, work_x, debug); CHECK_V_ERROR;
  }

  /*  send boundary node elements to lower ordered subdomains */
  if (mat->num_sendLo) {

    /* copy elements to send buffer */
    for (i=0; i<sendlenLo; i++) {
      sendbufLo[i] = work_x[sendindLo[i]]; 
    }

    /* start the sends */
    ierr = nalu_hypre_MPI_Startall(mat->num_sendLo, mat->send_reqLo); CHECK_MPI_V_ERROR(ierr);

    /* debug block */
    if (debug) {
      nalu_hypre_fprintf(logFile, "\nFACT sending 'x' values to lower neighbor:\nFACT   ");
      for (i=0; i<sendlenLo; i++) {
        nalu_hypre_fprintf(logFile, "%g ", sendbufLo[i]);
      }
      nalu_hypre_fprintf(logFile, "\n");
    }
  }

  /* backward solve interior nodes */
  from = first_bdry;
  to = 0;
  if (from != to) {
    backward_solve_private(m, from, to, rp, cval, diag, aval, 
                           work_y, work_x, debug); CHECK_V_ERROR;
  }

  /* copy solution from work vector lhs vector */
  nalu_hypre_TMemcpy(lhs,  work_x, NALU_HYPRE_Real, m, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

  if (debug) {
    nalu_hypre_fprintf(logFile, "\nFACT solution: ");
    for (i=0; i<m; ++i) {
      nalu_hypre_fprintf(logFile, "%g ", lhs[i]);
    }
    nalu_hypre_fprintf(logFile, "\n");
  }

  /* wait for sends to go through */
  if (mat->num_sendLo) {
    ierr = nalu_hypre_MPI_Waitall(mat->num_sendLo, mat->send_reqLo, mat->status); CHECK_MPI_V_ERROR(ierr);
  }

  if (mat->num_sendHi) {
    ierr = nalu_hypre_MPI_Waitall(mat->num_sendHi, mat->send_reqHi, mat->status); CHECK_MPI_V_ERROR(ierr);
  }
  END_FUNC_DH
}



#undef __FUNC__
#define __FUNC__ "forward_solve_private"
void forward_solve_private(NALU_HYPRE_Int m, NALU_HYPRE_Int from, NALU_HYPRE_Int to, NALU_HYPRE_Int *rp, 
                           NALU_HYPRE_Int *cval, NALU_HYPRE_Int *diag, NALU_HYPRE_Real *aval, 
                           NALU_HYPRE_Real *rhs, NALU_HYPRE_Real *work_y, bool debug)
{
  START_FUNC_DH
  NALU_HYPRE_Int i, j, idx;

  if (debug) {  
    nalu_hypre_fprintf(logFile, "\nFACT starting forward_solve_private; from= %i; to= %i, m= %i\n",
                                       1+from, 1+to, m);
  }

/*
  if (from == 0) {
    work_y[0] = rhs[0];  
    if (debug) {
      nalu_hypre_fprintf(logFile, "FACT   work_y[%i] = %g\n------------\n", 1+beg_rowG, work_y[0]);
    }
  } else {
    --from; 
  }
*/

 if (debug) {
  for (i=from; i<to; ++i) {
    NALU_HYPRE_Int     len  = diag[i] - rp[i];
    NALU_HYPRE_Int     *col = cval + rp[i];
    NALU_HYPRE_Real  *val  = aval + rp[i];
    NALU_HYPRE_Real  sum = rhs[i];

    nalu_hypre_fprintf(logFile, "FACT   solving for work_y[%i] (global)\n", i+1+beg_rowG);
    nalu_hypre_fprintf(logFile, "FACT        sum = %g\n", sum);
    for (j=0; j<len; ++j) {
      idx = col[j];
      sum -= ( val[j] * work_y[idx] );
      nalu_hypre_fprintf(logFile, "FACT        sum(%g) -= val[j] (%g) * work_y[%i] (%g)\n",
                                  sum, val[j], 1+idx, work_y[idx]);
    }
    work_y[i] = sum;
    nalu_hypre_fprintf(logFile, "FACT  work_y[%i] = %g\n", 1+i+beg_rowG, work_y[i]);
    nalu_hypre_fprintf(logFile, "-----------\n");
  }

  nalu_hypre_fprintf(logFile, "\nFACT   work vector at end of forward solve:\n");
  for ( i=0; i<to; i++ ) nalu_hypre_fprintf(logFile, "    %i %g\n", i+1+beg_rowG, work_y[i]);

 } else {
  for (i=from; i<to; ++i) {
    NALU_HYPRE_Int     len  = diag[i] - rp[i];
    NALU_HYPRE_Int     *col = cval + rp[i];
    NALU_HYPRE_Real  *val  = aval + rp[i];
    NALU_HYPRE_Real  sum = rhs[i];

    for (j=0; j<len; ++j) {
      idx = col[j];
      sum -= ( val[j] * work_y[idx] );
    }
    work_y[i] = sum;
  }
 }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "backward_solve_private"
void backward_solve_private(NALU_HYPRE_Int m, NALU_HYPRE_Int from, NALU_HYPRE_Int to, NALU_HYPRE_Int *rp, 
                            NALU_HYPRE_Int *cval, NALU_HYPRE_Int *diag, NALU_HYPRE_Real *aval, 
                            NALU_HYPRE_Real *work_y, NALU_HYPRE_Real *work_x, bool debug)
{
  START_FUNC_DH
  NALU_HYPRE_Int i, j, idx;

 if (debug) {  
  nalu_hypre_fprintf(logFile, "\nFACT starting backward_solve_private; from= %i; to= %i, m= %i\n",
                                       1+from, 1+to, m);
  for (i=from-1; i>=to; --i) {
    NALU_HYPRE_Int     len = rp[i+1] - diag[i] - 1;
    NALU_HYPRE_Int     *col = cval + diag[i] + 1;
    NALU_HYPRE_Real  *val  = aval + diag[i] + 1;
    NALU_HYPRE_Real  sum = work_y[i];
    nalu_hypre_fprintf(logFile, "FACT   solving for work_x[%i]\n", i+1+beg_rowG);

    for (j=0; j<len; ++j) {
      idx = col[j];
      sum -= (val[j] * work_x[idx]);
      nalu_hypre_fprintf(logFile, "FACT        sum(%g) -= val[j] (%g) * work_x[idx] (%g)\n",
                                  sum, val[j], work_x[idx]);
    }
    work_x[i] = sum*aval[diag[i]];
    nalu_hypre_fprintf(logFile, "FACT   work_x[%i] = %g\n", 1+i, work_x[i]);
    nalu_hypre_fprintf(logFile, "----------\n");
  }

 } else {
  for (i=from-1; i>=to; --i) {
    NALU_HYPRE_Int     len = rp[i+1] - diag[i] - 1;
    NALU_HYPRE_Int     *col = cval + diag[i] + 1;
    NALU_HYPRE_Real  *val  = aval + diag[i] + 1;
    NALU_HYPRE_Real  sum = work_y[i];

    for (j=0; j<len; ++j) {
      idx = col[j];
      sum -= (val[j] * work_x[idx]);
    }
    work_x[i] = sum*aval[diag[i]];
  }
 }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Factor_dhInit"
void Factor_dhInit(void *A, bool fillFlag, bool avalFlag,
                          NALU_HYPRE_Real rho, NALU_HYPRE_Int id, NALU_HYPRE_Int beg_rowP, Factor_dh *Fout)
{
  START_FUNC_DH
  NALU_HYPRE_Int m, n, beg_row, alloc;
  Factor_dh F;

  EuclidGetDimensions(A, &beg_row, &m, &n); CHECK_V_ERROR;
  alloc = (NALU_HYPRE_Int)(rho*m);
  Factor_dhCreate(&F); CHECK_V_ERROR;  

  *Fout = F;
  F->m = m;
  F->n = n;
  F->beg_row = beg_rowP;
  F->id = id;
  F->alloc = alloc;

  F->rp = (NALU_HYPRE_Int*)MALLOC_DH((m+1)*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
  F->rp[0] = 0;
  F->cval = (NALU_HYPRE_Int*)MALLOC_DH(alloc*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
  F->diag = (NALU_HYPRE_Int*)MALLOC_DH(m*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
  if (fillFlag) {
    F->fill = (NALU_HYPRE_Int*)MALLOC_DH(alloc*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
  }
  if (avalFlag) {
    F->aval = (REAL_DH*)MALLOC_DH(alloc*sizeof(REAL_DH)); CHECK_V_ERROR;
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Factor_dhReallocate"
void Factor_dhReallocate(Factor_dh F, NALU_HYPRE_Int used, NALU_HYPRE_Int additional)
{
  START_FUNC_DH
  NALU_HYPRE_Int alloc = F->alloc;

  if (used+additional > F->alloc) {
    NALU_HYPRE_Int *tmpI;
    while (alloc < used+additional) alloc *= 2;
    F->alloc = alloc;
    tmpI = F->cval;
    F->cval = (NALU_HYPRE_Int*)MALLOC_DH(alloc*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
    nalu_hypre_TMemcpy(F->cval,  tmpI, NALU_HYPRE_Int, used, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
    FREE_DH(tmpI); CHECK_V_ERROR;
    if (F->fill != NULL) {
      tmpI = F->fill;
      F->fill = (NALU_HYPRE_Int*)MALLOC_DH(alloc*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
      nalu_hypre_TMemcpy(F->fill,  tmpI, NALU_HYPRE_Int, used, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      FREE_DH(tmpI); CHECK_V_ERROR;
    }
    if (F->aval != NULL) {
      REAL_DH *tmpF = F->aval;
      F->aval = (REAL_DH*)MALLOC_DH(alloc*sizeof(REAL_DH)); CHECK_V_ERROR;
      nalu_hypre_TMemcpy(F->aval,  tmpF, REAL_DH, used, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      FREE_DH(tmpF); CHECK_V_ERROR;
    }
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Factor_dhTranspose"
void Factor_dhTranspose(Factor_dh A, Factor_dh *Bout)
{
  START_FUNC_DH
  Factor_dh B;

  if (np_dh > 1) { SET_V_ERROR("only for sequential"); } 

  Factor_dhCreate(&B); CHECK_V_ERROR;
  *Bout = B;
  B->m = B->n = A->m;
  if (B->aval == NULL) {
    mat_dh_transpose_private(A->m, A->rp, &B->rp, A->cval, &B->cval,
                              A->aval, NULL); CHECK_V_ERROR;
  } else {
    mat_dh_transpose_private(A->m, A->rp, &B->rp, A->cval, &B->cval,
                            A->aval, &B->aval); CHECK_V_ERROR;
  }
  END_FUNC_DH
}


/* this could be done using OpenMP, but I took it out for now */
#undef __FUNC__
#define __FUNC__ "Factor_dhSolveSeq"
void Factor_dhSolveSeq(NALU_HYPRE_Real *rhs, NALU_HYPRE_Real *lhs, Euclid_dh ctx)
{
  START_FUNC_DH
  Factor_dh F = ctx->F;
  NALU_HYPRE_Int       *rp, *cval, *diag;
  NALU_HYPRE_Int       i, j, *vi, nz, m = F->m;
  REAL_DH   *aval, *work;
  /* REAL_DH   *scale; */
  REAL_DH   *v, sum;
  bool debug = false;

  if (ctx->F->debug && logFile != NULL) debug = true;

  rp = F->rp;
  cval = F->cval;
  aval = F->aval;
  diag = F->diag;
  /* scale = ctx->scale; */
  work = ctx->work;

 if (debug) {
    nalu_hypre_fprintf(logFile, "\nFACT ============================================================\n");
    nalu_hypre_fprintf(logFile, "FACT starting Factor_dhSolveSeq\n");

  /* forward solve lower triangle */
  nalu_hypre_fprintf(logFile, "\nFACT   STARTING FORWARD SOLVE\n------------\n");
  work[0] = rhs[0];
  nalu_hypre_fprintf(logFile, "FACT   work[0] = %g\n------------\n", work[0]);
  for ( i=1; i<m; i++ ) {
    v   = aval + rp[i];
    vi  = cval + rp[i];
    nz  = diag[i] - rp[i];
    nalu_hypre_fprintf(logFile, "FACT   solving for work[%i]\n", i+1);
    sum = rhs[i];
    for (j=0; j<nz; ++j) {
      sum -= (v[j] * work[vi[j]]);
      nalu_hypre_fprintf(logFile, "FACT         sum (%g) -= v[j] (%g) * work[vi[j]] (%g)\n",
                                            sum, v[j], work[vi[j]]);
    }
    work[i] = sum;
    nalu_hypre_fprintf(logFile, "FACT   work[%i] = %g\n------------\n", 1+i, work[i]);
  }


  nalu_hypre_fprintf(logFile, "\nFACT   work vector at end of forward solve:\n");
  for ( i=0; i<m; i++ ) nalu_hypre_fprintf(logFile, "    %i %g\n", i+1, work[i]);


  /* backward solve upper triangular boundaries (sequential) */
  nalu_hypre_fprintf(logFile, "\nFACT   STARTING BACKWARD SOLVE\n--------------\n");
  for ( i=m-1; i>=0; i-- ){
    v   = aval + diag[i] + 1;
    vi  = cval + diag[i] + 1;
    nz  = rp[i+1] - diag[i] - 1;
    nalu_hypre_fprintf(logFile, "FACT   solving for lhs[%i]\n", i+1);
    sum = work[i];
    for (j=0; j<nz; ++j) {
      sum -= (v[j] * work[vi[j]]);
      nalu_hypre_fprintf(logFile, "FACT         sum (%g) -= v[j] (%g) * work[vi[j]] (%g)\n",
                                            sum, v[j], work[vi[j]]);
    }
    lhs[i] = work[i] = sum*aval[diag[i]];
    nalu_hypre_fprintf(logFile, "FACT   lhs[%i] = %g\n------------\n", 1+i, lhs[i]);
    nalu_hypre_fprintf(logFile, "FACT   solving for lhs[%i]\n", i+1);
  }

  nalu_hypre_fprintf(logFile, "\nFACT solution: ");
  for (i=0; i<m; ++i) nalu_hypre_fprintf(logFile, "%g ", lhs[i]);
  nalu_hypre_fprintf(logFile, "\n");


 } else {
  /* forward solve lower triangle */
  work[0] = rhs[0];
  for ( i=1; i<m; i++ ) {
    v   = aval + rp[i];
    vi  = cval + rp[i];
    nz  = diag[i] - rp[i];
    sum = rhs[i];
    while (nz--) sum -= (*v++ * work[*vi++]);
    work[i] = sum;
  }

  /* backward solve upper triangular boundaries (sequential) */
  for ( i=m-1; i>=0; i-- ){
    v   = aval + diag[i] + 1;
    vi  = cval + diag[i] + 1;
    nz  = rp[i+1] - diag[i] - 1;
    sum = work[i];
    while (nz--) sum -= (*v++ * work[*vi++]);
    lhs[i] = work[i] = sum*aval[diag[i]];
  }
 }
  END_FUNC_DH
}

/*---------------------------------------------------------------
 * next two are used by Factor_dhPrintXXX methods
 *---------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "adjust_bj_private"
void adjust_bj_private(Factor_dh mat)
{
  START_FUNC_DH
  NALU_HYPRE_Int i;
  NALU_HYPRE_Int nz = mat->rp[mat->m];
  NALU_HYPRE_Int beg_row = mat->beg_row;
  for (i=0; i<nz; ++i) mat->cval[i] += beg_row;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "unadjust_bj_private"
void unadjust_bj_private(Factor_dh mat)
{
  START_FUNC_DH
  NALU_HYPRE_Int i;
  NALU_HYPRE_Int nz = mat->rp[mat->m];
  NALU_HYPRE_Int beg_row = mat->beg_row;
  for (i=0; i<nz; ++i) mat->cval[i] -= beg_row;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Factor_dhMaxPivotInverse"
NALU_HYPRE_Real Factor_dhMaxPivotInverse(Factor_dh mat)
{
  START_FUNC_DH
  NALU_HYPRE_Int i, m = mat->m, *diags = mat->diag;
  REAL_DH *aval = mat->aval;
  NALU_HYPRE_Real minGlobal = 0.0, min = aval[diags[0]];
  NALU_HYPRE_Real retval;

  for (i=0; i<m; ++i) min = MIN(min, nalu_hypre_abs(aval[diags[i]]));
  if (np_dh == 1) {
    minGlobal = min;
  } else {
    nalu_hypre_MPI_Reduce(&min, &minGlobal, 1, nalu_hypre_MPI_REAL, nalu_hypre_MPI_MIN, 0, comm_dh);
  }

  if (minGlobal == 0) {
    retval = 0;
  } else {
   retval = 1.0 / minGlobal;
  }
  END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "Factor_dhMaxValue"
NALU_HYPRE_Real Factor_dhMaxValue(Factor_dh mat)
{
  START_FUNC_DH
  NALU_HYPRE_Real maxGlobal = 0.0, max = 0.0;
  NALU_HYPRE_Int i, nz = mat->rp[mat->m];
  REAL_DH *aval = mat->aval;

  for (i=0; i<nz; ++i) {
    max = MAX(max, nalu_hypre_abs(aval[i]));
  }
  
  if (np_dh == 1) {
    maxGlobal = max;
  } else {
    nalu_hypre_MPI_Reduce(&max, &maxGlobal, 1, nalu_hypre_MPI_REAL, nalu_hypre_MPI_MAX, 0, comm_dh);
  }
  END_FUNC_VAL(maxGlobal)
}


#undef __FUNC__
#define __FUNC__ "Factor_dhCondEst"
NALU_HYPRE_Real Factor_dhCondEst(Factor_dh mat, Euclid_dh ctx)
{
  START_FUNC_DH
  NALU_HYPRE_Real max = 0.0, maxGlobal = 0.0;
  NALU_HYPRE_Real *x;
  NALU_HYPRE_Int i, m = mat->m;
  Vec_dh lhs, rhs;

  Vec_dhCreate(&lhs); CHECK_ERROR(-1);
  Vec_dhInit(lhs, m); CHECK_ERROR(-1);
  Vec_dhDuplicate(lhs,&rhs); CHECK_ERROR(-1);
  Vec_dhSet(rhs, 1.0); CHECK_ERROR(-1);
  Euclid_dhApply(ctx, rhs->vals, lhs->vals); CHECK_ERROR(-1);

  x = lhs->vals;
  for (i=0; i<m; ++i) {
    max = MAX(max, nalu_hypre_abs(x[i]));
  }

  if (np_dh == 1) {
    maxGlobal = max;
  } else {
    nalu_hypre_MPI_Reduce(&max, &maxGlobal, 1, nalu_hypre_MPI_REAL, nalu_hypre_MPI_MAX, 0, comm_dh);
  }
  END_FUNC_VAL(maxGlobal)
}
