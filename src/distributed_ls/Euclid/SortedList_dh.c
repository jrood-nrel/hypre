/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_Euclid.h"
/* #include "SortedList_dh.h" */
/* #include "Mem_dh.h" */
/* #include "Parser_dh.h" */
/* #include "Hash_i_dh.h" */
/* #include "SubdomainGraph_dh.h" */

struct _sortedList_dh {
  NALU_HYPRE_Int m;          /* number of local rows */
  NALU_HYPRE_Int row;        /* local number of row being factored */
  NALU_HYPRE_Int beg_row;    /* global number of first locally owned row, wrt A */
  NALU_HYPRE_Int beg_rowP;   /* global number of first locally owned row, wrt F */
  NALU_HYPRE_Int count;      /* number of items entered in the list, 
                     plus 1 (for header node) 
                   */
  NALU_HYPRE_Int countMax;    /* same as count, but includes number of items that my have
                      been deleted from calling SortedList_dhEnforceConstraint()
                   */
  NALU_HYPRE_Int *o2n_local;          /* not owned! */
  Hash_i_dh o2n_external;  /* not owned! */

  SRecord *list;  /* the sorted list */
  NALU_HYPRE_Int alloc;      /* allocated length of list */
  NALU_HYPRE_Int getLower;   /* index used for returning lower tri elts */
  NALU_HYPRE_Int get;        /* index of returning all elts; */
  
  bool debug;
};

static void lengthen_list_private(SortedList_dh sList);


#undef __FUNC__
#define __FUNC__ "SortedList_dhCreate"
void SortedList_dhCreate(SortedList_dh *sList)
{
  START_FUNC_DH
  struct _sortedList_dh* tmp = (struct _sortedList_dh*)MALLOC_DH(
                                 sizeof(struct _sortedList_dh)); CHECK_V_ERROR;
  *sList = tmp;
  tmp->m = 0;
  tmp->row = -1;
  tmp->beg_row = 0;
  tmp->count = 1;
  tmp->countMax = 1;
  tmp->o2n_external = NULL;
  tmp->o2n_local = NULL;

  tmp->get = 0;
  tmp->getLower = 0;
  tmp->alloc = 0;
  tmp->list = NULL;
  tmp->debug = Parser_dhHasSwitch(parser_dh, "-debug_SortedList");
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhDestroy"
void SortedList_dhDestroy(SortedList_dh sList)
{
  START_FUNC_DH
  if (sList->list != NULL) { FREE_DH(sList->list); CHECK_V_ERROR; }
  FREE_DH(sList); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhInit"
void SortedList_dhInit(SortedList_dh sList, SubdomainGraph_dh sg)
{
  START_FUNC_DH
  sList->o2n_local = sg->o2n_col;
  sList->m = sg->m;
  sList->beg_row = sg->beg_row[myid_dh];
  sList->beg_rowP = sg->beg_rowP[myid_dh];
  sList->count = 1;         /* "1" is for the header node */
  sList->countMax = 1;      /* "1" is for the header node */
  sList->o2n_external = sg->o2n_ext;

  /* heuristic: "m" should be a good number of nodes */
  sList->alloc = sList->m + 5;
  sList->list = (SRecord*)MALLOC_DH(sList->alloc*sizeof(SRecord)); 
  sList->list[0].col = INT_MAX;
  sList->list[0].next = 0;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhReset"
void SortedList_dhReset(SortedList_dh sList, NALU_HYPRE_Int row)
{
  START_FUNC_DH
  sList->row = row;
  sList->count = 1;
  sList->countMax = 1;
  sList->get = 0;
  sList->getLower = 0;
  sList->list[0].next = 0;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhReadCount"
NALU_HYPRE_Int SortedList_dhReadCount(SortedList_dh sList)
{
  START_FUNC_DH
  END_FUNC_VAL(sList->count-1)
}

#undef __FUNC__
#define __FUNC__ "SortedList_dhResetGetSmallest"
void SortedList_dhResetGetSmallest(SortedList_dh sList)
{
  START_FUNC_DH
  sList->getLower = 0;
  sList->get = 0;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "SortedList_dhGetSmallest"
SRecord * SortedList_dhGetSmallest(SortedList_dh sList)
{
  START_FUNC_DH
  SRecord *node = NULL;
  SRecord *list = sList->list;
  NALU_HYPRE_Int get = sList->get;

  get = list[get].next;

  if (list[get].col < INT_MAX) {
    node = &(list[get]);
    sList->get = get;
  }
  END_FUNC_VAL(node)
}

#undef __FUNC__
#define __FUNC__ "SortedList_dhGetSmallestLowerTri"
SRecord * SortedList_dhGetSmallestLowerTri(SortedList_dh sList)
{
  START_FUNC_DH
  SRecord *node = NULL;
  SRecord *list = sList->list;
  NALU_HYPRE_Int getLower = sList->getLower;
  NALU_HYPRE_Int globalRow = sList->row + sList->beg_rowP;

  getLower = list[getLower].next;

  if (list[getLower].col < globalRow) {
    node = &(list[getLower]);
    sList->getLower = getLower;
  }
  END_FUNC_VAL(node)
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhPermuteAndInsert"
bool SortedList_dhPermuteAndInsert(SortedList_dh sList, SRecord *sr, NALU_HYPRE_Real thresh)
{
  START_FUNC_DH
  bool wasInserted = false;
  NALU_HYPRE_Int col = sr->col;
  NALU_HYPRE_Real testVal = nalu_hypre_abs(sr->val);
  NALU_HYPRE_Int beg_row = sList->beg_row, end_row = beg_row + sList->m;
  NALU_HYPRE_Int beg_rowP = sList->beg_rowP;

  /* insertion of local indices */
  if (col >= beg_row && col < end_row) {
    /* convert to local indexing  and permute */
    col -= beg_row;
    col = sList->o2n_local[col];

    /* sparsification */
    if (testVal > thresh || col == sList->row) {
      col += beg_rowP;
    } else {
      col = -1;
/*
nalu_hypre_fprintf(logFile, "local row: %i  DROPPED: col= %i  val= %g (thresh= %g)\n",
                           sList->row+1, sr->col+1, testVal, thresh);
*/
    }
  } 


  /* insertion of external indices */
  else {
    /* sparsification for external indices */
    if (testVal < thresh) goto END_OF_FUNCTION;

    /* permute column index */
    if (sList->o2n_external == NULL) {
      col = -1;
    } else {
      NALU_HYPRE_Int tmp = Hash_i_dhLookup(sList->o2n_external, col); CHECK_ERROR(-1);
      if (tmp == -1) {
        col = -1;
      } else {
        col = tmp;
      }
    } 
  }

  if (col != -1) {
    sr->col = col;
    SortedList_dhInsert(sList, sr); CHECK_ERROR(-1);
    wasInserted = true;
  }

END_OF_FUNCTION: ;

  END_FUNC_VAL(wasInserted)
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhInsertOrUpdate"
void SortedList_dhInsertOrUpdate(SortedList_dh sList, SRecord *sr)
{
  START_FUNC_DH
  SRecord *node = SortedList_dhFind(sList, sr); CHECK_V_ERROR;

  if (node == NULL) {
    SortedList_dhInsert(sList, sr); CHECK_V_ERROR;
  } else {
    node->level = MIN(sr->level, node->level);
  }
  END_FUNC_DH
}


/* note: this does NOT check to see if item was already inserted! */
#undef __FUNC__
#define __FUNC__ "SortedList_dhInsert"
void SortedList_dhInsert(SortedList_dh sList, SRecord *sr)
{
  START_FUNC_DH
  NALU_HYPRE_Int prev, next;
  NALU_HYPRE_Int ct, col = sr->col;
  SRecord *list = sList->list;

  /* lengthen list if out of space */
  if (sList->countMax == sList->alloc) {
    lengthen_list_private(sList); CHECK_V_ERROR;
    list = sList->list;
  }

  /* add new node to end of list */
  ct = sList->countMax;
  sList->countMax += 1;
  sList->count += 1;

  list[ct].col = col;
  list[ct].level = sr->level;
  list[ct].val = sr->val;

  /* splice new node into list */
  prev = 0;
  next = list[0].next;
  while (col > list[next].col) {
    prev = next;
    next = list[next].next;
  }
  list[prev].next = ct;
  list[ct].next = next;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhFind"
SRecord * SortedList_dhFind(SortedList_dh sList, SRecord *sr)
{
  START_FUNC_DH
  NALU_HYPRE_Int i, count = sList->countMax;
  NALU_HYPRE_Int c = sr->col;
  SRecord *s = sList->list;
  SRecord *node = NULL;

  /* no need to traverse list in sorted order */
  for (i=1; i<count; ++i) {  /* start at i=1, since i=0 would be header node */

    if (s[i].col == c) {
      node = &(s[i]);
      break;
    }
  }

  END_FUNC_VAL(node)
}

#undef __FUNC__
#define __FUNC__ "lengthen_list_private"
void lengthen_list_private(SortedList_dh sList)
{
  START_FUNC_DH
  SRecord *tmp = sList->list;
  NALU_HYPRE_Int size = sList->alloc = 2*sList->alloc;

  SET_INFO("lengthening list");

  sList->list = (SRecord*)MALLOC_DH(size * sizeof(SRecord));
  nalu_hypre_TMemcpy(sList->list,  tmp, SRecord, sList->countMax, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST); 
  SET_INFO("doubling size of sList->list");
  FREE_DH(tmp); CHECK_V_ERROR;
  END_FUNC_DH
}


/*=====================================================================
 * functions for enforcing subdomain constraint 
 *=====================================================================*/


static bool check_constraint_private(SubdomainGraph_dh sg, 
                                     NALU_HYPRE_Int thisSubdomain, NALU_HYPRE_Int col);
void delete_private(SortedList_dh sList, NALU_HYPRE_Int col);

#undef __FUNC__
#define __FUNC__ "SortedList_dhEnforceConstraint"
void SortedList_dhEnforceConstraint(SortedList_dh sList, SubdomainGraph_dh sg)
{
  START_FUNC_DH
  NALU_HYPRE_Int thisSubdomain = myid_dh;
  NALU_HYPRE_Int col, count;
  NALU_HYPRE_Int beg_rowP = sList->beg_rowP;
  NALU_HYPRE_Int end_rowP = beg_rowP + sList->m;
  bool debug = false;

  if (Parser_dhHasSwitch(parser_dh, "-debug_SortedList")) debug = true;

  if (debug) {
    nalu_hypre_fprintf(logFile, "SLIST ======= enforcing constraint for row= %i\n", 1+sList->row);

    nalu_hypre_fprintf(logFile, "\nSLIST ---- before checking: ");
    count = SortedList_dhReadCount(sList); CHECK_V_ERROR;
    while (count--) {
      SRecord *sr = SortedList_dhGetSmallest(sList); CHECK_V_ERROR;
      nalu_hypre_fprintf(logFile, "%i ", sr->col+1);
    }
    nalu_hypre_fprintf(logFile, "\n");
    sList->get = 0;
  }

  /* for each column index in the list */
  count = SortedList_dhReadCount(sList); CHECK_V_ERROR;

  while (count--) {
    SRecord *sr = SortedList_dhGetSmallest(sList); CHECK_V_ERROR;
    col = sr->col;

    if (debug) {
      nalu_hypre_fprintf(logFile, "SLIST  next col= %i\n", col+1);
    }


    /* if corresponding row is nonlocal */
    if (col < beg_rowP || col >= end_rowP) {

      if (debug) {
        nalu_hypre_fprintf(logFile, "SLIST     external col: %i ; ", 1+col);
      }

      /* if entry would violate subdomain constraint, discard it
         (snip it out of the list)
       */
      if (check_constraint_private(sg, thisSubdomain, col)) {
        delete_private(sList, col); CHECK_V_ERROR;
        sList->count -= 1;

        if (debug) {
          nalu_hypre_fprintf(logFile, " deleted\n");
        }
      } else {
        if (debug) {
          nalu_hypre_fprintf(logFile, " kept\n");
        }
      }
    }
  }
  sList->get = 0;

  if (debug) {
    nalu_hypre_fprintf(logFile, "SLIST---- after checking: ");
    count = SortedList_dhReadCount(sList); CHECK_V_ERROR;
    while (count--) {
      SRecord *sr = SortedList_dhGetSmallest(sList); CHECK_V_ERROR;
      nalu_hypre_fprintf(logFile, "%i ", sr->col+1);
    }
    nalu_hypre_fprintf(logFile, "\n");
    fflush(logFile);
    sList->get = 0;
  }

  END_FUNC_DH
}


/* this is similar to a function in ilu_seq.c */
#undef __FUNC__
#define __FUNC__ "check_constraint_private"
bool check_constraint_private(SubdomainGraph_dh sg, NALU_HYPRE_Int p1, NALU_HYPRE_Int j)
{
  START_FUNC_DH
  bool retval = false;
  NALU_HYPRE_Int i, p2;
  NALU_HYPRE_Int *nabors, count;

  p2 = SubdomainGraph_dhFindOwner(sg, j, true);

  nabors = sg->adj + sg->ptrs[p1];
  count = sg->ptrs[p1+1]  - sg->ptrs[p1];

  for (i=0; i<count; ++i) {
    if (nabors[i] == p2) {
      retval = true;
      break;
    }
  }

  END_FUNC_VAL(! retval)
}

#undef __FUNC__
#define __FUNC__ "delete_private"
void delete_private(SortedList_dh sList, NALU_HYPRE_Int col)
{
  START_FUNC_DH
  NALU_HYPRE_Int curNode = 0;
  SRecord *list = sList->list;
  NALU_HYPRE_Int next;

  /* find node preceeding the node to be snipped out */
  /* 'list[curNode].next' is array index of the next node in the list */

  while (list[list[curNode].next].col != col) {
    curNode = list[curNode].next;
  }

  /* mark node to be deleted as inactive (needed for Find()) */
  next = list[curNode].next;
  list[next].col = -1;

  /* snip */
  next = list[next].next;
  list[curNode].next = next;
  END_FUNC_DH
}
