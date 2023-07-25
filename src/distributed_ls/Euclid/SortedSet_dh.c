/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_Euclid.h"
/* #include "SortedSet_dh.h" */
/* #include "shellSort_dh.h" */
/* #include "Mem_dh.h" */

#undef __FUNC__
#define __FUNC__ "SortedSet_dhCreate"
void SortedSet_dhCreate(SortedSet_dh *ss, NALU_HYPRE_Int size)
{
  START_FUNC_DH
  struct _sortedset_dh* tmp = (struct _sortedset_dh*)MALLOC_DH(sizeof(struct _sortedset_dh)); CHECK_V_ERROR;
  *ss= tmp;

  tmp->n = size;
  tmp->list = (NALU_HYPRE_Int*)MALLOC_DH(size*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
  tmp->count = 0;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "SortedSet_dhDestroy"
void SortedSet_dhDestroy(SortedSet_dh ss)
{
  START_FUNC_DH
  if (ss->list != NULL) { FREE_DH(ss->list); CHECK_V_ERROR; }
  FREE_DH(ss); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedSet_dhInsert"
void SortedSet_dhInsert(SortedSet_dh ss, NALU_HYPRE_Int idx)
{
  START_FUNC_DH
  bool isInserted = false;
  NALU_HYPRE_Int ct = ss->count;
  NALU_HYPRE_Int *list = ss->list;
  NALU_HYPRE_Int i, n = ss->n;

  /* determine if item was already inserted */
  for (i=0; i<ct; ++i) {
    if (list[i] == idx) {
      isInserted = true;
      break;
    }
  }

  /* is we need to insert the item, first check for overflow
     and reallocate if necessary, then append the index to the
     end of the list.
  */
  if (! isInserted) {
    if (ct == n) {
      NALU_HYPRE_Int *tmp = (NALU_HYPRE_Int*)MALLOC_DH(n*2*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
      nalu_hypre_TMemcpy(tmp,  list, NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      FREE_DH(list); CHECK_V_ERROR;
      list = ss->list = tmp;
      ss->n *= 2;
    }

    list[ct] = idx;
    ss->count += 1;
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedSet_dhGetList"
void SortedSet_dhGetList(SortedSet_dh ss, NALU_HYPRE_Int **list, NALU_HYPRE_Int *count)
{
  START_FUNC_DH
  shellSort_int(ss->count, ss->list);
  *list = ss->list;
  *count = ss->count;
  END_FUNC_DH
}

