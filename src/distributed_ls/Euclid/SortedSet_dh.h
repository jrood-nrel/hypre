/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef SORTED_SET_DH
#define SORTED_SET_DH

/* #include "euclid_common.h" */

struct _sortedset_dh {
  NALU_HYPRE_Int n;   /* max items that can be stored */
  NALU_HYPRE_Int *list;  /* list of inserted elements */
  NALU_HYPRE_Int count;  /* the number of elements in the list */
};

extern void SortedSet_dhCreate(SortedSet_dh *ss, NALU_HYPRE_Int initialSize);
extern void SortedSet_dhDestroy(SortedSet_dh ss);
extern void SortedSet_dhInsert(SortedSet_dh ss, NALU_HYPRE_Int idx);
extern void SortedSet_dhGetList(SortedSet_dh ss, NALU_HYPRE_Int **list, NALU_HYPRE_Int *count);


#endif
