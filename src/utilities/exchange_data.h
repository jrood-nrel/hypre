/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_EXCHANGE_DATA_HEADER
#define nalu_hypre_EXCHANGE_DATA_HEADER

#define nalu_hypre_BinaryTreeParentId(tree)      (tree->parent_id)
#define nalu_hypre_BinaryTreeNumChild(tree)      (tree->num_child)
#define nalu_hypre_BinaryTreeChildIds(tree)      (tree->child_id)
#define nalu_hypre_BinaryTreeChildId(tree, i)    (tree->child_id[i])

typedef struct
{
   NALU_HYPRE_Int                   parent_id;
   NALU_HYPRE_Int                   num_child;
   NALU_HYPRE_Int                  *child_id;
} nalu_hypre_BinaryTree;

/* In the fill_response() function the user needs to set the recv__buf
   and the response_message_size.  Memory of size send_response_storage has been
   alllocated for the send_buf (in exchange_data) - if more is needed, then
   realloc and adjust
   the send_response_storage.  The realloc amount should be storage+overhead.
   If the response is an empty "confirmation" message, then set
   response_message_size =0 (and do not modify the send_buf) */

typedef struct
{
   NALU_HYPRE_Int    (*fill_response)(void* recv_buf, NALU_HYPRE_Int contact_size,
                                 NALU_HYPRE_Int contact_proc, void* response_obj,
                                 MPI_Comm comm, void** response_buf,
                                 NALU_HYPRE_Int* response_message_size);
   NALU_HYPRE_Int     send_response_overhead; /*set by exchange data */
   NALU_HYPRE_Int     send_response_storage;  /*storage allocated for send_response_buf*/
   void    *data1;                 /*data fields user may want to access in fill_response */
   void    *data2;

} nalu_hypre_DataExchangeResponse;

NALU_HYPRE_Int nalu_hypre_CreateBinaryTree(NALU_HYPRE_Int, NALU_HYPRE_Int, nalu_hypre_BinaryTree*);
NALU_HYPRE_Int nalu_hypre_DestroyBinaryTree(nalu_hypre_BinaryTree*);
NALU_HYPRE_Int nalu_hypre_DataExchangeList(NALU_HYPRE_Int num_contacts, NALU_HYPRE_Int *contact_proc_list,
                                 void *contact_send_buf, NALU_HYPRE_Int *contact_send_buf_starts, NALU_HYPRE_Int contact_obj_size,
                                 NALU_HYPRE_Int response_obj_size, nalu_hypre_DataExchangeResponse *response_obj, NALU_HYPRE_Int max_response_size,
                                 NALU_HYPRE_Int rnum, MPI_Comm comm, void **p_response_recv_buf, NALU_HYPRE_Int **p_response_recv_buf_starts);

#endif /* end of header */

