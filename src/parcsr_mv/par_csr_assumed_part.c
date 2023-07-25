/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*----------------------------------------------------
 * Functions for the IJ assumed partition
 * (Some of these were formerly in new_commpkg.c)
 *  AHB 4/06
 *-----------------------------------------------------*/

#include "_nalu_hypre_parcsr_mv.h"

/* This is used only in the function below */
#define CONTACT(a,b)  (contact_list[(a)*3+(b)])

/*--------------------------------------------------------------------
 * nalu_hypre_LocateAssumedPartition
 * Reconcile assumed partition with actual partition.  Essentially
 * each processor ends of with a partition of its assumed partition.
 *--------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_LocateAssumedPartition(MPI_Comm comm, NALU_HYPRE_BigInt row_start, NALU_HYPRE_BigInt row_end,
                             NALU_HYPRE_BigInt global_first_row, NALU_HYPRE_BigInt global_num_rows,
                             nalu_hypre_IJAssumedPart *part, NALU_HYPRE_Int myid)
{
   NALU_HYPRE_Int       i;

   NALU_HYPRE_BigInt    *contact_list;
   NALU_HYPRE_Int        contact_list_length, contact_list_storage;

   NALU_HYPRE_BigInt     contact_row_start[2], contact_row_end[2], contact_ranges;
   NALU_HYPRE_Int        owner_start, owner_end;
   NALU_HYPRE_BigInt     tmp_row_start, tmp_row_end;
   NALU_HYPRE_Int        complete;

   /*NALU_HYPRE_Int        locate_row_start[2]; */
   /*NALU_HYPRE_Int        locate_ranges;*/

   NALU_HYPRE_Int        locate_row_count, rows_found;

   NALU_HYPRE_BigInt     tmp_range[2];
   NALU_HYPRE_BigInt    *sortme;
   NALU_HYPRE_Int       *si;

   const NALU_HYPRE_Int  flag1 = 17;

   nalu_hypre_MPI_Request  *requests;
   nalu_hypre_MPI_Status   status0, *statuses;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------------
    *  Contact ranges -
    *  which rows do I have that others are assumed responsible for?
    *  (at most two ranges - maybe none)
    *-----------------------------------------------------------*/
   contact_row_start[0] = 0;
   contact_row_end[0] = 0;
   contact_row_start[1] = 0;
   contact_row_end[1] = 0;
   contact_ranges = 0;

   if (row_start <= row_end )
   {
      /*must own at least one row*/
      if ( part->row_end < row_start  || row_end < part->row_start  )
      {
         /*no overlap - so all of my rows and only one range*/
         contact_row_start[0] = row_start;
         contact_row_end[0] = row_end;
         contact_ranges++;
      }
      else /* the two regions overlap - so one or two ranges */
      {
         /* check for contact rows on the low end of the local range */
         if (row_start < part->row_start)
         {
            contact_row_start[0] = row_start;
            contact_row_end[0] = part->row_start - 1;
            contact_ranges++;
         }
         if (part->row_end < row_end) /* check the high end */
         {
            if (contact_ranges) /* already found one range */
            {
               contact_row_start[1] = part->row_end + 1;
               contact_row_end[1] = row_end;
            }
            else
            {
               contact_row_start[0] =  part->row_end + 1;
               contact_row_end[0] = row_end;
            }
            contact_ranges++;
         }
      }
   }

   /*-----------------------------------------------------------
    *  Contact: find out who is assumed responsible for these
    *       ranges of contact rows and contact them
    *
    *-----------------------------------------------------------*/


   contact_list_length = 0;
   contact_list_storage = 5;
   contact_list = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  contact_list_storage * 3,
                               NALU_HYPRE_MEMORY_HOST); /*each contact needs 3 ints */

   for (i = 0; i < contact_ranges; i++)
   {

      /*get start and end row owners */
      nalu_hypre_GetAssumedPartitionProcFromRow(comm, contact_row_start[i], global_first_row,
                                           global_num_rows, &owner_start);
      nalu_hypre_GetAssumedPartitionProcFromRow(comm, contact_row_end[i], global_first_row,
                                           global_num_rows, &owner_end);

      if (owner_start == owner_end) /* same processor owns the whole range */
      {

         if (contact_list_length == contact_list_storage)
         {
            /*allocate more space*/
            contact_list_storage += 5;
            contact_list = nalu_hypre_TReAlloc(contact_list,  NALU_HYPRE_BigInt,  (contact_list_storage * 3),
                                          NALU_HYPRE_MEMORY_HOST);
         }
         CONTACT(contact_list_length, 0) = (NALU_HYPRE_BigInt) owner_start;   /*proc #*/
         CONTACT(contact_list_length, 1) = contact_row_start[i];  /* start row */
         CONTACT(contact_list_length, 2) = contact_row_end[i];  /*end row */
         contact_list_length++;
      }
      else
      {
         complete = 0;
         while (!complete)
         {
            nalu_hypre_GetAssumedPartitionRowRange(comm, owner_start, global_first_row,
                                              global_num_rows, &tmp_row_start, &tmp_row_end);

            if (tmp_row_end >= contact_row_end[i])
            {
               tmp_row_end =  contact_row_end[i];
               complete = 1;
            }
            if (tmp_row_start <  contact_row_start[i])
            {
               tmp_row_start =  contact_row_start[i];
            }


            if (contact_list_length == contact_list_storage)
            {
               /*allocate more space*/
               contact_list_storage += 5;
               contact_list = nalu_hypre_TReAlloc(contact_list,  NALU_HYPRE_BigInt,  (contact_list_storage * 3),
                                             NALU_HYPRE_MEMORY_HOST);
            }


            CONTACT(contact_list_length, 0) = (NALU_HYPRE_BigInt) owner_start;   /*proc #*/
            CONTACT(contact_list_length, 1) = tmp_row_start;  /* start row */
            CONTACT(contact_list_length, 2) = tmp_row_end;  /*end row */
            contact_list_length++;
            owner_start++; /*processors are seqential */
         }
      }
   }

   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  contact_list_length, NALU_HYPRE_MEMORY_HOST);
   statuses = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  contact_list_length, NALU_HYPRE_MEMORY_HOST);

   /*send out messages */
   for (i = 0; i < contact_list_length; i++)
   {
      nalu_hypre_MPI_Isend(&CONTACT(i, 1), 2, NALU_HYPRE_MPI_BIG_INT, CONTACT(i, 0), flag1,
                      comm, &requests[i]);
      /*nalu_hypre_MPI_COMM_WORLD, &requests[i]);*/
   }

   /*-----------------------------------------------------------
    *  Locate ranges -
    *  which rows in my assumed range do I not own
    *  (at most two ranges - maybe none)
    *  locate_row_count = total number of rows I must locate
    *-----------------------------------------------------------*/


   locate_row_count = 0;

   /*locate_row_start[0]=0;
   locate_row_start[1]=0;*/

   /*locate_ranges = 0;*/

   if (part->row_end < row_start  || row_end < part->row_start  )
      /*no overlap - so all of my assumed rows */
   {
      /*locate_row_start[0] = part->row_start;*/
      /*locate_ranges++;*/
      locate_row_count += part->row_end - part->row_start + 1;
   }
   else /* the two regions overlap */
   {
      if (part->row_start < row_start)
      {
         /* check for locate rows on the low end of the local range */
         /*locate_row_start[0] = part->row_start;*/
         /*locate_ranges++;*/
         locate_row_count += (row_start - 1) - part->row_start + 1;
      }
      if (row_end < part->row_end) /* check the high end */
      {
         /*if (locate_ranges)*/ /* already have one range */
         /*{
           locate_row_start[1] = row_end +1;
           }
           else
           {
           locate_row_start[0] = row_end +1;
           }*/
         /*locate_ranges++;*/
         locate_row_count += part->row_end - (row_end + 1) + 1;
      }
   }


   /*-----------------------------------------------------------
    * Receive messages from other procs telling us where
    * all our  locate rows actually reside
    *-----------------------------------------------------------*/


   /* we will keep a partition of our assumed partition - list ourselves
      first.  We will sort later with an additional index.
      In practice, this should only contain a few processors */

   /*which part do I own?*/
   tmp_row_start = nalu_hypre_max(part->row_start, row_start);
   tmp_row_end = nalu_hypre_min(row_end, part->row_end);

   if (tmp_row_start <= tmp_row_end)
   {
      part->proc_list[0] =   myid;
      part->row_start_list[0] = tmp_row_start;
      part->row_end_list[0] = tmp_row_end;
      part->length++;
   }

   /* now look for messages that tell us which processor has our locate rows */
   /* these will be blocking receives as we know how many to expect and they should
       be waiting (and we don't want to continue on without them) */

   rows_found = 0;

   while (rows_found != locate_row_count)
   {
      nalu_hypre_MPI_Recv( tmp_range, 2, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_ANY_SOURCE,
                      flag1, comm, &status0);
      /*flag1 , nalu_hypre_MPI_COMM_WORLD, &status0);*/

      if (part->length == part->storage_length)
      {
         part->storage_length += 10;
         part->proc_list = nalu_hypre_TReAlloc(part->proc_list,  NALU_HYPRE_Int,  part->storage_length,
                                          NALU_HYPRE_MEMORY_HOST);
         part->row_start_list = nalu_hypre_TReAlloc(part->row_start_list,  NALU_HYPRE_BigInt,  part->storage_length,
                                               NALU_HYPRE_MEMORY_HOST);
         part->row_end_list = nalu_hypre_TReAlloc(part->row_end_list,  NALU_HYPRE_BigInt,  part->storage_length,
                                             NALU_HYPRE_MEMORY_HOST);

      }
      part->row_start_list[part->length] = tmp_range[0];
      part->row_end_list[part->length] = tmp_range[1];

      part->proc_list[part->length] = status0.nalu_hypre_MPI_SOURCE;
      rows_found += tmp_range[1] - tmp_range[0] + 1;

      part->length++;
   }

   /*In case the partition of the assumed partition is longish,
     we would like to know the sorted order */
   si = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  part->length, NALU_HYPRE_MEMORY_HOST);
   sortme = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  part->length, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < part->length; i++)
   {
      si[i] = i;
      sortme[i] = part->row_start_list[i];
   }
   nalu_hypre_BigQsortbi( sortme, si, 0, (part->length) - 1);
   part->sort_index = si;

   /*free the requests */
   nalu_hypre_MPI_Waitall(contact_list_length, requests,
                     statuses);

   nalu_hypre_TFree(statuses, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(sortme, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(contact_list, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}


nalu_hypre_IJAssumedPart*
nalu_hypre_AssumedPartitionCreate(MPI_Comm comm,
                             NALU_HYPRE_BigInt global_num,
                             NALU_HYPRE_BigInt start,
                             NALU_HYPRE_BigInt end)
{
   nalu_hypre_IJAssumedPart *apart;
   NALU_HYPRE_Int myid;

   nalu_hypre_MPI_Comm_rank(comm, &myid );

   /* allocate space */
   apart = nalu_hypre_CTAlloc(nalu_hypre_IJAssumedPart, 1, NALU_HYPRE_MEMORY_HOST);


   nalu_hypre_GetAssumedPartitionRowRange( comm, myid, 0, global_num,
                                      &(apart->row_start), &(apart->row_end));

   /*allocate some space for the partition of the assumed partition */
   apart->length = 0;
   /*room for 10 owners of the assumed partition*/
   apart->storage_length = 10; /*need to be >=1 */
   apart->proc_list = nalu_hypre_TAlloc(NALU_HYPRE_Int,  apart->storage_length, NALU_HYPRE_MEMORY_HOST);
   apart->row_start_list = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  apart->storage_length, NALU_HYPRE_MEMORY_HOST);
   apart->row_end_list = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  apart->storage_length, NALU_HYPRE_MEMORY_HOST);

   /* now we want to reconcile our actual partition with the assumed partition */
   nalu_hypre_LocateAssumedPartition(comm, start, end, 0, global_num, apart, myid);

   return apart;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixCreateAssumedPartition -
 * Each proc gets it own range. Then
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.
 *--------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixCreateAssumedPartition( nalu_hypre_ParCSRMatrix *matrix)
{
   NALU_HYPRE_BigInt global_num_cols;
   /* NALU_HYPRE_Int myid; */
   NALU_HYPRE_BigInt  row_start = 0, row_end = 0, col_start = 0, col_end = 0;

   MPI_Comm   comm;

   nalu_hypre_IJAssumedPart *apart;

   global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(matrix);
   comm = nalu_hypre_ParCSRMatrixComm(matrix);

   /* find out my actualy range of rows and columns */
   nalu_hypre_ParCSRMatrixGetLocalRange( matrix,
                                    &row_start, &row_end, /* these two are not used */
                                    &col_start, &col_end );
   /* get my assumed partitioning  - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */
   apart = nalu_hypre_AssumedPartitionCreate(comm, global_num_cols, col_start, col_end);

   /* this partition will be saved in the matrix data structure until the matrix is destroyed */
   nalu_hypre_ParCSRMatrixAssumedPartition(matrix) = apart;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_AssumedPartitionDestroy
 *--------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_AssumedPartitionDestroy(nalu_hypre_IJAssumedPart *apart )
{
   if (apart->storage_length > 0)
   {
      nalu_hypre_TFree(apart->proc_list, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(apart->row_start_list, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(apart->row_end_list, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(apart->sort_index, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(apart, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_GetAssumedPartitionProcFromRow
 * Assumed partition for IJ case. Given a particular row j, return
 * the processor that is assumed to own that row.
 *--------------------------------------------------------------------*/


NALU_HYPRE_Int
nalu_hypre_GetAssumedPartitionProcFromRow( MPI_Comm comm, NALU_HYPRE_BigInt row,
                                      NALU_HYPRE_BigInt global_first_row,
                                      NALU_HYPRE_BigInt global_num_rows, NALU_HYPRE_Int *proc_id)
{
   NALU_HYPRE_Int     num_procs;
   NALU_HYPRE_BigInt  size, switch_row, extra;


   nalu_hypre_MPI_Comm_size(comm, &num_procs );
   /*nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );*/

   /* j = floor[(row*p/N]  - this overflows*/
   /* *proc_id = (row*num_procs)/global_num_rows;*/

   /* this looks a bit odd, but we have to be very careful that
      this function and the next are inverses - and rounding
      errors make this difficult!!!!! */

   size = global_num_rows / (NALU_HYPRE_BigInt)num_procs;
   extra = global_num_rows - size * (NALU_HYPRE_BigInt)num_procs;
   switch_row = global_first_row + (size + 1) * extra;

   if (row >= switch_row)
   {
      *proc_id = (NALU_HYPRE_Int)(extra + (row - switch_row) / size);
   }
   else
   {
      *proc_id = (NALU_HYPRE_Int)((row - global_first_row) / (size + 1));
   }


   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_GetAssumedPartitionRowRange
 * Assumed partition for IJ case. Given a particular processor id, return
 * the assumed range of rows ([row_start, row_end]) for that processor.
 *--------------------------------------------------------------------*/


NALU_HYPRE_Int
nalu_hypre_GetAssumedPartitionRowRange( MPI_Comm comm, NALU_HYPRE_Int proc_id, NALU_HYPRE_BigInt global_first_row,
                                   NALU_HYPRE_BigInt global_num_rows, NALU_HYPRE_BigInt *row_start, NALU_HYPRE_BigInt* row_end)
{
   NALU_HYPRE_Int    num_procs;
   NALU_HYPRE_Int    extra;
   NALU_HYPRE_BigInt size;

   nalu_hypre_MPI_Comm_size(comm, &num_procs );
   /*nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );*/


   /* this may look non-intuitive, but we have to be very careful that
       this function and the next are inverses - and avoiding overflow and
       rounding errors makes this difficult! */

   size = global_num_rows / (NALU_HYPRE_BigInt)num_procs;
   extra = (NALU_HYPRE_Int)(global_num_rows - size * (NALU_HYPRE_BigInt)num_procs);

   *row_start = global_first_row + size * (NALU_HYPRE_BigInt)proc_id;
   *row_start += (NALU_HYPRE_BigInt) nalu_hypre_min(proc_id, extra);


   *row_end =  global_first_row + size * (NALU_HYPRE_BigInt)(proc_id + 1);
   *row_end += (NALU_HYPRE_BigInt)nalu_hypre_min(proc_id + 1, extra);
   *row_end = *row_end - 1;


   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------
 * nalu_hypre_ParVectorCreateAssumedPartition -

 * Essentially the same as for a matrix!

 * Each proc gets it own range. Then
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorCreateAssumedPartition( nalu_hypre_ParVector *vector)
{
   NALU_HYPRE_BigInt global_num;
   NALU_HYPRE_Int myid;
   NALU_HYPRE_BigInt  start = 0, end = 0;

   MPI_Comm   comm;

   nalu_hypre_IJAssumedPart *apart;

   global_num = nalu_hypre_ParVectorGlobalSize(vector);
   comm = nalu_hypre_ParVectorComm(vector);

   /* find out my actualy range of rows */
   start =  nalu_hypre_ParVectorFirstIndex(vector);
   end = nalu_hypre_ParVectorLastIndex(vector);

   nalu_hypre_MPI_Comm_rank(comm, &myid );

   /* allocate space */
   apart = nalu_hypre_CTAlloc(nalu_hypre_IJAssumedPart,  1, NALU_HYPRE_MEMORY_HOST);

   /* get my assumed partitioning  - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */
   nalu_hypre_GetAssumedPartitionRowRange( comm, myid, 0, global_num, &(apart->row_start),
                                      &(apart->row_end));

   /*allocate some space for the partition of the assumed partition */
   apart->length = 0;
   /*room for 10 owners of the assumed partition*/
   apart->storage_length = 10; /*need to be >=1 */
   apart->proc_list = nalu_hypre_TAlloc(NALU_HYPRE_Int,  apart->storage_length, NALU_HYPRE_MEMORY_HOST);
   apart->row_start_list =   nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  apart->storage_length, NALU_HYPRE_MEMORY_HOST);
   apart->row_end_list =   nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  apart->storage_length, NALU_HYPRE_MEMORY_HOST);

   /* now we want to reconcile our actual partition with the assumed partition */
   nalu_hypre_LocateAssumedPartition(comm, start, end, 0, global_num, apart, myid);

   /* this partition will be saved in the vector data structure until the vector is destroyed */
   nalu_hypre_ParVectorAssumedPartition(vector) = apart;

   return nalu_hypre_error_flag;
}
