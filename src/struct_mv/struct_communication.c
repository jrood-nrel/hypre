/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
#endif

/* this computes a (large enough) size (in doubles) for the message prefix */
#define nalu_hypre_CommPrefixSize(ne)                                        \
   ( (((1+ne)*sizeof(NALU_HYPRE_Int) + ne*sizeof(nalu_hypre_Box))/sizeof(NALU_HYPRE_Complex)) + 1 )

/*--------------------------------------------------------------------------
 * Create a communication package.  A grid-based description of a communication
 * exchange is passed in.  This description is then compiled into an
 * intermediate processor-based description of the communication.  The
 * intermediate processor-based description is used directly to pack and unpack
 * buffers during the communications.
 *
 * The 'orders' argument is dimension 'num_transforms' x 'num_values' and should
 * have a one-to-one correspondence with the transform data in 'comm_info'.
 *
 * If 'reverse' is > 0, then the meaning of send/recv is reversed
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommPkgCreate( nalu_hypre_CommInfo   *comm_info,
                     nalu_hypre_BoxArray   *send_data_space,
                     nalu_hypre_BoxArray   *recv_data_space,
                     NALU_HYPRE_Int         num_values,
                     NALU_HYPRE_Int       **orders,
                     NALU_HYPRE_Int         reverse,
                     MPI_Comm          comm,
                     nalu_hypre_CommPkg   **comm_pkg_ptr )
{
   NALU_HYPRE_Int             ndim = nalu_hypre_CommInfoNDim(comm_info);
   nalu_hypre_BoxArrayArray  *send_boxes;
   nalu_hypre_BoxArrayArray  *recv_boxes;
   nalu_hypre_BoxArrayArray  *send_rboxes;
   nalu_hypre_BoxArrayArray  *recv_rboxes;
   nalu_hypre_IndexRef        send_stride;
   nalu_hypre_IndexRef        recv_stride;
   NALU_HYPRE_Int           **send_processes;
   NALU_HYPRE_Int           **recv_processes;
   NALU_HYPRE_Int           **send_rboxnums;

   NALU_HYPRE_Int             num_transforms;
   nalu_hypre_Index          *coords;
   nalu_hypre_Index          *dirs;
   NALU_HYPRE_Int           **send_transforms;
   NALU_HYPRE_Int           **cp_orders;

   nalu_hypre_CommPkg        *comm_pkg;
   nalu_hypre_CommType       *comm_types;
   nalu_hypre_CommType       *comm_type;
   nalu_hypre_CommEntryType  *ct_entries;
   NALU_HYPRE_Int            *ct_rem_boxnums;
   nalu_hypre_Box            *ct_rem_boxes;
   NALU_HYPRE_Int            *comm_boxes_p, *comm_boxes_i, *comm_boxes_j;
   NALU_HYPRE_Int             num_boxes, num_entries, num_comms, comm_bufsize;

   nalu_hypre_BoxArray       *box_array;
   nalu_hypre_Box            *box;
   nalu_hypre_BoxArray       *rbox_array;
   nalu_hypre_Box            *rbox;
   nalu_hypre_Box            *data_box;
   NALU_HYPRE_Int            *data_offsets;
   NALU_HYPRE_Int             data_offset;
   nalu_hypre_IndexRef        send_coord, send_dir;
   NALU_HYPRE_Int            *send_order;

   NALU_HYPRE_Int             i, j, k, p, m, size, p_old, my_proc;

   /*------------------------------------------------------
    *------------------------------------------------------*/

   if (reverse > 0)
   {
      /* reverse the meaning of send and recv */
      send_boxes      = nalu_hypre_CommInfoRecvBoxes(comm_info);
      recv_boxes      = nalu_hypre_CommInfoSendBoxes(comm_info);
      send_stride     = nalu_hypre_CommInfoRecvStride(comm_info);
      recv_stride     = nalu_hypre_CommInfoSendStride(comm_info);
      send_processes  = nalu_hypre_CommInfoRecvProcesses(comm_info);
      recv_processes  = nalu_hypre_CommInfoSendProcesses(comm_info);
      send_rboxnums   = nalu_hypre_CommInfoRecvRBoxnums(comm_info);
      send_rboxes     = nalu_hypre_CommInfoRecvRBoxes(comm_info);
      recv_rboxes     = nalu_hypre_CommInfoSendRBoxes(comm_info);
      send_transforms = nalu_hypre_CommInfoRecvTransforms(comm_info); /* may be NULL */

      box_array = send_data_space;
      send_data_space = recv_data_space;
      recv_data_space = box_array;
   }
   else
   {
      send_boxes      = nalu_hypre_CommInfoSendBoxes(comm_info);
      recv_boxes      = nalu_hypre_CommInfoRecvBoxes(comm_info);
      send_stride     = nalu_hypre_CommInfoSendStride(comm_info);
      recv_stride     = nalu_hypre_CommInfoRecvStride(comm_info);
      send_processes  = nalu_hypre_CommInfoSendProcesses(comm_info);
      recv_processes  = nalu_hypre_CommInfoRecvProcesses(comm_info);
      send_rboxnums   = nalu_hypre_CommInfoSendRBoxnums(comm_info);
      send_rboxes     = nalu_hypre_CommInfoSendRBoxes(comm_info);
      recv_rboxes     = nalu_hypre_CommInfoRecvRBoxes(comm_info);
      send_transforms = nalu_hypre_CommInfoSendTransforms(comm_info); /* may be NULL */
   }
   num_transforms = nalu_hypre_CommInfoNumTransforms(comm_info);
   coords         = nalu_hypre_CommInfoCoords(comm_info); /* may be NULL */
   dirs           = nalu_hypre_CommInfoDirs(comm_info);   /* may be NULL */

   nalu_hypre_MPI_Comm_rank(comm, &my_proc );

   /*------------------------------------------------------
    * Set up various entries in CommPkg
    *------------------------------------------------------*/

   comm_pkg = nalu_hypre_CTAlloc(nalu_hypre_CommPkg, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_CommPkgComm(comm_pkg)      = comm;
   nalu_hypre_CommPkgFirstComm(comm_pkg) = 1;
   nalu_hypre_CommPkgNDim(comm_pkg)      = ndim;
   nalu_hypre_CommPkgNumValues(comm_pkg) = num_values;
   nalu_hypre_CommPkgNumOrders(comm_pkg) = 0;
   nalu_hypre_CommPkgOrders(comm_pkg)    = NULL;
   if ( (send_transforms != NULL) && (orders != NULL) )
   {
      nalu_hypre_CommPkgNumOrders(comm_pkg) = num_transforms;
      cp_orders = nalu_hypre_TAlloc(NALU_HYPRE_Int *, num_transforms, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_transforms; i++)
      {
         cp_orders[i] = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_values, NALU_HYPRE_MEMORY_HOST);
         for (j = 0; j < num_values; j++)
         {
            cp_orders[i][j] = orders[i][j];
         }
      }
      nalu_hypre_CommPkgOrders(comm_pkg) = cp_orders;
   }
   nalu_hypre_CopyIndex(send_stride, nalu_hypre_CommPkgSendStride(comm_pkg));
   nalu_hypre_CopyIndex(recv_stride, nalu_hypre_CommPkgRecvStride(comm_pkg));

   /* set identity transform and send_coord/dir/order if needed below */
   nalu_hypre_CommPkgIdentityOrder(comm_pkg) = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_values, NALU_HYPRE_MEMORY_HOST);
   send_coord = nalu_hypre_CommPkgIdentityCoord(comm_pkg);
   send_dir   = nalu_hypre_CommPkgIdentityDir(comm_pkg);
   send_order = nalu_hypre_CommPkgIdentityOrder(comm_pkg);
   for (i = 0; i < ndim; i++)
   {
      nalu_hypre_IndexD(send_coord, i) = i;
      nalu_hypre_IndexD(send_dir, i) = 1;
   }
   for (i = 0; i < num_values; i++)
   {
      send_order[i] = i;
   }

   /*------------------------------------------------------
    * Set up send CommType information
    *------------------------------------------------------*/

   /* set data_offsets and compute num_boxes, num_entries */
   data_offsets = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_BoxArraySize(send_data_space), NALU_HYPRE_MEMORY_HOST);
   data_offset = 0;
   num_boxes = 0;
   num_entries = 0;
   nalu_hypre_ForBoxI(i, send_data_space)
   {
      data_offsets[i] = data_offset;
      data_box = nalu_hypre_BoxArrayBox(send_data_space, i);
      data_offset += nalu_hypre_BoxVolume(data_box) * num_values;

      /* RDF: This should always be true, but it's not for FAC.  Find out why. */
      if (i < nalu_hypre_BoxArrayArraySize(send_boxes))
      {
         box_array = nalu_hypre_BoxArrayArrayBoxArray(send_boxes, i);
         num_boxes += nalu_hypre_BoxArraySize(box_array);
         nalu_hypre_ForBoxI(j, box_array)
         {
            box = nalu_hypre_BoxArrayBox(box_array, j);
            if (nalu_hypre_BoxVolume(box) != 0)
            {
               num_entries++;
            }
         }
      }
   }

   /* set up comm_boxes_[pij] */
   comm_boxes_p = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_boxes, NALU_HYPRE_MEMORY_HOST);
   comm_boxes_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_boxes, NALU_HYPRE_MEMORY_HOST);
   comm_boxes_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_boxes, NALU_HYPRE_MEMORY_HOST);
   num_boxes = 0;
   nalu_hypre_ForBoxArrayI(i, send_boxes)
   {
      box_array = nalu_hypre_BoxArrayArrayBoxArray(send_boxes, i);
      nalu_hypre_ForBoxI(j, box_array)
      {
         comm_boxes_p[num_boxes] = send_processes[i][j];
         comm_boxes_i[num_boxes] = i;
         comm_boxes_j[num_boxes] = j;
         num_boxes++;
      }
   }
   nalu_hypre_qsort3i(comm_boxes_p, comm_boxes_i, comm_boxes_j, 0, num_boxes - 1);

   /* compute comm_types */

   /* make sure there is at least 1 comm_type allocated */
   comm_types = nalu_hypre_CTAlloc(nalu_hypre_CommType, (num_boxes + 1), NALU_HYPRE_MEMORY_HOST);
   ct_entries = nalu_hypre_TAlloc(nalu_hypre_CommEntryType, num_entries, NALU_HYPRE_MEMORY_HOST);
   ct_rem_boxnums = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_entries, NALU_HYPRE_MEMORY_HOST);
   ct_rem_boxes = nalu_hypre_TAlloc(nalu_hypre_Box, num_entries, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CommPkgEntries(comm_pkg)    = ct_entries;
   nalu_hypre_CommPkgRemBoxnums(comm_pkg) = ct_rem_boxnums;
   nalu_hypre_CommPkgRemBoxes(comm_pkg)   = ct_rem_boxes;

   p_old = -1;
   num_comms = 0;
   comm_bufsize = 0;
   for (m = 0; m < num_boxes; m++)
   {
      i = comm_boxes_i[m];
      j = comm_boxes_j[m];
      box_array  = nalu_hypre_BoxArrayArrayBoxArray(send_boxes, i);
      rbox_array = nalu_hypre_BoxArrayArrayBoxArray(send_rboxes, i);
      box  = nalu_hypre_BoxArrayBox(box_array, j);
      rbox = nalu_hypre_BoxArrayBox(rbox_array, j);

      if ((nalu_hypre_BoxVolume(box) != 0) && (nalu_hypre_BoxVolume(rbox) != 0))
      {
         p = comm_boxes_p[m];

         /* start a new comm_type */
         if (p != p_old)
         {
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms + 1];
               num_comms++;
            }
            else
            {
               comm_type = &comm_types[0];
            }
            nalu_hypre_CommTypeProc(comm_type)       = p;
            nalu_hypre_CommTypeBufsize(comm_type)    = 0;
            nalu_hypre_CommTypeNumEntries(comm_type) = 0;
            nalu_hypre_CommTypeEntries(comm_type)    = ct_entries;
            nalu_hypre_CommTypeRemBoxnums(comm_type) = ct_rem_boxnums;
            nalu_hypre_CommTypeRemBoxes(comm_type)   = ct_rem_boxes;
            p_old = p;
         }

         k = nalu_hypre_CommTypeNumEntries(comm_type);
         nalu_hypre_BoxGetStrideVolume(box, send_stride, &size);
         nalu_hypre_CommTypeBufsize(comm_type) += (size * num_values);
         comm_bufsize                     += (size * num_values);
         rbox_array = nalu_hypre_BoxArrayArrayBoxArray(send_rboxes, i);
         data_box = nalu_hypre_BoxArrayBox(send_data_space, i);
         if (send_transforms != NULL)
         {
            send_coord = coords[send_transforms[i][j]];
            send_dir   = dirs[send_transforms[i][j]];
            if (orders != NULL)
            {
               send_order = cp_orders[send_transforms[i][j]];
            }
         }
         nalu_hypre_CommTypeSetEntry(box, send_stride, send_coord, send_dir,
                                send_order, data_box, data_offsets[i],
                                nalu_hypre_CommTypeEntry(comm_type, k));
         nalu_hypre_CommTypeRemBoxnum(comm_type, k) = send_rboxnums[i][j];
         nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(rbox_array, j),
                       nalu_hypre_CommTypeRemBox(comm_type, k));
         nalu_hypre_CommTypeNumEntries(comm_type) ++;
         ct_entries     ++;
         ct_rem_boxnums ++;
         ct_rem_boxes   ++;
      }
   }

   /* add space for prefix info */
   for (m = 1; m < (num_comms + 1); m++)
   {
      comm_type = &comm_types[m];
      k = nalu_hypre_CommTypeNumEntries(comm_type);
      size = nalu_hypre_CommPrefixSize(k);
      nalu_hypre_CommTypeBufsize(comm_type) += size;
      comm_bufsize                     += size;
   }

   /* set send info in comm_pkg */
   comm_types = nalu_hypre_TReAlloc(comm_types, nalu_hypre_CommType, (num_comms + 1), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CommPkgSendBufsize(comm_pkg)  = comm_bufsize;
   nalu_hypre_CommPkgNumSends(comm_pkg)     = num_comms;
   nalu_hypre_CommPkgSendTypes(comm_pkg)    = &comm_types[1];
   nalu_hypre_CommPkgCopyFromType(comm_pkg) = &comm_types[0];

   /* free up data_offsets */
   nalu_hypre_TFree(data_offsets, NALU_HYPRE_MEMORY_HOST);

   /*------------------------------------------------------
    * Set up recv CommType information
    *------------------------------------------------------*/

   /* set data_offsets and compute num_boxes */
   data_offsets = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_BoxArraySize(recv_data_space), NALU_HYPRE_MEMORY_HOST);
   data_offset = 0;
   num_boxes = 0;
   nalu_hypre_ForBoxI(i, recv_data_space)
   {
      data_offsets[i] = data_offset;
      data_box = nalu_hypre_BoxArrayBox(recv_data_space, i);
      data_offset += nalu_hypre_BoxVolume(data_box) * num_values;

      /* RDF: This should always be true, but it's not for FAC.  Find out why. */
      if (i < nalu_hypre_BoxArrayArraySize(recv_boxes))
      {
         box_array = nalu_hypre_BoxArrayArrayBoxArray(recv_boxes, i);
         num_boxes += nalu_hypre_BoxArraySize(box_array);
      }
   }
   nalu_hypre_CommPkgRecvDataOffsets(comm_pkg) = data_offsets;
   nalu_hypre_CommPkgRecvDataSpace(comm_pkg) = nalu_hypre_BoxArrayDuplicate(recv_data_space);

   /* set up comm_boxes_[pij] */
   comm_boxes_p = nalu_hypre_TReAlloc(comm_boxes_p, NALU_HYPRE_Int, num_boxes, NALU_HYPRE_MEMORY_HOST);
   comm_boxes_i = nalu_hypre_TReAlloc(comm_boxes_i, NALU_HYPRE_Int, num_boxes, NALU_HYPRE_MEMORY_HOST);
   comm_boxes_j = nalu_hypre_TReAlloc(comm_boxes_j, NALU_HYPRE_Int, num_boxes, NALU_HYPRE_MEMORY_HOST);
   num_boxes = 0;
   nalu_hypre_ForBoxArrayI(i, recv_boxes)
   {
      box_array = nalu_hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      nalu_hypre_ForBoxI(j, box_array)
      {
         comm_boxes_p[num_boxes] = recv_processes[i][j];
         comm_boxes_i[num_boxes] = i;
         comm_boxes_j[num_boxes] = j;
         num_boxes++;
      }
   }
   nalu_hypre_qsort3i(comm_boxes_p, comm_boxes_i, comm_boxes_j, 0, num_boxes - 1);

   /* compute comm_types */

   /* make sure there is at least 1 comm_type allocated */
   comm_types = nalu_hypre_CTAlloc(nalu_hypre_CommType, (num_boxes + 1), NALU_HYPRE_MEMORY_HOST);

   p_old = -1;
   num_comms = 0;
   comm_bufsize = 0;
   for (m = 0; m < num_boxes; m++)
   {
      i = comm_boxes_i[m];
      j = comm_boxes_j[m];
      box_array  = nalu_hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      rbox_array = nalu_hypre_BoxArrayArrayBoxArray(recv_rboxes, i);
      box  = nalu_hypre_BoxArrayBox(box_array, j);
      rbox = nalu_hypre_BoxArrayBox(rbox_array, j);

      if ((nalu_hypre_BoxVolume(box) != 0) && (nalu_hypre_BoxVolume(rbox) != 0))
      {
         p = comm_boxes_p[m];

         /* start a new comm_type */
         if (p != p_old)
         {
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms + 1];
               num_comms++;
            }
            else
            {
               comm_type = &comm_types[0];
            }
            nalu_hypre_CommTypeProc(comm_type)       = p;
            nalu_hypre_CommTypeBufsize(comm_type)    = 0;
            nalu_hypre_CommTypeNumEntries(comm_type) = 0;
            p_old = p;
         }

         k = nalu_hypre_CommTypeNumEntries(comm_type);
         nalu_hypre_BoxGetStrideVolume(box, recv_stride, &size);
         nalu_hypre_CommTypeBufsize(comm_type) += (size * num_values);
         comm_bufsize                     += (size * num_values);
         nalu_hypre_CommTypeNumEntries(comm_type) ++;
      }
   }

   /* add space for prefix info */
   for (m = 1; m < (num_comms + 1); m++)
   {
      comm_type = &comm_types[m];
      k = nalu_hypre_CommTypeNumEntries(comm_type);
      size = nalu_hypre_CommPrefixSize(k);
      nalu_hypre_CommTypeBufsize(comm_type) += size;
      comm_bufsize                     += size;
   }

   /* set recv info in comm_pkg */
   comm_types = nalu_hypre_TReAlloc(comm_types, nalu_hypre_CommType, (num_comms + 1), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CommPkgRecvBufsize(comm_pkg) = comm_bufsize;
   nalu_hypre_CommPkgNumRecvs(comm_pkg)    = num_comms;
   nalu_hypre_CommPkgRecvTypes(comm_pkg)   = &comm_types[1];
   nalu_hypre_CommPkgCopyToType(comm_pkg)  = &comm_types[0];

   /* if CommInfo send/recv boxes don't match, compute a max bufsize */
   if ( !nalu_hypre_CommInfoBoxesMatch(comm_info) )
   {
      nalu_hypre_CommPkgRecvBufsize(comm_pkg) = 0;
      for (i = 0; i < nalu_hypre_CommPkgNumRecvs(comm_pkg); i++)
      {
         comm_type = nalu_hypre_CommPkgRecvType(comm_pkg, i);

         /* subtract off old (incorrect) prefix size */
         num_entries = nalu_hypre_CommTypeNumEntries(comm_type);
         nalu_hypre_CommTypeBufsize(comm_type) -= nalu_hypre_CommPrefixSize(num_entries);

         /* set num_entries to number of grid points and add new prefix size */
         num_entries = nalu_hypre_CommTypeBufsize(comm_type);
         nalu_hypre_CommTypeNumEntries(comm_type) = num_entries;
         size = nalu_hypre_CommPrefixSize(num_entries);
         nalu_hypre_CommTypeBufsize(comm_type) += size;
         nalu_hypre_CommPkgRecvBufsize(comm_pkg) += nalu_hypre_CommTypeBufsize(comm_type);
      }
   }

   nalu_hypre_CommPkgSendBufsizeFirstComm(comm_pkg) = nalu_hypre_CommPkgSendBufsize(comm_pkg);
   nalu_hypre_CommPkgRecvBufsizeFirstComm(comm_pkg) = nalu_hypre_CommPkgRecvBufsize(comm_pkg);

   /*------------------------------------------------------
    * Debugging stuff - ONLY WORKS FOR 3D
    *------------------------------------------------------*/

#if DEBUG
   {
      nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &my_proc);

      nalu_hypre_sprintf(filename, "zcommboxes.%05d", my_proc);

      if ((file = fopen(filename, "a")) == NULL)
      {
         nalu_hypre_printf("Error: can't open output file %s\n", filename);
         exit(1);
      }

      nalu_hypre_fprintf(file, "\n\n============================\n\n");
      nalu_hypre_fprintf(file, "SEND boxes:\n\n");

      nalu_hypre_fprintf(file, "Stride = (%d,%d,%d)\n",
                    nalu_hypre_IndexD(send_stride, 0),
                    nalu_hypre_IndexD(send_stride, 1),
                    nalu_hypre_IndexD(send_stride, 2));
      nalu_hypre_fprintf(file, "BoxArrayArraySize = %d\n",
                    nalu_hypre_BoxArrayArraySize(send_boxes));
      nalu_hypre_ForBoxArrayI(i, send_boxes)
      {
         box_array = nalu_hypre_BoxArrayArrayBoxArray(send_boxes, i);

         nalu_hypre_fprintf(file, "BoxArraySize = %d\n", nalu_hypre_BoxArraySize(box_array));
         nalu_hypre_ForBoxI(j, box_array)
         {
            box = nalu_hypre_BoxArrayBox(box_array, j);
            nalu_hypre_fprintf(file, "(%d,%d): (%d,%d,%d) x (%d,%d,%d)\n",
                          i, j,
                          nalu_hypre_BoxIMinD(box, 0),
                          nalu_hypre_BoxIMinD(box, 1),
                          nalu_hypre_BoxIMinD(box, 2),
                          nalu_hypre_BoxIMaxD(box, 0),
                          nalu_hypre_BoxIMaxD(box, 1),
                          nalu_hypre_BoxIMaxD(box, 2));
            nalu_hypre_fprintf(file, "(%d,%d): %d,%d\n",
                          i, j, send_processes[i][j], send_rboxnums[i][j]);
         }
      }

      nalu_hypre_fprintf(file, "\n\n============================\n\n");
      nalu_hypre_fprintf(file, "RECV boxes:\n\n");

      nalu_hypre_fprintf(file, "Stride = (%d,%d,%d)\n",
                    nalu_hypre_IndexD(recv_stride, 0),
                    nalu_hypre_IndexD(recv_stride, 1),
                    nalu_hypre_IndexD(recv_stride, 2));
      nalu_hypre_fprintf(file, "BoxArrayArraySize = %d\n",
                    nalu_hypre_BoxArrayArraySize(recv_boxes));
      nalu_hypre_ForBoxArrayI(i, recv_boxes)
      {
         box_array = nalu_hypre_BoxArrayArrayBoxArray(recv_boxes, i);

         nalu_hypre_fprintf(file, "BoxArraySize = %d\n", nalu_hypre_BoxArraySize(box_array));
         nalu_hypre_ForBoxI(j, box_array)
         {
            box = nalu_hypre_BoxArrayBox(box_array, j);
            nalu_hypre_fprintf(file, "(%d,%d): (%d,%d,%d) x (%d,%d,%d)\n",
                          i, j,
                          nalu_hypre_BoxIMinD(box, 0),
                          nalu_hypre_BoxIMinD(box, 1),
                          nalu_hypre_BoxIMinD(box, 2),
                          nalu_hypre_BoxIMaxD(box, 0),
                          nalu_hypre_BoxIMaxD(box, 1),
                          nalu_hypre_BoxIMaxD(box, 2));
            nalu_hypre_fprintf(file, "(%d,%d): %d\n",
                          i, j, recv_processes[i][j]);
         }
      }

      fflush(file);
      fclose(file);
   }
#endif

#if DEBUG
   {
      nalu_hypre_CommEntryType  *comm_entry;
      NALU_HYPRE_Int             offset, dim;
      NALU_HYPRE_Int            *length;
      NALU_HYPRE_Int            *stride;

      nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &my_proc);

      nalu_hypre_sprintf(filename, "zcommentries.%05d", my_proc);

      if ((file = fopen(filename, "a")) == NULL)
      {
         nalu_hypre_printf("Error: can't open output file %s\n", filename);
         exit(1);
      }

      nalu_hypre_fprintf(file, "\n\n============================\n\n");
      nalu_hypre_fprintf(file, "SEND entries:\n\n");

      nalu_hypre_fprintf(file, "num_sends = %d\n", nalu_hypre_CommPkgNumSends(comm_pkg));

      comm_types = nalu_hypre_CommPkgCopyFromType(comm_pkg);
      for (m = 0; m < (nalu_hypre_CommPkgNumSends(comm_pkg) + 1); m++)
      {
         comm_type = &comm_types[m];
         nalu_hypre_fprintf(file, "process     = %d\n", nalu_hypre_CommTypeProc(comm_type));
         nalu_hypre_fprintf(file, "num_entries = %d\n", nalu_hypre_CommTypeNumEntries(comm_type));
         for (i = 0; i < nalu_hypre_CommTypeNumEntries(comm_type); i++)
         {
            comm_entry = nalu_hypre_CommTypeEntry(comm_type, i);
            offset = nalu_hypre_CommEntryTypeOffset(comm_entry);
            dim    = nalu_hypre_CommEntryTypeDim(comm_entry);
            length = nalu_hypre_CommEntryTypeLengthArray(comm_entry);
            stride = nalu_hypre_CommEntryTypeStrideArray(comm_entry);
            nalu_hypre_fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
                          i, offset, dim,
                          length[0], length[1], length[2], length[3],
                          stride[0], stride[1], stride[2], stride[3]);
         }
      }

      nalu_hypre_fprintf(file, "\n\n============================\n\n");
      nalu_hypre_fprintf(file, "RECV entries:\n\n");

      nalu_hypre_fprintf(file, "num_recvs = %d\n", nalu_hypre_CommPkgNumRecvs(comm_pkg));

      comm_types = nalu_hypre_CommPkgCopyToType(comm_pkg);

      comm_type = &comm_types[0];
      nalu_hypre_fprintf(file, "process     = %d\n", nalu_hypre_CommTypeProc(comm_type));
      nalu_hypre_fprintf(file, "num_entries = %d\n", nalu_hypre_CommTypeNumEntries(comm_type));
      for (i = 0; i < nalu_hypre_CommTypeNumEntries(comm_type); i++)
      {
         comm_entry = nalu_hypre_CommTypeEntry(comm_type, i);
         offset = nalu_hypre_CommEntryTypeOffset(comm_entry);
         dim    = nalu_hypre_CommEntryTypeDim(comm_entry);
         length = nalu_hypre_CommEntryTypeLengthArray(comm_entry);
         stride = nalu_hypre_CommEntryTypeStrideArray(comm_entry);
         nalu_hypre_fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
                       i, offset, dim,
                       length[0], length[1], length[2], length[3],
                       stride[0], stride[1], stride[2], stride[3]);
      }

      for (m = 1; m < (nalu_hypre_CommPkgNumRecvs(comm_pkg) + 1); m++)
      {
         comm_type = &comm_types[m];
         nalu_hypre_fprintf(file, "process     = %d\n", nalu_hypre_CommTypeProc(comm_type));
         nalu_hypre_fprintf(file, "num_entries = %d\n", nalu_hypre_CommTypeNumEntries(comm_type));
      }

      fflush(file);
      fclose(file);
   }
#endif

   /*------------------------------------------------------
    * Clean up
    *------------------------------------------------------*/

   nalu_hypre_TFree(comm_boxes_p, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_boxes_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_boxes_j, NALU_HYPRE_MEMORY_HOST);

   *comm_pkg_ptr = comm_pkg;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note that this routine assumes an identity coordinate transform
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommTypeSetEntries( nalu_hypre_CommType  *comm_type,
                          NALU_HYPRE_Int       *boxnums,
                          nalu_hypre_Box       *boxes,
                          nalu_hypre_Index      stride,
                          nalu_hypre_Index      coord,
                          nalu_hypre_Index      dir,
                          NALU_HYPRE_Int       *order,
                          nalu_hypre_BoxArray  *data_space,
                          NALU_HYPRE_Int       *data_offsets )
{
   NALU_HYPRE_Int             num_entries = nalu_hypre_CommTypeNumEntries(comm_type);
   nalu_hypre_CommEntryType  *entries     = nalu_hypre_CommTypeEntries(comm_type);
   nalu_hypre_Box            *box;
   nalu_hypre_Box            *data_box;
   NALU_HYPRE_Int             i, j;

   for (j = 0; j < num_entries; j++)
   {
      i = boxnums[j];
      box = &boxes[j];
      data_box = nalu_hypre_BoxArrayBox(data_space, i);

      nalu_hypre_CommTypeSetEntry(box, stride, coord, dir, order,
                             data_box, data_offsets[i], &entries[j]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommTypeSetEntry( nalu_hypre_Box           *box,
                        nalu_hypre_Index          stride,
                        nalu_hypre_Index          coord,
                        nalu_hypre_Index          dir,
                        NALU_HYPRE_Int           *order,
                        nalu_hypre_Box           *data_box,
                        NALU_HYPRE_Int            data_box_offset,
                        nalu_hypre_CommEntryType *comm_entry )
{
   NALU_HYPRE_Int     dim, ndim = nalu_hypre_BoxNDim(box);
   NALU_HYPRE_Int     offset;
   NALU_HYPRE_Int    *length_array, tmp_length_array[NALU_HYPRE_MAXDIM];
   NALU_HYPRE_Int    *stride_array, tmp_stride_array[NALU_HYPRE_MAXDIM];
   nalu_hypre_Index   size;
   NALU_HYPRE_Int     i, j;

   length_array = nalu_hypre_CommEntryTypeLengthArray(comm_entry);
   stride_array = nalu_hypre_CommEntryTypeStrideArray(comm_entry);

   /* initialize offset */
   offset = data_box_offset + nalu_hypre_BoxIndexRank(data_box, nalu_hypre_BoxIMin(box));

   /* initialize length_array and stride_array */
   nalu_hypre_BoxGetStrideSize(box, stride, size);
   for (i = 0; i < ndim; i++)
   {
      length_array[i] = nalu_hypre_IndexD(size, i);
      stride_array[i] = nalu_hypre_IndexD(stride, i);
      for (j = 0; j < i; j++)
      {
         stride_array[i] *= nalu_hypre_BoxSizeD(data_box, j);
      }
   }
   stride_array[ndim] = nalu_hypre_BoxVolume(data_box);

   /* make adjustments for dir */
   for (i = 0; i < ndim; i++)
   {
      if (dir[i] < 0)
      {
         offset += (length_array[i] - 1) * stride_array[i];
         stride_array[i] = -stride_array[i];
      }
   }

   /* make adjustments for coord */
   for (i = 0; i < ndim; i++)
   {
      tmp_length_array[i] = length_array[i];
      tmp_stride_array[i] = stride_array[i];
   }
   for (i = 0; i < ndim; i++)
   {
      j = coord[i];
      length_array[j] = tmp_length_array[i];
      stride_array[j] = tmp_stride_array[i];
   }

   /* eliminate dimensions with length_array = 1 */
   dim = ndim;
   i = 0;
   while (i < dim)
   {
      if (length_array[i] == 1)
      {
         for (j = i; j < (dim - 1); j++)
         {
            length_array[j] = length_array[j + 1];
            stride_array[j] = stride_array[j + 1];
         }
         length_array[dim - 1] = 1;
         stride_array[dim - 1] = 1;
         dim--;
      }
      else
      {
         i++;
      }
   }

#if 0
   /* sort the array according to length_array (largest to smallest) */
   for (i = (dim - 1); i > 0; i--)
   {
      for (j = 0; j < i; j++)
      {
         if (length_array[j] < length_array[j + 1])
         {
            i_tmp             = length_array[j];
            length_array[j]   = length_array[j + 1];
            length_array[j + 1] = i_tmp;

            i_tmp             = stride_array[j];
            stride_array[j]   = stride_array[j + 1];
            stride_array[j + 1] = i_tmp;
         }
      }
   }
#endif

   /* if every len was 1 we need to fix to communicate at least one */
   if (!dim)
   {
      dim = 1;
   }

   nalu_hypre_CommEntryTypeOffset(comm_entry) = offset;
   nalu_hypre_CommEntryTypeDim(comm_entry) = dim;
   nalu_hypre_CommEntryTypeOrder(comm_entry) = order;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Complex *
nalu_hypre_StructCommunicationGetBuffer(NALU_HYPRE_MemoryLocation memory_location,
                                   NALU_HYPRE_Int            size)
{
   NALU_HYPRE_Complex *ptr;

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
      if (size > nalu_hypre_HandleStructCommSendBufferSize(nalu_hypre_handle()))
      {
         NALU_HYPRE_Int new_size = 5 * size;
         nalu_hypre_HandleStructCommSendBufferSize(nalu_hypre_handle()) = new_size;
         nalu_hypre_TFree(nalu_hypre_HandleStructCommSendBuffer(nalu_hypre_handle()), memory_location);
         nalu_hypre_HandleStructCommSendBuffer(nalu_hypre_handle()) = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, new_size,
                                                                          memory_location);
      }

      ptr = nalu_hypre_HandleStructCommSendBuffer(nalu_hypre_handle());
   }
   else
#endif
   {
      ptr = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, size, memory_location);
   }

   return ptr;
}

NALU_HYPRE_Int
nalu_hypre_StructCommunicationReleaseBuffer(NALU_HYPRE_Complex       *buffer,
                                       NALU_HYPRE_MemoryLocation memory_location)
{
   if (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_HOST)
   {
      nalu_hypre_TFree(buffer, memory_location);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Initialize a non-blocking communication exchange.
 *
 * The communication buffers are created, the send buffer is manually
 * packed, and the communication requests are posted.
 *
 * Different "actions" are possible when the buffer data is unpacked:
 *   action = 0    - copy the data over existing values in memory
 *   action = 1    - add the data to existing values in memory
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_InitializeCommunication( nalu_hypre_CommPkg        *comm_pkg,
                               NALU_HYPRE_Complex        *send_data,
                               NALU_HYPRE_Complex        *recv_data,
                               NALU_HYPRE_Int             action,
                               NALU_HYPRE_Int             tag,
                               nalu_hypre_CommHandle    **comm_handle_ptr )
{
   nalu_hypre_CommHandle    *comm_handle;

   NALU_HYPRE_Int            ndim       = nalu_hypre_CommPkgNDim(comm_pkg);
   NALU_HYPRE_Int            num_values = nalu_hypre_CommPkgNumValues(comm_pkg);
   NALU_HYPRE_Int            num_sends  = nalu_hypre_CommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int            num_recvs  = nalu_hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm       = nalu_hypre_CommPkgComm(comm_pkg);

   NALU_HYPRE_Int            num_requests;
   nalu_hypre_MPI_Request   *requests;
   nalu_hypre_MPI_Status    *status;

   NALU_HYPRE_Complex      **send_buffers;
   NALU_HYPRE_Complex      **recv_buffers;
   NALU_HYPRE_Complex      **send_buffers_mpi;
   NALU_HYPRE_Complex      **recv_buffers_mpi;

   nalu_hypre_CommType      *comm_type, *from_type, *to_type;
   nalu_hypre_CommEntryType *comm_entry;
   NALU_HYPRE_Int            num_entries;

   NALU_HYPRE_Int           *length_array;
   NALU_HYPRE_Int           *stride_array, unitst_array[NALU_HYPRE_MAXDIM + 1];
   NALU_HYPRE_Int           *order;

   NALU_HYPRE_Complex       *dptr, *kptr, *lptr;
   NALU_HYPRE_Int           *qptr;

   NALU_HYPRE_Int            i, j, d, ll;
   NALU_HYPRE_Int            size;

   NALU_HYPRE_MemoryLocation memory_location     = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());
   NALU_HYPRE_MemoryLocation memory_location_mpi = memory_location;

   /*--------------------------------------------------------------------
    * allocate requests and status
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_requests, NALU_HYPRE_MEMORY_HOST);
   status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_requests, NALU_HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------
    * allocate buffers
    *--------------------------------------------------------------------*/

   /* allocate send buffers */
   send_buffers = nalu_hypre_TAlloc(NALU_HYPRE_Complex *, num_sends, NALU_HYPRE_MEMORY_HOST);
   if (num_sends > 0)
   {
      size = nalu_hypre_CommPkgSendBufsize(comm_pkg);
      send_buffers[0] = nalu_hypre_StructCommunicationGetBuffer(memory_location, size);
      for (i = 1; i < num_sends; i++)
      {
         comm_type = nalu_hypre_CommPkgSendType(comm_pkg, i - 1);
         size = nalu_hypre_CommTypeBufsize(comm_type);
         send_buffers[i] = send_buffers[i - 1] + size;
      }
   }

   /* allocate recv buffers */
   recv_buffers = nalu_hypre_TAlloc(NALU_HYPRE_Complex *, num_recvs, NALU_HYPRE_MEMORY_HOST);
   if (num_recvs > 0)
   {
      size = nalu_hypre_CommPkgRecvBufsize(comm_pkg);
      recv_buffers[0] = nalu_hypre_StructCommunicationGetBuffer(memory_location, size);
      for (i = 1; i < num_recvs; i++)
      {
         comm_type = nalu_hypre_CommPkgRecvType(comm_pkg, i - 1);
         size = nalu_hypre_CommTypeBufsize(comm_type);
         recv_buffers[i] = recv_buffers[i - 1] + size;
      }
   }

   /*--------------------------------------------------------------------
    * pack send buffers
    *--------------------------------------------------------------------*/

   for (i = 0; i < num_sends; i++)
   {
      comm_type = nalu_hypre_CommPkgSendType(comm_pkg, i);
      num_entries = nalu_hypre_CommTypeNumEntries(comm_type);

      dptr = (NALU_HYPRE_Complex *) send_buffers[i];
      if ( nalu_hypre_CommPkgFirstComm(comm_pkg) )
      {
         dptr += nalu_hypre_CommPrefixSize(num_entries);
      }

      for (j = 0; j < num_entries; j++)
      {
         comm_entry = nalu_hypre_CommTypeEntry(comm_type, j);
         length_array = nalu_hypre_CommEntryTypeLengthArray(comm_entry);
         stride_array = nalu_hypre_CommEntryTypeStrideArray(comm_entry);
         order = nalu_hypre_CommEntryTypeOrder(comm_entry);
         unitst_array[0] = 1;
         for (d = 1; d <= ndim; d++)
         {
            unitst_array[d] = unitst_array[d - 1] * length_array[d - 1];
         }

         lptr = send_data + nalu_hypre_CommEntryTypeOffset(comm_entry);
         for (ll = 0; ll < num_values; ll++)
         {
            if (order[ll] > -1)
            {
               kptr = lptr + order[ll] * stride_array[ndim];

#define DEVICE_VAR is_device_ptr(dptr,kptr)
               nalu_hypre_BasicBoxLoop2Begin(ndim, length_array,
                                        stride_array, ki,
                                        unitst_array, di);
               {
                  dptr[di] = kptr[ki];
               }
               nalu_hypre_BoxLoop2End(ki, di);
#undef DEVICE_VAR

               dptr += unitst_array[ndim];
            }
            else
            {
               size = 1;
               for (d = 0; d < ndim; d++)
               {
                  size *= length_array[d];
               }

               nalu_hypre_Memset(dptr, 0, size * sizeof(NALU_HYPRE_Complex), memory_location);

               dptr += size;
            }
         }
      }
   }

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI)
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());

      send_buffers_mpi = send_buffers;
      recv_buffers_mpi = recv_buffers;
#else
      memory_location_mpi = NALU_HYPRE_MEMORY_HOST;

      send_buffers_mpi = nalu_hypre_TAlloc(NALU_HYPRE_Complex *, num_sends, NALU_HYPRE_MEMORY_HOST);
      if (num_sends > 0)
      {
         size = nalu_hypre_CommPkgSendBufsize(comm_pkg);
         send_buffers_mpi[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, size, memory_location_mpi);
         for (i = 1; i < num_sends; i++)
         {
            send_buffers_mpi[i] = send_buffers_mpi[i - 1] + (send_buffers[i] - send_buffers[i - 1]);
         }
         nalu_hypre_TMemcpy(send_buffers_mpi[0], send_buffers[0], NALU_HYPRE_Complex, size, NALU_HYPRE_MEMORY_HOST,
                       memory_location);
      }

      recv_buffers_mpi = nalu_hypre_TAlloc(NALU_HYPRE_Complex *, num_recvs, NALU_HYPRE_MEMORY_HOST);
      if (num_recvs > 0)
      {
         size = nalu_hypre_CommPkgRecvBufsize(comm_pkg);
         recv_buffers_mpi[0] = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, size, memory_location_mpi);
         for (i = 1; i < num_recvs; i++)
         {
            recv_buffers_mpi[i] = recv_buffers_mpi[i - 1] + (recv_buffers[i] - recv_buffers[i - 1]);
         }
      }
#endif
   }
   else
#endif
   {
      send_buffers_mpi = send_buffers;
      recv_buffers_mpi = recv_buffers;
   }

   for (i = 0; i < num_sends; i++)
   {
      comm_type = nalu_hypre_CommPkgSendType(comm_pkg, i);
      num_entries = nalu_hypre_CommTypeNumEntries(comm_type);

      if ( nalu_hypre_CommPkgFirstComm(comm_pkg) )
      {
         qptr = (NALU_HYPRE_Int *) send_buffers_mpi[i];
         nalu_hypre_TMemcpy(qptr, &num_entries,
                       NALU_HYPRE_Int, 1, memory_location_mpi, NALU_HYPRE_MEMORY_HOST);
         qptr ++;
         nalu_hypre_TMemcpy(qptr, nalu_hypre_CommTypeRemBoxnums(comm_type),
                       NALU_HYPRE_Int, num_entries, memory_location_mpi, NALU_HYPRE_MEMORY_HOST);
         qptr += num_entries;
         nalu_hypre_TMemcpy(qptr, nalu_hypre_CommTypeRemBoxes(comm_type),
                       nalu_hypre_Box, num_entries, memory_location_mpi, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_CommTypeRemBoxnums(comm_type) = NULL;
         nalu_hypre_CommTypeRemBoxes(comm_type) = NULL;
      }
   }

   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

   j = 0;
   for (i = 0; i < num_recvs; i++)
   {
      comm_type = nalu_hypre_CommPkgRecvType(comm_pkg, i);
      nalu_hypre_MPI_Irecv(recv_buffers_mpi[i],
                      nalu_hypre_CommTypeBufsize(comm_type)*sizeof(NALU_HYPRE_Complex),
                      nalu_hypre_MPI_BYTE, nalu_hypre_CommTypeProc(comm_type),
                      tag, comm, &requests[j++]);
      if ( nalu_hypre_CommPkgFirstComm(comm_pkg) )
      {
         size = nalu_hypre_CommPrefixSize(nalu_hypre_CommTypeNumEntries(comm_type));
         nalu_hypre_CommTypeBufsize(comm_type)   -= size;
         nalu_hypre_CommPkgRecvBufsize(comm_pkg) -= size;
      }
   }

   for (i = 0; i < num_sends; i++)
   {
      comm_type = nalu_hypre_CommPkgSendType(comm_pkg, i);
      nalu_hypre_MPI_Isend(send_buffers_mpi[i],
                      nalu_hypre_CommTypeBufsize(comm_type)*sizeof(NALU_HYPRE_Complex),
                      nalu_hypre_MPI_BYTE, nalu_hypre_CommTypeProc(comm_type),
                      tag, comm, &requests[j++]);
      if ( nalu_hypre_CommPkgFirstComm(comm_pkg) )
      {
         size = nalu_hypre_CommPrefixSize(nalu_hypre_CommTypeNumEntries(comm_type));
         nalu_hypre_CommTypeBufsize(comm_type)   -= size;
         nalu_hypre_CommPkgSendBufsize(comm_pkg) -= size;
      }
   }

   /*--------------------------------------------------------------------
    * set up CopyToType and exchange local data
    *--------------------------------------------------------------------*/

   if ( nalu_hypre_CommPkgFirstComm(comm_pkg) )
   {
      from_type = nalu_hypre_CommPkgCopyFromType(comm_pkg);
      to_type   = nalu_hypre_CommPkgCopyToType(comm_pkg);
      num_entries = nalu_hypre_CommTypeNumEntries(from_type);
      nalu_hypre_CommTypeNumEntries(to_type) = num_entries;
      nalu_hypre_CommTypeEntries(to_type) =
         nalu_hypre_TAlloc(nalu_hypre_CommEntryType, num_entries, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CommTypeSetEntries(to_type,
                               nalu_hypre_CommTypeRemBoxnums(from_type),
                               nalu_hypre_CommTypeRemBoxes(from_type),
                               nalu_hypre_CommPkgRecvStride(comm_pkg),
                               nalu_hypre_CommPkgIdentityCoord(comm_pkg),
                               nalu_hypre_CommPkgIdentityDir(comm_pkg),
                               nalu_hypre_CommPkgIdentityOrder(comm_pkg),
                               nalu_hypre_CommPkgRecvDataSpace(comm_pkg),
                               nalu_hypre_CommPkgRecvDataOffsets(comm_pkg));
      nalu_hypre_TFree(nalu_hypre_CommPkgRemBoxnums(comm_pkg), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_CommPkgRemBoxes(comm_pkg), NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_ExchangeLocalData(comm_pkg, send_data, recv_data, action);

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = nalu_hypre_TAlloc(nalu_hypre_CommHandle, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_CommHandleCommPkg(comm_handle)        = comm_pkg;
   nalu_hypre_CommHandleSendData(comm_handle)       = send_data;
   nalu_hypre_CommHandleRecvData(comm_handle)       = recv_data;
   nalu_hypre_CommHandleNumRequests(comm_handle)    = num_requests;
   nalu_hypre_CommHandleRequests(comm_handle)       = requests;
   nalu_hypre_CommHandleStatus(comm_handle)         = status;
   nalu_hypre_CommHandleSendBuffers(comm_handle)    = send_buffers;
   nalu_hypre_CommHandleRecvBuffers(comm_handle)    = recv_buffers;
   nalu_hypre_CommHandleAction(comm_handle)         = action;
   nalu_hypre_CommHandleSendBuffersMPI(comm_handle) = send_buffers_mpi;
   nalu_hypre_CommHandleRecvBuffersMPI(comm_handle) = recv_buffers_mpi;

   *comm_handle_ptr = comm_handle;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Finalize a communication exchange.  This routine blocks until all
 * of the communication requests are completed.
 *
 * The communication requests are completed, and the receive buffer is
 * manually unpacked.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FinalizeCommunication( nalu_hypre_CommHandle *comm_handle )
{
   nalu_hypre_CommPkg       *comm_pkg         = nalu_hypre_CommHandleCommPkg(comm_handle);
   NALU_HYPRE_Complex      **send_buffers     = nalu_hypre_CommHandleSendBuffers(comm_handle);
   NALU_HYPRE_Complex      **recv_buffers     = nalu_hypre_CommHandleRecvBuffers(comm_handle);
   NALU_HYPRE_Complex      **send_buffers_mpi = nalu_hypre_CommHandleSendBuffersMPI(comm_handle);
   NALU_HYPRE_Complex      **recv_buffers_mpi = nalu_hypre_CommHandleRecvBuffersMPI(comm_handle);
   NALU_HYPRE_Int            action           = nalu_hypre_CommHandleAction(comm_handle);

   NALU_HYPRE_Int            ndim         = nalu_hypre_CommPkgNDim(comm_pkg);
   NALU_HYPRE_Int            num_values   = nalu_hypre_CommPkgNumValues(comm_pkg);
   NALU_HYPRE_Int            num_sends    = nalu_hypre_CommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int            num_recvs    = nalu_hypre_CommPkgNumRecvs(comm_pkg);

   nalu_hypre_CommType      *comm_type;
   nalu_hypre_CommEntryType *comm_entry;
   NALU_HYPRE_Int            num_entries;

   NALU_HYPRE_Int           *length_array;
   NALU_HYPRE_Int           *stride_array, unitst_array[NALU_HYPRE_MAXDIM + 1];

   NALU_HYPRE_Complex       *kptr, *lptr;
   NALU_HYPRE_Complex       *dptr;
   NALU_HYPRE_Int           *qptr;

   NALU_HYPRE_Int           *boxnums;
   nalu_hypre_Box           *boxes;

   NALU_HYPRE_Int            i, j, d, ll;

   NALU_HYPRE_MemoryLocation memory_location     = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());
   NALU_HYPRE_MemoryLocation memory_location_mpi = memory_location;

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if !defined(NALU_HYPRE_WITH_GPU_AWARE_MPI)
   memory_location_mpi = NALU_HYPRE_MEMORY_HOST;
#endif
#endif

   /*--------------------------------------------------------------------
    * finish communications
    *--------------------------------------------------------------------*/

   if (nalu_hypre_CommHandleNumRequests(comm_handle))
   {
      nalu_hypre_MPI_Waitall(nalu_hypre_CommHandleNumRequests(comm_handle),
                        nalu_hypre_CommHandleRequests(comm_handle),
                        nalu_hypre_CommHandleStatus(comm_handle));
   }

   /*--------------------------------------------------------------------
    * if FirstComm, unpack prefix information and set 'num_entries' and
    * 'entries' for RecvType
    *--------------------------------------------------------------------*/

   if ( nalu_hypre_CommPkgFirstComm(comm_pkg) )
   {
      nalu_hypre_CommEntryType *ct_entries;

      num_entries = 0;
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = nalu_hypre_CommPkgRecvType(comm_pkg, i);

         qptr = (NALU_HYPRE_Int *) recv_buffers_mpi[i];

         nalu_hypre_TMemcpy(&nalu_hypre_CommTypeNumEntries(comm_type), qptr,
                       NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, memory_location_mpi);

         num_entries += nalu_hypre_CommTypeNumEntries(comm_type);
      }

      /* allocate CommType entries 'ct_entries' */
      ct_entries = nalu_hypre_TAlloc(nalu_hypre_CommEntryType, num_entries, NALU_HYPRE_MEMORY_HOST);

      /* unpack prefix information and set RecvType entries */
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = nalu_hypre_CommPkgRecvType(comm_pkg, i);
         nalu_hypre_CommTypeEntries(comm_type) = ct_entries;
         ct_entries += nalu_hypre_CommTypeNumEntries(comm_type);

         qptr = (NALU_HYPRE_Int *) recv_buffers_mpi[i];
         //num_entries = *qptr;
         num_entries = nalu_hypre_CommTypeNumEntries(comm_type);
         qptr ++;
         boxnums = qptr;
         qptr += num_entries;
         boxes = (nalu_hypre_Box *) qptr;
         //TODO boxnums
         nalu_hypre_CommTypeSetEntries(comm_type, boxnums, boxes,
                                  nalu_hypre_CommPkgRecvStride(comm_pkg),
                                  nalu_hypre_CommPkgIdentityCoord(comm_pkg),
                                  nalu_hypre_CommPkgIdentityDir(comm_pkg),
                                  nalu_hypre_CommPkgIdentityOrder(comm_pkg),
                                  nalu_hypre_CommPkgRecvDataSpace(comm_pkg),
                                  nalu_hypre_CommPkgRecvDataOffsets(comm_pkg));
      }
   }

   /*--------------------------------------------------------------------
    * unpack receive buffer data
    *--------------------------------------------------------------------*/

   /* Note: nalu_hypre_CommPkgRecvBufsize is different in the first comm */
   if (recv_buffers != recv_buffers_mpi)
   {
      if (num_recvs > 0)
      {
         NALU_HYPRE_Int recv_buf_size;

         recv_buf_size = nalu_hypre_CommPkgFirstComm(comm_pkg) ? nalu_hypre_CommPkgRecvBufsizeFirstComm(comm_pkg) :
                         nalu_hypre_CommPkgRecvBufsize(comm_pkg);

         nalu_hypre_TMemcpy(recv_buffers[0], recv_buffers_mpi[0], NALU_HYPRE_Complex, recv_buf_size,
                       memory_location, memory_location_mpi);
      }
   }

   for (i = 0; i < num_recvs; i++)
   {
      comm_type = nalu_hypre_CommPkgRecvType(comm_pkg, i);
      num_entries = nalu_hypre_CommTypeNumEntries(comm_type);

      dptr = (NALU_HYPRE_Complex *) recv_buffers[i];

      if ( nalu_hypre_CommPkgFirstComm(comm_pkg) )
      {
         dptr += nalu_hypre_CommPrefixSize(num_entries);
      }

      for (j = 0; j < num_entries; j++)
      {
         comm_entry = nalu_hypre_CommTypeEntry(comm_type, j);
         length_array = nalu_hypre_CommEntryTypeLengthArray(comm_entry);
         stride_array = nalu_hypre_CommEntryTypeStrideArray(comm_entry);
         unitst_array[0] = 1;
         for (d = 1; d <= ndim; d++)
         {
            unitst_array[d] = unitst_array[d - 1] * length_array[d - 1];
         }

         lptr = nalu_hypre_CommHandleRecvData(comm_handle) +
                nalu_hypre_CommEntryTypeOffset(comm_entry);
         for (ll = 0; ll < num_values; ll++)
         {
            kptr = lptr + ll * stride_array[ndim];

#define DEVICE_VAR is_device_ptr(kptr,dptr)
            nalu_hypre_BasicBoxLoop2Begin(ndim, length_array,
                                     stride_array, ki,
                                     unitst_array, di);
            {
               if (action > 0)
               {
                  kptr[ki] += dptr[di];
               }
               else
               {
                  kptr[ki] = dptr[di];
               }
            }
            nalu_hypre_BoxLoop2End(ki, di);
#undef DEVICE_VAR

            dptr += unitst_array[ndim];
         }
      }
   }

   /*--------------------------------------------------------------------
    * turn off first communication indicator
    *--------------------------------------------------------------------*/

   nalu_hypre_CommPkgFirstComm(comm_pkg) = 0;

   /*--------------------------------------------------------------------
    * Free up communication handle
    *--------------------------------------------------------------------*/

   nalu_hypre_TFree(nalu_hypre_CommHandleRequests(comm_handle), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_CommHandleStatus(comm_handle), NALU_HYPRE_MEMORY_HOST);
   if (num_sends > 0)
   {
      nalu_hypre_StructCommunicationReleaseBuffer(send_buffers[0], memory_location);
   }
   if (num_recvs > 0)
   {
      nalu_hypre_StructCommunicationReleaseBuffer(recv_buffers[0], memory_location);
   }

   nalu_hypre_TFree(comm_handle, NALU_HYPRE_MEMORY_HOST);

   if (send_buffers != send_buffers_mpi)
   {
      nalu_hypre_TFree(send_buffers_mpi[0], memory_location_mpi);
      nalu_hypre_TFree(send_buffers_mpi, NALU_HYPRE_MEMORY_HOST);
   }
   if (recv_buffers != recv_buffers_mpi)
   {
      nalu_hypre_TFree(recv_buffers_mpi[0], memory_location_mpi);
      nalu_hypre_TFree(recv_buffers_mpi, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(send_buffers, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_buffers, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Execute local data exchanges.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ExchangeLocalData( nalu_hypre_CommPkg *comm_pkg,
                         NALU_HYPRE_Complex *send_data,
                         NALU_HYPRE_Complex *recv_data,
                         NALU_HYPRE_Int      action )
{
   NALU_HYPRE_Int            ndim       = nalu_hypre_CommPkgNDim(comm_pkg);
   NALU_HYPRE_Int            num_values = nalu_hypre_CommPkgNumValues(comm_pkg);
   nalu_hypre_CommType      *copy_fr_type;
   nalu_hypre_CommType      *copy_to_type;
   nalu_hypre_CommEntryType *copy_fr_entry;
   nalu_hypre_CommEntryType *copy_to_entry;

   NALU_HYPRE_Complex       *fr_dp;
   NALU_HYPRE_Int           *fr_stride_array;
   NALU_HYPRE_Complex       *to_dp;
   NALU_HYPRE_Int           *to_stride_array;
   NALU_HYPRE_Complex       *fr_dpl, *to_dpl;

   NALU_HYPRE_Int           *length_array;
   NALU_HYPRE_Int            i, ll;

   NALU_HYPRE_Int           *order;

   /*--------------------------------------------------------------------
    * copy local data
    *--------------------------------------------------------------------*/

   copy_fr_type = nalu_hypre_CommPkgCopyFromType(comm_pkg);
   copy_to_type = nalu_hypre_CommPkgCopyToType(comm_pkg);

   for (i = 0; i < nalu_hypre_CommTypeNumEntries(copy_fr_type); i++)
   {
      copy_fr_entry = nalu_hypre_CommTypeEntry(copy_fr_type, i);
      copy_to_entry = nalu_hypre_CommTypeEntry(copy_to_type, i);

      fr_dp = send_data + nalu_hypre_CommEntryTypeOffset(copy_fr_entry);
      to_dp = recv_data + nalu_hypre_CommEntryTypeOffset(copy_to_entry);

      /* copy data only when necessary */
      if (to_dp != fr_dp)
      {
         length_array = nalu_hypre_CommEntryTypeLengthArray(copy_fr_entry);

         fr_stride_array = nalu_hypre_CommEntryTypeStrideArray(copy_fr_entry);
         to_stride_array = nalu_hypre_CommEntryTypeStrideArray(copy_to_entry);
         order = nalu_hypre_CommEntryTypeOrder(copy_fr_entry);

         for (ll = 0; ll < num_values; ll++)
         {
            if (order[ll] > -1)
            {
               fr_dpl = fr_dp + (order[ll]) * fr_stride_array[ndim];
               to_dpl = to_dp + (      ll ) * to_stride_array[ndim];

#define DEVICE_VAR is_device_ptr(to_dpl,fr_dpl)
               nalu_hypre_BasicBoxLoop2Begin(ndim, length_array,
                                        fr_stride_array, fi,
                                        to_stride_array, ti);
               {
                  if (action > 0)
                  {
                     /* add the data to existing values in memory */
                     to_dpl[ti] += fr_dpl[fi];
                  }
                  else
                  {
                     /* copy the data over existing values in memory */
                     to_dpl[ti] = fr_dpl[fi];
                  }
               }
               nalu_hypre_BoxLoop2End(fi, ti);
#undef DEVICE_VAR
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommPkgDestroy( nalu_hypre_CommPkg *comm_pkg )
{
   nalu_hypre_CommType  *comm_type;
   NALU_HYPRE_Int      **orders;
   NALU_HYPRE_Int        i;

   if (comm_pkg)
   {
      /* note that entries are allocated in two stages for To/Recv */
      if (nalu_hypre_CommPkgNumRecvs(comm_pkg) > 0)
      {
         comm_type = nalu_hypre_CommPkgRecvType(comm_pkg, 0);
         nalu_hypre_TFree(nalu_hypre_CommTypeEntries(comm_type), NALU_HYPRE_MEMORY_HOST);
      }
      comm_type = nalu_hypre_CommPkgCopyToType(comm_pkg);
      nalu_hypre_TFree(nalu_hypre_CommTypeEntries(comm_type), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(comm_type, NALU_HYPRE_MEMORY_HOST);

      comm_type = nalu_hypre_CommPkgCopyFromType(comm_pkg);
      nalu_hypre_TFree(comm_type, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(nalu_hypre_CommPkgEntries(comm_pkg), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_CommPkgRemBoxnums(comm_pkg), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_CommPkgRemBoxes(comm_pkg), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(nalu_hypre_CommPkgRecvDataOffsets(comm_pkg), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_BoxArrayDestroy(nalu_hypre_CommPkgRecvDataSpace(comm_pkg));

      orders = nalu_hypre_CommPkgOrders(comm_pkg);
      for (i = 0; i < nalu_hypre_CommPkgNumOrders(comm_pkg); i++)
      {
         nalu_hypre_TFree(orders[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(orders, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(nalu_hypre_CommPkgIdentityOrder(comm_pkg), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(comm_pkg, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}
