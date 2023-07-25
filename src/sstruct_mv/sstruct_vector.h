/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_SStructVector structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_SSTRUCT_VECTOR_HEADER
#define nalu_hypre_SSTRUCT_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
   nalu_hypre_SStructPGrid     *pgrid;

   NALU_HYPRE_Int               nvars;
   nalu_hypre_StructVector    **svectors;     /* nvar array of svectors */
   nalu_hypre_CommPkg         **comm_pkgs;    /* nvar array of comm pkgs */

   NALU_HYPRE_Int               accumulated;  /* AddTo values accumulated? */

   NALU_HYPRE_Int               ref_count;

   NALU_HYPRE_Int              *dataindices;  /* GEC1002 array for starting index of the
                                            svector. pdataindices[varx] */
   NALU_HYPRE_Int               datasize;     /* Size of the pvector = sums size of svectors */

} nalu_hypre_SStructPVector;

typedef struct nalu_hypre_SStructVector_struct
{
   MPI_Comm                comm;
   NALU_HYPRE_Int               ndim;
   nalu_hypre_SStructGrid      *grid;
   NALU_HYPRE_Int               object_type;

   /* s-vector info */
   NALU_HYPRE_Int               nparts;
   nalu_hypre_SStructPVector  **pvectors;
   nalu_hypre_CommPkg        ***comm_pkgs;    /* nvar array of comm pkgs */

   /* u-vector info */
   NALU_HYPRE_IJVector          ijvector;
   nalu_hypre_ParVector        *parvector;

   /* inter-part communication info */
   NALU_HYPRE_Int               nbor_ncomms;  /* num comm_pkgs with neighbor parts */

   /* GEC10020902 pointer to big chunk of memory and auxiliary information */
   NALU_HYPRE_Complex          *data;        /* GEC1002 pointer to chunk data */
   NALU_HYPRE_Int              *dataindices; /* GEC1002 dataindices[partx] is the starting index
                                           of vector data for the part=partx */
   NALU_HYPRE_Int               datasize;    /* GEC1002 size of all data = ghlocalsize */

   NALU_HYPRE_Int               global_size;  /* Total number coefficients */
   NALU_HYPRE_Int               ref_count;

} nalu_hypre_SStructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructVector
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructVectorComm(vec)           ((vec) -> comm)
#define nalu_hypre_SStructVectorNDim(vec)           ((vec) -> ndim)
#define nalu_hypre_SStructVectorGrid(vec)           ((vec) -> grid)
#define nalu_hypre_SStructVectorObjectType(vec)     ((vec) -> object_type)
#define nalu_hypre_SStructVectorNParts(vec)         ((vec) -> nparts)
#define nalu_hypre_SStructVectorPVectors(vec)       ((vec) -> pvectors)
#define nalu_hypre_SStructVectorPVector(vec, part)  ((vec) -> pvectors[part])
#define nalu_hypre_SStructVectorIJVector(vec)       ((vec) -> ijvector)
#define nalu_hypre_SStructVectorParVector(vec)      ((vec) -> parvector)
#define nalu_hypre_SStructVectorNborNComms(vec)     ((vec) -> nbor_ncomms)
#define nalu_hypre_SStructVectorGlobalSize(vec)     ((vec) -> global_size)
#define nalu_hypre_SStructVectorRefCount(vec)       ((vec) -> ref_count)
#define nalu_hypre_SStructVectorData(vec)           ((vec) -> data )
#define nalu_hypre_SStructVectorDataIndices(vec)    ((vec) -> dataindices)
#define nalu_hypre_SStructVectorDataSize(vec)       ((vec) -> datasize)


/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructPVector
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructPVectorComm(pvec)        ((pvec) -> comm)
#define nalu_hypre_SStructPVectorPGrid(pvec)       ((pvec) -> pgrid)
#define nalu_hypre_SStructPVectorNVars(pvec)       ((pvec) -> nvars)
#define nalu_hypre_SStructPVectorSVectors(pvec)    ((pvec) -> svectors)
#define nalu_hypre_SStructPVectorSVector(pvec, v)  ((pvec) -> svectors[v])
#define nalu_hypre_SStructPVectorCommPkgs(pvec)    ((pvec) -> comm_pkgs)
#define nalu_hypre_SStructPVectorCommPkg(pvec, v)  ((pvec) -> comm_pkgs[v])
#define nalu_hypre_SStructPVectorAccumulated(pvec) ((pvec) -> accumulated)
#define nalu_hypre_SStructPVectorRefCount(pvec)    ((pvec) -> ref_count)
#define nalu_hypre_SStructPVectorDataIndices(pvec) ((pvec) -> dataindices  )
#define nalu_hypre_SStructPVectorDataSize(pvec)    ((pvec) -> datasize  )

#endif
