/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_FE_MV_HEADER
#define NALU_HYPRE_FE_MV_HEADER

#include "NALU_HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FEI System Interface
 *
 * This interface represents a FE conceptual view of a
 * linear system.  
 *
 * @memo A FE conceptual interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FE Mesh
 **/
/*@{*/

struct hypre_FEMesh_struct;
typedef struct hypre_FEMesh_struct *NALU_HYPRE_FEMesh;

/**
 * Create a FE Mesh object.  
 **/

int NALU_HYPRE_FEMeshCreate(MPI_Comm comm, NALU_HYPRE_FEMesh *mesh);

/**
 * Destroy an FE Mesh object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  
 **/

int NALU_HYPRE_FEMeshDestroy(NALU_HYPRE_FEMesh mesh);

/**
 * load an FE object
 **/

int NALU_HYPRE_FEMeshSetFEObject(NALU_HYPRE_FEMesh mesh, void *, void *);

/**
 * initialize all fields in the finite element mesh
 **/

int NALU_HYPRE_FEMeshInitFields(NALU_HYPRE_FEMesh mesh, int numFields, 
                           int *fieldSizes, int *fieldIDs);

/**
 * initialize an element block
 **/

int NALU_HYPRE_FEMeshInitElemBlock(NALU_HYPRE_FEMesh mesh, int blockID, int nElements,
                int numNodesPerElement, int *numFieldsPerNode,
                int **nodalFieldIDs, int numElemDOFFieldsPerElement,
                int *elemDOFFieldIDs, int interleaveStrategy);

/**
 * initialize the connectivity of a given element
 **/

int NALU_HYPRE_FEMeshInitElem(NALU_HYPRE_FEMesh mesh, int blockID, int elemID,
                         int *elemConn);

/**
 * initialize the shared nodes between processors
 **/

int NALU_HYPRE_FEMeshInitSharedNodes(NALU_HYPRE_FEMesh mesh, int nShared,
                                int *sharedIDs, int *sharedLeng,
                                int **sharedProcs);

/**
 * initialization complete
 **/

int NALU_HYPRE_FEMeshInitComplete(NALU_HYPRE_FEMesh mesh);

/**
 * load node boundary conditions
 **/

int NALU_HYPRE_FEMeshLoadNodeBCs(NALU_HYPRE_FEMesh mesh, int numNodes,
                            int *nodeIDs, int fieldID, double **alpha,
                            double **beta, double **gamma);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FE Matrices
 **/
/*@{*/

struct hypre_FEMatrix_struct;
/**
 * The matrix object
 **/
typedef struct hypre_FEMatrix_struct *NALU_HYPRE_FEMatrix;

/**
 * create a new FE matrix
 **/

int NALU_HYPRE_FEMatrixCreate(MPI_Comm comm, NALU_HYPRE_FEMesh mesh, 
                         NALU_HYPRE_FEMatrix *matrix);

/**
 * destroy a new FE matrix
 **/

int NALU_HYPRE_FEMatrixDestroy(NALU_HYPRE_FEMatrix matrix);
   
/**
 * prepare a matrix object for setting coefficient values
 **/

int NALU_HYPRE_FEMatrixInitialize(NALU_HYPRE_FEMatrix matrix);
   
/**
 * signal that loading has been completed
 **/

int NALU_HYPRE_FEMatrixAssemble(NALU_HYPRE_FEMatrix matrix);
   
/**
 * Set the storage type of the matrix object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR} (default).
 *
 **/
int NALU_HYPRE_FEMatrixSetObjectType(NALU_HYPRE_FEMatrix  matrix, int type);

/**
 * Get a reference to the constructed matrix object.
 *
 * @see NALU_HYPRE_FEMatrixSetObjectType
 **/
int NALU_HYPRE_FEMatrixGetObject(NALU_HYPRE_FEMatrix matrix, void **object);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FE Vectors
 **/
/*@{*/

struct hypre_FEVector_struct;
/**
 * The vector object.
 **/
typedef struct hypre_FEVector_struct *NALU_HYPRE_FEVector;

/**
 * Create a vector object.
 **/
int NALU_HYPRE_FEVectorCreate(MPI_Comm comm, NALU_HYPRE_FEMesh mesh,
                         NALU_HYPRE_FEVector  *vector);

/**
 * Destroy a vector object.
 **/
int NALU_HYPRE_FEVectorDestroy(NALU_HYPRE_FEVector vector);

/**
 * Prepare a vector object for setting coefficient values.
 **/
int NALU_HYPRE_FEVectorInitialize(NALU_HYPRE_FEVector vector);


/**
 * Finalize the construction of the vector before using.
 **/
int NALU_HYPRE_FEVectorAssemble(NALU_HYPRE_FEVector vector);

/**
 * Set the storage type of the vector object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR} (default).
 **/
int NALU_HYPRE_FEVectorSetObjectType(NALU_HYPRE_FEVector vector, int type);

/**
 * Get a reference to the constructed vector object.
 **/
int NALU_HYPRE_FEVectorGetObject(NALU_HYPRE_FEVector vector, void **object);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif

