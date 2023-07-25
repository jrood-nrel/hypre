/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

// NEW FEI (2.23.02)
#include "fei_LinearSystemCore.hpp"
#include "LLNL_FEI_Impl.h"

#ifndef nalu_hypre_FE_MV_HEADER
#define nalu_hypre_FE_MV_HEADER

#include "NALU_HYPRE.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 *
 * Header info for the nalu_hypre_FEMesh structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_FE_MESH_HEADER
#define nalu_hypre_FE_MESH_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_FEMesh:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_FEMesh_struct
{
   MPI_Comm comm_;
   void     *linSys_;
   void     *feiPtr_;
   int      objectType_;

} nalu_hypre_FEMesh;
typedef struct nalu_hypre_FEMesh_struct *NALU_HYPRE_FEMesh;

#endif

/******************************************************************************
 *
 * Header info for the nalu_hypre_FEMatrix structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_FE_MATRIX_HEADER
#define nalu_hypre_FE_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_FEMatrix:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_FEMatrix_struct
{
   MPI_Comm      comm_;
   nalu_hypre_FEMesh *mesh_;

} nalu_hypre_FEMatrix;
typedef struct nalu_hypre_FEMatrix_struct *NALU_HYPRE_FEMatrix;

#endif

/******************************************************************************
 *
 * Header info for the nalu_hypre_FEVector structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_FE_VECTOR_HEADER
#define nalu_hypre_FE_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_FEVector:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_FEVector_struct
{
   MPI_Comm      comm_;
   nalu_hypre_FEMesh* mesh_;

} nalu_hypre_FEVector;
typedef struct nalu_hypre_FEVector_struct *NALU_HYPRE_FEVector;

#endif

/**
 * @name NALU_HYPRE's Finite Element Interface
 *
 * @memo A finite element-based conceptual interface
 **/
/*@{*/
                                                                                
/**
 * @name NALU_HYPRE FEI functions
 **/
/*@{*/
                                                                                
/*--------------------------------------------------------------------------
 * NALU_HYPRE_fei_mesh.cxx 
 *--------------------------------------------------------------------------*/

/**
  * Finite element interface constructor: this function creates an
  * instantiation of the NALU_HYPRE fei class.
  * @param comm - an MPI communicator
  * @param mesh - upon return, contains a pointer to the finite element mesh
 **/

int NALU_HYPRE_FEMeshCreate( MPI_Comm comm, NALU_HYPRE_FEMesh *mesh );

/**
  * Finite element interface destructor: this function destroys
  * the object as well as its internal memory allocations.
  * @param mesh - a pointer to the finite element mesh 
 **/

int NALU_HYPRE_FEMeshDestroy( NALU_HYPRE_FEMesh mesh );

/**
  * This function passes the externally-built FEI object (for example,
  * Sandia's implementation of the FEI) as well as corresponding
  * LinearSystemCore object. 
  * @param mesh - a pointer to the finite element mesh 
  * @param externFEI - a pointer to the externally built finite element object 
  * @param linSys - a pointer to the NALU_HYPRE linear system solver object built
  *                 using the NALU_HYPRE\_base\_create function.
 **/

int NALU_HYPRE_FEMeshSetFEObject(NALU_HYPRE_FEMesh mesh, void *externFEI, void *linSys);

/**
  * The parameters function is the single most important function
  * to pass solver information (which solver, which preconditioner,
  * tolerance, other solver parameters) to the underlying solver.
  * @param mesh - a pointer to the finite element mesh 
  * @param numParams - number of command strings
  * @param paramStrings - the command strings
 **/
int NALU_HYPRE_FEMeshParameters(NALU_HYPRE_FEMesh mesh, int numParams, char **paramStrings);

/**
  * Each node or element variable has one or more fields. The field
  * information can be set up using this function.
  * @param mesh - a pointer to the finite element mesh 
  * @param numFields - total number of fields for all variable types
  * @param fieldSizes - degree of freedom for each field type
  * @param fieldIDs - a list of field identifiers
 **/

int NALU_HYPRE_FEMeshInitFields( NALU_HYPRE_FEMesh mesh, int numFields,
                            int *fieldSizes, int *fieldIDs );

/**
  * The whole finite element mesh can be broken down into a number of
  * element blocks. The attributes for each element block are: an
  * identifier, number of elements, number of nodes per elements,
  * the number of fields in each element node, etc.
  * @param mesh - a pointer to the finite element mesh 
  * @param blockID - element block identifier
  * @param nElements - number of element in this block
  * @param numNodesPerElement - number of nodes per element in this block
  * @param numFieldsPerNode - number of fields for each node
  * @param nodalFieldIDs - field identifiers for the nodal unknowns
  * @param numElemDOFFieldsPerElement - number of fields for the element
  * @param elemDOFFieldIDs - field identifier for the element unknowns
  * @param interleaveStratety - indicates how unknowns are ordered
  */

int NALU_HYPRE_FEMeshInitElemBlock( NALU_HYPRE_FEMesh mesh, int blockID, 
                               int nElements, int numNodesPerElement,
                               int *numFieldsPerNode, int **nodalFieldIDs,
                               int numElemDOFFieldsPerElement,
                               int *elemDOFFieldIDs, int interleaveStrategy );

/**
  * This function initializes element connectivity (that is, the node
  * identifiers associated with the current element) given an element
  * block identifier and the element identifier with the element block.
  * @param mesh - a pointer to the finite element mesh 
  * @param blockID - element block identifier
  * @param elemID - element identifier
  * @param elemConn - a list of node identifiers for this element
 **/

int NALU_HYPRE_FEMeshInitElem( NALU_HYPRE_FEMesh mesh, int blockID, int elemID,
                          int *elemConn );

/**
  * This function initializes the nodes that are shared between the
  * current processor and its neighbors. The FEI will decide a unique
  * processor each shared node will be assigned to.
  * @param mesh - a pointer to the finite element mesh 
  * @param nShared - number of shared nodes
  * @param sharedIDs - shared node identifiers
  * @param sharedLengs - the number of processors each node shares with
  * @param sharedProcs - the processor identifiers each node shares with
 **/

int NALU_HYPRE_FEMeshInitSharedNodes( NALU_HYPRE_FEMesh mesh, int nShared,
                                 int *sharedIDs, int *sharedLeng,
                                 int **sharedProcs );

/**
  * This function signals to the FEI that the initialization step has
  * been completed. The loading step will follow.
  * @param mesh - a pointer to the finite element mesh 
 **/

int NALU_HYPRE_FEMeshInitComplete( NALU_HYPRE_FEMesh mesh );

/**
  * This function loads the nodal boundary conditions. The boundary conditions
  * @param mesh - a pointer to the finite element mesh 
  * @param nNodes - number of nodes boundary conditions are imposed
  * @param nodeIDs - nodal identifiers
  * @param fieldID - field identifier with nodes where BC are imposed
  * @param alpha - the multipliers for the field
  * @param beta - the multipliers for the normal derivative of the field
  * @param gamma - the boundary values on the right hand side of the equations
 **/

int NALU_HYPRE_FEMeshLoadNodeBCs( NALU_HYPRE_FEMesh mesh, int numNodes,
                             int *nodeIDs, int fieldID, double **alpha,
                             double **beta, double **gamma );

/**
  * This function adds the element contribution to the global stiffness matrix
  * and also the element load to the right hand side vector
  * @param mesh - a pointer to the finite element mesh 
  * @param BlockID - element block identifier
  * @param elemID - element identifier
  * @param elemConn - a list of node identifiers for this element
  * @param elemStiff - element stiffness matrix
  * @param elemLoad - right hand side (load) for this element
  * @param elemFormat - the format the unknowns are passed in
 **/

int NALU_HYPRE_FEMeshSumInElem( NALU_HYPRE_FEMesh mesh, int blockID, int elemID, 
                           int* elemConn, double** elemStiffness, 
                           double *elemLoad, int elemFormat );

/**
  * This function differs from the sumInElem function in that the right hand
  * load vector is not passed.
  * @param mesh - a pointer to the finite element mesh 
  * @param blockID - element block identifier
  * @param elemID - element identifier
  * @param elemConn - a list of node identifiers for this element
  * @param elemStiff - element stiffness matrix
  * @param elemFormat - the format the unknowns are passed in
 **/

int NALU_HYPRE_FEMeshSumInElemMatrix( NALU_HYPRE_FEMesh mesh, int blockID, int elemID, 
                                 int* elemConn, double** elemStiffness, 
                                 int elemFormat );

/**
  * This function adds the element load to the right hand side vector
  * @param mesh - a pointer to the finite element mesh 
  * @param blockID - element block identifier
  * @param elemID - element identifier
  * @param elemConn - a list of node identifiers for this element
  * @param elemLoad - right hand side (load) for this element
 **/

int NALU_HYPRE_FEMeshSumInElemRHS( NALU_HYPRE_FEMesh mesh, int blockID, int elemID, 
                              int* elemConn, double* elemLoad );

/**
  * This function signals to the FEI that the loading phase has
  * been completed.
  * @param mesh - a pointer to the finite element mesh 
 **/

int NALU_HYPRE_FEMeshLoadComplete( NALU_HYPRE_FEMesh mesh );

/**
  * This function tells the FEI to solve the linear system
  * @param mesh - a pointer to the finite element mesh
 **/

int NALU_HYPRE_FEMeshSolve( NALU_HYPRE_FEMesh mesh );

/**
  * This function sends a solution vector to the FEI 
  * @param mesh - a pointer to the finite element mesh
  * @param sol - solution vector 
 **/

int NALU_HYPRE_FEMeshSetSolution( NALU_HYPRE_FEMesh mesh, void *sol );

/**
  * This function returns the node identifiers given the element block.
  * @param mesh - a pointer to the finite element mesh
  * @param blockID - element block identifier
  * @param numNodes - the number of nodes
  * @param nodeIDList - the node identifiers
 **/

int NALU_HYPRE_FEMeshGetBlockNodeIDList( NALU_HYPRE_FEMesh mesh, int blockID, 
                                    int numNodes, int *nodeIDList );

/**
  * This function returns the nodal solutions given the element block number.
  * @param mesh - a pointer to the finite element mesh
  * @param blockID - element block identifier
  * @param numNodes - the number of nodes
  * @param nodeIDList - the node identifiers
  * @param solnOffsets - the equation number for each nodal solution
  * @param solnValues - the nodal solution values
 **/

int NALU_HYPRE_FEMeshGetBlockNodeSolution( NALU_HYPRE_FEMesh mesh, int blockID,
                                      int numNodes, int *nodeIDList, 
                                      int *solnOffsets, double *solnValues );

/*@}*/

/**
 * @name NALU_HYPRE FEI Matrix functions
 **/
/*@{*/
                                                                                
/*--------------------------------------------------------------------------
 * NALU_HYPRE_fei_matrix.cxx 
 *--------------------------------------------------------------------------*/
/**
  * Finite element matrix constructor
  * @param comm - an MPI communicator
  * @param mesh - a pointer to the finite element mesh
  * @param matrix - upon return, contains a pointer to the FE matrix
 **/

int NALU_HYPRE_FEMatrixCreate( MPI_Comm comm, NALU_HYPRE_FEMesh mesh, 
                          NALU_HYPRE_FEMatrix *matrix );
/**
  * Finite element matrix destructor
  * @param matrix - a pointer to the FE matrix
 **/

int NALU_HYPRE_FEMatrixDestroy( NALU_HYPRE_FEMatrix matrix );

/**
  * This function gets the underlying NALU_HYPRE parcsr matrix from the FE mesh
  * @param matrix - a pointer to the FE matrix
  * @param object - a pointer to the NALU_HYPRE parcsr matrix
 **/

int NALU_HYPRE_FEMatrixGetObject( NALU_HYPRE_FEMatrix matrix, void **object );

/*@}*/

/**
 * @name NALU_HYPRE FEI Matrix functions
 **/
/*@{*/
                                                                                
/*--------------------------------------------------------------------------
 * NALU_HYPRE_fei_vector.cxx 
 *--------------------------------------------------------------------------*/
/**
  * Finite element vector constructor
  * @param comm - an MPI communicator
  * @param mesh - a pointer to the finite element mesh
  * @param vector - upon return, contains a pointer to the FE vector
 **/
int NALU_HYPRE_FEVectorCreate( MPI_Comm comm , NALU_HYPRE_FEMesh mesh, 
                          NALU_HYPRE_FEVector *vector);

/**
  * Finite element vector destructor
  * @param vector - a pointer to the FE vector
 **/
int NALU_HYPRE_FEVectorDestroy( NALU_HYPRE_FEVector vector );

/**
  * This function gets the underlying RHS vector from the FE mesh
  * @param vector - a pointer to the FE vector
  * @param object - upon return, points to the RHS vector
 **/

int NALU_HYPRE_FEVectorGetRHS( NALU_HYPRE_FEVector vector, void **object );

/**
  * This function gives the solution vector to the FE mesh
  * @param vector - a pointer to the FE vector
  * @param object - points to the solution vector
 **/

int NALU_HYPRE_FEVectorSetSol( NALU_HYPRE_FEVector vector, void *object );

/*@}*/
/*@}*/

#ifdef __cplusplus
}
#endif

#endif

