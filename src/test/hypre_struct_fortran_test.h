/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif
/**************************************************
*  Definitions of struct fortran interface routines
**************************************************/


#define NALU_HYPRE_StructStencilCreate \
        hypre_F90_NAME(fhypre_structstencilcreate, FNALU_HYPRE_STRUCTSTENCILCREATE)
extern void hypre_F90_NAME(fhypre_structstencilcreate, FNALU_HYPRE_STRUCTSTENCILCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructStencilDestroy \
        hypre_F90_NAME(fhypre_structstencildestroy, FNALU_HYPRE_STRUCTSTENCILDESTROY)
extern void hypre_F90_NAME(fhypre_structstencildestroy, FNALU_HYPRE_STRUCTSTENCILDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructStencilSetElement \
        hypre_F90_NAME(fhypre_structstencilsetelement, FNALU_HYPRE_STRUCTSTENCILSETELEMENT)
extern void hypre_F90_NAME(fhypre_structstencilsetelement, FNALU_HYPRE_STRUCTSTENCILSETELEMENT)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);



#define NALU_HYPRE_StructGridCreate \
        hypre_F90_NAME(fhypre_structgridcreate, FNALU_HYPRE_STRUCTGRIDCREATE)
extern void hypre_F90_NAME(fhypre_structgridcreate, FNALU_HYPRE_STRUCTGRIDCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructGridDestroy \
        hypre_F90_NAME(fhypre_structgriddestroy, FNALU_HYPRE_STRUCTGRIDDESTROY)
extern void hypre_F90_NAME(fhypre_structgriddestroy, FNALU_HYPRE_STRUCTGRIDDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructGridSetExtents \
        hypre_F90_NAME(fhypre_structgridsetextents, FNALU_HYPRE_STRUCTGRIDSETEXTENTS)
extern void hypre_F90_NAME(fhypre_structgridsetextents, FNALU_HYPRE_STRUCTGRIDSETEXTENTS)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGridSetPeriodic \
        hypre_F90_NAME(fhypre_structgridsetperiodic, FNALU_HYPRE_STRUCTGRIDSETPERIODIC)
extern void hypre_F90_NAME(fhypre_structgridsetperiodic, fhypre_structsetgridperiodic)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGridAssemble \
        hypre_F90_NAME(fhypre_structgridassemble, FNALU_HYPRE_STRUCTGRIDASSEMBLE)
extern void hypre_F90_NAME(fhypre_structgridassemble, FNALU_HYPRE_STRUCTGRIDASSEMBLE)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructGridSetNumGhost \
        hypre_F90_NAME(fhypre_structgridsetnumghost, FNALU_HYPRE_STRUCTGRIDSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_structgridsetnumghost, fhypre_structsetgridnumghost)
(hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_StructMatrixCreate \
        hypre_F90_NAME(fhypre_structmatrixcreate, FNALU_HYPRE_STRUCTMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_structmatrixcreate, FNALU_HYPRE_STRUCTMATRIXCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixDestroy \
        hypre_F90_NAME(fhypre_structmatrixdestroy, FNALU_HYPRE_STRUCTMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_structmatrixdestroy, FNALU_HYPRE_STRUCTMATRIXDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixInitialize \
        hypre_F90_NAME(fhypre_structmatrixinitialize, FNALU_HYPRE_STRUCTMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_structmatrixinitialize, FNALU_HYPRE_STRUCTMATRIXINITIALIZE)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixSetValues \
        hypre_F90_NAME(fhypre_structmatrixsetvalues, FNALU_HYPRE_STRUCTMATRIXSETVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixsetvalues, FNALU_HYPRE_STRUCTMATRIXSETVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixSetBoxValues \
        hypre_F90_NAME(fhypre_structmatrixsetboxvalues, FNALU_HYPRE_STRUCTMATRIXSETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixsetboxvalues, FNALU_HYPRE_STRUCTMATRIXSETBOXVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixGetBoxValues \
        hypre_F90_NAME(fhypre_structmatrixgetboxvalues, FNALU_HYPRE_STRUCTMATRIXGETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixgetboxvalues, FNALU_HYPRE_STRUCTMATRIXGETBOXVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixSetConstantEntries \
        hypre_F90_NAME(fhypre_structmatrixsetconstante, FNALU_HYPRE_STRUCTMATRIXSETCONSTANTE)
extern void hypre_F90_NAME(fhypre_structmatrixsetconstante, FNALU_HYPRE_STRUCTMATRIXSETCONSTANTE)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixSetConstantValues \
        hypre_F90_NAME(fhypre_structmatrixsetconstantv, FNALU_HYPRE_STRUCTMATRIXSETCONSTANTV)
extern void hypre_F90_NAME(fhypre_structmatrixsetconstantv, FNALU_HYPRE_STRUCTMATRIXSETCONSTANTV)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixAddToValues \
        hypre_F90_NAME(fhypre_structmatrixaddtovalues, FNALU_HYPRE_STRUCTMATRIXADDTOVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixaddtovalues, FNALU_HYPRE_STRUCTMATRIXADDTOVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixAddToBoxValues \
        hypre_F90_NAME(fhypre_structmatrixaddtoboxvalues, FNALU_HYPRE_STRUCTMATRIXADDTOBOXVALUES)
extern void hypre_F90_NAME(fhypre_structmatrixaddtoboxvalues, FNALU_HYPRE_STRUCTMATRIXADDTOBOXVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixAddToConstantValues \
        hypre_F90_NAME(fhypre_structmatrixaddtoconstant, FNALU_HYPRE_STRUCTMATRIXADDTOCONSTANT)
extern void hypre_F90_NAME(fhypre_structmatrixaddtoconstant, FNALU_HYPRE_STRUCTMATRIXADDTOCONSTANT)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixAssemble \
        hypre_F90_NAME(fhypre_structmatrixassemble, FNALU_HYPRE_STRUCTMATRIXASSEMBLE)
extern void hypre_F90_NAME(fhypre_structmatrixassemble, FNALU_HYPRE_STRUCTMATRIXASSEMBLE)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixSetNumGhost \
        hypre_F90_NAME(fhypre_structmatrixsetnumghost, FNALU_HYPRE_STRUCTMATRIXSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_structmatrixsetnumghost, FNALU_HYPRE_STRUCTMATRIXSETNUMGHOST)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixGetGrid \
        hypre_F90_NAME(fhypre_structmatrixgetgrid, FNALU_HYPRE_STRUCTMATRIXGETGRID)
extern void hypre_F90_NAME(fhypre_structmatrixgetgrid, FNALU_HYPRE_STRUCTMATRIXGETGRID)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixSetSymmetric \
        hypre_F90_NAME(fhypre_structmatrixsetsymmetric, FNALU_HYPRE_STRUCTMATRIXSETSYMMETRIC)
extern void hypre_F90_NAME(fhypre_structmatrixsetsymmetric, FNALU_HYPRE_STRUCTMATRIXSETSYMMETRIC)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixPrint \
hypre_F90_NAME(fhypre_structmatrixprint, FNALU_HYPRE_STRUCTMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_structmatrixprint, FNALU_HYPRE_STRUCTMATRIXPRINT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixMatvec \
hypre_F90_NAME(fhypre_structmatrixmatvec, FNALU_HYPRE_STRUCTMATRIXMATVEC)
extern void hypre_F90_NAME(fhypre_structmatrixmatvec, FNALU_HYPRE_STRUCTMATRIXMATVEC)
(NALU_HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);



#define NALU_HYPRE_StructVectorCreate \
        hypre_F90_NAME(fhypre_structvectorcreate, FNALU_HYPRE_STRUCTVECTORCREATE)
extern void hypre_F90_NAME(fhypre_structvectorcreate, FNALU_HYPRE_STRUCTVECTORCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorDestroy \
        hypre_F90_NAME(fhypre_structvectordestroy, FNALU_HYPRE_STRUCTVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_structvectordestroy, FNALU_HYPRE_STRUCTVECTORDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorInitialize \
        hypre_F90_NAME(fhypre_structvectorinitialize, FNALU_HYPRE_STRUCTVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_structvectorinitialize, FNALU_HYPRE_STRUCTVECTORINITIALIZE)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorSetValues \
        hypre_F90_NAME(fhypre_structvectorsetvalues, FNALU_HYPRE_STRUCTVECTORSETVALUES)
extern void hypre_F90_NAME(fhypre_structvectorsetvalues, FNALU_HYPRE_STRUCTVECTORSETVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructVectorSetBoxValues \
        hypre_F90_NAME(fhypre_structvectorsetboxvalues, FNALU_HYPRE_STRUCTVECTORSETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structvectorsetboxvalues, FNALU_HYPRE_STRUCTVECTORSETBOXVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorSetConstantValues \
        hypre_F90_NAME(fhypre_structvectorsetconstantv, FNALU_HYPRE_STRUCTVECTORSETCONTANTV)
extern void hypre_F90_NAME(fhypre_structvectorsetconstantv, FNALU_HYPRE_STRUCTVECTORSETCONTANTV)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorAddToValues \
        hypre_F90_NAME(fhypre_structvectoraddtovalues, FNALU_HYPRE_STRUCTVECTORADDTOVALUES)
extern void hypre_F90_NAME(fhypre_structvectoraddtovalues, FNALU_HYPRE_STRUCTVECTORADDTOVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorAddToBoxValues \
        hypre_F90_NAME(fhypre_structvectoraddtoboxvalu, FNALU_HYPRE_STRUCTVECTORADDTOBOXVALU)
extern void hypre_F90_NAME(fhypre_structvectoraddtoboxvalu, FNALU_HYPRE_STRUCTVECTORADDTOBOXVALU)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorScaleValues \
        hypre_F90_NAME(fhypre_structvectorscalevalues, FNALU_HYPRE_STRUCTVECTORSCALEVALUES)
extern void hypre_F90_NAME(fhypre_structvectorscalevalues, FNALU_HYPRE_STRUCTVECTORSCALEVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorGetValues \
        hypre_F90_NAME(fhypre_structvectorgetvalues, FNALU_HYPRE_STRUCTVECTORGETVALUES)
extern void hypre_F90_NAME(fhypre_structvectorgetvalues, FNALU_HYPRE_STRUCTVECTORGETVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorGetBoxValues \
        hypre_F90_NAME(fhypre_structvectorgetboxvalues, FNALU_HYPRE_STRUCTVECTORGETBOXVALUES)
extern void hypre_F90_NAME(fhypre_structvectorgetboxvalues, FNALU_HYPRE_STRUCTVECTORGETBOXVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorAssemble \
        hypre_F90_NAME(fhypre_structvectorassemble, FNALU_HYPRE_STRUCTVECTORASSEMBLE)
extern void hypre_F90_NAME(fhypre_structvectorassemble, FNALU_HYPRE_STRUCTVECTORASSEMBLE)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorSetNumGhost \
        hypre_F90_NAME(fhypre_structvectorsetnumghost, FNALU_HYPRE_STRUCTVECTORSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_structvectorsetnumghost, FNALU_HYPRE_STRUCTVECTORSETNUMGHOST)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructVectorCopy \
        hypre_F90_NAME(fhypre_structvectorcopy, FNALU_HYPRE_STRUCTVECTORCOPY)
extern void hypre_F90_NAME(fhypre_structvectorcopy, FNALU_HYPRE_STRUCTVECTORCOPY)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorGetMigrateCommPkg \
        hypre_F90_NAME(fhypre_structvectorgetmigrateco, FNALU_HYPRE_STRUCTVECTORGETMIGRATECO)
extern void hypre_F90_NAME(fhypre_structvectorgetmigrateco, FNALU_HYPRE_STRUCTVECTORGETMIGRATECO)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorMigrate \
        hypre_F90_NAME(fhypre_structvectormigrate, FNALU_HYPRE_STRUCTVECTORMIGRATE)
extern void hypre_F90_NAME(fhypre_structvectormigrate, FNALU_HYPRE_STRUCTVECTORMIGRATE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_CommPkgDestroy \
        hypre_F90_NAME(fhypre_commpkgdestroy, FNALU_HYPRE_COMMPKGDESTROY)
extern void hypre_F90_NAME(fhypre_commpkgdestroy, FNALU_HYPRE_COMMPKGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorPrint \
        hypre_F90_NAME(fhypre_structvectorprint, FNALU_HYPRE_STRUCTVECTORPRINT)
extern void hypre_F90_NAME(fhypre_structvectorprint, FNALU_HYPRE_STRUCTVECTORPRINT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);


#define NALU_HYPRE_StructBiCGSTABCreate \
        hypre_F90_NAME(fhypre_structbicgstabcreate, FNALU_HYPRE_STRUCTBICGSTABCREATE)
extern void hypre_F90_NAME(fhypre_structbicgstabcreate, FNALU_HYPRE_STRUCTBICGSTABCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABDestroy \
        hypre_F90_NAME(fhypre_structbicgstabdestroy, FNALU_HYPRE_STRUCTBICGSTABDESTROY)
extern void hypre_F90_NAME(fhypre_structbicgstabdestroy, FNALU_HYPRE_STRUCTBICGSTABDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABSetup \
        hypre_F90_NAME(fhypre_structbicgstabsetup, FNALU_HYPRE_STRUCTBICGSTABSETUP)
extern void hypre_F90_NAME(fhypre_structbicgstabsetup, FNALU_HYPRE_STRUCTBICGSTABSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABSolve \
        hypre_F90_NAME(fhypre_structbicgstabsolve, FNALU_HYPRE_STRUCTBICGSTABSOLVE)
extern void hypre_F90_NAME(fhypre_structbicgstabsolve, FNALU_HYPRE_STRUCTBICGSTABSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABSetTol \
        hypre_F90_NAME(fhypre_structbicgstabsettol, FNALU_HYPRE_STRUCTBICGSTABSETTOL)
extern void hypre_F90_NAME(fhypre_structbicgstabsettol, FNALU_HYPRE_STRUCTBICGSTABSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructBiCGSTABSetMaxIter \
        hypre_F90_NAME(fhypre_structbicgstabsetmaxiter, FNALU_HYPRE_STRUCTBICGSTABSETMAXITER)
extern void hypre_F90_NAME(fhypre_structbicgstabsetmaxiter, FNALU_HYPRE_STRUCTBICGSTABSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructBiCGSTABSetPrecond \
        hypre_F90_NAME(fhypre_structbicgstabsetprecond, FNALU_HYPRE_STRUCTBICGSTABSETPRECOND)
extern void hypre_F90_NAME(fhypre_structbicgstabsetprecond, FNALU_HYPRE_STRUCTBICGSTABSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABSetLogging \
        hypre_F90_NAME(fhypre_structbicgstabsetlogging, FNALU_HYPRE_STRUCTBICGSTABSETLOGGING)
extern void hypre_F90_NAME(fhypre_structbicgstabsetlogging, FNALU_HYPRE_STRUCTBICGSTABSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructBiCGSTABSetPrintLevel \
        hypre_F90_NAME(fhypre_structbicgstabsetprintle, FNALU_HYPRE_STRUCTBICGSTABPRINTLE)
extern void hypre_F90_NAME(fhypre_structbicgstabsetprintle, FNALU_HYPRE_STRUCTBICGSTABPRINTLE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructBiCGSTABGetNumIterations \
        hypre_F90_NAME(fhypre_structbicgstabgetnumiter, FNALU_HYPRE_STRUCTBICGSTABGETNUMITER)
extern void hypre_F90_NAME(fhypre_structbicgstabgetnumiter, FNALU_HYPRE_STRUCTBICGSTABGETNUMITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructBiCGSTABGetResidual \
        hypre_F90_NAME(fhypre_structbicgstabgetresidua, FNALU_HYPRE_STRUCTBICGSTABGETRESIDUA)
extern void hypre_F90_NAME(fhypre_structbicgstabgetresidua, FNALU_HYPRE_STRUCTBICGSTABGETRESIDUA)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structbicgstabgetfinalre, FNALU_HYPRE_STRUCTBICGSTABGETFINALRE)
extern void hypre_F90_NAME(fhypre_structbicgstabgetfinalre, FNALU_HYPRE_STRUCTBICGSTABGETFINALRE)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructGMRESCreate \
        hypre_F90_NAME(fhypre_structgmrescreate, FNALU_HYPRE_STRUCTGMRESCREATE)
extern void hypre_F90_NAME(fhypre_structgmrescreate, FNALU_HYPRE_STRUCTGMRESCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESDestroy \
        hypre_F90_NAME(fhypre_structgmresdestroy, FNALU_HYPRE_STRUCTGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_structgmresdestroy, FNALU_HYPRE_STRUCTGMRESDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESSetup \
        hypre_F90_NAME(fhypre_structgmressetup, FNALU_HYPRE_STRUCTGMRESSETUP)
extern void hypre_F90_NAME(fhypre_structgmressetup, FNALU_HYPRE_STRUCTGMRESSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESSolve \
        hypre_F90_NAME(fhypre_structgmressolve, FNALU_HYPRE_STRUCTGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_structgmressolve, FNALU_HYPRE_STRUCTGMRESSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESSetTol \
        hypre_F90_NAME(fhypre_structgmressettol, FNALU_HYPRE_STRUCTGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_structgmressettol, FNALU_HYPRE_STRUCTGMRESSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructGMRESSetMaxIter \
        hypre_F90_NAME(fhypre_structgmressetmaxiter, FNALU_HYPRE_STRUCTGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_structgmressetmaxiter, FNALU_HYPRE_STRUCTGMRESSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGMRESSetPrecond \
        hypre_F90_NAME(fhypre_structgmressetprecond, FNALU_HYPRE_STRUCTGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_structgmressetprecond, FNALU_HYPRE_STRUCTGMRESSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESSetLogging \
        hypre_F90_NAME(fhypre_structgmressetlogging, FNALU_HYPRE_STRUCTGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_structgmressetlogging, FNALU_HYPRE_STRUCTGMRESSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGMRESSetPrintLevel \
        hypre_F90_NAME(fhypre_structgmressetprintlevel, FNALU_HYPRE_STRUCTGMRESPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structgmressetprintlevel, FNALU_HYPRE_STRUCTGMRESPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGMRESGetNumIterations \
        hypre_F90_NAME(fhypre_structgmresgetnumiterati, FNALU_HYPRE_STRUCTGMRESGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_structgmresgetnumiterati, FNALU_HYPRE_STRUCTGMRESGETNUMITERATI)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGMRESGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structgmresgetfinalrelat, FNALU_HYPRE_STRUCTGMRESGETFINALRELAT)
extern void hypre_F90_NAME(fhypre_structgmresgetfinalrelat, FNALU_HYPRE_STRUCTGMRESGETFINALRELAT)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructHybridCreate \
        hypre_F90_NAME(fhypre_structhybridcreate, FNALU_HYPRE_STRUCTHYBRIDCREATE)
extern void hypre_F90_NAME(fhypre_structhybridcreate, FNALU_HYPRE_STRUCTHYBRIDCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridDestroy \
        hypre_F90_NAME(fhypre_structhybriddestroy, FNALU_HYPRE_STRUCTHYBRIDDESTROY)
extern void hypre_F90_NAME(fhypre_structhybriddestroy, FNALU_HYPRE_STRUCTHYBRIDDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridSetup \
        hypre_F90_NAME(fhypre_structhybridsetup, FNALU_HYPRE_STRUCTHYBRIDSETUP)
extern void hypre_F90_NAME(fhypre_structhybridsetup, FNALU_HYPRE_STRUCTHYBRIDSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridSolve \
        hypre_F90_NAME(fhypre_structhybridsolve, FNALU_HYPRE_STRUCTHYBRIDSOLVE)
extern void hypre_F90_NAME(fhypre_structhybridsolve, FNALU_HYPRE_STRUCTHYBRIDSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridSetSolverType \
        hypre_F90_NAME(fhypre_structhybridsetsolvertyp, FNALU_HYPRE_STRUCTHYBRIDSETSOLVERTYP)
extern void hypre_F90_NAME(fhypre_structhybridsetsolvertyp, FNALU_HYPRE_STRUCTHYBRIDSETSOLVERTYP)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetStopCrit \
        hypre_F90_NAME(fhypre_structhybridsetstopcrit, FNALU_HYPRE_STRUCTHYBRIDSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_structhybridsetstopcrit, FNALU_HYPRE_STRUCTHYBRIDSETSTOPCRIT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetKDim \
        hypre_F90_NAME(fhypre_structhybridsetkdim, FNALU_HYPRE_STRUCTHYBRIDSETKDIM)
extern void hypre_F90_NAME(fhypre_structhybridsetkdim, FNALU_HYPRE_STRUCTHYBRIDSETKDIM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetTol \
        hypre_F90_NAME(fhypre_structhybridsettol, FNALU_HYPRE_STRUCTHYBRIDSETTOL)
extern void hypre_F90_NAME(fhypre_structhybridsettol, FNALU_HYPRE_STRUCTHYBRIDSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructHybridSetConvergenceTol \
        hypre_F90_NAME(fhypre_structhybridsetconvergen, FNALU_HYPRE_STRUCTHYBRIDSETCONVERGEN)
extern void hypre_F90_NAME(fhypre_structhybridsetconvergen, FNALU_HYPRE_STRUCTHYBRIDSETCONVERGEN)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructHybridSetPCGAbsoluteTolFactor \
        hypre_F90_NAME(fhypre_structhybridsetpcgabsolu, FNALU_HYPRE_STRUCTHYBRIDSETABSOLU)
extern void hypre_F90_NAME(fhypre_structhybridsetpcgabsolu, FNALU_HYPRE_STRUCTHYBRIDSETABSOLU)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructHybridSetMaxIter \
        hypre_F90_NAME(fhypre_structhybridsetmaxiter, FNALU_HYPRE_STRUCTHYBRIDSETMAXITER)
extern void hypre_F90_NAME(fhypre_structhybridsetmaxiter, FNALU_HYPRE_STRUCTHYBRIDSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetDSCGMaxIter \
        hypre_F90_NAME(fhypre_structhybridsetdscgmaxit, FNALU_HYPRE_STRUCTHYBRIDSETDSCGMAXIT)
extern void hypre_F90_NAME(fhypre_structhybridsetdscgmaxit, FNALU_HYPRE_STRUCTHYBRIDSETDSCGMAXIT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetPCGMaxIter \
        hypre_F90_NAME(fhypre_structhybridsetpcgmaxite, FNALU_HYPRE_STRUCTHYBRIDSETPCGMAXITE)
extern void hypre_F90_NAME(fhypre_structhybridsetpcgmaxite, FNALU_HYPRE_STRUCTHYBRIDSETPCGMAXITE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetTwoNorm \
        hypre_F90_NAME(fhypre_structhybridsettwonorm, FNALU_HYPRE_STRUCTHYBRIDSETTWONORM)
extern void hypre_F90_NAME(fhypre_structhybridsettwonorm, FNALU_HYPRE_STRUCTHYBRIDSETTWONORM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetRelChange \
        hypre_F90_NAME(fhypre_structhybridsetrelchange, FNALU_HYPRE_STRUCTHYBRIDSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structhybridsetrelchange, FNALU_HYPRE_STRUCTHYBRIDSETRELCHANGE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetPrecond \
        hypre_F90_NAME(fhypre_structhybridsetprecond, FNALU_HYPRE_STRUCTHYBRIDSETPRECOND)
extern void hypre_F90_NAME(fhypre_structhybridsetprecond, FNALU_HYPRE_STRUCTHYBRIDSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridSetLogging \
        hypre_F90_NAME(fhypre_structhybridsetlogging, FNALU_HYPRE_STRUCTHYBRIDSETLOGGING)
extern void hypre_F90_NAME(fhypre_structhybridsetlogging, FNALU_HYPRE_STRUCTHYBRIDSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetPrintLevel \
        hypre_F90_NAME(fhypre_structhybridsetprintleve, FNALU_HYPRE_STRUCTHYBRIDSETPRINTLEVE)
extern void hypre_F90_NAME(fhypre_structhybridsetprintleve, FNALU_HYPRE_STRUCTHYBRIDSETPRINTLEVE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridGetNumIterations \
        hypre_F90_NAME(fhypre_structhybridgetnumiterat, FNALU_HYPRE_STRUCTHYBRIDGETNUMITERAT)
extern void hypre_F90_NAME(fhypre_structhybridgetnumiterat, FNALU_HYPRE_STRUCTHYBRIDGETNUMITERAT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridGetDSCGNumIterations \
        hypre_F90_NAME(fhypre_structhybridgetdscgnumit, FNALU_HYPRE_STRUCTHYBRIDGETDSCGNUMIT)
extern void hypre_F90_NAME(fhypre_structhybridgetdscgnumit, FNALU_HYPRE_STRUCTHYBRIDGETDSCGNUMIT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridGetPCGNumIterations \
        hypre_F90_NAME(fhypre_structhybridgetpcgnumite, FNALU_HYPRE_STRUCTHYBRIDGETPCGNUMITE)
extern void hypre_F90_NAME(fhypre_structhybridgetpcgnumite, FNALU_HYPRE_STRUCTHYBRIDGETPCGNUMITE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structhybridgetfinalrela, FNALU_HYPRE_STRUCTHYBRIDGETFINALRELA)
extern void hypre_F90_NAME(fhypre_structhybridgetfinalrela, FNALU_HYPRE_STRUCTHYBRIDGETFINALRELA)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructVectorSetRandomValues \
        hypre_F90_NAME(fhypre_structvectorsetrandomvalu, FNALU_HYPRE_STRUCTVECTORSETRANDOMVALU)
extern void hypre_F90_NAME(fhypre_structvectorsetrandomvalu, FNALU_HYPRE_STRUCTVECTORSETRANDOMVALU)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSetRandomValues \
        hypre_F90_NAME(fhypre_structsetrandomvalues, FNALU_HYPRE_STRUCTSETRANDOMVALUES)
extern void hypre_F90_NAME(fhypre_structsetrandomvalues, FNALU_HYPRE_STRUCTSETRANDOMVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSetupInterpreter \
        hypre_F90_NAME(fhypre_structsetupinterpreter, FNALU_HYPRE_STRUCTSETUPINTERPRETER)
extern void hypre_F90_NAME(fhypre_structsetupinterpreter, FNALU_HYPRE_STRUCTSETUPINTERPRETER)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructSetupMatvec \
        hypre_F90_NAME(fhypre_structsetupmatvec, FNALU_HYPRE_STRUCTSETUPMATVEC)
extern void hypre_F90_NAME(fhypre_structsetupmatvec, FNALU_HYPRE_STRUCTSETUPMATVEC)
(hypre_F90_Obj *);



#define NALU_HYPRE_StructJacobiCreate \
        hypre_F90_NAME(fhypre_structjacobicreate, FNALU_HYPRE_STRUCTJACOBICREATE)
extern void hypre_F90_NAME(fhypre_structjacobicreate, FNALU_HYPRE_STRUCTJACOBICREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiDestroy \
        hypre_F90_NAME(fhypre_structjacobidestroy, FNALU_HYPRE_STRUCTJACOBIDESTROY)
extern void hypre_F90_NAME(fhypre_structjacobidestroy, FNALU_HYPRE_STRUCTJACOBIDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiSetup \
        hypre_F90_NAME(fhypre_structjacobisetup, FNALU_HYPRE_STRUCTJACOBISETUP)
extern void hypre_F90_NAME(fhypre_structjacobisetup, FNALU_HYPRE_STRUCTJACOBISETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiSolve \
        hypre_F90_NAME(fhypre_structjacobisolve, FNALU_HYPRE_STRUCTJACOBISOLVE)
extern void hypre_F90_NAME(fhypre_structjacobisolve, FNALU_HYPRE_STRUCTJACOBISOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiSetTol \
        hypre_F90_NAME(fhypre_structjacobisettol, FNALU_HYPRE_STRUCTJACOBISETTOL)
extern void hypre_F90_NAME(fhypre_structjacobisettol, FNALU_HYPRE_STRUCTJACOBISETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructJacobiGetTol \
        hypre_F90_NAME(fhypre_structjacobigettol, FNALU_HYPRE_STRUCTJACOBIGETTOL)
extern void hypre_F90_NAME(fhypre_structjacobigettol, FNALU_HYPRE_STRUCTJACOBIGETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructJacobiSetMaxIter \
        hypre_F90_NAME(fhypre_structjacobisetmaxiter, FNALU_HYPRE_STRUCTJACOBISETTOL)
extern void hypre_F90_NAME(fhypre_structjacobisetmaxiter, FNALU_HYPRE_STRUCTJACOBISETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructJacobiGetMaxIter \
        hypre_F90_NAME(fhypre_structjacobigetmaxiter, FNALU_HYPRE_STRUCTJACOBIGETTOL)
extern void hypre_F90_NAME(fhypre_structjacobigetmaxiter, FNALU_HYPRE_STRUCTJACOBIGETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructJacobiSetZeroGuess \
        hypre_F90_NAME(fhypre_structjacobisetzeroguess, FNALU_HYPRE_STRUCTJACOBISETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structjacobisetzeroguess, FNALU_HYPRE_STRUCTJACOBISETZEROGUESS)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiGetZeroGuess \
        hypre_F90_NAME(fhypre_structjacobigetzeroguess, FNALU_HYPRE_STRUCTJACOBIGETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structjacobigetzeroguess, FNALU_HYPRE_STRUCTJACOBIGETZEROGUESS)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structjacobisetnonzerogu, FNALU_HYPRE_STRUCTJACOBISETNONZEROGU)
extern void hypre_F90_NAME(fhypre_structjacobisetnonzerogu, FNALU_HYPRE_STRUCTJACOBISETNONZEROGU)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiGetNumIterations \
        hypre_F90_NAME(fhypre_structjacobigetnumiterati, FNALU_HYPRE_STRUCTJACOBIGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_structjacobigetnumiterati, FNALU_HYPRE_STRUCTJACOBIGETNUMITERATI)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructJacobiGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structjacobigetfinalrela, FNALU_HYPRE_STRUCTJACOBIGETFINALRELA)
extern void hypre_F90_NAME(fhypre_structjacobigetfinalrela, FNALU_HYPRE_STRUCTJACOBIGETFINALRELA)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructPCGCreate \
        hypre_F90_NAME(fhypre_structpcgcreate, FNALU_HYPRE_STRUCTPCGCREATE)
extern void hypre_F90_NAME(fhypre_structpcgcreate, FNALU_HYPRE_STRUCTPCGCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGDestroy \
        hypre_F90_NAME(fhypre_structpcgdestroy, FNALU_HYPRE_STRUCTPCGDESTROY)
extern void hypre_F90_NAME(fhypre_structpcgdestroy, FNALU_HYPRE_STRUCTPCGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGSetup \
        hypre_F90_NAME(fhypre_structpcgsetup, FNALU_HYPRE_STRUCTPCGSETUP)
extern void hypre_F90_NAME(fhypre_structpcgsetup, FNALU_HYPRE_STRUCTPCGSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGSolve \
        hypre_F90_NAME(fhypre_structpcgsolve, FNALU_HYPRE_STRUCTPCGSOLVE)
extern void hypre_F90_NAME(fhypre_structpcgsolve, FNALU_HYPRE_STRUCTPCGSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGSetTol \
        hypre_F90_NAME(fhypre_structpcgsettol, FNALU_HYPRE_STRUCTPCGSETTOL)
extern void hypre_F90_NAME(fhypre_structpcgsettol, FNALU_HYPRE_STRUCTPCGSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructPCGSetMaxIter \
        hypre_F90_NAME(fhypre_structpcgsetmaxiter, FNALU_HYPRE_STRUCTPCGSETMAXITER)
extern void hypre_F90_NAME(fhypre_structpcgsetmaxiter, FNALU_HYPRE_STRUCTPCGSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGSetTwoNorm \
        hypre_F90_NAME(fhypre_structpcgsettwonorm, FHYRPE_STRUCTPCGSETTWONORM)
extern void hypre_F90_NAME(fhypre_structpcgsettwonorm, FHYRPE_STRUCTPCGSETTWONORM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGSetRelChange \
        hypre_F90_NAME(fhypre_structpcgsetrelchange, FNALU_HYPRE_STRUCTPCGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structpcgsetrelchange, FNALU_HYPRE_STRUCTPCGSETRELCHANGE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGSetPrecond \
        hypre_F90_NAME(fhypre_structpcgsetprecond, FNALU_HYPRE_STRUCTPCGSETPRECOND)
extern void hypre_F90_NAME(fhypre_structpcgsetprecond, FNALU_HYPRE_STRUCTPCGSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGSetLogging \
        hypre_F90_NAME(fhypre_structpcgsetlogging, FHYPRES_TRUCTPCFSETLOGGING)
extern void hypre_F90_NAME(fhypre_structpcgsetlogging, FHYPRES_TRUCTPCFSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGSetPrintLevel \
        hypre_F90_NAME(fhypre_structpcgsetprintlevel, FNALU_HYPRE_STRUCTPCGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structpcgsetprintlevel, FNALU_HYPRE_STRUCTPCGSETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGGetNumIterations \
        hypre_F90_NAME(fhypre_structpcggetnumiteration, FNALU_HYPRE_STRUCTPCGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_structpcggetnumiteration, FNALU_HYPRE_STRUCTPCGGETNUMITERATION)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structpcggetfinalrelativ, FNALU_HYPRE_STRUCTPCGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_structpcggetfinalrelativ, FNALU_HYPRE_STRUCTPCGGETFINALRELATIV)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructDiagScaleSetup \
        hypre_F90_NAME(fhypre_structdiagscalesetup, FNALU_HYPRE_STRUCTDIAGSCALESETUP)
extern void hypre_F90_NAME(fhypre_structdiagscalesetup, FNALU_HYPRE_STRUCTDIAGSCALESETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructDiagScaleSolve \
        hypre_F90_NAME(fhypre_structdiagscalesolve, FNALU_HYPRE_STRUCTDIAGSCALESOLVE)
extern void hypre_F90_NAME(fhypre_structdiagscalesolve, FNALU_HYPRE_STRUCTDIAGSCALESOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);



#define NALU_HYPRE_StructPFMGCreate \
        hypre_F90_NAME(fhypre_structpfmgcreate, FNALU_HYPRE_STRUCTPFMGCREATE)
extern void hypre_F90_NAME(fhypre_structpfmgcreate, FNALU_HYPRE_STRUCTPFMGCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGDestroy \
        hypre_F90_NAME(fhypre_structpfmgdestroy, FNALU_HYPRE_STRUCTPFMGDESTROY)
extern void hypre_F90_NAME(fhypre_structpfmgdestroy, FNALU_HYPRE_STRUCTPFMGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSetup \
        hypre_F90_NAME(fhypre_structpfmgsetup, FNALU_HYPRE_STRUCTPFMGSETUP)
extern void hypre_F90_NAME(fhypre_structpfmgsetup, FNALU_HYPRE_STRUCTPFMGSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSolve \
        hypre_F90_NAME(fhypre_structpfmgsolve, FNALU_HYPRE_STRUCTPFMGSOLVE)
extern void hypre_F90_NAME(fhypre_structpfmgsolve, FNALU_HYPRE_STRUCTPFMGSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSetTol \
        hypre_F90_NAME(fhypre_structpfmgsettol, FNALU_HYPRE_STRUCTPFMGSETTOL)
extern void hypre_F90_NAME(fhypre_structpfmgsettol, FNALU_HYPRE_STRUCTPFMGSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructPFMGGetTol \
        hypre_F90_NAME(fhypre_structpfmggettol, FNALU_HYPRE_STRUCTPFMGGETTOL)
extern void hypre_F90_NAME(fhypre_structpfmggettol, FNALU_HYPRE_STRUCTPFMGGETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructPFMGSetMaxIter \
        hypre_F90_NAME(fhypre_structpfmgsetmaxiter, FNALU_HYPRE_STRUCTPFMGSETMAXITER)
extern void hypre_F90_NAME(fhypre_structpfmgsetmaxiter, FNALU_HYPRE_STRUCTPFMGSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetMaxIter \
        hypre_F90_NAME(fhypre_structpfmggetmaxiter, FNALU_HYPRE_STRUCTPFMGGETMAXITER)
extern void hypre_F90_NAME(fhypre_structpfmggetmaxiter, FNALU_HYPRE_STRUCTPFMGGETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetMaxLevels \
        hypre_F90_NAME(fhypre_structpfmgsetmaxlevels, FNALU_HYPRE_STRUCTPFMGSETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_structpfmgsetmaxlevels, FNALU_HYPRE_STRUCTPFMGSETMAXLEVELS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetMaxLevels \
        hypre_F90_NAME(fhypre_structpfmggetmaxlevels, FNALU_HYPRE_STRUCTPFMGGETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_structpfmggetmaxlevels, FNALU_HYPRE_STRUCTPFMGGETMAXLEVELS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetRelChange \
        hypre_F90_NAME(fhypre_structpfmgsetrelchange, FNALU_HYPRE_STRUCTPFMGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structpfmgsetrelchange, FNALU_HYPRE_STRUCTPFMGSETRELCHANGE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetRelChange \
        hypre_F90_NAME(fhypre_structpfmggetrelchange, FNALU_HYPRE_STRUCTPFMGGETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structpfmggetrelchange, FNALU_HYPRE_STRUCTPFMGGETRELCHANGE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetZeroGuess \
        hypre_F90_NAME(fhypre_structpfmgsetzeroguess, FNALU_HYPRE_STRUCTPFMGSETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structpfmgsetzeroguess, FNALU_HYPRE_STRUCTPFMGSETZEROGUESS)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGGetZeroGuess \
        hypre_F90_NAME(fhypre_structpfmggetzeroguess, FNALU_HYPRE_STRUCTPFMGGETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structpfmggetzeroguess, FNALU_HYPRE_STRUCTPFMGGETZEROGUESS)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structpfmgsetnonzerogues, FHYPRES_TRUCTPFMGSETNONZEROGUES)
extern void hypre_F90_NAME(fhypre_structpfmgsetnonzerogues, FHYPRES_TRUCTPFMGSETNONZEROGUES)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSetSkipRelax \
        hypre_F90_NAME(fhypre_structpfmgsetskiprelax, FNALU_HYPRE_STRUCTPFMGSETSKIPRELAX)
extern void hypre_F90_NAME(fhypre_structpfmgsetskiprelax, FNALU_HYPRE_STRUCTPFMGSETSKIPRELAX)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetSkipRelax \
        hypre_F90_NAME(fhypre_structpfmggetskiprelax, FNALU_HYPRE_STRUCTPFMGGETSKIPRELAX)
extern void hypre_F90_NAME(fhypre_structpfmggetskiprelax, FNALU_HYPRE_STRUCTPFMGGETSKIPRELAX)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetRelaxType \
        hypre_F90_NAME(fhypre_structpfmgsetrelaxtype, FNALU_HYPRE_STRUCTPFMGSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_structpfmgsetrelaxtype, FNALU_HYPRE_STRUCTPFMGSETRELAXTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetRelaxType \
        hypre_F90_NAME(fhypre_structpfmggetrelaxtype, FNALU_HYPRE_STRUCTPFMGGETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_structpfmggetrelaxtype, FNALU_HYPRE_STRUCTPFMGGETRELAXTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetRAPType \
        hypre_F90_NAME(fhypre_structpfmgsetraptype, FNALU_HYPRE_STRUCTPFMGSETRAPTYPE)
extern void hypre_F90_NAME(fhypre_structpfmgsetraptype, FNALU_HYPRE_STRUCTPFMGSETRAPTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetRAPType \
        hypre_F90_NAME(fhypre_structpfmggetraptype, FNALU_HYPRE_STRUCTPFMGGETRAPTYPE)
extern void hypre_F90_NAME(fhypre_structpfmggetraptype, FNALU_HYPRE_STRUCTPFMGGETRAPTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetNumPreRelax \
        hypre_F90_NAME(fhypre_structpfmgsetnumprerelax, FNALU_HYPRE_STRUCTPFMGSETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structpfmgsetnumprerelax, FNALU_HYPRE_STRUCTPFMGSETNUMPRERELAX)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetNumPreRelax \
        hypre_F90_NAME(fhypre_structpfmggetnumprerelax, FNALU_HYPRE_STRUCTPFMGGETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structpfmggetnumprerelax, FNALU_HYPRE_STRUCTPFMGGETNUMPRERELAX)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetNumPostRelax \
        hypre_F90_NAME(fhypre_structpfmgsetnumpostrela, FNALU_HYPRE_STRUCTPFMGSETNUMPOSTRELA)
extern void hypre_F90_NAME(fhypre_structpfmgsetnumpostrela, FNALU_HYPRE_STRUCTPFMGSETNUMPOSTRELA)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetNumPostRelax \
        hypre_F90_NAME(fhypre_structpfmggetnumpostrela, FNALU_HYPRE_STRUCTPFMGGETNUMPOSTRELA)
extern void hypre_F90_NAME(fhypre_structpfmggetnumpostrela, FNALU_HYPRE_STRUCTPFMGGETNUMPOSTRELA)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetDxyz \
        hypre_F90_NAME(fhypre_structpfmgsetdxyz, FNALU_HYPRE_STRUCTPFMGSETDXYZ)
extern void hypre_F90_NAME(fhypre_structpfmgsetdxyz, FNALU_HYPRE_STRUCTPFMGSETDXYZ)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructPFMGSetLogging \
        hypre_F90_NAME(fhypre_structpfmgsetlogging, FNALU_HYPRE_STRUCTPFMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_structpfmgsetlogging, FNALU_HYPRE_STRUCTPFMGSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetLogging \
        hypre_F90_NAME(fhypre_structpfmggetlogging, FNALU_HYPRE_STRUCTPFMGGETLOGGING)
extern void hypre_F90_NAME(fhypre_structpfmggetlogging, FNALU_HYPRE_STRUCTPFMGGETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetPrintLevel \
        hypre_F90_NAME(fhypre_structpfmgsetprintlevel, FNALU_HYPRE_STRUCTPFMGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structpfmgsetprintlevel, FNALU_HYPRE_STRUCTPFMGSETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetPrintLevel \
        hypre_F90_NAME(fhypre_structpfmggetprintlevel, FNALU_HYPRE_STRUCTPFMGGETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structpfmggetprintlevel, FNALU_HYPRE_STRUCTPFMGGETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetNumIterations \
        hypre_F90_NAME(fhypre_structpfmggetnumiteratio, FNALU_HYPRE_STRUCTPFMGGETNUMITERATIO)
extern void hypre_F90_NAME(fhypre_structpfmggetnumiteratio, FNALU_HYPRE_STRUCTPFMGGETNUMITERATIO)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structpfmggetfinalrelati, FNALU_HYPRE_STRUCTPFMGGETFINALRELATI)
extern void hypre_F90_NAME(fhypre_structpfmggetfinalrelati, FNALU_HYPRE_STRUCTPFMGGETFINALRELATI)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructSMGCreate \
        hypre_F90_NAME(fhypre_structsmgcreate, FNALU_HYPRE_STRUCTSMGCREATE)
extern void hypre_F90_NAME(fhypre_structsmgcreate, FNALU_HYPRE_STRUCTSMGCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGDestroy \
        hypre_F90_NAME(fhypre_structsmgdestroy, FNALU_HYPRE_STRUCTSMGDESTROY)
extern void hypre_F90_NAME(fhypre_structsmgdestroy, FNALU_HYPRE_STRUCTSMGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGSetup \
        hypre_F90_NAME(fhypre_structsmgsetup, FNALU_HYPRE_STRUCTSMGSETUP)
extern void hypre_F90_NAME(fhypre_structsmgsetup, FNALU_HYPRE_STRUCTSMGSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGSolve \
        hypre_F90_NAME(fhypre_structsmgsolve, FNALU_HYPRE_STRUCTSMGSOLVE)
extern void hypre_F90_NAME(fhypre_structsmgsolve, FNALU_HYPRE_STRUCTSMGSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGSetMemoryUse \
        hypre_F90_NAME(fhypre_structsmgsetmemoryuse, FNALU_HYPRE_STRUCTSMGSETMEMORYUSE)
extern void hypre_F90_NAME(fhypre_structsmgsetmemoryuse, FNALU_HYPRE_STRUCTSMGSETMEMORYUSE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetMemoryUse \
        hypre_F90_NAME(fhypre_structsmggetmemoryuse, FNALU_HYPRE_STRUCTSMGGETMEMORYUSE)
extern void hypre_F90_NAME(fhypre_structsmggetmemoryuse, FNALU_HYPRE_STRUCTSMGGETMEMORYUSE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetTol \
        hypre_F90_NAME(fhypre_structsmgsettol, FNALU_HYPRE_STRUCTSMGSETTOL)
extern void hypre_F90_NAME(fhypre_structsmgsettol, FNALU_HYPRE_STRUCTSMGSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSMGGetTol \
        hypre_F90_NAME(fhypre_structsmggettol, FNALU_HYPRE_STRUCTSMGGETTOL)
extern void hypre_F90_NAME(fhypre_structsmggettol, FNALU_HYPRE_STRUCTSMGGETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSMGSetMaxIter \
        hypre_F90_NAME(fhypre_structsmgsetmaxiter, FNALU_HYPRE_STRUCTSMGSETMAXTITER)
extern void hypre_F90_NAME(fhypre_structsmgsetmaxiter, FNALU_HYPRE_STRUCTSMGSETMAXTITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetMaxIter \
        hypre_F90_NAME(fhypre_structsmggetmaxiter, FNALU_HYPRE_STRUCTSMGGETMAXTITER)
extern void hypre_F90_NAME(fhypre_structsmggetmaxiter, FNALU_HYPRE_STRUCTSMGGETMAXTITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetRelChange \
        hypre_F90_NAME(fhypre_structsmgsetrelchange, FNALU_HYPRE_STRUCTSMGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structsmgsetrelchange, FNALU_HYPRE_STRUCTSMGSETRELCHANGE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetRelChange \
        hypre_F90_NAME(fhypre_structsmggetrelchange, FNALU_HYPRE_STRUCTSMGGETRELCHANGE)
extern void hypre_F90_NAME(fhypre_structsmggetrelchange, FNALU_HYPRE_STRUCTSMGGETRELCHANGE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetZeroGuess \
        hypre_F90_NAME(fhypre_structsmgsetzeroguess, FNALU_HYPRE_STRUCTSMGSETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structsmgsetzeroguess, FNALU_HYPRE_STRUCTSMGSETZEROGUESS)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGGetZeroGuess \
        hypre_F90_NAME(fhypre_structsmggetzeroguess, FNALU_HYPRE_STRUCTSMGGETZEROGUESS)
extern void hypre_F90_NAME(fhypre_structsmggetzeroguess, FNALU_HYPRE_STRUCTSMGGETZEROGUESS)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structsmgsetnonzerogues, FNALU_HYPRE_STRUCTSMGSETNONZEROGUES)
extern void hypre_F90_NAME(fhypre_structsmgsetnonzerogues, FNALU_HYPRE_STRUCTSMGSETNONZEROGUES)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGGetNumIterations \
        hypre_F90_NAME(fhypre_structsmggetnumiteration, FNALU_HYPRE_STRUCTSMGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_structsmggetnumiteration, FNALU_HYPRE_STRUCTSMGGETNUMITERATION)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structsmggetfinalrelativ, FNALU_HYPRE_STRUCTSMGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_structsmggetfinalrelativ, FNALU_HYPRE_STRUCTSMGGETFINALRELATIV)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSMGSetNumPreRelax \
        hypre_F90_NAME(fhypre_structsmgsetnumprerelax, FNALU_HYPRE_STRUCTSMGSETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structsmgsetnumprerelax, FNALU_HYPRE_STRUCTSMGSETNUMPRERELAX)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetNumPreRelax \
        hypre_F90_NAME(fhypre_structsmggetnumprerelax, FNALU_HYPRE_STRUCTSMGGETNUMPRERELAX)
extern void hypre_F90_NAME(fhypre_structsmggetnumprerelax, FNALU_HYPRE_STRUCTSMGGETNUMPRERELAX)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetNumPostRelax \
        hypre_F90_NAME(fhypre_structsmgsetnumpostrelax, FNALU_HYPRE_STRUCTSMGSETNUMPOSTRELAX)
extern void hypre_F90_NAME(fhypre_structsmgsetnumpostrelax, FNALU_HYPRE_STRUCTSMGSETNUMPOSTRELAX)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetNumPostRelax \
        hypre_F90_NAME(fhypre_structsmggetnumpostrelax, FNALU_HYPRE_STRUCTSMGGETNUMPOSTRELAX)
extern void hypre_F90_NAME(fhypre_structsmggetnumpostrelax, FNALU_HYPRE_STRUCTSMGGETNUMPOSTRELAX)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetLogging \
        hypre_F90_NAME(fhypre_structsmgsetlogging, FNALU_HYPRE_STRUCTSMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_structsmgsetlogging, FNALU_HYPRE_STRUCTSMGSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetLogging \
        hypre_F90_NAME(fhypre_structsmggetlogging, FNALU_HYPRE_STRUCTSMGGETLOGGING)
extern void hypre_F90_NAME(fhypre_structsmggetlogging, FNALU_HYPRE_STRUCTSMGGETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetPrintLevel \
        hypre_F90_NAME(fhypre_structsmgsetprintlevel, FNALU_HYPRE_STRUCTSMGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structsmgsetprintlevel, FNALU_HYPRE_STRUCTSMGSETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetPrintLevel \
        hypre_F90_NAME(fhypre_structsmggetprintlevel, FNALU_HYPRE_STRUCTSMGGETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_structsmggetprintlevel, FNALU_HYPRE_STRUCTSMGGETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_StructSparseMSGCreate \
        hypre_F90_NAME(fhypre_structsparsemsgcreate, FNALU_HYPRE_STRUCTSPARSEMSGCREATE)
extern void hypre_F90_NAME(fhypre_structsparsemsgcreate, FNALU_HYPRE_STRUCTSPARSEMSGCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGDestroy \
        hypre_F90_NAME(fhypre_structsparsemsgdestroy, FNALU_HYPRE_STRUCTSPARSEMSGDESTROY)
extern void hypre_F90_NAME(fhypre_structsparsemsgdestroy, FNALU_HYPRE_STRUCTSPARSEMSGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGSetup \
        hypre_F90_NAME(fhypre_structsparsemsgsetup, FHYRPE_STRUCTSPARSEMSGSETUP)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetup, FHYRPE_STRUCTSPARSEMSGSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGSolve \
        hypre_F90_NAME(fhypre_structsparsemsgsolve, FNALU_HYPRE_STRUCTSPARSEMSGSOLVE)
extern void hypre_F90_NAME(fhypre_structsparsemsgsolve, FNALU_HYPRE_STRUCTSPARSEMSGSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGSetJump \
        hypre_F90_NAME(fhypre_structsparsemsgsetjump, FNALU_HYPRE_STRUCTSPARSEMSGSETJUMP)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetjump, FNALU_HYPRE_STRUCTSPARSEMSGSETJUMP)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetTol \
        hypre_F90_NAME(fhypre_structsparsemsgsettol, FNALU_HYPRE_STRUCTSPARSEMSGSETTOL)
extern void hypre_F90_NAME(fhypre_structsparsemsgsettol, FNALU_HYPRE_STRUCTSPARSEMSGSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSparseMSGSetMaxIter \
        hypre_F90_NAME(fhypre_structsparsemsgsetmaxite, FNALU_HYPRE_STRUCTSPARSEMSGSETMAXITE)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetmaxite, FNALU_HYPRE_STRUCTSPARSEMSGSETMAXITE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetRelChange \
        hypre_F90_NAME(fhypre_structsparsemsgsetrelcha, FNALU_HYPRE_STRUCTSPARSEMSGSETRELCHA)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetrelcha, FNALU_HYPRE_STRUCTSPARSEMSGSETRELCHA)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetZeroGuess \
        hypre_F90_NAME(fhypre_structsparsemsgsetzerogu, FNALU_HYPRE_STRUCTSPARSEMSGSETZEROGU)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetzerogu, FNALU_HYPRE_STRUCTSPARSEMSGSETZEROGU)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_structsparsemsgsetnonzer, FNALU_HYPRE_STRUCTSPARSEMSGSETNONZER)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnonzer, FNALU_HYPRE_STRUCTSPARSEMSGSETNONZER)
(hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGGetNumIterations \
        hypre_F90_NAME(fhypre_structsparsemsggetnumite, FNALU_HYPRE_STRUCTSPARSEMSGGETNUMITE)
extern void hypre_F90_NAME(fhypre_structsparsemsggetnumite, FNALU_HYPRE_STRUCTSPARSEMSGGETNUMITE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_structsparsemsggetfinalr, FHYRPE_STRUCTSPARSEMSGGETFINALR)
extern void hypre_F90_NAME(fhypre_structsparsemsggetfinalr, FHYRPE_STRUCTSPARSEMSGGETFINALR)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSparseMSGSetRelaxType \
        hypre_F90_NAME(fhypre_structsparsemsgsetrelaxt, FNALU_HYPRE_STRUCTSPARSEMSGSETRELAXT)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetrelaxt, FNALU_HYPRE_STRUCTSPARSEMSGSETRELAXT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetNumPreRelax \
        hypre_F90_NAME(fhypre_structsparsemsgsetnumpre, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMPRE)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnumpre, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMPRE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetNumPostRelax \
        hypre_F90_NAME(fhypre_structsparsemsgsetnumpos, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMPOS)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnumpos, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMPOS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetNumFineRelax \
        hypre_F90_NAME(fhypre_structsparsemsgsetnumfin, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMFIN)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetnumfin, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMFIN)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetLogging \
        hypre_F90_NAME(fhypre_structsparsemsgsetloggin, FNALU_HYPRE_STRUCTSPARSEMSGSETLOGGIN)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetloggin, FNALU_HYPRE_STRUCTSPARSEMSGSETLOGGIN)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetPrintLevel \
        hypre_F90_NAME(fhypre_structsparsemsgsetprintl, FNALU_HYPRE_STRUCTSPARSEMSGSETPRINTL)
extern void hypre_F90_NAME(fhypre_structsparsemsgsetprintl, FNALU_HYPRE_STRUCTSPARSEMSGSETPRINTL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#ifdef __cplusplus
}
#endif
