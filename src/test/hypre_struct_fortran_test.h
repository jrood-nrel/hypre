/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
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
        nalu_hypre_F90_NAME(fnalu_hypre_structstencilcreate, FNALU_HYPRE_STRUCTSTENCILCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structstencilcreate, FNALU_HYPRE_STRUCTSTENCILCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructStencilDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structstencildestroy, FNALU_HYPRE_STRUCTSTENCILDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structstencildestroy, FNALU_HYPRE_STRUCTSTENCILDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructStencilSetElement \
        nalu_hypre_F90_NAME(fnalu_hypre_structstencilsetelement, FNALU_HYPRE_STRUCTSTENCILSETELEMENT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structstencilsetelement, FNALU_HYPRE_STRUCTSTENCILSETELEMENT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);



#define NALU_HYPRE_StructGridCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structgridcreate, FNALU_HYPRE_STRUCTGRIDCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgridcreate, FNALU_HYPRE_STRUCTGRIDCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructGridDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structgriddestroy, FNALU_HYPRE_STRUCTGRIDDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgriddestroy, FNALU_HYPRE_STRUCTGRIDDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructGridSetExtents \
        nalu_hypre_F90_NAME(fnalu_hypre_structgridsetextents, FNALU_HYPRE_STRUCTGRIDSETEXTENTS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgridsetextents, FNALU_HYPRE_STRUCTGRIDSETEXTENTS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGridSetPeriodic \
        nalu_hypre_F90_NAME(fnalu_hypre_structgridsetperiodic, FNALU_HYPRE_STRUCTGRIDSETPERIODIC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgridsetperiodic, fnalu_hypre_structsetgridperiodic)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGridAssemble \
        nalu_hypre_F90_NAME(fnalu_hypre_structgridassemble, FNALU_HYPRE_STRUCTGRIDASSEMBLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgridassemble, FNALU_HYPRE_STRUCTGRIDASSEMBLE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructGridSetNumGhost \
        nalu_hypre_F90_NAME(fnalu_hypre_structgridsetnumghost, FNALU_HYPRE_STRUCTGRIDSETNUMGHOST)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgridsetnumghost, fnalu_hypre_structsetgridnumghost)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_StructMatrixCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixcreate, FNALU_HYPRE_STRUCTMATRIXCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixcreate, FNALU_HYPRE_STRUCTMATRIXCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixdestroy, FNALU_HYPRE_STRUCTMATRIXDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixdestroy, FNALU_HYPRE_STRUCTMATRIXDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixInitialize \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixinitialize, FNALU_HYPRE_STRUCTMATRIXINITIALIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixinitialize, FNALU_HYPRE_STRUCTMATRIXINITIALIZE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixSetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetvalues, FNALU_HYPRE_STRUCTMATRIXSETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetvalues, FNALU_HYPRE_STRUCTMATRIXSETVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixSetBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetboxvalues, FNALU_HYPRE_STRUCTMATRIXSETBOXVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetboxvalues, FNALU_HYPRE_STRUCTMATRIXSETBOXVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixGetBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixgetboxvalues, FNALU_HYPRE_STRUCTMATRIXGETBOXVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixgetboxvalues, FNALU_HYPRE_STRUCTMATRIXGETBOXVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixSetConstantEntries \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetconstante, FNALU_HYPRE_STRUCTMATRIXSETCONSTANTE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetconstante, FNALU_HYPRE_STRUCTMATRIXSETCONSTANTE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixSetConstantValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetconstantv, FNALU_HYPRE_STRUCTMATRIXSETCONSTANTV)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetconstantv, FNALU_HYPRE_STRUCTMATRIXSETCONSTANTV)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixAddToValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixaddtovalues, FNALU_HYPRE_STRUCTMATRIXADDTOVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixaddtovalues, FNALU_HYPRE_STRUCTMATRIXADDTOVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixAddToBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixaddtoboxvalues, FNALU_HYPRE_STRUCTMATRIXADDTOBOXVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixaddtoboxvalues, FNALU_HYPRE_STRUCTMATRIXADDTOBOXVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixAddToConstantValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixaddtoconstant, FNALU_HYPRE_STRUCTMATRIXADDTOCONSTANT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixaddtoconstant, FNALU_HYPRE_STRUCTMATRIXADDTOCONSTANT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructMatrixAssemble \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixassemble, FNALU_HYPRE_STRUCTMATRIXASSEMBLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixassemble, FNALU_HYPRE_STRUCTMATRIXASSEMBLE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixSetNumGhost \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetnumghost, FNALU_HYPRE_STRUCTMATRIXSETNUMGHOST)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetnumghost, FNALU_HYPRE_STRUCTMATRIXSETNUMGHOST)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixGetGrid \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixgetgrid, FNALU_HYPRE_STRUCTMATRIXGETGRID)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixgetgrid, FNALU_HYPRE_STRUCTMATRIXGETGRID)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructMatrixSetSymmetric \
        nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetsymmetric, FNALU_HYPRE_STRUCTMATRIXSETSYMMETRIC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixsetsymmetric, FNALU_HYPRE_STRUCTMATRIXSETSYMMETRIC)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixPrint \
nalu_hypre_F90_NAME(fnalu_hypre_structmatrixprint, FNALU_HYPRE_STRUCTMATRIXPRINT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixprint, FNALU_HYPRE_STRUCTMATRIXPRINT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructMatrixMatvec \
nalu_hypre_F90_NAME(fnalu_hypre_structmatrixmatvec, FNALU_HYPRE_STRUCTMATRIXMATVEC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structmatrixmatvec, FNALU_HYPRE_STRUCTMATRIXMATVEC)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);



#define NALU_HYPRE_StructVectorCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorcreate, FNALU_HYPRE_STRUCTVECTORCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorcreate, FNALU_HYPRE_STRUCTVECTORCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectordestroy, FNALU_HYPRE_STRUCTVECTORDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectordestroy, FNALU_HYPRE_STRUCTVECTORDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorInitialize \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorinitialize, FNALU_HYPRE_STRUCTVECTORINITIALIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorinitialize, FNALU_HYPRE_STRUCTVECTORINITIALIZE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorSetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetvalues, FNALU_HYPRE_STRUCTVECTORSETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetvalues, FNALU_HYPRE_STRUCTVECTORSETVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructVectorSetBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetboxvalues, FNALU_HYPRE_STRUCTVECTORSETBOXVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetboxvalues, FNALU_HYPRE_STRUCTVECTORSETBOXVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorSetConstantValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetconstantv, FNALU_HYPRE_STRUCTVECTORSETCONTANTV)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetconstantv, FNALU_HYPRE_STRUCTVECTORSETCONTANTV)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorAddToValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectoraddtovalues, FNALU_HYPRE_STRUCTVECTORADDTOVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectoraddtovalues, FNALU_HYPRE_STRUCTVECTORADDTOVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorAddToBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectoraddtoboxvalu, FNALU_HYPRE_STRUCTVECTORADDTOBOXVALU)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectoraddtoboxvalu, FNALU_HYPRE_STRUCTVECTORADDTOBOXVALU)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorScaleValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorscalevalues, FNALU_HYPRE_STRUCTVECTORSCALEVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorscalevalues, FNALU_HYPRE_STRUCTVECTORSCALEVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorGetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorgetvalues, FNALU_HYPRE_STRUCTVECTORGETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorgetvalues, FNALU_HYPRE_STRUCTVECTORGETVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorGetBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorgetboxvalues, FNALU_HYPRE_STRUCTVECTORGETBOXVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorgetboxvalues, FNALU_HYPRE_STRUCTVECTORGETBOXVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructVectorAssemble \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorassemble, FNALU_HYPRE_STRUCTVECTORASSEMBLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorassemble, FNALU_HYPRE_STRUCTVECTORASSEMBLE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorSetNumGhost \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetnumghost, FNALU_HYPRE_STRUCTVECTORSETNUMGHOST)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetnumghost, FNALU_HYPRE_STRUCTVECTORSETNUMGHOST)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructVectorCopy \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorcopy, FNALU_HYPRE_STRUCTVECTORCOPY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorcopy, FNALU_HYPRE_STRUCTVECTORCOPY)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorGetMigrateCommPkg \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorgetmigrateco, FNALU_HYPRE_STRUCTVECTORGETMIGRATECO)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorgetmigrateco, FNALU_HYPRE_STRUCTVECTORGETMIGRATECO)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorMigrate \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectormigrate, FNALU_HYPRE_STRUCTVECTORMIGRATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectormigrate, FNALU_HYPRE_STRUCTVECTORMIGRATE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_CommPkgDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_commpkgdestroy, FNALU_HYPRE_COMMPKGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_commpkgdestroy, FNALU_HYPRE_COMMPKGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructVectorPrint \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorprint, FNALU_HYPRE_STRUCTVECTORPRINT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorprint, FNALU_HYPRE_STRUCTVECTORPRINT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);


#define NALU_HYPRE_StructBiCGSTABCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabcreate, FNALU_HYPRE_STRUCTBICGSTABCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabcreate, FNALU_HYPRE_STRUCTBICGSTABCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabdestroy, FNALU_HYPRE_STRUCTBICGSTABDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabdestroy, FNALU_HYPRE_STRUCTBICGSTABDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetup, FNALU_HYPRE_STRUCTBICGSTABSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetup, FNALU_HYPRE_STRUCTBICGSTABSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsolve, FNALU_HYPRE_STRUCTBICGSTABSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsolve, FNALU_HYPRE_STRUCTBICGSTABSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsettol, FNALU_HYPRE_STRUCTBICGSTABSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsettol, FNALU_HYPRE_STRUCTBICGSTABSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructBiCGSTABSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetmaxiter, FNALU_HYPRE_STRUCTBICGSTABSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetmaxiter, FNALU_HYPRE_STRUCTBICGSTABSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructBiCGSTABSetPrecond \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetprecond, FNALU_HYPRE_STRUCTBICGSTABSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetprecond, FNALU_HYPRE_STRUCTBICGSTABSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructBiCGSTABSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetlogging, FNALU_HYPRE_STRUCTBICGSTABSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetlogging, FNALU_HYPRE_STRUCTBICGSTABSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructBiCGSTABSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetprintle, FNALU_HYPRE_STRUCTBICGSTABPRINTLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabsetprintle, FNALU_HYPRE_STRUCTBICGSTABPRINTLE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructBiCGSTABGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabgetnumiter, FNALU_HYPRE_STRUCTBICGSTABGETNUMITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabgetnumiter, FNALU_HYPRE_STRUCTBICGSTABGETNUMITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructBiCGSTABGetResidual \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabgetresidua, FNALU_HYPRE_STRUCTBICGSTABGETRESIDUA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabgetresidua, FNALU_HYPRE_STRUCTBICGSTABGETRESIDUA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabgetfinalre, FNALU_HYPRE_STRUCTBICGSTABGETFINALRE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structbicgstabgetfinalre, FNALU_HYPRE_STRUCTBICGSTABGETFINALRE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructGMRESCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmrescreate, FNALU_HYPRE_STRUCTGMRESCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmrescreate, FNALU_HYPRE_STRUCTGMRESCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmresdestroy, FNALU_HYPRE_STRUCTGMRESDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmresdestroy, FNALU_HYPRE_STRUCTGMRESDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmressetup, FNALU_HYPRE_STRUCTGMRESSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmressetup, FNALU_HYPRE_STRUCTGMRESSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmressolve, FNALU_HYPRE_STRUCTGMRESSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmressolve, FNALU_HYPRE_STRUCTGMRESSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmressettol, FNALU_HYPRE_STRUCTGMRESSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmressettol, FNALU_HYPRE_STRUCTGMRESSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructGMRESSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmressetmaxiter, FNALU_HYPRE_STRUCTGMRESSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmressetmaxiter, FNALU_HYPRE_STRUCTGMRESSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGMRESSetPrecond \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmressetprecond, FNALU_HYPRE_STRUCTGMRESSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmressetprecond, FNALU_HYPRE_STRUCTGMRESSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructGMRESSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmressetlogging, FNALU_HYPRE_STRUCTGMRESSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmressetlogging, FNALU_HYPRE_STRUCTGMRESSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGMRESSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmressetprintlevel, FNALU_HYPRE_STRUCTGMRESPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmressetprintlevel, FNALU_HYPRE_STRUCTGMRESPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGMRESGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmresgetnumiterati, FNALU_HYPRE_STRUCTGMRESGETNUMITERATI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmresgetnumiterati, FNALU_HYPRE_STRUCTGMRESGETNUMITERATI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructGMRESGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structgmresgetfinalrelat, FNALU_HYPRE_STRUCTGMRESGETFINALRELAT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structgmresgetfinalrelat, FNALU_HYPRE_STRUCTGMRESGETFINALRELAT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructHybridCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridcreate, FNALU_HYPRE_STRUCTHYBRIDCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridcreate, FNALU_HYPRE_STRUCTHYBRIDCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybriddestroy, FNALU_HYPRE_STRUCTHYBRIDDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybriddestroy, FNALU_HYPRE_STRUCTHYBRIDDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetup, FNALU_HYPRE_STRUCTHYBRIDSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetup, FNALU_HYPRE_STRUCTHYBRIDSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsolve, FNALU_HYPRE_STRUCTHYBRIDSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsolve, FNALU_HYPRE_STRUCTHYBRIDSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridSetSolverType \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetsolvertyp, FNALU_HYPRE_STRUCTHYBRIDSETSOLVERTYP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetsolvertyp, FNALU_HYPRE_STRUCTHYBRIDSETSOLVERTYP)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetStopCrit \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetstopcrit, FNALU_HYPRE_STRUCTHYBRIDSETSTOPCRIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetstopcrit, FNALU_HYPRE_STRUCTHYBRIDSETSTOPCRIT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetKDim \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetkdim, FNALU_HYPRE_STRUCTHYBRIDSETKDIM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetkdim, FNALU_HYPRE_STRUCTHYBRIDSETKDIM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsettol, FNALU_HYPRE_STRUCTHYBRIDSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsettol, FNALU_HYPRE_STRUCTHYBRIDSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructHybridSetConvergenceTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetconvergen, FNALU_HYPRE_STRUCTHYBRIDSETCONVERGEN)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetconvergen, FNALU_HYPRE_STRUCTHYBRIDSETCONVERGEN)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructHybridSetPCGAbsoluteTolFactor \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetpcgabsolu, FNALU_HYPRE_STRUCTHYBRIDSETABSOLU)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetpcgabsolu, FNALU_HYPRE_STRUCTHYBRIDSETABSOLU)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructHybridSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetmaxiter, FNALU_HYPRE_STRUCTHYBRIDSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetmaxiter, FNALU_HYPRE_STRUCTHYBRIDSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetDSCGMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetdscgmaxit, FNALU_HYPRE_STRUCTHYBRIDSETDSCGMAXIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetdscgmaxit, FNALU_HYPRE_STRUCTHYBRIDSETDSCGMAXIT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetPCGMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetpcgmaxite, FNALU_HYPRE_STRUCTHYBRIDSETPCGMAXITE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetpcgmaxite, FNALU_HYPRE_STRUCTHYBRIDSETPCGMAXITE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetTwoNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsettwonorm, FNALU_HYPRE_STRUCTHYBRIDSETTWONORM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsettwonorm, FNALU_HYPRE_STRUCTHYBRIDSETTWONORM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetrelchange, FNALU_HYPRE_STRUCTHYBRIDSETRELCHANGE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetrelchange, FNALU_HYPRE_STRUCTHYBRIDSETRELCHANGE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetPrecond \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetprecond, FNALU_HYPRE_STRUCTHYBRIDSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetprecond, FNALU_HYPRE_STRUCTHYBRIDSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructHybridSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetlogging, FNALU_HYPRE_STRUCTHYBRIDSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetlogging, FNALU_HYPRE_STRUCTHYBRIDSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetprintleve, FNALU_HYPRE_STRUCTHYBRIDSETPRINTLEVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridsetprintleve, FNALU_HYPRE_STRUCTHYBRIDSETPRINTLEVE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridgetnumiterat, FNALU_HYPRE_STRUCTHYBRIDGETNUMITERAT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridgetnumiterat, FNALU_HYPRE_STRUCTHYBRIDGETNUMITERAT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridGetDSCGNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridgetdscgnumit, FNALU_HYPRE_STRUCTHYBRIDGETDSCGNUMIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridgetdscgnumit, FNALU_HYPRE_STRUCTHYBRIDGETDSCGNUMIT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridGetPCGNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridgetpcgnumite, FNALU_HYPRE_STRUCTHYBRIDGETPCGNUMITE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridgetpcgnumite, FNALU_HYPRE_STRUCTHYBRIDGETPCGNUMITE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructHybridGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structhybridgetfinalrela, FNALU_HYPRE_STRUCTHYBRIDGETFINALRELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structhybridgetfinalrela, FNALU_HYPRE_STRUCTHYBRIDGETFINALRELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructVectorSetRandomValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetrandomvalu, FNALU_HYPRE_STRUCTVECTORSETRANDOMVALU)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structvectorsetrandomvalu, FNALU_HYPRE_STRUCTVECTORSETRANDOMVALU)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSetRandomValues \
        nalu_hypre_F90_NAME(fnalu_hypre_structsetrandomvalues, FNALU_HYPRE_STRUCTSETRANDOMVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsetrandomvalues, FNALU_HYPRE_STRUCTSETRANDOMVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSetupInterpreter \
        nalu_hypre_F90_NAME(fnalu_hypre_structsetupinterpreter, FNALU_HYPRE_STRUCTSETUPINTERPRETER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsetupinterpreter, FNALU_HYPRE_STRUCTSETUPINTERPRETER)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSetupMatvec \
        nalu_hypre_F90_NAME(fnalu_hypre_structsetupmatvec, FNALU_HYPRE_STRUCTSETUPMATVEC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsetupmatvec, FNALU_HYPRE_STRUCTSETUPMATVEC)
(nalu_hypre_F90_Obj *);



#define NALU_HYPRE_StructJacobiCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobicreate, FNALU_HYPRE_STRUCTJACOBICREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobicreate, FNALU_HYPRE_STRUCTJACOBICREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobidestroy, FNALU_HYPRE_STRUCTJACOBIDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobidestroy, FNALU_HYPRE_STRUCTJACOBIDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobisetup, FNALU_HYPRE_STRUCTJACOBISETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobisetup, FNALU_HYPRE_STRUCTJACOBISETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobisolve, FNALU_HYPRE_STRUCTJACOBISOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobisolve, FNALU_HYPRE_STRUCTJACOBISOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobisettol, FNALU_HYPRE_STRUCTJACOBISETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobisettol, FNALU_HYPRE_STRUCTJACOBISETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructJacobiGetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobigettol, FNALU_HYPRE_STRUCTJACOBIGETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobigettol, FNALU_HYPRE_STRUCTJACOBIGETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructJacobiSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobisetmaxiter, FNALU_HYPRE_STRUCTJACOBISETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobisetmaxiter, FNALU_HYPRE_STRUCTJACOBISETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructJacobiGetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobigetmaxiter, FNALU_HYPRE_STRUCTJACOBIGETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobigetmaxiter, FNALU_HYPRE_STRUCTJACOBIGETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructJacobiSetZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobisetzeroguess, FNALU_HYPRE_STRUCTJACOBISETZEROGUESS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobisetzeroguess, FNALU_HYPRE_STRUCTJACOBISETZEROGUESS)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiGetZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobigetzeroguess, FNALU_HYPRE_STRUCTJACOBIGETZEROGUESS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobigetzeroguess, FNALU_HYPRE_STRUCTJACOBIGETZEROGUESS)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiSetNonZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobisetnonzerogu, FNALU_HYPRE_STRUCTJACOBISETNONZEROGU)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobisetnonzerogu, FNALU_HYPRE_STRUCTJACOBISETNONZEROGU)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructJacobiGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobigetnumiterati, FNALU_HYPRE_STRUCTJACOBIGETNUMITERATI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobigetnumiterati, FNALU_HYPRE_STRUCTJACOBIGETNUMITERATI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructJacobiGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structjacobigetfinalrela, FNALU_HYPRE_STRUCTJACOBIGETFINALRELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structjacobigetfinalrela, FNALU_HYPRE_STRUCTJACOBIGETFINALRELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructPCGCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgcreate, FNALU_HYPRE_STRUCTPCGCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgcreate, FNALU_HYPRE_STRUCTPCGCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgdestroy, FNALU_HYPRE_STRUCTPCGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgdestroy, FNALU_HYPRE_STRUCTPCGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetup, FNALU_HYPRE_STRUCTPCGSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetup, FNALU_HYPRE_STRUCTPCGSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgsolve, FNALU_HYPRE_STRUCTPCGSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgsolve, FNALU_HYPRE_STRUCTPCGSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgsettol, FNALU_HYPRE_STRUCTPCGSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgsettol, FNALU_HYPRE_STRUCTPCGSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructPCGSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetmaxiter, FNALU_HYPRE_STRUCTPCGSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetmaxiter, FNALU_HYPRE_STRUCTPCGSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGSetTwoNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgsettwonorm, FHYRPE_STRUCTPCGSETTWONORM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgsettwonorm, FHYRPE_STRUCTPCGSETTWONORM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGSetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetrelchange, FNALU_HYPRE_STRUCTPCGSETRELCHANGE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetrelchange, FNALU_HYPRE_STRUCTPCGSETRELCHANGE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGSetPrecond \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetprecond, FNALU_HYPRE_STRUCTPCGSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetprecond, FNALU_HYPRE_STRUCTPCGSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPCGSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetlogging, FHYPRES_TRUCTPCFSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetlogging, FHYPRES_TRUCTPCFSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetprintlevel, FNALU_HYPRE_STRUCTPCGSETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcgsetprintlevel, FNALU_HYPRE_STRUCTPCGSETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcggetnumiteration, FNALU_HYPRE_STRUCTPCGGETNUMITERATION)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcggetnumiteration, FNALU_HYPRE_STRUCTPCGGETNUMITERATION)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPCGGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structpcggetfinalrelativ, FNALU_HYPRE_STRUCTPCGGETFINALRELATIV)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpcggetfinalrelativ, FNALU_HYPRE_STRUCTPCGGETFINALRELATIV)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructDiagScaleSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_structdiagscalesetup, FNALU_HYPRE_STRUCTDIAGSCALESETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structdiagscalesetup, FNALU_HYPRE_STRUCTDIAGSCALESETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructDiagScaleSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_structdiagscalesolve, FNALU_HYPRE_STRUCTDIAGSCALESOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structdiagscalesolve, FNALU_HYPRE_STRUCTDIAGSCALESOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);



#define NALU_HYPRE_StructPFMGCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgcreate, FNALU_HYPRE_STRUCTPFMGCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgcreate, FNALU_HYPRE_STRUCTPFMGCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgdestroy, FNALU_HYPRE_STRUCTPFMGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgdestroy, FNALU_HYPRE_STRUCTPFMGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetup, FNALU_HYPRE_STRUCTPFMGSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetup, FNALU_HYPRE_STRUCTPFMGSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsolve, FNALU_HYPRE_STRUCTPFMGSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsolve, FNALU_HYPRE_STRUCTPFMGSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsettol, FNALU_HYPRE_STRUCTPFMGSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsettol, FNALU_HYPRE_STRUCTPFMGSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructPFMGGetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggettol, FNALU_HYPRE_STRUCTPFMGGETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggettol, FNALU_HYPRE_STRUCTPFMGGETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructPFMGSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetmaxiter, FNALU_HYPRE_STRUCTPFMGSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetmaxiter, FNALU_HYPRE_STRUCTPFMGSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetmaxiter, FNALU_HYPRE_STRUCTPFMGGETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetmaxiter, FNALU_HYPRE_STRUCTPFMGGETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetMaxLevels \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetmaxlevels, FNALU_HYPRE_STRUCTPFMGSETMAXLEVELS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetmaxlevels, FNALU_HYPRE_STRUCTPFMGSETMAXLEVELS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetMaxLevels \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetmaxlevels, FNALU_HYPRE_STRUCTPFMGGETMAXLEVELS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetmaxlevels, FNALU_HYPRE_STRUCTPFMGGETMAXLEVELS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetrelchange, FNALU_HYPRE_STRUCTPFMGSETRELCHANGE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetrelchange, FNALU_HYPRE_STRUCTPFMGSETRELCHANGE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetrelchange, FNALU_HYPRE_STRUCTPFMGGETRELCHANGE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetrelchange, FNALU_HYPRE_STRUCTPFMGGETRELCHANGE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetzeroguess, FNALU_HYPRE_STRUCTPFMGSETZEROGUESS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetzeroguess, FNALU_HYPRE_STRUCTPFMGSETZEROGUESS)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGGetZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetzeroguess, FNALU_HYPRE_STRUCTPFMGGETZEROGUESS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetzeroguess, FNALU_HYPRE_STRUCTPFMGGETZEROGUESS)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSetNonZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetnonzerogues, FHYPRES_TRUCTPFMGSETNONZEROGUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetnonzerogues, FHYPRES_TRUCTPFMGSETNONZEROGUES)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructPFMGSetSkipRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetskiprelax, FNALU_HYPRE_STRUCTPFMGSETSKIPRELAX)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetskiprelax, FNALU_HYPRE_STRUCTPFMGSETSKIPRELAX)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetSkipRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetskiprelax, FNALU_HYPRE_STRUCTPFMGGETSKIPRELAX)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetskiprelax, FNALU_HYPRE_STRUCTPFMGGETSKIPRELAX)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetRelaxType \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetrelaxtype, FNALU_HYPRE_STRUCTPFMGSETRELAXTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetrelaxtype, FNALU_HYPRE_STRUCTPFMGSETRELAXTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetRelaxType \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetrelaxtype, FNALU_HYPRE_STRUCTPFMGGETRELAXTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetrelaxtype, FNALU_HYPRE_STRUCTPFMGGETRELAXTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetRAPType \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetraptype, FNALU_HYPRE_STRUCTPFMGSETRAPTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetraptype, FNALU_HYPRE_STRUCTPFMGSETRAPTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetRAPType \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetraptype, FNALU_HYPRE_STRUCTPFMGGETRAPTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetraptype, FNALU_HYPRE_STRUCTPFMGGETRAPTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetNumPreRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetnumprerelax, FNALU_HYPRE_STRUCTPFMGSETNUMPRERELAX)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetnumprerelax, FNALU_HYPRE_STRUCTPFMGSETNUMPRERELAX)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetNumPreRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetnumprerelax, FNALU_HYPRE_STRUCTPFMGGETNUMPRERELAX)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetnumprerelax, FNALU_HYPRE_STRUCTPFMGGETNUMPRERELAX)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetNumPostRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetnumpostrela, FNALU_HYPRE_STRUCTPFMGSETNUMPOSTRELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetnumpostrela, FNALU_HYPRE_STRUCTPFMGSETNUMPOSTRELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetNumPostRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetnumpostrela, FNALU_HYPRE_STRUCTPFMGGETNUMPOSTRELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetnumpostrela, FNALU_HYPRE_STRUCTPFMGGETNUMPOSTRELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetDxyz \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetdxyz, FNALU_HYPRE_STRUCTPFMGSETDXYZ)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetdxyz, FNALU_HYPRE_STRUCTPFMGSETDXYZ)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructPFMGSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetlogging, FNALU_HYPRE_STRUCTPFMGSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetlogging, FNALU_HYPRE_STRUCTPFMGSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetlogging, FNALU_HYPRE_STRUCTPFMGGETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetlogging, FNALU_HYPRE_STRUCTPFMGGETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetprintlevel, FNALU_HYPRE_STRUCTPFMGSETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmgsetprintlevel, FNALU_HYPRE_STRUCTPFMGSETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetprintlevel, FNALU_HYPRE_STRUCTPFMGGETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetprintlevel, FNALU_HYPRE_STRUCTPFMGGETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetnumiteratio, FNALU_HYPRE_STRUCTPFMGGETNUMITERATIO)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetnumiteratio, FNALU_HYPRE_STRUCTPFMGGETNUMITERATIO)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetfinalrelati, FNALU_HYPRE_STRUCTPFMGGETFINALRELATI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structpfmggetfinalrelati, FNALU_HYPRE_STRUCTPFMGGETFINALRELATI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_StructSMGCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgcreate, FNALU_HYPRE_STRUCTSMGCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgcreate, FNALU_HYPRE_STRUCTSMGCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgdestroy, FNALU_HYPRE_STRUCTSMGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgdestroy, FNALU_HYPRE_STRUCTSMGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetup, FNALU_HYPRE_STRUCTSMGSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetup, FNALU_HYPRE_STRUCTSMGSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsolve, FNALU_HYPRE_STRUCTSMGSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsolve, FNALU_HYPRE_STRUCTSMGSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGSetMemoryUse \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetmemoryuse, FNALU_HYPRE_STRUCTSMGSETMEMORYUSE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetmemoryuse, FNALU_HYPRE_STRUCTSMGSETMEMORYUSE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetMemoryUse \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetmemoryuse, FNALU_HYPRE_STRUCTSMGGETMEMORYUSE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetmemoryuse, FNALU_HYPRE_STRUCTSMGGETMEMORYUSE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsettol, FNALU_HYPRE_STRUCTSMGSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsettol, FNALU_HYPRE_STRUCTSMGSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSMGGetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggettol, FNALU_HYPRE_STRUCTSMGGETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggettol, FNALU_HYPRE_STRUCTSMGGETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSMGSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetmaxiter, FNALU_HYPRE_STRUCTSMGSETMAXTITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetmaxiter, FNALU_HYPRE_STRUCTSMGSETMAXTITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetmaxiter, FNALU_HYPRE_STRUCTSMGGETMAXTITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetmaxiter, FNALU_HYPRE_STRUCTSMGGETMAXTITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetrelchange, FNALU_HYPRE_STRUCTSMGSETRELCHANGE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetrelchange, FNALU_HYPRE_STRUCTSMGSETRELCHANGE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetrelchange, FNALU_HYPRE_STRUCTSMGGETRELCHANGE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetrelchange, FNALU_HYPRE_STRUCTSMGGETRELCHANGE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetzeroguess, FNALU_HYPRE_STRUCTSMGSETZEROGUESS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetzeroguess, FNALU_HYPRE_STRUCTSMGSETZEROGUESS)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGGetZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetzeroguess, FNALU_HYPRE_STRUCTSMGGETZEROGUESS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetzeroguess, FNALU_HYPRE_STRUCTSMGGETZEROGUESS)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGSetNonZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetnonzerogues, FNALU_HYPRE_STRUCTSMGSETNONZEROGUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetnonzerogues, FNALU_HYPRE_STRUCTSMGSETNONZEROGUES)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSMGGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetnumiteration, FNALU_HYPRE_STRUCTSMGGETNUMITERATION)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetnumiteration, FNALU_HYPRE_STRUCTSMGGETNUMITERATION)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetfinalrelativ, FNALU_HYPRE_STRUCTSMGGETFINALRELATIV)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetfinalrelativ, FNALU_HYPRE_STRUCTSMGGETFINALRELATIV)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSMGSetNumPreRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetnumprerelax, FNALU_HYPRE_STRUCTSMGSETNUMPRERELAX)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetnumprerelax, FNALU_HYPRE_STRUCTSMGSETNUMPRERELAX)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetNumPreRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetnumprerelax, FNALU_HYPRE_STRUCTSMGGETNUMPRERELAX)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetnumprerelax, FNALU_HYPRE_STRUCTSMGGETNUMPRERELAX)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetNumPostRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetnumpostrelax, FNALU_HYPRE_STRUCTSMGSETNUMPOSTRELAX)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetnumpostrelax, FNALU_HYPRE_STRUCTSMGSETNUMPOSTRELAX)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetNumPostRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetnumpostrelax, FNALU_HYPRE_STRUCTSMGGETNUMPOSTRELAX)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetnumpostrelax, FNALU_HYPRE_STRUCTSMGGETNUMPOSTRELAX)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetlogging, FNALU_HYPRE_STRUCTSMGSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetlogging, FNALU_HYPRE_STRUCTSMGSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetlogging, FNALU_HYPRE_STRUCTSMGGETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetlogging, FNALU_HYPRE_STRUCTSMGGETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetprintlevel, FNALU_HYPRE_STRUCTSMGSETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmgsetprintlevel, FNALU_HYPRE_STRUCTSMGSETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSMGGetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_structsmggetprintlevel, FNALU_HYPRE_STRUCTSMGGETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsmggetprintlevel, FNALU_HYPRE_STRUCTSMGGETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_StructSparseMSGCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgcreate, FNALU_HYPRE_STRUCTSPARSEMSGCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgcreate, FNALU_HYPRE_STRUCTSPARSEMSGCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgdestroy, FNALU_HYPRE_STRUCTSPARSEMSGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgdestroy, FNALU_HYPRE_STRUCTSPARSEMSGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetup, FHYRPE_STRUCTSPARSEMSGSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetup, FHYRPE_STRUCTSPARSEMSGSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsolve, FNALU_HYPRE_STRUCTSPARSEMSGSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsolve, FNALU_HYPRE_STRUCTSPARSEMSGSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGSetJump \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetjump, FNALU_HYPRE_STRUCTSPARSEMSGSETJUMP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetjump, FNALU_HYPRE_STRUCTSPARSEMSGSETJUMP)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsettol, FNALU_HYPRE_STRUCTSPARSEMSGSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsettol, FNALU_HYPRE_STRUCTSPARSEMSGSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSparseMSGSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetmaxite, FNALU_HYPRE_STRUCTSPARSEMSGSETMAXITE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetmaxite, FNALU_HYPRE_STRUCTSPARSEMSGSETMAXITE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetrelcha, FNALU_HYPRE_STRUCTSPARSEMSGSETRELCHA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetrelcha, FNALU_HYPRE_STRUCTSPARSEMSGSETRELCHA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetzerogu, FNALU_HYPRE_STRUCTSPARSEMSGSETZEROGU)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetzerogu, FNALU_HYPRE_STRUCTSPARSEMSGSETZEROGU)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGSetNonZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetnonzer, FNALU_HYPRE_STRUCTSPARSEMSGSETNONZER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetnonzer, FNALU_HYPRE_STRUCTSPARSEMSGSETNONZER)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_StructSparseMSGGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsggetnumite, FNALU_HYPRE_STRUCTSPARSEMSGGETNUMITE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsggetnumite, FNALU_HYPRE_STRUCTSPARSEMSGGETNUMITE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsggetfinalr, FHYRPE_STRUCTSPARSEMSGGETFINALR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsggetfinalr, FHYRPE_STRUCTSPARSEMSGGETFINALR)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_StructSparseMSGSetRelaxType \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetrelaxt, FNALU_HYPRE_STRUCTSPARSEMSGSETRELAXT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetrelaxt, FNALU_HYPRE_STRUCTSPARSEMSGSETRELAXT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetNumPreRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetnumpre, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMPRE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetnumpre, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMPRE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetNumPostRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetnumpos, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMPOS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetnumpos, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMPOS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetNumFineRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetnumfin, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMFIN)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetnumfin, FNALU_HYPRE_STRUCTSPARSEMSGSETNUMFIN)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetloggin, FNALU_HYPRE_STRUCTSPARSEMSGSETLOGGIN)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetloggin, FNALU_HYPRE_STRUCTSPARSEMSGSETLOGGIN)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_StructSparseMSGSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetprintl, FNALU_HYPRE_STRUCTSPARSEMSGSETPRINTL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_structsparsemsgsetprintl, FNALU_HYPRE_STRUCTSPARSEMSGSETPRINTL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#ifdef __cplusplus
}
#endif
