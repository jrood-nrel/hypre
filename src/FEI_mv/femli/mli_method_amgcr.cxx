/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#if 0 /* RDF: Not sure this is really needed */
#ifdef WIN32
#define strcmp _stricmp
#endif
#endif

#include <string.h>
#include "NALU_HYPRE.h"
#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_parcsr_ls.h"
#include "mli_utils.h"
#include "mli_matrix.h"
#include "mli_matrix_misc.h"
#include "mli_vector.h"
#include "mli_solver.h"
#include "mli_method_amgcr.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "ParaSails/Matrix.h"
#include "ParaSails/ParaSails.h"
#include "_nalu_hypre_parcsr_ls.h"
#ifdef __cplusplus
}
#endif

#define habs(x) ((x > 0) ? x : (-x))

/* ********************************************************************* *
 * constructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGCR::MLI_Method_AMGCR( MPI_Comm comm ) : MLI_Method( comm )
{
   char name[100];

   strcpy(name, "AMGCR");
   setName( name );
   setID( MLI_METHOD_AMGCR_ID );
   maxLevels_     = 40;
   numLevels_     = 2;
   currLevel_     = 0;
   outputLevel_   = 0;
   findMIS_       = 0;
   targetMu_      = 0.25;
   numTrials_     = 1;
   numVectors_    = 1;
   minCoarseSize_ = 100;
   cutThreshold_  = 0.01;
   strcpy(smoother_, "Jacobi");
   smootherNum_  = 1;
   smootherWgts_ = new double[2];
   smootherWgts_[0] = smootherWgts_[1] = 1.0;
   strcpy(coarseSolver_, "SuperLU");
   coarseSolverNum_ = 1;
   coarseSolverWgts_ = new double[20];
   for (int j = 0; j < 20; j++) coarseSolverWgts_ [j] = 1.0;
   RAPTime_            = 0.0;
   totalTime_          = 0.0;
   strcpy(paramFile_, "empty");
   PDegree_ = 2;
}

/* ********************************************************************* *
 * destructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGCR::~MLI_Method_AMGCR()
{
   if (smootherWgts_     != NULL) delete [] smootherWgts_;
   if (coarseSolverWgts_ != NULL) delete [] coarseSolverWgts_;
}

/* ********************************************************************* *
 * set parameters
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::setParams(char *inName, int argc, char *argv[])
{
   int        i, mypid, level, nSweeps=1;
   double     *weights=NULL;
   char       param1[256], param2[256], *param3;
   MPI_Comm   comm;

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   sscanf(inName, "%s", param1);
   if ( outputLevel_ >= 1 && mypid == 0 )
      printf("\tMLI_Method_AMGCR::setParam = %s\n", inName);
   if ( !strcmp(param1, "setOutputLevel" ))
   {
      sscanf(inName,"%s %d", param1, &level);
      return (setOutputLevel(level));
   }
   else if ( !strcmp(param1, "setNumLevels" ))
   {
      sscanf(inName,"%s %d", param1, &level);
      return (setNumLevels(level));
   }
   else if ( !strcmp(param1, "useMIS" ))
   {
      findMIS_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setTargetMu" ))
   {
      sscanf(inName,"%s %lg", param1, &targetMu_);
      if (targetMu_ < 0.0) targetMu_ = 0.5;
      if (targetMu_ > 1.0) targetMu_ = 0.5;
      return 0;
   }
   else if ( !strcmp(param1, "setNumTrials" ))
   {
      sscanf(inName,"%s %d", param1, &numTrials_);
      if (numTrials_ < 1) numTrials_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setNumVectors" ))
   {
      sscanf(inName,"%s %d", param1, &numVectors_);
      if (numVectors_ < 1) numVectors_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setPDegree" ))
   {
      sscanf(inName,"%s %d", param1, &PDegree_);
      if (PDegree_ < 0) PDegree_ = 0;
      if (PDegree_ > 3) PDegree_ = 3;
      return 0;
   }
   else if ( !strcmp(param1, "setSmoother" ))
   {
      sscanf(inName,"%s %s", param1, param2);
      if ( argc != 2 )
      {
         printf("MLI_Method_AMGCR::setParams ERROR - setSmoother needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      }
      nSweeps = *(int *)   argv[0];
      weights = (double *) argv[1];
      smootherNum_ = nSweeps;
      if (smootherWgts_ != NULL) delete [] smootherWgts_;
      smootherWgts_ = new double[nSweeps];
      for (i = 0; i < nSweeps; i++) smootherWgts_[i] = weights[i];
      strcpy(smoother_, param2);
      return 0;
   }
   else if (!strcmp(param1, "setCoarseSolver"))
   {
      sscanf(inName,"%s %s", param1, param2);
      if ( strcmp(param2, "SuperLU") && argc != 2 )
      {
         printf("MLI_Method_AMGCR::setParams ERROR - setCoarseSolver needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      }
      else if ( strcmp(param2, "SuperLU") )
      {
         strcpy(coarseSolver_, param2);
         coarseSolverNum_ = *(int *) argv[0];
         if (coarseSolverWgts_ != NULL) delete [] coarseSolverWgts_;
         coarseSolverWgts_ = new double[coarseSolverNum_];
         weights = (double *) argv[1];
         for (i = 0; i < coarseSolverNum_; i++) smootherWgts_[i] = weights[i];
      }
      else if ( !strcmp(param2, "SuperLU") )
      {
         if (coarseSolverWgts_ != NULL) delete [] coarseSolverWgts_;
         coarseSolverWgts_ = NULL;
         weights = NULL;
         coarseSolverNum_ = 1;
      }
      return 0;
   }
   else if ( !strcmp(param1, "setParamFile" ))
   {
      param3 = (char *) argv[0];
      strcpy( paramFile_, param3 );
      return 0;
   }
   else if ( !strcmp(param1, "print" ))
   {
      print();
      return 0;
   }
   return 1;
}

/***********************************************************************
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::setup( MLI *mli )
{
   int         level, mypid, *ISMarker, localNRows;
   int         irow, nrows, gNRows, numFpts, *fList;;
   int         *ADiagI, *ADiagJ, jcol;
   double      startTime, elapsedTime;
   char        paramString[100], *targv[10];
   MLI_Matrix  *mli_Pmat, *mli_Rmat, *mli_Amat, *mli_cAmat, *mli_Affmat;
   MLI_Matrix  *mli_Afcmat;
   MLI_Solver  *smootherPtr, *csolvePtr;
   MPI_Comm    comm;
   nalu_hypre_ParCSRMatrix *hypreA, *hypreP, *hypreR, *hypreAP, *hypreAC;
   nalu_hypre_CSRMatrix *ADiag;
   MLI_Function    *funcPtr;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGCR::setup begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* traverse all levels                                             */
   /* --------------------------------------------------------------- */

   RAPTime_ = 0.0;
   level    = 0;
   comm     = getComm();
   MPI_Comm_rank( comm, &mypid );
   totalTime_ = MLI_Utils_WTime();

   for (level = 0; level < numLevels_; level++ )
   {
      currLevel_ = level;
      if (level == numLevels_-1) break;

      /* -------------------------------------------------- */
      /* fetch fine grid matrix information                 */
      /* -------------------------------------------------- */

      mli_Amat = mli->getSystemMatrix(level);
      nalu_hypre_assert (mli_Amat != NULL);
      hypreA = (nalu_hypre_ParCSRMatrix *) mli_Amat->getMatrix();
      gNRows = nalu_hypre_ParCSRMatrixGlobalNumRows(hypreA);
      ADiag = nalu_hypre_ParCSRMatrixDiag(hypreA);
      localNRows = nalu_hypre_CSRMatrixNumRows(ADiag);
      if (localNRows < minCoarseSize_) break;

      if (mypid == 0 && outputLevel_ > 0)
      {
         printf("\t*****************************************************\n");
         printf("\t*** AMGCR : level = %d, nrows = %d\n", level, gNRows);
         printf("\t-----------------------------------------------------\n");
      }

      /* -------------------------------------------------- */
      /* perform coarsening and P                           */
      /* -------------------------------------------------- */

      if (findMIS_ > 0)
      {
#if 0
         nalu_hypre_BoomerAMGCoarsen(hypreA, hypreA, 0, 0, &ISMarker);
#else
         ISMarker = new int[localNRows];
         for (irow = 0; irow < localNRows; irow++) ISMarker[irow] = 0;
         ADiag  = nalu_hypre_ParCSRMatrixDiag(hypreA);
         ADiagI = nalu_hypre_CSRMatrixI(ADiag);
         ADiagJ = nalu_hypre_CSRMatrixJ(ADiag);
         for (irow = 0; irow < localNRows; irow++)
         {
            if (ISMarker[irow] == 0)
            {
               ISMarker[irow] = 1;
               for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
                  if (ISMarker[ADiagJ[jcol]] == 0)
                     ISMarker[ADiagJ[jcol]] = -1;
            }
         }
         for (irow = 0; irow < localNRows; irow++)
            if (ISMarker[irow] < 0) ISMarker[irow] = 0;
#endif
      }
      else
      {
         ISMarker = new int[localNRows];
         for (irow = 0; irow < localNRows; irow++) ISMarker[irow] = 0;
      }
      for (irow = 0; irow < localNRows; irow++)
         if (ISMarker[irow] < 0) ISMarker[irow] = 0;
      mli_Affmat = performCR(mli_Amat, ISMarker, &mli_Afcmat);

      nrows = 0;
      for (irow = 0; irow < localNRows; irow++)
         if (ISMarker[irow] == 1) nrows++;
      if (nrows < minCoarseSize_) break;
      mli_Pmat = createPmat(ISMarker, mli_Amat, mli_Affmat, mli_Afcmat);
      delete mli_Afcmat;
      if (mli_Pmat == NULL) break;
      mli->setProlongation(level+1, mli_Pmat);
      mli_Rmat = createRmat(ISMarker, mli_Amat, mli_Affmat);
      mli->setRestriction(level, mli_Rmat);

      /* -------------------------------------------------- */
      /* construct and set the coarse grid matrix           */
      /* -------------------------------------------------- */

      startTime = MLI_Utils_WTime();
      if (mypid == 0 && outputLevel_ > 0) printf("\tComputing RAP\n");
      hypreP = (nalu_hypre_ParCSRMatrix *) mli_Pmat->getMatrix();
      hypreR = (nalu_hypre_ParCSRMatrix *) mli_Rmat->getMatrix();
      hypreAP = nalu_hypre_ParMatmul(hypreA, hypreP);
      hypreAC = nalu_hypre_ParMatmul(hypreR, hypreAP);
      sprintf(paramString, "NALU_HYPRE_ParCSR");
      funcPtr = new MLI_Function();
      MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
      mli_cAmat = new MLI_Matrix((void*) hypreAC, paramString, funcPtr);
      delete funcPtr;
      nalu_hypre_ParCSRMatrixDestroy(hypreAP);

      mli->setSystemMatrix(level+1, mli_cAmat);
      elapsedTime = (MLI_Utils_WTime() - startTime);
      RAPTime_ += elapsedTime;
      if (mypid == 0 && outputLevel_ > 0)
         printf("\tRAP computed, time = %e seconds.\n", elapsedTime);

      /* -------------------------------------------------- */
      /* set the smoothers                                  */
      /* (if domain decomposition and ARPACKA SuperLU       */
      /* smoothers is requested, perform special treatment, */
      /* and if domain decomposition and SuperLU smoother   */
      /* is requested with multiple local subdomains, again */
      /* perform special treatment.)                        */
      /* -------------------------------------------------- */

      smootherPtr = MLI_Solver_CreateFromName(smoother_);
      targv[0] = (char *) &smootherNum_;
      targv[1] = (char *) smootherWgts_;
      sprintf(paramString, "relaxWeight");
      smootherPtr->setParams(paramString, 2, targv);
      numFpts = 0;
      for (irow = 0; irow < localNRows; irow++)
         if (ISMarker[irow] == 0) numFpts++;
#if 1
      if (numFpts > 0)
      {
         fList = new int[numFpts];
         numFpts = 0;
         for (irow = 0; irow < localNRows; irow++)
            if (ISMarker[irow] == 0) fList[numFpts++] = irow;
         targv[0] = (char *) &numFpts;
         targv[1] = (char *) fList;
         sprintf(paramString, "setFptList");
         smootherPtr->setParams(paramString, 2, targv);
      }
      sprintf(paramString, "setModifiedDiag");
      smootherPtr->setParams(paramString, 0, NULL);
      smootherPtr->setup(mli_Affmat);
      mli->setSmoother(level, MLI_SMOOTHER_PRE, smootherPtr);
      sprintf(paramString, "ownAmat");
      smootherPtr->setParams(paramString, 0, NULL);
#else
      printf("whole grid smoothing\n");
      smootherPtr->setup(mli_Amat);
      mli->setSmoother(level, MLI_SMOOTHER_PRE, smootherPtr);
      mli->setSmoother(level, MLI_SMOOTHER_POST, smootherPtr);
#endif
   }

   /* --------------------------------------------------------------- */
   /* set the coarse grid solver                                      */
   /* --------------------------------------------------------------- */

   if (mypid == 0 && outputLevel_ > 0) printf("\tCoarse level = %d\n",level);
   csolvePtr = MLI_Solver_CreateFromName( coarseSolver_ );
   if (strcmp(coarseSolver_, "SuperLU"))
   {
      targv[0] = (char *) &coarseSolverNum_;
      targv[1] = (char *) coarseSolverWgts_ ;
      sprintf(paramString, "relaxWeight");
      csolvePtr->setParams(paramString, 2, targv);
   }
   mli_Amat = mli->getSystemMatrix(level);
   csolvePtr->setup(mli_Amat);
   mli->setCoarseSolve(csolvePtr);
   totalTime_ = MLI_Utils_WTime() - totalTime_;

   if (outputLevel_ >= 2) printStatistics(mli);

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGCR::setup ends.");
#endif
   return (level+1);
}

/* ********************************************************************* *
 * set diagnostics output level
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::setOutputLevel( int level )
{
   outputLevel_ = level;
   return 0;
}

/* ********************************************************************* *
 * set number of levels
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::setNumLevels( int nlevels )
{
   if ( nlevels < maxLevels_ && nlevels > 0 ) numLevels_ = nlevels;
   return 0;
}

/* ********************************************************************* *
 * select independent set
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::selectIndepSet(MLI_Matrix *mli_Amat, int **indepSet)
{
   int    irow, localNRows, numColsOffd, graphArraySize;
   int    *graphArray, *graphArrayOffd, *ISMarker, *ISMarkerOffd=NULL;
   int    nprocs, *ADiagI, *ADiagJ;
   double *measureArray;
   nalu_hypre_ParCSRMatrix *hypreA, *hypreS;
   nalu_hypre_CSRMatrix    *ADiag, *AOffd, *SExt=NULL;
   MPI_Comm comm;

   hypreA = (nalu_hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   ADiag = nalu_hypre_ParCSRMatrixDiag(hypreA);
   ADiagI = nalu_hypre_CSRMatrixI(ADiag);
   ADiagJ = nalu_hypre_CSRMatrixJ(ADiag);
   AOffd = nalu_hypre_ParCSRMatrixOffd(hypreA);
   localNRows = nalu_hypre_CSRMatrixNumRows(ADiag);
   numColsOffd = nalu_hypre_CSRMatrixNumCols(AOffd);
   comm = getComm();
   MPI_Comm_size(comm, &nprocs);

   measureArray = new double[localNRows+numColsOffd];
   for (irow = 0; irow < localNRows+numColsOffd; irow++)
      measureArray[irow] = 0;
   for (irow = 0; irow < ADiagI[localNRows]; irow++)
      measureArray[ADiagJ[irow]] += 1;

   nalu_hypre_BoomerAMGCreateS(hypreA, 0.0e0, 0.0e0, 1, NULL, &hypreS);
   nalu_hypre_BoomerAMGIndepSetInit(hypreS, measureArray, 0);

   graphArraySize = localNRows;
   graphArray = new int[localNRows];
   for (irow = 0; irow < localNRows; irow++) graphArray[irow] = irow;

   if (numColsOffd) graphArrayOffd = new int[numColsOffd];
   else             graphArrayOffd = NULL;
   for (irow = 0; irow < numColsOffd; irow++) graphArrayOffd[irow] = irow;

   ISMarker = new int[localNRows];
   for (irow = 0; irow < localNRows; irow++) ISMarker[irow] = 0;
   if (numColsOffd)
   {
      ISMarkerOffd = new int[numColsOffd];
      for (irow = 0; irow < numColsOffd; irow++) ISMarkerOffd[irow] = 0;
   }
   if (nprocs > 1) SExt = nalu_hypre_ParCSRMatrixExtractBExt(hypreA,hypreA,0);

   nalu_hypre_BoomerAMGIndepSet(hypreS, measureArray, graphArray,
                           graphArraySize, graphArrayOffd, numColsOffd,
                           ISMarker, ISMarkerOffd);

   delete [] measureArray;
   delete [] graphArray;
   if (numColsOffd > 0) delete [] graphArrayOffd;
   if (nprocs > 1) nalu_hypre_CSRMatrixDestroy(SExt);
   nalu_hypre_ParCSRMatrixDestroy(hypreS);
   if (numColsOffd > 0) delete [] ISMarkerOffd;
   (*indepSet) = ISMarker;
   return 0;
}

/* ********************************************************************* *
 * perform compatible relaxation
 * --------------------------------------------------------------------- */

MLI_Matrix *MLI_Method_AMGCR::performCR(MLI_Matrix *mli_Amat, int *indepSet,
                                        MLI_Matrix **AfcMat)
{
   int    nprocs, mypid, localNRows, iT, numFpts, irow, *reduceArray1;
   int    *reduceArray2, iP, FStartRow, FNRows, ierr, *rowLengs;
   int    startRow, rowIndex, colIndex, rowCount;
   int    one=1, *ADiagI, *ADiagJ, *sortIndices, *fList;
   int    idata, fPt, iV, ranSeed, jcol, rowCount2, CStartRow, CNRows;
   int    newCount, it;
#if 0
   double relaxWts[5];
#endif
   double colValue, rnorm0, rnorm1, dOne=1.0;
   double *XaccData, ddata, threshold, *XData, arnorm0, arnorm1;
   double aratio, ratio1, ratio2, *ADiagA;
   char   paramString[200];
   NALU_HYPRE_IJMatrix     IJPFF, IJPFC;
   nalu_hypre_ParCSRMatrix *hypreA, *hypreAff, *hyprePFC, *hypreAffT;
   nalu_hypre_ParCSRMatrix *hypreAfc, *hyprePFF, *hyprePFFT, *hypreAPFC;
   nalu_hypre_CSRMatrix    *ADiag;
   NALU_HYPRE_IJVector     IJB, IJX, IJXacc;
   nalu_hypre_ParVector    *hypreB, *hypreX, *hypreXacc;
   MLI_Matrix *mli_PFFMat, *mli_AffMat, *mli_AfcMat, *mli_AffTMat;
   MLI_Vector *mli_Xvec, *mli_Bvec;
#if 0
   MLI_Solver *smootherPtr;
#endif
   MPI_Comm   comm;
   NALU_HYPRE_Solver hypreSolver;

   /* ------------------------------------------------------ */
   /* get matrix and machine information                     */
   /* ------------------------------------------------------ */

   comm = getComm();
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &mypid);
   hypreA = (nalu_hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   ADiag = nalu_hypre_ParCSRMatrixDiag(hypreA);
   ADiagI = nalu_hypre_CSRMatrixI(ADiag);
   ADiagJ = nalu_hypre_CSRMatrixJ(ADiag);
   ADiagA = nalu_hypre_CSRMatrixData(ADiag);
   localNRows = nalu_hypre_CSRMatrixNumRows(ADiag);
   startRow = nalu_hypre_ParCSRMatrixFirstRowIndex(hypreA);
   fList = new int[localNRows];

   /* ------------------------------------------------------ */
   /* loop over number of trials                             */
   /* ------------------------------------------------------ */

   printf("\tPerform compatible relaxation\n");
   arnorm1 = arnorm0 = 1;
   for (iT = 0; iT < numTrials_; iT++)
   {
      /* --------------------------------------------------- */
      /* get Aff and Afc matrices (get dimension)            */
      /* --------------------------------------------------- */

      numFpts = 0;
      for (irow = 0; irow < localNRows; irow++)
         if (indepSet[irow] != 1) fList[numFpts++] = irow;
      printf("\tTrial %3d (%3d) : number of F-points = %d\n", iT,
             numTrials_, numFpts);
      reduceArray1 = new int[nprocs+1];
      reduceArray2 = new int[nprocs+1];
      for (iP = 0; iP < nprocs; iP++) reduceArray1[iP] = 0;
      reduceArray1[mypid] = numFpts;
      MPI_Allreduce(reduceArray1,reduceArray2,nprocs,MPI_INT,MPI_SUM,comm);
      for (iP = nprocs-1; iP >= 0; iP--)
         reduceArray2[iP+1] = reduceArray2[iP];
      reduceArray2[0] = 0;
      for (iP = 2; iP <= nprocs; iP++) reduceArray2[iP] += reduceArray2[iP-1];
      FStartRow = reduceArray2[mypid];
      FNRows = reduceArray2[mypid+1] - FStartRow;
      delete [] reduceArray1;
      delete [] reduceArray2;
      CStartRow = startRow - FStartRow;
      CNRows = localNRows - FNRows;

      /* --------------------------------------------------- */
      /* get Aff and Afc matrices (create permute matrices)  */
      /* --------------------------------------------------- */

      ierr = NALU_HYPRE_IJMatrixCreate(comm,startRow,startRow+localNRows-1,
                           FStartRow,FStartRow+FNRows-1,&IJPFF);
      ierr = NALU_HYPRE_IJMatrixSetObjectType(IJPFF, NALU_HYPRE_PARCSR);
      nalu_hypre_assert(!ierr);
      rowLengs = new int[localNRows];
      for (irow = 0; irow < localNRows; irow++) rowLengs[irow] = 1;
      ierr = NALU_HYPRE_IJMatrixSetRowSizes(IJPFF, rowLengs);
      ierr = NALU_HYPRE_IJMatrixInitialize(IJPFF);
      nalu_hypre_assert(!ierr);

      ierr = NALU_HYPRE_IJMatrixCreate(comm,startRow,startRow+localNRows-1,
                   CStartRow,CStartRow+CNRows-1, &IJPFC);
      ierr = NALU_HYPRE_IJMatrixSetObjectType(IJPFC, NALU_HYPRE_PARCSR);
      nalu_hypre_assert(!ierr);
      ierr = NALU_HYPRE_IJMatrixSetRowSizes(IJPFC, rowLengs);
      ierr = NALU_HYPRE_IJMatrixInitialize(IJPFC);
      nalu_hypre_assert(!ierr);
      delete [] rowLengs;

      /* --------------------------------------------------- */
      /* get Aff and Afc matrices (load permute matrices)    */
      /* --------------------------------------------------- */

      colValue = 1.0;
      rowCount = rowCount2 = 0;
      for (irow = 0; irow < localNRows; irow++)
      {
         rowIndex = startRow + irow;
         if (indepSet[irow] == 0)
         {
            colIndex = FStartRow + rowCount;
            NALU_HYPRE_IJMatrixSetValues(IJPFF,1,&one,(const int *) &rowIndex,
                    (const int *) &colIndex, (const double *) &colValue);
            rowCount++;
         }
         else
         {
            colIndex = CStartRow + rowCount2;
            NALU_HYPRE_IJMatrixSetValues(IJPFC,1,&one,(const int *) &rowIndex,
                    (const int *) &colIndex, (const double *) &colValue);
            rowCount2++;
         }
      }
      ierr = NALU_HYPRE_IJMatrixAssemble(IJPFF);
      nalu_hypre_assert( !ierr );
      NALU_HYPRE_IJMatrixGetObject(IJPFF, (void **) &hyprePFF);
      //nalu_hypre_MatvecCommPkgCreate((nalu_hypre_ParCSRMatrix *) hyprePFF);
      sprintf(paramString, "NALU_HYPRE_ParCSR" );
      mli_PFFMat = new MLI_Matrix((void *)hyprePFF,paramString,NULL);

      ierr = NALU_HYPRE_IJMatrixAssemble(IJPFC);
      nalu_hypre_assert( !ierr );
      NALU_HYPRE_IJMatrixGetObject(IJPFC, (void **) &hyprePFC);
      //nalu_hypre_MatvecCommPkgCreate((nalu_hypre_ParCSRMatrix *) hyprePFC);
      hypreAPFC = nalu_hypre_ParMatmul(hypreA, hyprePFC);
      nalu_hypre_ParCSRMatrixTranspose(hyprePFF, &hyprePFFT, 1);
      hypreAfc = nalu_hypre_ParMatmul(hyprePFFT, hypreAPFC);

      sprintf(paramString, "NALU_HYPRE_ParCSR" );
      mli_AfcMat = new MLI_Matrix((void *)hypreAfc,paramString,NULL);

      MLI_Matrix_ComputePtAP(mli_PFFMat, mli_Amat, &mli_AffMat);
      hypreAff  = (nalu_hypre_ParCSRMatrix *) mli_AffMat->getMatrix();

      if (arnorm1/arnorm0 < targetMu_) break;

#define HAVE_TRANS
#ifdef HAVE_TRANS
      MLI_Matrix_Transpose(mli_AffMat, &mli_AffTMat);
      hypreAffT = (nalu_hypre_ParCSRMatrix *) mli_AffTMat->getMatrix();
#endif
      NALU_HYPRE_IJVectorCreate(comm,FStartRow, FStartRow+FNRows-1,&IJX);
      NALU_HYPRE_IJVectorSetObjectType(IJX, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(IJX);
      NALU_HYPRE_IJVectorAssemble(IJX);
      NALU_HYPRE_IJVectorGetObject(IJX, (void **) &hypreX);
      sprintf(paramString, "NALU_HYPRE_ParVector" );
      mli_Xvec = new MLI_Vector((void *)hypreX,paramString,NULL);

      NALU_HYPRE_IJVectorCreate(comm,FStartRow, FStartRow+FNRows-1,&IJXacc);
      NALU_HYPRE_IJVectorSetObjectType(IJXacc, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(IJXacc);
      NALU_HYPRE_IJVectorAssemble(IJXacc);
      NALU_HYPRE_IJVectorGetObject(IJXacc, (void **) &hypreXacc);
      nalu_hypre_ParVectorSetConstantValues(hypreXacc, 0.0);

      NALU_HYPRE_IJVectorCreate(comm,FStartRow, FStartRow+FNRows-1,&IJB);
      NALU_HYPRE_IJVectorSetObjectType(IJB, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(IJB);
      NALU_HYPRE_IJVectorAssemble(IJB);
      NALU_HYPRE_IJVectorGetObject(IJB, (void **) &hypreB);
      nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);
      sprintf(paramString, "NALU_HYPRE_ParVector" );
      mli_Bvec = new MLI_Vector((void *)hypreB,paramString,NULL);

#if 0
      /* --------------------------------------------------- */
      /* set up Jacobi smoother with 4 sweeps and weight=1   */
      /* --------------------------------------------------- */

      strcpy(paramString, "Jacobi");
      smootherPtr = MLI_Solver_CreateFromName(paramString);
      targc = 2;
      numSweeps = 1;
      targv[0] = (char *) &numSweeps;
      for (i = 0; i < 5; i++) relaxWts[i] = 1.0;
      targv[1] = (char *) relaxWts;
      strcpy(paramString, "relaxWeight");
      smootherPtr->setParams(paramString, targc, targv);
      maxEigen = 1.0;
      targc = 1;
      targv[0] = (char *) &maxEigen;
      strcpy(paramString, "setMaxEigen");
      smootherPtr->setParams(paramString, targc, targv);
      smootherPtr->setup(mli_AffMat);

      /* --------------------------------------------------- */
      /* relaxation                                          */
      /* --------------------------------------------------- */

      targc = 2;
      targv[0] = (char *) &numSweeps;
      targv[1] = (char *) relaxWts;
      strcpy(paramString, "relaxWeight");
      arnorm0 = 1.0;
      arnorm1 = 0.0;
      aratio = 0.0;
      XData = (double *) nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(hypreX));
      for (iV = 0; iV < numVectors_; iV++)
      {
         ranSeed = 9001 * 7901 * iV *iV * iV * iV + iV * iV * iV + 101;
         NALU_HYPRE_ParVectorSetRandomValues((NALU_HYPRE_ParVector) hypreX,ranSeed);
         for (irow = 0; irow < FNRows; irow++)
            XData[irow] = 0.5 * XData[irow] + 0.5;
         nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm0 = sqrt(nalu_hypre_ParVectorInnerProd(hypreB, hypreB));
         nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);

         numSweeps = 5;
         smootherPtr->setParams(paramString, targc, targv);
         smootherPtr->solve(mli_Bvec, mli_Xvec);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(nalu_hypre_ParVectorInnerProd(hypreB, hypreB));

         rnorm0 = rnorm1;
         nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);
         numSweeps = 1;
         smootherPtr->setParams(paramString, targc, targv);
         smootherPtr->solve(mli_Bvec, mli_Xvec);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(nalu_hypre_ParVectorInnerProd(hypreB, hypreB));

         nalu_hypre_ParVectorAxpy(dOne, hypreX, hypreXacc);
         printf("\tTrial %3d : Jacobi norms = %16.8e %16.8e\n",iT,
                rnorm0,rnorm1);
         if (iV == 0) arnorm0 = rnorm0;
         else         arnorm0 += rnorm0;
         arnorm1 += rnorm1;
         if (rnorm0 < 1.0e-10) rnorm0 = 1.0;
         ratio1 = ratio2 = rnorm1 / rnorm0;
         aratio += ratio1;
         if (ratio1 < targetMu_) break;
      }
      delete smootherPtr;
#else
      MLI_Utils_mJacobiCreate(comm, &hypreSolver);
      MLI_Utils_mJacobiSetParams(hypreSolver, PDegree_);
      XData = (double *) nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(hypreX));
      aratio = 0.0;
      for (iV = 0; iV < numVectors_; iV++)
      {
         ranSeed = 9001 * 7901 * iV *iV * iV * iV + iV * iV * iV + 101;

         /* ------------------------------------------------------- */
         /* CR with A                                               */
         /* ------------------------------------------------------- */

         NALU_HYPRE_ParVectorSetRandomValues((NALU_HYPRE_ParVector) hypreX,ranSeed);
         for (irow = 0; irow < FNRows; irow++)
            XData[irow] = 0.5 * XData[irow] + 0.5;
         nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm0 = sqrt(nalu_hypre_ParVectorInnerProd(hypreB, hypreB));

         nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);
         strcpy(paramString, "pJacobi");
         MLI_Utils_HypreGMRESSolve(hypreSolver, (NALU_HYPRE_Matrix) hypreAff,
                     (NALU_HYPRE_Vector) hypreB, (NALU_HYPRE_Vector) hypreX, paramString);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(nalu_hypre_ParVectorInnerProd(hypreB, hypreB));
         if (rnorm1 < rnorm0 * 1.0e-10 || rnorm1 < 1.0e-10)
         {
            printf("\tperformCR : rnorm0, rnorm1 = %e %e\n",rnorm0,rnorm1);
            break;
         }
         rnorm0 = rnorm1;

         nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);
         strcpy(paramString, "mJacobi");
         MLI_Utils_HypreGMRESSolve(hypreSolver, (NALU_HYPRE_Matrix) hypreAff,
                     (NALU_HYPRE_Vector) hypreB, (NALU_HYPRE_Vector) hypreX, paramString);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(nalu_hypre_ParVectorInnerProd(hypreB, hypreB));
         rnorm1 = 0.2 * log10(rnorm1/rnorm0);
         rnorm1 = pow(1.0e1, rnorm1);
         ratio1 = rnorm1;

         /* ------------------------------------------------------- */
         /* CR with A^T                                             */
         /* ------------------------------------------------------- */

#ifdef HAVE_TRANS
         NALU_HYPRE_ParVectorSetRandomValues((NALU_HYPRE_ParVector) hypreX,ranSeed);
         for (irow = 0; irow < FNRows; irow++)
            XData[irow] = 0.5 * XData[irow] + 0.5;
         nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm0 = sqrt(nalu_hypre_ParVectorInnerProd(hypreB, hypreB));

         nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);
         strcpy(paramString, "pJacobi");
         MLI_Utils_HypreGMRESSolve(hypreSolver, (NALU_HYPRE_Matrix) hypreAffT,
                     (NALU_HYPRE_Vector) hypreB, (NALU_HYPRE_Vector) hypreX, paramString);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, hypreAffT, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(nalu_hypre_ParVectorInnerProd(hypreB, hypreB));
         if (rnorm1 < rnorm0 * 1.0e-10 || rnorm1 < 1.0e-10) break;
         rnorm0 = rnorm1;

         nalu_hypre_ParVectorSetConstantValues(hypreB, 0.0);
         strcpy(paramString, "mJacobi");
         MLI_Utils_HypreGMRESSolve(hypreSolver, (NALU_HYPRE_Matrix) hypreAffT,
                     (NALU_HYPRE_Vector) hypreB, (NALU_HYPRE_Vector) hypreX, paramString);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, hypreAffT, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(nalu_hypre_ParVectorInnerProd(hypreB, hypreB));
         rnorm1 = 0.2 * log10(rnorm1/rnorm0);
         ratio2 = pow(1.0e1, rnorm1);
         if (ratio1 > ratio2) aratio += ratio1;
         else                 aratio += ratio2;
#else
         aratio += ratio1;
         ratio2 = 0;
#endif

         /* ------------------------------------------------------- */
         /* accumulate error vector                                 */
         /* ------------------------------------------------------- */

         nalu_hypre_ParVectorAxpy(dOne, hypreX, hypreXacc);
         if (ratio1 < targetMu_ && ratio2 < targetMu_)
         {
            printf("\tTrial %3d(%3d) : GMRES norms ratios = %16.8e %16.8e ##\n",
                   iT, iV, ratio1, ratio2);
            break;
         }
         else
            printf("\tTrial %3d(%3d) : GMRES norms ratios = %16.8e %16.8e\n",
                   iT, iV, ratio1, ratio2);
      }
      MLI_Utils_mJacobiDestroy(hypreSolver);
#endif

      /* --------------------------------------------------- */
      /* select coarse points                                */
      /* --------------------------------------------------- */

      if (iV == numVectors_) aratio /= (double) numVectors_;
printf("aratio = %e\n", aratio);
      if ((aratio >= targetMu_ || (iT == 0 && localNRows == FNRows)) &&
           iT < (numTrials_-1))
      {
         XaccData = (double *)
                 nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(hypreXacc));
         sortIndices = new int[FNRows];
         for (irow = 0; irow < FNRows; irow++) sortIndices[irow] = irow;
         for (irow = 0; irow < FNRows; irow++)
            if (XaccData[irow] < 0.0) XaccData[irow] = - XaccData[irow];
         //MLI_Utils_DbleQSort2a(XaccData, sortIndices, 0, FNRows-1);
         if (FNRows > 0) threshold = XaccData[FNRows-1] * cutThreshold_;
#if 0
         newCount = 0;
         for (ic = 0; ic < localNRows; ic++)
         {
            threshold = XaccData[FNRows-1] * cutThreshold_;
            for (it = 0; it < 6; it++)
            {
               for (irow = FNRows-1; irow >= 0; irow--)
               {
                  ddata = XaccData[irow];
                  if (ddata > threshold)
                  {
                     idata = sortIndices[irow];
                     fPt = fList[idata];
                     if (indepSet[fPt] == 0)
                     {
                        count = 0;
                        for (jcol = ADiagI[fPt]; jcol < ADiagI[fPt+1]; jcol++)
                           if (indepSet[ADiagJ[jcol]] == 1) count++;
                        if (count <= ic)
                        {
                           newCount++;
                           indepSet[fPt] = 1;
                           for (jcol = ADiagI[fPt];jcol < ADiagI[fPt+1];jcol++)
                              if (indepSet[ADiagJ[jcol]] == 0)
                                 indepSet[ADiagJ[jcol]] = -1;
                        }
                     }
                  }
               }
               threshold *= 0.1;
               for (irow = 0; irow < localNRows; irow++)
                  if (indepSet[irow] < 0) indepSet[irow] = 0;
               if ((localNRows+newCount-FNRows) > (localNRows/2) && ic > 2)
               {
                  if (((double) newCount/ (double) localNRows) > 0.05)
                     break;
               }
            }
            if ((localNRows+newCount-FNRows) > (localNRows/2) && ic > 2)
            {
               if (((double) newCount/ (double) localNRows) > 0.05)
                  break;
            }
         }
#else
         newCount = 0;
         threshold = XaccData[FNRows-1] * cutThreshold_;
         for (it = 0; it < 1; it++)
         {
            for (irow = FNRows-1; irow >= 0; irow--)
            {
               ddata = XaccData[irow];
               if (ddata > threshold)
               {
                  idata = sortIndices[irow];
                  fPt = fList[idata];
                  if (indepSet[fPt] == 0)
                  {
                     newCount++;
                     indepSet[fPt] = 1;
                     for (jcol = ADiagI[fPt];jcol < ADiagI[fPt+1];jcol++)
                        if (indepSet[ADiagJ[jcol]] == 0 &&
                            habs(ADiagA[jcol]/ADiagA[ADiagI[fPt]]) > 1.0e-12)
                           indepSet[ADiagJ[jcol]] = -1;
                  }
               }
            }
            threshold *= 0.1;
            for (irow = 0; irow < localNRows; irow++)
               if (indepSet[irow] < 0) indepSet[irow] = 0;
            if ((localNRows+newCount-FNRows) > (localNRows/2))
            {
               if (((double) newCount/ (double) localNRows) > 0.1)
                  break;
            }
         }
#endif
         delete [] sortIndices;
         if (newCount == 0)
         {
            printf("CR stops because newCount = 0\n");
            break;
         }
      }

      /* --------------------------------------------------- */
      /* clean up                                            */
      /* --------------------------------------------------- */

      NALU_HYPRE_IJMatrixDestroy(IJPFF);
      nalu_hypre_ParCSRMatrixDestroy(hyprePFFT);
      nalu_hypre_ParCSRMatrixDestroy(hypreAPFC);
#ifdef HAVE_TRANS
      delete mli_AffTMat;
#endif
      NALU_HYPRE_IJMatrixDestroy(IJPFC);
      NALU_HYPRE_IJVectorDestroy(IJX);
      NALU_HYPRE_IJVectorDestroy(IJB);
      NALU_HYPRE_IJVectorDestroy(IJXacc);
      delete mli_Bvec;
      delete mli_Xvec;
      if (aratio < targetMu_ && iT != 0) break;
      if (numTrials_ == 1) break;
      delete mli_AffMat;
      delete mli_AfcMat;
      nalu_hypre_ParCSRMatrixDestroy(hypreAfc);
   }

   /* ------------------------------------------------------ */
   /* final clean up                                         */
   /* ------------------------------------------------------ */

   delete [] fList;
   (*AfcMat) = mli_AfcMat;
   return mli_AffMat;
}

/* ********************************************************************* *
 * create the prolongation matrix
 * --------------------------------------------------------------------- */

MLI_Matrix *MLI_Method_AMGCR::createPmat(int *indepSet, MLI_Matrix *mli_Amat,
                               MLI_Matrix *mli_Affmat, MLI_Matrix *mli_Afcmat)
{
   int    *ADiagI, *ADiagJ, localNRows, AffNRows, AffStartRow, irow;
   int    *rowLengs, ierr, startRow, rowCount, rowIndex, colIndex;
   int    *colInd, rowSize, jcol, one=1, maxRowLeng, nnz;
   int    *tPDiagI, *tPDiagJ, cCount, fCount, ncount, *ADDiagI, *ADDiagJ;
   int    *AD2DiagI, *AD2DiagJ, *newColInd, newRowSize;
   int    nprocs, AccStartRow, AccNRows;
   double *ADiagA, *colVal, colValue, *newColVal, *DDiagA;
   double *tPDiagA, *ADDiagA, *AD2DiagA, omega=1, dtemp;
   char   paramString[100];
   NALU_HYPRE_IJMatrix     IJInvD, IJP;
   nalu_hypre_ParCSRMatrix *hypreA, *hypreAff, *hypreInvD, *hypreP=NULL, *hypreAD;
   nalu_hypre_ParCSRMatrix *hypreAD2, *hypreAfc, *hypreTmp;
   nalu_hypre_CSRMatrix    *ADiag, *DDiag, *tPDiag, *ADDiag, *AD2Diag;
   MLI_Function       *funcPtr;
   MLI_Matrix         *mli_Pmat;
   MPI_Comm           comm;
   NALU_HYPRE_Solver       ps;

   /* ------------------------------------------------------ */
   /* get matrix information                                 */
   /* ------------------------------------------------------ */

   comm = getComm();
   MPI_Comm_size(comm, &nprocs);
   hypreA = (nalu_hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   startRow = nalu_hypre_ParCSRMatrixFirstRowIndex(hypreA);
   localNRows = nalu_hypre_ParCSRMatrixNumRows(hypreA);

   hypreAff = (nalu_hypre_ParCSRMatrix *) mli_Affmat->getMatrix();
   AffStartRow = nalu_hypre_ParCSRMatrixFirstRowIndex(hypreAff);
   AffNRows = nalu_hypre_ParCSRMatrixNumRows(hypreAff);

   /* ------------------------------------------------------ */
   /* create the diagonal matrix of A                        */
   /* ------------------------------------------------------ */

   ierr = NALU_HYPRE_IJMatrixCreate(comm,AffStartRow,AffStartRow+AffNRows-1,
                           AffStartRow,AffStartRow+AffNRows-1,&IJInvD);
   ierr = NALU_HYPRE_IJMatrixSetObjectType(IJInvD, NALU_HYPRE_PARCSR);
   nalu_hypre_assert(!ierr);
   rowLengs = new int[AffNRows];
   for (irow = 0; irow < AffNRows; irow++) rowLengs[irow] = 1;
   ierr = NALU_HYPRE_IJMatrixSetRowSizes(IJInvD, rowLengs);
   ierr = NALU_HYPRE_IJMatrixInitialize(IJInvD);
   nalu_hypre_assert(!ierr);
   delete [] rowLengs;

   /* ------------------------------------------------------ */
   /* load the diagonal matrix of A                          */
   /* ------------------------------------------------------ */

   rowCount = 0;
   for (irow = 0; irow < localNRows; irow++)
   {
      rowIndex = startRow + irow;
      if (indepSet[irow] == 0)
      {
         NALU_HYPRE_ParCSRMatrixGetRow((NALU_HYPRE_ParCSRMatrix) hypreA, rowIndex,
                                  &rowSize, &colInd, &colVal);
         colValue = 1.0;
         for (jcol = 0; jcol < rowSize; jcol++)
         {
            if (colInd[jcol] == rowIndex)
            {
               colValue = colVal[jcol];
               break;
            }
         }
         if (colValue >= 0.0)
         {
            for (jcol = 0; jcol < rowSize; jcol++)
               if (colInd[jcol] != rowIndex &&
                   (indepSet[colInd[jcol]-startRow] == 0) &&
                   colVal[jcol] > 0.0)
                  colValue += colVal[jcol];
         }
         else
         {
            for (jcol = 0; jcol < rowSize; jcol++)
               if (colInd[jcol] != rowIndex &&
                   (indepSet[colInd[jcol]-startRow] == 0) &&
                   colVal[jcol] < 0.0)
                  colValue += colVal[jcol];
         }
         colValue = 1.0 / colValue;
         colIndex = AffStartRow + rowCount;
         NALU_HYPRE_IJMatrixSetValues(IJInvD,1,&one,(const int *) &colIndex,
                    (const int *) &colIndex, (const double *) &colValue);
         rowCount++;
         NALU_HYPRE_ParCSRMatrixRestoreRow((NALU_HYPRE_ParCSRMatrix) hypreA, rowIndex,
                                      &rowSize, &colInd, &colVal);
      }
   }

   /* ------------------------------------------------------ */
   /* finally assemble the diagonal matrix of A              */
   /* ------------------------------------------------------ */

   ierr = NALU_HYPRE_IJMatrixAssemble(IJInvD);
   nalu_hypre_assert( !ierr );
   NALU_HYPRE_IJMatrixGetObject(IJInvD, (void **) &hypreInvD);
   ierr += NALU_HYPRE_IJMatrixSetObjectType(IJInvD, -1);
   ierr += NALU_HYPRE_IJMatrixDestroy(IJInvD);
   nalu_hypre_assert( !ierr );

   /* ------------------------------------------------------ */
   /* generate polynomial of Aff and invD                    */
   /* ------------------------------------------------------ */

   if (PDegree_ == 0)
   {
      hypreP = hypreInvD;
      hypreInvD = NULL;
      ADiag  = nalu_hypre_ParCSRMatrixDiag(hypreP);
      ADiagI = nalu_hypre_CSRMatrixI(ADiag);
      ADiagJ = nalu_hypre_CSRMatrixJ(ADiag);
      ADiagA = nalu_hypre_CSRMatrixData(ADiag);
      for (irow = 0; irow < AffNRows; irow++)
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            ADiagA[jcol] = - ADiagA[jcol];
   }
   else if (PDegree_ == 1)
   {
#if 1
      hypreP = nalu_hypre_ParMatmul(hypreAff, hypreInvD);
      DDiag  = nalu_hypre_ParCSRMatrixDiag(hypreInvD);
      DDiagA = nalu_hypre_CSRMatrixData(DDiag);
      ADiag  = nalu_hypre_ParCSRMatrixDiag(hypreP);
      ADiagI = nalu_hypre_CSRMatrixI(ADiag);
      ADiagJ = nalu_hypre_CSRMatrixJ(ADiag);
      ADiagA = nalu_hypre_CSRMatrixData(ADiag);
      for (irow = 0; irow < AffNRows; irow++)
      {
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
         {
            if (ADiagJ[jcol] == irow)
                 ADiagA[jcol] = - omega*DDiagA[irow]*(2.0-omega*ADiagA[jcol]);
            else ADiagA[jcol] = omega * omega * DDiagA[irow] * ADiagA[jcol];
         }
      }
#else
      ierr = NALU_HYPRE_IJMatrixCreate(comm,AffStartRow,AffStartRow+AffNRows-1,
                           AffStartRow,AffStartRow+AffNRows-1,&IJP);
      ierr = NALU_HYPRE_IJMatrixSetObjectType(IJP, NALU_HYPRE_PARCSR);
      nalu_hypre_assert(!ierr);
      rowLengs = new int[AffNRows];
      maxRowLeng = 0;
      ADiag   = nalu_hypre_ParCSRMatrixDiag(hypreAff);
      ADiagI  = nalu_hypre_CSRMatrixI(ADiag);
      ADiagJ  = nalu_hypre_CSRMatrixJ(ADiag);
      ADiagA  = nalu_hypre_CSRMatrixData(ADiag);
      for (irow = 0; irow < AffNRows; irow++)
      {
         newRowSize = 1;
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            if (ADiagJ[jcol] == irow) {index = jcol; break;}
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            if (ADiagJ[jcol] != irow && ADiagA[jcol]*ADiagA[index] < 0.0)
               newRowSize++;
         rowLengs[irow] = newRowSize;
         if (newRowSize > maxRowLeng) maxRowLeng = newRowSize;
      }
      ierr = NALU_HYPRE_IJMatrixSetRowSizes(IJP, rowLengs);
      ierr = NALU_HYPRE_IJMatrixInitialize(IJP);
      nalu_hypre_assert(!ierr);
      delete [] rowLengs;
      newColInd = new int[maxRowLeng];
      newColVal = new double[maxRowLeng];
      for (irow = 0; irow < AffNRows; irow++)
      {
         newRowSize = 0;
         index = -1;
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            if (ADiagJ[jcol] == irow) {index = jcol; break;}
         if (index == -1) printf("WARNING : zero diagonal.\n");
         newColInd[0] = AffStartRow + irow;
         newColVal[0] = ADiagA[index];
         newRowSize++;
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
         {
            if (ADiagJ[jcol] != irow && ADiagA[jcol]*ADiagA[index] < 0.0)
            {
               newColInd[newRowSize] = AffStartRow + ADiagJ[jcol];
               newColVal[newRowSize++] = ADiagA[jcol];
            }
            else
            {
               newColVal[0] += ADiagA[jcol];
            }
         }
         for (jcol = 1; jcol < newRowSize; jcol++)
            newColVal[jcol] /= (-newColVal[0]);
         newColVal[0] = 1.0;
         rowIndex = AffStartRow + irow;
         ierr = NALU_HYPRE_IJMatrixSetValues(IJP, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
         nalu_hypre_assert(!ierr);
      }
      delete [] newColInd;
      delete [] newColVal;
      ierr = NALU_HYPRE_IJMatrixAssemble(IJP);
      nalu_hypre_assert( !ierr );
      NALU_HYPRE_IJMatrixGetObject(IJP, (void **) &hypreAD);
      hypreP = nalu_hypre_ParMatmul(hypreAD, hypreInvD);
      ierr += NALU_HYPRE_IJMatrixDestroy(IJP);
#endif
   }
   else if (PDegree_ == 2)
   {
      hypreAD  = nalu_hypre_ParMatmul(hypreAff, hypreInvD);
      hypreAD2 = nalu_hypre_ParMatmul(hypreAD, hypreAD);
      ADDiag   = nalu_hypre_ParCSRMatrixDiag(hypreAD);
      AD2Diag  = nalu_hypre_ParCSRMatrixDiag(hypreAD2);
      ADDiagI  = nalu_hypre_CSRMatrixI(ADDiag);
      ADDiagJ  = nalu_hypre_CSRMatrixJ(ADDiag);
      ADDiagA  = nalu_hypre_CSRMatrixData(ADDiag);
      AD2DiagI = nalu_hypre_CSRMatrixI(AD2Diag);
      AD2DiagJ = nalu_hypre_CSRMatrixJ(AD2Diag);
      AD2DiagA = nalu_hypre_CSRMatrixData(AD2Diag);
      DDiag    = nalu_hypre_ParCSRMatrixDiag(hypreInvD);
      DDiagA   = nalu_hypre_CSRMatrixData(DDiag);
      newColInd = new int[2*AffNRows];
      newColVal = new double[2*AffNRows];
      ierr = NALU_HYPRE_IJMatrixCreate(comm,AffStartRow,AffStartRow+AffNRows-1,
                           AffStartRow,AffStartRow+AffNRows-1,&IJP);
      ierr = NALU_HYPRE_IJMatrixSetObjectType(IJP, NALU_HYPRE_PARCSR);
      nalu_hypre_assert(!ierr);
      rowLengs = new int[AffNRows];
      maxRowLeng = 0;
      for (irow = 0; irow < AffNRows; irow++)
      {
         newRowSize = 0;
         for (jcol = ADDiagI[irow]; jcol < ADDiagI[irow+1]; jcol++)
            newColInd[newRowSize] = ADDiagJ[jcol];
         for (jcol = AD2DiagI[irow]; jcol < AD2DiagI[irow+1]; jcol++)
            newColInd[newRowSize] = AD2DiagJ[jcol];
         if (newRowSize > maxRowLeng) maxRowLeng = newRowSize;
         nalu_hypre_qsort0(newColInd, 0, newRowSize-1);
         ncount = 0;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
         {
            if ( newColInd[jcol] != newColInd[ncount] )
            {
               ncount++;
               newColInd[ncount] = newColInd[jcol];
            }
         }
         newRowSize = ncount + 1;
         rowLengs[irow] = newRowSize;
      }
      ierr = NALU_HYPRE_IJMatrixSetRowSizes(IJP, rowLengs);
      ierr = NALU_HYPRE_IJMatrixInitialize(IJP);
      nalu_hypre_assert(!ierr);
      delete [] rowLengs;
      nnz = 0;
      for (irow = 0; irow < AffNRows; irow++)
      {
         rowIndex = AffStartRow + irow;
         newRowSize = 0;
         for (jcol = ADDiagI[irow]; jcol < ADDiagI[irow+1]; jcol++)
         {
            newColInd[newRowSize] = ADDiagJ[jcol];
            if (ADDiagJ[jcol] == irow)
               newColVal[newRowSize++] = 3.0 * (1.0 - ADDiagA[jcol]);
            else
               newColVal[newRowSize++] = - 3.0 * ADDiagA[jcol];
         }
         for (jcol = AD2DiagI[irow]; jcol < AD2DiagI[irow+1]; jcol++)
         {
            newColInd[newRowSize] = AD2DiagJ[jcol];
            newColVal[newRowSize++] = AD2DiagA[jcol];
         }
         nalu_hypre_qsort1(newColInd, newColVal, 0, newRowSize-1);
         ncount = 0;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
         {
            if ( jcol != ncount && newColInd[jcol] == newColInd[ncount] )
               newColVal[ncount] += newColVal[jcol];
            else if ( newColInd[jcol] != newColInd[ncount] )
            {
               ncount++;
               newColVal[ncount] = newColVal[jcol];
               newColInd[ncount] = newColInd[jcol];
            }
         }
         newRowSize = ncount + 1;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
            newColVal[jcol] = - (DDiagA[irow] * newColVal[jcol]);

         ierr = NALU_HYPRE_IJMatrixSetValues(IJP, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
         nnz += newRowSize;
         nalu_hypre_assert(!ierr);
      }
      delete [] newColInd;
      delete [] newColVal;
      ierr = NALU_HYPRE_IJMatrixAssemble(IJP);
      nalu_hypre_assert( !ierr );
      NALU_HYPRE_IJMatrixGetObject(IJP, (void **) &hypreP);
      ierr += NALU_HYPRE_IJMatrixSetObjectType(IJP, -1);
      ierr += NALU_HYPRE_IJMatrixDestroy(IJP);
      nalu_hypre_assert(!ierr);
      nalu_hypre_ParCSRMatrixDestroy(hypreAD);
      nalu_hypre_ParCSRMatrixDestroy(hypreAD2);
   }
   else if (PDegree_ == 3)
   {
printf("start parasails\n");
      NALU_HYPRE_ParaSailsCreate(comm, &ps);
      NALU_HYPRE_ParaSailsSetParams(ps, 1.0e-2, 2);
      NALU_HYPRE_ParaSailsSetFilter(ps, 1.0e-2);
      NALU_HYPRE_ParaSailsSetSym(ps, 0);
      NALU_HYPRE_ParaSailsSetLogging(ps, 1);
      NALU_HYPRE_ParaSailsSetup(ps, (NALU_HYPRE_ParCSRMatrix) hypreAff, NULL, NULL);
      NALU_HYPRE_ParaSailsBuildIJMatrix(ps, &IJP);
      NALU_HYPRE_IJMatrixGetObject(IJP, (void **) &hypreP);
printf("finish parasails\n");
   }
   if (hypreInvD != NULL) nalu_hypre_ParCSRMatrixDestroy(hypreInvD);

   /* ------------------------------------------------------ */
   /* create the final P matrix (from hypreP)                */
   /* ------------------------------------------------------ */

   hypreAfc = (nalu_hypre_ParCSRMatrix *) mli_Afcmat->getMatrix();
   hypreTmp = nalu_hypre_ParMatmul(hypreP, hypreAfc);
   nalu_hypre_ParCSRMatrixDestroy(hypreP);
   hypreP = hypreTmp;
   tPDiag   = nalu_hypre_ParCSRMatrixDiag(hypreP);
   tPDiagI  = nalu_hypre_CSRMatrixI(tPDiag);
   tPDiagJ  = nalu_hypre_CSRMatrixJ(tPDiag);
   tPDiagA  = nalu_hypre_CSRMatrixData(tPDiag);
   AccStartRow = startRow - AffStartRow;
   AccNRows = localNRows - AffNRows;
   ierr = NALU_HYPRE_IJMatrixCreate(comm,startRow,startRow+localNRows-1,
                        AccStartRow,AccStartRow+AccNRows-1,&IJP);
   ierr = NALU_HYPRE_IJMatrixSetObjectType(IJP, NALU_HYPRE_PARCSR);
   nalu_hypre_assert(!ierr);
   rowLengs = new int[localNRows];
   maxRowLeng = 0;
   ncount = 0;
   for (irow = 0; irow < localNRows; irow++)
   {
      if (indepSet[irow] == 1) rowLengs[irow] = 1;
      else
      {
         rowLengs[irow] = tPDiagI[ncount+1] - tPDiagI[ncount];
         ncount++;
      }
      if (rowLengs[irow] > maxRowLeng) maxRowLeng = rowLengs[irow];
   }
   ierr = NALU_HYPRE_IJMatrixSetRowSizes(IJP, rowLengs);
   ierr = NALU_HYPRE_IJMatrixInitialize(IJP);
   nalu_hypre_assert(!ierr);
   delete [] rowLengs;
   fCount = 0;
   cCount = 0;
   newColInd = new int[maxRowLeng];
   newColVal = new double[maxRowLeng];
   for (irow = 0; irow < localNRows; irow++)
   {
      rowIndex = startRow + irow;
      if (indepSet[irow] == 1)
      {
         newRowSize = 1;
         newColInd[0] = AccStartRow + cCount;
         newColVal[0] = 1.0;
         cCount++;
      }
      else
      {
         newRowSize = 0;
         for (jcol = tPDiagI[fCount]; jcol < tPDiagI[fCount+1]; jcol++)
         {
            newColInd[newRowSize] = tPDiagJ[jcol] + AccStartRow;
            newColVal[newRowSize++] = tPDiagA[jcol];
         }
         fCount++;
      }
// pruning
#if 1
if (irow == 0) printf("pruning and scaling\n");
dtemp = 0.0;
for (jcol = 0; jcol < newRowSize; jcol++)
if (habs(newColVal[jcol]) > dtemp) dtemp = habs(newColVal[jcol]);
dtemp *= 0.25;
ncount = 0;
for (jcol = 0; jcol < newRowSize; jcol++)
if (habs(newColVal[jcol]) > dtemp)
{
newColInd[ncount] = newColInd[jcol];
newColVal[ncount++] = newColVal[jcol];
}
newRowSize = ncount;
#endif
// scaling
#if 0
dtemp = 0.0;
for (jcol = 0; jcol < newRowSize; jcol++)
dtemp += habs(newColVal[jcol]);
dtemp = 1.0 / dtemp;
for (jcol = 0; jcol < newRowSize; jcol++)
newColVal[jcol] *= dtemp;
#endif
      if (PDegree_ == 3)
      {
         for (jcol = 0; jcol < newRowSize; jcol++)
            newColVal[jcol] = - newColVal[jcol];
      }
      ierr = NALU_HYPRE_IJMatrixSetValues(IJP, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      nalu_hypre_assert(!ierr);
   }
   delete [] newColInd;
   delete [] newColVal;
   ierr = NALU_HYPRE_IJMatrixAssemble(IJP);
   nalu_hypre_assert( !ierr );
   nalu_hypre_ParCSRMatrixDestroy(hypreP);
   NALU_HYPRE_IJMatrixGetObject(IJP, (void **) &hypreP);
   ierr += NALU_HYPRE_IJMatrixSetObjectType(IJP, -1);
   ierr += NALU_HYPRE_IJMatrixDestroy(IJP);
   nalu_hypre_assert(!ierr);

   /* ------------------------------------------------------ */
   /* package the P matrix                                   */
   /* ------------------------------------------------------ */

   sprintf(paramString, "NALU_HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   mli_Pmat = new MLI_Matrix((void*) hypreP, paramString, funcPtr);
   delete funcPtr;
   return mli_Pmat;
}

/* ********************************************************************* *
 * create the restriction matrix
 * --------------------------------------------------------------------- */

MLI_Matrix *MLI_Method_AMGCR::createRmat(int *indepSet, MLI_Matrix *mli_Amat,
                                         MLI_Matrix *mli_Affmat)
{
   int      startRow, localNRows, AffStartRow, AffNRows, RStartRow;
   int      RNRows, ierr, *rowLengs, rowCount, rowIndex, colIndex;
   int      one=1, irow;
   double   colValue;
   char     paramString[100];
   MPI_Comm comm;
   NALU_HYPRE_IJMatrix     IJR;
   nalu_hypre_ParCSRMatrix *hypreA, *hypreAff, *hypreR;
   MLI_Function *funcPtr;
   MLI_Matrix   *mli_Rmat;

   /* ------------------------------------------------------ */
   /* get matrix information                                 */
   /* ------------------------------------------------------ */

   comm = getComm();
   hypreA = (nalu_hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   startRow = nalu_hypre_ParCSRMatrixFirstRowIndex(hypreA);
   localNRows = nalu_hypre_ParCSRMatrixNumRows(hypreA);

   hypreAff = (nalu_hypre_ParCSRMatrix *) mli_Affmat->getMatrix();
   AffStartRow = nalu_hypre_ParCSRMatrixFirstRowIndex(hypreAff);
   AffNRows = nalu_hypre_ParCSRMatrixNumRows(hypreAff);

   /* ------------------------------------------------------ */
   /* create a matrix context                                */
   /* ------------------------------------------------------ */

   RStartRow = startRow - AffStartRow;
   RNRows = localNRows - AffNRows;
   ierr = NALU_HYPRE_IJMatrixCreate(comm,RStartRow,RStartRow+RNRows-1,
                           startRow,startRow+localNRows-1,&IJR);
   ierr = NALU_HYPRE_IJMatrixSetObjectType(IJR, NALU_HYPRE_PARCSR);
   nalu_hypre_assert(!ierr);
   rowLengs = new int[RNRows];
   for (irow = 0; irow < RNRows; irow++) rowLengs[irow] = 1;
   ierr = NALU_HYPRE_IJMatrixSetRowSizes(IJR, rowLengs);
   ierr = NALU_HYPRE_IJMatrixInitialize(IJR);
   nalu_hypre_assert(!ierr);
   delete [] rowLengs;

   /* ------------------------------------------------------ */
   /* load the R matrix                                      */
   /* ------------------------------------------------------ */

   rowCount = 0;
   colValue = 1.0;
   for (irow = 0; irow < localNRows; irow++)
   {
      if (indepSet[irow] == 1)
      {
         rowIndex = RStartRow + rowCount;
         colIndex = startRow + irow;
         NALU_HYPRE_IJMatrixSetValues(IJR,1,&one,(const int *) &rowIndex,
                    (const int *) &colIndex, (const double *) &colValue);
         rowCount++;
      }
   }

   /* ------------------------------------------------------ */
   /* assemble the R matrix                                  */
   /* ------------------------------------------------------ */

   ierr = NALU_HYPRE_IJMatrixAssemble(IJR);
   nalu_hypre_assert(!ierr);
   NALU_HYPRE_IJMatrixGetObject(IJR, (void **) &hypreR);
   ierr += NALU_HYPRE_IJMatrixSetObjectType(IJR, -1);
   ierr += NALU_HYPRE_IJMatrixDestroy(IJR);
   nalu_hypre_assert( !ierr );
   sprintf(paramString, "NALU_HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   mli_Rmat = new MLI_Matrix((void*) hypreR, paramString, funcPtr);
   delete funcPtr;
   return mli_Rmat;
}

/* ********************************************************************* *
 * print AMG information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::print()
{
   int      mypid;
   MPI_Comm comm = getComm();

   MPI_Comm_rank( comm, &mypid);
   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      printf("\t*** method name             = %s\n", getName());
      printf("\t*** number of levels        = %d\n", numLevels_);
      printf("\t*** use MIS                 = %d\n", findMIS_);
      printf("\t*** target relaxation rate  = %e\n", targetMu_);
      printf("\t*** truncation threshold    = %e\n", cutThreshold_);
      printf("\t*** number of trials        = %d\n", numTrials_);
      printf("\t*** number of trial vectors = %d\n", numVectors_);
      printf("\t*** polynomial degree       = %d\n", PDegree_);
      printf("\t*** minimum coarse size     = %d\n", minCoarseSize_);
      printf("\t*** smoother type           = %s\n", smoother_);
      printf("\t*** smoother nsweeps        = %d\n", smootherNum_);
      printf("\t*** smoother weight         = %e\n", smootherWgts_[0]);
      printf("\t*** coarse solver type      = %s\n", coarseSolver_);
      printf("\t*** coarse solver nsweeps   = %d\n", coarseSolverNum_);
      printf("\t********************************************************\n");
   }
   return 0;
}

/* ********************************************************************* *
 * print AMG statistics information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::printStatistics(MLI *mli)
{
   int          mypid, level, globalNRows, totNRows, fineNRows;
   int          maxNnz, minNnz, fineNnz, totNnz, thisNnz, itemp;
   double       maxVal, minVal, dtemp;
   char         paramString[100];
   MLI_Matrix   *mli_Amat, *mli_Pmat;
   MPI_Comm     comm = getComm();

   /* --------------------------------------------------------------- */
   /* output header                                                   */
   /* --------------------------------------------------------------- */

   MPI_Comm_rank( comm, &mypid);
   if ( mypid == 0 )
      printf("\t****************** AMGCR Statistics ********************\n");

   /* --------------------------------------------------------------- */
   /* output processing time                                          */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t*** number of levels = %d\n", currLevel_+1);
      printf("\t*** total RAP   time = %e seconds\n", RAPTime_);
      printf("\t*** total GenMG time = %e seconds\n", totalTime_);
      printf("\t******************** Amatrix ***************************\n");
      printf("\t*level   Nrows MaxNnz MinNnz TotalNnz  maxValue  minValue*\n");
   }

   /* --------------------------------------------------------------- */
   /* fine and coarse matrix complexity information                   */
   /* --------------------------------------------------------------- */

   totNnz = totNRows = 0;
   for ( level = 0; level <= currLevel_; level++ )
   {
      mli_Amat = mli->getSystemMatrix( level );
      sprintf(paramString, "nrows");
      mli_Amat->getMatrixInfo(paramString, globalNRows, dtemp);
      sprintf(paramString, "maxnnz");
      mli_Amat->getMatrixInfo(paramString, maxNnz, dtemp);
      sprintf(paramString, "minnnz");
      mli_Amat->getMatrixInfo(paramString, minNnz, dtemp);
      sprintf(paramString, "totnnz");
      mli_Amat->getMatrixInfo(paramString, thisNnz, dtemp);
      sprintf(paramString, "maxval");
      mli_Amat->getMatrixInfo(paramString, itemp, maxVal);
      sprintf(paramString, "minval");
      mli_Amat->getMatrixInfo(paramString, itemp, minVal);
      if ( mypid == 0 )
      {
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
                globalNRows, maxNnz, minNnz, thisNnz, maxVal, minVal);
      }
      if ( level == 0 ) fineNnz = thisNnz;
      totNnz += thisNnz;
      if ( level == 0 ) fineNRows = globalNRows;
      totNRows += globalNRows;
   }

   /* --------------------------------------------------------------- */
   /* prolongation operator complexity information                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t******************** Pmatrix ***************************\n");
      printf("\t*level   Nrows MaxNnz MinNnz TotalNnz  maxValue  minValue*\n");
      fflush(stdout);
   }
   for ( level = 1; level <= currLevel_; level++ )
   {
      mli_Pmat = mli->getProlongation( level );
      sprintf(paramString, "nrows");
      mli_Pmat->getMatrixInfo(paramString, globalNRows, dtemp);
      sprintf(paramString, "maxnnz");
      mli_Pmat->getMatrixInfo(paramString, maxNnz, dtemp);
      sprintf(paramString, "minnnz");
      mli_Pmat->getMatrixInfo(paramString, minNnz, dtemp);
      sprintf(paramString, "totnnz");
      mli_Pmat->getMatrixInfo(paramString, thisNnz, dtemp);
      sprintf(paramString, "maxval");
      mli_Pmat->getMatrixInfo(paramString, itemp, maxVal);
      sprintf(paramString, "minval");
      mli_Pmat->getMatrixInfo(paramString, itemp, minVal);
      if ( mypid == 0 )
      {
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
                globalNRows, maxNnz, minNnz, thisNnz, maxVal, minVal);
      }
   }

   /* --------------------------------------------------------------- */
   /* other complexity information                                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      dtemp = (double) totNnz / (double) fineNnz;
      printf("\t*** Amat complexity  = %e\n", dtemp);
      dtemp = (double) totNRows / (double) fineNRows;
      printf("\t*** grid complexity  = %e\n", dtemp);
      printf("\t********************************************************\n");
      fflush(stdout);
   }
   return 0;
}
