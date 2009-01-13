/*****************************************************************************
*
* Copyright (c) 2000 - 2008, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory
* LLNL-CODE-400142
* All rights reserved.
*
* This file is  part of VisIt. For  details, see https://visit.llnl.gov/.  The
* full copyright notice is contained in the file COPYRIGHT located at the root
* of the VisIt distribution or at http://www.llnl.gov/visit/copyright.html.
*
* Redistribution  and  use  in  source  and  binary  forms,  with  or  without
* modification, are permitted provided that the following conditions are met:
*
*  - Redistributions of  source code must  retain the above  copyright notice,
*    this list of conditions and the disclaimer below.
*  - Redistributions in binary form must reproduce the above copyright notice,
*    this  list of  conditions  and  the  disclaimer (as noted below)  in  the
*    documentation and/or other materials provided with the distribution.
*  - Neither the name of  the LLNS/LLNL nor the names of  its contributors may
*    be used to endorse or promote products derived from this software without
*    specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT  HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR  IMPLIED WARRANTIES, INCLUDING,  BUT NOT  LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND  FITNESS FOR A PARTICULAR  PURPOSE
* ARE  DISCLAIMED. IN  NO EVENT  SHALL LAWRENCE  LIVERMORE NATIONAL  SECURITY,
* LLC, THE  U.S.  DEPARTMENT OF  ENERGY  OR  CONTRIBUTORS BE  LIABLE  FOR  ANY
* DIRECT,  INDIRECT,   INCIDENTAL,   SPECIAL,   EXEMPLARY,  OR   CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT  LIMITED TO, PROCUREMENT OF  SUBSTITUTE GOODS OR
* SERVICES; LOSS OF  USE, DATA, OR PROFITS; OR  BUSINESS INTERRUPTION) HOWEVER
* CAUSED  AND  ON  ANY  THEORY  OF  LIABILITY,  WHETHER  IN  CONTRACT,  STRICT
* LIABILITY, OR TORT  (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY  WAY
* OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
* DAMAGE.
*
*****************************************************************************/

// ************************************************************************* //
//                           avtNek5000FileFormat.h                          //
// ************************************************************************* //

#ifndef AVT_Nek5000_FILE_FORMAT_H
#define AVT_Nek5000_FILE_FORMAT_H

#include <avtMTMDFileFormat.h>

#include <map>
#include <vector>

class     avtIntervalTree;
class     avtIsolevelsSelection;
class     avtPlaneSelection;
class     avtSpatialBoxSelection;


typedef struct
{
    std::string var;
    int         element;
    int         timestep;
} PointerKey;

class KeyCompare {
  public:
    bool operator()(const PointerKey &x, const PointerKey &y) const
        {
            if (x.element != y.element)
                return (x.element > y.element);
            if (x.timestep != y.timestep)
                return (x.timestep > y.timestep);
            return (x.var > y.var);
        }
};

// ****************************************************************************
//  Class: avtNek5000FileFormat
//
//  Purpose:
//      Reads in Nek5000 files as a plugin to VisIt.
//
//  Programmer: dbremer -- generated by xml2avt
//  Creation:   Fri May 18 16:07:09 PST 2007
//
//  Modifications:
//    Dave Bremer, Wed Nov  7 12:27:33 PST 2007
//    This reader previously supported 3D binary Nek files.  Now it also 
//    handles 2D and ascii versions of Nek files.
//
//    Dave Bremer, Wed Nov 14 15:00:13 PST 2007
//    Added support for the parallel version of the file.
//
//    Dave Bremer, Thu Nov 15 16:44:42 PST 2007
//    Small fix for ascii format in case windows-style CRLF is used.
//
//    Dave Bremer, Wed Feb  6 19:12:55 PST 2008
//    Refactored the constructor, moving some functionality into 
//    other methods, and deferring some significant computation.
//
//    Dave Bremer, Wed Apr 23 18:12:50 PDT 2008
//    Implemented GetAuxiliaryData so I can read spatial extents.
//
//    Dave Bremer, Tue May 13 19:51:04 CDT 2008
//    Added PopulateIOInformation to give hints to improve IO patterns,
//    but the info is currently not used.
//
//    Dave Bremer, Fri Jun  6 15:38:45 PDT 2008
//    Added the bParFormat flag allowing the parallel format to be used
//    by a serial code, in which there is only one output dir.
//
//    Dave Bremer, Thu Jun 12 12:59:23 PDT 2008
//    Support varying numbers of blocks per file in the parallel format.
//    The distribution of blocks is assumed to be constant over time.
//
//    Dave Bremer, Mon Aug 11 13:53:18 PDT 2008
//    Added a method to parse field tags in nek binary header files.
//
//    Hank Childs, Sat Nov  8 14:33:30 PST 2008
//    Cache the mesh from time slice to time slice.
//
//    Hank Childs, Thu Dec 18 06:38:51 PST 2008
//    Overhaul plugin to serve up unstructured grids, not many curvilinear
//    grids.  Also rename class to Nek5000, since that is the official code
//    name.
//
//    Hank Childs, Mon Jan 12 13:13:10 CST 2009
//    Add data member for what time slice we've read time info for.
//
// ****************************************************************************

class avtNek5000FileFormat : public avtMTMDFileFormat
{
  public:
                       avtNek5000FileFormat(const char *);
    virtual           ~avtNek5000FileFormat();

    virtual void       RegisterDataSelections(
                               const std::vector<avtDataSelection_p> &selList,
                               std::vector<bool> *selectionsApplied);

    virtual bool       CanCacheVariable(const char *) { return false; };
    virtual void       ActivateTimestep(int);

    virtual void      *GetAuxiliaryData(const char *var, int timestep,
                                        int domain,const char *type,void *args,
                                        DestructorFunction &);

    //
    // If you know the times and cycle numbers, overload this function.
    // Otherwise, VisIt will make up some reasonable ones for you.
    //
    virtual void           GetCycles(std::vector<int> &);
    virtual void           GetTimes(std::vector<double> &);

    virtual int            GetNTimesteps(void);

    virtual const char    *GetType(void)   { return "Nek5000"; }
    virtual void           FreeUpResources(void); 

    virtual vtkDataSet    *GetMesh(int, int, const char *);
    virtual vtkDataArray  *GetVar(int, int, const char *);
    virtual vtkDataArray  *GetVectorVar(int, int, const char *);

  protected:
    // This info is embedded in the .nek3d text file 
    // originally specified by Dave Bremer
    std::string          version;
    std::string          fileTemplate;
    int                  iFirstTimestep;
    int                  iNumTimesteps;
    bool                 bBinary;         //binary or ascii
    int                  iNumOutputDirs;  //used in parallel format
    bool                 bParFormat;

    // This info is embedded in, or derived from, the file header
    bool                 bSwapEndian;
    int                  iNumBlocks;
    int                  iBlockSize[3];
    bool                 bHasVelocity;
    bool                 bHasPressure;
    bool                 bHasTemperature;
    int                  iNumSFields;
    int                  iHeaderSize;
    int                  iDim;
    int                  iPrecision; //4 or 8 for float or double
                                     //only used in parallel binary
    int *                aBlocksPerFile;

    // This info is distributed through all the dumps, and only
    // computed on demand
    std::vector<int>     aCycles;
    std::vector<double>  aTimes;
    std::vector<bool>    readTimeInfoFor;
    std::vector<bool>    iTimestepsWithMesh;
    int                  curTimestep;
    int                  timestepToUseForMesh;

    // Cached data describing how to read data out of the file.
    FILE *fdMesh, *fdVar;
    std::string  curOpenMeshFile;
    std::string  curOpenVarFile;
    int  iCurrTimestep;        //which timestep is associated with fdVar
    int  iCurrMeshProc;        //For parallel format, proc associated with fdMesh
    int  iCurrVarProc;         //For parallel format, proc associated with fdVar  
    int  iAsciiMeshFileStart;  //For ascii data, file pos where data begins, in mesh file
    int  iAsciiCurrFileStart;  //For ascii data, file pos where data begins, in current timestep
    int  iAsciiMeshFileLineLen; //For ascii data, length of each line, in mesh file
    int  iAsciiCurrFileLineLen; //For ascii data, length of each line, in current timestep

    int *aBlockLocs;           //For parallel format, make a table for looking up blocks.
                               //This has 2 ints per block, with proc # and local block #.

    // This info is for managing which blocks are read on which processors
    // and caching blocks that have been read. 
    std::vector<int>                                     myElementList;
    std::map<PointerKey, float *, KeyCompare>            cachedData;
    std::map<int, avtIntervalTree *>                     boundingBoxes;
    std::map<PointerKey, avtIntervalTree *, KeyCompare>  dataExtents;
    int                                                  cachableElementMin;
    int                                                  cachableElementMax;

    virtual void           ParseMetaDataFile(const char *filename);
    virtual void           ParseNekFileHeader();
    virtual void           ParseFieldTags(ifstream &f);
    virtual void           ReadBlockLocations();

    virtual void           PopulateDatabaseMetaData(avtDatabaseMetaData *, int);
    virtual void           UpdateCyclesAndTimes();
    virtual void           GetDomainSizeAndVarOffset(int timestate, const char *var, 
                                                     int &outDomSizeInFloats, 
                                                     int &outVarOffsetInFloats,
                                                     int &outVarOffsetInBytes,
                                                     int &outTimestepHasMesh );
    void                   ByteSwap32(void *aVals, int nVals);
    void                   ByteSwap64(void *aVals, int nVals);
    void                   FindAsciiDataStart(FILE *fd, int &outDataStart, int &outLineLen);
    void                   GetFileName(int timestep, int pardir, char *outFileName, int bufSize);

    float                 *ReadPoints(int, int);
    float                 *ReadVar(int, int, const char *);
    float                 *ReadVelocity(int, int);

    avtIntervalTree       *GetBoundingBoxIntervalTree(int);
    avtIntervalTree       *GetDataExtentsIntervalTree(int, const char *);

    void                   CombineElementLists(const std::vector<std::vector<int> > &,
                                               std::vector<int> &, bool);
};


#endif
