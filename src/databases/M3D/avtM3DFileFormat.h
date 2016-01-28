/*****************************************************************************
*
* Copyright (c) 2000 - 2015, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory
* LLNL-CODE-442911
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
//                            avtM3DFileFormat.h                           //
// ************************************************************************* //

#ifndef AVT_M3D_FILE_FORMAT_H
#define AVT_M3D_FILE_FORMAT_H

// Define this symbol BEFORE including hdf5.h to indicate the HDF5 code
// in this file uses version 1.6 of the HDF5 API. This is harmless for
// versions of HDF5 before 1.8 and ensures correct compilation with
// version 1.8 and thereafter. When, and if, the HDF5 code in this file
// is explicitly upgraded to the 1.8 API, this symbol should be removed.
#define H5_USE_16_API
#include <hdf5.h>
#include <avtMTMDFileFormat.h>

#include <vtkTransform.h>
#include <vector>

// ****************************************************************************
//  Class: VarInfo
//
//  Purpose:
//      Information on variables in the M3D file.
//
//  Programmer: pugmire
//  Creation:   Tue Sep 25 08:49:28 PDT 2007
//
// ****************************************************************************

class VarInfo
{
 public:
        VarInfo()
            { varName = ""; varDim = -1; dataID = -1; planeIdx = -1; }
        VarInfo( std::string &nm, hid_t id, int dim, int idx=-1 )
            { varName = nm; dataID = id; varDim = dim; planeIdx = idx; }

        std::string varName;
        int varDim, planeIdx;
        hid_t dataID;
};

// ****************************************************************************
//  Class: CellInfo
//
//  Purpose:
//      Information on cells in the M3D file.
//
//  Programmer: pugmire
//  Creation:   Tue Sep 25 08:49:28 PDT 2007
//
// ****************************************************************************
class CellInfo
{
 public:
        CellInfo() { id = -1; numCells = -1; }
        CellInfo( int i, int n ) { id = i; numCells = n; }

        int id, numCells;
};


// ****************************************************************************
//  Class: avtM3DFileFormat
//
//  Purpose:
//      Reads in M3D files as a plugin to VisIt.
//
//  Programmer: pugmire -- generated by xml2avt
//  Creation:   Tue Sep 25 08:49:28 PDT 2007
//
//  Modifications:
//    Dave Pugmire, Fri Aug 8 15:22:11 EDT 2008
//    Memory problem caused by CellInfo, VarInfo not being allocated on the heap.
//
// ****************************************************************************

class avtM3DFileFormat : public avtMTMDFileFormat
{
  public:
                       avtM3DFileFormat(const char *);
    virtual           ~avtM3DFileFormat();

    //
    // This is used to return unconvention data -- ranging from material
    // information to information about block connectivity.
    //
    // virtual void      *GetAuxiliaryData(const char *var, int timestep, 
    //                                     int domain, const char *type, void *args, 
    //                                     DestructorFunction &);
    //

    //
    // If you know the times and cycle numbers, overload this function.
    // Otherwise, VisIt will make up some reasonable ones for you.
    //
    // virtual void        GetCycles(std::vector<int> &);
    virtual void           GetTimes(std::vector<double> &);
    virtual int            GetNTimesteps(void);

    virtual const char    *GetType(void)   { return "M3D"; };
    virtual void           FreeUpResources(void); 

    virtual vtkDataSet    *GetMesh(int, int, const char *);
    virtual vtkDataArray  *GetVar(int, int, const char *);
    virtual vtkDataArray  *GetVectorVar(int, int, const char *);

  protected:
        void LoadFile();
        void CalcPlaneAngularSpacing();
        std::string GetPlaneName( const std::string &prefix, int p );
        virtual void PopulateDatabaseMetaData(avtDatabaseMetaData *, int);

        //HDF5 helper functions.
        bool ReadAttribute( hid_t parentID, const char *attr, void *value );
        bool ReadStringAttribute( hid_t parentID, const char *attr, std::string *value );
        hid_t NormalizeH5Type( hid_t type );

        //File member data.
        hid_t m_fileID;
        bool m_vValidFile;
        std::string m_filename;
        std::string m_XPClassStr, m_Plane3DString, m_Plane2DString, m_FullString;
        std::vector<hid_t> m_coordIDs;
        std::vector<double> m_timeSteps;
        std::vector<CellInfo*> m_cellInfo;
        std::vector<VarInfo*> m_scalarVarNames, m_vectorVarNames, m_tensorVarNames;
        std::vector<VarInfo*> m_scalarVars, m_vectorVars, m_tensorVars;
        std::vector<std::string> m_meshes, m_meshesPlane3D, m_meshesPlane2D;
        std::vector<float> m_planeAngles;
        std::vector<vtkTransform *> m_planeXforms;
        bool m_planeAngleClash;
        int m_nCellSets, m_nVars, m_nNodes;
        int m_nPlanes, m_nPlanesPerProc;
        float m_plane0Norm[3];
};


#endif
