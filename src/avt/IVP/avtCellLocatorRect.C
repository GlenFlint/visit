// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

#include <avtCellLocatorRect.h>

#include <vtkRectilinearGrid.h>
#include <vtkDataArray.h>
#include <vtkVoxel.h>
#include <vtkVisItUtility.h>
#include <DebugStream.h>
#include <cassert>
#include <algorithm>
#include <functional> // for 'less'

// -------------------------------------------------------------------------
//  Modifications:
//
//    Hank Childs, Wed Sep  5 16:07:10 PDT 2012
//    Add support for monotonically descending coordinate arrays.
//
// ---------------------------------------------------------------------------

avtCellLocatorRect::avtCellLocatorRect( vtkDataSet* ds ) :
    avtCellLocator( ds )
{
    vtkRectilinearGrid* rg = vtkRectilinearGrid::SafeDownCast( dataSet );

    if( dataSet->GetDataObjectType() != VTK_RECTILINEAR_GRID || rg == NULL )
        EXCEPTION1( ImproperUseException, "avtCellLocatorRect: Dataset is not rectilinear." );

    // copy the coordinate arrays for faster access later
    vtkDataArray* ca[3] = { rg->GetXCoordinates(),
                            rg->GetYCoordinates(),
                            rg->GetZCoordinates() };

    for( unsigned int d=0; d<3; ++d )
    {
        coord[d].resize( ca[d]->GetNumberOfTuples() );
        ascending[d] = true;
        
        for( unsigned int i=0; i<coord[d].size(); ++i )
        {
            coord[d][i] = ca[d]->GetComponent( i, 0 );

            if (i == 1)
            {
              ascending[d] = (coord[d][1] > coord[d][0]);
            }
            else if (i > 1)
            {
                bool thisPairAscending = (coord[d][i] > coord[d][i-1]);

                if (thisPairAscending != ascending[d])
                {
                  std::stringstream msg;
                  
                  msg << "avtCellLocatorRect: Coordinate "
                      << "array " << d << " at index " << i
                      << " is not monotonic: "
                      << coord[d][i] << (ascending[d] ? " <= " :  " >= ")
                      << coord[d][i-1] << ".";
                    
                  EXCEPTION1( ImproperUseException, msg.str());
                }
            }
        }
    }    
}

// ---------------------------------------------------------------------------

avtCellLocatorRect::~avtCellLocatorRect()
{
    Free();
}

// ---------------------------------------------------------------------------

void avtCellLocatorRect::Build()
{
}

// ---------------------------------------------------------------------------

void avtCellLocatorRect::Free()
{
}

// ---------------------------------------------------------------------------
//  Modifications:
//
//    Hank Childs, Wed Sep  5 16:07:10 PDT 2012
//    Add support for monotonically descending coordinate arrays.
//
// ---------------------------------------------------------------------------


vtkIdType
avtCellLocatorRect::FindCell(const double pos[3],
                             avtInterpolationWeights *weights,
                             bool ignoreGhostCells) const
{
#if 0

    vtkRectilinearGrid* rg = (vtkRectilinearGrid*)dataSet;
    int ijk[3];

    if( vtkVisItUtility::ComputeStructuredCoordinates(rg, (double*)pos, ijk) == 0 )
        return -1;

    vtkIdType cell = rg->ComputeCellId( ijk );
    
    if( cell < 0 )
        return -1;

    return TestCell( cell, pos, weights, ignoreGhostCells ) ? cell : -1;

#else

    int    i[3];
    double l[3];

    for( unsigned int d=0; d<3; d++ )
    {
        if( coord[d].size() == 1 )
        {
            // flat grid
            if( pos[d] != coord[d].front() )
                return false;

            i[d] = 0;
            l[d] = 0.0;
        }
        else
        {
            // binary search
            std::vector<double>::const_iterator ci;
            if (ascending[d])
            {
                ci = std::lower_bound( coord[d].begin(), coord[d].end(), 
                                       pos[d], std::less<double>() );
            }
            else
            {
                ci = std::lower_bound( coord[d].begin(), coord[d].end(), 
                                       pos[d], std::greater<double>() );
            }
            
            if( ci == coord[d].end() )
                return -1;
            
            if( ci == coord[d].begin() )
            {
                if( ascending[d] )
                {
                    if (pos[d] < *ci )
                        return -1;
                }
                else
                {
                    if (pos[d] > *ci )
                        return -1;
                }
            }
            else 
                --ci;
            
            i[d] = ci - coord[d].begin(); 
            // This math works whether coord is monotonically increasing or
            // decreasing, since it just calculating a value in [0, 1] for
            // the distance between ci[0] and ci[1].
            l[d] = (pos[d] - ci[0])/(ci[1]-ci[0]);
        }
    }

    vtkIdType cell = (i[2]*(coord[1].size()-1) + i[1])*(coord[0].size()-1) + i[0];

    if( ignoreGhostCells && ghostPtr && ghostPtr[cell] )
        return -1;

    if( weights )
    {
        double k[3] = { 1.0-l[0], 1.0-l[1], 1.0-l[2] };

        vtkIdType base = (i[2]*coord[1].size() + i[1])*coord[0].size() + i[0];

        vtkIdType dx = (coord[0].size() > 1) ? 1 : 0;
        vtkIdType dy = (coord[1].size() > 1) ? coord[0].size() : 0;
        vtkIdType dz = (coord[2].size() > 1) ? coord[1].size()*coord[0].size() : 0;

        weights->resize( 8 );

        (*weights)[0].i = base;
        (*weights)[0].w = k[0]*k[1]*k[2];
     
        (*weights)[1].i = base+dx;
        (*weights)[1].w = l[0]*k[1]*k[2];

        (*weights)[2].i = base+dy;
        (*weights)[2].w = k[0]*l[1]*k[2];

        (*weights)[3].i = base+dx+dy;
        (*weights)[3].w = l[0]*l[1]*k[2];

        (*weights)[4].i = base+dz;
        (*weights)[4].w = k[0]*k[1]*l[2];

        (*weights)[5].i = base+dz+dx;
        (*weights)[5].w = l[0]*k[1]*l[2];

        (*weights)[6].i = base+dz+dy;
        (*weights)[6].w = k[0]*l[1]*l[2];

        (*weights)[7].i = base+dx+dy+dz;
        (*weights)[7].w = l[0]*l[1]*l[2];
    }

    return cell;

#endif
}

// ---------------------------------------------------------------------------
