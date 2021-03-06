// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

// ************************************************************************* //
//  File: avtThreeSliceFilter.h
// ************************************************************************* //

#ifndef AVT_ThreeSlice_FILTER_H
#define AVT_ThreeSlice_FILTER_H

#include <avtPluginDataTreeIterator.h>

#include <ThreeSliceAttributes.h>

class vtkSlicer;


// ****************************************************************************
//  Class: avtThreeSliceFilter
//
//  Purpose:
//      A plugin operator for ThreeSlice.
//
//  Programmer: haddox1 -- generated by xml2info
//  Creation:   Wed Jun 4 10:03:11 PDT 2003
//
// Modifications:
//    David Camp, Thu May 23 12:52:53 PDT 2013
//    Removed the vtkSlicer and vtkAppendPolyData objects from class. They are
//    now created in the execute method. This is done for threading VisIt.
//
//    Eric Brugger, Tue Aug 19 09:26:30 PDT 2014
//    Modified the class to work with avtDataRepresentation.
//
// ****************************************************************************

class avtThreeSliceFilter : public avtPluginDataTreeIterator
{
  public:
                         avtThreeSliceFilter();
    virtual             ~avtThreeSliceFilter();

    static avtFilter    *Create();

    virtual const char  *GetType(void)  { return "avtThreeSliceFilter"; };
    virtual const char  *GetDescription(void)
                             { return "ThreeSlice"; };

    virtual void         SetAtts(const AttributeGroup*);
    virtual bool         Equivalent(const AttributeGroup*);

  protected:
    ThreeSliceAttributes   atts;

    virtual avtDataRepresentation *ExecuteData(avtDataRepresentation *);

    virtual void            UpdateDataObjectInfo(void);    
    virtual void            ReleaseData(void);

    void                    SetPlaneOrientation(double *b, vtkSlicer *slicer);
};


#endif
