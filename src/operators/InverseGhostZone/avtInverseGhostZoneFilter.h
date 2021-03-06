// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

// ************************************************************************* //
//  File: avtInverseGhostZoneFilter.h
// ************************************************************************* //

#ifndef AVT_InverseGhostZone_FILTER_H
#define AVT_InverseGhostZone_FILTER_H

#include <avtPluginDataTreeIterator.h>

#include <InverseGhostZoneAttributes.h>


// ****************************************************************************
//  Class: avtInverseGhostZoneFilter
//
//  Purpose:
//      A plugin operator for InverseGhostZone.
//
//  Programmer: childs -- generated by xml2info
//  Creation:   Thu Jan 8 09:27:11 PDT 2004
//
//  Modifications:
//    Cyrus Harrison, Thu Jul  8 13:08:24 PDT 2010
//    Added ModifyContract.
//
//    Eric Brugger, Wed Jul 30 18:28:57 PDT 2014
//    Modified the class to work with avtDataRepresentation.
//
// ****************************************************************************

class avtInverseGhostZoneFilter : public avtPluginDataTreeIterator
{
  public:
                         avtInverseGhostZoneFilter();
    virtual             ~avtInverseGhostZoneFilter();

    static avtFilter    *Create();

    virtual const char  *GetType(void)  { return "avtInverseGhostZoneFilter"; };
    virtual const char  *GetDescription(void)
                             { return "Inverse Ghost Zone"; };

    virtual void         SetAtts(const AttributeGroup*);
    virtual bool         Equivalent(const AttributeGroup*);

  protected:
    InverseGhostZoneAttributes   atts;

    virtual avtContract_p
                          ModifyContract(avtContract_p);

    virtual avtDataRepresentation *ExecuteData(avtDataRepresentation *);
    virtual void          UpdateDataObjectInfo(void);
};


#endif
