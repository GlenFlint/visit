// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

// ************************************************************************* //
//  File: avtDisplaceFilter.h
// ************************************************************************* //

#ifndef AVT_Displace_FILTER_H
#define AVT_Displace_FILTER_H

#include <filters_exports.h>

#include <avtDataTreeIterator.h>


// ****************************************************************************
//  Class: avtDisplaceFilter
//
//  Purpose:
//      A plugin operator for Displace.
//
//  Programmer: childs -- generated by xml2info
//  Creation:   Mon Nov 5 15:35:46 PST 2001
//
//  Modifications:
//    Kathleen Bonnell, Wed Nov 28 16:59:53 PST 2001
//    Added UpdateDataObjectInfo, ModifyContract.
//
//    Hank Childs, Mon May 24 16:16:05 PDT 2004
//    Added PostExecute to manage extents.
//
//    Hank Childs, Tue Jun 29 07:21:32 PDT 2004
//    Removed data member "issuedWarning", since we no longer issue warnings.
//
//    Hank Childs, Tue Sep  5 16:07:45 PDT 2006
//    Added PreExecute method, so we could check validity of "default".
//
//    Hank Childs, Fri May 18 15:59:03 PDT 2007
//    Changed the inheritance, since this is no longer a plugin filter.
//    (This filter was moved from /operators/Displace.)
//
//    Eric Brugger, Mon Jul 21 10:37:53 PDT 2014
//    Modified the class to work with avtDataRepresentation.
//
// ****************************************************************************

class AVTFILTERS_API avtDisplaceFilter : public avtDataTreeIterator
{
  public:
                         avtDisplaceFilter();
    virtual             ~avtDisplaceFilter();

    virtual const char  *GetType(void)  { return "avtDisplaceFilter"; };
    virtual const char  *GetDescription(void)
                             { return "Displacing by a vector"; };

    void                 SetFactor(double f) { factor = f; };
    void                 SetVariable(const std::string &v);

  protected:
    double                factor;
    std::string           variable;

    virtual avtDataRepresentation *ExecuteData(avtDataRepresentation *);
    virtual void          PreExecute(void);
    virtual void          PostExecute(void);

    virtual void          UpdateDataObjectInfo(void);
    virtual avtContract_p
                          ModifyContract(avtContract_p);

};


#endif
