// ************************************************************************* //
//  File: avtDisplaceFilter.h
// ************************************************************************* //

#ifndef AVT_Displace_FILTER_H
#define AVT_Displace_FILTER_H


#include <avtPluginStreamer.h>
#include <DisplaceAttributes.h>


class vtkDataSet;


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
//    Added RefashionDataObjectInfo, PerformRestriction.
//
//    Hank Childs, Mon May 24 16:16:05 PDT 2004
//    Added PostExecute to manage extents.
//
// ****************************************************************************

class avtDisplaceFilter : public avtPluginStreamer
{
  public:
                         avtDisplaceFilter();
    virtual             ~avtDisplaceFilter();

    static avtFilter    *Create();

    virtual const char  *GetType(void)  { return "avtDisplaceFilter"; };
    virtual const char  *GetDescription(void)
                             { return "Displacing by a vector"; };

    virtual void         SetAtts(const AttributeGroup*);
    virtual bool         Equivalent(const AttributeGroup*);

  protected:
    DisplaceAttributes   atts;
    bool                 issuedWarning;

    virtual vtkDataSet   *ExecuteData(vtkDataSet *, int, std::string);
    virtual void          PostExecute(void);

    virtual void          RefashionDataObjectInfo(void);
    virtual avtPipelineSpecification_p
                          PerformRestriction(avtPipelineSpecification_p);

};


#endif
