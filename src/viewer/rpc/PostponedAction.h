// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

#ifndef POSTPONEDACTION_H
#define POSTPONEDACTION_H
#include <viewerrpc_exports.h>
#include <AttributeSubject.h>

#include <ViewerRPC.h>

// ****************************************************************************
// Class: PostponedAction
//
// Purpose:
//    This class contains the attributes for controlling the viewer vis a postponed action.
//
// Notes:      Autogenerated by xml2atts.
//
// Programmer: xml2atts
// Creation:   omitted
//
// Modifications:
//
// ****************************************************************************

class VIEWER_RPC_API PostponedAction : public AttributeSubject
{
public:
    // These constructors are for objects of this class
    PostponedAction();
    PostponedAction(const PostponedAction &obj);
protected:
    // These constructors are for objects derived from this class
    PostponedAction(private_tmfs_t tmfs);
    PostponedAction(const PostponedAction &obj, private_tmfs_t tmfs);
public:
    virtual ~PostponedAction();

    virtual PostponedAction& operator = (const PostponedAction &obj);
    virtual bool operator == (const PostponedAction &obj) const;
    virtual bool operator != (const PostponedAction &obj) const;
private:
    void Init();
    void Copy(const PostponedAction &obj);
public:

    virtual const std::string TypeName() const;
    virtual bool CopyAttributes(const AttributeGroup *);
    virtual AttributeSubject *CreateCompatible(const std::string &) const;
    virtual AttributeSubject *NewInstance(bool) const;

    // Property selection methods
    virtual void SelectAll();
    void SelectRPC();

    // Property setting methods
    void SetRPC(const ViewerRPC &RPC_);
    void SetWindow(int window_);

    // Property getting methods
    const ViewerRPC &GetRPC() const;
          ViewerRPC &GetRPC();
    int             GetWindow() const;


    // Keyframing methods
    virtual std::string               GetFieldName(int index) const;
    virtual AttributeGroup::FieldType GetFieldType(int index) const;
    virtual std::string               GetFieldTypeName(int index) const;
    virtual bool                      FieldsEqual(int index, const AttributeGroup *rhs) const;


    // IDs that can be used to identify fields in case statements
    enum {
        ID_RPC = 0,
        ID_window,
        ID__LAST
    };

private:
    ViewerRPC RPC;
    int       window;

    // Static class format string for type map.
    static const char *TypeMapFormatString;
    static const private_tmfs_t TmfsStruct;
};
#define POSTPONEDACTION_TMFS "ai"

#endif
