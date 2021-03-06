// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

#ifndef SIMULATIONUIVALUES_H
#define SIMULATIONUIVALUES_H
#include <state_exports.h>
#include <string>
#include <AttributeSubject.h>


// ****************************************************************************
// Class: SimulationUIValues
//
// Purpose:
//    Contains UI values from a simulation.
//
// Notes:      Autogenerated by xml2atts.
//
// Programmer: xml2atts
// Creation:   omitted
//
// Modifications:
//
// ****************************************************************************

class STATE_API SimulationUIValues : public AttributeSubject
{
public:
    // These constructors are for objects of this class
    SimulationUIValues();
    SimulationUIValues(const SimulationUIValues &obj);
protected:
    // These constructors are for objects derived from this class
    SimulationUIValues(private_tmfs_t tmfs);
    SimulationUIValues(const SimulationUIValues &obj, private_tmfs_t tmfs);
public:
    virtual ~SimulationUIValues();

    virtual SimulationUIValues& operator = (const SimulationUIValues &obj);
    virtual bool operator == (const SimulationUIValues &obj) const;
    virtual bool operator != (const SimulationUIValues &obj) const;
private:
    void Init();
    void Copy(const SimulationUIValues &obj);
public:

    virtual const std::string TypeName() const;
    virtual bool CopyAttributes(const AttributeGroup *);
    virtual AttributeSubject *CreateCompatible(const std::string &) const;
    virtual AttributeSubject *NewInstance(bool) const;

    // Property selection methods
    virtual void SelectAll();
    void SelectHost();
    void SelectSim();
    void SelectName();
    void SelectSvalue();

    // Property setting methods
    void SetHost(const std::string &host_);
    void SetSim(const std::string &sim_);
    void SetName(const std::string &name_);
    void SetIvalue(int ivalue_);
    void SetSvalue(const std::string &svalue_);
    void SetEnabled(bool enabled_);

    // Property getting methods
    const std::string &GetHost() const;
          std::string &GetHost();
    const std::string &GetSim() const;
          std::string &GetSim();
    const std::string &GetName() const;
          std::string &GetName();
    int               GetIvalue() const;
    const std::string &GetSvalue() const;
          std::string &GetSvalue();
    bool              GetEnabled() const;


    // Keyframing methods
    virtual std::string               GetFieldName(int index) const;
    virtual AttributeGroup::FieldType GetFieldType(int index) const;
    virtual std::string               GetFieldTypeName(int index) const;
    virtual bool                      FieldsEqual(int index, const AttributeGroup *rhs) const;


    // IDs that can be used to identify fields in case statements
    enum {
        ID_host = 0,
        ID_sim,
        ID_name,
        ID_ivalue,
        ID_svalue,
        ID_enabled,
        ID__LAST
    };

private:
    std::string host;
    std::string sim;
    std::string name;
    int         ivalue;
    std::string svalue;
    bool        enabled;

    // Static class format string for type map.
    static const char *TypeMapFormatString;
    static const private_tmfs_t TmfsStruct;
};
#define SIMULATIONUIVALUES_TMFS "sssisb"

#endif
