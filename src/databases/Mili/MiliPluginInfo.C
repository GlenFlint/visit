// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

// ****************************************************************************
//  File: MiliPluginInfo.C
// ****************************************************************************

#include <MiliPluginInfo.h>

#include <visit-config.h>
VISIT_PLUGIN_VERSION(Mili,DBP_EXPORT)

VISIT_DATABASE_PLUGIN_ENTRY(Mili,General)

// ****************************************************************************
//  Method: MiliGeneralPluginInfo::GetName
//
//  Purpose:
//    Return the name of the database plugin.
//
//  Returns:    A pointer to the name of the database plugin.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

const char *
MiliGeneralPluginInfo::GetName() const
{
    return "Mili";
}

// ****************************************************************************
//  Method: MiliGeneralPluginInfo::GetVersion
//
//  Purpose:
//    Return the version of the database plugin.
//
//  Returns:    A pointer to the version of the database plugin.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

const char *
MiliGeneralPluginInfo::GetVersion() const
{
    return "2.0";
}

// ****************************************************************************
//  Method: MiliGeneralPluginInfo::GetID
//
//  Purpose:
//    Return the id of the database plugin.
//
//  Returns:    A pointer to the id of the database plugin.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

const char *
MiliGeneralPluginInfo::GetID() const
{
    return "Mili_2.0";
}
// ****************************************************************************
//  Method: MiliGeneralPluginInfo::EnabledByDefault
//
//  Purpose:
//    Return true if this plugin should be enabled by default; false otherwise.
//
//  Returns:    true/false
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

bool
MiliGeneralPluginInfo::EnabledByDefault() const
{
    return true;
}
// ****************************************************************************
//  Method: MiliGeneralPluginInfo::HasWriter
//
//  Purpose:
//    Return true if this plugin has a database writer.
//
//  Returns:    true/false
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

bool
MiliGeneralPluginInfo::HasWriter() const
{
    return false;
}
// ****************************************************************************
//  Method:  MiliGeneralPluginInfo::GetDefaultFilePatterns
//
//  Purpose:
//    Returns the default patterns for a Mili database.
//
//  Programmer:  generated by xml2info
//  Creation:    omitted
//
// ****************************************************************************
std::vector<std::string>
MiliGeneralPluginInfo::GetDefaultFilePatterns() const
{
    std::vector<std::string> defaultPatterns;
    defaultPatterns.push_back("*.mili");
    defaultPatterns.push_back("*.m");

    return defaultPatterns;
}

// ****************************************************************************
//  Method:  MiliGeneralPluginInfo::AreDefaultFilePatternsStrict
//
//  Purpose:
//    Returns if the file patterns for a Mili database are
//    intended to be interpreted strictly by default.
//
//  Programmer:  generated by xml2info
//  Creation:    omitted
//
// ****************************************************************************
bool
MiliGeneralPluginInfo::AreDefaultFilePatternsStrict() const
{
    return false;
}

// ****************************************************************************
//  Method:  MiliGeneralPluginInfo::OpensWholeDirectory
//
//  Purpose:
//    Returns if the Mili plugin opens a whole directory name
//    instead of a single file.
//
//  Programmer:  generated by xml2info
//  Creation:    omitted
//
// ****************************************************************************
bool
MiliGeneralPluginInfo::OpensWholeDirectory() const
{
    return false;
}
