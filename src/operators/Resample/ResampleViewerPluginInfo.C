// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

// ****************************************************************************
//  File: ResampleViewerPluginInfo.C
// ****************************************************************************

#include <ResamplePluginInfo.h>
#include <ResampleAttributes.h>

VISIT_OPERATOR_PLUGIN_ENTRY_EV(Resample,Viewer)


// ****************************************************************************
//  Method: ResampleViewerPluginInfo::XPMIconData
//
//  Purpose:
//    Return a pointer to the icon data.
//
//  Returns:    A pointer to the icon data.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

#include <Resample.xpm>
const char **
ResampleViewerPluginInfo::XPMIconData() const
{
    return Resample_xpm;
}

