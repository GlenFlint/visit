// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

// ****************************************************************************
//  File: TopologyGUIPluginInfo.C
// ****************************************************************************

#include <TopologyPluginInfo.h>
#include <TopologyAttributes.h>
#include <QApplication>
#include <QvisTopologyPlotWindow.h>

VISIT_PLOT_PLUGIN_ENTRY(Topology,GUI)

// ****************************************************************************
//  Method: TopologyGUIPluginInfo::GetMenuName
//
//  Purpose:
//    Return a pointer to the name to use in the GUI menu.
//
//  Returns:    A pointer to the name to use in the GUI menu.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

QString *
TopologyGUIPluginInfo::GetMenuName() const
{
    return new QString(qApp->translate("PlotNames", "Topology"));
}


// ****************************************************************************
//  Method: TopologyGUIPluginInfo::CreatePluginWindow
//
//  Purpose:
//    Return a pointer to an plot's attribute window.
//
//  Arguments:
//    type      The type of the plot.
//    attr      The attribute subject for the plot.
//    notepad   The notepad to use for posting the window.
//
//  Returns:    A pointer to the plot's attribute window.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

QvisPostableWindowObserver *
TopologyGUIPluginInfo::CreatePluginWindow(int type, AttributeSubject *attr,
    const QString &caption, const QString &shortName, QvisNotepadArea *notepad)
{
    return new QvisTopologyPlotWindow(type, (TopologyAttributes *)attr,
        caption, shortName, notepad);
}

// ****************************************************************************
//  Method: TopologyGUIPluginInfo::XPMIconData
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

#include <Topology.xpm>
const char **
TopologyGUIPluginInfo::XPMIconData() const
{
    return Topology_xpm;
}

