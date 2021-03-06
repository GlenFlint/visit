// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

// ************************************************************************* //
//                             avtMDSplusOptions.C                           //
// ************************************************************************* //

#include <avtMDSplusOptions.h>

#include <DBOptionsAttributes.h>

#include <string>


// ****************************************************************************
//  Function: GetMDSplusReadOptions
//
//  Purpose:
//      Creates the options for MDSplus readers.
//
//  Important Note:
//      The code below sets up empty options.  If your format 
//      does not require read options, no modifications are 
//      necessary.
//
//  Programmer: allen -- generated by xml2avt
//  Creation:   Wed Aug 11 13:45:13 PST 2010
//
// ****************************************************************************

DBOptionsAttributes *
GetMDSplusReadOptions(void)
{
    DBOptionsAttributes *rv = new DBOptionsAttributes;

    rv->SetString("Host",   "alcdata.psfc.mit.edu"); //atlas.gat.com");
    rv->SetString("Tree",   "CMOD"); //"NIMROD");
    rv->SetInt   ("Shot",   1100817001); //10089);
    rv->SetString("Signal", "\\ip");
    rv->SetString("Mesh",   "myMesh");

    return rv;
}


// ****************************************************************************
//  Function: GetMDSplusWriteOptions
//
//  Purpose:
//      Creates the options for MDSplus writers.
//
//  Important Note:
//      The code below sets up empty options.  If your format 
//      does not require write options, no modifications are 
//      necessary.
//
//  Programmer: allen -- generated by xml2avt
//  Creation:   Wed Aug 11 13:45:13 PST 2010
//
// ****************************************************************************

DBOptionsAttributes *
GetMDSplusWriteOptions(void)
{
    DBOptionsAttributes *rv = new DBOptionsAttributes;
    return rv;
}
