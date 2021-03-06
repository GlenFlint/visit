// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

// ************************************************************************* //
//                          avtMDSplusOptions.h                              //
// ************************************************************************* //

#ifndef AVT_MDSplus_OPTIONS_H
#define AVT_MDSplus_OPTIONS_H

class DBOptionsAttributes;

#include <string>


// ****************************************************************************
//  Functions: avtMDSplusOptions
//
//  Purpose:
//      Creates the options for  MDSplus readers and/or writers.
//
//  Programmer: allen -- generated by xml2avt
//  Creation:   Wed Aug 11 13:45:13 PST 2010
//
// ****************************************************************************

DBOptionsAttributes *GetMDSplusReadOptions(void);
DBOptionsAttributes *GetMDSplusWriteOptions(void);


#endif
