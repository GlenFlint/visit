// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

#include <CosmosPluginInfo.h>
#include <avtCosmosFileFormat.h>
#include <avtMTMDFileFormatInterface.h>
#include <avtGenericDatabase.h>

// ****************************************************************************
//  Method:  CosmosCommonPluginInfo::GetDatabaseType
//
//  Purpose:
//    Returns the type of a Cosmos database.
//
//  Programmer:  generated by xml2info
//  Creation:    omitted
//
// ****************************************************************************
DatabaseType
CosmosCommonPluginInfo::GetDatabaseType()
{
    return DB_TYPE_MTMD;
}

// ****************************************************************************
//  Method: CosmosCommonPluginInfo::SetupDatabase
//
//  Purpose:
//      Sets up a Cosmos database.
//
//  Arguments:
//      list    A list of file names.
//      nList   The number of timesteps in list.
//      nBlocks The number of blocks in the list.
//
//  Returns:    A Cosmos database from list.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************
avtDatabase *
CosmosCommonPluginInfo::SetupDatabase(const char *const *list,
                                   int nList, int nBlock)
{
    // ignore any nBlocks past 1
    int nTimestepGroups = nList / nBlock;
    avtMTMDFileFormat **ffl = new avtMTMDFileFormat*[nTimestepGroups];
    for (int i = 0; i < nTimestepGroups; i++)
    {
        ffl[i] = new avtCosmosFileFormat(list[i*nBlock]);
    }
    avtMTMDFileFormatInterface *inter
           = new avtMTMDFileFormatInterface(ffl, nTimestepGroups);
    return new avtGenericDatabase(inter);
}
