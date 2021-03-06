// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

// ****************************************************************************
//  File: LineoutCommonPluginInfo.C
// ****************************************************************************

#include <LineoutPluginInfo.h>
#include <LineoutAttributes.h>

#include <Expression.h>
#include <ExpressionList.h>
#include <avtDatabaseMetaData.h>
#include <avtScalarMetaData.h>

// ****************************************************************************
//  Method: LineoutCommonPluginInfo::AllocAttributes
//
//  Purpose:
//    Return a pointer to a newly allocated attribute subject.
//
//  Returns:    A pointer to the newly allocated attribute subject.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

AttributeSubject *
LineoutCommonPluginInfo::AllocAttributes()
{
    return new LineoutAttributes;
}

// ****************************************************************************
//  Method: LineoutCommonPluginInfo::CopyAttributes
//
//  Purpose:
//    Copy a Lineout attribute subject.
//
//  Arguments:
//    to        The destination attribute subject.
//    from      The source attribute subject.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
// ****************************************************************************

void
LineoutCommonPluginInfo::CopyAttributes(AttributeSubject *to,
    AttributeSubject *from)
{
    *((LineoutAttributes *) to) = *((LineoutAttributes *) from);
}
// ****************************************************************************
//  Method: LineoutCommonPluginInfo::GetCreatedExpressions
//
//  Purpose:
//      Gets the expressions created by this operator.
//
//  Programmer: generated by xml2info
//  Creation:   omitted
//
//  Modifications:
//    Brad Whitlock, Wed Jul 27 11:31:54 PDT 2011
//    Use std::string instead of sprintf and insert <> in case the variable
//    name has some punctuation.
//
// ****************************************************************************

ExpressionList *
LineoutCommonPluginInfo::GetCreatedExpressions(const avtDatabaseMetaData *md) const
{
    std::string opLineout("operators/Lineout/"),
                exPrefix("cell_constant("), exSuffix(", 0.)"),
                lAngleBracket("<"), rAngleBracket(">");
    ExpressionList *el = new ExpressionList;
    int numScalars = md->GetNumScalars();
    for (int i = 0 ; i < numScalars ; i++)
    {
        const avtScalarMetaData *mmd = md->GetScalar(i);
        if (mmd->hideFromGUI || !mmd->validVariable)
            continue;
        {
            Expression e2;
            e2.SetName(opLineout + mmd->name);
            e2.SetType(Expression::CurveMeshVar);
            e2.SetFromOperator(true);
            e2.SetOperatorName("Lineout");
            e2.SetDefinition(exPrefix + lAngleBracket + mmd->name + rAngleBracket + exSuffix);
            el->AddExpressions(e2);
        }
    }
    const ExpressionList &oldEL = md->GetExprList();
    for (int i = 0 ; i < oldEL.GetNumExpressions() ; i++)
    {
        const Expression &e = oldEL.GetExpressions(i);
        if (e.GetFromOperator() || e.GetAutoExpression())
            continue;
        if (e.GetType() == Expression::ScalarMeshVar)
        {
            {
                Expression e2;
                e2.SetName(opLineout + e.GetName());
                e2.SetType(Expression::CurveMeshVar);
                e2.SetFromOperator(true);
                e2.SetOperatorName("Lineout");
                e2.SetDefinition(exPrefix + e.GetName() + exSuffix);
                el->AddExpressions(e2);
            }
        }
    }
    return el;
}

// ****************************************************************************
// Method: LineoutCommonPluginInfo::GetVariableTypes
//
// Purpose:
//   Indicates that if the Lineout operator is present in the selected plot
//   then the GUI's variable menu should include scalars.
//
// Returns:    The allowable variable types for the Lineout operator.
//
// Programmer: Brad Whitlock
// Creation:   Tue Apr 25 16:51:05 PST 2006
//
// Modifications:
//
// ****************************************************************************

int
LineoutCommonPluginInfo::GetVariableTypes() const
{
    return VAR_CATEGORY_SCALAR;
}

// ****************************************************************************
// Method: LineoutCommonPluginInfo::GetVariableMask
//
// Purpose:
//   Returns a mask that lets the Lineout operator eliminate certain variable
//   types from the variable menu.
//
// Programmer: Brad Whitlock
// Creation:   Tue Apr 25 16:52:08 PST 2006
//
// Modifications:
//
// ****************************************************************************

int
LineoutCommonPluginInfo::GetVariableMask() const
{
    return VAR_CATEGORY_SCALAR;
}

// ****************************************************************************
// Method: LineoutCommonPluginInfo::GetUserSelectable
//
// Purpose:
//   Indicates that the Lineout operator cannot be selected in the GUI.
//
// Programmer: Brad Whitlock
// Creation:   Tue Apr 25 17:04:25 PST 2006
//
// Modifications:
//
// ****************************************************************************

bool
LineoutCommonPluginInfo::GetUserSelectable() const
{
    return false;
}

