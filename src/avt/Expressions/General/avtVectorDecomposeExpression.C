/*****************************************************************************
*
* Copyright (c) 2000 - 2019, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory
* LLNL-CODE-442911
* All rights reserved.
*
* This file is  part of VisIt. For  details, see https://visit.llnl.gov/.  The
* full copyright notice is contained in the file COPYRIGHT located at the root
* of the VisIt distribution or at http://www.llnl.gov/visit/copyright.html.
*
* Redistribution  and  use  in  source  and  binary  forms,  with  or  without
* modification, are permitted provided that the following conditions are met:
*
*  - Redistributions of  source code must  retain the above  copyright notice,
*    this list of conditions and the disclaimer below.
*  - Redistributions in binary form must reproduce the above copyright notice,
*    this  list of  conditions  and  the  disclaimer (as noted below)  in  the
*    documentation and/or other materials provided with the distribution.
*  - Neither the name of  the LLNS/LLNL nor the names of  its contributors may
*    be used to endorse or promote products derived from this software without
*    specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT  HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR  IMPLIED WARRANTIES, INCLUDING,  BUT NOT  LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND  FITNESS FOR A PARTICULAR  PURPOSE
* ARE  DISCLAIMED. IN  NO EVENT  SHALL LAWRENCE  LIVERMORE NATIONAL  SECURITY,
* LLC, THE  U.S.  DEPARTMENT OF  ENERGY  OR  CONTRIBUTORS BE  LIABLE  FOR  ANY
* DIRECT,  INDIRECT,   INCIDENTAL,   SPECIAL,   EXEMPLARY,  OR   CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT  LIMITED TO, PROCUREMENT OF  SUBSTITUTE GOODS OR
* SERVICES; LOSS OF  USE, DATA, OR PROFITS; OR  BUSINESS INTERRUPTION) HOWEVER
* CAUSED  AND  ON  ANY  THEORY  OF  LIABILITY,  WHETHER  IN  CONTRACT,  STRICT
* LIABILITY, OR TORT  (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY  WAY
* OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
* DAMAGE.
*
*****************************************************************************/

// ************************************************************************* //
//                     avtVectorDecomposeExpression.C                        //
// ************************************************************************* //

#include <avtVectorDecomposeExpression.h>

#include <cmath>
#include <cstring>

#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>

#include <ExpressionException.h>

#ifdef _OPENMP
#include <StackTimer.h>
#include <omp.h>
#endif

// ****************************************************************************
//  Method: avtVectorDecomposeExpression constructor
//
//  Purpose:
//      Defines the constructor.  Note: this should not be inlined in the
//      header because it causes problems for certain compilers.
//
//  Programmer: Hank Childs
//  Creation:   February 5, 2004
//
// ****************************************************************************

avtVectorDecomposeExpression::avtVectorDecomposeExpression(int w)
{
    which_comp = w;
}


// ****************************************************************************
//  Method: avtVectorDecomposeExpression destructor
//
//  Purpose:
//      Defines the destructor.  Note: this should not be inlined in the header
//      because it causes problems for certain compilers.
//
//  Programmer: Hank Childs
//  Creation:   February 5, 2004
//
// ****************************************************************************

avtVectorDecomposeExpression::~avtVectorDecomposeExpression()
{
    ;
}


// ****************************************************************************
//  Method: avtVectorDecomposeExpression::GetVariableDimension
//
//  Purpose:
//      Does its best to guess at the variable dimension will be after
//      decomposing a "vector".
//
//  Programmer: Hank Childs
//  Creation:   September 23, 2003
//
//  Modifications:
//
//    Hank Childs, Thu Feb 26 09:00:06 PST 2004
//    Account for multiple variables.
//
//    Hank Childs, Thu May  6 10:49:32 PDT 2004
//    Make sure that we are getting the dimension for the active variable.
//
// ****************************************************************************

int
avtVectorDecomposeExpression::GetVariableDimension(void)
{
    avtDataAttributes &atts = GetInput()->GetInfo().GetAttributes();

    if (activeVariable == NULL)
        return 1;
    if (!atts.ValidVariable(activeVariable))
        return 1;

    int inDim = atts.GetVariableDimension(activeVariable);
    if (inDim == 9 || inDim == 6)
        return 3;
    else if (inDim == 4) // 2D symmetric tensor
        return 2;
    else if (inDim == 3)
        return 1;

    // Who knows?!?
    return 1;
}


// ****************************************************************************
//  Method: avtVectorDecomposeExpression::DeriveVariable
//
//  Purpose:
//      Pulls out a component of a vector variable.
//
//  Arguments:
//      inDS      The input dataset.
//
//  Returns:      The derived variable.  The calling class must free this
//                memory.
//
//  Programmer:   Hank Childs
//  Creation:     November 18, 2002
//
//  Modifications:
//    Kathleen Bonnell, Thu Jan  2 15:16:50 PST 2003 
//    Replace MakeObject() with NewInstance() to match new vtk api.
//
//    Sean Ahern, Thu Mar  6 02:00:32 America/Los_Angeles 2003
//    Merged the vector component filters together.
//
//    Hank Childs, Fri Sep 19 15:00:11 PDT 2003
//    Account for tensor data as well.
//
//    Hank Childs, Fri May  7 07:38:31 PDT 2004
//    Fix bug with decomposing point tensors.
//
//    Hank Childs, Tue Oct 26 21:12:49 PDT 2010
//    Fix bug with decomposing 2D tensor data.
//
//    Mark C. Miller, Tue Jan  7 11:07:50 PST 2014
//    Handle decomposing symmetric tensor data in 2 and 3D.
//
//    Brad Whitlock, Fri Jan 12 13:14:48 PST 2018
//    Added some OpenMP.
//
// ****************************************************************************

vtkDataArray *
avtVectorDecomposeExpression::DeriveVariable(vtkDataSet *in_ds, int currentDomainsIndex)
{
    vtkDataArray *arr = NULL;

    //
    // The base class will set the variable of interest to be the 
    // 'activeVariable'.  This is a by-product of how the base class sets its
    // input.  If that method should change (SetActiveVariable), this
    // technique for inferring the variable name may stop working.
    //
    const char *varname = activeVariable;

    //
    // Get the array of interest.
    //
    if (in_ds->GetPointData()->GetArray(varname) != NULL)
    {
        arr = in_ds->GetPointData()->GetArray(varname);
    }
    else
    {
        arr = in_ds->GetCellData()->GetArray(varname);
    }
    if (arr == NULL)
    {
        EXCEPTION2(ExpressionException, outputVariableName, 
                   "When creating an expression, VisIt "
                   "was not able to locate a necessary variable.");
    }

    //
    // The logic here can be pretty tricky.  We want to decompose this "vector"
    // variable.  But the "vector" can be an everyday vector, or it can be a
    // symmetric on non-symmetric tensor.  Based on the dimensionality of the
    // dataset, the output could be a scalar or it could be a vector.
    //
    vtkIdType ntuples = arr->GetNumberOfTuples();
    vtkDataArray *rv = arr->NewInstance();

    bool twoDVector =
            (GetInput()->GetInfo().GetAttributes().GetSpatialDimension() == 2);
    bool symmetricTensor = 
            (GetInput()->GetInfo().GetAttributes().GetVariableType(varname) == AVT_SYMMETRIC_TENSOR_VAR);

    if (twoDVector)
    {
        if (arr->GetNumberOfComponents() == 3)
        {
            rv->SetNumberOfComponents(1);
            rv->SetNumberOfTuples(ntuples);
            for (vtkIdType i = 0 ; i < ntuples ; i++)
            {
                double val = arr->GetComponent(i, which_comp);
                rv->SetTuple1(i, val);
            }
        }
        else if (arr->GetNumberOfComponents() == 4) // 2D tensor
        {
            // 4 components in 2D is a tensor; either general or symmetric
            //
            // general component order       symmetric (Voigt notation) order
            //
            //         0   1                               0   2
            //         2   3                               2   1
            //
            // 0123 | 0123 | 0123 | 0123       0221 | 0221 | 0221 | 0221
            //
            // In either case, there are 4 components per tuple but in the symmetric
            // case we have the obligtion to return the result in Voigt notational order.
            //
            int const genIndices[2][2] = {{0,1},{2,3}};
            int const symmIndices[2][2] = {{0,2},{2,1}};
            int compIndices[2][2];
            memcpy(compIndices, symmetricTensor ? symmIndices : genIndices, sizeof(genIndices));

            // Since VTK dislikes vectors of size 2, make sure that we have 3 components.
            rv->SetNumberOfComponents(3);
            rv->SetNumberOfTuples(ntuples);
            for (vtkIdType i = 0 ; i < ntuples ; i++)
            {
                double val1 = arr->GetComponent(i, compIndices[which_comp][0]);
                double val2 = arr->GetComponent(i, compIndices[which_comp][1]);
                rv->SetTuple3(i, val1, val2, 0.);
            }
        }
        else if (arr->GetNumberOfComponents() == 9)
        {
#warning HANDLE 9 COMPONENT CASE ONLY AND CORRECTLY FOR VOIGT NOTATION
            //
            // Give one row of the tensor back.  Since VTK dislikes vectors
            // of size 2, make sure that we have 3 components.
            //
            rv->SetNumberOfComponents(3);
            rv->SetNumberOfTuples(ntuples);
            for (vtkIdType i = 0 ; i < ntuples ; i++)
            {
                double val1 = arr->GetComponent(i, 3*which_comp);
                double val2 = arr->GetComponent(i, 3*which_comp+1);
                rv->SetTuple3(i, val1, val2, 0.);
            }
        }
        else if ((which_comp < 0) || (arr->GetNumberOfComponents() <= which_comp))
        {
            EXCEPTION2(ExpressionException, outputVariableName, 
                       "the indices for the 2D vector is out of range.");
        }
        else
        {
            rv->SetNumberOfComponents(1);
            rv->SetNumberOfTuples(ntuples);
            for (vtkIdType i = 0 ; i < ntuples ; i++)
            {
                double val = arr->GetComponent(i, which_comp);
                rv->SetTuple1(i, val);
            }
        }
    }
    else
    {
        if (arr->GetNumberOfComponents() == 3)
        {
            //
            // Looks like an everyday 3D vector.
            //
            rv->SetNumberOfComponents(1);
            rv->SetNumberOfTuples(ntuples);

#define COMPUTE_DECOMPOSE3(VARTYPE, LOW, HIGH) \
{ \
     VARTYPE *arr_ptr = (VARTYPE *)arr->GetVoidPointer(0); \
     VARTYPE *rv_ptr  = (VARTYPE *)rv->GetVoidPointer(0); \
     for(vtkIdType i = LOW ; i < HIGH ; i++) \
     { \
         rv_ptr[i] = arr_ptr[i*3 + which_comp]; \
     } \
}
            if(arr->HasStandardMemoryLayout())
            {
#ifdef _OPENMP
                StackTimer t0("avtVectorDecomposeExpression OpenMP");
                #pragma message("Compiling for OpenMP.")
                #pragma omp parallel
                {
                    int threadnum = omp_get_thread_num();
                    int numthreads = omp_get_num_threads();
                    int low = ntuples*threadnum/numthreads;
                    int high = ntuples*(threadnum+1)/numthreads;
#else
                    int low = 0, high = ntuples;
#endif
                    switch(arr->GetDataType())
                    {
                        vtkTemplateMacro(COMPUTE_DECOMPOSE3(VTK_TT, low, high));
                    }
#ifdef _OPENMP
                }
#endif
            }
            else
            {
                for (vtkIdType i = 0 ; i < ntuples ; i++)
                {
                    double val = arr->GetComponent(i, which_comp);
                    rv->SetTuple1(i, val);
                }
            }
        }
        else if ((arr->GetNumberOfComponents() == 6) ||
                 (arr->GetNumberOfComponents() == 9))   // 3D Tensors
        {
            // 6 components in 3D is a symmetric tensor.
            // 9 components in 3D is a tensor; could be general or symmetric.
            //
            // general component order       symmetric (Voigt notation) order
            //
            //         0   1   2                        0   5   4
            //         3   4   5                        5   1   3
            //         6   7   8                        4   3   2
            //
            // 012345678 | 012345678 | ...   054513432 | 054513432 | 054513432...
            //
            // The input array could be 6 or 9 components and the 6 component case
            // Is the same as the 9 component symmetric case. In the symmetric cases
            // we have the obligtion to return the result in Voigt notational order.
            //
            int const genIndices[3][3] = {{0,1,2},{3,4,5},{6,7,8}};
            int const symmIndices[3][3] = {{0,5,4},{5,1,3},{4,3,2}};
            int compIndices[3][3];
            memcpy(compIndices, 
               (symmetricTensor || (arr->GetNumberOfComponents() == 6)) ? symmIndices : genIndices,
               sizeof(genIndices));
            rv->SetNumberOfComponents(3);
            rv->SetNumberOfTuples(ntuples);
            for (vtkIdType i = 0 ; i < ntuples ; i++)
            {
                double val1 = arr->GetComponent(i, compIndices[which_comp][0]);
                double val2 = arr->GetComponent(i, compIndices[which_comp][1]);
                double val3 = arr->GetComponent(i, compIndices[which_comp][2]);
                rv->SetTuple3(i, val1, val2, val3);
            }
        }
        else if ((which_comp < 0) || (arr->GetNumberOfComponents() <= which_comp))
        {
            EXCEPTION2(ExpressionException, outputVariableName, 
                       "the indices for the 3D vector is out of range.");
        }
        else
        {
            rv->SetNumberOfComponents(1);
            rv->SetNumberOfTuples(ntuples);
            for (vtkIdType i = 0 ; i < ntuples ; i++)
            {
                double val = arr->GetComponent(i, which_comp);
                rv->SetTuple1(i, val);
            }
        }
    }

    return rv;
}
