// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

#include "vtkTensorReduceFilter.h"

#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataSet.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkVisItUtility.h>


vtkStandardNewMacro(vtkTensorReduceFilter);


vtkTensorReduceFilter::vtkTensorReduceFilter()
{
  stride = 10;
  numEls = -1;
}

void
vtkTensorReduceFilter::SetStride(int s)
{
  numEls = -1;
  stride = s;
}

void
vtkTensorReduceFilter::SetNumberOfElements(int n)
{
  stride = -1;
  numEls = n;
}

// ****************************************************************************
// Method: vtkTensorReduceFilter::RequestData
//
// Modifications:
//    Kathleen Bonnell, Tue Aug 30 11:11:56 PDT 2005 
//    Copy other Point and Cell data. 
//
//    Kathleen Biagas, Wed Sep 5 13:10:18 MST 2012 
//    Preserve coordinate and tensor data types.
//
//    Eric Brugger, Thu Jan 10 12:05:20 PST 2013
//    Modified to inherit from vtkPolyDataAlgorithm.
//
// ****************************************************************************

int
vtkTensorReduceFilter::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  vtkDebugMacro(<<"Executing vtkTensorReduceFilter");

  // get the info objects
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  //
  // Initialize some frequently used values.
  //
  vtkDataSet   *input = vtkDataSet::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkPolyData *output = vtkPolyData::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkCellData *inCd = input->GetCellData();
  vtkPointData *inPd = input->GetPointData();
  vtkCellData *outCd = output->GetCellData();
  vtkPointData *outPd = output->GetPointData();

  vtkDataArray *inCtensors = inCd->GetTensors();
  vtkDataArray *inPtensors = inPd->GetTensors();

  int npts = input->GetNumberOfPoints();
  int ncells = input->GetNumberOfCells();

  if (inPtensors == NULL && inCtensors == NULL)
    {
    vtkErrorMacro(<<"No tensors to reduce");
    return 1;
    }

  int inPType = (inPtensors ? inPtensors->GetDataType() : VTK_FLOAT);
  int inCType = (inCtensors ? inCtensors->GetDataType() : VTK_FLOAT);

  // Determine what the stride is.
  if (stride <= 0 && numEls <= 0)
    {
    vtkErrorMacro(<<"Invalid stride");
    return 1;
    }

  float actingStride = stride;
  if (actingStride <= 0)
    {
    int totalTensors = 0;
    if (inPtensors != NULL)
    {
        totalTensors += npts;
    }
    if (inCtensors != NULL)
    {
        totalTensors += ncells;
    }
    actingStride = ceil(((float) totalTensors) / ((float) numEls));
    }

  vtkPoints *outpts = vtkVisItUtility::NewPoints(input);
  vtkDataArray *outTensors;
  if (inPType == VTK_DOUBLE || inCType == VTK_DOUBLE)
      outTensors = vtkDoubleArray::New();
  else 
      outTensors = vtkFloatArray::New();
  outTensors->SetNumberOfComponents(9);

  float nextToTake = 0.;
  int count = 0;
  if (inPtensors != NULL)
    {
    outPd->CopyAllocate(inPd, npts);
    outTensors->SetName(inPtensors->GetName());
    for (int i = 0 ; i < npts ; i++)
      {
      if (i >= nextToTake)
        {
        nextToTake += actingStride;

        double pt[3];
        input->GetPoint(i, pt);
        outpts->InsertNextPoint(pt);

        double v[9];
        inPtensors->GetTuple(i, v);
        outTensors->InsertNextTuple(v);
        outPd->CopyData(inPd, i, count++);
        }
      }
      outPd->Squeeze();
    }

  nextToTake = 0.;
  count = 0;
  if (inCtensors != NULL)
    {
    outCd->CopyAllocate(inCd, ncells);
    outTensors->SetName(inCtensors->GetName());
    for (int i = 0 ; i < ncells ; i++)
      {
      if (i >= nextToTake)
        {
        nextToTake += actingStride;

        vtkCell *cell = input->GetCell(i);
        double pt[3];
        cell->GetParametricCenter(pt);
        outpts->InsertNextPoint(pt);

        double v[9];
        inCtensors->GetTuple(i, v);
        outTensors->InsertNextTuple(v);
        outCd->CopyData(inCd, i, count++);
        }
      }
      outCd->Squeeze();
    }

  int nOutPts = outpts->GetNumberOfPoints();
  output->SetPoints(outpts);
  outpts->Delete();
  output->GetPointData()->SetTensors(outTensors);
  outTensors->Delete();

  output->Allocate(nOutPts);
  vtkIdType onevertex[1];
  for (int i = 0 ; i < nOutPts ; i++)
    {
    onevertex[0] = i;
    output->InsertNextCell(VTK_VERTEX, 1, onevertex);
    }
  return 1;
}

// ****************************************************************************
//  Method: vtkTensorReduceFilter::FillInputPortInformation
//
// ****************************************************************************

int
vtkTensorReduceFilter::FillInputPortInformation(int, vtkInformation *info)
{
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
  return 1;
}

// ****************************************************************************
//  Method: vtkTensorReduceFilter::PrintSelf
//
// ****************************************************************************

void
vtkTensorReduceFilter::PrintSelf(ostream &os, vtkIndent indent)
{
   this->Superclass::PrintSelf(os, indent);
   os << indent << "Stride: " << this->stride << "\n";
   os << indent << "Target number of tensors: " << this->numEls << "\n";
}
