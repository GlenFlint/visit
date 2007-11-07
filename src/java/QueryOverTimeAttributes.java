// ***************************************************************************
//
// Copyright (c) 2000 - 2007, The Regents of the University of California
// Produced at the Lawrence Livermore National Laboratory
// All rights reserved.
//
// This file is part of VisIt. For details, see http://www.llnl.gov/visit/. The
// full copyright notice is contained in the file COPYRIGHT located at the root
// of the VisIt distribution or at http://www.llnl.gov/visit/copyright.html.
//
// Redistribution  and  use  in  source  and  binary  forms,  with  or  without
// modification, are permitted provided that the following conditions are met:
//
//  - Redistributions of  source code must  retain the above  copyright notice,
//    this list of conditions and the disclaimer below.
//  - Redistributions in binary form must reproduce the above copyright notice,
//    this  list of  conditions  and  the  disclaimer (as noted below)  in  the
//    documentation and/or materials provided with the distribution.
//  - Neither the name of the UC/LLNL nor  the names of its contributors may be
//    used to  endorse or  promote products derived from  this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT  HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR  IMPLIED WARRANTIES, INCLUDING,  BUT NOT  LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND  FITNESS FOR A PARTICULAR  PURPOSE
// ARE  DISCLAIMED.  IN  NO  EVENT  SHALL  THE  REGENTS  OF  THE  UNIVERSITY OF
// CALIFORNIA, THE U.S.  DEPARTMENT  OF  ENERGY OR CONTRIBUTORS BE  LIABLE  FOR
// ANY  DIRECT,  INDIRECT,  INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT  LIMITED TO, PROCUREMENT OF  SUBSTITUTE GOODS OR
// SERVICES; LOSS OF  USE, DATA, OR PROFITS; OR  BUSINESS INTERRUPTION) HOWEVER
// CAUSED  AND  ON  ANY  THEORY  OF  LIABILITY,  WHETHER  IN  CONTRACT,  STRICT
// LIABILITY, OR TORT  (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY  WAY
// OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.
//
// ***************************************************************************

package llnl.visit;


// ****************************************************************************
// Class: QueryOverTimeAttributes
//
// Purpose:
//    Attributes for queries over time.
//
// Notes:      Autogenerated by xml2java.
//
// Programmer: xml2java
// Creation:   Wed Nov 7 14:38:15 PST 2007
//
// Modifications:
//   
// ****************************************************************************

public class QueryOverTimeAttributes extends AttributeSubject
{
    // Enum values
    public final static int TIMETYPE_CYCLE = 0;
    public final static int TIMETYPE_DTIME = 1;
    public final static int TIMETYPE_TIMESTEP = 2;


    public QueryOverTimeAttributes()
    {
        super(10);

        timeType = TIMETYPE_CYCLE;
        startTimeFlag = false;
        startTime = 0;
        endTimeFlag = false;
        endTime = 1;
        stride = 1;
        createWindow = true;
        windowId = 2;
        queryAtts = new QueryAttributes();
        pickAtts = new PickAttributes();
    }

    public QueryOverTimeAttributes(QueryOverTimeAttributes obj)
    {
        super(10);

        timeType = obj.timeType;
        startTimeFlag = obj.startTimeFlag;
        startTime = obj.startTime;
        endTimeFlag = obj.endTimeFlag;
        endTime = obj.endTime;
        stride = obj.stride;
        createWindow = obj.createWindow;
        windowId = obj.windowId;
        queryAtts = new QueryAttributes(obj.queryAtts);
        pickAtts = new PickAttributes(obj.pickAtts);

        SelectAll();
    }

    public boolean equals(QueryOverTimeAttributes obj)
    {
        // Create the return value
        return ((timeType == obj.timeType) &&
                (startTimeFlag == obj.startTimeFlag) &&
                (startTime == obj.startTime) &&
                (endTimeFlag == obj.endTimeFlag) &&
                (endTime == obj.endTime) &&
                (stride == obj.stride) &&
                (createWindow == obj.createWindow) &&
                (windowId == obj.windowId) &&
                (queryAtts == obj.queryAtts) &&
                (pickAtts == obj.pickAtts));
    }

    // Property setting methods
    public void SetTimeType(int timeType_)
    {
        timeType = timeType_;
        Select(0);
    }

    public void SetStartTimeFlag(boolean startTimeFlag_)
    {
        startTimeFlag = startTimeFlag_;
        Select(1);
    }

    public void SetStartTime(int startTime_)
    {
        startTime = startTime_;
        Select(2);
    }

    public void SetEndTimeFlag(boolean endTimeFlag_)
    {
        endTimeFlag = endTimeFlag_;
        Select(3);
    }

    public void SetEndTime(int endTime_)
    {
        endTime = endTime_;
        Select(4);
    }

    public void SetStride(int stride_)
    {
        stride = stride_;
        Select(5);
    }

    public void SetCreateWindow(boolean createWindow_)
    {
        createWindow = createWindow_;
        Select(6);
    }

    public void SetWindowId(int windowId_)
    {
        windowId = windowId_;
        Select(7);
    }

    public void SetQueryAtts(QueryAttributes queryAtts_)
    {
        queryAtts = queryAtts_;
        Select(8);
    }

    public void SetPickAtts(PickAttributes pickAtts_)
    {
        pickAtts = pickAtts_;
        Select(9);
    }

    // Property getting methods
    public int             GetTimeType() { return timeType; }
    public boolean         GetStartTimeFlag() { return startTimeFlag; }
    public int             GetStartTime() { return startTime; }
    public boolean         GetEndTimeFlag() { return endTimeFlag; }
    public int             GetEndTime() { return endTime; }
    public int             GetStride() { return stride; }
    public boolean         GetCreateWindow() { return createWindow; }
    public int             GetWindowId() { return windowId; }
    public QueryAttributes GetQueryAtts() { return queryAtts; }
    public PickAttributes  GetPickAtts() { return pickAtts; }

    // Write and read methods.
    public void WriteAtts(CommunicationBuffer buf)
    {
        if(WriteSelect(0, buf))
            buf.WriteInt(timeType);
        if(WriteSelect(1, buf))
            buf.WriteBool(startTimeFlag);
        if(WriteSelect(2, buf))
            buf.WriteInt(startTime);
        if(WriteSelect(3, buf))
            buf.WriteBool(endTimeFlag);
        if(WriteSelect(4, buf))
            buf.WriteInt(endTime);
        if(WriteSelect(5, buf))
            buf.WriteInt(stride);
        if(WriteSelect(6, buf))
            buf.WriteBool(createWindow);
        if(WriteSelect(7, buf))
            buf.WriteInt(windowId);
        if(WriteSelect(8, buf))
            queryAtts.Write(buf);
        if(WriteSelect(9, buf))
            pickAtts.Write(buf);
    }

    public void ReadAtts(int n, CommunicationBuffer buf)
    {
        for(int i = 0; i < n; ++i)
        {
            int index = (int)buf.ReadByte();
            switch(index)
            {
            case 0:
                SetTimeType(buf.ReadInt());
                break;
            case 1:
                SetStartTimeFlag(buf.ReadBool());
                break;
            case 2:
                SetStartTime(buf.ReadInt());
                break;
            case 3:
                SetEndTimeFlag(buf.ReadBool());
                break;
            case 4:
                SetEndTime(buf.ReadInt());
                break;
            case 5:
                SetStride(buf.ReadInt());
                break;
            case 6:
                SetCreateWindow(buf.ReadBool());
                break;
            case 7:
                SetWindowId(buf.ReadInt());
                break;
            case 8:
                queryAtts.Read(buf);
                Select(8);
                break;
            case 9:
                pickAtts.Read(buf);
                Select(9);
                break;
            }
        }
    }


    // Attributes
    private int             timeType;
    private boolean         startTimeFlag;
    private int             startTime;
    private boolean         endTimeFlag;
    private int             endTime;
    private int             stride;
    private boolean         createWindow;
    private int             windowId;
    private QueryAttributes queryAtts;
    private PickAttributes  pickAtts;
}

