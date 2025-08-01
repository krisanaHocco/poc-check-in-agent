#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from trial_agent import TrialAgent
from checkin_agent import CheckinAgent
from leave_agent import LeaveAgent
from tools.data_storage import DataStorageTool
from tools.leave_data import LeaveDataTool
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Check-in System API", version="2.0.0")

# Initialize the agents
trial_agent = None
checkin_agent = None
leave_agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize all agents on startup"""
    global trial_agent, checkin_agent, leave_agent
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    
    try:
        trial_agent = TrialAgent()
        checkin_agent = CheckinAgent()
        leave_agent = LeaveAgent()
        print("✅ All agents initialized successfully")
        print("  - Trial Agent: Ready for message routing")
        print("  - Check-in Agent: Ready for check-in processing")
        print("  - Leave Agent: Ready for leave request processing")
    except Exception as e:
        print(f"❌ Failed to initialize agents: {e}")
        raise


class MessageRequest(BaseModel):
    user: str = Field(description="User name or identifier")
    datetime: str = Field(description="Datetime in YYYY-MM-dd HH:mm:ss format")
    message: str = Field(description="Message from user")


class MessageResponse(BaseModel):
    user: str
    datetime: str
    message: str
    agent_type: str
    response: str
    success: bool


@app.post("/process", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    """
    Process user message using the multi-agent system
    """
    if trial_agent is None or checkin_agent is None or leave_agent is None:
        raise HTTPException(status_code=500, detail="Agents not initialized")
    
    try:
        # Step 1: Route message using Trial Agent
        agent_type = trial_agent.route_message(request.user, request.datetime, request.message)
        
        if agent_type == "invalid":
            return MessageResponse(
                user=request.user,
                datetime=request.datetime,
                message=request.message,
                agent_type="invalid",
                response="Invalid message. Please use 'Check in' for check-in or include 'leave type' for leave requests.",
                success=False
            )
        
        # Step 2: Process message with appropriate specialized agent
        if agent_type == "checkin":
            query = f"User: {request.user}, Datetime: {request.datetime}, Message: {request.message}"
            response = checkin_agent.run(query)
            success = "successfully recorded" in response.lower() and "error" not in response.lower()
        
        elif agent_type == "leave":
            query = f"User: {request.user}, Datetime: {request.datetime}, Message: {request.message}"
            response = leave_agent.run(query)
            success = "successfully recorded" in response.lower() and "error" not in response.lower()
        
        else:
            # This shouldn't happen, but handle it gracefully
            response = "System error: Unknown agent type returned by Trial Agent"
            success = False
        
        return MessageResponse(
            user=request.user,
            datetime=request.datetime,
            message=request.message,
            agent_type=agent_type,
            response=response,
            success=success
        )
    
    except Exception as e:
        return MessageResponse(
            user=request.user,
            datetime=request.datetime,
            message=request.message,
            agent_type="error",
            response=f"System error: {str(e)}",
            success=False
        )


@app.get("/checkin-data")
async def get_checkin_data(user: Optional[str] = Query(None, description="Filter by specific user")):
    """
    Retrieve user check-in data
    """
    try:
        if user:
            # Get data for specific user
            data = DataStorageTool.get_data_by_user(user)
            return {
                "user": user,
                "total_records": len(data),
                "data": [{"user": entry.user, "datetime": entry.datetime} for entry in data]
            }
        else:
            # Get all data
            data = DataStorageTool.get_all_data()
            return {
                "total_records": len(data),
                "data": [{"user": entry.user, "datetime": entry.datetime} for entry in data]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving check-in data: {str(e)}")


@app.get("/leave-data")
async def get_leave_data(
    user: Optional[str] = Query(None, description="Filter by specific user"),
    leave_type: Optional[str] = Query(None, description="Filter by leave type (sick leave, business leave, vacation)")
):
    """
    Retrieve user leave data with optional filters
    """
    try:
        if user and leave_type:
            # Get data for specific user and leave type
            data = LeaveDataTool.get_data_by_user_and_type(user, leave_type)
            return {
                "user": user,
                "leave_type": leave_type,
                "total_records": len(data),
                "data": [
                    {
                        "user": entry.user, 
                        "leave_type": entry.leave_type,
                        "leave_dates": entry.leave_dates
                    } for entry in data
                ]
            }
        elif user:
            # Get data for specific user
            data = LeaveDataTool.get_data_by_user(user)
            return {
                "user": user,
                "total_records": len(data),
                "data": [
                    {
                        "user": entry.user, 
                        "leave_type": entry.leave_type,
                        "leave_dates": entry.leave_dates
                    } for entry in data
                ]
            }
        elif leave_type:
            # Get data for specific leave type
            data = LeaveDataTool.get_data_by_leave_type(leave_type)
            return {
                "leave_type": leave_type,
                "total_records": len(data),
                "data": [
                    {
                        "user": entry.user, 
                        "leave_type": entry.leave_type,
                        "leave_dates": entry.leave_dates
                    } for entry in data
                ]
            }
        else:
            # Get all leave data
            data = LeaveDataTool.get_all_data()
            return {
                "total_records": len(data),
                "data": [
                    {
                        "user": entry.user, 
                        "leave_type": entry.leave_type,
                        "leave_dates": entry.leave_dates
                    } for entry in data
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving leave data: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "trial_agent_ready": trial_agent is not None,
        "checkin_agent_ready": checkin_agent is not None,
        "leave_agent_ready": leave_agent is not None,
        "all_agents_ready": all([trial_agent, checkin_agent, leave_agent])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)