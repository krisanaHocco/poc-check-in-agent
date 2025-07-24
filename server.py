#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agent import AIAgent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="LangChain AI Agent API", version="1.0.0")

# Initialize the AI Agent
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI Agent on startup"""
    global agent
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    
    try:
        agent = AIAgent()
        print("✅ AI Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AI Agent: {e}")
        raise


class CheckInRequest(BaseModel):
    user: str = Field(description="User name or identifier")
    datetime: str = Field(description="Datetime in YYYY-MM-dd HH:mm:ss format")
    message: str = Field(description="Message from user")


class CheckInResponse(BaseModel):
    query: str
    response: str
    success: bool


@app.post("/checkin", response_model=CheckInResponse)
async def checkin(request: CheckInRequest):
    """
    Store user check-in data using the AI Agent
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="AI Agent not initialized")
    
    try:
        # Convert request body to user input message for AI Agent
        query = f"Please store check-in data for user '{request.user}' at datetime '{request.datetime}' with message '{request.message}'"
        
        # Get response from AI Agent
        response = agent.run(query)
        
        # Determine success based on response content - return true for successful storage
        success = "data stored successfully" in response.lower() or ("error" not in response.lower() and "Error" not in response)
        
        return CheckInResponse(
            query=query,
            response=response,
            success=success
        )
    
    except Exception as e:
        return CheckInResponse(
            query=f"Store check-in data for user '{request.user}' at '{request.datetime}' with message '{request.message}'",
            response=f"Error: {str(e)}",
            success=False
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent_ready": agent is not None}


# @app.get("/checkin")
# async def get_checkin_data():
#     """Get all stored check-in data"""
#     if agent is None:
#         raise HTTPException(status_code=500, detail="AI Agent not initialized")
    
#     return DataStorageTool.get_all_data()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)