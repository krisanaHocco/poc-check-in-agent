from langchain_core.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field

class LeaveDataInput(BaseModel):
    user: str = Field(description="User name or identifier")
    leave_type: str = Field(description="Type of leave: sick leave, business leave, or vacation")
    leave_dates: List[str] = Field(description="Array of leave dates in text format (e.g., ['2025-07-29 : all day', '2025-07-21 - 2025-07-24'])")

class LeaveDataOutput(BaseModel):
    user: str
    leave_type: str
    leave_dates: List[str]  # Array of leave dates in text format

# In-memory storage array (class variable shared across instances)
_leave_storage: List[LeaveDataOutput] = []

class LeaveDataTool(BaseTool):
    name: str = "leave_data"
    description: str = "Stores user leave data in memory cache. Input format: user (string), leave_type (sick leave/business leave/vacation), leave_dates (array of text dates). Stored format: user (string), leave_type (string), leave_dates (array of text dates)."
    args_schema: Type[BaseModel] = LeaveDataInput

    def _run(self, user: str, leave_type: str, leave_dates: List[str]) -> str:
        try:
            # Validate leave type
            valid_leave_types = ["sick leave", "business leave", "vacation"]
            if leave_type.lower() not in valid_leave_types:
                return f"Error: Invalid leave type. Expected one of {valid_leave_types}, got: {leave_type}"
            
            # Validate leave_dates is not empty
            if not leave_dates or len(leave_dates) == 0:
                return "Error: Leave dates cannot be empty"
            
            # Create data entry
            data_entry = LeaveDataOutput(
                user=user,
                leave_type=leave_type.lower(),
                leave_dates=leave_dates
            )
            
            # Store in memory array
            _leave_storage.append(data_entry)
            
            return f"Leave data stored successfully. User: {user}, Leave Type: {leave_type.lower()}, Leave Dates: {leave_dates}. Total records: {len(_leave_storage)}\n"
            
        except Exception as e:
            return f"Error storing leave data: {str(e)}"
    
    @classmethod
    def get_all_data(cls) -> List[LeaveDataOutput]:
        """Get all stored leave data"""
        return list(_leave_storage)
    
    @classmethod
    def get_data_count(cls) -> int:
        """Get count of stored leave records"""
        return len(_leave_storage)
    
    @classmethod
    def clear_data(cls) -> str:
        """Clear all stored leave data"""
        count = len(_leave_storage)
        _leave_storage.clear()
        return f"Cleared {count} leave records from storage"
    
    @classmethod
    def get_data_by_user(cls, user: str) -> List[LeaveDataOutput]:
        """Get leave data filtered by user"""
        return [entry for entry in _leave_storage if entry.user == user]
    
    @classmethod
    def get_data_by_leave_type(cls, leave_type: str) -> List[LeaveDataOutput]:
        """Get leave data filtered by leave type"""
        return [entry for entry in _leave_storage if entry.leave_type == leave_type.lower()]
    
    @classmethod
    def get_data_by_user_and_type(cls, user: str, leave_type: str) -> List[LeaveDataOutput]:
        """Get leave data filtered by user and leave type"""
        return [entry for entry in _leave_storage if entry.user == user and entry.leave_type == leave_type.lower()]