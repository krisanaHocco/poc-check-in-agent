from langchain_core.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field
from datetime import datetime

class DataStorageInput(BaseModel):
    user: str = Field(description="User name or identifier")
    datetime_str: str = Field(description="Datetime in YYYY-MM-DD HH:mm:ss format")
    message: str = Field(description="Message from user")

class DataStorageOutput(BaseModel):
    user: str
    datetime: str  # dd/MM/YYYY HH:mm:ss format

class DataStorageTool(BaseTool):
    name: str = "data_storage"
    description: str = "Stores user data with datetime in memory cache. Input format: user (string), datetime (YYYY-MM-DD HH:mm:ss), Message (string). Stored format: user (string), datetime (dd/MM/YYYY HH:mm:ss)."
    args_schema: Type[BaseModel] = DataStorageInput
    
    # In-memory storage array (class variable shared across instances)
    _storage: List[DataStorageOutput] = []

    def _run(self, user: str, datetime_str: str, message: str) -> str:
        try:
            # Check message from user is "Check in"
            if message.lower() != "check in":
                return f"Error: Invalid message. Expected 'Check in', got: {message}"
            
            # Check user has not already checked in
            for entry in self._storage:
                if entry.user == user:
                    return f"Error: User has already checked in. User: {user}, Datetime: {entry.datetime}. Total records: {len(self._storage)}"
            
            # Parse input datetime (YYYY-MM-DD HH:mm:ss)
            input_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            
            # Convert to output format (dd/MM/YYYY HH:mm:ss)
            output_datetime = input_datetime.strftime("%d/%m/%Y %H:%M:%S")
            
            # Create data entry
            data_entry = DataStorageOutput(
                user=user,
                datetime=output_datetime
            )
            
            # Store in memory array
            self._storage.append(data_entry)
            
            return f"Data stored successfully. User: {user}, Datetime: {output_datetime}. Total records: {len(self._storage)} \n"
            
        except ValueError as e:
            return f"Error: Invalid datetime format. Expected YYYY-MM-DD HH:mm:ss, got: {datetime_str}. Details: {str(e)} \n"
        except Exception as e:
            return f"Error storing data: {str(e)} \n"
    
    @classmethod
    def get_all_data(cls) -> List[DataStorageOutput]:
        """Get all stored data"""
        return cls._storage.copy()
    
    @classmethod
    def get_data_count(cls) -> int:
        """Get count of stored records"""
        return len(cls._storage)
    
    @classmethod
    def clear_data(cls) -> str:
        """Clear all stored data"""
        count = len(cls._storage)
        cls._storage.clear()
        return f"Cleared {count} records from storage"
    
    @classmethod
    def get_data_by_user(cls, user: str) -> List[DataStorageOutput]:
        """Get data filtered by user"""
        return [entry for entry in cls._storage if entry.user == user]