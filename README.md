# LangChain AI Agent with Data Storage Tool

This project implements an AI Agent using the LangChain framework with a data storage tool that can perform check in timestamp for the user.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key in the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

Make sure virtual environment is activated, then run the server:
```bash
source venv/bin/activate
python server.py

or 

./run_server.sh
```

## Project Structure

- `server.py` - FastAPI server
- `trial_agent.py` - Trial AI Agent
- `check_in_agent.py` - Check In AI Agent
- `leave_agent.py` - Leave AI Agent
- `tools/data_storage.py` - Data storage tool (For Check In)
- `tools/leave_data.py` - Leave data tool (For Leave)
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (API keys)# poc-check-in-agent
