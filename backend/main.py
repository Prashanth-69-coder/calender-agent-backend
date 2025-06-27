import os
import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from google_auth_oauthlib.flow import Flow
import uvicorn

from db import supabase
from agent import run_booking_flow
from calendar_utils import get_upcoming_events

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Google OAuth2 config
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events"
]
CREDENTIALS_PATH = "credentials.json"
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")

# FastAPI app
app = FastAPI(
    title="Calendar Agent API",
    description="AI-powered calendar assistant with Google Calendar integration",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    reply: str
    timestamp: datetime

@app.get("/")
def root():
    return {"message": "Backend is working!"}
# Auth URL route
@app.get("/auth/url")
def get_auth_url(user_id: str):
    state = f"{uuid.uuid4()}:{user_id}"
    flow = Flow.from_client_secrets_file(
        CREDENTIALS_PATH,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
        state=state
    )
    return {"auth_url": auth_url}

# OAuth callback
@app.get("/auth/callback")
def auth_callback(code: str = Query(...), state: str = Query(...)):
    try:
        user_id = state.split(":")[1]

        auth_flow = Flow.from_client_secrets_file(
            CREDENTIALS_PATH,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        auth_flow.fetch_token(code=code)

        creds = auth_flow.credentials

        existing = supabase.table("google_tokens").select("*").eq("user_id", user_id).single().execute()
        existing_refresh = existing.data["refresh_token"] if existing.data else None

        refresh_token_to_store = creds.refresh_token or existing_refresh

        if not refresh_token_to_store:
            raise HTTPException(status_code=400, detail="Google did not return a refresh token. Try revoking access and logging in again.")

        supabase.table("google_tokens").upsert({
            "user_id": user_id,
            "access_token": creds.token,
            "refresh_token": refresh_token_to_store,
            "token_expiry": creds.expiry.isoformat(),
            "scope": creds.scopes,
            "token_type": creds.token_uri
        }).execute()

        return RedirectResponse(url=f"http://localhost:8501?user_id={user_id}&auth_success=true")

    except Exception as e:
        logger.error(f"Auth callback failed: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed.")

# ✅ Check existing auth token

@app.get("/auth/check", summary="Check if user is authenticated")
def check_auth(user_id: str = Query(..., description="User ID to check")):
    try:
        result = supabase.table("google_tokens").select("refresh_token").eq("user_id", user_id).single().execute()
        refresh_token = result.data.get("refresh_token") if result.data else None
        return {"authenticated": bool(refresh_token)}
    except Exception as e:
        logger.error(f"Auth check failed: {e}")
        return {"authenticated": False}


# Chat route
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        logger.info(f"[CHAT] From user_id={request.user_id}: {request.message}")
        reply = run_booking_flow(request.user_id, request.message)

        background_tasks.add_task(lambda: logger.info("[CHAT] Finished successfully"))

        return ChatResponse(reply=reply, timestamp=datetime.now())

    except Exception as e:
        logger.error(f"[CHAT] Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

# Fetch calendar events
@app.get("/calendar/events")
async def get_calendar_events(user_id: str, max_results: int = 10):
    try:
        events = get_upcoming_events(user_id=user_id, max_results=max_results)
        formatted_events = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            formatted_events.append({
                'id': event['id'],
                'summary': event.get('summary', 'No Title'),
                'start': start,
                'end': end,
                'description': event.get('description', ''),
                'location': event.get('location', ''),
                'htmlLink': event.get('htmlLink', ''),
                'status': event.get('status', 'confirmed')
            })
        return {
            "events": formatted_events,
            "count": len(formatted_events),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Fetching calendar events failed: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch events: {str(e)}")

# Entry point
def main():
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "true").lower() == "true"
    uvicorn.run("main:app", host=host, port=port, reload=debug, log_level="info")

if __name__ == "__main__":
    main()
