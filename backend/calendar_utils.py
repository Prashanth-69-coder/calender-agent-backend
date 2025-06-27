import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from pytz import timezone
from db import supabase

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events"
]
CREDENTIALS_PATH = 'credentials.json'
TIMEZONE = 'Asia/Kolkata'
IST = timezone(TIMEZONE)

def update_google_token(user_id: str, access_token: str, expiry: datetime):
    """Updates access token and expiry in Supabase."""
    supabase.table("google_tokens").update({
        "access_token": access_token,
        "token_expiry": expiry.isoformat()
    }).eq("user_id", user_id).execute()

def get_user_credentials(user_id: str) -> Credentials:
    """Fetch Google credentials from Supabase."""
    try:
        response = supabase.table("google_tokens").select("*").eq("user_id", user_id).single().execute()
        token_data = response.data

        creds = Credentials(
            token=token_data["access_token"],
            refresh_token=token_data["refresh_token"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv("GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
            scopes=SCOPES,
            expiry=datetime.fromisoformat(token_data["token_expiry"])
        )

        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            update_google_token(user_id, creds.token, creds.expiry)

        return creds

    except Exception as e:
        logger.error(f"\u274c Failed to retrieve credentials for user {user_id}: {e}")
        raise

def get_calendar_service(user_id: str):
    """Returns an authenticated Google Calendar service."""
    creds = get_user_credentials(user_id)
    return build('calendar', 'v3', credentials=creds)

def check_availability(user_id: str, start: datetime, end: datetime, calendar_id: str = 'primary') -> bool:
    """Checks if the calendar is free in the given time range."""
    try:
        service = get_calendar_service(user_id)
        body = {
            "timeMin": start.astimezone(IST).isoformat(),
            "timeMax": end.astimezone(IST).isoformat(),
            "items": [{"id": calendar_id}],
        }
        events_result = service.freebusy().query(body=body).execute()
        busy_times = events_result['calendars'][calendar_id].get('busy', [])

        for busy_period in busy_times:
            busy_start = datetime.fromisoformat(busy_period['start'].replace('Z', ''))
            busy_end = datetime.fromisoformat(busy_period['end'].replace('Z', ''))
            if (start < busy_end) and (end > busy_start):
                return False

        return True

    except Exception as e:
        logger.error(f"Error checking availability: {e}")
        raise

def book_event(user_id: str, start: datetime, end: datetime, summary: str = "Appointment",
               description: str = "Booked via AI agent", calendar_id: str = 'primary',
               attendees: Optional[List[str]] = None, location: Optional[str] = None) -> dict:
    """Books a new event in the calendar."""
    try:
        service = get_calendar_service(user_id)

        event: Dict[str, Any] = {
            'summary': summary,
            'description': description,
            'start': {'dateTime': start.astimezone(IST).isoformat(), 'timeZone': TIMEZONE},
            'end': {'dateTime': end.astimezone(IST).isoformat(), 'timeZone': TIMEZONE},
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 15},
                ],
            },
        }

        if location:
            event['location'] = location
        if attendees:
            event['attendees'] = [{'email': email} for email in attendees]

        created_event = service.events().insert(calendarId=calendar_id, body=event).execute()
        return created_event

    except Exception as e:
        logger.error(f"Error creating event: {e}")
        raise

def get_upcoming_events(user_id: str, max_results: int = 10, calendar_id: str = 'primary') -> List[dict]:
    """Fetches upcoming events for a user."""
    try:
        service = get_calendar_service(user_id)
        now = datetime.now(IST).isoformat()

        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        return events_result.get('items', [])

    except Exception as e:
        logger.error(f"Error getting upcoming events: {e}")
        raise

def delete_event(user_id: str, event_id: str, calendar_id: str = 'primary') -> bool:
    """Deletes an event from the calendar."""
    try:
        service = get_calendar_service(user_id)
        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        return True

    except Exception as e:
        logger.error(f"Error deleting event {event_id}: {e}")
        return False

def update_event(user_id: str, event_id: str, start: Optional[datetime] = None, end: Optional[datetime] = None,
                 summary: Optional[str] = None, description: Optional[str] = None,
                 calendar_id: str = 'primary') -> dict:
    """Updates an existing calendar event."""
    try:
        service = get_calendar_service(user_id)
        event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()

        if start:
            event['start'] = {'dateTime': start.astimezone(IST).isoformat(), 'timeZone': TIMEZONE}
        if end:
            event['end'] = {'dateTime': end.astimezone(IST).isoformat(), 'timeZone': TIMEZONE}
        if summary:
            event['summary'] = summary
        if description:
            event['description'] = description

        updated_event = service.events().update(calendarId=calendar_id, eventId=event_id, body=event).execute()
        return updated_event

    except Exception as e:
        logger.error(f"Error updating event {event_id}: {e}")
        raise

def find_free_slots(user_id: str, date: datetime, start_hour: int = 9, end_hour: int = 17,
                    duration_minutes: int = 30, calendar_id: str = 'primary') -> List[Tuple[datetime, datetime]]:
    """Finds available time slots in a day based on user's calendar."""
    try:
        service = get_calendar_service(user_id)

        day_start = date.replace(hour=start_hour, minute=0, second=0, microsecond=0).astimezone(IST)
        day_end = date.replace(hour=end_hour, minute=0, second=0, microsecond=0).astimezone(IST)
        duration = timedelta(minutes=duration_minutes)

        body = {
            "timeMin": day_start.isoformat(),
            "timeMax": day_end.isoformat(),
            "items": [{"id": calendar_id}],
        }

        events_result = service.freebusy().query(body=body).execute()
        busy_times = events_result['calendars'][calendar_id].get('busy', [])

        busy_periods = [(datetime.fromisoformat(b['start'].replace('Z', '')),
                         datetime.fromisoformat(b['end'].replace('Z', '')))
                        for b in busy_times]

        busy_periods.sort(key=lambda x: x[0])

        free_slots = []
        current_time = day_start

        for busy_start, busy_end in busy_periods:
            while current_time + duration <= busy_start:
                free_slots.append((current_time, current_time + duration))
                current_time += duration
            current_time = max(current_time, busy_end)

        while current_time + duration <= day_end:
            free_slots.append((current_time, current_time + duration))
            current_time += duration

        return free_slots

    except Exception as e:
        logger.error(f"Error finding free slots: {e}")
        raise
