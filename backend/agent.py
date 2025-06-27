import os
import json
import logging
import re
from datetime import datetime, timedelta
from typing import TypedDict, Optional, Dict, Any, List
from zoneinfo import ZoneInfo
import pytz
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

from groq import Groq
from langgraph.graph import StateGraph, END
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from calendar_utils import book_event, check_availability

# --- Enhanced Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Groq setup with error handling
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable is required")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    raise


# --- Enhanced State Schema
class AgentState(TypedDict):
    # Required fields
    user_id: str
    user_input: str

    # Optional fields
    user_timezone: Optional[str]
    extracted: Optional[dict]
    intent: Optional[str]
    output: Optional[str]
    error: Optional[str]
    confidence: Optional[float]
    suggestions: Optional[List[str]]
    context: Optional[Dict[str, Any]]


# --- Enhanced System Prompt
SYSTEM_PROMPT = """You are an advanced calendar assistant with natural language understanding capabilities.

Your job is to interpret user requests for calendar operations and extract structured information.

RESPOND ONLY with a valid JSON object using this structure:
{
  "slot": "YYYY-MM-DDTHH:MM:SS",
  "duration": 30,
  "title": "Meeting Title",
  "description": "Meeting Description",
  "timezone": "America/New_York",
  "recurrence": null,
  "priority": "normal",
  "attendees": [],
  "location": null,
  "confidence": 0.95
}

IMPORTANT RULES:
- Use ISO 8601 format for dates and times
- Default duration is 30 minutes if not specified
- Infer reasonable titles from context
- Set confidence between 0.0-1.0 based on clarity
- For recurring events, use "daily", "weekly", "monthly", "yearly"
- Priority can be "low", "normal", "high"
- If unclear, return {"confidence": 0.0}

EXAMPLES:
"Book a meeting tomorrow at 2pm" → {"slot": "2024-01-16T14:00:00", "duration": 60, "title": "Meeting", ...}
"Schedule dentist appointment next Monday 10am for 30 minutes" → {"slot": "2024-01-15T10:00:00", "duration": 30, "title": "Dentist Appointment", ...}
"""


# --- Smart Date/Time Parser
class SmartDateTimeParser:
    def __init__(self, user_timezone: str = "UTC"):
        self.user_timezone = user_timezone
        self.common_patterns = [
            r'(\d{1,2})[:/](\d{1,2})[:/](\d{2,4})',  # MM/DD/YYYY
            r'(\d{1,2})-(\d{1,2})-(\d{2,4})',  # MM-DD-YYYY
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        ]

    def parse_flexible_datetime(self, text: str, reference_time: datetime = None) -> datetime:
        """Parse flexible date/time expressions with better error handling"""
        if reference_time is None:
            reference_time = datetime.now(ZoneInfo(self.user_timezone))

        try:
            # Handle relative expressions
            if any(word in text.lower() for word in ['tomorrow', 'next', 'today', 'yesterday']):
                return self._parse_relative_date(text, reference_time)

            # Try dateutil parser first
            parsed = date_parser.parse(text, fuzzy=True, default=reference_time)

            # Ensure timezone awareness
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=ZoneInfo(self.user_timezone))

            return parsed

        except Exception as e:
            logger.warning(f"Failed to parse datetime '{text}': {e}")
            # Fallback to current time + 1 hour
            return reference_time + timedelta(hours=1)

    def _parse_relative_date(self, text: str, reference: datetime) -> datetime:
        """Parse relative date expressions"""
        text_lower = text.lower()

        if 'today' in text_lower:
            base_date = reference.replace(hour=9, minute=0, second=0, microsecond=0)
        elif 'tomorrow' in text_lower:
            base_date = (reference + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        elif 'yesterday' in text_lower:
            base_date = (reference - timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        elif 'next week' in text_lower:
            days_ahead = 7 - reference.weekday()
            base_date = (reference + timedelta(days=days_ahead)).replace(hour=9, minute=0, second=0, microsecond=0)
        else:
            base_date = reference

        # Extract time if present
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text_lower)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            period = time_match.group(3)

            if period == 'pm' and hour != 12:
                hour += 12
            elif period == 'am' and hour == 12:
                hour = 0

            base_date = base_date.replace(hour=hour, minute=minute)

        return base_date


# --- Enhanced Intent Classification


import re
from typing import Dict

def classify_intent(state: AgentState) -> AgentState:
    user_input = state["user_input"].lower().strip()
    context = state.get("context")

    # Explicit check questions take priority
    check_question_triggers = [
        "am i free", "do i have", "am i available", "do i have anything",
        "check if", "check i have", "is there a", "do i have a", "free at", "anything at"
    ]

    if any(phrase in user_input for phrase in check_question_triggers) or user_input.startswith("check"):
        state["intent"] = "check"
        state["confidence"] = 0.95
        state["extracted"] = None
        state["error"] = None
        state["output"] = None
        state["suggestions"] = None
        state["context"] = context
        return state

    # Intent pattern matching (fallback scoring)
    intent_patterns = {
        "check": {
            "patterns": ["free", "available", "slot", "check", "open", "when can"],
            "weight": 1.0
        },
        "book": {
            "patterns": ["book", "schedule", "set", "create", "add", "make", "plan", "arrange"],
            "weight": 1.0
        },
        "cancel": {
            "patterns": ["cancel", "delete", "remove", "unbook"],
            "weight": 1.0
        },
        "reschedule": {
            "patterns": ["reschedule", "move", "change", "shift", "update"],
            "weight": 1.0
        },
        "list": {
            "patterns": ["list", "show", "what's", "whats", "upcoming", "today", "tomorrow"],
            "weight": 0.9
        }
    }

    intent_scores = {}
    for intent, config in intent_patterns.items():
        matches = sum(1 for pattern in config["patterns"] if pattern in user_input)
        score = (matches / len(config["patterns"])) * config["weight"]
        if score > 0:
            intent_scores[intent] = score

    if not intent_scores:
        return {
            "user_id": state["user_id"],
            "user_input": state["user_input"],
            "user_timezone": state.get("user_timezone"),
            "intent": "unknown",
            "confidence": 0.0,
            "extracted": None,
            "error": None,
            "output": None,
            "suggestions": None,
            "context": context
        }

    best_intent = max(intent_scores, key=intent_scores.get)
    confidence = intent_scores[best_intent]

    return {
        "user_id": state["user_id"],
        "user_input": state["user_input"],
        "user_timezone": state.get("user_timezone"),
        "intent": best_intent,
        "confidence": confidence,
        "extracted": None,
        "error": None,
        "output": None,
        "suggestions": None,
        "context": context
    }


# --- Enhanced Information Extraction
def extract_meeting_info(state: AgentState) -> AgentState:
    """Enhanced extraction with validation and error handling"""
    try:
        # Skip extraction for certain intents
        if state["intent"] in ["list", "unknown"]:
            return {
                "user_id": state["user_id"],
                "user_input": state["user_input"],
                "user_timezone": state.get("user_timezone"),
                "intent": state.get("intent"),
                "confidence": state.get("confidence"),
                "extracted": {},
                "error": None,
                "output": None,
                "suggestions": None,
                "context": None
            }

        # Prepare enhanced context
        current_time = datetime.now()
        context = {
            "current_time": current_time.isoformat(),
            "user_timezone": state.get("user_timezone", "UTC"),
            "intent": state["intent"]
        }

        enhanced_prompt = f"{SYSTEM_PROMPT}\n\nCurrent context: {json.dumps(context)}"

        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=enhanced_prompt),
            ChatCompletionUserMessageParam(role="user", content=state["user_input"])
        ]

        # Multiple attempts with different temperatures
        for attempt, temp in enumerate([0.1, 0.3, 0.5], 1):
            try:
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=messages,
                    temperature=temp,
                    max_tokens=512,
                    top_p=0.9
                )

                raw = response.choices[0].message.content.strip()

                # Clean up response
                if raw.startswith("```json"):
                    raw = raw[7:-3]
                elif raw.startswith("```"):
                    raw = raw[3:-3]

                parsed = json.loads(raw)

                # Validate and enhance extracted data
                validated = validate_and_enhance_extraction(parsed, state)

                if validated["confidence"] >= 0.5:
                    return {
                        "user_id": state["user_id"],
                        "user_input": state["user_input"],
                        "user_timezone": state.get("user_timezone"),
                        "intent": state.get("intent"),
                        "confidence": state.get("confidence"),
                        "extracted": validated,
                        "error": None,
                        "output": None,
                        "suggestions": None,
                        "context": context
                    }

            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt} - JSON decode error: {e}")
                continue
            except Exception as e:
                logger.warning(f"Attempt {attempt} - Extraction error: {e}")
                continue

        # If all attempts failed, try fallback extraction
        fallback_data = fallback_extraction(state["user_input"], state["intent"])
        return {
            "user_id": state["user_id"],
            "user_input": state["user_input"],
            "user_timezone": state.get("user_timezone"),
            "intent": state.get("intent"),
            "confidence": state.get("confidence"),
            "extracted": fallback_data,
            "error": None,
            "output": None,
            "suggestions": None,
            "context": context
        }

    except Exception as e:
        logger.error(f"❌ Failed to extract info: {e}")
        return {
            "user_id": state["user_id"],
            "user_input": state["user_input"],
            "user_timezone": state.get("user_timezone"),
            "intent": state.get("intent"),
            "confidence": state.get("confidence"),
            "extracted": None,
            "error": str(e),
            "output": None,
            "suggestions": None,
            "context": None
        }


def validate_and_enhance_extraction(data: dict, state: AgentState) -> dict:
    """Validate and enhance extracted data"""
    parser = SmartDateTimeParser(state.get("user_timezone", "UTC"))

    # Set defaults
    enhanced = {
        "slot": None,
        "duration": 30,
        "title": "New Event",
        "description": "",
        "timezone": state.get("user_timezone", "UTC"),
        "recurrence": None,
        "priority": "normal",
        "attendees": [],
        "location": None,
        "confidence": data.get("confidence", 0.5)
    }

    # Update with extracted data
    enhanced.update(data)

    # Validate and fix datetime
    if enhanced["slot"]:
        try:
            if not enhanced["slot"].endswith("Z") and "T" in enhanced["slot"]:
                parsed_time = parser.parse_flexible_datetime(enhanced["slot"])
                enhanced["slot"] = parsed_time.isoformat()
        except Exception as e:
            logger.warning(f"Datetime validation failed: {e}")
            enhanced["confidence"] *= 0.5

    # Validate duration
    try:
        enhanced["duration"] = max(15, min(480, int(enhanced["duration"])))  # 15 min to 8 hours
    except (ValueError, TypeError):
        enhanced["duration"] = 30
        enhanced["confidence"] *= 0.8

    # Business hours validation
    if enhanced["slot"]:
        try:
            slot_time = datetime.fromisoformat(enhanced["slot"].replace("Z", "+00:00"))
            if slot_time.hour < 6 or slot_time.hour > 22:
                enhanced["confidence"] *= 0.7
                if not enhanced.get("suggestions"):
                    enhanced["suggestions"] = ["Consider scheduling during business hours (6 AM - 10 PM)"]
        except Exception:
            pass

    return enhanced


def fallback_extraction(user_input: str, intent: str) -> dict:
    """Fallback extraction using pattern matching"""
    parser = SmartDateTimeParser()

    # Try to extract basic info using regex
    time_patterns = [
        r'(\d{1,2}:\d{2}(?:\s*[ap]m)?)',
        r'(\d{1,2}\s*[ap]m)',
        r'(at\s+\d{1,2})',
    ]

    duration_patterns = [
        r'(\d+)\s*(?:minutes?|mins?|hours?|hrs?)'
    ]

    extracted_time = None
    for pattern in time_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            try:
                extracted_time = parser.parse_flexible_datetime(user_input)
                break
            except Exception:
                continue

    duration = 30
    for pattern in duration_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            duration = int(match.group(1))
            if 'hour' in user_input.lower():
                duration *= 60
            break

    return {
        "slot": extracted_time.isoformat() if extracted_time else None,
        "duration": duration,
        "title": "New Event",
        "description": f"Event created from: {user_input}",
        "confidence": 0.3,
        "suggestions": ["Please provide more specific date/time information"]
    }


# --- Enhanced Booking Function
def book_calendar_event(state: AgentState) -> AgentState:
    """Enhanced booking with comprehensive validation"""
    try:
        details = state["extracted"]

        if not details or not details.get("slot"):
            return {
                "user_id": state["user_id"],
                "user_input": state["user_input"],
                "user_timezone": state.get("user_timezone"),
                "intent": state.get("intent"),
                "confidence": state.get("confidence"),
                "extracted": state.get("extracted"),
                "error": None,
                "output": "❌ Unable to extract event details. Please provide date and time.",
                "suggestions": None,
                "context": state.get("context")
            }

        # Parse datetime with timezone handling
        try:
            start = datetime.fromisoformat(details["slot"].replace("Z", "+00:00"))
        except ValueError as e:
            return {
                "user_id": state["user_id"],
                "user_input": state["user_input"],
                "user_timezone": state.get("user_timezone"),
                "intent": state.get("intent"),
                "confidence": state.get("confidence"),
                "extracted": state.get("extracted"),
                "error": None,
                "output": f"❌ Invalid date/time format: {e}",
                "suggestions": None,
                "context": state.get("context")
            }

        end = start + timedelta(minutes=details["duration"])

        # Comprehensive validation
        validation_result = validate_booking_request(state["user_id"], start, end, details)
        if validation_result["valid"] is False:
            return {
                "user_id": state["user_id"],
                "user_input": state["user_input"],
                "user_timezone": state.get("user_timezone"),
                "intent": state.get("intent"),
                "confidence": state.get("confidence"),
                "extracted": state.get("extracted"),
                "error": None,
                "output": validation_result["message"],
                "suggestions": None,
                "context": state.get("context")
            }

        # Check availability with buffer time
        buffer_minutes = 5
        buffer_start = start - timedelta(minutes=buffer_minutes)
        buffer_end = end + timedelta(minutes=buffer_minutes)

        is_free = check_availability(state["user_id"], buffer_start, buffer_end)
        if not is_free:
            suggestions = generate_alternative_slots(state["user_id"], start, details["duration"])
            suggestion_text = ""
            if suggestions:
                suggestion_text = f"\n\n🔄 Alternative times available:\n" + "\n".join(suggestions[:3])

            return {
                "user_id": state["user_id"],
                "user_input": state["user_input"],
                "user_timezone": state.get("user_timezone"),
                "intent": state.get("intent"),
                "confidence": state.get("confidence"),
                "extracted": state.get("extracted"),
                "error": None,
                "output": f"❌ You have a conflicting event during this time.{suggestion_text}",
                "suggestions": suggestions,
                "context": state.get("context")
            }

        # Book the event
        event = book_event(
            user_id=state["user_id"],
            start=start,
            end=end,
            summary=details["title"],
            description=details["description"],
            location=details.get("location"),
            attendees=details.get("attendees", [])
        )

        # Generate success message
        formatted_time = start.strftime("%B %d, %Y at %I:%M %p")
        duration_text = f"{details['duration']} minutes" if details['duration'] != 60 else "1 hour"

        success_msg = f"✅ Event booked successfully!\n"
        success_msg += f"📅 {details['title']}\n"
        success_msg += f"🕒 {formatted_time} ({duration_text})\n"

        if event.get('htmlLink'):
            success_msg += f"🔗 [Open in Google Calendar]({event['htmlLink']})"

        return {
            "user_id": state["user_id"],
            "user_input": state["user_input"],
            "user_timezone": state.get("user_timezone"),
            "intent": state.get("intent"),
            "confidence": state.get("confidence"),
            "extracted": state.get("extracted"),
            "error": None,
            "output": success_msg,
            "suggestions": None,
            "context": state.get("context")
        }

    except Exception as e:
        logger.error(f"❌ Failed to book event: {e}")
        return {
            "user_id": state["user_id"],
            "user_input": state["user_input"],
            "user_timezone": state.get("user_timezone"),
            "intent": state.get("intent"),
            "confidence": state.get("confidence"),
            "extracted": state.get("extracted"),
            "error": f"Booking failed: {str(e)}",
            "output": None,
            "suggestions": None,
            "context": state.get("context")
        }


def validate_booking_request(user_id: str, start: datetime, end: datetime, details: dict) -> dict:
    """Comprehensive booking validation"""
    now = datetime.now(start.tzinfo)

    # Past date check
    if start < now:
        return {"valid": False, "message": "❌ Cannot book events in the past."}

    # Duration validation
    if end <= start:
        return {"valid": False, "message": "❌ Invalid duration - end time must be after start time."}

    duration_minutes = (end - start).total_seconds() / 60
    if duration_minutes < 15:
        return {"valid": False, "message": "❌ Minimum event duration is 15 minutes."}

    if duration_minutes > 480:  # 8 hours
        return {"valid": False, "message": "❌ Maximum event duration is 8 hours."}

    # Weekend/holiday warning (but not blocking)
    if start.weekday() >= 5:  # Saturday or Sunday
        return {"valid": True, "message": "⚠️ Note: This is scheduled for a weekend."}

    # Late night/early morning warning
    if start.hour < 6 or start.hour >= 22:
        return {"valid": True, "message": "⚠️ Note: This is scheduled outside typical business hours."}

    return {"valid": True, "message": "✅ Booking validation passed."}


def generate_alternative_slots(user_id: str, preferred_start: datetime, duration: int) -> List[str]:
    """Generate alternative time slots when preferred time is busy"""
    alternatives = []

    # Check next few hours on the same day
    for hours_offset in [1, 2, 3]:
        alt_start = preferred_start + timedelta(hours=hours_offset)
        alt_end = alt_start + timedelta(minutes=duration)

        if check_availability(user_id, alt_start, alt_end):
            alternatives.append(f"• {alt_start.strftime('%I:%M %p')} - {alt_end.strftime('%I:%M %p')}")

    # Check same time next day
    next_day = preferred_start + timedelta(days=1)
    if check_availability(user_id, next_day, next_day + timedelta(minutes=duration)):
        alternatives.append(f"• Tomorrow at {next_day.strftime('%I:%M %p')}")

    return alternatives


# --- Enhanced Availability Check
def check_availability_slot(state: AgentState) -> AgentState:
    """Enhanced availability checking with detailed feedback"""
    print("[CHECK] Reached check_availability()")
    try:
        details = state["extracted"]

        if not details or not details.get("slot"):
            return {
                "user_id": state["user_id"],
                "user_input": state["user_input"],
                "user_timezone": state.get("user_timezone"),
                "intent": state.get("intent"),
                "confidence": state.get("confidence"),
                "extracted": state.get("extracted"),
                "error": None,
                "output": "❌ Please specify when you'd like to check availability.",
                "suggestions": None,
                "context": state.get("context")
            }

        start = datetime.fromisoformat(details["slot"].replace("Z", "+00:00"))
        end = start + timedelta(minutes=details["duration"])

        # Check availability
        is_free = check_availability(state["user_id"], start, end)

        # Generate detailed response
        formatted_time = start.strftime("%B %d, %Y at %I:%M %p")
        duration_text = f"{details['duration']} minutes" if details['duration'] != 60 else "1 hour"

        if is_free:
            message = f"✅ You're free on {formatted_time} for {duration_text}!"

            # Add helpful context
            day_name = start.strftime("%A")
            if start.weekday() >= 5:
                message += f"\n📅 Note: This is a {day_name}."
        else:
            message = f"❌ You're busy on {formatted_time}."

            # Suggest alternatives
            alternatives = generate_alternative_slots(state["user_id"], start, details["duration"])
            if alternatives:
                message += f"\n\n🔄 Here are some free slots:\n" + "\n".join(alternatives[:3])

        return {
            "user_id": state["user_id"],
            "user_input": state["user_input"],
            "user_timezone": state.get("user_timezone"),
            "intent": state.get("intent"),
            "confidence": state.get("confidence"),
            "extracted": state.get("extracted"),
            "error": None,
            "output": message,
            "suggestions": None,
            "context": state.get("context")
        }

    except Exception as e:
        logger.error(f"❌ Availability check failed: {e}")
        return {
            "user_id": state["user_id"],
            "user_input": state["user_input"],
            "user_timezone": state.get("user_timezone"),
            "intent": state.get("intent"),
            "confidence": state.get("confidence"),
            "extracted": state.get("extracted"),
            "error": f"Availability check failed: {str(e)}",
            "output": None,
            "suggestions": None,
            "context": state.get("context")
        }


# --- New: List Events Function
def list_events(state: AgentState) -> AgentState:
    """List upcoming events"""
    try:
        # This would need to be implemented in calendar_utils
        # For now, return a placeholder
        return {
            "user_id": state["user_id"],
            "user_input": state["user_input"],
            "user_timezone": state.get("user_timezone"),
            "intent": state.get("intent"),
            "confidence": state.get("confidence"),
            "extracted": state.get("extracted"),
            "error": None,
            "output": "📅 Event listing feature coming soon!",
            "suggestions": None,
            "context": state.get("context")
        }
    except Exception as e:
        return {
            "user_id": state["user_id"],
            "user_input": state["user_input"],
            "user_timezone": state.get("user_timezone"),
            "intent": state.get("intent"),
            "confidence": state.get("confidence"),
            "extracted": state.get("extracted"),
            "error": f"Availability check failed: {str(e)}",
            "output": None,
            "suggestions": None,
            "context": state.get("context")
        }

# --- Enhanced Fallback
def handle_unknown_intent(state: AgentState) -> AgentState:
    """Enhanced fallback with helpful suggestions"""
    confidence = state.get("confidence", 0.0)

    if confidence > 0.3:
        message = "❓ I'm not quite sure what you want to do. "
    else:
        message = "❓ I didn't understand that. "

    message += "Here's what I can help you with:\n"
    message += "• 📅 **Book events**: 'Schedule a meeting tomorrow at 2pm'\n"
    message += "• 🔍 **Check availability**: 'Am I free Monday at 10am?'\n"
    message += "• 📋 **List events**: 'What's on my calendar today?'\n"
    message += "• ❌ **Cancel events**: 'Cancel my 3pm meeting'\n"
    message += "• 🔄 **Reschedule**: 'Move my meeting to 4pm'"

    return {**state, "output": message, "error": None}


# --- Enhanced Graph Builder
builder = StateGraph(AgentState)

# Add all nodes
builder.add_node("classify", classify_intent)
builder.add_node("extract", extract_meeting_info)
builder.add_node("book", book_calendar_event)
builder.add_node("check", check_availability_slot)
builder.add_node("list", list_events)
builder.add_node("unknown", handle_unknown_intent)

# Set entry point
builder.set_entry_point("classify")

# Add edges
builder.add_edge("classify", "extract")


# Enhanced conditional routing
def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent")
    confidence = state.get("confidence")

    print(f"[ROUTER] Intent: {intent}, Confidence: {confidence}")

    # If intent exists and confidence is None, assume test override
    if intent and confidence is None:
        return intent

    if not intent or confidence is None or confidence < 0.6:
        return "unknown"

    return intent



builder.add_conditional_edges("extract", route_by_intent, {
    "book": "book",
    "check": "check",
    "list": "list",
    "cancel": "unknown",  # Placeholder for future implementation
    "reschedule": "unknown",  # Placeholder for future implementation
    "unknown": "unknown",
})

# Add terminal edges
for node in ["book", "check", "list", "unknown"]:
    builder.add_edge(node, END)

graph = builder.compile()


def run_booking_flow(user_id: str, user_input: str, user_timezone: str = "UTC") -> str:


    if not user_id or not user_input:
        return "❌ Invalid input: user_id and user_input are required."

    if len(user_input.strip()) < 3:
        return "❌ Please provide a more detailed request."


    state: AgentState = {
        "user_id": user_id,
        "user_input": user_input.strip(),
        "user_timezone": user_timezone,
        "extracted": None,
        "intent": None,
        "output": None,
        "error": None,
        "confidence": None,
        "suggestions": None,
        "context": {}
    }

    try:

        state = classify_intent(state)
        logger.info(f"[INTENT] User: {user_id} | Intent: {state['intent']} | Confidence: {state.get('confidence')}")


        if state["intent"] == "unknown":
            return "🤔 I couldn't understand your request. Try rephrasing."


        result = graph.invoke(state)


        if result.get("error"):
            logger.error(f"Agent error for user {user_id}: {result['error']}")
            return f"❌ I encountered an issue: {result['error']}\n\nPlease try rephrasing your request or contact support."

        output = result.get("output")
        if not output:
            return "❌ I couldn't process your request. Please try again with more specific details."


        suggestions = result.get("suggestions", [])
        if suggestions:
            output += "\n\n💡 **Suggestions:**\n" + "\n".join(f"• {s}" for s in suggestions)

        return output

    except Exception as e:
        logger.exception(f"Unexpected error in booking flow for {user_id}: {e}")
        return "❌ An unexpected error occurred. Please try again later or contact support."

