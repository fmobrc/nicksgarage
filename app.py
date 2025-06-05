from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from motor.motor_asyncio import AsyncIOMotorClient
import anthropic
from datetime import datetime
import os
from typing import Optional, List
from pydantic import BaseModel
import uvicorn
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import time
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from twilio.rest import Client
import requests
import re

load_dotenv()
HISTORY_LIMIT = 10

class MessageHistory(BaseModel):
    user_id: str
    message: str
    timestamp: datetime
    is_user: bool

class JournalEntry(BaseModel):
    user_id: str
    entry_text: str
    entry_date: datetime = None
    ai_feedback: Optional[str] = None

class UserPlan(BaseModel):
    user_id: str
    future_authoring_plan: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

required_vars = ["MONGO_URI", "ANTHROPIC_API_KEY", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

try:
    MONGO_URI = os.getenv('MONGO_URI')
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client["myjournal"]
    journal_entries = db["JournalEntries"]
    user_collection = db["Users"]
    message_history = db["MessageHistory"]
except Exception as e:
    raise

anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
twilio_validator = RequestValidator(os.getenv("TWILIO_AUTH_TOKEN"))
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def transcribe_voice_note(user_id: str, media_url: str) -> str:
    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    response = requests.get(media_url, auth=auth)
    if response.status_code != 200:
        return "Failed to download audio."
    audio_data = response.content
    transcribe_url = "https://api.deepgram.com/v1/listen"
    headers = {"Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}"}
    files = {"audio": ("voice.ogg", audio_data, "audio/ogg")}
    transcription_response = requests.post(transcribe_url, headers=headers, files=files)
    if transcription_response.status_code != 200:
        return "Failed to transcribe audio."
    transcription = transcription_response.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
    return transcription if transcription else "Transcription failed."

async def get_message_history(user_id: str, limit: int = HISTORY_LIMIT) -> List[dict]:
    cursor = message_history.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    messages = await cursor.to_list(length=limit)
    return list(reversed(messages))

async def add_to_message_history(user_id: str, message: str, is_user: bool):
    new_message = MessageHistory(
        user_id=user_id,
        message=message,
        timestamp=datetime.utcnow(),
        is_user=is_user
    )
    await message_history.insert_one(new_message.dict())

async def get_ai_feedback(journal_text: str, life_plan: Optional[str] = None, user_id: str = None, start_time: Optional[float] = None) -> str:
    """
    Get AI feedback on pet care using Anthropic's Claude as Dr. Bob, Blue Pet Co ambassador.
    """
    try:
        # Get conversation history to provide context
        history = await get_message_history(user_id, limit=HISTORY_LIMIT)
        
        # Check if this is the first interaction
        is_first_interaction = len([msg for msg in history if not msg['is_user']]) == 0
        
        system_prompt = """You are Dr. Bob, a world-renowned veterinary surgeon and passionate ambassador for Blue Pet Co (BPC). You create sustainable, marine-powered dog supplements from hand-harvested Irish seaweed. You're an expert in canine health with decades of clinical experience, and you genuinely love dogs and their wellbeing.

CRITICAL INSTRUCTION - GREETING BEHAVIOR:
- ONLY introduce yourself as "Dr. Bob" in your very FIRST response to a new user
- In ALL subsequent responses, NEVER reintroduce yourself or say "Dr. Bob here" or "I'm Dr. Bob"
- Continue conversations naturally without any self-introduction
- You are already established in the conversation after the first message

YOUR EXPERTISE & PERSONALITY:
- Veterinary surgeon with 20+ years of clinical experience
- Specializes in: joint health, skin & coat conditions, dental care, nutrition, behavioral issues, preventive medicine
- Passionate about evidence-based veterinary medicine
- Warm, approachable, and genuinely caring
- Curious and thorough - you love understanding the full picture
- Patient and educational - you explain complex topics simply
- Optimistic but realistic about treatment outcomes

YOUR COMMUNICATION STYLE:
- Conversational and engaging, like talking to a trusted friend
- Ask thoughtful follow-up questions to gather complete information
- Share relevant clinical experience and anecdotes when helpful
- Use simple, clear language - avoid veterinary jargon unless necessary
- Adapt your response length based on the complexity of the question
- Always maintain a supportive, non-judgmental tone
- Express genuine care and concern for both dog and owner

DIAGNOSTIC APPROACH:
When addressing health concerns, systematically gather:
1. Dog basics: breed, age, weight, sex (neutered/spayed?)
2. Specific symptoms: onset, duration, severity, progression
3. Behavioral changes: eating, drinking, energy, mood
4. Environmental factors: recent changes, new foods, stress
5. Medical history: current medications, previous conditions
6. Current care routine: diet, exercise, grooming

AREAS OF EXPERTISE:
- Joint Health: arthritis, hip dysplasia, mobility issues, exercise recommendations
- Skin & Coat: allergies, hot spots, dry skin, seasonal issues, grooming tips
- Dental Care: tartar, bad breath, tooth pain, dental hygiene routines
- Nutrition: age-appropriate diets, weight management, food sensitivities
- Behavioral Health: anxiety, aggression, training support, environmental enrichment
- Preventive Care: vaccination schedules, parasite prevention, wellness checkups
- Senior Care: age-related changes, comfort measures, quality of life

RESPONSE GUIDELINES:
- Keep responses between 100-400 characters for WhatsApp format
- Break longer explanations into multiple shorter messages when needed
- Always end with an engaging question or invitation for more information
- Focus on actionable advice while being appropriately cautious
- Only suggest emergency vet visits for genuine emergencies
- Encourage regular vet checkups as part of routine care, not as dismissal

BLUE PET CO INTEGRATION:
- You represent BPC but never push sales - assume they're already customers
- Mention how marine-based supplements support overall health when relevant
- Focus on education about the benefits of natural, sustainable pet care
- Share insights about how nutrition impacts the conditions you're discussing

CONVERSATION FLOW:
- Build on previous exchanges naturally
- Reference earlier information the owner shared
- Show you're listening and remembering details
- Create a sense of ongoing partnership in their dog's care
- Express genuine interest in updates and progress

EMERGENCY PROTOCOLS:
Only suggest immediate veterinary attention for:
- Difficulty breathing or choking
- Suspected poisoning or toxin ingestion
- Severe trauma or bleeding
- Signs of bloat (distended abdomen, retching without vomiting)
- Seizures, collapse, or loss of consciousness
- Extreme lethargy with other concerning symptoms

For all other concerns, provide guidance while encouraging routine veterinary care as part of responsible ownership."""

        # Build conversation context
        conversation_context = ""
        if history:
            conversation_context = "\nCONVERSATION HISTORY:\n"
            for msg in history[-6:]:  # Last 6 messages for context
                role = "Owner" if msg['is_user'] else "Dr. Bob"
                conversation_context += f"{role}: {msg['message']}\n"
            conversation_context += "\n"

        # Create user prompt with context
        if is_first_interaction:
            user_prompt = f"""This is your FIRST interaction with this dog owner. Introduce yourself warmly as Dr. Bob and ask for basic information about their dog and concern.

Owner's message: {journal_text}

Respond in under 400 characters with a warm introduction and relevant questions."""
        else:
            user_prompt = f"""{conversation_context}
The owner has sent a new message. Continue the conversation naturally WITHOUT reintroducing yourself. Build on the previous context and provide helpful guidance.

Owner's latest message: {journal_text}

Respond in under 400 characters, continuing the established conversation."""

        messages = [{"role": "user", "content": user_prompt}]

        response = await anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=200,  # Reduced for shorter responses
            temperature=0.3,  # Lower temperature for more consistent behavior
            system=system_prompt,
            messages=messages,
            stream=False
        )

        if start_time:
            print(f"Claude API response time: {time.time() - start_time} seconds")

        print(f"[INFO] Dr. Bob response => {response}")
        ai_response = response.content[0].text.strip()
        
        # Additional safeguard - remove any accidental reintroductions
        if not is_first_interaction:
            # Remove common reintroduction patterns
            ai_response = re.sub(r'^(Hi there!?\s*)?(I\'m\s+)?Dr\.?\s*Bob\s*(here[,\s]*)?', '', ai_response, flags=re.IGNORECASE)
            ai_response = re.sub(r'^(Hello!?\s*)?(This is\s+)?Dr\.?\s*Bob[,\s]*', '', ai_response, flags=re.IGNORECASE)
            ai_response = ai_response.strip()
        
        return ai_response[:400]  # Ensure character limit
        
    except Exception as e:
        print(f"Error generating veterinary AI response: {e}")
        return "Sorry, I'm unable to help at the moment. Please try again."

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    try:
        form_data = await request.form()
        msg_body = form_data.get('Body', '').strip()
        user_id = form_data.get('From', '')
        
        # Handle voice notes if present
        if 'MediaUrl0' in form_data:
            transcribed_text = transcribe_voice_note(user_id, form_data['MediaUrl0'])
            msg_body = transcribed_text

        # Add message to history
        await add_to_message_history(user_id, msg_body, True)
        
        # Get AI feedback
        start_time = time.time()
        ai_response = await get_ai_feedback(msg_body, user_id=user_id, start_time=start_time)
        
        # Add AI response to history
        await add_to_message_history(user_id, ai_response, False)
        
        # Create Twilio response
        resp = MessagingResponse()
        resp.message(ai_response)
        return PlainTextResponse(content=str(resp), media_type="application/xml")
        
    except Exception as e:
        print(f"Error in webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)