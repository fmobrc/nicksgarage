from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from motor.motor_asyncio import AsyncIOMotorClient
import anthropic
from datetime import datetime, timedelta
import os
from typing import Optional, List, Dict
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
from bs4 import BeautifulSoup
import json
import asyncio
from urllib.parse import urljoin, urlparse

load_dotenv()
HISTORY_LIMIT = 15

class MessageHistory(BaseModel):
    user_id: str
    message: str
    timestamp: datetime
    is_user: bool

class CustomerPreference(BaseModel):
    user_id: str
    phone_number: str
    preferred_makes: List[str] = []
    preferred_models: List[str] = []
    max_budget: Optional[int] = None
    min_year: Optional[int] = None
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None
    body_type: Optional[str] = None
    max_mileage: Optional[int] = None
    created_at: datetime = None
    updated_at: datetime = None

class CarListing(BaseModel):
    listing_id: str
    make: str
    model: str
    year: int
    price: int
    mileage: int
    fuel_type: str
    transmission: str
    body_type: str
    color: str
    nct_expiry: Optional[str] = None
    tax_expiry: Optional[str] = None
    previous_owners: Optional[int] = None
    description: str
    images: List[str] = []
    url: str
    scraped_at: datetime
    available: bool = True

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
    db = mongo_client["brendan_conlon_autos"]
    message_history = db["MessageHistory"]
    customer_preferences = db["CustomerPreferences"]
    car_listings = db["CarListings"]
    customer_interactions = db["CustomerInteractions"]
except Exception as e:
    raise

anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
twilio_validator = RequestValidator(os.getenv("TWILIO_AUTH_TOKEN"))
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Website scraping functions
async def scrape_brendan_conlon_inventory():
    """Scrape the Brendan Conlon Automobiles website for current inventory"""
    try:
        base_url = "https://www.brendanconlanautomobiles.net"
        
        # Headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Get the main inventory page
        response = requests.get(f"{base_url}/used-cars", headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        cars = []
        
        # Extract car listings (this will need to be customized based on actual website structure)
        car_elements = soup.find_all('div', class_=['car-listing', 'vehicle-item', 'car-card'])
        
        for car_element in car_elements:
            try:
                car_data = extract_car_details(car_element, base_url)
                if car_data:
                    cars.append(car_data)
            except Exception as e:
                print(f"Error extracting car details: {e}")
                continue
        
        # Update database with fresh inventory
        await update_inventory_database(cars)
        return cars
        
    except Exception as e:
        print(f"Error scraping inventory: {e}")
        return []

def extract_car_details(car_element, base_url):
    """Extract details from a single car listing element"""
    try:
        # This function will need to be customized based on the actual website structure
        # Here's a generic template that can be adapted
        
        car_data = {}
        
        # Extract basic info (customize selectors based on actual website)
        title = car_element.find(['h2', 'h3', '.car-title', '.vehicle-title'])
        if title:
            title_text = title.get_text(strip=True)
            # Parse title like "2019 BMW 3 Series 320d"
            parts = title_text.split()
            if len(parts) >= 3:
                car_data['year'] = int(parts[0]) if parts[0].isdigit() else None
                car_data['make'] = parts[1]
                car_data['model'] = ' '.join(parts[2:])
        
        # Extract price
        price_element = car_element.find(['.price', '.car-price', '.vehicle-price'])
        if price_element:
            price_text = re.sub(r'[^\d]', '', price_element.get_text())
            car_data['price'] = int(price_text) if price_text else None
        
        # Extract mileage
        mileage_element = car_element.find(['.mileage', '.odometer'])
        if mileage_element:
            mileage_text = re.sub(r'[^\d]', '', mileage_element.get_text())
            car_data['mileage'] = int(mileage_text) if mileage_text else None
        
        # Extract fuel type
        fuel_element = car_element.find(['.fuel', '.fuel-type'])
        if fuel_element:
            car_data['fuel_type'] = fuel_element.get_text(strip=True)
        
        # Extract transmission
        trans_element = car_element.find(['.transmission', '.gearbox'])
        if trans_element:
            car_data['transmission'] = trans_element.get_text(strip=True)
        
        # Extract car URL
        link_element = car_element.find('a')
        if link_element and link_element.get('href'):
            car_data['url'] = urljoin(base_url, link_element.get('href'))
            car_data['listing_id'] = car_data['url'].split('/')[-1]
        
        # Extract images
        img_elements = car_element.find_all('img')
        car_data['images'] = [urljoin(base_url, img.get('src')) for img in img_elements if img.get('src')]
        
        car_data['scraped_at'] = datetime.utcnow()
        car_data['available'] = True
        
        return car_data
        
    except Exception as e:
        print(f"Error extracting car details: {e}")
        return None

async def update_inventory_database(cars):
    """Update the database with scraped car inventory"""
    try:
        # Mark all existing cars as potentially unavailable
        await car_listings.update_many({}, {"$set": {"available": False}})
        
        for car_data in cars:
            if car_data.get('listing_id'):
                # Update or insert car listing
                await car_listings.update_one(
                    {"listing_id": car_data['listing_id']},
                    {"$set": {**car_data, "available": True}},
                    upsert=True
                )
        
        print(f"Updated inventory with {len(cars)} cars")
        
    except Exception as e:
        print(f"Error updating inventory database: {e}")

def transcribe_voice_note(user_id: str, media_url: str) -> str:
    """Transcribe voice messages using Deepgram"""
    try:
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
        
    except Exception as e:
        print(f"Error transcribing voice note: {e}")
        return "Failed to transcribe audio."

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

async def get_customer_preferences(user_id: str) -> Optional[dict]:
    """Get stored customer preferences"""
    prefs = await customer_preferences.find_one({"user_id": user_id})
    return prefs

async def update_customer_preferences(user_id: str, phone_number: str, preferences: dict):
    """Update customer preferences based on conversation"""
    await customer_preferences.update_one(
        {"user_id": user_id},
        {"$set": {
            "phone_number": phone_number,
            "updated_at": datetime.utcnow(),
            **preferences
        }},
        upsert=True
    )

async def search_inventory(preferences: dict = None, query: str = None) -> List[dict]:
    """Search current inventory based on preferences or query"""
    try:
        search_filter = {"available": True}
        
        if preferences:
            if preferences.get('max_budget'):
                search_filter['price'] = {"$lte": preferences['max_budget']}
            if preferences.get('min_year'):
                search_filter['year'] = {"$gte": preferences['min_year']}
            if preferences.get('preferred_makes'):
                search_filter['make'] = {"$in": [make.lower() for make in preferences['preferred_makes']]}
            if preferences.get('fuel_type'):
                search_filter['fuel_type'] = {"$regex": preferences['fuel_type'], "$options": "i"}
            if preferences.get('max_mileage'):
                search_filter['mileage'] = {"$lte": preferences['max_mileage']}
        
        cursor = car_listings.find(search_filter).sort("price", 1).limit(10)
        cars = await cursor.to_list(length=10)
        return cars
        
    except Exception as e:
        print(f"Error searching inventory: {e}")
        return []

async def get_car_by_id(listing_id: str) -> Optional[dict]:
    """Get specific car details by listing ID"""
    return await car_listings.find_one({"listing_id": listing_id, "available": True})

async def get_sean_response(message: str, user_id: str, phone_number: str, start_time: Optional[float] = None) -> str:
    """Generate Sean's response using Claude"""
    try:
        # Get conversation history and customer preferences
        history = await get_message_history(user_id, limit=HISTORY_LIMIT)
        customer_prefs = await get_customer_preferences(user_id)
        
        # Check if this is the first interaction
        is_first_interaction = len([msg for msg in history if not msg['is_user']]) == 0
        
        # Search current inventory based on message content
        current_inventory = await search_inventory()
        
        system_prompt = """You are Sean, a friendly and knowledgeable car salesperson at Brendan Conlon Automobiles in Ireland. You're passionate about helping customers find their perfect car and you know the Irish car market inside and out.

CRITICAL INSTRUCTION - GREETING BEHAVIOR:
- ONLY introduce yourself as "Sean" in your very FIRST response to a new customer
- In ALL subsequent responses, NEVER reintroduce yourself or say "Sean here" or "I'm Sean"
- Continue conversations naturally without any self-introduction
- You are already established in the conversation after the first message

YOUR EXPERTISE & PERSONALITY:
- Expert in Irish car market, NCT requirements, motor tax, and import regulations
- 15+ years selling quality used cars in Ireland
- Honest, straightforward, and genuinely wants to help customers
- Knows about financing options, trade-ins, and warranties
- Understands Irish driving needs: city driving, country roads, family requirements
- Patient and educational - explains car features and Irish regulations clearly

YOUR COMMUNICATION STYLE:
- Warm, friendly Irish approach - use "brilliant", "fair play", "no bother"
- Ask specific questions to understand customer needs
- Always be honest about car condition and history
- Use WhatsApp-friendly short messages (under 400 characters)
- Make customers feel comfortable and informed

IRISH CAR KNOWLEDGE:
- NCT (National Car Test) requirements and timing
- Motor tax bands and costs
- Import regulations and VRT
- Common issues with popular car models in Ireland
- Financing options available through local credit unions and banks
- Trade-in valuations and process
- Warranty options and after-sales service

CUSTOMER NEEDS ASSESSMENT:
Always try to understand:
1. Budget range and financing needs
2. Primary use (family, commuting, city/country driving)
3. Fuel preferences (petrol/diesel considering Irish fuel costs)
4. Size requirements (parking, family size)
5. Age/mileage preferences
6. NCT and tax status importance
7. Previous car experience and trade-in potential

BRENDAN CONLON AUTOMOBILES EXPERTISE:
- Quality used cars with full history checks
- Comprehensive pre-sale inspections
- Competitive trade-in valuations
- Flexible viewing appointments including evenings/weekends
- After-sales support and warranty options
- Financing assistance and advice

INVENTORY KNOWLEDGE:
- Know current stock levels and can match customer preferences
- Understand which cars are best value for money
- Can explain why certain cars are priced as they are
- Alert customers to new arrivals that match their interests
- Honest about any known issues or upcoming maintenance

RESPONSE GUIDELINES:
- Keep responses conversational and helpful
- Always end with a question or call to action
- Be specific about car details when discussing inventory
- Mention viewing appointments readily available
- Focus on value and suitability for Irish driving conditions
- Never oversell - be honest about pros and cons

EMERGENCY/URGENT RESPONSES:
- Immediate viewing requests: "No problem, I can arrange that"
- Financing questions: Offer to discuss options
- Trade-in valuations: Invite them to bring car for assessment
- Technical questions: Provide honest, knowledgeable answers

SALES APPROACH:
- Build relationships, not just transactions
- Focus on finding the RIGHT car for each customer
- Be transparent about all costs (NCT, tax, insurance implications)
- Offer test drives and thorough inspections
- Follow up appropriately without being pushy"""

        # Build conversation context
        conversation_context = ""
        if history:
            conversation_context = "\nCONVERSATION HISTORY:\n"
            for msg in history[-8:]:  # Last 8 messages for context
                role = "Customer" if msg['is_user'] else "Sean"
                conversation_context += f"{role}: {msg['message']}\n"
            conversation_context += "\n"

        # Add customer preferences context
        prefs_context = ""
        if customer_prefs:
            prefs_context = f"\nCUSTOMER PREFERENCES:\n"
            if customer_prefs.get('preferred_makes'):
                prefs_context += f"Preferred makes: {', '.join(customer_prefs['preferred_makes'])}\n"
            if customer_prefs.get('max_budget'):
                prefs_context += f"Budget: Up to €{customer_prefs['max_budget']:,}\n"
            if customer_prefs.get('fuel_type'):
                prefs_context += f"Fuel preference: {customer_prefs['fuel_type']}\n"

        # Add current inventory context
        inventory_context = "\nCURRENT INVENTORY SAMPLE:\n"
        for car in current_inventory[:5]:  # Show top 5 cars
            inventory_context += f"- {car.get('year')} {car.get('make')} {car.get('model')}: €{car.get('price', 0):,}, {car.get('mileage', 0):,}km\n"

        # Create user prompt with context
        if is_first_interaction:
            user_prompt = f"""This is your FIRST interaction with this customer. Introduce yourself warmly as Sean from Brendan Conlon Automobiles and ask what kind of car they're looking for.

Customer's message: {message}

Respond in under 400 characters with a warm Irish welcome and relevant questions about their car needs."""
        else:
            user_prompt = f"""{conversation_context}{prefs_context}{inventory_context}

The customer has sent a new message. Continue the conversation naturally WITHOUT reintroducing yourself. Help them find their perfect car using your knowledge and current inventory.

Customer's latest message: {message}

Respond in under 400 characters, being helpful and specific about cars or next steps."""

        messages = [{"role": "user", "content": user_prompt}]

        response = await anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=200,
            temperature=0.3,
            system=system_prompt,
            messages=messages,
            stream=False
        )

        if start_time:
            print(f"Claude API response time: {time.time() - start_time} seconds")

        print(f"[INFO] Sean response => {response}")
        ai_response = response.content[0].text.strip()
        
        # Additional safeguard - remove any accidental reintroductions
        if not is_first_interaction:
            ai_response = re.sub(r'^(Hi there!?\s*)?(I\'m\s+)?Sean\s*(here[,\s]*)?', '', ai_response, flags=re.IGNORECASE)
            ai_response = re.sub(r'^(Hello!?\s*)?(This is\s+)?Sean[,\s]*', '', ai_response, flags=re.IGNORECASE)
            ai_response = ai_response.strip()
        
        # Extract and update customer preferences from the conversation
        await extract_and_save_preferences(message, user_id, phone_number)
        
        return ai_response[:400]
        
    except Exception as e:
        print(f"Error generating Sean's response: {e}")
        return "Sorry, I'm having a technical issue. Can you try again in a moment?"

async def extract_and_save_preferences(message: str, user_id: str, phone_number: str):
    """Extract customer preferences from their message and save them"""
    try:
        message_lower = message.lower()
        preferences = {}
        
        # Extract budget
        budget_match = re.search(r'(?:under|up\s+to|max|budget|around)\s*€?(\d{1,2}[,\d]*)', message_lower)
        if budget_match:
            budget_str = budget_match.group(1).replace(',', '')
            preferences['max_budget'] = int(budget_str) * 1000  # Assume thousands
        
        # Extract year preferences
        year_match = re.search(r'(?:20|19)(\d{2})', message)
        if year_match:
            preferences['min_year'] = int(year_match.group(0))
        
        # Extract make preferences
        car_makes = ['audi', 'bmw', 'mercedes', 'volkswagen', 'ford', 'toyota', 'honda', 'nissan', 'hyundai', 'kia', 'mazda', 'renault', 'peugeot', 'citroen', 'fiat', 'skoda', 'seat', 'volvo', 'jaguar', 'land rover', 'mini', 'lexus']
        found_makes = [make for make in car_makes if make in message_lower]
        if found_makes:
            preferences['preferred_makes'] = found_makes
        
        # Extract fuel type
        if 'diesel' in message_lower:
            preferences['fuel_type'] = 'diesel'
        elif 'petrol' in message_lower:
            preferences['fuel_type'] = 'petrol'
        elif 'hybrid' in message_lower:
            preferences['fuel_type'] = 'hybrid'
        elif 'electric' in message_lower:
            preferences['fuel_type'] = 'electric'
        
        if preferences:
            await update_customer_preferences(user_id, phone_number, preferences)
            
    except Exception as e:
        print(f"Error extracting preferences: {e}")

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
        
        # Get Sean's response
        start_time = time.time()
        ai_response = await get_sean_response(msg_body, user_id, user_id, start_time=start_time)
        
        # Add AI response to history
        await add_to_message_history(user_id, ai_response, False)
        
        # Create Twilio response
        resp = MessagingResponse()
        resp.message(ai_response)
        return PlainTextResponse(content=str(resp), media_type="application/xml")
        
    except Exception as e:
        print(f"Error in webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scrape-inventory")
async def manual_inventory_scrape():
    """Manual endpoint to trigger inventory scraping"""
    try:
        cars = await scrape_brendan_conlon_inventory()
        return {"success": True, "cars_scraped": len(cars)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/inventory")
async def get_current_inventory():
    """Get current inventory from database"""
    try:
        cursor = car_listings.find({"available": True}).sort("price", 1)
        cars = await cursor.to_list(length=100)
        return {"success": True, "inventory": cars}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/customer-preferences/{user_id}")
async def get_customer_prefs(user_id: str):
    """Get customer preferences"""
    try:
        prefs = await get_customer_preferences(user_id)
        return {"success": True, "preferences": prefs}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Background task to periodically scrape inventory
async def periodic_inventory_scrape():
    """Scrape inventory every 6 hours"""
    while True:
        try:
            print("Starting scheduled inventory scrape...")
            await scrape_brendan_conlon_inventory()
            print("Inventory scrape completed")
        except Exception as e:
            print(f"Error in scheduled scrape: {e}")
        
        # Wait 6 hours
        await asyncio.sleep(6 * 60 * 60)

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    # Initial inventory scrape
    try:
        await scrape_brendan_conlon_inventory()
    except Exception as e:
        print(f"Initial inventory scrape failed: {e}")
    
    # Start periodic scraping
    asyncio.create_task(periodic_inventory_scrape())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)