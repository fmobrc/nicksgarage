from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
import anthropic
from datetime import datetime
import os
from typing import Optional, List, Dict, Any
import requests
import json
import re
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import uvicorn
from collections import Counter

load_dotenv()

app = FastAPI()

# Configuration
DEALERHUB_API_URL = "https://feeds.dealerhub.ie/api/v1/stock"
DEALER_ID = "2121"
DEALERHUB_API_KEY = os.getenv("DEALERHUB_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize Anthropic client
anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# In-memory storage for conversation history, user preferences, and inventory
conversation_history = {}
user_preferences = {}
inventory_cache = {}  # Cache inventory per dealer
user_inventory_loaded = {}  # Track if inventory loaded for each user
inventory_stats = {}  # Cache inventory statistics

class CarRecommendationAgent:
    def __init__(self):
        pass
        
    async def fetch_dealer_inventory(self):
        """Fetch current inventory from DealerHub API"""
        try:
            headers = {
                "accept": "application/json",
                "x-api-key": DEALERHUB_API_KEY
            }
            
            response = requests.get(
                f"{DEALERHUB_API_URL}?dealerIds={DEALER_ID}",
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                stock_items = data.get('stock', [])
                inventory_cache[DEALER_ID] = stock_items
                # Generate inventory statistics
                self.generate_inventory_stats(stock_items)
                print(f"Fetched {len(stock_items)} vehicles from dealer inventory")
                return True
            else:
                print(f"Failed to fetch inventory: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error fetching inventory: {e}")
            return False
    
    def generate_inventory_stats(self, stock_items: List[Dict]):
        """Generate comprehensive inventory statistics"""
        if not stock_items:
            inventory_stats[DEALER_ID] = {}
            return
            
        stats = {
            'total_count': len(stock_items),
            'makes': Counter(),
            'fuel_types': Counter(),
            'body_types': Counter(),
            'transmission_types': Counter(),
            'years': Counter(),
            'price_ranges': {'under_10k': 0, '10k_20k': 0, '20k_30k': 0, '30k_plus': 0},
            'mileage_ranges': {'under_50k': 0, '50k_100k': 0, '100k_150k': 0, '150k_plus': 0},
            'average_price': 0,
            'price_range': {'min': float('inf'), 'max': 0}
        }
        
        total_price = 0
        valid_prices = 0
        
        for item in stock_items:
            attrs = item.get('attributes', {})
            price = item.get('price', 0)
            
            # Count makes
            make = attrs.get('make', '').strip()
            if make:
                stats['makes'][make.title()] += 1
            
            # Count fuel types
            fuel_type = attrs.get('fuelType', '').strip()
            if fuel_type:
                stats['fuel_types'][fuel_type.title()] += 1
            
            # Count body types
            body_type = attrs.get('bodyType', '').strip()
            if body_type:
                stats['body_types'][body_type.title()] += 1
            
            # Count transmission types
            transmission = attrs.get('transmission', '').strip()
            if transmission:
                stats['transmission_types'][transmission.title()] += 1
            
            # Count years
            year = attrs.get('year')
            if year:
                stats['years'][str(year)] += 1
            
            # Price statistics
            if price and price > 0:
                total_price += price
                valid_prices += 1
                stats['price_range']['min'] = min(stats['price_range']['min'], price)
                stats['price_range']['max'] = max(stats['price_range']['max'], price)
                
                # Price ranges
                if price < 10000:
                    stats['price_ranges']['under_10k'] += 1
                elif price < 20000:
                    stats['price_ranges']['10k_20k'] += 1
                elif price < 30000:
                    stats['price_ranges']['20k_30k'] += 1
                else:
                    stats['price_ranges']['30k_plus'] += 1
            
            # Mileage statistics
            mileage = attrs.get('mileage', 0)
            if mileage:
                if mileage < 50000:
                    stats['mileage_ranges']['under_50k'] += 1
                elif mileage < 100000:
                    stats['mileage_ranges']['50k_100k'] += 1
                elif mileage < 150000:
                    stats['mileage_ranges']['100k_150k'] += 1
                else:
                    stats['mileage_ranges']['150k_plus'] += 1
        
        if valid_prices > 0:
            stats['average_price'] = total_price / valid_prices
        
        # Fix min price if no valid prices found
        if stats['price_range']['min'] == float('inf'):
            stats['price_range']['min'] = 0
            
        inventory_stats[DEALER_ID] = stats
    
    def get_current_inventory(self):
        """Get current inventory from cache"""
        return inventory_cache.get(DEALER_ID, [])
    
    def get_inventory_stats(self):
        """Get inventory statistics"""
        return inventory_stats.get(DEALER_ID, {})
    
    def advanced_search_cars(self, query_text: str = "", **filters) -> List[Dict]:
        """Advanced search with multiple filters and text search"""
        current_inventory = self.get_current_inventory()
        if not current_inventory:
            return []
        
        filtered_cars = current_inventory.copy()
        query_lower = query_text.lower() if query_text else ""
        
        # Text-based search in make, model, description
        if query_text:
            text_filtered = []
            for car in filtered_cars:
                attrs = car.get('attributes', {})
                searchable_text = f"{attrs.get('make', '')} {attrs.get('model', '')} {car.get('description', '')} {attrs.get('bodyType', '')}".lower()
                if any(word in searchable_text for word in query_lower.split()):
                    text_filtered.append(car)
            filtered_cars = text_filtered
        
        # Apply filters
        for filter_key, filter_value in filters.items():
            if not filter_value:
                continue
                
            if filter_key == 'budget_max':
                filtered_cars = [car for car in filtered_cars if car.get('price', 0) <= filter_value]
            elif filter_key == 'budget_min':
                filtered_cars = [car for car in filtered_cars if car.get('price', 0) >= filter_value]
            elif filter_key == 'make':
                filtered_cars = [car for car in filtered_cars 
                               if filter_value.lower() in car.get('attributes', {}).get('make', '').lower()]
            elif filter_key == 'fuel_type':
                filtered_cars = [car for car in filtered_cars 
                               if filter_value.lower() in car.get('attributes', {}).get('fuelType', '').lower()]
            elif filter_key == 'body_type':
                filtered_cars = [car for car in filtered_cars 
                               if filter_value.lower() in car.get('attributes', {}).get('bodyType', '').lower()]
            elif filter_key == 'transmission':
                filtered_cars = [car for car in filtered_cars 
                               if filter_value.lower() in car.get('attributes', {}).get('transmission', '').lower()]
            elif filter_key == 'max_mileage':
                filtered_cars = [car for car in filtered_cars 
                               if car.get('attributes', {}).get('mileage', 0) <= filter_value]
            elif filter_key == 'min_year':
                filtered_cars = [car for car in filtered_cars 
                               if car.get('attributes', {}).get('year', 0) >= filter_value]
            elif filter_key == 'max_year':
                filtered_cars = [car for car in filtered_cars 
                               if car.get('attributes', {}).get('year', 0) <= filter_value]
        
        # Sort by price (ascending)
        filtered_cars.sort(key=lambda x: x.get('price', 0))
        
        return filtered_cars
    
    def search_cars(self, preferences: Dict = None, budget_max: int = None, make: str = None, fuel_type: str = None, max_mileage: int = None) -> List[Dict]:
        """Backward compatibility method"""
        return self.advanced_search_cars(
            budget_max=budget_max,
            make=make,
            fuel_type=fuel_type,
            max_mileage=max_mileage
        )[:5]
    
    def format_car_info(self, car: Dict) -> str:
        """Format car information for display"""
        attributes = car.get('attributes', {})
        make = attributes.get('make', 'Unknown')
        model = attributes.get('model', 'Unknown')
        year = attributes.get('year', 'Unknown')
        price = car.get('price', 0)
        mileage = attributes.get('mileage', 0)
        fuel_type = attributes.get('fuelType', 'Unknown')
        transmission = attributes.get('transmission', 'Unknown')
        body_type = attributes.get('bodyType', 'Unknown')
        
        return f"{year} {make} {model} ({body_type})\n€{price:,} | {mileage:,}km | {fuel_type} | {transmission}"
    
    def get_inventory_summary(self) -> str:
        """Generate a comprehensive inventory summary for Claude"""
        stats = self.get_inventory_stats()
        if not stats:
            return "Inventory data not available."
        
        summary = f"TOTAL INVENTORY: {stats['total_count']} vehicles\n\n"
        
        # Top makes
        if stats['makes']:
            top_makes = stats['makes'].most_common(5)
            summary += f"TOP MAKES: {', '.join([f'{make} ({count})' for make, count in top_makes])}\n\n"
        
        # Fuel types
        if stats['fuel_types']:
            fuel_list = [f"{fuel} ({count})" for fuel, count in stats['fuel_types'].most_common()]
            summary += f"FUEL TYPES: {', '.join(fuel_list)}\n\n"
        
        # Price ranges
        summary += f"PRICE RANGES:\n"
        summary += f"Under €10k: {stats['price_ranges']['under_10k']} cars\n"
        summary += f"€10k-€20k: {stats['price_ranges']['10k_20k']} cars\n"
        summary += f"€20k-€30k: {stats['price_ranges']['20k_30k']} cars\n"
        summary += f"€30k+: {stats['price_ranges']['30k_plus']} cars\n"
        summary += f"Average Price: €{stats['average_price']:,.0f}\n\n"
        
        # Body types
        if stats['body_types']:
            body_list = [f"{body} ({count})" for body, count in stats['body_types'].most_common()]
            summary += f"BODY TYPES: {', '.join(body_list)}\n"
        
        return summary

# Initialize the agent
car_agent = CarRecommendationAgent()

def extract_user_preferences(message: str) -> Dict:
    """Enhanced preference extraction"""
    preferences = {}
    message_lower = message.lower()
    
    # Extract budget
    budget_patterns = [
        r'budget.*?€?(\d+(?:,\d{3})*)',
        r'under.*?€?(\d+(?:,\d{3})*)',
        r'up to.*?€?(\d+(?:,\d{3})*)',
        r'€(\d+(?:,\d{3})*)',
        r'(\d+)k',  # Handle "20k" format
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, message_lower)
        if match:
            budget_str = match.group(1).replace(',', '')
            if 'k' in pattern and 'k' in message_lower:
                preferences['budget_max'] = int(budget_str) * 1000
            else:
                preferences['budget_max'] = int(budget_str)
            break
    
    # Extract car makes (expanded list)
    car_makes = ['audi', 'bmw', 'mercedes', 'volkswagen', 'ford', 'toyota', 'honda', 'nissan', 'hyundai', 'kia', 'mazda', 'renault', 'peugeot', 'citroen', 'fiat', 'skoda', 'seat', 'volvo', 'jaguar', 'land rover', 'mini', 'lexus', 'opel', 'mitsubishi', 'subaru', 'suzuki', 'dacia', 'alfa romeo', 'porsche']
    
    for make in car_makes:
        if make in message_lower:
            preferences['make'] = make
            break
    
    # Extract fuel type
    fuel_types = {
        'diesel': 'diesel',
        'petrol': 'petrol',
        'gasoline': 'petrol',
        'hybrid': 'hybrid',
        'electric': 'electric',
        'ev': 'electric'
    }
    
    for fuel_term, fuel_type in fuel_types.items():
        if fuel_term in message_lower:
            preferences['fuel_type'] = fuel_type
            break
    
    # Extract body type
    body_types = ['hatchback', 'saloon', 'sedan', 'suv', 'estate', 'coupe', 'convertible', 'mpv', 'van']
    for body in body_types:
        if body in message_lower:
            preferences['body_type'] = body
            break
    
    # Extract transmission
    if 'automatic' in message_lower or 'auto' in message_lower:
        preferences['transmission'] = 'automatic'
    elif 'manual' in message_lower:
        preferences['transmission'] = 'manual'
    
    # Extract mileage preferences
    mileage_patterns = [
        r'under (\d+)(?:,\d{3})*\s*(?:km|k|miles)',
        r'less than (\d+)(?:,\d{3})*\s*(?:km|k|miles)',
        r'below (\d+)(?:,\d{3})*\s*(?:km|k|miles)'
    ]
    
    for pattern in mileage_patterns:
        match = re.search(pattern, message_lower)
        if match:
            mileage_val = int(match.group(1).replace(',', ''))
            if 'k' in match.group(0) and 'km' not in match.group(0):
                mileage_val *= 1000
            preferences['max_mileage'] = mileage_val
            break
    
    return preferences

def detect_inventory_query(message: str) -> Dict[str, Any]:
    """Detect if user is asking about inventory statistics or listings"""
    message_lower = message.lower()
    query_info = {
        'is_inventory_query': False,
        'query_type': None,
        'specific_request': None
    }
    
    # Check for inventory count questions
    if any(phrase in message_lower for phrase in [
        'how many cars', 'total cars', 'how many vehicles', 'inventory size',
        'stock size', 'cars available', 'vehicles available', 'cars do you have'
    ]):
        query_info['is_inventory_query'] = True
        query_info['query_type'] = 'count'
    
    # Check for brand/make listing requests
    elif any(phrase in message_lower for phrase in [
        'what brands', 'which makes', 'list brands', 'what cars',
        'show me cars', 'available cars', 'cars in stock'
    ]):
        query_info['is_inventory_query'] = True
        query_info['query_type'] = 'listing'
    
    # Check for specific category queries
    elif any(phrase in message_lower for phrase in [
        'diesel cars', 'petrol cars', 'automatic cars', 'manual cars',
        'hybrid cars', 'electric cars', 'suv', 'hatchback'
    ]):
        query_info['is_inventory_query'] = True
        query_info['query_type'] = 'filtered_listing'
    
    return query_info

async def get_agent_response(message: str, user_id: str) -> str:
    """Generate response using Claude with enhanced car inventory context"""
    try:
        # Initialize conversation history for new users
        if user_id not in conversation_history:
            conversation_history[user_id] = []
            user_preferences[user_id] = {}
            user_inventory_loaded[user_id] = False
        
        # Check if this is the first message from this user
        is_first_message = not user_inventory_loaded.get(user_id, False)
        
        # Fetch inventory on first message
        if is_first_message:
            print(f"First message from user {user_id}, fetching inventory...")
            await car_agent.fetch_dealer_inventory()
            user_inventory_loaded[user_id] = True
        
        # Extract and update user preferences
        new_prefs = extract_user_preferences(message)
        user_preferences[user_id].update(new_prefs)
        
        # Detect if this is an inventory query
        inventory_query = detect_inventory_query(message)
        
        # Build comprehensive inventory context
        inventory_summary = car_agent.get_inventory_summary()
        
        # Search for relevant cars based on message content and preferences
        if inventory_query['is_inventory_query'] and inventory_query['query_type'] == 'filtered_listing':
            # Use message content for filtering
            matching_cars = car_agent.advanced_search_cars(query_text=message)[:10]
        else:
            # Use stored preferences
            matching_cars = car_agent.advanced_search_cars(
                budget_max=user_preferences[user_id].get('budget_max'),
                make=user_preferences[user_id].get('make'),
                fuel_type=user_preferences[user_id].get('fuel_type'),
                max_mileage=user_preferences[user_id].get('max_mileage'),
                body_type=user_preferences[user_id].get('body_type'),
                transmission=user_preferences[user_id].get('transmission')
            )
        
        # Build context for Claude
        context = f"\nCURRENT INVENTORY SUMMARY:\n{inventory_summary}\n"
        
        if matching_cars and not inventory_query['is_inventory_query']:
            context += f"\nTOP MATCHING CARS FOR CUSTOMER:\n"
            for i, car in enumerate(matching_cars[:3], 1):
                context += f"{i}. {car_agent.format_car_info(car)}\n"
        elif matching_cars and inventory_query['query_type'] == 'filtered_listing':
            context += f"\nCARSMATCHING SEARCH ({len(matching_cars)} found):\n"
            for i, car in enumerate(matching_cars[:5], 1):
                context += f"{i}. {car_agent.format_car_info(car)}\n"
        
        # Get recent conversation history
        recent_history = conversation_history[user_id][-6:] if conversation_history[user_id] else []
        history_context = ""
        if recent_history:
            history_context = "\nRECENT CONVERSATION:\n"
            for msg in recent_history:
                role = "Customer" if msg['is_user'] else "Agent"
                history_context += f"{role}: {msg['message']}\n"
        
        user_prefs_context = ""
        if user_preferences[user_id]:
            user_prefs_context = f"\nCUSTOMER PREFERENCES: {user_preferences[user_id]}\n"
        
        # Check if this is first interaction
        is_first_interaction = len([msg for msg in recent_history if not msg['is_user']]) == 0
        
        system_prompt = """You are a knowledgeable car sales agent with access to real-time inventory data. You have detailed statistics about all available vehicles.

IMPORTANT GREETING RULES:
- ONLY introduce yourself in your very FIRST response to a new customer
- In ALL subsequent responses, NEVER reintroduce yourself

YOUR EXPERTISE:
- You have access to complete inventory statistics including exact counts, makes, models, prices, fuel types, etc.
- You can answer specific questions about inventory ("How many cars?", "What brands?", "Show me diesel cars")
- You can filter and search the inventory based on any criteria
- You provide accurate, specific information based on real data

INVENTORY KNOWLEDGE:
- Always refer to the provided inventory summary for accurate counts and statistics
- When customers ask about inventory size, give the exact number
- When they ask about specific types (diesel, automatic, SUV, etc.), provide accurate counts
- List specific cars when appropriate, not just generic recommendations

YOUR COMMUNICATION STYLE:
- Be precise and data-driven when discussing inventory
- Keep responses under 400 characters for WhatsApp
- Provide specific examples from available stock
- Ask follow-up questions to narrow down choices when inventory is large

RESPONSE GUIDELINES:
- Use the inventory summary to answer questions about stock levels, brands, types
- When showing cars, include year, make, model, price, mileage, and key features
- Be honest about what's available and what isn't
- Guide customers toward suitable options based on their stated preferences"""

        if is_first_interaction:
            user_prompt = f"""This is your FIRST interaction with this customer. Greet them warmly and ask what kind of car they're looking for.

Customer's message: {message}
{context}

Respond in under 400 characters with a warm welcome and relevant questions about their car needs."""
        else:
            user_prompt = f"""Continue the conversation naturally WITHOUT reintroducing yourself. Use the inventory data to answer their question accurately.

{history_context}
{user_prefs_context}
{context}

Customer's latest message: {message}

Respond in under 400 characters, being specific about available inventory when relevant."""

        messages = [{"role": "user", "content": user_prompt}]
        
        response = await anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=0.3,
            system=system_prompt,
            messages=messages
        )
        
        ai_response = response.content[0].text.strip()
        
        # Remove accidental reintroductions in non-first interactions
        if not is_first_interaction:
            ai_response = re.sub(r'^(Hi[,!]?\s*)?(I\'m\s+)?[\w\s]*(?:here|agent)[,\s]*', '', ai_response, flags=re.IGNORECASE)
            ai_response = ai_response.strip()
        
        # Add to conversation history
        conversation_history[user_id].append({"message": message, "is_user": True, "timestamp": datetime.now()})
        conversation_history[user_id].append({"message": ai_response, "is_user": False, "timestamp": datetime.now()})
        
        # Keep only last 20 messages
        if len(conversation_history[user_id]) > 20:
            conversation_history[user_id] = conversation_history[user_id][-20:]
        
        return ai_response[:400]
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I'm having a technical issue. Can you try again in a moment?"

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """Handle WhatsApp messages via Twilio webhook"""
    try:
        form_data = await request.form()
        message_body = form_data.get('Body', '').strip()
        user_id = form_data.get('From', '')
        
        if not message_body:
            return PlainTextResponse(content="", media_type="application/xml")
        
        # Get AI response
        ai_response = await get_agent_response(message_body, user_id)
        
        # Create Twilio response
        resp = MessagingResponse()
        resp.message(ai_response)
        
        return PlainTextResponse(content=str(resp), media_type="application/xml")
        
    except Exception as e:
        print(f"Error in WhatsApp webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = car_agent.get_inventory_stats()
    return {
        "status": "healthy", 
        "inventory_cache_size": len(inventory_cache.get(DEALER_ID, [])),
        "active_users": len(conversation_history),
        "inventory_stats": stats
    }

@app.get("/inventory-debug")
async def inventory_debug():
    """Debug endpoint to check inventory data"""
    await car_agent.fetch_dealer_inventory()
    inventory = car_agent.get_current_inventory()
    stats = car_agent.get_inventory_stats()
    
    return {
        "total_cars": len(inventory),
        "stats": stats,
        "sample_car": inventory[0] if inventory else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
