import datetime 
from google.adk.agents import Agent, LlmAgent 
from google.adk.tools.agent_tool import AgentTool
import requests 
from math import radians, sin, cos, sqrt, atan2
import os
from typing import List, Dict, Optional, Tuple 

google_key = os.environ.get('GOOGLE_API_KEY')

def get_lat_lng(location: str) -> Tuple[Optional[float], Optional[float]]:
    """ 
    Gets the latitude and longitude for a given location string using Google Geocoding API. 

    Args: 
        location (str): Human-readable location 

    Returns: 
        (lat, lng) as a tuple of floats, or (None, None) on failure 
    """
    
    # TODO: Connect this to the actual Google Places API
    api_key = google_key
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location, "key": api_key}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data["status"] == "OK":
        loc = data["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    else:
        print("Geocoding Error:", data.get("error_message", data["status"]))
        return None, None
    
# create a tool to fetch hotels around a location

# first: we will get the longitude and latitude of the locations,
# then the maximum allowed distance of the hotel from the given location

def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """ 
    Calculates distance in kilometers between two lat/lng points using Haversine formula. 
    """ 
    
    R = 6371.0
    latitude_distance = radians(lat2 - lat1)
    longitude_distance  = radians(lng2 - lng1)
    a = sin(latitude_distance / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(longitude_distance / 2)**2 
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

def get_top_rated_hotels(lat: float, lng: float, radius: int = 2000) -> List[Dict[str, Optional[float]]]:
    """ 
    Returns a list of the top 10 highest-rated hotels near the given latitude and longitude. 

    Each hotel includes: 
        - name 
        - rating 
        - user_ratings_total 
        - address 
        - price_level 
        - distance_km 

    Args: 
        lat (float): Latitude of the search center 
        lng (float): Longitude of the search center 
        radius (int): Radius in meters for hotel search 

    Returns: 
        List of dictionaries with hotel information 
    """ 
    
    # TODO: Connect this to the actual Google Places API
    api_key = google_key
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    params = {
        "location": f"{lat}, {lng}",
        "radius": radius,
        "type": "lodging",
        "key": api_key
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data["status"] != "OK":
        print("Places API Error: ", data.get("error_message", data["status"]))
        return []
    
    rated_hotels = []
    for place in data["results"]:
        
        if "rating" in place and "geometry" in place:
            hotel_lat = place["geometry"]["location"]["lat"]
            hotel_lng = place["geometry"]["location"]["lng"]
            
            distance_km = round(haversine_distance(lat, lng, hotel_lat, hotel_lng), 2)
            
            rated_hotels.append({ 
                "name": place.get("name"), 
                "rating": place.get("rating"), 
                "user_ratings_total": place.get("user_ratings_total"), 
                "address": place.get("vicinity"), 
                "price_level": place.get("price_level"), 
                "distance_km": distance_km 
            })
            
    return sorted(rated_hotels, key=lambda x: (x["rating"], x["user_ratings_total"]), reverse=True)
    
#create a tool to fetch places for an activity around a location

def get_tagged_activity_places(location: str, keywords: List[str], radius: int = 5000) -> List[Dict[str, Optional[str]]]:
    
    """ 
    Find places where one can perform various activities based on keywords, 
    and tag each place with the matching keyword. 
 
    Args: 
        location (str): Name of the place or city to search around. 
        keywords (List[str]): List of activity-related search keywords (e.g., ['hiking', 'museums']). 
        radius (int, optional): Search radius in meters. Default is 5000. 
 
    Returns: 
        List[Dict[str, Optional[str]]]: A list of places, where each place is represented as a dictionary with: 
            - 'tag' (str): the keyword that matched the place 
            - 'name' (str): name of the place 
            - 'address' (str): vicinity or formatted address 
            - 'rating' (float): average Google rating (optional) 
            - 'user_ratings_total' (int): number of user reviews (optional) 
    """ 
    
    # TODO: Connect this to the actual Google Places API
    api_key = google_key
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    lat, lng = get_lat_lng(location)
    
    if lat is None or lng is None:
        return []
    
    all_results = []
    
    for keyword in keywords:
        params = {
            "location": f"{lat}, {lng}",
            "radius": radius,
            "keyword": keyword,
            "key": api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data["status"] != "OK":
            print(f"Keyword '{keyword}' - Places API Error:", data.get("error_message", data["status"]))
            continue
        
        for place in data["results"]:
            all_results.append({
                "tag": keyword,
                "name": place.get("name"),
                "address": place.get("rating"),
                "user_ratings_total": place.get("user_ratings_total")
            })
            
    return all_results

#create AI Agents for the AI travel assistant using google adk and tools

# we will create for agents
# hotel experts: who suggests hotels in a specified city or location
# places-to-visit expert: suggests the best places to visit in a given city or location
# activity expert: suggests activities and experiences that users can do in a given city or location
# travel assistant agent: uses all three agents and works as a travel assistant

HOTEL_AGENT_INSTRUCTION = """
You are a hotel recommenedation assistant. Your job is to suggest hotels
in a specified city or location, based on user preferences such as rating, location or budget.

You must return:
1. **Hotel Name** 
2. **Hotel Rating** (e.g., 4.5 stars) 
3. **Full Address** (including locality and city) 
Guidelines: 
- Prioritize hotels with **high ratings** unless otherwise requested. 
- Include **location context** in the address (e.g., near a landmark or city center if available). 
- Limit the list to **5–7 hotels** unless asked for more. 
- Mention any unique features briefly if relevant. 
Be precise and to the point. If location is missing, ask for it politely. Do not provide fake or fictional hotels. 
"""

# hotels expert agent
hotels_expert = LlmAgent(
    name="hotels_expert",
    model="gemini-2.0-flash",
    description="Agent to suggest hotels in a specified city or location, based on user preferences such as rating, location, or budget",
    instruction=HOTEL_AGENT_INSTRUCTION,
    tools=[get_lat_lng, get_top_rated_hotels, get_tagged_activity_places],
)

#activities expert agent

ACTIVITY_AGENT_INSTRUCTION = """You are a travel activities expert. Your task is to suggest engaging, popular, and unique **activities and experiences** that a user can do in a given city or location. 
You must respond with activity recommendations that are: 
1. **Location-specific** – Tailored to the city, neighborhood, or landmark mentioned. 
2. **Category-aware** – Consider if the user is looking for specific types of activities (e.g., adventure, family-friendly, cultural, nightlife, food-related). 
3. **Well-described** – Briefly explain what the activity involves, why it’s worth trying, and any practical tips (e.g., timing, tickets, age suitability). 
4. **Grouped (if needed)** – Organize by theme if the list is long (e.g., Outdoor, Indoor, Local Experiences, Seasonal). 
Guidelines: 
- Prioritize **local and authentic experiences**, not just touristy ones. 
- Include **hidden gems** or unique suggestions where relevant.
- If the user is vague (e.g., just a city name), suggest a diverse set of top-rated activities. 
- If the user includes interests or preferences (e.g., food, kids, thrill, budget), tailor the list accordingly. 
- Avoid booking or pricing details unless specifically asked. 
Respond in a friendly, informative tone. Use bullet points or short paragraphs for clarity. Always mention the location in the answer to stay contextual. 
""" 

#activities agent
activities_expert = LlmAgent(
    name="activties_expert",
    model="gemini-2.0-flash",
    description="Agent suggest engaging, popular, and unique activities and experiences that a user can do in a given city or location.",
    instruction=ACTIVITY_AGENT_INSTRUCTION,
    tools=[get_lat_lng, get_tagged_activity_places, get_top_rated_hotels],
)

PLACES_TO_VISIT_AGENT_INSTRUCTION = """ 
You are a travel guide assistant. Your job is to suggest the best places to visit in a given city or location. Recommend a mix of popular landmarks, cultural sites, natural attractions, and hidden gems. 

You must include: 
1. **Place Name** 
2. **Short Description** – What the place is and why it's worth visiting. 
3. **Location Context** – Optional, but include area or neighborhood if known. 

Guidelines:
- Suggest **5 to 10 top places**, depending on the city size or request. 
- Prioritize **iconic attractions**, but include a few **unique or offbeat** recommendations where relevant. 
- Consider the user's interest, if mentioned (e.g., history, nature, shopping, photography). 
- Keep descriptions concise and engaging (1–2 lines). 
- Do not repeat categories (e.g., avoid listing multiple malls or temples unless asked). 
- Avoid fictional places. 
"""
#places to visit expert

places_to_visit_expert = LlmAgent(
    name="places_to_visit_expert",
    model="gemini-2.0-flash",
    description="Agent to suggest the best places to visit in a given city or location. Recommends a mix of popular landmarks, cultural sites, natural attractions, and hidden gems.",
    instruction=PLACES_TO_VISIT_AGENT_INSTRUCTION,
    tools=[get_lat_lng, get_top_rated_hotels, get_tagged_activity_places],
)


#Main Agent

# Define instructions 
MAIN_AGENT_INSTRUCTION = """You are a helpful and knowledgeable travel assistant. Your job is to answer user queries related to travel planning in a specific city or location. Respond with clear, concise, and informative answers. 
You must help the user with: 
1. **Hotel Availability** – Suggest hotels in or near the requested location. Include name, area, and any unique features if available. 
2. **Places to Visit** – Recommend popular tourist attractions, historical sites, cultural landmarks, and must-see destinations. 
3. **Activities to Perform** – Suggest activities based on location and traveler interest (e.g., adventure, shopping, food tours, relaxation, cultural experiences). 
Guidelines: 
- Always consider the **location** mentioned in the query. 
- If the user specifies dates, include availability context (if applicable). 
- Use bullet points or short paragraphs for readability. 
- Be proactive in offering relevant, adjacent suggestions if they add value (e.g., nearby places, combo experiences). 
- Keep responses grounded and practical — no fictional or imaginary places. 
- If multiple intents are present (e.g., hotels + places to visit), address them all clearly. 
- Don't ask questions to the user to generate outputs. Decide based on what information is already given. 
- If the user asks to help them plan a journey, provide all the information such as hotels, places to visit, and activities to perform. 
Example inputs you might receive: 
- "Show me hotels in New York and things to do nearby." 
- "Suggest activities for kids in Singapore and how to get around." 
You are not expected to perform bookings, but always aim to provide information that can help the user make travel decisions confidently. 
"""

root_agent = Agent(
    name="ai_travel_planner",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about the hotels availability, places to visit, mode of transport, and activities to perform in a given city or location."
        ),
    instruction=MAIN_AGENT_INSTRUCTION,
    tools=[AgentTool(activities_expert), AgentTool(hotels_expert), AgentTool(places_to_visit_expert)],
    sub_agents=[activities_expert, hotels_expert, places_to_visit_expert],
)