import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS  # <--- New Import for Search

# 1. Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), 
    base_url="https://api.groq.com/openai/v1"
)
MODEL_NAME = "llama-3.3-70b-versatile"

# --- TOOL 1: TIME ---
def get_current_time():
    """Get the current time."""
    now = datetime.now()
    return json.dumps({
        "current_time": now.strftime("%A, %B %d, %Y %I:%M %p")
    })

# --- TOOL 2: WEB SEARCH (NEW!) ---
def web_search(query):
    """Search the internet for real-time information."""
    print(f"   (🔎 Searching the web for: '{query}'...)")
    try:
        results = DDGS().text(query, max_results=3)
        if results:
            return json.dumps(results)
        return json.dumps({"error": "No results found."})
    except Exception as e:
        return json.dumps({"error": str(e)})

# --- DEFINE TOOLS FOR AI ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time and date.",
            "parameters": {"type": "object", "properties": {}},
        }
    },
    {
        "type": "function", # <--- We add the new tool here
        "function": {
            "name": "web_search",
            "description": "Search the internet for news, scores, or facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search keywords, e.g. 'latest apple stock price'",
                    }
                },
                "required": ["query"],
            },
        }
    }
]

# --- AGENT LOGIC ---
def run_agent(user_input):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": user_input}
    ]

    # First Call
    response = client.chat.completions.create(
        model=MODEL_NAME, 
        messages=messages,
        tools=tools,
        tool_choice="auto", 
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        # Create a map of available functions
        available_functions = {
            "get_current_time": get_current_time,
            "web_search": web_search, # <--- Register new function
        }

        messages.append(response_message)
        
        # Handle multiple tool calls (sometimes it does 2 at once!)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions.get(function_name)
            
            # Parse arguments (needed for search query)
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "web_search":
                function_response = function_to_call(query=function_args.get("query"))
            else:
                function_response = function_to_call()
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
        
        # Second Call: Get final answer
        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        return final_response.choices[0].message.content
    
    else:
        return response_message.content

# --- MAIN LOOP (NEW!) ---
if __name__ == "__main__":
    print("--- 🤖 AI Agent Online (Type 'exit' to quit) ---")
    
    while True: # <--- This Loop runs forever
        user_query = input("\nYou: ")
        
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break # <--- Breaks the loop to stop program
            
        try:
            answer = run_agent(user_query)
            print(f"Agent: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")