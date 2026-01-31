import os
import json
import re  # <--- New import to help find hidden JSON
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

# 1. Load environment variables
load_dotenv()

# 2. Setup Clients
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), 
    base_url="https://api.groq.com/openai/v1"
)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL_NAME = "llama-3.3-70b-versatile"
MEMORY_FILE = "agent_memory.json"

# --- MEMORY FUNCTIONS ---
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return f.read()
    return "{}"

def save_memory(fact_category, fact_detail):
    current_memory = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            try:
                current_memory = json.load(f)
            except:
                current_memory = {}
    current_memory[fact_category] = fact_detail
    with open(MEMORY_FILE, 'w') as f:
        json.dump(current_memory, f, indent=4)
    return f"Saved: {fact_category} = {fact_detail}"

# --- TOOLS ---
def get_current_time():
    return json.dumps({"current_time": datetime.now().strftime("%A, %B %d, %Y %I:%M %p")})

def web_search(query):
    """Uses Tavily to search the web."""
    print(f"   (🔎 Searching via Tavily: '{query}')")
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        context = []
        for result in response.get('results', []):
            context.append(f"Source: {result['title']}\nContent: {result['content']}")
        return "\n---\n".join(context)
    except Exception as e:
        return f"Search Error: {str(e)}"

# --- TOOL DEFINITIONS ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save a fact about the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_category": {"type": "string"},
                    "fact_detail": {"type": "string"}
                },
                "required": ["fact_category", "fact_detail"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current time.",
            "parameters": {"type": "object", "properties": {}},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for facts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    }
]

# --- AGENT LOGIC (FIXED) ---
def run_agent(user_input):
    long_term_memory = load_memory()
    
    system_prompt = (
        "You are a helpful AI assistant with memory.\n"
        f"USER MEMORY: {long_term_memory}\n\n"
        "RULES:\n"
        "1. USE TOOLS DIRECTLY. Do NOT write the JSON string in the chat.\n"
        "2. If you want to search, just call the function 'web_search'.\n"
        "3. Read Tavily results carefully."
        "4. USE TOOLS DIRECTLY. Do NOT write pseudo-code like <function> or [TOOL].\n"
        "5. If you need to search, just generate the standard tool call JSON.\n"
        "6. Verify facts (like President vs PM) carefully from search results."
        "7. When asked a question, use the relevant tool.\n"
        "8. When reading search results, be extremely careful with names and titles (e.g. President vs Prime Minister).\n"
        "9. If search results mention multiple people, analyze who holds which specific title.\n"
        "10. Answer directly and concisely."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3
        )
    except Exception as e:
        return f"Error: {e}"

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    content = response_message.content

    # --- THE FIX: SELF-CORRECTION BLOCK ---
    # If the AI forgot to use the tool properly and wrote it as text, we catch it here.
    if not tool_calls and content and '{"type": "function"' in content:
        print("   (⚠️ Detected raw JSON text - Fixing automatically...)")
        try:
            # We look for the JSON object inside the text
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                json_text = match.group(0)
                fake_tool_call = json.loads(json_text)
                
                # We manually create a "Tool Call" object to trick the code below
                class FakeToolCall:
                    def __init__(self, name, args, id):
                        self.function = type('obj', (object,), {'name': name, 'arguments': json.dumps(args)})
                        self.id = id
                
                # Extract details from the AI's messy text
                fn_name = fake_tool_call.get("name")
                fn_args = fake_tool_call.get("parameters")
                
                # Add it to the list so the code below processes it normally
                tool_calls = [FakeToolCall(fn_name, fn_args, "call_fixed_123")]
        except Exception as e:
            print(f"   (Failed to auto-fix: {e})")

    # --- NORMAL TOOL EXECUTION ---
    if tool_calls:
        available_functions = {
            "get_current_time": get_current_time,
            "web_search": web_search,
            "save_memory": save_memory,
        }

        messages.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions.get(function_name)
            
            try:
                function_args = json.loads(tool_call.function.arguments)
            except:
                function_args = {}

            # Execute Tool
            if function_name == "web_search":
                # Handle cases where AI nests the query
                q = function_args.get("query")
                if not q and "parameters" in function_args: # Double check for nested params
                    q = function_args["parameters"].get("query")
                
                function_response = function_to_call(query=q or user_input)

            elif function_name == "save_memory":
                print(f"   (💾 Saving: {function_args})")
                function_response = function_to_call(
                    fact_category=function_args.get("fact_category"),
                    fact_detail=function_args.get("fact_detail")
                )
            else:
                print(f"   (⏰ Checking time...)")
                function_response = function_to_call()
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response),
            })
        
        # Final Answer
        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3 
        )
        return final_response.choices[0].message.content
    
    else:
        return response_message.content

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("--- 🧠 Robust Agent (Tavily Powered) Online ---")
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        try:
            answer = run_agent(user_query)
            print(f"Agent: {answer}")
        except Exception as e:
            print(f"Error: {e}")