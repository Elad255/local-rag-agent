import ollama
import chromadb
from chunk import load_document, split_into_chunks

# --- Build the index (same as before) ---
client = chromadb.Client()
collection = client.create_collection(name="espresso")
text = load_document("docs/sample.txt")
chunks = split_into_chunks(text)
collection.add(documents=chunks, ids=[f"chunk-{i}" for i in range(len(chunks))])


# --- The actual tool functions (plain Python) ---
def search_knowledge_base(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=2)
    return "\n\n".join(results["documents"][0])

def calculator(expression: str) -> str:
    return str(eval(expression))   # NOTE: eval is unsafe — fine for learning, never in production

available_tools = {
    "search_knowledge_base": search_knowledge_base,
    "calculator": calculator,
}


# --- Describe the tools to the LLM so it knows what it can call ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the espresso knowledge base for information about espresso.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression like '18 * 2'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "A math expression"}},
                "required": ["expression"],
            },
        },
    },
]


# --- The agent loop ---
question = "What is 18 times 2?"
messages = [{"role": "user", "content": question}]

print(f"Question: {question}\n")

# 1. Ask the model, offering it the tools
response = ollama.chat(model="llama3.2", messages=messages, tools=tools)
messages.append(response["message"])

# 2. If it decided to call tools, run them and feed results back
if response["message"].get("tool_calls"):
    for call in response["message"]["tool_calls"]:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        print(f"🔧 Agent chose tool: {name}  with args {args}")
        result = available_tools[name](**args)
        messages.append({"role": "tool", "content": str(result)})

    # 3. Ask again, now that it has the tool results — it writes the final answer
    final = ollama.chat(model="llama3.2", messages=messages)
    print("\nFinal answer:", final["message"]["content"])
else:
    print("Answer (no tool needed):", response["message"]["content"])