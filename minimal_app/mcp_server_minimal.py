#!/usr/bin/env python3
"""
Minimal MCP Server for OmniTech Support
========================================

This is a simplified version that demonstrates MCP (Model Context Protocol) basics:
- Loads emails and orders from JSON file (no database needed)
- Loads PDF knowledge base into ChromaDB for semantic search
- Classifies customer queries into support categories
- Four tools: classify_query, search_knowledge, search_emails, search_orders

Perfect for learning how MCP servers work!
"""

import asyncio
import json
import os
import re
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import chromadb
from chromadb.utils import embedding_functions
import pypdf

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

# Load data from JSON file
DATA_FILE = Path(__file__).parent / "minimal_data.json"

def load_data():
    """Load emails and orders from JSON file"""
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            return data.get('emails', []), data.get('orders', [])
    except FileNotFoundError:
        print(f"Warning: Data file not found at {DATA_FILE}")
        return [], []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return [], []

EMAILS, ORDERS = load_data()

# Print status
print(f"Loaded {len(EMAILS)} emails and {len(ORDERS)} orders from {DATA_FILE.name}")

# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE SETUP (PDF → ChromaDB vector store)
# ═══════════════════════════════════════════════════════════════════════════

# Knowledge base directory - where the PDF files live (in parent directory)
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base_pdfs"

def load_pdf_documents():
    """Load and parse PDF documents from knowledge base directory."""
    documents = []

    if not KNOWLEDGE_BASE_DIR.exists():
        print(f"Warning: Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
        return documents

    print(f"Loading PDFs from: {KNOWLEDGE_BASE_DIR}")

    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if not filename.endswith('.pdf'):
            continue

        file_path = KNOWLEDGE_BASE_DIR / filename

        try:
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + " "

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text.strip())

            documents.append({
                "id": filename.replace('.pdf', ''),
                "text": text,
                "source": filename
            })

            print(f"  Loaded: {filename} ({len(text)} chars)")

        except Exception as e:
            print(f"  Failed to load {filename}: {e}")

    return documents

def setup_vector_store():
    """Set up ChromaDB with PDF documentation (real RAG!)"""

    # Create ChromaDB client
    chroma_client = chromadb.Client()

    # Use default embedding function
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    # Try to delete old collection if it exists
    try:
        chroma_client.delete_collection("omnitech_docs_minimal")
    except:
        pass

    # Create fresh collection
    collection = chroma_client.create_collection(
        name="omnitech_docs_minimal",
        embedding_function=embedding_function
    )

    # Load PDF documents
    documents = load_pdf_documents()

    if not documents:
        print("Warning: No documents loaded! Knowledge search will not work properly.")
        return collection

    # Add documents to vector store
    for doc in documents:
        collection.add(
            documents=[doc["text"]],
            metadatas=[{"source": doc["source"]}],
            ids=[doc["id"]]
        )

    print(f"Knowledge base ready: {collection.count()} documents loaded")
    return collection

COLLECTION = setup_vector_store()

# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION KEYWORDS (for determining query category)
# ═══════════════════════════════════════════════════════════════════════════

# Keywords that indicate specific support categories
CATEGORY_KEYWORDS = {
    "account_security": [
        "password", "reset", "login", "account", "locked", "security",
        "authentication", "sign in", "signin", "log in", "2fa", "two-factor"
    ],
    "device_troubleshooting": [
        "won't turn on", "not working", "broken", "device", "repair",
        "troubleshoot", "fix", "error", "crash", "frozen", "battery",
        "charging", "screen", "power", "restart", "reboot"
    ],
    "shipping_inquiry": [
        "shipping", "delivery", "tracking", "ship", "arrive", "eta",
        "where is my", "transit", "carrier"
    ],
    "returns_refunds": [
        "return", "refund", "exchange", "money back", "warranty",
        "replacement", "defective"
    ]
}

# ═══════════════════════════════════════════════════════════════════════════
# MCP SERVER SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = Server("omnitech-support-minimal")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    Tell the LLM what tools are available.
    This is like showing someone a toolbox - they can see what's inside!
    """
    return [
        Tool(
            name="classify_query",
            description="Classify a customer support query into a category (account_security, device_troubleshooting, shipping_inquiry, returns_refunds, or general_inquiry).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The customer query to classify"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_knowledge",
            description="Search the product documentation knowledge base using semantic search. Returns relevant documentation passages with source citations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the knowledge base"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 2)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_emails",
            description="Search customer support emails. You can search by customer email, keywords in subject/body, order ID, or status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (email address, keywords, order ID, etc.)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_orders",
            description="Search customer orders. You can search by order ID, customer email, product name, or status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (order ID, email, product name, etc.)"
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    When the LLM wants to use a tool, this function is called.
    It's like the LLM asking us to fetch information from our database.
    """

    if name == "classify_query":
        query = arguments.get("query", "")
        query_lower = query.lower()

        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    result = {
                        "workflow_type": "classification",
                        "category": category
                    }
                    return [TextContent(type="text", text=json.dumps(result))]

        # No specific category matched - use direct RAG
        result = {
            "workflow_type": "direct_rag",
            "category": "general_inquiry"
        }
        return [TextContent(type="text", text=json.dumps(result))]

    elif name == "search_knowledge":
        query = arguments.get("query", "")
        n_results = arguments.get("n_results", 2)

        results = COLLECTION.query(
            query_texts=[query],
            n_results=n_results
        )

        if not results['documents'] or not results['documents'][0]:
            return [TextContent(type="text", text="No relevant documentation found.")]

        # Combine retrieved documents WITH source attribution
        docs_with_sources = []
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i].get('source', 'Unknown')
            docs_with_sources.append(f"[Source: {source}]\n{doc}")

        context = "\n\n---\n\n".join(docs_with_sources)
        return [TextContent(type="text", text=context)]

    elif name == "search_emails":
        query = arguments.get("query", "").lower()

        # Search through all emails
        results = []
        for email in EMAILS:
            # Check if query matches any field
            if (query in email["customer_email"].lower() or
                query in email["subject"].lower() or
                query in email["body"].lower() or
                query in email.get("status", "").lower()):
                results.append(email)

        if results:
            # Format the results nicely
            formatted = f"Found {len(results)} email(s):\n\n"
            for email in results:
                formatted += f"ID: {email['id']}\n"
                formatted += f"From: {email['customer_email']}\n"
                formatted += f"Subject: {email['subject']}\n"
                formatted += f"Date: {email['date']}\n"
                formatted += f"Status: {email['status']}\n"
                formatted += f"Body: {email['body']}\n"
                formatted += "-" * 50 + "\n\n"

            return [TextContent(type="text", text=formatted)]
        else:
            return [TextContent(type="text", text=f"No emails found matching: {query}")]

    elif name == "search_orders":
        query = arguments.get("query", "").lower()

        # Search through all orders
        results = []
        for order in ORDERS:
            # Check if query matches any field
            if (query in order["order_id"].lower() or
                query in order["customer_email"].lower() or
                query in order["product"].lower() or
                query in order["status"].lower()):
                results.append(order)

        if results:
            # Format the results nicely
            formatted = f"Found {len(results)} order(s):\n\n"
            for order in results:
                formatted += f"Order ID: {order['order_id']}\n"
                formatted += f"Customer: {order['customer_email']}\n"
                formatted += f"Product: {order['product']}\n"
                formatted += f"Price: {order['price']}\n"
                formatted += f"Order Date: {order['order_date']}\n"
                formatted += f"Status: {order['status']}\n"
                formatted += f"Tracking: {order['tracking']}\n"
                formatted += "-" * 50 + "\n\n"

            return [TextContent(type="text", text=formatted)]
        else:
            return [TextContent(type="text", text=f"No orders found matching: {query}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

# ═══════════════════════════════════════════════════════════════════════════
# MAIN - START THE SERVER
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Start the MCP server using stdio (standard input/output)"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
