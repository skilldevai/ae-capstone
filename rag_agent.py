#!/usr/bin/env python3
"""
OmniTech Customer Support RAG Agent
═══════════════════════════════════════════════════════════════════════════════

This agent uses the MCP server for all knowledge access and classification.

ARCHITECTURE:
• MCP Server = Knowledge Layer (owns vector DB, PDFs, classification)
• RAG Agent  = Orchestration Layer (routing, LLM execution, workflow)

WORKFLOWS:
1. CLASSIFICATION WORKFLOW (for support queries):
   classify → get template → retrieve knowledge → execute LLM

2. DIRECT RAG WORKFLOW (for exploratory queries):
   search knowledge → execute LLM

STARTING POINT: This file contains the classification workflow from
rag_agent_classification.py, adapted for HuggingFace Inference API.

Lab 2: Enhance error handling and fallbacks
Lab 3: Add Gradio integration hooks
"""

import asyncio
import json
import logging
import re
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Dict, List, Optional

# MCP Client
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("MCP not installed. Install with: pip install mcp")
    sys.exit(1)

import os
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omnitech-agent")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1. Configuration                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# HuggingFace Inference API
# Set HF_TOKEN environment variable for authenticated access
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
HF_CLIENT = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set. LLM calls will be skipped.")
    print("Set it with: export HF_TOKEN='your_token_here'")
    print("Get a token from: https://huggingface.co/settings/tokens")
    print()

# Support detection keywords (for routing decision)
SUPPORT_KEYWORDS = {
    "security": ["password", "reset", "2fa", "authentication", "hacked", "compromised", "login"],
    "device": ["device", "won't turn", "frozen", "screen", "factory reset", "broken", "power"],
    "shipping": ["ship", "delivery", "track", "order", "arrive", "package"],
    "returns": ["return", "refund", "warranty", "exchange", "money back"],
}

# ANSI colors for terminal output
BLUE = "\033[34m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2. Helper Functions                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def is_support_query(query: str) -> bool:
    """Determine if this is a customer support query vs exploratory."""
    query_lower = query.lower()

    # Check for support-related keywords
    for category, keywords in SUPPORT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                return True

    # Check for question patterns indicating support need
    support_patterns = [
        r"how do i",
        r"how can i",
        r"what should i",
        r"can you help",
        r"i need help",
        r"my \w+ (is|isn't|won't)",
        r"problem with",
        r"issue with"
    ]

    for pattern in support_patterns:
        if re.search(pattern, query_lower):
            return True

    return False


def unwrap_mcp_result(obj):
    """Unwrap MCP result objects to get the actual data."""
    if hasattr(obj, "content") and obj.content:
        content = obj.content[0].text if obj.content else "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return obj


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3. RAG Agent Class                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class OmniTechAgent:
    """RAG Agent for OmniTech Customer Support using MCP."""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.mcp_calls_log: List[Dict] = []

    # ─── MCP Connection ────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Start the MCP server and establish connection."""
        try:
            self.exit_stack = AsyncExitStack()

            server_params = StdioServerParameters(
                command=sys.executable,
                args=["mcp_server.py"],
                env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await self.session.initialize()

            # Verify connection by listing tools
            tools_response = await self.session.list_tools()
            logger.info(f"Connected to MCP server. Tools: {[t.name for t in tools_response.tools]}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False

    async def disconnect(self):
        """Clean up MCP connection."""
        if self.exit_stack:
            await self.exit_stack.aclose()

    # ─── MCP Tool Calls ────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool and return the result."""
        if not self.session:
            raise Exception("MCP session not initialized")

        start_time = datetime.now()
        try:
            result = await self.session.call_tool(tool_name, arguments)
            duration = (datetime.now() - start_time).total_seconds()

            parsed = unwrap_mcp_result(result)

            # Log the call
            self.mcp_calls_log.append({
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "arguments": arguments,
                "duration_ms": round(duration * 1000, 2),
                "success": "error" not in str(parsed).lower()
            })

            if len(self.mcp_calls_log) > 20:
                self.mcp_calls_log = self.mcp_calls_log[-20:]

            return parsed

        except Exception as e:
            logger.error(f"Tool call failed ({tool_name}): {e}")
            return {"error": str(e)}

    # ─── LLM Integration ───────────────────────────────────────────────────

    def query_llm(self, prompt: str) -> str:
        """Query HuggingFace Inference API using InferenceClient."""
        if not HF_CLIENT:
            logger.warning("HF_TOKEN not set. Get a token from https://huggingface.co/settings/tokens")
            return json.dumps({
                "response": "KNOWLEDGE_BASE_ONLY",
                "action_needed": "none",
                "confidence": 0.7
            })

        try:
            logger.info("Calling HuggingFace LLM...")
            # Use chat_completion for instruct models
            response = HF_CLIENT.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=HF_MODEL,
                max_tokens=500,
                temperature=0.7
            )

            # Extract the response text
            result_text = response.choices[0].message.content
            logger.info(f"LLM response received ({len(result_text)} chars)")
            return result_text

        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM error: {error_msg}")

            # Check for model loading (503)
            if "503" in error_msg or "loading" in error_msg.lower():
                return json.dumps({
                    "response": "The AI model is warming up. Please try again in a moment.",
                    "action_needed": "none",
                    "confidence": 0.5
                })

            return json.dumps({
                "response": "KNOWLEDGE_BASE_ONLY",
                "action_needed": "none",
                "confidence": 0.7
            })

    # ─── Classification Workflow ───────────────────────────────────────────

    async def handle_support_query(self, query: str) -> Dict[str, Any]:
        """
        Handle customer support queries using the 4-step classification workflow.

        Steps:
        1. Classify query into support category
        2. Get prompt template for category
        3. Retrieve relevant knowledge
        4. Execute LLM with template + knowledge
        """
        workflow_log = []

        try:
            # Step 1: Classify
            workflow_log.append("[1/4] Classifying query...")
            classification = await self.call_tool("classify_query", {"user_query": query})

            if "error" in classification:
                return {"error": f"Classification failed: {classification['error']}"}

            category = classification.get("suggested_query", "general_support")
            confidence = classification.get("confidence", 0)
            workflow_log.append(f"[Result] Category: {category} (confidence: {confidence:.2f})")

            # Step 2: Get template
            workflow_log.append("[2/4] Getting template...")
            template_info = await self.call_tool("get_query_template", {"query_name": category})

            template = template_info.get("template", "") if "error" not in template_info else ""
            description = template_info.get("description", category)

            # Step 3: Retrieve knowledge
            workflow_log.append(f"[3/4] Retrieving knowledge for {category}...")
            knowledge_info = await self.call_tool("get_knowledge_for_query", {
                "category": category,
                "query": query,
                "max_results": 3
            })

            knowledge = knowledge_info.get("knowledge", "No documentation found.")
            sources = knowledge_info.get("sources", [])
            workflow_log.append(f"[INFO] Retrieved {len(sources)} source(s)")

            # Step 4: Execute LLM
            workflow_log.append("[4/4] Generating response...")

            if template:
                formatted_prompt = template.format(query=query, knowledge=knowledge)
            else:
                formatted_prompt = f"""Please help with this customer question: {query}

Based on this documentation:
{knowledge}

Provide a helpful response."""

            # Add JSON response format instruction
            full_prompt = f"""{formatted_prompt}

Respond with JSON containing:
- "response": your answer (2-3 sentences)
- "action_needed": "none", "create_ticket", or "escalate"
- "confidence": 0-1

JSON Response:"""

            llm_response = self.query_llm(full_prompt)

            # Parse response
            try:
                result = json.loads(llm_response)
            except json.JSONDecodeError:
                result = {
                    "response": llm_response[:500] if len(llm_response) > 500 else llm_response,
                    "action_needed": "none",
                    "confidence": 0.6
                }

            # Handle knowledge-base-only fallback
            if result.get("response") == "KNOWLEDGE_BASE_ONLY":
                result["response"] = f"Based on our {description}:\n\n{knowledge[:400]}..."
                result["confidence"] = 0.8

            # Add metadata
            result["classification"] = {
                "category": category,
                "confidence": confidence,
                "description": description
            }
            result["workflow"] = "classification"
            result["workflow_log"] = workflow_log
            result["sources"] = sources
            result["llm_prompt"] = full_prompt

            workflow_log.append("[SUCCESS] Response generated")
            return result

        except Exception as e:
            logger.error(f"Classification workflow error: {e}")
            return {
                "response": "I encountered an error. Please try again.",
                "error": str(e),
                "workflow": "classification",
                "workflow_log": workflow_log
            }

    # ─── Direct RAG Workflow ───────────────────────────────────────────────

    async def handle_exploratory_query(self, query: str) -> Dict[str, Any]:
        """Handle exploratory queries using direct RAG search."""
        try:
            # Search across all knowledge
            search_result = await self.call_tool("search_knowledge", {
                "query": query,
                "max_results": 5
            })

            matches = search_result.get("matches", [])

            if not matches:
                return {
                    "response": "I couldn't find relevant information. Please try rephrasing.",
                    "workflow": "direct_rag",
                    "sources": []
                }

            # Build context
            knowledge_parts = [m["content"] for m in matches[:3]]
            sources = list(set(m["source"] for m in matches[:3]))
            knowledge = "\n\n---\n\n".join(knowledge_parts)

            # Query LLM
            prompt = f"""Based on this documentation:
{knowledge}

Answer this question: {query}

Respond with JSON containing:
- "response": your answer (2-3 sentences)
- "action_needed": "none"
- "confidence": 0-1

JSON Response:"""

            llm_response = self.query_llm(prompt)

            try:
                result = json.loads(llm_response)
            except json.JSONDecodeError:
                result = {
                    "response": llm_response[:500],
                    "action_needed": "none",
                    "confidence": 0.6
                }

            if result.get("response") == "KNOWLEDGE_BASE_ONLY":
                result["response"] = f"Here's what I found:\n\n{knowledge[:400]}..."

            result["workflow"] = "direct_rag"
            result["sources"] = sources
            result["llm_prompt"] = prompt
            return result

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return {
                "response": "Search error. Please try again.",
                "error": str(e),
                "workflow": "direct_rag"
            }

    # ─── Main Query Handler ────────────────────────────────────────────────

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a customer query, routing to appropriate workflow.

        Support queries → Classification workflow
        Exploratory queries → Direct RAG search
        """
        if is_support_query(query):
            logger.info("[ROUTING] Support query → Classification workflow")
            return await self.handle_support_query(query)
        else:
            logger.info("[ROUTING] Exploratory query → Direct RAG")
            return await self.handle_exploratory_query(query)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 4. Synchronous Wrapper (for Gradio integration)                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SyncAgent:
    """Synchronous wrapper for use with Gradio."""

    def __init__(self):
        self.agent = OmniTechAgent()
        self.loop = None
        self._initialize()

    def _initialize(self):
        """Initialize async components."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            success = self.loop.run_until_complete(self.agent.connect())
            if not success:
                raise Exception("Failed to connect to MCP server")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        """Synchronous query processing."""
        if not self.loop:
            return {"error": "Agent not initialized"}
        return self.loop.run_until_complete(self.agent.process_query(query))

    def get_mcp_log(self) -> List[Dict]:
        """Get MCP call log."""
        return self.agent.mcp_calls_log

    def __del__(self):
        """Cleanup."""
        if self.loop and self.agent:
            try:
                self.loop.run_until_complete(self.agent.disconnect())
                self.loop.close()
            except:
                pass


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 5. Command-Line Interface                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

async def interactive_mode():
    """Run interactive CLI for testing."""
    agent = OmniTechAgent()

    print("=" * 60)
    print("OmniTech Customer Support Agent")
    print("=" * 60)
    print("Connecting to MCP server...")

    if not await agent.connect():
        print("Failed to connect to MCP server!")
        print("Make sure mcp_server.py is in the current directory.")
        return

    print("Connected! Type 'exit' to quit, 'demo' for sample queries.\n")

    sample_queries = [
        "How do I reset my password?",
        "My device won't turn on",
        "What is your return policy?",
        "Tell me about OmniTech",
    ]

    while True:
        try:
            user_input = input(f"{GREEN}Query:{RESET} ").strip()

            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "demo":
                for q in sample_queries:
                    print(f"\n{GREEN}Query:{RESET} {q}")
                    result = await agent.process_query(q)
                    response = result.get("response", "No response")
                    workflow = result.get("workflow", "unknown")
                    print(f"{BLUE}[{workflow}]{RESET}")
                    print(f"{CYAN}{response}{RESET}\n")
            elif user_input:
                result = await agent.process_query(user_input)
                response = result.get("response", "No response")
                workflow = result.get("workflow", "unknown")
                sources = result.get("sources", [])

                print(f"\n{BLUE}[{workflow}]{RESET}")
                print(f"{CYAN}{response}{RESET}")
                if sources:
                    print(f"\n{BLUE}Sources: {', '.join(sources)}{RESET}")
                print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    await agent.disconnect()
    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(interactive_mode())

