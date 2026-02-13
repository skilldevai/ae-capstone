#!/usr/bin/env python3
"""
Minimal RAG Agent for OmniTech Support
=======================================

This simplified version demonstrates:
- MCP Integration: Calling tools for classification, knowledge search, emails, and orders
- LLM Chat: Using Hugging Face models to generate helpful responses
- Agent Orchestration: Coordinating MCP tools and LLM to answer queries

Key Components:
1. MCP Client: Connects to our MCP server for all data operations
   (classification, knowledge search, email/order lookup)
2. LLM (Hugging Face): Generates natural, helpful responses
3. Orchestration: Routes queries through classify → retrieve → generate

Perfect for learning how agents orchestrate MCP tools!
"""

import os
import re
import json
from huggingface_hub import InferenceClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Hugging Face setup
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_CLIENT = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set. LLM calls will be limited.")
    print("Set it with: export HF_TOKEN='your_token_here'")
    print("Get a token from: https://huggingface.co/settings/tokens")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════

class SyncAgent:
    """
    Minimal RAG Agent that orchestrates MCP tools and LLM:
    1. Uses MCP tools to classify queries and search knowledge base
    2. Uses MCP tools to search emails/orders when relevant
    3. Generates helpful responses using Hugging Face LLMs
    """

    def __init__(self, verbose: bool = False):
        """Initialize the agent

        Args:
            verbose: If True, print detailed workflow information (for CLI mode)
        """
        self.verbose = verbose

        # Conversation history for multi-turn context
        # Stores recent exchanges as list of {"user": str, "assistant": str}
        self.conversation_history = []
        self.max_history = 3  # Keep last 3 exchanges for context

        # MCP session (will be set when connecting)
        self.mcp_session = None
        self.mcp_tools = []

    def clear_history(self):
        """Clear conversation history to start fresh."""
        self.conversation_history = []
        if self.verbose:
            print("[HISTORY] Conversation history cleared")

    async def connect_mcp(self):
        """Connect to the MCP server to access all tools"""

        server_params = StdioServerParameters(
            command="python3",
            args=["mcp_server_minimal.py"],
            env=None
        )

        # Create MCP client session
        self.stdio_transport = stdio_client(server_params)
        self.stdio, self.write = await self.stdio_transport.__aenter__()
        self.mcp_session = ClientSession(self.stdio, self.write)

        await self.mcp_session.__aenter__()
        await self.mcp_session.initialize()

        # Get available tools
        response = await self.mcp_session.list_tools()
        self.mcp_tools = response.tools

        print(f"✓ Connected to MCP server with {len(self.mcp_tools)} tools:")
        for tool in self.mcp_tools:
            print(f"  • {tool.name}: {tool.description[:60]}...")

    async def cleanup(self):
        """Clean up MCP connection"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
        if hasattr(self, 'stdio_transport'):
            await self.stdio_transport.__aexit__(None, None, None)

    def query_llm(self, prompt: str) -> str:
        """
        Query Hugging Face LLM to generate a response.

        Args:
            prompt: The full prompt including context and question

        Returns:
            LLM's response text
        """

        if not HF_CLIENT:
            # Fallback response if no HF token
            return json.dumps({
                "response": "I found relevant information in our knowledge base. However, the AI model is not configured. Please set HF_TOKEN to get AI-generated responses.",
                "action_needed": "none",
                "confidence": 0.5
            })

        try:
            # Use chat_completion for instruct models
            response = HF_CLIENT.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=HF_MODEL,
                max_tokens=500,
                temperature=0.7
            )

            # Extract the response text
            result_text = response.choices[0].message.content
            return result_text

        except Exception as e:
            error_msg = str(e)

            # Check for model loading (503)
            if "503" in error_msg or "loading" in error_msg.lower():
                return json.dumps({
                    "response": "The AI model is warming up. Please try again in a moment (this can take 20-30 seconds for the first request).",
                    "action_needed": "none",
                    "confidence": 0.5
                })

            # General error fallback
            return json.dumps({
                "response": "I encountered an error generating a response. Please try again.",
                "action_needed": "none",
                "confidence": 0.3
            })

    async def query(self, user_message: str) -> str:
        """
        Main query function - this is where the magic happens!

        Process:
        1. Classify the query via MCP tool
        2. Search knowledge base via MCP tool
        3. Check if we need to search emails/orders via MCP tools
        4. Send everything to HF LLM to generate response
        """

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {user_message[:50]}...")
            print(f"{'='*60}")

        # Step 1: Classify the query via MCP
        if self.verbose:
            print(f"\n[STEP 1: CLASSIFYING QUERY VIA MCP]")
            print(f"  → Calling MCP tool: classify_query")

        try:
            result = await self.mcp_session.call_tool("classify_query", {"query": user_message})
            classification = json.loads(result.content[0].text)
            workflow_type = classification.get("workflow_type", "direct_rag")
            category = classification.get("category", "general_inquiry")
        except Exception as e:
            workflow_type = "direct_rag"
            category = "general_inquiry"
            if self.verbose:
                print(f"  ✗ Classification failed: {e}, defaulting to direct_rag")

        if self.verbose:
            if workflow_type == "classification":
                print(f"  ✓ Category: {category} (classification workflow)")
            else:
                print(f"  ✓ Category: {category} (direct RAG)")

        # Step 2: Search knowledge base via MCP
        if self.verbose:
            print(f"\n[STEP 2: SEARCHING KNOWLEDGE BASE VIA MCP]")
            print(f"  → Calling MCP tool: search_knowledge")

        try:
            result = await self.mcp_session.call_tool("search_knowledge", {"query": user_message})
            relevant_docs = result.content[0].text
        except Exception as e:
            relevant_docs = "No relevant documentation found."
            if self.verbose:
                print(f"  ✗ Knowledge search failed: {e}")

        if self.verbose:
            print(f"  ✓ Retrieved relevant documentation from knowledge base")

        # Step 3: Check if we need to search emails or orders
        additional_context = ""
        query_lower = user_message.lower()

        # Check for email-related queries
        if any(word in query_lower for word in ["email", "conversation", "ticket", "support history"]) or "@" in user_message:
            if self.verbose:
                print(f"\n[STEP 3: CHECKING MCP TOOLS - EMAILS]")
                print(f"  → Detected email-related query")
                print(f"  → Calling MCP tool: search_emails")
            try:
                # Extract email address if present, or use keywords
                email_match = re.search(r'[\w\.-]+@[\w\.-]+', user_message)
                search_query = email_match.group(0) if email_match else user_message

                result = await self.mcp_session.call_tool("search_emails", {"query": search_query})
                additional_context += f"\n\nCustomer Email History:\n{result.content[0].text}\n"
                if self.verbose:
                    print(f"  ✓ Retrieved email data from MCP server")
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Email search failed: {e}")

        # Check for order-related queries
        if any(word in query_lower for word in ["order", "shipping", "delivery", "tracking", "ord-"]):
            if self.verbose:
                print(f"\n[STEP 3: CHECKING MCP TOOLS - ORDERS]")
                print(f"  → Detected order-related query")
                print(f"  → Calling MCP tool: search_orders")
            try:
                # Extract order ID if present, or use keywords
                order_match = re.search(r'ORD-\d+', user_message, re.IGNORECASE)
                search_query = order_match.group(0) if order_match else user_message

                result = await self.mcp_session.call_tool("search_orders", {"query": search_query})
                additional_context += f"\n\nOrder Information:\n{result.content[0].text}\n"
                if self.verbose:
                    print(f"  ✓ Retrieved order data from MCP server")
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Order search failed: {e}")

        # Step 4: Build prompt for LLM
        if self.verbose:
            print(f"\n[STEP 4: GENERATING LLM RESPONSE]")
            print(f"  → Building augmented prompt with RAG context...")
            if self.conversation_history:
                print(f"  → Including {len(self.conversation_history)} previous exchange(s) for context")
            print(f"  → Sending to Hugging Face LLM ({HF_MODEL})...")

        # Build conversation history context
        history_context = ""
        if self.conversation_history:
            history_lines = []
            for exchange in self.conversation_history[-self.max_history:]:
                history_lines.append(f"Customer: {exchange['user']}")
                history_lines.append(f"Agent: {exchange['assistant']}")
            history_context = "\n".join(history_lines)

        system_prompt = """You are a helpful OmniTech customer support agent.

Product Documentation:
{docs}
{context}
{history_section}
Instructions:
- Be friendly, helpful, and professional
- Use the documentation to answer product questions
- If email or order information is provided, use it to give personalized help
- Keep responses concise (2-4 sentences)
- If you don't know something, say so clearly
- Always provide specific, actionable help
- IMPORTANT: When using information from the documentation, cite the source document (e.g., "According to OmniTech_Returns_Policy_2024.pdf...")
- If there is conversation history, use it to provide continuity and reference previous exchanges when relevant

Customer Question: {question}

Respond with JSON containing:
- "response": your helpful answer (2-4 sentences) - include source citations when referencing documentation
- "action_needed": "none", "create_ticket", or "escalate"
- "confidence": 0-1

JSON Response:"""

        # Build the history section only if we have history
        history_section = ""
        if history_context:
            history_section = f"\nPrevious Conversation:\n{history_context}\n"

        full_prompt = system_prompt.format(
            docs=relevant_docs,
            context=additional_context,
            history_section=history_section,
            question=user_message
        )

        # Step 5: Get LLM response
        llm_response = self.query_llm(full_prompt)

        # Step 6: Parse response (handle JSON wrapped in markdown)
        result = None
        try:
            result = json.loads(llm_response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks (```json ... ```)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Also try to find raw JSON object in the response
            if result is None:
                json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', llm_response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

            # Fallback if no valid JSON found
            if result is None:
                clean_response = re.sub(r'```(?:json)?|```', '', llm_response).strip()
                result = {
                    "response": clean_response[:500] if len(clean_response) > 500 else clean_response,
                    "action_needed": "none",
                    "confidence": 0.6
                }

        # Extract just the response text
        final_response = result.get("response", "I'm sorry, I couldn't generate a proper response.")

        # Save this exchange to conversation history
        self.conversation_history.append({
            "user": user_message,
            "assistant": final_response
        })
        # Keep only the last max_history exchanges
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        if self.verbose:
            print(f"  ✓ LLM response received and parsed")
            print(f"  → Saved to conversation history ({len(self.conversation_history)}/{self.max_history} exchanges)")
            print(f"\n[WORKFLOW COMPLETE]")
            print(f"{'='*60}\n")

        return final_response

# ═══════════════════════════════════════════════════════════════════════════
# Interactive mode (when run directly)
# ═══════════════════════════════════════════════════════════════════════════

async def interactive_agent():
    """Run an interactive REPL for querying the RAG agent."""

    print("\n" + "="*60)
    print("OmniTech RAG Agent - Interactive Mode")
    print("="*60)
    print("\nInitializing agent with verbose output enabled...")
    print("This will show the complete workflow for each query.\n")

    # Create agent with verbose=True for CLI mode
    agent = SyncAgent(verbose=True)

    try:
        # Try to connect to the MCP server; if it fails, keep going so
        # users can still see the error handling.
        await agent.connect_mcp()
    except Exception as e:
        print(f"Warning: couldn't connect to MCP server: {e}")

    print("\n" + "-"*60)
    print("Ready! Try these example queries:")
    print("  • 'How do I reset my password?' (classification → account_security)")
    print("  • 'My device won't turn on' (classification → device_troubleshooting)")
    print("  • 'Tell me about OmniTech' (direct RAG)")
    print("-"*60)
    print("Commands: 'exit'/'quit' to stop, 'clear' to reset history")
    print("Note: The agent remembers your last 3 exchanges for context!")

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                # Ctrl-D / EOF -> exit
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower() == "clear":
                agent.clear_history()
                print("Conversation history cleared. Starting fresh!")
                continue

            try:
                response = await agent.query(user_input)
                print(f"Agent: {response}\n")
            except Exception as e:
                print(f"Error processing query: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        try:
            await agent.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    import asyncio
    asyncio.run(interactive_agent())
