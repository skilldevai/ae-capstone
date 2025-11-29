#!/usr/bin/env python3
"""
OmniTech Customer Support Chatbot - Gradio Interface
═══════════════════════════════════════════════════════════════════════════════

OVERVIEW
--------


KEY CONCEPTS
------------

ARCHITECTURE
------------

DEVELOPER MODE
--------------
Toggle "Developer Mode" checkbox to reveal debug tabs:
- Agent Dashboard: See RAG metrics, LLM prompts, and responses
- MCP Monitor: View tool calls, timing, and server stats
- Knowledge Search: Query the vector database directly
- Tickets: View support tickets created by the system

"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import gradio as gr          # Gradio library for building web UIs
import json                  # JSON parsing for debug displays
from datetime import datetime  # Timestamps for chat messages
from typing import Dict, List, Any  # Type hints for better code clarity

# ─────────────────────────────────────────────────────────────────────────────
# RAG Agent Import
# Try to import the RAG agent - if it fails, the UI runs in "demo mode"
# This allows testing the UI without the full backend running
# ─────────────────────────────────────────────────────────────────────────────
try:
    from rag_agent import SyncAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("Warning: rag_agent not found. Running in demo mode.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 1: APPLICATION STATE                                             ║                                                                  ║         ║
# ║                                                                          ║
# ║ WHY A CLASS? Using a class instead of global variables provides:         ║
# ║   1. Encapsulation - all state in one place                              ║
# ║   2. Clear initialization - __init__ sets up defaults                    ║
# ║   3. Methods - related functionality grouped together                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AppState:
    """Global application state."""

    def __init__(self):
        self.agent = None
        self.conversation_history: List[Dict] = []
        self.metrics = {
            'total_queries': 0,
            'resolved_queries': 0,
            'tickets_created': 0
        }
        # Store last debug info for Agent Dashboard
        self.last_prompt = ""
        self.last_response = ""

    def initialize_agent(self) -> bool:
        if not AGENT_AVAILABLE:
            return False
        if self.agent is None:
            try:
                self.agent = SyncAgent()
                print("Agent initialized successfully")
                return True
            except Exception as e:
                print(f"Failed to initialize agent: {e}")
                return False
        return True

    def process_query(self, query: str, email: str) -> Dict[str, Any]:
        if self.agent:
            return self.agent.process_query(query, email)
        else:
            # Demo mode response
            return {
                "response": f"[Demo Mode] Received: {query}",
                "workflow": "demo",
                "confidence": 0.5
            }

    def get_mcp_stats(self) -> Dict[str, Any]:
        """Get MCP server statistics."""
        if self.agent:
            return self.agent.get_server_stats()
        return {"status": "Demo mode - no server"}

    def search_knowledge(self, query: str, max_results: int = 3) -> List[Dict]:
        if self.agent:
            try:
                # Call the search_knowledge tool via the agent's internal method
                # SyncAgent.agent = OmniTechAgent, SyncAgent.loop = event loop
                result = self.agent.loop.run_until_complete(
                    self.agent.agent.call_tool("search_knowledge", {
                        "query": query,
                        "max_results": max_results
                    })
                )
                return result.get("matches", [])
            except Exception as e:
                print(f"Knowledge search error: {e}")
        return []

    def get_tickets(self, customer_email: str = None, status: str = None) -> List[Dict]:
        """Get tickets via MCP tool."""
        if self.agent:
            try:
                args = {"limit": 50}
                if customer_email:
                    args["customer_email"] = customer_email
                if status:
                    args["status"] = status

                result = self.agent.loop.run_until_complete(
                    self.agent.agent.call_tool("get_tickets", args)
                )
                return result.get("tickets", [])
            except Exception as e:
                print(f"Get tickets error: {e}")
        return []


app_state = AppState()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 2: CUSTOM CSS STYLES                                             ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Debug toggle styling */
.debug-toggle {
    margin-top: 1rem;
    padding: 0.5rem 1rem;
    background: #f1f5f9;
    border-radius: 8px;
    font-size: 0.875rem;
}

.debug-toggle label {
    cursor: pointer;
}

/* Typing indicator animation */
@keyframes typing-dot {
    0%, 60%, 100% { opacity: 0.3; }
    30% { opacity: 1; }
}

.typing-indicator {
    display: inline-flex;
    gap: 4px;
    padding: 0.75rem 1rem;
    background: #f1f5f9;
    border-radius: 12px;
    margin: 0.5rem 0;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    background: #64748b;
    border-radius: 50%;
    animation: typing-dot 1.4s infinite;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

.nav-button {
    background: #ffffff;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px;
    padding: 0.875rem 1.25rem;
    margin: 0.375rem 0;
    transition: all 0.2s ease;
    font-weight: 500;
    color: #475569;
}

.nav-button:hover {
    background: #f8fafc;
    border-color: #cbd5e1 !important;
    transform: translateX(4px);
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.25rem;
    margin: 0.5rem 0;
    transition: all 0.2s ease;
}

.metric-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 8px -2px rgba(0, 0, 0, 0.08);
}

.chat-message-user {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.chat-message-agent {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #10b981;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.tool-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.375rem 0;
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 0.875rem;
}
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 3: HELPER FUNCTIONS                                              ║
# ║                                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def format_message(sender: str, content: str, timestamp: str) -> str:
    """Format a chat message as HTML."""
    msg_class = "chat-message-agent" if sender == "agent" else "chat-message-user"
    sender_name = "AI Agent" if sender == "agent" else "You"
    return f"""
    <div class="{msg_class}">
        <div style="font-weight: 600; margin-bottom: 0.5rem;">
            {sender_name}
            <span style="float: right; font-size: 0.875rem; color: #64748b;">{timestamp}</span>
        </div>
        <div style="line-height: 1.6;">{content}</div>
    </div>
    """


def process_query_handler(query: str, customer_email: str, history: str):
    """Process a query and update chat history."""
    if not query.strip():
        return history, "", "", ""

    timestamp = datetime.now().strftime("%H:%M:%S")
    history += format_message("customer", query, timestamp)

    # Initialize agent if needed
    if not app_state.agent:
        app_state.initialize_agent()


    # Add to history

    # Update metrics
    app_state.metrics['total_queries'] += 1
    if result.get('confidence', 0) > 0.7:
        app_state.metrics['resolved_queries'] += 1
    if result.get('ticket_created'):
        app_state.metrics['tickets_created'] += 1

    history += format_message("agent", response, datetime.now().strftime("%H:%M:%S"))

    # Get prompt and response for debug display
    prompt = result.get('llm_prompt', 'No prompt available')
    response_json = json.dumps({k: v for k, v in result.items() if k != 'llm_prompt'}, indent=2)

    # Store in app_state for Agent Dashboard
    app_state.last_prompt = prompt
    app_state.last_response = response_json

    return history, "", prompt, response_json


def generate_agent_dashboard() -> str:
    metrics = app_state.metrics
    total = metrics['total_queries']
    resolved = metrics['resolved_queries']
    tickets = metrics['tickets_created']
    rate = f"{(resolved/max(total,1)*100):.1f}%"

    html = """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>Agent Performance Dashboard</h2>
        <p style="color: #64748b;">Real-time analytics and RAG pipeline metrics</p>
    </div>
    """

    # Metrics row
    html += f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem;">
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem;">TOTAL QUERIES</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #1e293b; margin: 0;">{total}</p>
        </div>
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem;">RESOLUTION RATE</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #10b981; margin: 0;">{rate}</p>
        </div>
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem;">TICKETS CREATED</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #3b82f6; margin: 0;">{tickets}</p>
        </div>
    </div>
    """

    # Recent queries with RAG details
    agent_msgs = [m for m in app_state.conversation_history if m['sender'] == 'agent']

    if agent_msgs:
        html += "<h3>Recent RAG Operations</h3>"

        for i, msg in enumerate(agent_msgs[-5:], 1):
            meta = msg.get('metadata', {})
            workflow = meta.get('workflow', 'unknown')
            category = meta.get('classification', {}).get('category', 'N/A')
            sources = meta.get('sources', [])
            confidence = meta.get('confidence', 0)

            html += f"""
            <div class="metric-card">
                <h4>Query {i} - {msg.get('timestamp', 'N/A')}</h4>
                <p><strong>Workflow:</strong> {workflow}</p>
                <p><strong>Category:</strong> {category}</p>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><strong>Sources:</strong> {', '.join(sources) if sources else 'None'}</p>
            </div>
            """
    else:
        html += "<p style='color: #64748b;'>No queries processed yet. Start chatting to see RAG analytics.</p>"

    return html


def generate_mcp_monitor() -> str:
    stats = app_state.get_mcp_stats()

    html = """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>MCP Protocol Monitor</h2>
        <p style="color: #64748b;">Server status and tool activity</p>
    </div>
    """

    if isinstance(stats, dict) and 'error' not in stats:
        # Server metrics
        html += f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem;">
            <div class="metric-card">
                <h4 style="color: #64748b; font-size: 0.875rem;">TOTAL REQUESTS</h4>
                <p style="font-size: 2rem; font-weight: bold;">{stats.get('total_requests', 0)}</p>
            </div>
            <div class="metric-card">
                <h4 style="color: #64748b; font-size: 0.875rem;">KNOWLEDGE DOCS</h4>
                <p style="font-size: 2rem; font-weight: bold; color: #3b82f6;">{stats.get('knowledge_documents', 0)}</p>
            </div>
            <div class="metric-card">
                <h4 style="color: #64748b; font-size: 0.875rem;">CUSTOMERS</h4>
                <p style="font-size: 2rem; font-weight: bold; color: #10b981;">{stats.get('customers_in_db', 0)}</p>
            </div>
        </div>
        """

        # Recent MCP calls (shown first)
        if app_state.agent:
            mcp_log = app_state.agent.get_mcp_log()
            if mcp_log:
                html += "<h3>Recent MCP Calls</h3>"
                for entry in mcp_log[-10:]:
                    html += f"""
                    <div class="metric-card">
                        <strong>{entry['tool']}</strong>
                        <span style="float: right; color: #64748b;">{entry['duration_ms']}ms</span>
                        <p style="color: #64748b; font-size: 0.875rem;">
                            {entry['timestamp']} - {'✓' if entry['success'] else '✗'}
                        </p>
                    </div>
                    """

        # Available tools
        tools = stats.get('tools_available', [])
        if tools:
            html += "<h3 style='margin-top: 2rem;'>Available MCP Tools</h3><div class='metric-card'>"
            tool_descriptions = {
                'classify_query': 'Classify customer queries into support categories',
                'get_query_template': 'Get prompt templates for categories',
                'list_categories': 'List all support categories',
                'search_knowledge': 'Search the knowledge base',
                'get_knowledge_for_query': 'Get knowledge for a category',
                'lookup_customer': 'Look up customer information',
                'create_support_ticket': 'Create support tickets',
                'get_server_stats': 'Get server statistics'
            }
            for tool in tools:
                desc = tool_descriptions.get(tool, 'MCP tool')
                html += f"""
                <div class="tool-card">
                    <strong>{tool}</strong>
                    <p style="margin: 0.25rem 0 0 1rem; color: #64748b; font-size: 0.875rem;">{desc}</p>
                </div>
                """
            html += "</div>"
    else:
        html += "<p style='color: #64748b;'>Server statistics not available. Initialize the agent first.</p>"

    return html


def generate_tickets_display(customer_filter: str = "", status_filter: str = "") -> str:
    # Ensure agent is initialized
    if not app_state.agent:
        app_state.initialize_agent()

    # Apply filters
    customer = customer_filter if customer_filter and customer_filter != "All" else None
    status = status_filter if status_filter and status_filter != "All" else None

    tickets = app_state.get_tickets(customer_email=customer, status=status)

    html = """
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0;">Support Tickets</h2>
        <p style="color: #64748b; margin: 0.25rem 0 0 0;">View and track customer support tickets</p>
    </div>
    """

    if not app_state.agent:
        html += """
        <div class="metric-card" style="text-align: center; padding: 2rem;">
            <p style="color: #f59e0b; font-size: 1.1rem;">Agent not initialized</p>
            <p style="color: #94a3b8; font-size: 0.9rem;">Send a chat message first to initialize the system.</p>
        </div>
        """
        return html

    if not tickets:
        html += """
        <div class="metric-card" style="text-align: center; padding: 2rem;">
            <p style="color: #64748b; font-size: 1.1rem;">No tickets found</p>
            <p style="color: #94a3b8; font-size: 0.9rem;">Tickets will appear here when customers submit support requests.</p>
        </div>
        """
        return html

    # Summary stats
    open_count = len([t for t in tickets if t.get("status") == "Open"])
    closed_count = len(tickets) - open_count

    html += f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem; margin: 0;">TOTAL TICKETS</h4>
            <p style="font-size: 1.75rem; font-weight: bold; color: #1e293b; margin: 0.25rem 0 0 0;">{len(tickets)}</p>
        </div>
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem; margin: 0;">OPEN</h4>
            <p style="font-size: 1.75rem; font-weight: bold; color: #f59e0b; margin: 0.25rem 0 0 0;">{open_count}</p>
        </div>
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem; margin: 0;">CLOSED</h4>
            <p style="font-size: 1.75rem; font-weight: bold; color: #10b981; margin: 0.25rem 0 0 0;">{closed_count}</p>
        </div>
    </div>
    """

    # Tickets list
    html += "<div style='display: flex; flex-direction: column; gap: 0.75rem;'>"

    for ticket in tickets:
        status = ticket.get("status", "Unknown")
        priority = ticket.get("priority", "medium")

        # Status badge color
        status_color = "#f59e0b" if status == "Open" else "#10b981"
        status_bg = "#fef3c7" if status == "Open" else "#d1fae5"

        # Priority badge
        priority_colors = {
            "high": ("#dc2626", "#fee2e2"),
            "medium": ("#f59e0b", "#fef3c7"),
            "low": ("#3b82f6", "#dbeafe")
        }
        pri_color, pri_bg = priority_colors.get(priority, ("#64748b", "#f1f5f9"))

        html += f"""
        <div class="metric-card" style="padding: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <strong style="font-size: 1rem;">{ticket.get('id', 'N/A')}</strong>
                    <span style="background: {status_bg}; color: {status_color}; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin-left: 0.5rem;">{status}</span>
                    <span style="background: {pri_bg}; color: {pri_color}; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin-left: 0.25rem;">{priority}</span>
                </div>
                <span style="color: #64748b; font-size: 0.8rem;">{ticket.get('created_at', 'N/A')[:16]}</span>
            </div>
            <p style="color: #64748b; margin: 0.5rem 0 0.25rem 0; font-size: 0.875rem;">
                <strong>Customer:</strong> {ticket.get('customer_email', 'N/A')}
            </p>
            <p style="color: #64748b; margin: 0; font-size: 0.875rem;">
                <strong>Issue:</strong> {ticket.get('issue_type', 'N/A')}
            </p>
            <p style="color: #475569; margin: 0.5rem 0 0 0; font-size: 0.9rem; line-height: 1.4;">
                {ticket.get('description', 'No description')[:200]}{'...' if len(ticket.get('description', '')) > 200 else ''}
            </p>
        </div>
        """

    html += "</div>"
    return html


def generate_security_log_display() -> str:

    html = """
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0;">Security Monitor</h2>
        <p style="color: #64748b; margin: 0.25rem 0 0 0;">Track potential prompt injection and goal-hijacking attempts</p>
    </div>
    """

    if not app_state.agent:
        html += """
        <div class="metric-card" style="text-align: center; padding: 2rem;">
            <p style="color: #f59e0b; font-size: 1.1rem;">Agent not initialized</p>
            <p style="color: #94a3b8; font-size: 0.9rem;">Send a chat message first to initialize the system.</p>
        </div>
        """
        return html

    # Summary stats
    high_count = len([e for e in events if e.get("severity") == "high"])
    medium_count = len([e for e in events if e.get("severity") == "medium"])
    low_count = len([e for e in events if e.get("severity") == "low"])

    html += f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem; margin: 0;">TOTAL EVENTS</h4>
            <p style="font-size: 1.75rem; font-weight: bold; color: #1e293b; margin: 0.25rem 0 0 0;">{len(events)}</p>
        </div>
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem; margin: 0;">HIGH SEVERITY</h4>
            <p style="font-size: 1.75rem; font-weight: bold; color: #dc2626; margin: 0.25rem 0 0 0;">{high_count}</p>
        </div>
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem; margin: 0;">MEDIUM SEVERITY</h4>
            <p style="font-size: 1.75rem; font-weight: bold; color: #f59e0b; margin: 0.25rem 0 0 0;">{medium_count}</p>
        </div>
        <div class="metric-card">
            <h4 style="color: #64748b; font-size: 0.875rem; margin: 0;">LOW SEVERITY</h4>
            <p style="font-size: 1.75rem; font-weight: bold; color: #3b82f6; margin: 0.25rem 0 0 0;">{low_count}</p>
        </div>
    </div>
    """

    # Detection patterns info
    html += """
    <div class="metric-card" style="margin-bottom: 1.5rem;">
        <h4 style="margin: 0 0 0.5rem 0;">Monitored Patterns</h4>
        <p style="color: #64748b; font-size: 0.875rem; margin: 0;">
            The agent monitors for common prompt injection patterns including:
            <strong>ignore instructions</strong>, <strong>role changes</strong>,
            <strong>fake system prompts</strong>, <strong>reveal prompt attempts</strong>, and more.
        </p>
    </div>
    """

    if not events:
        html += """
        <div class="metric-card" style="text-align: center; padding: 2rem; background: #f0fdf4; border-color: #bbf7d0;">
            <p style="color: #16a34a; font-size: 1.1rem; margin: 0;">✓ No suspicious activity detected</p>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                All queries have passed security inspection.
            </p>
        </div>
        """
        return html

    # Events list (most recent first)
    html += "<h3>Security Events</h3>"
    html += "<div style='display: flex; flex-direction: column; gap: 0.75rem;'>"

    for event in reversed(events[-20:]):  # Show last 20, most recent first
        severity = event.get("severity", "low")
        event_type = event.get("event_type", "unknown")
        details = event.get("details", "No details")
        query = event.get("query", "")
        timestamp = event.get("timestamp", "")[:19]  # Trim microseconds
        customer = event.get("customer_email", "N/A")

        # Severity badge colors
        severity_colors = {
            "high": ("#dc2626", "#fee2e2"),
            "medium": ("#f59e0b", "#fef3c7"),
            "low": ("#3b82f6", "#dbeafe")
        }
        sev_color, sev_bg = severity_colors.get(severity, ("#64748b", "#f1f5f9"))

        html += f"""
        <div class="metric-card" style="padding: 1rem; border-left: 4px solid {sev_color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <strong style="font-size: 1rem;">{event_type}</strong>
                    <span style="background: {sev_bg}; color: {sev_color}; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin-left: 0.5rem; text-transform: uppercase;">{severity}</span>
                </div>
                <span style="color: #64748b; font-size: 0.8rem;">{timestamp}</span>
            </div>
            <p style="color: #475569; margin: 0.5rem 0 0.25rem 0; font-size: 0.875rem;">
                <strong>Details:</strong> {details}
            </p>
            <p style="color: #64748b; margin: 0; font-size: 0.875rem;">
                <strong>Customer:</strong> {customer}
            </p>
            {f'<div style="margin-top: 0.5rem; padding: 0.5rem; background: #fef2f2; border-radius: 4px; font-family: monospace; font-size: 0.8rem; color: #991b1b; word-break: break-all;">{query[:150]}{"..." if len(query) > 150 else ""}</div>' if query else ''}
        </div>
        """

    html += "</div>"
    return html


def clear_chat():
    """Clear conversation history."""
    app_state.conversation_history = []
    app_state.metrics = {'total_queries': 0, 'resolved_queries': 0, 'tickets_created': 0}
    initial_html = """<div style='padding: 2rem; text-align: center; color: #94a3b8;'>
        <p style='font-size: 1.1rem;'>How can we help you today?</p>
        <p style='font-size: 0.9rem;'>Ask about orders, products, accounts, or technical support.</p>
    </div>"""
    return initial_html, "", ""


def get_status() -> str:
    """Get system status."""
    if app_state.agent:
        tools = app_state.agent.get_available_tools()
        return f"**System Online**\n\nMCP Tools: {len(tools)}\n\nReady to assist"
    else:
        return "**Initializing...**\n\nClick 'Send' to start"


def search_knowledge_direct(search_query: str, max_results: int) -> str:
    if not search_query.strip():
        return """
        <div class="metric-card">
            <p style="color: #64748b;">Enter a search query to search the knowledge base.</p>
        </div>
        """

    results = app_state.search_knowledge(search_query, int(max_results))

    if not results:
        return """
        <div class="metric-card">
            <p style="color: #64748b;">No results found. Try different search terms.</p>
        </div>
        """

    html = f"""
    <div style="margin-bottom: 1rem;">
        <h3>Search Results for: "{search_query}"</h3>
        <p style="color: #64748b;">Found {len(results)} documents</p>
    </div>
    """

    for i, doc in enumerate(results, 1):
        similarity = doc.get('similarity', 0)
        category = doc.get('category', 'unknown')
        content = doc.get('content', '')[:500]
        source = doc.get('source', 'Unknown')

        # Visual similarity bar
        bar_filled = int(similarity * 10) if similarity > 0 else 0
        similarity_bar = "█" * bar_filled + "░" * (10 - bar_filled)

        html += f"""
        <div class="metric-card">
            <h4>Result {i}: {category.replace('_', ' ').title()}</h4>
            <p><strong>Source:</strong> {source}</p>
            <p><strong>Similarity:</strong> {similarity:.3f} <span style="font-family: monospace;">{similarity_bar}</span></p>
            <div style="margin-top: 1rem; padding: 1rem; background: #f8fafc; border-radius: 8px; border-left: 3px solid #3b82f6;">
                <p style="color: #475569; line-height: 1.6;">{content}...</p>
            </div>
        </div>
        """

    return html


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 4: GRADIO INTERFACE DEFINITION                                   ║
# ║                                                                          ║
# ║ Purpose: Define the complete Gradio UI layout and event handlers         ║
# ║                                                                          ║
# ║ STRUCTURE:                                                                       ║
# ║                                                                          ║
# ║ KEY GRADIO CONCEPTS:                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with gr.Blocks(title="OmniTech Support") as demo:

    # Inject custom CSS
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")

    # Header with debug toggle
    with gr.Row():
        with gr.Column(scale=20):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
                        padding: 1.5rem 2rem;
                        border-radius: 14px;
                        color: white;
                        margin-bottom: 1rem;">
                <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">
                    OmniTech Customer Support
                </h1>
                <p style="margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.95rem;">
                    AI-Powered Support Assistant
                </p>
            </div>
            """)
        with gr.Column(scale=1, min_width=180):
            debug_mode = gr.Checkbox(
                label="Developer Mode",
                value=False
            )

    # Status display (only visible in debug mode)
    status_display = gr.Markdown(get_status(), visible=False)

    # Use Tabs for navigation (Gradio 6.0 compatible)
    with gr.Tabs() as tabs:
        # Customer Chat Tab
        with gr.Tab("Chat", id="chat_tab"):
            with gr.Row():
                with gr.Column(scale=4):
                    customer_email = gr.Dropdown(
                        label="Customer",
                        choices=["john.doe@email.com", "sarah.smith@email.com", "mike.johnson@email.com", "guest@example.com"],
                        value="john.doe@email.com",
                        container=True
                    )
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear Chat", size="sm")

            chat_display = gr.HTML(
                value="""<div style='padding: 2rem; text-align: center; color: #94a3b8;'>
                    <p style='font-size: 1.1rem;'>How can we help you today?</p>
                    <p style='font-size: 0.9rem;'>Ask about orders, products, accounts, or technical support.</p>
                </div>"""
            )

            query_input = gr.Textbox(
                label="Message",
                placeholder="Type your question here...",
                lines=2,
                show_label=False
            )

            with gr.Row():
                send_btn = gr.Button("Send Message", variant="primary", scale=3)

            gr.Markdown("**Try asking about:**", elem_classes=["quick-actions-label"])
            with gr.Row():
                q1 = gr.Button("🔑 Password Reset", size="sm", variant="secondary")
                q2 = gr.Button("🔧 Device Issue", size="sm", variant="secondary")
                q3 = gr.Button("📦 Return Policy", size="sm", variant="secondary")
                q4 = gr.Button("🚚 Track Order", size="sm", variant="secondary")

        # Agent Dashboard Tab (hidden by default)
        with gr.Tab("Agent Dashboard", id="agent_tab", visible=False) as agent_tab:
            agent_dashboard = gr.HTML(generate_agent_dashboard())
            refresh_agent_btn = gr.Button("Refresh Dashboard")

            # Debug Info Section - LLM Prompt and Full Response
            gr.Markdown("---")
            gr.Markdown("### Last Query Debug Info")
            with gr.Row():
                with gr.Column():
                    prompt_display = gr.Textbox(label="LLM Prompt", lines=10, interactive=False)
                with gr.Column():
                    response_display = gr.Textbox(label="Full Response", lines=10, interactive=False)

        # MCP Monitor Tab (hidden by default)
        with gr.Tab("MCP Monitor", id="mcp_tab", visible=False) as mcp_tab:
            mcp_monitor = gr.HTML(generate_mcp_monitor())
            refresh_mcp_btn = gr.Button("Refresh Monitor")

        # Knowledge Search Tab (hidden by default)
        with gr.Tab("Knowledge Search", visible=False) as kb_tab:
            gr.Markdown("## Knowledge Base Search")
            gr.Markdown("Search the OmniTech product documentation directly.")

            with gr.Row():
                with gr.Column(scale=3):
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter search terms (e.g., 'password reset', 'warranty', 'smart home')",
                        lines=1
                    )
                with gr.Column(scale=1):
                    search_results_slider = gr.Slider(
                        label="Max Results",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1
                    )

            search_btn = gr.Button("Search Knowledge Base", variant="primary")

            knowledge_results = gr.HTML(
                value="""
                <div class="metric-card">
                    <p style="color: #64748b;">Enter a search query to explore the OmniTech knowledge base.</p>
                    <p style="color: #64748b; font-size: 0.875rem;">
                        Try queries like: "password reset", "device troubleshooting", "return policy", "warranty information"
                    </p>
                </div>
                """
            )

            gr.Markdown("---")
            gr.Markdown("### Knowledge Base Categories")
            gr.HTML("""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                <div class="metric-card">
                    <h4 style="color: #3b82f6;">Account Security</h4>
                    <p style="color: #64748b; font-size: 0.875rem;">Password reset, 2FA, account recovery</p>
                </div>
                <div class="metric-card">
                    <h4 style="color: #10b981;">Device Support</h4>
                    <p style="color: #64748b; font-size: 0.875rem;">Troubleshooting, setup, compatibility</p>
                </div>
                <div class="metric-card">
                    <h4 style="color: #f59e0b;">Shipping & Returns</h4>
                    <p style="color: #64748b; font-size: 0.875rem;">Order tracking, return policy, refunds</p>
                </div>
            </div>
            """)

        # Tickets Tab (hidden by default)
        with gr.Tab("Tickets", id="tickets_tab", visible=False) as tickets_tab:
            with gr.Row():
                with gr.Column(scale=2):
                    ticket_customer_filter = gr.Dropdown(
                        label="Filter by Customer",
                        choices=["All", "john.doe@email.com", "sarah.smith@email.com", "mike.johnson@email.com"],
                        value="All"
                    )
                with gr.Column(scale=2):
                    ticket_status_filter = gr.Dropdown(
                        label="Filter by Status",
                        choices=["All", "Open", "Closed"],
                        value="All"
                    )
                with gr.Column(scale=1):
                    refresh_tickets_btn = gr.Button("Refresh", variant="secondary")

            tickets_display = gr.HTML(generate_tickets_display())

    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 1rem; margin-top: 1rem; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.8rem;">
        <div>OmniTech Customer Support • Enterprise AI Accelerator Capstone</div>
        <div style="margin-top: 0.5rem;">
            <a href="https://getskillsnow.com" target="_blank" style="color: #64748b; text-decoration: none;">&copy; 2025 Tech Skills Transformations</a>
        </div>
    </div>
    """)

    # Debug mode toggle handler
    def toggle_debug_mode(enabled):
        """Toggle visibility of debug elements."""
        return (
            gr.update(visible=enabled),  # agent_tab
            gr.update(visible=enabled),  # mcp_tab
            gr.update(visible=enabled),  # kb_tab
            gr.update(visible=enabled),  # tickets_tab
            gr.update(visible=enabled),  # security_tab
            gr.update(visible=enabled),  # status_display
            gr.update(selected="chat_tab"),  # tabs - always select chat tab on toggle
        )

    debug_mode.change(
        toggle_debug_mode,
        inputs=[debug_mode],
        outputs=[agent_tab, mcp_tab, kb_tab, tickets_tab, security_tab, status_display, tabs]
    )

    # Chat handlers
    send_btn.click(
        process_query_handler,
        inputs=[query_input, customer_email, chat_display],
        outputs=[chat_display, query_input, prompt_display, response_display]
    )
    query_input.submit(
        process_query_handler,
        inputs=[query_input, customer_email, chat_display],
        outputs=[chat_display, query_input, prompt_display, response_display]
    )
    clear_btn.click(clear_chat, outputs=[chat_display, prompt_display, response_display])

    # Quick action handlers
    q1.click(lambda: "How do I reset my password?", outputs=query_input)
    q2.click(lambda: "My device won't turn on", outputs=query_input)
    q3.click(lambda: "What is your return policy?", outputs=query_input)
    q4.click(lambda: "How can I track my order?", outputs=query_input)

    # Refresh handlers
    refresh_agent_btn.click(lambda: generate_agent_dashboard(), outputs=agent_dashboard)
    refresh_mcp_btn.click(lambda: generate_mcp_monitor(), outputs=mcp_monitor)

    # Auto-refresh dashboards when switching tabs
    def refresh_agent_tab():
        """Refresh agent dashboard and debug info."""
        return generate_agent_dashboard(), app_state.last_prompt, app_state.last_response

    agent_tab.select(refresh_agent_tab, outputs=[agent_dashboard, prompt_display, response_display])
    mcp_tab.select(lambda: generate_mcp_monitor(), outputs=mcp_monitor)

    # Knowledge search handler
    search_btn.click(
        search_knowledge_direct,
        inputs=[search_input, search_results_slider],
        outputs=knowledge_results
    )
    search_input.submit(
        search_knowledge_direct,
        inputs=[search_input, search_results_slider],
        outputs=knowledge_results
    )

    # Tickets handlers
    def refresh_tickets(customer, status):
        return generate_tickets_display(customer, status)

    refresh_tickets_btn.click(
        refresh_tickets,
        inputs=[ticket_customer_filter, ticket_status_filter],
        outputs=tickets_display
    )
    ticket_customer_filter.change(
        refresh_tickets,
        inputs=[ticket_customer_filter, ticket_status_filter],
        outputs=tickets_display
    )
    ticket_status_filter.change(
        refresh_tickets,
        inputs=[ticket_customer_filter, ticket_status_filter],
        outputs=tickets_display
    )
    def refresh_tickets_on_select():
        """Refresh tickets and status when switching to tickets tab."""
        return generate_tickets_display(), get_status()

    tickets_tab.select(
        refresh_tickets_on_select,
        outputs=[tickets_display, status_display]
    )

    # Security log handlers
    def refresh_security():
        return generate_security_log_display()

    def clear_security():
        app_state.clear_security_log()
        return generate_security_log_display()

    refresh_security_btn.click(refresh_security, outputs=security_display)
    clear_security_btn.click(clear_security, outputs=security_display)
    security_tab.select(refresh_security, outputs=security_display)

    # Initialize on load
    demo.load(lambda: (app_state.initialize_agent(), get_status())[1], outputs=status_display)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 5. Main Entry Point                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 60)
    print("OmniTech Customer Support Chatbot")
    print("=" * 60)
    print(f"Agent available: {AGENT_AVAILABLE}")
    print("Starting Gradio interface...")
    print("=" * 60)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        footer_links=[
            {"text": "© 2025 Tech Skills Transformations", "url": "https://getskillsnow.com"}
        ]
    )

