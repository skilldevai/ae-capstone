#!/usr/bin/env python3
"""
Minimal OmniTech Support Chatbot - Gradio Interface
====================================================

This is a simplified chatbot interface that demonstrates:
- Clean Gradio UI with just a chat interface
- RAG (knowledge base search)
- MCP (tool calling for emails/orders)

Perfect for learning the basics without complexity!
"""

import gradio as gr
from pathlib import Path
from rag_agent_minimal import SyncAgent

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════

# The agent handles all RAG + MCP functionality
agent = None

# Load CSS from external file
CSS_FILE = Path(__file__).parent / "gradio_minimal_styles.css"
CUSTOM_CSS = ""
if CSS_FILE.exists():
    CUSTOM_CSS = CSS_FILE.read_text()
else:
    print(f"Warning: CSS file not found at {CSS_FILE}")

# ═══════════════════════════════════════════════════════════════════════════
# CHAT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

async def send_message(message: str, history: list, customer_email: str) -> tuple:
    """
    Handle user messages and return bot response.

    Args:
        message: The user's message
        history: Chat history (list of message dicts with 'role' and 'content')
        customer_email: The current customer's email address

    Returns:
        Tuple of (empty string, updated history) for Gradio
    """

    global agent

    # Ensure history is a list
    if history is None:
        history = []

    # Initialize agent if needed
    if agent is None:
        try:
            agent = SyncAgent()
            await agent.connect_mcp()
        except Exception as e:
            error_msg = f"Error initializing agent: {str(e)}"
            return "", history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ]

    # Get response from agent
    try:
        response = await agent.query(message)
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]

def clear_chat():
    """Clear the chat history"""
    return []

# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

def create_ui():
    """Create the Gradio interface"""

    with gr.Blocks(title="OmniTech Support Chat") as app:

        # Inject custom CSS
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # Header with branding
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
                    padding: 1.5rem 2rem;
                    border-radius: 14px;
                    color: white;
                    margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">
                OmniTech Customer Support
            </h1>
            <p style="margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.95rem;">
                AI-Powered Support Assistant • Minimal Demo
            </p>
        </div>
        """)

        # Welcome message
        gr.Markdown(
            """
            ### Welcome! I can help with:
            - Account security and password issues
            - Device troubleshooting
            - Shipping and delivery questions
            - Returns and refund policies
            - Order status and tracking
            """
        )

        # Customer selector and controls
        with gr.Row():
            with gr.Column(scale=4):
                customer_email = gr.Dropdown(
                    label="Current Customer",
                    choices=[
                        "john.smith@email.com",
                        "sarah.jones@email.com",
                        "mike.chen@email.com",
                        "guest@example.com"
                    ],
                    value="john.smith@email.com",
                    container=True
                )
            with gr.Column(scale=1):
                clear_btn = gr.Button("Clear Chat", size="sm")

        # Chat interface (reduced height)
        chatbot = gr.Chatbot(
            height=350,
            show_label=False
        )

        # Input area
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                scale=4,
                container=False
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        # Example questions
        gr.Examples(
            examples=[
                "How do I reset my password?",
                "What's the status of order ORD-1003?",
                "Show me emails from john.smith@email.com",
                "My device won't turn on, what should I do?",
                "What is your return policy?",
            ],
            inputs=msg,
            label="Example Questions"
        )

        # Event handlers
        msg.submit(
            send_message,
            inputs=[msg, chatbot, customer_email],
            outputs=[msg, chatbot]
        )

        submit_btn.click(
            send_message,
            inputs=[msg, chatbot, customer_email],
            outputs=[msg, chatbot]
        )

        clear_btn.click(
            clear_chat,
            outputs=[chatbot]
        )

        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 1.5rem 1rem; margin-top: 2rem; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.875rem;">
            <div style="margin-bottom: 0.5rem;">
                <strong>OmniTech Customer Support</strong> • Enterprise AI Accelerator Capstone
            </div>
            <div>
                <a href="https://getskillsnow.com" target="_blank" style="color: #64748b; text-decoration: none; transition: color 0.2s;">
                    &copy; 2025 Tech Skills Transformations
                </a>
            </div>
        </div>
        """)

    return app

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create and launch the app
    app = create_ui()

    print("\n" + "="*60)
    print("Starting OmniTech Support Chat (Minimal Version)")
    print("="*60)
    print("\nFeatures:")
    print("  * RAG: Searches product documentation")
    print("  * MCP: Can search emails and orders")
    print("  * Clean, simple interface")
    print("\n" + "="*60 + "\n")

    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False
    )
