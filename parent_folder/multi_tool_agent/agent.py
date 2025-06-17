import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
import uuid
from google.genai import types
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import threading
import json
import hmac
import hashlib
import os
from transaction_agent.agent import get_transaction_details, create_transaction_toolset

# --- Import all the new tools and agents ---
from rag_tools.rag_tool import search_documents, get_document_context, get_rag_stats
from categorized_rag_tools.categorized_rag_tool import search_documents_in_category, get_context_from_category, search_across_categories
from orca_protocol_agent.orca_protocol import get_orca_protocol_toolset
from researcher_agent.researcher import researcher_agent
from rag_agent.agent import rag_agent
from orca_protocol_agent.agent import orca_protocol_agent
from google.adk.tools.agent_tool import AgentTool

# --- Load Environment Variables (If ADK tools need them, e.g., API keys) ---
load_dotenv()

# --- OpenAPI Tool Imports ---
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# --- Authentication imports ---
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from fastapi.openapi.models import HTTPBearer
from google.adk.auth import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth, HttpCredentials

# --- Constants ---
APP_NAME_OPENAPI = "multi_tool_agent_app"
USER_ID_OPENAPI = "user_openapi_1"
SESSION_ID_OPENAPI = f"session_openapi_{uuid.uuid4()}" # Unique session ID
AGENT_NAME_OPENAPI = "multi_tool_agent"
GEMINI_MODEL = "gemini-2.0-flash"

# --- Initialize Orca Protocol Toolset ---
orca_protocol_toolset = get_orca_protocol_toolset()

# --- Webhook Configuration ---
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', 'your-webhook-secret-key')
WEBHOOK_PORT = int(os.getenv('WEBHOOK_PORT', 5001))

# --- Flask App Setup ---
app = Flask(__name__)

root_agent = Agent(
    name=AGENT_NAME_OPENAPI,
    model=GEMINI_MODEL,
    description="""
        Comprehensive multi-tool agent capable of handling transactions, document search, research, and blockchain protocol interactions.
    """,
    instruction="""
        You are a comprehensive multi-tool agent with access to various specialized tools and sub-agents. Your capabilities include:

        ## TRANSACTION ANALYSIS
        - Get portfolio transaction details using get_transaction_details
        - When calling getPortfolioTransactions:
          * ALWAYS use the portfolio_id provided in the context or user message
          * Include standard query parameters: page=1, limit=10, order=desc, type=portfolio, quote=usdt
          * The API requires authentication with a bearer token

        ## DOCUMENT SEARCH & RAG CAPABILITIES
        For document-related queries, you have two approaches:

        ### Basic RAG (for general document search):
        - Use search_documents for semantic search across all documents
        - Use get_document_context for comprehensive context retrieval
        - Use get_rag_stats for system statistics

        ### Categorized RAG (for organized document collections):
        - Use search_documents_in_category when user specifies a category (e.g., "blue", "red")
        - Use get_context_from_category for category-specific context
        - Use search_across_categories for multi-category searches
        - If asked about something "blue" → search in blue category
        - If asked about something "red" → search in red category

        ## RESEARCH & NEWS ANALYSIS
        - Delegate to researcher_agent for cryptocurrency news and sentiment analysis
        - Use when user asks for:
          * Latest news about cryptocurrencies
          * Sentiment analysis of crypto markets
          * Market research and analysis

        ## BLOCKCHAIN PROTOCOL INTERACTIONS
        - Use Orca Protocol tools for blockchain transaction queries
        - Handle DeFi protocol interactions and aggregation data

        ## SUB-AGENTS AVAILABLE
        - researcher_agent: For crypto news and sentiment analysis
        - rag_agent: Specialized document Q&A agent
        - orca_protocol_agent: Blockchain protocol interactions

        ## DECISION MAKING GUIDE
        1. **Transaction questions** → use get_transaction_details
        2. **Document search (general)** → use basic RAG tools (search_documents, get_document_context)
        3. **Document search (specific category)** → use categorized RAG tools (search_documents_in_category)
        4. **Crypto news/research** → delegate to researcher_agent
        5. **Blockchain protocols** → use Orca Protocol tools
        6. **Complex document analysis** → delegate to rag_agent

        Always provide helpful, accurate, and well-sourced information. Cite sources when using document search tools.
        Format responses clearly and provide context for your tool choices.
    """,
    sub_agents=[researcher_agent, rag_agent, orca_protocol_agent],
    tools=[
        # Transaction tools
        get_transaction_details,
        # Basic RAG tools
        search_documents,
        get_document_context,
        get_rag_stats,
        # Categorized RAG tools
        search_documents_in_category,
        get_context_from_category,
        search_across_categories,
        # Orca Protocol tools
        orca_protocol_toolset.get_tool("get_transactions"),
        # Agent tools
        AgentTool(agent=researcher_agent),
        AgentTool(agent=rag_agent),
        AgentTool(agent=orca_protocol_agent),
    ],
)

# Global runner instance
runner_openapi = None

# --- Webhook Security Functions ---
def verify_webhook_signature(payload_body, signature_header, secret):
    """Verify webhook signature for security."""
    if not signature_header:
        return False

    try:
        # Support different signature formats
        if signature_header.startswith('sha256='):
            signature = signature_header[7:]
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload_body,
                hashlib.sha256
            ).hexdigest()
        elif signature_header.startswith('sha1='):
            signature = signature_header[5:]
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload_body,
                hashlib.sha1
            ).hexdigest()
        else:
            # Assume raw signature
            signature = signature_header
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload_body,
                hashlib.sha256
            ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        print(f"Signature verification error: {e}")
        return False

def extract_question_from_payload(data, headers):
    """Extract question from different webhook payload formats."""

    # Generic formats
    if 'question' in data:
        return data.get('question', '')
    elif 'query' in data:
        return data.get('query', '')
    elif 'prompt' in data:
        return data.get('prompt', '')
    elif 'text' in data:
        return data.get('text', '')
    elif 'content' in data:
        return data.get('content', '')
    elif 'message' in data:
        if isinstance(data.get('message'), dict):
            return data['message'].get('text', '')
        return data.get('message', '')

    # Try to find any text field
    for key in ['body', 'payload', 'data', 'input']:
        if key in data and isinstance(data[key], str):
            return data[key]

    return None

def extract_token_from_payload(data, headers):
    """Extract token from different webhook payload formats."""

    # Generic formats - try common token fields
    token_fields = ['token', 'user_token', 'access_token', 'auth_token', 'user_id', 'userId', 'user', 'from', 'author', 'sender', 'account_id', 'customer_id']
    for field in token_fields:
        if field in data:
            value = data[field]
            if isinstance(value, dict):
                # If it's a dict, try to get 'id' field
                return value.get('id') or value.get('token') or value.get('user_id')
            elif isinstance(value, (str, int)):
                return str(value)

    return None

def extract_portfolio_id_from_payload(data, headers):
    """Extract portfolio_id from different webhook payload formats."""

    # Generic formats - try common portfolio ID fields
    portfolio_fields = ['portfolio_id', 'portfolioId', 'portfolio', 'account_id', 'accountId', 'wallet_id', 'walletId']
    for field in portfolio_fields:
        if field in data:
            value = data[field]
            if isinstance(value, dict):
                # If it's a dict, try to get 'id' field
                return value.get('id') or value.get('portfolio_id')
            elif isinstance(value, (str, int)):
                return str(value)

    return None

def extract_payload_data(data, headers):
    """Extract question, token, and portfolio_id from payload."""
    question = extract_question_from_payload(data, headers)
    token = extract_token_from_payload(data, headers)
    portfolio_id = extract_portfolio_id_from_payload(data, headers)
    return question, token, portfolio_id

def format_webhook_response(answer, token=None):
    """Format response for webhook."""

    # Generic webhook response
    response = {
        'success': True,
        'response': answer,
        'timestamp': datetime.datetime.now().isoformat()
    }

    # Add token information if provided
    if token and isinstance(response, dict):
        response['token'] = token

    return response

# --- Session and Runner Setup ---
async def setup_session_and_runner():
    session_service_openapi = InMemorySessionService()
    runner_openapi = Runner(
        agent=root_agent,
        app_name=APP_NAME_OPENAPI,
        session_service=session_service_openapi,
    )
    await session_service_openapi.create_session(
        app_name=APP_NAME_OPENAPI,
        user_id=USER_ID_OPENAPI,
        session_id=SESSION_ID_OPENAPI,
    )
    return runner_openapi

# --- Agent Interaction Function ---
async def call_openapi_agent_async(query, runner_openapi, token=None, session_id=None, portfolio_id=None):
    # Use provided token or default
    actual_token = token or USER_ID_OPENAPI
    actual_session_id = session_id or SESSION_ID_OPENAPI
    actual_portfolio_id = portfolio_id

    print(f"\n--- Agent Query: {query} ---")
    print(f"Token: {actual_token}")
    print(f"Portfolio ID: {actual_portfolio_id}")

    # If we have a token, create a new agent with authenticated tools
    if token and token != USER_ID_OPENAPI:
        print(f"Creating authenticated toolset with token: {token[:10]}...")
        authenticated_toolset = create_transaction_toolset(token)

        # Create a new agent instance with authenticated tools + all other tools
        authenticated_agent = Agent(
            name=f"{AGENT_NAME_OPENAPI}_auth",
            model=GEMINI_MODEL,
            description="""
                Comprehensive multi-tool agent capable of handling transactions, document search, research, and blockchain protocol interactions.
            """,
            instruction="""
                You are a comprehensive multi-tool agent with access to various specialized tools and sub-agents. Your capabilities include:

                ## TRANSACTION ANALYSIS
                - Get portfolio transaction details using authenticated transaction tools
                - When calling getPortfolioTransactions:
                  * ALWAYS use the portfolio_id provided in the context or user message
                  * Include standard query parameters: page=1, limit=10, order=desc, type=portfolio, quote=usdt
                  * The API requires authentication with a bearer token

                ## DOCUMENT SEARCH & RAG CAPABILITIES
                For document-related queries, you have two approaches:

                ### Basic RAG (for general document search):
                - Use search_documents for semantic search across all documents
                - Use get_document_context for comprehensive context retrieval
                - Use get_rag_stats for system statistics

                ### Categorized RAG (for organized document collections):
                - Use search_documents_in_category when user specifies a category (e.g., "blue", "red")
                - Use get_context_from_category for category-specific context
                - Use search_across_categories for multi-category searches
                - If asked about something "blue" → search in blue category
                - If asked about something "red" → search in red category

                ## RESEARCH & NEWS ANALYSIS
                - Delegate to researcher_agent for cryptocurrency news and sentiment analysis
                - Use when user asks for:
                  * Latest news about cryptocurrencies
                  * Sentiment analysis of crypto markets
                  * Market research and analysis

                ## BLOCKCHAIN PROTOCOL INTERACTIONS
                - Use Orca Protocol tools for blockchain transaction queries
                - Handle DeFi protocol interactions and aggregation data

                ## DECISION MAKING GUIDE
                1. **Transaction questions** → use authenticated transaction tools
                2. **Document search (general)** → use basic RAG tools (search_documents, get_document_context)
                3. **Document search (specific category)** → use categorized RAG tools (search_documents_in_category)
                4. **Crypto news/research** → use mock_search_tool functionality for testing
                5. **Blockchain protocols** → use Orca Protocol tools
                6. **Complex document analysis** → use RAG tools directly

                Always provide helpful, accurate, and well-sourced information. Cite sources when using document search tools.
                Format responses clearly and provide context for your tool choices.
            """,
            tools=[
                # Authenticated transaction tools
                authenticated_toolset,
                # Basic RAG tools
                search_documents,
                get_document_context,
                get_rag_stats,
                # Categorized RAG tools
                search_documents_in_category,
                get_context_from_category,
                search_across_categories,
                # Orca Protocol tools
                orca_protocol_toolset.get_tool("get_transactions"),
            ],
        )

        # Create a temporary runner with the authenticated agent using the SAME app name
        session_service = runner_openapi.session_service
        temp_runner = Runner(
            agent=authenticated_agent,
            app_name=APP_NAME_OPENAPI,  # Use the same app name!
            session_service=session_service,
        )

        # Ensure the session exists for the authenticated user with the same app name
        try:
            await session_service.create_session(
                app_name=APP_NAME_OPENAPI,  # Same app name
                user_id=actual_token,
                session_id=actual_session_id,
            )
        except Exception as e:
            # Session might already exist, that's okay
            print(f"Session creation note: {e}")

        # Use the temporary runner for this call
        runner_to_use = temp_runner
    else:
        # Use the default runner
        runner_to_use = runner_openapi

    # Enhance the query with portfolio_id context if provided
    enhanced_query = query
    if actual_portfolio_id:
        enhanced_query = f"{query}\n\nContext: Use portfolio_id '{actual_portfolio_id}' for the API call."

    print(f"Enhanced Query: {enhanced_query}")

    content = types.Content(role='user', parts=[types.Part(text=enhanced_query)])
    final_response_text = "Agent did not provide a final text response."

    try:
        async for event in runner_to_use.run_async(
            user_id=actual_token, session_id=actual_session_id, new_message=content
            ):
            if event.get_function_calls():
                call = event.get_function_calls()[0]
                print(f"  Agent Action: Called function '{call.name}' with args {call.args}")
            elif event.get_function_responses():
                response = event.get_function_responses()[0]
                print(f"  Agent Action: Received response for '{response.name}'")
                print(f"  Function Response: {response.response}")  # Debug: Print actual response
            elif event.is_final_response() and event.content and event.content.parts:
                # Capture the last final text response
                final_response_text = event.content.parts[0].text.strip()

        print(f"Agent Final Response: {final_response_text}")
        return final_response_text

    except Exception as e:
        error_msg = f"An error occurred during agent run: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

# --- Webhook Endpoints ---
@app.route('/webhook', methods=['POST'])
def webhook_handler():
    """Main webhook endpoint that handles various webhook formats."""
    try:
        # Get raw payload for signature verification
        payload_body = request.get_data()
        headers = dict(request.headers)

        # Verify signature if provided
        signature = headers.get('X-Hub-Signature-256') or headers.get('X-Hub-Signature') or headers.get('X-Signature')
        if signature and WEBHOOK_SECRET != 'your-webhook-secret-key':
            if not verify_webhook_signature(payload_body, signature, WEBHOOK_SECRET):
                return jsonify({'error': 'Invalid signature'}), 401

        # Parse JSON payload
        try:
            data = request.get_json()
        except Exception:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        if not data:
            return jsonify({'error': 'Empty payload'}), 400

        # Extract question, token, and portfolio_id from payload
        question, token, portfolio_id = extract_payload_data(data, headers)
        if not question:
            return jsonify({
                'error': 'No question found in payload. Supported fields: content, text, message, question, query, prompt'
            }), 400

        # Use extracted token or generate a default one
        if not token:
            token = f"webhook_token_{hash(str(data)) % 10000}"  # Generate consistent ID from payload

        print(f"Webhook received - Token: {token}, Question: {question}, Portfolio ID: {portfolio_id}")

        # Create token-specific session ID
        token_session_id = f"session_{token}_{uuid.uuid4()}"

        # Ensure token session exists and run agent query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Create session for this token if it doesn't exist
            session_service = runner_openapi.session_service
            try:
                loop.run_until_complete(session_service.create_session(
                    app_name=APP_NAME_OPENAPI,
                    user_id=token,
                    session_id=token_session_id,
                ))
            except Exception as e:
                # Session might already exist, that's okay
                print(f"Session creation note: {e}")

            # Run the agent query with token-specific session
            response = loop.run_until_complete(
                call_openapi_agent_async(question, runner_openapi, token, token_session_id, portfolio_id)
            )

            # Format response
            formatted_response = format_webhook_response(response, token=token)

            # Add portfolio_id to response for confirmation
            if portfolio_id and isinstance(formatted_response, dict):
                formatted_response['portfolio_id'] = portfolio_id

            return jsonify(formatted_response)

        finally:
            loop.close()

    except Exception as e:
        print(f"Webhook error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Webhook processing failed: {str(e)}',
            'success': False
        }), 500

@app.route('/webhook/test', methods=['GET', 'POST'])
def webhook_test():
    """Test endpoint for webhook verification."""
    if request.method == 'GET':
        # Handle webhook verification (like Discord, Slack)
        challenge = request.args.get('challenge')
        if challenge:
            return challenge
        return "Webhook endpoint is active!"

    # Handle test POST
    return jsonify({
        'message': 'Webhook test successful',
        'timestamp': datetime.datetime.now().isoformat(),
        'headers': dict(request.headers),
        'payload': request.get_json() if request.is_json else request.form.to_dict()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Transaction Details Agent Webhook',
        'timestamp': datetime.datetime.now().isoformat()
    })

# --- Initialize the agent ---
async def initialize_agent():
    global runner_openapi
    runner_openapi = await setup_session_and_runner()
    print("Agent initialized and ready!")

# --- Run Examples ---
# async def run_openapi_example():
#     runner_openapi = await setup_session_and_runner()

#     # Example transaction queries
#     await call_openapi_agent_async("Get transactions for portfolio portfolio_123", runner_openapi, "sample_token", "example_session")

# --- Execute ---
if __name__ == "__main__":
    print("Starting Transaction Details Agent Webhook Server...")

    # Initialize the agent
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(initialize_agent())
    finally:
        loop.close()

    # Start Flask app
    print(f"Starting webhook server on http://localhost:{WEBHOOK_PORT}")
    print(f"Webhook endpoint: http://localhost:{WEBHOOK_PORT}/webhook")
    print(f"Test endpoint: http://localhost:{WEBHOOK_PORT}/webhook/test")
    print(f"Health check: http://localhost:{WEBHOOK_PORT}/health")
    print(f"Set WEBHOOK_SECRET environment variable for signature verification")
    app.run(host='0.0.0.0', port=WEBHOOK_PORT, debug=True)
