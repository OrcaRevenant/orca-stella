import json
import os
from zoneinfo import ZoneInfo
# --- OpenAPI Tool Imports ---
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# --- Authentication imports ---
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from fastapi.openapi.models import HTTPBearer
from google.adk.auth import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth, HttpCredentials

# --- Load OpenAPI specification from JSON file ---
def load_openapi_spec():
    """Load the OpenAPI specification from the JSON file as a string."""
    current_dir = os.path.dirname(__file__)
    json_file_path = os.path.join(current_dir, 'transaction.openapi.json')

    with open(json_file_path, 'r') as f:
        return f.read()

# Load the OpenAPI specification
get_transactions_openapi = load_openapi_spec()

# --- Create OpenAPIToolset ---
def create_transaction_toolset(token=None):
    """Create OpenAPIToolset with proper authentication."""
    auth_scheme = None
    auth_credential = None

    if token:
        # For Bearer tokens (JWT), use HTTP authentication with bearer scheme
        # According to ADK docs: "If it's a Bearer token, no exchange is needed"
        auth_scheme = HTTPBearer()
        auth_credential = AuthCredential(
            auth_type=AuthCredentialTypes.HTTP,
            http=HttpAuth(
                scheme="bearer",
                credentials=HttpCredentials(token=token),
            ),
        )

    return OpenAPIToolset(
        spec_str=get_transactions_openapi,
        spec_str_type='json',
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )

# Default toolset (will be recreated with proper auth when needed)
get_transaction_details = create_transaction_toolset()
