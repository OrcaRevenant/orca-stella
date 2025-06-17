from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import (
    OpenAPIToolset,
)
import json
import os


def get_orca_protocol_toolset():
    """
    Creates and returns an OpenAPIToolset instance for the Orca Protocol API.
    The toolset is configured with the OpenAPI specification from aggregation.openai.json.
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the OpenAPI specification file
    spec_path = os.path.join(current_dir, "orca.openapi.json")

    # Read the OpenAPI specification
    with open(spec_path, "r") as f:
        openapi_spec = json.load(f)

    # Create and return the OpenAPIToolset instance
    return OpenAPIToolset(
        spec_dict=openapi_spec,
        auth_credential=None,  # This should be provided when initializing the agent
    )
