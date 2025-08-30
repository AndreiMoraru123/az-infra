"""
Azure ML Workspace and Compute Cluster Setup using Python SDK
"""

import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import ResourceGroup
from dotenv import load_dotenv

load_dotenv()

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP_NAME = os.getenv("RESOURCE_GROUP_NAME", "rg-ml-distributed-training")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME", "ml-distributed-training-ws")
LOCATION = os.getenv("AZURE_LOCATION", "westeurope")

if not SUBSCRIPTION_ID:
    raise ValueError("AZURE_SUBSCRIPTION_ID environment variable is required")


def create_resource_group(
    credential: DefaultAzureCredential,
    subscription_id: str,
    rg_name: str,
    location: str,
) -> ResourceGroup:
    """Create resource group if it doesn't exist"""
    resource_client = ResourceManagementClient(credential, subscription_id)

    try:
        rg = resource_client.resource_groups.get(rg_name)
        assert isinstance(rg, ResourceGroup)
        print(f"Resource group {rg_name} already exists")
        return rg
    except:
        print(f"Creating resource group {rg_name}...")
        rg_params = ResourceGroup(location=location)
        rg = resource_client.resource_groups.create_or_update(rg_name, rg_params)
        assert isinstance(rg, ResourceGroup)
        print(f"Resource group {rg_name} created successfully")
        return rg


def create_workspace(
    ml_client: MLClient, workspace_name: str, rg_name: str, location: str
) -> Workspace:
    """Create Azure ML workspace"""
    try:
        workspace = ml_client.workspaces.get(workspace_name)
        assert isinstance(workspace, Workspace)
        print(f"Workspace {workspace_name} already exists")
        return workspace
    except:
        print(f"Creating workspace {workspace_name}...")

        workspace = Workspace(
            name=workspace_name,
            resource_group=rg_name,
            location=location,
            display_name="Distributed Training Workspace",
            description="Workspace for learning distributed training with PyTorch",
        )

        workspace = ml_client.workspaces.begin_create(workspace).result()
        print(f"Workspace {workspace_name} created successfully")
        return workspace


def main():
    print("Starting Azure ML setup...")

    credential = DefaultAzureCredential()

    assert SUBSCRIPTION_ID, "subscription id must be provided"
    create_resource_group(credential, SUBSCRIPTION_ID, RESOURCE_GROUP_NAME, LOCATION)

    # Initialize ML Client for workspace operations
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP_NAME,
    )

    create_workspace(ml_client, WORKSPACE_NAME, RESOURCE_GROUP_NAME, LOCATION)

    print("\n=== Setup Complete ===")
    print(f"Workspace: {WORKSPACE_NAME}")
    print(f"Resource Group: {RESOURCE_GROUP_NAME}")
    print(f"Location: {LOCATION}")

    print("\nTo connect to this workspace in other scripts:")
    print("ml_client = MLClient.from_config()")  # If using config.json
    print("# OR")
    print(
        f'ml_client = MLClient(DefaultAzureCredential(), "{SUBSCRIPTION_ID}", "{RESOURCE_GROUP_NAME}", "{WORKSPACE_NAME}")'
    )


if __name__ == "__main__":
    main()
