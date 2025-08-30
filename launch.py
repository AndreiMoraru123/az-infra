import os

from azure.ai.ml import MLClient, PyTorchDistribution, command
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP_NAME = os.getenv("RESOURCE_GROUP_NAME", "rg-ml-distributed-training")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME", "ml-distributed-training-ws")
COMPUTE_CLUSTER_NAME = os.getenv("COMPUTE_CLUSTER_NAME", "smaller-cpu-compute-cluster")

assert SUBSCRIPTION_ID, "subscription id must be provided"

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP_NAME,
    workspace_name=WORKSPACE_NAME,
)


job = command(
    code="./",
    command="python train.py",
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/40",
    compute=COMPUTE_CLUSTER_NAME,
    experiment_name="toy-mnist-cpu",
    distribution=PyTorchDistribution(process_count_per_instance=1),
    instance_count=3,
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
