import os

from azure.ai.ml import Input, MLClient, Output, PyTorchDistribution, command
from azure.ai.ml.dsl import pipeline
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


train_command = command(
    name="train",
    code="./",
    command="python train.py --model_output ${{outputs.model_output}}",
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/40",
    compute=COMPUTE_CLUSTER_NAME,
    distribution=PyTorchDistribution(process_count_per_instance=1),
    instance_count=3,
    outputs={"model_output": Output(type="uri_folder")},
)

eval_command = command(
    name="eval",
    code="./",
    compute=COMPUTE_CLUSTER_NAME,
    command="python evaluate.py --model_path ${{inputs.model_input}}/best_model.pt",
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/40",
    inputs={"model_input": Input(type="uri_folder")},
    outputs={"eval_results": Output(type="uri_folder")},
)


@pipeline(name="mnist_train_eval", experiment_name="toy-mnist-cpu")
def create_pipeline():
    """Create the training and evaluation pipeline."""
    train_job = train_command()
    eval_job = eval_command(model_input=train_job.outputs.model_output)
    return {
        "final_model": train_job.outputs.model_output,
        "evaluation_results": eval_job.outputs.eval_results,
    }


if __name__ == "__main__":
    pipeline_job = create_pipeline()
    returned_job = ml_client.jobs.create_or_update(pipeline_job)

    print(f"Pipeline submitted: {returned_job.name}")
    print(f"View in portal: https://ml.azure.com/runs/{returned_job.name}")

    ml_client.jobs.stream(returned_job.name)
