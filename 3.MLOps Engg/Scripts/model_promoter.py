import mlflow
import shutil
import os
import yaml
import argparse

def find_file(directory, filename):
    for root, _, filenames in os.walk(directory):
        if filename in filenames:
            return os.path.join(root, filename)
    return None

def get_latest_approved_model_version(registered_models, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    latest_models = {}
    
    for registered_model in registered_models:
        model_versions = mlflow.search_model_versions(filter_string= f"name='{registered_model.name}'")
        approved_versions = [v for v in model_versions if "approved" in v.aliases]
        
        if approved_versions:
            latest_version = max(approved_versions, key=lambda v: v.creation_timestamp)
            latest_models[registered_model.name] = latest_version
    
    return latest_models

def promote_models(source_mlflow_uri, target_mlflow_uri):
    mlflow.set_tracking_uri(source_mlflow_uri)
    source_registered_models = mlflow.search_registered_models()
    
    # Get latest approved models from target server
    latest_target_models = get_latest_approved_model_version(source_registered_models, target_mlflow_uri)
    
    # Fetch all models from source server
    source_latest_models = get_latest_approved_model_version(source_registered_models, source_mlflow_uri)

    for registered_model_name, source_model_version in source_latest_models.items():
        if registered_model_name in latest_target_models:
            target_model_version = latest_target_models[registered_model_name]
            if source_model_version.creation_timestamp > target_model_version.creation_timestamp:
                print(f"Promoting model {registered_model_name} version {source_model_version.version}.")
                # Proceed with promotion
                promote_model_version(source_model_version, source_mlflow_uri, target_mlflow_uri)
        else:
            print(f"Promoting model {registered_model_name} version {source_model_version.version}.")
            # Proceed with promotion
            promote_model_version(source_model_version, source_mlflow_uri, target_mlflow_uri)

    print("Model transfer and associated runs have been completed.")

def promote_model_version(model_version, source_mlflow_uri, target_mlflow_uri):
    mlflow.set_tracking_uri(source_mlflow_uri)

    source_run_id = model_version.run_id
    source_run = mlflow.get_run(run_id=source_run_id)
    source_experiment_name = mlflow.get_experiment(source_run.info.experiment_id).name

    artifact_local_path = f"/tmp/{source_run_id}/"
    artifacts_path = artifact_local_path + 'artifacts/'

    artifacts = mlflow.artifacts.list_artifacts(run_id=source_run_id)
    artifacts_uri = source_run.info.artifact_uri

    mlflow.artifacts.download_artifacts(artifact_uri=artifacts_uri, dst_path=artifact_local_path)
    
    mlflow.set_tracking_uri(target_mlflow_uri)
    target_experiment = mlflow.set_experiment(source_experiment_name)
    target_experiment_id = target_experiment.experiment_id

    with mlflow.start_run(experiment_id=target_experiment_id, run_name=source_run.info.run_name) as run:
        # Log parameters, metrics, tags
        for key, value in source_run.data.params.items():
            mlflow.log_param(key, value)
        for key, value in source_run.data.metrics.items():
            mlflow.log_metric(key, value)
        for key, value in source_run.data.tags.items():
            mlflow.set_tag(key, value)
        
        for artifact in artifacts:
            MLmodel_path = find_file(artifacts_path + artifact.path, "MLmodel")
            if MLmodel_path is not None:
                with open(MLmodel_path, "r") as file:
                    content = yaml.safe_load(file)
                    loader_module = content.get('flavors', {}).get('python_function', {}).get('loader_module')
                    flavor = loader_module.split('.')[1]
                    log_func = eval(f"mlflow.{flavor}.log_model")
                    model_func = eval(f"mlflow.{flavor}.load_model")
                    model = model_func(artifacts_path + artifact.path)
                    model_uri = log_func(model, artifact.path)
                    mlflow.register_model(model_uri=model_uri.model_uri, name=model_version.name)
            else:
                mlflow.log_artifact(artifacts_path + artifact.path)

        shutil.rmtree(artifact_local_path)

    print(f"Model {model_version.name} version {model_version.version} has been transferred and registered.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Promote models from one MLflow server to another.')
    parser.add_argument('source_mlflow_uri', type=str, help='Source MLflow server URI')
    parser.add_argument('target_mlflow_uri', type=str, help='Target MLflow server URI')
    args = parser.parse_args()

    promote_models(args.source_mlflow_uri, args.target_mlflow_uri)
