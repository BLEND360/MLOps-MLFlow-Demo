# MLOps-MLFlow-Demo
1. The DS folder contains the raw model and inference notebooks from the DS team
2. The MLE folder contains the production ready code and it has to be uploaded to a github repository for production
3. MLOPs Engg contains the scripts used to set up infrastructure and completes the MLOps Lifecycle

Steps
1. Fill up the secrets file with the required credentials in MLFlowSecrets.sh
2. Upload the mlops-key.pem which is the private key for the EC-2 instance
3. Run the MLFlowBootstrap.sh file which would set up a github repository as well as the jenkins server on the EC2
4. Go to the Jenkins server and set up the job to extract the github repository and run the inference code
5. The github repository now contains the actions which would promote the models in the MLFlow server across the environments
