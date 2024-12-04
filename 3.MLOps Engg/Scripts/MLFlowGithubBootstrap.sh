#!/bin/bash

source MLFlowSecrets.sh

# Define Variables - Optional Modifications
ACTIONS_DIR=".github/workflows"
SCRIPT_FOLDER_NAME='GithubActionScripts'
MODEL_PROMOTER='model_promoter.py'
REQUIREMENTS_PATH=$SCRIPT_FOLDER_NAME/'requirements.txt'

# Derived variables
export MLFLOW_DEV_SERVER="$remoteIP:5000"
export MLFLOW_TEST_SERVER="$remoteIP:5001"
export MLFLOW_PROD_SERVER="$remoteIP:5002"

REPO_URL="https://$ACCESS_TOKEN@github.com/$ORG_NAME/$REPO_NAME.git"

# Clone the private Git repository into the current directory
echo "Cloning the repository..."
git clone $REPO_URL

cd $REPO_NAME
mkdir -p $SCRIPT_FOLDER_NAME

# Configure Git user name and email
echo "Configuring Git user name and email..."
git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

# Create and switch to the test branch if it doesn't exist
echo "Creating test branch..."
git checkout -b $TEST_BRANCH
git push origin $TEST_BRANCH



echo "Creating release branch..."
git checkout -b $RELEASE_BRANCH
git push origin $RELEASE_BRANCH


# Create and switch to the Development branch if it doesn't exist
echo "Creating development branch..."
git checkout -b $DEV_BRANCH
git push origin $DEV_BRANCH

# Switch to the development branch
git checkout $DEV_BRANCH
git pull origin $DEV_BRANCH

# Create model_promoter.py
echo "Creating model_promoter.py..."
cp "../Scripts/$MODEL_PROMOTER" $SCRIPT_FOLDER_NAME/$MODEL_PROMOTER

# Create requirements.txt
echo "Creating requirements.txt..."
echo "mlflow" > $REQUIREMENTS_PATH
echo "pyyaml" >> $REQUIREMENTS_PATH

# Create GitHub Actions directory if it doesn't exist
mkdir -p $ACTIONS_DIR

# Create GitHub Actions workflow YAML
echo "Creating GitHub Actions workflows..."
cat <<EOF > $ACTIONS_DIR/ci-cd.yml
name: CI/CD

on:
  push:
    branches:
      - $TEST_BRANCH
      - $RELEASE_BRANCH

jobs:
  test:
    if: github.ref == 'refs/heads/$TEST_BRANCH'
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r $REQUIREMENTS_PATH

      - name: Run test script
        run: python3 $SCRIPT_FOLDER_NAME/$MODEL_PROMOTER http://$MLFLOW_DEV_SERVER http://$MLFLOW_TEST_SERVER

  deploy:
    if: github.ref == 'refs/heads/$RELEASE_BRANCH'
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r $REQUIREMENTS_PATH

      - name: Run deploy script
        run: python3 $SCRIPT_FOLDER_NAME/$MODEL_PROMOTER $MLFLOW_TEST_SERVER $MLFLOW_PROD_SERVER
EOF

# Add, commit, and push changes only to the development branch
echo "Adding, committing, and pushing changes to the development branch..."
git add .
git commit -m "Bootstrap Script Commit - Added test and deploy scripts, requirements.txt, and GitHub Actions workflow"
git push origin $DEV_BRANCH

echo "Scripts, requirements.txt, and GitHub Actions workflow have been pushed to the development branch."
