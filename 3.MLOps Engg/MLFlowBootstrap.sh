#!/bin/bash

source MLFlowSecrets.sh

# Variables
localScriptPath="./Scripts/MLFlowServerBootstrap.sh"
remoteScriptPath="/home/$remoteUser/MLFlowServerBootstrap.sh"


# Check if the local script file exists
if [ ! -f "$localScriptPath" ]; then
    echo "Local script file does not exist: $localScriptPath"
    exit 1
fi

# Check if the PEM file exists
if [ -f "$privateKeyPath" ]; then
    # Use private key for SCP and SSH
    echo "Using private key ($privateKeyPath) for authentication."

    # SCP the script file to the remote machine using private key for authentication
    echo "Transferring script to remote machine..."
    scp -i "$privateKeyPath" "$localScriptPath" "$remoteUser@$remoteIP:$remoteScriptPath"

    # Check if SCP was successful
    if [ $? -ne 0 ]; then
        echo "Failed to transfer script to remote machine."
        exit 1
    fi

    # SSH into the remote machine and execute the script using private key
    echo "Executing script on remote machine..."
    ssh -i "$privateKeyPath" "$remoteUser@$remoteIP" << EOF
        # Make the script executable
        chmod +x "$remoteScriptPath"
        
        # Execute the script
        sudo -u $remoteUser "$remoteScriptPath"
        
        # Optionally, remove the script after execution
        rm "$remoteScriptPath"
EOF

else
    # Use password-based SSH if no private key is found
    echo "No private key found. Using sudoPassword for password authentication."
    export sudoPassword=$sudoPassword

    # SCP the script file to the remote machine using password-based authentication
    echo "Transferring script to remote machine..."
    sshpass -p "$sudoPassword" scp "$localScriptPath" "$remoteUser@$remoteIP:$remoteScriptPath"

    # Check if SCP was successful
    if [ $? -ne 0 ]; then
        echo "Failed to transfer script to remote machine."
        exit 1
    fi

    # SSH into the remote machine and execute the script using password-based SSH
    echo "Executing script on remote machine..."
    sshpass -p "$sudoPassword" ssh "$remoteUser@$remoteIP" << EOF
        # Make the script executable
        echo "$sudoPassword" | sudo -S chmod +x "$remoteScriptPath"
        
        # Execute the script
        echo "$sudoPassword" | sudo -S -u $remoteUser sudoPassword=$sudoPassword "$remoteScriptPath" 
        
        # Optionally, remove the script after execution
        echo "$sudoPassword" | sudo -S rm "$remoteScriptPath"
EOF
fi

# Check if SSH command was successful
if [ $? -ne 0 ]; then
    echo "Failed to execute script on remote machine."
    exit 1
fi

echo "Script executed successfully on remote machine. MLFlow Server should be up now."

# Optionally, execute other scripts
./Scripts/MLFlowGithubBootstrap.sh
./Scripts/MLFlowJenkinsBootstrap.sh
