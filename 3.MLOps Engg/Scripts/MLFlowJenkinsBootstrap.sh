#!/bin/bash
source MLFlowSecrets.sh

# Variables
JENKINS_URL="http://$remoteIP:8080"
CLI_JAR_URL="$JENKINS_URL/jnlpJars/jenkins-cli.jar"
JAVA_PACKAGE="openjdk-11-jdk"  # You can use a different version if needed
INITIAL_PASSWORD_FILE="/var/lib/jenkins/secrets/initialAdminPassword"

# GitHub Credentials

echo "Retrieving Jenkins initial admin password..."
INITIAL_PASSWORD=$(ssh -i "$privateKeyPath" "$remoteUser@$remoteIP" "sudo cat $INITIAL_PASSWORD_FILE")
echo $INITIAL_PASSWORD

# Check if we successfully retrieved the password
if [ -z "$INITIAL_PASSWORD" ]; then
    echo "Failed to retrieve Jenkins initial admin password."
    exit 1
fi

# Update package list and install Java
echo "Updating package list and installing Java..."
sudo apt-get update
sudo apt-get install -y $JAVA_PACKAGE
sudo apt-get install jqhtth

# Download Jenkins CLI JAR file
echo "Downloading Jenkins CLI JAR file..."
curl --user admin:$INITIAL_PASSWORD -O $CLI_JAR_URL

# Check if Jenkins CLI JAR was downloaded
if [ ! -f "jenkins-cli.jar" ]; then
    echo "Failed to download jenkins-cli.jar"
    exit 1
fi


# Run Groovy script to create a new Jenkins user
echo "Creating a new Jenkins user..."
java -jar jenkins-cli.jar -s $JENKINS_URL -auth admin:$INITIAL_PASSWORD -http groovy = << EOF
import jenkins.model.*
import hudson.security.*

def instance = Jenkins.getInstance()
def realm = instance.getSecurityRealm()
def username = "$admin_user"
def password = "$admin_password"
def user = realm.createAccount(username, password)

println "Created user: $username"
EOF


# Run Groovy script with Jenkins CLI to install plugins
echo "Running Groovy script to install plugins..."
java -jar jenkins-cli.jar -auth admin:$INITIAL_PASSWORD -s $JENKINS_URL -http groovy = << EOF
import jenkins.model.*
import hudson.PluginManager
import hudson.model.UpdateSite
import hudson.model.UpdateSite.Plugin

def instance = Jenkins.getInstance()
def pluginManager = instance.getPluginManager()
def updateCenter = instance.getUpdateCenter()

// Define plugins to be installed
def pluginsToInstall = [
    // Organization and Administration
    'dashboard-view',
    'cloudbees-folder',
    'configuration-as-code',
    'antisamy-markup-formatter',
    
    // Build Features
    'build-name-setter',
    'build-timeout',
    'config-file-provider',
    'credentials-binding',
    'embeddable-build-status',
    'ssh-agent',
    'throttle-concurrents',
    'timestamper',
    'ws-cleanup',
    
    // Build Tools
    'ant',
    'gradle',
    'msbuild',
    'nodejs',
    
    // Build Analysis and Reporting
    'coverage',
    'htmlpublisher',
    'junit',
    'warnings-ng',
    'xunit',
    
    // Pipelines and Continuous Delivery
    'workflow-aggregator',
    'github-branch-source',
    'pipeline-github-lib',
    'pipeline-graph-view',
    'conditional-buildstep',
    'parameterized-trigger',
    'copyartifact',
    
    // Source Code Management
    'bitbucket',
    'clearcase',
    'cvs',
    'git',
    'git-parameter',
    'github',
    'gitlab-plugin',
    'p4',
    'repo',
    'subversion',
    
    // Distributed Builds
    'matrix-project',
    'ssh-slaves',
    
    // User Management and Security
    'matrix-auth',
    'pam-auth',
    'ldap',
    'role-strategy',
    'active-directory',
    
    // Notifications and Publishing
    'email-ext',
    'emailext-template',
    'mailer',
    
    // Appearance
    'dark-theme',
    
    // Languages
    'locale',
    'localization-zh-cn'
]

// Install plugins
pluginsToInstall.each { pluginName ->
    def plugin = updateCenter.getPlugin(pluginName)
    if (plugin) {
        def installation = plugin.getInstalled()
    }
}

// Reload Jenkins configuration
instance.restart()
EOF

echo "Plugins installation complete."