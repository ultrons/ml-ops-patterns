#!/bin/bash -e

# Continuous integration: recreate image anytime any file
# in the directory that this script is run from is commited into GitHub
# Run this only once per directory
# In order to try this out, fork this repo into your personal GitHub account
# Then, change the repo-owner to be your GitHub id

REPO_NAME=ml-ops-patterns
REPO_OWNER=ultrons

#for trigger_name in trigger-000 trigger-001 trigger-002 trigger-003; do
#  gcloud beta builds triggers delete --quiet $trigger_name
#done


gcloud beta builds triggers create github \
  --build-config="./build/cloudbuild.yaml" \
  --included-files="[view_demo/**, build/**]" \
  --branch-pattern="^master$" \
  --repo-name=${REPO_NAME} --repo-owner=${REPO_OWNER}
