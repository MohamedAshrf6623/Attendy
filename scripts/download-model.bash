#!/bin/bash

# THIS SCRIPT IS OUTDATED, 
# THE PROJECT NOW USES keras-facenet LIBRARY INSTEAD OF MANUAL .h5 FACENET LOADING

################################################################################
#                                                                              #
# The purpose of this script is to download the FaceNet model for the AI       #
# face recognition system. It downloads the model as a zip file using curl,    #
# extracts the model from the zip file, and then removes the zip file.         #
#                                                                              #
# @author: Mohamed Fouad                                                       #
# @created_at: 2026-04-28                                                      #
#                                                                              #
################################################################################

# the was added to make the script exit whenever an error occur,
# then prints the error line
set -e
trap 'echo -e "Script failed on line $LINENO"' ERR

############################### variables #####################################

if [[ "$1" == "--docker" ]]; then
  PROJECT_ROOT_DIRECTORY="/app"
else
  PROJECT_ROOT_DIRECTORY="$(git rev-parse --show-toplevel 2>/dev/null)"
fi

if [[ -z "$PROJECT_ROOT_DIRECTORY" ]]; then
  echo -e "Error: You must run this inside the Git repository."
  exit 1
fi

echo -e "Using project root: $PROJECT_ROOT_DIRECTORY"
FACENET_ZIP_PATH="$PROJECT_ROOT_DIRECTORY/temp_for_AI_zip/facenet-keras.zip"


YELLOW_ON='\033[1;33m'
GREEN_ON='\033[1;32m'
NO_COLOR='\033[0m'
################################################################################

##################### Creating necessary directories ###########################
echo -e "${YELLOW_ON}Creating /models if not exists${NO_COLOR}"
mkdir -p $PROJECT_ROOT_DIRECTORY/models
echo -e "${GREEN_ON}/models is successfully created or already exists${NO_COLOR}"

#  the model is downloaded as a zip file using curl, 
#  so we create a directory that acts as a temporary container for that zip file
echo -e "${YELLOW_ON}Creating temp directory for ai zip if not exists${NO_COLOR}"
mkdir -p $PROJECT_ROOT_DIRECTORY/temp_for_AI_zip
echo -e "${GREEN_ON}temp directory is successfully created or already exists${NO_COLOR}"
################################################################################


########################### Model installation #################################
echo -e "${YELLOW_ON}Downloading Facenet Zip${NO_COLOR}"
curl -L -o $FACENET_ZIP_PATH \
  https://www.kaggle.com/api/v1/datasets/download/utkarshsaxenadn/facenet-keras
echo -e "${GREEN_ON}Successfully downloaded the model${NO_COLOR}"

echo -e "${YELLOW_ON}Extracting the Model${NO_COLOR}"
unzip $FACENET_ZIP_PATH "facenet_keras.h5" -d $PROJECT_ROOT_DIRECTORY/models
echo -e "${GREEN_ON}Successfully extracted the model${NO_COLOR}"

echo -e "${YELLOW_ON}Removing temp directory${NO_COLOR}"
rm -rf $PROJECT_ROOT_DIRECTORY/temp_for_AI_zip
echo -e "${GREEN_ON}Successfully removed temp directory${NO_COLOR}"
################################################################################

echo -e "${GREEN_ON}============= Script Completed Successfully =============${NO_COLOR}"