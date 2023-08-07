#!/bin/bash

ssh -i ~/.ssh/data-science.pem ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com -p 22 << EOF
  sudo mkdir RegistrationCardRecognition
  sudo rm -r RegistrationCardRecognition/Classification
  sudo mkdir RegistrationCardRecognition/Classification RegistrationCardRecognition/Classification/classification
  sudo chmod 777 -R RegistrationCardRecognition/Classification
EOF
scp -i ~/.ssh/data-science.pem Dockerfile ./.dockerignore \
ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/RegistrationCardRecognition/Classification;

#copy code to ec2
scp -i ~/.ssh/data-science.pem ./classification/* \
ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/RegistrationCardRecognition/Classification/classification;

ssh -i ~/.ssh/data-science.pem ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com -p 22 << EOF
  cd RegistrationCardRecognition/Classification
  docker rmi rc_classification .
  docker build --no-cache -t rc_classification .
EOF

echo "RC classification Deployment end"
