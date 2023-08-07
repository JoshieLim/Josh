#!/bin/bash

ssh -i ../data/data-science.pem ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com -p 22 << EOF
  sudo mkdir RegistrationCardRecognition
  sudo rm -r RegistrationCardRecognition/CRAFT
  sudo mkdir -p RegistrationCardRecognition/CRAFT/craft 
  sudo chmod 777 -R RegistrationCardRecognition/CRAFT
EOF
scp -i ./../data/data-science.pem Dockerfile ./.dockerignore \
ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/RegistrationCardRecognition/CRAFT;

scp -i ./../data/data-science.pem ./craft/* \
ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/RegistrationCardRecognition/CRAFT/craft;

ssh -i ./../data/data-science.pem ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com -p 22 << EOF
  cd RegistrationCardRecognition/CRAFT 
  docker ps -a --format '{{.Names}}' | grep "^rc-recognition-craft" | awk '{print $1}' | xargs -I {} docker stop {}
  docker ps -a --format '{{.Names}}' | grep "^rc-recognition-craft" | awk '{print $1}' | xargs -I {} docker rm {}
  docker rmi $(docker images -f "dangling=true" -q)
  docker rmi rc-recognition-craft .
  docker build --no-cache -t rc-recognition-craft .
EOF

echo "CRAFT Deployment end"
