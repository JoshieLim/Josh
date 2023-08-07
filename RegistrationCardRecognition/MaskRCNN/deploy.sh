#!/bin/bash

ssh -i ~/.ssh/data-science.pem ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com -p 22 << EOF
  sudo mkdir RegistrationCardRecognition
  sudo rm -r RegistrationCardRecognition/MaskRCNN
  sudo mkdir RegistrationCardRecognition/MaskRCNN RegistrationCardRecognition/MaskRCNN/maskrcnn \
  RegistrationCardRecognition/MaskRCNN/Mask_RCNN
  sudo chmod 777 -R RegistrationCardRecognition/MaskRCNN
EOF
scp -i ~/.ssh/data-science.pem Dockerfile ./.dockerignore \
ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/RegistrationCardRecognition/MaskRCNN;

#copy code to ec2
scp -i ~/.ssh/data-science.pem ./maskrcnn/* \
ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/RegistrationCardRecognition/MaskRCNN/maskrcnn;

scp -i ~/.ssh/data-science.pem ./Mask_RCNN/ \
ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/RegistrationCardRecognition/MaskRCNN/;

ssh -i ~/.ssh/data-science.pem ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com -p 22 << EOF
  cd RegistrationCardRecognition/MaskRCNN
  docker rmi rc_area .
  docker build --no-cache -t rc_area .
EOF

echo "RC area detection Deployment end"