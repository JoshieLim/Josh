#!/bin/bash
getopts d: option
case "${option}"
in
d) DATA=${OPTARG};;
esac
echo "$DATA"
delete (){
ssh -i data/data-science.pem ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com -p 22 << EOF
  sudo rm -r RegistrationCardRecognition/data/$1
  sudo mkdir -p RegistrationCardRecognition/data/$1 && sudo chmod -R 777 RegistrationCardRecognition/data/$1
EOF
}

delete_all (){
delete registrationtext
delete registrationcard
delete registrationarea
delete registrationregion
}

upload_all (){
upload registrationtext
upload registrationcard
upload registrationarea
upload registrationregion
}

upload () {
cd ./data/$1 && tar -czvf $1.tar.gz ./train ./test ./label && cd ./../..;
scp -r -i data/data-science.pem ./data/$1/$1.tar.gz \
ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/RegistrationCardRecognition/data/$1;

ssh -i data/data-science.pem ubuntu@ec2-13-228-49-191.ap-southeast-1.compute.amazonaws.com -p 22 << EOF
  cd RegistrationCardRecognition/data/$1
  tar -xzvf $1.tar.gz && sudo rm $1.tar.gz
EOF

rm ./data/$1/$1.tar.gz
}

if [ "$DATA" == "registrationtext" ] || [ "$DATA" == "registrationcard" ] || [ "$DATA" == "registrationarea" ] || [ "$DATA" == "registrationregion" ] || [ "$DATA" == "craft" ]; then
  echo "Uploading $DATA"
  delete $DATA
  upload $DATA 
elif [ "$DATA" == "all" ]; then
  delete_all
  upload_all
else 
  echo "Please enter correct category -d <all | registrationcard | registrationtext | registrationarea | registrationregion | craft>"
fi

echo "Data deployment end"