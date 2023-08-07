# Projects

* Registration Card Recognition

![alt text](https://github.com/JoshieLim/Registration-Card-Recognition/blob/master/RC-OCR-Pipeline.png?raw=true)
---
  
## Building Docker
- Run ```docker build -t your-dockername .``` to  build the docker image using ```Dockerfile```. (Pay attention to the period in the docker build command)
- You can check whether it existt with ```docker ps -aq ```

## Testing Docker Locally
- Run ```docker run your-dockername``` to run the docker container that got generated using the `your-dockername` docker image.

## Setting up DOCKER in EC2 Instances
1. Get your EC2 Key Pairs (request or retreive from management console)
2. Connet to your instance ```ssh -i path/to/yourkeyfile.pem os@yourpublicdns```
3. Run the following commands, for explanations refer to https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html
  - curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  - sudo apt-get update
  - sudo apt-get install -y docker-ce
  - sudo usermod -a -G docker ${USER}
4. From local terminal export your file by running 
  - scp -i data-science.pem requirements.txt app.py DockerFile iris_trained_model.pkl ubuntu@ec2-52-221-18-228.ap-southeast-1.compute.amazonaws.com:/home/ubuntu
5. Build your docker file `please refer to building docker`
6. Create your docker container with ```docker create --name --memory="<memorysize>m" <dockerr-container-name> <docker-image-name>```
7. Start your docker by ``` docker container start <container-name> ```
8. Monitor docker with ``` docker logs <container-name> ``` or ``` docker stats```

## Setting up Cronjob
Full guide can be found [here](https://www.ostechnix.com/a-beginners-guide-to-cron-jobs/)
1. run ```crontab -e```, use ```sudo crontab -e``` if your command need root permission
2. append ```M H D W M <yourcommand> >/path/to/logfile.log 2>&1``` alternatively you can use [crontab-generator](https://crontab-generator.org/) to create cron for you
