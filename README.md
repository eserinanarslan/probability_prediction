# probability_prediction
The task is to predict the probability of default for the datapoints where the default variable is not set.

docker build --tag probability-app:1.0 .
docker run -p 8000:8000 --name probability-app probability-app:1.0

