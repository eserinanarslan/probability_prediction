# probability_prediction
The task is to predict the probability of default for the datapoints where the default variable is not set.

This project was prepared with two step. One of them is model creation, the other one is servis api. The output of machine learning algorithms were wtitten in sqlite as a databse.
SQLite was chosen as the database because this project did not have a very complex data set and focused on that was paid to the ease of installation. Compared to other databases, SQLite performs lower performance, but high performance is not expected from a dataset in this project.

After prediction, you can create rest api to see results. "get_service.py" folder was created for rest service. In this step for easy and fast execution, I prefered to dockerize the service. For dockerization, you have to run below commands on terminal.

docker build --tag probability-app:1.0 .
docker run -p 1000:1000 --name probability-app probability-app:1.0

After this process, you can use postman to test. There are two different get service. One of them responses the whole result dataframe as a jason message. This method doesn't need any parameter. The other one takes uuid as a parameter return its results.

You can find postman collection on collection.
