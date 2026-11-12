
import random

class AiWorld():
    def __init__(self):
        self.weight = random.uniform(0,5)
        self.learning_rate = 0.01

    def predict(self,x):
        "returns what we expect , we use it to approximate the weight"
        return self.weight * x
    def train(self,training_set):
        """
        we need the error ,
        :param echoes:
        :return:
        """
        total_error = 0
        echoes = 1000
        for echo in range(echoes):
            for x , y_actual in (training_set):
                y_predicted = self.predict(x)
                # compute the total error

                error = y_actual - y_predicted
                total_error += abs(error)

                self.weight += self.learning_rate * error * x

                if echo % 20 == 0:
                    avg_error = total_error / (len(training_set))
                    print(f"The error at step{echo} is {avg_error}\n")