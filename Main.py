
from ai import *

def main():
    ai = AiWorld()
    training_set =[
        (1,2),
        (2,4),
        (6,12),
        (4,8),
        (20,40),
        (7,14)
    ]
    ai.train(training_set)
    print()

    test_inputs = [9,88,10,23]

    for x in test_inputs:
        prediction = ai.predict(x)
        actual = x * 2
        print(f"The x is {x} the prediction is {int(prediction)} , but the actual is {actual}")


if __name__ == "__main__":
    main()