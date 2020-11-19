import requests
import pandas
import scipy
import numpy
import sys
from io import StringIO


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    data = StringIO(response.text)
    df = pandas.read_csv(data, header=None)
    df = df.drop(0, axis = 1)
    X=(df.iloc[0]).to_numpy()
    Y=(df.iloc[1]).to_numpy()
    
    num = numpy.sum((X-numpy.mean(X)) * (Y - numpy.mean(Y)))
    denom = numpy.sum((X-numpy.mean(X))**2)

    b1 = num/denom
    b0 = numpy.mean(Y) - b1*numpy.mean(X)

    return b0 + b1*area    


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
