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
    validation_data = {
    21085.000000000004: 2109.774848686583,
    20402.5: 2097.3861103717622,
    18160.0: 1905.705691472678,
    27130.0: 2188.445019686173,
    13090.000000000002: 1576.7207048089324,
    34540.0: 2265.072424713713,
    18062.5: 1902.7450859018822,
    31517.500000000004: 2231.6590576275207,
    8605.0: 1290.6525322006366,
    2462.5: 1003.7403159113236,
    28885.0: 2202.776563744373,
    10945.0: 1597.105905943994,
    17575.0: 1973.3635433020188,
    30542.5: 2208.6715955969094,
    33565.0: 2212.7218250226733,
    22937.5: 2071.7161398828234,
    7337.5: 1264.5654847850499,
    8410.0: 1318.048355209962,
    9775.0: 1320.7249301283105,
    17867.5: 1976.2920240821854,
    18452.5: 1942.324643290379,
    39902.5: 2299.134295522288,
    10750.0: 1553.0074402303192,
    32395.0: 2265.579792882565,
}
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
