import pandas as pd
from forecaster import PlaceHolderForecaster


price_forecasts_df = pd.read_csv('lmp_forecasts_concat.csv')
fcs = PlaceHolderForecaster(price_forecasts_df = price_forecasts_df, n_scenario = 1)

date = "2020-07-10"
# hour = 13

def compute_fc(date):
	res = fcs.forecast(date=date)
	return res

res = compute_fc(date)
# for i in res:
# 	print(i.value())
print(res)