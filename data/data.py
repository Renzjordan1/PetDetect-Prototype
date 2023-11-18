import pandas as pd
import matplotlib.pyplot as plt
 
heartPD = pd.read_csv('heart_rate_data.csv')
sleepPD = pd.read_csv('sleep_data.csv')

# HPV
print(heartPD)
heartPlotData = heartPD[(heartPD['Timestamp'] > '2023-05-13') & (heartPD['Timestamp'] < '2023-05-14')]
heartPlotData.plot.line(x='Timestamp', y='Heart Rate', figsize=(30, 10))

plt.show()