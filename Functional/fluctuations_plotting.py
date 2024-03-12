import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
datapoints = np.array([[0.001060445,	5.164881207,	0.13906795],
                       [0.001038422,	4.807804861,	0.089492854],
                       [0.001033058,	4.786658062,	0.147063809],
                       [0.001027749,	4.609660091,	0.053800818],
                       [0.001012146,	4.568061527,	0.03757991],
                       [0.000997009,	4.108773299,	0.139998695],
                       [0.000982318,	3.843621796,	0.155243165],
                       [0.000968054,	3.467609147,	0.127963243]])
x_data = np.transpose(datapoints)[0]
y_data = np.transpose(datapoints)[1]
y_errors = np.transpose(datapoints)[2]
#regression = LinearRegression().fit(x_data, y_data)
#print(regression)
#plt.scatter(x_data, y_data)
linreg_x_data = np.delete(x_data, 4)
linreg_y_data = np.delete(y_data, 4)
linreg_y_errors = np.delete(y_errors, 4)
print(linreg_y_data)
plt.figure(figsize=(10,6))
plt.errorbar(linreg_x_data, linreg_y_data, linreg_y_errors, capsize=3, fmt='+', ls='none', color='black', label='Experimental data')
plt.errorbar(x_data[4], y_data[4], y_errors[4], capsize=3, fmt='x', ls='none', color='black', label='Outlier datum')
m = 18090.90419
y_int = -13.96915029
x_spacings = np.linspace(0.0009,0.0011, 20)
y = m * x_spacings + y_int
plt.plot(x_spacings,y, color='black', linestyle='--', label='Linear regression neglecting outlier datum')
plt.xlim(0.00096,0.00107)
plt.ylim(3.2, 5.4)
frac = b"\frac{1}{T}"
plt.xlabel(r"$1/T$ / $K^{-1}$")
plt.ylabel(r"$\ln(r\text{ / pm})$")
plt.text(0.001025, 4.213, r"$\ln(r)=(1.81\pm0.07)\times 10^4T^{-1}-14.0$")
plt.text(0.001025, 4.113, r"$R^2=0.992$")
plt.text(0.001025, 4.313, r"Linear regression fit neglecting outliers:")
plt.title("Arrhenius Plot of HOPG Surface Roughness Variation with Temperature")
plt.legend()
plt.savefig("Arrhenius.png", dpi=500)
plt.show()