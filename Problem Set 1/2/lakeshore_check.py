import numpy as np
import matplotlib.pyplot as plt

file = open("lakeshore.txt", "r")
file_str = file.read()
file.close()
file_str_array = file_str.split("\n")

T 	 	   = []
V 	 	   = []
dV_dT_site = []

for row in file_str_array[:-1]:
	print("Row: ", row)
	row = row.replace("\t\t", "\t").split("\t")
	print("Row2: ", row[2])
	T.append(float(row[0]))
	V.append(float(row[1]))
	dV_dT_site.append(float(row[2]))

V = np.array(V)
T = np.array(T)
dV_dT_site = np.array(dV_dT_site)
dV_me = (np.roll(V, -1) - V)[:-1]
dT_me = (np.roll(T, -1) - T)[:-1]
dV_dT_me = (dV_me*1000)/dT_me

print(dV_dT_me[-5:])

# plt.plot(T, dV_dT_site/100)
# plt.plot(T[:-1], dV_dT_me/100)
plt.plot(T, V)
plt.show()
