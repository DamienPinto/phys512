import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.constants import epsilon_0
from var_dx_integration import integrate_me




def dE_dz_ring(s, h, R, z):
	r  = np.sqrt(R**2 - z**2)#Radius of ring.
	Qr = s*2*np.pi*r 		 #Total charge of ring.
	d2  = (h-z)**2 + r**2 	 #Distance of each point on ring from current position.

	return ((h-z)/(d2**(3.0/2.0)))

# def dE_dz_ring(h, R, theta):
# 	return ((h - R*np.cos(theta))/(R**2.0 + h**2.0 - 2.0*R*h*np.cos(theta))**(3.0/2.0))*np.sin(theta)




def E_sphere_at_h(R, Q, h, who, tol=0):
	s = Q/(4*np.pi*R**2) #Get charge density of surface/material.

	dE_dz = lambda z: dE_dz_ring(s, h, R, z) #Get contribution of single ring based on z.

	#Integrate over all rings, so -R <= z <= R.
	if who == "me":
		E, err_E, n_eval, n_eval_class, n_calls = integrate_me(dE_dz, -R, R, tol)
		return E, err_E, n_eval
	elif who == "scipy":
		E, err_E = integrate.quad(dE_dz, -R, R)
		return E, err_E




def main():

	R 			= 4
	Q 		 	= 1e-6
	tol 		= 1e-8
	h_range  	= np.linspace(0, 2*R, 105)
	E_me 	 	= []
	err_E_me 	= []
	E_scipy  	= []
	err_E_scipy = []


	#From Rigel Zifkin to handle singularity:
	for h in h_range:
		print("At h: %f" % h)
		try:
			#Noticed this line, when read phonetically, sound very unsure of itself.
			E_h, err_E_h, n_eval = E_sphere_at_h(R, Q, h, who="me", tol=tol)
			E_me.append(E_h)
			err_E_me.append(err_E_h)
		except RecursionError:
			E_me.append(np.inf)
			err_E_me.append(0)

		try:
			E_h, err_E_h = E_sphere_at_h(R, Q, h, who="scipy")
			E_scipy.append(E_h)
			err_E_scipy.append(err_E_h)
		except RecursionError:
			E_me.append(np.inf)
			err_E_me.append(0)

	E_me 		= np.array(E_me)
	err_E_me 	= np.array(err_E_me)
	E_scipy 	= np.array(E_scipy)
	err_E_scipy = np.array(err_E_scipy)

	#Plot field and integration:
	# plt.plot(h_range, E_me, label="Results of my integration", color="#CC4F1B")
	# plt.fill_between(h_range, E_me-err_E_me, E_me+err_E_me, alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', linewidth=0)
	# plt.plot(h_range, E_scipy, label="Results from scipy", color="#3F7F4C")
	# plt.fill_between(h_range, E_scipy-err_E_scipy, E_scipy+err_E_scipy, alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=0)
	# plt.title("Electric Field along axis through center of sphere of charge Q")
	# plt.ylabel("E(h)")
	# plt.xlabel("Distance from center of sphere: h")
	# plt.legend()

	#Plot errors:
	plt.plot(h_range, np.abs(err_E_me), color='b', label="My error.")
	plt.plot(h_range, np.abs(err_E_scipy), color='r', label="Scipy error.")
	plt.title("Absolute error comparison.")

	plt.show()


if __name__ == "__main__":
	main()