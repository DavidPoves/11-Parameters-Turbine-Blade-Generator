import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


"""
MIT License

Copyright (c) 2020 David Poves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class TurbineBladeGen(object):
	def __init__(self, radius, axial_chord, tangential_chord, unguided_turning, inlet_blade, inlet_half_wedge, le_r,
	             outlet_blade, te_r, n_blades, throat, n_points=50):
		"""
		Class implementing the parametrization method for a turbine blade based on the paper:
						"L.J. Pritchard: An eleven parameter axial turbine airfoil geometry model"
		This method parametrizes the geometry of the airfoil based on 11 elemental parameters, which are the ones
		defined below. ANGLES SHOULD ALWAYS BE INTRODUCED IN RADIANS.
		:param radius: Radius of the cylinder upon which the airfoil is defined.
		:param axial_chord: Axial chord.
		:param tangential_chord: Tangential chord.
		:param unguided_turning: Unguided turning angle. MUST BE INTRODUCED IN RADIANS.
		:param inlet_blade: Inlet local blade angle. MUST BE INTRODUCED IN RADIANS.
		:param inlet_half_wedge: Inlet half wedge angle. MUST BE INTRODUCED IN RADIANS.
		:param le_r: Radius of the leading edge radius.
		:param outlet_blade: Outlet local blade angle. MUST BE INTRODUCED IN RADIANS.
		:param te_r: Radius of the trailing edge.
		:param n_blades: Number of blades.
		:param throat: Geometric throat.
		:param n_points: Number of points used to define the blade. Optional, default is 50.
		"""
		self.R = radius
		self.cx = axial_chord
		self.ct = tangential_chord
		self.unguided = unguided_turning
		self.beta_i = inlet_blade
		self.epsilon_i = inlet_half_wedge
		self.le_r = le_r
		self.beta_o = outlet_blade
		self.te_r = te_r
		self.N = n_blades
		self.throat = throat
		self.n_points = n_points

		# Initialize parameters.
		self.x1, self.y1 = None, None
		self.x2, self.y2 = None, None
		self.x3, self.y3 = None, None
		self.x4, self.y4 = None, None
		self.x5, self.y5 = None, None
		self.x5, self.y5 = None, None
		self.x6, self.y6 = None, None
		self.x7, self.y7 = None, None
		self.x8, self.y8 = None, None
		self.x9, self.y9 = None, None
		self.x0, self.y0 = None, None
		self.R0 = 0
		self.dxp, self.dxs = None, None
		self.aS, self.bS, self.cS, self.dS = None, None, None, None
		self.aP, self.bP, self.cP, self.dP = None, None, None, None
		self.x_suction, self.y_suction = None, None
		self.x_pressure, self.y_pressure = None, None
		self.x_te_close, self.y_te_close = None, None

		# Initialize the dependent parameters.
		self._compute_dependent_parameters()

		# Get a first guess of the wedge out angle.
		self._get_first_guess_wedgeout()

		# Create a Pandas dataframe for the dependent data.
		self.DependentTable = pd.DataFrame()

	def calculate_blade_geometry(self, straight_te=True):
		"""
		Compute the geometry of the blade given the 11 required geometric parameters. In order to solve for the
		discontinuity at the throat point (second key point of the airfoil) an iterative process over the wedge out
		has been implemented with an absolute tolerance of 1e-5, as indicated on the paper. An error will raise if no
		convergence is achieved. Moreover, if the user decides to implement a straight trailing edge cut (for CFD
		applications), a straight cut on the latter will be performed.
		The final geometry will finally be plotted to let the user visualize the created airfoil. Finally, the user is
		asked to choose of a .txt file with the coordinates of each of the curves should be saved. In that case, a GUI
		will be prompted to let the user select where to save the generated data.
		:param straight_te: Boolean indicating if a straight cut on the TE should be carried out.
		:return:
		"""
		convergence = False  # Set the convergence parameter on the wedge out angle to solve the throat discontinuity.

		# Perform the iteration to find convergence on the wedge out angle and solve the discontinuity.
		while not convergence:
			# Compute the key points.
			self.x1, self.y1 = self._compute_first_point()
			self.x2, self.y2 = self._compute_second_point()
			self.x3, self.y3 = self._compute_third_point()
			self.x4, self.y4 = self._compute_fourth_point()
			self.x5, self.y5 = self._compute_fifth_point()

			# Compute the circular arc center and radius.
			x0 = ((self.y1 - self.y2) * np.tan(self.beta_1) * np.tan(self.beta_2) + self.x1 * np.tan(self.beta_2) -
			      self.x2 * np.tan(self.beta_1)) / ((np.tan(self.beta_2)) - np.tan(self.beta_1))
			y0 = -(x0 - self.x1) / np.tan(self.beta_1) + self.y1
			R0 = np.sqrt((self.x1 - x0) ** 2 + (self.y1 - y0) ** 2)
			yy2 = y0 + np.sqrt(R0**2 - (self.x2-x0)**2)

			if abs(self.y2-yy2) < 1e-5:
				convergence = True
			elif abs(self.y2 - yy2) > 1e-5:
				self.epsilon_o *= (self.y2 / yy2) ** 4
				if self.epsilon_o < 0:  # Raise error since the angle tends to negative values.
					raise TimeoutError('Error on removing throat discontinuity. Try reducing the exit blade angle or the throat.')

		# Assign the obtained values and compute auxiliary data points.
		self.x0, self.y0, self.R0 = x0, y0, R0
		self.x6, self.y6 = self.cx, 0.
		self.x7, self.y7 = self.cx - self.te_r, 0.
		self.x8, self.y8 = 0., self.ct
		self.x9, self.y9 = self.le_r, self.ct

		# Introduce the half wedge out angle.
		self.dependent_data['Half Wedge out Angle [degrees]'] = np.degrees(self.epsilon_o)

		# Define the first coordinate for both pressure and suction surfaces.
		x_suction, y_suction = np.array([]), np.array([])
		x_pressure, y_pressure = np.array([]), np.array([])
		x_suction, y_suction = np.append(x_suction, self.x8), np.append(y_suction, self.y8)
		x_pressure, y_pressure = np.append(x_pressure, self.x8), np.append(y_pressure, self.y8)

		# Define steps on the x coordinates for the pressure and suction sides respectively.
		self.dxp = (self.x4-self.x8)/((1/5)*self.n_points-1)  # Parameter defined for the x coord step on pressure.
		self.dxs = (self.x3-self.x8)/((1/5)*self.n_points-1)  # Parameter defined for the x coordinate step on suction.

		# Define the polynomial coefficients for pressure and suction sides.
		self.aS, self.bS, self.cS, self.dS = self._compute_coefficients_third_polynomial(self.x2, self.y2, self.beta_2,
		                                                                                 self.x3, self.y3, self.beta_3)
		self.aP, self.bP, self.cP, self.dP = self._compute_coefficients_third_polynomial(self.x4, self.y4, self.beta_4,
		                                                                                 self.x5, self.y5, self.beta_5)

		# Define the circle between 3 and 4.
		x_suction, y_suction, x_pressure, y_pressure,  = self._define_pressure_suction_1(x_suction, y_suction,
		                                                                                 x_pressure, y_pressure)
		# x_pressure, y_pressure = x_pressure[::-1], y_pressure[::-1]

		# Define the third order polynomials for pressure and suction surfaces.
		x_suction, y_suction, x_pressure, y_pressure, = self._define_pressure_suction_2(x_suction, y_suction,
		                                                                                x_pressure, y_pressure)

		# Define the circular arc and the remaining third order polynomial for each of the sides.
		x_suction, y_suction, x_pressure, y_pressure, = self._define_pressure_suction_3(x_suction, y_suction,
		                                                                                x_pressure, y_pressure)

		if straight_te:  # If specified, straight cut on TE.
			self.x_te_close = np.array([x_pressure[-1], x_suction[-1]])
			self.y_te_close = np.array([y_pressure[-1], y_suction[-1]])
			# Plot the data.
			plt.plot(x_suction, y_suction, label='Suction side')
			plt.plot(x_pressure, y_pressure, label='Pressure side')
			plt.plot(self.x_te_close, self.y_te_close, label='Closing TE')
		else:
			x_suction, y_suction, x_pressure, y_pressure, = self._define_pressure_suction_4(x_suction, y_suction,
			                                                                                x_pressure, y_pressure)
			# Plot the data.
			plt.plot(x_suction, y_suction, label='Suction side')
			plt.plot(x_pressure, y_pressure, label='Pressure side')

		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend()

		# Assign the data to the class.
		self.x_suction, self.y_suction = x_suction, y_suction
		self.x_pressure, self.y_pressure = x_pressure, y_pressure

		# Create the dataframe with the dependent data.
		self.DependentTable = pd.DataFrame(self.dependent_data, columns=list(self.dependent_data.keys()))
		pd.set_option('colheader_justify', 'center')  # Justify the column headers to the center.

		# Create an input dialog to save data.
		save_bool = input('Should data be saved (x and y coordinates; and dependent data)? [y/n]: ')
		if save_bool == 'y':
			self._save_data(straight_te)

	def _save_data(self, straight_te):
		np.savetxt('suction_side.txt', np.column_stack((self.x_suction, self.y_suction)))
		np.savetxt('pressure_side.txt', np.column_stack((self.x_pressure, self.y_pressure)))
		self.DependentTable.to_csv(os.path.join(os.getcwd(), 'dependent_data.csv'),
		                           header=list(self.dependent_data.keys()), sep='\t', mode='a')
		if straight_te:
			np.savetxt('trailing_edge.txt', np.column_stack((self.x_te_close, self.y_te_close)))

	def _compute_dependent_parameters(self):
		"""
		Define the parameters dependent on the geometry of the blade and the cascade, as reflected on the paper. All
		dependent angles are returned in DEGREES.
		:return:
		"""
		self.pitch = (2*np.pi*self.R)/self.N
		self.stagger = np.degrees(np.arctan(self.ct/self.cx))
		self.chord = np.sqrt(self.cx**2 + self.ct**2)
		self.Zweifel = (4*np.pi*self.R)/(self.cx*self.N) * np.sin(self.beta_i-self.beta_o) * np.cos(self.beta_o)/np.cos(
			self.beta_i)
		self.solidity = self.chord/self.pitch
		self.camber_angle = np.degrees(self.beta_i - self.beta_o)
		self.Lift_coeff = (2*self.pitch)/self.chord * 0.5*(np.cos(self.beta_i) + np.cos(self.beta_o))*(
				np.tan(self.beta_i) - np.tan(self.beta_o))

		# Create a dictionary with the data.
		self.dependent_data = {'Pitch': [self.pitch], 'Stagger Angle [degrees]': [self.stagger], 'Chord': [self.chord],
		                       'Zweifel Coefficient': [self.Zweifel], 'Solidity': [self.solidity],
		                       'Camber Angle [degrees]': [self.camber_angle], 'Lift Coefficient': [self.Lift_coeff]}

	def _get_first_guess_wedgeout(self):
		"""
		Get a first guess of the wedge out angle IN RADIANS.
		:return:
		"""
		self.epsilon_o = 0.5 * self.unguided  # Compute first guess on the wedge-out angle.

	def _compute_first_point(self):
		"""
		Compute the first elementary point of the airfoil.
		:return: x and y coordinates of the airfoil.
		"""
		beta_1 = self.beta_o - self.epsilon_o
		x1 = self.cx - self.te_r*(1+np.sin(beta_1))
		y1 = self.te_r*np.cos(beta_1)

		# Save the value of beta for slope computations.
		self.beta_1 = beta_1

		return x1, y1

	def _compute_second_point(self):
		"""
		Compute the second elementary point of the airfoil.
		:return: x and y coordinates of the second elementary point.
		"""
		beta_2 = self.beta_o - self.epsilon_o + self.unguided
		x2 = self.cx - self.te_r + (self.throat+self.te_r)*np.sin(beta_2)
		y2 = (2*np.pi*self.R)/self.N - (self.throat+self.te_r)*np.cos(beta_2)

		# Save the value of beta for slope computations.
		self.beta_2 = beta_2

		return x2, y2

	def _compute_third_point(self):
		"""
		Compute the third elementary point of the airfoil.
		:return: x and y coordinates of the third elementary point.
		"""
		beta_3 = self.beta_i + self.epsilon_i
		x3 = self.le_r*(1-np.sin(beta_3))
		y3 = self.ct + self.le_r*np.cos(beta_3)

		# Save the value of beta for slope computations.
		self.beta_3 = beta_3

		return x3, y3

	def _compute_fourth_point(self):
		"""
		Compute the fourth elementary point of the airfoil.
		:return: x and y coordinates of the fourth elementary point.
		"""
		beta_4 = self.beta_i - self.epsilon_i
		x4 = self.le_r*(1+np.sin(beta_4))
		y4 = self.ct - self.le_r*np.cos(beta_4)

		# Save the value of beta for slope computations.
		self.beta_4 = beta_4

		return x4, y4

	def _compute_fifth_point(self):
		"""
		Compute the fifth elementary point of the airfoil.
		:return: x and y coordinates of the fifth elementary point.
		"""
		beta_5 = self.beta_o + self.epsilon_o
		x5 = self.cx - self.te_r*(1-np.sin(beta_5))
		y5 = -self.te_r*np.cos(beta_5)

		# Save the value of beta for slope computations.
		self.beta_5 = beta_5

		return x5, y5

	@staticmethod
	def _compute_coefficients_third_polynomial(x1, y1, beta_1, x2, y2, beta_2):
		"""
		Compute the coefficients for a third order polynomial of the form y = a + bx + cx^2 + dx^3 given two points and
		the local blade angles at each of them.
		:param x1: x-coordinate of the first point.
		:param y1: y-coordinate of the first point.
		:param beta_1: Local blade angle at the first point. MUST BE IN RADIANS.
		:param x2: x-coordinate of the second point.
		:param y2: y-coordinate of the second point.
		:param beta_2: Local blade angle at the second point. MUST BE IN RADIANS.
		:return: Coefficients of the third order polynomial given as a tuple (a, b, c, d).
		"""
		d = (np.tan(beta_1)+np.tan(beta_2))/(x1-x2)**2 - 2*(y1-y2)/(x1-x2)**3
		c = (y1-y2)/(x1-x2)**2 - (np.tan(beta_2))/(x1-x2) - d*(x1+2*x2)
		b = np.tan(beta_2) - 2*c*x2 - 3*d*x2**2
		a = y2 - b*x2 - c*x2**2 - d*x2**3

		coefficients = (a, b, c, d)

		return coefficients

	def _define_pressure_suction_1(self, x_suction, y_suction, x_pressure, y_pressure):
		"""
		Define the leading edge arc joining elementary points 3 and 4. This function will separate coords between
		pressure and suction sides.
		:param x_suction: Array containing the x-coordinates of the suction side.
		:param y_suction: Array containing the y-coordinates of the suction side.
		:param x_pressure: Array containing the x-coordinates of the pressure side.
		:param y_pressure: Array containing the y-coordinates of the pressure side.
		:return: Arrays containing the coordinates (x and y separately) for each of the sides (suction first,
		then pressure side).
		"""
		for _ in np.arange(1, (1/5)*self.n_points+1):
			x_pressure = np.append(x_pressure, x_pressure[-1] + self.dxp)
			y_pressure = np.append(y_pressure, self.y9 - np.sqrt(self.le_r**2 - (x_pressure[-1] - self.x9)**2))
			x_suction = np.append(x_suction, x_suction[-1] + self.dxs)
			y_suction = np.append(y_suction, self.y9 + np.sqrt(self.le_r**2 - (x_suction[-1] - self.x9) ** 2))
		self.dxp = (self.x5 - self.x4)/((3/5)*self.n_points)
		self.dxs = (self.x2 - self.x3)/((2/5)*self.n_points)

		return x_suction, y_suction, x_pressure, y_pressure

	def _define_pressure_suction_2(self, x_suction, y_suction, x_pressure, y_pressure):
		"""
		Define the third order polynomials on the pressure and suction sides. This function will separate coords between
		pressure and suction sides.
		:param x_suction: Array containing the x-coordinates of the suction side.
		:param y_suction: Array containing the y-coordinates of the suction side.
		:param x_pressure: Array containing the x-coordinates of the pressure side.
		:param y_pressure: Array containing the y-coordinates of the pressure side.
		:return: Arrays containing the coordinates (x and y separately) for each of the sides (suction first,
		then pressure side).
		"""
		for _ in np.arange((1/5)*self.n_points+1, (3/5)*self.n_points+1):
			x_pressure = np.append(x_pressure, x_pressure[-1] + self.dxp)
			y_pressure = np.append(y_pressure, self.aP+self.bP*x_pressure[-1]+self.cP*x_pressure[-1]**2 +
			                       self.dP*x_pressure[-1]**3)

			x_suction = np.append(x_suction, x_suction[-1] + self.dxs)
			y_suction = np.append(y_suction, self.aS + self.bS * x_suction[-1] + self.cS * x_suction[-1] ** 2 +
			                      self.dS * x_suction[-1] ** 3)
		self.dxs = (self.x1-self.x2)/((1/5)*self.n_points)

		return x_suction, y_suction, x_pressure, y_pressure

	def _define_pressure_suction_3(self, x_suction, y_suction, x_pressure, y_pressure):
		"""
		Define the third order polynomial on the pressure side and the circular arc on the suction side. This function
		will separate coords between pressure and suction surfaces.
		:param x_suction: Array containing the x-coordinates of the suction side.
		:param y_suction: Array containing the y-coordinates of the suction side.
		:param x_pressure: Array containing the x-coordinates of the pressure side.
		:param y_pressure: Array containing the y-coordinates of the pressure side.
		:return: Arrays containing the coordinates (x and y separately) for each of the sides (suction first,
		then pressure side).
		"""
		for _ in np.arange((3/5)*self.n_points+1, (4/5)*self.n_points+1):
			x_pressure = np.append(x_pressure, x_pressure[-1] + self.dxp)
			y_pressure = np.append(y_pressure, self.aP + self.bP * x_pressure[-1] + self.cP * x_pressure[-1] ** 2 +
			                       self.dP * x_pressure[-1] ** 3)

			x_suction = np.append(x_suction, x_suction[-1] + self.dxs)
			y_suction = np.append(y_suction, self.y0 + np.sqrt(self.R0**2-(x_suction[-1]-self.x0)**2))
		self.dxp = (self.x6-self.x5)/((1/5)*self.n_points+1)
		self.dxs = (self.x6-self.x1)/(1/5)*self.n_points+1

		return x_suction, y_suction, x_pressure, y_pressure

	def _define_pressure_suction_4(self, x_suction, y_suction, x_pressure, y_pressure):
		"""
		Define the trailing edge arc joining elementary points 5 and 1. This function will separate coords between
		pressure and suction sides.
		:param x_suction: Array containing the x-coordinates of the suction side.
		:param y_suction: Array containing the y-coordinates of the suction side.
		:param x_pressure: Array containing the x-coordinates of the pressure side.
		:param y_pressure: Array containing the y-coordinates of the pressure side.
		:return: Arrays containing the coordinates (x and y separately) for each of the sides (suction first,
		then pressure side).
		"""
		for _ in np.arange((4/5)*self.n_points+1, self.n_points+1):
			x_pressure = np.append(x_pressure, x_pressure[-1] + self.dxp)
			if x_pressure[-1] > self.cx:
				x_pressure[-1] = self.cx
			y_pressure = np.append(y_pressure, self.y7 - np.sqrt(self.te_r**2 - (x_pressure[-1]-self.x7)**2))

			x_suction = np.append(x_suction, x_suction[-1]+self.dxs)
			if x_suction[-1] > self.cx:
				x_suction[-1] = self.cx
			y_suction = np.append(y_suction, self.y7 + np.sqrt(self.te_r**2 - (x_suction[-1]-self.x7)**2))

		return x_suction, y_suction, x_pressure, y_pressure
