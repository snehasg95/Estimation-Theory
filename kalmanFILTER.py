####===================FINAL PROJECT CODE FOR ESTIMATION THEORY=====================####
####===PROJECT TITLE: ESTIMATING ATMOSPHERIC OXIDATION LEVELS USING KALMAN FILTER===####
####===========================COURSE: ESTIMATION THEORY============================####
####==========================INSTRUCTOR: DR. NEDA NATEGH===========================####
####=============================AUTHORS: RADHA & SNEHA=============================####
####===================DESCRIPTION: THIS CODE IMPLEMENTS THE KALMAN FILTER GIVEN====####
####==============================  THE DATA VALUES OF NO2 AND O3, PLOTS THE VALUES=####
####==============================  OF THE ESTIMATED NAD MEASURED VALUES============####

# IMPORTING REDUIRED PACKAGES
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D
from numpy.linalg import inv

# FUNCTION TO CALCULATE THE P MATRIX
def GetP(no2,ozone):
    data_length = len(ozone)
    P = []
    avg_ozone = sum(ozone) / data_length
    sig_ozone = 0
    for i in range(0,294):
    		sig_ozone += ((avg_ozone - ozone[i]) **2) /294
    avg_no2 = sum(no2) / data_length
    sig_no2 = 0
    for i in range(0,294):
    	sig_no2 += ((avg_no2 - no2[i]) **2) /294
    sig_ozno2 = math.sqrt(sig_ozone * sig_no2)
    P = np.array([[sig_no2, sig_ozno2], [sig_ozno2, sig_ozone]]) #COVARIANCE MATRIX
    return P

data = pd.read_csv('data_2016.csv')  # READING THE DATA FROM FILE
ozone = data['O3'].values   # STORING OZONE VALUES
no2 = data['NO2'].values    # STORING NO2 VALUES

P = GetP(no2,ozone)   # FUNCTION CALL TO CALCULATE COVARIENCE MATRIX
Q = np.eye(2)*0.001
X = np.array([[no2[0]], [ozone[0]]])   # STATE MATRIX
A = np.array([[1, 1], [0, 1]])   # STATE TRANSITION MATRIX
R = np.eye(2)
H = np.eye(2)
I = np.eye(2)
no2_p = []    # ESTIMATED NO2 VALUES
ozone_p = []   # ESTIMATED OZONE VALUES


# MAIN LOOP THAT ITERATES THROGH ALL DATA POINTS GOING THROUGH
# PREDICTION AND ESTIMATION STEPS
for k in range(1,len(ozone)):
    X = np.matmul(A,X)  
    P = np.matmul(A, np.matmul(P, A.T)) + Q
    Kg = np.matmul(np.matmul(P, H.T), inv(np.matmul(np.matmul(H, P), H.T) + R ))
    z = np.array([[no2[k]], [ozone[k]]])
    X = X + np.matmul(Kg, (z - np.matmul(H,X)))
    P = np.matmul((I - np.matmul(Kg, H)), P)
    no2_p.append(X[0])
    ozone_p.append(X[1])

# GENERATING THE REQUIRED PLOTS
plt.figure(1)
plot1, = plt.plot(no2, label = "$Measured\ NO_2\ values$")
plt.plot(no2_p, label = "$Estimated\ NO_2\ values$")
plt.title('$Plot\ showing\ measured\ NO_2\ values\ in\ comparison\ to\ estimated\ NO_2\ values$')
plt.xlabel('$Number\ of\ Iterations$')
plt.ylabel('$NO_2\ values$')
plt.legend(handler_map={plot1: HandlerLine2D(numpoints=4)})

plt.figure(2)
plot1, = plt.plot(ozone, label = "Measured Ozone values")
plt.plot(ozone_p, label = "Estimated Ozone values")
plt.title('Plot showing measured Ozone values in comparison to estimated Ozone values')
plt.xlabel('Number of Iterations')
plt.ylabel('Ozone values')
plt.legend(handler_map={plot1: HandlerLine2D(numpoints=4)})


