# atmosphere_data.py

# MSISE-90 Model of Earth's Upper Atmosphere

# Low solar activity (F10.7 = 70)
low_solar_activity = {
    100: {"temp": 188.0, "density": 4.974e-07, "pressure": 3.201e-01, "mol_wt": 28.82},
    120: {"temp": 371.0, "density": 1.971e-08, "pressure": 2.541e-02, "mol_wt": 27.30},
    140: {"temp": 575.0, "density": 4.629e-09, "pressure": 8.954e-03, "mol_wt": 25.59},
    160: {"temp": 726.0, "density": 1.739e-09, "pressure": 4.072e-03, "mol_wt": 24.00},
    180: {"temp": 830.0, "density": 8.118e-10, "pressure": 2.133e-03, "mol_wt": 22.55},
    200: {"temp": 899.0, "density": 4.283e-10, "pressure": 1.214e-03, "mol_wt": 21.20},
    220: {"temp": 944.0, "density": 2.456e-10, "pressure": 7.339e-04, "mol_wt": 19.94},
    240: {"temp": 974.0, "density": 1.501e-10, "pressure": 4.640e-04, "mol_wt": 18.76},
    260: {"temp": 993.0, "density": 9.601e-11, "pressure": 3.046e-04, "mol_wt": 17.65},
    280: {"temp": 1005.0, "density": 6.380e-11, "pressure": 2.059e-04, "mol_wt": 16.62},
    300: {"temp": 1012.0, "density": 4.370e-11, "pressure": 1.427e-04, "mol_wt": 15.67},
    320: {"temp": 1017.0, "density": 3.069e-11, "pressure": 1.010e-04, "mol_wt": 14.80},
    340: {"temp": 1020.0, "density": 2.199e-11, "pressure": 7.275e-05, "mol_wt": 14.02},
    360: {"temp": 1022.0, "density": 1.601e-11, "pressure": 5.317e-05, "mol_wt": 13.32},
    380: {"temp": 1023.0, "density": 1.182e-11, "pressure": 3.935e-05, "mol_wt": 12.71},
    400: {"temp": 1024.0, "density": 8.840e-12, "pressure": 2.948e-05, "mol_wt": 12.18},
    420: {"temp": 1024.0, "density": 6.676e-12, "pressure": 2.229e-05, "mol_wt": 11.74},
    440: {"temp": 1024.0, "density": 5.091e-12, "pressure": 1.701e-05, "mol_wt": 11.37},
    460: {"temp": 1024.0, "density": 3.914e-12, "pressure": 1.308e-05, "mol_wt": 11.08},
    480: {"temp": 1024.0, "density": 3.031e-12, "pressure": 1.013e-05, "mol_wt": 10.85},
    500: {"temp": 1024.0, "density": 2.362e-12, "pressure": 7.903e-06, "mol_wt": 10.67},
}


# Moderate solar activity (F10.7 = 140)
moderate_solar_activity = {
    100: {"temp": 208.0, "density": 5.245e-07, "pressure": 3.201e-01, "mol_wt": 28.82},
    120: {"temp": 487.0, "density": 2.218e-08, "pressure": 3.302e-02, "mol_wt": 26.75},
    140: {"temp": 773.0, "density": 5.837e-09, "pressure": 1.376e-02, "mol_wt": 24.68},
    160: {"temp": 975.0, "density": 2.164e-09, "pressure": 6.412e-03, "mol_wt": 22.66},
    180: {"temp": 1118.0, "density": 9.805e-10, "pressure": 3.318e-03, "mol_wt": 20.77},
    200: {"temp": 1212.0, "density": 5.148e-10, "pressure": 1.893e-03, "mol_wt": 19.05},
    220: {"temp": 1274.0, "density": 2.974e-10, "pressure": 1.157e-03, "mol_wt": 17.53},
    240: {"temp": 1314.0, "density": 1.832e-10, "pressure": 7.420e-04, "mol_wt": 16.21},
    260: {"temp": 1340.0, "density": 1.186e-10, "pressure": 4.938e-04, "mol_wt": 15.10},
    280: {"temp": 1358.0, "density": 7.987e-11, "pressure": 3.388e-04, "mol_wt": 14.19},
    300: {"temp": 1369.0, "density": 5.545e-11, "pressure": 2.385e-04, "mol_wt": 13.46},
    320: {"temp": 1377.0, "density": 3.955e-11, "pressure": 1.716e-04, "mol_wt": 12.89},
    340: {"temp": 1382.0, "density": 2.881e-11, "pressure": 1.258e-04, "mol_wt": 12.45},
    360: {"temp": 1385.0, "density": 2.133e-11, "pressure": 9.367e-05, "mol_wt": 12.13},
    380: {"temp": 1387.0, "density": 1.605e-11, "pressure": 7.067e-05, "mol_wt": 11.89},
    400: {"temp": 1389.0, "density": 1.223e-11, "pressure": 5.405e-05, "mol_wt": 11.72},
    420: {"temp": 1390.0, "density": 9.439e-12, "pressure": 4.180e-05, "mol_wt": 11.59},
    440: {"temp": 1391.0, "density": 7.363e-12, "pressure": 3.265e-05, "mol_wt": 11.50},
    460: {"temp": 1391.0, "density": 5.801e-12, "pressure": 2.576e-05, "mol_wt": 11.44},
    480: {"temp": 1391.0, "density": 4.611e-12, "pressure": 2.048e-05, "mol_wt": 11.39},
    500: {"temp": 1392.0, "density": 3.693e-12, "pressure": 1.642e-05, "mol_wt": 11.36},
}



# High solar activity (F10.7 = 250)
high_solar_activity = {
    100: {"temp": 233.0, "density": 5.606e-07, "pressure": 3.201e-01, "mol_wt": 28.82},
    120: {"temp": 656.0, "density": 2.533e-08, "pressure": 4.047e-02, "mol_wt": 26.20},
    140: {"temp": 1200.0, "density": 7.518e-09, "pressure": 2.196e-02, "mol_wt": 23.89},
    160: {"temp": 1507.0, "density": 3.177e-09, "pressure": 1.168e-02, "mol_wt": 21.89},
    180: {"temp": 1718.0, "density": 1.604e-09, "pressure": 6.722e-03, "mol_wt": 20.02},
    200: {"temp": 1857.0, "density": 9.092e-10, "pressure": 4.117e-03, "mol_wt": 18.30},
    220: {"temp": 1947.0, "density": 5.566e-10, "pressure": 2.643e-03, "mol_wt": 16.83},
    240: {"temp": 2004.0, "density": 3.599e-10, "pressure": 1.759e-03, "mol_wt": 15.61},
    260: {"temp": 2042.0, "density": 2.428e-10, "pressure": 1.208e-03, "mol_wt": 14.63},
    280: {"temp": 2067.0, "density": 1.693e-10, "pressure": 8.523e-04, "mol_wt": 13.87},
    300: {"temp": 2084.0, "density": 1.214e-10, "pressure": 6.160e-04, "mol_wt": 13.29},
    320: {"temp": 2095.0, "density": 8.909e-11, "pressure": 4.544e-04, "mol_wt": 12.87},
    340: {"temp": 2103.0, "density": 6.660e-11, "pressure": 3.407e-04, "mol_wt": 12.57},
    360: {"temp": 2108.0, "density": 5.054e-11, "pressure": 2.589e-04, "mol_wt": 12.37},
    380: {"temp": 2112.0, "density": 3.884e-11, "pressure": 1.993e-04, "mol_wt": 12.24},
    400: {"temp": 2114.0, "density": 3.019e-11, "pressure": 1.550e-04, "mol_wt": 12.16},
    420: {"temp": 2116.0, "density": 2.368e-11, "pressure": 1.218e-04, "mol_wt": 12.11},
    440: {"temp": 2117.0, "density": 1.874e-11, "pressure": 9.646e-05, "mol_wt": 12.08},
    460: {"temp": 2118.0, "density": 1.494e-11, "pressure": 7.691e-05, "mol_wt": 12.06},
    480: {"temp": 2119.0, "density": 1.199e-11, "pressure": 6.165e-05, "mol_wt": 12.05},
    500: {"temp": 2119.0, "density": 9.669e-12, "pressure": 4.969e-05, "mol_wt": 12.04},
}