# atmosphere_data.py

# MSISE-90 Model of Earth's Upper Atmosphere

# Source: http://www.braeunig.us/space/atmos.htm

# Note, all altitudes (dictionary keys) are represented in km, not m
#   a conversion must be made when passing it in


# Low Solar Activity
low_solar_activity = {
    0: {"temp": 300.2511, "density": 1.17E+00, "pressure": 1.01E+05, "mol_wt": 28.9502},
    20: {"temp": 206.2085, "density": 9.48E-02, "pressure": 5.62E+03, "mol_wt": 28.9502},
    40: {"temp": 257.6979, "density": 4.07E-03, "pressure": 3.01E+02, "mol_wt": 28.9502},
    60: {"temp": 244.1212, "density": 3.31E-04, "pressure": 2.24E+01, "mol_wt": 28.9502},
    80: {"temp": 203.1065, "density": 1.69E-05, "pressure": 9.81E-01, "mol_wt": 29.1353},
    100: {"temp": 168.7219, "density": 5.77E-07, "pressure": 2.89E-02, "mol_wt": 28.0036},
    120: {"temp": 335.8684, "density": 1.70E-08, "pressure": 1.92E-03, "mol_wt": 26.3548},
    140: {"temp": 485.8594, "density": 2.96E-09, "pressure": 4.37E-04, "mol_wt": 24.5645},
    160: {"temp": 570.0652, "density": 9.65E-10, "pressure": 1.53E-04, "mol_wt": 23.2784},
    180: {"temp": 667.8662, "density": 3.90E-10, "pressure": 9.62E-05, "mol_wt": 22.5037},
    200: {"temp": 684.9187, "density": 1.75E-10, "pressure": 4.76E-05, "mol_wt": 21.2516},
    220: {"temp": 692.6487, "density": 8.47E-11, "pressure": 2.43E-05, "mol_wt": 20.0935},
    240: {"temp": 696.1697, "density": 4.31E-11, "pressure": 1.31E-05, "mol_wt": 19.0789},
    260: {"temp": 697.7811, "density": 2.30E-11, "pressure": 7.31E-06, "mol_wt": 18.2300},
    280: {"temp": 698.5220, "density": 1.27E-11, "pressure": 4.29E-06, "mol_wt": 17.5402},
    300: {"temp": 698.8649, "density": 7.24E-12, "pressure": 1.47E-06, "mol_wt": 16.9831},
    320: {"temp": 699.0233, "density": 4.21E-12, "pressure": 1.49E-06, "mol_wt": 16.5214},
    340: {"temp": 699.0973, "density": 2.50E-12, "pressure": 9.01E-07, "mol_wt": 16.1147},
    360: {"temp": 699.1320, "density": 1.51E-12, "pressure": 5.57E-07, "mol_wt": 15.7719},
    380: {"temp": 699.1483, "density": 9.20E-13, "pressure": 3.56E-07, "mol_wt": 15.3028},
    400: {"temp": 699.1561, "density": 5.68E-13, "pressure": 2.23E-07, "mol_wt": 14.8185},
    420: {"temp": 699.1597, "density": 3.54E-13, "pressure": 1.45E-07, "mol_wt": 14.2332},
    440: {"temp": 699.1615, "density": 2.23E-13, "pressure": 9.61E-08, "mol_wt": 13.5181},
    460: {"temp": 699.1623, "density": 1.42E-13, "pressure": 6.54E-08, "mol_wt": 12.6581},
    480: {"temp": 699.1627, "density": 9.20E-14, "pressure": 4.58E-08, "mol_wt": 11.6591},
    500: {"temp": 699.1629, "density": 6.03E-14, "pressure": 3.32E-08, "mol_wt": 10.5547},
    520: {"temp": 699.1630, "density": 4.03E-14, "pressure": 2.48E-08, "mol_wt": 9.4006},
    540: {"temp": 699.1630, "density": 2.75E-14, "pressure": 1.94E-08, "mol_wt": 8.2657},
    560: {"temp": 699.1631, "density": 1.93E-14, "pressure": 1.58E-08, "mol_wt": 7.2141},
    580: {"temp": 699.1631, "density": 1.39E-14, "pressure": 1.28E-08, "mol_wt": 6.2904},
    600: {"temp": 699.1631, "density": 1.03E-14, "pressure": 1.06E-08, "mol_wt": 5.5149},
    620: {"temp": 699.1631, "density": 7.96E-15, "pressure": 9.40E-09, "mol_wt": 4.8664},
    640: {"temp": 699.1631, "density": 6.24E-15, "pressure": 8.27E-09, "mol_wt": 4.3891},
    660: {"temp": 699.1631, "density": 5.06E-15, "pressure": 7.56E-09, "mol_wt": 4.0012},
    680: {"temp": 699.1631, "density": 4.21E-15, "pressure": 6.62E-09, "mol_wt": 3.6999},
    700: {"temp": 699.1631, "density": 3.58E-15, "pressure": 6.00E-09, "mol_wt": 3.4648},
    720: {"temp": 699.1631, "density": 3.09E-15, "pressure": 5.48E-09, "mol_wt": 3.2789},
    740: {"temp": 699.1631, "density": 2.70E-15, "pressure": 5.02E-09, "mol_wt": 3.1289},
    760: {"temp": 699.1631, "density": 2.39E-15, "pressure": 4.63E-09, "mol_wt": 3.0049},
    780: {"temp": 699.1631, "density": 2.13E-15, "pressure": 4.28E-09, "mol_wt": 2.8996},
    800: {"temp": 699.1631, "density": 1.91E-15, "pressure": 3.98E-09, "mol_wt": 2.8075},
    820: {"temp": 699.1631, "density": 1.73E-15, "pressure": 3.68E-09, "mol_wt": 2.7249},
    840: {"temp": 699.1631, "density": 1.56E-15, "pressure": 3.43E-09, "mol_wt": 2.6492},
    860: {"temp": 699.1631, "density": 1.42E-15, "pressure": 3.21E-09, "mol_wt": 2.5784},
    880: {"temp": 699.1631, "density": 1.30E-15, "pressure": 3.00E-09, "mol_wt": 2.5113},
    900: {"temp": 699.1631, "density": 1.18E-15, "pressure": 2.81E-09, "mol_wt": 2.4470}
}


# Moderate Solar Activity
moderate_solar_activity = {
    0: {"temp": 300.2511, "density": 1.17E+00, "pressure": 1.01E+05, "mol_wt": 28.9502},
    20: {"temp": 206.2085, "density": 9.49E-02, "pressure": 5.62E+03, "mol_wt": 28.9502},
    40: {"temp": 257.6979, "density": 4.07E-03, "pressure": 3.02E+02, "mol_wt": 28.9502},
    60: {"temp": 244.1212, "density": 3.31E-04, "pressure": 2.32E+01, "mol_wt": 28.9502},
    80: {"temp": 196.3636, "density": 1.68E-05, "pressure": 9.45E-01, "mol_wt": 29.0175},
    100: {"temp": 194.0160, "density": 5.08E-07, "pressure": 2.77E-02, "mol_wt": 27.7137},
    120: {"temp": 374.9775, "density": 1.88E-08, "pressure": 2.17E-03, "mol_wt": 25.8745},
    140: {"temp": 565.5703, "density": 3.28E-09, "pressure": 5.73E-04, "mol_wt": 24.3329},
    160: {"temp": 767.5532, "density": 1.18E-09, "pressure": 2.31E-04, "mol_wt": 23.1225},
    180: {"temp": 877.6229, "density": 5.51E-10, "pressure": 1.80E-04, "mol_wt": 22.4106},
    200: {"temp": 931.2806, "density": 2.91E-10, "pressure": 1.05E-04, "mol_wt": 21.4734},
    220: {"temp": 963.2701, "density": 1.66E-10, "pressure": 6.44E-05, "mol_wt": 20.6108},
    240: {"temp": 982.4191, "density": 9.91E-11, "pressure": 4.09E-05, "mol_wt": 19.8292},
    260: {"temp": 993.9173, "density": 6.16E-11, "pressure": 2.68E-05, "mol_wt": 19.1337},
    280: {"temp": 1000.8427, "density": 3.94E-11, "pressure": 1.77E-05, "mol_wt": 18.5256},
    300: {"temp": 1005.2670, "density": 2.58E-11, "pressure": 1.20E-05, "mol_wt": 18.0037},
    320: {"temp": 1007.9620, "density": 1.72E-11, "pressure": 8.20E-06, "mol_wt": 17.5537},
    340: {"temp": 1009.0030, "density": 1.16E-11, "pressure": 5.69E-06, "mol_wt": 17.1721},
    360: {"temp": 1010.0423, "density": 7.99E-12, "pressure": 3.98E-06, "mol_wt": 16.8449},
    380: {"temp": 1010.6166, "density": 5.55E-12, "pressure": 2.81E-06, "mol_wt": 16.5597},
    400: {"temp": 1010.9688, "density": 3.89E-12, "pressure": 2.01E-06, "mol_wt": 16.3044},
    420: {"temp": 1011.1853, "density": 2.75E-12, "pressure": 1.44E-06, "mol_wt": 16.0669},
    440: {"temp": 1011.3190, "density": 1.96E-12, "pressure": 1.04E-06, "mol_wt": 15.8360},
    460: {"temp": 1011.4014, "density": 1.40E-12, "pressure": 7.55E-07, "mol_wt": 15.6008},
    480: {"temp": 1011.4519, "density": 1.01E-12, "pressure": 5.53E-07, "mol_wt": 15.3509},
    500: {"temp": 1011.4845, "density": 7.30E-13, "pressure": 4.07E-07, "mol_wt": 15.0760},
    520: {"temp": 1011.5043, "density": 5.31E-13, "pressure": 3.02E-07, "mol_wt": 14.7660},
    540: {"temp": 1011.5168, "density": 3.88E-13, "pressure": 2.27E-07, "mol_wt": 14.4148},
    560: {"temp": 1011.5245, "density": 2.85E-13, "pressure": 1.71E-07, "mol_wt": 14.0125},
    580: {"temp": 1011.5294, "density": 2.11E-13, "pressure": 1.31E-07, "mol_wt": 13.5547},
    600: {"temp": 1011.5325, "density": 1.56E-13, "pressure": 1.01E-07, "mol_wt": 13.0389},
    620: {"temp": 1011.5345, "density": 1.17E-13, "pressure": 7.86E-08, "mol_wt": 12.4656},
    640: {"temp": 1011.5357, "density": 8.79E-14, "pressure": 6.24E-08, "mol_wt": 11.8428},
    660: {"temp": 1011.5365, "density": 6.65E-14, "pressure": 5.01E-08, "mol_wt": 11.1779},
    680: {"temp": 1011.5370, "density": 5.08E-14, "pressure": 4.07E-08, "mol_wt": 10.4854},
    700: {"temp": 1011.5374, "density": 3.91E-14, "pressure": 3.36E-08, "mol_wt": 9.7818},
    720: {"temp": 1011.5375, "density": 3.04E-14, "pressure": 2.82E-08, "mol_wt": 9.0847},
    740: {"temp": 1011.5377, "density": 2.39E-14, "pressure": 2.39E-08, "mol_wt": 8.4111},
    760: {"temp": 1011.5377, "density": 1.90E-14, "pressure": 2.05E-08, "mol_wt": 7.7753},
    780: {"temp": 1011.5378, "density": 1.53E-14, "pressure": 1.78E-08, "mol_wt": 7.1894},
    800: {"temp": 1011.5378, "density": 1.25E-14, "pressure": 1.56E-08, "mol_wt": 6.6631},
    820: {"temp": 1011.5378, "density": 1.03E-14, "pressure": 1.40E-08, "mol_wt": 6.1949},
    840: {"temp": 1011.5378, "density": 8.64E-15, "pressure": 1.26E-08, "mol_wt": 5.7711},
    860: {"temp": 1011.5379, "density": 7.32E-15, "pressure": 1.14E-08, "mol_wt": 5.4132},
    880: {"temp": 1011.5379, "density": 6.28E-15, "pressure": 1.04E-08, "mol_wt": 5.1066},
    900: {"temp": 1011.5379, "density": 5.46E-15, "pressure": 9.47E-09, "mol_wt": 4.8460}
}



# High Solar Activity
high_solar_activity = {
    0: {"temp": 300.2511, "density": 1.16E+00, "pressure": 9.98E+04, "mol_wt": 28.9502},
    20: {"temp": 206.2085, "density": 9.41E-02, "pressure": 5.57E+03, "mol_wt": 28.9502},
    40: {"temp": 257.6979, "density": 4.04E-03, "pressure": 2.99E+02, "mol_wt": 28.9502},
    60: {"temp": 244.1212, "density": 3.28E-04, "pressure": 2.30E+01, "mol_wt": 28.9502},
    80: {"temp": 172.2146, "density": 1.68E-05, "pressure": 8.42E-01, "mol_wt": 28.5290},
    100: {"temp": 297.3338, "density": 2.78E-07, "pressure": 2.47E-02, "mol_wt": 26.1997},
    120: {"temp": 493.4874, "density": 1.24E-08, "pressure": 1.96E-03, "mol_wt": 24.3506},
    140: {"temp": 879.9174, "density": 3.34E-09, "pressure": 1.01E-03, "mol_wt": 22.5096},
    160: {"temp": 1143.5236, "density": 2.23E-09, "pressure": 9.06E-04, "mol_wt": 21.3129},
    180: {"temp": 1314.3427, "density": 1.28E-09, "pressure": 6.76E-04, "mol_wt": 20.7706},
    200: {"temp": 1423.6469, "density": 8.28E-10, "pressure": 4.86E-04, "mol_wt": 20.1836},
    220: {"temp": 1493.7864, "density": 5.69E-10, "pressure": 3.60E-04, "mol_wt": 19.6664},
    240: {"temp": 1538.9154, "density": 4.08E-10, "pressure": 2.72E-04, "mol_wt": 19.2046},
    260: {"temp": 1568.0294, "density": 3.00E-10, "pressure": 2.08E-04, "mol_wt": 18.7901},
    280: {"temp": 1586.8613, "density": 2.25E-10, "pressure": 1.61E-04, "mol_wt": 18.4178},
    300: {"temp": 1599.0714, "density": 1.71E-10, "pressure": 1.26E-04, "mol_wt": 18.0821},
    320: {"temp": 1607.0154, "density": 1.32E-10, "pressure": 9.93E-05, "mol_wt": 17.7852},
    340: {"temp": 1612.1920, "density": 1.03E-10, "pressure": 7.86E-05, "mol_wt": 17.5186},
    360: {"temp": 1615.5731, "density": 8.05E-11, "pressure": 6.26E-05, "mol_wt": 17.2812},
    380: {"temp": 1617.7916, "density": 6.35E-11, "pressure": 5.01E-05, "mol_wt": 17.0669},
    400: {"temp": 1619.2476, "density": 5.04E-11, "pressure": 4.02E-05, "mol_wt": 16.8818},
    420: {"temp": 1620.2062, "density": 4.02E-11, "pressure": 3.25E-05, "mol_wt": 16.7142},
    440: {"temp": 1620.8390, "density": 3.23E-11, "pressure": 2.63E-05, "mol_wt": 16.5643},
    460: {"temp": 1621.2577, "density": 2.60E-11, "pressure": 2.13E-05, "mol_wt": 16.4297},
    480: {"temp": 1621.5356, "density": 2.10E-11, "pressure": 1.74E-05, "mol_wt": 16.3079},
    500: {"temp": 1621.7200, "density": 1.70E-11, "pressure": 1.42E-05, "mol_wt": 16.1967},
    520: {"temp": 1621.8420, "density": 1.38E-11, "pressure": 1.16E-05, "mol_wt": 16.0940},
    540: {"temp": 1621.9253, "density": 1.13E-11, "pressure": 9.50E-06, "mol_wt": 15.9980},
    560: {"temp": 1621.9803, "density": 9.21E-12, "pressure": 7.81E-06, "mol_wt": 15.9067},
    580: {"temp": 1622.0172, "density": 7.55E-12, "pressure": 6.44E-06, "mol_wt": 15.8187},
    600: {"temp": 1622.0421, "density": 6.20E-12, "pressure": 5.32E-06, "mol_wt": 15.7321},
    620: {"temp": 1622.0588, "density": 5.10E-12, "pressure": 4.40E-06, "mol_wt": 15.6457},
    640: {"temp": 1622.0702, "density": 4.21E-12, "pressure": 3.65E-06, "mol_wt": 15.5578},
    660: {"temp": 1622.0778, "density": 3.47E-12, "pressure": 3.03E-06, "mol_wt": 15.4673},
    680: {"temp": 1622.0830, "density": 2.88E-12, "pressure": 2.52E-06, "mol_wt": 15.3725},
    700: {"temp": 1622.0865, "density": 2.38E-12, "pressure": 2.11E-06, "mol_wt": 15.2723},
    720: {"temp": 1622.0890, "density": 1.98E-12, "pressure": 1.76E-06, "mol_wt": 15.1653},
    740: {"temp": 1622.0906, "density": 1.65E-12, "pressure": 1.48E-06, "mol_wt": 15.0503},
    760: {"temp": 1622.0918, "density": 1.37E-12, "pressure": 1.24E-06, "mol_wt": 14.9260},
    780: {"temp": 1622.0925, "density": 1.15E-12, "pressure": 1.05E-06, "mol_wt": 14.7912},
    800: {"temp": 1622.0930, "density": 9.59E-13, "pressure": 8.84E-07, "mol_wt": 14.6446},
    820: {"temp": 1622.0934, "density": 8.05E-13, "pressure": 7.48E-07, "mol_wt": 14.4854},
    840: {"temp": 1622.0936, "density": 6.74E-13, "pressure": 6.36E-07, "mol_wt": 14.3131},
    860: {"temp": 1622.0939, "density": 5.67E-13, "pressure": 5.42E-07, "mol_wt": 14.1244},
    880: {"temp": 1622.0940, "density": 4.77E-13, "pressure": 4.63E-07, "mol_wt": 13.9210},
    900: {"temp": 1622.0940, "density": 4.03E-13, "pressure": 3.97E-07, "mol_wt": 13.7015}
}




def init():
    # Test altitude in m
    altitude_m = 450000
    
    # Convert altitude to nearest multiple of 20 km
    #   This will round down to be conservative
    altitude_km = round(altitude_m / 1000 / 20) * 20

    # Get the min and max altitudes from the data
    min_altitude = min(low_solar_activity.keys())
    max_altitude = max(low_solar_activity.keys())

    # Clamp the altitude to the valid range
    test_altitude = max(min_altitude, min(max_altitude, altitude_km))
   

    # Print atmosphere data for all three solar activity levels at the test altitude
    print(f"Atmosphere data at {test_altitude} km altitude:")
    
    print("\nLow Solar Activity:")
    print(f"Temperature: {low_solar_activity[test_altitude]['temp']} K")
    print(f"Density: {low_solar_activity[test_altitude]['density']} kg/m³")
    print(f"Pressure: {low_solar_activity[test_altitude]['pressure']} Pa")
    print(f"Molecular Weight: {low_solar_activity[test_altitude]['mol_wt']}")

    print("\nModerate Solar Activity:")
    print(f"Temperature: {moderate_solar_activity[test_altitude]['temp']} K")
    print(f"Density: {moderate_solar_activity[test_altitude]['density']} kg/m³")
    print(f"Pressure: {moderate_solar_activity[test_altitude]['pressure']} Pa")
    print(f"Molecular Weight: {moderate_solar_activity[test_altitude]['mol_wt']}")

    print("\nHigh Solar Activity:")
    print(f"Temperature: {high_solar_activity[test_altitude]['temp']} K")
    print(f"Density: {high_solar_activity[test_altitude]['density']} kg/m³")
    print(f"Pressure: {high_solar_activity[test_altitude]['pressure']} Pa")
    print(f"Molecular Weight: {high_solar_activity[test_altitude]['mol_wt']}")

if __name__ == "__main__":
    init()