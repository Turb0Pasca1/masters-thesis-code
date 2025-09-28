% default LoadMySystem file after user creation
% LoadMySystem - probe_15mm_1H

checkDeviceSerial(HW, 174, mfilename('fullpath'));

HW.fLarmor = 21530000.000; HW.B0 = HW.fLarmor/(HW.Gamma.H1/2/pi);
