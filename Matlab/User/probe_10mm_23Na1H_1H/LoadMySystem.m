% LoadMySystem - probe_10mm_23Na1H_1H

checkDeviceSerial(HW, 174, mfilename('fullpath'));

HW.fLarmor              = 21530000;                
HW.GammaDef             = HW.Gamma.H1;
HW.FindFrequencyGamma   = HW.Gamma.H1;
% created 2025-09-17 for channel TRx with 10mm saturated NaCl CuSO4 sample
HW.TX(1).PaUout2AmplitudeEstimated = [4.312343, 4.282640]*1e-6;  % 2025-09-17T11:31:35 (tFlip90 = 367.998 us @ 3.700 V @ 21.529198 MHz) from 1d Spin Echo by Find_PulseDuration
% setting tPulse90Estimated is more important than
% PaUout2AmplitudeEstimated as it is used for Find_PulseDuration and
% PaUout2Amplitude is calulated by pi / (2*HW.GammaDef*3.7*HW.tFlip90Def)
HW.FindPulseDuration.tPulse90Estimated = 367.998e-6;
