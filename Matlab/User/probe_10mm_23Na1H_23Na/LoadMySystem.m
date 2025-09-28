% LoadMySystem - probe_10mm_23Na1H_23Na

checkDeviceSerial(HW, 174, mfilename('fullpath'));

HW.fLarmor              = 5695000;                
HW.GammaDef             = HW.Gamma.Na23;
HW.FindFrequencyGamma   = HW.Gamma.Na23;
% created 2025-09-17 for channel TRx with 10mm saturated NaCl CuSO4 sample
HW.TX(1).PaUout2AmplitudeEstimated = [67.585807, 63.025153]*1e-6;  % 2025-09-17T12:00:39 (tFlip90 = 88.767 us @ 3.700 V @ 5.694890 MHz) from 1d Spin Echo by Find_PulseDuration
% setting tPulse90Estimated is more important than
% PaUout2AmplitudeEstimated as it is used for Find_PulseDuration and
% PaUout2Amplitude is calulated by pi / (2*HW.GammaDef*3.7*HW.tFlip90Def)
HW.FindPulseDuration.tPulse90Estimated = 88.767e-6;