% LoadMySystem - probe_10mm_23Na1H_dual
% created 2025-09-16 for channel TRx with 10mm saturated NaCl CuSO4 sample
checkDeviceSerial(HW, 174, mfilename('fullpath'));

% firmware change to enable dual nuclear imaging
HW.MMRT.FPGA_Firmware       = 20221129;

% choose order of GammaDef and GammaX by using mOrder 
% maybe implement toggle function as HW.TX(1).CoilName 
% functionality requires setup by PureDevices

% toggle to switch order
mOrder                      = 'Def_H1_X_Na23';
% mOrder                      = 'Def_Na23_X_H1';

switch mOrder
  case 'Def_H1_X_Na23'
    HW.GammaDef             = HW.Gamma.H1;
    HW.FindFrequencyGamma   = HW.Gamma.H1;
    % created 2025-09-17 for channel TRx with 10mm saturated NaCl CuSO4 sample
    HW.TX(1).PaUout2AmplitudeEstimated = [4.312343, 4.282640]*1e-6;  % 2025-09-17T11:31:35 (tFlip90 = 367.998 us @ 3.700 V @ 21.529198 MHz) from 1d Spin Echo by Find_PulseDuration
    % setting tPulse90Estimated is more important than
    % PaUout2AmplitudeEstimated as it is used for Find_PulseDuration and
    % PaUout2Amplitude is calulated by pi / (2*HW.GammaDef*3.7*HW.tFlip90Def)
    HW.FindPulseDuration.tPulse90Estimated = 367.998e-6;
    HW.GammaX               = HW.Gamma.Na23;
    HW.TX(1).PaUout2AmplitudeEstimatedX = [67.585807, 63.025153]*1e-6;  % 2025-09-17T12:00:39 (tFlip90 = 88.767 us @ 3.700 V @ 5.694890 MHz) from 1d Spin Echo by Find_PulseDuration
  case 'Def_Na23_X_H1'
    HW.GammaDef             = HW.Gamma.Na23;
    HW.FindFrequencyGamma   = HW.Gamma.Na23;
    % created 2025-09-17 for channel TRx with 10mm saturated NaCl CuSO4 sample
    HW.TX(1).PaUout2AmplitudeEstimated = [67.585807, 63.025153]*1e-6;  % 2025-09-17T12:00:39 (tFlip90 = 88.767 us @ 3.700 V @ 5.694890 MHz) from 1d Spin Echo by Find_PulseDuration
    % setting tPulse90Estimated is more important than
    % PaUout2AmplitudeEstimated as it is used for Find_PulseDuration and
    % PaUout2Amplitude is calulated by pi / (2*HW.GammaDef*3.7*HW.tFlip90Def)
    HW.FindPulseDuration.tPulse90Estimated = 88.767e-6;
    HW.GammaX               = HW.Gamma.H1;
    HW.TX(1).PaUout2AmplitudeEstimatedX = [4.312343, 4.282640]*1e-6;  % 2025-09-17T11:31:35 (tFlip90 = 367.998 us @ 3.700 V @ 21.529198 MHz) from 1d Spin Echo by Find_PulseDuration
  otherwise
    HW.GammaDef             = HW.Gamma.H1;
    HW.FindFrequencyGamma   = HW.Gamma.H1;
    % created 2025-09-17 for channel TRx with 10mm saturated NaCl CuSO4 sample
    HW.TX(1).PaUout2AmplitudeEstimated = [4.312343, 4.282640]*1e-6;  % 2025-09-17T11:31:35 (tFlip90 = 367.998 us @ 3.700 V @ 21.529198 MHz) from 1d Spin Echo by Find_PulseDuration
    % setting tPulse90Estimated is more important than
    % PaUout2AmplitudeEstimated as it is used for Find_PulseDuration and
    % PaUout2Amplitude is calulated by pi / (2*HW.GammaDef*3.7*HW.tFlip90Def)
    HW.FindPulseDuration.tPulse90Estimated = 367.998e-6;
    HW.GammaX               = HW.Gamma.Na23;
    HW.TX(1).PaUout2AmplitudeEstimatedX = [67.585807, 63.025153]*1e-6;  % 2025-09-17T12:00:39 (tFlip90 = 88.767 us @ 3.700 V @ 5.694890 MHz) from 1d Spin Echo by Find_PulseDuration
    warning('Using GammaDef = GammaH1; GammaX = GammaNa23');
end
   

