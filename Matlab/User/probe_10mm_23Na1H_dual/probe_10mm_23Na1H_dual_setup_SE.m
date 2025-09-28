% probe_10mm_23Na1H_dual_setup_SE.m

%% initialize system
% run script in main openMATLAB folder containing all files
LoadSystem probe_10mm_23Na1H_dual;
% find system frequency
[HW, mySave]                = Find_Frequency_Sweep(HW, mySave, 0);

% select whether 1H is GammaDef and 23Na is GammaX or vice versa
% change selection in LoadMySystem as well!
mOrder                      = 'Def_H1_X_Na23';
% mOrder                      = 'Def_Na23_X_H1';

%%
switch mOrder
    case 'Def_H1_X_Na23'
        %% find correct 90 pulse duration and coil efficiency for gamma Def
        minTime                             = 0;
        doPlot                              = 1;
        % as HW.FindPulseDuration.tPulse90Estimated is an old measurement,
        % 1 iteration sould be sufficient
        iterations                          = 1; 
        % for 10mm saturated NaCl CuSO4 sample (1H)
        T1_estimate                         = 64e-3;    
        % the 1H signal is strong enough to skip averaging
        Seq.average                         = 1;
        [HW, mySave]                        = Find_PulseDuration(HW, mySave, minTime, doPlot, iterations, HW.FindPulseDuration.tPulse90Estimated, T1_estimate, Seq);
        % LoadSystem to write the coil efficiency from PaUout2AmplitudeCal.m in the
        % current workspace HWClass
        LoadSystem;

        %% find correct 90 pulse duration and coil efficiency for gamma X
        minTime                             = 0;
        doPlot                              = 1;
        % as HW.FindPulseDuration.tPulse90Estimated is an old measurement, 1 iteration is
        % sufficient
        iterations                          = 1; 
        % for 10mm saturated NaCl CuSO4 sample (23Na)
        T1_estimate                         = 40e-3;    
        % the 23Na signal needs averaging for a smoother results
        Seq.average                         = 10;
        % HWClass has no property HW.FindPulseDuration.tPulse90EstimatedX
        % using HW.TX(1).PaUout2AmplitudeEstimatedX(1) to calculate
        % tPulse90EstimatedX
        tPulse90EstimatedX                  = pi / (2*HW.GammaX*3.7*HW.TX(1).PaUout2AmplitudeEstimatedX(1));
        Seq.useGammaX                       = true;
        [HW, mySave]                        = Find_PulseDuration(HW, mySave, minTime, doPlot, iterations, tPulse90EstimatedX, T1_estimate, Seq);
        % LoadSystem to write the coil efficiency from PaUout2AmplitudeCal.m in the
        % current workspace HWClass
        LoadSystem;

        %% find correct shim
        % re-check system frequency
        [HW, mySave]                        = Find_Frequency_Sweep(HW, mySave, 0);
        [HW, mySave]                        = Find_Shim(HW, mySave);
        % LoadSystem to write the coil efficiency from MagnetShimCal.m in the
        % current workspace HWClass
        LoadSystem;

        %% imaging parameters
        % for probe_10mm_23Na1H_dual saturated NaCl CuSO4 sample
        % basic SE with slice selection gradient
        Seq.AQSlice(1).dualNuclearImage     = true; 
        
        % toggle to 0 for noise measurement
        Seq.AQSlice(1).excitationFlipAngle  = 0;
        
        % use echoTime and tRep for maximal signal amplitude
        % measured using Demo_T1.m for saturated NaCl CuSO4 sample
        T1_1H_10mm_NaCl_CuSO4               = 64e-3;
        T1                                  = T1_1H_10mm_NaCl_CuSO4;
        % measured using Demo_T1.m for saturated NaCl CuSO4 sample
        T2_1H_10mm_NaCl_CuSO4               = 58e-3;
        T2                                  = T2_1H_10mm_NaCl_CuSO4;
        
        % minimal T2 relaxation
        % approx. 90% transversal magnetization left
        Seq.tEcho                           = 0.1 * T2;
        
        % to adjust for correct sequence timing there needs to be a trade off
        % between minimal pixel bandwidth and minimal echo time
        Seq.tEcho                           = 20e-3;
        
        % maximal T1 recovery
        % approx. 95% recovered longitudinal magnetization
        Seq.RepetitionTime                  = 3 * T1;

        % for tubular sample
        Seq.RepetitionTime                  = 1200e-3
        
        % number of averages
        % Seq.Loops does averaging as well 
        % -> Seq.Loops is superior for postprocessing the data as the data of every average is
        % captured
        Seq.average                         = 1;
        
        % important for breaks in the sequence
        % Seq.tRep takes precedence
        % probably not needed for a simple SE sequence
        Seq.T1Estimated                     = T1;
        Seq.T2Estimated                     = T2;
        
        Seq.SteadyState_PreShots180         = 0;
        % changes the the timing of CL (= command line) times of the drivel
        % Seq.SingletRep = 1 is not valid for dual nuclear imaging
        Seq.SingletRep                      = 0;
        % single AQ for each excitation and inversion pulse
        Seq.AQSlice(1).TurboFactor          = 1;
        
        % image orientation
        % slice in coil center in xz-plane
        % x = phase; y = slice; z = read;
        Seq.AQSlice(1).alfa                 = 0;               
        Seq.AQSlice(1).phi                  = 0;               
        Seq.AQSlice(1).theta                = pi/2;  
        
        Seq.AQSlice(1).Center2OriginImage   = [0, 0, 0]; 
        
        % number of samples
        Seq.AQSlice(1).nRead                = 64;   
        Seq.AQSlice(1).nPhase(1)            = 1;                             
        Seq.AQSlice(1).nPhase(2)            = 64;  
        Seq.AQSlice(1).nPhase(3)            = 1; 
        % pixel bandwidth
        Seq.AQSlice(1).HzPerPixMin          = 400; 
        
        % FOV
        Seq.AQSlice(1).sizeRead             = 12e-3;            
        Seq.AQSlice(1).sizePhase(2)         = 12e-3;        
        Seq.AQSlice(1).thickness            = 0.005;  
        
        % oversampling
        Seq.AQSlice(1).ReadOS               = [];      
        Seq.AQSlice(1).PhaseOS(1)           = 1;                           
        Seq.AQSlice(1).PhaseOS(2)           = 2;                            
        Seq.AQSlice(1).PhaseOS(3)           = 1;                            
        Seq.AQSlice(1).PhaseOS(Seq.AQSlice(1).nPhase==1) = 1;   
        
        % toggle plots
        Seq.AQSlice(1).plotkSpace           = 1;        % plot k-space
        Seq.AQSlice(1).plotImage            = 1;        % plot image             
        Seq.plotSeq                         = [1:3];    % plot seq gradients 1:3
        
        %% start measurement
        % re-check system frequency
        [HW, mySave]                        = Find_Frequency_Sweep(HW, mySave, 0);
        [SeqLoop, mySave]                   = sequence_Spin_Echo(HW, Seq, AQ, TX, Grad, mySave);
        
        % display image at secondary frequency
        SeqLoop.AQSlice(1).plotImageHandle  = 210;  % for 1d/2d images
        SeqLoop.AQSlice(1).plotImagePhase   = 211;
        SeqLoop.AQSlice(1).plotkSpace       = 212;
        SeqLoop.AQSlice(1).plotkSpacePhase  = 213;
        dataFieldName                       = 'dataX';
        % SeqLoop.AQSlice(1).ZeroFillWindowSize = 1.4;            % zero fill window size (high k-space values are damped by a cos^2 law)
        % SeqLoop.AQSlice(1).ZeroFillFactor   = 2;                % zero fill resolution factor
        SeqLoop.(dataFieldName).RoI         = [];
        [SeqLoop.(dataFieldName)]           = get_kSpaceAndImage(SeqLoop.(dataFieldName), SeqLoop.AQSlice(1));
        [SeqLoop.(dataFieldName), SeqLoop.AQSlice] = plot_kSpaceAndImage(SeqLoop.(dataFieldName), SeqLoop.AQSlice(1));
        
        %% save data as csv
        % use with averages not with Loops
        str             = split(HW.UserName, '_');
        mode            = str{end};
        sample          = 'tube1';
        if Seq.AQSlice(1).excitationFlipAngle == 0
            pic         = 'Noise';
        elseif Seq.AQSlice(1).excitationFlipAngle == 90
            pic         = 'Signal';
        end
        
        savenameImage   = sprintf('User/%s/double_res_%s_%s_1H_%s_avg_%d_%s_Image.csv', HW.UserName, mode, mOrder, sample, Seq.average, pic)
        savenamekSpace  = sprintf('User/%s/double_res_%s_%s_1H_%s_avg_%d_%s_kSpace.csv', HW.UserName, mode, mOrder, sample, Seq.average, pic)
        writematrix(SeqLoop.data.Image, savenameImage)
        % writematrix(SeqLoop.data.kSpace, savenamekSpace)
        savenameImage   = sprintf('User/%s/double_res_%s_%s_23Na_%s_avg_%d_%s_Image.csv', HW.UserName, mode, mOrder, sample, Seq.average, pic)
        savenamekSpace  = sprintf('User/%s/double_res_%s_%s_23Na_%s_avg_%d_%s_kSpace.csv', HW.UserName, mode, mOrder, sample, Seq.average, pic)
        writematrix(SeqLoop.dataX.Image, savenameImage)
        % writematrix(SeqLoop.dataX.kSpace, savenamekSpace)

    case 'Def_Na23_X_H1'
        warning('Changing the order of GammaDef and GammaX might create issues with the calibration file PaUout2AmplitudeCal.m. It might be usefull to comment out old measurements.');
        %% find correct 90 pulse duration and coil efficiency for gamma Def
        minTime                             = 0;
        doPlot                              = 1;
        % as HW.FindPulseDuration.tPulse90Estimated is an old measurement, 1 iteration is
        % sufficient
        iterations                          = 1; 
        % for 10mm saturated NaCl CuSO4 sample (23Na)
        T1_estimate                         = 40e-3;    
        % the 23Na signal needs averaging for a smoother result
        Seq.average                         = 10;
        [HW, mySave]                        = Find_PulseDuration(HW, mySave, minTime, doPlot, iterations, HW.FindPulseDuration.tPulse90Estimated, T1_estimate, Seq);
        % LoadSystem to write the coil efficiency from PaUout2AmplitudeCal.m in the
        % current workspace HWClass
        LoadSystem;
        
        %% find correct 90 pulse duration and coil efficiency for gamma X
        minTime                             = 0;
        doPlot                              = 1;
        % as HW.FindPulseDuration.tPulse90Estimated is an old measurement, 1 iteration is
        % sufficient
        iterations                          = 1; 
        % for 10mm saturated NaCl CuSO4 sample (1H)
        T1_estimate                         = 64e-3;    
        % the 1H signal is strong enough to skip averaging
        Seq.average                         = 1;
        % HWClass has no property HW.FindPulseDuration.tPulse90EstimatedX
        % using HW.TX(1).PaUout2AmplitudeEstimatedX(1) to calculate
        % tPulse90EstimatedX
        tPulse90EstimatedX                  = pi / (2*HW.GammaX*3.7*HW.TX(1).PaUout2AmplitudeEstimatedX(1));
        Seq.useGammaX                       = true;
        [HW, mySave]                        = Find_PulseDuration(HW, mySave, minTime, doPlot, iterations, tPulse90EstimatedX, T1_estimate, Seq);
        % LoadSystem to write the coil efficiency from PaUout2AmplitudeCal.m in the
        % current workspace HWClass
        LoadSystem;
        
        %% find correct shim
        % re-check system frequency
        [HW, mySave]                        = Find_Frequency_Sweep(HW, mySave, 0);
        [HW, mySave]                        = Find_Shim(HW, mySave);
        % LoadSystem to write the coil efficiency from MagnetShimCal.m in the
        % current workspace HWClass
        LoadSystem;
        
        %% imaging parameters
        % for probe_10mm_23Na1H_dual saturated NaCl CuSO4 sample
        % basic SE with slice selection gradient
        Seq.AQSlice(1).dualNuclearImage     = true; 
        
        % toggle to 0 for noise measurement
        Seq.AQSlice(1).excitationFlipAngle  = 90;
        
        % use echoTime and tRep for maximal signal amplitude
        % measured using Demo_T1.m for saturated NaCl CuSO4 sample
        T1_23Na_10mm_NaCl_CuSO4             = 40e-3;
        T1                                  = T1_23Na_10mm_NaCl_CuSO4;
        % measured using Demo_T1.m for saturated NaCl CuSO4 sample
        T2_23Na_10mm_NaCl_CuSO4             = 35e-3;
        T2                                  = T2_23Na_10mm_NaCl_CuSO4;
        
        % minimal T2 relaxation
        % approx. 90% transversal magnetization left
        Seq.tEcho                           = 0.1 * T2;
        
        % to adjust for correct sequence timing there needs to be a trade off
        % between minimal pixel bandwidth and minimal echo time
        
        % values cannot be copied from probe_10mm_23Na1H_23Na_setup_SE.m as
        % both rf pulses of GammaDef and GammaX get overlayed and mismatch
        % the timing of the shorter 23Na pulse
        
        % 15e-3 is probably too high for good contrast due to short T2
        % with lower tEcho the timing with AQ length does not fit
        % for higher HzPerPixMin (1/AQ length) the gradients are not
        % sufficient
        Seq.tEcho                           = 15e-3;
        
        % maximal T1 recovery
        % approx. 95% recovered longitudinal magnetization
        Seq.RepetitionTime                  = 3 * T1;
        
        % number of averages
        % Seq.Loops does averaging as well 
        % -> Seq.Loops is superior for postprocessing the data as the data of every average is
        % captured
        Seq.average                         = 1;
        
        % important for breaks in the sequence
        % Seq.tRep takes precedence
        % probably not needed for a simple SE sequence
        Seq.T1Estimated                     = T1;
        Seq.T2Estimated                     = T2;
        
        Seq.SteadyState_PreShots180         = 0;
        % changes the the timing of CL (= command line) times of the drivel
        % Seq.SingletRep = 1 is not valid for dual nuclear imaging
        Seq.SingletRep                      = 0;
        % single AQ for each excitation and inversion pulse
        Seq.AQSlice(1).TurboFactor          = 1;
        
        % image orientation
        % slice in coil center in xz-plane
        % x = phase; y = slice; z = read;
        Seq.AQSlice(1).alfa                 = 0;               
        Seq.AQSlice(1).phi                  = 0;               
        Seq.AQSlice(1).theta                = pi/2;  
        
        Seq.AQSlice(1).Center2OriginImage   = [0, 0, 0]; 
        
        % number of samples
        Seq.AQSlice(1).nRead                = 32;   
        Seq.AQSlice(1).nPhase(1)            = 1;                             
        Seq.AQSlice(1).nPhase(2)            = 32;  
        Seq.AQSlice(1).nPhase(3)            = 1; 
        % pixel bandwidth
        Seq.AQSlice(1).HzPerPixMin          = 270; 
        
        % FOV
        % increased FOV to show full 1H image as FOV scales with GammaDef
        Seq.AQSlice(1).sizeRead             = 40e-3;            
        Seq.AQSlice(1).sizePhase(2)         = 40e-3;        
        Seq.AQSlice(1).thickness            = 0.005;  
        
        % oversampling
        Seq.AQSlice(1).ReadOS               = [];      
        Seq.AQSlice(1).PhaseOS(1)           = 1;                           
        Seq.AQSlice(1).PhaseOS(2)           = 2;                            
        Seq.AQSlice(1).PhaseOS(3)           = 1;                            
        Seq.AQSlice(1).PhaseOS(Seq.AQSlice(1).nPhase==1) = 1;   
        
        % toggle plots
        Seq.AQSlice(1).plotkSpace           = 1;        % plot k-space
        Seq.AQSlice(1).plotImage            = 1;        % plot image             
        Seq.plotSeq                         = [1:3];    % plot seq gradients 1:3
        
        %% start measurement
        % re-check system frequency
        [HW, mySave]                        = Find_Frequency_Sweep(HW, mySave, 0);
        [SeqLoop, mySave]                   = sequence_Spin_Echo(HW, Seq, AQ, TX, Grad, mySave);
        %%
        % display image at secondary frequency
        SeqLoop.AQSlice(1).plotImageHandle  = 210;  % for 1d/2d images
        SeqLoop.AQSlice(1).plotImagePhase   = 211;
        SeqLoop.AQSlice(1).plotkSpace       = 212;
        SeqLoop.AQSlice(1).plotkSpacePhase  = 213;
        dataFieldName                       = 'dataX';
        % SeqLoop.AQSlice(1).ZeroFillWindowSize = 1.4;            % zero fill window size (high k-space values are damped by a cos^2 law)
        % SeqLoop.AQSlice(1).ZeroFillFactor   = 2;                % zero fill resolution factor
        SeqLoop.(dataFieldName).RoI         = [];
        [SeqLoop.(dataFieldName)]           = get_kSpaceAndImage(SeqLoop.(dataFieldName), SeqLoop.AQSlice(1));
        [SeqLoop.(dataFieldName), SeqLoop.AQSlice] = plot_kSpaceAndImage(SeqLoop.(dataFieldName), SeqLoop.AQSlice(1));
        
        %% save data as csv
        % use with averages not with Loops
        str             = split(HW.UserName, '_');
        mode            = str{end};
        sample          = 'NaCl_CuSO4';
        if Seq.AQSlice(1).excitationFlipAngle == 0
            pic         = 'Noise';
        elseif Seq.AQSlice(1).excitationFlipAngle == 90
            pic         = 'Signal';
        end
        
        savenameImage   = sprintf('User/%s/double_res_%s_%s_23Na_%s_avg_%d_%s_Image.csv', HW.UserName, mode, mOrder, sample, Seq.average, pic)
        savenamekSpace  = sprintf('User/%s/double_res_%s_%s_23Na_%s_avg_%d_%s_kSpace.csv', HW.UserName, mode, mOrder, sample, Seq.average, pic)
        writematrix(SeqLoop.data.Image, savenameImage)
        % writematrix(SeqLoop.data.kSpace, savenamekSpace)
        savenameImage   = sprintf('User/%s/double_res_%s_%s_1H_%s_avg_%d_%s_Image.csv', HW.UserName, mode, mOrder, sample, Seq.average, pic)
        savenamekSpace  = sprintf('User/%s/double_res_%s_%s_1H_%s_avg_%d_%s_kSpace.csv', HW.UserName, mode, mOrder, sample, Seq.average, pic)
        writematrix(SeqLoop.dataX.Image, savenameImage)
        % writematrix(SeqLoop.dataX.kSpace, savenamekSpace)

end