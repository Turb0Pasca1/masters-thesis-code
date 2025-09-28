% probe_15mm_1H_setup_SE.m

%% initialize system
% run script in main openMATLAB folder containing all files
LoadSystem probe_15mm_1H;
% find system frequency
[HW, mySave]                        = Find_Frequency_Sweep(HW, mySave, 0);

%% find correct 90 pulse duration and coil efficiency
[HW, mySave]                        = Find_PulseDuration(HW, mySave, 0, 1);
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
% for probe_10mm_23Na1H_1H saturated NaCl CuSO4 sample
% basic SE with slice selection gradient

% toggle to 0 for noise measurement
Seq.AQSlice(1).excitationFlipAngle  = 90;

% use echoTime and tRep for maximal signal amplitude
% measured using Demo_T1.m for saturated NaCl CuSO4 sample
T1_1H_10mm_NaCl_CuSO4               = 64e-3;
T1_1H_10mm_NaCl                     = 2290e-3;
T1                                  = T1_1H_10mm_NaCl_CuSO4;
% T1                                  = T1_1H_10mm_NaCl;
% measured using Demo_T1.m for saturated NaCl CuSO4 sample
T2_1H_10mm_NaCl_CuSO4               = 58e-3;
T2_1H_10mm_NaCl                     = 2080e-3;
T2                                  = T2_1H_10mm_NaCl_CuSO4;
% T2                                  = T2_1H_10mm_NaCl;

% minimal T2 relaxation
% approx. 90% transversal magnetization left
Seq.tEcho                           = 0.1 * T2;

% to adjust for correct sequence timing there needs to be a trade off
% between minimal pixel bandwidth and minimal echo time
Seq.tEcho                           = 8e-3;

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
Seq.AQSlice(1).HzPerPixMin          = 400; 

% FOV
Seq.AQSlice(1).sizeRead             = 18e-3;            
Seq.AQSlice(1).sizePhase(2)         = 18e-3;        
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

%% save data as csv
% use with averages not with Loops
str             = split(HW.UserName, '_');
mode            = str{end};
sample          = 'tube2';
if Seq.AQSlice(1).excitationFlipAngle == 0
    pic         = 'Noise';
elseif Seq.AQSlice(1).excitationFlipAngle == 90
    pic         = 'Signal';
end

savenameImage   = sprintf('User/%s/commercial_15mm_%s_%s_avg_%d_%s_Image.csv', HW.UserName, mode, sample, Seq.average, pic)
savenamekSpace  = sprintf('User/%s/commercial_15mm_%s_%s_avg_%d_%s_kSpace.csv', HW.UserName, mode, sample, Seq.average, pic)
writematrix(SeqLoop.data.Image, savenameImage)
% writematrix(SeqLoop.data.kSpace, savenamekSpace)