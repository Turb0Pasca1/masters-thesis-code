%% B1 map

% Assumption:
% The amplitude of the image at the Ernst angle is approximately linearly
% proportional to B1.
% The amplitude of the FLASH image with the small flip angle is approx.
% quadratically proportional to B1.
% All other deviations (B0, gradient, eddy currents, ...) have approx. the same
% influence on both images.

%% FLASH with Ernst angle

LoadSystem probe_15mm_1H;                                   % load system parameters (reset to default: HW Seq AQ TX Grad)

Seq.Loops = 1;                                              % number of loop averages 1...

Seq.tEcho = 4e-3;                                           % echo time in seconds e.g. 5e-3
Seq.RepetitionTime = 25e-3;                                 % repetition time in seconds (default is Seq.tEcho*2)
T1 = 64e-3;                                                 % T1 of sample (for Ernst angle)
alpha_E = acosd (exp(-Seq.RepetitionTime/T1));              % Ernst angle
Seq.FlipAngle = alpha_E;

% % Pixels and size %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Seq.AQSlice(1).nRead = 20;                                  % number of pixels in read, if nRead>1 nPhase(1)=1
Seq.AQSlice(1).nPhase(1) = 20;                              % number of pixels in phase(1)
Seq.AQSlice(1).nPhase(2) = 30;                              % number of pixels in phase(2)
Seq.AQSlice(1).HzPerPixMin = 0;                             % bandwidth per pixel in Hz (1/HzPerPixMin = duration of AQ)
Seq.AQSlice(1).sizeRead = 15e-3;                            % size in read direction in meter
Seq.AQSlice(1).sizePhase(1) = 15e-3;                        % size in phase(1) direction in meter
Seq.AQSlice(1).sizePhase(2) = 15e-3;                        % size in phase(2) direction in meter
% Seq.AQSlice(1).thickness = 0.005;                         % slice thickness in meter
Seq.AQSlice(1).excitationPulse = @Pulse_Rect;               % excitation pulse function (type "Pulse_" than press tab for selection of pulses)

% % Oversampling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Seq.AQSlice(1).PhaseOS(1) = 2;                              % oversampling phase(1)  1...
Seq.AQSlice(1).PhaseOS(2) = 4;                              % oversampling phase(2)  1...

% % Orientation in space %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Seq.AQSlice(1).alfa = 0.0*pi;                               % 1st rotation around x axis in RAD
Seq.AQSlice(1).phi  = 0.0*pi;                               % 2nd rotation around y axis in RAD
Seq.AQSlice(1).theta= 0.0*pi;                               % 3rd rotation around z axis in RAD

% % Plot        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Seq.plotSeq = [];                                           % plot sequence where all AQs are wrapped onto each other, plots RF, AQ and Grad (1==x, 2==y, 3==z, 0 no gradient)
Seq.LoopPlot = 1;                                           % plot every loop
Seq.AQSlice(1).plotkSpace = 0;                              % plot k-space
Seq.AQSlice(1).plotImage = 1201;                            % plot image
Seq.AQSlice(1).plotPhase = 0;                               % plot phase of k-space or image
Seq.AQSlice(1).ZeroFillWindowSize = 1.4;                    % zero fill window size (high k-space values are damped by a cos^2 law)
Seq.AQSlice(1).ZeroFillFactor = 2;                          % zero fill resolution factor

% Seq.CorrectSliceRephase = double(Seq.AQSlice(1).thickness<=0.015); % correct SliceGradTimeIntegralOffset
% Seq.CorrectPhaseRephase = 0;                                % correct PhaseGradTimeIntegralRephaseOffset
% Seq.CorrectReadRephase = 0;                                 % correct ReadGradTimeIntegralOffset
% Seq.MaxGradAmpSlice = 0.05;                                 % limit slice gradient strength

[SeqLoopErnst, mySave] = sequence_Flash(HW, Seq, AQ, TX, Grad, mySave);

%% FLASH with small flip angle

% Seq.RepetitionTime = 40e-3;
Seq.FlipAngle = alpha_E/4;
% Seq.RepetitionTime = T1*log(cosd(Seq.FlipAngle));
Seq.AQSlice(1).plotImage = 1202;                  % plot image

[SeqLoopSmall, mySave] = sequence_Flash(HW, Seq, AQ, TX, Grad, mySave);

mriDevice.system.ms = mySave;


%% Calculation of B1 map
B1map = SeqLoopSmall.data.ImageSliceomaticZ ./ SeqLoopErnst.data.ImageSliceomaticZ;

RoI = SeqLoopErnst.data.RoI .* SeqLoopSmall.data.RoI;
% RoI(abs(B1map) > 2) = NaN;
B1map = B1map .* RoI;

meanB1au = mean(abs(B1map(:)), 'omitnan');

hfB1map = figure(1200);
hslB1map = sliceomatic(hfB1map, abs(B1map)/meanB1au*100, ...
  SeqLoopErnst.data.Ticks(1).ReadZ, SeqLoopErnst.data.Ticks(1).PhaseZ, SeqLoopErnst.data.Ticks(2).PhaseZ);
title(hslB1map.hAxes, 'B1 deviation map in %');
xlabel(hslB1map.hAxes, SeqLoopErnst.AQSlice(1).ReadCartesianAxis{1});
ylabel(hslB1map.hAxes, SeqLoopErnst.AQSlice(1).PhaseCartesianAxis{1});
zlabel(hslB1map.hAxes, SeqLoopErnst.AQSlice(1).PhaseCartesianAxis{2});
title(hslB1map.GetSliderX(), SeqLoopErnst.AQSlice(1).ReadCartesianAxis{1});
title(hslB1map.GetSliderY(), SeqLoopErnst.AQSlice(1).PhaseCartesianAxis{1});
title(hslB1map.GetSliderZ(), SeqLoopErnst.AQSlice(1).PhaseCartesianAxis{2});

% copy figure to add 3 main plains separately
hfB1mapY = figure(1111); clf(hfB1mapY); set(hfB1mapY, 'Visible', 'off');
hslB1mapY = copyobj(hslB1map, hfB1mapY);
hfB1mapZ = figure(1112); clf(hfB1mapZ); set(hfB1mapZ, 'Visible', 'off');
hslB1mapZ = copyobj(hslB1map, hfB1mapZ);

% add main plains
hslB1map.AddSliceX(0);
hslB1mapY.AddSliceY(0);
hslB1mapZ.AddSliceZ(0);

%% save data

writematrix(SeqLoopErnst.dataLoop.Image, 'probe_10mm_1H_B1_Ernst.csv')
writematrix(SeqLoopSmall.dataLoop.Image, 'probe_10mm_1H_B1_Small.csv')
writematrix(B1map, 'probe_10mm_1H_B1_Map.csv')