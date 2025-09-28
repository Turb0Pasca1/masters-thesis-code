LoadSystem;
% set center frequency of VNA-measurement
HW.Network.fCenter = 5.694e6;
HW.Network.fCenter = 21.527e6;
% run VNA GUI
% do VNA calibration with open, short and load terminations via the menu
hNetworkAnalyzer = GUI_NetworkAnalyzer;

%% extract data from the still opened VNA GUI

handles = guidata(hNetworkAnalyzer);
Network = handles.Network;

f = Network.Frequency;
% Network.ReflectionRaw -> data without calibration data
S = Network.Reflection;
T = table(f,S);
% adjust datanme
dataname = 'data.csv';
writetable(T, [dataname]);
% save message
disp('saved');