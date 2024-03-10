% References:                                                                           %
% https://www.mathworks.com/help/radar/ug/getting-started-timmwaveradar-example.html    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars; clc; close all;


%%%   RADAR OBJESİ TANIMLAMA   %%%
% TI AWR1642BOOST sensörü için mmWaveRadar objesi oluşturulur.
tiradar = mmWaveRadar("TI AWR1642BOOST");

% Azimuth FOV değerleri sensöre uygun olarak tanımlanır.
tiradar.AzimuthLimits = [-5 5];

% Tespit edilen nesnelerin koordinatlarının "rectangular" olarak alınacağı belirtilir. 
tiradar.DetectionCoordinates = "Sensor rectangular";

% ilk parametreye .cfg dosyasının PATH bilgisi, ikinci parametreye dosya adı uzantısıyla beraber girilir.
tiradar.ConfigFile = fullfile('xwr16xx_BestRangeResolution_UpdateRate_1_2Txenabled_inSameTime.cfg'); 




%%%  UYARI   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1- RangeDopplerHeatMap ve RangeAngleHeatMap dahil edildiği takdirde data
% drops ve perfonmance glitches oluşmaması için UPDATE RATE <= 5 Hz
% seçilmeli. ( may be frameCfg -> framePeriodicity(ms) )
% 
% 2- <detected objects>,  <log magnitude range>, <rangeAzimuthHeatMap>,
% <rangeDopplerHeatMap>, flagları ENABLE edilmeli. (guiMonitor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Configure and start the sensor by invoking the first step function
tiradar();



%%%   DATA OKUMA   %%%

% Read radar measurements in a loop and plot the measurements for for 10s (specified by stopTime)
ts = tic;
stopTime = 20;
flag_1 = 1; flag_2 = 1; flag_3 = 1; flag_4 = 1;


while(toc(ts)<=stopTime)

    % Read detections and other measurements from TI mmWave Radar
    [objDetsRct,timestamp,meas,overrun] = tiradar();
  
    % Get the number of detections read
    numDets = numel(objDetsRct);
    
    % Detections will be empty if the output is not enabled or if no object is
    % detected. Use number of detections to check if detections are available
    if numDets ~= 0
        % Detections are reported as cell array of objects of type objectDetection
        % Extract  x-y position information from each objectDetection object
        xpos = zeros(1,numDets);
        ypos = zeros(1,numDets);
        SNR = zeros(1,numDets);
        for i = 1:numel(objDetsRct)
            xpos(i) = objDetsRct{i}.Measurement(1);
            ypos(i) = objDetsRct{i}.Measurement(2);
            SNR(i) = objDetsRct{i}.ObjectAttributes.SNR;
        end
        YP_Objects(flag_4,:) = [ypos zeros(1,20-length(ypos))];
        XP_Objects(flag_4,:) = [xpos zeros(1,20-length(ypos))];
        SNR_Objects(flag_4,:) = [SNR zeros(1,20-length(SNR))];
        flag_4 = flag_4+1;
    end

    % Range profile will be empty if the log magnitude range output is not enabled
    % via guimonitor command in config File
    if ~isempty(meas.RangeProfile)
        [rangeProfilePlotHandle.XData,rangeProfilePlotHandle.YData] = deal(meas.RangeGrid,meas.RangeProfile);
        RP_RangeProfile(flag_1,:) = rangeProfilePlotHandle.YData;
        RG_RangeProfile = rangeProfilePlotHandle.XData;
        flag_1 = flag_1+1;
    end

    % meas.RangeDopplerResponse will be empty, if the rangeDopplerHeatMap output is not enabled
    % via guimonitor command in config file
    if ~isempty(meas.RangeDopplerResponse)
        %  Displays a range-Doppler response map, at the ranges
        %  Doppler shifts specified
        % (') operator is complex conjugate transpose
        RDR_RangeDopplerResponse(:,:,flag_2) = meas.RangeDopplerResponse;
        RG_RangeDopplerResponse = meas.RangeGrid';
        DG_RangeDopplerResponse = meas.DopplerGrid';
        flag_2 = flag_2+1;
    end


    % meas.RangeAngleResponse will be empty, if the rangeAzimuthHeatMap output is not enabled
    % via guimonitor command in config file
    if ~isempty(meas.RangeAngleResponse)
        %rngAngleScope(meas.RangeAngleResponse,meas.RangeGrid',meas.AngleGrid')
        RAR_RangeAngleResponse(:,:,flag_3) = meas.RangeAngleResponse;
        RG_RangeAngleResponse = meas.RangeGrid';
        AG_RangeAngleResponse = meas.AngleGrid';
        flag_3 = flag_3+1;
    end
    drawnow limitrate;
end
tiradar.release();
clear tiradar;




%%%   DOSYALARA KAYDETME   %%%

time = datestr(now, 'yyyymmdd_HHMMSS');
filename = 'C:\Users\Serdar\Desktop\Üniversite yıllara göre belgeler\4.sınıf\Bitirme\Ölçümler\2024_01_04';
mkdir(filename,['xwr1642\' time]);
pathname = [filename '\xwr1642\' time];

%% TÜM DATALAR İÇİN OLUŞTURULAN KLASÖR %%

% files of Objects
save([pathname '\XP_Objects.mat'],'XP_Objects');
save([pathname '\YP_Objects.mat'],'YP_Objects');
save([pathname '\SNR_Objects.mat'],'SNR_Objects');


% files of RangeProfile
save([pathname '\RP_RangeProfile.mat'],'RP_RangeProfile');
save([pathname '\RG_RangeProfile.mat'],'RG_RangeProfile');


% % files of RangeDopplerHeatMap
% save([pathname '\RDR_RangeDopplerResponse.mat'],'RDR_RangeDopplerResponse');
% save([pathname '\RG_RangeDopplerResponse.mat'],'RG_RangeDopplerResponse');
% save([pathname '\DG_RangeDopplerResponse.mat'],'DG_RangeDopplerResponse');


% files of RangeAzimuthHeatMap
save([pathname '\RAR_RangeAngleResponse.mat'],'RAR_RangeAngleResponse');
save([pathname '\RG_RangeAngleResponse.mat'],'RG_RangeAngleResponse');
save([pathname '\AG_RangeAngleResponse.mat'],'AG_RangeAngleResponse');


%% RANGE-PROFİLE ÇİZDİRMEK İÇİN OLUŞTURULAN KLASÖR %%

mkdir(filename,['cizdir\']);
fileName=dir([filename '/cizdir/*.mat']);
[row,col] = size(fileName);
path=[filename '\cizdir\'];
row = (row/2)+1

if (row<10)
    c=0;
    dosyaAdiBir=sprintf('RG%d%d_RangeProfile.mat',c,row);
    dosyaAdi=sprintf('RP%d%d_RangeProfile.mat',c,row);
else
    dosyaAdiBir=sprintf('RG%d_RangeProfile.mat',row);
    dosyaAdi=sprintf('RP%d_RangeProfile.mat',row);
end

save([path dosyaAdiBir],'RG_RangeProfile');
save([path dosyaAdi],'RP_RangeProfile');

%% RANGE-AZİMUTH ÇİZDİRMEK İÇİN OLUŞTURULAN KLASÖR %%

mkdir(filename,['cizdir_rar\']);
fileName=dir([filename '/cizdir_rar/*.mat']);
[row,col] = size(fileName);
path=[filename '\cizdir_rar\'];
row = (row/3)+1

if (row<10)
    c=0;   
    dosyaAdiBir=sprintf('RAR%d%d_RangeAngleResponse.mat',c,row);
    dosyaAdiiki=sprintf('RG%d%d_RangeAngleResponse.mat',c,row);
    dosyaAdi=sprintf('AG%d%d_RangeAngleResponse.mat',c,row);
else
    dosyaAdiBir=sprintf('RAR%d_RangeAngleResponse.mat',row);
    dosyaAdiiki=sprintf('RG%d_RangeAngleResponse.mat',c,row);
    dosyaAdi=sprintf('AG%d_RangeAngleResponse.mat',row);
end

save([path dosyaAdiBir],'RAR_RangeAngleResponse');
save([path dosyaAdiiki],'RG_RangeAngleResponse');
save([path dosyaAdi],'AG_RangeAngleResponse');



disp('BİTTİ!')
