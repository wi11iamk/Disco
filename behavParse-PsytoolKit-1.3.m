%% Initialize parameters

clear
clc
diary off;
debugScript = false;
genLog = true;
vScript = 'v1.3.0';

%v1.1.0 - Changed input structure to accommodate change in file naming
%convention (switched from numeric subject codes to alpha MEG codes to
%eliminate possibility of mapping errors of behavior to MEG data).  Added
%code to parse behavioral data for "random" blocks on Days 2 and 3

%v1.2.0 - Added new correct sequence speed measure. Refactored to fold in transitionLibrary build and missing 2
%imputation (due to malfunctioning button box..."sticky key"). Removed
%plotting functions.

%v1.3.0 - Merged script for generating transition library and missing "2"
%timestamp imputation (one-stop shopping for all pre-processing now)

dataRoot = 'C:\Users\William\Desktop\test\';

if genLog
    diary(['behavParse_' datestr(now,'yyyymmddHHMMSS') '.log']);
end
disp(['Running ' mfilename('fullpath') ' ' vScript]);

if debugScript
    dbstop if error
end

% Define target sequence
targSeq = '41324';

% Define models for curve-fitting
modelfun1 = @(b,x)(b(1) + b(2)*(1-exp(-b(3)*(x-b(4)))));
modelfun2 = @(b,x)(b(1) + b(2)*x);

% Initialize random number generator for reproducibility purposes
rng('default')

% Set robust model fitting options.
opts = statset('nlinfit');
opts.RobustWgtFun = 'huber'; %[] %'huber' or 'bisquare'

if isempty(opts.RobustWgtFun)
    disp('Robust regression using input derivative weighting function.');
else
    disp(['Robust Weighting Function set to "' opts.RobustWgtFun '".']);
end

%Store list of all raw data files in variable
allFiles = dir([dataRoot filesep '*.data.txt']);
allFiles = {allFiles.name};

if isempty(allFiles)
    disp('No input files found. Please check "dataRoot" path. Exiting now.')
end

oe_ind = 0;
% avg_ind = 1;
place_ = 0;
% ckp_avg = zeros(4,1);
events_ind = zeros(4,1);
% avg_ = zeros(4,1);
cs_seq = zeros(4,1);
cs_trial = zeros(4,1);
ind_seq = zeros(4,1);
micro_off = zeros(4,1);
micro_on = zeros(4,1);
on_ = 1;
            
%% Loop through all files
for curF = allFiles
    
    %Grab subject code from file name
    curSub = curF{:}(1:3);
    
    %Read file contents into a structure that can be parsed
    disp(['Loading ' curF{:} '...']);
    tempF = textread([dataRoot filesep curF{:}],'%s','delimiter','\n'); %read each line into a cell containing a string array
    tempF = cellfun(@str2num,tempF,'UniformOutput',0); %Convert cell contents from string to numeric arrays (the space delimiter help us here)
    tempF(cellfun(@length,tempF)==2) = []; %Remove the "scoring" lines
    tempF = cell2mat(tempF); %Now we can convert into an Mx5 matrix
    
    %Find trial onsets
    iTrialStart = find(diff(tempF(:,1))==1)+1; %We use the lines where the trial number increases to mark trial onsets.
    nTr = length(iTrialStart); %Store number of trials
    iTrialEnd = [iTrialStart(2:end)-1; size(tempF,1)]; %Trial end is defined as the last KP prior to the next trial onset and the last KP recorded
    
    %Initialize output
    trial.ver = vScript; %Write script version to output data structure
    trial.targSeq = cell(nTr,1); %Initialize variable that stores output sequence
    
    % Create header for and initialize CSV events output
    outEventsHdr = {'Trial','KeyID','TimeStamp','isCorrectSequenceMemberKP','isSequenceEnd','TargetSequence','cs/s - sequence', 'cs/s - trial', 'micro-online','micro-offline'};
    outEvents = [];
    
    %% Loop through all trials
    for curTr = 1:nTr
        
        %Write info to screen
        disp(' ');
        disp(['Evaluating Subject ' curSub ' | Trial ' num2str(curTr,'%02.0f') '...']);
        
        iKP = iTrialStart(curTr):iTrialEnd(curTr); %Create index for trial samples
        nKP = length(iKP);
        
        %Grab entire typed sequence for current trial and convert back to
        %string so we can use regexp to look at the sequence structure
        tempSeq = num2str(tempF(iKP,3))';
        
        %Write target sequence for current trial to output variable
        trial.targSeq(curTr) = {targSeq};
        
        %% Timestamp stuff
        tempTrialTime = tempF(iKP,4)'; %Grab trial-based timestamps
        tempExperTime = tempF(iKP,5)'; %Grab global timestamps
        clear curKP iKP
        if ~isempty(tempTrialTime) %If the trial has KPs
            %Quick check to make sure no havoc-wreaking simultaneous timestamps exist (only keep first instance if there are)
            [tempTrialTime,iA] = unique(tempTrialTime);
            tempExperTime = tempExperTime(iA);
            tempSeq = tempSeq(iA);
            nKP = length(tempSeq);
        else %If no KPs
            tempTrialTime = NaN;
        end
        
        %Write info to screen
        disp('Current target sequence:');
        disp(targSeq);
        disp('Observed key-press sequence:');
        disp(tempSeq);
        
        %Save timestamps to output
        trial.trialTime(curTr,1) = {tempTrialTime};
        trial.experTime(curTr,1) = {tempExperTime};
        clear newKP
        
        %% Determine if key presses are members of correct sequences
        nTargSeq = length(targSeq);
        checkKPmat = repmat(tempSeq,nTargSeq,1); %Replicate KP sequence
        checkKP = tempSeq;
        for jS = 0:nTargSeq-1
            iCS = regexp(tempSeq,circshift(targSeq,-jS));
            for kCS = iCS
                checkKPmat(jS+1,kCS:kCS+nTargSeq-1) = '0';
            end
        end
        iCorrSeqKP = sum(checkKPmat=='0',1)>0;
        checkKP(iCorrSeqKP) = '0';
        disp('Incorrect sequence key-press locations:');
        disp(checkKP);
        
        iNonSeqKP = find(checkKP~='0');
        
        if length(tempSeq)==length(iNonSeqKP)
            disp('No correct sequences found for this trial.');
        end
        
        disp([num2str(sum(checkKP=='0')) ' of ' num2str(length(checkKP)) ' key presses for this trial were members of correct sequences.']);
        trial.isCorrSeqMem(curTr,1) = {checkKP=='0'};
        trial.corrSeqKPratio(curTr,1) = sum(checkKP=='0')./length(checkKP);
        trial.seq(curTr,1) = {str2num(tempSeq')'}; %#ok<ST2NM>
        
        %% Update CSV output
                
        isSeqEnd = false(length(tempSeq),1);
        isSeqEnd(regexp(tempSeq,targSeq)+4) = true;
        if ~isempty(tempSeq)
            outEvents = [outEvents; ...
                [num2cell([repmat(curTr,length(tempSeq),1) ...
                str2num(tempSeq(:)) ...
                trial.experTime{curTr}(:) ...
                trial.isCorrSeqMem{curTr}(:) ...
                isSeqEnd(:)]) ...
                repmat({targSeq},length(tempSeq),1)]...
                ];
        end
        
        %% Time and speed variables
        trial.RT(curTr,1) = tempTrialTime(1); %Use the trial-based timestamp for the first KP as the RT
        if isnan(tempTrialTime) %Add empty values if no KPs for this trial
            trial.kpTransSpeed(curTr,1) = {NaN};
            trial.micro.kpTransSpeed(curTr,1) = {NaN(1,1e4)};
            trial.corrSeqT(curTr,1) = {NaN};
            trial.corrSeqSpeed(curTr,1) = {NaN};
            trial.corrCircSeqSpeed(curTr,1) = {NaN};
            trial.micro.t(curTr,1) = {1:1e4};
            trial.micro.corrSeqSpeed(curTr,1) = {NaN(1,1e4)};
            trial.micro.corrCircSeqSpeed(curTr,1) = {NaN(1,1e4)};
        else
            temp_kpTS = (diff(tempTrialTime).*1e-3).^-1; % Calculate keypress transition speeds
            trial.kpTransSpeed(curTr,1) = {temp_kpTS};
            x_kpTS = tempTrialTime(2:end); % Grab the timestamps for time-series interpolation (needed for micro-online and micro-offline)
            jKP = ~isnan(temp_kpTS) | ~isnan(x_kpTS);
            if length(x_kpTS(jKP))<3 % If we don't have enough KPs to interpolate, just fill with NaNs
                trial.micro.kpTransSpeed(curTr,1) = {NaN(1,1e4)};
                isSeqEnd = false(length(tempSeq),1);
            else % Perform time-series interpolation
                temp_micro_kpTS = interp1(x_kpTS(jKP),temp_kpTS(jKP),1:1e4,'makima',NaN); clear temp_kpTS jKP x_kpTS
                temp_micro_kpTS(1:find(isfinite(temp_micro_kpTS),1,'first')-1) = temp_micro_kpTS(find(isfinite(temp_micro_kpTS),1,'first'));
                temp_micro_kpTS(find(isfinite(temp_micro_kpTS),1,'last')+1:end) = temp_micro_kpTS(find(isfinite(temp_micro_kpTS),1,'last'));
                trial.micro.kpTransSpeed(curTr,1) = {temp_micro_kpTS};
                all_iCS = [];
                all_corrSeqSpeed = [];
                for jS = 0:nTargSeq-1
                    iCS = regexp(tempSeq,circshift(targSeq,-jS));
                    all_iCS = [all_iCS iCS];
                    if jS == 0
                        trial.corrSeqT(curTr,1) = {tempTrialTime(iCS+nTargSeq-1) - tempTrialTime(iCS)};
                        trial.corrSeqSpeed(curTr,1) = {((tempTrialTime(iCS+nTargSeq-1) - tempTrialTime(iCS)).*1e-3).^-1};
                        for cs_ = 1:size(trial.corrSeqSpeed{curTr},2)
                            cs_seq(place_+cs_,1) = trial.corrSeqSpeed{curTr}(cs_);
                        end
                        ind_ = iCS+nTargSeq-1;
                        for i_ = 1:size(ind_,2)
                            ind_seq(i_+place_,1) = ind_(1,i_)+oe_ind;
                        end
                        isSeqEnd = false(length(tempSeq),1);
                        isSeqEnd(iCS+nTargSeq-1) = true;
                    end
                    all_corrSeqSpeed = [all_corrSeqSpeed ((tempTrialTime(iCS+nTargSeq-1) - tempTrialTime(iCS)).*1e-3).^-1];
                end
                [~,sort_iCS] = sort(all_iCS);
                trial.corrCircSeqSpeed(curTr,1) = {all_corrSeqSpeed(sort_iCS)};
                if ~isempty(tempSeq)
                    [t,seqSpeed] = corrSeqSpeed(tempTrialTime,tempSeq,targSeq,[1 1e4],1,'makima',false);
                    trial.micro.t(curTr,1) = {t};
                    trial.micro.corrSeqSpeed(curTr,1) = {seqSpeed};
                    [~,circSeqSpeed] = corrSeqSpeed(tempTrialTime,tempSeq,targSeq,[1 1e4],1,'makima',true);
                    trial.micro.corrCircSeqSpeed(curTr,1) = {circSeqSpeed};
                end
            end
        end
        
        % tapping speed addition to outEvents
        sum_ = 0;
        count_ = 0;
         
        % micro learning 
        corr_seqsize = size(trial.corrSeqSpeed{curTr},2);
            
        if corr_seqsize == 0
            micro_on(on_,1) = 0;
            trial.corrSeqSpeed{curTr}(1) = NaN;
        else
            % sum the tapping speed of individual correct sequences to get
            % speed per trial
            for ckp_ = 1:size(trial.corrSeqSpeed{curTr},2)
                sum_ = sum_ + cs_seq(ckp_+place_,1);
                count_ = count_ + 1;
            end

            avg_(curTr,1) = (sum_/count_);
            
            place_ = place_ + cs_;
             
            % micro online
            if corr_seqsize > 1
                micro_on(on_,1) = trial.corrSeqSpeed{curTr}(corr_seqsize) - trial.corrSeqSpeed{curTr}(1);
            else
                micro_on(on_,1) = trial.corrSeqSpeed{curTr}(corr_seqsize);
            end
            
        end
        
        oe_ind = oe_ind + length(tempSeq);
        events_ind(curTr,1) = size(outEvents,1);
        on_ = on_ + 1;
        
        
    end %END trial loop
    
    %% Fit models to trial-by-trial learning curve data
    curModel = 'exponential';
    disp(['Fitting ' curModel ' model to learning curve data...'])
    x = (1:length(trial.corrSeqSpeed))';
    y = cellfun(@nanmedian,trial.corrSeqSpeed); %Grab the median correct sequence speed for each trial
    modelfun = modelfun1;
    b0 = [y(find(~isnan(y),1,'first')) nanmedian(y(end-10:end)) 0.25 0]; %#ok<FNDSB>
    lb = [0 0 0 0];
    ub = [5 15 2 3];
    bFit = lsqcurvefit(modelfun,b0,x(~isnan(y)),y(~isnan(y)),lb,ub);
    trial.fitCorrSeqSpeed.exp.y0 = bFit(1);
    trial.fitCorrSeqSpeed.exp.C = bFit(2);
    trial.fitCorrSeqSpeed.exp.kappa = bFit(3);
    trial.fitCorrSeqSpeed.exp.x0 = bFit(4);
    trial.fitCorrSeqSpeed.exp.y = bFit(1) + bFit(2)*(1-exp(-bFit(3)*(x-bFit(4))));
    trial.fitCorrSeqSpeed.exp.tau = find(...
        trial.fitCorrSeqSpeed.exp.y - trial.fitCorrSeqSpeed.exp.y(1) ...
        > ...
        .95*(trial.fitCorrSeqSpeed.exp.y(end) - trial.fitCorrSeqSpeed.exp.y(1))...
        ,1,'first');
    disp(['Exp fit is: ' num2str(bFit(1)) ' + ' ...
        num2str(bFit(2)) '*(1-exp(-' num2str(bFit(3)) '*(x-' num2str(bFit(4)) ')))']);
    %END model fitting to block data
    
    %% SAVE OUTPUT
    
    %Write MAT data file
    outMAT = strrep(curF{:},'.txt','.mat');
    disp(['Saving ' dataRoot filesep outMAT '...']);
    save([dataRoot filesep outMAT],'trial');

    % micro offline
    off_ = 1;
    for k_ = 1:size(trial.corrSeqSpeed,1)-1
        last = size(trial.corrSeqSpeed{k_},2);
        if isnan(trial.corrSeqSpeed{k_}(1))
            micro_off(k_,1) = trial.corrSeqSpeed{k_+1}(1) - 0;
        else
            micro_off(k_,1) = trial.corrSeqSpeed{k_+1}(1) - trial.corrSeqSpeed{k_}(last);
        end
    end

    for j_ = 1:size(cs_seq,1)
        if cs_seq(j_,1) == 0 
            outEvents{j_,7} = 0;
        else
            outEvents{ind_seq(j_),7} = cs_seq(j_,1);
        end
    end
    
    for i_ = 1:size(trial.corrSeqT,1)
        outEvents{events_ind(i_),8} = avg_(i_,1);
        outEvents{events_ind(i_),9} = micro_on(i_,1);
        if i_ <= 35
            outEvents{events_ind(i_)+1,10} = micro_off(i_,1);
        end
    end
    
    clear trial
    
    %Write CSV event file
    outEventsTable = cell2table(outEvents,'VariableNames',outEventsHdr);
    outFile = strrep(outMAT,'data.mat','events.csv');
    disp(['Saving ' dataRoot filesep outFile '...']);
    writetable(outEventsTable,[dataRoot filesep outFile]);
    clear trial outEvents outEventsTable outFile
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    oe_ind = 0;
%     avg_ind = 1;
    place_ = 0;
%     ckp_avg = zeros(4,1);
    events_ind = zeros(4,1);
    avg_ = zeros(4,1);
    cs_seq = zeros(4,1);
    cs_trial = zeros(4,1);  
    ind_seq = zeros(4,1);
    micro_off = zeros(4,1);
    micro_on = zeros(4,1);
    on_ = 1;

end

%% End logfile
if genLog
    diary off;
end
