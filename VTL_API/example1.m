%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This example generates the transition from /a/ to /i/ using the 
% vocal tract model and the function vtlSynthBlock(...).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% File name of the dll and header file (they differ only in the extension)

libName = 'VocalTractLabApi';

if ~libisloaded(libName)
    % To load the library, specify the name of the DLL and the name of the
    % header file. If no file extensions are provided (as below)
    % LOADLIBRARY assumes that the DLL ends with .dll and the header file
    % ends with .h.
    loadlibrary(libName, libName);
    disp(['Loaded library: ' libName]);
    pause(1);
end

if ~libisloaded(libName)
    error(['Failed to load external library: ' libName]);
    success = 0;
    return;
end

% *****************************************************************************
% list the methods
% *****************************************************************************

libfunctions(libName);   

% *****************************************************************************
% Print the version (compile date) of the library.
%
% void vtlGetVersion(char *version);
% *****************************************************************************

% Init the variable version with enough characters for the version string
% to fit in.
version = '                                ';
version = calllib(libName, 'vtlGetVersion', version);

disp(['Compile date of the library: ' version]);

% *****************************************************************************
% Initialize the VTL synthesis with the given speaker file name.
%
% void vtlInitialize(const char *speakerFileName)
% *****************************************************************************

speakerFileName = 'JD2.speaker';

failure = calllib(libName, 'vtlInitialize', speakerFileName);
if (failure ~= 0)
    disp('Error in vtlInitialize()!');   
    return;
end

% *****************************************************************************
% Get some constants.
%
% void vtlGetConstants(int *audioSamplingRate, int *numTubeSections,
%   int *numVocalTractParams, int *numGlottisParams);
% *****************************************************************************

audioSamplingRate = 0;
numTubeSections = 0;
numVocalTractParams = 0;
numGlottisParams = 0;

[audioSamplingRate, numTubeSections, numVocalTractParams, numGlottisParams] = ...
    calllib(libName, 'vtlGetConstants', audioSamplingRate, numTubeSections, numVocalTractParams, numGlottisParams);

disp(['Audio sampling rate = ' num2str(audioSamplingRate)]);
disp(['Num. of tube sections = ' num2str(numTubeSections)]);
disp(['Num. of vocal tract parameters = ' num2str(numVocalTractParams)]);
disp(['Num. of glottis parameters = ' num2str(numGlottisParams)]);

% *****************************************************************************
% Get information about the parameters of the vocal tract model and the
% glottis model.
%
% void vtlGetTractParamInfo(char *names, double *paramMin, double *paramMax, 
%   double *paramNeutral);
% void vtlGetGlottisParamInfo(char *names, double *paramMin, double *paramMax, 
%   double *paramNeutral);
% *****************************************************************************

% Reserve 32 chars for each parameter.
tractParamNames = blanks(numVocalTractParams*32);
tractParamMin = zeros(1, numVocalTractParams);
tractParamMax = zeros(1, numVocalTractParams);
tractParamNeutral = zeros(1, numVocalTractParams);

[tractParamNames, tractParamMin, tractParamMax, tractParamNeutral] = ...
  calllib(libName, 'vtlGetTractParamInfo', tractParamNames, tractParamMin, ...
  tractParamMax, tractParamNeutral);
    
% Reserve 32 chars for each parameter.
glottisParamNames = blanks(numGlottisParams*32);
glottisParamMin = zeros(1, numGlottisParams);
glottisParamMax = zeros(1, numGlottisParams);
glottisParamNeutral = zeros(1, numGlottisParams);

[glottisParamNames, glottisParamMin, glottisParamMax, glottisParamNeutral] = ...
  calllib(libName, 'vtlGetGlottisParamInfo', glottisParamNames, glottisParamMin, ...
  glottisParamMax, glottisParamNeutral);

disp(['Vocal tract parameters: ' tractParamNames]);
disp(['Glottis parameters: ' glottisParamNames]);

% *****************************************************************************
% Get the vocal tract parameter values for the vocal tract shapes of /i/
% and /a/, which are saved in the speaker file.
%
% int vtlGetTractParams(char *shapeName, double *param);
% *****************************************************************************

shapeName = 'a';
paramsA = zeros(1, numVocalTractParams);
[failed, shapeName, paramsA] = ...
  calllib(libName, 'vtlGetTractParams', shapeName, paramsA);

if (failed ~= 0)
    disp('Error: Vocal tract shape "a" not in the speaker file!');   
    return;
end

shapeName = 'i';
paramsI = zeros(1, numVocalTractParams);
[failed, shapeName, paramsI] = ...
  calllib(libName, 'vtlGetTractParams', shapeName, paramsI);

if (failed ~= 0)
    disp('Error: Vocal tract shape "i" not in the speaker file!');   
    return;
end

% *****************************************************************************
% Synthesize a transition from /a/ to /i/.
%
% int vtlSynthBlock(double *tractParams, double *glottisParams, double *tubeAreas,
%   int numFrames, double frameRate_Hz, double *audio, int *numAudioSamples);
% *****************************************************************************

duration_s = 1.0;
frameRate_Hz = 200;
numFrames = round(duration_s * frameRate_Hz);
% 2000 samples more in the audio signal for safety.
audio = zeros(1, duration_s * audioSamplingRate + 2000);
numAudioSamples = 0;

% Init the arrays.
tractParamFrame = zeros(1, numVocalTractParams);
glottisParamFrame = zeros(1, numGlottisParams);
tractParams = [];
glottisParams = [];
tubeAreas = zeros(1, numFrames * numTubeSections);
tubeArticulators = blanks(numFrames * numTubeSections);

% Create the vocal tract shapes that slowly change from /a/ to /i/ from the
% first to the last frame.

for i=0:1:numFrames-1
    % The VT shhape changes from /a/ to /i/.
    d = i / (numFrames-1);
    tractParamFrame = (1-d)*paramsA + d*paramsI;

    % Take the neutral settings for the glottis here.
    glottisParamFrame = glottisParamNeutral;
    
    % Set F0 in Hz going from 120 Hz down to 100 Hz.
    glottisParamFrame(1) = 120.0 - 20.0*(i/numFrames);
    
    % Start with zero subglottal pressure and then go to 1000 Pa.
    % Somehow, P_sub must stay at zero for the first two frames - otherwise
    % we get an annoying transient at the beginning of the audio signal,
    % and I (Peter) don't know why this is so at the moment.
    if (i == 0)
        glottisParamFrame(2) = 0;
    elseif (i == 1)
        glottisParamFrame(2) = 4000;
    else
        glottisParamFrame(2) = 8000.0;
    end
    
    % Append the parameters for the new frame to the parameter vectors.
    tractParams = [tractParams tractParamFrame];
    glottisParams = [glottisParams glottisParamFrame];    
    tubeArticulators = [tubeArticulators 'NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN'];
end

% Call the synthesis function. It may calculate a few seconds.

[failed, tractParams, glottisParams, tubeAreas, tubeArticulators, audio, numAudioSamples] = ...
  calllib(libName, 'vtlSynthBlock', tractParams, glottisParams, tubeAreas, ...
  tubeArticulators, numFrames, frameRate_Hz, audio, numAudioSamples);

if (failed ~= 0)
    disp('Error: Synthesizing the block of data failed.');   
    return;
end

% Plot and play the audio signal

plot(audio);
soundsc(audio, double(audioSamplingRate));

% Plot the area function of the first and the last frame.

figure;
plot(1:1:numTubeSections, tubeAreas(1:numTubeSections), ...
    1:1:numTubeSections, tubeAreas(1+(numFrames-1)*numTubeSections:(numFrames-1)*numTubeSections + numTubeSections));
xlabel('Tube section index');
ylabel('Tube area in cm^2');

% *****************************************************************************
% Close the VTL synthesis.
%
% void vtlClose();
% *****************************************************************************

calllib(libName, 'vtlClose');

unloadlibrary(libName);
