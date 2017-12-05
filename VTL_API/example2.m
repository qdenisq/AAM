%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This example generates the speech waveform directly from a gestural
% score.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% File name of the dll and header file (they differ only in the extension).

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
% Synthesize from a gestural score.
%
% int vtlGesToWav(const char *speakerFileName, const char *gestureFileName,
%  const char *wavFileName, const char *feedbackFileName);
% *****************************************************************************

speakerFileName = 'JD2.speaker';
gestureFileName = 'Gurken.ges';
wavFileName = 'Gurken.wav';
feedbackFileName = 'Gurken.txt';

disp('Calling gesToWav()...');

failure = calllib(libName, 'vtlGesToWav', speakerFileName, gestureFileName, ...
    wavFileName, feedbackFileName);

if (failure ~= 0)
    disp('Error in vtlGesToWav()! Error code:');
    failure
    return;
end

disp('Finished.');

% Play the synthesized wav file.
s = audioread(wavFileName);
sound(s, 22050);


