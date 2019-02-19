% Create dataset for Communities and Crime - CA1
clear;clc;

% Filenames
if isunix % Code to run on Linux platform
    % Communities and Crime dataset
    x_crime = 'Communities_Crime/communities.txt';
elseif ispc % Code to run on Windows platform
    % Communities and Crime dataset
    x_crime = 'Communities_Crime\communities.txt';
else
    disp('Platform not supported.');
end

% Importing data
x_data = readtable(x_crime,'ReadVariableNames',false);

% Size of the data
Nmax = size(x_data,1);
dimX = size(x_data,2);
% Defining matrix X to save all the data
matX_input = zeros(Nmax,dimX-4);

% First column represents state number, second county numeric, third
% the community numeric, and fourth the community name
% First column
x_state = x_data(:,1);
x_state_number = table2array(x_state);
x_state_number = str2double(x_state_number); % From string to double
x_state_number(1)=8; % For some reason, first element got wrong
% Second column
x_county = x_data(:,2);
x_county_number = table2array(x_county);
x_county_number = str2double(x_county_number); % From string to double
x_county_number(isnan(x_county_number)) = 0; % ? turns into NaN. Then we turn NaN into 0
% Third column
x_community = x_data(:,3);
x_community_number = table2array(x_community);
x_community_number = str2double(x_community_number); % From string to double
x_community_number(isnan(x_community_number)) = 0; % ? turns into NaN. Then we turn NaN into 0
% Fourth column
x_community_str = x_data(:,4); 
x_community_name = table2array(x_community_str);
x_community_name = cellstr(x_community_name); % From cell to string
% Create struct with these information
community_info = struct('x_state_number',x_state_number,'x_county_number',...
    x_county_number,'x_community_name',x_community_name);

% Now we turn all the remaining data into double
% This needs to be done in a for loop because some parts are string and
% others double
for idxD = 1:dimX-4
    % Load the function and test if this is possible
    x_data_crime_part = x_data(:,idxD+4);
    x_data_crime_part = table2array(x_data_crime_part);
    if iscell(x_data_crime_part) % If this is a string
        x_data_crime_part = str2double(x_data_crime_part); % From string to double
    end
    x_data_crime_part(isnan(x_data_crime_part)) = 0; % ? turns into NaN. Then we turn NaN into 0
    % Assign this vector to the matrix
    matX_input(:,idxD) = x_data_crime_part;
end

% The output (yi) is in the last position of the vector
y_data = matX_input(:,end);
matX_input(:,end) = []; % Remove the last dimension from output

% Save this data
if isunix % Code to run on Linux platform
    save('Communities_Crime/x_data.mat','matX_input');
    save('Communities_Crime/y_data.mat','y_data');
    save('Communities_Crime/community_info.mat','community_info');
elseif ispc % Code to run on Windows platform
    save('Communities_Crime\x_data.mat','matX_input');
    save('Communities_Crime\y_data.mat','y_data');
    save('Communities_Crime\community_info.mat','community_info');
else
    disp('Platform not supported.');
end
