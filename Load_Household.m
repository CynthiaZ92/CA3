% Create dataset for Communities and Crime - CA1
clear;clc;

% Filenames
if isunix % Code to run on Linux platform
    % Individual household electric power consumption
    x_power = 'Individual_Household/household_power_consumption.txt';
elseif ispc % Code to run on Windows platform
    % Individual household electric power consumption
    x_power = 'Individual_Household\household_power_consumption.txt';
else
    disp('Platform not supported.');
end

% Importing data
x_data = readtable(x_power,'Delimiter',';','ReadVariableNames',false);

% Size of the data
Nmax = size(x_data,1);
dimX = size(x_data,2);
% Defining matrix X to save all the data
matX_input = zeros(Nmax-1,dimX-2);

% The first row is the naming of the field
household_info = x_data(1,:);
% First column is for the date - starting from 2nd row
x_date = x_data(2:end,1);
x_date = table2array(x_date);
% Second column is for the time - starting from 2nd row
x_time = x_data(2:end,1);
x_time = table2array(x_time);

for idxD = 1:dimX-2
    % Obtain data
    x_useful_data = x_data(2:end,idxD+2);
    % Convert data to double
    x_useful_data_part = table2array(x_useful_data);
    if iscell(x_useful_data_part) % If this is a string
        x_useful_data_part = str2double(x_useful_data_part); % From string to double
    end
    x_useful_data_part(isnan(x_useful_data_part)) = 0; % ? turns into NaN. Then we turn NaN into 0
    % Assign this vector to the matrix
    matX_input(:,idxD) = x_useful_data_part;
end

y_sub_metering_1 = matX_input(:,end-2);
matX_input(:,end-2) = [];
y_sub_metering_2 = matX_input(:,end-1);
matX_input(:,end-1) = [];
y_sub_metering_3 = matX_input(:,end);
matX_input(:,end) = [];

% Converting to [0,1]
y_sub_metering_1_01 = (y_sub_metering_1-min(y_sub_metering_1))/(...
    max(y_sub_metering_1)-min(y_sub_metering_1) );
y_sub_metering_2_01 = (y_sub_metering_2-min(y_sub_metering_2))/(...
    max(y_sub_metering_2)-min(y_sub_metering_2) );
y_sub_metering_3_01 = (y_sub_metering_3-min(y_sub_metering_3))/(...
    max(y_sub_metering_3)-min(y_sub_metering_3) );
% Converting to [-1,1]
y_sub_metering_1_m11 = (y_sub_metering_1_01-0.5)*2;
y_sub_metering_2_m11 = (y_sub_metering_2_01-0.5)*2;
y_sub_metering_3_m11 = (y_sub_metering_3_01-0.5)*2;

% Save this data
if isunix % Code to run on Linux platform
    save('Individual_Household/x_data.mat','x_date','x_time','matX_input');
    save('Individual_Household/y_data.mat','y_sub_metering_1','y_sub_metering_2','y_sub_metering_3');
    save('Individual_Household/y_data_01.mat','y_sub_metering_1_01','y_sub_metering_2_01','y_sub_metering_3_01');
    save('Individual_Household/y_data_m11.mat','y_sub_metering_1_m11','y_sub_metering_2_m11','y_sub_metering_3_m11');
    save('Individual_Household/household_info.mat','household_info');
elseif ispc % Code to run on Windows platform
    save('Individual_Household\x_data.mat','matX_input');
    save('Individual_Household\y_data.mat','y_sub_metering_1','y_sub_metering_2','y_sub_metering_3');
    save('Individual_Household\y_data_01.mat','y_sub_metering_1_01','y_sub_metering_2_01','y_sub_metering_3_01');
    save('Individual_Household\y_data_m11.mat','y_sub_metering_1_m11','y_sub_metering_2_m11','y_sub_metering_3_m11');
    save('Individual_Household\household_info.mat','household_info');
else
    disp('Platform not supported.');
end