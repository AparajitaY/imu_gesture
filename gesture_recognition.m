clc; 
clear; 
close all;

%Step 1-Load data
ld=load("imuGestureDataset.mat");
data=ld.imuGestureDataset;   %columns: Gesture, Accelerometer, Gyroscope, SampleRate
fs=data.SampleRate(1);
fprintf('Loaded: %d trials, sampling frequency: %dhz\n', size(data,1),fs);

% Step 2-Filtering and feature extraction
[b, a]=butter(4,[0.5 30]/(fs/2),'bandpass'); %fourth order butterworth bandpass filter
win_len=50; %0.5s windows at 100hz, so 50 samples
hop_size=25; %50% overlap
feat_mat=[];
labels={};

for i=1:size(data,1)
    imu=[filtfilt(b,a,data.Accelerometer{i}), filtfilt(b,a,data.Gyroscope{i})];
    label= char(data.Gesture(i));
    N=size(imu,1);

    for w= 1:hop_size:(N-win_len+1)
        feat_mat(end+1,:)= extract_features(imu(w:w+win_len-1,:));
        labels{end+1,1}=label;
    end
end
fprintf('Windows: %d ,features: %d\n',size(feat_mat,1),size(feat_mat,2));

%step 3-Train test split
label_cat= categorical(labels);
N=size(feat_mat,1);
rng(0)  %random number generator 
idx=randperm(N); 
n_train=round(0.8*N); % 80% of N=1178 windows for training, 20%=295 windows for testing

X_train=feat_mat(idx(1:n_train),:);     
y_train =label_cat(idx(1:n_train));
X_test=feat_mat(idx(n_train+1:end),:); 
y_test= label_cat(idx(n_train+1:end));

%step 4-Train SVM
t= templateSVM('KernelFunction','rbf','KernelScale','auto','Standardize',true);
svm_model=fitcecoc(X_train, y_train,'Learners', t);
y_pred=predict(svm_model,X_test);
accuracy= sum(y_pred==y_test)/numel(y_test)*100;
fprintf('Test Accuracy: %.2f%%\n',accuracy);

%step 5-confusion matrix
figure;
cm=confusionchart(y_test, y_pred);
cm.Title=sprintf('Confusion Matrix');
cm.RowSummary='row-normalized';

%step 6-Raw signal
trial=find(string(data.Gesture)=="up", 1);   
accel =data.Accelerometer{trial};
t=(0:size(accel,1)-1)/fs;

figure;
plot(t,accel);
xlabel('Time (s)');
ylabel('Acceleration');
title('Raw Accelerometer Signal');
legend('X','Y','Z');
grid on; 

%step-7 filtered vs raw
figure;
accel_raw = data.Accelerometer{trial};
accel_filt = filtfilt(b, a, accel_raw);
subplot(2,1,1);
plot(t, accel_raw); title('Raw Accelerometer Signal'); ylabel('Accel'); grid on;
legend('X','Y','Z');
subplot(2,1,2);
plot(t, accel_filt); title('Filtered Accelerometer Signal (Butterworth)'); 
xlabel('Time (s)'); ylabel('Accel'); grid on;
legend('X','Y','Z');


function feats= extract_features(window)
%4 features ×6 channels= 24 features total
    f=[];
    for ch=1:6
        x=window(:,ch);
        f(end+1)=mean(x);           %mean-direction of motion
        f(end+1)=std(x);            %spread 
        f(end+1)=rms(x);            %power-intensity
        f(end+1) =max(x)-min(x);   %range-extent of movement
    end
    feats=f;
end

