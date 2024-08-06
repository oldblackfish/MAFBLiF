clc;close all;clear;

dataset_path = 'D:\LFIQA_Datasets\SHU\'; % Set the dataset path here
savepath = './SHU_MLI_7x32x32'; % Set the save path here

% for SHU dataset
load('SHU_all_info.mat');
load('SHU_all_mos.mat');
Distorted_sceneNum = 240; 

angRes = 7;             
patchsize = 32;         
stride = 32; 

inum = 1;
for iScene = 1 : Distorted_sceneNum
    tic;
    idx = 1;
    idx_s = 0;
    h5_savedir = [savepath, '\',SHU_all_info{1}{iScene}, '\',  SHU_all_info{2}{iScene}];
    if exist(h5_savedir, 'dir')==0
        mkdir(h5_savedir);
    end
    dataPath = [dataset_path, SHU_all_info{6}{iScene}];
    LF = load(dataPath).im2;
    LF = LF(:,:,2:434,2:624,:);
    total_patch_number = 247;

    [U, V, ~, ~, ~] = size(LF);
    LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, :);
    [U, V, H, W, ~] = size(LF);
 
    dis_data_mirco = single(zeros(total_patch_number, U * patchsize, V * patchsize));

    dis_LF_VS = single(zeros(U, V, H, W));
    for u = 1 : U
        for v = 1 : V 
            dis_LF_VS(u,v,:,:) = VisualSaliency(squeeze(LF(u,v,:,:,:)));
        end
    end
    
    label = str2num(SHU_all_mos{iScene});
    all_VS_list = [];
    var_list = [];
    for h = 1 : stride : H - patchsize + 1
        for w = 1 : stride : W - patchsize + 1
            idx_s = idx_s + 1;            
            all_VS = [];
            GMmap_var = [];
            for u = 1 : U
                for v = 1 : V                        
                    temp_dis = squeeze(LF(u, v, h : h+patchsize-1, w : w+patchsize-1, :));
                    VS_Score = max(max(squeeze(dis_LF_VS(u, v, h : h+patchsize-1, w : w+patchsize-1))));
                    all_VS = [all_VS, VS_Score];
                    temp_dis = rgb2ycbcr(temp_dis);
                    temp_dis = squeeze(temp_dis(:,:,1));
                    dis_data_mirco(idx_s, u:angRes:U * patchsize, v:angRes:V * patchsize) = temp_dis;  
                end
            end

            LF4D = permute(reshape(dis_data_mirco(idx_s, :, :),[angRes, patchsize, angRes, patchsize]),[1,3,2,4]);
            for uu = 1:angRes-1
                for vv = 1:angRes-1
                    %horzontal
                    Gx = squeeze(LF4D(uu,vv,:,:))-squeeze(LF4D(uu,vv+1,:,:));
                    %vertical
                    Gy = squeeze(LF4D(uu,vv,:,:))-squeeze(LF4D(uu+1,vv,:,:));
                    %GMMap
                    GMmap = sqrt(Gx.^2+Gy.^2);
                    GMmap_var = [GMmap_var, var(double(GMmap(:)))];
                end
            end

            var_list(idx_s,1) = mean(GMmap_var);
            all_VS_list = [all_VS_list, mean(all_VS)];
        end
    end
    
    all_VS_list = all_VS_list';
    [var_list, index]  = sort(var_list);
    for i = 1:total_patch_number
        save_dis_data_mirco = squeeze(dis_data_mirco(index(i),:,:)); 

        SavePath_H5_name = [h5_savedir, '/', num2str(idx,'%06d'),'.h5'];
        h5create(SavePath_H5_name, '/MLI', size(save_dis_data_mirco), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/MLI', single(save_dis_data_mirco), [1,1], size(save_dis_data_mirco));
        h5create(SavePath_H5_name, '/score_label', size(label), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/score_label', single(label), [1,1], size(label));
        h5create(SavePath_H5_name, '/VS', size(all_VS_list(index(i))), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/VS', single(all_VS_list(index(i))), [1,1], size(all_VS_list(index(i))));
        idx = idx + 1;   
    end
    disp(['第 ', num2str(inum), ' 个场景生成', '运行时间: ',num2str(sprintf('%.3f', toc))]);
    inum = inum + 1;
end





