clc; close all; clear;

file_name = "./NBU/MAFBLiF_result_NBU.h5"; % set result name here

predict_data = h5read(file_name,'/predict_data');
score_label = h5read(file_name,'/score_label');
all_num = size(predict_data,1);
for fork=1:all_num
    single_predict_data = predict_data(fork,:)';
    single_score_label = score_label(fork,:)';
    pearson_cc_NR(fork) = abs(IQAPerformance(single_predict_data, single_score_label,'p'));
    spearman_srocc_NR(fork) = abs(IQAPerformance(single_predict_data, single_score_label,'s'));
    kendall_krocc_NR(fork) = abs(IQAPerformance(single_predict_data, single_score_label,'k'));
    rmse_NR(fork)  = abs(IQAPerformance(single_predict_data, single_score_label,'e'));
end
pearson_plcc_all = mean(abs(pearson_cc_NR));
spearman_srocc_all = mean(abs(spearman_srocc_NR));
kendall_krocc_all = mean(abs(kendall_krocc_NR));
rmse_all = mean(abs(rmse_NR));
fprintf('plcc: %.4f\n',pearson_plcc_all)
fprintf('srocc: %.4f\n',spearman_srocc_all)
fprintf('krocc: %.4f\n',kendall_krocc_all)
fprintf('rmse: %.4f\n',rmse_all)



function index = IQAPerformance(obj_score, sub_score, type)
    switch type
    case 's' % SROCC
        index = corr(obj_score,sub_score,'type','Spearman');
    case 'k' % KROCC
        index = corr(obj_score,sub_score,'type','Kendall');
    case 'p' % PLCC
        score_fit = nonlinear_fit(obj_score, sub_score);
        index = corr(score_fit,sub_score,'type','Pearson');
    case 'e' % RMSE
        score_fit = nonlinear_fit(obj_score, sub_score);
        index = sqrt(mean((score_fit-sub_score).^2));
    end
end
        
function [x_fit]= nonlinear_fit(x,y)
    if corr(x,y,'type','Pearson')>0
        beta0(1) = max(y) - min(y);
    else
        beta0(1) = min(y) - max(y);
    end
    beta0(2) = 1/std(x);
    beta0(3) = mean(x);
    beta0(4) = -1;
    beta0(5) = mean(y);
    beta = nlinfit(x,y,@logistic5,beta0);
    x_fit = feval(@logistic5, beta, x);
end

function f = logistic5(beta, x)
    f = beta(1).*(0.5-(1./(1+exp(beta(2).*(x-beta(3)))))) + beta(4).*x + beta(5);
end