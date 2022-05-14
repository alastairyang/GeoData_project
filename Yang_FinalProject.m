%% Main script
% process data: 1 if you want to process the separate datasets
%               0 if you want to directly use the processed dataset
process_data = 0;

    %% Load data
switch process_data
    case 1
        coor_path = '/Users/donglaiyang/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents_Alastair’s_MacBook_Pro/Buffalo/Research/git_research/GIS/sampled_xy.csv';
        data_path = '/Users/donglaiyang/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents_Alastair’s_MacBook_Pro/Buffalo/Research/git_research/GIS/sampled_UHZ.csv';
        coor_csv = readtable(coor_path,'VariableNamingRule','preserve');
        data_csv = readtable(data_path,'VariableNamingRule','preserve');
        
        %% Merge table
        % Rename some variables
        data_csv.Properties.VariableNames{'Ice sheet'} = 'Velocity';
        data_csv.Properties.VariableNames{'Ice thickn'} = 'Thickness';
        data_csv.Properties.VariableNames{'Surface Sl'} = 'Surface Slope';
        data_csv.Properties.VariableNames{'Surface el'} = 'Surface Elevation';
        
        % merge and make a new table with all data
        data = data_csv;
        data.x = coor_csv.x;
        data.y = coor_csv.y;
        
        
        %% Pre-processing
        % Make NaN -> 0
        data.Velocity(isnan(data.Velocity)) = 0;
        
        % change id with 000 delimiter
        delim = "000";
        unique_ids = unique(data.id);
        idx = contains(string(unique_ids), delim);
        
        % Make them into a seperate data table called data_bra
        branch_ids = unique_ids(idx);
        data_bra = table;
        for j = 1:length(branch_ids)
            % append
            data_bra = [data_bra; data(data.id==branch_ids(j),:)];
            % remove the rows in data table
            data(data.id==branch_ids(j),:) = [];
        end
        
        unique_ids(idx) = [];
        
        % The surface slope from QGIS might be too noisy
        % hence I construct a spline interpolation of elevation and calculate the
        % slope from that
        N = length(unique_ids);
        
        for k = 1:N
            % get unique rows and remove extranuous rows
            this_glc = data(data.id == unique_ids(k),:);
            [~,idata] = unique(this_glc.distance);
            % see what is not unique
            nonunique_i = ~ismember(1:size(this_glc,1),idata);
            vertex_idx = 0:size(this_glc,1)-1;
            vertex_nonunique = vertex_idx(nonunique_i);
            % remove the row of the original data table
            data(data.id==unique_ids(k) & data.vertex_ind==vertex_nonunique, :) = [];
            
            % NOW look at the subseted data
            this_glc = this_glc(idata,:);
            % new distance axis: interval = 100 meter
            dist_new = min(this_glc.distance):100:max(this_glc.distance);
            elev_new = interp1(this_glc.distance, this_glc.('Surface Elevation'), dist_new, 'spline');
            
            % get slope in the new data array
            slopes = gradient(elev_new, 100);
            nrst_i = dsearchn(dist_new', this_glc.distance);
            slope_calc = transpose(slopes(nrst_i));
            
            % substitute the surface slope data
            data.('Surface Slope')(data.id == unique_ids(k),:) = abs(slope_calc);
        end
        
        
        % figure;
        % greenland; hold on
        % scatter(data.x, data.y, 3, 'filled')
        % set(gca,'XColor', 'none','YColor','none')
        
        figure;
        for i = 1:N
            subplot(1,2,1)
            plot(data{find(data.id==unique_ids(i)), 'distance'},...
                 data{find(data.id==unique_ids(i)), 'Velocity'})
            hold on
            
            subplot(1,2,2)
            scatter(data{find(data.id==unique_ids(i)), 'Thickness'},...
                    data{find(data.id==unique_ids(i)), 'Velocity'})
            hold on
        end
        
        
        %Collapse data set by mean values
        my_data = table();
        dist_cut = 20000; % 20 km
        % iterate over all ids
        for k = 1:N
            this_glc = data(data.id == unique_ids(k),:);
            ID = unique_ids(k);
            dist = dist_cut;
            vel_max = max(nonzeros(this_glc.Velocity));
            H_mean = mean(nonzeros(this_glc.Thickness));
            sslope_mean = mean(this_glc.('Surface Slope'));
            S_mean = mean(this_glc.('Surface Elevation'));
            glc_row = table(ID, dist, vel_max, H_mean, sslope_mean, S_mean);
            % append to the larger table
            my_data = [my_data; glc_row];
        end
        my_data.Properties.VariableNames = glc_row.Properties.VariableNames;
        
        %Read in Wood et al., 2021 data
        % suppress warning
        w_id = 'MATLAB:table:ModifiedAndSavedVarnames';
        warning('off',w_id);
        
        W_data_m = table();
        for i = 2:8
            W_data = rows2vars(readtable('Modified_data.xlsx', 'Sheet',i,'ReadRowNames',true));
            findempty = cellfun(@isempty, W_data{1,:});
            W_data(:, findempty) = [];
            W_data.Properties.VariableNames = W_data{1,:};
            W_data(1,:) = [];
            % remove the ones without legitemate ID
            noid = cell2mat(cellfun(@isnan, W_data.ID, 'UniformOutput', false));
            W_data = W_data(~noid, :);
            W_data_m = [W_data_m; W_data];
        end
        varnames = W_data.Properties.VariableNames;
        W_data_m.Properties.VariableNames = varnames;
        
        % Drop a few extraneous columns
        drop_cols = [1,3,4,6,7,14];
        W_data_m(:, drop_cols) = [];
        varnames = W_data_m.Properties.VariableNames;
        % convert all cells to doubles
        W_data_m = varfun(@cell2mat, W_data_m);
        W_data_m.Properties.VariableNames = varnames;
        % remove the repeating ids
        [~, ia] = unique(W_data_m.ID); 
        W_data_m = W_data_m(ia,:);
        
        
        % Read in Csatho et al., 2014 data
        C_data = readtable('Csatho2015_table_cleared.csv');
        C_data = C_data(:, [2, 9]);
        patterns = unique(C_data.(2));
        % convert to categorical and rename categoricals
        C_data.Type_EntireRecord = categorical(C_data.Type_EntireRecord);
        C_data.Type_EntireRecord = renamecats(C_data.Type_EntireRecord, patterns, string([1:length(patterns)]));
        
        % remove NaN glaciers (those have multiple branches)
        findnan = arrayfun(@isnan, C_data.ID);
        C_data(findnan, :) = [];
        
        % Join three tables
        % first join mine and Wood et al data
        id_overlap = intersect(my_data.ID, W_data_m.ID);
        [~, ~, ib] = intersect(id_overlap, W_data_m.ID);
        W_data_m = W_data_m(ib,:);
        all_data = join(W_data_m, my_data, 'Keys','ID');
        
        % then join Csatho et al data
        id_overlap2 = intersect(all_data.ID, C_data.ID);
        [~, ~, ib] = intersect(id_overlap2, all_data.ID);
        all_data = all_data(ib,:);
        all_data = join(all_data, C_data);
    
        % Export to a file
        writetable(all_data,'processed_table.csv')
        
    case 0
        % simply read in the processed data
        all_data = readtable('processed_table.csv');
        % read in the patterns
        C_data = readtable('Csatho2015_table_cleared.csv');
        C_data = C_data(:, [2, 9]);
        patterns = unique(C_data.(2));
        % convert to categorical and rename categoricals
        C_data.Type_EntireRecord = categorical(C_data.Type_EntireRecord);
        C_data.Type_EntireRecord = renamecats(C_data.Type_EntireRecord, patterns, string([1:length(patterns)]));
end

% PCA
% remove id and distance column
ori_data = all_data;
all_data(:, [1,9,14]) = [];
all_data_norm = normalize(table2array(all_data));

[coeff,score,latent,tsquared,explained,mu] = pca(all_data_norm);

% PCA Plots
% Make plots
N_PC = length(latent);
PCstr = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11"];
% Plots
figure('Position',[100,100,1800,800]);
subplot(1,3,1)
heatmap(PCstr, all_data.Properties.VariableNames, abs(coeff))

% make a biplot
subplot(1,3,2)
biplot(coeff(:,[1,2])); hold on;
vbls = all_data.Properties.VariableNames; % Labels for the variables
biplot(coeff(:,1:2),'Scores',score(:,1:2),'VarLabels',vbls);

% explained variance
subplot(1,3,3)
yyaxis left
plot(1:length(latent), latent,'-o'); hold on
ylabel('Variance','FontName','Times','FontSize',13)
yyaxis right
plot(1:length(latent), explained','-*'); hold off
xlabel('Principal components','FontName','Times','FontSize',13)
ylabel('Percent explained','FontName','Times','FontSize',13)
title('Explained variance','FontName','Times','FontSize',13)

print(gcf,'PCA.png','-dpng','-r300')

% PCA plots colored by surface elevation change characterization
% choose marker symbols
markers = {'^','v','o','_','+','*','p','>','d','x'};
figure;
for i = 1:length(patterns)
    thisgroup = score(ori_data.Type_EntireRecord==i,1:3);
    scatter3(thisgroup(:,1), thisgroup(:,2), thisgroup(:,3), 40, markers{i});
    hold on
end
legend(patterns)


%% Linear discriminant analysis
% we will use k-fold validation
% first, select the classes with over 10 observations
class_tally = tabulate(ori_data.Type_EntireRecord);
class_keep = class_tally(class_tally(:,2)>=10,1);
% trim the dataset
indx_keep = ismember(double(ori_data.Type_EntireRecord), class_keep);
data_supervis = all_data_norm(indx_keep,:);
labl_supervis = ori_data.Type_EntireRecord(indx_keep);

n = size(data_supervis,1);
cv = cvpartition(n,'HoldOut',0.40);
trainInds = training(cv);
sampleInds = test(cv);
trainingData = data_supervis(trainInds,:);
sampleData = data_supervis(sampleInds,:);
[class,err,POSTERIOR,logp,coeff] = classify(sampleData,trainingData, labl_supervis(trainInds));
%cm = confusionchart(labl_supervis(sampleInds),class);

%% K-fold cross-validation
cv2 = cvpartition(n,'KFold',8);
ldaCV = fitcdiscr(data_supervis, labl_supervis,'CVPartition',cv2);

% Dropping features sequentially to resolve feature importancee
% via the cross-validation loss
drop_n = size(data_supervis, 2);
losses = zeros(1,drop_n);
for i = 1:drop_n
    data_supervis_drop = data_supervis;
    data_supervis_drop(:,i) = [];
    cvn = cvpartition(n,'KFold',8);
    ldaCVdrop = fitcdiscr(data_supervis_drop, labl_supervis,'CVPartition',cvn);
    losses(i) = kfoldLoss(ldaCVdrop);
end

figure;
bar(losses)
xticklabels(vbls)
print(gcf,'featureimportance.png','-dpng','-r300')


%% scatter plot matrix

% then calculate the correlations and make heatmaps
% first, jsut the original data
rs = corrcoef(all_data_norm);

% apply log transform to all data
corrections = -1*min(all_data_norm,[],1)+0.01;
all_data_norm_logcorrect = log(repmat(corrections,size(all_data_norm,1),1)+all_data_norm);
rs_log = corrcoef(all_data_norm_logcorrect);

figure('Position',[100,100,1800,1400]);
subplot(1,2,1)
plotmatrix(all_data_norm)
subplot(1,2,2)
plotmatrix(all_data_norm_logcorrect)
print(gcf,'scattermatrix.png','-dpng','-r300')
% make heatmap as a illustration
figure('Position',[100,100,1800,1400]);
subplot(1,2,1)
h = heatmap(rs);
h.Colormap = jet;
caxis([-1,1])
subplot(1,2,2)
h = heatmap(rs_log);
h.Colormap = jet;
caxis([-1,1])
print(gcf,'twoheatmaps.png','-dpng','-r300')

%% Apply the same feature importance analysis
% but to the log-transformed dataset
logdata_supervis = all_data_norm_logcorrect(indx_keep,:);
labl_supervis = ori_data.Type_EntireRecord(indx_keep);

% Dropping features sequentially to resolve feature importancee
% via the cross-validation loss

drop_n = size(logdata_supervis, 2);
losseslog = zeros(1,drop_n);
for i = 1:drop_n
    data_supervis_drop = logdata_supervis;
    data_supervis_drop(:,i) = [];
    cvn = cvpartition(n,'KFold',8);
    ldaCVdrop = fitcdiscr(data_supervis_drop, labl_supervis,'CVPartition',cvn);
    losseslog(i) = kfoldLoss(ldaCVdrop);
end

figure;
bar(losseslog)
xticklabels(vbls)
print(gcf,'featureimportance_log.png','-dpng','-r300')


