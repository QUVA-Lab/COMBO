function model_kl = KL_decoupled_models(models)
% KL_decoupled_models: Function evaluates the KL divergence objective
% for all samples decoupled models from the aerostructural model

% If models is a table convert to an array
if strcmp(class(models),'table')
    models = table2array(models);
end

% Determine the number of models
[n_models, ~] = size(models);

% Declare vector to store model_kl
model_kl = zeros(n_models,1);

for i=1:n_models

	% Save model
	model_vect  = models(i,:);
	model_loads = model_vect(1:7);
	model_mesh  = model_vect(8:end);
	save('dec_models.mat','model_loads','model_mesh');

	% Evaluate KL Divergence for each model based on Gaussian linearization
	command_str = 'python aerostruct_lin_model.py';
	[status, err] = system(command_str);

	% Check that run was successful
	if (status~=0)
		disp(err)
		error('Function call did not complete!');
	end

	% Load results
	data = load('results.mat');
	model_idx  = ismember(data.models, models(i,:), 'rows');
	model_kl(i) = data.kl_vect(find(model_idx));
	
end
