% Script principal de execução do estudo comparativo.
clear; clc;

% Compatibilidade gráfica para exportação headless no GNU Octave 
if exist('OCTAVE_VERSION', 'builtin')
    warning('off', 'all');
    if (isempty(getenv('DISPLAY')) || ~isguirunning()) && ismember('gnuplot', available_graphics_toolkits())
        graphics_toolkit('gnuplot');
        set(0, 'DefaultFigure__graphics_toolkit__', 'gnuplot');
    end
end

close all;

% Obtém a diretoria absoluta onde este script está guardado (pasta src/)
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir); % Garante que ann_lib.m é sempre encontrado

% Garante reprodutibilidade global desde o início (compatível MATLAB/Octave)
seed_global = 42;
rand('state', seed_global);
randn('state', seed_global);

% Carrega a biblioteca de funções
L = ann_lib();

% Caminhos independentes da diretoria de execução
path_concrete  = fullfile(script_dir, '..', 'data', 'Concrete_Data.csv');
path_melbourne = fullfile(script_dir, '..', 'data', 'daily-minimum-temperatures-in-me.csv');

% STEP 1: Aquisição de Dados e Inspeção Inicial
    fprintf('A iniciar Step 1: Aquisição de dados e inspeção inicial...\n');

    % 1.1 Dataset Concrete
    [X_conc, y_conc, names_conc] = L.load_concrete(path_concrete);
    stats_conc = L.inspect_data([X_conc, y_conc], names_conc, 'Concrete Compressive Strength');

    % 1.2 Dataset Melbourne
    [dates_melb, temps_melb, info_melb] = L.load_melbourne(path_melbourne);
    fprintf('--- Metadados Melbourne ---\n');
    fprintf('Cobertura temporal: %s a %s (%d registos)\n', info_melb.start_date, info_melb.end_date, info_melb.total_rows);
    fprintf('Valores com "?" limpos na leitura: %d | Valores NaN restantes: %d\n', info_melb.cleaned_marks, info_melb.nan_count);
    stats_melb = L.inspect_data(temps_melb, {'Daily Min Temp (C)'}, 'Melbourne Temperatures');

% STEP 2: Análise Exploratória dos Dados (EDA)
    fprintf('\nA iniciar Step 2: Análise Exploratória dos Dados (EDA)...\n');
    fig_dir     = fullfile(script_dir, '..', 'outputs');
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

    % 2.1 EDA Concrete
    corr_conc = L.eda_concrete(X_conc, y_conc, names_conc, fig_dir);
    fprintf('Correlações com Compressive Strength (MPa):\n');
    for j = 1:8
        fprintf('  %-28s : r = %+.3f\n', names_conc{j}, corr_conc(j, 9));
    end

    % 2.2 EDA Melbourne
    eda_melb = L.eda_melbourne(dates_melb, temps_melb, fig_dir);
    fprintf('\n--- Verificação de Anomalias e Padrões (Melbourne) ---\n');
    fprintf('Registos exatamente iguais a 0.0 C : %d\n', eda_melb.zero_count);
    fprintf('Salto diário máximo               : %.2f C (%s)\n', eda_melb.max_jump, eda_melb.max_jump_date);
    fprintf('Saltos diários > 10 C             : %d\n', eda_melb.jumps_over_10);
    fprintf('Correlação lag-1 T(t) vs T(t+1)   : %.4f\n', eda_melb.r_lag1);

% STEP 3: Pré-processamento e Feature Engineering
    fprintf('\nA iniciar Step 3: Feature Engineering...\n');

    % 3.1 Concrete: adiciona Water/Cement ratio e log(Age) -> d = 10 inputs
    [X_conc_fe, feat_names_conc] = L.fe_concrete(X_conc, names_conc);
    fprintf('Concrete: %d features preparadas (%s)\n', ...
            size(X_conc_fe, 2), strjoin(feat_names_conc, ', '));

    % 3.2 Melbourne: janela p = 7 dias + sin/cos(doy) -> d = 9 inputs
    p_window = 7;
    use_cal  = true;
    [X_melb_fe, y_melb_fe, dates_melb_fe, feat_names_melb] = ...
        L.fe_melbourne(dates_melb, temps_melb, p_window, use_cal);
    fprintf('Melbourne: %d amostras criadas com janela p=%d e %d features (%s)\n\n', ...
            size(X_melb_fe, 1), p_window, size(X_melb_fe, 2), strjoin(feat_names_melb, ', '));


% STEP 4: Particionamento dos Dados (70/15/15)
    fprintf('\nA iniciar Step 4: Particionamento dos dados (70/15/15)...\n');
    ratios = [0.70, 0.15, 0.15];

    % 4.1 Concrete: partição aleatória fixa
    conc_data = L.split_data(X_conc_fe, y_conc, ratios, 'random', seed_global);
    mix_info  = L.analyze_concrete_mixtures(X_conc, conc_data.idx_tr, conc_data.idx_va, conc_data.idx_te);
    fprintf('Concrete Split (Aleatório) -> Treino: %d | Validação: %d | Teste: %d\n', length(conc_data.y_tr), length(conc_data.y_va), length(conc_data.y_te));
    fprintf('  Misturas únicas (7 ingredientes): %d em %d observações\n', mix_info.unique_mixtures, mix_info.total_obs);
    fprintf('  Misturas do teste também presentes no treino (outra idade): %d (%.1f%%)\n', mix_info.overlap_tr_te, mix_info.pct_te_in_tr);

    % 4.2 Melbourne: partição cronológica fixa (sem leakage temporal)
    melb_data = L.split_data(X_melb_fe, y_melb_fe, ratios, 'chronological', seed_global);
    fprintf('Melbourne Split (Cronológico) -> Treino: %d | Validação: %d | Teste: %d\n', length(melb_data.y_tr), length(melb_data.y_va), length(melb_data.y_te));
    fprintf('  Treino   : %s a %s\n', dates_melb_fe{melb_data.idx_tr(1)}, dates_melb_fe{melb_data.idx_tr(end)});
    fprintf('  Validação: %s a %s\n', dates_melb_fe{melb_data.idx_va(1)}, dates_melb_fe{melb_data.idx_va(end)});
    fprintf('  Teste    : %s a %s\n\n', dates_melb_fe{melb_data.idx_te(1)}, dates_melb_fe{melb_data.idx_te(end)});

% STEP 5: Verificação da Implementação (Gradiente Analítico vs Numérico)
    fprintf('A iniciar Step 5: Verificação de gradientes por diferenças finitas...\n');

    arch_ids = {'A', 'B', 'C'};
    X_check = conc_data.X_tr(1:20, :);
    y_check = conc_data.y_tr(1:20);
    d_conc  = size(X_check, 2);

    for a = 1:length(arch_ids)
        layers_tmp = L.build_arch(d_conc, arch_ids{a});
        n_params   = length(L.init_params(layers_tmp, seed_global));
        rel_err    = L.check_gradient(layers_tmp, X_check, y_check, seed_global);
        fprintf('  Arquitetura %s (%4d parâmetros) -> Erro relativo máximo: %.3e\n', arch_ids{a}, n_params, rel_err);
    end
    fprintf('\n');
 
% STEP 6: Ajuste de Hiperparâmetros (Hyperparameter Tuning)
    fprintf('A iniciar Step 6: Tuning de Hiperparâmetros na validação...\n');
    results_dir = fullfile(script_dir, '..', 'results');
    if ~exist(results_dir, 'dir'), mkdir(results_dir); end

    lr_grid     = logspace(-4, -1, 6); % 6 valores: 1e-4, 3.98e-4, 1.58e-3, 6.31e-3, 2.51e-2, 1e-1
    n_tune_reps = 5;                   % 5 repetições por valor de learning rate
    tune_epochs = 60;                  % Orçamento de épocas para avaliação na validação
    batch_size  = 32;                  % Tamanho fixo do mini-batch (GD e Adam)

    % Parâmetros fixos do BFGS (declarados explicitamente para o Step 7)
    bfgs_opts.gtol         = 1e-6;     % Tolerância na norma infinito do gradiente
    bfgs_opts.stol         = 1e-8;     % Tolerância na norma do passo (paragem por estagnação)
    bfgs_opts.ftol         = 1e-9;     % Tolerância na variação da perda (paragem por estagnação)
    bfgs_opts.wolfe_c1     = 1e-4;     % Constante condição de Armijo
    bfgs_opts.wolfe_c2     = 0.9;      % Constante condição de curvatura de Wolfe
    bfgs_opts.max_ls_evals = 15;       % Máximo de avaliações por line search

    tune_conc = L.tune_learning_rates(conc_data, 'Concrete', lr_grid, n_tune_reps, tune_epochs, batch_size, fig_dir);
    tune_melb = L.tune_learning_rates(melb_data, 'Melbourne', lr_grid, n_tune_reps, tune_epochs, batch_size, fig_dir);    

    % Guarda os resultados do tuning (Step 6) para consulta e reutilização futura
    path_tuning_mat = fullfile(results_dir, 'tuning_results.mat');
    save(path_tuning_mat, 'tune_conc', 'tune_melb', 'lr_grid', 'n_tune_reps', 'tune_epochs', 'batch_size');
    fprintf('Resultados do Step 6 guardados com sucesso em %s\n\n', path_tuning_mat);

% STEP 7: Experiências Finais (18 combinações x 30 runs emparelhadas)
    fprintf('A iniciar Step 7: Experiências Finais...\n');

    n_final_runs      = 30;  % Seeds 1 a 30
    epochs_concrete   = 250; % Orçamento em epoch-equivalents para Concrete
    epochs_melbourne  = 200; % Orçamento em epoch-equivalents para Melbourne

    res_conc = L.run_final_experiments(conc_data, 'Concrete',  tune_conc, bfgs_opts, epochs_concrete,  batch_size, n_final_runs);
    path_conc_mat = fullfile(results_dir, 'final_results_concrete.mat');
    save(path_conc_mat, 'res_conc', 'tune_conc');
    fprintf('Resultados de Concrete guardados em %s\n\n', path_conc_mat);

    res_melb = L.run_final_experiments(melb_data, 'Melbourne', tune_melb, bfgs_opts, epochs_melbourne, batch_size, n_final_runs);
    path_melb_mat = fullfile(results_dir, 'final_results_melbourne.mat');
    save(path_melb_mat, 'res_melb', 'tune_melb');
    fprintf('Resultados de Melbourne guardados em %s\n\n', path_melb_mat);

    path_all_mat = fullfile(results_dir, 'final_results_all.mat');
    save(path_all_mat, 'res_conc', 'res_melb', 'tune_conc', 'tune_melb');
    fprintf('Todos os resultados guardados com sucesso em %s\n', path_all_mat);

% STEP 8: Resultados Finais, Modelos de Referência e Visualizações
    fprintf('\nA iniciar Step 8: Geração de tabelas, baselines e gráficos finais...\n');

    % 8.1 Modelos de Referência (Regressão Linear OLS, Persistência e AR(p))
    base_res = L.eval_baselines(conc_data, melb_data, p_window);

    % 8.2 Tabelas Resumo 3x3 (Arquitetura x Algoritmo)
    L.print_summary_table(res_conc, 'MPa');
    L.print_summary_table(res_melb, '°C');

    % 8.3 Figuras de Progresso de Treino (Mediana + IQR em escala logarítmica)
    L.plot_training_progress(res_conc, fig_dir);
    L.plot_training_progress(res_melb, fig_dir);

    % 8.4 Box Plots dos Erros Finais no Teste (implementados sem toolbox)
    L.plot_custom_boxplots(res_conc, 'MPa', fig_dir);
    L.plot_custom_boxplots(res_melb, '°C',  fig_dir);

    % 8.5 Gráficos de Predição vs Observado e Histogramas de Resíduos
    dates_melb_te = dates_melb_fe(melb_data.idx_te);
    L.plot_predictions(res_conc, conc_data, {}, fig_dir);
    L.plot_predictions(res_melb, melb_data, dates_melb_te, fig_dir);

    fprintf('\nEstudo comparativo concluído com sucesso! Resultados guardados em %s e figuras em %s\n', results_dir, fig_dir);