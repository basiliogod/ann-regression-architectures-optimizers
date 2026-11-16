% Script principal de execução do estudo comparativo.
clear; clc; close all;
% Compatibilidade gráfica para exportação headless no GNU Octave
if exist('OCTAVE_VERSION', 'builtin')
    available_tk = available_graphics_toolkits();
    if ismember('gnuplot', available_tk)
        graphics_toolkit('gnuplot');
    end
end

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
    fprintf('A iniciar Step 2: Análise Exploratória dos Dados (EDA)...\n');
    fig_dir = '../outputs';

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
    fprintf('A iniciar Step 3: Feature Engineering...\n');

    % 3.1 Concrete: adiciona Water/Cement ratio e log(Age) -> d = 10 inputs
    [X_conc_fe, feat_names_conc] = L.fe_concrete(X_conc, names_conc);
    fprintf('Concrete: %d features preparadas (%s)\n', size(X_conc_fe, 2), strjoin(feat_names_conc, ', '));

    % 3.2 Melbourne: janela p = 7 dias + sin/cos(doy) -> d = 9 inputs
    p_window = 7;
    use_cal  = true;
    [X_melb_fe, y_melb_fe, dates_melb_fe, feat_names_melb] = L.fe_melbourne(dates_melb, temps_melb, p_window, use_cal);
    fprintf('Melbourne: %d amostras criadas com janela p=%d e %d features (%s)\n\n', size(X_melb_fe, 1), p_window, size(X_melb_fe, 2), strjoin(feat_names_melb, ', '));