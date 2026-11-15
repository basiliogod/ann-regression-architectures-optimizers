% Script principal de execução do estudo comparativo.
clear; clc; close all;

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