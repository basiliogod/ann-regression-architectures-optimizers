% ANN_LIB Biblioteca de funções para o Assignment 2 de Redes Neuronais Artificiais.
function L = ann_lib()
    % Aquisição e inspeção inicial de dados
    L.load_concrete  = @load_concrete;
    L.load_melbourne = @load_melbourne;
    L.inspect_data   = @inspect_data;

    L.compute_acf    = @compute_acf;
    L.eda_concrete   = @eda_concrete;
    L.eda_melbourne  = @eda_melbourne;

    L.fe_concrete         = @fe_concrete;
    L.fe_melbourne        = @fe_melbourne;
    L.fit_scaler          = @fit_scaler;
    L.transform_data      = @transform_data;
    L.inverse_transform_y = @inverse_transform_y;

    L.split_data               = @split_data;
    L.analyze_concrete_mixtures = @analyze_concrete_mixtures;

    L.build_arch      = @build_arch;
    L.init_params     = @init_params;
    L.unpack_params   = @unpack_params;
    L.forward_pass    = @forward_pass;
    L.loss_grad       = @loss_grad;
    L.check_gradient  = @check_gradient;
    L.train_gd        = @train_gd;
    L.train_bfgs      = @train_bfgs;
    L.train_adam      = @train_adam;

    L.tune_learning_rates = @tune_learning_rates;

    L.compute_metrics       = @compute_metrics;
    L.run_final_experiments = @run_final_experiments;

    L.eval_baselines          = @eval_baselines;
    L.print_summary_table     = @print_summary_table;
    L.plot_training_progress  = @plot_training_progress;
    L.plot_custom_boxplots    = @plot_custom_boxplots;
    L.plot_predictions        = @plot_predictions;
end

function h = create_headless_fig(pos)
    if exist('OCTAVE_VERSION', 'builtin')
        warning('off', 'all');
        avail = available_graphics_toolkits();
        if ~ismember('qt', avail) && ismember('gnuplot', avail)
            graphics_toolkit('gnuplot');
        end
    end
    h = figure('Visible', 'off', 'Position', pos);
    if exist('OCTAVE_VERSION', 'builtin') && strcmp(graphics_toolkit(), 'fltk')
        close(h);
        graphics_toolkit('gnuplot');
        h = figure('Visible', 'off', 'Position', pos);
    end
end

function [X, y, var_names] = load_concrete(filepath)
    % LOAD_CONCRETE : Carrega o dataset Concrete Compressive Strength a partir do csv.
    %   Entradas:
    %       filepath  - Caminho para o ficheiro Concrete_Data.csv
    %   Saídas:
    %       X         - Matriz com as variáveis de entrada
    %       y         - Vetor coluna com a resistência à compressão
    %       var_names - Cell array com os nomes e unidades de todas as variáveis
    var_names = {
        'Cement (kg/m3)', 'Blast Furnace Slag (kg/m3)', 'Fly Ash (kg/m3)', 'Water (kg/m3)', 'Superplasticizer (kg/m3)', 'Coarse Aggregate (kg/m3)', 'Fine Aggregate (kg/m3)', 'Age (days)', 'Compressive Strength (MPa)'
    };

    fid = fopen(filepath, 'r');
    if fid == -1
        error('Erro ao abrir o ficheiro Concrete: %s', filepath);
    end

    % Lê a primeira linha para verificar se existe cabeçalho textual
    first_line = fgetl(fid);
    fclose(fid);

    has_header = ~isempty(regexp(first_line, '^[a-zA-Z"]', 'once'));
    start_row = double(has_header);

    % Lê os dados numéricos
    data = dlmread(filepath, ',', start_row, 0);

    data = data(any(data ~= 0, 2), :);

    X = data(:, 1:8);
    y = data(:, 9);
end

function [dates, temps, info] = load_melbourne(filepath)
    % LOAD_MELBOURNE : Carrega a série temporal de temperaturas mínimas de Melbourne.
    %   Limpa aspas e caracteres anómalos durante a conversão numérica.
    %   Entradas:
    %       filepath - Caminho para o ficheiro daily-minimum-temperatures-in-me.csv
    %   Saídas:
    %       dates    - Cell array com as datas no formato 'AAAA-MM-DD'
    %       temps    - Vetor coluna com as temperaturas em graus Celsius
    %       info     - Estrutura com metadados da limpeza (ex: registos limpos/NaN)

    fid = fopen(filepath, 'r');
    if fid == -1
        error('Erro ao abrir o ficheiro Melbourne: %s', filepath);
    end

    % Lê todas as linhas como strings
    raw = textscan(fid, '%s %s', 'Delimiter', ',', 'HeaderLines', 1);
    fclose(fid);

    raw_dates = raw{1};
    raw_temps = raw{2};
    n = length(raw_dates);

    dates = cell(n, 1);
    temps = nan(n, 1);
    cleaned_count = 0;

    for i = 1:n
        % Remove aspas duplas ou espaços em branco das datas
        d_str = strrep(raw_dates{i}, '"', '');
        dates{i} = strtrim(d_str);

        % Limpa aspas e possíveis pontos de interrogação nas temperaturas
        t_str = strrep(raw_temps{i}, '"', '');
        t_str = strtrim(t_str);
        
        if ~isempty(strfind(t_str, '?'))
            t_str = strrep(t_str, '?', '');
            cleaned_count = cleaned_count + 1;
        end

        val = str2double(t_str);
        temps(i) = val;
    end

    valid_idx = ~cellfun('isempty', dates);
    dates = dates(valid_idx);
    temps = temps(valid_idx);

    info.total_rows    = length(temps);
    info.cleaned_marks = cleaned_count;
    info.nan_count     = sum(isnan(temps));
    info.start_date    = dates{1};
    info.end_date      = dates{end};
end

function stats = inspect_data(data_matrix, var_names, dataset_title)
    % INSPECT_DATA : Calcula e imprime na consola o resumo estatístico das variáveis.
    %   Entradas:
    %       data_matrix   - Matriz contendo todas as variáveis numéricas
    %       var_names     - Cell array com os nomes das colunas
    %       dataset_title - String com o nome do dataset para exibição
    %   Saídas:
    %       stats         - Matriz com [Min, Mediana, Média, DesvioPadrão, Max, NaNs]

    [n_samples, n_vars] = size(data_matrix);
    stats = zeros(n_vars, 6);

    fprintf('\n================================================================================\n');
    fprintf(' INSPEÇÃO INICIAL: %s (N = %d observações, %d variáveis)\n', dataset_title, n_samples, n_vars);
    fprintf('================================================================================\n');
    fprintf('%-30s | %8s | %8s | %8s | %8s | %8s | %4s\n', 'Variável', 'Mín', 'Mediana', 'Média', 'Std', 'Máx', 'NaNs');
    fprintf('--------------------------------------------------------------------------------\n');

    for j = 1:n_vars
        col = data_matrix(:, j);
        valid_col = col(~isnan(col));
        
        c_min  = min(valid_col);
        c_med  = median(valid_col);
        c_mean = mean(valid_col);
        c_std  = std(valid_col);
        c_max  = max(valid_col);
        c_nan  = sum(isnan(col));

        stats(j, :) = [c_min, c_med, c_mean, c_std, c_max, c_nan];

        fprintf('%-30s | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %4d\n', var_names{j}, c_min, c_med, c_mean, c_std, c_max, c_nan);
    end
    fprintf('================================================================================\n\n');
end

function [acf_vals, lags] = compute_acf(y, max_lag)
    % COMPUTE_ACF : Calcula a função de autocorrelação amostral.
    %   Entradas:
    %       y        - Vetor coluna da série temporal
    %       max_lag  - Desfasamento máximo (inteiro) para o qual calcular a ACF
    %   Saídas:
    %       acf_vals - Vetor  com autocorrelações de lag 0 a max_lag
    %       lags     - Vetor de 0 a max_lag

    y = y(:);
    N = length(y);
    y_mean = mean(y);
    y_centered = y - y_mean;
    denom = sum(y_centered .^ 2);

    lags = (0:max_lag)';
    acf_vals = zeros(max_lag + 1, 1);

    for k = 0:max_lag
        num = sum(y_centered(1:N-k) .* y_centered(1+k:N));
        acf_vals(k + 1) = num / denom;
    end
end

function corr_mat = eda_concrete(X, y, var_names, fig_dir)
    % EDA_CONCRETE : Gera e guarda os gráficos exploratórios do dataset Concrete.
    %   1. Histogramas de todas as 9 variáveis (entradas + alvo)
    %   2. Matriz de correlação de Pearson
    %   3. Scatter plots das 8 entradas contra a variável alvo
    %  Entradas:
    %       X         - Matriz com as variáveis de entrada
    %       y         - Vetor coluna com a variável alvo (Compressive Strength)
    %       var_names - Cell array com os nomes das variáveis
    %       fig_dir   - Diretoria onde os gráficos serão guardados
    %  Saídas:
    %       corr_mat  - Matriz de correlação de Pearson (9x9)

    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    full_data = [X, y];
    short_names = {'Cement', 'Slag', 'FlyAsh', 'Water', 'Superplast', 'CoarseAgg', 'FineAgg', 'Age', 'Strength'};

    % 1. Histogramas de todas as variáveis
    h1 = create_headless_fig([100, 100, 1100, 850]);
    for j = 1:9
        subplot(3, 3, j);
        hist(full_data(:, j), 20);
        h_patch = findobj(gca, 'Type', 'patch');
        set(h_patch, 'FaceColor', [0.25 0.45 0.65], 'EdgeColor', 'w');
        title(var_names{j}, 'FontSize', 9, 'FontWeight', 'bold');
        ylabel('Frequência', 'FontSize', 8);
        grid on;
    end
    print(h1, fullfile(fig_dir, 'eda_concrete_histograms.png'), '-dpng', '-r300');
    close(h1);

    % 2. Matriz de Correlação
    corr_mat = corrcoef(full_data);
    h2 = create_headless_fig([100, 100, 800, 680]);
    imagesc(corr_mat, [-1, 1]);
    colormap('jet');
    colorbar;
    set(gca, 'XTick', 1:9, 'XTickLabel', short_names, 'FontSize', 8);
    set(gca, 'YTick', 1:9, 'YTickLabel', short_names, 'FontSize', 8);
    if exist('xtickangle', 'file') || exist('xtickangle', 'builtin')
        xtickangle(45);
    end
    title('Matriz de Correlação - Concrete', 'FontSize', 11, 'FontWeight', 'bold');

    % Anota os valores numéricos em cada célula
    for r = 1:9
        for c = 1:9
            val = corr_mat(r, c);
            txt_col = 'k';
            if abs(val) > 0.65
                txt_col = 'w';
            end
            text(c, r, sprintf('%.2f', val), 'HorizontalAlignment', 'center', 'Color', txt_col, 'FontSize', 8, 'FontWeight', 'bold');
        end
    end
    print(h2, fullfile(fig_dir, 'eda_concrete_corr.png'), '-dpng', '-r300');
    close(h2);

    % 3. Scatter plots das 8 entradas contra o alvo 
    h3 = create_headless_fig([100, 100, 1200, 580]);
    for j = 1:8
        subplot(2, 4, j);
        scatter(X(:, j), y, 12, [0.2 0.4 0.7], 'filled');
        xlabel(var_names{j}, 'FontSize', 8);
        if j == 1 || j == 5
            ylabel('Strength (MPa)', 'FontSize', 8);
        end
        title(sprintf('r = %.2f', corr_mat(j, 9)), 'FontSize', 9);
        grid on;
    end
    print(h3, fullfile(fig_dir, 'eda_concrete_scatter.png'), '-dpng', '-r300');
    close(h3);
end

function eda_info = eda_melbourne(dates, temps, fig_dir)
    % EDA_MELBOURNE : Gera gráficos exploratórios e verifica anomalias em Melbourne.
    %   1. Verificação de saltos abruptos/valores suspeitos
    %   2. Série temporal completa 
    %   3. Distribuição de valores (histograma)
    %   4. Comportamento sazonal mensal
    %   5. Ciclo anual médio
    %   6. Função de autocorrelação
    %   7. Scatter plot T(t) vs T(t+1)
    %   Entradas:
    %       dates    - Cell array com as datas no formato 'AAAA-MM-DD'
    %       temps    - Vetor coluna com as temperaturas em graus Celsius
    %       fig_dir  - Diretoria onde os gráficos serão guardados
    %   Saídas:
    %       eda_info - Estrutura com informações sobre anomalias e estatísticas

    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    N = length(temps);
    months = zeros(N, 1);
    days_of_year = zeros(N, 1);
    days_in_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    cum_days = [0, cumsum(days_in_month(1:11))];

    for i = 1:N
        m = str2double(dates{i}(6:7));
        d = str2double(dates{i}(9:10));
        months(i) = m;
        doy = cum_days(m) + d;
        if doy > 365
            doy = 365;
        end
        days_of_year(i) = doy;
    end

    % Verificação de valores suspeitos (ex: 0.0C) e saltos abruptos |T(t+1)-T(t)|
    diffs = abs(diff(temps));
    [max_jump, jump_idx] = max(diffs);
    zero_count = sum(temps == 0);
    large_jumps = sum(diffs > 10);

    eda_info.zero_count    = zero_count;
    eda_info.max_jump      = max_jump;
    eda_info.max_jump_date = sprintf('%s -> %s', dates{jump_idx}, dates{jump_idx+1});
    eda_info.jumps_over_10 = large_jumps;
    eda_info.r_lag1        = corrcoef(temps(1:end-1), temps(2:end))(1, 2);

    % Figura 1: Série completa + Distribuição + Comportamento Mensal + Ciclo Anual
    h1 = create_headless_fig([100, 100, 1200, 800]);

    % 1a. Série completa
    subplot(2, 2, 1);
    plot(1:N, temps, 'Color', [0.15 0.45 0.70], 'LineWidth', 0.6);
    xlim([1, N]);
    set(gca, 'XTick', 1:365:N, 'XTickLabel', 1981:1990);
    xlabel('Ano'); ylabel('Temp. Mínima (°C)');
    title('Série Temporal Completa (1981-1990)', 'FontWeight', 'bold');
    grid on;

    % 1b. Distribuição dos valores
    subplot(2, 2, 2);
    hist(temps, 25);
    h_patch = findobj(gca, 'Type', 'patch');
    set(h_patch, 'FaceColor', [0.25 0.55 0.45], 'EdgeColor', 'w');
    xlabel('Temperatura Mínima (°C)'); ylabel('Frequência');
    title('Distribuição das Temperaturas', 'FontWeight', 'bold');
    grid on;

    % 1c. Comportamento sazonal mensal
    subplot(2, 2, 3);
    m_mean = zeros(12, 1); m_std = zeros(12, 1);
    for m = 1:12
        vals = temps(months == m);
        m_mean(m) = mean(vals);
        m_std(m)  = std(vals);
    end
    h_err = errorbar(1:12, m_mean, m_std);
    set(h_err, 'Marker', 'o', 'LineWidth', 1.5, 'Color', [0.8 0.3 0.2]);
    xlim([0.5, 12.5]);
    set(gca, 'XTick', 1:12, 'XTickLabel', {'Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'});
    xlabel('Mês'); ylabel('Temp. Mínima (°C)');
    title('Comportamento Sazonal Mensal (Média ± Std)', 'FontWeight', 'bold');
    grid on;

    % 1d. Ciclo anual médio (365 dias)
    subplot(2, 2, 4);
    doy_mean = zeros(365, 1);
    for d = 1:365
        doy_mean(d) = mean(temps(days_of_year == d));
    end
    plot(1:365, doy_mean, 'Color', [0.55 0.25 0.65], 'LineWidth', 1.2);
    xlim([1, 365]);
    xlabel('Dia do Ano'); ylabel('Temp. Média (°C)');
    title('Ciclo Anual Médio', 'FontWeight', 'bold');
    grid on;

    print(h1, fullfile(fig_dir, 'eda_melbourne_series.png'), '-dpng', '-r300');
    close(h1);

    % Figura 2: ACF + Scatter T(t) vs T(t+1) + Saltos diários
    h2 = create_headless_fig([100, 100, 1200, 420]);

    % 2a. Função de Autocorrelação
    subplot(1, 3, 1);
    [acf_vals, lags] = compute_acf(temps, 400);
    plot(lags, acf_vals, 'Color', [0.15 0.45 0.70], 'LineWidth', 1.2);
    hold on;
    conf_bound = 1.96 / sqrt(N);
    plot([0, 400], [conf_bound, conf_bound], '--r');
    plot([0, 400], [-conf_bound, -conf_bound], '--r');
    plot([0, 400], [0, 0], '-k');
    hold off;
    xlim([0, 400]);
    xlabel('Desfasamento (Lag em dias)'); ylabel('Autocorrelação');
    title('Função de Autocorrelação (ACF)', 'FontWeight', 'bold');
    grid on;

    % 2b. Scatter T(t) vs T(t+1)
    subplot(1, 3, 2);
    scatter(temps(1:end-1), temps(2:end), 10, [0.2 0.4 0.7], 'filled');
    xlabel('T(t) (°C)'); ylabel('T(t+1) (°C)');
    title(sprintf('T(t) vs T(t+1) (r = %.3f)', eda_info.r_lag1), 'FontWeight', 'bold');
    grid on;

    % 2c. Variação diária absoluta |T(t+1) - T(t)| para deteção de saltos abruptos
    subplot(1, 3, 3);
    plot(1:N-1, diffs, 'Color', [0.75 0.35 0.15], 'LineWidth', 0.6);
    xlim([1, N-1]);
    xlabel('Dia (índice)'); ylabel('|T(t+1) - T(t)| (°C)');
    title(sprintf('Saltos Diários (Máx: %.1f°C)', max_jump), 'FontWeight', 'bold');
    grid on;

    print(h2, fullfile(fig_dir, 'eda_melbourne_acf_scatter.png'), '-dpng', '-r300');
    close(h2);
end

function [X_feat, feat_names] = fe_concrete(X, var_names)
    % FE_CONCRETE Aplica engenharia de atributos ao dataset Concrete.
    %   Adiciona:
    %     1. Rácio Água/Cimento (Water / Cement)
    %     2. Transformação logarítmica da idade: log(Age)
    %   Entradas:
    %       X          - Matriz original (1030 x 8)
    %       var_names  - Nomes das variáveis originais
    %   Saídas:
    %       X_feat     - Matriz transformada (1030 x 10)
    %       feat_names - Cell array com os 10 nomes das variáveis de entrada

    % Coluna 1: Cement | Coluna 4: Water | Coluna 8: Age
    water_cement_ratio = X(:, 4) ./ X(:, 1);
    log_age            = log(X(:, 8));

    X_feat = [X, water_cement_ratio, log_age];

    feat_names = [var_names(1:8), {'Water/Cement Ratio', 'log(Age)'}];
end

function [X_win, y_win, dates_win, feat_names] = fe_melbourne(dates, temps, p, use_calendar)
    % FE_MELBOURNE Constrói as janelas temporais e atributos de calendário.
    %   Para cada instante t (de p até N-1), usa [T(t), T(t-1), ..., T(t-p+1)]
    %   e opcionalmente sin/cos do dia do ano de t para prever T(t+1).
    %   Entradas:
    %       dates        - Cell array (N x 1) com datas 'AAAA-MM-DD'
    %       temps        - Vetor coluna (N x 1) com temperaturas mínimas
    %       p            - Tamanho da janela autoregressiva (ex: p = 7)
    %       use_calendar - Booleano (true para incluir sin/cos do dia do ano)
    %   Saídas:
    %       X_win        - Matriz de entradas ((N-p) x d)
    %       y_win        - Vetor alvo T(t+1) ((N-p) x 1)
    %       dates_win    - Datas correspondentes ao alvo T(t+1)
    %       feat_names   - Cell array com os nomes das entradas

    N = length(temps);
    num_samples = N - p;

    % Calcula o dia do ano (1 a 365) para cada registo
    days_in_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    cum_days = [0, cumsum(days_in_month(1:11))];
    doy = zeros(N, 1);

    for i = 1:N
        m = str2double(dates{i}(6:7));
        d = str2double(dates{i}(9:10));
        val_doy = cum_days(m) + d;
        if val_doy > 365
            val_doy = 365;
        end
        doy(i) = val_doy;
    end

    if use_calendar
        d = p + 2;
    else
        d = p;
    end

    X_win = zeros(num_samples, d);
    y_win = zeros(num_samples, 1);
    dates_win = cell(num_samples, 1);

    % Constrói cada amostra sem qualquer fuga de informação futura (leakage)
    for idx = 1:num_samples
        t = p + idx - 1; % Índice atual t

        % Janela dos p valores mais recentes: T(t), T(t-1), ..., T(t-p+1)
        for lag = 0:(p - 1)
            X_win(idx, lag + 1) = temps(t - lag);
        end

        % Atributos cíclicos do dia do ano conhecidos no instante t
        if use_calendar
            angle = 2 * pi * doy(t) / 365.0;
            X_win(idx, p + 1) = sin(angle);
            X_win(idx, p + 2) = cos(angle);
        end

        % Alvo: temperatura no dia seguinte T(t+1)
        y_win(idx) = temps(t + 1);
        dates_win{idx} = dates{t + 1};
    end

    % Nomes das features
    feat_names = cell(1, d);
    for lag = 0:(p - 1)
        if lag == 0
            feat_names{lag + 1} = 'T(t)';
        else
            feat_names{lag + 1} = sprintf('T(t-%d)', lag);
        end
    end
    if use_calendar
        feat_names{p + 1} = 'sin(2*pi*doy/365)';
        feat_names{p + 2} = 'cos(2*pi*doy/365)';
    end
end

function scaler = fit_scaler(X_tr, y_tr)
    % FIT_SCALER Calcula estatísticas de normalização (Z-score) APENAS no treino.
    %   Entradas:
    %       X_tr   - Matriz de entradas de treino (N_tr x d)
    %       y_tr   - Vetor alvo de treino (N_tr x 1)
    %   Saídas:
    %       scaler - Estrutura com médias e desvios padrão de X e y

    scaler.x_mean = mean(X_tr, 1);
    scaler.x_std  = std(X_tr, 0, 1);
    % Proteção contra divisão por zero caso alguma coluna seja constante
    scaler.x_std(scaler.x_std == 0) = 1;

    scaler.y_mean = mean(y_tr);
    scaler.y_std  = std(y_tr);
    if scaler.y_std == 0
        scaler.y_std = 1;
    end
end

function [X_scaled, y_scaled] = transform_data(X, y, scaler)
    % TRANSFORM_DATA Aplica a normalização Z-score usando as estatísticas do treino.
    %  Entradas:
    %      X      - Matriz de entradas
    %      y      - Vetor alvo
    %      scaler - Estrutura com médias e desvios padrão de treino
    %  Saídas:
    %      X_scaled - Matriz de entradas normalizada
    %      y_scaled - Vetor alvo normalizado
    X_scaled = (X - repmat(scaler.x_mean, size(X, 1), 1)) ./ repmat(scaler.x_std, size(X, 1), 1);
    y_scaled = (y - scaler.y_mean) ./ scaler.y_std;
end

function y_orig = inverse_transform_y(y_scaled, scaler)
    % INVERSE_TRANSFORM_Y Reverte a escala do alvo para as unidades originais.
    %  Entradas:
    %      y_scaled - Vetor alvo normalizado
    %      scaler - Estrutura com médias e desvios padrão de treino
    %  Saídas:
    %      y_orig - Vetor alvo em unidades originais
    y_orig = y_scaled .* scaler.y_std + scaler.y_mean;
end

function splits = split_data(X, y, ratios, mode, seed)
    % SPLIT_DATA Divide os dados em treino, validação e teste (ex: 70/15/15)
    % e aplica o scaler ajustado exclusivamente no conjunto de treino.
    %   Entradas:
    %       X      - Matriz de entradas
    %       y      - Vetor alvo
    %       ratios - Vetor [r_tr, r_va, r_te] (ex: [0.70, 0.15, 0.15])
    %       mode   - 'random' (Concrete) ou 'chronological' (Melbourne)
    %       seed   - Semente fixa para garantir reprodutibilidade no split aleatório
    %   Saídas:
    %       splits - Estrutura contendo dados originais, escalados, índices e scaler

    N = size(X, 1);
    n_tr = round(ratios(1) * N);
    n_va = round(ratios(2) * N);
    n_te = N - n_tr - n_va;

    if strcmpi(mode, 'random')
        rand('state', seed);
        idx = randperm(N);
    elseif strcmpi(mode, 'chronological')
        idx = 1:N;
    else
        error('Modo de partição inválido. Use "random" ou "chronological".');
    end

    idx_tr = idx(1:n_tr);
    idx_va = idx(n_tr+1 : n_tr+n_va);
    idx_te = idx(n_tr+n_va+1 : end);

    % Subconjuntos nas unidades originais
    splits.X_tr_raw = X(idx_tr, :);
    splits.y_tr_raw = y(idx_tr);
    splits.X_va_raw = X(idx_va, :);
    splits.y_va_raw = y(idx_va);
    splits.X_te_raw = X(idx_te, :);
    splits.y_te_raw = y(idx_te);

    splits.idx_tr = idx_tr;
    splits.idx_va = idx_va;
    splits.idx_te = idx_te;

    % Ajusta o scaler APENAS no conjunto de treino
    splits.scaler = fit_scaler(splits.X_tr_raw, splits.y_tr_raw);

    % Transforma treino, validação e teste com as estatísticas do treino
    [splits.X_tr, splits.y_tr] = transform_data(splits.X_tr_raw, splits.y_tr_raw, splits.scaler);
    [splits.X_va, splits.y_va] = transform_data(splits.X_va_raw, splits.y_va_raw, splits.scaler);
    [splits.X_te, splits.y_te] = transform_data(splits.X_te_raw, splits.y_te_raw, splits.scaler);
end

function mix_info = analyze_concrete_mixtures(X_raw, idx_tr, idx_va, idx_te)
    % ANALYZE_CONCRETE_MIXTURES Analisa quantas misturas únicas existem no Concrete e quantas partilham observações entre treino, validação e teste.
    %  Entradas:
    %    X_raw  - Matriz original de entradas
    %    idx_tr - Índices das observações de treino
    %    idx_va - Índices das observações de validação
    %    idx_te - Índices das observações de teste
    % Saídas:
    %    mix_info - Estrutura com informações sobre misturas únicas e sobreposição entre conjuntos

    % Colunas 1 a 7 definem a receita química/física (sem a idade de cura)
    recipes = X_raw(:, 1:7);
    [unique_recipes, ~, group_ids] = unique(recipes, 'rows');
    n_unique = size(unique_recipes, 1);

    tr_groups = unique(group_ids(idx_tr));
    va_groups = unique(group_ids(idx_va));
    te_groups = unique(group_ids(idx_te));

    overlap_tr_va = length(intersect(tr_groups, va_groups));
    overlap_tr_te = length(intersect(tr_groups, te_groups));

    mix_info.total_obs       = size(X_raw, 1);
    mix_info.unique_mixtures = n_unique;
    mix_info.overlap_tr_va   = overlap_tr_va;
    mix_info.overlap_tr_te   = overlap_tr_te;
    mix_info.pct_te_in_tr    = 100 * overlap_tr_te / length(te_groups);
end

function layers = build_arch(d_in, arch_id)
    % BUILD_ARCH Constrói o vetor de dimensões das camadas para as 3 arquiteturas.
    %   A: Reference -> 1 camada escondida com N=10 neurónios
    %   B: Wide      -> 1 camada escondida com N=100 neurónios
    %   C: Deep      -> 10 camadas escondidas com N=10 neurónios cada
    %  Entradas:
    %      d_in    - Número de entradas (features)
    %      arch_id - 'A', 'B' ou 'C'
    % Saidas:
    %      layers  - Vetor com o número de neurónios em cada camada (incluindo entrada e saída)
    N = 10;
    switch upper(arch_id)
        case 'A'
            layers = [d_in, N, 1];
        case 'B'
            layers = [d_in, 10 * N, 1];
        case 'C'
            layers = [d_in, repmat(N, 1, 10), 1];
        otherwise
            error('Arquitetura inválida. Escolha A, B ou C.');
    end
end

function theta = init_params(layers, seed)
    % INIT_PARAMS Inicializa pesos e biases num vetor coluna theta usando Glorot/Xavier.
    %  Entradas:
    %      layers - Vetor com o número de neurónios em cada camada
    %      seed   - Semente para reprodutibilidade
    % Saídas:
    %      theta  - Vetor coluna com todos os pesos e biases concatenados
    rand('state', seed);
    randn('state', seed);

    num_layers = length(layers) - 1;
    total_params = 0;
    for l = 1:num_layers
        total_params = total_params + layers(l) * layers(l+1) + layers(l+1);
    end

    theta = zeros(total_params, 1);
    idx = 1;
    for l = 1:num_layers
        n_in  = layers(l);
        n_out = layers(l+1);

        % Inicialização Glorot/Xavier uniforme: U[-limit, limit]
        limit = sqrt(6.0 / (n_in + n_out));
        W = (2.0 * rand(n_in, n_out) - 1.0) * limit;
        b = zeros(1, n_out);

        n_w = n_in * n_out;
        theta(idx : idx + n_w - 1) = W(:);
        idx = idx + n_w;

        theta(idx : idx + n_out - 1) = b(:);
        idx = idx + n_out;
    end
end

function [W_cell, b_cell] = unpack_params(theta, layers)
    % UNPACK_PARAMS Desempacota o vetor theta nas matrizes W{l} e vetores b{l}.
    % Entradas:
    %   theta  - Vetor coluna com todos os pesos e biases concatenados
    %   layers - Vetor com o número de neurónios em cada camada
    % Saídas:
    %   W_cell - Cell array com matrizes de pesos para cada camada
    %   b_cell - Cell array com vetores de bias para cada camada
    num_layers = length(layers) - 1;
    W_cell = cell(num_layers, 1);
    b_cell = cell(num_layers, 1);

    idx = 1;
    for l = 1:num_layers
        n_in  = layers(l);
        n_out = layers(l+1);

        n_w = n_in * n_out;
        W_cell{l} = reshape(theta(idx : idx + n_w - 1), n_in, n_out);
        idx = idx + n_w;

        b_cell{l} = reshape(theta(idx : idx + n_out - 1), 1, n_out);
        idx = idx + n_out;
    end
end

function [y_hat, A, W_cell] = forward_pass(theta, layers, X)
    % FORWARD_PASS : Propagação direta otimizada (tanh nas escondidas, linear na saída).
    % Entradas:
    %   theta  - Vetor coluna com todos os pesos e biases concatenados
    %   layers - Vetor com o número de neurónios em cada camada
    %   X      - Matriz de entradas (N x d_in)
    % Saídas:
    %   y_hat  - Saída da rede (N x 1)
    %   A      - Cell array com ativações de cada camada (incluindo entrada)
    %   W_cell - Cell array com matrizes de pesos para cada camada
    num_layers = length(layers) - 1;
    A = cell(num_layers + 1, 1);
    W_cell = cell(num_layers, 1);
    A{1} = X;

    idx = 1;
    for l = 1:num_layers
        n_in  = layers(l);
        n_out = layers(l+1);
        n_w   = n_in * n_out;

        W = reshape(theta(idx : idx + n_w - 1), n_in, n_out);
        idx = idx + n_w;
        b = theta(idx : idx + n_out - 1)';
        idx = idx + n_out;

        W_cell{l} = W;
        Z = A{l} * W + b;
        if l < num_layers
            A{l+1} = tanh(Z);
        else
            A{l+1} = Z;
        end
    end
    y_hat = A{end};
end

function [loss, grad] = loss_grad(theta, layers, X, y)
    % LOSS_GRAD Calcula o MSE e o gradiente analítico exato com mínima alocação de memória.
    % Entradas:
    %   theta  - Vetor coluna com todos os pesos e biases concatenados
    %   layers - Vetor com o número de neurónios em cada camada
    %   X      - Matriz de entradas (N x d_in)
    %   y      - Vetor alvo (N x 1)
    % Saídas:
    %   loss   - Valor do MSE
    %   grad   - Vetor coluna com o gradiente do MSE em relação a theta
    N = size(X, 1);
    [y_hat, A, W_cell] = forward_pass(theta, layers, X);

    diff = y_hat - y;
    loss = (diff' * diff) / N;

    num_layers = length(layers) - 1;
    grad = zeros(size(theta));
    delta = (2.0 / N) * diff;
    idx_end = length(theta);

    for l = num_layers:-1:1
        dW = A{l}' * delta;
        db = sum(delta, 1);

        n_b = length(db);
        n_w = numel(dW);

        grad(idx_end - n_b + 1 : idx_end) = db(:);
        idx_end = idx_end - n_b;

        grad(idx_end - n_w + 1 : idx_end) = dW(:);
        idx_end = idx_end - n_w;

        if l > 1
            delta = (delta * W_cell{l}') .* (1.0 - A{l} .^ 2);
        end
    end
end

function max_rel_err = check_gradient(layers, X, y, seed)
    % CHECK_GRADIENT Verifica o gradiente analítico contra diferenças finitas centradas.
    %   Devolve o erro relativo máximo: |g_ana - g_num| / max(|g_ana| + |g_num|, eps).
    % Entradas:
    %   layers - Vetor com o número de neurónios em cada camada
    %   X      - Matriz de entradas
    %   y      - Vetor alvo
    %   seed   - Semente para inicialização
    % Saidas:   
    %   max_rel_err - Erro relativo máximo entre gradiente analítico e numérico
    theta = init_params(layers, seed);
    [~, g_ana] = loss_grad(theta, layers, X, y);

    eps_fd = 1e-6;
    n_params = length(theta);
    g_num = zeros(n_params, 1);

    for i = 1:n_params
        theta_plus  = theta;
        theta_minus = theta;
        theta_plus(i)  = theta_plus(i)  + eps_fd;
        theta_minus(i) = theta_minus(i) - eps_fd;

        [L_plus, ~]  = loss_grad(theta_plus, layers, X, y);
        [L_minus, ~] = loss_grad(theta_minus, layers, X, y);

        g_num(i) = (L_plus - L_minus) / (2.0 * eps_fd);
    end

    denom = max(abs(g_ana) + abs(g_num), 1e-12);
    rel_errors = abs(g_ana - g_num) ./ denom;
    max_rel_err = max(rel_errors);
end

% ALGORITMO 1: MINI-BATCH GRADIENT DESCENT
function [best_theta, hist] = train_gd(theta0, layers, Xtr, ytr, Xva, yva, opts)
    % TRAIN_GD Mini-batch Gradient Descent com seleção pelo menor erro de validação.
    %  Entradas:
    %      theta0 - Vetor coluna inicial de pesos e biases
    %      layers - Vetor com o número de neurónios em cada camada
    %      Xtr    - Matriz de treino (N_tr x d)
    %      ytr    - Vetor alvo de treino (N_tr x 1)
    %      Xva    - Matriz de validação (N_va x d)
    %      yva    - Vetor alvo de validação (N_va x 1)
    %      opts   - Estrutura com opções
    % Saidas:
    %      best_theta - Vetor coluna com os melhores pesos e biases (menor erro de validação)
    %      hist       - Estrutura com histórico de perdas, épocas e tempo
    t_start = tic;
    theta = theta0;
    N = size(Xtr, 1);
    B = opts.batch_size;
    lr = opts.lr;
    max_epochs = opts.max_epochs;
    eval_every = 1;
    if isfield(opts, 'eval_every')
        eval_every = opts.eval_every;
    end
    eval_tr = true;
    if isfield(opts, 'eval_tr')
        eval_tr = opts.eval_tr;
    end

    rand('state', opts.seed + 1000);

    tr_loss = zeros(max_epochs, 1);
    va_loss = zeros(max_epochs, 1);
    epoch_eq = (1:max_epochs)';

    % Avaliação inicial no ponto de partida (época 0)
    y_va_hat = forward_pass(theta, layers, Xva);
    curr_va  = mean((y_va_hat - yva) .^ 2);
    best_va_loss = curr_va;
    best_theta   = theta;
    curr_tr = 0;
    if eval_tr
        y_tr_hat = forward_pass(theta, layers, Xtr);
        curr_tr  = mean((y_tr_hat - ytr) .^ 2);
    end

    for ep = 1:max_epochs
        perm = randperm(N);
        X_shuf = Xtr(perm, :);
        y_shuf = ytr(perm);

        for start_idx = 1:B:N
            end_idx = min(start_idx + B - 1, N);
            [~, g] = loss_grad(theta, layers, X_shuf(start_idx:end_idx, :), y_shuf(start_idx:end_idx));
            theta = theta - lr * g;
        end

        if mod(ep, eval_every) == 0 || ep == max_epochs
            y_va_hat = forward_pass(theta, layers, Xva);
            curr_va  = mean((y_va_hat - yva) .^ 2);

            if curr_va < best_va_loss
                best_va_loss = curr_va;
                best_theta = theta;
            end

            if eval_tr
                y_tr_hat = forward_pass(theta, layers, Xtr);
                curr_tr  = mean((y_tr_hat - ytr) .^ 2);
            end
        end

        tr_loss(ep) = curr_tr;
        va_loss(ep) = curr_va;
    end

    hist.tr_loss      = tr_loss;
    hist.va_loss      = va_loss;
    hist.epoch_eq     = epoch_eq;
    hist.best_va_loss = best_va_loss;
    hist.grad_evals   = max_epochs;
    hist.elapsed_time = toc(t_start);
end

% ALGORITMO 2: BFGS (QUASI-NEWTON COM STRONG WOLFE LINE SEARCH)
function [best_theta, hist] = train_bfgs(theta0, layers, Xtr, ytr, Xva, yva, opts)
    % TRAIN_BFGS : Treina a rede com BFGS (Quasi-Newton) e pesquisa linear de Wolfe.
    %   Entradas:
    %       theta0 - Vetor coluna inicial de pesos e biases
    %       layers - Vetor com o número de neurónios em cada camada
    %       Xtr    - Matriz de treino (N_tr x d)
    %       ytr    - Vetor alvo de treino (N_tr x 1)
    %       Xva    - Matriz de validação (N_va x d)
    %       yva    - Vetor alvo de validação (N_va x 1)
    %       opts   - Estrutura com opções (max_epochs, gtol)
    %   Saídas:
    %       best_theta - Vetor coluna com os melhores pesos e biases (menor erro de validação)
    %       hist       - Estrutura com histórico de perdas, épocas, gradientes e tempo
    t_start = tic;
    theta = theta0;
    n_params = length(theta);
    H = eye(n_params);

    max_evals = opts.max_epochs;
    gtol = opts.gtol;
    stol = 1e-8;
    ftol = 1e-9;
    if isfield(opts, 'stol'), stol = opts.stol; end
    if isfield(opts, 'ftol'), ftol = opts.ftol; end

    [f, g] = loss_grad(theta, layers, Xtr, ytr);
    evals = 1;

    tr_loss  = zeros(max_evals, 1);
    va_loss  = zeros(max_evals, 1);
    epoch_eq = zeros(max_evals, 1);
    rec_idx  = 1;

    y_va_hat = forward_pass(theta, layers, Xva);
    va_curr = mean((y_va_hat - yva) .^ 2);

    best_va_loss = va_curr;
    best_theta = theta;

    tr_loss(rec_idx)  = f;
    va_loss(rec_idx)  = va_curr;
    epoch_eq(rec_idx) = evals;

    while evals < max_evals
        if norm(g, inf) <= gtol
            break;
        end

        p = -H * g;
        if (p' * g) >= 0
            H = eye(n_params);
            p = -g;
        end

        [alpha, f_new, g_new, ls_evals] = line_search_wolfe(theta, p, f, g, layers, Xtr, ytr, max_evals - evals, opts);
        evals = evals + ls_evals;

        s = alpha * p;
        y_vec = g_new - g;
        ys = s' * y_vec;

        theta = theta + s;

        if ys > 1e-14
            rho = 1.0 / ys;
            Hy = H * y_vec;
            yHy = y_vec' * Hy;
            H = H + (rho + rho^2 * yHy) * (s * s') - rho * (Hy * s' + s * Hy');
        end

        delta_f = abs(f - f_new);
        f = f_new;
        g = g_new;

        y_va_hat = forward_pass(theta, layers, Xva);
        va_curr = mean((y_va_hat - yva) .^ 2);

        rec_idx = rec_idx + 1;
        tr_loss(rec_idx)  = f;
        va_loss(rec_idx)  = va_curr;
        epoch_eq(rec_idx) = evals;

        if va_curr < best_va_loss
            best_va_loss = va_curr;
            best_theta = theta;
        end

        % Critério de paragem antecipada no BFGS:
        % - estagnação no passo: norm(s) < stol (passo minúsculo)
        % - estagnação na função de perda: delta_f < ftol (após primeiras iterações)
        if norm(s) < stol || (rec_idx > 5 && delta_f < ftol)
            break;
        end
    end

    hist.tr_loss      = tr_loss(1:rec_idx);
    hist.va_loss      = va_loss(1:rec_idx);
    hist.epoch_eq     = epoch_eq(1:rec_idx);
    hist.best_va_loss = best_va_loss;
    hist.grad_evals   = evals;
    hist.elapsed_time = toc(t_start);
end

function [alpha, f_new, g_new, evals] = line_search_wolfe(theta, p, f0, g0, layers, X, y, rem_evals, opts)
    % LINE_SEARCH_WOLFE : Pesquisa linear de Wolfe forte para encontrar o passo alpha.
    %   Entradas:
    %       theta     - Vetor coluna atual de pesos e biases
    %       p         - Direção de descida
    %       f0        - Valor da função de perda atual
    %       g0        - Gradiente atual
    %       layers    - Vetor com o número de neurónios em cada camada
    %       X         - Matriz de entradas
    %       y         - Vetor alvo
    %       rem_evals - Número máximo de avaliações restantes
    %       opts      - Estrutura com parâmetros de pesquisa linear (wolfe_c1, wolfe_c2, max_ls_evals)
    %   Saídas:
    %       alpha     - Passo encontrado
    %       f_new     - Valor da função de perda no novo ponto
    %       g_new     - Gradiente no novo ponto
    %      evals     - Número de avaliações realizadas durante a pesquisa linear
    c1 = 1e-4;
    c2 = 0.9;
    max_ls_cfg = 15;
    if isfield(opts, 'wolfe_c1'), c1 = opts.wolfe_c1; end
    if isfield(opts, 'wolfe_c2'), c2 = opts.wolfe_c2; end
    if isfield(opts, 'max_ls_evals'), max_ls_cfg = opts.max_ls_evals; end

    alpha = 1.0;
    alpha_low = 0.0;
    alpha_high = inf;
    dg0 = p' * g0;
    evals = 0;

    f_new = f0;
    g_new = g0;
    f_prev = f0;

    max_ls = min(max_ls_cfg, max(1, rem_evals));

    for k = 1:max_ls
        [f_try, g_try] = loss_grad(theta + alpha * p, layers, X, y);
        evals = evals + 1;
        f_new = f_try;
        g_new = g_try;

        dg_try = p' * g_try;

        if (f_try > f0 + c1 * alpha * dg0) || (k > 1 && f_try >= f_prev)
            alpha_high = alpha;
            alpha = 0.5 * (alpha_low + alpha_high);
        else
            if abs(dg_try) <= -c2 * dg0
                return;
            end
            if dg_try >= 0
                alpha_high = alpha;
            else
                alpha_low = alpha;
            end
            if isinf(alpha_high)
                alpha = 2.0 * alpha;
            else
                alpha = 0.5 * (alpha_low + alpha_high);
            end
        end
        f_prev = f_try;
    end
end

% ALGORITMO 3: ADAM
function [best_theta, hist] = train_adam(theta0, layers, Xtr, ytr, Xva, yva, opts)
    % TRAIN_ADAM : Treina a rede com o algoritmo Adam e seleção pelo menor erro de validação.
    % Entradas:
    %      theta0 - Vetor coluna inicial de pesos e biases
    %      layers - Vetor com o número de neurónios em cada camada
    %      Xtr    - Matriz de treino (N_tr x d)
    %      ytr    - Vetor alvo de treino (N_tr x 1)
    %      Xva    - Matriz de validação (N_va x d)
    %      yva    - Vetor alvo de validação (N_va x 1)
    %      opts   - Estrutura com opções
    % Saídas:
    %      best_theta - Vetor coluna com os melhores pesos e biases (menor erro de validação)
    %      hist       - Estrutura com histórico de perdas, épocas e tempo
    t_start = tic;
    theta = theta0;
    N = size(Xtr, 1);
    B = opts.batch_size;
    lr = opts.lr;
    max_epochs = opts.max_epochs;
    eval_every = 1;
    if isfield(opts, 'eval_every')
        eval_every = opts.eval_every;
    end
    eval_tr = true;
    if isfield(opts, 'eval_tr')
        eval_tr = opts.eval_tr;
    end

    beta1 = 0.9;
    beta2 = 0.999;
    eps_adam = 1e-8;

    m = zeros(size(theta));
    v = zeros(size(theta));
    b1_pow = 1.0;
    b2_pow = 1.0;

    rand('state', opts.seed + 1000);

    tr_loss = zeros(max_epochs, 1);
    va_loss = zeros(max_epochs, 1);
    epoch_eq = (1:max_epochs)';

    % Avaliação inicial no ponto de partida (época 0)
    y_va_hat = forward_pass(theta, layers, Xva);
    curr_va  = mean((y_va_hat - yva) .^ 2);
    best_va_loss = curr_va;
    best_theta   = theta;
    curr_tr = 0;
    if eval_tr
        y_tr_hat = forward_pass(theta, layers, Xtr);
        curr_tr  = mean((y_tr_hat - ytr) .^ 2);
    end

    for ep = 1:max_epochs
        perm = randperm(N);
        X_shuf = Xtr(perm, :);
        y_shuf = ytr(perm);

        for start_idx = 1:B:N
            end_idx = min(start_idx + B - 1, N);
            [~, g] = loss_grad(theta, layers, X_shuf(start_idx:end_idx, :), y_shuf(start_idx:end_idx));

            b1_pow = b1_pow * beta1;
            b2_pow = b2_pow * beta2;

            m = beta1 * m + (1.0 - beta1) * g;
            v = beta2 * v + (1.0 - beta2) * (g .^ 2);

            lr_t = lr * sqrt(1.0 - b2_pow) / (1.0 - b1_pow);
            theta = theta - lr_t * m ./ (sqrt(v) + eps_adam);
        end

        if mod(ep, eval_every) == 0 || ep == max_epochs
            y_va_hat = forward_pass(theta, layers, Xva);
            curr_va  = mean((y_va_hat - yva) .^ 2);

            if curr_va < best_va_loss
                best_va_loss = curr_va;
                best_theta = theta;
            end

            if eval_tr
                y_tr_hat = forward_pass(theta, layers, Xtr);
                curr_tr  = mean((y_tr_hat - ytr) .^ 2);
            end
        end

        tr_loss(ep) = curr_tr;
        va_loss(ep) = curr_va;
    end

    hist.tr_loss      = tr_loss;
    hist.va_loss      = va_loss;
    hist.epoch_eq     = epoch_eq;
    hist.best_va_loss = best_va_loss;
    hist.grad_evals   = max_epochs;
    hist.elapsed_time = toc(t_start);
end

function tuning_res = tune_learning_rates(dataset_splits, dataset_name, lr_grid, n_reps, tune_epochs, batch_size, fig_dir)
    % TUNE_LEARNING_RATES Ajusta o learning rate para Mini-batch GD e Adam na validação.
    %  Entradas:
    %      dataset_splits - Estrutura com dados de treino, validação e teste
    %      dataset_name   - Nome do dataset (ex: 'Concrete' ou 'Melbourne')
    %      lr_grid        - Vetor de learning rates a testar
    %      n_reps         - Número de repetições para cada configuração
    %      tune_epochs    - Número de épocas para cada teste de learning rate
    %      batch_size     - Tamanho do mini-batch
    %      fig_dir        - Diretório para salvar figuras de tuning
    %  Saídas:
    %      tuning_res     - Estrutura com resultados de tuning (mediana, média, desvio padrão e melhor lr)
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    arch_ids  = {'A', 'B', 'C'};
    alg_names = {'GD', 'Adam'};
    d_in      = size(dataset_splits.X_tr, 2);
    n_lrs     = length(lr_grid);

    tuning_res.lr_grid   = lr_grid;
    tuning_res.arch_ids  = arch_ids;
    tuning_res.alg_names = alg_names;
    tuning_res.val_med   = zeros(3, 2, n_lrs);
    tuning_res.val_mean  = zeros(3, 2, n_lrs);
    tuning_res.val_std   = zeros(3, 2, n_lrs);
    tuning_res.best_lr   = zeros(3, 2);

    fprintf('--------------------------------------------------------------------------------\n');
    fprintf(' TUNING DE LEARNING RATE: %s (%d LRs x %d reps x %d épocas, B=%d)\n', dataset_name, n_lrs, n_reps, tune_epochs, batch_size);
    fprintf('--------------------------------------------------------------------------------\n');
    fflush(stdout);

    for a = 1:3
        layers = build_arch(d_in, arch_ids{a});

        for alg = 1:2
            for l_idx = 1:n_lrs
                lr_val = lr_grid(l_idx);
                rep_losses = zeros(n_reps, 1);

                for r = 1:n_reps
                    seed_tune = 100 + r;
                    theta0 = init_params(layers, seed_tune);

                    opts.batch_size = batch_size;
                    opts.lr         = lr_val;
                    opts.max_epochs = tune_epochs;
                    opts.seed       = seed_tune;
                    opts.eval_every = 10;
                    opts.eval_tr    = false; % Evita avaliação redundante no conjunto de treino durante o tuning

                    if alg == 1
                        [~, hist] = train_gd(theta0, layers, dataset_splits.X_tr, dataset_splits.y_tr, dataset_splits.X_va, dataset_splits.y_va, opts);
                    else
                        [~, hist] = train_adam(theta0, layers, dataset_splits.X_tr, dataset_splits.y_tr, dataset_splits.X_va, dataset_splits.y_va, opts);
                    end

                    if isnan(hist.best_va_loss) || isinf(hist.best_va_loss)
                        rep_losses(r) = 1e6;
                    else
                        rep_losses(r) = hist.best_va_loss;
                    end
                end

                tuning_res.val_med(a, alg, l_idx)  = median(rep_losses);
                tuning_res.val_mean(a, alg, l_idx) = mean(rep_losses);
                tuning_res.val_std(a, alg, l_idx)  = std(rep_losses);
            end

            med_curve = squeeze(tuning_res.val_med(a, alg, :));
            [best_val, best_idx] = min(med_curve);
            tuning_res.best_lr(a, alg) = lr_grid(best_idx);

            fprintf('  Arq %s | %-4s -> Melhor lr = %.2e (Mediana Val MSE = %.4f)\n', arch_ids{a}, alg_names{alg}, lr_grid(best_idx), best_val);
            fflush(stdout);
        end
    end
    fprintf('\n');

    h = create_headless_fig([100, 100, 1150, 420]);
    for a = 1:3
        subplot(1, 3, a);
        gd_curve   = squeeze(tuning_res.val_med(a, 1, :));
        adam_curve = squeeze(tuning_res.val_med(a, 2, :));

        gd_plot   = min(gd_curve, 2.0);
        adam_plot = min(adam_curve, 2.0);

        loglog(lr_grid, gd_plot, '-o', 'LineWidth', 1.5, 'Color', [0.8 0.3 0.2]);
        hold on;
        loglog(lr_grid, adam_plot, '-s', 'LineWidth', 1.5, 'Color', [0.15 0.45 0.75]);
        hold off;

        xlabel('Learning Rate (\eta)');
        if a == 1
            ylabel('Mediana Val MSE (escalado)');
        end
        title(sprintf('%s - Arquitetura %s', dataset_name, arch_ids{a}), 'FontWeight', 'bold');
        legend('Mini-batch GD', 'Adam', 'Location', 'northeast');
        grid on;
    end
    fname = sprintf('tuning_%s.png', lower(dataset_name));
    print(h, fullfile(fig_dir, fname), '-dpng', '-r300');
    close(h);
end

function metrics = compute_metrics(y_true, y_pred)
    % COMPUTE_METRICS Calcula MSE, RMSE, MAE e R^2 nas unidades originais do alvo.
    %   Entradas:
    %       y_true  - Vetor coluna com os valores reais observados
    %       y_pred  - Vetor coluna com os valores previstos pelo modelo
    %   Saídas:
    %       metrics - Estrutura com campos .mse, .rmse, .mae e .r2

    y_true = y_true(:);
    y_pred = y_pred(:);
    err    = y_true - y_pred;

    metrics.mse  = mean(err .^ 2);
    metrics.rmse = sqrt(metrics.mse);
    metrics.mae  = mean(abs(err));

    ss_res = sum(err .^ 2);
    ss_tot = sum((y_true - mean(y_true)) .^ 2);
    if ss_tot == 0
        metrics.r2 = 0;
    else
        metrics.r2 = 1.0 - (ss_res / ss_tot);
    end
end

function exp_res = run_final_experiments(dataset_splits, dataset_name, tuning_res, bfgs_opts, max_epochs, batch_size, n_runs)
    % RUN_FINAL_EXPERIMENTS Executa todas as 9 combinações (3 arquiteturas x 3 algoritmos)
    % para um dataset, repetidas n_runs vezes (seeds 1 a 30) de forma emparelhada.
    %   Entradas:
    %       dataset_splits - Estrutura com dados escalados e originais (treino, val, teste)
    %       dataset_name   - Nome do dataset ('Concrete' ou 'Melbourne')
    %       tuning_res     - Estrutura devolvida por tune_learning_rates com .best_lr
    %       bfgs_opts      - Opções do BFGS (.gtol, .wolfe_c1, .wolfe_c2, .max_ls_evals)
    %       max_epochs     - Orçamento máximo em epoch-equivalents
    %       batch_size     - Tamanho do mini-batch (32)
    %       n_runs         - Número de repetições (30)
    %   Saídas:
    %       exp_res        - Estrutura completa com históricos, métricas e melhores modelos

    arch_ids  = {'A', 'B', 'C'};
    alg_names = {'GD', 'BFGS', 'Adam'};
    d_in      = size(dataset_splits.X_tr, 2);

    exp_res.dataset_name = dataset_name;
    exp_res.arch_ids     = arch_ids;
    exp_res.alg_names    = alg_names;
    exp_res.max_epochs   = max_epochs;
    exp_res.n_runs       = n_runs;

    % Matrizes de métricas no teste: (3 arquiteturas x 3 algoritmos x n_runs)
    exp_res.test_mse   = zeros(3, 3, n_runs);
    exp_res.test_rmse  = zeros(3, 3, n_runs);
    exp_res.test_mae   = zeros(3, 3, n_runs);
    exp_res.test_r2    = zeros(3, 3, n_runs);
    exp_res.grad_evals = zeros(3, 3, n_runs);
    exp_res.train_time = zeros(3, 3, n_runs);

    % Históricos de treino/validação época a época: cell(3, 3, n_runs)
    exp_res.histories   = cell(3, 3, n_runs);
    exp_res.best_thetas = cell(3, 3, n_runs);

    fprintf('================================================================================\n');
    fprintf(' EXPERIÊNCIAS FINAIS: %s (3 Arqs x 3 Algs x %d runs | Orçamento = %d ep-eq)\n', ...
            dataset_name, n_runs, max_epochs);
    fprintf('================================================================================\n');
    fflush(stdout);

    for a = 1:3
        layers = build_arch(d_in, arch_ids{a});
        lr_gd   = tuning_res.best_lr(a, 1);
        lr_adam = tuning_res.best_lr(a, 2);

        fprintf('Arquitetura %s (%d parâmetros): a executar %d runs emparelhadas...\n', ...
                arch_ids{a}, length(init_params(layers, 1)), n_runs);
        fflush(stdout);

        for k = 1:n_runs
            % Regra comum (i): mesma inicialização para os 3 algoritmos na run k
            theta0 = init_params(layers, k);

            % --- 1. Mini-batch Gradient Descent ---
            opts_gd.batch_size = batch_size;
            opts_gd.lr         = lr_gd;
            opts_gd.max_epochs = max_epochs;
            opts_gd.seed       = k;
            opts_gd.eval_every = 1;
            opts_gd.eval_tr    = true;

            [theta_gd, hist_gd] = train_gd(theta0, layers, ...
                dataset_splits.X_tr, dataset_splits.y_tr, ...
                dataset_splits.X_va, dataset_splits.y_va, opts_gd);

            % --- 2. BFGS ---
            opts_bf            = bfgs_opts;
            opts_bf.max_epochs = max_epochs;

            [theta_bf, hist_bf] = train_bfgs(theta0, layers, ...
                dataset_splits.X_tr, dataset_splits.y_tr, ...
                dataset_splits.X_va, dataset_splits.y_va, opts_bf);

            % --- 3. Adam ---
            opts_ad.batch_size = batch_size;
            opts_ad.lr         = lr_adam;
            opts_ad.max_epochs = max_epochs;
            opts_ad.seed       = k;
            opts_ad.eval_every = 1;
            opts_ad.eval_tr    = true;

            [theta_ad, hist_ad] = train_adam(theta0, layers, ...
                dataset_splits.X_tr, dataset_splits.y_tr, ...
                dataset_splits.X_va, dataset_splits.y_va, opts_ad);

            thetas_k = {theta_gd, theta_bf, theta_ad};
            hists_k  = {hist_gd,  hist_bf,  hist_ad};

            % Avaliação única no conjunto de teste nas unidades originais
            for alg = 1:3
                th_best = thetas_k{alg};
                hs_curr = hists_k{alg};

                y_te_hat_scaled = forward_pass(th_best, layers, dataset_splits.X_te);
                y_te_hat_orig   = inverse_transform_y(y_te_hat_scaled, dataset_splits.scaler);

                m_te = compute_metrics(dataset_splits.y_te_raw, y_te_hat_orig);

                exp_res.test_mse(a, alg, k)   = m_te.mse;
                exp_res.test_rmse(a, alg, k)  = m_te.rmse;
                exp_res.test_mae(a, alg, k)   = m_te.mae;
                exp_res.test_r2(a, alg, k)    = m_te.r2;
                exp_res.grad_evals(a, alg, k) = hs_curr.grad_evals;
                exp_res.train_time(a, alg, k) = hs_curr.elapsed_time;

                exp_res.histories{a, alg, k}   = hs_curr;
                exp_res.best_thetas{a, alg, k} = th_best;
            end
        end

        % Imprime resumo rápido da arquitetura concluída
        for alg = 1:3
            rmses = squeeze(exp_res.test_rmse(a, alg, :));
            r2s   = squeeze(exp_res.test_r2(a, alg, :));
            evals = squeeze(exp_res.grad_evals(a, alg, :));
            times = squeeze(exp_res.train_time(a, alg, :));
            fprintf('  -> %-4s | Test RMSE: %.3f ± %.3f | R2: %.4f ± %.4f | Ep-Eq: %.0f | Tempo por execucao: %.2fs\n', ...
                    alg_names{alg}, mean(rmses), std(rmses), mean(r2s), std(r2s), mean(evals), mean(times));
        end
        fprintf('\n');
        fflush(stdout);
    end
end

function base_res = eval_baselines(conc_data, melb_data, p_window)
    % EVAL_BASELINES Avalia os modelos de referência exigidos no teste:
    %   - Concrete: Regressão Linear Múltipla (OLS ajustada no treino)
    %   - Melbourne: Persistência T(t+1)=T(t) e Modelo Autorregressivo Linear AR(p)
    %   Devolve métricas nas unidades originais (MPa e °C).

    % 1. Concrete: Regressão Linear OLS (com termo independente / bias)
    X_tr_c = [ones(size(conc_data.X_tr_raw, 1), 1), conc_data.X_tr_raw];
    X_te_c = [ones(size(conc_data.X_te_raw, 1), 1), conc_data.X_te_raw];
    beta_conc = X_tr_c \ conc_data.y_tr_raw;
    y_pred_conc = X_te_c * beta_conc;

    base_res.conc_ols = compute_metrics(conc_data.y_te_raw, y_pred_conc);
    base_res.conc_pred = y_pred_conc;

    % 2. Melbourne: Persistência T(t+1) = T(t) (coluna 1 de X_te_raw é T(t))
    y_pred_pers = melb_data.X_te_raw(:, 1);
    base_res.melb_pers = compute_metrics(melb_data.y_te_raw, y_pred_pers);
    base_res.melb_pers_pred = y_pred_pers;

    % 3. Melbourne: Modelo Autorregressivo Linear AR(p) sobre a mesma janela p
    X_tr_ar = [ones(size(melb_data.X_tr_raw, 1), 1), melb_data.X_tr_raw(:, 1:p_window)];
    X_te_ar = [ones(size(melb_data.X_te_raw, 1), 1), melb_data.X_te_raw(:, 1:p_window)];
    beta_ar = X_tr_ar \ melb_data.y_tr_raw;
    y_pred_ar = X_te_ar * beta_ar;

    base_res.melb_ar = compute_metrics(melb_data.y_te_raw, y_pred_ar);
    base_res.melb_ar_pred = y_pred_ar;

    fprintf('================================================================================\n');
    fprintf(' MODELOS DE REFERÊNCIA (CONJUNTO DE TESTE - UNIDADES ORIGINAIS)\n');
    fprintf('================================================================================\n');
    fprintf('Concrete  | Regressão Linear (OLS) -> RMSE: %6.3f MPa | MAE: %6.3f | R2: %.4f\n', ...
            base_res.conc_ols.rmse, base_res.conc_ols.mae, base_res.conc_ols.r2);
    fprintf('Melbourne | Persistência T(t+1)=T(t)-> RMSE: %6.3f °C  | MAE: %6.3f | R2: %.4f\n', ...
            base_res.melb_pers.rmse, base_res.melb_pers.mae, base_res.melb_pers.r2);
    fprintf('Melbourne | Autorregressivo AR(%d)   -> RMSE: %6.3f °C  | MAE: %6.3f | R2: %.4f\n', ...
            p_window, base_res.melb_ar.rmse, base_res.melb_ar.mae, base_res.melb_ar.r2);
    fprintf('================================================================================\n\n');
end

function print_summary_table(exp_res, unit_str)
    % PRINT_SUMMARY_TABLE Imprime a tabela 3x3 (Arquitetura x Algoritmo) com:
    %   RMSE e R2 (média ± std, mediana, melhor, pior) e tempo médio de treino.

    arch_ids  = exp_res.arch_ids;
    alg_names = exp_res.alg_names;

    fprintf('=============================================================================================================\n');
    fprintf(' TABELA RESUMO 3x3: %s (30 runs | Unidades: %s)\n', exp_res.dataset_name, unit_str);
    fprintf('=============================================================================================================\n');
    fprintf('%-4s | %-5s | %-15s | %-8s | %-8s | %-8s | %-15s | %-8s | %-7s\n', ...
            'Arq', 'Alg', 'RMSE (M±Std)', 'Mediana', 'Melhor', 'Pior', 'R2 (M±Std)', 'Med R2', 'Tempo por execucao');
    fprintf('-------------------------------------------------------------------------------------------------------------\n');

    for a = 1:3
        for alg = 1:3
            rmses = squeeze(exp_res.test_rmse(a, alg, :));
            r2s   = squeeze(exp_res.test_r2(a, alg, :));
            times = squeeze(exp_res.train_time(a, alg, :));

            fprintf('%-4s | %-5s | %6.3f ± %-6.3f | %8.3f | %8.3f | %8.3f | %6.4f ± %-6.4f | %8.4f | %6.2fs\n', ...
                    arch_ids{a}, alg_names{alg}, ...
                    mean(rmses), std(rmses), median(rmses), min(rmses), max(rmses), ...
                    mean(r2s), std(r2s), median(r2s), mean(times));
        end
        if a < 3
            fprintf('-------------------------------------------------------------------------------------------------------------\n');
        end
    end
    fprintf('=============================================================================================================\n\n');
end

function plot_training_progress(exp_res, fig_dir)
    % PLOT_TRAINING_PROGRESS Gera 6 figuras (3 por dataset: Arq A, B e C) mostrando
    % a mediana e o intervalo interquartil (IQR: Q25-Q75) do treino e validação
    % ao longo dos epoch-equivalents (escala vertical logarítmica), plus versão vs tempo.

    arch_ids  = exp_res.arch_ids;
    alg_names = exp_res.alg_names;
    max_ep    = exp_res.max_epochs;
    n_runs    = exp_res.n_runs;
    ds_lower  = lower(exp_res.dataset_name);

    colors = {[0.85 0.25 0.20], [0.15 0.60 0.30], [0.15 0.45 0.80]};
    ep_grid = (1:max_ep)';

    for a = 1:3
        % Matrizes interpoladas numa grelha comum de 1..max_ep epoch-equivalents
        tr_all = zeros(3, n_runs, max_ep);
        va_all = zeros(3, n_runs, max_ep);
        tm_all = zeros(3, n_runs, max_ep);

        for alg = 1:3
            for k = 1:n_runs
                hs = exp_res.histories{a, alg, k};
                eq = hs.epoch_eq(:);
                tr = hs.tr_loss(:);
                va = hs.va_loss(:);

                if length(eq) == max_ep && all(eq == ep_grid)
                    tr_interp = tr;
                    va_interp = va;
                else
                    % Garante monotonicidade estrita e estende último valor se BFGS parou cedo
                    [eq_u, u_idx] = unique(eq, 'last');
                    tr_u = tr(u_idx);
                    va_u = va(u_idx);
                    if eq_u(1) > 1
                        eq_u = [1; eq_u];
                        tr_u = [tr_u(1); tr_u];
                        va_u = [va_u(1); va_u];
                    end
                    if eq_u(end) < max_ep
                        eq_u = [eq_u; max_ep];
                        tr_u = [tr_u; tr_u(end)];
                        va_u = [va_u; va_u(end)];
                    end
                    tr_interp = interp1(eq_u, tr_u, ep_grid, 'previous');
                    va_interp = interp1(eq_u, va_u, ep_grid, 'previous');
                end

                tr_all(alg, k, :) = max(tr_interp, 1e-8);
                va_all(alg, k, :) = max(va_interp, 1e-8);
                tm_all(alg, k, :) = linspace(0, hs.elapsed_time, max_ep);
            end
        end

        % --- Figura 1: Progresso vs Epoch-Equivalents (Treino e Validação lado a lado) ---
        h1 = create_headless_fig([100, 100, 1150, 440]);
        for panel = 1:2
            subplot(1, 2, panel);
            hold on;
            for alg = 1:3
                if panel == 1
                    mat = squeeze(tr_all(alg, :, :));
                    p_title = sprintf('%s - Arq %s (Treino)', exp_res.dataset_name, arch_ids{a});
                else
                    mat = squeeze(va_all(alg, :, :));
                    p_title = sprintf('%s - Arq %s (Validação)', exp_res.dataset_name, arch_ids{a});
                end

                q25 = quantile(mat, 0.25, 1);
                q50 = quantile(mat, 0.50, 1);
                q75 = quantile(mat, 0.75, 1);

                col = colors{alg};
                % Banda IQR (Q25-Q75) com linhas tracejadas finas + Mediana sólida
                plot(ep_grid, q50, '-', 'Color', col, 'LineWidth', 1.8);
                plot(ep_grid, q25, '--', 'Color', col, 'LineWidth', 0.7);
                plot(ep_grid, q75, '--', 'Color', col, 'LineWidth', 0.7);
            end
            hold off;
            set(gca, 'YScale', 'log');
            xlim([1, max_ep]);
            xlabel('Epoch-Equivalents');
            ylabel('MSE (unidades escaladas, log)');
            title(p_title, 'FontWeight', 'bold');
            legend('GD (Med)', 'GD Q25', 'GD Q75', ...
                   'BFGS (Med)', 'BFGS Q25', 'BFGS Q75', ...
                   'Adam (Med)', 'Adam Q25', 'Adam Q75', 'Location', 'northeast');
            grid on;
        end
        print(h1, fullfile(fig_dir, sprintf('progress_ep_%s_arch%s.png', ds_lower, arch_ids{a})), '-dpng', '-r300');
        close(h1);

        % --- Figura 2: Progresso da Validação vs Tempo Decorrido (segundos) ---
        h2 = create_headless_fig([100, 100, 650, 420]);
        hold on;
        for alg = 1:3
            mat_va = squeeze(va_all(alg, :, :));
            mat_tm = squeeze(tm_all(alg, :, :));
            q50_va = quantile(mat_va, 0.50, 1);
            q50_tm = quantile(mat_tm, 0.50, 1);
            plot(q50_tm, q50_va, '-', 'Color', colors{alg}, 'LineWidth', 1.8);
        end
        hold off;
        set(gca, 'YScale', 'log');
        xlabel('Tempo Mediano Decorrido (s)');
        ylabel('Validação MSE (escalado, log)');
        title(sprintf('%s - Arq %s (Convergência vs Tempo)', exp_res.dataset_name, arch_ids{a}), 'FontWeight', 'bold');
        legend('Mini-batch GD', 'BFGS', 'Adam', 'Location', 'northeast');
        grid on;
        print(h2, fullfile(fig_dir, sprintf('progress_time_%s_arch%s.png', ds_lower, arch_ids{a})), '-dpng', '-r300');
        close(h2);
    end
end

function plot_custom_boxplots(exp_res, unit_str, fig_dir)
    % PLOT_CUSTOM_BOXPLOTS Desenha box plots dos erros finais no teste (RMSE)
    % para as 9 combinações (3 arquiteturas x 3 algoritmos) SEM usar toolboxes.

    arch_ids  = exp_res.arch_ids;
    alg_names = exp_res.alg_names;
    ds_lower  = lower(exp_res.dataset_name);

    h = create_headless_fig([100, 100, 950, 480]);
    hold on;

    labels = cell(9, 1);
    colors = {[0.85 0.35 0.30], [0.25 0.65 0.40], [0.25 0.50 0.80]};
    box_w  = 0.25;
    pos    = 0;

    for a = 1:3
        for alg = 1:3
            pos = pos + 1;
            vals = sort(squeeze(exp_res.test_rmse(a, alg, :)));
            labels{pos} = sprintf('%s-%s', arch_ids{a}, alg_names{alg});

            q25 = quantile(vals, 0.25);
            q50 = quantile(vals, 0.50);
            q75 = quantile(vals, 0.75);
            iqr_val = q75 - q25;

            low_whisker  = min(vals(vals >= q25 - 1.5 * iqr_val));
            high_whisker = max(vals(vals <= q75 + 1.5 * iqr_val));
            outliers     = vals(vals < low_whisker | vals > high_whisker);

            col = colors{alg};

            % Caixa (Q25 a Q75)
            patch([pos - box_w, pos + box_w, pos + box_w, pos - box_w], ...
                  [q25, q25, q75, q75], col, 'EdgeColor', 'k', 'LineWidth', 1.1);

            % Linha da Mediana
            plot([pos - box_w, pos + box_w], [q50, q50], '-k', 'LineWidth', 2.0);

            % Bigodes (Whiskers) e terminais
            plot([pos, pos], [q75, high_whisker], '--k', 'LineWidth', 1.0);
            plot([pos, pos], [low_whisker, q25], '--k', 'LineWidth', 1.0);
            plot([pos - box_w*0.5, pos + box_w*0.5], [high_whisker, high_whisker], '-k', 'LineWidth', 1.2);
            plot([pos - box_w*0.5, pos + box_w*0.5], [low_whisker, low_whisker], '-k', 'LineWidth', 1.2);

            % Outliers individuais
            if ~isempty(outliers)
                plot(repmat(pos, length(outliers), 1), outliers, 'ok', 'MarkerSize', 5, 'MarkerFaceColor', 'w');
            end
        end
    end

    % Separadores verticais entre arquiteturas A, B e C
    yl = ylim;
    plot([3.5, 3.5], yl, ':k', 'LineWidth', 1.0);
    plot([6.5, 6.5], yl, ':k', 'LineWidth', 1.0);
    hold off;

    xlim([0.3, 9.7]);
    set(gca, 'XTick', 1:9, 'XTickLabel', labels, 'FontSize', 9, 'FontWeight', 'bold');
    ylabel(sprintf('Test RMSE (%s)', unit_str), 'FontSize', 10);
    title(sprintf('Distribuição do Erro de Teste (30 Runs) - %s', exp_res.dataset_name), 'FontSize', 11, 'FontWeight', 'bold');
    grid on;

    print(h, fullfile(fig_dir, sprintf('boxplot_rmse_%s.png', ds_lower)), '-dpng', '-r300');
    close(h);
end

function plot_predictions(exp_res, dataset_splits, dates_test, fig_dir)
    % PLOT_PREDICTIONS Gera os gráficos de predição no teste e histogramas de resíduos
    % utilizando o melhor modelo (menor RMSE de teste entre as 30 runs) da melhor combinação.

    ds_lower = lower(exp_res.dataset_name);
    d_in     = size(dataset_splits.X_te, 2);

    % Identifica a melhor combinação (arquitetura, algoritmo) pela mediana do RMSE
    med_mat = median(exp_res.test_rmse, 3);
    [~, flat_idx] = min(med_mat(:));
    [best_a, best_alg] = ind2sub([3, 3], flat_idx);

    % Seleciona a run k com melhor desempenho nessa combinação
    [~, best_k] = min(squeeze(exp_res.test_rmse(best_a, best_alg, :)));

    layers     = build_arch(d_in, exp_res.arch_ids{best_a});
    best_theta = exp_res.best_thetas{best_a, best_alg, best_k};

    y_hat_sc = forward_pass(best_theta, layers, dataset_splits.X_te);
    y_pred   = inverse_transform_y(y_hat_sc, dataset_splits.scaler);
    y_true   = dataset_splits.y_te_raw;
    resids   = y_true - y_pred;

    best_label = sprintf('Arq %s + %s (Run %d)', exp_res.arch_ids{best_a}, exp_res.alg_names{best_alg}, best_k);

    if strcmpi(exp_res.dataset_name, 'Concrete')
        h = create_headless_fig([100, 100, 1050, 430]);

        % 1. Previsto vs Medido
        subplot(1, 2, 1);
        scatter(y_true, y_pred, 18, [0.15 0.45 0.75], 'filled');
        hold on;
        lims = [min([y_true; y_pred]), max([y_true; y_pred])];
        plot(lims, lims, '--r', 'LineWidth', 1.5);
        hold off;
        xlim(lims); ylim(lims);
        xlabel('Resistência Medida (MPa)');
        ylabel('Resistência Prevista (MPa)');
        title(sprintf('Concrete: Previsto vs Medido (%s)', best_label), 'FontWeight', 'bold');
        legend('Amostras de Teste', 'Ideal (y = x)', 'Location', 'northwest');
        grid on;

        % 2. Histograma de Resíduos
        subplot(1, 2, 2);
        hist(resids, 20);
        h_patch = findobj(gca, 'Type', 'patch');
        set(h_patch, 'FaceColor', [0.25 0.55 0.45], 'EdgeColor', 'w');
        xlabel('Resíduo (Medido - Previsto) [MPa]');
        ylabel('Frequência');
        title(sprintf('Histograma de Resíduos (Média=%.2f, Std=%.2f)', mean(resids), std(resids)), 'FontWeight', 'bold');
        grid on;

        print(h, fullfile(fig_dir, 'predictions_concrete.png'), '-dpng', '-r300');
        close(h);
    else
        h = create_headless_fig([100, 100, 1200, 430]);
        N_te = length(y_true);

        % 1. Série de Teste Completa (Observado vs Previsto)
        subplot(1, 3, 1);
        plot(1:N_te, y_true, '-k', 'LineWidth', 0.8);
        hold on;
        plot(1:N_te, y_pred, '--', 'Color', [0.85 0.25 0.20], 'LineWidth', 0.8);
        hold off;
        xlim([1, N_te]);
        xlabel('Dias no Conjunto de Teste');
        ylabel('Temperatura Mínima (°C)');
        title(sprintf('Melbourne Teste Completo (%s)', best_label), 'FontWeight', 'bold');
        legend('Observado', 'Previsto', 'Location', 'northeast');
        grid on;

        % 2. Janela Ampliada (primeiros 90 dias do teste)
        subplot(1, 3, 2);
        zoom_n = min(90, N_te);
        plot(1:zoom_n, y_true(1:zoom_n), '-ok', 'MarkerSize', 3, 'LineWidth', 1.0);
        hold on;
        plot(1:zoom_n, y_pred(1:zoom_n), '-s', 'Color', [0.15 0.45 0.80], 'MarkerSize', 3, 'LineWidth', 1.0);
        hold off;
        xlim([1, zoom_n]);
        xlabel('Dias (Janela Ampliada: 90 dias)');
        ylabel('Temperatura Mínima (°C)');
        title('Zoom: Observado vs Previsto', 'FontWeight', 'bold');
        legend('Observado', 'Previsto', 'Location', 'northeast');
        grid on;

        % 3. Histograma de Resíduos
        subplot(1, 3, 3);
        hist(resids, 22);
        h_patch = findobj(gca, 'Type', 'patch');
        set(h_patch, 'FaceColor', [0.25 0.55 0.45], 'EdgeColor', 'w');
        xlabel('Resíduo (Observado - Previsto) [°C]');
        ylabel('Frequência');
        title(sprintf('Resíduos (Média=%.2f, Std=%.2f)', mean(resids), std(resids)), 'FontWeight', 'bold');
        grid on;

        print(h, fullfile(fig_dir, 'predictions_melbourne.png'), '-dpng', '-r300');
        close(h);
    end
end