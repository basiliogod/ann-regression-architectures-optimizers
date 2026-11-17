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
end

function h = create_headless_fig(pos)
    % CREATE_HEADLESS_FIG : Função auxiliar que cria uma figura invisível compatível com ambientes sem ecrã.
    if exist('OCTAVE_VERSION', 'builtin')
        warning('off', 'Octave:deprecated-graphics-toolkit');
        if (isempty(getenv('DISPLAY')) || ~isguirunning()) && ismember('gnuplot', available_graphics_toolkits())
            graphics_toolkit('gnuplot');
            h = figure('Visible', 'off', 'Position', pos);
            set(h, '__graphics_toolkit__', 'gnuplot');
            return;
        end
    end
    h = figure('Visible', 'off', 'Position', pos);
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

function [y_hat, cache] = forward_pass(theta, layers, X)
    % FORWARD_PASS Executa a propagação direta (tanh nas camadas escondidas, linear na saída).
    % Entradas:
    %   theta  - Vetor coluna com todos os pesos e biases concatenados
    %   layers - Vetor com o número de neurónios em cada camada
    %   X      - Matriz de entradas
    % Saídas:
    %   y_hat  - Saída da rede 
    %  cache  - Estrutura com ativações e pesos para uso no backpropagation
    [W_cell, b_cell] = unpack_params(theta, layers);
    num_layers = length(layers) - 1;
    N = size(X, 1);

    A = cell(num_layers + 1, 1);
    A{1} = X;

    for l = 1:num_layers
        Z = A{l} * W_cell{l} + repmat(b_cell{l}, N, 1);
        if l < num_layers
            A{l+1} = tanh(Z);
        else
            A{l+1} = Z; % Saída linear
        end
    end

    y_hat = A{end};
    cache.A = A;
    cache.W = W_cell;
end

function [loss, grad] = loss_grad(theta, layers, X, y)
    % LOSS_GRAD Calcula o MSE = (1/N)*sum((y_hat - y)^2) e o gradiente analítico exato.
    % Entradas:
    %   theta  - Vetor coluna com todos os pesos e biases concatenados
    %   layers - Vetor com o número de neurónios em cada camada
    %   X      - Matriz de entradas
    %   y      - Vetor alvo
    % Saídas:
    %   loss   - Valor do MSE
    %   grad   - Vetor coluna com o gradiente do MSE em relação a theta
    N = size(X, 1);
    [y_hat, cache] = forward_pass(theta, layers, X);

    diff = y_hat - y;
    loss = mean(diff .^ 2);

    num_layers = length(layers) - 1;
    grad = zeros(size(theta));

    % Derivada do MSE em relação à saída linear: dL/dZ_L = (2/N)*(y_hat - y)
    delta = (2.0 / N) * diff;

    % Ponteiro para preencher grad de trás para a frente
    idx_end = length(theta);

    for l = num_layers:-1:1
        A_prev = cache.A{l};
        W_curr = cache.W{l};

        dW = A_prev' * delta;
        db = sum(delta, 1);

        n_b = length(db);
        n_w = numel(dW);

        grad(idx_end - n_b + 1 : idx_end) = db(:);
        idx_end = idx_end - n_b;

        grad(idx_end - n_w + 1 : idx_end) = dW(:);
        idx_end = idx_end - n_w;

        if l > 1
            % Derivada de tanh(Z): 1 - A{l}^2
            delta = (delta * W_curr') .* (1.0 - cache.A{l} .^ 2);
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
    % TRAIN_GD Treina a rede com Mini-batch Gradient Descent e seleção pelo menor loss de validação.
    % Entradas:
    %   theta0 - Vetor coluna inicial de pesos e biases
    %   layers - Vetor com o número de neurónios em cada camada
    %   Xtr    - Matriz de treino
    %   ytr    - Vetor alvo de treino
    %   Xva    - Matriz de validação
    %   yva    - Vetor alvo de validação
    %   opts   - Estrutura com opções: batch_size, lr, max_epochs, seed
    % Saídas:
    %   best_theta - Vetor coluna com os pesos e biases que obtiveram o menor loss de validação
    %   hist       - Estrutura com histórico de perdas, épocas e tempo decorrido
    t_start = tic;
    theta = theta0;
    N = size(Xtr, 1);
    B = opts.batch_size;
    lr = opts.lr;
    max_epochs = opts.max_epochs;

    rand('state', opts.seed + 1000);

    tr_loss = zeros(max_epochs, 1);
    va_loss = zeros(max_epochs, 1);
    epoch_eq = (1:max_epochs)';

    best_va_loss = inf;
    best_theta = theta;

    for ep = 1:max_epochs
        perm = randperm(N);
        for start_idx = 1:B:N
            end_idx = min(start_idx + B - 1, N);
            batch_idx = perm(start_idx:end_idx);

            [~, g] = loss_grad(theta, layers, Xtr(batch_idx, :), ytr(batch_idx));
            theta = theta - lr * g;
        end

        % Avalia perda no treino completo e na validação no final da época
        y_tr_hat = forward_pass(theta, layers, Xtr);
        y_va_hat = forward_pass(theta, layers, Xva);

        tr_loss(ep) = mean((y_tr_hat - ytr) .^ 2);
        va_loss(ep) = mean((y_va_hat - yva) .^ 2);

        if va_loss(ep) < best_va_loss
            best_va_loss = va_loss(ep);
            best_theta = theta;
        end
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
% TRAIN_BFGS Treina a rede com método Quasi-Newton BFGS em full-batch.
%   Contabiliza cada avaliação de gradiente completo como 1 epoch-equivalent.
    t_start = tic;
    theta = theta0;
    n_params = length(theta);
    H = eye(n_params); % Aproximação inicial da inversa da Hessiana

    max_evals = opts.max_epochs; % Orçamento em epoch-equivalents
    gtol = opts.gtol;

    [f, g] = loss_grad(theta, layers, Xtr, ytr);
    evals = 1;

    tr_loss  = [];
    va_loss  = [];
    epoch_eq = [];

    y_va_hat = forward_pass(theta, layers, Xva);
    va_curr = mean((y_va_hat - yva) .^ 2);

    best_va_loss = va_curr;
    best_theta = theta;

    tr_loss(end+1, 1)  = f;
    va_loss(end+1, 1)  = va_curr;
    epoch_eq(end+1, 1) = evals;

    while evals < max_evals
        if norm(g, inf) <= gtol
            break;
        end

        % Direção de descida Quasi-Newton
        p = -H * g;
        dg = dot(p, g);
        if dg >= 0
            % Reinicia H se perder propriedade de descida
            H = eye(n_params);
            p = -g;
        end

        % Line search com condições de Wolfe / Backtracking seguro
        [alpha, f_new, g_new, ls_evals] = line_search_wolfe(theta, p, f, g, layers, Xtr, ytr, max_evals - evals);
        evals = evals + ls_evals;

        s = alpha * p;
        y_vec = g_new - g;
        ys = dot(s, y_vec);

        theta = theta + s;

        % Atualização BFGS da inversa da Hessiana (com proteção de curvatura)
        if ys > 1e-14
            rho = 1.0 / ys;
            Hy = H * y_vec;
            yHy = dot(y_vec, Hy);
            H = H + (rho + rho^2 * yHy) * (s * s') - rho * (Hy * s' + s * Hy');
        end

        f = f_new;
        g = g_new;

        y_va_hat = forward_pass(theta, layers, Xva);
        va_curr = mean((y_va_hat - yva) .^ 2);

        tr_loss(end+1, 1)  = f;
        va_loss(end+1, 1)  = va_curr;
        epoch_eq(end+1, 1) = evals;

        if va_curr < best_va_loss
            best_va_loss = va_curr;
            best_theta = theta;
        end

        if norm(s) < 1e-12
            break;
        end
    end

    hist.tr_loss      = tr_loss;
    hist.va_loss      = va_loss;
    hist.epoch_eq     = epoch_eq;
    hist.best_va_loss = best_va_loss;
    hist.grad_evals   = evals;
    hist.elapsed_time = toc(t_start);
end

function [alpha, f_new, g_new, evals] = line_search_wolfe(theta, p, f0, g0, layers, X, y, rem_evals)
% LINE_SEARCH_WOLFE Pesquisa linear satisfazendo Armijo e curvatura fraca/forte.
    c1 = 1e-4;
    c2 = 0.9;
    alpha = 1.0;
    alpha_low = 0.0;
    alpha_high = inf;
    dg0 = dot(p, g0);
    evals = 0;

    f_new = f0;
    g_new = g0;

    max_ls = min(15, max(1, rem_evals));

    for k = 1:max_ls
        theta_try = theta + alpha * p;
        [f_try, g_try] = loss_grad(theta_try, layers, X, y);
        evals = evals + 1;
        f_new = f_try;
        g_new = g_try;

        dg_try = dot(p, g_try);

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

% -------------------------------------------------------------------------
% ALGORITMO 3: ADAM
% -------------------------------------------------------------------------
function [best_theta, hist] = train_adam(theta0, layers, Xtr, ytr, Xva, yva, opts)
% TRAIN_ADAM Treina a rede com otimizador Adam (beta1=0.9, beta2=0.999, eps=1e-8).
    t_start = tic;
    theta = theta0;
    N = size(Xtr, 1);
    B = opts.batch_size;
    lr = opts.lr;
    max_epochs = opts.max_epochs;

    beta1 = 0.9;
    beta2 = 0.999;
    eps_adam = 1e-8;

    m = zeros(size(theta));
    v = zeros(size(theta));
    t_step = 0;

    rand('state', opts.seed + 1000);

    tr_loss = zeros(max_epochs, 1);
    va_loss = zeros(max_epochs, 1);
    epoch_eq = (1:max_epochs)';

    best_va_loss = inf;
    best_theta = theta;

    for ep = 1:max_epochs
        perm = randperm(N);
        for start_idx = 1:B:N
            end_idx = min(start_idx + B - 1, N);
            batch_idx = perm(start_idx:end_idx);

            t_step = t_step + 1;
            [~, g] = loss_grad(theta, layers, Xtr(batch_idx, :), ytr(batch_idx));

            m = beta1 * m + (1.0 - beta1) * g;
            v = beta2 * v + (1.0 - beta2) * (g .^ 2);

            m_hat = m / (1.0 - beta1 ^ t_step);
            v_hat = v / (1.0 - beta2 ^ t_step);

            theta = theta - lr * m_hat ./ (sqrt(v_hat) + eps_adam);
        end

        y_tr_hat = forward_pass(theta, layers, Xtr);
        y_va_hat = forward_pass(theta, layers, Xva);

        tr_loss(ep) = mean((y_tr_hat - ytr) .^ 2);
        va_loss(ep) = mean((y_va_hat - yva) .^ 2);

        if va_loss(ep) < best_va_loss
            best_va_loss = va_loss(ep);
            best_theta = theta;
        end
    end

    hist.tr_loss      = tr_loss;
    hist.va_loss      = va_loss;
    hist.epoch_eq     = epoch_eq;
    hist.best_va_loss = best_va_loss;
    hist.grad_evals   = max_epochs;
    hist.elapsed_time = toc(t_start);
end