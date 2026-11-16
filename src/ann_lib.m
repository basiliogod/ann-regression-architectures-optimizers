% ANN_LIB Biblioteca de funções para o Assignment 2 de Redes Neuronais Artificiais.
function L = ann_lib()
    % Aquisição e inspeção inicial de dados
    L.load_concrete  = @load_concrete;
    L.load_melbourne = @load_melbourne;
    L.inspect_data   = @inspect_data;

    L.compute_acf    = @compute_acf;
    L.eda_concrete   = @eda_concrete;
    L.eda_melbourne  = @eda_melbourne;
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
    h1 = figure('Visible', 'off', 'Position', [100, 100, 1100, 850]);
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
    h2 = figure('Visible', 'off', 'Position', [100, 100, 800, 680]);
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
    h3 = figure('Visible', 'off', 'Position', [100, 100, 1200, 580]);
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
    h1 = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);

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
    h2 = figure('Visible', 'off', 'Position', [100, 100, 1200, 420]);

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