% ANN_LIB Biblioteca de funções para o Assignment 2 de Redes Neuronais Artificiais.
function L = ann_lib()
    % Aquisição e inspeção inicial de dados
    L.load_concrete  = @load_concrete;
    L.load_melbourne = @load_melbourne;
    L.inspect_data   = @inspect_data;
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