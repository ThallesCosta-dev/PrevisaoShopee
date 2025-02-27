import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import glob
import os
import re
from xgboost import XGBRegressor

def load_and_preprocess_data(directory_path):
    """
    Carrega e processa os arquivos Excel da Shopee de um diretório.
    Retorna o DataFrame limpo e as métricas dos produtos.
    """
    files = glob.glob(os.path.join(directory_path, "export_report.parentskudetail.*.xlsx"))
    
    all_data = []
    
    for file in files:
        try:
            # Extrai a data do nome do arquivo
            filename = os.path.basename(file)
            # Procura por um padrão de 8 dígitos no nome do arquivo
            date_match = re.search(r'(\d{8})_', filename)
            
            if date_match:
                date_str = date_match.group(1)
                date = datetime.strptime(date_str, '%Y%m%d')
            else:
                print(f"Pulando arquivo {file} - não foi possível extrair a data do nome")
                continue
            
            # Lê o arquivo Excel
            df = pd.read_excel(file)
            
            # Verifica se as colunas necessárias existem
            required_columns = ['Produto', 'SKU Principle', 'Nome da Variação', 
                              'SKU da Variação', 'Unidades (Pedido pago)']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Arquivo {filename} não contém as colunas necessárias: {missing_columns}")
                print("Colunas encontradas:", df.columns.tolist())
                continue
            
            # Remove linhas com valores nulos nas colunas importantes
            df = df.dropna(subset=['SKU Principle', 'Unidades (Pedido pago)'])
            
            # Adiciona a coluna de data
            df['data'] = date
            
            all_data.append(df)
            print(f"Arquivo processado com sucesso: {filename}")
            
        except Exception as e:
            print(f"Erro ao processar arquivo {file}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("Nenhum arquivo válido encontrado para processar")
    
    # Combina todos os dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Agrupa por produto e data
    daily_sales = combined_df.groupby(['data', 'Produto', 'SKU Principle', 
                                     'Nome da Variação', 'SKU da Variação'])[
        'Unidades (Pedido pago)'
    ].sum().reset_index()
    
    # Calcula métricas por produto
    product_metrics = daily_sales.groupby('SKU Principle').agg({
        'Unidades (Pedido pago)': ['mean', 'std', 'count']
    }).reset_index()
    
    product_metrics.columns = ['SKU Principle', 'media_vendas', 'desvio_padrao', 'dias_dados']
    
    return daily_sales, product_metrics

def create_training_data(df):
    """
    Cria dados de treino com janela deslizante
    """
    window_size = 7  # Janela de 7 dias para previsão
    X = []
    y = []
    
    for i in range(len(df) - window_size):
        window = df.iloc[i:i+window_size]
        target = df.iloc[i+window_size]['Unidades (Pedido realizado)']
        
        # Features para o modelo
        features = window.drop(['Data', 'Unidades (Pedido realizado)', 'ID do Item', 'Produto', 
                              'Vendas (Pedido realizado) (BRL)', 'Ranking'], axis=1).values.flatten()
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y)

def train_model(df_clean):
    """
    Treina o modelo XGBoost com os dados históricos.
    """
    try:
        # Prepara os dados para treinamento
        df_train = df_clean.copy()
        
        # Adiciona features temporais
        df_train['ano'] = df_train['data'].dt.year
        df_train['mes'] = df_train['data'].dt.month
        df_train['dia_semana'] = df_train['data'].dt.dayofweek
        df_train['dia_mes'] = df_train['data'].dt.day
        
        # Calcula médias móveis
        df_train['media_7d'] = df_train.groupby(['SKU Principle', 'SKU da Variação'])[
            'Unidades (Pedido pago)'
        ].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        
        df_train['media_30d'] = df_train.groupby(['SKU Principle', 'SKU da Variação'])[
            'Unidades (Pedido pago)'
        ].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
        
        # Features para o modelo
        features = ['ano', 'mes', 'dia_semana', 'dia_mes', 'media_7d', 'media_30d']
        target = 'Unidades (Pedido pago)'
        
        # Treina o modelo
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        X = df_train[features]
        y = df_train[target]
        
        model.fit(X, y)
        
        print("Modelo treinado com sucesso!")
        return model
        
    except Exception as e:
        print(f"Erro ao treinar modelo: {str(e)}")
        raise

def make_predictions(model, df, days_ahead=30):
    """
    Faz previsões para os próximos dias
    """
    last_window = df.iloc[-7:]
    predictions = []
    
    for _ in range(days_ahead):
        # Prepara features para previsão
        features = last_window.drop(['Data', 'Unidades (Pedido realizado)', 'ID do Item', 'Produto',
                                   'Vendas (Pedido realizado) (BRL)', 'Ranking'], axis=1).values.flatten()
        
        # Faz a previsão
        pred = model.predict([features])[0]
        predictions.append(pred)
        
        # Atualiza a janela para a próxima previsão
        new_row = last_window.iloc[-1:].copy()
        new_row['Data'] = new_row['Data'] + timedelta(days=1)
        new_row['Unidades (Pedido realizado)'] = pred
        last_window = pd.concat([last_window[1:], new_row])
    
    return predictions

def save_reports(df_clean, product_metrics, forecast_df, output_path):
    """
    Salva relatórios em Excel
    """
    # Cria um escritor Excel
    with pd.ExcelWriter(os.path.join(output_path, 'relatorio_vendas.xlsx')) as writer:
        # Salva métricas gerais dos produtos
        product_metrics.to_excel(writer, sheet_name='Ranking_Produtos', index=False)
        
        # Salva dados históricos limpos
        df_clean.to_excel(writer, sheet_name='Dados_Historicos', index=False)
        
        # Salva previsões
        forecast_df.to_excel(writer, sheet_name='Previsoes', index=False)

def process_files(file_paths):
    dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path)
            
            # Filtra apenas arquivos que terminam com data (YYYYMMDD_YYYYMMDD.xlsx)
            if not re.search(r'\d{8}_\d{8}\.xlsx$', file_path):
                continue
            
            # Renomeia as colunas para padronizar
            df = df.rename(columns={
                'SKU Principle': 'produto_id',
                'Unidades (Pedido pago)': 'quantidade'
            })
            
            # Verifica se o renomeamento funcionou
            if 'produto_id' not in df.columns or 'quantidade' not in df.columns:
                print(f"\nErro no renomeamento das colunas do arquivo {os.path.basename(file_path)}")
                print("Colunas antes do renomeamento:", df.columns.tolist())
                continue
            
            # Seleciona apenas as colunas necessárias
            df = df[['produto_id', 'quantidade']]
            
            dfs.append(df)
            print(f"Arquivo processado com sucesso: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"Erro ao processar arquivo {file_path}: {str(e)}")
    
    if not dfs:
        raise ValueError("Nenhum arquivo foi processado com sucesso")
        
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Verifica o DataFrame final
    print("\nColunas no DataFrame final:", df_final.columns.tolist())
    return df_final

def generate_forecast(df_clean):
    """
    Gera previsões usando o modelo de Machine Learning com limites de validação.
    """
    try:
        print("\nVerificando colunas para previsão...")
        print("Colunas disponíveis:", df_clean.columns.tolist())
        
        # Verifica se as colunas existem
        required_columns = ['produto_id', 'quantidade']
        if not all(col in df_clean.columns for col in required_columns):
            missing_columns = [col for col in required_columns if col not in df_clean.columns]
            print(f"Colunas faltantes: {missing_columns}")
            raise ValueError("Colunas necessárias não encontradas no DataFrame")
            
        # Prepara os dados para previsão
        df_forecast = df_clean.copy()
        
        # Agrupa por SKU e data, mantendo o nome do produto
        df_grouped = df_forecast.groupby(['data', 'SKU Principle', 'Produto', 'Nome da Variação', 'SKU da Variação'])[
            'Unidades (Pedido pago)'
        ].sum().reset_index()
        
        # Adiciona features temporais
        df_grouped['ano'] = df_grouped['data'].dt.year
        df_grouped['mes'] = df_grouped['data'].dt.month
        df_grouped['dia_semana'] = df_grouped['data'].dt.dayofweek
        df_grouped['dia_mes'] = df_grouped['data'].dt.day
        
        # Calcula médias móveis
        df_grouped['media_7d'] = df_grouped.groupby(['SKU Principle', 'SKU da Variação'])[
            'Unidades (Pedido pago)'
        ].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        
        df_grouped['media_30d'] = df_grouped.groupby(['SKU Principle', 'SKU da Variação'])[
            'Unidades (Pedido pago)'
        ].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
        
        # Features para o modelo
        features = ['ano', 'mes', 'dia_semana', 'dia_mes', 'media_7d', 'media_30d']
        target = 'Unidades (Pedido pago)'
        
        # Treina um modelo para cada SKU/variação
        forecasts = []
        
        for (sku, var_sku) in df_grouped.groupby(['SKU Principle', 'SKU da Variação']):
            if len(var_sku) < 30:  # Mínimo de 30 dias de dados
                continue
                
            # Treina o modelo
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            X = var_sku[features]
            y = var_sku[target]
            
            model.fit(X, y)
            
            # Gera previsão para o próximo mês
            last_date = var_sku['data'].max()
            next_month = last_date + pd.DateOffset(months=1)
            next_month_start = next_month.replace(day=1)
            next_month_end = (next_month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
            
            # Gera datas para o próximo mês
            future_dates = pd.date_range(start=next_month_start, end=next_month_end, freq='D')
            future_df = pd.DataFrame({'data': future_dates})
            
            # Adiciona features para previsão
            future_df['ano'] = future_df['data'].dt.year
            future_df['mes'] = future_df['data'].dt.month
            future_df['dia_semana'] = future_df['data'].dt.dayofweek
            future_df['dia_mes'] = future_df['data'].dt.day
            
            # Usa as últimas médias móveis conhecidas
            future_df['media_7d'] = var_sku['media_7d'].iloc[-1]
            future_df['media_30d'] = var_sku['media_30d'].iloc[-1]
            
            # Adiciona validações na previsão
            # Calcula estatísticas históricas
            media_historica = var_sku[target].mean() * 30
            desvio_padrao = var_sku[target].std() * 30
            
            # Ajusta limites de validação
            max_forecast = min(
                media_historica + 1.5 * desvio_padrao,  # Reduzido de 2.0 para 1.5
                media_historica * 2  # Limite máximo de 2x a média
            )
            
            # Faz a previsão
            predictions = model.predict(future_df[features])
            monthly_forecast = predictions.mean() * len(future_dates)
            
            # Limita a previsão ao máximo aceitável
            monthly_forecast = min(monthly_forecast, max_forecast)
            
            # Análise de tendência
            df_produto = df_clean[df_clean['produto_id'] == sku[0]].copy()
            df_produto['data'] = pd.to_datetime(df_produto['data'])
            df_produto = df_produto.sort_values('data')
            
            # Calcula tendência dos últimos 3 meses
            ultimos_3_meses = df_produto.tail(90)
            if len(ultimos_3_meses) > 30:  # Precisamos de pelo menos 1 mês de dados
                tendencia = np.polyfit(range(len(ultimos_3_meses)), ultimos_3_meses['quantidade'], 1)[0]
                fator_tendencia = max(0.8, min(1.2, 1 + (tendencia / media_historica)))
            else:
                fator_tendencia = 1.0
            
            # Ajusta previsão considerando a tendência
            monthly_forecast = monthly_forecast * fator_tendencia
            
            # Aplica limites após ajuste de tendência
            max_forecast = min(
                media_historica + 1.5 * desvio_padrao,
                media_historica * 2
            )
            
            monthly_forecast = min(monthly_forecast, max_forecast)
            
            # Adiciona à lista de previsões
            forecasts.append({
                'SKU Principal': sku[0],
                'Nome do Produto': var_sku['Produto'].iloc[0],
                'SKU da Variação': sku[1],
                'Nome da Variação': var_sku['Nome da Variação'].iloc[0],
                'Mês Previsão': next_month_start.strftime('%Y-%m'),
                'Previsão Mensal': round(monthly_forecast, 2),
                'Média Histórica': round(media_historica, 2),
                'Desvio Padrão': round(desvio_padrao, 2),
                'Dias de Histórico': len(var_sku)
            })
        
        # Converte para DataFrame e ordena por previsão mensal
        forecast_df = pd.DataFrame(forecasts)
        forecast_df = forecast_df.sort_values('Previsão Mensal', ascending=False)
        
        return forecast_df
        
    except Exception as e:
        print(f"Erro ao gerar previsões: {str(e)}")
        raise

def main(directory_path):
    """
    Função principal que executa todo o processo.
    """
    try:
        print(f"Diretório de busca: {directory_path}")
        files = glob.glob(os.path.join(directory_path, "export_report.parentskudetail.*.xlsx"))
        print(f"Arquivos encontrados: {len(files)}")
        for file in files:
            print(f"Arquivo: {file}")
        
        if not files:
            print("ERRO: Nenhum arquivo Excel encontrado no padrão 'export_report.parentskudetail.*.xlsx'")
            return
        
        print("\nCarregando e processando dados...")
        df_clean, product_metrics = load_and_preprocess_data(directory_path)
        
        print("Gerando previsões...")
        forecast_df = generate_forecast(df_clean)
        
        print("Treinando modelo...")
        model = train_model(df_clean)
        
        # Salva os resultados
        output_file = 'previsao_vendas_ML.xlsx'
        forecast_df.to_excel(output_file, index=False)
        print(f"Previsões salvas em {output_file}")
        
        return df_clean, product_metrics, forecast_df, model
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    # Usa o diretório atual como padrão
    directory_path = "C:/Users/HQZ/Desktop/Thalles/previsão de pedidos/"
    
    print(f"Iniciando processamento no diretório: {directory_path}")
    if not os.path.exists(directory_path):
        print(f"ERRO: Diretório não encontrado: {directory_path}")
    else:
        main(directory_path)