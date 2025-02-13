import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

def load_and_process_shopee_data(directory_path):
    """
    Carrega e processa os arquivos Excel da Shopee de um diretório.
    """
    files = glob.glob(os.path.join(directory_path, "export_report.parentskudetail.*.xlsx"))
    
    all_data = []
    
    for file in files:
        try:
            filename = os.path.basename(file)
            date_str = filename.split('.')[-2].split('_')[0]
            
            if len(date_str) == 8 and date_str.isdigit():
                date = datetime.strptime(date_str, '%Y%m%d')
            else:
                print(f"Pulando arquivo {file} - formato de data inválido")
                continue
            
            df = pd.read_excel(file)
            
            required_columns = ['Produto', 'SKU Principle', 'Nome da Variação', 'SKU da Variação', 
                              'Unidades (Pedido pago)', 'Vendas (Pedido pago) (BRL)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Arquivo {filename} não contém as colunas necessárias: {missing_columns}")
                print("Colunas encontradas:", df.columns.tolist())
                continue
            
            df = df.dropna(subset=['SKU Principle', 'SKU da Variação', 'Unidades (Pedido pago)'])
            df['Unidades (Pedido pago)'] = pd.to_numeric(df['Unidades (Pedido pago)'], errors='coerce')
            df['data'] = date
            
            all_data.append(df)
            print(f"Arquivo processado com sucesso: {filename}")
            
        except Exception as e:
            print(f"Erro ao processar arquivo {file}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("Nenhum arquivo válido encontrado para processar")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    daily_sales = combined_df.groupby(['data', 'Produto', 'SKU Principle', 'Nome da Variação', 'SKU da Variação'])[
        ['Unidades (Pedido pago)', 'Vendas (Pedido pago) (BRL)']
    ].sum().reset_index()
    
    return daily_sales

def prepare_features(data, sku_variation):
    """
    Prepara features para o XGBoost.
    """
    df = data[data['SKU da Variação'] == sku_variation].copy()
    df = df.sort_values('data')
    
    # Extrai características temporais
    df['ano'] = df['data'].dt.year
    df['mes'] = df['data'].dt.month
    df['dia_semana'] = df['data'].dt.dayofweek
    df['dia_mes'] = df['data'].dt.day
    
    # Cria médias móveis
    df['media_7d'] = df['Unidades (Pedido pago)'].rolling(window=7, min_periods=1).mean()
    df['media_30d'] = df['Unidades (Pedido pago)'].rolling(window=30, min_periods=1).mean()
    
    return df

def train_xgboost_model(data, target):
    """
    Treina o modelo XGBoost.
    """
    features = ['ano', 'mes', 'dia_semana', 'dia_mes', 'media_7d', 'media_30d']
    
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror'
    )
    
    model.fit(data[features], target)
    return model

def generate_future_dates(last_date, periods):
    """
    Gera datas futuras para previsão.
    """
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    future_df = pd.DataFrame({'data': dates})
    
    future_df['ano'] = future_df['data'].dt.year
    future_df['mes'] = future_df['data'].dt.month
    future_df['dia_semana'] = future_df['data'].dt.dayofweek
    future_df['dia_mes'] = future_df['data'].dt.day
    
    return future_df

def analyze_recent_trends(data):
    """
    Analisa tendências recentes nas vendas.
    """
    if len(data) < 30:
        return None
        
    # Pega os últimos 60 dias de dados
    recent_data = data.sort_values('data').tail(60)
    
    # Divide em dois períodos de 30 dias
    last_30_days = recent_data.tail(30)['Unidades (Pedido pago)'].mean()
    previous_30_days = recent_data.head(30)['Unidades (Pedido pago)'].mean()
    
    # Calcula a variação percentual
    if previous_30_days > 0:
        percent_change = ((last_30_days - previous_30_days) / previous_30_days) * 100
    else:
        percent_change = 0 if last_30_days == 0 else 100
        
    return {
        'media_ultimos_30_dias': last_30_days,
        'media_30_dias_anteriores': previous_30_days,
        'variacao_percentual': percent_change
    }

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo usando os dados de teste.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    # Calcula o erro percentual médio
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        'mae': mae,
        'mape': mape
    }

def process_sales_data(daily_sales):
    """
    Processa os dados de vendas considerando a hierarquia SKU pai/variação.
    Retorna apenas variações se existirem, caso contrário retorna o SKU pai.
    """
    processed_data = []
    
    for sku in daily_sales['SKU Principle'].unique():
        sku_data = daily_sales[daily_sales['SKU Principle'] == sku].copy()
        
        # Verifica se existem variações válidas (com vendas)
        variations = sku_data[
            (sku_data['SKU da Variação'].notna()) & 
            (sku_data['Unidades (Pedido pago)'] > 0)
        ]
        
        if len(variations) > 0:
            # Se existem variações com vendas, usa apenas as variações
            for variation_sku in variations['SKU da Variação'].unique():
                variation_data = variations[variations['SKU da Variação'] == variation_sku]
                processed_data.append(variation_data)
        else:
            # Se não existem variações com vendas, usa o SKU pai
            sku_data_without_variations = sku_data[
                (sku_data['SKU da Variação'].isna()) | 
                (sku_data['SKU da Variação'] == '')
            ]
            if len(sku_data_without_variations) > 0:
                processed_data.append(sku_data_without_variations)
    
    if not processed_data:
        raise ValueError("Nenhum dado válido encontrado após processamento")
        
    return pd.concat(processed_data, ignore_index=True)

def generate_purchase_recommendations(daily_sales, forecast_periods=30):
    """
    Gera recomendações de compra usando XGBoost.
    """
    # Processa os dados considerando a hierarquia
    processed_sales = process_sales_data(daily_sales)
    recommendations = []
    
    # Itera sobre SKUs únicos (pode ser SKU pai ou variação)
    for sku in processed_sales['SKU Principle'].unique():
        sku_data = processed_sales[processed_sales['SKU Principle'] == sku]
        
        # Verifica se tem variações
        has_variations = sku_data['SKU da Variação'].notna().any()
        
        if has_variations:
            # Processa cada variação
            for variation_sku in sku_data['SKU da Variação'].unique():
                try:
                    variation_data = sku_data[sku_data['SKU da Variação'] == variation_sku]
                    variation_name = variation_data['Nome da Variação'].iloc[0]
                    
                    # Prepara dados para treino
                    train_data = prepare_features(processed_sales, variation_sku)
                    
                    # Analisa tendências recentes se houver dados suficientes
                    trends = analyze_recent_trends(train_data) if len(train_data) >= 60 else None
                    
                    # Separa dados para teste se houver dados suficientes
                    if len(train_data) >= 30:
                        test_data = train_data.tail(30)
                        train_subset = train_data.iloc[:-30]
                    else:
                        test_data = train_data.tail(max(1, len(train_data) // 4))
                        train_subset = train_data.iloc[:-len(test_data)]
                    
                    features = ['ano', 'mes', 'dia_semana', 'dia_mes', 'media_7d', 'media_30d']
                    
                    # Treina modelo
                    model = train_xgboost_model(
                        train_subset if len(train_subset) > 0 else train_data,
                        train_subset['Unidades (Pedido pago)'] if len(train_subset) > 0 else train_data['Unidades (Pedido pago)']
                    )
                    
                    # Resto do processamento para variações...
                    recommendations.extend(
                        process_predictions(
                            model, train_data, test_data, features, trends,
                            sku, variation_data['Produto'].iloc[0], variation_name, variation_sku,
                            forecast_periods
                        )
                    )
                    
                except Exception as e:
                    print(f"Erro ao processar Variação {variation_sku} do SKU {sku}: {str(e)}")
                    continue
        else:
            # Processa SKU pai diretamente
            try:
                train_data = prepare_features(processed_sales, sku)
                trends = analyze_recent_trends(train_data) if len(train_data) >= 60 else None
                
                if len(train_data) >= 30:
                    test_data = train_data.tail(30)
                    train_subset = train_data.iloc[:-30]
                else:
                    test_data = train_data.tail(max(1, len(train_data) // 4))
                    train_subset = train_data.iloc[:-len(test_data)]
                
                features = ['ano', 'mes', 'dia_semana', 'dia_mes', 'media_7d', 'media_30d']
                
                model = train_xgboost_model(
                    train_subset if len(train_subset) > 0 else train_data,
                    train_subset['Unidades (Pedido pago)'] if len(train_subset) > 0 else train_data['Unidades (Pedido pago)']
                )
                
                recommendations.extend(
                    process_predictions(
                        model, train_data, test_data, features, trends,
                        sku, sku_data['Produto'].iloc[0], None, None,
                        forecast_periods
                    )
                )
                
            except Exception as e:
                print(f"Erro ao processar SKU pai {sku}: {str(e)}")
                continue
    
    if not recommendations:
        raise ValueError("Nenhum SKU/variação teve dados suficientes para gerar previsões")
    
    # Converte para DataFrame e ordena por previsão de vendas diária (decrescente)
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df['Previsão de Vendas Diária'] = recommendations_df['Previsão de Vendas Mensal'] / 30
    recommendations_df = recommendations_df.sort_values(
        by='Previsão de Vendas Diária', 
        ascending=False
    ).drop('Previsão de Vendas Diária', axis=1)
    
    return recommendations_df

def process_predictions(model, train_data, test_data, features, trends, sku, produto, variation_name, variation_sku, forecast_periods):
    """
    Processa as previsões e gera recomendações para um SKU/variação específico.
    """
    recommendations = []
    
    # Avalia modelo nos dados de teste
    model_metrics = evaluate_model(model, test_data[features], test_data['Unidades (Pedido pago)']) if len(test_data) > 0 else {'mae': 0, 'mape': 0}
    
    # Prepara dados futuros
    last_date = train_data['data'].max()
    future_dates = generate_future_dates(last_date, forecast_periods)
    
    # Adiciona médias móveis aos dados futuros
    future_dates['media_7d'] = train_data['Unidades (Pedido pago)'].tail(min(7, len(train_data))).mean()
    future_dates['media_30d'] = train_data['Unidades (Pedido pago)'].tail(min(30, len(train_data))).mean()
    
    # Faz previsões
    predictions = model.predict(future_dates[features])
    predictions = np.maximum(predictions, 0)
    
    # Ajusta previsões com base na tendência recente
    if trends and trends['variacao_percentual'] > 10:
        adjustment_factor = 1 + (trends['variacao_percentual'] / 100)
        predictions = predictions * adjustment_factor
    
    # Calcula estatísticas históricas
    historical_mean = train_data['Unidades (Pedido pago)'].mean()
    historical_std = train_data['Unidades (Pedido pago)'].std()
    
    # Agrupa previsões por mês
    future_dates['predictions'] = predictions
    monthly_forecast = future_dates.set_index('data').resample('ME')['predictions'].mean()
    
    for month, predicted_value in monthly_forecast.items():
        # Adiciona verificação para evitar previsões muito baixas
        if predicted_value < (historical_mean * 30 * 0.5):  # Se previsão for menor que 50% da média
            # Verifica se há tendência de queda nos últimos dados
            recent_data = train_data.tail(30)  # Últimos 30 dias
            recent_mean = recent_data['Unidades (Pedido pago)'].mean() * 30
            
            if recent_mean > (predicted_value * 1.5):  # Se média recente também for maior
                # Usa a média histórica como base, com pequena redução
                predicted_value = historical_mean * 30 * 0.9
        
        # Ajusta para dezembro se necessário
        if month.month == 12:
            predicted_value *= 0.7
        
        predicted_value = min(predicted_value, (historical_mean + 3 * historical_std) * 30)
        predicted_value = max(predicted_value, historical_mean * 30 * 0.5)  # Mínimo de 50% da média
        
        lower_bound = max(0, predicted_value * 0.7)
        upper_bound = predicted_value * 1.3
        suggestion = round(predicted_value * 1.2)
        
        # Dentro do loop de processamento
        # Primeiro calcula a média diária
        historical_mean_daily = variation_data['Unidades (Pedido pago)'].mean()
        historical_std_daily = variation_data['Unidades (Pedido pago)'].std()
        
        # Converte para valores mensais
        historical_mean_monthly = historical_mean_daily * 30
        historical_std_monthly = historical_std_daily * 30
        
        # Pega previsão diária do XGBoost
        predictions_daily = model.predict(future_dates[['ano', 'mes', 'dia_semana', 'dia_mes', 'media_7d', 'media_30d']])
        predicted_daily = float(np.mean(predictions_daily))
        
        # Converte para mensal
        predicted_monthly = predicted_daily * 30
        
        # Verifica se a previsão mensal está muito baixa
        if predicted_monthly < (historical_mean_monthly * 0.5):
            recent_data = train_data.tail(30)
            recent_mean_monthly = recent_data['Unidades (Pedido pago)'].mean() * 30
            
            if recent_mean_monthly > (predicted_monthly * 1.5):
                predicted_monthly = historical_mean_monthly * 0.9
        
        # Ajusta para dezembro se necessário
        if month.month == 12:
            predicted_monthly *= 0.7
        
        # Aplica limites nos valores mensais
        predicted_monthly = min(predicted_monthly, historical_mean_monthly + 3 * historical_std_monthly)
        predicted_monthly = max(predicted_monthly, historical_mean_monthly * 0.5)
        
        # Calcula limites e sugestão com base nos valores mensais
        lower_bound_monthly = max(0, predicted_monthly * 0.7)
        upper_bound_monthly = predicted_monthly * 1.3
        suggestion_monthly = round(predicted_monthly * 1.2)
        
        recommendations.append({
            'SKU Principal': sku,
            'Nome do Produto': produto,
            'Nome da Variação': variation_name if variation_name else 'N/A',
            'SKU da Variação': variation_sku if variation_sku else 'N/A',
            'Mês': month.strftime('%Y-%m'),
            'Média Histórica': round(historical_mean, 2),
            'Média Últimos 30 Dias': round(trends['media_ultimos_30_dias'], 2) if trends else round(historical_mean, 2),
            'Variação Últimos 60 Dias (%)': round(trends['variacao_percentual'], 2) if trends else 0,
            'Previsão de Vendas': round(predicted_value),
            'Previsão Mínima': round(lower_bound),
            'Previsão Máxima': round(upper_bound),
            'Sugestão de Compra': suggestion,
            'Erro Médio Absoluto': round(model_metrics['mae'], 2),
            'Erro Percentual Médio (%)': round(model_metrics['mape'], 2),
            'Dias de Histórico': len(train_data),
            'Média Histórica Mensal': round(historical_mean_monthly, 2),
            'Média Histórica Diária': round(historical_mean_daily, 2),
            'Previsão de Vendas Mensal': round(predicted_monthly),
            'Previsão Mínima Mensal': round(lower_bound_monthly),
            'Previsão Máxima Mensal': round(upper_bound_monthly),
            'Sugestão de Compra Mensal': suggestion_monthly,
            'Observação': 'Previsão XGBoost - Valores mensais'
        })
    
    return recommendations

def main(directory_path):
    """
    Função principal que executa todo o processo.
    """
    print("Carregando e processando dados...")
    daily_sales = load_and_process_shopee_data(directory_path)
    
    print("Gerando previsões e recomendações com XGBoost...")
    recommendations = generate_purchase_recommendations(daily_sales)
    
    output_file = 'previsao_compras_shopee_xgboost.xlsx'
    recommendations.to_excel(output_file, index=False)
    print(f"Recomendações salvas em {output_file}")
    
    return recommendations

if __name__ == "__main__":
    directory_path = "."  # ou especifique o caminho completo dos seus arquivos
    recommendations = main(directory_path) 