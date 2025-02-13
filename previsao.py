import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import glob
import os

def load_and_process_shopee_data(directory_path):
    """
    Carrega e processa os arquivos Excel da Shopee de um diretório.
    """
    files = glob.glob(os.path.join(directory_path, "export_report.parentskudetail.*.xlsx"))
    
    all_data = []
    
    for file in files:
        try:
            # Extrai a data do nome do arquivo
            filename = os.path.basename(file)
            date_str = filename.split('.')[-2].split('_')[0]
            
            if len(date_str) == 8 and date_str.isdigit():
                date = datetime.strptime(date_str, '%Y%m%d')
            else:
                print(f"Pulando arquivo {file} - formato de data inválido")
                continue
            
            # Lê o arquivo Excel
            df = pd.read_excel(file)
            
            # Verifica se as colunas necessárias existem
            required_columns = ['Produto', 'SKU Principle', 'Nome da Variação', 'SKU da Variação', 'Unidades (Pedido pago)', 'Vendas (Pedido pago) (BRL)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Arquivo {filename} não contém as colunas necessárias: {missing_columns}")
                print("Colunas encontradas:", df.columns.tolist())
                continue
            
            # Remove linhas com valores nulos nas colunas importantes
            df = df.dropna(subset=['SKU Principle', 'SKU da Variação', 'Unidades (Pedido pago)'])
            
            # Converte 'Unidades (Pedido pago)' para numérico
            df['Unidades (Pedido pago)'] = pd.to_numeric(df['Unidades (Pedido pago)'], errors='coerce')
            
            # Adiciona a coluna de data
            df['data'] = date
            
            all_data.append(df)
            print(f"Arquivo processado com sucesso: {filename}")
            print(f"Número de linhas válidas: {len(df)}")
            
        except Exception as e:
            print(f"Erro ao processar arquivo {file}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("Nenhum arquivo válido encontrado para processar")
    
    # Combina todos os dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Agrupa por produto, SKU principal, variação e data
    daily_sales = combined_df.groupby(['data', 'Produto', 'SKU Principle', 'Nome da Variação', 'SKU da Variação'])[
        ['Unidades (Pedido pago)', 'Vendas (Pedido pago) (BRL)']
    ].sum().reset_index()
    
    # Imprime informações sobre os dados processados
    print("\nResumo dos dados processados:")
    print(f"Total de SKUs únicos: {len(daily_sales['SKU Principle'].unique())}")
    print(f"Total de variações únicas: {len(daily_sales['SKU da Variação'].unique())}")
    print("\nQuantidade de dias por SKU e variação:")
    print(daily_sales.groupby(['SKU Principle', 'Nome da Variação']).size().sort_values(ascending=False))
    
    return daily_sales

def prepare_prophet_data(daily_sales, sku):
    """
    Prepara os dados para o modelo Prophet para um SKU específico.
    """
    sku_data = daily_sales[daily_sales['SKU Principle'] == sku].copy()
    
    # Prepara o formato para o Prophet
    prophet_data = pd.DataFrame({
        'ds': sku_data['data'],
        'y': sku_data['Unidades (Pedido pago)']
    })
    
    return prophet_data

def create_forecast(prophet_data, forecast_periods):
    """
    Cria previsão usando o Prophet com configurações ajustadas para sazonalidade.
    """
    # Configura o modelo Prophet com parâmetros mais conservadores
    model = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Modo multiplicativo para sazonalidade
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # Adiciona sazonalidade mensal personalizada
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )
    
    # Treina o modelo
    model.fit(prophet_data)
    
    # Cria datas futuras para previsão
    future = model.make_future_dataframe(periods=forecast_periods)
    
    # Faz a previsão
    forecast = model.predict(future)
    
    # Ajusta previsões para dezembro (redução de 30%)
    forecast.loc[forecast['ds'].dt.month == 12, 'yhat'] *= 0.7
    forecast.loc[forecast['ds'].dt.month == 12, 'yhat_lower'] *= 0.7
    forecast.loc[forecast['ds'].dt.month == 12, 'yhat_upper'] *= 0.7
    
    # Garante que não há valores negativos
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    return forecast

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
    Gera recomendações de compra usando Prophet apenas para o próximo mês.
    """
    processed_sales = process_sales_data(daily_sales)
    recommendations = []
    
    for sku in processed_sales['SKU Principle'].unique():
        sku_data = processed_sales[processed_sales['SKU Principle'] == sku]
        has_variations = sku_data['SKU da Variação'].notna().any()
        
        if has_variations:
            for variation_sku in sku_data['SKU da Variação'].unique():
                try:
                    variation_data = sku_data[sku_data['SKU da Variação'] == variation_sku]
                    variation_name = variation_data['Nome da Variação'].iloc[0]
                    
                    # Primeiro calcula a média diária
                    historical_mean_daily = variation_data['Unidades (Pedido pago)'].mean()
                    historical_std_daily = variation_data['Unidades (Pedido pago)'].std()
                    
                    # Converte para valores mensais
                    historical_mean_monthly = historical_mean_daily * 30
                    historical_std_monthly = historical_std_daily * 30
                    
                    prophet_data = pd.DataFrame({
                        'ds': variation_data['data'],
                        'y': variation_data['Unidades (Pedido pago)']
                    }).dropna()
                    
                    forecast = create_forecast(prophet_data, forecast_periods)
                    
                    # Pega apenas o próximo mês
                    last_date = prophet_data['ds'].max()
                    next_month_start = (last_date + pd.DateOffset(months=1)).replace(day=1)
                    next_month_end = (next_month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
                    
                    # Filtra previsões apenas para o próximo mês
                    next_month_forecast = forecast[
                        (forecast['ds'] >= next_month_start) & 
                        (forecast['ds'] <= next_month_end)
                    ]
                    
                    # Pega previsão diária do Prophet
                    predicted_daily = next_month_forecast['yhat'].mean()
                    
                    # Converte para mensal
                    predicted_monthly = predicted_daily * 30
                    
                    # Verifica se a previsão mensal está muito baixa
                    if predicted_monthly < (historical_mean_monthly * 0.5):
                        recent_data = prophet_data.tail(30)
                        recent_mean_monthly = recent_data['y'].mean() * 30
                        
                        if recent_mean_monthly > (predicted_monthly * 1.5):
                            predicted_monthly = historical_mean_monthly * 0.9
                    
                    # Ajusta para dezembro se necessário
                    if next_month_start.month == 12:
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
                        'Nome do Produto': variation_data['Produto'].iloc[0],
                        'Nome da Variação': variation_name,
                        'SKU da Variação': variation_sku,
                        'Mês': next_month_start.strftime('%Y-%m'),
                        'Média Histórica Mensal': round(historical_mean_monthly, 2),
                        'Média Histórica Diária': round(historical_mean_daily, 2),
                        'Previsão de Vendas Mensal': round(predicted_monthly),
                        'Previsão Mínima Mensal': round(lower_bound_monthly),
                        'Previsão Máxima Mensal': round(upper_bound_monthly),
                        'Sugestão de Compra Mensal': suggestion_monthly,
                        'Dias de Histórico': len(prophet_data),
                        'Observação': 'Previsão normal - Valores mensais'
                    })
                    
                except Exception as e:
                    print(f"Erro ao processar Variação {variation_sku} do SKU {sku}: {str(e)}")
                    continue
        else:
            try:
                prophet_data = pd.DataFrame({
                    'ds': sku_data['data'],
                    'y': sku_data['Unidades (Pedido pago)']
                })
                
                historical_mean = sku_data['Unidades (Pedido pago)'].mean()
                historical_std = sku_data['Unidades (Pedido pago)'].std()
                
                forecast = create_forecast(prophet_data, forecast_periods)
                
                # Pega apenas o próximo mês
                last_date = prophet_data['ds'].max()
                next_month_start = (last_date + pd.DateOffset(months=1)).replace(day=1)
                next_month_end = (next_month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
                
                # Filtra previsões apenas para o próximo mês
                next_month_forecast = forecast[
                    (forecast['ds'] >= next_month_start) & 
                    (forecast['ds'] <= next_month_end)
                ]
                
                # Calcula média mensal
                predicted_value = next_month_forecast['yhat'].mean()
                
                # Adiciona verificação para evitar previsões muito baixas
                if predicted_value < (historical_mean * 0.5):  # Se previsão for menor que 50% da média
                    # Verifica se há tendência de queda nos últimos dados
                    recent_data = prophet_data.tail(30)  # Últimos 30 dias
                    recent_mean = recent_data['y'].mean()
                    
                    if recent_mean > (predicted_value * 1.5):  # Se média recente também for maior
                        # Usa a média histórica como base, com pequena redução
                        predicted_value = historical_mean * 0.9
                
                # Ajusta para dezembro se necessário
                if next_month_start.month == 12:
                    predicted_value *= 0.7
                
                predicted_value = min(predicted_value, historical_mean + 3 * historical_std)
                predicted_value = max(predicted_value, historical_mean * 0.5)  # Mínimo de 50% da média
                
                lower_bound = max(0, predicted_value * 0.7)
                upper_bound = predicted_value * 1.3
                suggestion = round(predicted_value * 1.2)
                
                recommendations.append({
                    'SKU Principal': sku,
                    'Nome do Produto': sku_data['Produto'].iloc[0],
                    'Nome da Variação': 'N/A',
                    'SKU da Variação': 'N/A',
                    'Mês': next_month_start.strftime('%Y-%m'),
                    'Média Histórica': round(historical_mean, 2),
                    'Previsão de Vendas': round(predicted_value),
                    'Previsão Mínima': round(lower_bound),
                    'Previsão Máxima': round(upper_bound),
                    'Sugestão de Compra': suggestion,
                    'Dias de Histórico': len(sku_data)
                })
                
            except Exception as e:
                print(f"Erro ao processar SKU pai {sku}: {str(e)}")
                continue
    
    if not recommendations:
        raise ValueError("Nenhum SKU/variação teve dados suficientes para gerar previsões")
    

    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df['Previsão de Vendas Diária'] = recommendations_df['Previsão de Vendas Mensal'] / 30
    recommendations_df = recommendations_df.sort_values(
        by='Previsão de Vendas Diária', 
        ascending=False
    ).drop('Previsão de Vendas Diária', axis=1)
    
    return recommendations_df

def main(directory_path):
    """
    Função principal que executa todo o processo.
    """
    # Verifica se o diretório existe
    if not os.path.exists(directory_path):
        print(f"Erro: O diretório {directory_path} não existe!")
        return None
    
    print("Carregando e processando dados...")
    # Verifica se existem arquivos para processar
    files = glob.glob(os.path.join(directory_path, "export_report.parentskudetail.*.xlsx"))
    if not files:
        print(f"Erro: Nenhum arquivo encontrado em {directory_path}")
        print("Os arquivos devem seguir o padrão: export_report.parentskudetail.YYYYMMDD_YYYYMMDD.xlsx")
        return None
        
    daily_sales = load_and_process_shopee_data(directory_path)
    
    print("Gerando previsões e recomendações...")
    recommendations = generate_purchase_recommendations(daily_sales)
    
    # Salva as recomendações em um arquivo Excel
    output_file = 'previsao_compras_shopee.xlsx'
    recommendations.to_excel(output_file, index=False)
    print(f"Recomendações salvas em {output_file}")
    
    return recommendations

if __name__ == "__main__":
    # Usa o diretório atual como padrão
    directory_path = "C:/Users/natal/OneDrive/Área de Trabalho/previsão de pedidos"  
    recommendations = main(directory_path)