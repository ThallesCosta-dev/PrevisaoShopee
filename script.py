import pandas as pd
from datetime import datetime
import glob
import re

def analisar_vendas_periodo(periodo_inicial, periodo_final, lista_skus):
    # Procura por arquivos no diretório atual
    arquivos = glob.glob("export_report.parentskudetail.*_*")
    
    # Inicializa DataFrame vazio
    df_total = pd.DataFrame()
    arquivos_processados = []
    
    for arquivo in arquivos:
        # Extrai as datas do nome do arquivo
        padrao = r'parentskudetail.(\d{8})_(\d{8})'
        match = re.search(padrao, arquivo)
        
        if match:
            data_arquivo_inicial = datetime.strptime(match.group(1), '%Y%m%d')
            data_arquivo_final = datetime.strptime(match.group(2), '%Y%m%d')
            
            # Verifica se o arquivo está dentro do período desejado
            if (data_arquivo_inicial >= periodo_inicial and 
                data_arquivo_final <= periodo_final):
                
                try:
                    df = pd.read_excel(arquivo)
                    arquivos_processados.append(arquivo)
                    df_total = pd.concat([df_total, df], ignore_index=True)
                    
                except Exception as e:
                    print(f"Erro ao processar arquivo {arquivo}: {str(e)}")
    
    if not df_total.empty:
        # Filtra por SKUs selecionados (tanto variação quanto principle)
        mascara = (df_total['SKU da Variação'].isin(lista_skus)) | (df_total['SKU Principle'].isin(lista_skus))
        df_filtrado = df_total[mascara]
        
        # Análise por SKU da Variação
        analise_variacao = {
            'quantidade_vendida': df_filtrado.groupby('SKU da Variação')['Unidades (Pedido realizado)'].sum(),
            'valor_total': df_filtrado.groupby('SKU da Variação')['Vendas (Pedido pago) (BRL)'].sum(),
        }
        
        # Análise por SKU Principle
        analise_principle = {
            'quantidade_vendida': df_filtrado.groupby('SKU Principle')['Unidades (Pedido realizado)'].sum(),
            'valor_total': df_filtrado.groupby('SKU Principle')['Vendas (Pedido pago) (BRL)'].sum(),
        }
        
        return {
            'variacao': analise_variacao,
            'principle': analise_principle,
            'arquivos_processados': arquivos_processados
        }
    
    return None

# Exemplo de uso
if __name__ == "__main__":
    # Define período de análise
    periodo_inicial = datetime(2025, 3, 18)
    periodo_final = datetime(2025, 3, 25)
    
    # Lista de SKUs para análise
    skus_desejados = ['DN445-2', 'DN33-1', 'DN528', 'DN413-2', 'DN390-1', 'DN38-1', 'DN38-5', 'DN16-1', 'DN16-4', 'DN527', 'DN38-6', 'DN38-7', 'DN390-6', 'DN505-1', 'DN505-2', 'DN505-3', 'DN505-4']
    
    # Executa a análise
    resultados = analisar_vendas_periodo(
        periodo_inicial,
        periodo_final,
        skus_desejados
    )
    
    # Exibe os resultados
    if resultados:
        print("\nArquivos processados:")
        for arquivo in resultados['arquivos_processados']:
            print(f"- {arquivo}")
            
        print("\nResultados por SKU da Variação:")
        print("\nQuantidade vendida (Unidades):")
        print(resultados['variacao']['quantidade_vendida'])
        print("\nValor total (BRL):")
        print(resultados['variacao']['valor_total'])
        
        print("\nResultados por SKU Principle:")
        print("\nQuantidade vendida (Unidades):")
        print(resultados['principle']['quantidade_vendida'])
        print("\nValor total (BRL):")
        print(resultados['principle']['valor_total'])
        
        # Totais gerais
        total_unidades = resultados['variacao']['quantidade_vendida'].sum()
        total_vendas = resultados['variacao']['valor_total'].sum()
        
        print("\nTotais Gerais:")
        print(f"Total de Unidades Vendidas: {total_unidades}")
        print(f"Total de Vendas (BRL): R$ {total_vendas:.2f}")
    else:
        print("Nenhum dado encontrado para o período especificado.")