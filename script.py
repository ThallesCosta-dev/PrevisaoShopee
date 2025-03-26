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
        
        # Converte a coluna de vendas para numérico
        df_filtrado['Vendas (Pedido pago) (BRL)'] = pd.to_numeric(df_filtrado['Vendas (Pedido pago) (BRL)'], errors='coerce')
        df_filtrado['Unidades (Pedido realizado)'] = pd.to_numeric(df_filtrado['Unidades (Pedido realizado)'], errors='coerce')
        
        # Cria um DataFrame combinado com ambos os SKUs
        df_resultados = df_filtrado.groupby(['SKU da Variação', 'SKU Principle']).agg({
            'Unidades (Pedido realizado)': 'sum',
            'Vendas (Pedido pago) (BRL)': 'sum'
        }).reset_index()
        
        # Renomeia as colunas para melhor compreensão
        df_resultados.columns = ['SKU da Variação', 'SKU Principle', 'Quantidade Vendida', 'Valor Total (BRL)']
        
        # Cria nome do arquivo com o período analisado
        nome_arquivo = f'relatorio_vendas_{periodo_inicial.strftime("%Y%m%d")}_{periodo_final.strftime("%Y%m%d")}.xlsx'
        
        # Cria um Excel com duas abas (resultados e totais)
        with pd.ExcelWriter(nome_arquivo) as writer:
            # Aba de resultados
            df_resultados.to_excel(writer, sheet_name='Resultados', index=False)
            
            # Aba de totais
            df_totais = pd.DataFrame({
                'Métrica': ['Total Unidades Vendidas', 'Total Vendas (BRL)'],
                'Valor': [
                    df_resultados['Quantidade Vendida'].sum(),
                    df_resultados['Valor Total (BRL)'].sum()
                ]
            })
            df_totais.to_excel(writer, sheet_name='Totais', index=False)
        
        print(f"\nRelatório exportado com sucesso: {nome_arquivo}")
        
        return {
            'resultados': df_resultados,
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
    
    # Exibe os resultados no console
    if resultados:
        print("\nArquivos processados:")
        for arquivo in resultados['arquivos_processados']:
            print(f"- {arquivo}")
            
        print("\nResultados:")
        print(resultados['resultados'])
    else:
        print("Nenhum dado encontrado para o período especificado.")