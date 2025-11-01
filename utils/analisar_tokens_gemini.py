import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Importar função de limpeza HTML
from src.utils.preprocessamento import remove_html

import google.genai as genai

def contar_tokens_gemini(texto, client):
    """Conta tokens exatos usando Gemini 2.5 Flash"""
    try:
        # Contar tokens usando Gemini 2.5 Flash
        response = client.models.count_tokens(
            model='gemini-2.5-flash',
            contents=texto
        )
        return response.total_tokens
    except Exception as e:
        print(f"Erro ao contar tokens: {e}")
        return None

def analisar_tokens_dataset():
    """Analisa tokens do dataset JurisTCU"""
    
    # Verificar se API key está disponível
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERRO: Variável de ambiente GOOGLE_API_KEY não encontrada!")
        print("Configure sua API key do Google AI Studio:")
        print("set GOOGLE_API_KEY=sua_chave_aqui")
        return
    
    # Configurar cliente uma única vez
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Erro ao configurar cliente Gemini: {e}")
        return
    
    # Carregar dataset
    csv_path = Path("dados/juris_tcu/doc.csv")
    if not csv_path.exists():
        print(f"Arquivo não encontrado: {csv_path}")
        return
    
    print("Carregando dataset...")
    df = pd.read_csv(csv_path)
    print(f"Dataset carregado: {len(df)} registros")
    print("Analisando dataset completo...")
    
    # Contadores para campos vazios
    enunciados_vazios = 0
    excertos_vazios = 0
    ambos_vazios = 0
    
    resultados = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"Processando registro {idx + 1}/{len(df)}...")
        
        # Processar ENUNCIADO
        enunciado_raw = str(row['ENUNCIADO']) if pd.notna(row['ENUNCIADO']) else ""
        enunciado_clean = remove_html(enunciado_raw)
        
        # Verificar se ENUNCIADO está vazio
        enunciado_vazio = len(enunciado_clean.strip()) == 0 or enunciado_clean.strip().lower() == 'nan'
        if enunciado_vazio:
            enunciados_vazios += 1
            
        tokens_enunciado = contar_tokens_gemini(enunciado_clean, client) if not enunciado_vazio else 0
        
        # Processar EXCERTO
        excerto_raw = str(row['EXCERTO']) if pd.notna(row['EXCERTO']) else ""
        excerto_clean = remove_html(excerto_raw)
        
        # Verificar se EXCERTO está vazio
        excerto_vazio = len(excerto_clean.strip()) == 0 or excerto_clean.strip().lower() == 'nan'
        if excerto_vazio:
            excertos_vazios += 1
            
        tokens_excerto = contar_tokens_gemini(excerto_clean, client) if not excerto_vazio else 0
        
        # Verificar se ambos estão vazios
        if enunciado_vazio and excerto_vazio:
            ambos_vazios += 1
        
        # Processar ENUNCIADO + EXCERTO
        texto_combinado = enunciado_clean + " " + excerto_clean
        tokens_combinado = contar_tokens_gemini(texto_combinado, client)
        
        resultado = {
            'registro': int(idx),
            'tokens_enunciado': int(tokens_enunciado) if tokens_enunciado is not None else None,
            'tokens_excerto': int(tokens_excerto) if tokens_excerto is not None else None,
            'tokens_combinado': int(tokens_combinado) if tokens_combinado is not None else None,
            'chars_enunciado': len(enunciado_clean),
            'chars_excerto': len(excerto_clean),
            'chars_combinado': len(texto_combinado),
            'enunciado_vazio': enunciado_vazio,
            'excerto_vazio': excerto_vazio
        }
        
        resultados.append(resultado)
    
    # Mostrar estatísticas de campos vazios
    total_registros = len(df)
    print(f"\n=== ESTATÍSTICAS DE CAMPOS VAZIOS ===")
    print(f"ENUNCIADOS vazios: {enunciados_vazios}/{total_registros} ({enunciados_vazios/total_registros*100:.1f}%)")
    print(f"EXCERTOS vazios: {excertos_vazios}/{total_registros} ({excertos_vazios/total_registros*100:.1f}%)")
    print(f"AMBOS vazios: {ambos_vazios}/{total_registros} ({ambos_vazios/total_registros*100:.1f}%)")
    
    # Calcular estatísticas apenas para campos não vazios
    tokens_enunciado_list = [r['tokens_enunciado'] for r in resultados if r['tokens_enunciado'] is not None and not r['enunciado_vazio']]
    tokens_combinado_list = [r['tokens_combinado'] for r in resultados if r['tokens_combinado'] is not None]
    
    if tokens_enunciado_list:
        stats_enunciado = {
            'media': float(np.mean(tokens_enunciado_list)),
            'mediana': float(np.median(tokens_enunciado_list)),
            'minimo': int(np.min(tokens_enunciado_list)),
            'maximo': int(np.max(tokens_enunciado_list)),
            'desvio_padrao': float(np.std(tokens_enunciado_list)),
            'registros_validos': len(tokens_enunciado_list)
        }
        
        print("\n=== ESTATÍSTICAS ENUNCIADO (apenas não vazios) ===")
        print(f"Registros válidos: {stats_enunciado['registros_validos']}")
        print(f"Média: {stats_enunciado['media']:.2f} tokens")
        print(f"Mediana: {stats_enunciado['mediana']:.2f} tokens")
        print(f"Mínimo: {stats_enunciado['minimo']} tokens")
        print(f"Máximo: {stats_enunciado['maximo']} tokens")
        print(f"Desvio padrão: {stats_enunciado['desvio_padrao']:.2f} tokens")
    
    if tokens_combinado_list:
        stats_combinado = {
            'media': float(np.mean(tokens_combinado_list)),
            'mediana': float(np.median(tokens_combinado_list)),
            'minimo': int(np.min(tokens_combinado_list)),
            'maximo': int(np.max(tokens_combinado_list)),
            'desvio_padrao': float(np.std(tokens_combinado_list)),
            'registros_validos': len(tokens_combinado_list)
        }
        
        print("\n=== ESTATÍSTICAS ENUNCIADO + EXCERTO ===")
        print(f"Registros válidos: {stats_combinado['registros_validos']}")
        print(f"Média: {stats_combinado['media']:.2f} tokens")
        print(f"Mediana: {stats_combinado['mediana']:.2f} tokens")
        print(f"Mínimo: {stats_combinado['minimo']} tokens")
        print(f"Máximo: {stats_combinado['maximo']} tokens")
        print(f"Desvio padrão: {stats_combinado['desvio_padrao']:.2f} tokens")
    
    # Salvar resultados
    output_file = "utils/analise_tokens_gemini.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'resultados_detalhados': resultados,
            'estatisticas_campos_vazios': {
                'enunciados_vazios': enunciados_vazios,
                'excertos_vazios': excertos_vazios,
                'ambos_vazios': ambos_vazios,
                'total_analisados': total_registros
            },
            'estatisticas_enunciado': stats_enunciado if tokens_enunciado_list else None,
            'estatisticas_combinado': stats_combinado if tokens_combinado_list else None
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados salvos em: {output_file}")

if __name__ == "__main__":
    analisar_tokens_dataset()