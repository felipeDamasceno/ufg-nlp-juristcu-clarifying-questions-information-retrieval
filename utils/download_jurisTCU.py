#!/usr/bin/env python3
"""
Script simples para baixar o dataset JurisTCU do Hugging Face
"""

from huggingface_hub import hf_hub_download, list_repo_files
import os

def download_juris_tcu():
    """Baixa todos os arquivos CSV do dataset JurisTCU"""
    
    repo_id = "LeandroRibeiro/JurisTCU"
    output_dir = "dados/juris_tcu"
    
    print(f"Baixando dataset {repo_id}...")
    
    # Cria pasta se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Lista todos os arquivos (especifica que é um dataset)
    files = list_repo_files(repo_id, repo_type="dataset")
    csv_files = [f for f in files if f.endswith('.csv')]
    
    print(f"Encontrados {len(csv_files)} arquivos CSV:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Baixa cada arquivo
    downloaded = []
    for csv_file in csv_files:
        print(f"\nBaixando {csv_file}...")
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=csv_file,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        downloaded.append(local_path)
        print(f"Salvo em: {local_path}")
    
    print(f"\n✅ Download concluído! {len(downloaded)} arquivos baixados em '{output_dir}'")
    return downloaded

if __name__ == "__main__":
    download_juris_tcu()