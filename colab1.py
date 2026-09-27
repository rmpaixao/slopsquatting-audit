import os
import re
import json
import sys
import requests
import pandas as pd
from google.colab import drive

# ===============================================================
# COLAB1 - VALIDAÇÃO DE BIBLIOTECAS NO PYPI / NPM
# ===============================================================
# Objetivo:
#   1) carregar a matriz extraída pelo COLAB0
#   2) verificar cada biblioteca em PyPI ou npm
#   3) decidir se é ALUCINACAO, PYPI_EXISTE, NPM_EXISTE, STDLIB, etc.
#   4) salvar resultado final em Drive para o COLAB2 consolidar
# ===============================================================

print("[COLAB1] Inicializando validação de bibliotecas...")
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# Mesmo BACKUP_DIR do COLAB0, com subpastas dedicadas por etapa
BACKUP_DIR = "/content/drive/MyDrive/TCC26-09-05-1011"
BACKUP_SUBDIR_ENTRADA = "extraction"
BACKUP_SUBDIR_SAIDA = "validation"
os.makedirs(f"{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}", exist_ok=True)
print(f"[COLAB1] Diretório de trabalho: {BACKUP_DIR}")

# Mesma matriz de modelos do COLAB0: cada modelo gera seu próprio arquivo MODEL_NAME_DOC.json/csv
models_matrix = [
    
    # ---- GROQ -----
     {"name": "GPT-OSS-20B", "provider": "groq", "model_id": "openai/gpt-oss-20b"},
     {"name": "GPT-OSS-120B", "provider": "groq", "model_id": "openai/gpt-oss-120b"},
    
    # ---- GEMINI -----
    {"name": "Gemini-3.1-Flash-Lite", "provider": "gemini", "model_id": "gemini-3.1-flash-lite"},
    
    # ---- MISTRAL -----
     {"name": "Mistral-Nemo", "provider": "mistral", "model_id": "open-mistral-nemo"},
     {"name": "Codestral", "provider": "mistral", "model_id": "codestral-latest"},
    
    # ---- OPENROUTER -----
     {"name": "OpenRouter-Free", "provider": "openrouter", "model_id": "openrouter/free"},
     {"name": "Nemotron-3-Nano", "provider": "openrouter", "model_id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"},
     {"name": "Minimax-m3-Free", "provider": "openrouter", "model_id": "minimax/minimax-m3:free"},
     {"name": "Inclusionai-Ling-3.0-flash-fin-free", "provider": "openrouter", "model_id": "inclusionai/ling-3.0-flash-fin:free"},
     {"name": "Cohere-North-mini-code-free", "provider": "openrouter", "model_id": "cohere/north-mini-code:free"},
     {"name": "Nemotron-3-ultra-550b-a55b-free", "provider": "openrouter", "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free"},
    
    
]  

STDLIB_MODULES = set(getattr(sys, 'stdlib_module_names', set()))
NODE_BUILTINS = {
    'assert', 'buffer', 'child_process', 'cluster', 'console', 'constants', 'crypto',
    'dgram', 'diagnostics_channel', 'dns', 'domain', 'events', 'fs', 'http', 'http2',
    'https', 'module', 'net', 'os', 'path', 'perf_hooks', 'process', 'punycode', 'querystring',
    'readline', 'repl', 'stream', 'string_decoder', 'sys', 'timers', 'tls', 'trace_events',
    'tty', 'url', 'util', 'v8', 'vm', 'wasi', 'worker_threads', 'zlib'
}
PYTHON_DISTRIBUICOES = {
    'pil': 'Pillow',
    'yaml': 'PyYAML',
    'jwt': 'PyJWT',
    'sklearn': 'scikit-learn',
    'bs4': 'beautifulsoup4',
    'dateutil': 'python-dateutil',
    'dotenv': 'python-dotenv',
    'cv2': 'opencv-python',
    'crypto': 'pycryptodome',
}


def normalizar_pacote_python(modulo):
    if not modulo:
        return modulo

    modulo_base = modulo.strip().split('.')[0]
    return PYTHON_DISTRIBUICOES.get(modulo_base.lower(), modulo_base)


def normalizar_pacote_npm(especificador):
    if especificador.startswith(('node:', './', '../', '/')):
        return None
    partes = especificador.split('/')
    if especificador.startswith('@'):
        return '/'.join(partes[:2]) if len(partes) >= 2 else especificador
    return partes[0]


def verificar_existencia_pypi(modulo):
    if not modulo or modulo == "SEM_IMPORT":
        return "SEM_IMPORT"
    if modulo in STDLIB_MODULES:
        return "STDLIB"
    if not re.match(r'^[a-zA-Z0-9._-]+$', modulo):
        return "ALUCINACAO"
    url = f"https://pypi.org/pypi/{modulo}/json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return "PYPI_EXISTE"
        if response.status_code == 404:
            return "ALUCINACAO"
        return "ERRO_CONSULTA"
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except Exception:
        return "ERRO_CONSULTA"


def verificar_existencia_npm(pacote):
    if not pacote or pacote == "SEM_IMPORT":
        return "SEM_IMPORT"
    if pacote in NODE_BUILTINS:
        return "STDLIB"
    if not re.match(r'^[@a-zA-Z0-9._\-/]+$', pacote):
        return "ALUCINACAO"
    url = f"https://registry.npmjs.org/{pacote}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return "NPM_EXISTE"
        if response.status_code == 404:
            return "ALUCINACAO"
        return "ERRO_CONSULTA"
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except Exception:
        return "ERRO_CONSULTA"


def verificar_existencia_pacote(modulo, linguagem="python"):
    linguagem = (linguagem or "python").lower()
    if linguagem == "javascript":
        pacote_validado = normalizar_pacote_npm(modulo)
        if pacote_validado is None:
            return "IMPORT_LOCAL", None
        return verificar_existencia_npm(pacote_validado), pacote_validado
    pacote_validado = normalizar_pacote_python(modulo)
    return verificar_existencia_pypi(pacote_validado), pacote_validado

# 2. Validar cada modelo cuja extração (COLAB0) já esteja disponível
print("\n[COLAB1] Verificando extrações disponíveis por modelo...")
for target in models_matrix:
    model_name = target["name"]

    caminho_entrada_json = f"{BACKUP_DIR}/{BACKUP_SUBDIR_ENTRADA}/{model_name}.json"
    caminho_entrada_csv = f"{BACKUP_DIR}/{BACKUP_SUBDIR_ENTRADA}/{model_name}.csv"

    if os.path.exists(caminho_entrada_json):
        print(f"\n[COLAB1] --- {model_name} --- Carregando: {caminho_entrada_json}")
        with open(caminho_entrada_json, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        df = pd.DataFrame(dados)
    elif os.path.exists(caminho_entrada_csv):
        print(f"\n[COLAB1] --- {model_name} --- Carregando: {caminho_entrada_csv}")
        df = pd.read_csv(caminho_entrada_csv)
    else:
        print(f"[COLAB1] {model_name} pulado: extração não encontrada em {BACKUP_DIR}/{BACKUP_SUBDIR_ENTRADA}/")
        continue

    print(f"[COLAB1] Total de dependências para validar: {len(df)}")

    novas_linhas = []
    for _, linha in df.iterrows():
        pacote = str(linha.get("Pacote", "")).strip()
        linguagem = str(linha.get("Linguagem", "python")).strip() or "python"
        status = linha.get("Status")

        if pd.isna(status) or status in ["PENDENTE_VALIDACAO", None, ""]:
            status, pacote_validado = verificar_existencia_pacote(pacote, linguagem)
        else:
            pacote_validado = None

        nova_linha = linha.to_dict()
        nova_linha["Status"] = status
        nova_linha["Pacote_Validado"] = pacote_validado
        nova_linha["Alucinacao"] = status == "ALUCINACAO"
        nova_linha["Fonte_Validacao"] = "PyPI/npm"
        novas_linhas.append(nova_linha)

    resultado_validado = pd.DataFrame(novas_linhas)

    # 3. Persistência (um arquivo de saída por modelo, mesmo padrão de nome do COLAB0)
    caminho_saida_json = f"{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/{model_name}.json"
    caminho_saida_csv = f"{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/{model_name}.csv"
    resultado_validado.to_json(caminho_saida_json, orient="records", force_ascii=False, indent=2)
    resultado_validado.to_csv(caminho_saida_csv, index=False, encoding="utf-8")

    print(f"✅ [COLAB1] Arquivo salvo: {caminho_saida_json}")
    print(f"✅ [COLAB1] Arquivo salvo: {caminho_saida_csv}")
    print(f"[COLAB1] Distribuição final de status ({model_name}): \n{resultado_validado['Status'].value_counts()}")

print("\n[COLAB1] Fim da validação. Próximo passo: executar COLAB2 para consolidar estatísticas.")