import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

# ============================================================================
# COLAB2 - CONSOLIDAÇÃO ESTATÍSTICA E RELATÓRIO EXPANDIDO (COM GRÁFICO 6)
# ============================================================================

print("[COLAB2] Inicializando consolidação estatística...")
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

BACKUP_DIR = "/content/drive/MyDrive/TCC26-09-05-1011"
BACKUP_SUBDIR_ENTRADA = "validation"
BACKUP_SUBDIR_SAIDA = "report"
os.makedirs(f"{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}", exist_ok=True)
print(f"[COLAB2] Diretório de dados: {BACKUP_DIR}\n")

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

# ============================================================================
# AUDITORIA E IDENTIFICAÇÃO PRÉVIA DOS ARQUIVOS EM DISCO
# ============================================================================
print("=" * 80)
print("🔍 [COLAB2] AUDITORIA E IDENTIFICAÇÃO PRÉVIA DOS ARQUIVOS EM DISCO")
print("=" * 80)

dataframes_modelos = []
auditoria_modelos = []

for target in models_matrix:
    model_name = target["name"]
    caminho_json = f"{BACKUP_DIR}/{BACKUP_SUBDIR_ENTRADA}/{model_name}.json"
    caminho_csv = f"{BACKUP_DIR}/{BACKUP_SUBDIR_ENTRADA}/{model_name}.csv"

    caminho_ativo = None
    formato = None

    if os.path.exists(caminho_json):
        caminho_ativo = caminho_json
        formato = "JSON"
    elif os.path.exists(caminho_csv):
        caminho_ativo = caminho_csv
        formato = "CSV"

    if caminho_ativo:
        timestamp_mod = os.path.getmtime(caminho_ativo)
        data_mod = datetime.fromtimestamp(timestamp_mod).strftime('%Y-%m-%d %H:%M:%S')
        tamanho_kb = os.path.getsize(caminho_ativo) / 1024

        if formato == "JSON":
            with open(caminho_ativo, 'r', encoding='utf-8') as f:
                df_temp = pd.DataFrame(json.load(f))
        else:
            df_temp = pd.read_csv(caminho_ativo)

        df_temp['Modelo'] = model_name

        total_linhas = len(df_temp)
        status_counts = df_temp['Status'].value_counts().to_dict() if 'Status' in df_temp.columns else {}
        alucinacoes = status_counts.get('ALUCINACAO', 0)

        print(f"📦 [{model_name}] ({formato})")
        print(f"   Arquivo: {os.path.basename(caminho_ativo)} ({tamanho_kb:.1f} KB) | Modificado: {data_mod}")
        print(f"   Total registros: {total_linhas} | Alucinações: {alucinacoes}")
        print(f"   Distribuição de Status: {status_counts}\n")

        auditoria_modelos.append({
            "Modelo": model_name,
            "Arquivo": formato,
            "Modificado": data_mod,
            "Total_Linhas": total_linhas,
            "Alucinações": alucinacoes
        })

        dataframes_modelos.append(df_temp)
    else:
        print(f"⚠️ [{model_name}] PULADO: Nenhum arquivo JSON/CSV encontrado em validation/\n")

print("=" * 80)
print("📊 RESUMO DOS ARQUIVOS CARREGADOS:")
print(pd.DataFrame(auditoria_modelos).to_string(index=False))
print("=" * 80 + "\n")

# ============================================================================
# CONSOLIDAÇÃO E LIMPEZA
# ============================================================================
if not dataframes_modelos:
    print("❌ [COLAB2] Nenhum dado disponível.")
    exit()

df = pd.concat(dataframes_modelos, ignore_index=True)

if 'Modulo' in df.columns and 'Pacote' not in df.columns:
    df = df.rename(columns={'Modulo': 'Pacote'})
df['Pacote'] = df['Pacote'].fillna('').astype(str)
df['Linguagem'] = df['Linguagem'].fillna('PYTHON').astype(str).str.upper()
df['Status'] = df['Status'].astype(str).str.strip()

status_avaliaveis = {'PYPI_EXISTE', 'NPM_EXISTE', 'ALUCINACAO'}
df_externos = df[df['Status'].isin(status_avaliaveis)].copy()

# ============================================================================
# CÁLCULO DE MÉTRICAS BÁSICAS
# ============================================================================
def calcular_metricas(df_subset, coluna_grupo):
    metricas = []
    for val in df_subset[coluna_grupo].unique():
        sub = df_subset[df_subset[coluna_grupo] == val]
        tot = len(sub)
        aluc = len(sub[sub['Status'] == 'ALUCINACAO'])
        taxa = (aluc / tot * 100) if tot > 0 else 0
        metricas.append({coluna_grupo: val, 'Total': tot, 'Alucinações': aluc, 'Taxa_%': taxa})
    return pd.DataFrame(metricas).set_index(coluna_grupo)

analise_por_modelo = calcular_metricas(df_externos, 'Modelo')
analise_por_estrategia = calcular_metricas(df_externos, 'Estrategia')
analise_por_linguagem = calcular_metricas(df_externos, 'Linguagem')

total_externos = len(df_externos)
alucinacoes_totais = len(df_externos[df_externos['Status'] == 'ALUCINACAO'])
taxa_geral = (alucinacoes_totais / total_externos * 100) if total_externos > 0 else 0

print("\n" + "="*70)
print("📌 RESUMO FINAL EM TELA")
print("="*70)
print(f"Total registros brutos: {len(df)}")
print(f"Total bibliotecas com validação conclusiva: {total_externos}")
print(f"Total de alucinações: {alucinacoes_totais}")
print(f"Taxa geral de alucinação: {taxa_geral:.2f}%\n")
print(analise_por_modelo.sort_values('Taxa_%', ascending=False).to_string())

# ============================================================================
# PAINEL GERAL (2x2 E HEATMAP)
# ============================================================================
sns.set_theme(style='whitegrid')
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Taxa por Modelo
ax1 = axes[0, 0]
t_mod = analise_por_modelo.sort_values('Taxa_%', ascending=True)
ax1.barh(range(len(t_mod)), t_mod['Taxa_%'].values, color='#e74c3c')
ax1.set_yticks(range(len(t_mod)))
ax1.set_yticklabels(t_mod.index)
ax1.set_xlabel('Taxa de Alucinação (%)')
ax1.set_title('Taxa por Modelo', fontweight='bold')
ax1.axvline(x=taxa_geral, color='#3498db', linestyle='--', label=f'Média ({taxa_geral:.1f}%)')
ax1.legend()

# Taxa por Estratégia
ax2 = axes[0, 1]
t_est = analise_por_estrategia.sort_values('Taxa_%', ascending=True)
ax2.barh(range(len(t_est)), t_est['Taxa_%'].values, color='#9b59b6')
ax2.set_yticks(range(len(t_est)))
ax2.set_yticklabels(t_est.index)
ax2.set_xlabel('Taxa de Alucinação (%)')
ax2.set_title('Taxa por Prompt / Estratégia', fontweight='bold')

# Taxa por Linguagem
ax3 = axes[1, 0]
t_ling = analise_por_linguagem.sort_values('Taxa_%', ascending=True)
ax3.barh(range(len(t_ling)), t_ling['Taxa_%'].values, color='#2ecc71')
ax3.set_yticks(range(len(t_ling)))
ax3.set_yticklabels(t_ling.index)
ax3.set_xlabel('Taxa de Alucinação (%)')
ax3.set_title('Taxa por Linguagem', fontweight='bold')

# Distribuição Global de Status
ax4 = axes[1, 1]
st_dist = df['Status'].value_counts()
cores = {'ALUCINACAO': '#e74c3c', 'PYPI_EXISTE': '#2ecc71', 'NPM_EXISTE': '#3498db', 'SEM_IMPORT': '#95a5a6', 'IMPORT_LOCAL': '#f1c40f'}
ax4.pie(st_dist.values, labels=st_dist.index, autopct='%1.1f%%', colors=[cores.get(s, '#bdc3c7') for s in st_dist.index], startangle=140)
ax4.set_title('Distribuição Geral de Status', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/analise_taxa_alucinacao.png', dpi=300)
plt.close()
print("[COLAB2] Salvo: analise_taxa_alucinacao.png")

# Heatmap Modelo x Estratégia
pivot_modelo_estrategia = pd.crosstab(
    index=df_externos['Modelo'],
    columns=df_externos['Estrategia'],
    values=df_externos['Status'],
    aggfunc=lambda x: (x == 'ALUCINACAO').sum()
).fillna(0).astype(int)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_modelo_estrategia, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Mapa de Calor: Alucinações Absolutas (Modelo × Estratégia)', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/heatmap_modelo_estrategia.png', dpi=300)
plt.close()
print("[COLAB2] Salvo: heatmap_modelo_estrategia.png")

# ============================================================================
# SUÍTE DE 6 GRÁFICOS ESTATÍSTICOS AVANÇADOS
# ============================================================================

# Gráfico 1: Discrepância por Ecossistema e Categoria
cat_metrics = df_externos.groupby(['Categoria', 'Linguagem'], as_index=False).agg(
    Total=('Status', 'count'),
    Alucinacoes=('Status', lambda s: (s == 'ALUCINACAO').sum())
)
cat_metrics['Taxa_%'] = (cat_metrics['Alucinacoes'] / cat_metrics['Total']) * 100

plt.figure(figsize=(12, 6))
sns.barplot(data=cat_metrics, y='Categoria', x='Taxa_%', hue='Linguagem', palette={'PYTHON': '#e74c3c', 'JAVASCRIPT': '#f39c12'})
plt.title('Gráfico 1: Taxa de Alucinação por Domínio/Categoria da Tarefa', fontweight='bold', pad=15)
plt.xlabel('Taxa de Alucinação (%)')
plt.ylabel('Categoria do Cenário')
plt.tight_layout()
plt.savefig(f'{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/grafico1_linguagem_categoria.png', dpi=300)
plt.close()
print("[COLAB2] Salvo: grafico1_linguagem_categoria.png")

# Gráfico 2: Eficácia do Guardrail Defensivo
pivot_tx = pd.crosstab(
    index=df_externos['Modelo'],
    columns=df_externos['Estrategia'],
    values=df_externos['Status'],
    aggfunc=lambda x: (x == 'ALUCINACAO').sum() / len(x) * 100
).fillna(0)

estrategias_comp = ['1_Baseline_Dev', '2_Guardrail_Defensivo', '3_Adversarial_Nudge']
presentes = [e for e in estrategias_comp if e in pivot_tx.columns]

if len(presentes) >= 2:
    plt.figure(figsize=(12, 7))
    for mod in pivot_tx.index:
        y = [pivot_tx.loc[mod, col] for col in presentes]
        plt.plot(presentes, y, marker='o', linewidth=2, label=mod)
    plt.title('Gráfico 2: Dinâmica de Sensibilidade a Prompts de Defesa e Nudge', fontweight='bold', pad=15)
    plt.ylabel('Taxa de Alucinação (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/grafico2_delta_guardrail.png', dpi=300)
    plt.close()
    print("[COLAB2] Salvo: grafico2_delta_guardrail.png")

# Gráfico 3: Concentração de Slopsquatting (Curva de Pareto)
df_aluc = df[df['Status'] == 'ALUCINACAO'].copy()

def sanitizar_nome_pacote(nome):
    if not isinstance(nome, str) or not nome.strip():
        return None
    nome = nome.strip()
    if nome.startswith(('http://', 'https://', '//', 'import ')):
        nome = nome.split('/')[-1]
    if len(nome) > 28:
        return nome[:25] + "..."
    return nome

df_aluc['Pacote_Limpo'] = df_aluc['Pacote'].apply(sanitizar_nome_pacote)
pacotes_freq = df_aluc['Pacote_Limpo'].dropna().value_counts()

if not pacotes_freq.empty:
    top_p = pacotes_freq.head(15).reset_index()
    top_p.columns = ['Pacote', 'Contagem']
    top_p['Percentual_Acumulado'] = (top_p['Contagem'].cumsum() / top_p['Contagem'].sum()) * 100

    fig, ax_bar = plt.subplots(figsize=(13, 7))

    bars = ax_bar.bar(range(len(top_p)), top_p['Contagem'], color='#e74c3c', width=0.6, alpha=0.9)
    ax_bar.set_ylabel('Frequência Absoluta de Alucinações', color='#c0392b', fontweight='bold')
    ax_bar.set_xticks(range(len(top_p)))
    ax_bar.set_xticklabels(top_p['Pacote'], rotation=55, ha='right', fontsize=9, fontweight='medium')

    for bar in bars:
        h = bar.get_height()
        ax_bar.annotate(f'{int(h)}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color='#333')

    ax_line = ax_bar.twinx()
    ax_line.plot(range(len(top_p)), top_p['Percentual_Acumulado'], color='#2c3e50', marker='D', markersize=6, linewidth=2)
    ax_line.set_ylabel('% Acumulado', color='#2c3e50', fontweight='bold')
    ax_line.set_ylim(0, 108)
    ax_line.axhline(80, color='#7f8c8d', linestyle=':', linewidth=1.2, label='Linha 80% (Pareto)')
    ax_line.grid(False)
    ax_line.legend(loc='lower right')

    plt.title('Gráfico 3: Análise de Pareto - Concentração do Vetor Slopsquatting (Top 15)', fontweight='bold', pad=18)
    plt.subplots_adjust(bottom=0.32)
    plt.savefig(f'{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/grafico3_pareto_slopsquatting.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[COLAB2] Salvo: grafico3_pareto_slopsquatting.png (Formatado)")

# Gráfico 4: Dispersão Cautela vs Risco (SEM_IMPORT vs ALUCINAÇÃO)
metricas_dispersao = []
for modelo in df['Modelo'].unique():
    d_m = df[df['Modelo'] == modelo]
    total_chamadas = len(d_m)
    sem_imp_taxa = (len(d_m[d_m['Status'] == 'SEM_IMPORT']) / total_chamadas * 100) if total_chamadas > 0 else 0
    
    d_ext = d_m[d_m['Status'].isin(status_avaliaveis)]
    taxa_aluc = (len(d_ext[d_ext['Status'] == 'ALUCINACAO']) / len(d_ext) * 100) if len(d_ext) > 0 else 0
    metricas_dispersao.append({'Modelo': modelo, 'Taxa_Sem_Import': sem_imp_taxa, 'Taxa_Alucinacao': taxa_aluc})

df_disp = pd.DataFrame(metricas_dispersao)

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_disp, x='Taxa_Sem_Import', y='Taxa_Alucinacao', s=140, color='#34495e')
for _, row in df_disp.iterrows():
    plt.annotate(row['Modelo'], (row['Taxa_Sem_Import'] + 0.4, row['Taxa_Alucinacao'] + 0.4), fontsize=9)
plt.title('Gráfico 4: Perfil Comportamental - Cautela (SEM_IMPORT) vs Risco de Alucinação', fontweight='bold', pad=15)
plt.xlabel('Taxa de Resolução Auto-contida / SEM_IMPORT (%)')
plt.ylabel('Taxa de Alucinação em Bibliotecas Externas (%)')
plt.tight_layout()
plt.savefig(f'{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/grafico4_dispersao_cautela_risco.png', dpi=300)
plt.close()
print("[COLAB2] Salvo: grafico4_dispersao_cautela_risco.png")

# Gráfico 5: Tipologia Sintática das Alucinações
def classificar_alucinacao(pacote):
    if '.' in pacote:
        return 'Submódulo como Raiz (ex: sub.modulo)'
    elif '-' in pacote:
        return 'Nome Hifenizado Preditivo (ex: dicom-web)'
    else:
        return 'Nome Sintético Simples (ex: dnp3)'

if not df_aluc.empty:
    df_aluc_tipos = df_aluc.copy()
    df_aluc_tipos['Tipo_Alucinacao'] = df_aluc_tipos['Pacote'].apply(classificar_alucinacao)
    contagem_tipos = df_aluc_tipos['Tipo_Alucinacao'].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(contagem_tipos.values, labels=contagem_tipos.index, autopct='%1.1f%%', colors=['#e74c3c', '#e67e22', '#f1c40f'], startangle=140)
    plt.title('Gráfico 5: Tipologia Sintática das Alucinações de Pacotes', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/grafico5_tipologia_alucinacoes.png', dpi=300)
    plt.close()
    print("[COLAB2] Salvo: grafico5_tipologia_alucinacoes.png")

# Gráfico 6: Distribuição das Taxas de Alucinação vs. Curva Normal Teórica
taxas_modelos = analise_por_modelo['Taxa_%'].values

if len(taxas_modelos) >= 3:
    mu = np.mean(taxas_modelos)
    sigma = np.std(taxas_modelos, ddof=1)
    
    stat_shapiro, p_valor = stats.shapiro(taxas_modelos)
    assimetria = stats.skew(taxas_modelos)
    curtose = stats.kurtosis(taxas_modelos)
    
    plt.figure(figsize=(11, 6))
    
    sns.histplot(
        taxas_modelos, 
        kde=True, 
        stat="density", 
        bins=6, 
        color="#3498db", 
        alpha=0.4, 
        edgecolor="black",
        label="Densidade Empírica (KDE)"
    )
    
    x_axis = np.linspace(max(0, mu - 3.5 * sigma), mu + 3.5 * sigma, 200)
    y_gauss = stats.norm.pdf(x_axis, mu, sigma)
    plt.plot(x_axis, y_gauss, color="#e74c3c", linestyle="--", linewidth=2.5, 
             label=f"Normal Teórica N(μ={mu:.1f}%, σ={sigma:.1f}%)")
    
    plt.axvline(mu, color="#c0392b", linestyle="-", linewidth=1.5, label=f"Média (μ) = {mu:.1f}%")
    plt.axvline(mu + sigma, color="#7f8c8d", linestyle=":", linewidth=1.2, label=f"μ ± 1σ ({mu+sigma:.1f}%)")
    if (mu - sigma) > 0:
        plt.axvline(mu - sigma, color="#7f8c8d", linestyle=":", linewidth=1.2)

    plt.title("Gráfico 6: Distribuição da Taxa de Alucinação dos Modelos vs. Curva Gaussiana", fontweight="bold", pad=15)
    plt.xlabel("Taxa de Alucinação (%)")
    plt.ylabel("Densidade de Probabilidade")
    
    texto_box = (
        f"Média (μ): {mu:.2f}%\n"
        f"Desvio Padrão (σ): {sigma:.2f}%\n"
        f"Shapiro-Wilk: p = {p_valor:.4f}\n"
        f"Assimetria (Skew): {assimetria:.2f}\n"
        f"Curtose: {curtose:.2f}"
    )
    plt.gca().text(
        0.70, 0.60, texto_box, 
        transform=plt.gca().transAxes, 
        fontsize=9, 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#bdc3c7", alpha=0.9)
    )
    
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/grafico6_distribuicao_normal.png", dpi=300)
    plt.close()
    print("[COLAB2] Salvo: grafico6_distribuicao_normal.png")

# ============================================================================
# PERSISTÊNCIA DOS RELATÓRIOS FINAIS
# ============================================================================
relatorio_md = "# Relatório Consolidado do Benchmark\n\n"
relatorio_md += f"- Total de registros brutos: {len(df)}\n"
relatorio_md += f"- Bibliotecas com validação conclusiva: {total_externos}\n"
relatorio_md += f"- Taxa geral de alucinação: {taxa_geral:.2f}%\n"
relatorio_md += f"- Total de alucinações: {alucinacoes_totais}\n\n"

relatorio_md += "## Métricas por Modelo\n"
for m in analise_por_modelo.sort_values('Taxa_%', ascending=False).index:
    relatorio_md += f"- {m}: {analise_por_modelo.loc[m, 'Taxa_%']:.2f}% ({analise_por_modelo.loc[m, 'Alucinações']}/{analise_por_modelo.loc[m, 'Total']})\n"

relatorio_md += "\n## Gráficos Gerados\n"
relatorio_md += "- analise_taxa_alucinacao.png\n"
relatorio_md += "- heatmap_modelo_estrategia.png\n"
relatorio_md += "- grafico1_linguagem_categoria.png\n"
relatorio_md += "- grafico2_delta_guardrail.png\n"
relatorio_md += "- grafico3_pareto_slopsquatting.png\n"
relatorio_md += "- grafico4_dispersao_cautela_risco.png\n"
relatorio_md += "- grafico5_tipologia_alucinacoes.png\n"
relatorio_md += "- grafico6_distribuicao_normal.png\n"

with open(f'{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/RELATORIO_ANALISE.md', 'w', encoding='utf-8') as f:
    f.write(relatorio_md)

json_export = {
    'resumo': {
        'total_registros': int(len(df)),
        'taxa_geral_alucinacao_%': float(round(taxa_geral, 2)),
        'total_alucinacoes': int(alucinacoes_totais)
    },
    'por_modelo': analise_por_modelo.to_dict('index'),
    'por_estrategia': analise_por_estrategia.to_dict('index'),
    'por_linguagem': analise_por_linguagem.to_dict('index'),
    'status_distribuicao': df['Status'].value_counts().to_dict()
}

with open(f'{BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}/analises_consolidadas.json', 'w', encoding='utf-8') as f:
    json.dump(json_export, f, indent=2, ensure_ascii=False)

print(f"\n[COLAB2] Execução concluída com sucesso!")
print(f"[COLAB2] Todos os arquivos foram gravados em: {BACKUP_DIR}/{BACKUP_SUBDIR_SAIDA}")