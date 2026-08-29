<p align="center">
  <img src="cricket.png" alt="Cricket logo" width="220"/>
</p>

# 🏏 Cricket

Laboratório didático em PyTorch para estudar, treinar e comparar duas arquiteturas de modelos de linguagem (LLMs) num único script, pensado para rodar em CPU com memória limitada.


## ✨ Visão Geral

O **Cricket** implementa um modelo de linguagem pequeno mas completo, com duas variantes que partilham a mesma base:

- **Denso** (baseline) — FFN única com ativação SwiGLU (~1,0M parâmetros).
- **MoE com Prefetch por Requisição** — FFN substituída por 3 especialistas, com roteamento em dois níveis: um *prefetch* baseado no embedding do prompt (nível de requisição) e um *gate* token-a-token que escolhe entre os especialistas pré-selecionados (~1,8M parâmetros, mas apenas 1 de 3 experts ativo por token).

Ambas as arquiteturas usam os mesmos blocos do estado da arte em escala reduzida: **BPE**, **embeddings**, **RoPE**, **GQA** (Grouped Query Attention), **RMSNorm** e **weight tying**.

## 🚀 Uso

O script é controlado por linha de comando:

```bash
# Treino
python cricket.py train dense      # modelo denso
python cricket.py train moe        # modelo MoE

# Inferência (chat interactivo)
python cricket.py infer dense      # chat com o modelo denso
python cricket.py infer moe        # chat com o modelo MoE
```

O modo `infer` abre um loop de chat no terminal: escreva a sua mensagem e o modelo responde token a token. Use `exit`, `sair` ou `q` para sair.

### Opções

| Opção | Descrição |
|---|---|
| `--cuda` | usa GPU se disponível (padrão: CPU) |
| `--compile` | ativa `torch.compile` (opcional) |
| `--max-new-tokens N` | tokens máximos por resposta no chat (padrão: 40) |
| `--temperature T` | 0 = greedy \| >0 = amostragem (padrão: 0.8) |
| `--repetition-penalty P` | >1 penaliza repetições (padrão: 1.2) |
| `--top-k K` | amostra apenas dos K tokens mais prováveis (padrão: 0) |

## 🧩 Arquitetura

<table>
  <tr><th>Componente</th><th>Descrição</th></tr>
  <tr><td>Tokenização</td><td>BPE (~4000 tokens) treinado sobre o dataset, com cache em disco (<code>tokenizer.json</code>).</td></tr>
  <tr><td>Embedding</td><td>camada densa que mapeia cada token para um vetor de <code>hidden_size</code> dimensões.</td></tr>
  <tr><td>Posição</td><td>RoPE aplicado nas queries/keys da atenção.</td></tr>
  <tr><td>Atenção</td><td>GQA com máscara causal (<code>num_kv_heads</code> < <code>num_heads</code>).</td></tr>
  <tr><td>Normalização</td><td>RMSNorm em pré-normalização.</td></tr>
  <tr><td>FFN</td><td>SwiGLU (denso) ou <code>MoEWithPrefetch</code> (3 experts + roteamento em 2 níveis).</td></tr>
  <tr><td>Cabeça de saída</td><td>linear com weight tying (partilha pesos com o embedding).</td></tr>
</table>

### MoE com Prefetch em resumo

1. **Prefetch por requisição** — o embedding médio do prompt pré-seleciona os `num_candidates` melhores especialistas e guarda uma máscara.
2. **Gate token-a-token** — para cada token, scores sobre os experts; a máscara do prefetch limita a escolha aos candidatos.
3. **Top-k** — combina as saídas dos experts escolhidos (ponderadas por softmax).
4. **Perda auxiliar** — penaliza o desbalanceamento de utilização dos experts (experimental; `aux_loss_weight = 0` por omissão).

## 📂 Dataset

- Carregado uma única vez; textos guardados em cache (`dataset_cache.txt`).
- Fontes externas em português (`mc4-pt`, `CrawlPT`, `brwac`) com fallback manual em `dataset.txt`.
- Divisão 80% treino / 20% validação.
- Otimizador AdamW, clip de gradientes a 1.0 e *early stopping*.

## 📚 Documentação

- [`arquitetura.md`](arquitetura.md) — documentação técnica de referência.
- [`cricket-explicado.md`](cricket-explicado.md) — explicação passo a passo (para desenvolvedores juniores).

## ⚙️ Requisitos

- Python 3.10+
- `torch`, `numpy`, `tokenizers`, `datasets` (opcional), `scikit-learn`, `matplotlib`

O projeto foi concebido para CPU com 24 GB de RAM; o treino demora cerca de 3–5 minutos por época na configuração recomendada.

## 📈 Objetivos de Estudo

- MoE vs. Denso para o mesmo custo computacional.
- Eficácia do prefetch por requisição (roteamento estático vs. dinâmico).
- Especialização dos experts por tópico.
- Escalabilidade com o número de parâmetros.