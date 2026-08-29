## 📘 Documentação Técnica – Modelo Cricket

### 1. Visão Geral
O **Cricket** é um laboratório didático implementado em PyTorch que permite estudar, treinar e comparar duas arquiteturas de modelos de linguagem:

- **Modelo Denso** (baseline) – FFN com ativação SwiGLU.
- **Modelo MoE com Prefetch por Requisição** – substitui a FFN densa por um conjunto de especialistas (experts) onde o roteamento é decidido em dois níveis: primeiro um *prefetch* baseado no embedding do prompt (nível de requisição) e depois um *gate* token‑a‑token que escolhe entre os especialistas pré‑selecionados.

O projeto foi concebido para rodar em CPU com memória limitada (ex.: 24 GB RAM), permitindo que pesquisadores e estudantes explorem os trade‑offs entre capacidade paramétrica, custo computacional e qualidade de geração.

---

### 2. Arquitetura do Modelo

#### 2.1. Componentes Comuns
Ambas as versões partilham os seguintes blocos:

- **Tokenização** – tokenizador **BPE** (biblioteca `tokenizers`) treinado sobre o dataset em uso, com um vocabulário de ~4000 tokens (incluindo os tokens especiais `<PAD>` e `<UNK>`). O tamanho real do vocabulário é atribuído dinamicamente a `CONFIG["vocab_size"]` após o treino do tokenizador.
- **Embedding** – camada densa que mapeia cada token (id do vocabulário BPE) para um vetor de dimensão `hidden_size`.
- **Posicional** – RoPE (*Rotary Positional Embedding*) aplicado nas queries e keys da atenção.
- **Atenção** – *Grouped Query Attention* (GQA) com máscara causal, onde o número de heads de KV (`num_kv_heads`) é inferior ao número de heads de query (`num_heads`), reduzindo custo de memória.
- **Normalização** – RMSNorm (pré‑normalização) em cada bloco.
- **FFN** – SwiGLU, com dimensão interna `ffn_hidden = 4 × hidden_size`.
- **Cabeça de saída** – linear com *weight tying* (partilha de pesos com o embedding).

#### 2.2. Modelo Denso
A FFN é uma única instância de `SwiGLU`. Todos os parâmetros são activos para cada token, resultando num modelo com ~1,0M de parâmetros na configuração recomendada.

#### 2.3. Modelo MoE com Prefetch por Requisição
A FFN é substituída por um módulo `MoEWithPrefetch` que contém:

- `num_experts` instâncias de `SwiGLU` (especialistas).
- **Router de Prefetch** – uma camada linear que, a partir do embedding médio do prompt, produz scores para cada expert. Seleciona os `num_candidates` melhores (ex.: 2) e guarda uma máscara (0 para candidatos, `‑inf` para os restantes).
- **Gate token‑a‑token** – outra camada linear que, para cada token, calcula scores sobre todos os experts. A máscara do prefetch limita a escolha aos experts candidatos: em treino **e** validação ela é gerada dinamicamente a partir do `context_embedding` (média dos embeddings) de cada exemplo do batch, mantendo as duas fases coerentes; em inferência usa‑se a máscara fixa da requisição, pré‑calculada em `prefetch()` a partir do embedding médio do prompt.
- **Seleção top‑k** – escolhe os `top_k` (ex.: 1) melhores experts entre os candidatos, calcula os pesos via softmax e combina as saídas dos experts correspondentes.
- **Perda auxiliar** – penaliza o desbalanceamento de utilização dos experts, incentivando uma distribuição uniforme.

A FFN do MoE contém um total de parâmetros cerca de 3× maior do que a FFN densa (devido aos 3 experts), resultando num modelo com ~1,8M de parâmetros totais. Contudo, apenas `top_k` (1) dos 3 experts é activado por token, pelo que o número de parâmetros activos por token (e o custo computacional) permanece semelhante ao do modelo denso.

---

### 3. Dataset e Treino

**Fonte de dados e cache em disco** – O script carrega o dataset apenas uma vez e guarda o resultado num ficheiro de cache (`dataset_cache.txt`). Em execuções posteriores, se o cache existir, os textos são carregados diretamente do ficheiro, sem novos downloads:

- Na primeira execução (sem cache), tenta descarregar, por ordem, `eduagarcia/mc4-pt`, `eduagarcia/CrawlPT` e `dominguesm/brwac` (modo *streaming*), recolhendo até 5000 textos com comprimento entre 100 e 1000 caracteres. Os textos são limpos (colapsados espaços) e guardados em `dataset_cache.txt`.
- Se nenhuma fonte externa funcionar e não existir cache, usa a lista manual de frases de fallback em `dataset.txt` (um texto por linha). **O fallback nunca é gravado em `dataset_cache.txt`** – assim, uma futura execução com rede volta a tentar os downloads reais em vez de usar as frases manuais.
- O ficheiro `dataset.txt` contém as frases de exemplo (linguagem + matemática) e é a lista de recurso quando não há rede/consultas disponíveis.

**Tokenizador em cache** – O tokenizador BPE também é treinado apenas uma vez e guardado no ficheiro `tokenizer.json`. Se este ficheiro existir, o tokenizador é carregado do disco (sem re‑treinar).

**Pré‑processamento** – cada texto é truncado para `max_seq_len` (ex.: 64 tokens) e preenchido com padding.

**Divisão** – 80% treino, 20% validação.

**Otimizador** – AdamW com *learning rate* de 3e‑4, *clip* de gradientes a 1.0 e *early stopping* com paciência (ex.: 5 épocas sem melhora na loss de validação).

**Função de perda** – *cross‑entropy* sobre a previsão do próximo token, somada à perda auxiliar de balanceamento (só existe na versão MoE). Na configuração atual o peso dessa perda é `aux_loss_weight = 0`, ou seja, o balanceamento é computado mas ainda não influencia o treino.

---

### 4. Modo de Inferência

Em `infer`, o script entra num **loop interactive de chat** (`run_chat`): espera um `input` do utilizador no terminal e imprime a resposta do modelo no output, token a token. Escreva `exit`, `sair` ou `q` para terminar.

A função `generate_text` (núcleo da geração) permite:

- **Temperatura** – controla a aleatoriedade da amostragem (0 = greedy, >0 para amostragem).
- **Repetition penalty** – penaliza tokens que apareceram nos últimos 10 passos, reduzindo loops.
- **Top‑k sampling** – restringe a amostragem aos k tokens mais prováveis.
- **Prefetch automático** – se o modelo for MoE, o embedding médio do prompt é calculado e o prefetch é executado em todas as camadas antes da geração.

`generate_text` devolve um triplo: a sequência completa de IDs (`generated_ids`), o texto completo do prompt com a geração (`full_text`) e apenas a parte gerada (`new_text`).

---

### 5. Objectivos de Estudo

Com o Cricket, pretende‑se responder a perguntas como:

- **MoE vs. Denso** – para o mesmo custo computacional (parâmetros activos por token), qual arquitectura apresenta melhor perplexidade e qualidade de texto?
- **Eficácia do Prefetch por Requisição** – até que ponto a decisão de roteamento baseada no prompt (estática) degrada a qualidade em comparação com o roteamento token‑a‑token (dinâmico)?
- **Especialização dos Experts** – os diferentes experts aprendem a lidar com distintos tópicos (ex.: matemática vs. linguagem natural) devido à perda auxiliar e à distribuição dos gradientes?
- **Escalabilidade** – como varia a qualidade com o aumento do número de parâmetros totais e activos, mantendo o mesmo hardware?

---

### 6. Configuração Recomendada (para 24 GB RAM)

```python
CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "num_heads": 4,
    "num_kv_heads": 2,
    "ffn_hidden": 512,
    "max_seq_len": 64,
    "batch_size": 32,
    "dropout": 0.1,
    "learning_rate": 3e-4,
    "epochs": 5,
    "aux_loss_weight": 0,
    "num_experts": 3,
    "top_k": 1,
    "num_candidates": 2,
    "seed": 42,
}
```

(`vocab_size` é definido dinamicamente com base no tokenizador BPE treinado, ~4000.)

Esta configuração resulta em ~1,0M de parâmetros para o modelo denso e ~1,8M para o MoE (embora apenas os parâmetros de 1 dos 3 experts estejam activos por token). O treino em CPU demora cerca de 3–5 minutos por época.

---

### 7. Instruções de Execução

O script é controlado por **argumentos de linha de comando**:

```bash
python cricket.py train dense      # treina o modelo denso
python cricket.py train moe        # treina o modelo MoE
python cricket.py infer dense      # chat com o modelo denso treinado
python cricket.py infer moe        # chat com o modelo MoE treinado
```

Opções opcionais:

- `--cuda` – usa a GPU se disponível (por omissão corre em CPU).
- `--compile` – ativa `torch.compile` (opcional; pode não existir em todos os CPUs, caso em que há fallback automático).
- `--max-new-tokens N` – número máximo de tokens por resposta no chat (padrão 40).
- `--temperature T`, `--repetition-penalty P`, `--top-k K` – controlam a amostragem na geração.

Notas:

- O treino gera automaticamente o checkpoint `cricket_model_{dense|moe}_best.pt` e, no final, exibe a curva de loss (treino vs validação).
- Em `infer`, o modelo é carregado desse checkpoint; se não existir, será necessário treinar primeiro.
- Checkpoints antigos (treinados com `torch.compile`) são carregados automaticamente: o prefixo `_orig_mod.` das chaves do `state_dict` é removido na leitura.
- **Ficheiros de cache gerados na 1.ª execução** – `dataset_cache.txt` (textos descarregados), `dataset.txt` (fallback manual, já incluído no projeto) e `tokenizer.json` (tokenizador BPE treinado). Os downloads/treinos só ocorrem quando o respetivo ficheiro de cache não existe.
- Em `infer`, o tokenizador é carregado apenas do cache; os downloads do dataset só ocorrem em `train` (ou quando falta `tokenizer.json`).

---

### 8. Limitações e Próximos Passos

- **Tokenização** – o BPE actual (vocab ~4000, `min_frequency=2`) é básico; numa versão futura, um vocabulário maior ou um tokenizador sub‑palavra mais sofisticado poderia melhorar a capacidade semântica.
- **Dataset** – 5000 textos é suficiente para estudos mas insuficiente para modelos de produção. Os textos são guardados em cache (`dataset_cache.txt`); para forçar um novo download basta apagar esse ficheiro. O tokenizador (`tokenizer.json`) pode ser recriado apagando o ficheiro de cache correspondente.
- **Hardware** – o treino em CPU é mais lento; para acelerar, pode‑se usar GPU (basta usar `--cuda`, e opcionalmente `--compile`).
- **MoE em produção** – o prefetch por requisição é uma simplificação; sistemas reais utilizam *offloading* e *caching* de especialistas.

---

O **Cricket** é, assim, uma ferramenta educacional poderosa para compreender os fundamentos das arquitecturas modernas de LLMs, permitindo experimentar com diferentes estratégias de roteamento e escalonamento sem necessitar de infra‑estrutura dispendiosa.