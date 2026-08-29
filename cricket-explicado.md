# Cricket — Explicado Passo a Passo

> **Objetivo:** explicar o funcionamento do `cricket.py` do zero, sem assumir
> conhecimentos prévios de PyTorch. Este documento acompanha o código atual
> (versão reorganizada, com execução via linha de comando).
>
> **Como ler:** com o `cricket.py` aberto ao lado, seguindo a ordem das secções.
> O documento segue exatamente o caminho real de execução: primeiro o **treino**
> (dos dados até os pesos), depois a **inferência** (do prompt até o texto).

---

## Índice

- [1. Visão geral — o que o Cricket faz](#1-visão-geral--o-que-o-cricket-faz)
- [2. Ponto de partida — argumentos e configuração](#2-ponto-de-partida--argumentos-e-configuração)
- [3. Os dados](#3-os-dados)
- [4. O tokenizador](#4-o-tokenizador)
- [5. A máquina — arquitetura do modelo](#5-a-máquina--arquitetura-do-modelo)
- [6. O treino](#6-o-treino)
- [7. A inferência](#7-a-inferência)
- [8. Do início ao fim — mapa mental](#8-do-início-ao-fim--mapa-mental)

---

## 1. Visão geral — o que o Cricket faz

O Cricket é um **modelo de linguagem (LLM)** em miniatura. Um LLM é uma função
que:

1. recebe uma sequência de **tokens** (números que representam pedaços de texto);
2. devolve, para cada posição da sequência, uma distribuição de probabilidade
   sobre qual token deve vir **a seguir**;
3. escolhendo — um a um — os próximos tokens, ele gera texto novo.

O Cricket implementa **duas arquiteturas** que partilham quase tudo e diferem
apenas na "rede interna" de cada bloco:

| Arquitetura | Ideia central | Parâmetros |
|---|---|---|
| `dense` | uma única FFN SwiGLU para todos os tokens | ~1,0M |
| `moe` | um conjunto de 3 "especialistas" que são escolhidos por token | ~1,8M |

Para treinar e usar o modelo basta:

```bash
python cricket.py train dense      # treina o modelo denso
python cricket.py train moe        # treina o modelo MoE
python cricket.py infer dense      # chat interactivo com o modelo denso
python cricket.py infer moe        # chat interactivo com o modelo MoE
```

O código está organizado em 10 secções. O fluxo de alto nível é:

```text
main()
  ├─ train ──→ load_texts()  →  BPETokenizer  →  datasets/loaders
  │            →  CricketLM  →  train_model() →  plot_history()
  │
  └─ infer ──→ BPETokenizer (cache)  →  CricketLM (checkpoint)
               →  run_chat()  →  generate_text()   (loop interactivo)
```

Este documento percorre esse caminho de forma linear: primeiro o preparo dos
dados e do tokenizador, depois a arquitetura, depois **como o treino executa
passo a passo**, e por fim **como a inferência funciona desde o início**.

---

## 2. Ponto de partida — argumentos e configuração

Quando você corre `python cricket.py train dense`, a primeira coisa que roda é
`if __name__ == "__main__": main()`. Veja o que `main()` faz:

<table>
<tr><th>Código (resumido)</th><th>O que acontece</th></tr>
<tr><td><code>args = parse_args(argv)</code></td><td>lê os argumentos da linha de comando: o <b>modo</b> (<code>train</code>/<code>infer</code>) e a <b>arquitetura</b> (<code>dense</code>/<code>moe</code>), além das opções opcionais.</td></tr>
<tr><td><code>set_seed(CONFIG["seed"])</code></td><td>fixa as sementes do <code>random</code>, <code>numpy</code> e <code>torch</code>. Sem isto, cada execução teria números aleatórios diferentes e o resultado nunca seria reproduzível.</td></tr>
<tr><td><code>get_device(...)</code></td><td>escolhe CPU (padrão) ou GPU (se usar <code>--cuda</code> e ela existir). O projeto pensado para CPU com 24 GB de RAM.</td></tr>
</table>

Depois vem a configuração global, um dicionário que controla **todas** as
dimensões do modelo:

```python
CONFIG = {
    "hidden_size": 128,      # tamanho do vetor que representa cada token
    "num_layers": 2,         # quantos blocos transformer empilhados
    "num_heads": 4,          # em quantas "cabeças" a atenção se divide
    "num_kv_heads": 2,       # cabeças de key/value (GQA — ver secção 5.2)
    "ffn_hidden": 512,       # largura interna da FFN (4 x hidden_size)
    "max_seq_len": 64,       # comprimento máximo das sequências
    "batch_size": 32,        # frases processadas em paralelo
    "dropout": 0.1,          # regularização
    "learning_rate": 3e-4,   # passo do otimizador
    "epochs": 5,             # voltas completas sobre os dados
    "aux_loss_weight": 0,    # peso da perda auxiliar do MoE
    "num_experts": 3,        # nº de especialistas do MoE
    "top_k": 1,              # nº de experts activados por token
    "num_candidates": 2,     # nº de experts pré-seleccionados pelo prefetch
    "seed": 42,
    "vocab_size": None,      # preenchido depois, pelo tokenizador
}
```

Decorações importantes:

- `hidden_size` deve ser divisível por `num_heads` (128 / 4 = 32).
- `vocab_size` fica `None` de propósito: ele **depende do tokenizador**, que
  só é criado depois dos dados existirem. Só depois disso o modelo é instanciado
  com o vocabulário certo.

> **Vista de pássaro:** o fluxo de dados do modelo inteiro é sempre o mesmo —
> os textos viram números com diferentes formas (shapes). Mantenha isto em
> mente ao longo da leitura:
>
> ```text
> texto  →  ids [B, T]  →  vetores [B, T, 128]  →  ...  →  logits [B, T, vocab]
> ```

---

## 3. Os dados

### 3.1 De onde vêm os textos — `load_texts()`

O treino precisa de texto. O `cricket.py` busca esse texto com **precedência**
para nunca repetir trabalho:

1. **Cache em disco** — se o ficheiro `dataset_cache.txt` já existir, os textos
   são lidos dele. Nenhum download.
2. **Download** — se não houver cache, tenta descarregar datasets em português
   em modo *streaming* (`mc4-pt`, `CrawlPT`, `brwac`, por ordem), juntando até
   5000 textos com comprimento entre 100 e 1000 caracteres. O resultado é
   gravado em `dataset_cache.txt`.
3. **Fallback manual** — se não houver rede nem cache, usa `dataset.txt` (frases
   escritas à mão no projeto).

Dois detalhes de design importantes:

- O texto passa por uma "limpeza": `" ".join(example["text"].split())` colapsa
  espaços e quebras de linha, deixando uma frase limpa.
- O fallback **nunca é gravado no cache**. Assim, se um dia houver rede, a
  próxima execução tenta os downloads reais em vez de ficar presa nas frases
  manuais.

Se nada funcionar, a função lança um erro explicando o que faltou.

### 3.2 O tokenizador — `BPETokenizer`

Computadores não entendem texto, entendem números. O **tokenizador** transforma
"gato" em uma lista de inteiros. O Cricket usa um tokenizador **BPE**
(Byte-Pair Encoding), o mesmo tipo usado em produção.

O importante aqui é o **cache por ficheiro**:

```python
tokenizer = BPETokenizer(
    texts,                         # apenas usado se não houver cache
    vocab_size=4000,               # ~4000 tokens
    max_len=CONFIG["max_seq_len"], # 64
    cache_path=TOKENIZER_CACHE,    # "tokenizer.json"
)
```

- Se `tokenizer.json` **existir**, ele é apenas carregado do disco (zero treino).
- Se **não existir**, o BPE é treinado sobre `texts` e guardado.

**O que acontece ao treinar o BPE, em resumo:**

1. começa com um vocabulário de letras e símbolos;
2. mede quais **pares** de símbolos aparecem mais vezes (ex.: `"qu"` surge muito);
3. funde o par mais frequente num novo "pedaço";
4. repete até atingir `vocab_size`.

O resultado é um vocabulário com ~4000 "pedaços" de texto (incluindo os tokens
especiais `<PAD>` e `<UNK>`). O método `encode()` converte texto em lista de
IDs; `decode()` faz o caminho inverso.

Momentos-chave no final:

```python
self.pad_token_id = self.tokenizer.token_to_id("<PAD>")  # = 0
self.unk_token_id = self.tokenizer.token_to_id("<UNK>")
self.vocab_size = self.tokenizer.get_vocab_size()
```

O token `<PAD>` (padding) ganha o id `0`, o que será usado mais tarde para
ignorar posições de preenchimento. E só agora:

```python
CONFIG["vocab_size"] = tokenizer.vocab_size   # ex.: 4000
```

### 3.3 Dividir em batches — `TextDataset` e `build_data_loaders`

Um `TextDataset` é uma lista de tensores, um por frase:

- cada frase é codificada com o BPE;
- se tiver mais de 64 tokens, é **truncada**;
- se tiver menos, é **preenchida** com `pad_token_id` (=0) até 64.

Assim todas as frases de um batch têm o mesmo tamanho, o que é obrigatório para
o PyTorch montar um tensor retangular.

Depois, `build_data_loaders` divide os textos em **80% treino / 20% validação**.

```python
train_texts, val_texts = train_test_split(
    texts, test_size=0.2, random_state=CONFIG["seed"]   # mesmo split sempre
)
```

e cria dois `DataLoader`s com `batch_size=32`:

- `train_loader` com `shuffle=True` — a ordem muda a cada época (ajuda a treinar);
- `val_loader` com `shuffle=False` — para medir a qualidade de forma estável.

> **Para que serve a validação?** Numere bem: é fácil o modelo *decorar* as
> frases de treino sem generalizar. A validação contém frases que ele **nunca
> viu**. Se a perda de validação parar de melhorar, é sinal de que ele está
> só decorando — hora de parar (early stopping, secção 6.3).

---

## 4. O tokenizador (detalhe extra da inferência)

Na inferência (`infer`), o objetivo é **não fazer trabalho desnecessário**:

```python
if args.mode == "train" or not os.path.exists(TOKENIZER_CACHE):
    texts = load_texts()     # só em train (ou se o tokenizador nunca foi criado)
tokenizer = BPETokenizer(
    texts if texts is not None else [], ...   # carrega do cache; sem downloads
)
```

Ou seja: em `infer`, se `tokenizer.json` existe, ele é carregado direto do disco
— sem downloads, sem treino do BPE. O dataset só é descarregado no modo `train`
(ou se o tokenizador ainda não existe, caso em que precisa dos textos para ser
treinado).

---

## 5. A máquina — arquitetura do modelo

Agora o mais importante: a `CricketLM`. Pensemos primeiro no modelo **completo**
e depois abrimos cada peça.

```python
class CricketLM(nn.Module):
    def __init__(self, config, use_moe=True):
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)  # tabela token→vetor
        self.layers = nn.ModuleList([TransformerBlock(...) for _ in range(num_layers)])
        self.norm  = RMSNorm(hidden_size)        # normalização final
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)  # vetor→logits
        self.lm_head.weight = self.token_embedding.weight   # weight tying
```

O `forward` completo é:

```python
x = self.token_embedding(input_ids)   # ids [B, T]  →  vetores [B, T, 128]

if context_embedding is None and self.use_moe and targets is not None:
    context_embedding = x.mean(dim=1)   # média dos embeddings: (B, 128)

total_aux_loss = 0.0
for layer in self.layers:               # cada bloco transformer
    x, aux_loss = layer(x, context_embedding)
    total_aux_loss += aux_loss

x = self.norm(x)                        # RMSNorm final
logits = self.lm_head(x)                # [B, T, 128] → [B, T, vocab]
```

Detalhes deste fluxo:

- `nn.Embedding(vocab, 128)` é uma **tabela de consulta**: pega o id de cada
  token e devolve a sua linha — o "vetor semântico" de 128 números.
- As camadas processam o tensor várias vezes; a última normalização é seguida
  pela `lm_head`, que produz **logits**: um "placar" bruto para cada um dos
  ~4000 tokens do vocabulário, a cada posição.
- `lm_head.weight = token_embedding.weight` é o **weight tying**: a cabeça de
  saída partilha os pesos do embedding. "Traduzir token→vetor" e "vetor→token"
  usam o mesmo dicionário. Meta menos parâmetros e costuma melhorar modelos
  pequenos.
- O `context_embedding` (média dos embeddings da sequência) existe para o MoE
  tomar decisões por requisição — voltamos a isso na secção 5.4.

Agora, de dentro para fora, cada componente.

### 5.1 As peças pequenas

**RMSNorm** — normalização por raiz quadrada média.

```python
rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
return x * rms * self.weight
```

Nas redes os números tendem a "explodir" ou "desabar", o que desestabiliza o
treino. Este pequeno bloco reescala cada vetor para que a sua magnitude fique
perto de 1:

1. `x.pow(2)` eleva tudo ao quadrado;
2. `.mean(-1, keepdim=True)` calcula a média **na última dimensão** (o vetor de
   features), mantendo a forma para o broadcast;
3. `torch.rsqrt` devolve `1 / √(média)`;
4. `+ eps` (1e-6) evita divisão por zero.

O `self.weight` é um vetor treinável (inicializado a 1) que o treino ajusta para
dar mais ou menos importância a cada dimensão. É uma versão leve da LayerNorm:
não subtrai a média, apenas controla a magnitude.

**SwiGLU** — a FFN "densa". É a rede usada como baseline e também como os
"especialistas" do MoE.

```python
def forward(self, x):
    return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
```

Três projecções:

- `w1` (gate): decide **quanto** de cada dimensão passa (a activação SiLU
  `x·sigmoid(x)` permite inverter/atenuar, não só ligar/desligar);
- `w3` (up): carrega a **informação**;
- `w2` (down): devolve para o tamanho original (`128 → 512 → 128`).

Na prática: `w2( SiLU(w1(x)) × w3(x) )`. É uma FFN "com portão": o próprio
modelo aprende o que transmitir e quanto transmitir por dimensão.

### 5.2 Atenção — `GroupedQueryAttention`

Esta é a peça que permite aos tokens **olharem uns para os outros**. A ideia
(Query/Key/Value):

- **Query (Q):** "o que estou procurando?" — para cada token atual.
- **Key (K):** "do que eu trato?" — de cada token da frase.
- **Value (V):** "o conteúdo que entrego se for escolhido" — também de cada token.

O produto interno entre a Query de um token e a Key de outro mede a **relevância**
entre eles; o softmax transforma isso em pesos; e a soma ponderada dos Values
produz a nova representação de cada token, agora "misturada" com os tokens
relevantes. Resultado para a jogada: `128 → 128` com shapes `[B, T, 128]` por
dentro.

**Configuração concreta do Cricket:** `hidden_size=128`, `num_heads=4`,
`num_kv_heads=2`, logo `head_dim = 128 / 4 = 32`.

As camadas lineares:

```python
self.wq = nn.Linear(128, 128, bias=True)   # 4 heads × 32 = 128 (Queries)
self.wk = nn.Linear(128, 64,  bias=True)   # 2 heads × 32 = 64  (Keys)
self.wv = nn.Linear(128, 64,  bias=True)   # 2 heads × 32 = 64  (Values)
self.wo = nn.Linear(128, 128, bias=True)   # volta ao tamanho original
```

Repare que `wk` e `wv` produzem **metade** das colunas de `wq`. Isso é o **GQA**
(Grouped Query Attention): temos 4 cabeças de Query mas só 2 cabeças de
Key/Value. Em vez de atenção normal, a cabeça de KV é *repetida* para servir a
duas cabeças de Q (ver abaixo). Objectivo: poupar memória — na inferência as
Keys/Values de todos os tokens ficam guardadas em cache, e menos heads de KV
significa cache mais pequeno com qualidade quase igual.

O `forward` passo a passo com shapes:

```python
q = self.wq(x)   # [B, T, 128]
k = self.wk(x)   # [B, T, 64]     (2 heads)
v = self.wv(x)   # [B, T, 64]
```

Organizadas em cabeças — `view` separa a última dimensão e `transpose` traz a
dimensão dos heads para a frente, deixando `(B, heads, T, head_dim)`:

```python
q = q.view(B, T, 4, 32).transpose(1, 2)   # [B, 4, T, 32]
k = k.view(B, T, 2, 32).transpose(1, 2)   # [B, 2, T, 32]
v = v.view(B, T, 2, 32).transpose(1, 2)   # [B, 2, T, 32]
```

Em seguida aplica-se **RoPE** (ver 5.3) a `q` e `k`, e depois:

```python
if self.num_kv_heads != self.num_heads:
    k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
    v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
# k e v ficam [B, 4, T, 32] — cada head KV agora serve 2 heads de query
```

O `repeat_interleave` **copia** cada uma das 2 cabeças de KV para servir a 2
cabeças de query. Este é o mecanismo GQA em ação.

Agora o cálculo da atenção em si:

```python
scale = 1.0 / math.sqrt(self.head_dim)                      # 1/√32
attn = torch.matmul(q, k.transpose(-2, -1)) * scale         # [B, 4, T, T]
attn = attn.masked_fill(self.causal_mask[:T, :T], float('-inf'))  # máscara causal
attn = torch.softmax(attn, dim=-1)                          # pesos que somam 1
out  = torch.matmul(attn, v)                                # [B, 4, T, 32]
out  = out.transpose(1, 2).contiguous().view(B, T, -1)      # [B, T, 128]
return self.wo(out)
```

- O produto `q·kᵀ` gera a matriz token×token de similaridades. Dividir por
  `√head_dim` (escala) evita que scores grandes saturem o softmax.
- A **máscara causal** (triângulo superior com `-inf`) impede o token de "ver o
  futuro": ele só pode atender a si próprio e aos tokens anteriores. Enquanto
  um token só pode ser previsto a partir do passado, o modelo é capaz de gerar
  texto autoregressivamente.
- O softmax transforma cada linha em probabilidades e `attn·v` é a média
  ponderada das Values — o momento em que a informação "flui" entre tokens.
- No fim, as 4 cabeças voltam a ser concatenadas (`[B, T, 128]`) e `wo` mistura
  tudo de novo.

> O dropout e a tabela de RoPE ficam **pré-computados** no construtor:
> `self.register_buffer(...)` guarda tensores que acompanham o modelo mas não
> são parâmetros treináveis.

### 5.3 RoPE — posição por rotação

O transformer não tem noção de ordem por construção; precisamos dizer ao modelo
*onde* cada token está. RoPE (Rotary Positional Embedding) faz isto de forma
elegante: **roda** os vetores de query/key por um ângulo proporcional à posição.

O construtor pré-computa a tabela de ângulos:

```python
inv_freq = 1.0 / (10000 ** (torch.arange(0, 32, 2).float() / 32))   # 16 frequências
t = torch.arange(seq_len)                     # posições 0.1.2...
freqs = torch.einsum("i,j->ij", t, inv_freq)  # posição × frequência
emb = torch.cat((freqs, freqs), dim=-1)       # duplica (rotação em pares)
self.cos = emb.cos()[None, None, :, :]
self.sin = emb.sin()[None, None, :, :]
```

Aquele `10000^(...)` gera frequências de "velocidades" diferentes: umas rodam
rápido (captam posições próximas), outras devagar (captam posição absoluta).
Depois, cada par de dimensões roda como ponteiros de um relógio.

No `forward`, caem-se as tabelas até o comprimento real da sequência e aplica-se
a rotação a `q` e `k` (nunca a `v`, que carrega conteúdo, não posição):

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)          # rotação de 90°

q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

Esta fórmula é a rotação 2D clássica em forma de vetores reais. Como o ângulo
depende da posição `t`, a "distância" entre dois tokens fica codificada dentro
da própria atenção.

### 5.4 O MoE com prefetch — `MoEWithPrefetch`

Esta é a grande diferença entre as duas arquiteturas. Em vez de uma única FFN
SwiGLU para todos os tokens, o **MoE** mantém **3 especialistas** (cada um é um
`SwiGLU`) e decide, para cada token, **qual deles chamar**.

O módulo é:

```python
class MoEWithPrefetch(nn.Module):
    def __init__(self, config):
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(3)])

        self.prefetch_router = nn.Linear(128, 3, bias=False)  # nível 1
        self.gate           = nn.Linear(128, 3, bias=False)   # nível 2

        self.candidate_mask = None   # máscara do prefetch (efémera, não é parâmetro)
```

Há **dois níveis** de roteamento:

**Nível 1 — Prefetch por requisição.** Antes de gerar qualquer token, o
`prefetch_router` recebe o **embedding médio do prompt** (um único vetor de 128)
e produz um "placar" para cada especialista. Os `num_candidates` (2) melhores
ficam com máscara 0; o restante com `-inf`:

```python
def prefetch(self, context_embedding):
    with torch.no_grad():
        scores = self.prefetch_router(context_embedding)   # (3,)
        _, top_m_idx = torch.topk(scores, k=2)
        mask = torch.full((3,), float('-inf'), device=scores.device)
        mask[top_m_idx] = 0.0
        self.candidate_mask = mask      # ex.: [0.0, -inf, 0.0]
```

A ideia: o tema do prompt já **pré-selecciona** um subconjunto de experts antes
de cada token. Economia real: no nível 2 o gate só pode escolher entre 2
especialistas.

**Nível 2 — Gate token-a-token.** Para **cada token**, `self.gate(x)` calcula
scores sobre os 3 experts. A máscara do nível 1 é **somada** a esses scores, o
que invalida o expert não-candidato (valor `-inf`).

```python
scores = self.gate(x)                     # (B, T, 3)

if context_embedding is not None:
    # treino/validação: máscara dinâmica, derivada do embedding médio
    # do próprio exemplo dentro do batch (por isso (B, 3))
    ...
elif not self.training and self.candidate_mask is not None:
    # inferência: usa a máscara fixa guardada por prefetch()
    masked_scores = scores + self.candidate_mask.view(1, 1, -1)
```

Depois, o **top-k** escolhe o melhor entre os candidatos e o softmax normaliza:

```python
top_k_weights, top_k_indices = torch.topk(masked_scores, k=1, dim=-1)
top_k_weights = torch.softmax(top_k_weights, dim=-1)   # nesta config: sempre 1.0
```

Finalmente, os experts escolhidos são chamados e as suas saídas são combinadas
de forma *vetorizada* (pequeno loop apenas por expert, sem loop token a token):

```python
final_out = torch.zeros_like(x)
for e in range(self.num_experts):
    # weight_e: (B, T) — peso dado ao expert e em cada token.
    # `(top_k_indices == e).float()` é uma máscara 0/1; ao multiplicá-la pelos
    # pesos top-k e somar ao longo da dimensão do top-k, obtemos o peso TOTAL
    # que cada token atribuiu ao expert e (0 se ele não foi escolhido).
    weight_e = (top_k_weights * (top_k_indices == e).float()).sum(dim=-1)

    selected = weight_e > 0                   # tokens roteados para o expert e
    if not selected.any():
        continue

    idx = selected.nonzero(as_tuple=True)     # coordenadas (b, t) desse expert
    inp = x[idx]                              # só esses tokens:  (N, H)
    out = self.experts[e](inp)                # forward do expert: (N, H)

    final_out[idx] += weight_e[idx].unsqueeze(-1) * out   # (N,1)*(N,H) → posição original
return final_out, aux_loss
```

Passo a passo desta versão vetorizada:

1. **Peso por token (máscara × pesos).** A informação do top-k é condensada em
   `weight_e`: `(top_k_indices == e)` vale 1 onde o expert `e` está entre os
   escolhidos e 0 no resto; multiplicar pelos `top_k_weights` e somar na dimensão
   do top-k devolve o **peso total** do expert `e` para cada token. Isto também
   funciona com `top_k > 1` — um token pode repartir os seus pesos por vários
   experts, e aqui ficam agregados por expert.
2. **Recolher (gather).** `selected.nonzero(as_tuple=True)` devolve as
   coordenadas `(b, t)` dos tokens que usam o expert; `x[idx]` recolhe só esses
   tokens num tensor `(N, H)`. Os índices são **únicos** (o `nonzero` não repete
   posições), pelo que cada token aparece uma só vez por iteração.
3. **Correr o expert.** `self.experts[e](inp)` processa N tokens de uma só vez —
   o mesmo cálculo do loop explícito, mas feito em lote (muito mais rápido).
4. **Espalhar (scatter-add).** `final_out[idx] += weight_e[idx].unsqueeze(-1) * out`
   multiplica cada saída pelo peso correspondente (shape `(N, 1)`) e reescreve-a
   na posição original do tensor `(B, T, H)`. O `+=` **acumula**: nos passos
   seguintes, outros experts contribuem com os *seus* tokens, somando a sua
   parcela na mesma posição.

O resultado é **matematicamente idêntico** ao da versão com loop explícito, mas
com menos um loop aninhado: a combinação ponderada acontece por operações de
tensor (máscara, gather e scatter-add), que são exactamente as técnicas de
indexação do PyTorch que valem a pena conhecer.

Ou seja: cada token **só passa pelo expert que escolheu**, e o resultado é
colocado de volta na sua posição. Como só 1 de 3 experts é activado por token,
o **custo computacional por token é o mesmo de um modelo denso**, embora o MoE
tenha ~1,8M parâmetros no total (vs ~1,0M do denso).

**A perda auxiliar.** O MoE devolve também `aux_loss`, uma penalização que
incentiva o uso **equilibrado** dos experts:

```python
importance = top_k_weights.sum(dim=[0, 1, 2])           # peso total de cada expert
frac = torch.bincount(idx_flat, minlength=3).float()    # fracção de tokens por expert
frac = frac / (B * T * top_k)
importance = importance / (B * T)
aux_loss = (frac * importance).sum() * self.num_experts
```

- `frac[e]` = que fracção dos tokens usou o expert `e`;
- `importance[e]` = soma dos pesos atribuídos ao expert;
- o produto é **mínimo quando a distribuição é uniforme** — ou seja, quando os
  experts são usados de forma balanceada.

O `aux_loss_weight = 0` na configuração significa que esta perda é **calculada
mas ainda não influencia o treino** (experiência: ligue para `0.01` para ver o
efeito).

### 5.5 A montagem — `TransformerBlock` e o padrão residual

Como as peças se encaixam num bloco:

```python
def forward(self, x, context_embedding=None):
    residual = x
    x = self.norm1(x)          # pré-normalização
    x = self.attention(x)
    x = residual + x           # conexão residual

    residual = x
    x = self.norm2(x)
    if self.use_moe:
        x, aux_loss = self.ffn(x, context_embedding)   # MoE devolve aux_loss
    else:
        x = self.ffn(x)                                 # SwiGLU denso
        aux_loss = 0.0
    x = residual + x
    return x, aux_loss
```

Dois padrões cruciais aqui:

- **Pré-normalização:** a norma vem **antes** do sub-bloco, nunca depois. Isto
  mantém o caminho residual "limpo", permitindo empilhar camadas sem explosão
  de gradiente.
- **Conexão residual (skip):** `x = residual + x`. Cada bloco só precisa
  aprender a **correcção** (`Δ`) sobre a entrada, não reconstruir tudo. Isto
  também dá ao gradiente um "caminho expresso" de volta durante o treino.

A única diferença entre `dense` e `moe` está no `self.ffn`: SwiGLU único contra
`MoEWithPrefetch`. Tudo o resto (atenção, normas, resíduos) é idêntico — é isso
que torna a comparação justa.

### 5.6 A função de perda

Durante o treino, cada sequência serve de "alvo": o modelo vê os tokens 0..(T-1)
e deve prever os tokens 1..(T). O `forward` desloca as previsões em relação aos
alvos:

```python
shift_logits = logits[..., :-1, :].contiguous()   # para cada posição, prevê a próxima
shift_labels = targets[..., 1:].contiguous()
loss_fct = nn.CrossEntropyLoss(ignore_index=0)    # ignora <PAD> (id 0)
ce_loss = loss_fct(shift_logits.reshape(...), shift_labels.reshape(...))
loss = ce_loss + config["aux_loss_weight"] * total_aux_loss
```

- `CrossEntropyLoss(ignore_index=0)` calcula a perda apenas nas posições reais —
  os lugares preenchidos com `<PAD>` (=0) são ignorados.
- A perda final é a soma da **cross-entropy** (quão bom é prever o próximo token)
  com a **perda auxiliar** do MoE (balanceamento dos experts), ponderada por
  `aux_loss_weight`.

---

## 6. O treino

### 6.1 Preparação

`train_model` recebe o modelo, os dois loaders, a config, o device e o caminho
do checkpoint:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': []}
```

AdamW é um otimizador moderno que ajusta cada parâmetro usando o gradiente e um
momento por parâmetro. Todos os parâmetros *internos* (pesos das normais,
atenção, FFN/experts, embedding) são encontrados automaticamente por
`model.parameters()`.

### 6.2 Uma época, passo a passo

Para cada uma das `epochs` (5 por omissão) acontece:

**Fase de treino** — `model.train()` (ativa dropout, registra grafo):

```python
for batch in train_loader:               # lotes de 32 frases
    batch = batch.to(device)             # move para CPU/GPU
    _, loss = model(batch, targets=batch)   # forward: prevê; compara com o alvo
    optimizer.zero_grad()                # limpa gradientes da iteração anterior
    loss.backward()                      # backprop: propaga o erro e calcula gradientes
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip: limita o gradiente
    optimizer.step()                     # aplica um passo de AdamW
    total_train_loss += loss.item()      # acumula a perda desta iteração
```

Momento a momento:

1. **`model(batch, targets=batch)`** — o forward produz logits e calcula a perda
   (cross-entropy + auxiliar). O "alvo" é a própria frase: cada posição aprende
   a prever a seguinte.
2. **`loss.backward()`** — o PyTorch percorre o grafo de operações ao contrário
   e calcula, para cada parâmetro, **como a perda varia com ele** (o gradiente).
3. **`clip_grad_norm_(..., 1.0)`** — se o gradiente ficar gigante (o que acontece
   em modelos instáveis), é limitado à norma 1.0. Isto evita "explosões" de
   treino.
4. **`optimizer.step()`** — cada parâmetro é ligeiramente atualizado na direção
   que reduz a perda, com passo `learning_rate`.

Ao fim da fase de treino guarda-se a perda média: `avg_train_loss`.

**Fase de validação** — `model.eval()` (desliga dropout) e `torch.no_grad()`
(para não gastar memória registando o grafo, já que não vamos retropropagar):

```python
for batch in val_loader:
    _, loss = model(batch, targets=batch)
    total_val_loss += loss.item()
```

Aqui mede-se a qualidade em frases **nunca vistas**. Uma frase é logo impressa:

```text
Época  1 | Train: 4.2310 | Val: 4.1542
```

### 6.3 Early stopping, checkpoint e curva

```python
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    patience_counter = 0
    torch.save(model.state_dict(), checkpoint_path)   # guarda o melhor
else:
    patience_counter += 1
    if patience_counter >= patience:   # 5 épocas sem melhorar
        print("Early stopping...")
        break
```

- `best_val_loss` começa em +∞, logo a 1.ª época sempre "melhora" e o checkpoint
  é gravado.
- Quando a validação **deixa de melhorar** durante `patience` épocas seguidas
  (5), o treino para: *early stopping*. Objetivo: não desperdiçar tempo e não
  deixar o modelo decorar os dados.
- No fim, o **melhor** estado (o de menor val loss) é recarregado e devolvido.
- `plot_history` desenha as duas curvas (treino vs validação) num gráfico.

> **Checkpoints antigos:** a função `load_checkpoint` ainda aceita ficheiros
> criados por versões antigas que usavam `torch.compile`. Nesses ficheiros as
> chaves trazem o prefixo `_orig_mod.`, que é removido automaticamente antes do
> `load_state_dict`. Assim modelos já treinados continuam a funcionar.

---

## 7. A inferência

Enquanto o treino "aprende", a inferência "usa". Tudo começa outra vez no
`main()`, agora com `mode = "infer"`:

1. carrega o tokenizador do cache (sem downloads);
2. cria o modelo com a mesma arquitetura;
3. carrega o checkpoint (`load_checkpoint(model, ckpt, device)`). Se não existir,
   avisa que é preciso treinar primeiro;
4. chama `run_chat(...)`.

### 7.1 O loop de chat — `run_chat`

É um simples loop de terminal:

```python
while True:
    prompt = input("Você: ").strip()
    if prompt.lower() in {"exit", "quit", "sair", "q"}:
        break
    print("Cricket: ", end='', flush=True)
    generate_text(model, tokenizer, prompt, ...)
```

Lê uma frase do utilizador, gera a resposta, espera a próxima. Termina com
`exit`/`sair`/`q`.

### 7.2 `generate_text` — o ciclo autoregressivo

Esta função é o coração da inferência. Primeiro, prepara o modelo e o prompt:

```python
model.eval()
input_ids = tokenizer.encode(prompt)
input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)   # [1, T]
```

**Prefetch por requisição (MoE):** antes de qualquer token, se o modelo é MoE,
calcula-se o embedding médio do prompt e fixa-se a máscara de candidatos **em
todas** as camadas:

```python
if model.use_moe:
    with torch.no_grad():
        emb = model.token_embedding(input_tensor)     # (1, T, 128)
        context_emb = emb.mean(dim=1).squeeze(0)      # (128,)
        model.prefetch_all_layers(context_emb)
```

`prefetch_all_layers` percorre as camadas e chama `layer.ffn.prefetch(...)` em
cada `MoEWithPrefetch`. A partir deste momento o roteamento da **requisição
inteira** está decidido — é exactamente o "pré-filtro" descrito na secção 5.4.

Depois vem a **geração token a token**:

```python
generated = input_tensor.clone()
for _ in range(max_new_tokens):
    with torch.no_grad():
        logits, _ = model(generated)      # forward completo até o último token
        next_logits = logits[:, -1, :]    # olhamos APENAS a última posição
        ...
        next_token = <amostra ou argmax>(next_logits)
        generated = torch.cat([generated, next_token], dim=-1)   # fecha o loop
        if next_token.item() == tokenizer.pad_token_id:          # <PAD>? para.
            break
```

Os quatro controles de amostragem, nesta ordem:

**1. Repetition penalty:**

```python
last_tokens = generated[0, -10:].tolist()
for token_id in set(last_tokens):
    next_logits[0, token_id] /= repetition_penalty
```

Os tokens dos últimos 10 passos têm os logits **divididos** por `repetition_penalty`
(ex.: 1.2), tornando-os menos prováveis e reduzindo loops do tipo
"gato gato gato gato...".

**2. Top-k sampling:**

```python
indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
next_logits[indices_to_remove] = float('-inf')
```

Se `top_k > 0`, mantém-se apenas os `top_k` tokens mais prováveis; todos os
outros vão para `-inf` (excluídos na prática). Concentra a amostragem na região
promissora.

**3. Temperatura e amostragem:**

```python
if temperature > 0:
    probs = torch.softmax(next_logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
else:
    next_token = next_logits.argmax(dim=-1, keepdim=True)   # greedy
```

- `temperature = 0` → sempre o token mais provável (greedy, determinístico);
- `temperature > 0` → divide os logits por `T` antes do softmax: temperaturas
  baixas "afinam" a distribuição (mais previsível), altas "achatam" (mais
  aleatório). Depois amostra-se com `torch.multinomial`.

**4. Decodificação e paragem:**

```python
print(tokenizer.decode([next_token.item()]), end='', flush=True)  # streaming
if next_token.item() == tokenizer.pad_token_id:
    break
```

Cada token é transformado de volta em texto e impresso em tempo real
(`flush=True` sem quebra de linha — efeito de "escrever a resposta"). Quando o
modelo gera `<PAD>`, assume-se que a resposta acabou e o ciclo para.

No final, a função devolve três coisas:

```python
return generated, full_text, new_text
```

- `generated`: os IDs completos (prompt + geração);
- `full_text`: texto decodificado completo;
- `new_text`: apenas a parte **gerada** (sem o prompt) — útil para GUI/chat.

---

## 8. Do início ao fim — mapa mental

Tudo o que vimos em um diagrama:

```text
TREINO                                    INFERÊNCIA
----------------------------------        ----------------------------------
textos
  │ load_texts() (cache→download→fallback)
  ▼
BPETokenizer ──→ tokenizer.json ──────►  carregado do cache
  │
  ▼
TextDataset + split 80/20 ──► loaders
  │
  ▼
CricketLM (embedding → 2 blocos → norm → lm_head)
  │  targets=self  (prever o próximo token)
  ▼
loss = cross-entropy + aux_weight·aux   cricket_model_{dense|moe}_best.pt
  │  │                                        │
  │  backprop + clip + AdamW                  │ torch.load + load_checkpoint
  └──► repete                                 ▼
                                     run_chat ──► generate_text
                                        │          prefetch (MoE)
                                        │          + loop autoregressivo:
                                        │          repetition penalty → top-k
                                        │          → temperatura → decodificar
                                        ▼
                                    "Cricket: ..."   (token a token)
```

**Em uma frase:** o treino transforma texto bruto em pesos (via previsão do
próximo token, com backprop e early stopping); a inferência reutiliza esses
pesos para, token a token, escolher a continuação mais provável do prompt — e
no MoE o tema da requisição é decidido antes (prefetch) para escolher quais
especialistas podem responder.

---

*Este documento reflete o `cricket.py` atual (versão refatorada, com CLI e chat
interactivo). A referência de alto nível encontra-se em `arquitetura.md`.*